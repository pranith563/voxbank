"""
orchestrator/src/app.py

FastAPI application for VoxBank Orchestrator
Main entry point for LLM orchestration and conversation engine
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any
from dotenv import load_dotenv
import os
import logging
import asyncio
import httpx
import json
# Load environment
load_dotenv()

# Set up file-based logging
from logging_config import setup_logging, get_logger
setup_logging()

# Logging
logger = get_logger("voxbank.orchestrator")

from agent.agent import VoxBankAgent
from agent.orchestrator import ConversationOrchestrator
from gemini_llm_client import GeminiLLMClient
from clients.mcp_client import MCPClient
from context.session_manager import get_session_manager
from voice_processing import (
    transcribe_audio_to_text,
    synthesize_text_to_audio,
    audio_bytes_to_data_url,
    extract_voice_embedding,
)
from agent.helpers import translate_text

VOX_BANK_BASE_URL = os.getenv("VOX_BANK_BASE_URL", "http://localhost:9000")

LANGUAGE_PRESETS = {
    "en": {"preferred": "en", "stt": "en-IN", "tts": "en-IN"},
    "hi": {"preferred": "hi", "stt": "hi-IN", "tts": "hi-IN"},
}
LANGUAGE_DEFAULT = LANGUAGE_PRESETS["en"]


def _resolve_language_preset(lang_value: Optional[str]) -> Dict[str, str]:
    if not lang_value:
        return LANGUAGE_DEFAULT
    code = lang_value.strip().lower()
    if code.startswith("hi"):
        return LANGUAGE_PRESETS["hi"]
    return LANGUAGE_PRESETS["en"]


def _apply_language_settings(
    session: Dict[str, Any],
    session_id: str,
    *candidates: Optional[str],
) -> Dict[str, str]:
    chosen: Optional[Dict[str, str]] = None
    for candidate in candidates:
        if candidate and candidate.strip():
            chosen = _resolve_language_preset(candidate)
            break
    if chosen is None:
        chosen = _resolve_language_preset(session.get("preferred_language"))

    changed = (
        session.get("preferred_language") != chosen["preferred"]
        or session.get("stt_lang") != chosen["stt"]
        or session.get("tts_lang") != chosen["tts"]
    )

    session["preferred_language"] = chosen["preferred"]
    session["stt_lang"] = chosen["stt"]
    session["tts_lang"] = chosen["tts"]

    if changed:
        session_manager.save_session(session_id, session)
        logger.info(
            "Session %s language -> preferred=%s stt=%s tts=%s",
            session_id,
            session["preferred_language"],
            session["stt_lang"],
            session["tts_lang"],
        )
    return chosen


async def _translate_with_fallback(
    text: Optional[str],
    source_lang: str,
    target_lang: str,
    agent: VoxBankAgent,
    log_prefix: str,
) -> str:
    if not text:
        return ""
    if source_lang.lower() == target_lang.lower():
        return text
    try:
        translated = await translate_text(text, source_lang, target_lang, agent.call_llm)
        logger.info(
            "%s translation %s->%s succeeded",
            log_prefix,
            source_lang,
            target_lang,
        )
        return translated.strip() or text
    except Exception as exc:
        logger.exception(
            "%s translation failed (%s->%s): %s",
            log_prefix,
            source_lang,
            target_lang,
            exc,
        )
        return text



# Pydantic models
# for text endpoints
class TextRequest(BaseModel):
    transcript: str
    session_id: str
    user_id: str
    # optional flag to request a short/long reply
    reply_style: Optional[str] = "concise"  # 'concise' | 'detailed'
    # optional flag: also request audio TTS for this reply
    output_audio: Optional[bool] = False
    # optional user language hint ("en", "hi", etc.)
    language: Optional[str] = None
    preferred_language: Optional[str] = None

class TextResponse(BaseModel):
    response_text: str
    session_id: str
    requires_confirmation: bool = False
    audio_url: Optional[str] = None
    meta: Optional[Dict[str, Any]] = None
    logged_out: bool = False

class VoiceRequest(BaseModel):
    audio_data: Optional[str] = None  # base64-encoded audio (optional)
    transcript: Optional[str] = None  # optional pre-transcribed text
    session_id: str
    user_id: str
    language: Optional[str] = None
    preferred_language: Optional[str] = None

class ConfirmRequest(BaseModel):
    session_id: str
    confirm: bool
    user_id: Optional[str] = None


class VoiceResponse(BaseModel):
    response_text: str
    audio_url: Optional[str] = None
    session_id: str
    requires_confirmation: bool = False
    meta: Optional[Dict[str, Any]] = None
    logged_out: bool = False


class RegisterRequest(BaseModel):
    username: str
    passphrase: str
    email: Optional[str] = None
    full_name: Optional[str] = None
    phone_number: Optional[str] = None
    address: Optional[str] = None
    date_of_birth: Optional[str] = None  # ISO date string (not used by mock-bank today)
    audio_data: Optional[str] = None  # base64-encoded audio for embedding
    session_id: Optional[str] = None  # orchestrator session to bind user to


class RegisterResponse(BaseModel):
    status: str
    user: Dict[str, Any]


class LogoutRequest(BaseModel):
    session_id: str


class LoginRequest(BaseModel):
    username: str
    passphrase: str
    session_id: str


# FastAPI app
app = FastAPI(title="VoxBank Orchestrator", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Session management (supports in-memory or Redis based on env vars)
session_manager = get_session_manager()

# Backwards-compat alias so any code that still reads SESSIONS sees the same storage.
SESSIONS: Any = getattr(session_manager, "sessions", None)


def perform_session_logout(session_id: str) -> None:
    """
    Clear authentication/session state for a given session_id.
    Used by explicit logout endpoint and the logout_user MCP tool.
    """
    logger.info("Performing session logout for session_id=%s", session_id)
    session = session_manager.get_session(session_id)
    if session:
        session["user_id"] = None
        session["username"] = None
        session["is_authenticated"] = False
        session["is_voice_verified"] = False
        session["primary_account"] = None
        session["accounts"] = []
        session["beneficiaries"] = []
        session["user_profile"] = {}
        session["pending_action"] = None
        session["pending_clarification"] = None
        session["context"] = {}
        session["history"] = []
        session["conversation_history"] = []
        session["preferred_language"] = "en"
        session["stt_lang"] = "en-IN"
        session["tts_lang"] = "en-IN"
        session_manager.save_session(session_id, session)

    try:
        agent: VoxBankAgent = app.state.agent
        agent.auth_state.pop(session_id, None)
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("Failed to clear agent auth state for session %s: %s", session_id, exc)


def get_session_profile(session_id: str) -> Dict[str, Any]:
    """
    Return a compact session profile structure for the given session_id.

    This is used by downstream endpoints so they don't need to know the
    internal layout of the SessionManager's session dict.
    """
    sess = session_manager.get_session(session_id)
    if not sess:
        logger.info("Session profile requested for unknown session_id=%s", session_id)
        return {
            "user_id": None,
            "username": None,
            "is_authenticated": False,
            "is_voice_verified": False,
            "primary_account": None,
            "accounts": [],
            "beneficiaries": [],
            "preferred_language": "en",
            "stt_lang": "en-IN",
            "tts_lang": "en-IN",
        }

    profile = {
        "user_id": sess.get("user_id"),
        "username": sess.get("username"),
        "is_authenticated": bool(sess.get("is_authenticated")),
        "is_voice_verified": bool(sess.get("is_voice_verified")),
        "primary_account": sess.get("primary_account"),
        "accounts": sess.get("accounts") or [],
        "beneficiaries": sess.get("beneficiaries") or [],
        "user_profile": sess.get("user_profile") or {},
        "preferred_language": sess.get("preferred_language") or "en",
        "stt_lang": sess.get("stt_lang") or "en-IN",
        "tts_lang": sess.get("tts_lang") or "en-IN",
    }
    logger.info(
        "Session profile for %s -> user_id=%s username=%s primary_account=%s accounts=%d",
        session_id,
        profile["user_id"],
        profile["username"],
        profile["primary_account"],
        len(profile["accounts"]),
    )
    return profile




async def hydrate_session_profile_from_mock_bank(session_id: str, user_id: str) -> None:
    """
    Fetch user profile + accounts (and beneficiaries) from mock-bank and cache
    them on the session.

    This is used after a successful login (either via HTTP login endpoint or
    conversational auth) so that subsequent LLM/tool calls can rely on a
    populated session_profile.
    """
    accounts_compact: list[Dict[str, Any]] = []
    beneficiaries: list[Dict[str, Any]] = []
    primary_account: Optional[str] = None

    profile_data: Dict[str, Any] = {}

    # Prefer MCP tools for profile + accounts (+ beneficiaries)
    mcp = getattr(app.state, "mcp_client", None)
    if mcp is not None:
        logger.info(
            "hydrate_session_profile: using MCP tools get_user_profile/get_user_accounts for user_id=%s",
            user_id,
        )
        # Fetch profile via MCP (FastMCP returns CallToolResult; HTTP fallback returns dict)
        try:
            prof_res = await mcp.call_tool("get_user_profile", {"user_id": user_id})
            prof_payload = getattr(prof_res, "data", prof_res)
            if isinstance(prof_payload, dict) and prof_payload.get("status") == "success":
                profile_data = prof_payload.get("user") or {}
                logger.info(
                    "hydrate_session_profile: MCP profile fetch successful for user_id=%s username=%s",
                    user_id,
                    profile_data.get("username"),
                )
            else:
                logger.warning(
                    "hydrate_session_profile: MCP get_user_profile returned non-success for user_id=%s: %s",
                    user_id,
                    prof_res,
                )
        except Exception as e:
            logger.exception("hydrate_session_profile: MCP get_user_profile failed for user_id=%s error=%s", user_id, e)

        # Fetch accounts via MCP
        accounts_full: list[Dict[str, Any]] = []
        try:
            acc_res = await mcp.call_tool("get_user_accounts", {"user_id": user_id})
            acc_payload = getattr(acc_res, "data", acc_res)
            if isinstance(acc_payload, dict) and acc_payload.get("status") == "success":
                accounts_full = acc_payload.get("accounts") or []
            else:
                logger.warning(
                    "hydrate_session_profile: MCP get_user_accounts returned non-success for user_id=%s: %s",
                    user_id,
                    acc_res,
                )
        except Exception as e:
            logger.exception("hydrate_session_profile: MCP get_user_accounts failed for user_id=%s error=%s", user_id, e)

        # Fetch beneficiaries via MCP (optional, for richer USER CONTEXT)
        try:
            ben_res = await mcp.call_tool("get_user_beneficiaries", {"user_id": user_id})
            ben_payload = getattr(ben_res, "data", ben_res)
            if isinstance(ben_payload, dict) and ben_payload.get("status") == "success":
                beneficiaries = ben_payload.get("beneficiaries") or []
                logger.info(
                    "hydrate_session_profile: MCP beneficiary fetch successful for user_id=%s count=%d",
                    user_id,
                    len(beneficiaries),
                )
            else:
                logger.warning(
                    "hydrate_session_profile: MCP get_user_beneficiaries returned non-success for user_id=%s: %s",
                    user_id,
                    ben_res,
                )
        except Exception as e:
            logger.exception(
                "hydrate_session_profile: MCP get_user_beneficiaries failed for user_id=%s error=%s",
                user_id,
                e,
            )

        # Build compact accounts + primary from accounts_full
        try:
            for acc in accounts_full:
                acct_num = acc.get("account_number")
                if not acct_num:
                    continue
                accounts_compact.append(
                    {
                        "account_number": acct_num,
                        "account_type": acc.get("account_type"),
                        "currency": acc.get("currency"),
                    }
                )
            for acc in accounts_full:
                if (acc.get("account_type") or "").strip().lower() == "savings":
                    primary_account = acc.get("account_number")
                    break
            if not primary_account and accounts_full:
                primary_account = accounts_full[0].get("account_number")

            logger.info(
                "hydrate_session_profile: cached %d accounts for user_id=%s primary_account=%s via MCP",
                len(accounts_compact),
                user_id,
                primary_account,
            )
        except Exception as e:
            logger.exception(
                "hydrate_session_profile: error while building account summary from MCP data for user_id=%s: %s",
                user_id,
                e,
            )
    else:
        # Fallback to direct HTTP if MCP client is not available
        base = VOX_BANK_BASE_URL.rstrip("/") if VOX_BANK_BASE_URL else None
        if not base:
            logger.error(
                "hydrate_session_profile: neither MCP client nor VOX_BANK_BASE_URL configured; cannot hydrate profile"
            )
        else:
            try:
                async with httpx.AsyncClient(timeout=5.0) as client:
                    profile_url = f"{base}/api/users/{user_id}"
                    accounts_url = f"{base}/api/users/{user_id}/accounts"
                    beneficiaries_url = f"{base}/api/users/{user_id}/beneficiaries"

                    logger.info("hydrate_session_profile: HTTP fetching profile from %s", profile_url)
                    try:
                        profile_resp = await client.get(profile_url)
                        profile_resp.raise_for_status()
                        profile_data = profile_resp.json() or {}
                        logger.info(
                            "hydrate_session_profile: HTTP profile fetch successful for user_id=%s username=%s",
                            user_id,
                            profile_data.get("username"),
                        )
                    except Exception as e:
                        logger.warning(
                            "hydrate_session_profile: HTTP profile fetch failed for user_id=%s error=%s",
                            user_id,
                            e,
                        )

                    logger.info("hydrate_session_profile: HTTP fetching accounts from %s", accounts_url)
                    try:
                        accounts_resp = await client.get(accounts_url)
                        accounts_resp.raise_for_status()
                        accounts_full = accounts_resp.json() or []

                        for acc in accounts_full:
                            acct_num = acc.get("account_number")
                            if not acct_num:
                                continue
                            accounts_compact.append(
                                {
                                    "account_number": acct_num,
                                    "account_type": acc.get("account_type"),
                                    "currency": acc.get("currency"),
                                }
                            )
                        for acc in accounts_full:
                            if (acc.get("account_type") or "").strip().lower() == "savings":
                                primary_account = acc.get("account_number")
                                break
                        if not primary_account and accounts_full:
                            primary_account = accounts_full[0].get("account_number")

                        logger.info(
                            "hydrate_session_profile: HTTP cached %d accounts for user_id=%s primary_account=%s",
                            len(accounts_compact),
                            user_id,
                            primary_account,
                        )
                    except Exception as e:
                        logger.warning(
                            "hydrate_session_profile: HTTP accounts fetch failed for user_id=%s error=%s",
                            user_id,
                            e,
                        )
                    # Beneficiaries via HTTP (optional)
                    logger.info("hydrate_session_profile: HTTP fetching beneficiaries from %s", beneficiaries_url)
                    try:
                        ben_resp = await client.get(beneficiaries_url)
                        ben_resp.raise_for_status()
                        beneficiaries = ben_resp.json() or []
                        logger.info(
                            "hydrate_session_profile: HTTP cached %d beneficiaries for user_id=%s",
                            len(beneficiaries),
                            user_id,
                        )
                    except Exception as e:
                        logger.warning(
                            "hydrate_session_profile: HTTP beneficiaries fetch failed for user_id=%s error=%s",
                            user_id,
                            e,
                        )
            except Exception as e:
                logger.exception(
                    "hydrate_session_profile: unexpected HTTP error while fetching profile/accounts: %s", e
                )

    # Update session profile fields
    sess = session_manager.ensure_session(session_id, user_id=user_id)
    # Prefer username from profile if available
    if profile_data.get("username"):
        sess["username"] = profile_data["username"]
    # Store a safe copy of the user profile (without large/secret fields)
    safe_profile = dict(profile_data) if profile_data else {}
    safe_profile.pop("audio_embedding", None)
    safe_profile.pop("passphrase", None)
    sess["user_profile"] = safe_profile
    sess["accounts"] = accounts_compact
    sess["beneficiaries"] = beneficiaries
    sess["primary_account"] = primary_account
    sess["is_authenticated"] = True
    session_manager.save_session(session_id, sess)

# Instantiate clients on startup
@app.on_event("startup")
async def startup_event():
    """
    Startup wiring:
     - Initialize Gemini LLM client
     - Initialize MCP client (discovery)
     - Instantiate the VoxBankAgent + ConversationOrchestrator
     - Attempt to list available MCP tools and log them
    """
    # Load GEMINI settings
    gemini_key = os.getenv("GEMINI_API_KEY")
    gemini_model = os.getenv("GEMINI_MODEL", "gemini-2.5-pro")
    if not gemini_key:
        logger.error("GEMINI_API_KEY not set. LLM generation will not work until key is provided.")
    # Create clients, session manager and agent singletons
    app.state.gemini_client = GeminiLLMClient(api_key=gemini_key, model=gemini_model)
    app.state.mcp_client = MCPClient(base_url=os.getenv("MCP_TOOL_BASE_URL"))
    app.state.session_manager = session_manager

    # Initialize MCP client (discovery / open transport)
    tool_spec: Dict[str, Any] = {}
    try:
        await app.state.mcp_client.initialize()
        logger.info(
            "MCP client initialized (base=%s transport=%s)",
            app.state.mcp_client.base_url,
            getattr(app.state.mcp_client, "transport_mode", "unknown"),
        )

        # Discover tools from MCP `list_tools` (primary source of tool metadata)
        try:
            raw = await app.state.mcp_client.call_tool("list_tools", {})
            if isinstance(raw, dict) and isinstance(raw.get("tools"), dict):
                tool_spec = raw["tools"]
            else:
                logger.info("MCP list_tools returned unexpected shape: %s", raw)
        except Exception as exc:
            logger.exception("Failed to call MCP list_tools: %s", exc)

    except Exception as e:
        logger.exception("MCP client initialize failed: %s", e)

    # Instantiate VoxBankAgent with dynamic tool_spec (falls back internally if empty)
    app.state.agent = VoxBankAgent(
        model_name=gemini_model,
        llm_client=app.state.gemini_client,
        mcp_client=app.state.mcp_client,
        tool_spec=tool_spec or None,
    )
    # Attach a ConversationOrchestrator for ReAct loop and MCP tool execution
    app.state.orchestrator = ConversationOrchestrator(app.state.agent, app.state.mcp_client)

    # Log discovered tools for observability
    if tool_spec:
        logger.info("Startup: loaded %d tools from MCP list_tools:", len(tool_spec))
        for name, meta in tool_spec.items():
            logger.info(" - %s: %s", name, meta.get("description", ""))
    else:
        logger.warning("Startup: no tool metadata loaded from MCP; VoxBankAgent will use fallback tool spec")

    logger.info(
        "Orchestrator started. Gemini model=%s MCP base=%s",
        gemini_model,
        getattr(app.state.mcp_client, "base_url", "unknown"),
    )
    logger.info("=" * 80)


@app.on_event("shutdown")
async def shutdown_event():
    logger.info("=" * 80)
    logger.info("ORCHESTRATOR SHUTDOWN")
    logger.info("=" * 80)
    
    # close httpx client
    mcp = getattr(app.state, "mcp_client", None)
    if mcp:
        try:
            logger.info("Closing MCP client...")
            await mcp.close()
            logger.info("✓ MCP client closed")
        except Exception as e:
            logger.exception("Error closing MCP client: %s", e)
    
    logger.info("Orchestrator shutdown complete.")
    logger.info("=" * 80)


@app.get("/")
async def root():
    return {"message": "VoxBank Orchestrator API", "status": "running"}


@app.get("/api/health")
async def health_check():
    return {"status": "healthy"}


@app.get("/api/session/me")
async def session_me(session_id: str) -> Dict[str, Any]:
    """
    Return the authenticated user (if any) for a given session_id.

    This endpoint is used by the frontend on load to determine whether a
    user is already logged in for the current session.
    """
    logger.info("API Request: GET /api/session/me | session_id=%s", session_id)
    sess = session_manager.get_session(session_id)
    if not sess or not sess.get("user_id"):
        return {"authenticated": False, "user": None}
    return {
        "authenticated": True,
        "user": {
            "user_id": sess.get("user_id"),
            "username": sess.get("username"),
        },
    }


@app.post("/api/auth/logout")
async def logout(req: LogoutRequest):
    """
    Clear authentication/session state for a given session_id.
    Used by the frontend when the user clicks Logout.
    """
    session_id = req.session_id
    logger.info("API Request: POST /api/auth/logout | session_id=%s", session_id)

    perform_session_logout(session_id)

    return {"status": "ok"}


@app.post("/api/auth/register", response_model=RegisterResponse)
async def register_user(request: RegisterRequest):
    """
    Register a new VoxBank user.

    This endpoint is deterministic and does NOT use the LLM.
    It proxies to the mock-bank /api/register endpoint and, on success,
    binds the returned user_id/username to the given session_id (if provided).
    """
    logger.info("API Request: POST /api/auth/register | username=%s", request.username)

    base = VOX_BANK_BASE_URL.rstrip("/") if VOX_BANK_BASE_URL else None
    if not base:
        logger.error("VOX_BANK_BASE_URL is not configured; cannot register user")
        raise HTTPException(status_code=500, detail="mock-bank base URL not configured")

    username_norm = (request.username or "").strip().lower()
    passphrase_norm = (request.passphrase or "").strip().lower()

    payload: Dict[str, Any] = {
        "username": username_norm,
        "passphrase": passphrase_norm,
        "email": request.email,
        "full_name": request.full_name,
        "phone_number": request.phone_number,
        "address": request.address,
        "date_of_birth": request.date_of_birth,
        "audio_data": request.audio_data,
    }

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            url = f"{base}/api/register"
            logger.info("Register: calling mock-bank %s", url)
            resp = await client.post(url, json=payload)
            if resp.status_code != 200:
                try:
                    detail = resp.json().get("detail")
                except Exception:
                    detail = resp.text
                logger.warning("Register: mock-bank error status=%s detail=%s", resp.status_code, detail)
                raise HTTPException(status_code=resp.status_code, detail=detail or "Registration failed")
            user = resp.json()

        # Bind session -> user if a session_id was provided
        if request.session_id:
            user_id = str(user.get("user_id"))
            sess = session_manager.ensure_session(request.session_id, user_id=user_id)
            sess["username"] = (user.get("username") or username_norm)
            session_manager.save_session(request.session_id, sess)
            # Hydrate session profile (user_profile, accounts, primary_account, is_authenticated)
            try:
                await hydrate_session_profile_from_mock_bank(request.session_id, user_id)
            except Exception as e:
                logger.exception("Register: failed to hydrate session profile after registration: %s", e)
            # Keep agent auth_state in sync so tools can be used without conversational login
            try:
                agent: VoxBankAgent = app.state.agent
                state = agent._get_auth_state(request.session_id)
                state["authenticated"] = True
                state["user_id"] = user_id
                state["flow_stage"] = None
                state["temp"] = {}
            except Exception as e:
                logger.exception("Register: failed to sync agent auth_state: %s", e)

        return RegisterResponse(status="success", user=user)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error in /api/auth/register: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/auth/login")
async def login_user(request: LoginRequest) -> Dict[str, Any]:
    """
    Login endpoint for the frontend.

    This validates credentials against mock-bank /api/login and, on success,
    records {user_id, username} in the orchestrator session and agent auth_state.
    """
    logger.info("API Request: POST /api/auth/login | username=%s session_id=%s", request.username, request.session_id)

    base = VOX_BANK_BASE_URL.rstrip("/") if VOX_BANK_BASE_URL else None
    if not base:
        logger.error("VOX_BANK_BASE_URL is not configured; cannot perform login")
        raise HTTPException(status_code=500, detail="mock-bank base URL not configured")

    username_norm = (request.username or "").strip().lower()
    passphrase_norm = (request.passphrase or "").strip().lower()
    payload = {"username": username_norm, "passphrase": passphrase_norm}

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            url = f"{base}/api/login"
            logger.info("Login: calling mock-bank %s", url)
            resp = await client.post(url, json=payload)
            if resp.status_code != 200:
                try:
                    detail = resp.json().get("detail")
                except Exception:
                    detail = resp.text
                logger.warning("Login: mock-bank error status=%s detail=%s", resp.status_code, detail)
                raise HTTPException(status_code=resp.status_code, detail=detail or "Login failed")
        data = resp.json()

        user_id = str(data.get("user_id"))
        username = (data.get("username") or username_norm)

        # Bind basic session state
        sess = session_manager.ensure_session(request.session_id, user_id=user_id)
        sess["username"] = username
        # hydrate profile (includes setting is_authenticated)
        await hydrate_session_profile_from_mock_bank(request.session_id, user_id)

        # Sync agent auth state
        try:
            agent: VoxBankAgent = app.state.agent
            state = agent._get_auth_state(request.session_id)
            state["authenticated"] = True
            state["user_id"] = user_id
            state["flow_stage"] = None
            state["temp"] = {}
        except Exception as e:
            logger.exception("Login: failed to sync agent auth_state: %s", e)

        return {
            "status": "ok",
            "user": {"user_id": user_id, "username": username},
            "session_id": request.session_id,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error in /api/auth/login: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


# WebSocket contract (voice/text streaming)
# - Client connects to: ws://<host>:<port>/ws
# - First message (JSON, text frame) SHOULD be:
#     { "type": "meta", "sampleRate": 16000, "channels": 1, "encoding": "pcm16", "lang": "en" }
#   The orchestrator currently ignores audio frames and uses text only.
# - Client sends JSON events such as:
#     { "event": "end" }   # signals end of an utterance (optional)
#   or a plain text transcript message:
#     { "type": "transcript", "text": "user question here", "session_id": "<optional>" }
# - Server replies with JSON messages:
#     { "type": "meta_ack", "message": "meta received" }
#     { "type": "reply", "text": "<assistant reply text>" }
#     { "type": "error", "message": "<error description>" }

@app.websocket("/ws")
async def websocket_chat(ws: WebSocket):
    """
    WebSocket endpoint used by the frontend voice/chat UI.

    It currently treats incoming text messages as complete user transcripts and
    routes them through agent.orchestrate(), returning a single "reply"
    message for each transcript.
    """
    await ws.accept()
    logger.info("WS: connection opened from %s", ws.client)

    try:
        # Optional: track a session_id per connection
        current_session_id: Optional[str] = None
        ws_lang_hint: Optional[str] = None

        while True:
            message = await ws.receive()

            # Handle disconnect ping/pong or close codes
            if "text" not in message and "bytes" not in message:
                continue

            if message.get("bytes") is not None:
                # Audio frames are currently not processed in the orchestrator
                logger.debug("WS: received binary frame (%d bytes) - ignored", len(message["bytes"]) if message["bytes"] else 0)
                continue

            raw_text = message.get("text") or ""
            logger.info("WS: text frame received: %s", raw_text[:200])

            # Parse JSON if possible
            try:
                data = json.loads(raw_text)
            except json.JSONDecodeError:
                await ws.send_text(json.dumps({"type": "error", "message": "invalid_json"}))
                continue

            msg_type = data.get("type")

            # Handle initial meta message
            if msg_type == "meta":
                logger.info(
                    "WS: meta received sampleRate=%s channels=%s encoding=%s lang=%s",
                    data.get("sampleRate"),
                    data.get("channels"),
                    data.get("encoding"),
                    data.get("lang"),
                )
                ws_lang_hint = data.get("lang")
                await ws.send_text(json.dumps({"type": "meta_ack", "message": "meta received"}))
                continue

            # Handle end-of-utterance marker (no-op for now)
            if data.get("event") == "end":
                logger.debug("WS: received end-of-utterance event")
                continue

            # Treat transcript messages as user input to the LLM agent
            if msg_type == "transcript":
                transcript = (data.get("text") or "").strip()
                if not transcript:
                    await ws.send_text(json.dumps({"type": "error", "message": "empty_transcript"}))
                    continue

                output_audio = bool(data.get("output_audio"))

                # Resolve or create session_id for this connection
                session_id = data.get("session_id") or current_session_id or "ws-session"
                current_session_id = session_id

                # Ensure session exists and append history entry
                session = session_manager.ensure_session(session_id, user_id=None)
                _apply_language_settings(
                    session,
                    session_id,
                    data.get("preferred_language"),
                    data.get("language"),
                    ws_lang_hint,
                )
                session_manager.add_history_message(session_id, "user", transcript)

                try:
                    agent: VoxBankAgent = app.state.agent
                    logger.info("WS: calling agent.orchestrate() for session %s", session_id)
                    profile = get_session_profile(session_id)
                    user_lang = (profile.get("preferred_language") or "en").lower()
                    logger.info("WS: preferred_language=%s", user_lang)
                    effective_transcript = transcript
                    if user_lang != "en":
                        effective_transcript = await _translate_with_fallback(
                            transcript,
                            user_lang,
                            "en",
                            agent,
                            f"ws_input:{session_id}",
                        )
                    out = await agent.orchestrate(
                        effective_transcript,
                        session_id,
                        user_confirmation=None,
                        session_profile=profile,
                    )
                    logger.info("WS: orchestrate returned status=%s", out.get("status"))
                    logout_triggered = bool(out.get("logged_out"))
                    if logout_triggered:
                        perform_session_logout(session_id)
                        logger.info("Session %s logged out via MCP tool (ws)", session_id)

                    # If auth just completed, persist authenticated user_id into session and hydrate profile
                    auth_user_id = out.get("authenticated_user_id")
                    if auth_user_id:
                        sess = session_manager.ensure_session(session_id, user_id=auth_user_id)
                        session_manager.save_session(session_id, sess)
                        logger.info("Session %s authenticated as user %s (ws)", session_id, auth_user_id)
                        # best-effort: hydrate profile for this session
                        try:
                            await hydrate_session_profile_from_mock_bank(session_id, str(auth_user_id))
                        except Exception as e:
                            logger.exception("WS: failed to hydrate session profile after auth: %s", e)

                    # Normal reply path
                    response_text_en = out.get("response") or out.get("message") or "I couldn't process that request right now."
                    response_text = await _translate_with_fallback(
                        response_text_en,
                        "en",
                        user_lang,
                        agent,
                        f"ws_response:{session_id}",
                    )
                    if not logout_triggered:
                        session_manager.add_history_message(session_id, "assistant", response_text)

                    # Optionally generate server-side TTS audio for this reply
                    if output_audio and response_text:
                        try:
                            audio_bytes = await synthesize_text_to_audio(response_text)
                            if audio_bytes:
                                audio_url = audio_bytes_to_data_url(audio_bytes, mime="audio/wav")
                                await ws.send_text(
                                    json.dumps(
                                        {
                                            "type": "audio_base64",
                                            "audio": audio_url,
                                            "mime": "audio/wav",
                                            "session_id": session_id,
                                        }
                                    )
                                )
                                logger.info(
                                    "WS: sent TTS audio for session %s (len=%d bytes)",
                                    session_id,
                                    len(audio_bytes),
                                )
                        except Exception as e:
                            logger.exception("WS: TTS generation failed: %s", e)

                    reply_payload = {
                        "type": "reply",
                        "text": response_text,
                        "session_id": session_id,
                        "logged_out": logout_triggered,
                    }
                    await ws.send_text(json.dumps(reply_payload))
                    if logout_triggered:
                        await ws.send_text(json.dumps({"type": "logout", "session_id": session_id}))
                        # ensure a clean session on next turn
                        session_manager.ensure_session(session_id, user_id=None)
                except Exception as e:
                    logger.exception("WS: error during orchestrate: %s", e)
                    await ws.send_text(json.dumps({"type": "error", "message": "server_exception", "details": str(e)}))

                continue

            # Unknown message type
            await ws.send_text(json.dumps({"type": "error", "message": f"unknown_type_{msg_type}"}))

    except WebSocketDisconnect:
        logger.info("WS: client disconnected %s", ws.client)
    except Exception as e:
        logger.exception("WS: unexpected error: %s", e)
        try:
            await ws.close()
        except Exception:
            pass

@app.post("/api/text/process", response_model=TextResponse)
async def process_text(request: TextRequest, background_tasks: BackgroundTasks):
    """
    Process a text input (transcript) and return the assistant's response.
    Uses the same orchestration flow as voice but returns text only.
    """
    logger.info("=" * 80)
    logger.info("API Request: POST /api/text/process")
    logger.info("Session ID: %s | User ID: %s", request.session_id, request.user_id)
    logger.info("Transcript: %s", request.transcript)
    logger.info("Reply Style: %s", request.reply_style)
    
    transcript = request.transcript.strip()
    if not transcript:
        logger.warning("Empty transcript received")
        raise HTTPException(status_code=400, detail="transcript is required")

    session_id = request.session_id
    user_id = request.user_id

    # initialize session (only trust user_id when session is brand new)
    existing_session = session_manager.get_session(session_id)
    if existing_session is None:
        session = session_manager.ensure_session(session_id, user_id=user_id)
    else:
        session = session_manager.ensure_session(session_id, user_id=None)
    _apply_language_settings(session, session_id, request.preferred_language, request.language)
    session_manager.add_history_message(session_id, "user", transcript)
    logger.debug("Session state: %s", session)

    try:
        agent: VoxBankAgent = app.state.agent
        logger.info("Calling agent.orchestrate()")

        # Orchestrate (agent should return structure similar to voice flow)
        profile = get_session_profile(session_id)
        user_lang = (profile.get("preferred_language") or "en").lower()
        logger.info("Text process: preferred_language=%s", user_lang)
        text_for_llm = transcript
        if user_lang != "en":
            text_for_llm = await _translate_with_fallback(
                transcript,
                user_lang,
                "en",
                agent,
                f"text_input:{session_id}",
            )

        out = await agent.orchestrate(
            text_for_llm,
            session_id,
            user_confirmation=None,
            session_profile=profile,
        ) if hasattr(agent, "orchestrate") else await agent.process_user_input(text_for_llm, session_id)
        
        logger.info("Agent orchestrate returned: status=%s", out.get("status"))
        logger.debug("Full orchestrate response: %s", out)

        logout_triggered = bool(out.get("logged_out"))
        if logout_triggered:
            perform_session_logout(session_id)
            session = session_manager.ensure_session(session_id, user_id=None)
            profile = get_session_profile(session_id)
            logger.info("Session %s logged out via MCP tool", session_id)

        # If agent asks for confirmation (high-risk)
        if not logout_triggered and out.get("status") == "needs_confirmation":
            logger.info("Action requires confirmation")
            session["pending_action"] = {
                "parsed": out.get("parsed"),
                "transcript": transcript,
                "transcript_en": text_for_llm,
            }
            session_manager.save_session(session_id, session)
            resp_text_en = out.get("message", "Please confirm the action.")
            resp_text = await _translate_with_fallback(
                resp_text_en,
                "en",
                user_lang,
                agent,
                f"text_confirm:{session_id}",
            )
            logger.info("Returning confirmation request: %s", resp_text)
            audio_url = None
            if request.output_audio:
                try:
                    audio_bytes = await synthesize_text_to_audio(resp_text)
                    if audio_bytes:
                        audio_url = audio_bytes_to_data_url(audio_bytes, mime="audio/wav")
                        logger.info("Text TTS (confirm): generated audio (%d bytes) for session %s", len(audio_bytes), session_id)
                except Exception as e:
                    logger.exception("Text TTS (confirm) failed: %s", e)
            session_manager.add_history_message(session_id, "assistant", resp_text)
            return TextResponse(
                response_text=resp_text,
                session_id=session_id,
                requires_confirmation=True,
                audio_url=audio_url,
                meta={"parsed": out.get("parsed")},
                logged_out=logout_triggered,
            )
        
        # If agent needs clarification (missing information or auth/login)
        elif not logout_triggered and out.get("status") == "clarify":
            logger.info("Agent needs clarification from user")
            # Get the clarification message from the agent
            clarify_message_en = out.get(
                "message",
                "I need more information to help you. Could you please provide more details?",
            )
            clarify_message = await _translate_with_fallback(
                clarify_message_en,
                "en",
                user_lang,
                agent,
                f"text_clarify:{session_id}",
            )
            logger.info("Returning clarification request: %s", clarify_message)
            
            # Store the partial parsed intent in session for context (optional, for better continuity)
            if out.get("parsed"):
                session["pending_clarification"] = {
                    "parsed": out.get("parsed"),
                    "transcript": transcript,
                    "tool_name": out.get("parsed", {}).get("tool_name"),
                    "tool_input": out.get("parsed", {}).get("tool_input", {})
                }
                session_manager.save_session(session_id, session)
            
            # Append assistant's clarification question to history
            session["history"].append({"role": "assistant", "text": clarify_message})
            session_manager.save_session(session_id, session)
            
            # Return clarification response (no confirmation needed, just asking for more info)
            audio_url = None
            if request.output_audio:
                try:
                    audio_bytes = await synthesize_text_to_audio(clarify_message)
                    if audio_bytes:
                        audio_url = audio_bytes_to_data_url(audio_bytes, mime="audio/wav")
                        logger.info("Text TTS (clarify): generated audio (%d bytes) for session %s", len(audio_bytes), session_id)
                except Exception as e:
                    logger.exception("Text TTS (clarify) failed: %s", e)

            return TextResponse(
                response_text=clarify_message,
                session_id=session_id,
                requires_confirmation=False,
                audio_url=audio_url,
                meta={"parsed": out.get("parsed"), "status": "clarify"},
                logged_out=logout_triggered,
            )
        
        # If auth just completed, persist authenticated user_id into session
        auth_user_id = out.get("authenticated_user_id")
        if auth_user_id:
            session["user_id"] = auth_user_id
            logger.info("Session %s authenticated as user %s (text)", session_id, auth_user_id)
            try:
                await hydrate_session_profile_from_mock_bank(session_id, str(auth_user_id))
            except Exception as e:
                logger.exception("Text: failed to hydrate session profile after auth: %s", e)

        # Normal response path
        response_text_en = out.get("response")
        if not response_text_en:
            logger.error("Agent returned None response_text. Full output: %s", out)
            response_text_en = "I apologize, but I'm having trouble processing your request. Please try again."
        response_text = await _translate_with_fallback(
            response_text_en,
            "en",
            user_lang,
            agent,
            f"text_response:{session_id}",
        )

        audio_url = None
        if request.output_audio:
            try:
                audio_bytes = await synthesize_text_to_audio(response_text)
                if audio_bytes:
                    audio_url = audio_bytes_to_data_url(audio_bytes, mime="audio/wav")
                    logger.info("Text TTS: generated audio (%d bytes) for session %s", len(audio_bytes), session_id)
            except Exception as e:
                logger.exception("Text TTS failed: %s", e)

        if not logout_triggered:
            session_manager.add_history_message(session_id, "assistant", response_text)

        response_meta: Dict[str, Any] = {}
        if logout_triggered:
            response_meta["logged_out"] = True
        if out.get("tool_result") is not None:
            response_meta.setdefault("tool_result", out.get("tool_result"))
        logger.info("Returning successful response: %s", response_text[:100] + "..." if len(response_text) > 100 else response_text)
        logger.info("=" * 80)
        return TextResponse(
            response_text=response_text,
            session_id=session_id,
            requires_confirmation=False,
            audio_url=audio_url,
            meta=response_meta or None,
            logged_out=logout_triggered,
        )

    except Exception as e:
        logger.exception("Error in process_text: %s", e)
        logger.error("Request details - Session: %s, User: %s, Transcript: %s", session_id, user_id, transcript)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/text/intent")
async def parse_intent(req: TextRequest):
    """
    Intent extraction / parse endpoint.
    Returns intent, entities, and confidence without executing tools.
    Useful for unit tests and UI debugging.
    """
    logger.info("API Request: POST /api/text/intent | Session: %s | Transcript: %s", req.session_id, req.transcript)
    
    transcript = req.transcript.strip()
    if not transcript:
        logger.warning("Empty transcript in parse_intent")
        raise HTTPException(status_code=400, detail="transcript is required")

    agent: VoxBankAgent = app.state.agent

    try:
        # If agent exposes a direct intent function, prefer it
        if hasattr(agent, "process_user_input"):
            logger.debug("Using agent.process_user_input()")
            parsed = agent.process_user_input(transcript, req.session_id)
            # if process_user_input is sync, allow for coroutine
            if asyncio.iscoroutine(parsed):
                parsed = await parsed
            logger.info("Intent parsed: %s", parsed.get("intent"))
            return {"parsed": parsed}

        # Fallback: call orchestrate in "parse-only" mode if supported
        if hasattr(agent, "orchestrate"):
            logger.debug("Using agent.orchestrate(parse_only=True)")
            profile = get_session_profile(req.session_id)
            out = await agent.orchestrate(
                transcript,
                req.session_id,
                user_confirmation=None,
                parse_only=True,
                session_profile=profile,
            )
            parsed = out.get("parsed") or out
            logger.info("Intent parsed via orchestrate: %s", parsed.get("intent") if isinstance(parsed, dict) else "unknown")
            return {"parsed": parsed}

        # Last resort — return a simple echo-based parse
        logger.warning("No intent extractor available on agent")
        return {"parsed": {"intent": "unknown", "entities": {}, "confidence": 0.0, "note": "No intent extractor available on agent"}}
    except Exception as e:
        logger.exception("Error in parse_intent: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/text/respond", response_model=TextResponse)
async def text_respond(req: TextRequest):
    """
    Simpler endpoint: directly ask the LLM client for a reply (useful for fast dev iterations).
    This bypasses orchestration and tools — DO NOT use for actions (transfers).
    """
    logger.info("API Request: POST /api/text/respond | Session: %s | Transcript: %s", req.session_id, req.transcript)
    
    transcript = req.transcript.strip()
    if not transcript:
        logger.warning("Empty transcript in text_respond")
        raise HTTPException(status_code=400, detail="transcript is required")

    # use LLM client directly to get a text reply
    llm: GeminiLLMClient = app.state.gemini_client
    try:
        # Some LLM clients accept reply options; we pass a short system prompt to steer style
        system_prompt = "You are VoxBank assistant. Answer concisely and safely."
        user_prompt = f"{system_prompt}\nUser: {transcript}\nAssistant:"
        logger.debug("LLM prompt: %s", user_prompt)
        response_text = await llm.generate(user_prompt, max_tokens=256)
        logger.info("LLM response received: %s", response_text[:100] + "..." if len(response_text) > 100 else response_text)
        
        # update session history for continuity
        session_id = req.session_id
        session = session_manager.ensure_session(session_id, user_id=req.user_id)
        session["history"].append({"role": "assistant", "text": response_text})
        return TextResponse(
            response_text=response_text,
            session_id=session_id,
            requires_confirmation=False,
            logged_out=False,
        )
    except Exception as e:
        logger.exception("LLM generate failed: %s", e)
        raise HTTPException(status_code=500, detail="LLM error")

@app.post("/api/voice/process", response_model=VoiceResponse)
async def process_voice(request: VoiceRequest, background_tasks: BackgroundTasks):
    """
    Process voice input.

    - If `transcript` is provided, use it directly.
    - Else if `audio_data` (base64) is provided, run STT to obtain a transcript.
    - Run the LLM agent orchestration on the transcript.
    - Optionally run TTS on the assistant reply and return an audio data URL.

    NOTE: STT/TTS are currently stubbed in voice_processing.py and should be
    replaced with real engines for production use.
    """
    logger.info("=" * 80)
    logger.info("API Request: POST /api/voice/process")
    logger.info("Session ID: %s | User ID: %s", request.session_id, request.user_id)
    session_id = request.session_id
    user_id = request.user_id

    transcript: Optional[str] = None

    # 1) Prefer explicit transcript if provided
    if request.transcript and request.transcript.strip():
        transcript = request.transcript.strip()
        logger.info("Voice: using provided transcript: %s", transcript)
    # 2) Otherwise, try STT on provided audio_data
    elif request.audio_data:
        logger.info("Voice: no transcript provided, attempting STT on audio_data")
        try:
            import base64

            audio_bytes = base64.b64decode(request.audio_data)
        except Exception as e:
            logger.exception("Failed to decode base64 audio_data: %s", e)
            raise HTTPException(status_code=400, detail="invalid audio_data (base64 decode failed)")

        transcript = await transcribe_audio_to_text(audio_bytes)
        logger.info("Voice STT result: '%s'", transcript)
    else:
        logger.warning("Voice: neither transcript nor audio_data provided")
        raise HTTPException(status_code=400, detail="Either transcript or audio_data must be provided")

    if not transcript:
        logger.warning("Voice: STT returned empty transcript")
        raise HTTPException(status_code=400, detail="Could not derive transcript from audio")

    existing_session = session_manager.get_session(session_id)
    if existing_session is None:
        session = session_manager.ensure_session(session_id, user_id=user_id)
    else:
        session = session_manager.ensure_session(session_id, user_id=None)
    _apply_language_settings(session, session_id, request.preferred_language, request.language)
    session["history"].append({"role": "user", "text": transcript})
    session_manager.save_session(session_id, session)
    logger.debug("Session state: %s", session)

    try:
        agent: VoxBankAgent = app.state.agent
        logger.info("Calling agent.orchestrate() for voice")
        profile = get_session_profile(session_id)
        user_lang = (profile.get("preferred_language") or "en").lower()
        logger.info("Voice process: preferred_language=%s", user_lang)
        text_for_llm = transcript
        if user_lang != "en":
            text_for_llm = await _translate_with_fallback(
                transcript,
                user_lang,
                "en",
                agent,
                f"voice_input:{session_id}",
            )
        # Orchestrate: this will return needs_confirmation if LLM asks for it
        out = await agent.orchestrate(
            text_for_llm,
            session_id,
            user_confirmation=None,
            session_profile=profile,
        )
        logger.info("Agent orchestrate returned: status=%s", out.get("status"))
        logger.debug("Full orchestrate response: %s", out)

        logout_triggered = bool(out.get("logged_out"))
        if logout_triggered:
            perform_session_logout(session_id)
            session = session_manager.ensure_session(session_id, user_id=None)
            profile = get_session_profile(session_id)
            logger.info("Session %s logged out via MCP tool (voice)", session_id)

        # If agent returns needs_confirmation, store pending_action in session
        if out.get("status") == "needs_confirmation":
            logger.info("Action requires confirmation")
            session["pending_action"] = {
                "parsed": out.get("parsed"),
                "transcript": transcript,
                "transcript_en": text_for_llm,
            }
            session_manager.save_session(session_id, session)
            resp_text_en = out.get("message", "Please confirm the action.")
            resp_text = await _translate_with_fallback(
                resp_text_en,
                "en",
                user_lang,
                agent,
                f"voice_confirm:{session_id}",
            )
            logger.info("Returning confirmation request: %s", resp_text)
            if not logout_triggered:
                session["history"].append({"role": "assistant", "text": resp_text})
                session_manager.save_session(session_id, session)
            meta_payload: Dict[str, Any] = {"parsed": out.get("parsed")}
            if logout_triggered:
                meta_payload["logged_out"] = True
            return VoiceResponse(
                response_text=resp_text,
                session_id=session_id,
                requires_confirmation=True,
                meta=meta_payload,
                logged_out=logout_triggered,
            )

        # If agent needs clarification (missing information or auth/login)
        elif out.get("status") == "clarify":
            logger.info("Agent needs clarification from user")
            # Get the clarification message from the agent
            clarify_message_en = out.get(
                "message",
                "I need more information to help you. Could you please provide more details?",
            )
            clarify_message = await _translate_with_fallback(
                clarify_message_en,
                "en",
                user_lang,
                agent,
                f"voice_clarify:{session_id}",
            )
            logger.info("Returning clarification request: %s", clarify_message)
            
            # Store the partial parsed intent in session for context (optional, for better continuity)
            if out.get("parsed"):
                session["pending_clarification"] = {
                    "parsed": out.get("parsed"),
                    "transcript": transcript,
                    "tool_name": out.get("parsed", {}).get("tool_name"),
                    "tool_input": out.get("parsed", {}).get("tool_input", {})
                }
                session_manager.save_session(session_id, session)
            
            # Append assistant's clarification question to history
            if not logout_triggered:
                session["history"].append({"role": "assistant", "text": clarify_message})
            if not logout_triggered:
                session_manager.save_session(session_id, session)
            
            # Return clarification response (no confirmation needed, just asking for more info)
            meta_payload = {"parsed": out.get("parsed"), "status": "clarify"}
            if logout_triggered:
                meta_payload["logged_out"] = True
            return VoiceResponse(
                response_text=clarify_message,
                session_id=session_id,
                requires_confirmation=False,
                meta=meta_payload,
                logged_out=logout_triggered,
            )

        # If auth just completed, persist authenticated user_id into session and hydrate profile
        auth_user_id = out.get("authenticated_user_id")
        if auth_user_id:
            session["user_id"] = auth_user_id
            logger.info("Session %s authenticated as user %s (voice)", session_id, auth_user_id)
            try:
                await hydrate_session_profile_from_mock_bank(session_id, str(auth_user_id))
            except Exception as e:
                logger.exception("Voice: failed to hydrate session profile after auth: %s", e)

        # Normal completed response
        response_text_en = out.get("response")
        if not response_text_en:
            logger.error("Agent returned None response_text. Full output: %s", out)
            response_text_en = "I apologize, but I'm having trouble processing your request. Please try again."
        response_text = await _translate_with_fallback(
            response_text_en,
            "en",
            user_lang,
            agent,
            f"voice_response:{session_id}",
        )

        # Run TTS to generate audio for voice clients (stub for now)
        audio_url: Optional[str] = None
        try:
            audio_bytes = await synthesize_text_to_audio(response_text)
            if audio_bytes:
                audio_url = audio_bytes_to_data_url(audio_bytes, mime="audio/wav")
                logger.info("Voice TTS: generated audio (%d bytes) for session %s", len(audio_bytes), session_id)
        except Exception as e:
            logger.exception("Voice TTS failed: %s", e)

        response_meta: Dict[str, Any] = {}
        if out.get("tool_result") is not None:
            response_meta["tool_result"] = out.get("tool_result")
        if logout_triggered:
            response_meta["logged_out"] = True
        if not logout_triggered:
            session["history"].append({"role": "assistant", "text": response_text})
            session_manager.save_session(session_id, session)
        logger.info("Returning successful response: %s", response_text[:100] + "..." if len(response_text) > 100 else response_text)
        logger.info("=" * 80)
        # Optionally run background tasks such as storing audit logs (placeholder)
        # background_tasks.add_task(store_audit, user_id, session_id, transcript, out)

        return VoiceResponse(
            response_text=response_text,
            audio_url=audio_url,
            session_id=session_id,
            requires_confirmation=False,
            meta=response_meta or None,
        )

    except Exception as e:
        logger.exception("Error in process_voice: %s", e)
        logger.error("Request details - Session: %s, User: %s, Transcript: %s", session_id, user_id, transcript)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/voice/confirm", response_model=VoiceResponse)
async def confirm_action(request: ConfirmRequest):
    """
    Confirm or reject a previously requested high-risk action.
    - Expects a session with a pending_action stored by /api/voice/process
    - If confirm=True, the agent will re-run orchestration with user_confirmation=True
    """
    session_id = request.session_id
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    pending = session.get("pending_action")
    if not pending:
        raise HTTPException(status_code=400, detail="No pending action to confirm")

    # If user rejects
    if not request.confirm:
        session["pending_action"] = None
        deny_text = "Okay, I have cancelled that action."
        session["history"].append({"role": "assistant", "text": deny_text})
        return VoiceResponse(
            response_text=deny_text,
            session_id=session_id,
            requires_confirmation=False,
            logged_out=False,
        )

    # User confirmed: re-run orchestration with confirmation flag
    try:
        agent: VoxBankAgent = app.state.agent
        transcript = pending.get("transcript_en") or pending.get("transcript")
        profile = get_session_profile(session_id)
        user_lang = (profile.get("preferred_language") or "en").lower()
        # orchestrate with user_confirmation=True to perform the action
        out = await agent.orchestrate(transcript, session_id, user_confirmation=True, session_profile=profile)

        # clear pending action
        session["pending_action"] = None

        logout_triggered = bool(out.get("logged_out"))
        if logout_triggered:
            perform_session_logout(session_id)
            session = session_manager.ensure_session(session_id, user_id=None)

        # append assistant message
        response_text_en = out.get("response") or ""
        response_text = await _translate_with_fallback(
            response_text_en,
            "en",
            user_lang,
            agent,
            f"voice_confirm_final:{session_id}",
        )
        if not logout_triggered:
            session["history"].append({"role": "assistant", "text": response_text})
            session_manager.save_session(session_id, session)

        meta_payload: Dict[str, Any] = {}
        if out.get("tool_result") is not None:
            meta_payload["tool_result"] = out.get("tool_result")
        if logout_triggered:
            meta_payload["logged_out"] = True
        return VoiceResponse(
            response_text=response_text,
            session_id=session_id,
            requires_confirmation=False,
            meta=meta_payload or None,
            logged_out=logout_triggered,
        )
    except Exception as e:
        logger.exception("Error confirming action: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


# Optional endpoint to fetch session history (debug)
@app.get("/api/session/{session_id}/history")
async def session_history(session_id: str):
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"session_id": session_id, "history": session.get("history", []), "pending_action": session.get("pending_action")}
