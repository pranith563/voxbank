"""
orchestrator/src/app.py

FastAPI application for VoxBank Orchestrator
Main entry point for LLM orchestration and conversation engine
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any
from dotenv import load_dotenv, find_dotenv
import os
import logging
import asyncio
import httpx
import json
# Load environment
load_dotenv(find_dotenv(), override=False)

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
    initialize_tts_backends,
)
from agent.helpers import translate_text
from context.voice_profile import merge_voice_profile, get_voice_profile

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

    # Warm up TTS backends so first request doesn't pay cold-start cost
    try:
        tts_status = await initialize_tts_backends()
        logger.info("TTS backends initialized: %s", tts_status)
    except Exception as exc:
        logger.exception("Failed to initialize TTS backends: %s", exc)

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
                if isinstance(data.get("voice_profile"), dict):
                    session["voice_profile"] = merge_voice_profile(data["voice_profile"])
                    session_manager.save_session(session_id, session)
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
                    voice_profile = get_voice_profile(profile)
                    reply_style = voice_profile.get("style", "warm")
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
                            audio_bytes = await synthesize_text_to_audio(
                                response_text,
                                session_profile=profile,
                                reply_style=reply_style,
                            )
                            if audio_bytes:
                                audio_url = audio_bytes_to_data_url(audio_bytes, mime="audio/wav")
                                await ws.send_text(
                                    json.dumps(
                                        {
                                            "type": "audio_base64",
                                            "audio": audio_url,
                                            "mime": "audio/wav",
                                            "session_id": session_id,
                                            "voice_profile": voice_profile,
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
