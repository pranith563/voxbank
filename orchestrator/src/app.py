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

from llm_agent import LLMAgent
from gemini_llm_client import GeminiLLMClient
from clients.mcp_client import MCPClient
from context.session_manager import SessionManager
from voice_processing import (
    transcribe_audio_to_text,
    synthesize_text_to_audio,
    audio_bytes_to_data_url,
    extract_voice_embedding,
)



# Pydantic models
# for text endpoints
class TextRequest(BaseModel):
    transcript: str
    session_id: str
    user_id: str
    # optional flag to request a short/long reply
    reply_style: Optional[str] = "concise"  # 'concise' | 'detailed'

class TextResponse(BaseModel):
    response_text: str
    session_id: str
    requires_confirmation: bool = False
    meta: Optional[Dict[str, Any]] = None

class VoiceRequest(BaseModel):
    audio_data: Optional[str] = None  # base64-encoded audio (optional)
    transcript: Optional[str] = None  # optional pre-transcribed text
    session_id: str
    user_id: str

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


class RegisterRequest(BaseModel):
    username: str
    passphrase: str
    email: Optional[str] = None
    full_name: Optional[str] = None
    phone_number: Optional[str] = None
    address: Optional[str] = None
    date_of_birth: Optional[str] = None  # ISO date string (not used by mock-bank today)
    audio_data: Optional[str] = None  # base64-encoded audio for embedding


class RegisterResponse(BaseModel):
    status: str
    user: Dict[str, Any]


class LogoutRequest(BaseModel):
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

# Session management
# In-memory session manager (for demo/prototype). For prod, use Redis or DB.
session_manager = SessionManager(
    session_timeout_minutes=int(os.getenv("SESSION_TIMEOUT_MINUTES", "30"))
)

# Backwards-compat alias so any code that still reads SESSIONS
# sees the same underlying dictionary.
SESSIONS: Dict[str, Dict[str, Any]] = session_manager.sessions

# Instantiate clients on startup
@app.on_event("startup")
async def startup_event():
    """
    Startup wiring:
     - Initialize Gemini LLM client
     - Initialize MCP client (discovery)
     - Instantiate the LLMAgent with the clients
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

    # Instantiate LLMAgent with dynamic tool_spec (falls back internally if empty)
    app.state.agent = LLMAgent(
        model_name=gemini_model,
        llm_client=app.state.gemini_client,
        mcp_client=app.state.mcp_client,
        tool_spec=tool_spec or None,
    )

    # Log discovered tools for observability
    if tool_spec:
        logger.info("Startup: loaded %d tools from MCP list_tools:", len(tool_spec))
        for name, meta in tool_spec.items():
            logger.info(" - %s: %s", name, meta.get("description", ""))
    else:
        logger.warning("Startup: no tool metadata loaded from MCP; LLMAgent will use fallback tool spec")

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


@app.post("/api/auth/logout")
async def logout(req: LogoutRequest):
    """
    Clear authentication/session state for a given session_id.
    Used by the frontend when the user clicks Logout.
    """
    session_id = req.session_id
    logger.info("API Request: POST /api/auth/logout | session_id=%s", session_id)

    # Clear orchestrator session
    sess = session_manager.get_session(session_id)
    if sess:
        sess["user_id"] = None
        sess["pending_action"] = None
        sess["pending_clarification"] = None

    # Clear agent auth state
    agent: LLMAgent = app.state.agent
    if session_id in agent.auth_state:
        try:
            del agent.auth_state[session_id]
        except Exception:
            agent.auth_state[session_id] = {
                "authenticated": False,
                "user_id": None,
                "flow_stage": None,
                "temp": {},
            }

    return {"status": "ok"}


@app.post("/api/auth/register", response_model=RegisterResponse)
async def register_user(request: RegisterRequest):
    """
    Register a new VoxBank user.

    - Optionally accepts `audio_data` (base64) and extracts a voice embedding.
    - Calls MCP tool `register_user` to create the user in mock-bank.
    """
    logger.info("API Request: POST /api/auth/register | username=%s", request.username)

    # Prepare audio embedding if audio_data is provided
    audio_embedding = None
    if request.audio_data:
        try:
            import base64

            audio_bytes = base64.b64decode(request.audio_data)
            audio_embedding = await extract_voice_embedding(audio_bytes)
            logger.info(
                "Register: extracted voice embedding (len=%d) for username=%s",
                len(audio_embedding or []),
                request.username,
            )
        except Exception as e:
            logger.exception("Register: failed to extract voice embedding: %s", e)

    # Build payload for MCP register_user tool
    payload: Dict[str, Any] = {
        "username": request.username,
        "passphrase": request.passphrase,
        "email": request.email,
        "full_name": request.full_name,
        "phone_number": request.phone_number,
        "audio_embedding": audio_embedding,
    }

    try:
        mcp_client: MCPClient = app.state.mcp_client
        logger.info("Register: calling MCP register_user for username=%s", request.username)
        result = await mcp_client.call_tool("register_user", payload)
        logger.info("Register: MCP register_user result status=%s", result.get("status"))

        if result.get("status") != "success":
            msg = result.get("message") or "Registration failed"
            raise HTTPException(status_code=400, detail=msg)

        user = result.get("user") or {}
        return RegisterResponse(status="success", user=user)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error in /api/auth/register: %s", e)
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
    routes them through LLMAgent.orchestrate(), returning a single "reply"
    message for each transcript.
    """
    await ws.accept()
    logger.info("WS: connection opened from %s", ws.client)

    try:
        # Optional: track a session_id per connection
        current_session_id: Optional[str] = None

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

                # Resolve or create session_id for this connection
                session_id = data.get("session_id") or current_session_id or "ws-session"
                current_session_id = session_id

                # Ensure session exists and append history entry
                session = session_manager.ensure_session(session_id, user_id=None)
                session["history"].append({"role": "user", "text": transcript})

                try:
                    agent: LLMAgent = app.state.agent
                    logger.info("WS: calling agent.orchestrate() for session %s", session_id)
                    out = await agent.orchestrate(transcript, session_id, user_confirmation=None)
                    logger.info("WS: orchestrate returned status=%s", out.get("status"))

                    # If auth just completed, persist authenticated user_id into session
                    auth_user_id = out.get("authenticated_user_id")
                    if auth_user_id:
                        session["user_id"] = auth_user_id
                        logger.info("Session %s authenticated as user %s (ws)", session_id, auth_user_id)

                    # Normal reply path
                    response_text = out.get("response") or out.get("message") or "I couldn't process that request right now."
                    session["history"].append({"role": "assistant", "text": response_text})

                    await ws.send_text(json.dumps({"type": "reply", "text": response_text, "session_id": session_id}))
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

    # initialize session if missing
    session = session_manager.ensure_session(session_id, user_id=user_id)
    session["history"].append({"role": "user", "text": transcript})
    logger.debug("Session state: %s", session)

    try:
        agent: LLMAgent = app.state.agent
        logger.info("Calling agent.orchestrate()")

        # Orchestrate (agent should return structure similar to voice flow)
        out = await agent.orchestrate(transcript, session_id, user_confirmation=None) \
              if hasattr(agent, "orchestrate") else await agent.process_user_input(transcript, session_id)
        
        logger.info("Agent orchestrate returned: status=%s", out.get("status"))
        logger.debug("Full orchestrate response: %s", out)

        # If agent asks for confirmation (high-risk)
        if out.get("status") == "needs_confirmation":
            logger.info("Action requires confirmation")
            session["pending_action"] = {"parsed": out.get("parsed"), "transcript": transcript}
            resp_text = out.get("message", "Please confirm the action.")
            logger.info("Returning confirmation request: %s", resp_text)
            return TextResponse(response_text=resp_text, session_id=session_id, requires_confirmation=True, meta={"parsed": out.get("parsed")})
        
        # If agent needs clarification (missing information or auth/login)
        elif out.get("status") == "clarify":
            logger.info("Agent needs clarification from user")
            # Get the clarification message from the agent
            clarify_message = out.get("message", "I need more information to help you. Could you please provide more details?")
            logger.info("Returning clarification request: %s", clarify_message)
            
            # Store the partial parsed intent in session for context (optional, for better continuity)
            if out.get("parsed"):
                session["pending_clarification"] = {
                    "parsed": out.get("parsed"),
                    "transcript": transcript,
                    "tool_name": out.get("parsed", {}).get("tool_name"),
                    "tool_input": out.get("parsed", {}).get("tool_input", {})
                }
            
            # Append assistant's clarification question to history
            session["history"].append({"role": "assistant", "text": clarify_message})
            
            # Return clarification response (no confirmation needed, just asking for more info)
            return TextResponse(
                response_text=clarify_message, 
                session_id=session_id, 
                requires_confirmation=False, 
                meta={"parsed": out.get("parsed"), "status": "clarify"}
            )
        
        # If auth just completed, persist authenticated user_id into session
        auth_user_id = out.get("authenticated_user_id")
        if auth_user_id:
            session["user_id"] = auth_user_id
            logger.info("Session %s authenticated as user %s (text)", session_id, auth_user_id)

        # Normal response path
        response_text = out.get("response")
        if not response_text:
            logger.error("Agent returned None response_text. Full output: %s", out)
            response_text = "I apologize, but I'm having trouble processing your request. Please try again."
        
        session["history"].append({"role": "assistant", "text": response_text})
        logger.info("Returning successful response: %s", response_text[:100] + "..." if len(response_text) > 100 else response_text)
        logger.info("=" * 80)
        return TextResponse(response_text=response_text, session_id=session_id, requires_confirmation=False)

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

    agent: LLMAgent = app.state.agent

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
            out = await agent.orchestrate(transcript, req.session_id, user_confirmation=None, parse_only=True)
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
        return TextResponse(response_text=response_text, session_id=session_id, requires_confirmation=False)
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

    # initialize session if missing
    session = session_manager.ensure_session(session_id, user_id=user_id)
    session["history"].append({"role": "user", "text": transcript})
    logger.debug("Session state: %s", session)

    try:
        agent: LLMAgent = app.state.agent
        logger.info("Calling agent.orchestrate() for voice")
        # Orchestrate: this will return needs_confirmation if LLM asks for it
        out = await agent.orchestrate(transcript, session_id, user_confirmation=None)
        logger.info("Agent orchestrate returned: status=%s", out.get("status"))
        logger.debug("Full orchestrate response: %s", out)

        # If agent returns needs_confirmation, store pending_action in session
        if out.get("status") == "needs_confirmation":
            logger.info("Action requires confirmation")
            session["pending_action"] = {"parsed": out.get("parsed"), "transcript": transcript}
            resp_text = out.get("message", "Please confirm the action.")
            logger.info("Returning confirmation request: %s", resp_text)
            return VoiceResponse(response_text=resp_text, session_id=session_id, requires_confirmation=True, meta={"parsed": out.get("parsed")})

        # If agent needs clarification (missing information or auth/login)
        elif out.get("status") == "clarify":
            logger.info("Agent needs clarification from user")
            # Get the clarification message from the agent
            clarify_message = out.get("message", "I need more information to help you. Could you please provide more details?")
            logger.info("Returning clarification request: %s", clarify_message)
            
            # Store the partial parsed intent in session for context (optional, for better continuity)
            if out.get("parsed"):
                session["pending_clarification"] = {
                    "parsed": out.get("parsed"),
                    "transcript": transcript,
                    "tool_name": out.get("parsed", {}).get("tool_name"),
                    "tool_input": out.get("parsed", {}).get("tool_input", {})
                }
            
            # Append assistant's clarification question to history
            session["history"].append({"role": "assistant", "text": clarify_message})
            
            # Return clarification response (no confirmation needed, just asking for more info)
            return VoiceResponse(
                response_text=clarify_message, 
                session_id=session_id, 
                requires_confirmation=False, 
                meta={"parsed": out.get("parsed"), "status": "clarify"}
            )

        # If auth just completed, persist authenticated user_id into session
        auth_user_id = out.get("authenticated_user_id")
        if auth_user_id:
            session["user_id"] = auth_user_id
            logger.info("Session %s authenticated as user %s (voice)", session_id, auth_user_id)

        # Normal completed response
        response_text = out.get("response")
        if not response_text:
            logger.error("Agent returned None response_text. Full output: %s", out)
            response_text = "I apologize, but I'm having trouble processing your request. Please try again."

        # Run TTS to generate audio for voice clients (stub for now)
        audio_url: Optional[str] = None
        try:
            audio_bytes = await synthesize_text_to_audio(response_text)
            if audio_bytes:
                audio_url = audio_bytes_to_data_url(audio_bytes, mime="audio/wav")
                logger.info("Voice TTS: generated audio (%d bytes) for session %s", len(audio_bytes), session_id)
        except Exception as e:
            logger.exception("Voice TTS failed: %s", e)

        session["history"].append({"role": "assistant", "text": response_text})
        logger.info("Returning successful response: %s", response_text[:100] + "..." if len(response_text) > 100 else response_text)
        logger.info("=" * 80)
        # Optionally run background tasks such as storing audit logs (placeholder)
        # background_tasks.add_task(store_audit, user_id, session_id, transcript, out)

        return VoiceResponse(
            response_text=response_text,
            audio_url=audio_url,
            session_id=session_id,
            requires_confirmation=False,
            meta={"tool_result": out.get("tool_result")},
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
        return VoiceResponse(response_text=deny_text, session_id=session_id, requires_confirmation=False)

    # User confirmed: re-run orchestration with confirmation flag
    try:
        agent: LLMAgent = app.state.agent
        transcript = pending.get("transcript")
        # orchestrate with user_confirmation=True to perform the action
        out = await agent.orchestrate(transcript, session_id, user_confirmation=True)

        # clear pending action
        session["pending_action"] = None

        # append assistant message
        session["history"].append({"role": "assistant", "text": out.get("response")})

        return VoiceResponse(response_text=out.get("response"), session_id=session_id, requires_confirmation=False, meta={"tool_result": out.get("tool_result")})
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
