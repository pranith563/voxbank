"""
orchestrator/src/app.py

FastAPI application for VoxBank Orchestrator
Main entry point for LLM orchestration and conversation engine
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any
from dotenv import load_dotenv
import os
import logging
import asyncio
import httpx

# Load environment
load_dotenv()

# Logging
logger = logging.getLogger("voxbank.orchestrator")
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))

# Local imports (make sure these modules exist in this package)
# from orchestrator.src.llm_agent import LLMAgent
# from orchestrator.src.gemini_llm_client import GeminiLLMClient
# from orchestrator.src.mcp_client import MCPClient  # optional when you build it
#
# If running as a package, adjust imports accordingly. The below assumes app.py is in the same folder
# as llm_agent.py and gemini_llm_client.py. If not, adapt import paths.

try:
    from llm_agent import LLMAgent
    from gemini_llm_client import GeminiLLMClient
except Exception as e:
    # fallback relative imports if run as module
    from .llm_agent import LLMAgent  # type: ignore
    from .gemini_llm_client import GeminiLLMClient  # type: ignore

# Simple MCP HTTP client stub (replace with FastMCP client later)
class MCPHttpClient:
    """
    Minimal MCP client that calls tool endpoints over HTTP.
    Expects MCP_TOOL_BASE_URL environment var like: http://localhost:9000
    Tool endpoints expected: {MCP_TOOL_BASE_URL}/{tool_name}
    """
    def __init__(self, base_url: Optional[str] = None, timeout: int = 10):
        self.base_url = base_url or os.getenv("MCP_TOOL_BASE_URL", "http://localhost:9000")
        self.timeout = timeout
        self._client = httpx.AsyncClient(timeout=self.timeout)

    async def execute(self, tool_name: str, payload: Dict[str, Any], session_id: str = "") -> Dict[str, Any]:
        url = f"{self.base_url.rstrip('/')}/{tool_name}"
        try:
            resp = await self._client.post(url, json=payload)
            if resp.status_code == 200:
                return resp.json()
            else:
                logger.warning("MCP tool %s returned status %s: %s", tool_name, resp.status_code, resp.text)
                return {"status": "error", "message": f"MCP call returned {resp.status_code}"}
        except Exception as e:
            logger.exception("Error calling MCP tool %s: %s", tool_name, e)
            return {"status": "error", "message": str(e)}

    async def close(self):
        await self._client.aclose()


# Pydantic models
class VoiceRequest(BaseModel):
    audio_data: Optional[str] = None  # base64 placeholder (ASR handled elsewhere)
    transcript: Optional[str] = None
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

# In-memory session store (for demo/prototype). For prod, use Redis or DB.
SESSIONS: Dict[str, Dict[str, Any]] = {}

# Instantiate clients on startup
@app.on_event("startup")
async def startup_event():
    # Load GEMINI settings
    gemini_key = os.getenv("GEMINI_API_KEY")
    gemini_model = os.getenv("GEMINI_MODEL", "gemini-2.5-pro")
    if not gemini_key:
        logger.error("GEMINI_API_KEY not set. LLM generation will not work until key is provided.")
    # Create clients and agent singletons
    app.state.gemini_client = GeminiLLMClient(api_key=gemini_key, model=gemini_model)
    app.state.mcp_client = MCPHttpClient(base_url=os.getenv("MCP_TOOL_BASE_URL"))
    app.state.agent = LLMAgent(model_name=gemini_model, llm_client=app.state.gemini_client, mcp_client=app.state.mcp_client)
    logger.info("Orchestrator started. Gemini model=%s MCP base=%s", gemini_model, app.state.mcp_client.base_url)


@app.on_event("shutdown")
async def shutdown_event():
    # close httpx client
    mcp = getattr(app.state, "mcp_client", None)
    if mcp:
        await mcp.close()
    logger.info("Orchestrator shutdown.")


@app.get("/")
async def root():
    return {"message": "VoxBank Orchestrator API", "status": "running"}


@app.get("/api/health")
async def health_check():
    return {"status": "healthy"}


@app.post("/api/voice/process", response_model=VoiceResponse)
async def process_voice(request: VoiceRequest, background_tasks: BackgroundTasks):
    """
    Process voice input (transcript required for prototype).
    - If `transcript` is provided, orchestrate immediately.
    - If high-risk action requires confirmation, the response will indicate `requires_confirmation=True`.
    """
    if not request.transcript or request.transcript.strip() == "":
        raise HTTPException(status_code=400, detail="transcript is required in this prototype")

    session_id = request.session_id
    user_id = request.user_id
    transcript = request.transcript.strip()

    # initialize session if missing
    session = SESSIONS.setdefault(session_id, {"user_id": user_id, "history": [], "pending_action": None})
    session["history"].append({"role": "user", "text": transcript})

    try:
        agent: LLMAgent = app.state.agent
        # Orchestrate: this will return needs_confirmation if LLM asks for it
        out = await agent.orchestrate(transcript, session_id, user_confirmation=None)

        # If agent returns needs_confirmation, store pending_action in session
        if out.get("status") == "needs_confirmation":
            session["pending_action"] = {"parsed": out.get("parsed"), "transcript": transcript}
            resp_text = out.get("message", "Please confirm the action.")
            return VoiceResponse(response_text=resp_text, session_id=session_id, requires_confirmation=True, meta={"parsed": out.get("parsed")})

        # Normal completed response
        session["history"].append({"role": "assistant", "text": out.get("response")})
        # Optionally run background tasks such as storing audit logs (placeholder)
        # background_tasks.add_task(store_audit, user_id, session_id, transcript, out)

        return VoiceResponse(response_text=out.get("response"), session_id=session_id, requires_confirmation=False, meta={"tool_result": out.get("tool_result")})

    except Exception as e:
        logger.exception("Error in process_voice: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/voice/confirm", response_model=VoiceResponse)
async def confirm_action(request: ConfirmRequest):
    """
    Confirm or reject a previously requested high-risk action.
    - Expects a session with a pending_action stored by /api/voice/process
    - If confirm=True, the agent will re-run orchestration with user_confirmation=True
    """
    session_id = request.session_id
    session = SESSIONS.get(session_id)
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
    session = SESSIONS.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"session_id": session_id, "history": session.get("history", []), "pending_action": session.get("pending_action")}