# Orchestrator – LLM Agent & Conversation Engine

The orchestrator is a FastAPI service that hosts VoxBank’s LLM-based assistant. It manages sessions, login/registration gating, ReAct-style tool calling via MCP, and voice/WebSocket entry points.

## Overview

- **LLMAgent (`orchestrator/src/llm_agent.py`)**
  - ReAct loop: decides `respond` / `call_tool` / `ask_user` / `ask_confirmation`.
  - Uses a Gemini-based `GeminiLLMClient` for both decision and final reply polishing.
  - Maintains per-session conversation history and a small auth state.
  - Gates account tools (`balance`, `transactions`, `transfer`) behind login.
  - Dynamically injects MCP tool metadata into prompts (no hard-coded tool list).

- **App (`orchestrator/src/app.py`)**
  - REST endpoints:
    - `POST /api/text/process` – main text entry (ReAct + tools).
    - `POST /api/text/respond` – direct LLM response (no tools, for quick dev).
    - `POST /api/voice/process` – voice entry: STT → ReAct → TTS.
    - `POST /api/voice/confirm` – confirm high-risk actions (e.g., transfers).
    - `GET /api/health` – health check.
  - WebSocket endpoint:
    - `GET /ws` – accepts transcript messages and returns replies for streaming UI.
  - Session management:
    - In-memory `SessionManager` keyed by `session_id` with `user_id`, history, and pending actions.

- **MCP Client (`orchestrator/src/clients/mcp_client.py`)**
  - One shared client, initialized on app startup.
  - Calls MCP HTTP server (`mcp-tools`) to execute tools like `balance`, `transfer`, `register_user`, `login_user`, etc.
  - On startup, calls `list_tools` and passes the resulting tool spec into `LLMAgent`.

## New Auth & Voice Behavior

- **Login/Registration (rule-based, no LLM)**
  - Before calling account tools, `LLMAgent` checks if the session is authenticated.
  - If not, it runs a deterministic login/registration flow:
    - Prompts user to choose “login” or “register”.
    - For login: asks for username, validates against mock-bank `/api/users`, then asks for a passphrase and marks the session authenticated.
    - For registration: collects username + passphrase, and marks session authenticated (mock-bank user creation is available via MCP and REST).
  - On successful auth, `authenticated_user_id` is returned and persisted into the session (`session["user_id"]`).

- **Voice**
  - `POST /api/voice/process` accepts:
    - `transcript` (text) **or**
    - `audio_data` (base64-encoded audio).
  - Uses `voice_processing.py`:
    - `transcribe_audio_to_text(audio_bytes)` – STT stub, ready to wire to a real engine.
    - `synthesize_text_to_audio(text)` – TTS stub.
  - Replies include:
    - `response_text` – assistant message.
    - `audio_url` – optional `data:audio/wav;base64,...` URL if TTS produced audio.

## Setup & Running

Environment:

- `GEMINI_API_KEY` - Gemini / GenAI API key.
- `GEMINI_MODEL` - model name (e.g. `gemini-2.5-pro`).
- `MCP_TOOL_BASE_URL` - base URL for MCP tools (default `http://localhost:9100`).
- `SESSION_TIMEOUT_MINUTES` - session expiry window in minutes.
- `SESSION_BACKEND` - `memory` (default) or `redis` for multi-user / multi-instance deployments.
- `SESSION_REDIS_URL` / `SESSION_REDIS_PREFIX` - required when using the Redis backend; lets all orchestrator instances share the same session map.

Install & run (dev):

```bash
cd orchestrator
python -m pip install -r requirements.txt

# Use uvicorn to run FastAPI
uvicorn src.app:app --reload --port 8000
```

The orchestrator expects:

- mock-bank running on `MOCK_BANK_BASE_URL` (used by some legacy auth helpers).
- mcp-tools running and accessible via `MCP_TOOL_BASE_URL`.

## Integration Points

- **mock-bank**
  - Indirectly via MCP tools.
  - Direct HTTP calls from `LLMAgent` only for some legacy auth checks (e.g. listing users).

- **mcp-tools**
  - On startup, orchestrator calls `list_tools` to build the dynamic tool spec.
  - At runtime, tools like `balance`, `transactions`, `transfer`, `register_user`, `login_user`, and `set_user_audio_embedding` are invoked via MCP.

- **frontend**
  - Uses:
    - `POST /api/text/process` or `/api/text/respond` for chat.
    - `POST /api/voice/process` for voice.
    - `GET /ws` for low-latency streaming interactions.
