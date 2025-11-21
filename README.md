# VoxBank – AI Voice Banking Assistant

VoxBank is an AI-powered banking assistant that lets users interact with their accounts via chat or voice. It combines a central **orchestrator** (LLM + ReAct agent), a **mock-bank** API backed by PostgreSQL, and a **Model Context Protocol (MCP)** tools layer, with a React **frontend** for the voice/chat UI.

## Architecture Overview

- **frontend/** – React/TypeScript app for chat + voice UI (mic capture, transcripts, TTS playback).
- **orchestrator/** – FastAPI service that hosts the LLM agent, ReAct loop, session management, auth gating, and voice routing (incl. WebSocket).
- **mcp-tools/** – MCP HTTP server exposing tools like `balance`, `transactions`, `transfer`, plus auth tools like `register_user`, `login_user`, and embedding tools.
- **mock-bank/** – FastAPI + SQLAlchemy backend simulating a bank (users/accounts/transactions) on PostgreSQL, now with `passphrase` + `audio_embedding` fields.
- **data/** – SQL schema for mock-bank (`mock-bank/data/schema.sql`) and related assets.

Data flow:
- Frontend sends text/voice → Orchestrator (HTTP or WebSocket).
- Orchestrator’s `LLMAgent` runs a ReAct loop, calling MCP tools when needed.
- MCP server calls mock-bank REST APIs to execute banking/auth operations.
- Results are summarized by the LLM and returned as text (and optionally audio for voice flows).

## Key Features

- LLM ReAct agent with MCP tool calling.
- Rule-based login/registration gate before account tools.
- User `passphrase` and `audio_embedding` stored in PostgreSQL.
- Dynamic MCP tool discovery → prompt/tool spec auto-updated on restart.
- Voice support:
  - `/api/voice/process` can accept audio (base64) or transcript.
  - STT/TTS interfaces plumbed for future engines.
  - WebSocket `/ws` endpoint for streaming voice/chat.

## Running the Components (Dev)

Prerequisites:
- Python 3.11+
- PostgreSQL with a `voxbank` database (see `mock-bank/README.md`).
- Node.js 18+ (for frontend).

High-level steps (details in each component README):

```bash
# 1) Mock Bank – DB + API
cd mock-bank
python -m pip install -r requirements.txt
# create schema
psql $DATABASE_URL -f data/schema.sql
# seed demo users/accounts
uvicorn src.app:app --reload --port 9000

# 2) MCP Tools
cd mcp-tools
python -m pip install -r requirements.txt
python -m mcp_server  # or: python src/mcp_server.py

# 3) Orchestrator
cd orchestrator
python -m pip install -r requirements.txt
uvicorn src.app:app --reload --port 8000

# 4) Frontend
cd frontend
npm install
npm run dev
```

### Session storage & multi-user support

The orchestrator keeps per-session state (history, authentication, cached accounts). By default it stores this in-process (good for local dev). For concurrent users or multiple orchestrator replicas, switch to Redis:

```
SESSION_BACKEND=redis
SESSION_REDIS_URL=redis://localhost:6379/0  # or your managed Redis
SESSION_REDIS_PREFIX=voxbank:session
SESSION_TIMEOUT_MINUTES=30
```

With Redis enabled, all API workers share the same session map, so multiple tabs/browsers/users can interact simultaneously without clobbering each other.

## Component READMEs

- `orchestrator/README.md` – LLM/ReAct agent, auth, sessions, voice, WebSocket details.
- `mock-bank/README.md` – DB schema, seed data, and auth/embedding endpoints.
- `mcp-tools/README.md` – MCP HTTP server, tool catalog, and discovery.
- `frontend/README.md` – Running the UI, voice flows, and integration with `/api` and `/ws`.

Each sub-README explains how that service fits into the overall VoxBank architecture and how to run it independently.

## New Auth & Voice Capabilities (Summary)

- **Auth**
  - `users` table in mock-bank now has `passphrase` and `audio_embedding (jsonb)`.
  - REST endpoints:
    - `POST /api/users` / `POST /api/register` – create user with username + passphrase (+ optional audio embedding).
    - `POST /api/login` – validate username + passphrase.
    - `PUT/GET /api/users/{user_id}/audio-embedding` – manage voice embeddings.
  - MCP tools wrapping these APIs: `register_user`, `login_user`, `set_user_audio_embedding`, `get_user_profile`.
  - Orchestrator enforces login before calling account tools (`balance`, `transactions`, `transfer`).

- **Voice**
  - Orchestrator’s `/api/voice/process`:
    - Accepts `transcript` or base64 `audio_data`.
    - Calls stubbed `transcribe_audio_to_text` / `synthesize_text_to_audio` (see `orchestrator/src/voice_processing.py`).
    - Returns both `response_text` and optional `audio_url`.
  - WebSocket `/ws`:
    - Used by the frontend for low-latency, streaming-style interactions.
    - Currently text-first; binary audio frames are plumbed but ignored until STT is integrated.

## License

See [LICENSE](LICENSE) for licensing details.

