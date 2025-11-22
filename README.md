# VoxBank – AI Voice Banking Assistant

VoxBank is an end‑to‑end **AI voice banking** prototype.  
It lets users ask natural questions like “What’s my balance?” or “Do I have any loans?” and handles everything from:

- LLM intent understanding and tool calling (ReAct)
- Safe execution of banking operations via MCP tools
- A PostgreSQL‑backed mock‑bank core
- Streaming voice input/output with Whisper STT and multi‑engine TTS

At a glance:

- **frontend/** – React/TypeScript voice + chat UI (browser STT, fastText language detection, WS audio).
- **orchestrator/** – FastAPI LLM agent + conversation engine + voice router.
- **mcp-tools/** – MCP HTTP tool server (balance, transactions, transfer, “my_*” summary tools, auth).
- **mock-bank/** – Postgres‑backed banking API (users, accounts, transactions, loans, cards, reminders).
- **docs/** – Architecture, sequence, data‑model, MCP, and security diagrams.

> If you open this README on GitHub, you can expand the diagrams in `docs/` for a visual tour.

---

## 1. Architecture Overview

```text
┌───────────┐       WebSocket / HTTP        ┌───────────────────┐     MCP HTTP      ┌────────────────┐
│ Frontend  │  ───────────────────────────► │  Orchestrator     │ ────────────────► │  mcp-tools     │
│ (React)   │  ◄───────────────────────────  │  (LLM + ReAct)    │ ◄───────────────  │  (FastMCP)     │
└───────────┘                                └───────────────────┘                  └──────┬─────────┘
           ▲                                         │                                    │
           │                                         │ HTTP                               │
           │                                         ▼                                    │
           │                                  ┌───────────────┐                           │
           │                                  │  mock-bank    │◄──────────────────────────┘
           │                                  │  (FastAPI +   │
           │                                  │   PostgreSQL) │
           │                                  └───────────────┘
           │
           │ audio (PCM16), transcripts, TTS
           ▼
    Microphone + Speaker
```

- **Frontend**
  - Captures mic audio, runs browser STT for instant UI feedback.
  - Streams audio to the backend over WebSocket.
  - Performs client‑side language detection via `fasttext.js`.
  - Displays chat and plays audio replies (server TTS or browser TTS).

- **Orchestrator**
  - Runs the ReAct loop on top of Gemini (`GeminiLLMClient`).
  - Keeps sessions, auth state, and user profiles.
  - Calls MCP tools to perform banking operations.
  - Handles voice flows via WebSocket + Whisper STT + pluggable TTS.

- **MCP Tools**
  - Exposes a curated set of tools via the Model Context Protocol:
    - `balance`, `transactions`, `transfer`
    - `register_user`, `login_user`, `set_user_audio_embedding`, `get_user_profile`
    - `get_my_profile`, `get_my_accounts`, `get_my_beneficiaries`
    - `cards_summary`, `loans_summary`, `reminders_summary`, `logout_user`, `list_tools`
  - Acts as a security and compliance boundary: the LLM never talks to the DB directly.

- **Mock Bank**
  - Simulates a core banking system:
    - Users (with `passphrase` + `audio_embedding`)
    - Accounts & transactions
    - Beneficiaries, cards, loans, reminders
  - All side‑effects (e.g., transfers) go through here.

For visuals, see:

- `docs/architecture-diagram.png`
- `docs/sequence-diagram.png`
- `docs/data-model.png`
- `docs/mcp-flow.png`
- `docs/security-model.png`

---

## 2. Key Features

### 2.1 Safe LLM Tool‑Calling

- ReAct‑style loop:
  - LLM decides: `respond` / `call_tool` / `ask_user` / `ask_confirmation`.
  - Tools are validated against a schema from `mcp-tools` (`list_tools`).
  - Required parameters are enforced before any tool executes.
  - High‑risk tools (e.g., `transfer`) require explicit confirmation.

### 2.2 Deterministic Auth Flow (No LLM Loops)

- Auth is handled by a **Python state machine**, not the LLM:
  - Stages: `idle` → `await_username` → `await_passphrase`.
  - Username and passphrase are read directly from user turns.
  - `POST /api/login` on mock‑bank is used for verification.
  - On success, a session profile is hydrated with user + accounts + loans, etc.
- This design:
  - Eliminates prompt‑loop issues around login.
  - Makes auth behavior fully predictable and testable.

### 2.3 Voice: Whisper STT + Multi‑Engine TTS

- **Streaming STT with Whisper**
  - Frontend:
    - Sends a `meta` frame with audio config.
    - Streams PCM16 audio as binary WebSocket frames.
    - Sends `{"event": "end"}` to trigger STT.
  - Orchestrator:
    - Buffers audio, wraps it into WAV, calls `whisper-1`.
    - Logs transcript + latency:
      ```text
      Whisper STT: transcript='do I have any loans' duration_ms=132.4
      ```
    - Sends `{"type": "stt_latency", "whisper_ms": 132.4, ...}` back to the client.
    - Routes the transcription into the same ReAct loop as text chat.

- **Multi‑Engine TTS**
  - Pluggable engines in `orchestrator/src/voice_processing.py`:
    - `piper` – local CLI engine (default; low latency, on‑device).
    - `speechbrain` – local Tacotron2 + HiFi‑GAN (fully offline once models cached).
    - `openai` – OpenAI TTS (`gpt-4o-mini-tts`).
    - `gemini` – Gemini TTS (`gemini-2.5-flash-preview-tts`, etc.).
  - Per‑session **voice profiles**:
    ```jsonc
    {
      "voice_profile": {
        "engine": "piper",
        "voice_id": "en_IN_female1",
        "style": "warm"
      }
    }
    ```
  - The frontend can override this profile in each `transcript` message.

### 2.4 Client‑Side Language Detection

- `frontend/src/lib/langDetect.ts`:
  - Uses `fasttext.js` with `lid.176.ftz` hosted under `public/models/`.
  - Normalizes predictions (e.g., `__label__en`, `__label__hi`) to a small set:
    - `["en", "hi", "te", "kn", "ta", "ml"]`
  - Sends `language` with each `transcript`:
    ```json
    { "type": "transcript", "text": "क्या मेरे पास कोई लोन है?", "language": "hi", ... }
    ```
  - Backend uses this to:
    - Choose STT/TTS presets.
    - Translate between English and the user’s preferred language when needed.

### 2.5 Observability: STT Latencies

- Frontend sends `client_stt_ms` – how long browser STT took for each utterance.
- Backend sends `stt_latency` with `whisper_ms` – how long Whisper needed.
- This makes it easy to measure end‑to‑end latency and compare browser vs server STT.

---

## 3. Running the Components (Dev)

### 3.1 Prerequisites

- Python 3.11+
- PostgreSQL with a `voxbank` database
- Node.js 18+
- Valid API keys:
  - `GEMINI_API_KEY` for Gemini LLM/TTS.
  - `OPENAI_API_KEY` for Whisper STT and OpenAI TTS.

### 3.2 Core Services

#### 1) Mock Bank – DB + API

```bash
cd mock-bank
python -m pip install -r requirements.txt

# create schema
export DATABASE_URL="postgresql+asyncpg://postgres:password@localhost:5432/voxbank"
psql "postgres://postgres:password@localhost:5432/voxbank" -f data/schema.sql

# optional: seed demo users/accounts/loans
uvicorn src.app:app --reload --port 9000
# then:
curl -X POST "http://localhost:9000/api/admin/seed" \
  -H "Content-Type: application/json" \
  -d '{ "token": "letmein" }'
```

#### 2) MCP Tools

```bash
cd mcp-tools
python -m pip install -r requirements.txt

export VOX_BANK_BASE_URL="http://localhost:9000"
export MCP_HOST="0.0.0.0"
export MCP_PORT=9100

python -m src.mcp_server      # or: python src/mcp_server.py
```

#### 3) Orchestrator

```bash
cd orchestrator
python -m pip install -r requirements.txt

export GEMINI_API_KEY=...
export GEMINI_MODEL="gemini-2.5-pro"
export OPENAI_API_KEY=...
export MCP_TOOL_BASE_URL="http://localhost:9100"
export VOX_BANK_BASE_URL="http://localhost:9000"

uvicorn src.app:app --reload --port 8000
```

#### 4) Frontend

```bash
cd frontend
npm install
npm run dev
```

By default the frontend connects to:

- HTTP: `http://localhost:8000`
- WebSocket: `ws://localhost:8000/ws`

You can adjust these in `VoiceSearchGeminiBrowser.tsx` or via Vite env if you choose to externalize configuration.

---

## 4. Session Storage & Multi‑User Support

The orchestrator keeps per‑session state (history, auth, profiles, voice profile, etc.).  
By default this is in‑process memory (great for local dev).

To run with multiple orchestrator instances or survive restarts, enable Redis:

```bash
export SESSION_BACKEND=redis
export SESSION_REDIS_URL=redis://localhost:6379/0  # or your managed Redis
export SESSION_REDIS_PREFIX=voxbank:session
export SESSION_TIMEOUT_MINUTES=30
```

With Redis enabled:

- All API workers share the same sessions.
- Multiple tabs / devices can talk to the same session_id without clobbering state.

---

## 5. Component READMEs

Each sub‑directory has its own detailed README:

- `orchestrator/README.md` – LLM/ReAct agent, deterministic auth, sessions, Whisper STT, TTS engines, WebSocket protocol.
- `mock-bank/README.md` – schema, seed data, auth/embedding endpoints, and banking APIs.
- `mcp-tools/README.md` – MCP HTTP server, full tool catalog and how the orchestrator uses `list_tools`.
- `frontend/README.md` – voice UI, WebSocket integration, language detection, and voice profiles.
- `docs/README.md` – description of the diagrams and how to use them in reviews.

Start there if you want to deep‑dive into a specific piece of the system.

---

## 6. Security & Compliance Notes (Conceptual)

VoxBank is a demo, but the architecture borrows from real‑world secure assistant patterns:

- **Tool sandbox via MCP**
  - The LLM never sees DB credentials.
  - All side‑effects go through audited HTTP tools.

- **Auth gating & ownership checks**
  - Deterministic login flow; no “hallucinated” auth.
  - Transfers can only originate from accounts owned by the logged‑in user.

- **Separation of concerns**
  - Mock‑bank, mcp‑tools, and orchestrator are separate processes/services.
  - Easy to swap mock‑bank for a real core system behind the same tool surface.

These design choices are illustrated in `docs/security-model.png`.

---

## 7. License

See [LICENSE](LICENSE) for licensing details.

