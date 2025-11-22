# Orchestrator – LLM Agent, Conversation Engine & Voice Router

The orchestrator is a FastAPI service that hosts VoxBank’s LLM-based assistant. It manages:

- ReAct-style tool calling via MCP (Model Context Protocol)
- Deterministic login / registration flows
- Session + profile hydration from the mock-bank
- Text and voice entry points over HTTP and WebSocket

It is the “brain” that coordinates the frontend, MCP tools, and mock-bank.

---

## 1. High‑Level Components

- **LLM Agent (`src/agent/agent.py`)**
  - Runs the **decision loop** (`decision()`): chooses between `respond`, `call_tool`, `ask_user`, `ask_confirmation`.
  - Uses `GeminiLLMClient` (Gemini) for both decision and final response generation.
  - Maintains per-session conversation history and a small auth state.
  - Injects MCP tool metadata into prompts dynamically (no hard-coded tool list).

- **Conversation Orchestrator (`src/agent/orchestrator.py`)**
  - Owns the **ReAct loop** around the LLM:
    - Calls `agent.decision(...)`
    - Executes MCP tools via `MCPClient`
    - Feeds tool results back into the LLM or deterministic fallbacks
  - Enforces authentication for account-level intents.
  - Handles OTP flows and confirmation prompts for high‑risk operations (e.g., transfers).

- **FastAPI App (`src/app.py`)**
  - REST endpoints:
    - `POST /api/text/process` – main text entry (ReAct + tools).
    - `POST /api/text/respond` – direct LLM response (no tools; useful for quick dev).
    - `POST /api/voice/process` – voice entry: optional STT + ReAct + TTS (HTTP mode).
    - `POST /api/voice/confirm` – confirmation channel for high‑risk actions.
    - `POST /api/auth/login` – deterministic HTTP login endpoint.
    - `GET /api/health` – health check.
  - WebSocket endpoint:
    - `GET /ws` – full duplex voice/chat channel (binary audio + JSON messages).
  - Startup/shutdown:
    - Initializes Gemini client, MCP client, Whisper STT client, session manager, and tool spec.

- **MCP Client (`src/clients/mcp_client.py`)**
  - Uses `fastmcp` when available, with HTTP fallback.
  - Calls MCP HTTP server (`mcp-tools`) to execute tools like:
    - `balance`, `transactions`, `transfer`
    - `register_user`, `login_user`, `set_user_audio_embedding`
    - `get_my_profile`, `get_my_accounts`, `get_my_beneficiaries`
  - On startup, calls `list_tools` and passes the resulting tool spec into `VoxBankAgent`.

---

## 2. Auth Flow – Deterministic, Not LLM‑Driven

Authentication is deliberately **not** left to the LLM:

- Each session has an `auth_state`:
  ```jsonc
  {
    "authenticated": false,
    "user_id": null,
    "flow_stage": "idle" | "await_username" | "await_passphrase",
    "temp": { "username": "…" }
  }
  ```

- When the user asks for a protected intent (balance, transactions, loans, transfers, etc.) while unauthenticated:
  - The orchestrator sets `flow_stage = "await_username"` and prompts:
    > "You're not logged in yet. Please provide your username to continue."

- Once the username is provided:
  - `flow_stage` becomes `"await_passphrase"`.
  - The next utterance is treated as the passphrase and validated via **mock-bank**:
    - `POST /api/login` (wrapped by `_login_user_via_http`).

- On successful login:
  - `authenticated_user_id` is returned by the orchestrator.
  - It updates the session (`session["user_id"]`) and hydrates the profile from mock-bank.
  - `auth_state.authenticated = true` and `flow_stage = "idle"`.

- On failure:
  - The orchestrator stays in `"await_passphrase"` or resets to `"await_username"` and sends clear messages such as:
    > "That passphrase didn't match our records. Please try again, or say 'cancel login'."

Once the auth flow has started, the orchestrator **does not** call the LLM for these username/passphrase turns; it is a pure Python state machine.

---

## 3. Voice: STT, TTS & WebSocket Protocol

### 3.1 Server‑Side STT with Whisper

WebSocket `/ws` is the main path for real‑time voice:

1. Client opens a WebSocket and sends a **meta** frame:
   ```json
   {
     "type": "meta",
     "sampleRate": 16000,
     "channels": 1,
     "encoding": "pcm16",
     "lang": "en"
   }
   ```

2. Client streams **binary PCM16 audio** frames (ArrayBuffer) as the user speaks.

3. When the utterance ends (silence or button press), client sends:
   ```json
   { "event": "end", "session_id": "sess-123" }
   ```

4. The orchestrator:
   - Buffers all PCM16 chunks for that session.
   - Wraps them into a WAV container.
   - Calls **OpenAI Whisper** (`whisper-1`) via `AsyncOpenAI.audio.transcriptions.create`.
   - Logs the transcription and latency to the console:
     ```text
     Whisper STT: transcript='do I have any loans' duration_ms=132.4
     ```
   - Optionally sends a latency message back to the client:
     ```json
     { "type": "stt_latency", "whisper_ms": 132.4, "session_id": "sess-123" }
     ```
   - Treats the transcription as a normal user message and routes it into `agent.orchestrate(...)`.

HTTP voice endpoint (`/api/voice/process`) can still accept `audio_data`, but the most advanced STT behavior is on the WebSocket path.

### 3.2 Multi‑Engine TTS & Voice Profiles

TTS is implemented in `src/voice_processing.py`:

- Supported TTS engines:
  - `piper` – local CLI (`piper`) with on‑device models.
  - `speechbrain` – local Tacotron2 + HiFi‑GAN models via SpeechBrain.
  - `openai` – OpenAI TTS (`gpt-4o-mini-tts`).
  - `gemini` – Gemini TTS (e.g., `gemini-2.5-flash-preview-tts`).

- Per‑session **voice profile** stored in the session and exposed in the session profile:
  ```jsonc
  {
    "voice_profile": {
      "engine": "piper",             // or "speechbrain", "openai", "gemini"
      "voice_id": "en_IN_female1",   // engine-specific voice id / model
      "style": "warm"                // semantic style hint
    }
  }
  ```

- The frontend can override this profile by sending:
  ```json
  {
    "type": "transcript",
    "session_id": "sess-123",
    "text": "Send 50 dollars to John",
    "output_audio": true,
    "language": "en",
    "voice_profile": {
      "engine": "piper",
      "voice_id": "en_IN_male1",
      "style": "neutral"
    }
  }
  ```

- The orchestrator merges this with defaults (see `context/voice_profile.py`) and passes it into `synthesize_text_to_audio(...)`. If a local engine fails, it falls back to cloud TTS (OpenAI / Gemini).

### 3.3 WebSocket Message Flow (Example)

```jsonc
// 1) Client -> server: describe audio stream
{ "type": "meta", "sampleRate": 16000, "channels": 1, "encoding": "pcm16", "lang": "en" }

// 2) Client -> server: binary PCM16 audio frames (ArrayBuffer)
//  (sent as raw binary frames, not JSON)

// 3) Client -> server: optional transcript with language & client STT timing
{
  "type": "transcript",
  "session_id": "sess-123",
  "text": "Do I have any loans?",
  "output_audio": true,
  "language": "en",
  "client_stt_ms": 82,
  "voice_profile": {
    "engine": "piper",
    "voice_id": "en_IN_female1",
    "style": "warm"
  }
}

// 4) Client -> server: mark utterance end for Whisper
{ "event": "end", "session_id": "sess-123" }

// 5) Server -> client: STT latency (Whisper)
{ "type": "stt_latency", "whisper_ms": 131.7, "session_id": "sess-123" }

// 6) Server -> client: assistant reply text
{ "type": "reply", "text": "Yes, you currently have one active home loan.", "session_id": "sess-123" }

// 7) Server -> client: assistant audio (if output_audio=true)
{
  "type": "audio_base64",
  "audio": "data:audio/wav;base64,AAAA...",
  "mime": "audio/wav",
  "session_id": "sess-123",
  "voice_profile": {
    "engine": "piper",
    "voice_id": "en_IN_female1",
    "style": "warm"
  }
}
```

---

## 4. Session Management & Profiles

Sessions are managed by `context/session_manager.py`:

- Each session contains:
  - `session_id`, `user_id`, `username`
  - `is_authenticated`, `is_voice_verified`
  - `primary_account`, `accounts` (hydrated from mock-bank)
  - `preferred_language`, `stt_lang`, `tts_lang`
  - `voice_profile` (see above)
  - `history` (compact role/text history used in prompts)

The helper `context/session_profile.get_session_profile(session_id)` returns a compact profile used throughout the orchestrator and prompts.

Redis support:

- By default, sessions are stored in memory (good for local dev).
- To enable Redis:
  ```bash
  export SESSION_BACKEND=redis
  export SESSION_REDIS_URL=redis://localhost:6379/0
  export SESSION_REDIS_PREFIX=voxbank:session
  export SESSION_TIMEOUT_MINUTES=30
  ```
  This allows multiple orchestrator instances to share state.

---

## 5. Setup & Running

### Environment

Required / important env vars:

- `GEMINI_API_KEY` – Gemini / GenAI API key.
- `GEMINI_MODEL` – Gemini model name (e.g. `gemini-2.5-pro`).
- `OPENAI_API_KEY` – OpenAI API key (for Whisper STT and OpenAI TTS).
- `MCP_TOOL_BASE_URL` – base URL for MCP tools (default `http://localhost:9100`).
- `VOX_BANK_BASE_URL` – mock-bank base URL (default `http://localhost:9000`).
- `SESSION_BACKEND`, `SESSION_REDIS_URL`, `SESSION_REDIS_PREFIX` – session storage configuration.
- `LOG_LEVEL` – log level (`INFO` is a good default).

### Running (dev)

```bash
cd orchestrator
python -m pip install -r requirements.txt

# Run FastAPI with reload
uvicorn src.app:app --reload --port 8000
```

The orchestrator expects:

- mock-bank running at `VOX_BANK_BASE_URL` (for user/accounts/loans data).
- mcp-tools running at `MCP_TOOL_BASE_URL` (for tools like `balance`, `transfer`, `get_my_*`).

---

## 6. Integration Summary

- **mock-bank**
  - Used via MCP tools and direct HTTP for profile/accounts hydration.
  - Provides ground truth for users, accounts, loans, transactions, and reminders.

- **mcp-tools**
  - On startup, orchestrator calls `list_tools` to build the dynamic tool spec.
  - At runtime, calls tools such as:
    - `balance`, `transactions`, `transfer`
    - `register_user`, `login_user`
    - `set_user_audio_embedding`, `get_user_profile`
    - `get_my_profile`, `get_my_accounts`, `get_my_beneficiaries`

- **frontend**
  - Uses:
    - `POST /api/text/process` or `/api/text/respond` for chat.
    - `POST /api/voice/process` for simple voice integrations.
    - `GET /ws` for full streaming voice (audio + text + TTS + latencies).

