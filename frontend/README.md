# VoxBank Frontend – Voice & Chat UI

The frontend is a React/Vite application that provides a browser-based chat and voice interface to VoxBank.

## Overview

- Located in `frontend/src`.
- Key pieces:
  - `VoiceSearch.tsx` / `VoiceSearchGeminiBrowser.tsx` – voice interaction UIs.
  - `hooks/useVoiceSearch.ts` – coordinates WebSocket + audio sampling.
  - `hooks/useVoiceWebSocket.ts` – manages WebSocket connection and message handling.
  - `App.tsx` – main app container.

## Integration with Backend

- HTTP:
  - Uses orchestrator endpoints such as:
    - `POST /api/text/process` – text chat with full ReAct + tools.
    - `POST /api/voice/process` – voice interactions when sending recorded audio or transcripts.

- WebSocket:
  - Connects to `ws://<orchestrator-host>:8000/ws` by default.
  - Initial message: a `meta` JSON frame:
    ```json
    { "type": "meta", "sampleRate": 16000, "channels": 1, "encoding": "pcm16", "lang": "en" }
    ```
  - Then:
    - Sends binary audio chunks from the microphone.
    - Sends JSON events like `{ "event": "end" }` to mark utterance boundaries.
    - Sends transcript messages via `sendJSON({ type: "transcript", text: "..." })`.
  - Receives server messages:
    - `{ "type": "meta_ack", ... }`
    - `{ "type": "reply", "text": "..." }`
    - `{ "type": "error", "message": "..." }`

The frontend uses these replies to update transcripts and trigger browser TTS playback.

## New Auth & Voice Notes

- Login/registration:
  - The orchestrator enforces login before account-specific tools; the UI can detect clarification messages like:
    > "You're not logged in yet. Would you like to login or register?"
  - A future UI extension can expose username/passphrase flows and voice auth prompts.

- Voice:
  - Audio is captured (e.g. via Web Audio/MediaStream), down-sampled to 16k, and sent over WebSocket.
  - The app treats server `reply` messages as text to display and/or speak via browser TTS.

## Setup & Running

Environment:

- Node.js 18+.

Install and start dev server:

```bash
cd frontend
npm install
npm run dev
```

By default, the app will talk to the orchestrator running on `http://localhost:8000` and `ws://localhost:8000/ws`. Adjust URLs in hooks or environment variables as needed for your environment.

