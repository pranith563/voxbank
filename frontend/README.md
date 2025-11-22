# VoxBank Frontend – Voice & Chat UI

The frontend is a React/Vite application that provides a rich, browser‑based chat and voice interface to VoxBank.  
It showcases:

- Streaming voice input with SpeechRecognition + WebSocket audio
- Client‑side language detection using fastText.js (WASM)
- Integration with the orchestrator’s `/ws` and `/api/*` endpoints
- Optional server‑side TTS playback from the orchestrator

---

## 1. Project Layout

Located under `frontend/`:

- `src/App.tsx` – main app container, routes into voice/chat surfaces.
- `src/VoiceSearchGeminiBrowser.tsx`
  - Full‑featured voice experience:
    - Mic button UI.
    - Native SpeechRecognition transcripts.
    - WebSocket connection to `/ws`.
    - Audio playback for TTS responses.
    - Client‑side STT latency tracking.
  - Sends both **text transcripts** and **PCM16 audio** to the backend.
- `src/VoiceSearch.tsx`
  - Simpler voice search UI using the `useVoiceSearch` hook.
- `src/hooks/useVoiceSearch.ts`
  - Coordinates WebSocket communication and audio sampling via Web Audio.
  - Manages `listening`, `finalTranscript`, `interimTranscript`.
- `src/hooks/useVoiceWebSocket.ts`
  - Manages WebSocket connection lifecycle.
  - Sends/receives JSON messages and binary audio frames.
- `src/hooks/useAudioSampling.ts`
  - Handles microphone capture, downsampling, and silence detection.
- `src/lib/langDetect.ts`
  - Language detection using `fasttext.js` and a `lid.176.ftz` model loaded from `public/models/`.

---

## 2. Voice Flow (Browser + WebSocket)

On a typical voice turn:

1. User taps the mic button.
2. The app:
   - Starts a `SpeechRecognition` instance (browser STT).
   - Starts `MediaRecorder` to stream microphone audio as `audio/webm` chunks.
   - Opens a WebSocket to `ws://<orchestrator-host>:8000/ws`.
   - Sends a `meta` JSON frame describing audio:
     ```json
     { "type": "meta", "sampleRate": 16000, "channels": 1, "encoding": "pcm16", "lang": "en" }
     ```
3. While the user speaks:
   - `SpeechRecognition` populates interim and final transcripts.
   - MediaRecorder sends binary audio chunks over the WebSocket.
4. After silence (or when the user presses the button again):
   - The client:
     - Computes client‑side STT latency (`client_stt_ms`) based on recognition start time.
     - Runs `detectLanguage(text)` (fastText.js) for each new transcript chunk.
     - Sends a `transcript` message:
       ```jsonc
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
       ```
     - Sends `{ "event": "end", "session_id": "sess-123" }` to tell the backend to run Whisper STT over the buffered audio.
5. The backend orchestrator:
   - Runs Whisper, logs the transcription + latency, and runs the ReAct loop.
6. The frontend receives:
   - `reply` messages with `text` to display.
   - Optional `audio_base64` messages with server‑generated TTS audio.
   - Optional `stt_latency` messages with Whisper timing.
7. The UI:
   - Shows text in the transcript and chat.
   - Plays server audio (or uses browser `speechSynthesis` when server TTS is disabled).

---

## 3. Language Detection (fastText.js)

The frontend includes a lightweight language detector:

- File: `src/lib/langDetect.ts`
- Uses `fasttext.js` with a `lid.176.ftz` model served from `public/models/`.
- Supported languages (currently): `en`, `hi`, `te`, `kn`, `ta`, `ml`.
- Behavior:
  - Lazily loads the model on first use.
  - Predicts a label like `__label__en`, `__label__hi`.
  - Normalizes to the VoxBank language set with a simple confidence threshold.
  - Falls back to a heuristic (Latin vs non‑Latin script) and then to `"en"` if detection fails.

Every `transcript` WebSocket message includes this `language` field, which the backend uses to pick STT/TTS settings and to translate between English and the user’s preferred language.

---

## 4. Voice Profiles & TTS

The frontend can choose TTS characteristics by sending a `voice_profile` inside the transcript payload:

```jsonc
{
  "voice_profile": {
    "engine": "piper",          // "piper" | "speechbrain" | "openai" | "gemini"
    "voice_id": "en_IN_male1",  // model / voice name for the chosen engine
    "style": "neutral"          // "warm" | "neutral" | "formal" | "energetic" | …
  }
}
```

The orchestrator merges this with defaults and uses it to select the TTS backend and voice.

If `output_audio` is `true`, the frontend expects an `audio_base64` message and plays it back via an off‑DOM `<audio>` element.

---

## 5. HTTP Integration (Optional)

In addition to the WebSocket `/ws`, the frontend can call HTTP endpoints exposed by the orchestrator:

- `POST /api/text/process`
  - For pure text chat, with full ReAct + tools behavior.
- `POST /api/text/respond`
  - For a simpler LLM‑only response (no tools; useful for prototyping UI copy).
- `POST /api/voice/process`
  - For non‑streaming voice interactions (send base64 audio, receive text + optional audio URL).

These can be called from `services/` modules or via `fetch` directly.

---

## 6. Running the Frontend

### Prerequisites

- Node.js 18+

### Install & Run (dev)

```bash
cd frontend
npm install
npm run dev
```

By default, the app will talk to:

- Orchestrator HTTP at `http://localhost:8000`
- Orchestrator WebSocket at `ws://localhost:8000/ws`

You can adjust these in the components/hooks (or via environment/`VITE_` configuration if you wish to externalize URLs).

