from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=False)
# voiceConversion.py
"""
FastAPI WebSocket service that accepts {"type":"transcript","text":"..."} and replies
with Gemini-generated WAV audio (base64 encoded).

Requires:
  - Python 3.10+
  - Set environment variable GOOGLE_API_KEY to a valid API key with access to Gemini TTS.
  - Install dependencies: pip install -r requirements.txt

requirements.txt:
  fastapi
  uvicorn[standard]
  google-genai
  anyio
  httpx
"""

import os
import json
import base64
from datetime import datetime
from io import BytesIO
import logging
import asyncio

import wave
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import uvicorn

# NOTE: the GenAI Python package is imported lazily inside the tts function so that
# the app can still start and produce a clear error if the package is missing.
# The docs show using: from google import genai
# See: https://ai.google.dev/gemini-api/docs/speech-generation#python_2. (ref)
from typing import Optional

logger = logging.getLogger("voiceConversion")
logging.basicConfig(level=logging.INFO)

app = FastAPI()

# WAV parameters to match Gemini TTS (docs examples use 24000 Hz, mono, 16-bit)
SAMPLE_RATE = 24000
CHANNELS = 1
SAMPLE_WIDTH_BYTES = 2  # 16-bit = 2 bytes

# Model to use (per docs)
GEMINI_TTS_MODEL = "gemini-2.5-flash-preview-tts"

# API key - required from server environment only
API_KEY = os.getenv("GOOGLE_API_KEY")


def wave_file_bytes_from_pcm(pcm_bytes: bytes,
                             channels: int = CHANNELS,
                             rate: int = SAMPLE_RATE,
                             sample_width: int = SAMPLE_WIDTH_BYTES) -> bytes:
    """
    Wrap raw PCM (signed 16-bit little-endian) into a WAV file (bytes).
    """
    bio = BytesIO()
    with wave.open(bio, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(rate)
        wf.writeframes(pcm_bytes)
    return bio.getvalue()


async def generate_gemini_tts_wav_bytes(text: str, voice_name: Optional[str] = "Kore") -> bytes:
    """
    Calls Gemini TTS via the google-genai Python client and returns WAV bytes.
    - The Gemini client may be synchronous; run it in a threadpool to avoid blocking the event loop.
    - Handles returned data that can be bytes or base64-encoded string (be defensive).
    """
    if not API_KEY:
        raise RuntimeError("server_missing_api_key: Set GOOGLE_API_KEY in server environment")

    # Lazy import so missing library yields a clear runtime error
    try:
        from google import genai
        from google.genai import types
    except Exception as e:
        raise RuntimeError("google-genai package missing. Install with `pip install google-genai`.") from e

    # genai.Client() reads environment variable GOOGLE_API_KEY automatically, but we explicitly
    # set it on the client for clarity (the client supports passing api_key in constructor).
    def sync_call():
        client = genai.Client()  # will read GOOGLE_API_KEY env or application default
        # Build the speech config per docs
        cfg = types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name=voice_name
                    )
                )
            )
        )

        # The docs show contents as a simple string for Python. We'll pass the text directly.
        response = client.models.generate_content(
            model=GEMINI_TTS_MODEL,
            contents=text,
            config=cfg,
        )

        # Extract audio bytes from response. Docs show:
        # data = response.candidates[0].content.parts[0].inline_data.data
        # That may be bytes (preferred) or base64 str; handle both.
        try:
            cand = response.candidates[0]
            part = cand.content.parts[0]
            inline = getattr(part, "inline_data", None) or getattr(part, "inlineData", None) or {}
            data = inline.data if hasattr(inline, "data") else inline.get("data") if isinstance(inline, dict) else None
        except Exception as e:
            # Try alternate path in case the structure differs slightly
            raise RuntimeError(f"unexpected_gemini_response_shape: {e}")

        if data is None:
            raise RuntimeError("no_audio_in_gemini_response")

        # If data is a str, assume base64 and decode; if bytes-like, use directly.
        if isinstance(data, str):
            # docs sometimes show base64 in JS examples; be defensive
            try:
                pcm_bytes = base64.b64decode(data)
            except Exception:
                # maybe it's raw text -> convert to bytes
                pcm_bytes = data.encode("utf-8")
        elif isinstance(data, (bytes, bytearray)):
            pcm_bytes = bytes(data)
        else:
            # fallback: attempt to convert
            pcm_bytes = bytes(data)

        # The docs' JS curl example writes out 'out.pcm' then uses ffmpeg:
        # ffmpeg -f s16le -ar 24000 -ac 1 -i out.pcm out.wav
        # That means data is raw signed 16-bit little-endian PCM.
        # Wrap into WAV container:
        wav_bytes = wave_file_bytes_from_pcm(pcm_bytes, channels=CHANNELS,
                                            rate=SAMPLE_RATE, sample_width=SAMPLE_WIDTH_BYTES)
        return wav_bytes

    # Run the synchronous gemini client call in a thread to avoid blocking
    loop = asyncio.get_running_loop()
    wav_bytes = await loop.run_in_executor(None, sync_call)
    return wav_bytes


@app.websocket("/ws")
async def websocket_tts(ws: WebSocket):
    await ws.accept()

    # Only use server-side API key (no client-provided keys)
    if not API_KEY:
        await ws.send_text(json.dumps({
            "type": "error",
            "message": "server_missing_api_key",
            "details": "Set GOOGLE_API_KEY in server environment to a valid Gemini API key."
        }))
        await ws.close()
        return

    try:
        while True:
            raw = await ws.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await ws.send_text(json.dumps({"type": "error", "message": "invalid_json"}))
                continue

            mtype = msg.get("type")
            if mtype == "transcript":
                text = msg.get("text", "")
                if not text:
                    await ws.send_text(json.dumps({"type": "error", "message": "empty_text"}))
                    continue

                # Acknowledge/processing message
                await ws.send_text(json.dumps({"type": "processing", "message": "synthesizing"}))

                try:
                    wav_bytes = await generate_gemini_tts_wav_bytes(text, voice_name="Kore")
                except Exception as e:
                    # Expose helpful error to client logs (don't leak secrets)
                    logger.exception("Gemini TTS error")
                    await ws.send_text(json.dumps({
                        "type": "error",
                        "message": "gemini_tts_failed",
                        "details": str(e)
                    }))
                    continue

                # Base64-encode WAV bytes for JSON transport
                wav_b64 = base64.b64encode(wav_bytes).decode("ascii")
                ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
                filename = f"tts-{ts}.wav"

                resp = {
                    "type": "audio",
                    "format": "wav",
                    "mime": "audio/wav",
                    "filename": filename,
                    "audio": wav_b64
                }
                await ws.send_text(json.dumps(resp))
                continue

            else:
                await ws.send_text(json.dumps({"type": "error", "message": f"unknown_type_{mtype}"}))

    except WebSocketDisconnect:
        logger.info("Client disconnected")
        return
    except Exception as e:
        logger.exception("Unhandled server exception")
        try:
            await ws.send_text(json.dumps({"type": "error", "message": "server_exception", "details": str(e)}))
        except Exception:
            pass
        await ws.close()


# Simple test UI (server-side API key only)
@app.get("/")
async def root():
    html = """
    <!doctype html>
    <html>
      <head><meta charset="utf-8"/><title>Gemini TTS WS Test</title></head>
      <body>
        <h3>Gemini TTS WS (server-only key)</h3>
        <textarea id="text" rows="4" cols="80">hello can you hear me</textarea><br/>
        <button onclick="connectAndSend()">Connect & Send</button>
        <pre id="log"></pre>
        <audio id="player" controls></audio>
        <script>
          function log(s){ document.getElementById('log').innerText += s + "\\n"; }
          let ws;
          function connectAndSend(){
            const text = document.getElementById('text').value;
            ws = new WebSocket("ws://" + location.host + "/ws");
            ws.onopen = () => {
              log("ws open");
              ws.send(JSON.stringify({type:'transcript', text}));
            };
            ws.onmessage = (ev) => {
              try {
                const m = JSON.parse(ev.data);
                log("recv: " + JSON.stringify(m).slice(0,200));
                if(m.type === 'audio' && m.audio){
                  const b64 = m.audio;
                  const binary = atob(b64);
                  const len = binary.length;
                  const bytes = new Uint8Array(len);
                  for (let i = 0; i < len; i++) bytes[i] = binary.charCodeAt(i);
                  const blob = new Blob([bytes.buffer], {type: m.mime || 'audio/wav'});
                  const url = URL.createObjectURL(blob);
                  document.getElementById('player').src = url;
                }
              } catch(e) {
                log("non-json or parse error: " + ev.data);
              }
            };
            ws.onclose = () => log("ws closed");
            ws.onerror = (e) => log("ws error");
          }
        </script>
      </body>
    </html>
    """
    return HTMLResponse(content=html, status_code=200)


