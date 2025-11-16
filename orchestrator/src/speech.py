# app_multi_lang.py
"""
Patched FastAPI WebSocket ASR server (VOSK) - multi-language with safety guards.

Features:
- Model registry (env-driven) for en/hi/kn/te
- Lazy model loading with cache
- Verbose logging of received audio bytes, partial and final ASR JSON
- Server-side resampling fallback (scipy optional)
- Session safeguards:
  - inactivity timeout
  - max session bytes
  - max utterance duration
- Runs with: python app_multi_lang.py  OR uvicorn app_multi_lang:app --reload --host 0.0.0.0 --port 8765
"""

import asyncio
import json
import logging
import os
import math
from typing import Dict, Optional

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# VOSK ASR
from vosk import Model, KaldiRecognizer

# Optional server-side resampling
try:
    from scipy.signal import resample_poly
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False

# ---------- Configuration ----------
DEFAULT_SR = 16000  # VOSK models expect 16k by default
API_TOKEN = os.environ.get("VOICE_API_TOKEN")  # optional token check

# Model map: set env vars VOSK_EN_MODEL, VOSK_HI_MODEL, etc. or edit paths here.
MODEL_MAP: Dict[str, str] = {
    "en": os.environ.get("VOSK_EN_MODEL", "./models/vosk-model-small-en-us-0.15"),
    "hi": os.environ.get("VOSK_HI_MODEL", "./models/vosk-model-small-hi-0.22"),
    "te": os.environ.get("VOSK_TE_MODEL", "./models/vosk-model-small-te-0.4"),
    "kn": os.environ.get("VOSK_KN_MODEL", "./models/vosk-model-small-kn-0.4"),
}

# Session safeguard defaults
INACTIVITY_TIMEOUT = int(os.environ.get("INACTIVITY_TIMEOUT", 120))     # seconds
MAX_SESSION_BYTES = int(os.environ.get("MAX_SESSION_BYTES", 10 * 1024 * 1024))  # 10 MB
MAX_UTTERANCE_SECONDS = int(os.environ.get("MAX_UTTERANCE_SECONDS", 30))  # seconds

# ---------- Logging ----------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("voice-assistant-multilang")

# ---------- App ----------
app = FastAPI(title="Voice Assistant (VOSK Multi-lang)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model cache to avoid re-loading on every connection
_model_cache: Dict[str, Model] = {}

def model_path_available(lang: str) -> bool:
    p = MODEL_MAP.get(lang)
    return bool(p and os.path.isdir(p))

def load_vosk_model_for_lang(lang: str) -> Optional[Model]:
    """Lazy-load and return a VOSK Model for the requested language.
       Returns None if model path missing or load fails."""
    lang = (lang or "en").lower()
    path = MODEL_MAP.get(lang)
    if not path:
        logger.warning("No model path configured for lang='%s' in MODEL_MAP", lang)
        return None
    if not os.path.isdir(path):
        logger.warning("Configured path for lang '%s' does not exist: %s", lang, path)
        return None
    if lang in _model_cache:
        return _model_cache[lang]
    try:
        logger.info("Loading VOSK model for lang=%s from %s (this may take a while)...", lang, path)
        model = Model(path)
        _model_cache[lang] = model
        logger.info("Loaded model for lang=%s (cached).", lang)
        return model
    except Exception as e:
        logger.exception("Failed to load VOSK model for lang=%s at %s: %s", lang, path, e)
        return None

@app.get("/")
async def root():
    available = {k: (os.path.isdir(p) and p) or None for k, p in MODEL_MAP.items()}
    return {"status": "ok", "available_models": available}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket, token: Optional[str] = Query(None)):
    # Optional token check
    if API_TOKEN and token != API_TOKEN:
        logger.warning("WebSocket connection rejected due to invalid token from %s", websocket.client)
        await websocket.close(code=4001)
        return

    await websocket.accept()
    logger.info("Client connected: %s", websocket.client)

    client_sample_rate = DEFAULT_SR
    channels = 1
    recognizer: Optional[KaldiRecognizer] = None
    model_lang: Optional[str] = None

    # session safeguards
    session_bytes = 0
    utterance_bytes = 0
    utterance_start_time: Optional[float] = None

    try:
        while True:
            # Wait for a message, but timeout if idle for too long
            try:
                msg = await asyncio.wait_for(websocket.receive(), timeout=INACTIVITY_TIMEOUT)
            except asyncio.TimeoutError:
                logger.info("Closing websocket due to inactivity timeout (%ds) for %s", INACTIVITY_TIMEOUT, websocket.client)
                try:
                    await websocket.send_text(json.dumps({"type":"error","message":"session_timeout"}))
                except Exception:
                    pass
                try:
                    await websocket.close(code=4000)
                except Exception:
                    pass
                break

            # Text frame (control)
            if 'text' in msg and msg['text'] is not None:
                try:
                    obj = json.loads(msg['text'])
                except json.JSONDecodeError:
                    logger.debug("Received non-JSON text frame (ignored): %s", msg['text'])
                    continue

                # Meta: client tells us sampleRate/channels/lang
                if obj.get("type") == "meta":
                    sr = int(obj.get("sampleRate", DEFAULT_SR))
                    ch = int(obj.get("channels", 1))
                    enc = obj.get("encoding", "pcm16")
                    lang = obj.get("lang", "en")
                    logger.info("Meta received from client: sampleRate=%s channels=%s encoding=%s lang=%s",
                                sr, ch, enc, lang)
                    client_sample_rate = sr
                    channels = ch
                    model_lang = (lang or "en").lower()

                    # Reset utterance bookkeeping on new meta
                    utterance_bytes = 0
                    utterance_start_time = None

                    # Try loading requested language
                    model = load_vosk_model_for_lang(model_lang)
                    if model is None:
                        available = [k for k, p in MODEL_MAP.items() if os.path.isdir(p)]
                        await websocket.send_text(json.dumps({
                            "type": "error",
                            "message": f"Model for language '{model_lang}' not found on server.",
                            "available_languages": available,
                            "suggestion": "Upload model folder or use a supported language."
                        }))
                        # fallback to English if available
                        fallback = load_vosk_model_for_lang("en")
                        if fallback:
                            recognizer = KaldiRecognizer(fallback, DEFAULT_SR)
                            recognizer.SetWords(True)
                            model_lang = "en"
                            await websocket.send_text(json.dumps({"type": "meta_ack", "message": "fallback_to_en"}))
                        else:
                            await websocket.send_text(json.dumps({"type": "meta_ack", "message": "no_model_loaded"}))
                            recognizer = None
                    else:
                        recognizer = KaldiRecognizer(model, DEFAULT_SR)
                        recognizer.SetWords(True)
                        await websocket.send_text(json.dumps({"type": "meta_ack", "message": "ready", "lang": model_lang}))
                    continue

                # end of utterance
                if obj.get("event") == "end":
                    logger.info("Received end event from client")
                    # finalize utterance bookkeeping
                    utterance_start_time = None
                    utterance_bytes = 0

                    if recognizer:
                        final_res_json = recognizer.FinalResult()
                        logger.info("VOSK FinalResult JSON: %s", final_res_json)
                        try:
                            final_res = json.loads(final_res_json)
                            text = final_res.get("text", "")
                        except Exception:
                            logger.exception("Failed to parse FinalResult")
                            text = ""
                        await websocket.send_text(json.dumps({"type": "final", "text": text}))
                        reply_text = f"Got it: {text}" if text else "I didn't hear anything."
                        await websocket.send_text(json.dumps({"type": "reply", "text": reply_text}))
                    else:
                        await websocket.send_text(json.dumps({"type": "error", "message": "No recognizer available for this session."}))
                    continue

                # reset command
                if obj.get("event") == "reset":
                    logger.info("Received reset event, reinitializing recognizer for lang=%s", model_lang)
                    if model_lang:
                        model = load_vosk_model_for_lang(model_lang)
                    else:
                        model = load_vosk_model_for_lang("en")
                    if model:
                        recognizer = KaldiRecognizer(model, DEFAULT_SR)
                        recognizer.SetWords(True)
                        await websocket.send_text(json.dumps({"type": "reset_ack"}))
                    else:
                        await websocket.send_text(json.dumps({"type": "error", "message": "No model available to reset."}))
                    continue

                # ignore other control messages
                logger.debug("Ignored control message: %s", obj)
                continue

            # Binary frame (audio bytes)
            if 'bytes' in msg and msg['bytes'] is not None:
                audio_bytes = msg['bytes']
                # --- AUDIO DEBUG BLOCK (Paste this below audio_bytes line) ---
                arr = np.frombuffer(audio_bytes, dtype=np.int16)

                # RMS volume (if 0 or near 0 → silence)
                if arr.size > 0:
                    rms = float(np.sqrt(np.mean(arr.astype(np.float32)**2)))
                else:
                    rms = 0.0

                # Log audio stats
                logger.warning(
                    "AUDIO DEBUG: samples=%d min=%d max=%d rms=%.3f",
                    arr.size,
                    int(arr.min() if arr.size else 0),
                    int(arr.max() if arr.size else 0),
                    rms
                )

                # Print first 10 samples for inspection (shows if audio is valid or garbage)
                logger.warning("AUDIO DEBUG: first_samples=%s", arr[:10].tolist())

                chunk_len = len(audio_bytes)
                session_bytes += chunk_len
                utterance_bytes += chunk_len
                now = asyncio.get_event_loop().time()
                if utterance_start_time is None:
                    utterance_start_time = now

                logger.info("Received audio bytes: %d bytes (client_sr=%s, channels=%s) from %s",
                            chunk_len, client_sample_rate, channels, websocket.client)

                # Enforce max session bytes
                if session_bytes > MAX_SESSION_BYTES:
                    logger.warning("Session exceeded MAX_SESSION_BYTES (%d). Closing connection %s", MAX_SESSION_BYTES, websocket.client)
                    try:
                        await websocket.send_text(json.dumps({"type":"error","message":"session_too_large"}))
                    except Exception:
                        pass
                    try:
                        await websocket.close(code=4009)
                    except Exception:
                        pass
                    break

                # Enforce max utterance duration (based on wall-clock time)
                elapsed = now - utterance_start_time if utterance_start_time else 0.0
                if elapsed > MAX_UTTERANCE_SECONDS:
                    logger.warning("Utterance exceeded MAX_UTTERANCE_SECONDS (%d). Closing connection %s", MAX_UTTERANCE_SECONDS, websocket.client)
                    try:
                        await websocket.send_text(json.dumps({"type":"error","message":"utterance_too_long"}))
                    except Exception:
                        pass
                    try:
                        await websocket.close(code=4008)
                    except Exception:
                        pass
                    break

                # Ensure recognizer exists
                if not recognizer:
                    model = load_vosk_model_for_lang("en")
                    if model:
                        recognizer = KaldiRecognizer(model, DEFAULT_SR)
                        recognizer.SetWords(True)
                        logger.info("Auto-initialized English recognizer for session")
                    else:
                        await websocket.send_text(json.dumps({"type": "error", "message": "No ASR model available. Send meta with lang or upload models."}))
                        continue

                # Resample server-side if client SR != DEFAULT_SR
                if client_sample_rate != DEFAULT_SR:
                    if not SCIPY_AVAILABLE:
                        await websocket.send_text(json.dumps({"type": "error", "message": f"Server needs scipy to resample {client_sample_rate}Hz -> {DEFAULT_SR}Hz. Install scipy or send 16k audio."}))
                        logger.warning("Resample needed but scipy not available")
                        continue
                    data = np.frombuffer(audio_bytes, dtype=np.int16)
                    if channels > 1:
                        data = data[::channels]
                    gcd = math.gcd(client_sample_rate, DEFAULT_SR)
                    up = DEFAULT_SR // gcd
                    down = client_sample_rate // gcd
                    resampled = resample_poly(data, up, down).astype(np.int16)
                    audio_bytes_to_feed = resampled.tobytes()
                    logger.debug("Resampled audio: %d -> %d samples", len(data), len(resampled))
                else:
                    if channels > 1:
                        arr = np.frombuffer(audio_bytes, dtype=np.int16)
                        arr = arr[::channels]
                        audio_bytes_to_feed = arr.tobytes()
                    else:
                        audio_bytes_to_feed = audio_bytes

                # Feed bytes to recognizer
                try:
                    accept = recognizer.AcceptWaveform(audio_bytes_to_feed)
                except Exception as e:
                    logger.exception("Error feeding waveform to recognizer: %s", e)
                    try:
                        await websocket.send_text(json.dumps({"type": "error", "message": "ASR processing error"}))
                    except Exception:
                        pass
                    continue

                if accept:
                    res_json = recognizer.Result()
                    logger.info("VOSK Result JSON: %s", res_json)
                    try:
                        res = json.loads(res_json)
                        text = res.get("text", "")
                        logger.info("Final transcript: '%s'", text)
                        await websocket.send_text(json.dumps({"type": "final", "text": text}))
                    except Exception:
                        logger.exception("Failed to parse recognizer.Result JSON")
                else:
                    partial_json = recognizer.PartialResult()
                    logger.debug("VOSK Partial JSON: %s", partial_json)
                    try:
                        partial = json.loads(partial_json)
                        partial_text = partial.get("partial", "")
                        if partial_text:
                            await websocket.send_text(json.dumps({"type": "partial", "text": partial_text}))
                    except Exception:
                        logger.exception("Failed to parse recognizer.PartialResult JSON")
                continue

    except WebSocketDisconnect:
        logger.info("Client disconnected: %s", websocket.client)
    except Exception as e:
        logger.exception("WS handler exception: %s", e)
        try:
            await websocket.close(code=1011)
        except Exception:
            pass

# Run with: python app_multi_lang.py  OR uvicorn app_multi_lang:app --reload --host 0.0.0.0 --port 8765
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8765, log_level="info")
