"""
voice_processing.py

Lightweight STT/TTS plumbing for VoxBank.

This module defines small, swappable interfaces for:
- Speech-to-text (STT): transcribe_audio_to_text
- Text-to-speech (TTS): synthesize_text_to_audio
- Voice embedding (for future auth): extract_voice_embedding

All implementations are currently stubbed/minimal and should be replaced
with real engines (e.g. local ASR, cloud TTS, or embedding models) later.
"""

from typing import Optional
import base64
import logging
import os
import asyncio
import time
from io import BytesIO
import wave

logger = logging.getLogger("voxbank.voice_processing")

# Global TTS backend selector: "openai" or "gemini"
TTS_PROVIDER = (os.getenv("TTS_PROVIDER") or "openai").lower()

# OpenAI TTS configuration (for gpt-4o-mini-tts)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_TTS_MODEL = os.getenv("OPENAI_TTS_MODEL", "gpt-4o-mini-tts")
OPENAI_TTS_VOICE = os.getenv("OPENAI_TTS_VOICE", "alloy")
OPENAI_TTS_FORMAT = os.getenv("OPENAI_TTS_FORMAT", "wav")  # desired format; SDK may fall back

# Gemini TTS configuration (gemini-2.5-flash-lite-preview-tts)
GEMINI_API_KEY = (
  os.getenv("GEMINI_API_KEY")
  or os.getenv("GENAI_API_KEY")
  or os.getenv("GEMINI_TOKEN")
)
GEMINI_TTS_MODEL = os.getenv(
  "GEMINI_TTS_MODEL", "gemini-2.5-flash-preview-tts"
)

# WAV parameters to match Gemini TTS PCM output (per docs / voiceConversion.py)
SAMPLE_RATE = 24000
CHANNELS = 1
SAMPLE_WIDTH_BYTES = 2  # 16-bit = 2 bytes

# Lazy-initialized SDK clients (shared across calls)
_openai_client = None
_gemini_client = None
_genai_types = None


async def transcribe_audio_to_text(audio_bytes: bytes, *, lang: str = "en") -> str:
  """
  Stub STT implementation.

  - Input: raw audio bytes (e.g. PCM/WAV/Opus, depending on frontend).
  - Output: best-effort transcript string.

  TODO:
  - Integrate a real STT engine (e.g., local Whisper, cloud API, or Vosk).
  - Normalize sampling rate / channels before calling the engine.
  """
  if not audio_bytes:
    logger.warning("STT: received empty audio_bytes")
    return ""

  # For now, we cannot actually decode audio without an engine,
  # so we return a fixed placeholder and rely on the frontend to
  # also send a transcript when needed.
  logger.info("STT: stub called (len=%d bytes, lang=%s)", len(audio_bytes), lang)
  return ""


async def synthesize_text_to_audio(text: str, *, lang: str = "en") -> Optional[bytes]:
  """
  Text-to-speech implementation using the configured backend.

  - Input: assistant reply text.
  - Output: audio bytes or None if TTS is not configured/available.

  Backend is selected via TTS_PROVIDER env var:
    - "openai" (default): OpenAI gpt-4o-mini-tts
    - "gemini": Gemini gemini-2.5-flash-lite-preview-tts
  """
  if not text:
    return None

  requested_provider = (TTS_PROVIDER or "openai").lower()
  provider_used = None
  audio: Optional[bytes] = None
  started = time.perf_counter()

  # Try Gemini first if requested
  if requested_provider == "gemini":
    provider_used = "gemini"
    audio = await _synthesize_with_gemini(text, lang=lang)
    if audio is None:
      logger.warning(
        "TTS: Gemini backend failed or not configured; falling back to OpenAI if available"
      )

  # Fallback or default: OpenAI
  if audio is None:
    provider_used = "openai"
    audio = await _synthesize_with_openai(text, lang=lang)

  elapsed_ms = (time.perf_counter() - started) * 1000.0
  logger.info(
    "TTS: provider=%s requested=%s duration_ms=%.1f success=%s bytes=%s",
    provider_used,
    requested_provider,
    elapsed_ms,
    bool(audio),
    len(audio) if audio else 0,
  )
  return audio


async def _synthesize_with_openai(text: str, *, lang: str = "en") -> Optional[bytes]:
  """
  OpenAI-based TTS helper.
  """
  if not OPENAI_API_KEY:
    logger.warning("TTS(OpenAI): OPENAI_API_KEY not set; skipping TTS generation")
    return None

  try:
    # Import OpenAI SDK lazily so the module still loads if it's missing.
    try:
      from openai import OpenAI  # type: ignore
    except Exception as e:  # pragma: no cover - optional dependency
      logger.warning("TTS(OpenAI): OpenAI SDK not installed; cannot synthesize audio (%s)", e)
      return None

    global _openai_client
    if _openai_client is None:
      _openai_client = OpenAI(api_key=OPENAI_API_KEY)
      logger.info("TTS(OpenAI): created OpenAI client instance")

    def _call_openai_tts() -> Optional[bytes]:
      try:
        client = _openai_client
        logger.info(
          "TTS(OpenAI): calling audio.speech model=%s voice=%s format=%s text_len=%d",
          OPENAI_TTS_MODEL,
          OPENAI_TTS_VOICE,
          OPENAI_TTS_FORMAT,
          len(text),
        )
        # Some OpenAI Python SDK versions accept `response_format`, others do not.
        # First try with it; if the SDK rejects it, retry without and let server default.
        try:
          resp = client.audio.speech.create(
            model=OPENAI_TTS_MODEL,
            voice=OPENAI_TTS_VOICE,
            input=text,
            response_format=OPENAI_TTS_FORMAT,
          )
        except TypeError as te:
          logger.warning(
            "TTS(OpenAI): SDK does not support 'response_format' arg (%s); "
            "retrying without explicit format (server default).",
            te,
          )
          resp = client.audio.speech.create(
            model=OPENAI_TTS_MODEL,
            voice=OPENAI_TTS_VOICE,
            input=text,
          )
        # New OpenAI SDK typically returns raw bytes for audio.speech.create.
        if isinstance(resp, (bytes, bytearray)):
          return bytes(resp)
        # Fallbacks for older/variant SDK response types.
        content = getattr(resp, "content", None)
        if content is not None:
          return content  # type: ignore[return-value]
        if hasattr(resp, "read"):
          return resp.read()  # type: ignore[call-arg, return-value]
        logger.warning("TTS(OpenAI): unexpected response type from OpenAI TTS: %r", type(resp))
        return None
      except Exception as exc:  # pragma: no cover - external dependency
        logger.exception("TTS(OpenAI): synthesis failed (sync): %s", exc)
        return None

    audio_bytes = await asyncio.to_thread(_call_openai_tts)
    if audio_bytes:
      logger.info("TTS(OpenAI): received %d bytes from OpenAI TTS", len(audio_bytes))
      return audio_bytes
    return None
  except Exception as e:  # pragma: no cover - external dependency
    logger.exception("TTS(OpenAI): synthesis failed: %s", e)
    return None


async def _synthesize_with_gemini(text: str, *, lang: str = "en") -> Optional[bytes]:
  """
  Gemini-based TTS helper using gemini-2.5-flash-lite-preview-tts.
  """
  if not GEMINI_API_KEY:
    logger.warning("TTS(Gemini): GEMINI_API_KEY / GENAI_API_KEY not set; skipping TTS generation")
    return None

  try:
    try:
      from google import genai as google_genai  # type: ignore
      from google.genai import types as genai_types_mod  # type: ignore
    except Exception as e:  # pragma: no cover - optional dependency
      logger.warning("TTS(Gemini): google-genai SDK not installed; cannot synthesize audio (%s)", e)
      return None

    global _gemini_client, _genai_types
    if _gemini_client is None:
      _gemini_client = google_genai.Client(api_key=GEMINI_API_KEY)
      _genai_types = genai_types_mod
      logger.info("TTS(Gemini): created google-genai client instance for TTS")

    def _call_gemini_tts() -> Optional[bytes]:
      try:
        client = _gemini_client
        types_mod = _genai_types
        cfg = types_mod.GenerateContentConfig(
          response_modalities=["AUDIO"],
          speech_config=types_mod.SpeechConfig(
            language_code=lang,
            voice_config=types_mod.VoiceConfig(
              prebuilt_voice_config=types_mod.PrebuiltVoiceConfig(voice_name="Kore")
            ),
          ),
        )
        logger.info(
          "TTS(Gemini): calling model=%s lang=%s text_len=%d",
          GEMINI_TTS_MODEL,
          lang,
          len(text),
        )
        resp = client.models.generate_content(
          model=GEMINI_TTS_MODEL,
          contents=text,
          config=cfg,
        )
        candidates = getattr(resp, "candidates", None) or []
        if not candidates:
          logger.warning("TTS(Gemini): no candidates in response")
          return None
        cand = candidates[0]
        content = getattr(cand, "content", None)
        parts = getattr(content, "parts", None) or []
        for part in parts:
          inline = getattr(part, "inline_data", None) or getattr(part, "inlineData", None) or {}
          # Extract raw audio data; may be bytes or base64 string
          if hasattr(inline, "data"):
            data = inline.data
          elif isinstance(inline, dict):
            data = inline.get("data")
          else:
            data = None
          if data is None:
            continue
          # If data is a str, assume base64; else treat as bytes-like
          if isinstance(data, str):
            try:
              pcm_bytes = base64.b64decode(data)
            except Exception:
              pcm_bytes = data.encode("utf-8")
          elif isinstance(data, (bytes, bytearray)):
            pcm_bytes = bytes(data)
          else:
            pcm_bytes = bytes(data)

          # Wrap raw PCM (16-bit LE, 24kHz mono) into WAV container
          bio = BytesIO()
          with wave.open(bio, "wb") as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(SAMPLE_WIDTH_BYTES)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(pcm_bytes)
          wav_bytes = bio.getvalue()
          return wav_bytes

        logger.warning("TTS(Gemini): no inline_data audio blob found in response")
        return None
      except Exception as exc:  # pragma: no cover - external dependency
        logger.exception("TTS(Gemini): synthesis failed (sync): %s", exc)
        return None

    audio_bytes = await asyncio.to_thread(_call_gemini_tts)
    if audio_bytes:
      logger.info("TTS(Gemini): received %d bytes from Gemini TTS", len(audio_bytes))
      return audio_bytes
    return None
  except Exception as e:  # pragma: no cover - external dependency
    logger.exception("TTS(Gemini): synthesis failed: %s", e)
    return None


def _guess_audio_mime(audio_bytes: bytes, default: str = "audio/wav") -> str:
  """
  Best-effort MIME detection for common audio formats.
  """
  if not audio_bytes or len(audio_bytes) < 4:
    return default

  header4 = audio_bytes[:4]
  header3 = audio_bytes[:3]
  header2 = audio_bytes[:2]

  # WAV header: "RIFF"
  if header4 == b"RIFF":
    return "audio/wav"

  # Ogg container: "OggS"
  if header4 == b"OggS":
    return "audio/ogg"

  # MP3: ID3 tag or frame sync 0xFF 0xFB/0xF3/0xF2
  if header3 == b"ID3" or header2 in (b"\xff\xfb", b"\xff\xf3", b"\xff\xf2"):
    return "audio/mpeg"

  return default


def audio_bytes_to_data_url(audio_bytes: bytes, mime: str = "audio/wav") -> str:
  """
  Helper to wrap audio bytes into a data: URL that the frontend can play
  directly in an <audio> element.

  MIME is auto-detected for common formats (wav/mp3/ogg) and falls back
  to the provided `mime` if detection is inconclusive.
  """
  effective_mime = _guess_audio_mime(audio_bytes, default=mime)
  if effective_mime != mime:
    logger.info(
      "TTS: adjusted audio MIME from %s to %s based on header", mime, effective_mime
    )
  b64 = base64.b64encode(audio_bytes).decode("ascii")
  return f"data:{effective_mime};base64,{b64}"


async def extract_voice_embedding(audio_bytes: bytes) -> Optional[list[float]]:
  """
  TODO hook for voice-based authentication (embeddings).

  - Input: raw audio bytes from the user.
  - Output: a numeric embedding suitable for comparison against a stored
    embedding for the logged-in user, or None if extraction fails.

  This is NOT implemented yet; it only defines the contract.
  """
  if not audio_bytes:
    return None

  logger.info("VOICE AUTH: extract_voice_embedding stub called (len=%d bytes)", len(audio_bytes))
  # Placeholder: return a deterministic dummy vector based on length only.
  # Real implementation should call a speaker embedding model.
  length = float(len(audio_bytes))
  return [length, 0.0, 0.0]
