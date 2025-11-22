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

from typing import Optional, Dict, Any
import base64
import logging
import os
import asyncio
import time
from io import BytesIO
import wave
import tempfile
import subprocess
import shutil

from context.voice_profile import get_voice_profile

logger = logging.getLogger("voxbank.voice_processing")

# Hint for HF hub to behave nicely on Windows (also used in voice_auth)
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")

# Global TTS backend selector: "openai", "gemini", or "local"
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

# Local TTS (SpeechBrain) configuration
LOCAL_TTS_MODEL = os.getenv("LOCAL_TTS_MODEL", "speechbrain/tts-tacotron2-ljspeech")
LOCAL_VOCODER_MODEL = os.getenv("LOCAL_VOCODER_MODEL", "speechbrain/tts-hifigan-ljspeech")
LOCAL_SAMPLE_RATE = 22050  # typical for LJSpeech models

# Lazy-initialized SDK clients / models (shared across calls)
_openai_client = None
_gemini_client = None
_genai_types = None
_local_tts_model = None
_local_vocoder = None
_tts_initialized = False


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


async def synthesize_text_to_audio(
  text: str,
  *,
  session_profile: Optional[Dict[str, Any]] = None,
  reply_style: str = "warm",
  lang: str = "en",
) -> Optional[bytes]:
  """
  Text-to-speech implementation using the configured backend.

  - Input: assistant reply text.
  - Output: audio bytes or None if TTS is not configured/available.

  Backend is selected via session voice_profile engine with fallback to cloud:
    - "speechbrain" (default local)
    - "piper" (local CLI/binding)
    - "openai" (cloud)
    - "gemini" (cloud)
  """
  if not text:
    return None

  profile = get_voice_profile(session_profile)
  engine = (profile.get("engine") or "speechbrain").lower()
  voice_id = profile.get("voice_id") or "en_IN_female1"
  style = profile.get("style") or reply_style

  provider_used = engine
  audio: Optional[bytes] = None
  started = time.perf_counter()

  # Local-first
  if engine == "piper":
    audio = await _synthesize_with_piper(text, voice_id=voice_id, style=style)
  elif engine in {"speechbrain", "local"}:
    audio = await _synthesize_with_local(text, lang=lang)
  elif engine == "gemini":
    audio = await _synthesize_with_gemini(text, lang=lang)
  elif engine == "openai":
    audio = await _synthesize_with_openai(text, lang=lang)

  # Fallback to cloud if local failed or unsupported engine
  if audio is None:
    provider_used = "cloud_fallback"
    audio = await _synthesize_with_cloud_tts(text, voice_id=voice_id, style=style, lang=lang)

  elapsed_ms = (time.perf_counter() - started) * 1000.0
  logger.info(
    "TTS: provider=%s engine=%s duration_ms=%.1f success=%s bytes=%s",
    provider_used,
    engine,
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


async def _synthesize_with_local(text: str, *, lang: str = "en") -> Optional[bytes]:
  """
  Local TTS helper using SpeechBrain Tacotron2 + HiFi-GAN.

  This runs entirely on the local machine and can reduce latency by
  avoiding network calls once models are downloaded.
  """
  try:
    # Import heavy dependencies lazily so they don't affect startup
    from speechbrain.inference.TTS import Tacotron2  # type: ignore
    from speechbrain.inference.vocoders import HIFIGAN  # type: ignore
    import torch  # type: ignore
    import numpy as np  # type: ignore
  except Exception as e:  # pragma: no cover - optional dependency
    logger.warning("TTS(Local): SpeechBrain / torch not available; cannot synthesize audio (%s)", e)
    return None

  global _local_tts_model, _local_vocoder
  if _local_tts_model is None or _local_vocoder is None:
    try:
      logger.info(
        "TTS(Local): loading SpeechBrain models tts=%s vocoder=%s",
        LOCAL_TTS_MODEL,
        LOCAL_VOCODER_MODEL,
      )
      _local_tts_model = Tacotron2.from_hparams(
        source=LOCAL_TTS_MODEL,
        savedir="pretrained_models/tts-tacotron2-ljspeech",
      )
      _local_vocoder = HIFIGAN.from_hparams(
        source=LOCAL_VOCODER_MODEL,
        savedir="pretrained_models/tts-hifigan-ljspeech",
      )
      logger.info("TTS(Local): SpeechBrain models loaded")
    except Exception as e:  # pragma: no cover - model download/initialization issues
      logger.exception("TTS(Local): failed to load SpeechBrain models: %s", e)
      return None

  def _call_local_tts() -> Optional[bytes]:
    try:
      tts_model = _local_tts_model
      vocoder = _local_vocoder
      if tts_model is None or vocoder is None:
        return None

      logger.info("TTS(Local): synthesizing text_len=%d", len(text))
      with torch.no_grad():
        mel_output, mel_length, alignment = tts_model.encode_text(text)
        waveforms = vocoder.decode_batch(mel_output)  # (batch, 1, time)

      # Convert to mono waveform numpy array
      waveform = waveforms[0].squeeze().cpu().numpy()
      # Normalize to int16 PCM
      waveform = waveform / max(np.max(np.abs(waveform)), 1e-8)
      pcm_int16 = (waveform * 32767.0).astype("<i2")

      bio = BytesIO()
      with wave.open(bio, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(LOCAL_SAMPLE_RATE)
        wf.writeframes(pcm_int16.tobytes())
      return bio.getvalue()
    except Exception as exc:  # pragma: no cover - local TTS runtime issues
      logger.exception("TTS(Local): synthesis failed (sync): %s", exc)
      return None

  try:
    audio_bytes = await asyncio.to_thread(_call_local_tts)
    if audio_bytes:
      logger.info("TTS(Local): produced %d bytes of audio", len(audio_bytes))
      return audio_bytes
    return None
  except Exception as e:
    logger.exception("TTS(Local): synthesis failed: %s", e)
    return None


async def _synthesize_with_piper(text: str, *, voice_id: str, style: str = "warm") -> Optional[bytes]:
  """
  Piper CLI/binding helper. Uses subprocess if available.
  """
  piper_exe = shutil.which("piper")
  if not piper_exe:
    logger.warning("TTS(Piper): 'piper' executable not found in PATH; skipping")
    return None

  # Piper expects a voice model file; callers should supply a model path in voice_id.
  # We use a blocking subprocess executed in a worker thread for Windows compatibility.
  if not voice_id:
    logger.warning("TTS(Piper): no voice_id provided")
    return None

  async def _run_piper() -> Optional[bytes]:
    try:
      with tempfile.TemporaryDirectory() as tmpdir:
        out_path = os.path.join(tmpdir, "piper_out.wav")
        cmd = [piper_exe, "-m", voice_id, "-f", out_path]
        logger.info("TTS(Piper): invoking %s", cmd)
        try:
          result = await asyncio.to_thread(
            subprocess.run,
            cmd,
            input=text.encode("utf-8"),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
          )
        except FileNotFoundError:
          logger.warning("TTS(Piper): voice model file not found for voice_id=%s", voice_id)
          return None

        if result.returncode != 0:
          logger.warning("TTS(Piper): command failed with code %s", result.returncode)
          return None
        if not os.path.exists(out_path):
          logger.warning("TTS(Piper): output file not found after synthesis")
          return None
        with open(out_path, "rb") as f:
          audio_bytes = f.read()
        logger.info("TTS(Piper): produced %d bytes using model=%s", len(audio_bytes), voice_id)
        return audio_bytes
    except Exception as exc:  # pragma: no cover - external dependency
      logger.exception("TTS(Piper): synthesis failed: %s", exc)
      return None

  return await _run_piper()


async def _synthesize_with_cloud_tts(
  text: str,
  *,
  voice_id: Optional[str],
  style: str,
  lang: str = "en",
) -> Optional[bytes]:
  """
  Thin wrapper to reuse OpenAI/Gemini helpers as cloud fallback.
  """
  requested = (TTS_PROVIDER or "openai").lower()
  audio = None
  if requested == "gemini":
    audio = await _synthesize_with_gemini(text, lang=lang)
  if audio is None and requested in {"openai", "speechbrain", "local", "piper"}:
    audio = await _synthesize_with_openai(text, lang=lang)
  if audio is None and requested != "gemini":
    # final fallback to Gemini if configured provider failed and GEMINI key exists
    audio = await _synthesize_with_gemini(text, lang=lang)
  return audio


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


# ---------------------------------------------------------------------------
# Initialization helpers
# ---------------------------------------------------------------------------


def _ensure_openai_client() -> bool:
  global _openai_client
  if _openai_client is not None:
    return True
  if not OPENAI_API_KEY:
    logger.warning("TTS(OpenAI): OPENAI_API_KEY not set; cannot initialize OpenAI client")
    return False
  try:
    from openai import OpenAI  # type: ignore
    _openai_client = OpenAI(api_key=OPENAI_API_KEY)
    logger.info("TTS(OpenAI): client initialized")
    return True
  except Exception as exc:  # pragma: no cover - optional dependency
    logger.exception("TTS(OpenAI): failed to initialize client: %s", exc)
    return False


async def _ensure_gemini_client() -> bool:
  global _gemini_client, _genai_types
  if _gemini_client is not None:
    return True
  if not GEMINI_API_KEY:
    logger.warning("TTS(Gemini): GEMINI_API_KEY not set; cannot initialize Gemini client")
    return False
  try:
    from google import genai  # type: ignore
    from google.genai import types as genai_types  # type: ignore
  except Exception as exc:  # pragma: no cover - optional dependency
    logger.warning("TTS(Gemini): google-genai not installed; cannot initialize Gemini client (%s)", exc)
    return False

  def _init() -> bool:
    try:
      client = genai.Client(api_key=GEMINI_API_KEY)
      _gemini_client = client
      _genai_types = genai_types
      logger.info("TTS(Gemini): client initialized")
      return True
    except Exception as exc:  # pragma: no cover
      logger.exception("TTS(Gemini): failed to initialize client: %s", exc)
      return False

  return await asyncio.to_thread(_init)


async def _ensure_local_models() -> bool:
  global _local_tts_model, _local_vocoder
  if _local_tts_model is not None and _local_vocoder is not None:
    return True
  try:
    from speechbrain.inference.TTS import Tacotron2  # type: ignore
    from speechbrain.inference.vocoders import HIFIGAN  # type: ignore
  except Exception as exc:  # pragma: no cover - optional dependency
    logger.warning("TTS(Local): SpeechBrain not installed; cannot initialize local TTS (%s)", exc)
    return False

  def _load_models() -> bool:
    try:
      logger.info(
        "TTS(Local): loading SpeechBrain models tts=%s vocoder=%s",
        LOCAL_TTS_MODEL,
        LOCAL_VOCODER_MODEL,
      )
      _local_tts_model = Tacotron2.from_hparams(
        source=LOCAL_TTS_MODEL,
        savedir="pretrained_models/tts-tacotron2-ljspeech",
      )
      _local_vocoder = HIFIGAN.from_hparams(
        source=LOCAL_VOCODER_MODEL,
        savedir="pretrained_models/tts-hifigan-ljspeech",
      )
      logger.info("TTS(Local): models loaded")
      return True
    except Exception as exc:  # pragma: no cover
      logger.exception("TTS(Local): failed to load models: %s", exc)
      return False

  return await asyncio.to_thread(_load_models)


async def initialize_tts_backends(warm_fallback: bool = True) -> dict:
  """
  Warm up the configured TTS backend (and optional fallback) during app startup.
  Returns a dict of provider -> bool indicating initialization success.
  """
  global _tts_initialized
  if _tts_initialized:
    return {"status": "already_initialized"}

  requested = (TTS_PROVIDER or "openai").lower()
  statuses = {}

  if requested == "gemini":
    statuses["gemini"] = await _ensure_gemini_client()
  elif requested in {"local", "speechbrain"}:
    statuses["local"] = await _ensure_local_models()
  else:
    statuses["openai"] = _ensure_openai_client()

  # Warm OpenAI as fallback if configured provider is not OpenAI
  if warm_fallback and requested != "openai":
    statuses["openai"] = _ensure_openai_client()

  _tts_initialized = True
  return statuses
