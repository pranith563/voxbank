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

logger = logging.getLogger("voxbank.voice_processing")


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
  Stub TTS implementation.

  - Input: assistant reply text.
  - Output: audio bytes (e.g. WAV) or None if not available.

  TODO:
  - Integrate a real TTS engine (e.g., Gemini TTS as in voiceConversion.py,
    or any local TTS library) and return generated audio bytes.
  """
  if not text:
    return None

  logger.info("TTS: stub called for %d chars (lang=%s); returning None", len(text), lang)
  # No actual audio yet
  return None


def audio_bytes_to_data_url(audio_bytes: bytes, mime: str = "audio/wav") -> str:
  """
  Helper to wrap audio bytes into a data: URL that the frontend can play
  directly in an <audio> element.
  """
  b64 = base64.b64encode(audio_bytes).decode("ascii")
  return f"data:{mime};base64,{b64}"


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
  # Placeholder: return a dummy small vector
  return [0.0, 0.0, 0.0]

