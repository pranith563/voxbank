"""
Voice profile helpers for TTS configuration.
"""

from typing import Dict, Any

DEFAULT_VOICE_PROFILE: Dict[str, Any] = {
    "engine": "gemini",
    "voice_id": "en_IN_female1",
    "style": "warm",
}


def merge_voice_profile(profile: Dict[str, Any] | None) -> Dict[str, Any]:
    """
    Merge a provided profile with defaults to ensure required keys exist.
    """
    merged = dict(DEFAULT_VOICE_PROFILE)
    if profile:
        merged.update({k: v for k, v in profile.items() if v is not None})
    return merged


def get_voice_profile(session_profile: Dict[str, Any] | None) -> Dict[str, Any]:
    """
    Resolve the effective voice profile from a session profile-like dict.
    """
    vp = None
    if session_profile:
        vp = session_profile.get("voice_profile") or session_profile.get("voice") or {}
    return merge_voice_profile(vp if isinstance(vp, dict) else {})
