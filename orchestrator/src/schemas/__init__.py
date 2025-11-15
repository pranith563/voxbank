"""
Schemas for orchestrator
"""

from pydantic import BaseModel
from typing import Optional, Dict, Any, List


class VoiceInput(BaseModel):
    audio_data: Optional[str] = None
    transcript: Optional[str] = None
    session_id: str
    user_id: str


class VoiceOutput(BaseModel):
    response_text: str
    audio_url: Optional[str] = None
    session_id: str
    requires_confirmation: bool = False
    confirmation_details: Optional[Dict[str, Any]] = None

