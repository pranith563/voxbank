from typing import Optional, Dict, Any

from pydantic import BaseModel


class TextRequest(BaseModel):
    transcript: str
    session_id: str
    user_id: str
    # optional flag to request a short/long reply
    reply_style: Optional[str] = "concise"  # 'concise' | 'detailed'


class TextResponse(BaseModel):
    response_text: str
    session_id: str
    requires_confirmation: bool = False
    meta: Optional[Dict[str, Any]] = None


class VoiceRequest(BaseModel):
    audio_data: Optional[str] = None  # base64-encoded audio (optional)
    transcript: Optional[str] = None  # optional pre-transcribed text
    session_id: str
    user_id: str


class ConfirmRequest(BaseModel):
    session_id: str
    confirm: bool
    user_id: Optional[str] = None


class VoiceResponse(BaseModel):
    response_text: str
    audio_url: Optional[str] = None
    session_id: str
    requires_confirmation: bool = False
    meta: Optional[Dict[str, Any]] = None


class RegisterRequest(BaseModel):
    username: str
    passphrase: str
    email: Optional[str] = None
    full_name: Optional[str] = None
    phone_number: Optional[str] = None
    address: Optional[str] = None
    date_of_birth: Optional[str] = None  # ISO date string
    audio_data: Optional[str] = None  # base64-encoded audio for embedding
    session_id: Optional[str] = None  # orchestrator session to bind user to


class RegisterResponse(BaseModel):
    status: str
    user: Dict[str, Any]


class LogoutRequest(BaseModel):
    session_id: str


class LoginRequest(BaseModel):
    username: str
    passphrase: str
    session_id: str

