"""
Schemas for Auth Service
"""

from pydantic import BaseModel
from typing import Optional


class OTPRequest(BaseModel):
    user_id: str
    phone: str


class OTPVerifyRequest(BaseModel):
    user_id: str
    otp: str


class BiometricEnrollRequest(BaseModel):
    user_id: str
    audio_data: str


class BiometricVerifyRequest(BaseModel):
    user_id: str
    audio_data: str


class AuthResponse(BaseModel):
    success: bool
    verified: bool
    score: Optional[float] = None
    error: Optional[str] = None

