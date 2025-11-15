"""
Auth Service
OTP + Voice Biometrics (mock)
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

app = FastAPI(title="VoxBank Auth Service")


class OTPRequest(BaseModel):
    user_id: str
    phone: str


class OTPVerifyRequest(BaseModel):
    user_id: str
    otp: str


class BiometricVerifyRequest(BaseModel):
    user_id: str
    audio_data: str


class OTPResponse(BaseModel):
    success: bool
    otp_sent: bool
    error: Optional[str] = None


class VerifyResponse(BaseModel):
    success: bool
    verified: bool
    error: Optional[str] = None


@app.post("/api/otp/send", response_model=OTPResponse)
async def send_otp(request: OTPRequest):
    """
    Send OTP to user
    """
    try:
        # TODO: Implement OTP sending
        return OTPResponse(success=True, otp_sent=True)
    except Exception as e:
        return OTPResponse(success=False, otp_sent=False, error=str(e))


@app.post("/api/otp/verify", response_model=VerifyResponse)
async def verify_otp(request: OTPVerifyRequest):
    """
    Verify OTP
    """
    try:
        # TODO: Implement OTP verification
        return VerifyResponse(success=True, verified=True)
    except Exception as e:
        return VerifyResponse(success=False, verified=False, error=str(e))


@app.post("/api/biometric/verify", response_model=VerifyResponse)
async def verify_biometric(request: BiometricVerifyRequest):
    """
    Verify voice biometric
    """
    try:
        # TODO: Implement biometric verification
        return VerifyResponse(success=True, verified=True)
    except Exception as e:
        return VerifyResponse(success=False, verified=False, error=str(e))


@app.get("/api/health")
async def health():
    return {"status": "healthy"}

