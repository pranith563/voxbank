"""
OTP Manager
Handles OTP generation, storage, and validation
"""

import random
import string
from typing import Dict, Optional
from datetime import datetime, timedelta


class OTPManager:
    """
    Manages OTP lifecycle
    """
    
    def __init__(self, expiry_minutes: int = 5):
        self.otp_store: Dict[str, Dict] = {}
        self.expiry = timedelta(minutes=expiry_minutes)
    
    def generate_otp(self, user_id: str) -> str:
        """
        Generate 6-digit OTP
        """
        otp = ''.join(random.choices(string.digits, k=6))
        self.otp_store[user_id] = {
            "otp": otp,
            "created_at": datetime.now(),
            "verified": False
        }
        return otp
    
    def verify_otp(self, user_id: str, otp: str) -> bool:
        """
        Verify OTP
        """
        if user_id not in self.otp_store:
            return False
        
        stored = self.otp_store[user_id]
        
        # Check expiry
        if datetime.now() - stored["created_at"] > self.expiry:
            del self.otp_store[user_id]
            return False
        
        # Check OTP match
        if stored["otp"] == otp:
            stored["verified"] = True
            return True
        
        return False
    
    def is_verified(self, user_id: str) -> bool:
        """
        Check if OTP was verified
        """
        if user_id not in self.otp_store:
            return False
        return self.otp_store[user_id].get("verified", False)

