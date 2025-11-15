"""
Mock SMS Provider
Simulates SMS sending for OTP
"""

from typing import Dict


class MockSMSProvider:
    """
    Mock SMS provider for development/testing
    """
    
    def send_otp(self, phone: str, otp: str) -> Dict[str, bool]:
        """
        Send OTP via SMS (mock)
        """
        # In production, this would call actual SMS gateway
        print(f"[MOCK SMS] Sending OTP {otp} to {phone}")
        return {"success": True, "message_id": "MOCK123"}

