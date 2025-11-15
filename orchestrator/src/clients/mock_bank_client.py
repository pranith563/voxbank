"""
Mock Bank Client
Client for communicating with mock bank backend
"""

import httpx
from typing import Dict, Any, Optional


class MockBankClient:
    """
    HTTP client for mock bank API
    """
    
    def __init__(self, base_url: str = "http://localhost:8001"):
        self.base_url = base_url
        self.client = httpx.AsyncClient(timeout=30.0)
    
    async def get_user_accounts(self, user_id: str) -> Dict[str, Any]:
        """
        Get user's accounts
        """
        try:
            response = await self.client.get(f"{self.base_url}/api/users/{user_id}/accounts")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    async def verify_otp(self, user_id: str, otp: str) -> bool:
        """
        Verify OTP with auth service
        """
        try:
            response = await self.client.post(
                f"{self.base_url}/api/auth/verify-otp",
                json={"user_id": user_id, "otp": otp}
            )
            return response.status_code == 200
        except:
            return False

