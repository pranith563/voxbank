"""
MCP Client
Client for communicating with MCP tool services
"""

import httpx
from typing import Dict, Any, Optional
import json


class MCPClient:
    """
    HTTP client for MCP tool services
    """
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.client = httpx.AsyncClient(timeout=30.0)
    
    async def call_tool(self, tool_name: str, params: Dict[str, Any], user_id: str) -> Dict[str, Any]:
        """
        Call an MCP tool service
        """
        tool_url = f"{self.base_url}/mcp-tools/{tool_name}/execute"
        
        payload = {
            "user_id": user_id,
            "params": params
        }
        
        try:
            response = await self.client.post(tool_url, json=payload)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    async def get_balance(self, user_id: str, account_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get account balance
        """
        return await self.call_tool("balance", {"account_id": account_id}, user_id)
    
    async def transfer_funds(self, user_id: str, from_account: str, to_account: str, amount: float, description: str = "") -> Dict[str, Any]:
        """
        Transfer funds between accounts
        """
        return await self.call_tool("transfer", {
            "from_account": from_account,
            "to_account": to_account,
            "amount": amount,
            "description": description
        }, user_id)
    
    async def get_transactions(self, user_id: str, account_id: str, limit: int = 10) -> Dict[str, Any]:
        """
        Get transaction history
        """
        return await self.call_tool("transactions", {
            "account_id": account_id,
            "limit": limit
        }, user_id)

