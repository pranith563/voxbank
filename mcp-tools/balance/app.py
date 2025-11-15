"""
Balance Inquiry MCP Tool Service
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import httpx

app = FastAPI(title="Balance MCP Tool")


class BalanceRequest(BaseModel):
    user_id: str
    params: dict


class BalanceResponse(BaseModel):
    success: bool
    balance: Optional[float] = None
    account_id: Optional[str] = None
    currency: str = "INR"
    error: Optional[str] = None


@app.post("/execute", response_model=BalanceResponse)
async def get_balance(request: BalanceRequest):
    """
    Get account balance for user
    """
    try:
        # TODO: Call mock-bank API
        # For now, return mock data
        return BalanceResponse(
            success=True,
            balance=50000.00,
            account_id="ACC001",
            currency="INR"
        )
    except Exception as e:
        return BalanceResponse(success=False, error=str(e))


@app.get("/health")
async def health():
    return {"status": "healthy"}

