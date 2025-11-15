"""
Balance Inquiry MCP Tool Service
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import httpx

app = FastAPI(title="Balance MCP Tool")


class BalanceRequest(BaseModel):
    account_number: str
    session_id: Optional[str] = None
    user_id: Optional[str] = None


class BalanceResponse(BaseModel):
    account_number: str
    balance: float
    available_balance: Optional[float] = None
    currency: str
    status: str


# -----------------------
# Tools endpoints
# -----------------------

@app.post("/tools/balance", response_model=BalanceResponse, dependencies=[Depends(verify_mcp_key)])
async def tool_balance(req: BalanceRequest):
    """
    Return account balance for a given account_number.
    Orchestrator should call this to fetch latest balance before confirming high-risk actions.
    """
    # Query mock bank account endpoint
    data = await call_mock_bank("GET", f"/api/accounts/{req.account_number}")
    # data expected to match AccountOut serialization
    return BalanceResponse(
        account_number=req.account_number,
        balance=float(data.get("balance", 0.0)),
        available_balance=float(data.get("available_balance")) if data.get("available_balance") is not None else None,
        currency=data.get("currency", "USD"),
        status=data.get("status", "unknown")
    )

