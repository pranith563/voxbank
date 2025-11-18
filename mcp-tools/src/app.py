# mcp-tools/src/app.py
"""
MCP Tools adapter (FastAPI + FastMCP)
- Exposes HTTP endpoints under /tools/* (backwards-compatible)
- Registers curated LLM-friendly tools with FastMCP and mounts them under /mcp
- Uses a shared X-MCP-TOOL-KEY header for simple auth
"""

from fastapi import FastAPI, Header, HTTPException, Request, status, Depends
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
import os
from dotenv import load_dotenv
import logging
import httpx
import asyncio
import uuid

load_dotenv()

MCP_TOOL_KEY = os.getenv("MCP_TOOL_KEY", "mcp-test-key")          # shared secret between orchestrator and MCP tools
MOCK_BANK_BASE = os.getenv("MOCK_BANK_BASE_URL", "http://localhost:9000")
REQUEST_TIMEOUT = float(os.getenv("MCP_REQUEST_TIMEOUT", "10"))

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("mcp_tools")

app = FastAPI(title="MCP Tools - Mock Bank Adapter", version="1.0.0")

# Shared async client (reused)
_http_client = httpx.AsyncClient(timeout=REQUEST_TIMEOUT)


# -----------------------
# Security dependency
# -----------------------
async def verify_mcp_key(x_mcp_tool_key: Optional[str] = Header(None)):
    if not x_mcp_tool_key or x_mcp_tool_key != MCP_TOOL_KEY:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid MCP tool key")
    return True


# -----------------------
# Request / Response models
# -----------------------

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


class TransactionsRequest(BaseModel):
    account_number: str
    limit: int = Field(default=10, ge=1, le=200)
    session_id: Optional[str] = None


class TransactionItem(BaseModel):
    transaction_id: str
    transaction_reference: str
    amount: float
    currency: str
    entry_type: str
    transaction_type: str
    status: str
    created_at: Optional[str] = None


class TransactionsResponse(BaseModel):
    account_number: str
    transactions: list[TransactionItem]


class TransferRequest(BaseModel):
    from_account_number: str
    to_account_number: str
    amount: float
    currency: str = "USD"
    initiated_by_user_id: Optional[str] = None
    reference: Optional[str] = None
    session_id: Optional[str] = None


class TransferResponse(BaseModel):
    status: str
    transaction_reference: str
    txn_id: Optional[str] = None
    message: Optional[str] = None


# -----------------------
# Helpers
# -----------------------
async def call_mock_bank(method: str, path: str, json: Dict[str, Any] = None):
    url = MOCK_BANK_BASE.rstrip("/") + path
    try:
        resp = await _http_client.request(method, url, json=json)
        logger.info("Mock-bank %s %s -> %s", method.upper(), url, resp.status_code)
        if resp.status_code >= 400:
            raise HTTPException(status_code=502, detail=f"mock-bank error: {resp.status_code} {resp.text}")
        return resp.json()
    except httpx.RequestError as e:
        logger.exception("HTTPX request failed: %s", e)
        raise HTTPException(status_code=502, detail="mock-bank unreachable")


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


@app.post("/tools/transactions", response_model=TransactionsResponse, dependencies=[Depends(verify_mcp_key)])
async def tool_transactions(req: TransactionsRequest):
    """
    Return latest transactions for an account.
    """
    # call mock-bank transactions
    data = await call_mock_bank("GET", f"/api/accounts/{req.account_number}/transactions?limit={req.limit}")
    # data should be a list of tx dicts
    txs = []
    for t in data:
        txs.append(TransactionItem(
            transaction_id=t.get("transaction_id"),
            transaction_reference=t.get("transaction_reference"),
            amount=float(t.get("amount") or 0.0),
            currency=t.get("currency"),
            entry_type=t.get("entry_type"),
            transaction_type=t.get("transaction_type"),
            status=t.get("status"),
            created_at=t.get("created_at")
        ))
    return TransactionsResponse(account_number=req.account_number, transactions=txs)


@app.post("/tools/transfer", response_model=TransferResponse, dependencies=[Depends(verify_mcp_key)])
async def tool_transfer(req: TransferRequest):
    """
    Execute a transfer via the mock bank API.
    This endpoint is the high-risk tool — orchestrator must obtain user confirmation before calling.
    """
    # Server-side basic validation
    if req.amount <= 0:
        raise HTTPException(status_code=400, detail="Invalid amount")

    # Call mock-bank transfer
    payload = {
        "from_account_number": req.from_account_number,
        "to_account_number": req.to_account_number,
        "amount": req.amount,
        "currency": req.currency,
        "initiated_by_user_id": req.initiated_by_user_id,
        "reference": req.reference
    }
    data = await call_mock_bank("POST", "/api/transfer", json=payload)
    # expected mock-bank returns a TransferOut-like dict
    return TransferResponse(
        status=data.get("status", "failed"),
        transaction_reference=data.get("transaction_reference"),
        txn_id=data.get("txn_id"),
        message=data.get("message")
    )


# -----------------------
# Health + shutdown
# -----------------------
@app.get("/health")
async def health():
    return {"status": "healthy", "mock_bank": MOCK_BANK_BASE}

@app.on_event("shutdown")
async def shutdown_event():
    await _http_client.aclose()