# mcp-tools/src/app.py
"""
MCP Tools adapter (FastAPI + FastMCP)
- Exposes HTTP endpoints under /tools/* (backwards-compatible)
- Registers curated LLM-friendly tools with FastMCP and mounts them under /mcp
- Uses a shared X-MCP-TOOL-KEY header for simple auth
"""

import os
import logging
from typing import Optional, Dict, Any, List
from dotenv import load_dotenv, find_dotenv
from pydantic import BaseModel, Field
from fastapi import FastAPI, Header, HTTPException, status, Depends, Request
import httpx
import asyncio
import uuid

load_dotenv(find_dotenv(), override=False)

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
MCP_TOOL_KEY = os.getenv("MCP_TOOL_KEY", "mcp-test-key")
MOCK_BANK_BASE = os.getenv("MOCK_BANK_BASE_URL", "http://localhost:9000")
REQUEST_TIMEOUT = float(os.getenv("MCP_REQUEST_TIMEOUT", "10"))

logging.basicConfig(level=LOG_LEVEL)
logger = logging.getLogger("mcp_tools")

# Try to import fastmcp; if not installed we'll still provide the HTTP endpoints
try:
    from fastmcp import FastMCP
    FASTMCP_AVAILABLE = True
except Exception:
    FastMCP = None
    FASTMCP_AVAILABLE = False
    logger.warning("fastmcp not installed. MCP features will be disabled. Install via `pip install fastmcp` for LLM-friendly tools.")

# FastAPI app (keeps your existing HTTP surface)
app = FastAPI(title="MCP Tools (FastMCP-enabled)", version="1.1.0")

# Shared httpx client
_http_client = httpx.AsyncClient(timeout=REQUEST_TIMEOUT)


# -----------------------
# Security dependency for HTTP endpoints
# -----------------------
async def verify_mcp_key(x_mcp_tool_key: Optional[str] = Header(None)):
    if not x_mcp_tool_key or x_mcp_tool_key != MCP_TOOL_KEY:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid MCP tool key")
    return True


# -----------------------
# Helper: call mock-bank
# -----------------------
async def call_mock_bank(method: str, path: str, json: Dict[str, Any] = None) -> Any:
    url = MOCK_BANK_BASE.rstrip("/") + path
    try:
        resp = await _http_client.request(method, url, json=json)
        logger.debug("Mock-bank %s %s -> %s", method.upper(), url, resp.status_code)
        if resp.status_code >= 400:
            raise HTTPException(status_code=502, detail=f"mock-bank error: {resp.status_code} {resp.text}")
        return resp.json()
    except httpx.RequestError as e:
        logger.exception("HTTPX request failed: %s", e)
        raise HTTPException(status_code=502, detail="mock-bank unreachable")


# -----------------------
# HTTP request/response models (kept for backwards compatibility)
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
    transactions: List[TransactionItem]


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
# HTTP endpoints (backwards compatible)
# -----------------------
@app.post("/tools/balance", response_model=BalanceResponse, dependencies=[Depends(verify_mcp_key)])
async def tool_balance(req: BalanceRequest):
    data = await call_mock_bank("GET", f"/api/accounts/{req.account_number}")
    return BalanceResponse(
        account_number=req.account_number,
        balance=float(data.get("balance", 0.0)),
        available_balance=float(data.get("available_balance")) if data.get("available_balance") is not None else None,
        currency=data.get("currency", "USD"),
        status=data.get("status", "unknown")
    )


@app.post("/tools/transactions", response_model=TransactionsResponse, dependencies=[Depends(verify_mcp_key)])
async def tool_transactions(req: TransactionsRequest):
    data = await call_mock_bank("GET", f"/api/accounts/{req.account_number}/transactions?limit={req.limit}")
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
    if req.amount <= 0:
        raise HTTPException(status_code=400, detail="Invalid amount")
    payload = {
        "from_account_number": req.from_account_number,
        "to_account_number": req.to_account_number,
        "amount": req.amount,
        "currency": req.currency,
        "initiated_by_user_id": req.initiated_by_user_id,
        "reference": req.reference
    }
    data = await call_mock_bank("POST", "/api/transfer", json=payload)
    return TransferResponse(
        status=data.get("status", "failed"),
        transaction_reference=data.get("transaction_reference"),
        txn_id=data.get("txn_id"),
        message=data.get("message")
    )


# -----------------------
# Tools metadata endpoint (useful for discovery)
# -----------------------
@app.get("/tools/metadata")
async def tools_metadata():
    """
    Return a machine-readable description of curated tools.
    This helps the orchestrator / agent discover tool signatures when FastMCP client is not used.
    """
    return [
        {
            "name": "balance",
            "path": "/tools/balance",
            "method": "POST",
            "params": [{"name": "account_number", "type": "string", "required": True}],
            "description": "Get account balance by account number"
        },
        {
            "name": "transactions",
            "path": "/tools/transactions",
            "method": "POST",
            "params": [{"name": "account_number", "type": "string", "required": True}, {"name": "limit", "type": "integer", "required": False}],
            "description": "Fetch recent transactions"
        },
        {
            "name": "transfer",
            "path": "/tools/transfer",
            "method": "POST",
            "params": [
                {"name":"from_account_number","type":"string","required":True},
                {"name":"to_account_number","type":"string","required":True},
                {"name":"amount","type":"number","required":True},
                {"name":"currency","type":"string","required":False}
            ],
            "description": "Execute a funds transfer"
        }
    ]


# -----------------------
# FastMCP integration (curated tools)
# -----------------------
# If FastMCP is available, create an MCP instance from this FastAPI app
if FASTMCP_AVAILABLE:
    try:
        mcp = FastMCP.from_fastapi(app=app)  # auto-convert / integrate into same ASGI
        logger.info("FastMCP initialized and attached to FastAPI app.")
    except Exception as e:
        mcp = None
        logger.exception("Failed to initialize FastMCP from FastAPI app: %s", e)
else:
    mcp = None


# Example curated tool implementations using the mcp.tool decorator.
# These functions will be discoverable by the FastMCP runtime and are the
# LLM-friendly typed callables the agent can inspect and call.
if mcp:
    # balance tool
    @mcp.tool(name="balance", description="Get account balance by account number")
    async def balance_tool(account_number: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Returns {account_number, balance, available_balance, currency, status}
        """
        # call existing helper
        data = await call_mock_bank("GET", f"/api/accounts/{account_number}")
        return {
            "account_number": account_number,
            "balance": float(data.get("balance", 0.0)),
            "available_balance": float(data.get("available_balance")) if data.get("available_balance") is not None else None,
            "currency": data.get("currency", "USD"),
            "status": data.get("status", "unknown")
        }

    # transactions tool
    @mcp.tool(name="transactions", description="Fetch recent transactions for account")
    async def transactions_tool(account_number: str, limit: int = 10, session_id: Optional[str] = None) -> Dict[str, Any]:
        data = await call_mock_bank("GET", f"/api/accounts/{account_number}/transactions?limit={limit}")
        # return the list as-is; MCP client/agent can summarize
        return {"account_number": account_number, "transactions": data}

    # transfer tool
    @mcp.tool(name="transfer", description="Execute a funds transfer (high-risk)")
    async def transfer_tool(
        from_account_number: str,
        to_account_number: str,
        amount: float,
        currency: str = "USD",
        initiated_by_user_id: Optional[str] = None,
        reference: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        # Basic server-side policy: reject zero/negative amounts
        if amount <= 0:
            return {"status": "failed", "message": "Invalid amount"}
        payload = {
            "from_account_number": from_account_number,
            "to_account_number": to_account_number,
            "amount": amount,
            "currency": currency,
            "initiated_by_user_id": initiated_by_user_id,
            "reference": reference
        }
        # perform the transfer via mock-bank
        res = await call_mock_bank("POST", "/api/transfer", json=payload)
        # Keep the shape stable for agent consumption
        return {
            "status": res.get("status"),
            "transaction_reference": res.get("transaction_reference"),
            "txn_id": res.get("txn_id"),
            "message": res.get("message")
        }

    # Mount the MCP http app inside FastAPI under /mcp for HTTP-based discovery/calls
    try:
        app.mount("/mcp", mcp.http_app())
        logger.info("Mounted fastmcp http app at /mcp")
    except Exception as e:
        logger.exception("Failed to mount fastmcp http app: %s", e)


# -----------------------
# Shutdown
# -----------------------
@app.on_event("shutdown")
async def shutdown_event():
    try:
        await _http_client.aclose()
    except Exception:
        pass
