"""
Standalone FastMCP HTTP Server
--------------------------------
This runs the MCP tool server over HTTP ONLY (not stdio).

Tools exposed:
 - balance
 - transactions
 - transfer
 - list_tools

MCP endpoints created by FastMCP:
 - GET /mcp/manifest
 - POST /mcp/invoke/<tool_name>
 - POST /mcp/call   (JSON-RPC batch/traces)
"""

import os
import logging
import httpx
from dotenv import load_dotenv
from fastmcp import FastMCP


load_dotenv()

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
VOX_BANK_BASE = os.getenv("VOX_BANK_BASE_URL", "http://localhost:9000")
MCP_HOST = os.getenv("MCP_HOST", "0.0.0.0")
MCP_PORT = int(os.getenv("MCP_PORT", "9100"))
REQUEST_TIMEOUT = float(os.getenv("MCP_REQUEST_TIMEOUT", "10"))

logging.basicConfig(level=LOG_LEVEL)
logger = logging.getLogger("mcp_server")

# Shared HTTP client for mock-bank
_http = httpx.AsyncClient(timeout=REQUEST_TIMEOUT)

# Create MCP registry
mcp = FastMCP(name="vox_mcp")

# Canonical tool metadata (name -> description/params) exposed via list_tools.
# This is the primary source for orchestrator prompts and tool validation.
TOOL_METADATA = {
    "balance": {
        "description": "Get account balance",
        "params": {
            "account_number": {"type": "string", "required": True},
        },
    },
    "transactions": {
        "description": "Fetch recent transactions for an account",
        "params": {
            "account_number": {"type": "string", "required": True},
            "limit": {"type": "integer", "required": False},
        },
    },
    "transfer": {
        "description": "Execute money transfer (HIGH RISK)",
        "high_risk": True,
        "params": {
            "from_account_number": {"type": "string", "required": True},
            "to_account_number": {"type": "string", "required": True},
            "amount": {"type": "number", "required": True},
            "currency": {"type": "string", "required": False, "default": "USD"},
            "initiated_by_user_id": {"type": "string", "required": False},
            "reference": {"type": "string", "required": False},
        },
    },
    "register_user": {
        "description": "Register a new VoxBank user with username and passphrase",
        "params": {
            "username": {"type": "string", "required": True},
            "passphrase": {"type": "string", "required": True},
            "email": {"type": "string", "required": False},
            "full_name": {"type": "string", "required": False},
            "phone_number": {"type": "string", "required": False},
            "audio_embedding": {"type": "array[number]", "required": False},
        },
    },
    "login_user": {
        "description": "Validate username + passphrase and return user_id",
        "params": {
            "username": {"type": "string", "required": True},
            "passphrase": {"type": "string", "required": True},
        },
    },
    "set_user_audio_embedding": {
        "description": "Set or replace the audio embedding for a user",
        "params": {
            "user_id": {"type": "string", "required": True},
            "audio_embedding": {"type": "array[number]", "required": True},
        },
    },
    "get_user_profile": {
        "description": "Fetch a user's profile by user_id",
        "params": {
            "user_id": {"type": "string", "required": True},
        },
    },
    "list_tools": {
        "description": "List available MCP tools",
        "params": {},
    },
}

# -----------------------------
# Tools
# -----------------------------

@mcp.tool(name="balance", description="Get account balance")
async def balance(account_number: str) -> dict:
    url = f"{VOX_BANK_BASE}/api/accounts/{account_number}"
    try:
        r = await _http.get(url)
        r.raise_for_status()
        data = r.json()
        return {
            "account_number": account_number,
            "balance": float(data.get("balance", 0)),
            "available_balance": data.get("available_balance"),
            "currency": data.get("currency"),
            "status": data.get("status")
        }
    except Exception as e:
        logger.error("balance_tool error: %s", e)
        return {"error": str(e)}


@mcp.tool(name="transactions", description="Fetch recent transactions")
async def transactions(account_number: str, limit: int = 10) -> dict:
    url = f"{VOX_BANK_BASE}/api/accounts/{account_number}/transactions?limit={limit}"
    try:
        r = await _http.get(url)
        r.raise_for_status()
        return {
            "account_number": account_number,
            "transactions": r.json()
        }
    except Exception as e:
        logger.error("transactions_tool error: %s", e)
        return {"error": str(e)}


@mcp.tool(name="transfer", description="Execute money transfer (high risk)")
async def transfer(
    from_account_number: str,
    to_account_number: str,
    amount: float,
    currency: str = "USD",
    initiated_by_user_id: str = None,
    reference: str = None
) -> dict:
    payload = {
        "from_account_number": from_account_number,
        "to_account_number": to_account_number,
        "amount": amount,
        "currency": currency,
        "initiated_by_user_id": initiated_by_user_id,
        "reference": reference
    }

    url = f"{VOX_BANK_BASE}/api/transfer"
    try:
        r = await _http.post(url, json=payload)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logger.error("transfer_tool error: %s", e)
        return {"error": str(e)}


@mcp.tool(name="register_user", description="Register a new VoxBank user with username and passphrase")
async def register_user(
    username: str,
    passphrase: str,
    email: str | None = None,
    full_name: str | None = None,
    phone_number: str | None = None,
    audio_embedding: list[float] | None = None,
) -> dict:
    """
    Wraps POST /api/users on the mock-bank service.
    """
    url = f"{VOX_BANK_BASE}/api/users"
    payload = {
        "username": username,
        "passphrase": passphrase,
        "email": email,
        "full_name": full_name,
        "phone_number": phone_number,
        "audio_embedding": audio_embedding,
    }
    try:
        r = await _http.post(url, json=payload)
        r.raise_for_status()
        data = r.json()
        return {"status": "success", "user": data}
    except httpx.HTTPStatusError as e:
        logger.error("register_user error: %s", e)
        detail = e.response.json().get("detail") if e.response.headers.get("content-type", "").startswith("application/json") else str(e)
        return {"status": "error", "message": detail, "http_status": e.response.status_code}
    except Exception as e:
        logger.error("register_user unexpected error: %s", e)
        return {"status": "error", "message": str(e)}


@mcp.tool(name="login_user", description="Validate username + passphrase and return user_id")
async def login_user(username: str, passphrase: str) -> dict:
    """
    Wraps POST /api/login on the mock-bank service.
    """
    url = f"{VOX_BANK_BASE}/api/login"
    payload = {"username": username, "passphrase": passphrase}
    try:
        r = await _http.post(url, json=payload)
        r.raise_for_status()
        data = r.json()
        return {
            "status": data.get("status", "ok"),
            "user_id": data.get("user_id"),
            "username": data.get("username"),
            "has_audio_embedding": data.get("has_audio_embedding", False),
            "message": data.get("message"),
        }
    except httpx.HTTPStatusError as e:
        logger.error("login_user error: %s", e)
        detail = e.response.json().get("detail") if e.response.headers.get("content-type", "").startswith("application/json") else str(e)
        return {"status": "error", "message": detail, "http_status": e.response.status_code}
    except Exception as e:
        logger.error("login_user unexpected error: %s", e)
        return {"status": "error", "message": str(e)}


@mcp.tool(name="set_user_audio_embedding", description="Set or replace the audio embedding for a user")
async def set_user_audio_embedding(user_id: str, audio_embedding: list[float]) -> dict:
    """
    Wraps PUT /api/users/{user_id}/audio-embedding.
    """
    url = f"{VOX_BANK_BASE}/api/users/{user_id}/audio-embedding"
    payload = {"audio_embedding": audio_embedding}
    try:
        r = await _http.put(url, json=payload)
        r.raise_for_status()
        data = r.json()
        return {"status": "success", "user": data}
    except httpx.HTTPStatusError as e:
        logger.error("set_user_audio_embedding error: %s", e)
        detail = e.response.json().get("detail") if e.response.headers.get("content-type", "").startswith("application/json") else str(e)
        return {"status": "error", "message": detail, "http_status": e.response.status_code}
    except Exception as e:
        logger.error("set_user_audio_embedding unexpected error: %s", e)
        return {"status": "error", "message": str(e)}


@mcp.tool(name="get_user_profile", description="Fetch a user's profile by user_id")
async def get_user_profile(user_id: str) -> dict:
    """
    Wraps GET /api/users/{user_id} and returns profile info.
    """
    url = f"{VOX_BANK_BASE}/api/users/{user_id}"
    try:
        r = await _http.get(url)
        r.raise_for_status()
        data = r.json()
        return {"status": "success", "user": data}
    except httpx.HTTPStatusError as e:
        logger.error("get_user_profile error: %s", e)
        detail = e.response.json().get("detail") if e.response.headers.get("content-type", "").startswith("application/json") else str(e)
        return {"status": "error", "message": detail, "http_status": e.response.status_code}
    except Exception as e:
        logger.error("get_user_profile unexpected error: %s", e)
        return {"status": "error", "message": str(e)}


@mcp.tool(name="list_tools", description="List available MCP tools")
def list_tools() -> dict:
    """Return metadata for all available MCP tools."""
    try:
        # Prefer TOOL_METADATA as the canonical source. This ensures the
        # orchestrator can build prompts and validation rules dynamically.
        return {"tools": TOOL_METADATA}
    except Exception as e:
        logger.error(f"Error in list_tools: {e}")
        return {"tools": {}}


# -----------------------------
# Start MCP HTTP server
# -----------------------------
def main():
    logger.info(f"Starting MCP HTTP server on http://{MCP_HOST}:{MCP_PORT}")
    
    # The magic line:
    mcp.run(
        host=MCP_HOST,
        port=MCP_PORT,
        transport="streamable-http"
    )

if __name__ == "__main__":
    main()
