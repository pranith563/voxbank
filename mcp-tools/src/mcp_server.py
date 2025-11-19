"""
Standalone FastMCP HTTP Server
--------------------------------
HTTP-only MCP tool server for VoxBank.

Tools exposed:
 - balance
 - transactions
 - transfer
 - register_user
 - login_user
 - set_user_audio_embedding
 - get_user_profile
 - list_tools

MCP endpoints created by FastMCP:
 - GET /mcp/manifest
 - POST /mcp/invoke/<tool_name>
 - POST /mcp/call   (JSON-RPC batch/traces)
"""

import logging
import os

import httpx
from dotenv import load_dotenv
from fastmcp import FastMCP

from logging_config import get_logger, setup_logging

load_dotenv()

VOX_BANK_BASE = os.getenv("VOX_BANK_BASE_URL", "http://localhost:9000")
MCP_HOST = os.getenv("MCP_HOST", "0.0.0.0")
MCP_PORT = int(os.getenv("MCP_PORT", "9100"))
REQUEST_TIMEOUT = float(os.getenv("MCP_REQUEST_TIMEOUT", "10"))

# Configure structured logging (file + console)
setup_logging()
logger = get_logger("mcp_server")

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


def _mock_bank_url(path: str) -> str:
    base = VOX_BANK_BASE.rstrip("/")
    return f"{base}{path}"


async def _get_json(url: str, *, params: dict | None = None) -> dict:
    """
    Helper to GET JSON from mock-bank with consistent error handling.
    """
    try:
        logger.info("HTTP GET %s params=%s", url, params)
        r = await _http.get(url, params=params)
        r.raise_for_status()
        return {"ok": True, "data": r.json()}
    except httpx.HTTPStatusError as e:
        logger.error(
            "HTTP GET failed url=%s status=%s detail=%s",
            url,
            e.response.status_code,
            e,
        )
        detail = (
            e.response.json().get("detail")
            if e.response.headers.get("content-type", "").startswith("application/json")
            else str(e)
        )
        return {
            "ok": False,
            "status": e.response.status_code,
            "message": detail,
        }
    except Exception as e:  # pragma: no cover - network/lower level errors
        logger.error("HTTP GET unexpected error url=%s error=%s", url, e)
        return {"ok": False, "status": None, "message": str(e)}


async def _post_json(url: str, payload: dict) -> dict:
    """
    Helper to POST JSON to mock-bank with consistent error handling.
    """
    try:
        logger.info("HTTP POST %s payload_keys=%s", url, list(payload.keys()))
        r = await _http.post(url, json=payload)
        r.raise_for_status()
        return {"ok": True, "data": r.json()}
    except httpx.HTTPStatusError as e:
        logger.error(
            "HTTP POST failed url=%s status=%s detail=%s",
            url,
            e.response.status_code,
            e,
        )
        detail = (
            e.response.json().get("detail")
            if e.response.headers.get("content-type", "").startswith("application/json")
            else str(e)
        )
        return {
            "ok": False,
            "status": e.response.status_code,
            "message": detail,
        }
    except Exception as e:  # pragma: no cover
        logger.error("HTTP POST unexpected error url=%s error=%s", url, e)
        return {"ok": False, "status": None, "message": str(e)}


async def _put_json(url: str, payload: dict) -> dict:
    """
    Helper to PUT JSON to mock-bank with consistent error handling.
    """
    try:
        logger.info("HTTP PUT %s payload_keys=%s", url, list(payload.keys()))
        r = await _http.put(url, json=payload)
        r.raise_for_status()
        return {"ok": True, "data": r.json()}
    except httpx.HTTPStatusError as e:
        logger.error(
            "HTTP PUT failed url=%s status=%s detail=%s",
            url,
            e.response.status_code,
            e,
        )
        detail = (
            e.response.json().get("detail")
            if e.response.headers.get("content-type", "").startswith("application/json")
            else str(e)
        )
        return {
            "ok": False,
            "status": e.response.status_code,
            "message": detail,
        }
    except Exception as e:  # pragma: no cover
        logger.error("HTTP PUT unexpected error url=%s error=%s", url, e)
        return {"ok": False, "status": None, "message": str(e)}


# -----------------------------
# Tools
# -----------------------------


@mcp.tool(name="balance", description="Get account balance")
async def balance(account_number: str) -> dict:
    url = _mock_bank_url(f"/api/accounts/{account_number}")
    result = await _get_json(url)
    if not result["ok"]:
        return {"error": result["message"], "status": result.get("status")}

    data = result["data"]
    return {
        "account_number": account_number,
        "balance": float(data.get("balance", 0)),
        "available_balance": data.get("available_balance"),
        "currency": data.get("currency"),
        "status": data.get("status"),
    }


@mcp.tool(name="transactions", description="Get recent transactions")
async def transactions(account_number: str, limit: int = 10) -> dict:
    url = _mock_bank_url(f"/api/accounts/{account_number}/transactions")
    result = await _get_json(url, params={"limit": limit})
    if not result["ok"]:
        return {"error": result["message"], "status": result.get("status")}

    return {
        "account_number": account_number,
        "transactions": result["data"],
    }


@mcp.tool(name="transfer", description="Transfer funds between accounts")
async def transfer(
    from_account_number: str,
    to_account_number: str,
    amount: float,
    currency: str = "USD",
    initiated_by_user_id: str | None = None,
    reference: str | None = None,
) -> dict:
    url = _mock_bank_url("/api/transfer")
    payload = {
        "from_account_number": from_account_number,
        "to_account_number": to_account_number,
        "amount": amount,
        "currency": currency,
        "initiated_by_user_id": initiated_by_user_id,
        "reference": reference,
    }
    result = await _post_json(url, payload)
    if not result["ok"]:
        return {"error": result["message"], "status": result.get("status")}

    return result["data"]


@mcp.tool(name="register_user", description="Register a new VoxBank user")
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
    url = _mock_bank_url("/api/users")
    payload = {
        "username": username,
        "passphrase": passphrase,
        "email": email,
        "full_name": full_name,
        "phone_number": phone_number,
        "audio_embedding": audio_embedding,
    }
    result = await _post_json(url, payload)
    if not result["ok"]:
        return {
            "status": "error",
            "message": result["message"],
            "http_status": result.get("status"),
        }

    data = result["data"]
    return {"status": "success", "user": data}


@mcp.tool(name="login_user", description="Validate username + passphrase and return user_id")
async def login_user(username: str, passphrase: str) -> dict:
    """
    Wraps POST /api/login on the mock-bank service.
    """
    url = _mock_bank_url("/api/login")
    payload = {"username": username, "passphrase": passphrase}
    result = await _post_json(url, payload)
    if not result["ok"]:
        return {
            "status": "error",
            "message": result["message"],
            "http_status": result.get("status"),
        }

    data = result["data"]
    return {
        "status": data.get("status", "ok"),
        "user_id": data.get("user_id"),
        "username": data.get("username"),
        "has_audio_embedding": data.get("has_audio_embedding", False),
        "message": data.get("message"),
    }


@mcp.tool(name="set_user_audio_embedding", description="Set or replace the audio embedding for a user")
async def set_user_audio_embedding(user_id: str, audio_embedding: list[float]) -> dict:
    """
    Wraps PUT /api/users/{user_id}/audio-embedding.
    """
    url = _mock_bank_url(f"/api/users/{user_id}/audio-embedding")
    payload = {"audio_embedding": audio_embedding}
    result = await _put_json(url, payload)
    if not result["ok"]:
        return {
            "status": "error",
            "message": result["message"],
            "http_status": result.get("status"),
        }

    data = result["data"]
    return {"status": "success", "user": data}


@mcp.tool(name="get_user_profile", description="Fetch a user's profile by user_id")
async def get_user_profile(user_id: str) -> dict:
    """
    Wraps GET /api/users/{user_id} and returns profile info.
    """
    url = _mock_bank_url(f"/api/users/{user_id}")
    result = await _get_json(url)
    if not result["ok"]:
        return {
            "status": "error",
            "message": result["message"],
            "http_status": result.get("status"),
        }

    return {"status": "success", "user": result["data"]}


@mcp.tool(name="list_tools", description="List available MCP tools")
def list_tools() -> dict:
    """Return metadata for all available MCP tools."""
    try:
        # Prefer TOOL_METADATA as the canonical source. This ensures the
        # orchestrator can build prompts and validation rules dynamically.
        return {"tools": TOOL_METADATA}
    except Exception as e:  # pragma: no cover
        logger.error("Error in list_tools: %s", e)
        return {"tools": {}}


# -----------------------------
# Start MCP HTTP server
# -----------------------------
def main():
    logger.info("Starting MCP HTTP server on http://%s:%s", MCP_HOST, MCP_PORT)

    # The magic line:
    mcp.run(
        host=MCP_HOST,
        port=MCP_PORT,
        transport="streamable-http",
    )


if __name__ == "__main__":
    main()

