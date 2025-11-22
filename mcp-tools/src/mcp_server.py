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
    "get_user_beneficiaries": {
        "description": "List all beneficiaries (saved payees) for a given user_id",
        "params": {
            "user_id": {"type": "string", "required": True},
        },
    },
    "add_beneficiary": {
        "description": "Add a new beneficiary (saved payee) for a user",
        "params": {
            "user_id": {"type": "string", "required": True},
            "nickname": {"type": "string", "required": False},
            "account_number": {"type": "string", "required": True},
            "bank_name": {"type": "string", "required": False},
            "is_internal": {"type": "boolean", "required": False},
        },
    },
    "get_user_accounts": {
        "description": "Fetch all accounts for a user by user_id",
        "params": {
            "user_id": {"type": "string", "required": True},
        },
    },
    # Simple \"my_*\" info tools that do not require user_id in tool_input.
    "get_my_profile": {
        "description": "Get the current logged-in user's profile.",
        "params": {},
    },
    "get_my_accounts": {
        "description": "Get the current logged-in user's accounts.",
        "params": {},
    },
    "get_my_beneficiaries": {
        "description": "Get the current logged-in user's saved payees/beneficiaries.",
        "params": {},
    },
    "cards_summary": {
        "description": "List all cards for the current logged-in user.",
        "params": {},
    },
    "loans_summary": {
        "description": "List active loans for the current logged-in user.",
        "params": {},
    },
    "reminders_summary": {
        "description": "List reminders (optionally upcoming) for the current logged-in user.",
        "params": {
            "days": {"type": "integer", "required": False},
        },
    },
    "logout_user": {
        "description": "Log out the current VoxBank user/session.",
        "params": {
            "user_id": {"type": "string", "required": False},
            "session_id": {"type": "string", "required": False},
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


# ---------------------------------------------------------------------------
# \"My\" info tools
# ---------------------------------------------------------------------------


@mcp.tool(name="get_my_profile", description="Get the current logged-in user's profile")
async def get_my_profile(user_id: str | None = None, session_id: str | None = None) -> dict:  # noqa: ARG001
    """
    Resolve the current user's profile without exposing user_id in tool_input
    to the LLM. The orchestrator MUST inject user_id/session_id as hidden params.
    """
    if not user_id:
        return {
            "status": "error",
            "message": "No user context available for get_my_profile.",
        }
    return await get_user_profile(user_id)


@mcp.tool(name="get_user_accounts", description="Fetch all accounts for a user by user_id")
async def get_user_accounts(user_id: str) -> dict:
    """
    Wraps GET /api/users/{user_id}/accounts and returns the list of accounts.
    """
    url = _mock_bank_url(f"/api/users/{user_id}/accounts")
    result = await _get_json(url)
    if not result["ok"]:
        return {
            "status": "error",
            "message": result["message"],
            "http_status": result.get("status"),
        }

    return {"status": "success", "accounts": result["data"]}


@mcp.tool(name="get_my_accounts", description="Get the current logged-in user's accounts")
async def get_my_accounts(user_id: str | None = None, session_id: str | None = None) -> dict:  # noqa: ARG001
    if not user_id:
        return {
            "status": "error",
            "message": "No user context available for get_my_accounts.",
        }
    return await get_user_accounts(user_id)


@mcp.tool(name="cards_summary", description="List cards for the current user")
async def cards_summary(user_id: str | None = None, session_id: str | None = None) -> dict:  # noqa: ARG001
    if not user_id:
        return {
            "status": "error",
            "message": "No user context available for cards_summary.",
        }
    url = _mock_bank_url(f"/api/users/{user_id}/cards")
    result = await _get_json(url)
    if not result["ok"]:
        return {
            "status": "error",
            "message": result["message"],
            "http_status": result.get("status"),
        }
    return {
        "status": "success",
        "user_id": user_id,
        "cards": result["data"],
    }


@mcp.tool(name="loans_summary", description="List loans for the current user")
async def loans_summary(user_id: str | None = None, session_id: str | None = None) -> dict:  # noqa: ARG001
    if not user_id:
        return {
            "status": "error",
            "message": "No user context available for loans_summary.",
        }
    url = _mock_bank_url(f"/api/users/{user_id}/loans")
    result = await _get_json(url)
    if not result["ok"]:
        return {
            "status": "error",
            "message": result["message"],
            "http_status": result.get("status"),
        }
    return {
        "status": "success",
        "user_id": user_id,
        "loans": result["data"],
    }


@mcp.tool(name="reminders_summary", description="List reminders for the current user")
async def reminders_summary(
    days: int = 30,
    user_id: str | None = None,
    session_id: str | None = None,  # noqa: ARG001
) -> dict:
    if not user_id:
        return {
            "status": "error",
            "message": "No user context available for reminders_summary.",
        }
    params = {"upcoming": "true", "days": days}
    url = _mock_bank_url(f"/api/users/{user_id}/reminders")
    result = await _get_json(url, params=params)
    if not result["ok"]:
        return {
            "status": "error",
            "message": result["message"],
            "http_status": result.get("status"),
        }
    return {
        "status": "success",
        "user_id": user_id,
        "reminders": result["data"],
        "days": days,
    }


@mcp.tool(name="get_my_beneficiaries", description="List beneficiaries for the current user")
async def get_my_beneficiaries(
    limit: int = 50,
    offset: int = 0,
    user_id: str | None = None,
    session_id: str | None = None,  # noqa: ARG001
) -> dict:
    if not user_id:
        return {
            "status": "error",
            "message": "No user context available for get_my_beneficiaries.",
        }
    return await _get_user_beneficiaries(user_id=user_id, limit=limit, offset=offset)


async def _get_user_beneficiaries(user_id: str, limit: int = 50, offset: int = 0) -> dict:
    """
    Wraps GET /api/users/{user_id}/beneficiaries.
    """
    url = _mock_bank_url(f"/api/users/{user_id}/beneficiaries")
    result = await _get_json(url, params={"limit": limit, "offset": offset})
    if not result["ok"]:
        return {
            "status": "error",
            "message": result["message"],
            "http_status": result.get("status"),
        }

    return {
        "status": "success",
        "user_id": user_id,
        "beneficiaries": result["data"],
    }


@mcp.tool(name="get_user_beneficiaries", description="List beneficiaries for a user")
async def get_user_beneficiaries(user_id: str, limit: int = 50, offset: int = 0) -> dict:  # noqa: D401
    """Tool wrapper delegating to _get_user_beneficiaries."""
    return await _get_user_beneficiaries(user_id=user_id, limit=limit, offset=offset)


@mcp.tool(name="add_beneficiary", description="Add a new beneficiary for a user")
async def add_beneficiary(
    user_id: str,
    account_number: str,
    nickname: str | None = None,
    bank_name: str | None = None,
    is_internal: bool = True,
) -> dict:
    """
    Wraps POST /api/users/{user_id}/beneficiaries.
    """
    url = _mock_bank_url(f"/api/users/{user_id}/beneficiaries")
    payload = {
        "account_number": account_number,
        "nickname": nickname,
        "bank_name": bank_name,
        "is_internal": is_internal,
    }
    result = await _post_json(url, payload)
    if not result["ok"]:
        return {
            "status": "error",
            "message": result["message"],
            "http_status": result.get("status"),
        }

    return {"status": "success", "beneficiary": result["data"]}


@mcp.tool(name="logout_user", description="Log out the current VoxBank user/session")
async def logout_user(user_id: str | None = None, session_id: str | None = None) -> dict:
    """
    Simple acknowledgment tool. The orchestrator clears session/auth state after calling this.
    """
    logger.info("logout_user tool invoked (user_id=%s session_id=%s)", user_id, session_id)
    return {
        "status": "success",
        "message": "Logout acknowledged.",
        "user_id": user_id,
        "session_id": session_id,
    }


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
