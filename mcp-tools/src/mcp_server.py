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


@mcp.tool(name="list_tools", description="List available MCP tools")
def list_tools() -> dict:
    """List all available MCP tools"""
    try:
        # FastMCP stores tools in its internal registry
        # Access via the mcp instance's tools attribute
        if hasattr(mcp, 'tools') and mcp.tools:
            tool_names = list(mcp.tools.keys())
        elif hasattr(mcp, '_tools') and mcp._tools:
            tool_names = list(mcp._tools.keys())
        else:
            # Fallback: return known tools
            tool_names = [] #["balance", "transactions", "transfer", "list_tools"]
        return {"tools": tool_names}
    except Exception as e:
        logger.error(f"Error in list_tools: {e}")
        # Return known tools as fallback
        return {"tools": []}


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
