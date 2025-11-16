# orchestrator/src/clients/mcp_client.py
"""
MCP Client - uses FastMCP Client with StreamableHttpTransport when available,
otherwise falls back to an httpx-based invoke client.

Environment:
  MCP_TOOL_BASE_URL  (default: http://localhost:9100)   -- base URL to MCP HTTP server
  MCP_TOOL_KEY       (default: mcp-test-key)
"""

import os
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger("voxbank.mcp_client")
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))

# Try to import fastmcp client classes
try:
    from fastmcp import Client as FastMCPClient
    from fastmcp.client.transports import StreamableHttpTransport
    FASTMCP_AVAILABLE = True
except Exception:
    FastMCPClient = None
    StreamableHttpTransport = None
    FASTMCP_AVAILABLE = False

import httpx
import asyncio

DEFAULT_BASE = os.getenv("MCP_TOOL_BASE_URL", "http://localhost:9100").rstrip("/")
DEFAULT_TOOL_KEY = os.getenv("MCP_TOOL_KEY", "mcp-test-key")


# -------------------------
# FastMCP-based wrapper
# -------------------------
class FastMCPWrapper:
    """
    Wraps fastmcp.Client using StreamableHttpTransport to call remote MCP HTTP servers.
    """

    def __init__(self, base_url: str = DEFAULT_BASE, tool_key: str = DEFAULT_TOOL_KEY, timeout: float = 10.0):
        if not FASTMCP_AVAILABLE:
            raise RuntimeError("fastmcp not installed")
        # transport expects full URL to MCP mount (e.g. http://localhost:9100/mcp)
        # We'll allow base_url to be the root and try common mountpoints.
        # Prefer /mcp if not included
        if base_url.endswith("/mcp"):
            url = base_url
        else:
            url = base_url.rstrip("/") + "/mcp"

        headers = {"X-MCP-TOOL-KEY": tool_key}
        # StreamableHttpTransport accepts url and headers
        self.transport = StreamableHttpTransport(url=url, headers=headers)
        self.client = FastMCPClient(self.transport)
        logger.info("FastMCPWrapper initialized for %s", url)

    async def initialize(self):
        # `async with client` opens the transport / session context
        # We keep the client open for the process lifetime
        await self.client.__aenter__()  # start the client session
        logger.info("FastMCP client session started")

    async def close(self):
        try:
            await self.client.__aexit__(None, None, None)
            logger.info("FastMCP client closed")
        except Exception:
            logger.exception("Error closing FastMCP client")

    async def call_tool(self, tool_name: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Use fastmcp client.call_tool which returns results (and raises on transport/server errors).
        """
        try:
            # fastmcp client expects dict payload
            res = await self.client.call_tool(tool_name, payload)
            return res
        except Exception as e:
            logger.exception("FastMCP call_tool error: %s", e)
            raise


# -------------------------
# Fallback httpx-based client
# -------------------------
class HttpxMCPClient:
    """
    Simple HTTP fallback client that posts to discovered invoke endpoints (/mcp/invoke/<tool> or /invoke/<tool>)
    """

    def __init__(self, base_url: str = DEFAULT_BASE, tool_key: str = DEFAULT_TOOL_KEY, timeout: float = 10.0):
        self.base_url = base_url.rstrip("/")
        self.tool_key = tool_key
        self._client = httpx.AsyncClient(timeout=timeout)
        # prefer /mcp
        self.invoke_prefix = "/mcp/invoke"

    async def initialize(self):
        # no-op for fallback; optionally validate server reachable
        # try /mcp/invoke/<probe> and /invoke/<probe>
        probe_paths = ["/mcp/invoke/__probe__", "/invoke/__probe__", "/invoke/__probe__"]
        for p in probe_paths:
            try:
                url = f"{self.base_url}{p}"
                resp = await self._client.options(url, timeout=1.0)
                if resp.status_code in (200, 204, 405, 401):
                    if p.startswith("/mcp"):
                        self.invoke_prefix = "/mcp/invoke"
                    else:
                        self.invoke_prefix = "/invoke"
                    logger.info("HttpxMCPClient: chosen invoke_prefix=%s", self.invoke_prefix)
                    return
            except Exception:
                continue
        # default fallback
        logger.warning("HttpxMCPClient: defaulting invoke_prefix to /mcp/invoke")
        self.invoke_prefix = "/mcp/invoke"

    async def close(self):
        try:
            await self._client.aclose()
        except Exception:
            logger.exception("Error closing httpx client")

    def _invoke_url(self, tool_name: str) -> str:
        prefix = self.invoke_prefix if self.invoke_prefix else "/mcp/invoke"
        prefix = prefix.rstrip("/")
        return f"{self.base_url}{prefix}/{tool_name}"

    async def call_tool(self, tool_name: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        url = self._invoke_url(tool_name)
        headers = {"Content-Type": "application/json", "X-MCP-TOOL-KEY": self.tool_key}
        try:
            resp = await self._client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPStatusError as e:
            logger.error("HTTP invoke error %s %s -> %s", url, e.response.status_code, e.response.text)
            raise
        except Exception:
            logger.exception("HTTP invoke failed: %s", url)
            raise


# -------------------------
# Public MCPClient facade
# -------------------------
class MCPClient:
    """
    Facade: uses FastMCP client if available, otherwise httpx fallback.
    Exposes:
      - initialize()
      - close()
      - call_tool(tool_name, payload)
      - convenience wrappers get_balance/get_transactions/transfer_funds
    """

    def __init__(self, base_url: Optional[str] = None, tool_key: Optional[str] = None, timeout: float = 10.0):
        self.base_url = (base_url or DEFAULT_BASE).rstrip("/")
        self.tool_key = tool_key or DEFAULT_TOOL_KEY
        self.transport_mode = "fastmcp" if FASTMCP_AVAILABLE else "httpx"
        self._impl = None
        logger.info("MCPClient selecting transport: %s", self.transport_mode)
        if self.transport_mode == "fastmcp":
            self._impl = FastMCPWrapper(self.base_url, self.tool_key, timeout=timeout)
        else:
            self._impl = HttpxMCPClient(self.base_url, self.tool_key, timeout=timeout)

    async def initialize(self):
        await self._impl.initialize()

    async def close(self):
        await self._impl.close()

    async def call_tool(self, tool_name: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        return await self._impl.call_tool(tool_name, payload)
