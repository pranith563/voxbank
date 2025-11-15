# from orchestrator/src/mcp_client.py (example)
import os, httpx

class MCPHttpClient:
    def __init__(self, base_url=None, tool_key=None, timeout=10):
        self.base_url = base_url or os.getenv("MCP_TOOL_BASE_URL", "http://localhost:9100")
        self.tool_key = tool_key or os.getenv("MCP_TOOL_KEY", "mcp-test-key")
        self._client = httpx.AsyncClient(timeout=timeout)

    async def execute(self, tool_name: str, payload: dict, session_id: str = "") -> dict:
        url = f"{self.base_url.rstrip('/')}/tools/{tool_name}"
        headers = {"X-MCP-TOOL-KEY": self.tool_key, "Content-Type": "application/json"}
        r = await self._client.post(url, json=payload, headers=headers)
        r.raise_for_status()
        return r.json()
