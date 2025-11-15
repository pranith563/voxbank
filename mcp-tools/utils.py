import os
import httpx
from dotenv import load_dotenv
load_dotenv()

MCP_TOOL_KEY = os.getenv("MCP_TOOL_KEY", "mcp-test-key")          # shared secret between orchestrator and MCP tools
MOCK_BANK_BASE = os.getenv("MOCK_BANK_BASE_URL", "http://localhost:9000")
REQUEST_TIMEOUT = float(os.getenv("MCP_REQUEST_TIMEOUT", "10"))
# -----------------------
# Helpers
# -----------------------
# Shared async client (reused)
_http_client = httpx.AsyncClient(timeout=REQUEST_TIMEOUT)

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
