# MCP Tools – VoxBank MCP HTTP Server

The `mcp-tools` service is a FastMCP-based HTTP server that exposes VoxBank tools via the Model Context Protocol. The orchestrator discovers these tools at startup and uses them in the LLM ReAct loop.

## Overview

- Entry point: `mcp-tools/src/mcp_server.py`
- Uses `fastmcp.FastMCP` to register tools and serve:
  - `GET /mcp/manifest`
  - `POST /mcp/invoke/<tool_name>`
  - `POST /mcp/call`
- Shared `httpx.AsyncClient` talks to the mock-bank API (`VOX_BANK_BASE_URL`).
- Canonical tool metadata:
  - `TOOL_METADATA` in `mcp_server.py` is returned by `list_tools` and is the source of truth for:
    - Tool descriptions.
    - Parameter schemas (name, type, required).

## Tools

Banking:

- `balance(account_number)` → wraps `GET /api/accounts/{account_number}`.
- `transactions(account_number, limit=10)` → wraps `GET /api/accounts/{acct}/transactions`.
- `transfer(from_account_number, to_account_number, amount, currency="USD", initiated_by_user_id=None, reference=None)` → wraps `POST /api/transfer` (high risk).

Auth & Profile:

- `register_user(username, passphrase, email?, full_name?, phone_number?, audio_embedding?)` → wraps `POST /api/users`.
- `login_user(username, passphrase)` → wraps `POST /api/login`.
- `set_user_audio_embedding(user_id, audio_embedding)` → wraps `PUT /api/users/{user_id}/audio-embedding`.
- `get_user_profile(user_id)` → wraps `GET /api/users/{user_id}`.

Discovery:

- `list_tools()`:
  - Returns `{"tools": TOOL_METADATA}` so the orchestrator can dynamically build the tool spec for prompts and validation.

## Running the MCP Server

Environment:

- `VOX_BANK_BASE_URL` – base URL of mock-bank (default `http://localhost:9000`).
- `MCP_HOST` – host to bind (default `0.0.0.0`).
- `MCP_PORT` – port for the MCP HTTP server (default `9100`).
- `MCP_REQUEST_TIMEOUT` – HTTP timeout when calling mock-bank.
- `LOG_LEVEL` – log level (default `INFO`).

Install & run:

```bash
cd mcp-tools
python -m pip install -r requirements.txt

# Run the MCP HTTP server
python -m mcp_server          # if packaged as a module
# or
python src/mcp_server.py      # direct script
```

The orchestrator uses `MCP_TOOL_BASE_URL` (default `http://localhost:9100`) to reach this server and calls `list_tools` at startup to initialize the LLM tool spec.

