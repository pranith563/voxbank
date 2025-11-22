# mcp-tools – VoxBank MCP HTTP Server

The `mcp-tools` service is a FastMCP-based HTTP server that exposes VoxBank tools via the **Model Context Protocol (MCP)**.  
The orchestrator discovers these tools at startup and uses them from its ReAct loop.

---

## 1. Overview

- Entry point: `mcp-tools/src/mcp_server.py`
- Uses `fastmcp.FastMCP` to register tools and serve:
  - `GET  /mcp/manifest`
  - `POST /mcp/invoke/<tool_name>`
  - `POST /mcp/call` (JSON-RPC style)
- Shared `httpx.AsyncClient` talks to the mock-bank API (`VOX_BANK_BASE_URL`).
- Canonical tool metadata:
  - `TOOL_METADATA` in `mcp_server.py` is returned by the `list_tools` tool and is the **source of truth** for:
    - Tool descriptions.
    - Parameter schemas (name, type, required/default).

This service is the “safe execution layer” between the LLM and the mock-bank.

---

## 2. Tool Catalog (High-Level)

### 2.1 Core Banking

- `balance`
  - Description: Get account balance.
  - Params:
    - `account_number` (`string`, required)
  - Wraps: `GET /api/accounts/{account_number}` on mock-bank.

- `transactions`
  - Description: Fetch recent transactions for an account.
  - Params:
    - `account_number` (`string`, required)
    - `limit` (`integer`, optional, default 10)
  - Wraps: `GET /api/accounts/{account_number}/transactions?limit=…`.

- `transfer` (HIGH RISK)
  - Description: Execute a funds transfer between two accounts.
  - Params:
    - `from_account_number` (`string`, required)
    - `to_account_number` (`string`, required)
    - `amount` (`number`, required)
    - `currency` (`string`, optional, default `"USD"`)
    - `initiated_by_user_id` (`string`, optional)
    - `reference` (`string`, optional)
  - Wraps: `POST /api/transfer`.

### 2.2 User, Auth & Profile

- `register_user`
  - Description: Register a new VoxBank user with username + passphrase (and optional profile + audio embedding).
  - Params: `username`, `passphrase`, `email?`, `full_name?`, `phone_number?`, `audio_embedding?`.
  - Wraps: `POST /api/users` / `POST /api/register`.

- `login_user`
  - Description: Validate username + passphrase and return basic auth info.
  - Params: `username`, `passphrase`.
  - Wraps: `POST /api/login`.

- `set_user_audio_embedding`
  - Description: Store or replace the voice embedding for a user.
  - Params: `user_id`, `audio_embedding`.
  - Wraps: `PUT /api/users/{user_id}/audio-embedding`.

- `get_user_profile`
  - Description: Fetch a user profile by `user_id`.
  - Params: `user_id`.
  - Wraps: `GET /api/users/{user_id}`.

### 2.3 “My *” Summary Tools

These tools are **session-aware** and are designed to be called without the LLM needing to pass a `user_id`. The orchestrator injects session context when calling them.

- `get_my_profile`
  - Description: Get the current logged‑in user’s profile.
  - Params: none (orchestrator provides `user_id`, `session_id`).

- `get_my_accounts`
  - Description: List accounts for the current user.
  - Params: none.

- `get_my_beneficiaries`
  - Description: List saved beneficiaries/payees for the current user.
  - Params: none.

- `cards_summary`
  - Description: Summary of cards for the current user.
  - Params: none.

- `loans_summary`
  - Description: Summary of loans/EMIs for the current user.
  - Params: none.

- `reminders_summary`
  - Description: Summary of reminders for the current user.
  - Params:
    - `days` (`integer`, optional) – look‑ahead window for upcoming reminders.

### 2.4 Beneficiaries & Accounts

- `get_user_beneficiaries`
  - Description: List beneficiaries for a specific `user_id`.
  - Params: `user_id`, `limit?`, `offset?`.

- `add_beneficiary`
  - Description: Add a new beneficiary (saved payee) for a user.
  - Params: `user_id`, `account_number`, `nickname?`, `bank_name?`, `is_internal?`.

- `get_user_accounts`
  - Description: Fetch all accounts for a given `user_id`.
  - Params: `user_id`.

### 2.5 Session / Utility

- `logout_user`
  - Description: Log out the current VoxBank user/session.
  - Params: `user_id?`, `session_id?` (orchestrator usually supplies these).

- `list_tools`
  - Description: Enumerate all tools and their schemas.
  - Params: none.
  - Returns:
    ```jsonc
    {
      "tools": {
        "balance": {
          "description": "Get account balance",
          "params": {
            "account_number": {"type": "string", "required": true}
          }
        },
        "transfer": { ... },
        "get_my_profile": { ... }
      }
    }
    ```
  - Used by the orchestrator at startup to build its prompt/tool spec.

---

## 3. HTTP Surface

When running, the server exposes:

- `GET  /mcp/manifest`  
  Returns the MCP manifest describing this tool server.

- `POST /mcp/invoke/<tool_name>`  
  Single‑tool invocation endpoint. Body is the JSON params for that tool.

- `POST /mcp/call`  
  Batch/multiplexed invocation endpoint for more advanced MCP clients.

FastMCP handles the protocol details; `mcp_server.py` focuses on business logic and calling mock‑bank.

---

## 4. Example: Calling a Tool Manually

Although the orchestrator normally calls tools via MCP, you can exercise them directly for debugging:

```bash
curl -X POST "http://localhost:9100/mcp/invoke/balance" \
  -H "Content-Type: application/json" \
  -d '{ "account_number": "ACC000010" }'
```

Sample response (shape depends on mock‑bank response):

```json
{
  "status": "success",
  "account_number": "ACC000010",
  "balance": 4330.0,
  "currency": "USD"
}
```

Or list tools:

```bash
curl -X POST "http://localhost:9100/mcp/invoke/list_tools" -d '{}'
```

---

## 5. Running the MCP Server

### Environment

- `VOX_BANK_BASE_URL` – base URL of mock-bank (default `http://localhost:9000`).
- `MCP_HOST` – host to bind (default `0.0.0.0`).
- `MCP_PORT` – port for the MCP HTTP server (default `9100`).
- `MCP_REQUEST_TIMEOUT` – HTTP timeout when calling mock-bank (seconds).
- `MCP_TOOL_KEY` – shared key used by the orchestrator to authenticate (`X-MCP-TOOL-KEY` header).
- `LOG_LEVEL` – log level (default `INFO`).

### Install & Run (dev)

```bash
cd mcp-tools
python -m pip install -r requirements.txt

# Run the MCP HTTP server
python -m src.mcp_server
# or, depending on packaging:
python src/mcp_server.py
```

By default this binds to `http://0.0.0.0:9100`.  
The orchestrator points at it via `MCP_TOOL_BASE_URL` (e.g. `http://localhost:9100`) and calls `list_tools` during startup.

