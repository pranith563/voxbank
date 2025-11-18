# Mock Bank – PostgreSQL-backed Banking API

The mock-bank service simulates a core banking backend for VoxBank. It exposes REST APIs for users, accounts, transactions, and transfers, backed by PostgreSQL via SQLAlchemy.

## Overview

- FastAPI app in `mock-bank/src/app.py`.
- Async SQLAlchemy models in `mock-bank/src/db/models.py`.
  - `users` – user profiles, now with:
    - `passphrase` (string) for login.
    - `audio_embedding` (jsonb) for future voice authentication.
  - `accounts` – bank accounts linked to users.
  - `transactions` – debit/credit entries, including transfers.
- Shared DB session in `mock-bank/src/db/session.py`.

## Schema & Data

- Canonical schema file:
  - `mock-bank/data/schema.sql` – creates `users`, `accounts`, and `transactions` tables, including `passphrase` and `audio_embedding` on `users`.
- Seed endpoint:
  - `POST /api/admin/seed` – inserts demo users (with demo `passphrase="seeded"`) and can be extended to seed accounts/transactions.

To bootstrap a fresh DB:

```bash
cd mock-bank
export DATABASE_URL="postgresql://postgres:password@localhost:5432/voxbank"
psql $DATABASE_URL -f data/schema.sql
```

## Auth & Embedding Endpoints

- `GET /api/users` – list users.
- `GET /api/users/{user_id}` – get a single user; response includes `has_audio_embedding`.
- `POST /api/users` – create a user:
  - Body: `{ "username", "passphrase", "email"?, "full_name"?, "phone_number"?, "audio_embedding"? }`
  - Validates unique username.
- `POST /api/register` – alias of `POST /api/users`.
- `POST /api/login` – validate username + passphrase:
  - Body: `{ "username", "passphrase" }`
  - Returns: `{ "user_id", "username", "status", "message", "has_audio_embedding" }`
- `PUT /api/users/{user_id}/audio-embedding` – set/replace `audio_embedding`.
- `GET /api/users/{user_id}/audio-embedding` – retrieve `audio_embedding` (404 if missing).

These endpoints are wrapped by MCP tools (`register_user`, `login_user`, `set_user_audio_embedding`, `get_user_profile`) and used by the orchestrator’s login flow.

## Banking Endpoints

- `GET /api/users/{user_id}/accounts` – list accounts for a user.
- `GET /api/accounts/{account_number}` – get account details.
- `GET /api/accounts/{account_number}/transactions` – transaction history (with `limit`).
- `POST /api/transfer` – perform an atomic funds transfer between two accounts (with locking and balance checks).

These are wrapped by MCP tools: `balance`, `transactions`, `transfer`.

## Running the Service

Environment:

- `DATABASE_URL` – Postgres URI (required).
- `LOG_LEVEL` – log level (default `INFO`).
- `SIMPLE_ADMIN_TOKEN` – token for `/api/admin/seed` (default `"letmein"`).

Install & run:

```bash
cd mock-bank
python -m pip install -r requirements.txt

uvicorn src.app:app --reload --port 9000
```

The orchestrator and MCP tools expect this service at the base URL configured via `VOX_BANK_BASE_URL` / `MOCK_BANK_BASE_URL`.

