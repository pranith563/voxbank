# Mock Bank – PostgreSQL‑Backed Banking API

The **mock‑bank** service simulates a real core banking backend for VoxBank.  
It exposes REST APIs for users, accounts, transactions, loans, cards, reminders, and transfers, backed by PostgreSQL via async SQLAlchemy.

This is the system‑of‑record that MCP tools and the orchestrator use for all financial data.

---

## 1. Architecture Overview

- FastAPI app: `mock-bank/src/app.py`
- Database models: `mock-bank/src/db/models.py`
  - `User` – customer profiles:
    - `passphrase` – login credential (demo only; not a real password hash).
    - `audio_embedding` – JSONB vector for future voice authentication.
  - `Account` – savings/checking/business accounts linked to users.
  - `Transaction` – ledger entries (debit/credit, transfer, deposit, payment, etc.).
  - `Beneficiary` – saved payees.
  - `Card` – credit/debit cards.
  - `Loan` – loans with EMI details.
  - `Reminder` – reminders (e.g., EMI due).
- DB session:
  - `mock-bank/src/db/session.py` – async engine + session factory; loaded from `DATABASE_URL`.

The canonical schema lives in `data/schema.sql` and is designed for clarity and easy inspection.

---

## 2. Schema & Bootstrapping

### 2.1 Creating the Database

1. Create a Postgres database (e.g. `voxbank`).
2. Set `DATABASE_URL`:
   ```bash
   export DATABASE_URL="postgresql+asyncpg://postgres:password@localhost:5432/voxbank"
   ```
3. Apply the schema:
   ```bash
   cd mock-bank
   psql "postgres://postgres:password@localhost:5432/voxbank" -f data/schema.sql
   ```

### 2.2 Seeding Demo Users

The API includes a simple seeding endpoint:

- `POST /api/admin/seed` – inserts a set of demo users and related entities.
  - Protected by `SIMPLE_ADMIN_TOKEN` (default `"letmein"`).

Example:

```bash
curl -X POST "http://localhost:9000/api/admin/seed" \
  -H "Content-Type: application/json" \
  -d '{ "token": "letmein" }'
```

This ensures you have users with realistic accounts, loans, cards, and reminders for demos.

---

## 3. Auth & Voice Embedding Endpoints

These power the orchestrator’s login and future voice‑based auth flows.

- `POST /api/users` – create a user (registration)
  - Body:
    ```json
    {
      "username": "john_doe",
      "passphrase": "my secret phrase",
      "email": "john@example.com",
      "full_name": "John Doe",
      "phone_number": "9990001111",
      "audio_embedding": [0.1, 0.2, 0.3] // optional
    }
    ```
  - Alias: `POST /api/register`.

- `POST /api/login` – validate username + passphrase
  - Body:
    ```json
    { "username": "john_doe", "passphrase": "my secret phrase" }
    ```
  - Response:
    ```json
    {
      "user_id": "…",
      "username": "john_doe",
      "status": "ok",
      "message": "Login successful",
      "has_audio_embedding": true
    }
    ```

- `PUT /api/users/{user_id}/audio-embedding` – set or replace audio embedding.
- `GET /api/users/{user_id}/audio-embedding` – fetch audio embedding (404 if missing).

The MCP tools `register_user`, `login_user`, and `set_user_audio_embedding` are thin wrappers around these APIs.

---

## 4. Core Banking Endpoints

### 4.1 Accounts & Transactions

- `GET /api/users/{user_id}/accounts`
  - List accounts for a user (compact view).

- `GET /api/accounts/{account_number}`
  - Fetch a single account by account number.
  - Returns balances, currency, status, timestamps, etc.

- `GET /api/accounts/{account_number}/transactions?limit=20`
  - Recent transactions in reverse chronological order.

- `POST /api/transfer`
  - Perform an atomic funds transfer inside a DB transaction.
  - Body (simplified):
    ```json
    {
      "from_account_number": "ACC000010",
      "to_account_number": "ACC001001",
      "amount": 50.0,
      "currency": "USD",
      "initiated_by_user_id": "…"
    }
    ```
  - Locks both accounts (`SELECT … FOR UPDATE`), checks balances, updates balances, and inserts a debit + credit `Transaction` row pair.

These endpoints are wrapped by the `balance`, `transactions`, and `transfer` MCP tools.

### 4.2 Cards, Loans & Reminders

- `GET /api/users/{user_id}/cards`
  - Cards linked to the user (credit/debit, network, limits, dues).

- `GET /api/users/{user_id}/loans`
  - Loans with EMI info (principal, outstanding, interest_rate, next_due_date, etc.).

- `GET /api/users/{user_id}/reminders`
  - Reminders (optionally filtered to upcoming windows).

The MCP tools `cards_summary`, `loans_summary`, and `reminders_summary` are built on these APIs.

### 4.3 Beneficiaries

- `GET /api/users/{user_id}/beneficiaries`
  - List saved payees/beneficiaries.

- `POST /api/users/{user_id}/beneficiaries`
  - Add a beneficiary (internal/external).

These are wrapped by MCP tools `get_user_beneficiaries`, `get_my_beneficiaries`, and `add_beneficiary`.

---

## 5. Running the Service

### Environment

- `DATABASE_URL` – Postgres URI (required).
  - Example: `postgresql+asyncpg://postgres:password@localhost:5432/voxbank`
- `LOG_LEVEL` – e.g. `INFO`, `DEBUG`.
- `SIMPLE_ADMIN_TOKEN` – token for `/api/admin/seed` (default `"letmein"`).

### Install & Run (dev)

```bash
cd mock-bank
python -m pip install -r requirements.txt

uvicorn src.app:app --reload --port 9000
```

The orchestrator and MCP tools expect this service to be reachable at the base URL configured via:

- `VOX_BANK_BASE_URL` (for MCP tools)
- `VOX_BANK_BASE_URL` or `MOCK_BANK_BASE_URL` (for any direct HTTP calls)

---

## 6. How Orchestrator Uses Mock Bank

- During login:
  - Orchestrator calls `POST /api/login` to validate username/passphrase.
  - On success, it then fetches:
    - `GET /api/users/{user_id}`
    - `GET /api/users/{user_id}/accounts`
  - The results are cached in the session profile.

- During conversations:
  - All financial operations (balance, transactions, transfers, loans, cards, reminders) go through MCP tools which in turn call these REST endpoints.

This separation (mock‑bank ↔ mcp‑tools ↔ orchestrator ↔ LLM) keeps responsibilities clear and makes it easy to swap the mock‑bank for a real core system in the future.

