# mock-bank/src/app.py
import os
import logging
from decimal import Decimal
from typing import List, Optional
from uuid import UUID, uuid4

from fastapi import FastAPI, HTTPException, Depends, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import asyncio
from sqlalchemy import select, update, func
from sqlalchemy.orm import selectinload
from sqlalchemy.exc import SQLAlchemyError
from db.models import User, Account, Transaction
# load env
load_dotenv()

# logging
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("mock_bank")

# Async DB session & models (assumes these files exist under mock-bank/src/db/)
try:
    from db.session import AsyncSessionLocal, engine
    from db.models import User, Account, Transaction  # models should match your schema
except Exception as e:
    # If run as script, adjust sys.path externally or run with project root on path.
    logger.exception("Error importing DB modules: %s", e)
    raise

app = FastAPI(title="Mock Bank API (VoxBank)", version="1.0.0")

# CORS (open for demo)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

# --------------------
# Pydantic schemas
# --------------------
class UserOut(BaseModel):
    user_id: UUID
    username: str
    email: str
    full_name: Optional[str] = None
    phone_number: Optional[str] = None

class AccountOut(BaseModel):
    account_id: UUID
    account_number: str
    user_id: UUID
    account_type: str
    currency: str
    balance: float
    available_balance: Optional[float]
    status: str

class TransactionOut(BaseModel):
    transaction_id: UUID
    transaction_reference: str
    account_id: UUID
    entry_type: str
    transaction_type: str
    amount: float
    currency: str
    status: str
    description: Optional[str] = None
    created_at: Optional[str] = None

class TransferIn(BaseModel):
    from_account_number: str = Field(..., example="ACC-0001")
    to_account_number: str = Field(..., example="ACC-0002")
    amount: Decimal = Field(..., gt=0, example=100.00)
    currency: str = Field("USD")
    initiated_by_user_id: Optional[UUID] = None
    reference: Optional[str] = None

class TransferOut(BaseModel):
    status: str
    transaction_reference: str
    txn_id: Optional[UUID] = None
    message: Optional[str] = None

# --------------------
# DB helper dependency
# --------------------
async def get_db():
    async with AsyncSessionLocal() as session:
        yield session

# --------------------
# Utility: serialize helpers
# --------------------
def _serialize_user(u: User) -> dict:
    return {
        "user_id": str(u.user_id),
        "username": u.username,
        "email": u.email,
        "full_name": u.full_name,
        "phone_number": u.phone_number,
    }

def _serialize_account(a: Account) -> dict:
    return {
        "account_id": str(a.account_id),
        "account_number": a.account_number,
        "user_id": str(a.user_id),
        "account_type": a.account_type,
        "currency": a.currency,
        "balance": float(a.balance) if a.balance is not None else 0.0,
        "available_balance": float(a.available_balance) if a.available_balance is not None else None,
        "status": a.status,
    }

def _serialize_tx(t: Transaction) -> dict:
    return {
        "transaction_id": str(t.transaction_id),
        "transaction_reference": t.transaction_reference,
        "account_id": str(t.account_id),
        "entry_type": t.entry_type,
        "transaction_type": t.transaction_type,
        "amount": float(t.amount) if t.amount is not None else None,
        "currency": t.currency,
        "status": t.status,
        "description": t.description,
        "created_at": t.created_at.isoformat() if t.created_at else None
    }


from db import session as db_session_module
logger.info("Effective DATABASE_URL: %s", getattr(db_session_module, "DATABASE_URL", os.environ.get("DATABASE_URL")))

# simple request logging middleware (fast)
from starlette.requests import Request
@app.middleware("http")
async def log_requests(request: Request, call_next):
    try:
        body = await request.body()
        logger.info("HTTP %s %s from %s body=%s", request.method, request.url.path, request.client.host if request.client else "?", body.decode(errors="ignore")[:100])
    except Exception:
        logger.exception("Failed to read request body for logging")
    response = await call_next(request)
    return response
# --------------------
# Routes
# --------------------
@app.get("/api/health")
async def health():
    return {"status": "healthy"}

@app.get("/api/users", response_model=List[UserOut])
async def list_users(limit: int = 50, offset: int = 0, db=Depends(get_db)):
    # Use ORM select which returns mapped User objects
    stmt = select(User).order_by(User.created_at.desc()).limit(limit).offset(offset)
    res = await db.execute(stmt)
    users = res.scalars().all()
    return [_serialize_user(u) for u in users]

@app.get("/api/users/{user_id}", response_model=UserOut)
async def get_user(user_id: UUID, db=Depends(get_db)):
    stmt = select(User).where(User.user_id == user_id)
    res = await db.execute(stmt)
    u = res.scalars().first()
    if not u:
        raise HTTPException(status_code=404, detail="User not found")
    return _serialize_user(u)

@app.get("/api/users/{user_id}/accounts", response_model=List[AccountOut])
async def get_user_accounts(user_id: UUID, db=Depends(get_db)):
    stmt = select(Account).where(Account.user_id == user_id)
    res = await db.execute(stmt)
    accounts = res.scalars().all()
    return [_serialize_account(a) for a in accounts]

@app.get("/api/accounts/{account_number}", response_model=AccountOut)
async def get_account(account_number: str, db=Depends(get_db)):
    acct_num = account_number.strip()
    logger.info("Lookup account_number=%s (raw=%s)", acct_num, account_number)
    stmt = select(Account).where(Account.account_number == acct_num)
    res = await db.execute(stmt)
    a = res.scalars().first()
    if not a:
        # useful debug: log count check
        try:
            count_stmt = select(func.count()).select_from(Account).where(Account.account_number == acct_num)
            cnt_res = await db.execute(count_stmt)
            cnt = cnt_res.scalar_one()
        except Exception as ex:
            logger.exception("Error counting account rows: %s", ex)
            cnt = "error"
        logger.warning("Account not found: %s (count=%s)", acct_num, cnt)
        raise HTTPException(status_code=404, detail="Account not found")
    return _serialize_account(a)


@app.get("/api/accounts/{account_number}/transactions", response_model=List[TransactionOut])
async def get_account_transactions(account_number: str, limit: int = 20, db=Depends(get_db)):
    # fetch account first
    stmt_acct = select(Account).where(Account.account_number == account_number)
    acct_res = await db.execute(stmt_acct)
    acct = acct_res.scalars().first()
    if not acct:
        raise HTTPException(status_code=404, detail="Account not found")

    stmt = select(Transaction).where(Transaction.account_id == acct.account_id).order_by(Transaction.created_at.desc()).limit(limit)
    res = await db.execute(stmt)
    txs = res.scalars().all()
    return [_serialize_tx(t) for t in txs]

# --------------------
# Transfer endpoint (simulated)
# --------------------
@app.post("/api/transfer", response_model=TransferOut)
async def transfer_funds(payload: TransferIn, db=Depends(get_db)):
    """
    Perform a funds transfer inside a single DB transaction.
    Ensures rows are locked with FOR UPDATE and all changes are committed atomically.
    """
    # Prepare txn_ref before transaction so we can return it on failures
    txn_ref = payload.reference or f"TXN{str(uuid4())[:8].upper()}"

    try:
        # Start a transaction context BEFORE running any SELECT/UPDATE
        async with db.begin():
            # Lock source account row
            stmt_from = select(Account).where(Account.account_number == payload.from_account_number).with_for_update()
            res_from = await db.execute(stmt_from)
            acct_from = res_from.scalars().first()
            if not acct_from:
                raise HTTPException(status_code=404, detail="Source account not found")

            # Lock destination account row
            stmt_to = select(Account).where(Account.account_number == payload.to_account_number).with_for_update()
            res_to = await db.execute(stmt_to)
            acct_to = res_to.scalars().first()
            if not acct_to:
                raise HTTPException(status_code=404, detail="Destination account not found")

            # Currency check
            if acct_from.currency != payload.currency or acct_to.currency != payload.currency:
                raise HTTPException(status_code=400, detail="Currency mismatch between accounts and requested transfer")

            amount = Decimal(payload.amount)
            if acct_from.balance is None:
                raise HTTPException(status_code=400, detail="Source account balance unavailable")

            # Sufficient funds check
            if Decimal(acct_from.balance) < amount:
                return TransferOut(status="failed", transaction_reference=txn_ref, message="Insufficient funds")

            # Compute new balances
            new_balance_from = (Decimal(acct_from.balance) - amount).quantize(Decimal("0.01"))
            new_balance_to = (Decimal(acct_to.balance or 0) + amount).quantize(Decimal("0.01"))

            # Update balances (use update() so concurrent sessions see DB state)
            await db.execute(
                update(Account)
                .where(Account.account_id == acct_from.account_id)
                .values(balance=new_balance_from, updated_at=func.now())
            )
            await db.execute(
                update(Account)
                .where(Account.account_id == acct_to.account_id)
                .values(balance=new_balance_to, updated_at=func.now())
            )

            # Create transaction rows (debit + credit)
            tx_from = Transaction(
                transaction_id=uuid4(),
                transaction_reference=txn_ref,
                account_id=acct_from.account_id,
                from_account_id=acct_from.account_id,
                to_account_id=acct_to.account_id,
                entry_type="debit",
                transaction_type="transfer",
                amount=amount,
                currency=payload.currency,
                fee=Decimal("0.00"),
                status="completed",
                description=f"Transfer to {payload.to_account_number}",
                # if you renamed metadata column to metadata_json, use that attr
                **({"metadata_json": None} if hasattr(Transaction, "metadata_json") else {"metadata": None}),
                balance_after=new_balance_from,
                initiated_by=payload.initiated_by_user_id
            )
            tx_to = Transaction(
                transaction_id=uuid4(),
                transaction_reference=txn_ref,
                account_id=acct_to.account_id,
                from_account_id=acct_from.account_id,
                to_account_id=acct_to.account_id,
                entry_type="credit",
                transaction_type="transfer",
                amount=amount,
                currency=payload.currency,
                fee=Decimal("0.00"),
                status="completed",
                description=f"Transfer from {payload.from_account_number}",
                **({"metadata_json": None} if hasattr(Transaction, "metadata_json") else {"metadata": None}),
                balance_after=new_balance_to,
                initiated_by=payload.initiated_by_user_id
            )
            db.add_all([tx_from, tx_to])


        # If we reach here, the transaction committed successfully
        return TransferOut(status="success", transaction_reference=txn_ref, txn_id=tx_from.transaction_id, message="Transfer completed")

    except HTTPException:
        # re-raise HTTP errors to be returned as-is
        raise
    except SQLAlchemyError as e:
        # DB-level error; already rolled back by context manager
        logger.exception("Transfer failed (DB error): %s", e)
        return TransferOut(status="failed", transaction_reference=txn_ref, message="Database error during transfer")
    except Exception as e:
        logger.exception("Transfer failed: %s", e)
        return TransferOut(status="failed", transaction_reference=txn_ref, message=str(e))

# --------------------
# Admin seed endpoint (protected by SIMPLE_ADMIN_TOKEN in .env)
# --------------------
@app.post("/api/admin/seed")
async def seed_demo(token: str = Body(..., embed=True), db=Depends(get_db)):
    expected = os.getenv("SIMPLE_ADMIN_TOKEN", "letmein")
    if token != expected:
        raise HTTPException(status_code=401, detail="Unauthorized")

    # simple idempotent seed: insert users/accounts if missing
    from sqlalchemy import select
    async with db.begin():
        # create 5 demo users and accounts if not existing
        sample_users = [
            {"username": "john_doe", "email": "john.doe@email.com", "full_name": "John Doe", "phone_number": "9990001111"},
            {"username": "jane_smith", "email": "jane.smith@email.com", "full_name": "Jane Smith", "phone_number": "9990002222"},
            {"username": "mike_wilson", "email": "mike.wilson@email.com", "full_name": "Mike Wilson", "phone_number": "9990003333"},
            {"username": "sarah_brown", "email": "sarah.brown@email.com", "full_name": "Sarah Brown", "phone_number": "9990004444"},
            {"username": "david_jones", "email": "david.jones@email.com", "full_name": "David Jones", "phone_number": "9990005555"},
        ]
        created = 0
        for su in sample_users:
            stmt = select(User).where(User.username == su["username"])
            r = await db.execute(stmt)
            exists = r.scalars().first()
            if not exists:
                u = User(
                    user_id=uuid4(),
                    username=su["username"],
                    email=su["email"],
                    password_hash="seeded",  # demo only
                    full_name=su["full_name"],
                    phone_number=su["phone_number"],
                )
                db.add(u)
                created += 1
        # Note: accounts/transactions seeding could be more elaborate. Skipping here for brevity.

    return {"seeded_users_created": created}

# --------------------
# Startup / shutdown
# --------------------
@app.on_event("startup")
async def on_startup():
    logger.info("Mock-bank starting up")

@app.on_event("shutdown")
async def on_shutdown():
    # dispose engine
    try:
        await engine.dispose()
    except Exception:
        pass
    logger.info("Mock-bank shutting down")