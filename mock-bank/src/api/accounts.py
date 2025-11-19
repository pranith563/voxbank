from decimal import Decimal
from typing import List
from uuid import UUID, uuid4
import os

from fastapi import APIRouter, Body, Depends, HTTPException
from sqlalchemy import func, select, update
from sqlalchemy.exc import SQLAlchemyError

from db.models import Account, Transaction, User
from logging_config import get_logger
from .deps import get_db
from .schemas import AccountOut, TransactionOut, TransferIn, TransferOut
from .serializers import serialize_account, serialize_tx

logger = get_logger("mock_bank.api.accounts")

router = APIRouter(tags=["accounts"])


@router.get("/accounts/{account_number}", response_model=AccountOut)
async def get_account(account_number: str, db=Depends(get_db)):
    """
    Fetch a single account by account_number.
    """
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
    return serialize_account(a)


@router.get("/accounts/{account_number}/transactions", response_model=List[TransactionOut])
async def get_account_transactions(account_number: str, limit: int = 20, db=Depends(get_db)):
    """
    Return recent transactions for an account.
    """
    logger.info("Fetching transactions for account_number=%s limit=%s", account_number, limit)
    stmt_acct = select(Account).where(Account.account_number == account_number)
    acct_res = await db.execute(stmt_acct)
    acct = acct_res.scalars().first()
    if not acct:
        logger.warning("Account not found for transactions account_number=%s", account_number)
        raise HTTPException(status_code=404, detail="Account not found")

    stmt = (
        select(Transaction)
        .where(Transaction.account_id == acct.account_id)
        .order_by(Transaction.created_at.desc())
        .limit(limit)
    )
    res = await db.execute(stmt)
    txs = res.scalars().all()
    return [serialize_tx(t) for t in txs]


@router.post("/transfer", response_model=TransferOut)
async def transfer_funds(payload: TransferIn, db=Depends(get_db)):
    """
    Perform a funds transfer inside a single DB transaction.
    Ensures rows are locked with FOR UPDATE and all changes are committed atomically.
    """
    txn_ref = payload.reference or f"TXN{str(uuid4())[:8].upper()}"
    logger.info(
        "Transfer request from=%s to=%s amount=%s currency=%s",
        payload.from_account_number,
        payload.to_account_number,
        payload.amount,
        payload.currency,
    )

    try:
        async with db.begin():
            # Lock source account row
            stmt_from = (
                select(Account)
                .where(Account.account_number == payload.from_account_number)
                .with_for_update()
            )
            res_from = await db.execute(stmt_from)
            acct_from = res_from.scalars().first()
            if not acct_from:
                raise HTTPException(status_code=404, detail="Source account not found")

            # Lock destination account row
            stmt_to = (
                select(Account)
                .where(Account.account_number == payload.to_account_number)
                .with_for_update()
            )
            res_to = await db.execute(stmt_to)
            acct_to = res_to.scalars().first()
            if not acct_to:
                raise HTTPException(status_code=404, detail="Destination account not found")

            # Currency check
            if acct_from.currency != payload.currency or acct_to.currency != payload.currency:
                raise HTTPException(
                    status_code=400,
                    detail="Currency mismatch between accounts and requested transfer",
                )

            amount = Decimal(payload.amount)
            if acct_from.balance is None:
                raise HTTPException(status_code=400, detail="Source account has no balance set")

            if Decimal(acct_from.balance) < amount:
                logger.warning(
                    "Transfer failed - insufficient funds from=%s balance=%s amount=%s",
                    payload.from_account_number,
                    acct_from.balance,
                    amount,
                )
                return TransferOut(
                    status="failed",
                    transaction_reference=txn_ref,
                    message="Insufficient funds",
                )

            # Compute new balances
            new_balance_from = (Decimal(acct_from.balance) - amount).quantize(Decimal("0.01"))
            new_balance_to = (Decimal(acct_to.balance or 0) + amount).quantize(Decimal("0.01"))

            # Update balances
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
                **({"metadata_json": None} if hasattr(Transaction, "metadata_json") else {"metadata": None}),
                balance_after=new_balance_from,
                initiated_by=payload.initiated_by_user_id,
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
                initiated_by=payload.initiated_by_user_id,
            )
            db.add_all([tx_from, tx_to])

        logger.info(
            "Transfer success txn_ref=%s from=%s to=%s amount=%s",
            txn_ref,
            payload.from_account_number,
            payload.to_account_number,
            amount,
        )
        return TransferOut(
            status="success",
            transaction_reference=txn_ref,
            txn_id=tx_from.transaction_id,
            message="Transfer completed",
        )

    except HTTPException:
        # Re-raise HTTP errors directly
        raise
    except SQLAlchemyError as e:
        logger.exception("Transfer failed (DB error): %s", e)
        return TransferOut(
            status="failed",
            transaction_reference=txn_ref,
            message="Database error during transfer",
        )
    except Exception as e:
        logger.exception("Transfer failed: %s", e)
        return TransferOut(
            status="failed",
            transaction_reference=txn_ref,
            message=str(e),
        )


@router.post("/admin/seed")
async def seed_demo(token: str = Body(..., embed=True), db=Depends(get_db)):
    """
    Simple idempotent seeding of demo users.
    Protected by SIMPLE_ADMIN_TOKEN in environment.
    """
    expected = os.getenv("SIMPLE_ADMIN_TOKEN", "letmein")
    if token != expected:
        logger.warning("Admin seed unauthorized attempt")
        raise HTTPException(status_code=401, detail="Unauthorized")

    from sqlalchemy import select as sa_select

    created = 0
    async with db.begin():
        sample_users = [
            {
                "username": "john_doe",
                "email": "john.doe@email.com",
                "full_name": "John Doe",
                "phone_number": "9990001111",
            },
            {
                "username": "jane_smith",
                "email": "jane.smith@email.com",
                "full_name": "Jane Smith",
                "phone_number": "9990002222",
            },
            {
                "username": "mike_wilson",
                "email": "mike.wilson@email.com",
                "full_name": "Mike Wilson",
                "phone_number": "9990003333",
            },
            {
                "username": "sarah_brown",
                "email": "sarah.brown@email.com",
                "full_name": "Sarah Brown",
                "phone_number": "9990004444",
            },
            {
                "username": "david_jones",
                "email": "david.jones@email.com",
                "full_name": "David Jones",
                "phone_number": "9990005555",
            },
        ]
        for su in sample_users:
            stmt = sa_select(User).where(User.username == su["username"])
            r = await db.execute(stmt)
            exists = r.scalars().first()
            if not exists:
                u = User(
                    user_id=uuid4(),
                    username=su["username"],
                    email=su["email"],
                    passphrase="seeded",
                    password_hash="seeded",
                    full_name=su["full_name"],
                    phone_number=su["phone_number"],
                )
                db.add(u)
                created += 1

    logger.info("Admin seed complete; created=%s users", created)
    return {"seeded_users_created": created}
