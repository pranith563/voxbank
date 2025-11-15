# mock-bank/src/db/crud.py
from typing import List, Optional
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from models import User, Account, Transaction

async def get_user_by_id(db: AsyncSession, user_id):
    q = select(User).where(User.user_id == user_id)
    res = await db.execute(q)
    return res.scalars().first()

async def list_users(db: AsyncSession, limit: int = 50, offset: int = 0) -> List[User]:
    q = select(User).limit(limit).offset(offset)
    res = await db.execute(q)
    return res.scalars().all()

async def get_accounts_for_user(db: AsyncSession, user_id) -> List[Account]:
    q = select(Account).where(Account.user_id == user_id)
    res = await db.execute(q)
    return res.scalars().all()

async def get_account_by_number(db: AsyncSession, account_number: str) -> Optional[Account]:
    q = select(Account).where(Account.account_number == account_number)
    res = await db.execute(q)
    return res.scalars().first()

async def get_transactions_for_account(db: AsyncSession, account_id, limit: int = 20):
    q = select(Transaction).where(Transaction.account_id == account_id).order_by(Transaction.created_at.desc()).limit(limit)
    res = await db.execute(q)
    return res.scalars().all()
