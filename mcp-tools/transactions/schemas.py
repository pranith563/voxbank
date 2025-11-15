"""
Schemas for Transactions Tool
"""

from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime


class Transaction(BaseModel):
    transaction_id: str
    date: str
    amount: float
    type: str
    description: str
    balance: float


class TransactionsInput(BaseModel):
    account_id: str
    limit: int = 10
    start_date: Optional[str] = None
    end_date: Optional[str] = None


class TransactionsOutput(BaseModel):
    transactions: List[Transaction]
    total_count: int

