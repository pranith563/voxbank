"""
Schemas for Balance Tool
"""

from pydantic import BaseModel
from typing import Optional


class BalanceInput(BaseModel):
    account_id: Optional[str] = None


class BalanceOutput(BaseModel):
    balance: float
    account_id: str
    currency: str
    last_updated: str

