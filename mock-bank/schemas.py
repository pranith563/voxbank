"""
Schemas for Mock Bank API
"""

from pydantic import BaseModel
from typing import Optional
from datetime import datetime


class User(BaseModel):
    user_id: str
    name: str
    email: str
    phone: str


class Account(BaseModel):
    account_id: str
    user_id: str
    balance: float
    account_type: str
    currency: str = "INR"


class TransferRequest(BaseModel):
    from_account: str
    to_account: str
    amount: float
    description: Optional[str] = ""


class TransferResponse(BaseModel):
    transaction_id: str
    status: str
    timestamp: str

