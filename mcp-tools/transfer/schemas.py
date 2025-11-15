"""
Schemas for Transfer Tool
"""

from pydantic import BaseModel


class TransferInput(BaseModel):
    from_account: str
    to_account: str
    amount: float
    description: str = ""


class TransferOutput(BaseModel):
    transaction_id: str
    status: str
    from_account: str
    to_account: str
    amount: float
    timestamp: str

