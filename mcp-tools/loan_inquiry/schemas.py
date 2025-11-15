"""
Schemas for Loan Inquiry Tool
"""

from pydantic import BaseModel
from typing import Optional


class LoanDetails(BaseModel):
    loan_id: str
    loan_type: str
    principal_amount: float
    interest_rate: float
    emi_amount: float
    remaining_balance: float
    next_emi_date: str


class LoanInquiryInput(BaseModel):
    loan_id: Optional[str] = None


class LoanInquiryOutput(BaseModel):
    loans: list[LoanDetails]

