"""
Loan Inquiry MCP Tool Service
"""

from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

app = FastAPI(title="Loan Inquiry MCP Tool")


class LoanInquiryRequest(BaseModel):
    user_id: str
    params: dict


class LoanInquiryResponse(BaseModel):
    success: bool
    loan_details: Optional[dict] = None
    error: Optional[str] = None


@app.post("/execute", response_model=LoanInquiryResponse)
async def get_loan_info(request: LoanInquiryRequest):
    """
    Get loan information
    """
    try:
        # TODO: Implement loan inquiry logic
        return LoanInquiryResponse(
            success=True,
            loan_details={}
        )
    except Exception as e:
        return LoanInquiryResponse(success=False, error=str(e))


@app.get("/health")
async def health():
    return {"status": "healthy"}

