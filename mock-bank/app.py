"""
Mock Bank Backend
Fake bank system for testing MCP tools
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict
import json
import os

app = FastAPI(title="Mock Bank API")

# Simple in-memory database
db: Dict = {
    "users": {},
    "accounts": {},
    "transactions": [],
    "loans": {}
}


class Account(BaseModel):
    account_id: str
    user_id: str
    balance: float
    account_type: str
    currency: str = "INR"


class Transaction(BaseModel):
    transaction_id: str
    from_account: str
    to_account: str
    amount: float
    timestamp: str
    status: str


@app.get("/")
async def root():
    return {"message": "Mock Bank API", "status": "running"}


@app.get("/api/users/{user_id}/accounts")
async def get_user_accounts(user_id: str):
    """
    Get all accounts for a user
    """
    # TODO: Load from db.json
    return {
        "user_id": user_id,
        "accounts": [
            {
                "account_id": "ACC001",
                "balance": 50000.00,
                "account_type": "savings",
                "currency": "INR"
            }
        ]
    }


@app.get("/api/accounts/{account_id}/balance")
async def get_balance(account_id: str):
    """
    Get account balance
    """
    return {
        "account_id": account_id,
        "balance": 50000.00,
        "currency": "INR"
    }


@app.post("/api/transfers")
async def create_transfer(transfer: dict):
    """
    Execute a transfer
    """
    # TODO: Implement transfer logic
    return {
        "transaction_id": "TXN123456",
        "status": "completed"
    }


@app.get("/api/accounts/{account_id}/transactions")
async def get_transactions(account_id: str, limit: int = 10):
    """
    Get transaction history
    """
    return {
        "account_id": account_id,
        "transactions": []
    }


@app.get("/api/health")
async def health():
    return {"status": "healthy"}

