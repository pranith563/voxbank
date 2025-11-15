"""
Common utilities for MCP tools
"""

from typing import Dict, Any
from datetime import datetime


def format_currency(amount: float, currency: str = "INR") -> str:
    """
    Format amount as currency string
    """
    if currency == "INR":
        return f"₹{amount:,.2f}"
    return f"{currency} {amount:,.2f}"


def validate_amount(amount: float) -> bool:
    """
    Validate amount is positive
    """
    return amount > 0


def generate_transaction_id() -> str:
    """
    Generate unique transaction ID
    """
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    return f"TXN{timestamp}"

