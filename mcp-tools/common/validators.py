"""
Common validation utilities
"""

from typing import Dict, Any, List


def validate_account_id(account_id: str) -> bool:
    """
    Validate account ID format
    """
    return account_id is not None and len(account_id) >= 3


def validate_transfer_params(params: Dict[str, Any]) -> List[str]:
    """
    Validate transfer parameters
    Returns list of validation errors
    """
    errors = []
    
    if "from_account" not in params or not params["from_account"]:
        errors.append("from_account is required")
    
    if "to_account" not in params or not params["to_account"]:
        errors.append("to_account is required")
    
    if "amount" not in params:
        errors.append("amount is required")
    elif params["amount"] <= 0:
        errors.append("amount must be positive")
    
    return errors

