"""
Risk Rules
Defines risk assessment rules for different actions
"""

from typing import Dict, Any
from datetime import datetime, time


class RiskRules:
    """
    Applies risk assessment rules
    """
    
    def __init__(self):
        self.max_daily_transfer = 1000000  # 10 lakh
        self.max_single_transfer = 500000  # 5 lakh
        self.suspicious_hours = (time(2, 0), time(5, 0))  # 2 AM to 5 AM
    
    def check_transfer_limits(self, amount: float, user_id: str, daily_total: float) -> Dict[str, Any]:
        """
        Check if transfer amount is within limits
        """
        violations = []
        
        if amount > self.max_single_transfer:
            violations.append(f"Amount exceeds single transfer limit of ₹{self.max_single_transfer}")
        
        if daily_total + amount > self.max_daily_transfer:
            violations.append(f"Amount exceeds daily transfer limit of ₹{self.max_daily_transfer}")
        
        return {
            "allowed": len(violations) == 0,
            "violations": violations
        }
    
    def check_time_based_risk(self) -> Dict[str, Any]:
        """
        Check if current time is suspicious
        """
        current_time = datetime.now().time()
        is_suspicious = self.suspicious_hours[0] <= current_time <= self.suspicious_hours[1]
        
        return {
            "suspicious": is_suspicious,
            "requires_extra_verification": is_suspicious
        }

