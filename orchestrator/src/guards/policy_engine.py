"""
Policy Engine
Enforces security and compliance policies
"""

from typing import Dict, Any, List
from enum import Enum


class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class PolicyEngine:
    """
    Evaluates actions against security and compliance policies
    """
    
    def __init__(self):
        self.risk_rules = {
            "balance_inquiry": RiskLevel.LOW,
            "transactions": RiskLevel.LOW,
            "transfer": RiskLevel.HIGH,
            "loan_inquiry": RiskLevel.MEDIUM,
            "reminder": RiskLevel.LOW
        }
    
    def evaluate_action(self, intent: str, entities: Dict[str, Any], user_id: str) -> Dict[str, Any]:
        """
        Evaluate if action is allowed and what verification is required
        """
        risk_level = self.risk_rules.get(intent, RiskLevel.MEDIUM)
        
        result = {
            "allowed": True,
            "risk_level": risk_level.value,
            "requires_otp": False,
            "requires_biometric": False,
            "requires_confirmation": False
        }
        
        # High risk actions require additional verification
        if risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
            result["requires_otp"] = True
            result["requires_biometric"] = True
            result["requires_confirmation"] = True
        
        # Check amount thresholds
        if "amount" in entities:
            amount = entities["amount"]
            if amount > 100000:  # High value transaction
                result["requires_otp"] = True
                result["requires_confirmation"] = True
        
        return result

