"""
Intent Classification Module
Classifies user intents from transcripts
"""

from typing import Dict, List, Tuple
import json


class IntentClassifier:
    """
    Classifies user intents from natural language input
    """
    
    def __init__(self):
        self.intent_mappings = {
            "balance_inquiry": ["balance", "how much", "account balance"],
            "transfer": ["transfer", "send money", "pay"],
            "transactions": ["transactions", "history", "statement"],
            "loan_inquiry": ["loan", "emi", "interest"],
            "reminder": ["remind", "reminder", "schedule"]
        }
    
    def classify(self, transcript: str) -> Tuple[str, float]:
        """
        Classify intent from transcript
        Returns: (intent, confidence_score)
        """
        transcript_lower = transcript.lower()
        
        # TODO: Implement proper NLU with embeddings/LLM
        for intent, keywords in self.intent_mappings.items():
            if any(keyword in transcript_lower for keyword in keywords):
                return (intent, 0.85)
        
        return ("unknown", 0.0)

