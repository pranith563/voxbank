"""
Entity Resolution Module
Extracts and resolves entities from user input
"""

from typing import Dict, List, Optional
import re


class EntityResolver:
    """
    Extracts entities like amounts, account numbers, names, dates from transcripts
    """
    
    def extract_entities(self, transcript: str, intent: str) -> Dict[str, any]:
        """
        Extract relevant entities based on intent
        """
        entities = {}
        
        # Extract amount
        amount_pattern = r'[₹$]?(\d+(?:,\d{3})*(?:\.\d{2})?)'
        amount_matches = re.findall(amount_pattern, transcript)
        if amount_matches:
            entities["amount"] = float(amount_matches[0].replace(',', ''))
        
        # Extract recipient name (simple pattern)
        # TODO: Implement proper NER
        if "to" in transcript.lower():
            parts = transcript.lower().split("to")
            if len(parts) > 1:
                entities["recipient"] = parts[1].strip().split()[0]
        
        # Extract date/time references
        date_keywords = ["today", "tomorrow", "next week", "month"]
        for keyword in date_keywords:
            if keyword in transcript.lower():
                entities["date"] = keyword
                break
        
        return entities

