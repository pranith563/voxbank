"""
Conversation Memory
Long-term memory for user preferences and past interactions
"""

from typing import Dict, List, Optional
from datetime import datetime


class ConversationMemory:
    """
    Stores and retrieves long-term conversation context
    """
    
    def __init__(self):
        self.user_memories: Dict[str, List[Dict]] = {}
    
    def store_interaction(self, user_id: str, intent: str, entities: Dict, outcome: str):
        """
        Store user interaction for future reference
        """
        if user_id not in self.user_memories:
            self.user_memories[user_id] = []
        
        self.user_memories[user_id].append({
            "intent": intent,
            "entities": entities,
            "outcome": outcome,
            "timestamp": datetime.now().isoformat()
        })
    
    def get_user_preferences(self, user_id: str) -> Dict:
        """
        Retrieve user preferences from past interactions
        """
        if user_id not in self.user_memories:
            return {}
        
        # TODO: Analyze past interactions to extract preferences
        return {}
    
    def get_recent_interactions(self, user_id: str, limit: int = 5) -> List[Dict]:
        """
        Get recent interactions for context
        """
        if user_id not in self.user_memories:
            return []
        
        return self.user_memories[user_id][-limit:]

