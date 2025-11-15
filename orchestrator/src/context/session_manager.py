"""
Session Management
Manages user conversation sessions and context
"""

from typing import Dict, Optional
from datetime import datetime, timedelta
import uuid


class SessionManager:
    """
    Manages conversation sessions and maintains context
    """
    
    def __init__(self, session_timeout_minutes: int = 30):
        self.sessions: Dict[str, Dict] = {}
        self.session_timeout = timedelta(minutes=session_timeout_minutes)
    
    def create_session(self, user_id: str) -> str:
        """
        Create a new conversation session
        """
        session_id = str(uuid.uuid4())
        self.sessions[session_id] = {
            "user_id": user_id,
            "created_at": datetime.now(),
            "last_activity": datetime.now(),
            "conversation_history": [],
            "context": {}
        }
        return session_id
    
    def get_session(self, session_id: str) -> Optional[Dict]:
        """
        Get session data
        """
        session = self.sessions.get(session_id)
        if session:
            # Check if session expired
            if datetime.now() - session["last_activity"] > self.session_timeout:
                del self.sessions[session_id]
                return None
            session["last_activity"] = datetime.now()
        return session
    
    def add_to_history(self, session_id: str, user_input: str, bot_response: str):
        """
        Add conversation turn to history
        """
        session = self.get_session(session_id)
        if session:
            session["conversation_history"].append({
                "user": user_input,
                "bot": bot_response,
                "timestamp": datetime.now().isoformat()
            })
    
    def update_context(self, session_id: str, key: str, value: any):
        """
        Update session context
        """
        session = self.get_session(session_id)
        if session:
            session["context"][key] = value

