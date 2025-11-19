"""
Session Management
Manages user conversation sessions and context.

This module centralizes how we store and retrieve per-session state so that
the FastAPI app and the LLM agent can share a consistent session model.
"""

from typing import Dict, Optional, Any
from datetime import datetime, timedelta
import uuid


class SessionManager:
    """
    Manages conversation sessions and maintains context.

    Notes:
    - Sessions are kept in memory for now. For production, back this with
      Redis or a database.
    - Each session dict is expected to be compatible with the keys used in
      app.py (e.g., ``history``, ``pending_action``, ``pending_clarification``).
    """

    def __init__(self, session_timeout_minutes: int = 30):
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self.session_timeout = timedelta(minutes=session_timeout_minutes)

    # ------------------------------------------------------------------
    # Core helpers
    # ------------------------------------------------------------------
    def _init_session(self, session_id: str, user_id: Optional[str]) -> Dict[str, Any]:
        now = datetime.now()
        session: Dict[str, Any] = {
            "session_id": session_id,
            "user_id": user_id,
            "username": None,
            # Auth / profile flags
            "is_authenticated": False,
            "is_voice_verified": False,
            # Simple cached profile info for tools/LLM
            "primary_account": None,
            "accounts": [],
            "created_at": now,
            "last_activity": now,
            # High-level chat history used by the orchestrator (role/text)
            "history": [],
            # Optional richer history/context if needed by callers
            "conversation_history": [],
            "pending_action": None,
            "pending_clarification": None,
            "context": {},
        }
        self.sessions[session_id] = session
        return session

    def create_session(self, user_id: str) -> str:
        """
        Create a new conversation session with a generated session_id.
        """
        session_id = str(uuid.uuid4())
        self._init_session(session_id, user_id)
        return session_id

    def ensure_session(self, session_id: str, user_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Ensure a session exists for the given session_id.

        If the session is missing or expired, it will be (re)created.
        """
        session = self.get_session(session_id)
        if session is None:
            session = self._init_session(session_id, user_id)
        # Optionally refresh user_id if newly provided
        if user_id and not session.get("user_id"):
            session["user_id"] = user_id
        return session

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get session data, or None if not found/expired.
        """
        session = self.sessions.get(session_id)
        if session:
            now = datetime.now()
            # Check if session expired
            if now - session.get("last_activity", now) > self.session_timeout:
                # Drop expired session
                del self.sessions[session_id]
                return None
            session["last_activity"] = now
        return session

    # ------------------------------------------------------------------
    # History & context helpers
    # ------------------------------------------------------------------
    def add_history_message(self, session_id: str, role: str, text: str) -> None:
        """
        Append a message to the session's role-based history.
        """
        session = self.ensure_session(session_id, user_id=None)
        history = session.setdefault("history", [])
        history.append(
            {
                "role": role,
                "text": text,
                "timestamp": datetime.now().isoformat(),
            }
        )

    def add_to_history(self, session_id: str, user_input: str, bot_response: str) -> None:
        """
        Backwards-compatible helper that records a user/bot turn in
        ``conversation_history``. This is not used by app.py but remains
        available for any callers that prefer this shape.
        """
        session = self.ensure_session(session_id, user_id=None)
        convo = session.setdefault("conversation_history", [])
        convo.append(
            {
                "user": user_input,
                "bot": bot_response,
                "timestamp": datetime.now().isoformat(),
            }
        )

    def update_context(self, session_id: str, key: str, value: Any) -> None:
        """
        Update session context dictionary.
        """
        session = self.ensure_session(session_id, user_id=None)
        ctx = session.setdefault("context", {})
        ctx[key] = value

    # ------------------------------------------------------------------
    # Pending action helpers (used for confirmations/clarifications)
    # ------------------------------------------------------------------
    def set_pending_action(self, session_id: str, payload: Any) -> None:
        session = self.ensure_session(session_id, user_id=None)
        session["pending_action"] = payload

    def get_pending_action(self, session_id: str) -> Optional[Any]:
        session = self.get_session(session_id)
        if not session:
            return None
        return session.get("pending_action")

    def clear_pending_action(self, session_id: str) -> None:
        session = self.get_session(session_id)
        if session:
            session["pending_action"] = None

