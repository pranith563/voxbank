"""
Session Management
Manages user conversation sessions and context.

This module centralizes how we store and retrieve per-session state so that
the FastAPI app and the LLM agent can share a consistent session model.
"""

from __future__ import annotations

from typing import Dict, Optional, Any
from datetime import datetime, timedelta
import uuid
import json
import os

try:
    import redis  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    redis = None  # type: ignore


class SessionManager:
    """
    Manages conversation sessions and maintains context.

    Notes:
    - Sessions are kept in memory for now. For production, back this with
      Redis or a database.
    - Each session dict is expected to be compatible with the keys used in
      app.py (e.g., ``history``, ``pending_action``, ``pending_clarification``).
    """

    def __init__(self, session_timeout_minutes: int = 30, storage: Optional[Any] = None):
        self.sessions: Any = storage if storage is not None else {}
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
        self.save_session(session_id, session)
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
            self.save_session(session_id, session)
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
            self.save_session(session_id, session)
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
        self.save_session(session_id, session)

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
        self.save_session(session_id, session)

    def update_context(self, session_id: str, key: str, value: Any) -> None:
        """
        Update session context dictionary.
        """
        session = self.ensure_session(session_id, user_id=None)
        ctx = session.setdefault("context", {})
        ctx[key] = value
        self.save_session(session_id, session)

    # ------------------------------------------------------------------
    # Pending action helpers (used for confirmations/clarifications)
    # ------------------------------------------------------------------
    def set_pending_action(self, session_id: str, payload: Any) -> None:
        session = self.ensure_session(session_id, user_id=None)
        session["pending_action"] = payload
        self.save_session(session_id, session)

    def get_pending_action(self, session_id: str) -> Optional[Any]:
        session = self.get_session(session_id)
        if not session:
            return None
        return session.get("pending_action")

    def clear_pending_action(self, session_id: str) -> None:
        session = self.get_session(session_id)
        if session:
            session["pending_action"] = None
            self.save_session(session_id, session)

    def save_session(self, session_id: str, session: Dict[str, Any]) -> None:
        """
        Persist the session back to the underlying storage.
        """
        self.sessions[session_id] = session


class RedisSessionStorage:
    """
    Minimal dict-like storage that keeps session payloads in Redis as JSON.
    Each set operation refreshes the TTL so idle sessions eventually expire.
    """

    def __init__(
        self,
        redis_url: str,
        prefix: str = "voxbank:session",
        ttl_seconds: Optional[int] = None,
    ) -> None:
        if redis is None:  # pragma: no cover - defensive
            raise RuntimeError(
                "redis package is not installed. Install `redis` or set SESSION_BACKEND=memory."
            )
        self.client = redis.Redis.from_url(redis_url, decode_responses=True)
        self.prefix = prefix
        self.ttl_seconds = ttl_seconds

    def _key(self, session_id: str) -> str:
        return f"{self.prefix}:{session_id}"

    def _serialize(self, session: Dict[str, Any]) -> str:
        data = dict(session)
        for key in ("created_at", "last_activity"):
            val = data.get(key)
            if isinstance(val, datetime):
                data[key] = val.isoformat()
        return json.dumps(data)

    def _deserialize(self, payload: str) -> Dict[str, Any]:
        data = json.loads(payload)
        for key in ("created_at", "last_activity"):
            val = data.get(key)
            if isinstance(val, str):
                try:
                    data[key] = datetime.fromisoformat(val)
                except ValueError:
                    pass
        return data

    def get(self, session_id: str, default: Optional[Any] = None) -> Optional[Dict[str, Any]]:
        raw = self.client.get(self._key(session_id))
        if raw is None:
            return default
        return self._deserialize(raw)

    def __getitem__(self, session_id: str) -> Dict[str, Any]:
        session = self.get(session_id)
        if session is None:
            raise KeyError(session_id)
        return session

    def __setitem__(self, session_id: str, session: Dict[str, Any]) -> None:
        payload = self._serialize(session)
        if self.ttl_seconds and self.ttl_seconds > 0:
            self.client.setex(self._key(session_id), self.ttl_seconds, payload)
        else:
            self.client.set(self._key(session_id), payload)

    def __delitem__(self, session_id: str) -> None:
        self.client.delete(self._key(session_id))

    def touch(self, session_id: str) -> None:
        """
        Refresh TTL for a session without modifying payload.
        """
        if self.ttl_seconds and self.ttl_seconds > 0:
            self.client.expire(self._key(session_id), self.ttl_seconds)


_SESSION_MANAGER_SINGLETON: Optional[SessionManager] = None


def get_session_manager() -> SessionManager:
    """
    Build (and cache) the global SessionManager instance.
    Allows switching storage backends via environment variables.
    """
    global _SESSION_MANAGER_SINGLETON
    if _SESSION_MANAGER_SINGLETON is not None:
        return _SESSION_MANAGER_SINGLETON

    timeout = int(os.getenv("SESSION_TIMEOUT_MINUTES", "30"))
    backend = os.getenv("SESSION_BACKEND", "memory").lower()
    storage = None
    ttl_seconds = int(timedelta(minutes=timeout).total_seconds())

    if backend == "redis":
        redis_url = (
            os.getenv("SESSION_REDIS_URL")
            or os.getenv("REDIS_URL")
        )
        if not redis_url:
            raise RuntimeError("SESSION_BACKEND=redis requires SESSION_REDIS_URL or REDIS_URL to be set.")
        prefix = os.getenv("SESSION_REDIS_PREFIX", "voxbank:session")
        storage = RedisSessionStorage(redis_url=redis_url, prefix=prefix, ttl_seconds=ttl_seconds)

    _SESSION_MANAGER_SINGLETON = SessionManager(session_timeout_minutes=timeout, storage=storage)
    return _SESSION_MANAGER_SINGLETON


