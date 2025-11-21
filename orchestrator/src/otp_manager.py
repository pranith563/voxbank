from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple


@dataclass
class OtpChallenge:
    code: str
    purpose: str
    created_at: datetime
    expires_at: datetime
    attempts: int = 0
    max_attempts: int = 3
    pending_tool_name: Optional[str] = None
    pending_tool_input: Optional[dict] = None
    pending_intent: Optional[str] = None


class OtpManager:
    def __init__(self, ttl_seconds: int = 300, reuse_seconds: int = 300) -> None:
        self.ttl = ttl_seconds
        self.reuse_ttl = reuse_seconds
        self._challenges: Dict[str, OtpChallenge] = {}
        self._recent_success: Dict[str, datetime] = {}

    def _is_expired(self, challenge: OtpChallenge) -> bool:
        return datetime.utcnow() > challenge.expires_at

    def has_pending(self, session_id: str) -> bool:
        challenge = self._challenges.get(session_id)
        if not challenge:
            return False
        if self._is_expired(challenge):
            self._challenges.pop(session_id, None)
            return False
        return True

    def get_pending(self, session_id: str) -> Optional[OtpChallenge]:
        challenge = self._challenges.get(session_id)
        if not challenge:
            return None
        if self._is_expired(challenge):
            self._challenges.pop(session_id, None)
            return None
        return challenge

    def clear(self, session_id: str) -> None:
        self._challenges.pop(session_id, None)

    def create_challenge(
        self,
        session_id: str,
        code: str,
        purpose: str,
        *,
        pending_tool_name: Optional[str] = None,
        pending_tool_input: Optional[dict] = None,
        pending_intent: Optional[str] = None,
        max_attempts: int = 3,
    ) -> OtpChallenge:
        challenge = OtpChallenge(
            code=code,
            purpose=purpose,
            created_at=datetime.utcnow(),
            expires_at=datetime.utcnow() + timedelta(seconds=self.ttl),
            max_attempts=max_attempts,
            pending_tool_name=pending_tool_name,
            pending_tool_input=pending_tool_input,
            pending_intent=pending_intent,
        )
        self._challenges[session_id] = challenge
        return challenge

    def verify_code(self, session_id: str, code: str) -> Tuple[bool, Optional[OtpChallenge]]:
        challenge = self.get_pending(session_id)
        if not challenge:
            return False, None

        challenge.attempts += 1
        if challenge.attempts > challenge.max_attempts:
            self.clear(session_id)
            return False, challenge

        if challenge.code == code:
            self.clear(session_id)
            self._recent_success[session_id] = datetime.utcnow()
            return True, challenge

        return False, challenge

    def is_recently_verified(self, session_id: str) -> bool:
        ts = self._recent_success.get(session_id)
        if not ts:
            return False
        if datetime.utcnow() - ts > timedelta(seconds=self.reuse_ttl):
            self._recent_success.pop(session_id, None)
            return False
        return True

    def clear_recent(self, session_id: str) -> None:
        self._recent_success.pop(session_id, None)
