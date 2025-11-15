"""
Common authentication utilities
"""

from typing import Optional
import jwt


def verify_token(token: str, secret: str) -> Optional[dict]:
    """
    Verify JWT token
    """
    try:
        payload = jwt.decode(token, secret, algorithms=["HS256"])
        return payload
    except jwt.InvalidTokenError:
        return None


def require_auth(user_id: str) -> bool:
    """
    Check if user is authenticated
    """
    # TODO: Implement proper auth check
    return user_id is not None and len(user_id) > 0

