"""
Session profile helpers

Provides:
- session_manager: shared in-memory SessionManager instance
- get_session_profile(session_id)
- hydrate_session_profile_from_mock_bank(session_id, user_id)
"""

from typing import Dict, Any, Optional, List
import os
import httpx

from logging_config import get_logger
from .session_manager import get_session_manager
from .voice_profile import get_voice_profile


logger = get_logger("voxbank.orchestrator")

VOX_BANK_BASE_URL = os.getenv("VOX_BANK_BASE_URL", "http://localhost:9000")

session_manager = get_session_manager()


def get_session_profile(session_id: str) -> Dict[str, Any]:
    """
    Return a compact session profile structure for the given session_id.

    This is used by downstream endpoints so they don't need to know the
    internal layout of the SessionManager's session dict.
    """
    sess = session_manager.get_session(session_id)
    if not sess:
        logger.info("Session profile requested for unknown session_id=%s", session_id)
        return {
            "user_id": None,
            "username": None,
            "is_authenticated": False,
            "is_voice_verified": False,
            "primary_account": None,
            "accounts": [],
        }

    profile = {
        "user_id": sess.get("user_id"),
        "username": sess.get("username"),
        "is_authenticated": bool(sess.get("is_authenticated")),
        "is_voice_verified": bool(sess.get("is_voice_verified")),
        "primary_account": sess.get("primary_account"),
        "accounts": sess.get("accounts") or [],
        "voice_profile": get_voice_profile(sess),
    }
    logger.info(
        "Session profile for %s -> user_id=%s username=%s primary_account=%s accounts=%d",
        session_id,
        profile["user_id"],
        profile["username"],
        profile["primary_account"],
        len(profile["accounts"]),
    )
    return profile


async def hydrate_session_profile_from_mock_bank(session_id: str, user_id: str) -> None:
    """
    Fetch user profile + accounts from mock-bank and cache them on the session.

    This is used after a successful login (either via HTTP login endpoint or
    conversational auth) so that subsequent LLM/tool calls can rely on a
    populated session_profile.
    """
    base = VOX_BANK_BASE_URL.rstrip("/") if VOX_BANK_BASE_URL else None
    if not base:
        logger.error("hydrate_session_profile: VOX_BANK_BASE_URL not configured; cannot hydrate profile")
        return

    accounts_compact: List[Dict[str, Any]] = []
    primary_account: Optional[str] = None
    profile_data: Dict[str, Any] = {}

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            profile_url = f"{base}/api/users/{user_id}"
            accounts_url = f"{base}/api/users/{user_id}/accounts"

            logger.info("hydrate_session_profile: fetching profile from %s", profile_url)
            try:
                profile_resp = await client.get(profile_url)
                profile_resp.raise_for_status()
                profile_data = profile_resp.json() or {}
                logger.info(
                    "hydrate_session_profile: profile fetch successful for user_id=%s username=%s",
                    user_id,
                    profile_data.get("username"),
                )
            except Exception as e:
                logger.warning("hydrate_session_profile: profile fetch failed for user_id=%s error=%s", user_id, e)

            logger.info("hydrate_session_profile: fetching accounts from %s", accounts_url)
            try:
                accounts_resp = await client.get(accounts_url)
                accounts_resp.raise_for_status()
                accounts_full = accounts_resp.json() or []

                for acc in accounts_full:
                    acct_num = acc.get("account_number")
                    if not acct_num:
                        continue
                    accounts_compact.append(
                        {
                            "account_number": acct_num,
                            "account_type": acc.get("account_type"),
                            "currency": acc.get("currency"),
                        }
                    )
                # Decide primary account: prefer savings, else first
                for acc in accounts_full:
                    if (acc.get("account_type") or "").strip().lower() == "savings":
                        primary_account = acc.get("account_number")
                        break
                if not primary_account and accounts_full:
                    primary_account = accounts_full[0].get("account_number")

                logger.info(
                    "hydrate_session_profile: cached %d accounts for user_id=%s primary_account=%s",
                    len(accounts_compact),
                    user_id,
                    primary_account,
                )
            except Exception as e:
                logger.warning("hydrate_session_profile: accounts fetch failed for user_id=%s error=%s", user_id, e)
    except Exception as e:
        logger.exception("hydrate_session_profile: unexpected error while fetching profile/accounts: %s", e)

    # Update session profile fields
    sess = session_manager.ensure_session(session_id, user_id=user_id)
    # Prefer username from profile if available
    if profile_data.get("username"):
        sess["username"] = profile_data["username"]
    sess["accounts"] = accounts_compact
    sess["primary_account"] = primary_account
    sess["is_authenticated"] = True
