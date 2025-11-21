from __future__ import annotations

import logging
import os
from decimal import Decimal
from typing import Any, Awaitable, Callable, Dict, Optional

import httpx

from otp_manager import OtpManager
from .agent import VoxBankAgent
from .helpers import format_observation_for_history

logger = logging.getLogger("agent")

OTP_SMS_API_KEY = os.getenv("OTP_SMS_API_KEY")
OTP_SMS_BASE_URL = os.getenv("OTP_SMS_BASE_URL", "https://2factor.in/API/V1")
OTP_SMS_TEMPLATE = os.getenv("OTP_SMS_TEMPLATE", "VOXBANK_OTP")
HIGH_RISK_TRANSFER_THRESHOLD = Decimal(os.getenv("OTP_TRANSFER_THRESHOLD", "10000"))
OTP_CODE_MIN_LEN = 4
OTP_CODE_MAX_LEN = 8
OTP_REUSE_WINDOW_SECONDS = int(os.getenv("OTP_REUSE_WINDOW_SECONDS", "300"))


class OtpWorkflow:
    def __init__(
        self,
        agent: VoxBankAgent,
        otp_manager: OtpManager,
        execute_tool_cb: Callable[[str, Dict[str, Any]], Awaitable[Any]],
    ) -> None:
        self.agent = agent
        self.otp_manager = otp_manager
        self.execute_tool_cb = execute_tool_cb

    async def intercept_pending_challenge(
        self,
        session_id: str,
        transcript: str,
        reply_style: str,
    ) -> Optional[Dict[str, Any]]:
        challenge = self.otp_manager.get_pending(session_id)
        if not challenge:
            return None

        digits = self._extract_digits(transcript)
        if not (OTP_CODE_MIN_LEN <= len(digits) <= OTP_CODE_MAX_LEN):
            return None

        verified, challenge_state = self.otp_manager.verify_code(session_id, digits)
        if verified and challenge_state:
            return await self._complete_pending_action(session_id, challenge_state, reply_style)

        if challenge_state:
            if challenge_state.attempts >= challenge_state.max_attempts:
                msg = "Too many incorrect OTP attempts. I've cancelled the operation."
                self.agent._append_history(session_id, {"role": "assistant", "text": msg})
                logger.info("OTP verification failed due to max attempts for session %s", session_id)
                return {"status": "clarify", "message": msg}
            else:
                msg = "That OTP seems incorrect. Please try again."
                self.agent._append_history(session_id, {"role": "assistant", "text": msg})
                return {"status": "clarify", "message": msg}
        return None

    def should_trigger_transfer_otp(
        self,
        session_id: str,
        amount_value: Any,
        session_profile: Optional[Dict[str, Any]],
    ) -> bool:
        amount = self._extract_amount(amount_value)
        if amount is None or amount < HIGH_RISK_TRANSFER_THRESHOLD:
            return False
        if self.otp_manager.is_recently_verified(session_id):
            return False
        if not session_profile or not session_profile.get("user_id"):
            return False
        if not OTP_SMS_API_KEY:
            return False
        phone = self._extract_phone_number(session_profile)
        return bool(phone)

    async def initiate_transfer_otp(
        self,
        session_id: str,
        session_profile: Optional[Dict[str, Any]],
        tool_name: str,
        tool_input: Dict[str, Any],
        intent: Optional[str],
    ) -> Optional[Dict[str, Any]]:
        phone_number = self._extract_phone_number(session_profile)
        if not phone_number:
            msg = "I could not find a registered mobile number to send the OTP. Please update your profile."
            self.agent._append_history(session_id, {"role": "assistant", "text": msg})
            return {"status": "clarify", "message": msg}

        try:
            otp_code = await self._send_otp_sms(phone_number, "high_risk_transfer")
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("Failed to send OTP SMS for session %s: %s", session_id, exc)
            msg = "I couldn't send the OTP right now. Please try again shortly."
            self.agent._append_history(session_id, {"role": "assistant", "text": msg})
            return {"status": "clarify", "message": msg}

        self.otp_manager.create_challenge(
            session_id=session_id,
            code=otp_code,
            purpose="high_risk_transfer",
            pending_tool_name=tool_name,
            pending_tool_input=tool_input,
            pending_intent=intent,
        )
        otp_message = "I've sent a 6-digit OTP to your registered mobile number. Please share the code to continue."
        self.agent._append_history(session_id, {"role": "assistant", "text": otp_message})
        logger.info("OTP challenge created for session %s (transfer)", session_id)
        return {"status": "needs_confirmation", "message": otp_message}

    async def _complete_pending_action(
        self,
        session_id: str,
        challenge,
        reply_style: str,
    ) -> Dict[str, Any]:
        tool_name = challenge.pending_tool_name
        tool_input = challenge.pending_tool_input or {}
        if not tool_name:
            msg = "OTP verified, but I couldn't resume the pending action. Please try again."
            self.agent._append_history(session_id, {"role": "assistant", "text": msg})
            logger.warning("OTP verified but no pending tool stored for session %s", session_id)
            return {"status": "clarify", "message": msg}

        try:
            logger.info("OTP verified for session %s; executing pending tool %s", session_id, tool_name)
            tool_result = await self.execute_tool_cb(tool_name, tool_input)
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("Failed to execute pending tool %s after OTP verification: %s", tool_name, exc)
            msg = "I couldn't complete the action after verifying your OTP. Please try again later."
            self.agent._append_history(session_id, {"role": "assistant", "text": msg})
            return {"status": "ok", "response": msg}

        obs_summary = format_observation_for_history(tool_name, tool_result)
        self.agent._append_history(
            session_id,
            {"role": "tool", "text": obs_summary, "detail": tool_result},
        )

        response_text = await self.agent.generate_response(
            challenge.pending_intent or "transfer",
            tool_input,
            tool_result,
            reply_style=reply_style,
        )
        self.agent._append_history(session_id, {"role": "assistant", "text": response_text})
        return {"status": "ok", "response": response_text, "tool_result": tool_result}

    async def _send_otp_sms(self, phone_number: str, purpose: str) -> str:
        if not OTP_SMS_API_KEY:
            raise RuntimeError("OTP SMS API key not configured")
        normalized_phone = self._extract_digits(phone_number)
        if not normalized_phone:
            raise RuntimeError("Invalid phone number for OTP")

        base = OTP_SMS_BASE_URL.rstrip("/")
        template_name = OTP_SMS_TEMPLATE or "VOXBANK_OTP"
        url = f"{base}/{OTP_SMS_API_KEY}/SMS/{normalized_phone}/AUTOGEN2/{template_name}"
        async with httpx.AsyncClient(timeout=8.0) as client:
            response = await client.get(url)
        response.raise_for_status()
        data = response.json()
        if data.get("Status") != "Success" or "OTP" not in data:
            raise RuntimeError(f"OTP service returned error: {data}")
        logger.info("OTP SMS requested for phone ending in %s (purpose=%s)", normalized_phone[-4:], purpose)
        return str(data["OTP"])

    def _extract_amount(self, value: Any) -> Optional[Decimal]:
        if value is None:
            return None
        try:
            if isinstance(value, Decimal):
                return value
            if isinstance(value, (int, float)):
                return Decimal(str(value))
            if isinstance(value, str) and value.strip():
                return Decimal(value.strip())
        except Exception:
            return None
        return None

    def _extract_digits(self, text: Optional[str]) -> str:
        if not text:
            return ""
        return "".join(ch for ch in text if ch.isdigit())

    def _extract_phone_number(self, session_profile: Optional[Dict[str, Any]]) -> Optional[str]:
        if not session_profile:
            return None
        user_profile = session_profile.get("user_profile") or {}
        phone = user_profile.get("phone_number") or session_profile.get("phone_number")
        if not phone:
            return None
        digits = self._extract_digits(phone)
        return digits if digits else None
