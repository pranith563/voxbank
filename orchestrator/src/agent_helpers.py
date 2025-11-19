"""
agent_helpers.py

Utility helpers for LLMAgent to keep llm_agent.py focused on core
decision / orchestration logic.
"""

from typing import Any, Dict, List, Optional
import json
import re
from decimal import Decimal
import logging


logger = logging.getLogger("llm_agent.helpers")


def format_amount(value: Any, currency: str = "USD") -> Optional[str]:
    """Return a nicely formatted amount string or None if invalid."""
    if value is None:
        return None
    try:
        if isinstance(value, str):
            cleaned = re.sub(r"[^\d.\-]", "", value)
            num = Decimal(cleaned)
        else:
            num = Decimal(str(value))
    except Exception:
        return None

    if num == num.to_integral():
        return f"{num:.0f} {currency}"
    return f"{num.quantize(Decimal('0.01'))} {currency}"


def mask_account(acct: Optional[str]) -> Optional[str]:
    """Return masked account like ACC***9001 or 'primary' unchanged."""
    if not acct:
        return None
    acct_str = str(acct)
    if acct_str.lower() in ("primary", "default"):
        return acct_str

    last4_digits = re.sub(r"\D", "", acct_str)[-4:]
    prefix = acct_str.split(last4_digits)[0] if last4_digits and last4_digits in acct_str else acct_str[:-4]
    if prefix:
        return f"{prefix}****{last4_digits}"
    return f"****{last4_digits}"


def is_raw_tool_output(text: str) -> bool:
    """
    Heuristic to detect when an assistant response looks like an
    unpolished/raw tool dump (tables, bullets, lots of account numbers).
    """
    if not text or len(text) < 20:
        return False
    if re.search(r"\btransactions?\b", text, flags=re.I):
        return True
    if re.search(r"\bACC[0-9A-Za-z]{3,}\b", text):
        return True
    if re.search(r"^\s*[-\u2022]\s+", text, flags=re.M):
        return True
    if re.search(r"\d{4}-\d{2}-\d{2}", text):
        return True
    return False


def render_history_for_prompt(history: List[Dict[str, Any]], max_messages: int = 8) -> str:
    """
    Render a short text block from recent conversation history for use
    inside the decision prompt.
    """
    if not history:
        return ""
    recent = history[-max_messages:]
    lines: List[str] = []
    for msg in recent:
        role = msg.get("role", "user")
        text = msg.get("text", "")
        lines.append(f"{role}: {text}")
    return "\n".join(lines)


def format_observation_for_history(tool_name: str, observation: Any) -> str:
    """
    Summarise a tool observation so it can be stored compactly in history.
    """
    try:
        if isinstance(observation, dict):
            if "balance" in observation:
                return f"{tool_name} -> balance: {observation.get('balance')} {observation.get('currency', '')}"
            if "transactions" in observation and isinstance(observation.get("transactions"), list):
                return f"{tool_name} -> returned {len(observation.get('transactions'))} transactions"
            if "status" in observation and observation.get("status") != "success":
                return f"{tool_name} -> status: {observation.get('status')} - {str(observation.get('message', ''))}"
            short = json.dumps(observation, default=str)
            return f"{tool_name} -> {short[:200]}"
        if isinstance(observation, list):
            return f"{tool_name} -> list length {len(observation)}"
        return f"{tool_name} -> {str(observation)[:200]}"
    except Exception:
        return f"{tool_name} -> (unserializable observation)"


def resolve_account_from_profile(
    session_profile: Optional[Dict[str, Any]],
    label: Optional[str],
) -> Optional[str]:
    """
    Given a session_profile and an abstract label like "primary", "savings",
    or "current", return a concrete account_number if possible.
    """
    if not session_profile:
        return None

    primary = session_profile.get("primary_account")
    accounts = session_profile.get("accounts") or []

    if not label:
        return primary

    label_norm = str(label).strip().lower()

    if label_norm in {"primary", "default", "my account", "my primary account"}:
        return primary

    def find_by_type(substr: str) -> Optional[str]:
        for acc in accounts:
            t = (acc.get("account_type") or "").strip().lower()
            if substr in t:
                return acc.get("account_number")
        return None

    if "saving" in label_norm:
        return find_by_type("saving") or primary
    if "current" in label_norm or "checking" in label_norm:
        candidate = find_by_type("current") or find_by_type("checking")
        return candidate or primary

    if re.fullmatch(r"ACC[0-9A-Za-z\-]+", label_norm, flags=re.I):
        return label
    if re.fullmatch(r"\d{6,}", label_norm):
        return label

    return None


def deterministic_fallback(
    intent: str,
    entities: Dict[str, Any],
    tool_result: Optional[Any],
) -> str:
    """
    Deterministic safe messages when the LLM output is invalid or
    not trustworthy.
    """
    try:
        if intent == "balance" and isinstance(tool_result, dict):
            bal = tool_result.get("balance") or tool_result.get("available_balance")
            cur = tool_result.get("currency") or "USD"
            bal_str = format_amount(bal, cur)
            acct = mask_account(entities.get("account_number") or tool_result.get("account_number"))
            if bal_str:
                return f"Your account {acct} has a balance of {bal_str}."

        if intent == "transactions":
            if isinstance(tool_result, dict):
                txs = tool_result.get("transactions")
            elif isinstance(tool_result, list):
                txs = tool_result
            else:
                txs = None

            if txs:
                acct = mask_account(entities.get("account_number"))
                suffix = f" for account {acct}" if acct else ""
                return f"I've retrieved {len(txs)} recent transactions{suffix}."
            return "I couldn't find any recent transactions for your account."

        if intent == "transfer" and isinstance(tool_result, dict):
            status = (tool_result.get("status") or "").lower()
            amount = tool_result.get("amount")
            cur = tool_result.get("currency") or "USD"
            amt_str = format_amount(amount, cur)
            recipient = (
                tool_result.get("to")
                or entities.get("recipient_name")
                or entities.get("to_account_number")
                or "the recipient"
            )
            ref = (
                tool_result.get("txn_id")
                or tool_result.get("transaction_reference")
                or tool_result.get("reference")
            )

            if status == "success" and amt_str:
                if ref:
                    return f"Your transfer of {amt_str} to {recipient} was completed successfully. Reference: {ref}."
                return f"Your transfer of {amt_str} to {recipient} was completed successfully."
            if status:
                return f"The transfer status is '{status}'. Please check the details or try again."
            return "I couldn't confirm whether the transfer was completed. Please verify your transactions."

    except Exception:
        logger.exception("deterministic_fallback: error while building fallback response")

    return "I'm having trouble generating a safe response right now. Please try again or rephrase your request."
