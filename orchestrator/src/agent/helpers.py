"""
Shared utilities for VoxBank agent and orchestrator.

This file centralizes formatting, history rendering, and prompt-assembly
helpers so that the main classes stay focused on core logic.
"""

from typing import Any, Awaitable, Callable, Dict, List, Optional
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


def render_history_for_prompt(history: List[Dict[str, Any]], max_messages: int = 40) -> str:
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


def build_user_context_block(session_profile: Optional[Dict[str, Any]]) -> str:
    """
    Build a compact USER CONTEXT block for the decision prompt from a
    session_profile structure.
    """
    if not session_profile:
        return "- (none)"

    lines: List[str] = []
    lines.append(f"- user_id: {session_profile.get('user_id')}")
    lines.append(f"- username: {session_profile.get('username')}")

    primary_acct = session_profile.get("primary_account")
    lines.append(f"- primary_account: {primary_acct}")

    profile = session_profile.get("user_profile") or {}
    full_name = profile.get("full_name")
    if full_name:
        lines.append(f"- full_name: {full_name}")
    email = profile.get("email")
    if email:
        lines.append(f"- email: {email}")
    phone = profile.get("phone_number")
    if phone:
        lines.append(f"- phone_number: {phone}")
    preferred_lang = profile.get("preferred_language")
    if preferred_lang:
        lines.append(f"- preferred_language: {preferred_lang}")
    kyc_status = profile.get("kyc_status")
    if kyc_status:
        lines.append(f"- kyc_status: {kyc_status}")
    status = profile.get("status")
    if status:
        lines.append(f"- customer_status: {status}")
    address = profile.get("address")
    if address:
        lines.append(f"- address: {address}")
    dob = profile.get("date_of_birth")
    if dob:
        lines.append(f"- date_of_birth: {dob}")
    last_active = profile.get("last_active")
    if last_active:
        lines.append(f"- last_active: {last_active}")
    has_audio_embedding = profile.get("has_audio_embedding")
    if has_audio_embedding is not None:
        lines.append(f"- has_voice_print: {bool(has_audio_embedding)}")

    accounts = session_profile.get("accounts") or []
    acct_types: List[str] = []
    for acc in accounts:
        t = (acc.get("account_type") or "").strip().lower()
        if t and t not in acct_types:
            acct_types.append(t)
    if acct_types:
        lines.append(f"- other_accounts: {', '.join(acct_types)}")

    # Include a brief accounts summary (up to 3 accounts) for more context
    if accounts:
        summaries: List[str] = []
        for acc in accounts[:3]:
            acct_num = acc.get("account_number")
            acct_type = (acc.get("account_type") or "").strip().lower() or "account"
            curr = acc.get("currency") or ""
            label = f"{acct_type} ({acct_num})"
            if curr:
                label = f"{label} {curr}"
            summaries.append(label)
        lines.append(f"- accounts_summary: {', '.join(summaries)}")

    # Beneficiary summary if present
    beneficiaries = session_profile.get("beneficiaries") or []
    if beneficiaries:
        names: List[str] = []
        for b in beneficiaries[:3]:
            nick = b.get("nickname") or b.get("beneficiary_name") or b.get("account_number")
            acct = b.get("account_number")
            if acct:
                names.append(f"{nick} ({acct})")
            else:
                names.append(str(nick))
        lines.append(f"- saved_beneficiaries: {', '.join(names)}")

    return "\n".join(lines) if lines else "- (none)"


def build_tools_block(tool_spec: Dict[str, Any]) -> str:
    """
    Build a TOOLS block description from MCP tool metadata.
    """
    lines: List[str] = []
    for name, meta in (tool_spec or {}).items():
        params_meta = meta.get("params", {}) or {}
        params = ", ".join(
            f"{p} (required)" if (p_meta or {}).get("required") else f"{p} (optional)"
            for p, p_meta in params_meta.items()
        )
        desc = meta.get("description", "")
        lines.append(f"- {name}: {desc} Params: {params}")
    return "\n".join(lines)


# ------------------------------------------------------------------
# Translation helper
# ------------------------------------------------------------------

TranslatorCallable = Callable[..., Awaitable[str]]


async def translate_text(
    text: str,
    source_lang: str,
    target_lang: str,
    call_llm: TranslatorCallable,
) -> str:
    """
    Translate text between languages using the configured LLM client, keeping
    numeric identifiers intact. Returns the original text if translation fails.
    """
    if not text:
        return ""

    if source_lang.strip().lower() == target_lang.strip().lower():
        return text

    prompt = (
        "SYSTEM: You are a precise translator for a banking assistant.\n"
        "Instructions:\n"
        "- Translate from {source} to {target}.\n"
        "- Never alter account numbers, card numbers, phone numbers, OTPs, or dates.\n"
        "- Preserve numeric values exactly as provided.\n"
        "- Respond with translation only, without commentary.\n\n"
        "Input ({source}):\n"
        "{text}\n\n"
        "Output ({target}):"
    ).format(source=source_lang, target=target_lang, text=text)

    max_tokens = min(2048, max(60, len(text) + 50))

    try:
        translated = await call_llm(prompt, max_tokens=max_tokens, temperature=0.0)
        if isinstance(translated, str):
            return translated.strip() or text
        return str(translated).strip() or text
    except Exception:
        logger.exception(
            "translate_text: failed (%s -> %s)",
            source_lang,
            target_lang,
        )
        return text
