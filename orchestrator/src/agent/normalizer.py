"""
agent/normalizer.py

Small "normalizer LLM" used to clean up raw STT text and extract numeric
signals before the main VoxBank decision agent runs.

This module is intentionally lightweight and stateless. It relies on the
same LLM client used by VoxBankAgent (e.g., GeminiLLMClient) and returns
structured JSON describing the cleaned text and any numeric hints.
"""

from typing import Any, Dict, Optional
import logging

from guards.json_clean import validate_ai_json


logger = logging.getLogger("llm_agent.normalizer")


async def normalize_input(
    llm_client: Any,
    raw_text: str,
    last_assistant_msg: Optional[str] = None,
    language: str = "en",
) -> Dict[str, Any]:
    """
    Normalize raw STT text and extract numeric hints using a small LLM prompt.

    Returns a dict:
      {
        "cleaned_text": str,
        "numbers": [float|int],
        "primary_number": float|int|None,
        "currency_hint": str|None,
        "message_type": "number_only" | "account_like" | "amount_like" | "free_text",
      }

    On any error, returns a conservative default using raw_text.
    """
    fallback = {
        "cleaned_text": raw_text,
        "numbers": [],
        "primary_number": None,
        "currency_hint": None,
        "message_type": "free_text",
    }

    if not raw_text or not raw_text.strip():
        return fallback

    if llm_client is None or not hasattr(llm_client, "generate"):
        logger.warning("Normalizer: llm_client is not configured; skipping normalization")
        return fallback

    system_instructions = (
        "You are a small normalization helper for VoxBank's voice banking assistant.\n"
        "You receive the last assistant message and the user's raw speech-to-text output.\n"
        "Your job is to clean and normalize the user message and extract numeric information.\n\n"
        "Requirements:\n"
        "1) cleaned_text:\n"
        "   - Remove filler such as 'uh', 'please', 'can you', 'could you', 'just', 'like', etc.\n"
        "   - Keep important content and intent.\n"
        "   - Fix spacing inside account-like or number-like strings so '001 001' -> '001001'.\n"
        "   - Do NOT change the meaning.\n"
        "2) numbers:\n"
        "   - Extract all numeric values you can infer.\n"
        "   - Convert spoken numbers like 'hundred', 'one hundred and fifty',\n"
        "     'one thousand two hundred and fifty' into 100, 150, 1250, etc.\n"
        "   - For phrases like 'hundred dollars', treat it as a numeric value 100,\n"
        "     and set currency_hint accordingly if obvious (e.g., 'rupees', 'dollars').\n"
        "   - If multiple values are present and ambiguous, include them all in the 'numbers' list.\n"
        "3) primary_number:\n"
        "   - The most likely number the user intended given the context.\n"
        "   - If you are not sure, set it to null.\n"
        "4) currency_hint:\n"
        "   - 'USD', 'INR', etc. if the user clearly spoke a currency, otherwise null.\n"
        "5) message_type:\n"
        "   - 'number_only': short numeric answer, likely a direct answer to last question.\n"
        "   - 'account_like': looks like an account number or similar identifier (digit-heavy string).\n"
        "   - 'amount_like': looks like a monetary amount (with or without currency words).\n"
        "   - 'free_text': everything else.\n\n"
        "You MUST respond with a single JSON object and nothing else.\n"
        "JSON FORMAT:\n"
        "{\n"
        '  \"cleaned_text\": \"<string>\",\n'
        "  \"numbers\": [<float or int>],\n"
        "  \"primary_number\": <float or int or null>,\n"
        "  \"currency_hint\": \"<string or null>\",\n"
        "  \"message_type\": \"number_only\" | \"account_like\" | \"amount_like\" | \"free_text\"\n"
        "}\n"
    )

    last_assistant_msg = last_assistant_msg or ""
    language = language or "en"

    prompt = (
        f"{system_instructions}\n"
        "INPUT:\n"
        f"language: {language}\n"
        f"last_assistant_msg: {last_assistant_msg!r}\n"
        f"raw_text: {raw_text!r}\n"
        "Return only the JSON object.\n"
    )

    try:
        logger.info("Normalizer: calling LLM for input cleaning (len=%d chars)", len(raw_text))
        raw = await llm_client.generate(prompt, max_tokens=192)
    except Exception as e:  # pragma: no cover - external LLM failure
        logger.exception("Normalizer: LLM call failed: %s", e)
        return fallback

    if not raw:
        logger.warning("Normalizer: empty response from LLM; using fallback")
        return fallback

    try:
        validation = validate_ai_json(raw)
        if getattr(validation, "json_valid", False) and isinstance(validation.data, dict):
            data = validation.data  # type: ignore[assignment]
        else:
            logger.warning(
                "Normalizer: validate_ai_json reported invalid JSON; truncated=%s errors=%s",
                getattr(validation, "likely_truncated", False),
                getattr(validation, "errors", []),
            )
            return fallback
    except Exception as e:  # pragma: no cover - JSON validation failure
        logger.exception("Normalizer: validate_ai_json failed: %s", e)
        return fallback

    try:
        cleaned_text = data.get("cleaned_text") or raw_text
        numbers = data.get("numbers") or []
        if not isinstance(numbers, list):
            numbers = []

        primary = data.get("primary_number", None)
        currency_hint = data.get("currency_hint")
        msg_type = data.get("message_type") or "free_text"

        if not isinstance(cleaned_text, str):
            cleaned_text = raw_text
        if currency_hint is not None and not isinstance(currency_hint, str):
            currency_hint = None
        if msg_type not in ("number_only", "account_like", "amount_like", "free_text"):
            msg_type = "free_text"

        return {
            "cleaned_text": cleaned_text,
            "numbers": numbers,
            "primary_number": primary,
            "currency_hint": currency_hint,
            "message_type": msg_type,
        }
    except Exception as e:  # pragma: no cover - defensive
        logger.exception("Normalizer: error while normalizing LLM JSON output: %s", e)
        return fallback

