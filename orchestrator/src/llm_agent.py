"""
llm_agent.py

LLM Agent Controller
Handles conversation flow, intent understanding, and tool orchestration.

- Async-friendly (FastAPI compatible)
- Dependency-injectable: intent_classifier, llm_client, mcp_client
- Includes simple defaults for quick local/cursor development
"""

from typing import Dict, Any, Optional, List, Tuple
import asyncio
import logging
import re
import os
import json
from decimal import Decimal

import httpx
from pydantic.types import T
from gemini_llm_client import GeminiLLMClient

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY") or os.environ.get("GENAI_API_KEY") or os.environ.get("GEMINI_TOKEN")
DEFAULT_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-pro")
VOX_BANK_BASE_URL = os.environ.get("VOX_BANK_BASE_URL", "http://localhost:9000")

# Set up file-based logging
try:
    from logging_config import setup_logging, get_logger
    setup_logging()
    logger = get_logger("llm_agent")
except ImportError:
    # Fallback if logging_config not available
    logger = logging.getLogger("llm_agent")
    logging.basicConfig(level=logging.INFO)

import math
from decimal import Decimal

from prompts.vox_assistant import (
    TOOL_SPEC as FALLBACK_TOOL_SPEC,
    PRELOGIN_PROMPT_TEMPLATE,
    POSTLOGIN_PROMPT_TEMPLATE,
    SYSTEM_PROMPT,
)

def _format_amount(self, value, currency="USD"):
    """Return a nicely formatted amount string or None if invalid."""
    if value is None:
        return None
    # Accept Decimal, int, float, numeric strings
    try:
        if isinstance(value, str):
            # remove commas and currency symbols
            v = re.sub(r"[^\d.\-]", "", value)
            num = Decimal(v)
        else:
            num = Decimal(str(value))
    except Exception:
        return None
    # If it's a whole number show no decimals, else show 2 decimal places
    if num == num.to_integral():
        return f"{num:.0f} {currency}"
    else:
        return f"{num.quantize(Decimal('0.01'))} {currency}"

def _mask_account(self, acct: Optional[str]) -> Optional[str]:
    """Return masked account like ACC***9001 or 'primary' unchanged."""
    if not acct:
        return None
    acct = str(acct)
    if acct.lower() in ("primary", "default"):
        return acct
    # keep last 4 digits, preserve prefix if exists
    last4 = re.sub(r"\D", "", acct)[-4:]
    prefix = acct.split(last4)[0] if last4 and last4 in acct else acct[:-4]
    if prefix:
        return f"{prefix}****{last4}"
    return f"****{last4}"

class KeywordIntentClassifier:
    """
    Simple fallback intent classifier using keyword matching and regex.
    Returns: {"intent": str, "entities": dict, "confidence": float}
    """

    INTENT_KEYWORDS = {
        "balance": ["balance", "how much", "account balance", "what's my balance", "abalance"],
        "transfer": ["transfer", "send", "pay", "send money", "pay to"],
        "transactions": ["transactions", "statement", "history", "last transactions", "show me my"],
        "loan_inquiry": ["loan", "interest rate", "emi", "apply loan"],
        "reminder": ["remind", "reminder", "set reminder"],
        "greeting": ["hello", "hi", "hey", "good morning", "good evening"],
        "goodbye": ["bye", "goodbye", "see you"]
    }

    AMOUNT_RE = re.compile(r"(?:₹|\bINR\s?)?([0-9]+(?:[.,][0-9]{1,2})?)")
    PHONE_RE = re.compile(r"\b\d{10}\b")
    ACCOUNT_RE = re.compile(r"(?:account|a/c|acct)\s*(?:no\.?|number)?\s*:? *([0-9\-xX*]{4,})")

    async def classify(self, text: str) -> Dict[str, Any]:
        tx = text.lower()
        # detect intent by keywords
        best_intent = "unknown"
        for intent, keywords in self.INTENT_KEYWORDS.items():
            for kw in keywords:
                if kw in tx:
                    best_intent = intent
                    break
            if best_intent != "unknown":
                break

        # simple entities
        entities: Dict[str, Any] = {}
        # amount
        m = self.AMOUNT_RE.search(text.replace(",", ""))
        if m:
            try:
                entities["amount"] = float(m.group(1))
            except Exception:
                entities["amount_raw"] = m.group(1)
        # phone
        m2 = self.PHONE_RE.search(text)
        if m2:
            entities["phone"] = m2.group(0)
        # account
        m3 = self.ACCOUNT_RE.search(text)
        if m3:
            entities["account_masked"] = m3.group(1)

        # recipient name heuristic: "to <name>"
        to_match = re.search(r"\bto\s+([A-Z][a-zA-Z]+\s?[A-Za-z]*)", text)
        if to_match:
            entities["recipient_name"] = to_match.group(1)

        confidence = 0.6 if best_intent == "unknown" else 0.9
        return {"intent": best_intent, "entities": entities, "confidence": confidence}

class LLMAgent:
    """
    Main LLM agent that orchestrates conversations and tool calls.
    """

    # Map intents to (mcp_tool_name, input_builder)
    INTENT_TOOL_MAP = {
        # balance tool expects: {"account_number": "..."}
        "balance": ("balance", lambda ent: {"account_number": ent.get("account_masked") or ent.get("account_number") or "primary"}),
        # transfer expects: {"from_account_number","to_account_number","amount","currency","initiated_by_user_id","reference"}
        "transfer": ("transfer", lambda ent: {
            "from_account_number": ent.get("from_account_number") or ent.get("from_account") or "primary",
            "to_account_number": ent.get("to_account_number") or ent.get("recipient_account") or ent.get("phone") or ent.get("recipient_name"),
            "amount": float(ent.get("amount")) if ent.get("amount") is not None else None,
            "currency": ent.get("currency") or "USD",
            "initiated_by_user_id": ent.get("initiated_by_user_id"),
            "reference": ent.get("reference")
        }),
        # transactions expects: {"account_number","limit"}
        "transactions": ("transactions", lambda ent: {"account_number": ent.get("account_masked") or ent.get("account_number") or "primary", "limit": ent.get("limit", 5)}),
        # placeholders for other intents
        "loan_inquiry": ("loan_inquiry", lambda ent: {"loan_type": ent.get("loan_type")}),
        "reminder": ("set_reminder", lambda ent: {"when": ent.get("date") or ent.get("time"), "title": ent.get("title", "reminder")})
    }
    
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        intent_classifier: Optional[Any] = None,
        llm_client: Optional[Any] = None,
        mcp_client: Optional[Any] = None,
        high_risk_intents: Optional[List[str]] = None,
        tool_spec: Optional[Dict[str, Any]] = None,
    ):
        self.model_name = model_name
        # Keep intent_classifier as optional fallback, but LLM is primary
        self.intent_classifier = intent_classifier
        self.llm_client = llm_client or GeminiLLMClient(api_key=GEMINI_API_KEY, model=self.model_name)
        self.mcp_client = mcp_client  # expected to have `call_tool(tool_name, payload)` async method
        self.conversation_history: Dict[str, List[Dict[str, Any]]] = {}
        self.high_risk_intents = high_risk_intents or ["transfer", "payment", "loan_application"]
        # Simple in-memory auth state keyed by session_id
        # {
        #   "authenticated": bool,
        #   "user_id": Optional[str],
        #   "flow_stage": Optional[str],
        #   "temp": Dict[str, Any]
        # }
        self.auth_state: Dict[str, Dict[str, Any]] = {}
        # Separate prompt templates for pre-login and post-login
        self.prelogin_prompt_template = PRELOGIN_PROMPT_TEMPLATE
        self.postlogin_prompt_template = POSTLOGIN_PROMPT_TEMPLATE
        # Dynamic tool spec, primarily sourced from MCP list_tools; falls back to
        # static TOOL_SPEC from prompts.vox_assistant if not provided.
        self.tool_spec: Dict[str, Any] = {}
        self.tool_parameters: Dict[str, set] = {}
        self.set_tool_spec(tool_spec or FALLBACK_TOOL_SPEC)
        logger.debug("Tools block: %s", self.tools_block)


    def set_tool_spec(self, spec: Dict[str, Any]) -> None:
        """
        Update the tool specification used for prompts and validation.

        Expected shape (from MCP list_tools):
          {
            "tool_name": {
              "description": "...",
              "params": {
                 "param": {"type": "...", "required": bool, ...},
                 ...
              },
              ...
            },
            ...
          }
        """
        if not spec:
            logger.warning("LLMAgent.set_tool_spec called with empty spec; falling back to built-in spec")
            spec = FALLBACK_TOOL_SPEC

        self.tool_spec = spec
        # Precompute parameter sets for validation
        self.tool_parameters = {
            name: set((meta.get("params") or {}).keys())
            for name, meta in self.tool_spec.items()
        }

        # Render tools_block string used inside the LLM prompt
        lines: List[str] = []
        for name, meta in self.tool_spec.items():
            params_meta = meta.get("params", {}) or {}
            params = ", ".join(
                f"{p} (required)" if (p_meta or {}).get("required") else f"{p} (optional)"
                for p, p_meta in params_meta.items()
            )
            desc = meta.get("description", "")
            lines.append(f"- {name}: {desc} Params: {params}")
        self.tools_block = "\n".join(lines)

        logger.info("LLMAgent tool spec updated with %d tools", len(self.tool_spec))

    # helper method to render history for the prompt (keeps prompt size manageable)
    def _render_history_for_prompt(self, session_id: str, max_messages: int = 8) -> str:
        h = self.get_history(session_id) or []
        if not h:
            return ""
        recent = h[-max_messages:]
        lines = []
        for m in recent:
            role = m.get("role", "user")
            text = m.get("text", "")
            lines.append(f"{role}: {text}")
        return "\n".join(lines)
    
    # helper to format observation stored in history (short summary)
    def _format_observation_for_history(self, tool_name: str, observation: Any) -> str:
        try:
            if isinstance(observation, dict):
                # keep brief keys: status, message, balance, transactions count
                if "balance" in observation:
                    return f"{tool_name} -> balance: {observation.get('balance')} {observation.get('currency', '')}"
                if "transactions" in observation and isinstance(observation.get("transactions"), list):
                    return f"{tool_name} -> returned {len(observation.get('transactions'))} transactions"
                if "status" in observation and observation.get("status") != "success":
                    return f"{tool_name} -> status: {observation.get('status')} - {str(observation.get('message',''))}"
                # fallback to short json snippet
                short = json.dumps(observation, default=str)
                return f"{tool_name} -> {short[:200]}"
            if isinstance(observation, list):
                return f"{tool_name} -> list length {len(observation)}"
            return f"{tool_name} -> {str(observation)[:200]}"
        except Exception:
            return f"{tool_name} -> (unserializable observation)"

    # -----------------------
    # Auth / login helpers
    # -----------------------
    def _get_auth_state(self, session_id: str) -> Dict[str, Any]:
        """
        Return (and initialize if needed) the authentication state for a session.
        """
        if session_id not in self.auth_state:
            self.auth_state[session_id] = {
                "authenticated": False,
                "user_id": None,
                "flow_stage": None,
                "temp": {},
            }
        return self.auth_state[session_id]

    def is_authenticated(self, session_id: str) -> bool:
        state = self._get_auth_state(session_id)
        return bool(state.get("authenticated") and state.get("user_id"))

    def get_authenticated_user_id(self, session_id: str) -> Optional[str]:
        state = self._get_auth_state(session_id)
        return state.get("user_id") if state.get("authenticated") else None

    async def _find_mock_bank_user_by_username(self, username: str) -> Optional[Dict[str, Any]]:
        """
        Deterministic lookup of a user by username using mock-bank API.
        Uses GET /api/list_users and filters client-side.
        """
        base = VOX_BANK_BASE_URL.rstrip("/") if VOX_BANK_BASE_URL else None
        if not base:
            logger.warning("VOX_BANK_BASE_URL not configured; cannot validate username")
            return None

        username_norm = (username or "").strip().lower()

        # list endpoint returns a list of users; we filter by username here
        url = f"{base}/api/list_users"
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(url, params={"limit": 200, "offset": 0})
                resp.raise_for_status()
                users = resp.json()
        except Exception as e:
            logger.exception("Auth: failed to list users from mock-bank: %s", e)
            return None

        for u in users or []:
            if (u.get("username") or "").strip().lower() == username_norm:
                logger.info("Auth: found existing mock-bank user %s -> %s", username_norm, u.get("user_id"))
                return u
        logger.info("Auth: username %s not found in mock-bank", username_norm)
        return None

    async def _login_user_via_mcp_or_http(self, username: str, passphrase: str) -> Optional[Dict[str, Any]]:
        """
        Deterministic login helper: validate username + passphrase by calling
        mock-bank /api/login directly.

        Returns a dict with at least user_id/username on success, or None on failure.
        """
        # Normalise inputs to lower-case to be robust to STT case variation
        username_norm = (username or "").strip().lower()
        passphrase_norm = (passphrase or "").strip().lower()

        # Direct HTTP to mock-bank
        base = VOX_BANK_BASE_URL.rstrip("/") if VOX_BANK_BASE_URL else None
        if not base:
            logger.warning("VOX_BANK_BASE_URL not configured; cannot perform HTTP login")
            return None

        url = f"{base}/api/login"
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.post(url, json={"username": username_norm, "passphrase": passphrase_norm})
                if resp.status_code != 200:
                    logger.warning("AUTH: HTTP login failed for %s: %s", username_norm, resp.text)
                    return None
                data = resp.json()
                if data.get("user_id"):
                    logger.info("AUTH: HTTP login success for %s", username_norm)
                    return data
        except Exception as e:
            logger.exception("AUTH: HTTP login call failed for %s: %s", username_norm, e)
        return None

    async def _handle_auth_flow(self, transcript: str, session_id: str) -> Dict[str, Any]:
        """
        Rule-based login/registration dialogue handler.
        This is deterministic and does NOT call the LLM.
        """
        state = self._get_auth_state(session_id)
        text = (transcript or "").strip()
        lower = text.lower()
        stage = state.get("flow_stage") or "await_choice"

        logger.info("AUTH: handling auth flow stage=%s for session=%s", stage, session_id)

        # Stage 1: ask user to choose login or register
        if stage == "await_choice":
            if any(kw in lower for kw in ["login", "log in", "sign in"]):
                state["flow_stage"] = "await_login_username"
                msg = "Sure, let's get you logged in. Please tell me your username."
            elif any(kw in lower for kw in ["register", "sign up", "create account", "open account"]):
                state["flow_stage"] = "await_register_username"
                msg = "Great, let's create a new account. What username would you like to use?"
            else:
                msg = "You're not logged in yet. Please say 'login' to sign in or 'register' to create a new account."
            self._append_history(session_id, {"role": "assistant", "text": msg})
            return {"status": "clarify", "message": msg}

        # Stage 2: login - capture username and verify it exists
        if stage == "await_login_username":
            # Extract username from the utterance.
            # Prefer patterns like "my username is sowmy", otherwise fall back to last token
            # but avoid treating generic words like "username" or "login" as actual usernames.
            words = text.split()
            if not words:
                msg = "I didn't catch your username. Please say just your username, like 'sowmy'."
                self._append_history(session_id, {"role": "assistant", "text": msg})
                return {"status": "clarify", "message": msg}

            # Simple pattern: "... username is <value>"
            username = None
            for marker in ["username is", "user name is", "my name is", "i am"]:
                if marker in lower:
                    # take the text after the marker as the candidate
                    candidate = lower.split(marker, 1)[1].strip()
                    if candidate:
                        username = candidate.split()[0]
                        break

            if not username:
                username = words[-1]

            username = username.strip().lower()

            # Avoid obvious non-usernames that come from echoing the assistant prompt
            if username.lower() in {"username", "user", "login"}:
                msg = "Please say just your username, for example 'sowmy'."
                self._append_history(session_id, {"role": "assistant", "text": msg})
                return {"status": "clarify", "message": msg}

            state["temp"]["username"] = username
            logger.info("AUTH: login username candidate=%s", username)

            user = await self._find_mock_bank_user_by_username(username)
            if not user:
                msg = f"I couldn't find an account for username '{username}'. You can try a different username or say 'register' to create a new one."
                # Stay in the same stage so user can retry or switch to register
                self._append_history(session_id, {"role": "assistant", "text": msg})
                return {"status": "clarify", "message": msg}

            # Username exists; ask for passphrase (demo only – no real validation yet)
            state["flow_stage"] = "await_login_passphrase"
            state["temp"]["user_record"] = user
            msg = f"Thanks, {username}. Please provide your passphrase."
            self._append_history(session_id, {"role": "assistant", "text": msg})
            return {"status": "clarify", "message": msg}

        # Stage 3: login - check passphrase (validate via MCP/HTTP)
        if stage == "await_login_passphrase":
            if not text:
                msg = "I didn't catch your passphrase. Please say it again."
                self._append_history(session_id, {"role": "assistant", "text": msg})
                return {"status": "clarify", "message": msg}

            username = (state["temp"].get("username") or "user").strip().lower()
            login_res = await self._login_user_via_mcp_or_http(username, text)
            if not login_res:
                msg = "The passphrase you provided doesn't match our records. Please try again."
                self._append_history(session_id, {"role": "assistant", "text": msg})
                return {"status": "clarify", "message": msg}

            user_id = login_res.get("user_id") or username
            logger.info("AUTH: login success for username=%s user_id=%s", username, user_id)
            state["authenticated"] = True
            state["user_id"] = user_id
            state["flow_stage"] = None
            state["temp"] = {}

            msg = f"You're now logged in as {username}. How can I help you with your accounts?"
            self._append_history(session_id, {"role": "assistant", "text": msg})
            return {
                "status": "auth_ok",
                "message": msg,
                "authenticated_user_id": user_id,
            }

        # Stage 4: registration - capture desired username
        if stage == "await_register_username":
            username = text.split()[-1]
            state["temp"]["username"] = username
            logger.info("AUTH: registration username candidate=%s", username)

            user = await self._find_mock_bank_user_by_username(username)
            if user:
                msg = f"The username '{username}' is already taken. Please choose another username."
                self._append_history(session_id, {"role": "assistant", "text": msg})
                return {"status": "clarify", "message": msg}

            state["flow_stage"] = "await_register_passphrase"
            msg = f"Username '{username}' is available. Please choose a passphrase."
            self._append_history(session_id, {"role": "assistant", "text": msg})
            return {"status": "clarify", "message": msg}

        # Stage 5: registration - capture passphrase and mark session as logged in
        if stage == "await_register_passphrase":
            if not text:
                msg = "I didn't catch your passphrase. Please say it again."
                self._append_history(session_id, {"role": "assistant", "text": msg})
                return {"status": "clarify", "message": msg}

            username = state["temp"].get("username") or "user"
            # For now we simulate user creation; UI-based registration will hit mock-bank directly.
            user_id = username
            logger.info("AUTH: registration success for username=%s (session-only user_id=%s)", username, user_id)

            state["authenticated"] = True
            state["user_id"] = user_id
            state["flow_stage"] = None
            state["temp"] = {}

            msg = f"Welcome, {username}! Your VoxBank profile is ready and you're now logged in."
            self._append_history(session_id, {"role": "assistant", "text": msg})
            return {
                "status": "auth_ok",
                "message": msg,
                "authenticated_user_id": user_id,
            }

        # Fallback: reset auth flow if stage unknown
        logger.warning("AUTH: unknown auth flow stage '%s' for session=%s; resetting", stage, session_id)
        state["flow_stage"] = "await_choice"
        state["temp"] = {}
        msg = "You're not logged in yet. Would you like to login or register?"
        self._append_history(session_id, {"role": "assistant", "text": msg})
        return {"status": "clarify", "message": msg}
    
    def _is_raw_tool_output(self, text: str) -> bool:
        if not text or len(text) < 20:
            return False
        # heuristics
        if re.search(r"\btransactions?\b", text, flags=re.I):
            return True
        if re.search(r"\bACC[0-9A-Za-z]{3,}\b", text):
            return True
        if re.search(r"^\s*[-\u2022]\s+", text, flags=re.M):  # bullet lines
            return True
        if re.search(r"\[.*?\]|\{.*?\}", text):
            return True
        return False

    async def decision(
        self,
        transcript: str,
        context: str,
        session_profile: Optional[Dict[str, Any]] = None,
        observation: Optional[Any] = None,
        auth_state: Optional[bool] = None,
        max_tokens: int = 512,
    ) -> dict:
        """
        Ask the LLM to *decide* action: respond, call_tool, ask_user, or ask_confirmation.
        Returns parsed JSON dict with keys: action, intent, requires_tool, tool_name, tool_input,
        requires_confirmation, response
        """
        logger.info("=" * 80)
        logger.info("LLM DECISION - Starting")
        logger.info("Context: %s | Transcript: %s", context, transcript)

        # Determine authentication flag from explicit auth_state or session_profile
        if auth_state is None and session_profile is not None:
            is_auth = bool(session_profile.get("is_authenticated"))
        else:
            is_auth = bool(auth_state)

        if not is_auth:
            # PRE-LOGIN: no tools, no user-specific context
            logger.info("LLM DECISION - Using PRELOGIN prompt template (unauthenticated)")
            prompt = (
                self.prelogin_prompt_template
                .replace("{history}", context)
                .strip()
            )
        else:
            # POST-LOGIN: tools and user context available
            logger.info("LLM DECISION - Using POSTLOGIN prompt template (authenticated)")
            # Build a simple user context block from session_profile
            user_context_lines = []
            if session_profile:
                user_context_lines.append(f"- user_id: {session_profile.get('user_id')}")
                user_context_lines.append(f"- username: {session_profile.get('username')}")
                primary_acct = session_profile.get("primary_account")
                user_context_lines.append(f"- primary_account: {primary_acct}")
                # Derive account types summary
                acct_types = []
                for acc in session_profile.get("accounts") or []:
                    t = (acc.get("account_type") or "").strip().lower()
                    if t and t not in acct_types:
                        acct_types.append(t)
                if acct_types:
                    user_context_lines.append(f"- other_accounts: {', '.join(acct_types)}")
            user_context_block = "\n".join(user_context_lines) if user_context_lines else "- (none)"

            prompt = (
                self.postlogin_prompt_template
                .replace("{history}", context)
                .replace("{tools_block}", self.tools_block)
                .replace("{user_context_block}", user_context_block)
                .strip()
            )

        # Add the user request at the end
        prompt = f"{prompt}\n\nUser request: \"{transcript}\"\n"
        # If observation present, append observation JSON for LLM to reason about
        if observation is not None:
            try:
                obs_json = json.dumps(observation, default=str)
            except Exception:
                obs_json = str(observation)
            prompt += f"\nObservation (from tool): {obs_json}\n"

        prompt += "\nReturn JSON now."
        logger.info("Calling LLM with prompt (length: %d chars)", len(prompt))

        # call llm
        raw = await self.call_llm(prompt, max_tokens=max_tokens)
        logger.info("LLM raw response received (length: %d chars)", len(raw))
        logger.info("\n LLM raw resposne: %s\n", raw)
        logger.debug("LLM raw response: %s", raw[:500] + "..." if len(raw) > 500 else raw)
        # Extract JSON
        try:
            logger.debug("Extracting JSON from LLM response...")
            json_start = raw.find("{")
            json_end = raw.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                json_str = raw[json_start:json_end]
                logger.debug("Extracted JSON (start=%d, end=%d): %s", json_start, json_end, json_str[:200] + "..." if len(json_str) > 200 else json_str)
            else:
                json_str = raw
                logger.warning("Could not find JSON boundaries, using full response")
            parsed = json.loads(json_str)
            logger.info("✓ JSON parsed successfully")
            logger.debug("Parsed JSON: %s", parsed)
        except json.JSONDecodeError as e:
            logger.exception("Failed to parse LLM decision JSON: %s", e)
            logger.error("Raw LLM response: %s", raw)
            # fallback: treat as unknown / ask_user to clarify
            return {
                "action": "ask_user",
                "intent": "unknown",
                "requires_tool": False,
                "tool_name": None,
                "tool_input": {},
                "requires_confirmation": False,
                "response": "Sorry — can you rephrase that? I didn't understand."
            }
        except Exception as e:
            logger.exception("Unexpected error parsing LLM response: %s", e)
            logger.error("Raw LLM response: %s", raw)
            return {
                "action": "ask_user",
                "intent": "unknown",
                "requires_tool": False,
                "tool_name": None,
                "tool_input": {},
                "requires_confirmation": False,
                "response": "Sorry — can you rephrase that? I didn't understand."
            }

        # Normalize fields
        logger.debug("Normalizing parsed fields...")
        parsed.setdefault("action", "respond")
        parsed.setdefault("intent", "unknown")
        parsed.setdefault("requires_tool", bool(parsed.get("tool_name")))
        parsed.setdefault("tool_input", parsed.get("tool_input", {}))
        parsed.setdefault("requires_confirmation", bool(parsed.get("requires_confirmation", False)))
        parsed.setdefault("response", parsed.get("response", ""))
        
        logger.info("LLM Decision: action=%s, intent=%s, requires_tool=%s, tool_name=%s", 
                   parsed["action"], parsed["intent"], parsed["requires_tool"], parsed.get("tool_name"))
        logger.debug("Full parsed decision: %s", parsed)

        # Enforce pre-login restrictions: never call tools when not authenticated
        if not is_auth:
            if parsed.get("requires_tool"):
                logger.warning("Pre-login mode: overriding requires_tool=true to false")
            if parsed.get("tool_name"):
                logger.warning("Pre-login mode: clearing tool_name '%s'", parsed.get("tool_name"))
            # If the model requested a tool call, convert it into an ask_user prompting for login
            if parsed["action"] == "call_tool":
                parsed["action"] = "ask_user"
                parsed["response"] = (
                    parsed.get("response")
                    or "You need to log in or register before I can access your accounts. Please tell me your username to begin."
                )
            parsed["requires_tool"] = False
            parsed["tool_name"] = None
            parsed["tool_input"] = {}

        # Validate tool_input if action == call_tool (post-login only)
        if parsed["action"] == "call_tool" and parsed.get("tool_name"):
            tool = parsed["tool_name"]
            logger.info("Validating tool call: %s", tool)
            logger.debug("Tool input before validation: %s", parsed["tool_input"])
            
            if tool not in self.tool_parameters:
                logger.warning("Invalid tool requested: %s (not in TOOL_PARAMETERS)", tool)
                # invalid tool => ask user
                return {
                    "action": "ask_user",
                    "intent": parsed["intent"],
                    "requires_tool": False,
                    "tool_name": None,
                    "tool_input": {},
                    "requires_confirmation": False,
                    "response": f"The requested operation '{tool}' is not available. What would you like to do instead?"
                }
            
            # filter tool_input keys and ensure required params present
            expected = self.tool_parameters.get(tool, set())
            logger.debug("Expected parameters for %s: %s", tool, expected)
            filtered = {k: v for k, v in parsed["tool_input"].items() if k in expected}
            logger.debug("Filtered tool input: %s", filtered)
            
            # Use dynamic spec first; fall back to built-in spec if needed
            base_spec = (self.tool_spec.get(tool) or FALLBACK_TOOL_SPEC.get(tool) or {})
            param_spec = base_spec.get("params", {}) or {}
            missing = [
                p for p in expected
                if p not in filtered and (param_spec.get(p) or {}).get("required")
            ]
            if missing:
                logger.warning("Missing required parameters for %s: %s", tool, missing)
                # ask for missing params
                return {
                    "action": "ask_user",
                    "intent": parsed["intent"],
                    "requires_tool": True,
                    "tool_name": tool,
                    "tool_input": filtered,
                    "requires_confirmation": parsed["requires_confirmation"],
                    "response": f"I need the following information to proceed: {', '.join(missing)}. Please provide it."
                }
            parsed["tool_input"] = filtered
            logger.info("✓ Tool validation passed: %s with input %s", tool, filtered)

        logger.info("LLM DECISION - Complete")
        logger.info("=" * 80)
        return parsed

    async def orchestrate(
        self,
        transcript: str,
        session_id: str,
        user_confirmation: Optional[bool] = None,
        reply_style: Optional[str] = "concise",
        parse_only: bool = False,
        max_iterations: int = 4,
        session_profile: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        ReAct style orchestration loop:
        - call decision()
        - if decision -> call_tool: execute tool, append observation, call decision() again
        - otherwise handle respond / ask_user / ask_confirmation
        """
        logger.info("=" * 80)
        logger.info("ORCHESTRATE (ReAct) - Starting")
        logger.info("Session: %s | Transcript: %s", session_id, transcript)
        logger.info("User confirmation: %s | Parse only: %s | Reply style: %s", user_confirmation, parse_only, reply_style)

        if session_profile:
            logger.info(
                "ORCHESTRATE: session_profile user_id=%s username=%s primary_account=%s accounts=%d",
                session_profile.get("user_id"),
                session_profile.get("username"),
                session_profile.get("primary_account"),
                len(session_profile.get("accounts") or []),
            )

        auth_state = self._get_auth_state(session_id)
        # If we are currently in a login / registration flow, handle it
        if auth_state.get("flow_stage"):
            logger.info("ORCHESTRATE: routing input to auth flow (stage=%s)", auth_state.get("flow_stage"))
            result = await self._handle_auth_flow(transcript, session_id)
            logger.info("ORCHESTRATE: auth flow handled with status=%s", result.get("status"))
            logger.info("=" * 80)
            return result

        # If the session is unauthenticated and the user explicitly mentions login/registration
        # in this utterance, route directly into the deterministic auth flow instead of asking
        # the LLM to interpret it. This prevents the assistant from looping on generic
        # "please log in" messages without actually starting the login dialogue.
        if not self.is_authenticated(session_id):
            lower_tx = (transcript or "").lower()
            login_keywords = [
                "login",
                "log in",
                "sign in",
            ]
            register_keywords = [
                "register",
                "sign up",
                "create account",
                "open account",
            ]
            if any(kw in lower_tx for kw in login_keywords + register_keywords):
                logger.info(
                    "ORCHESTRATE: detected explicit login/register intent in transcript; "
                    "routing to auth flow with stage=%s",
                    auth_state.get("flow_stage") or "await_choice",
                )
                result = await self._handle_auth_flow(transcript, session_id)
                logger.info("ORCHESTRATE: auth flow handled with status=%s", result.get("status"))
                logger.info("=" * 80)
                return result

        iterations = 0
        parsed = None
        tool_result = None

        # observation is the last tool output passed back into decision
        observation = None

        while iterations < max_iterations:
            iterations += 1
            logger.info("ReAct loop iteration %d/%d", iterations, max_iterations)

            # Call decision with current transcript + context, auth state, and last observation (if any)
            context_str = self._render_history_for_prompt(session_id)
            auth_flag = self.is_authenticated(session_id)
            parsed = await self.decision(
                transcript,
                context_str,
                session_profile=session_profile,
                observation=observation,
                auth_state=auth_flag,
            )
            self._append_history(session_id, {"role": "user", "text": transcript})
            logger.debug("Added user message to history")
            action = parsed.get("action")
            intent = parsed.get("intent")
            requires_tool = parsed.get("requires_tool", False)
            tool_name = parsed.get("tool_name")
            tool_input = parsed.get("tool_input", {})
            requires_confirmation = parsed.get("requires_confirmation", False)
            assistant_response = parsed.get("response", "")

            logger.info("Decision (iter %d): action=%s intent=%s tool=%s requires_confirmation=%s", iterations, action, intent, tool_name, requires_confirmation)
            logger.debug("Parsed: %s", parsed)

            # Hard guard: if the session is not authenticated and the intent is
            # clearly account-specific, never allow a direct LLM response.
            # This prevents hallucinated balances/transactions when session_authenticated is false.
            unauthenticated = not self.is_authenticated(session_id)
            account_intents = {"balance", "transactions", "transfer"}
            account_tools = {"balance", "transactions", "transfer"}
            if (
                unauthenticated
                and action == "respond"
                and (
                    (intent in account_intents)
                    or (requires_tool and tool_name in account_tools)
                )
            ):
                state = self._get_auth_state(session_id)
                if not state.get("flow_stage"):
                    state["flow_stage"] = "await_choice"
                login_msg = "You're not logged in yet. Please login or register before I can access your accounts."
                self._append_history(session_id, {"role": "assistant", "text": login_msg})
                logger.info(
                    "ORCHESTRATE - Blocked unauthenticated account-level respond action for session %s (intent=%s, tool=%s)",
                    session_id,
                    intent,
                    tool_name,
                )
                logger.info("=" * 80)
                return {"status": "clarify", "message": login_msg}

            # If decision requests to ask user / confirm -> return immediately (assistant message already prepared)
            if action in ("ask_user", "ask_confirmation"):
                # append assistant text to history for continuity
                self._append_history(session_id, {"role": "assistant", "text": assistant_response})
                status = "needs_confirmation" if action == "ask_confirmation" else "clarify"
                logger.info("ORCHESTRATE - Completed (ask_user/ask_confirmation) at iter %d", iterations)
                logger.info("=" * 80)
                return {"status": status, "message": assistant_response}

            # If decision says respond -> append to history and return final response
            if action == "respond" and assistant_response:
                logger.info("Using LLM-provided response (action=respond) — checking if polishing needed")
                # If the response looks like raw tool output or is long/multi-line, polish it.
                should_polish = self._is_raw_tool_output(assistant_response)
                # Also polish if we have a tool_result/observation present (from previous loop)
                if not should_polish and (tool_result is not None or parsed.get("tool_input")):
                    # if tool_result present and assistant_response mentions data, polish
                    should_polish = True

                should_polish = False
                if should_polish:
                    logger.info("Polishing assistant response via generate_response_llm_first()")
                    # pass intent, parsed tool_input/entities and tool_result (if any) so LLM can craft final text
                    try:
                        polished = await self.generate_response(
                            intent or "unknown",
                            parsed.get("tool_input", {}),
                            tool_result or parsed.get("tool_output") or parsed.get("observation"),
                            reply_style=reply_style
                        )
                        # fallback to assistant_response if polishing fails
                        final_reply = polished or assistant_response
                    except Exception as e:
                        logger.exception("Polishing failed: %s. Falling back to raw assistant_response", e)
                        final_reply = assistant_response
                else:
                    final_reply = assistant_response

                # Append final reply and return
                self._append_history(session_id, {"role": "assistant", "text": final_reply})
                logger.info("ORCHESTRATE - Complete (respond)")
                return {"status": "ok", "response": final_reply}

            # If decision says call_tool -> enforce login for account tools, then execute tool
            if action == "call_tool" and requires_tool and tool_name:
                logger.info("Decision requested tool call: %s (iter %d)", tool_name, iterations)

                # Gate all tools that are not explicitly auth/utility tools behind login/registration.
                auth_tools = {"register_user", "login_user", "set_user_audio_embedding", "get_user_profile", "list_tools"}
                if not parse_only and not self.is_authenticated(session_id) and tool_name not in auth_tools:
                    logger.info(
                        "AUTH: session %s not authenticated; prompting for login/registration before tool %s",
                        session_id,
                        tool_name,
                    )
                    state = self._get_auth_state(session_id)
                    if not state.get("flow_stage"):
                        state["flow_stage"] = "await_choice"
                    msg = "You're not logged in yet. Would you like to login or register?"
                    self._append_history(session_id, {"role": "assistant", "text": msg})
                    logger.info("ORCHESTRATE - Completed (auth_required before tool exec) at iter %d", iterations)
                    logger.info("=" * 80)
                    return {"status": "clarify", "message": msg, "parsed": parsed}

                # Resolve abstract account labels (primary/savings/current) to concrete account_numbers
                # before executing tools. This biases tools toward the primary account unless the user
                # explicitly refers to another account type.
                if session_profile:
                    # Balance / transactions: map account_number
                    if tool_name in ("balance", "transactions"):
                        acct_label = tool_input.get("account_number")
                        acct_number = self._resolve_account_from_profile(session_profile, acct_label)
                        if not acct_number:
                            clarify = (
                                "I couldn't determine which account to use. "
                                "Please specify which account (for example, your savings or current account)."
                            )
                            self._append_history(session_id, {"role": "assistant", "text": clarify})
                            logger.info(
                                "ORCHESTRATE - Unable to resolve account label '%s' for tool %s; asking user to clarify",
                                acct_label,
                                tool_name,
                            )
                            logger.info("=" * 80)
                            return {"status": "clarify", "message": clarify, "parsed": parsed}
                        tool_input["account_number"] = acct_number
                        parsed["tool_input"]["account_number"] = acct_number

                    # Transfer: map from_account only; to_account may be external
                    if tool_name == "transfer":
                        from_label = tool_input.get("from_account_number")
                        from_number = self._resolve_account_from_profile(session_profile, from_label)
                        if not from_number:
                            clarify = (
                                "Which account would you like to send money from? "
                                "You can say your savings account or current account."
                            )
                            self._append_history(session_id, {"role": "assistant", "text": clarify})
                            logger.info(
                                "ORCHESTRATE - Unable to resolve from_account label '%s' for transfer; asking user to clarify",
                                from_label,
                            )
                            logger.info("=" * 80)
                            return {"status": "clarify", "message": clarify, "parsed": parsed}
                        tool_input["from_account_number"] = from_number
                        parsed["tool_input"]["from_account_number"] = from_number

                # Enforce confirmation for high-risk actions
                if requires_confirmation and not user_confirmation:
                    confirm_msg = f"I will perform: {intent}. {assistant_response or 'Do you want to proceed?'}"
                    # Do not execute; ask user for confirmation
                    self._append_history(session_id, {"role": "assistant", "text": confirm_msg})
                    logger.info("ORCHESTRATE - Completed (needs_confirmation before tool exec) at iter %d", iterations)
                    logger.info("=" * 80)
                    return {"status": "needs_confirmation", "message": confirm_msg}

                # ensure mcp_client present
                if not self.mcp_client:
                    logger.error("MCP client not configured - cannot execute tool")
                    observation = {"status": "not_configured", "message": "MCP client not set up."}
                else:
                    try:
                        logger.info("EXECUTING MCP TOOL %s with input %s", tool_name, tool_input)
                        tool_result = await self.mcp_client.call_tool(tool_name, tool_input)
                        logger.info("Tool result: %s", tool_result)
                        # Normalize observation to dict or list
                        if isinstance(tool_result, (dict, list)):
                            observation = tool_result
                        else:
                            # non-serializable -> stringify minimally
                            observation = {"status": "ok", "result": str(tool_result)}
                    except Exception as e:
                        logger.exception("Exception while executing tool %s: %s", tool_name, e)
                        observation = {"status": "error", "message": str(e)}

                # append a short tool observation into history as a tool message and continue loop
                obs_summary = self._format_observation_for_history(tool_name, observation)
                self._append_history(session_id, {"role": "tool", "text": obs_summary, "detail": observation})
                logger.info("Appended tool observation to history and continuing loop (iter %d)", iterations)
                # continue to next iteration so decision() can see observation and decide next action
                continue

            # If none of the above matched, break
            logger.warning("Decision returned unexpected action or no-op; breaking loop (iter %d): %s", iterations, parsed)
            break

        # Reached end of loop: either no decision or max iterations hit
        logger.info("Exited ReAct loop after %d iterations", iterations)

        # If we have a tool_result (observation), let generate_response combine it into final reply
        response_text = None
        try:
            response_text = await self.generate_response(intent or "unknown", parsed.get("tool_input", {}), observation)
        except Exception as e:
            logger.exception("Error generating final response: %s", e)
            response_text = parsed.get("response") or "Sorry, I'm having trouble right now."

        # Append assistant reply to history and return
        self._append_history(session_id, {"role": "assistant", "text": response_text})
        logger.info("ORCHESTRATE - Complete (final response)")
        logger.info("=" * 80)
        return {"status": "ok", "response": response_text}


    async def generate_response(
        self,
        intent: str,
        entities: Dict[str, Any],
        tool_result: Optional[Any] = None,
        reply_style: str = "concise",
        max_tokens: int = 150,
        temperature: float = 0.0
    ) -> str:
        """
        LLM-first response generator.
        The LLM is given a compact JSON context and expected to return 1-2 polite sentences.
        This wrapper sanitizes the LLM output and falls back to deterministic summaries when needed.
        """

        # Build context JSON for the LLM
        context = {
            "intent": intent,
            "entities": entities or {},
            "tool_result": tool_result if tool_result is not None else {}
        }


        # Safely insert the context JSON (avoid .format interpreting braces)
        context_json = json.dumps(context, default=str)
        prompt = SYSTEM_PROMPT.replace("{context_json}", context_json)

        # Call LLM - prefer passing temperature if call_llm supports it.
        try:
            # try to pass temperature if call_llm accepts it
            raw = await self.call_llm(prompt, max_tokens=max_tokens, temperature=temperature)
        except TypeError:
            # fallback: older call_llm without temperature parameter
            raw = await self.call_llm(prompt, max_tokens=max_tokens)

        raw = (raw or "").strip()
        logger.debug("LLM reply length=%d", len(raw))

        # Basic sanitation: if empty or appears to contain placeholders or a meta question -> fallback
        if not raw:
            logger.warning("LLM returned empty reply; falling back to deterministic summary.")
            return self._deterministic_fallback(intent, entities, tool_result)

        # If LLM asked an unrelated meta-question, fallback
        if re.search(r"\bWhen was this\b|\bWhen did\b|\bHow long ago\b|\bwhat time\b", raw, flags=re.I):
            logger.warning("LLM asked meta-question unexpectedly; using deterministic summary instead.")
            return self._deterministic_fallback(intent, entities, tool_result)

        # If LLM returned bracket placeholders or JSON fragments, fallback
        if re.search(r"\[.*?\]|\{.*?\}", raw):
            logger.warning("LLM returned placeholders or JSON-like text; using deterministic fallback.")
            return self._deterministic_fallback(intent, entities, tool_result)

        # Sanity: ensure the reply uses numeric digits where expected for amounts (for balance/transactions/transfer)
        if intent in ("balance", "transactions", "transfer"):
            # If the LLM response lacks any digit at all, assume it hallucinated and fallback
            if not re.search(r"\d", raw):
                logger.warning("LLM reply lacks digits for numeric intent; falling back.")
                return self._deterministic_fallback(intent, entities, tool_result)

        # Final clean-up: strip repeated whitespace, ensure single trailing punctuation
        reply = " ".join(raw.split())
        # ensure it ends with a period or question mark or exclamation
        if not re.search(r"[.!?]\s*$", reply):
            reply = reply.rstrip() + "."

        # Mask account numbers in the reply if the model echoed them unmasked
        try:
            # find token-like ACC tokens and mask last 4 digits
            reply = re.sub(r"\b(ACC[0-9A-Za-z\-]{4,})\b", lambda m: self._mask_account(m.group(1)) or m.group(1), reply)
        except Exception:
            # ignore any masking errors
            pass

        return reply


    # Helper deterministic fallback method (add to the class)
    def _deterministic_fallback(self, intent: str, entities: Dict[str, Any], tool_result: Optional[Any]) -> str:
        """
        Deterministic safe messages when the LLM output is invalid or not trustworthy.
        Keeps logic simple and predictable.
        """
        try:
            # BALANCE
            if intent == "balance" and isinstance(tool_result, dict):
                bal = tool_result.get("balance") or tool_result.get("available_balance")
                cur = tool_result.get("currency") or "USD"
                bal_str = self._format_amount(bal, cur)
                acct = self._mask_account(entities.get("account_number") or tool_result.get("account_number"))
                if bal_str:
                    return f"Your account {acct} has a balance of {bal_str}."

            # TRANSACTIONS
            if intent == "transactions":
                if isinstance(tool_result, dict):
                    txs = tool_result.get("transactions")
                elif isinstance(tool_result, list):
                    txs = tool_result
                else:
                    txs = None
                if isinstance(txs, list) and len(txs) > 0:
                    t = txs[0]
                    amt = t.get("amount") or t.get("value")
                    cur = t.get("currency") or t.get("currency_code") or ""
                    # direction: debit if negative or type indicates debit
                    try:
                        direction = "debit" if (t.get("type") == "debit" or (amt is not None and float(amt) < 0)) else "credit"
                    except Exception:
                        direction = "debit" if t.get("type") == "debit" else "credit"
                    desc = t.get("description") or t.get("merchant") or t.get("narration") or "a transaction"
                    acct = self._mask_account(t.get("account") or entities.get("account_number") or tool_result.get("account_number"))
                    amt_str = self._format_amount(amt, cur or "")
                    if amt_str:
                        return f"Your most recent transaction for {acct} was a completed {direction} of {amt_str} for '{desc}'."
                return "I couldn't find any recent transactions. Would you like me to fetch more details?"

            # TRANSFER - success or failure summary
            if intent == "transfer" and isinstance(tool_result, dict):
                status = tool_result.get("status")
                if status == "success":
                    amount = tool_result.get("amount") or entities.get("amount")
                    cur = tool_result.get("currency") or "USD"
                    amt_str = self._format_amount(amount, cur)
                    recipient = tool_result.get("to") or entities.get("recipient_name") or entities.get("to_account_number") or "the recipient"
                    ref = tool_result.get("txn_id") or tool_result.get("transaction_reference") or tool_result.get("reference")
                    if amt_str:
                        if ref:
                            return f"Transfer of {amt_str} to {recipient} completed successfully. Reference: {ref}."
                        return f"Transfer of {amt_str} to {recipient} completed successfully."
                # failure case
                if status in ("error", "failed") or tool_result.get("message"):
                    msg = tool_result.get("message") or "The transfer could not be completed."
                    return f"I couldn't complete the transfer: {msg}. Please check and try again."

            # Generic fallback when no specific formatting applies
            if isinstance(tool_result, dict) and tool_result.get("message"):
                return str(tool_result.get("message"))

        except Exception as e:
            logger.exception("Exception in deterministic fallback: %s", e)

        # ultimate generic fallback
        return "I couldn't retrieve the information right now. Would you like me to try again?"


    async def call_llm_deprecated(self, prompt: str, max_tokens: int = 256) -> str:
        """
        Wrapper around the LLM client. If a real llm_client is provided it will be used.
        Otherwise we use the GeminiLLMClient fallback.
        """
        logger.debug("=" * 60)
        logger.debug("CALL_LLM - Starting")
        logger.debug("Prompt length: %d chars | Max tokens: %d", len(prompt), max_tokens)
        
        if not self.llm_client:
            logger.warning("LLM client not set, attempting to create Gemini client...")
            # Attempt to create a local Gemini client if possible
            try:
                self.llm_client = GeminiLLMClient(api_key=GEMINI_API_KEY, model=self.model_name)
                logger.info("✓ Gemini client created")
            except Exception as e:
                logger.exception("Failed to create Gemini client: %s", e)
                raise RuntimeError("No LLM client available and Gemini client could not be created")

        try:
            logger.info("Calling LLM generate()...")
            text = await self.llm_client.generate(prompt, max_tokens=max_tokens)
            logger.info("✓ LLM response received (length: %d chars)", len(str(text)))
            
            # ensure we return a string
            if isinstance(text, str):
                result = text.strip()
                logger.debug("Response (string): %s", result[:300] + "..." if len(result) > 300 else result)
                logger.debug("CALL_LLM - Complete")
                logger.debug("=" * 60)
                return result
            # some clients return dict-like responses
            if isinstance(text, dict):
                logger.debug("Response (dict): %s", text)
                # try common fields
                result = (text.get("text") or text.get("content") or str(text)).strip()
                logger.debug("Extracted text: %s", result[:300] + "..." if len(result) > 300 else result)
                logger.debug("CALL_LLM - Complete")
                logger.debug("=" * 60)
                return result
            result = str(text)
            logger.debug("Response (other): %s", result[:300] + "..." if len(result) > 300 else result)
            logger.debug("CALL_LLM - Complete")
            logger.debug("=" * 60)
            return result
        except Exception as e:
            logger.exception("LLM client error; falling back to safe message. %s", e)
            logger.error("Prompt that failed: %s", prompt[:500] + "..." if len(prompt) > 500 else prompt)
            logger.debug("CALL_LLM - Error")
            logger.debug("=" * 60)
            return "Sorry, I'm having trouble right now. Please try again in a moment."

    async def call_llm(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        stop: Optional[list] = None,
        stream: Optional[bool] = False,
        **extra_kwargs
    ) -> str:
        """
        Wrapper around LLM client. Tries to pass optional generation kwargs but
        falls back to positional generate(prompt, max_tokens) if the client doesn't accept them.
        Returns a cleaned string.
        """
        if not self.llm_client:
            try:
                self.llm_client = GeminiLLMClient(api_key=GEMINI_API_KEY, model=self.model_name)
            except Exception as e:
                logger.exception("Failed to create Gemini client: %s", e)
                raise RuntimeError("No LLM client available and Gemini client could not be created")

        gen_kwargs = {"prompt": prompt, "max_tokens": max_tokens}
        if temperature is not None:
            gen_kwargs["temperature"] = float(temperature)
        if top_p is not None:
            gen_kwargs["top_p"] = float(top_p)
        if stop is not None:
            gen_kwargs["stop"] = stop
        if stream:
            gen_kwargs["stream"] = True
        gen_kwargs.update(extra_kwargs)

        try:
            try:
                text = await self.llm_client.generate(**gen_kwargs)
            except TypeError:
                # client doesn't accept kwargs — call positional fallback
                text = await self.llm_client.generate(prompt, max_tokens)
            # normalize text to string
            if isinstance(text, str):
                return text.strip()
            if isinstance(text, dict):
                return (text.get("text") or text.get("content") or str(text)).strip()
            return str(text)
        except Exception:
            logger.exception("LLM client error; falling back to safe message.")
            return "Sorry, I'm having trouble right now. Please try again in a moment."


    def should_confirm_action(self, intent: str, entities: Dict[str, Any]) -> bool:
        """
        Determine if action requires user confirmation (risk-based).
        """
        return intent in self.high_risk_intents

    def _append_history(self, session_id: str, entry: Dict[str, Any]) -> None:
        self.conversation_history.setdefault(session_id, []).append(entry)

    def get_history(self, session_id: str) -> List[Dict[str, Any]]:
        return self.conversation_history.get(session_id, [])

    # -----------------------
    # Account resolution helpers
    # -----------------------
    def _resolve_account_from_profile(
        self,
        session_profile: Optional[Dict[str, Any]],
        label: Optional[str],
    ) -> Optional[str]:
        """
        Given a session_profile and an abstract label like "primary", "savings",
        or "current", return a concrete account_number if possible.

        This helper is intentionally small and logic-only so it can be unit-tested.
        """
        if not session_profile:
            return None

        primary = session_profile.get("primary_account")
        accounts = session_profile.get("accounts") or []

        if not label:
            # No label -> use primary if available
            return primary

        label_norm = str(label).strip().lower()

        # Primary / default
        if label_norm in {"primary", "default", "my account", "my primary account"}:
            return primary

        # Savings / current mapping by account_type
        def find_by_type(substr: str) -> Optional[str]:
            for acc in accounts:
                t = (acc.get("account_type") or "").strip().lower()
                if substr in t:
                    return acc.get("account_number")
            return None

        if "saving" in label_norm:  # matches "savings", "saving"
            # Prefer savings; if not found, fall back to primary
            acct = find_by_type("saving")
            return acct or primary

        if "current" in label_norm or "checking" in label_norm:
            return find_by_type("current") or find_by_type("checking")

        # As a last resort, if label looks like a concrete account number, return as-is.
        # This allows the LLM or user to supply explicit account numbers.
        return label if label_norm.startswith("acc") or label_norm.isdigit() else None

    def get_tools(self) -> List[Tuple[str, str, str]]:
        """
        Return canonical list of tools (name, path, description) that this agent expects.
        Used by orchestrator discovery if present.
        """
        return [
            ("balance", "/tools/balance", "Get account balance by account number"),
            ("transactions", "/tools/transactions", "Fetch recent transactions for an account"),
            ("transfer", "/tools/transfer", "Execute a funds transfer (high-risk)")
        ]
