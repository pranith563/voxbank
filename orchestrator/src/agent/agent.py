"""
agent/agent.py

VoxBankAgent
------------

Core LLM agent for VoxBank:
- Handles conversation state and per-session memory.
- Builds prompts (pre-login and post-login).
- Parses LLM JSON decisions.
- Manages deterministic auth flow (login/registration).

Tool execution and the ReAct loop are handled by ConversationOrchestrator
in agent/orchestrator.py. This module focuses purely on the "brain" part.
"""

from typing import Dict, Any, Optional, List, Tuple
import asyncio
import logging
import re
import os
import json
import ast
from decimal import Decimal

import httpx
from pydantic.types import T
from gemini_llm_client import GeminiLLMClient

from prompts.tool_spec import TOOL_SPEC as FALLBACK_TOOL_SPEC
from prompts.decision_prelogin import PRELOGIN_PROMPT_TEMPLATE
from prompts.decision_postlogin import POSTLOGIN_PROMPT_TEMPLATE
from prompts.system_response import SYSTEM_PROMPT
from .helpers import (
    format_amount,
    mask_account,
    is_raw_tool_output,
    render_history_for_prompt,
    format_observation_for_history,
    resolve_account_from_profile,
    deterministic_fallback,
    build_user_context_block,
    build_tools_block,
)


GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY") or os.environ.get("GENAI_API_KEY") or os.environ.get("GEMINI_TOKEN")
DEFAULT_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-pro")
VOX_BANK_BASE_URL = os.environ.get("VOX_BANK_BASE_URL", "http://localhost:9000")


logger = logging.getLogger("llm_agent")


class VoxBankAgent:
    """
    Main LLM agent that orchestrates conversations and tool calls (logic only).
    """

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

    # ------------------------------------------------------------------
    # Tool specification
    # ------------------------------------------------------------------
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
            logger.warning("VoxBankAgent.set_tool_spec called with empty spec; falling back to built-in spec")
            spec = FALLBACK_TOOL_SPEC

        self.tool_spec = spec
        # Precompute parameter sets for validation
        self.tool_parameters = {
            name: set((meta.get("params") or {}).keys())
            for name, meta in self.tool_spec.items()
        }

        # Render tools_block string used inside the LLM prompt
        self.tools_block = build_tools_block(self.tool_spec)

        logger.info("VoxBankAgent tool spec updated with %d tools", len(self.tool_spec))

    # ------------------------------------------------------------------
    # History helpers
    # ------------------------------------------------------------------
    def _append_history(self, session_id: str, entry: Dict[str, Any]) -> None:
        self.conversation_history.setdefault(session_id, []).append(entry)

    def get_history(self, session_id: str) -> List[Dict[str, Any]]:
        return self.conversation_history.get(session_id, [])

    def _render_history_for_prompt(self, session_id: str, max_messages: int = 40) -> str:
        history = self.get_history(session_id) or []
        return render_history_for_prompt(history, max_messages=max_messages)

    def _format_observation_for_history(self, tool_name: str, observation: Any) -> str:
        return format_observation_for_history(tool_name, observation)

    # ------------------------------------------------------------------
    # Auth state / login helpers
    # ------------------------------------------------------------------
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
        return bool(state.get("authenticated"))

    async def _find_mock_bank_user_by_username(self, username: str) -> Optional[Dict[str, Any]]:
        """
        Helper to query mock-bank for a user by username.
        Returns user dict or None.
        """
        base = VOX_BANK_BASE_URL.rstrip("/") if VOX_BANK_BASE_URL else None
        if not base:
            logger.warning("VOX_BANK_BASE_URL not configured; cannot look up user")
            return None

        # The mock-bank API exposes /api/list_users (without a username filter),
        # so we fetch a reasonable page and filter client-side by username.
        url = f"{base}/api/list_users"
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(url, params={"limit": 100, "offset": 0})
                if resp.status_code != 200:
                    logger.warning("User lookup failed for %s: %s", username, resp.text)
                    return None
                data = resp.json() or []
                if not isinstance(data, list):
                    logger.warning("Unexpected payload from list_users: %s", data)
                    return None
                username_norm = (username or "").strip().lower()
                for u in data:
                    if (u.get("username") or "").strip().lower() == username_norm:
                        return u
        except Exception as e:
            logger.exception("Error during user lookup for %s: %s", username, e)
        return None

    async def _login_user_via_http(self, username: str, passphrase: str) -> Optional[Dict[str, Any]]:
        """
        Deterministic login helper: validate username + passphrase by calling
        mock-bank /api/login directly.
        """
        base = VOX_BANK_BASE_URL.rstrip("/") if VOX_BANK_BASE_URL else None
        if not base:
            logger.warning("VOX_BANK_BASE_URL not configured; cannot perform HTTP login")
            return None

        url = f"{base}/api/login"
        username_norm = (username or "").strip().lower()
        passphrase_norm = (passphrase or "").strip().lower()
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.post(url, json={"username": username_norm, "passphrase": passphrase_norm})
                if resp.status_code != 200:
                    logger.warning("AUTH: HTTP login failed for %s: %s", username_norm, resp.text)
                    return None
                data = resp.json() or {}
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
                msg = "Great, let's create a new VoxBank profile. What username would you like to use?"
            else:
                msg = (
                    "To access your accounts, you'll need to login or register. "
                    "Say 'login' if you already have an account or 'register' to create a new one."
                )
            self._append_history(session_id, {"role": "assistant", "text": msg})
            return {"status": "clarify", "message": msg}

        # Stage 2: login - capture username
        if stage == "await_login_username":
            username = text.split()[-1].strip().lower()
            state["temp"]["username"] = username
            logger.info("AUTH: login username candidate=%s", username)

            user = await self._find_mock_bank_user_by_username(username)
            if not user:
                msg = (
                    f"I couldn't find an account for username '{username}'. "
                    "You can try a different username or say 'register' to create a new one."
                )
                self._append_history(session_id, {"role": "assistant", "text": msg})
                return {"status": "clarify", "message": msg}

            state["flow_stage"] = "await_login_passphrase"
            state["temp"]["user_record"] = user
            msg = f"Thanks, {username}. Please provide your passphrase."
            self._append_history(session_id, {"role": "assistant", "text": msg})
            return {"status": "clarify", "message": msg}

        # Stage 3: login - check passphrase via HTTP
        if stage == "await_login_passphrase":
            if not text:
                msg = "I didn't catch your passphrase. Please say it again."
                self._append_history(session_id, {"role": "assistant", "text": msg})
                return {"status": "clarify", "message": msg}

            username = (state["temp"].get("username") or "user").strip().lower()
            login_res = await self._login_user_via_http(username, text)
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
            return {"status": "auth_ok", "message": msg, "authenticated_user_id": user_id}

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

        # Stage 5: registration - capture passphrase and mark session as logged in (session-only)
        if stage == "await_register_passphrase":
            if not text:
                msg = "I didn't catch your passphrase. Please say it again."
                self._append_history(session_id, {"role": "assistant", "text": msg})
                return {"status": "clarify", "message": msg}

            username = state["temp"].get("username") or "user"
            user_id = username
            logger.info("AUTH: registration success for username=%s (session-only user_id=%s)", username, user_id)

            state["authenticated"] = True
            state["user_id"] = user_id
            state["flow_stage"] = None
            state["temp"] = {}

            msg = f"Welcome, {username}! Your VoxBank profile is ready and you're now logged in."
            self._append_history(session_id, {"role": "assistant", "text": msg})
            return {"status": "auth_ok", "message": msg, "authenticated_user_id": user_id}

        # Fallback: reset auth flow if stage unknown
        logger.warning("AUTH: unknown auth flow stage '%s' for session=%s; resetting", stage, session_id)
        state["flow_stage"] = "await_choice"
        msg = (
            "To access your accounts, you'll need to login or register. "
            "Say 'login' if you already have an account or 'register' to create a new one."
        )
        self._append_history(session_id, {"role": "assistant", "text": msg})
        return {"status": "clarify", "message": msg}

    # ------------------------------------------------------------------
    # LLM decision and validation
    # ------------------------------------------------------------------
    def _is_raw_tool_output(self, text: str) -> bool:
        return is_raw_tool_output(text)

    async def call_llm(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        stop: Optional[list] = None,
        stream: Optional[bool] = False,
        **extra_kwargs,
    ) -> str:
        """
        Wrapper around LLM client. Tries to pass optional generation kwargs but
        falls back to positional generate(prompt, max_tokens) if the client doesn't accept them.
        Returns a cleaned string.
        """
        if not self.llm_client:
            raise RuntimeError("No LLM client configured for VoxBankAgent")

        # prefer keyword-style generation if supported
        try:
            raw = await self.llm_client.generate(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=stop,
                stream=stream,
                **extra_kwargs,
            )
        except TypeError:
            raw = await self.llm_client.generate(prompt, max_tokens=max_tokens)

        if raw is None:
            return ""
        if isinstance(raw, str):
            return raw
        if isinstance(raw, dict):
            return str(raw)
        return str(raw)

    async def decision(
        self,
        transcript: str,
        context: str,
        session_profile: Optional[Dict[str, Any]] = None,
        observation: Optional[Any] = None,
        auth_state: Optional[bool] = None,
        max_tokens: int = 512,
        user_context_block: Optional[str] = None,
        tools_block: Optional[str] = None,
    ) -> dict:
        """
        Ask the LLM to *decide* action: respond, call_tool, ask_user, or ask_confirmation.
        Returns parsed JSON dict with keys: action, intent, requires_tool, tool_name, tool_input,
        requires_confirmation, response.
        """
        logger.info("=" * 80)
        logger.info("LLM DECISION - Starting")
        logger.info("\nContext: %s ", context)

        if auth_state is None and session_profile is not None:
            is_auth = bool(session_profile.get("is_authenticated"))
        else:
            is_auth = bool(auth_state)

        if not is_auth:
            logger.info("LLM DECISION - Using PRELOGIN prompt template (unauthenticated)")
            prompt = (
                self.prelogin_prompt_template
                .replace("{history}", context)
                .strip()
            )
        else:
            logger.info("LLM DECISION - Using POSTLOGIN prompt template (authenticated)")
            if user_context_block is None:
                user_context_block = build_user_context_block(session_profile)
            effective_tools_block = tools_block if tools_block is not None else getattr(self, "tools_block", "")
            logger.info("user_context: %s\n",user_context_block)
            prompt = (
                self.postlogin_prompt_template
                .replace("{history}", context)
                .replace("{tools_block}", effective_tools_block)
                .replace("{user_context_block}", user_context_block)
                .strip()
            )

        prompt = f'{prompt}\n\nUser request: "{transcript}"\n'
        if observation is not None:
            try:
                obs_json = json.dumps(observation, default=str)
            except Exception:
                obs_json = str(observation)
            prompt += f"\nObservation (from tool): {obs_json}\n"
        prompt += "\nReturn JSON now."
        logger.info("Calling LLM with prompt (length: %d chars)", len(prompt))

        raw = await self.call_llm(prompt, max_tokens=max_tokens)
        logger.info(raw)
        logger.info("LLM raw response received (length: %d chars)", len(raw))

        # Some models wrap JSON in Markdown-style ```json ... ``` fences.
        # Strip those fences if present so json.loads can parse it.
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            # Drop the first line (e.g. ```json or ```).
            newline_idx = cleaned.find("\n")
            if newline_idx != -1:
                cleaned = cleaned[newline_idx + 1 :]
            else:
                # No newline, just strip the fence prefix.
                cleaned = cleaned.lstrip("`")
        if cleaned.endswith("```"):
            cleaned = cleaned[: cleaned.rfind("```")]
        cleaned = cleaned.strip()

        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            # Fallback: some models emit Python-style dicts with single quotes.
            # Try ast.literal_eval to recover, then normalise to a plain dict.
            try:
                logger.warning("Primary JSON parse failed; attempting literal_eval fallback")
                obj = ast.literal_eval(cleaned)
                if isinstance(obj, dict):
                    parsed = obj
                else:
                    raise ValueError("literal_eval did not return a dict")
            except Exception:
                logger.exception("Failed to parse LLM decision JSON; raw=%s", raw)
                raise

        # Normalisation and validation
        if not isinstance(parsed, dict):
            logger.warning("Parsed decision is not a dict; wrapping: %s", parsed)
            parsed = {"action": "respond", "intent": "unknown", "response": str(parsed)}

        for key in ("action", "intent", "response"):
            parsed.setdefault(key, "")
        parsed.setdefault("requires_tool", False)
        parsed.setdefault("tool_name", None)
        parsed.setdefault("tool_input", {})
        parsed.setdefault("requires_confirmation", False)

        # Enforce pre-login rules: no tools, no user-specific data when not authenticated
        if not is_auth:
            parsed["requires_tool"] = False
            parsed["tool_name"] = None
            parsed["tool_input"] = {}
            parsed["requires_confirmation"] = False
            if parsed.get("action") == "call_tool":
                parsed["action"] = "ask_user"
                parsed["response"] = (
                    "You need to login or register before I can access your accounts. "
                    "Please say 'login' if you already have an account or 'register' to create one."
                )

        # Validate tool input against spec
        if parsed.get("requires_tool") and parsed.get("tool_name"):
            tool = parsed["tool_name"]
            if tool not in self.tool_parameters:
                logger.warning("Invalid tool requested: %s (not in TOOL_PARAMETERS)", tool)
                return {
                    "action": "ask_user",
                    "intent": parsed["intent"],
                    "requires_tool": False,
                    "tool_name": None,
                    "tool_input": {},
                    "requires_confirmation": False,
                    "response": (
                        f"The requested operation '{tool}' is not available. "
                        "What would you like to do instead?"
                    ),
                }

            expected = self.tool_parameters.get(tool, set())
            logger.debug("Expected parameters for %s: %s", tool, expected)
            filtered = {k: v for k, v in (parsed["tool_input"] or {}).items() if k in expected}
            logger.debug("Filtered tool input: %s", filtered)

            base_spec = (self.tool_spec.get(tool) or FALLBACK_TOOL_SPEC.get(tool) or {})
            param_spec = base_spec.get("params", {}) or {}
            missing = [
                p for p in expected
                if p not in filtered and (param_spec.get(p) or {}).get("required")
            ]
            if missing:
                logger.warning("Missing required parameters for %s: %s", tool, missing)
                return {
                    "action": "ask_user",
                    "intent": parsed["intent"],
                    "requires_tool": True,
                    "tool_name": tool,
                    "tool_input": filtered,
                    "requires_confirmation": parsed["requires_confirmation"],
                    "response": (
                        "I need the following information to proceed: "
                        f"{', '.join(missing)}. Please provide it."
                    ),
                }
            parsed["tool_input"] = filtered
            logger.info("Tool validation passed: %s with input %s", tool, filtered)

        logger.info("LLM DECISION - Complete")
        logger.info("=" * 80)
        return parsed

    # ------------------------------------------------------------------
    # Orchestration delegator (keeps external API stable)
    # ------------------------------------------------------------------
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
        Thin delegator to ConversationOrchestrator.handle_turn().

        The full ReAct loop (decision + tool execution) is implemented in
        agent/orchestrator.py. This method exists to keep the public API
        compatible with the previous LLMAgent.orchestrate entrypoint.
        """
        from .orchestrator import ConversationOrchestrator  # local import to avoid cycles

        if not hasattr(self, "_orchestrator") or getattr(self, "_orchestrator", None) is None:
            self._orchestrator = ConversationOrchestrator(self, self.mcp_client, max_iters=max_iterations)

        return await self._orchestrator.handle_turn(
            transcript=transcript,
            session_id=session_id,
            session_profile=session_profile,
            user_confirmation=user_confirmation,
            reply_style=reply_style or "concise",
            parse_only=parse_only,
            max_iterations=max_iterations,
        )

    # ------------------------------------------------------------------
    # Response generation
    # ------------------------------------------------------------------
    async def generate_response(
        self,
        intent: str,
        entities: Dict[str, Any],
        tool_result: Optional[Any] = None,
        reply_style: str = "concise",
        max_tokens: int = 150,
        temperature: float = 0.0,
    ) -> str:
        """
        LLM-first response generator.
        The LLM is given a compact JSON context and expected to return 1-2 polite sentences.
        This wrapper sanitizes the LLM output and falls back to deterministic summaries when needed.
        """
        context = {
            "intent": intent,
            "entities": entities or {},
            "tool_result": tool_result if tool_result is not None else {},
        }

        context_json = json.dumps(context, default=str)
        prompt = SYSTEM_PROMPT.replace("{context_json}", context_json)

        try:
            raw = await self.call_llm(prompt, max_tokens=max_tokens, temperature=temperature)
        except TypeError:
            raw = await self.call_llm(prompt, max_tokens=max_tokens)

        raw = (raw or "").strip()
        logger.debug("LLM reply length=%d", len(raw))

        if not raw:
            logger.warning("LLM returned empty reply; falling back to deterministic summary.")
            return deterministic_fallback(intent, entities, tool_result)

        if re.search(r"\bWhen was this\b|\bWhen did\b|\bHow long ago\b|\bwhat time\b", raw, flags=re.I):
            logger.warning("LLM asked meta-question unexpectedly; using deterministic summary instead.")
            return deterministic_fallback(intent, entities, tool_result)

        if re.search(r"\[.*?\]|\{.*?\}", raw):
            logger.warning("LLM returned placeholders or JSON-like text; using deterministic fallback.")
            return deterministic_fallback(intent, entities, tool_result)

        if intent in ("balance", "transactions", "transfer"):
            if not re.search(r"\d", raw):
                logger.warning("LLM reply lacks digits for numeric intent; falling back.")
                return deterministic_fallback(intent, entities, tool_result)

        reply = " ".join(raw.split())
        if not re.search(r"[.!?]\s*$", reply):
            reply = reply.rstrip() + "."

        try:
            reply = re.sub(
                r"\b(ACC[0-9A-Za-z\-]{4,})\b",
                lambda m: mask_account(m.group(1)) or m.group(1),
                reply,
            )
        except Exception:
            pass

        return reply
