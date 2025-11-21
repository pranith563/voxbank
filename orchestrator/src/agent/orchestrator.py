"""
agent/orchestrator.py

ConversationOrchestrator:
- Runs the ReAct-style loop by delegating to the agent for decisions.
- Executes MCP tools between decision steps.

The ReAct loop logic that used to live inside LLMAgent.orchestrate has
been migrated here. VoxBankAgent is now responsible for LLM decisions,
while ConversationOrchestrator owns tool execution and loop control.
"""

import logging
from typing import Any, Optional, Dict

from otp_manager import OtpManager
from .agent import VoxBankAgent
from .helpers import (
    build_user_context_block,
    render_history_for_prompt,
)
from .normalizer import normalize_input
from .otp_workflow import OtpWorkflow
from prompts.tool_spec import TOOL_SPEC


logger = logging.getLogger("agent")


class ConversationOrchestrator:
    def __init__(self, agent: VoxBankAgent, mcp_client: Any, max_iters: int = 10) -> None:
        self.agent = agent
        self.mcp_client = mcp_client
        self.max_iters = max_iters
        self.otp_manager = OtpManager()
        self.otp_workflow = OtpWorkflow(
            agent=self.agent,
            otp_manager=self.otp_manager,
            execute_tool_cb=self.execute_tool,
        )

    async def execute_tool(self, tool_name: str, tool_input: Dict[str, Any]) -> Any:
        """
        Wrapper around MCP client for executing tools.
        """
        return await self.mcp_client.call_tool(tool_name, tool_input)

    async def handle_turn(
        self,
        transcript: str,
        session_id: str,
        session_profile: Optional[Dict[str, Any]] = None,
        user_confirmation: Optional[bool] = None,
        reply_style: str = "concise",
        parse_only: bool = False,
        max_iterations: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        ReAct style orchestration loop:
        - call agent.decision()
        - if decision -> call_tool: execute tool, call decision() again
        - otherwise handle respond / ask_user / ask_confirmation

        Behaviour and return shape are preserved from the legacy
        LLMAgent.orchestrate implementation.
        """
        logger.info("=" * 80)
        logger.info("ORCHESTRATE (ReAct) - Starting")
        logger.info("Session: %s | Transcript: %s", session_id, transcript)
        logger.info(
            "User confirmation: %s | Parse only: %s | Reply style: %s",
            user_confirmation,
            parse_only,
            reply_style,
        )

        eff_max_iterations = max_iterations if max_iterations is not None else self.max_iters

        if session_profile:
            logger.info(
                "ORCHESTRATE: session_profile user_id=%s username=%s primary_account=%s accounts=%d",
                session_profile.get("user_id"),
                session_profile.get("username"),
                session_profile.get("primary_account"),
                len(session_profile.get("accounts") or []),
            )

        otp_pending_result = await self.otp_workflow.intercept_pending_challenge(
            session_id=session_id,
            transcript=transcript,
            reply_style=reply_style,
        )
        if otp_pending_result:
            return otp_pending_result

        auth_state = self.agent._get_auth_state(session_id)
        # If we are currently in a login / registration flow, handle it
        if auth_state.get("flow_stage"):
            logger.info("ORCHESTRATE: routing input to auth flow (stage=%s)", auth_state.get("flow_stage"))
            result = await self.agent._handle_auth_flow(transcript, session_id)
            logger.info("ORCHESTRATE: auth flow handled with status=%s", result.get("status"))
            logger.info("=" * 80)
            return result

        # If the session is unauthenticated and the user explicitly mentions login/registration
        # in this utterance, route directly into the deterministic auth flow instead of asking
        # the LLM to interpret it.
        if not self.agent.is_authenticated(session_id):
            lower_tx = (transcript or "").lower()
            login_keywords = ["login", "log in", "sign in"]
            register_keywords = ["register", "sign up", "create account", "open account"]
            if any(kw in lower_tx for kw in login_keywords + register_keywords):
                logger.info(
                    "ORCHESTRATE: detected explicit login/register intent in transcript; "
                    "routing to auth flow with stage=%s",
                    auth_state.get("flow_stage") or "await_choice",
                )
                result = await self.agent._handle_auth_flow(transcript, session_id)
                logger.info("ORCHESTRATE: auth flow handled with status=%s", result.get("status"))
                logger.info("=" * 80)
                return result

        iterations = 0
        parsed: Optional[Dict[str, Any]] = None
        tool_result: Any = None
        logout_requested = False

        # Normalize raw transcript once per turn via the small normalizer LLM
        history_for_norm = self.agent.get_history(session_id)
        last_assistant_msg = None
        if history_for_norm:
            for msg in reversed(history_for_norm):
                if msg.get("role") == "assistant":
                    last_assistant_msg = msg.get("text")
                    break

        language = "en"
        if session_profile:
            language = (
                session_profile.get("user_profile", {}).get("preferred_language")
                or session_profile.get("language")
                or "en"
            )

        try:
            normalized = await normalize_input(
                llm_client=self.agent.llm_client,
                raw_text=transcript,
                last_assistant_msg=last_assistant_msg,
                language=language,
            )
        except Exception as e:  # pragma: no cover - defensive
            logger.exception("ORCHESTRATE: normalize_input failed; using raw transcript: %s", e)
            normalized = {
                "cleaned_text": transcript,
                "numbers": [],
                "primary_number": None,
                "currency_hint": None,
                "message_type": "free_text",
            }

        effective_transcript = normalized.get("cleaned_text") or transcript
        logger.info(
            "ORCHESTRATE: normalized transcript='%s' (raw='%s') message_type=%s numbers=%s primary=%s",
            effective_transcript,
            transcript,
            normalized.get("message_type"),
            normalized.get("numbers"),
            normalized.get("primary_number"),
        )

        while iterations < eff_max_iterations:
            iterations += 1
            logger.info("ReAct loop iteration %d/%d", iterations, eff_max_iterations)

            # Call decision with current transcript + context, auth state
            history = self.agent.get_history(session_id)
            context_str = render_history_for_prompt(history or [])
            user_context_block = build_user_context_block(session_profile) if session_profile else None
            tools_block = getattr(self.agent, "tools_block", "")
            auth_flag = self.agent.is_authenticated(session_id)
            
            parsed = await self.agent.decision(
                effective_transcript,
                context_str,
                session_profile=session_profile,
                auth_state=auth_flag,
                user_context_block=user_context_block,
                tools_block=tools_block,
                normalized_input=normalized,
            )
            self.agent._append_history(session_id, {"role": "user", "text": effective_transcript})
            logger.debug("Added user message to history")

            action = parsed.get("action")
            intent = parsed.get("intent")
            requires_tool = parsed.get("requires_tool", False)
            tool_name = parsed.get("tool_name")
            tool_input = parsed.get("tool_input", {}) or {}
            requires_confirmation = parsed.get("requires_confirmation", False)
            assistant_response = parsed.get("response", "")

            logger.info(
                "Decision (iter %d): action=%s intent=%s tool=%s requires_confirmation=%s",
                iterations,
                action,
                intent,
                tool_name,
                requires_confirmation,
            )
            logger.debug("Parsed: %s", parsed)

            # Hard guard: unauthenticated sessions must not get account-level answers.
            unauthenticated = not self.agent.is_authenticated(session_id)
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
                state = self.agent._get_auth_state(session_id)
                if not state.get("flow_stage"):
                    state["flow_stage"] = "await_choice"
                login_msg = "You're not logged in yet. Please login or register before I can access your accounts."
                self.agent._append_history(session_id, {"role": "assistant", "text": login_msg})
                logger.info(
                    "ORCHESTRATE - Blocked unauthenticated account-level respond action for session %s (intent=%s, tool=%s)",
                    session_id,
                    intent,
                    tool_name,
                )
                logger.info("=" * 80)
                return {"status": "clarify", "message": login_msg}

            # ask_user / ask_confirmation -> return immediately
            if action in ("ask_user", "ask_confirmation"):
                self.agent._append_history(session_id, {"role": "assistant", "text": assistant_response})
                status = "needs_confirmation" if action == "ask_confirmation" else "clarify"
                logger.info("ORCHESTRATE - Completed (ask_user/ask_confirmation) at iter %d", iterations)
                logger.info("=" * 80)
                return {"status": status, "message": assistant_response}

            if action == "respond" and assistant_response:
                respond_result = await self._handle_respond(
                    transcript=transcript,
                    session_id=session_id,
                    intent=intent,
                    assistant_response=assistant_response,
                    reply_style=reply_style,
                    logout_requested=logout_requested,
                    tool_result=tool_result,
                    parsed=parsed,
                )
                return respond_result

            if action == "call_tool" and requires_tool and tool_name:
                (
                    early_response,
                    new_tool_result,
                    logout_flag,
                ) = await self._handle_call_tool(
                    action=action,
                    intent=intent,
                    tool_name=tool_name,
                    tool_input=tool_input,
                    requires_tool=requires_tool,
                    requires_confirmation=requires_confirmation,
                    user_confirmation=user_confirmation,
                    assistant_response=assistant_response,
                    transcript=transcript,
                    session_id=session_id,
                    session_profile=session_profile,
                    parse_only=parse_only,
                    reply_style=reply_style,
                    iterations=iterations,
                    parsed=parsed,
                )
                if logout_flag:
                    logout_requested = True
                if early_response is not None:
                    return early_response
                tool_result = new_tool_result
                continue

            # If none of the above matched, break
            logger.warning(
                "Decision returned unexpected action or no-op; breaking loop (iter %d): %s",
                iterations,
                parsed,
            )
            break

        # Reached end of loop: either no decision or max iterations hit
        logger.info("Exited ReAct loop after %d iterations", iterations)

        # If we have a tool_result, let generate_response combine it into final reply
        response_text: Optional[str]
        try:
            response_text = await self.agent.generate_response(
                intent or "unknown",
                parsed.get("tool_input", {}) if parsed else {},
            )
        except Exception as e:
            logger.exception("Error generating final response: %s", e)
            response_text = (parsed or {}).get("response") or "Sorry, I'm having trouble right now."

        self.agent._append_history(session_id, {"role": "assistant", "text": response_text})
        logger.info("ORCHESTRATE - Complete (final response)")
        logger.info("=" * 80)
        result = {"status": "ok", "response": response_text}
        if logout_requested:
            result["logged_out"] = True
        return result

    async def _handle_respond(
        self,
        *,
        transcript: str,
        session_id: str,
        intent: Optional[str],
        assistant_response: str,
        reply_style: str,
        logout_requested: bool,
        tool_result: Any,
        parsed: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Handle the "respond" action returned by the decision LLM.
        Keeps the optional polishing hook (currently disabled) and
        centralises logout detection before returning the final payload.
        """
        logger.info("Using LLM-provided response (action=respond) - checking if polishing needed")
        should_polish = False  # Placeholder for future polishing heuristics
        final_reply = assistant_response
        if should_polish:
            try:
                polished = await self.agent.generate_response(
                    intent or "unknown",
                    (parsed or {}).get("tool_input", {}),
                    tool_result,
                    reply_style=reply_style,
                )
                final_reply = polished or assistant_response
            except Exception as exc:  # pragma: no cover - defensive
                logger.exception(
                    "Polishing failed: %s. Falling back to raw assistant_response",
                    exc,
                )
                final_reply = assistant_response

        logout_flag = logout_requested or self._detect_logout(transcript, final_reply, intent)

        self.agent._append_history(session_id, {"role": "assistant", "text": final_reply})
        logger.info("ORCHESTRATE - Complete (respond)")
        result = {"status": "ok", "response": final_reply}
        if logout_flag:
            result["logged_out"] = True
        return result

    def _detect_logout(self, transcript: str, response: str, intent: Optional[str]) -> bool:
        """
        Detect whether the user or agent text implies a logout so downstream
        handlers can clear the session.
        """
        lower_tx = (transcript or "").lower()
        lower_resp = (response or "").lower()
        intent_norm = (intent or "").lower() if intent else ""
        logout_phrases = (
            "logout",
            "logged_out",
            "log out",
            "sign out",
            "signout",
            "log me out",
            "log out now",
            "signed out",
        )
        logout_intents = {"logout", "signout", "reset_session"}
        return bool(
            any(phrase in lower_tx for phrase in logout_phrases)
            or any(phrase in lower_resp for phrase in logout_phrases)
            or intent_norm in logout_intents
        )

    async def _handle_call_tool(
        self,
        *,
        action: str,
        intent: Optional[str],
        tool_name: str,
        tool_input: Any,
        requires_tool: bool,
        requires_confirmation: bool,
        user_confirmation: Optional[bool],
        assistant_response: Optional[str],
        transcript: str,
        session_id: str,
        session_profile: Optional[Dict[str, Any]],
        parse_only: bool,
        reply_style: str,
        iterations: int,
        parsed: Optional[Dict[str, Any]],
    ) -> tuple[Optional[Dict[str, Any]], Any, bool]:
        """
        Handle tool execution including auth gating, confirmation prompts,
        OTP triggers, and logging. Returns a tuple of:
        (early_response_dict_or_None, tool_result, logout_flag)
        """
        logger.info("Decision requested tool call: %s (iter %d)", tool_name, iterations)

        if tool_name == "logout_user":
            response_text = assistant_response or "You are now logged out."
            result = {"status": "ok", "response": response_text, "logged_out": True}
            return result, None, True

        auth_block = self._maybe_block_for_auth(
            tool_name=tool_name,
            session_id=session_id,
            parse_only=parse_only,
            iterations=iterations,
            parsed=parsed,
        )
        if auth_block is not None:
            return auth_block, None, False

        otp_response = await self._maybe_trigger_otp(
            tool_name=tool_name,
            tool_input=tool_input,
            session_id=session_id,
            session_profile=session_profile,
            intent=intent,
            parse_only=parse_only,
        )
        if otp_response is not None:
            return otp_response, None, False

        # Validate required parameters before executing any tool.
        valid, missing = self._validate_tool_input(tool_name, tool_input)
        if not valid:
            tool_result = {
                "status": "error",
                "tool_error": "missing_required_params",
                "tool_name": tool_name,
                "missing_params": missing,
                "partial_input": tool_input or {},
            }
            summary = (
                f"Tool {tool_name} call rejected: missing required params: "
                f"{', '.join(missing)}. The assistant should obtain these values "
                "before retrying the tool."
            )
            self.agent._append_history(
                session_id,
                {"role": "tool", "text": summary, "detail": tool_result},
            )
            logger.info(
                "ORCHESTRATE - Tool %s missing required params %s; appended tool error to history",
                tool_name,
                missing,
            )
            return None, tool_result, False

        if requires_confirmation and not user_confirmation:
            confirm_response = self._build_confirmation_response(
                session_id=session_id,
                intent=intent,
                assistant_response=assistant_response,
                iterations=iterations,
            )
            return confirm_response, None, False

        tool_result = await self._execute_tool_safe(
            tool_name,
            tool_input,
            session_id=session_id,
            session_profile=session_profile,
        )

        # If a \"my_*\" or summary tool reports missing user context despite
        # reaching this point, surface a friendly login/register prompt.
        if (
            isinstance(tool_result, dict)
            and tool_result.get("status") == "error"
            and isinstance(tool_result.get("message"), str)
            and "No user context available" in tool_result.get("message", "")
        ):
            msg = "You're not logged in yet. Would you like to login or register?"
            self.agent._append_history(session_id, {"role": "assistant", "text": msg})
            logger.info(
                "ORCHESTRATE - Tool %s reported missing user context; prompting login/register",
                tool_name,
            )
            logger.info("=" * 80)
            return {"status": "clarify", "message": msg}, None, False

        return None, tool_result, False

    def _maybe_block_for_auth(
        self,
        *,
        tool_name: str,
        session_id: str,
        parse_only: bool,
        iterations: int,
        parsed: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        auth_tools = {
            "register_user",
            "login_user",
            "set_user_audio_embedding",
            "get_user_profile",
            "list_tools",
            "logout_user",
        }
        if not parse_only and not self.agent.is_authenticated(session_id) and tool_name not in auth_tools:
            logger.info(
                "AUTH: session %s not authenticated; prompting for login/registration before tool %s",
                session_id,
                tool_name,
            )
            state = self.agent._get_auth_state(session_id)
            if not state.get("flow_stage"):
                state["flow_stage"] = "await_choice"
            msg = "You're not logged in yet. Would you like to login or register?"
            self.agent._append_history(session_id, {"role": "assistant", "text": msg})
            logger.info(
                "ORCHESTRATE - Completed (auth_required before tool exec) at iter %d",
                iterations,
            )
            logger.info("=" * 80)
            return {"status": "clarify", "message": msg, "parsed": parsed}
        return None

    async def _maybe_trigger_otp(
        self,
        *,
        tool_name: str,
        tool_input: Dict[str, Any],
        session_id: str,
        session_profile: Optional[Dict[str, Any]],
        intent: Optional[str],
        parse_only: bool,
    ) -> Optional[Dict[str, Any]]:
        if tool_name != "transfer" or parse_only:
            return None
        if not self.otp_workflow.should_trigger_transfer_otp(
            session_id=session_id,
            amount_value=tool_input.get("amount"),
            session_profile=session_profile,
        ):
            return None
        return await self.otp_workflow.initiate_transfer_otp(
            session_id=session_id,
            session_profile=session_profile,
            tool_name=tool_name,
            tool_input=tool_input,
            intent=intent,
        )

    def _build_confirmation_response(
        self,
        *,
        session_id: str,
        intent: Optional[str],
        assistant_response: Optional[str],
        iterations: int,
    ) -> Dict[str, Any]:
        confirm_msg = f"I will perform: {intent}. {assistant_response or 'Do you want to proceed?'}"
        self.agent._append_history(session_id, {"role": "assistant", "text": confirm_msg})
        logger.info(
            "ORCHESTRATE - Completed (needs_confirmation before tool exec) at iter %d",
            iterations,
        )
        logger.info("=" * 80)
        return {"status": "needs_confirmation", "message": confirm_msg}

    async def _execute_tool_safe(
        self,
        tool_name: str,
        tool_input: Dict[str, Any],
        *,
        session_id: str,
        session_profile: Optional[Dict[str, Any]],
    ) -> Any:
        if not self.mcp_client:
            logger.error("MCP client not configured - cannot execute tool")
            return None

        payload: Dict[str, Any] = dict(tool_input or {})
        if tool_name in {
            "get_my_profile",
            "get_my_accounts",
            "get_my_beneficiaries",
            "cards_summary",
            "loans_summary",
            "reminders_summary",
            "logout_user",
        }:
            if session_profile and session_profile.get("user_id"):
                payload.setdefault("user_id", str(session_profile.get("user_id")))
            payload.setdefault("session_id", session_id)

        try:
            logger.info("EXECUTING MCP TOOL %s with input %s", tool_name, payload)
            tool_result = await self.execute_tool(tool_name, payload)
            logger.info("Tool result: %s", tool_result)
            return tool_result
        except Exception as exc:
            logger.exception("Exception while executing tool %s: %s", tool_name, exc)
            return None

    def _validate_tool_input(
        self,
        tool_name: str,
        tool_input: Any,
    ) -> tuple[bool, list[str]]:
        """
        Validate tool_input against TOOL_SPEC to ensure all required params
        are present before executing the tool.
        """
        spec = TOOL_SPEC.get(tool_name) or {}
        params_meta = spec.get("params") or {}
        required_params = [
            name for name, meta in params_meta.items() if (meta or {}).get("required")
        ]
        if not required_params:
            return True, []

        payload = tool_input if isinstance(tool_input, dict) else {}
        missing = [
            name
            for name in required_params
            if name not in payload or payload.get(name) in (None, "")
        ]
        return (len(missing) == 0, missing)
