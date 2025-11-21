"""
agent/orchestrator.py

ConversationOrchestrator:
- Runs the ReAct-style loop by delegating to the agent for decisions.
- Executes MCP tools between decision steps.

The ReAct loop logic that used to live inside LLMAgent.orchestrate has
been migrated here. VoxBankAgent is now responsible for LLM decisions,
while ConversationOrchestrator owns tool execution and loop control.
"""

from typing import Any, Optional, Dict
import logging

from .agent import VoxBankAgent
from .helpers import (
    build_user_context_block,
    render_history_for_prompt,
    format_observation_for_history,
)
from .normalizer import normalize_input


logger = logging.getLogger("llm_agent")


class ConversationOrchestrator:
    def __init__(self, agent: VoxBankAgent, mcp_client: Any, max_iters: int = 4) -> None:
        self.agent = agent
        self.mcp_client = mcp_client
        self.max_iters = max_iters

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
        - if decision -> call_tool: execute tool, append observation, call decision() again
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

        # observation is the last tool output passed back into decision
        observation: Any = None

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

            # Call decision with current transcript + context, auth state, and last observation (if any)
            history = self.agent.get_history(session_id)
            context_str = render_history_for_prompt(history or [])
            user_context_block = build_user_context_block(session_profile) if session_profile else None
            tools_block = getattr(self.agent, "tools_block", "")
            auth_flag = self.agent.is_authenticated(session_id)
            
            parsed = await self.agent.decision(
                effective_transcript,
                context_str,
                session_profile=session_profile,
                observation=observation,
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

            # respond -> final reply
            if action == "respond" and assistant_response:
                logger.info(
                    "Using LLM-provided response (action=respond) – checking if polishing needed",
                )
                should_polish = False  # previously computed but effectively disabled
                if should_polish:
                    logger.info("Polishing assistant response via generate_response()")
                    try:
                        polished = await self.agent.generate_response(
                            intent or "unknown",
                            parsed.get("tool_input", {}),
                            tool_result or parsed.get("tool_output") or parsed.get("observation"),
                            reply_style=reply_style,
                        )
                        final_reply = polished or assistant_response
                    except Exception as e:
                        logger.exception(
                            "Polishing failed: %s. Falling back to raw assistant_response",
                            e,
                        )
                        final_reply = assistant_response
                else:
                    final_reply = assistant_response

                self.agent._append_history(session_id, {"role": "assistant", "text": final_reply})
                logger.info("ORCHESTRATE - Complete (respond)")
                result = {"status": "ok", "response": final_reply}
                if logout_requested:
                    result["logged_out"] = True
                return result

            # call_tool -> gate by auth, resolve accounts, execute tool, loop again
            if action == "call_tool" and requires_tool and tool_name:
                logger.info("Decision requested tool call: %s (iter %d)", tool_name, iterations)
                if not isinstance(tool_input, dict):
                    tool_input = {}
                    if parsed is not None:
                        parsed["tool_input"] = {}
                if tool_name == "logout_user":
                    tool_input.setdefault("session_id", session_id)
                    if session_profile and session_profile.get("user_id"):
                        tool_input.setdefault("user_id", session_profile.get("user_id"))
                # Gate all non-auth tools behind login/registration.
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

                # Resolve abstract account labels based on session_profile
                if session_profile:
                    from agent.helpers import resolve_account_from_profile  # local import to avoid cycles

                    if tool_name in ("balance", "transactions"):
                        acct_label = tool_input.get("account_number")
                        acct_number = resolve_account_from_profile(session_profile, acct_label)
                        if not acct_number:
                            clarify = (
                                "I couldn't determine which account to use. "
                                "Please specify which account (for example, your savings or current account)."
                            )
                            self.agent._append_history(session_id, {"role": "assistant", "text": clarify})
                            logger.info(
                                "ORCHESTRATE - Unable to resolve account label '%s' for tool %s; asking user to clarify",
                                acct_label,
                                tool_name,
                            )
                            logger.info("=" * 80)
                            return {"status": "clarify", "message": clarify, "parsed": parsed}
                        tool_input["account_number"] = acct_number
                        parsed["tool_input"]["account_number"] = acct_number

                    if tool_name == "transfer":
                        from_label = tool_input.get("from_account_number")
                        from_number = resolve_account_from_profile(session_profile, from_label)
                        if not from_number:
                            clarify = (
                                "Which account would you like to send money from? "
                                "You can say your savings account or current account."
                            )
                            self.agent._append_history(session_id, {"role": "assistant", "text": clarify})
                            logger.info(
                                "ORCHESTRATE - Unable to resolve from_account label '%s' for transfer; asking user to clarify",
                                from_label,
                            )
                            logger.info("=" * 80)
                            return {"status": "clarify", "message": clarify, "parsed": parsed}

                        user_accounts = {
                            acc.get("account_number")
                            for acc in (session_profile.get("accounts") or [])
                            if acc.get("account_number")
                        }
                        if user_accounts and from_number not in user_accounts:
                            msg = (
                                "I can’t move money from an account that doesn’t belong to you. "
                                "Please choose one of your own accounts as the source."
                            )
                            self.agent._append_history(session_id, {"role": "assistant", "text": msg})
                            logger.info(
                                "SECURITY: blocked transfer from non-owned account %s for session %s",
                                from_number,
                                session_id,
                            )
                            logger.info("=" * 80)
                            return {"status": "ok", "response": msg}

                        tool_input["from_account_number"] = from_number
                        parsed["tool_input"]["from_account_number"] = from_number

                # Confirmation for high-risk actions
                if requires_confirmation and not user_confirmation:
                    confirm_msg = f"I will perform: {intent}. {assistant_response or 'Do you want to proceed?'}"
                    self.agent._append_history(session_id, {"role": "assistant", "text": confirm_msg})
                    logger.info(
                        "ORCHESTRATE - Completed (needs_confirmation before tool exec) at iter %d",
                        iterations,
                    )
                    logger.info("=" * 80)
                    return {"status": "needs_confirmation", "message": confirm_msg}

                # Execute tool via MCP
                if not self.mcp_client:
                    logger.error("MCP client not configured - cannot execute tool")
                    observation = {"status": "not_configured", "message": "MCP client not set up."}
                else:
                    try:
                        logger.info("EXECUTING MCP TOOL %s with input %s", tool_name, tool_input)
                        tool_result = await self.execute_tool(tool_name, tool_input)
                        logger.info("Tool result: %s", tool_result)
                        if isinstance(tool_result, (dict, list)):
                            observation = tool_result
                        else:
                            observation = {"status": "ok", "result": str(tool_result)}
                    except Exception as e:
                        logger.exception("Exception while executing tool %s: %s", tool_name, e)
                        observation = {"status": "error", "message": str(e)}

                # Append observation and continue loop
                obs_summary = format_observation_for_history(tool_name, observation)
                self.agent._append_history(
                    session_id,
                    {"role": "tool", "text": obs_summary, "detail": observation},
                )
                if (
                    tool_name == "logout_user"
                    and isinstance(observation, dict)
                    and observation.get("status") == "success"
                ):
                    logout_requested = True
                logger.info(
                    "Appended tool observation to history and continuing loop (iter %d)",
                    iterations,
                )
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

        # If we have a tool_result/observation, let generate_response combine it into final reply
        response_text: Optional[str]
        try:
            response_text = await self.agent.generate_response(
                intent or "unknown",
                parsed.get("tool_input", {}) if parsed else {},
                observation,
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
