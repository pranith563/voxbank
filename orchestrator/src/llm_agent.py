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

from gemini_llm_client import GeminiLLMClient

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY") or os.environ.get("GENAI_API_KEY") or os.environ.get("GEMINI_TOKEN")
DEFAULT_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-pro")

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

from prompts.vox_assistant import TOOL_SPEC,PROMPT_TEMPLATE, SYSTEM_PROMPT

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
    
    # Map tool names to their expected parameters (for filtering metadata)
    TOOL_PARAMETERS = {k: set(v["params"].keys()) for k, v in TOOL_SPEC.items()}

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        intent_classifier: Optional[Any] = None,
        llm_client: Optional[Any] = None,
        mcp_client: Optional[Any] = None,
        high_risk_intents: Optional[List[str]] = None,
    ):
        self.model_name = model_name
        # Keep intent_classifier as optional fallback, but LLM is primary
        self.intent_classifier = intent_classifier
        self.llm_client = llm_client or GeminiLLMClient(api_key=GEMINI_API_KEY, model=self.model_name)
        self.mcp_client = mcp_client  # expected to have `call_tool(tool_name, payload)` async method
        self.conversation_history: Dict[str, List[Dict[str, Any]]] = {}
        self.high_risk_intents = high_risk_intents or ["transfer", "payment", "loan_application"]
        self.tools_block = self._render_tools_block()
        logger.debug("Tools block: %s", self.tools_block)


    def _render_tools_block(self) -> str:
        lines = []
        for name, meta in TOOL_SPEC.items():
            params = ", ".join([f"{p} (required)" if p_meta.get("required") else f"{p} (optional)"
                                for p, p_meta in meta["params"].items()])
            lines.append(f"- {name}: {meta['description']} Params: {params}")
        return "\n".join(lines)
    
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

    async def decision(self, transcript: str, context: str, observation: Optional[Any] = None, max_tokens: int = 512) -> dict:
        """
        Ask the LLM to *decide* action: respond, call_tool, ask_user, or ask_confirmation.
        Returns parsed JSON dict with keys: action, intent, requires_tool, tool_name, tool_input,
        requires_confirmation, response
        """
        logger.info("=" * 80)
        logger.info("LLM DECISION - Starting")
        logger.info("Context: %s | Transcript: %s", context, transcript)
        
        # add context/history to the prompt, tools_block too
        prompt = PROMPT_TEMPLATE.replace("{history}",context).replace("{tools_block}", self.tools_block).strip()

        # Add the user request at the end
        prompt = f"{prompt}\n\nUser request: \"{transcript}\"\n"
        
        logger.info("LLM FULL PROMPT: %s",prompt)
        # If observation present, append observation JSON for LLM to reason about
        if observation is not None:
            try:
                obs_json = json.dumps(observation, default=str)
            except Exception:
                obs_json = str(observation)
            prompt += f"\nObservation (from tool): {obs_json}\n"

        prompt += "\nReturn JSON now."
        logger.info("Calling LLM with prompt (length: %d chars)", len(prompt))
        logger.debug("Full prompt: %s", prompt[:500] + "..." if len(prompt) > 500 else prompt)

        # call llm
        raw = await self.call_llm(prompt, max_tokens=max_tokens)
        logger.info("LLM raw response received (length: %d chars)", len(raw))
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

        # Validate tool_input if action == call_tool
        if parsed["action"] == "call_tool" and parsed.get("tool_name"):
            tool = parsed["tool_name"]
            logger.info("Validating tool call: %s", tool)
            logger.debug("Tool input before validation: %s", parsed["tool_input"])
            
            if tool not in self.TOOL_PARAMETERS:
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
            expected = self.TOOL_PARAMETERS.get(tool, set())
            logger.debug("Expected parameters for %s: %s", tool, expected)
            filtered = {k: v for k, v in parsed["tool_input"].items() if k in expected}
            logger.debug("Filtered tool input: %s", filtered)
            
            missing = [p for p in expected if p not in filtered and TOOL_SPEC[tool]["params"][p].get("required")]
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

    async def process_user_input(self, transcript: str, session_id: str) -> Dict[str, Any]:
        """
        Process user input using LLM to determine intent and required actions.
        This is a compatibility method that uses the LLM-based parser.
        Returns a dict with keys:
         - intent, entities, confidence, requires_tool (bool), tool_name (opt), tool_input (opt),
           confirmation_required (bool)
        """
        # Use LLM-based decision parsing
        parsed = await self.decision(transcript, session_id)
        # Convert to expected format for backward compatibility
        result = {
            "intent": parsed.get("intent", "unknown"),
            "entities": parsed.get("tool_input", {}),  # Use tool_input as entities
            "confidence": 0.9 if parsed.get("requires_tool") else 0.7,
            "requires_tool": parsed.get("requires_tool", False),
            "tool_name": parsed.get("tool_name"),
            "tool_input": parsed.get("tool_input", {}),
            "confirmation_required": parsed.get("requires_confirmation", False),
        }
        return result

    async def orchestrate(self, transcript: str, session_id: str, user_confirmation: Optional[bool] = None, reply_style: Optional[str] = "concise", parse_only: bool = False, max_iterations: int = 4) -> Dict[str, Any]:
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

        # Ensure user message added to history first so decision sees it
        self._append_history(session_id, {"role": "user", "text": transcript})
        logger.debug("Added user message to history")

        iterations = 0
        parsed = None
        tool_result = None

        # observation is the last tool output passed back into decision
        observation = None

        while iterations < max_iterations:
            iterations += 1
            logger.info("ReAct loop iteration %d/%d", iterations, max_iterations)

            # Call decision with current transcript + context and last observation (if any)
            parsed = await self.decision(transcript, self._render_history_for_prompt(session_id), observation=observation)

            action = parsed.get("action")
            intent = parsed.get("intent")
            requires_tool = parsed.get("requires_tool", False)
            tool_name = parsed.get("tool_name")
            tool_input = parsed.get("tool_input", {})
            requires_confirmation = parsed.get("requires_confirmation", False)
            assistant_response = parsed.get("response", "")

            logger.info("Decision (iter %d): action=%s intent=%s tool=%s requires_confirmation=%s", iterations, action, intent, tool_name, requires_confirmation)
            logger.debug("Parsed: %s", parsed)

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

            # If decision says call_tool -> execute tool, capture observation and loop again
            if action == "call_tool" and requires_tool and tool_name:
                logger.info("Decision requested tool call: %s (iter %d)", tool_name, iterations)

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
        logger.debug("LLM reply (raw): %s", raw[:1000])

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
        logger.debug("Prompt preview: %s", prompt[:200] + "..." if len(prompt) > 200 else prompt)
        
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