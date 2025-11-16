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

from prompts.vox_assistant import TOOL_SPEC,PROMPT_TEMPLATE

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
    
    def _render_tools_block(self) -> str:
        lines = []
        for name, meta in TOOL_SPEC.items():
            params = ", ".join([f"{p} (required)" if p_meta.get("required") else f"{p} (optional)"
                                for p, p_meta in meta["params"].items()])
            lines.append(f"- {name}: {meta['description']} Params: {params}")
        return "\n".join(lines)
    
    async def _llm_act_decision(self, transcript: str, session_id: str, max_tokens: int = 512) -> dict:
        """
        Ask the LLM to *decide* action: respond, call_tool, ask_user, or ask_confirmation.
        Returns parsed JSON dict with keys: action, intent, requires_tool, tool_name, tool_input,
        requires_confirmation, response
        """
        logger.info("=" * 80)
        logger.info("LLM ACT DECISION - Starting")
        logger.info("Session: %s | Transcript: %s", session_id, transcript)
        
        # include recent history for context (last few messages)
        history = self.get_history(session_id) or []
        logger.debug("History length: %d messages", len(history))
        
        history_context = ""
        if history:
            last = history[-6:]  # last up to 6 messages
            history_context = "\nConversation context:\n" + "\n".join([f"{m['role']}: {m['text']}" for m in last]) + "\n\n"
            logger.debug("History context: %s", history_context[:200] + "..." if len(history_context) > 200 else history_context)

        tools_block = self._render_tools_block()
        logger.debug("Tools block: %s", tools_block)
        
        # Use replace instead of format to avoid issues with JSON braces in template
        prompt = (history_context + PROMPT_TEMPLATE.replace("{tools_block}", tools_block)).strip()
        # Add the user request at the end
        prompt = f"{prompt}\n\nUser request: \"{transcript}\"\n\nReturn JSON now."
        
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

        logger.info("LLM ACT DECISION - Complete")
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
        parsed = await self._llm_act_decision(transcript, session_id)
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

    async def orchestrate(self, transcript: str, session_id: str, user_confirmation: Optional[bool] = None, reply_style: Optional[str] = "concise", parse_only: bool = False) -> Dict[str, Any]:
        """
        High-level orchestration method using LLM for intent classification:
         - Uses LLM to extract intent and entities from user input
         - Determines which tool to call (if any)
         - If confirmation required, returns needs_confirmation unless user_confirmation=True
         - Calls MCP tool (via mcp_client) if needed
         - Generates final response via LLM
        """
        logger.info("=" * 80)
        logger.info("ORCHESTRATE - Starting")
        logger.info("Session: %s | Transcript: %s", session_id, transcript)
        logger.info("User confirmation: %s | Parse only: %s | Reply style: %s", user_confirmation, parse_only, reply_style)
        
        # Maintain history
        self._append_history(session_id, {"role": "user", "text": transcript})
        logger.debug("Added user message to history")

        # Use LLM to extract intent, entities, and determine tool requirements
        logger.info("Calling _llm_act_decision()...")
        parsed = await self._llm_act_decision(transcript, session_id)
        
        action = parsed.get("action")
        intent = parsed.get("intent")
        requires_tool = parsed.get("requires_tool", False)
        tool_name = parsed.get("tool_name")
        tool_input = parsed.get("tool_input", {})
        requires_confirmation = parsed.get("requires_confirmation", False)
        assistant_response = parsed.get("response", "")
        
        logger.info("LLM Decision Summary:")
        logger.info("  Action: %s", action)
        logger.info("  Intent: %s", intent)
        logger.info("  Requires Tool: %s", requires_tool)
        logger.info("  Tool Name: %s", tool_name)
        logger.info("  Tool Input: %s", tool_input)
        logger.info("  Requires Confirmation: %s", requires_confirmation)
        logger.info("  Assistant Response: %s", assistant_response[:100] + "..." if len(assistant_response) > 100 else assistant_response)

        # If the LLM wants to ask user (clarify or confirm) -> return needs_confirmation or ask
        if action in ("ask_user", "ask_confirmation"):
            logger.info("LLM requested to ask user (action=%s)", action)
            # append assistant text to history and return
            self._append_history(session_id, {"role": "assistant", "text": assistant_response})
            status = "needs_confirmation" if action == "ask_confirmation" else "clarify"
            logger.info("Returning status=%s with message: %s", status, assistant_response)
            logger.info("ORCHESTRATE - Complete (ask_user/ask_confirmation)")
            logger.info("=" * 80)
            return {"status": status, "message": assistant_response, "parsed": parsed}

        # If the LLM decided to call a tool
        tool_result = None
        if action == "call_tool" and requires_tool and tool_name:
            logger.info("Tool execution required: %s", tool_name)
            
            if requires_confirmation and not user_confirmation:
                logger.info("Tool requires confirmation but user_confirmation=False")
                # ask for explicit confirmation in orchestrate layer
                confirm_msg = f"I will perform: {intent}. {assistant_response or 'Do you want to proceed?'}"
                logger.info("Asking for confirmation: %s", confirm_msg)
                self._append_history(session_id, {"role": "assistant", "text": confirm_msg})
                logger.info("ORCHESTRATE - Complete (needs_confirmation)")
                logger.info("=" * 80)
                return {"status": "needs_confirmation", "message": confirm_msg, "parsed": parsed}
            
            if not self.mcp_client:
                logger.error("MCP client not configured - cannot execute tool")
                tool_result = {"status": "not_configured", "message": "MCP client not set up."}
            else:
                try:
                    # filtered_input already performed in _llm_act_decision
                    logger.info("=" * 60)
                    logger.info("EXECUTING MCP TOOL")
                    logger.info("Tool: %s", tool_name)
                    logger.info("Input: %s", tool_input)
                    logger.info("=" * 60)
                    
                    tool_result = await self.mcp_client.call_tool(tool_name, tool_input)
                    
                    logger.info("=" * 60)
                    logger.info("MCP TOOL RESULT")
                    logger.info("Tool: %s", tool_name)
                    logger.info("Result: %s", tool_result)
                    logger.info("=" * 60)
                except Exception as e:
                    logger.exception("Error executing MCP tool %s: %s", tool_name, e)
                    logger.error("Tool input was: %s", tool_input)
                    tool_result = {"status": "error", "message": str(e)}

        # If action == "respond" just use assistant_response
        if action == "respond" and assistant_response:
            logger.info("Using LLM-provided response (action=respond)")
            self._append_history(session_id, {"role": "assistant", "text": assistant_response})
            logger.info("ORCHESTRATE - Complete (respond)")
            logger.info("=" * 80)
            return {"status": "ok", "response": assistant_response, "tool_result": tool_result, "parsed": parsed}

        # Otherwise, we still have to synthesize a response (maybe include tool_result)
        logger.info("Generating response via generate_response()...")
        logger.debug("Intent: %s, Entities: %s, Tool result: %s", intent, parsed.get("tool_input", {}), tool_result)
        response_text = await self.generate_response_llm_first(intent, parsed.get("tool_input", {}), tool_result)
        logger.info("Generated response: %s", response_text[:100] + "..." if len(response_text) > 100 else response_text)
        self._append_history(session_id, {"role": "assistant", "text": response_text})
        logger.info("ORCHESTRATE - Complete")
        logger.info("=" * 80)
        return {"status": "ok", "response": response_text, "tool_result": tool_result, "parsed": parsed}

    async def generate_response_llm_first(self, intent: str, entities: Dict[str, Any], tool_result: Optional[Any] = None, reply_style: str = "concise") -> str:
    # Build context JSON for the LLM
        context = {
            "intent": intent,
            "entities": entities or {},
            "tool_result": tool_result if tool_result is not None else {}
        }
        system_user_prompt = """SYSTEM: You are VoxBank's assistant. Your job is to produce a single short user-facing sentence or two based only on the provided JSON context. Do NOT invent, guess, or ask follow-up questions unless required input is missing. If required input is missing, return a single short clarifying question asking explicitly for the missing field(s) only.

    Rules:
    - Use only the values present in the JSON. Do not add new numbers, account ids, names, or dates.
    - Mask account numbers in the text (e.g. "ACC****9001" or "primary"). If the JSON contains an account field labelled "account_number" unmasked, display it masked.
    - Do not use square-bracket placeholders like [Balance Amount]. If a value is missing, ask for it (see above).
    - Keep the reply concise (1-2 sentences). Use polite, human tone.
    - For transactions: summarize most recent transaction(s) with amount, direction (debit/credit), short description/merchant, and status. Do not ask meta questions (e.g., "When was this created?") unless the user specifically asked for that field and it's missing from the data.
    - For failures: state the failure briefly and, if helpful, a single actionable next step.
    - For transfers or high-risk actions: include the transaction reference/id if present, and do not indicate success unless the tool_result clearly shows status == "success".
    - If tool_result is empty or not a dict/list, respond: "I couldn't retrieve the information right now. Would you like me to try again?"

    USER CONTEXT (JSON):
    {context_json}

    Return only the reply text (no JSON, no explanations).
    """
        prompt = system_user_prompt.format(context_json=json.dumps(context, default=str))

        # Call LLM with deterministic settings; update call_llm to accept temperature param if available
        raw = await self.call_llm(prompt, max_tokens=150)  # ensure call_llm uses low temperature (0) in the LLM client

        # Basic sanitation: remove extra whitespace, and ensure it doesn't ask an unrelated meta question
        reply = raw.strip()
        # If LLM produced a meta question asking about creation time without user asking, fallback to deterministic summary:
        if re.search(r"\bWhen was this\b|\bWhen did\b|\bHow long ago\b", reply, flags=re.I):
            # fallback: produce deterministic summary using available fields
            if isinstance(tool_result, dict):
                # attempt to build a concise transaction summary
                txs = tool_result.get("transactions") or (tool_result if isinstance(tool_result, list) else None)
                if isinstance(txs, list) and len(txs) > 0:
                    t = txs[0]
                    amt = t.get("amount") or t.get("value")
                    cur = t.get("currency") or t.get("currency_code") or ""
                    direction = "debit" if t.get("type") == "debit" or float(amt) < 0 else "credit"
                    desc = t.get("description") or t.get("merchant") or t.get("narration") or "a transaction"
                    acct = self._mask_account(t.get("account") or entities.get("account_number") or tool_result.get("account_number"))
                    amt_str = self._format_amount(amt, cur or "")
                    if amt_str:
                        return f"Your most recent transaction for {acct} was a completed {direction} of {amt_str} for '{desc}'."
            # otherwise return a generic safe message
            return "I couldn't determine the full transaction details. Would you like me to fetch more details?"

        # Ensure we don't return bracket placeholders etc.
        if re.search(r"\[.*?\]|\{.*?\}", reply):
            # fallback deterministic
            if intent == "balance" and isinstance(tool_result, dict):
                bal = tool_result.get("balance") or tool_result.get("available_balance")
                cur = tool_result.get("currency") or "USD"
                bal_str = self._format_amount(bal, cur)
                acct = self._mask_account(entities.get("account_number") or tool_result.get("account_number"))
                if bal_str:
                    return f"Your account {acct} has a balance of {bal_str}."
            return "Sorry, I couldn't generate the reply. Would you like me to try again?"

        return reply


    async def generate_response(self, intent: str, entities: Dict[str, Any], tool_result: Optional[Any] = None, reply_style: Optional[str] = "concise") -> str:
        """
        Generate natural language response based on intent and tool results.
        Uses llm_client.generate(...) if available.
        """
        logger.debug("=" * 60)
        logger.debug("GENERATE_RESPONSE - Starting")
        logger.debug("Intent: %s | Entities: %s | Tool result: %s", intent, entities, tool_result)
        
        # If tool result exists, try to produce a concise user-facing message
        try:
            if isinstance(tool_result, dict):
                logger.debug("Tool result is dict, attempting to format...")
                # Successful transfer
                if tool_result.get("status") == "success":
                    logger.info("Tool returned success status")
                    if intent == "transfer":
                        logger.debug("Formatting transfer success response")
                        amt = entities.get("amount") or (tool_result.get("amount") if isinstance(tool_result.get("amount"), (int, float)) else None)
                        amt_str = f"{amt}" if amt is not None else "the requested amount"
                        # get recipient info
                        to = tool_result.get("to") or entities.get("recipient_name") or tool_result.get("recipient") or tool_result.get("to_account_number") or tool_result.get("to_account")
                        txn_id = tool_result.get("txn_id") or tool_result.get("transaction_reference")
                        response = f"Transfer of {amt_str} completed successfully to {to}. Transaction reference {txn_id}."
                        logger.info("Generated transfer response: %s", response)
                        logger.debug("GENERATE_RESPONSE - Complete")
                        logger.debug("=" * 60)
                        return response
                    if intent == "balance":
                        logger.debug("Formatting balance response")
                        bal = tool_result.get("balance") or tool_result.get("available_balance")
                        cur = tool_result.get("currency") or "USD"
                        response = f"Your account balance is {bal} {cur}."
                        logger.info("Generated balance response: %s", response)
                        logger.debug("GENERATE_RESPONSE - Complete")
                        logger.debug("=" * 60)
                        return response
                    if intent == "transactions":
                        logger.debug("Formatting transactions response")
                        txs = tool_result.get("transactions") or tool_result
                        # summarize the most recent few transactions
                        if isinstance(txs, list) and len(txs) > 0:
                            first = txs[0]
                            response = f"I found {len(txs)} transactions. Most recent: {first.get('transaction_reference')} {first.get('amount')} {first.get('currency')} ({first.get('status')})."
                            logger.info("Generated transactions response: %s", response)
                            logger.debug("GENERATE_RESPONSE - Complete")
                            logger.debug("=" * 60)
                            return response
                        response = "No recent transactions found."
                        logger.info("Generated transactions response: %s", response)
                        logger.debug("GENERATE_RESPONSE - Complete")
                        logger.debug("=" * 60)
                        return response
                # Tool returned non-success
                logger.warning("Tool returned non-success status: %s", tool_result.get("status"))
                # Let LLM craft a helpful message about the failure
                prompt = f"Tool returned: {tool_result}. Please create a concise, user-friendly message explaining the result."
                logger.debug("Calling LLM to format tool failure message...")
                response = await self.call_llm(prompt)
                logger.debug("GENERATE_RESPONSE - Complete")
                logger.debug("=" * 60)
                return response
        except Exception as e:
            logger.exception("Error while formatting tool_result: %s", e)
            # continue to LLM fallback

        # No tool result or formatting failed -> ask LLM for a reply
        logger.debug("No tool result or formatting failed, calling LLM for general response...")
        prompt = f"User intent: {intent}. Entities: {entities}. Provide a helpful assistant reply. style={reply_style}"
        response = await self.call_llm(prompt)
        logger.debug("GENERATE_RESPONSE - Complete")
        logger.debug("=" * 60)
        return response

    async def call_llm(self, prompt: str, max_tokens: int = 256) -> str:
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