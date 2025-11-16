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
from decimal import Decimal

from gemini_llm_client import GeminiLLMClient

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY") or os.environ.get("GENAI_API_KEY") or os.environ.get("GEMINI_TOKEN")
DEFAULT_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-pro")

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)



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
        for name, meta in self.TOOL_SPEC.items():
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
        # include recent history for context (last few messages)
        history = self.get_history(session_id) or []
        history_context = ""
        if history:
            last = history[-6:]  # last up to 6 messages
            history_context = "\nConversation context:\n" + "\n".join([f"{m['role']}: {m['text']}" for m in last]) + "\n\n"

        tools_block = self._render_tools_block()
        prompt = (history_context + self.PROMPT_TEMPLATE.format(tools_block=tools_block)).strip()
        # Add the user request at the end
        prompt = f"{prompt}\n\nUser request: \"{transcript}\"\n\nReturn JSON now."

        # call llm
        raw = await self.call_llm(prompt, max_tokens=max_tokens)
        # Extract JSON
        try:
            json_start = raw.find("{")
            json_end = raw.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                json_str = raw[json_start:json_end]
            else:
                json_str = raw
            parsed = json.loads(json_str)
        except Exception as e:
            logger.exception("Failed to parse LLM decision JSON: %s | raw=%s", e, raw)
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

        # Normalize fields
        parsed.setdefault("action", "respond")
        parsed.setdefault("intent", "unknown")
        parsed.setdefault("requires_tool", bool(parsed.get("tool_name")))
        parsed.setdefault("tool_input", parsed.get("tool_input", {}))
        parsed.setdefault("requires_confirmation", bool(parsed.get("requires_confirmation", False)))
        parsed.setdefault("response", parsed.get("response", ""))

        # Validate tool_input if action == call_tool
        if parsed["action"] == "call_tool" and parsed.get("tool_name"):
            tool = parsed["tool_name"]
            if tool not in self.TOOL_PARAMETERS:
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
            filtered = {k: v for k, v in parsed["tool_input"].items() if k in expected}
            missing = [p for p in expected if p not in filtered and self.TOOL_SPEC[tool]["params"][p].get("required")]
            if missing:
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

        return parsed

    async def _llm_parse_intent(self, transcript: str, session_id: str) -> Dict[str, Any]:
        """
        Use LLM to parse user intent, extract entities, and determine tool requirements.
        Returns a dict with keys:
         - intent, entities, requires_tool (bool), tool_name (opt), tool_input (opt),
           confirmation_required (bool)
        """
        # Build prompt for LLM to extract intent and entities
        available_tools = list(self.INTENT_TOOL_MAP.keys())
        tools_description = "\n".join([
            f"- {tool}: {desc}" for tool, (_, _) in self.INTENT_TOOL_MAP.items()
            for desc in [f"Tool: {tool}"]
        ])
        
        prompt = f"""
You are a banking assistant. Analyze the user's request and extract:
1. Intent: one of {available_tools} or "greeting", "goodbye", "unknown"
2. Entities: Extract relevant information like amounts, account numbers, recipient names, etc.
3. Tool requirement: Does this request need a banking tool?

Available tools:
- balance: Get account balance (requires: account_number)
- transactions: Get transaction history (requires: account_number, optional: limit)
- transfer: Transfer funds (requires: from_account_number, to_account_number, amount, optional: currency, reference)

User request: "{transcript}"

Respond in JSON format:
{{
    "intent": "balance|transactions|transfer|greeting|goodbye|unknown",
    "entities": {{
        "account_number": "...",
        "amount": 0.0,
        "recipient_name": "...",
        "to_account_number": "...",
        "from_account_number": "...",
        "currency": "USD|INR",
        "limit": 10
    }},
    "requires_tool": true|false,
    "tool_name": "balance|transactions|transfer|null",
    "confirmation_required": true|false
}}"""

        try:
            # Get conversation history for context
            history = self.get_history(session_id)
            history_context = ""
            if len(history) > 1:  # More than just current message
                recent = history[-3:]  # Last 3 messages for context
                history_context = "\nRecent conversation:\n"
                for msg in recent:
                    role = msg.get("role", "user")
                    text = msg.get("text", "")
                    history_context += f"{role}: {text}\n"
            
            full_prompt = history_context + prompt if history_context else prompt
            
            # Call LLM
            response = await self.call_llm(full_prompt, max_tokens=512)
            
            # Parse JSON response
            import json
            # Try to extract JSON from response (might have extra text)
            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                parsed = json.loads(json_str)
            else:
                # Fallback: try to parse entire response
                parsed = json.loads(response)
            
            # Extract values
            intent = parsed.get("intent", "unknown")
            entities = parsed.get("entities", {})
            requires_tool = parsed.get("requires_tool", False)
            tool_name = parsed.get("tool_name")
            confirmation_required = parsed.get("confirmation_required", False)
            
            # Build tool input if tool is required
            tool_input = {}
            if requires_tool and tool_name and tool_name in self.INTENT_TOOL_MAP:
                _, input_builder = self.INTENT_TOOL_MAP[tool_name]
                tool_input = input_builder(entities) or {}
            
            # Override confirmation if intent is high-risk
            if not confirmation_required:
                confirmation_required = self.should_confirm_action(intent, entities)
            
            result = {
                "intent": intent,
                "entities": entities,
                "requires_tool": requires_tool,
                "tool_name": tool_name,
                "tool_input": tool_input,
                "confirmation_required": confirmation_required,
            }
            logger.debug("LLM parse result: %s", result)
            return result
            
        except Exception as e:
            logger.exception("Error parsing intent with LLM: %s", e)
            # Fallback to simple keyword-based classification if LLM fails
            if hasattr(self, 'intent_classifier') and self.intent_classifier:
                classification = await self.intent_classifier.classify(transcript)
                intent = classification.get("intent", "unknown")
                entities = classification.get("entities", {}) or {}
                requires_tool = intent in self.INTENT_TOOL_MAP
                tool_name = None
                tool_input = None
                if requires_tool:
                    tool_name, input_builder = self.INTENT_TOOL_MAP[intent]
                    tool_input = input_builder(entities) or {}
                return {
                    "intent": intent,
                    "entities": entities,
                    "requires_tool": requires_tool,
                    "tool_name": tool_name,
                    "tool_input": tool_input,
                    "confirmation_required": self.should_confirm_action(intent, entities),
                }
            # Ultimate fallback
            return {
                "intent": "unknown",
                "entities": {},
                "requires_tool": False,
                "tool_name": None,
                "tool_input": {},
                "confirmation_required": False,
            }

    async def process_user_input(self, transcript: str, session_id: str) -> Dict[str, Any]:
        """
        Process user input using LLM to determine intent and required actions.
        This is a compatibility method that uses the LLM-based parser.
        Returns a dict with keys:
         - intent, entities, confidence, requires_tool (bool), tool_name (opt), tool_input (opt),
           confirmation_required (bool)
        """
        # Use LLM-based parsing (same as orchestrate but without execution)
        parsed = await self._llm_parse_intent(transcript, session_id)
        # Add confidence for backward compatibility
        parsed["confidence"] = 0.9 if parsed.get("requires_tool") else 0.7
        return parsed

    async def orchestrate(self, transcript: str, session_id: str, user_confirmation: Optional[bool] = None, reply_style: Optional[str] = "concise", parse_only: bool = False) -> Dict[str, Any]:
        """
        High-level orchestration method using LLM for intent classification:
         - Uses LLM to extract intent and entities from user input
         - Determines which tool to call (if any)
         - If confirmation required, returns needs_confirmation unless user_confirmation=True
         - Calls MCP tool (via mcp_client) if needed
         - Generates final response via LLM
        """
        logger.info("Processing user input for session %s: %s", session_id, transcript)
        # Maintain history
        self._append_history(session_id, {"role": "user", "text": transcript})

        # Use LLM to extract intent, entities, and determine tool requirements
        parsed = await self._llm_parse_intent(transcript, session_id)

        # If parse_only requested, return parsed structure without executing tools
        if parse_only:
            return {"status": "parsed", "parsed": parsed}

        intent = parsed.get("intent", "unknown")
        entities = parsed.get("entities", {})
        requires_tool = parsed.get("requires_tool", False)
        tool_name = parsed.get("tool_name")
        tool_input = parsed.get("tool_input", {})
        confirmation_required = parsed.get("confirmation_required", False)

        tool_result = None

        # If a confirmation is required and we don't have confirmation, ask for confirmation
        if confirmation_required and not user_confirmation:
            prompt = f"I detected a high-risk action that needs confirmation: intent={intent}, entities={entities}. Do you want to proceed?"
            # craft LLM prompt to ask user in simple language
            llm_text = await self.call_llm(prompt)
            # Append assistant's prompt to history
            self._append_history(session_id, {"role": "assistant", "text": llm_text})
            return {"status": "needs_confirmation", "message": llm_text, "parsed": parsed}

        # If tool required, execute it
        if requires_tool:
            if not self.mcp_client:
                logger.warning("MCP client not configured; cannot execute tool %s", tool_name)
                tool_result = {"status": "not_configured", "message": "MCP client not set up in this environment."}
            else:
                try:
                    # Clean input: ensure numeric amount is present for transfer
                    if tool_name == "transfer" and (tool_input.get("amount") is None):
                        # try to extract from entities
                        tool_input["amount"] = entities.get("amount")
                    
                    # Filter tool_input to only include expected parameters for this tool
                    # Remove any metadata fields that tools don't expect
                    expected_params = self.TOOL_PARAMETERS.get(tool_name, set())
                    filtered_input = {k: v for k, v in tool_input.items() if k in expected_params}
                    
                    # Remove None values to avoid validation errors
                    filtered_input = {k: v for k, v in filtered_input.items() if v is not None}
                    
                    logger.info("Calling MCP tool %s with input %s", tool_name, filtered_input)
                    # Call the actual tool with the filtered input (only expected parameters)
                    tool_result = await self.mcp_client.call_tool(tool_name, filtered_input)
                    logger.info("Tool %s result: %s", tool_name, tool_result)
                except Exception as e:
                    logger.exception("Error executing MCP tool %s: %s", tool_name, e)
                    tool_result = {"status": "error", "message": str(e)}

        # If parse-only was requested above we already returned.
        # Generate final response via LLM (or simple generator)
        response_text = await self.generate_response(intent, entities, tool_result, reply_style=reply_style)
        self._append_history(session_id, {"role": "assistant", "text": response_text})

        return {"status": "ok", "response": response_text, "tool_result": tool_result, "parsed": parsed}

    async def generate_response(self, intent: str, entities: Dict[str, Any], tool_result: Optional[Any] = None, reply_style: Optional[str] = "concise") -> str:
        """
        Generate natural language response based on intent and tool results.
        Uses llm_client.generate(...) if available.
        """
        # If tool result exists, try to produce a concise user-facing message
        try:
            if isinstance(tool_result, dict):
                # Successful transfer
                if tool_result.get("status") == "success":
                    if intent == "transfer":
                        amt = entities.get("amount") or (tool_result.get("amount") if isinstance(tool_result.get("amount"), (int, float)) else None)
                        amt_str = f"{amt}" if amt is not None else "the requested amount"
                        # get recipient info
                        to = tool_result.get("to") or entities.get("recipient_name") or tool_result.get("recipient") or tool_result.get("to_account_number") or tool_result.get("to_account")
                        txn_id = tool_result.get("txn_id") or tool_result.get("transaction_reference")
                        return f"Transfer of {amt_str} completed successfully to {to}. Transaction reference {txn_id}."
                    if intent == "balance":
                        bal = tool_result.get("balance") or tool_result.get("available_balance")
                        cur = tool_result.get("currency") or "USD"
                        return f"Your account balance is {bal} {cur}."
                    if intent == "transactions":
                        txs = tool_result.get("transactions") or tool_result
                        # summarize the most recent few transactions
                        if isinstance(txs, list) and len(txs) > 0:
                            first = txs[0]
                            return f"I found {len(txs)} transactions. Most recent: {first.get('transaction_reference')} {first.get('amount')} {first.get('currency')} ({first.get('status')})."
                        return "No recent transactions found."
                # Tool returned non-success
                # Let LLM craft a helpful message about the failure
                prompt = f"Tool returned: {tool_result}. Please create a concise, user-friendly message explaining the result."
                return await self.call_llm(prompt)
        except Exception as e:
            logger.exception("Error while formatting tool_result: %s", e)
            # continue to LLM fallback

        # No tool result or formatting failed -> ask LLM for a reply
        prompt = f"User intent: {intent}. Entities: {entities}. Provide a helpful assistant reply. style={reply_style}"
        return await self.call_llm(prompt)

    async def call_llm(self, prompt: str, max_tokens: int = 256) -> str:
        """
        Wrapper around the LLM client. If a real llm_client is provided it will be used.
        Otherwise we use the GeminiLLMClient fallback.
        """
        if not self.llm_client:
            # Attempt to create a local Gemini client if possible
            try:
                self.llm_client = GeminiLLMClient(api_key=GEMINI_API_KEY, model=self.model_name)
            except Exception:
                raise RuntimeError("No LLM client available and Gemini client could not be created")

        try:
            text = await self.llm_client.generate(prompt, max_tokens=max_tokens)
            # ensure we return a string
            if isinstance(text, str):
                return text.strip()
            # some clients return dict-like responses
            if isinstance(text, dict):
                # try common fields
                return (text.get("text") or text.get("content") or str(text)).strip()
            return str(text)
        except Exception as e:
            logger.exception("LLM client error; falling back to safe message. %s", e)
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