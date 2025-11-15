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


class LLMAgent:
    """
    Main LLM agent that orchestrates conversations and tool calls.
    """

    INTENT_TOOL_MAP = {
        "balance": ("check_balance", lambda ent: {"account_id": ent.get("account_masked") or "default"}),
        "transfer": ("transfer_funds", lambda ent: {"to": ent.get("recipient_name") or ent.get("phone"),
                                                    "amount": ent.get("amount")}),
        "transactions": ("get_transactions", lambda ent: {"account_id": ent.get("account_masked") or "default",
                                                          "limit": ent.get("limit", 5)}),
        "loan_inquiry": ("loan_inquiry", lambda ent: {"loan_type": ent.get("loan_type")}),
        "reminder": ("set_reminder", lambda ent: {"when": ent.get("date") or ent.get("time"),
                                                  "title": ent.get("title", "reminder")})
    }

    def __init__(
        self,
        model_name: str = "gpt-4",
        intent_classifier: Optional[Any] = None,
        llm_client: Optional[Any] = None,
        mcp_client: Optional[Any] = None,
        high_risk_intents: Optional[List[str]] = None,
    ):
        self.model_name = model_name
        self.intent_classifier = intent_classifier or KeywordIntentClassifier()
        self.llm_client = llm_client
        self.mcp_client = mcp_client  # expected to have `execute(tool_name, payload, session_id)` async method
        self.conversation_history: Dict[str, List[Dict[str, Any]]] = {}
        self.high_risk_intents = high_risk_intents or ["transfer", "payment", "loan_application"]

    async def process_user_input(self, transcript: str, session_id: str) -> Dict[str, Any]:
        """
        Process user input and determine intent and required actions.
        Returns a dict with keys:
         - intent, entities, confidence, requires_tool (bool), tool_name (opt), tool_input (opt),
           confirmation_required (bool)
        """
        logger.info("Processing user input for session %s: %s", session_id, transcript)
        # maintain history
        self._append_history(session_id, {"role": "user", "text": transcript})

        # classify intent
        classification = await self.intent_classifier.classify(transcript)
        intent = classification.get("intent", "unknown")
        entities = classification.get("entities", {})
        confidence = float(classification.get("confidence", 0.0))

        # determine if requires tool and which
        requires_tool = intent in self.INTENT_TOOL_MAP
        tool_name = None
        tool_input = None
        confirmation_required = self.should_confirm_action(intent, entities)

        if requires_tool:
            tool_name, input_builder = self.INTENT_TOOL_MAP[intent]
            tool_input = input_builder(entities)
            # add session_id and metadata
            tool_input["session_id"] = session_id
            tool_input["source_transcript"] = transcript

        result = {
            "intent": intent,
            "entities": entities,
            "confidence": confidence,
            "requires_tool": requires_tool,
            "tool_name": tool_name,
            "tool_input": tool_input,
            "confirmation_required": confirmation_required,
        }
        logger.debug("Process result: %s", result)
        return result

    async def orchestrate(self, transcript: str, session_id: str, user_confirmation: Optional[bool] = None) -> Dict[str, Any]:
        """
        High-level orchestration method:
         - processes input
         - if confirmation required, prompts/ waits for confirmation (user_confirmation param)
         - calls MCP tool (via mcp_client) if needed
         - generates final response via LLM
        """
        parsed = await self.process_user_input(transcript, session_id)

        intent = parsed["intent"]
        entities = parsed["entities"]
        requires_tool = parsed["requires_tool"]
        tool_name = parsed["tool_name"]
        tool_input = parsed["tool_input"]
        confirmation_required = parsed["confirmation_required"]

        tool_result = None

        # If a confirmation is required and we don't have confirmation, ask for confirmation
        if confirmation_required and not user_confirmation:
            prompt = f"Action requires confirmation: intent={intent}, entities={entities}. Please confirm."
            llm_text = await self.call_llm(prompt)
            # Append assistant's prompt to history
            self._append_history(session_id, {"role": "assistant", "text": llm_text})
            return {"status": "needs_confirmation", "message": llm_text, "parsed": parsed}

        if requires_tool:
            if not self.mcp_client:
                logger.warning("MCP client not configured; cannot execute tool %s", tool_name)
                tool_result = {"status": "not_configured", "message": "MCP client not set up in this environment."}
            else:
                # Execute MCP tool - expected async API
                try:
                    logger.info("Calling MCP tool %s with input %s", tool_name, tool_input)
                    tool_result = await self.mcp_client.execute(tool_name, tool_input, session_id=session_id)
                    logger.info("Tool %s result: %s", tool_name, tool_result)
                except Exception as e:
                    logger.exception("Error executing MCP tool %s", tool_name)
                    tool_result = {"status": "error", "message": str(e)}

        # generate final response via LLM (or simple generator)
        response_text = await self.generate_response(intent, entities, tool_result)
        self._append_history(session_id, {"role": "assistant", "text": response_text})

        return {"status": "ok", "response": response_text, "tool_result": tool_result, "parsed": parsed}

    async def generate_response(self, intent: str, entities: Dict[str, Any], tool_result: Optional[Any] = None) -> str:
        """
        Generate natural language response based on intent and tool results.
        Uses llm_client.generate(...) if available.
        """
        if tool_result:
            # Simple templating for common tools
            if isinstance(tool_result, dict) and tool_result.get("status") == "success":
                if intent == "transfer":
                    txn_id = tool_result.get("txn_id") or tool_result.get("transaction_id")
                    amt = entities.get("amount")
                    to = tool_result.get("to") or entities.get("recipient_name") or tool_result.get("recipient")
                    return f"Transfer of ₹{amt} to {to} completed successfully. Transaction id {txn_id}."
                if intent == "balance":
                    bal = tool_result.get("balance") or tool_result.get("available_balance")
                    return f"Your account balance is ₹{bal}."
                # fallback
                return tool_result.get("message", "Operation completed.")
            else:
                # if the tool returned failure or not-configured, ask LLM to craft a message
                prompt = f"Tool result for intent {intent}: {tool_result}. Make a concise user-facing message."
                return await self.call_llm(prompt)

        # No tool result => generic response generated by LLM
        prompt = f"User intent: {intent}. Entities: {entities}. Produce a helpful assistant reply."
        return await self.call_llm(prompt)

    async def call_llm(self, prompt: str) -> str:
        """
        Wrapper around the LLM client. If a real llm_client is provided it will be used.
        Otherwise we use the SimpleLLMClient fallback.
        """
        if not self.llm_client:
            raise RuntimeError("No LLM client available")
        try:
            text = await self.llm_client.generate(prompt)
            return text
        except Exception as e:
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


# --------------------
# Example / local demo
# --------------------
if __name__ == "__main__":
    # Minimal demonstration of orchestrate flow without real MCP
    class MockMCPClient:
        async def execute(self, tool_name: str, payload: Dict[str, Any], session_id: str = ""):
            # very small mock behaviors
            await asyncio.sleep(0.05)
            if tool_name == "check_balance":
                return {"status": "success", "balance": 12500}
            if tool_name == "transfer_funds":
                if payload.get("amount") and payload["amount"] > 100000:
                    return {"status": "failed", "message": "Insufficient funds"}
                return {"status": "success", "txn_id": "TXN12345", "to": payload.get("to")}
            return {"status": "success", "message": "ok"}

    async def run_demo():
        agent = LLMAgent(model_name="dev", llm_client= GeminiLLMClient(api_key=GEMINI_API_KEY, model=DEFAULT_MODEL), mcp_client=MockMCPClient())
        sid = "session_demo_1"

        # 1) balance query
        out = await agent.orchestrate("What's my account balance?", sid)
        print("Balance flow:", out)

        # 2) transfer flow (requires confirmation)
        out2 = await agent.orchestrate("Transfer ₹500 to Ramesh", sid)
        print("Transfer-step1:", out2)

        # simulate user confirms (user_confirmation=True)
        out3 = await agent.orchestrate("Transfer ₹500 to Ramesh", sid, user_confirmation=True)
        print("Transfer-step2:", out3)

    asyncio.run(run_demo())
