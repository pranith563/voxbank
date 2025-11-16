# Put inside LLMAgent (class-level)
TOOL_SPEC = {
    "balance": {
        "description": "Get account balance",
        "params": {
            "account_number": {"type": "string", "required": True}
        }
    },
    "transactions": {
        "description": "Get recent transactions for an account",
        "params": {
            "account_number": {"type": "string", "required": True},
            "limit": {"type": "integer", "required": False}
        }
    },
    "transfer": {
        "description": "Execute a funds transfer (HIGH RISK)",
        "params": {
            "from_account_number": {"type": "string", "required": True},
            "to_account_number": {"type": "string", "required": True},
            "amount": {"type": "number", "required": True},
            "currency": {"type": "string", "required": False, "default": "USD"},
            "initiated_by_user_id": {"type": "string", "required": False},
            "reference": {"type": "string", "required": False}
        }
    }
}

PROMPT_TEMPLATE = """
SYSTEM: You are VoxBank's assistant. You may either:
 - directly respond to the user question (action = "respond"),
 - call a bank tool (action = "call_tool") — only when necessary to fulfill the user's request,
 - or ask the user a short clarifying question or confirmation (action = "ask_user" / "ask_confirmation").

When you choose to call a tool, produce `tool_name` and `tool_input` that match the tool's parameter schema (see TOOLS below). If the tool is HIGH RISK (like transfer), set `requires_confirmation` to true if user consent is not explicit.

Return *only* a single JSON object (no extra text) with the exact fields described in the JSON schema below.

TOOLS:
{tools_block}

JSON RESPONSE FORMAT:
{{
  "action": "respond" | "call_tool" | "ask_user" | "ask_confirmation",
  "intent": "<short-intent-label>",         # e.g. balance, transfer, transactions, greeting, unknown
  "requires_tool": true|false,
  "tool_name": "<tool-name-or-null>",
  "tool_input": {{ ... }},                 # only present if action == "call_tool"
  "requires_confirmation": true|false,     # set true for high-risk or when user hasn't confirmed
  "response": "<assistant message>"        # only present if action == "respond" or "ask_user" or "ask_confirmation"
}}

IMPORTANT:
- If you cannot build valid tool_input (missing required fields), do not call the tool; instead return action="ask_user" and ask a single short clarifying question.
- Ensure numeric fields are numbers in JSON (no currency symbols).
- Keep `response` short and actionable.
- Use low creativity (temperature ~0) — be deterministic.

EXAMPLES:
1) "What's my savings balance?"
=> the LLM should return action:"call_tool", tool_name:"balance", tool_input:{"account_number": "primary"} (or the extracted account mask)

2) "Send $50 to John (phone +91...)"
=> If user did not confirm, you can set requires_confirmation:true and action:"ask_confirmation" with a short message like "Do you want to transfer 50 INR to John ending ... ? Reply YES to proceed."

END
"""
