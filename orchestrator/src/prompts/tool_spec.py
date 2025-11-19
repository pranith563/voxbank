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

TOOL_SPEC_V2 = {
    "balance": {
        "description": "Get the current balance for a specified bank account.",
        "params": {
            "account_number": {
                "type": "string",
                "required": True,
                "description": "The unique identifier for the bank account (e.g., 'ACC12345')."
            }
        },
        "examples": [
            {"user_query": "What's my primary account balance?", "tool_input": {"account_number": "ACC12345"}},
            {"user_query": "balance for savings", "tool_input": {"account_number": "ACC12345"}}
        ]
    },
    "transactions": {
        "description": "Retrieve a list of recent transactions for a given bank account.",
        "params": {
            "account_number": {
                "type": "string",
                "required": True,
                "description": "The account to fetch transactions for (e.g., 'ACC12345')."
            },
            "limit": {
                "type": "integer",
                "required": False,
                "default": 10,
                "description": "The maximum number of transactions to return."
            }
        },
        "examples": [
            {"user_query": "Show my recent transactions", "tool_input": {"account_number": "ACC006566", "limit": 10}},
            {"user_query": "Get the last 5 transactions for my savings account", "tool_input": {"account_number": "ACC002001", "limit": 5}}
        ]
    },
    "transfer": {
        "description": "Initiate a funds transfer between two accounts. This is a HIGH RISK action.",
        "params": {
            "from_account_number": {
                "type": "string",
                "required": True,
                "description": "The account from which to transfer funds."
            },
            "to_account_number": {
                "type": "string",
                "required": True,
                "description": "The recipient's account number."
            },
            "amount": {
                "type": "number",
                "required": True,
                "description": "The amount of money to transfer."
            },
            "currency": {
                "type": "string",
                "required": False,
                "default": "USD",
                "description": "The currency of the transfer."
            }
        },
        "examples": [
            {"user_query": "Send $50 from my primary to savings", "tool_input": {"from_account_number": "ACC002002", "to_account_number": "ACC002001", "amount": 50, "currency": "USD"}},
            {"user_query": "transfer 1000 INR to ACC98765", "tool_input": {"from_account_number": "ACC08765", "to_account_number": "ACC98765", "amount": 1000, "currency": "INR"}}
        ]
    }
}

