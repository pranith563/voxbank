# Logging Configuration

## Overview

The orchestrator uses file-based logging with separate log files for different modules. This makes it easy to debug issues by tracing the flow of requests, LLM calls, and tool executions.

## Log Files

All log files are stored in `orchestrator/logs/` directory:

- **`app.log`** - FastAPI application logs (API requests, endpoints, startup/shutdown)
- **`agent.log`** - LLM agent orchestration logs (intent parsing, tool calls, LLM interactions)
- **`mcp_client.log`** - MCP client logs (tool execution, MCP server communication)
- **`gemini_client.log`** - Gemini LLM client logs (API calls, responses)

## Log Rotation

Log files automatically rotate when they reach 10MB, keeping up to 5 backup files:
- `app.log`
- `app.log.1`
- `app.log.2`
- etc.

## Log Levels

Set the log level using the `LOG_LEVEL` environment variable:
- `DEBUG` - Detailed information for debugging
- `INFO` - General informational messages (default)
- `WARNING` - Warning messages
- `ERROR` - Error messages only

Example:
```bash
export LOG_LEVEL=DEBUG
```

## What Gets Logged

### app.py
- API request details (endpoint, session ID, user ID, transcript)
- Agent orchestration calls and responses
- Session state changes
- Startup and shutdown events
- Error details with full context

### agent.py
- LLM decision-making process (`_llm_act_decision`)
- Full prompts sent to LLM
- LLM raw responses
- JSON parsing and validation
- Tool validation and filtering
- Tool execution requests and results
- Response generation flow
- Conversation history management

### Key Log Sections

Each major operation is logged with clear section markers:
```
================================================================================
ORCHESTRATE - Starting
Session: sess-123 | Transcript: What's my balance?
...
LLM ACT DECISION - Starting
...
EXECUTING MCP TOOL
Tool: balance
Input: {'account_number': 'primary'}
...
MCP TOOL RESULT
Tool: balance
Result: {'status': 'success', 'balance': 1000.0}
...
ORCHESTRATE - Complete
================================================================================
```

## Debugging Tips

1. **Find where a request failed**: Search for the session ID in the logs
2. **Trace LLM calls**: Look for "CALL_LLM" sections with full prompts and responses
3. **Check tool execution**: Search for "EXECUTING MCP TOOL" sections
4. **Understand decision flow**: Follow the "LLM ACT DECISION" sections to see how the LLM decided what to do
5. **Find errors**: Search for "ERROR" or "Exception" in the logs

## Example Log Flow

```
2025-01-XX 10:00:00 - voxbank.orchestrator - INFO - API Request: POST /api/text/process
2025-01-XX 10:00:00 - agent - INFO - ORCHESTRATE - Starting
2025-01-XX 10:00:00 - agent - INFO - LLM ACT DECISION - Starting
2025-01-XX 10:00:00 - agent - INFO - Calling LLM with prompt (length: 1234 chars)
2025-01-XX 10:00:01 - agent - INFO - LLM raw response received (length: 456 chars)
2025-01-XX 10:00:01 - agent - INFO - LLM Decision: action=call_tool, intent=balance, tool_name=balance
2025-01-XX 10:00:01 - agent - INFO - EXECUTING MCP TOOL
2025-01-XX 10:00:01 - agent - INFO - Tool: balance
2025-01-XX 10:00:01 - agent - INFO - Input: {'account_number': 'primary'}
2025-01-XX 10:00:01 - agent - INFO - MCP TOOL RESULT
2025-01-XX 10:00:01 - agent - INFO - Result: {'status': 'success', 'balance': 1000.0}
2025-01-XX 10:00:01 - agent - INFO - Generated response: Your account balance is 1000.0 USD.
2025-01-XX 10:00:01 - agent - INFO - ORCHESTRATE - Complete
```

## Console Output

By default, only WARNING and ERROR level messages are shown on the console. All other logs go to the log files. This keeps the console clean while maintaining detailed logs in files.

