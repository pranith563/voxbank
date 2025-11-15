# MCP Tools

Secure MCP Tool Services (microservices) for financial operations.

## Services

- **balance**: Account balance inquiries
- **transfer**: Fund transfers
- **transactions**: Transaction history
- **loan_inquiry**: Loan information
- **reminders**: Payment reminders

## Architecture

Each tool service has:
- `app.py` - FastAPI wrapper
- `schemas.py` - Input/output schemas
- `service.py` - Business logic
- `Dockerfile` - Container definition

## Common Utilities

Shared utilities in `common/`:
- `utils.py` - General utilities
- `auth.py` - Authentication
- `validators.py` - Input validation

