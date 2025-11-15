# VoxBank Orchestrator

The brain of VoxBank - LLM orchestration and conversation engine.

## Components

- **LLM Agent**: Main conversation controller
- **NLU**: Intent classification and entity extraction
- **Context Management**: Session and memory management
- **Policy Engine**: Security and compliance guards
- **MCP Client**: Communication with tool services

## Setup

```bash
pip install -r requirements.txt
uvicorn src.app:app --reload
```

## API Endpoints

- `POST /api/voice/process` - Process voice input
- `GET /api/health` - Health check

