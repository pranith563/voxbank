# VoxBank – Your AI Voice Banking Companion

VoxBank is an AI-powered voice banking assistant that enables users to perform secure financial operations through natural conversation. It reimagines digital banking by combining speech recognition, large language models (LLMs), and secure tool orchestration via the Model Context Protocol (MCP).

## 🎯 Features

- **Natural Voice Interaction**: Simply speak to perform banking operations
- **Secure Tool Execution**: MCP-based microservices ensure safe financial operations
- **Multi-turn Conversations**: Context-aware conversations with memory
- **Risk-based Security**: OTP and biometric verification for high-risk actions
- **Multilingual Support**: Supports multiple languages and regional accents

## 🏗️ Architecture

```
┌─────────────┐
│   Frontend  │ (Voice UI - React)
└──────┬──────┘
       │
┌──────▼──────────┐
│  Orchestrator   │ (LLM + NLU + Context)
└──────┬──────────┘
       │
┌──────▼─────────────────┐
│    MCP Tools           │ (Secure Microservices)
│  ┌──────┬──────┬─────┐ │
│  │Balance│Transfer│...│ │
│  └──────┴──────┴─────┘ │
└──────┬─────────────────┘
       │
┌──────▼──────────┐
│   Mock Bank     │ (Fake Bank Backend)
└─────────────────┘
```

## 📁 Project Structure

- **`/frontend`** - Voice UI (React/TypeScript)
- **`/orchestrator`** - LLM orchestration and conversation engine
- **`/mcp-tools`** - Secure MCP tool services (microservices)
- **`/mock-bank`** - Fake bank backend for testing
- **`/auth-service`** - OTP + Voice Biometrics
- **`/data`** - SQL schema and seed data
- **`/infra`** - Docker Compose and Kubernetes configs
- **`/docs`** - Architecture diagrams and documentation
- **`/demo`** - Demo scripts and assets

## 🚀 Quick Start

### Prerequisites

- Docker and Docker Compose
- Python 3.11+
- Node.js 18+ (for frontend)

### One-Command Setup

```bash
# Start all services
cd infra
docker-compose up -d

# Check services
docker-compose ps
```

### Manual Setup

1. **Orchestrator**
```bash
cd orchestrator
pip install -r requirements.txt
uvicorn src.app:app --reload
```

2. **Mock Bank**
```bash
cd mock-bank
pip install -r requirements.txt
uvicorn app:app --reload --port 8001
```

3. **Frontend**
```bash
cd frontend
npm install
npm run dev
```

## 📚 Documentation

- [Architecture Overview](docs/)
- [API Documentation](orchestrator/README.md)
- [MCP Tools Guide](mcp-tools/README.md)

## 🔒 Security

- All financial operations go through secure MCP tool services
- High-risk actions require OTP and biometric verification
- Policy engine enforces compliance rules
- All transactions are audited and logged

## 🧪 Testing

```bash
# Run tests
pytest

# Test with demo scripts
cd demo/scripts
```

## 📝 License

See [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions welcome! Please read our contributing guidelines first.

## 📧 Contact

For questions or support, please open an issue on GitHub.

