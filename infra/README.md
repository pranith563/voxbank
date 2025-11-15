# Infrastructure

Deployment, Infrastructure, and DevOps configuration.

## Docker Compose

One-command demo setup:

```bash
docker-compose up -d
```

This starts all services:
- Orchestrator (port 8000)
- Mock Bank (port 8001)
- MCP Tools (ports 8002-8007)
- Auth Service (port 8004)
- NGINX Reverse Proxy (port 80)

## Kubernetes

Optional Kubernetes manifests for production deployment:

- `orchestrator.yaml` - Orchestrator deployment
- `mock-bank.yaml` - Mock bank service
- `mcp-tools.yaml` - MCP tool services
- `ingress.yaml` - Ingress configuration

## Reverse Proxy

NGINX configuration for routing requests to appropriate services.

