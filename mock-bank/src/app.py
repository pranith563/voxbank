"""
mock-bank/src/app.py

FastAPI application entrypoint for the VoxBank mock-bank service.

This module wires together:
- Logging configuration (file-based under mock-bank/logs/)
- CORS and basic middleware
- Domain routers under src/api/ (users, accounts & transfers)
"""

import os

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.requests import Request

from db import session as db_session_module
from db.session import engine
from logging_config import get_logger, setup_logging
from api.accounts import router as accounts_router
from api.users import router as users_router

# Load environment variables early
load_dotenv()

# Configure logging before creating the app
setup_logging()
logger = get_logger("mock_bank")

app = FastAPI(title="Mock Bank API (VoxBank)", version="1.0.0")

# CORS (open for demo)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """
    Lightweight request logger to help trace mock-bank traffic.
    """
    try:
        body = await request.body()
        logger.info(
            "HTTP %s %s from %s body=%s",
            request.method,
            request.url.path,
            request.client.host if request.client else "?",
            body.decode(errors="ignore")[:200],
        )
    except Exception:
        logger.exception("Failed to read request body for logging")
    response = await call_next(request)
    return response


@app.get("/api/health")
async def health():
    """
    Simple health check endpoint.
    """
    return {"status": "healthy"}


# Log effective DB URL once at import time
logger.info(
    "Effective DATABASE_URL: %s",
    getattr(db_session_module, "DATABASE_URL", os.environ.get("DATABASE_URL")),
)

# Include domain routers
app.include_router(users_router, prefix="/api")
app.include_router(accounts_router, prefix="/api")


@app.on_event("startup")
async def on_startup():
    logger.info("Mock-bank starting up")


@app.on_event("shutdown")
async def on_shutdown():
    try:
        await engine.dispose()
    except Exception:
        logger.exception("Error disposing engine on shutdown")
    logger.info("Mock-bank shutting down")

