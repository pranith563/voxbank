# mock-bank/src/db/deps.py
from typing import AsyncGenerator
from .session import AsyncSessionLocal

async def get_db() -> AsyncGenerator:
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            # session closed automatically by context manager
            pass
