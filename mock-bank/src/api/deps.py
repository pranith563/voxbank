from typing import AsyncGenerator

from db.session import AsyncSessionLocal


async def get_db() -> AsyncGenerator:
    """
    Async DB session dependency for FastAPI routes.
    """
    async with AsyncSessionLocal() as session:
        yield session

