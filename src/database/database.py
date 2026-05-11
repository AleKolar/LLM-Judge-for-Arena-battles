# src/database/database.py
from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

# Синхронный движок – исключительно для Alembic и тестов
SYNC_DATABASE_URL = "sqlite:///./arena_battle.db"

sync_engine = create_engine(
    SYNC_DATABASE_URL,
    connect_args={"check_same_thread": False}
)

# Асинхронный движок – для приложения
ASYNC_DATABASE_URL = "sqlite+aiosqlite:///./arena_battle.db"
async_engine = create_async_engine(
    ASYNC_DATABASE_URL,
    echo=False,
)

AsyncSessionLocal = async_sessionmaker(
    bind=async_engine,
    class_=AsyncSession,
    expire_on_commit=False
)

class Base(DeclarativeBase):
    pass

# Алиас для Alembic - Нужен !
engine = sync_engine

async def get_async_db():
    """Dependency для получения сессии в эндпоинтах."""
    async with AsyncSessionLocal() as session:
        yield session
