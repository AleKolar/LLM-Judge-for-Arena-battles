# src/services/arena_result.py

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.models.db_models import ArenaResult


async def get_last_result_service(db: AsyncSession):
    stmt = select(ArenaResult).order_by(ArenaResult.created_at.desc()).limit(1)
    result = await db.execute(stmt)
    return result.scalars().first()


async def get_battle_by_id(db: AsyncSession, battle_id: int) -> ArenaResult | None:
    stmt = select(ArenaResult).where(ArenaResult.id == battle_id)
    result = await db.execute(stmt)
    return result.scalars().first()
