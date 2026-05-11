# src/services/arena_result.py

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.models.db_models import ArenaResult


async def get_last_result_service(db: AsyncSession):
    stmt = (
        select(ArenaResult)
        .order_by(ArenaResult.created_at.desc())
        .limit(1)
    )

    result = await db.execute(stmt)
    last_battle = result.scalars().first()

    if not last_battle:
        return None

    return last_battle
