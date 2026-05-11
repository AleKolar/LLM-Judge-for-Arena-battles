# src/routers/llm_arena.py

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.database.database import get_async_db
from src.models.db_models import ArenaResult
from src.models.models import BattleHistoryResponse, CompareRequest, WinnerResponse, ModelEvidence
from src.services.ai_service import (
    AVAILABLE_MODELS,
    DEFAULT_MODELS,
    judge_winner,
    run_arena_comparison,
)
from src.services.arena_result import get_last_result_service
from src.utils.normalize import normalize_evidence, to_md

router = APIRouter(prefix="/api/llm-arena", tags=["LLM Arena"])

@router.get("/models")
async def list_models():
    return AVAILABLE_MODELS

@router.post("/compare")
async def run_comparison(
    request: Request,
    payload: CompareRequest,
    db: AsyncSession = Depends(get_async_db)
):
    session = request.app.state.http_session
    models = payload.models or DEFAULT_MODELS

    if len(models) < 2:
        raise HTTPException(400, "Нужно выбрать две модели")

    result = await run_arena_comparison(models, session, payload.prompt)

    battle = ArenaResult(
        model1=models[0],
        model2=models[1],
        winner=None,
        message="Сравнение завершено",
        evidence=result.get("results", [])
    )
    db.add(battle)
    await db.commit()
    await db.refresh(battle)

    result["arena_result_id"] = battle.id
    return result

@router.post("/winner", response_model=WinnerResponse)
async def declare_winner(
    request: Request,
    db: AsyncSession = Depends(get_async_db)
):
    stmt = select(ArenaResult).order_by(ArenaResult.id.desc()).limit(1)
    result = await db.execute(stmt)
    last_battle = result.scalar()  # или result.scalars().first()

    if not last_battle:
        raise HTTPException(400, "Сначала запустите сравнение моделей.")

    results = last_battle.evidence
    if not isinstance(results, list) or len(results) < 2:
        raise HTTPException(400, "Некорректные данные битвы в БД")

    decision = judge_winner(results)

    last_battle.winner = decision["winners"][0] if decision["winners"] else "draw"
    last_battle.message = decision["message"]
    await db.commit()

    last_battle.winner = decision["winners"][0] if decision["winners"] else "draw"
    last_battle.message = decision["message"]
    await db.commit()

    return WinnerResponse(**decision)


@router.get("/history", response_model=list[BattleHistoryResponse])
async def get_history(db: AsyncSession = Depends(get_async_db)):
    stmt = select(ArenaResult).order_by(ArenaResult.id.desc()).limit(10)
    result = await db.execute(stmt)
    battles = result.scalars().all()
    return battles


@router.get("/last-result")
async def get_last_result(db: AsyncSession = Depends(get_async_db)):
    last_battle = await get_last_result_service(db)

    if not last_battle:
        raise HTTPException(404, "История битв пуста")

    # 1. нормализуем сырые данные из БД
    evidence = normalize_evidence(last_battle.evidence)

    # 2. превращаем evidence в markdown-блок
    evidence_md = to_md(evidence)

    # 3. собираем итоговый markdown документ
    markdown = (
        f"# 🧠 LLM Arena Result\n\n"
        f"## 🏆 Result\n{last_battle.message}\n\n"
        f"---\n\n"
        f"## 📊 Evidence\n\n"
        f"{evidence_md}"
    )

    return Response(
        content=markdown,
        media_type="text/markdown",
        headers={
            "Content-Disposition": "attachment; filename=last_result.md"
        }
    )