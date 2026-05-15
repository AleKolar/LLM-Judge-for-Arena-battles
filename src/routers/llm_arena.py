# src/routers/llm_arena.py
import logging
from slowapi import Limiter
from slowapi.util import get_remote_address
from fastapi import APIRouter, Depends, HTTPException, Request, Response
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from src.database.database import get_async_db
from src.models.db_models import ArenaResult
from src.models.models import BattleHistoryResponse, CompareRequest, WinnerResponse
from src.schemas.schemas import JudgeWinnerRequest
from src.services.ai_service import (
    AVAILABLE_MODELS, DEFAULT_MODELS,
    judge_winner, run_arena_comparison,
)
from src.services.arena_result import get_last_result_service, get_battle_by_id
from src.services.download_service import generate_battle_markdown

logger = logging.getLogger("llm_arena_router")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

router = APIRouter(prefix="/api/llm-arena", tags=["LLM Arena"])
limiter = Limiter(key_func=get_remote_address)

@router.get("/models")
async def list_models():
    return AVAILABLE_MODELS


@router.post("/compare")
@limiter.limit("6/minute")   # не более 6 запросов в минуту
async def run_comparison(
        request: Request,
        payload: CompareRequest,
        db: AsyncSession = Depends(get_async_db)
):
    session = request.app.state.http_session
    models = payload.models or DEFAULT_MODELS
    if len(models) < 2:
        logger.warning("compare: недостаточно моделей: %s", models)
        raise HTTPException(400, "Нужно выбрать две модели")
    logger.info("compare: запрос моделей %s", models)

    try:
        result = await run_arena_comparison(models, session, payload.prompt)
    except Exception as e:
        logger.exception("Ошибка при генерации: %s", e)
        raise HTTPException(500, "Внутренняя ошибка сервера") from e

    battle = ArenaResult(
        model1=models[0], model2=models[1],
        winner=None, message="Ожидает решения судьи",
        evidence=result.get("results", [])
    )
    db.add(battle)
    await db.commit()
    await db.refresh(battle)
    logger.info("compare: создан battle id=%s", battle.id)

    return {
        "arena_result_id": battle.id,
        "results": result.get("results", []),
        "elapsed": result.get("elapsed", 0),
    }


@router.post("/winner/{battle_id}", response_model=WinnerResponse)
@limiter.limit("10/minute")  # судейство 10 раз / запросов в минуту
async def declare_winner(
        battle_id: int,
        payload: JudgeWinnerRequest,
        request: Request,
        db: AsyncSession = Depends(get_async_db)
):
    battle = await get_battle_by_id(db, battle_id)
    if not battle:
        logger.warning("winner: битва %d не найдена", battle_id)
        raise HTTPException(404, f"Битва с id={battle_id} не найдена")

    results = battle.evidence
    if not isinstance(results, list) or len(results) < 2:
        logger.error("winner: некорректные evidence для битвы %d", battle_id)
        raise HTTPException(400, "Некорректные данные битвы в БД")

    logger.info("winner: судейство битвы %d, судья %s", battle_id, payload.judge_model)
    try:
        decision = await judge_winner(
            results,
            request.app.state.http_session,
            judge_model=payload.judge_model,
        )
    except Exception as e:
        logger.exception("Ошибка при судействе битвы %d: %s", battle_id, e)
        raise HTTPException(500, "Внутренняя ошибка сервера") from e

    battle.judge_model_name = payload.judge_model

    if decision.get("winners"):
        battle.winner = decision["winners"][0]
        battle.message = decision["message"]
        battle.judge_reason = decision.get("reason", "")
        battle.winner_position = decision.get("winner_position")
    else:
        battle.winner = None
        battle.message = decision["message"]
        battle.judge_reason = decision.get("reason", "")
        battle.winner_position = None

    await db.commit()
    logger.info("winner: результат битвы %d сохранён, победитель: %s", battle_id, battle.winner)

    decision["model_a_name"] = battle.model1
    decision["model_b_name"] = battle.model2
    decision["judge_model_name"] = payload.judge_model

    return WinnerResponse(**decision)


@router.get("/history", response_model=list[BattleHistoryResponse])
async def get_history(db: AsyncSession = Depends(get_async_db)):
    stmt = select(ArenaResult).order_by(ArenaResult.id.desc()).limit(10)
    result = await db.execute(stmt)
    battles = result.scalars().all()
    logger.info("history: возвращено %d записей", len(battles))
    return battles


@router.get("/download-result/{battle_id}")
async def download_result(battle_id: int, db: AsyncSession = Depends(get_async_db)):
    battle = await get_battle_by_id(db, battle_id)
    if not battle:
        logger.warning("download: битва %d не найдена", battle_id)
        raise HTTPException(404, f"Битва с id={battle_id} не найдена")
    logger.info("download: генерация markdown для битвы %d", battle_id)
    markdown = generate_battle_markdown(battle)
    return Response(
        content=markdown,
        media_type="text/markdown",
        headers={"Content-Disposition": f"attachment; filename=battle_{battle_id}_result.md"}
    )


@router.get("/last-result")
async def get_last_result(db: AsyncSession = Depends(get_async_db)):
    last_battle = await get_last_result_service(db)
    if not last_battle:
        logger.warning("last-result: нет ни одной битвы")
        raise HTTPException(404, "История битв пуста")
    logger.info("last-result: битва id=%d", last_battle.id)
    markdown = generate_battle_markdown(last_battle)
    return Response(
        content=markdown,
        media_type="text/markdown",
        headers={"Content-Disposition": "attachment; filename=last_result.md"}
    )
