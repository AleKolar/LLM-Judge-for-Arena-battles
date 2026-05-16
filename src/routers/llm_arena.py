# src/routers/llm_arena.py
import logging
from slowapi import Limiter
from slowapi.util import get_remote_address
from fastapi import APIRouter, Depends, HTTPException, Request, Response
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from src.database.database import get_async_db
from src.models.db_models import ArenaResult
from src.models.models import BattleHistoryResponse, CompareRequest, WinnerResponse, FullBattleRequest
from src.schemas.schemas import JudgeWinnerRequest
from src.services.ai_service import (
    AVAILABLE_MODELS, DEFAULT_MODELS,
    judge_winner, run_arena_comparison, JUDGE_MODEL,
)
from src.services.arena_result import get_last_result_service, get_battle_by_id
from src.services.download_service import generate_battle_markdown
from src.utils.normalize import normalize_decision, normalize_evidence

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

@router.get("/judge_models")
async def list_judge_models():
    return JUDGE_MODEL


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


@router.post(
    "/battle",
    response_model=WinnerResponse,
    summary="Full LLM Battle Pipeline",
    description=(
        "Full LLM Arena pipeline:\n\n"
        "1. Генерация ответов моделей\n"
        "2. Судейство (LLM Judge)\n"
        "3. Сохранение итогового результата в БД\n"
        "4. Возврат финального verdict\n\n"
        "### Пример запроса:\n"
        "```json\n"
        "{\n"
        '  "models": [\n'
        '    "gpt-4o-mini",\n'
        '    "deepseek-chat"\n'
        "  ],\n"
        '  "judge_model": "deepseek-chat",\n'
        '  "prompt": "Сгенерируй короткую Python-функцию, которая проверяет високосный год. Напиши pytest тесты. Верни только Python-код без markdown."\n'
        "}\n"
        "```"
    )
)
@limiter.limit("5/minute")
async def full_battle(
        payload: FullBattleRequest,
        request: Request,
        db: AsyncSession = Depends(get_async_db)
):
    session = request.app.state.http_session

    models = payload.models

    if len(models) != 2:
        raise HTTPException(400, "Нужно ровно две модели")

    logger.info("battle: models=%s judge=%s", models, payload.judge_model)

    # 1. GENERATION
    compare_result = await run_arena_comparison(
        models=models,
        session=session,
        prompt=payload.prompt
    )

    compare_result = compare_result or {}

    if compare_result.get("error"):
        raise HTTPException(500, compare_result["error"])

    raw_results = compare_result.get("results", [])
    results = normalize_evidence(raw_results)
    elapsed = compare_result.get("elapsed", 0)

    # 2. SAVE
    battle = ArenaResult(
        model1=models[0],
        model2=models[1],
        winner=None,
        message="Pending judge",
        evidence=results
    )

    db.add(battle)
    await db.commit()
    await db.refresh(battle)

    # 3. JUDGE
    decision_raw = await judge_winner(
        results=results,
        session=session,
        judge_model=payload.judge_model
    )

    decision = normalize_decision(decision_raw)

    # 4. UPDATE DB
    battle.judge_model_name = payload.judge_model
    battle.winner = decision["winners"][0] if decision["winners"] else None
    battle.message = decision.get("message", "")
    battle.judge_reason = decision.get("reason", "")
    battle.winner_position = decision.get("winner_position")

    await db.commit()

    # 5. RESPONSE ENRICHMENT
    decision.update({
        "arena_result_id": battle.id,
        "elapsed": elapsed,
        "model_a_name": battle.model1,
        "model_b_name": battle.model2,
        "judge_model_name": payload.judge_model,
    })

    return WinnerResponse(**decision)