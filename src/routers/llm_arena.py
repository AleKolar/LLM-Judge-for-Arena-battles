# src/routers/llm_arena.py
import logging

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from slowapi import Limiter
from slowapi.util import get_remote_address
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.database.database import get_async_db
from src.models.db_models import ArenaResult
from src.models.models import BattleHistoryResponse, FullBattleRequest, WinnerResponse
from src.schemas.schemas import CompareRequest, JudgeWinnerRequest
from src.services.ai_service import (
    AVAILABLE_MODELS,
    DEFAULT_MODELS,
    JUDGE_PROMPT_TEMPLATE,
    PUBLIC_JUDGE_MODELS,
    SYSTEM_PROMPT,
    judge_winner,
    resolve_judge_model_key,
    run_arena_comparison,
)
from src.services.arena_result import get_battle_by_id, get_last_result_service
from src.services.download_service import generate_battle_markdown
from src.utils.normalize import normalize_decision, normalize_evidence

logger = logging.getLogger("llm_arena_router")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

router = APIRouter(prefix="/api/llm-arena", tags=["LLM Arena"])
limiter = Limiter(key_func=get_remote_address)


def _fallback_error_evidence(models: list[str], error: str) -> list[dict[str, str]]:
    """Build one error evidence item per requested participant.

    This is a last-resort guard for orchestration-level failures. Ordinary
    Ollama errors are already returned by ai_service.fetch_from_ollama().
    The attempted battle is still persisted so the user can see the failure.
    """
    return [
        {
            "model": model_key,
            "content": error,
            "status": "error",
        }
        for model_key in models
    ]


@router.get("/models")
async def list_models():
    return AVAILABLE_MODELS


@router.get("/judge_models")
async def list_judge_models():
    # Hide legacy aliases from the UI; they remain accepted by old request
    # schemas and are resolved to the corresponding local Ollama model.
    return PUBLIC_JUDGE_MODELS


@router.get("/prompts/system")
async def get_system_prompt():
    """Return the default prompt used for battle participants."""
    return {"prompt": SYSTEM_PROMPT}


@router.get("/prompts/judge")
async def get_judge_prompt():
    """Return the default Judge prompt template."""
    return {"prompt": JUDGE_PROMPT_TEMPLATE}


@router.post("/compare")
@limiter.limit("6/minute")
async def run_comparison(
        request: Request,
        payload: CompareRequest,
        db: AsyncSession = Depends(get_async_db),
):
    session = request.app.state.http_session
    models = payload.models or DEFAULT_MODELS

    if len(models) < 2:
        logger.warning("compare: недостаточно моделей: %s", models)
        raise HTTPException(400, "Нужно выбрать две модели")

    logger.info("compare: запрос моделей %s", models)

    try:
        result = await run_arena_comparison(models, session, payload.prompt)
    except Exception as exc:
        # Preserve the attempted comparison even if orchestration itself
        # fails. Model-level failures normally arrive as status=error items.
        logger.exception("compare: ошибка orchestration: %s", exc)
        result = {
            "results": _fallback_error_evidence(
                models,
                f"Не удалось выполнить запрос к локальному Ollama: {type(exc).__name__}: {exc}",
            ),
            "elapsed": 0,
        }

    evidence = result.get("results") or _fallback_error_evidence(
        models,
        result.get("error", "Ollama не вернул результаты генерации."),
    )

    # IMPORTANT BUSINESS RULE:
    # Always persist the Arena attempt, including evidence with status=error.
    # The next /winner request uses this evidence to report unavailable models
    # or to identify the only participant that managed to answer.
    battle = ArenaResult(
        model1=models[0],
        model2=models[1],
        winner=None,
        message="Ожидает решения судьи",
        evidence=evidence,
    )
    db.add(battle)
    await db.commit()
    await db.refresh(battle)

    logger.info(
        "compare: создана запись попытки battle id=%s; statuses=%s",
        battle.id,
        [item.get("status") for item in evidence],
    )

    return {
        "arena_result_id": battle.id,
        "results": evidence,
        "elapsed": result.get("elapsed", 0),
    }


@router.post("/winner/{battle_id}", response_model=WinnerResponse)
@limiter.limit("10/minute")
async def declare_winner(
        battle_id: int,
        payload: JudgeWinnerRequest,
        request: Request,
        db: AsyncSession = Depends(get_async_db),
):
    battle = await get_battle_by_id(db, battle_id)
    if not battle:
        logger.warning("winner: битва %d не найдена", battle_id)
        raise HTTPException(404, f"Битва с id={battle_id} не найдена")

    results = battle.evidence
    if not isinstance(results, list) or len(results) < 2:
        logger.error("winner: некорректные evidence для битвы %d", battle_id)
        raise HTTPException(400, "Некорректные данные битвы в БД")

    judge_model_key = resolve_judge_model_key(payload.judge_model)
    logger.info(
        "winner: судейство битвы %d, Judge=%s",
        battle_id,
        judge_model_key,
    )

    try:
        decision = await judge_winner(
            results=results,
            session=request.app.state.http_session,
            judge_model=judge_model_key,
            judge_prompt=payload.judge_prompt,
        )
    except Exception as exc:
        logger.exception("winner: ошибка судейства битвы %d: %s", battle_id, exc)
        raise HTTPException(500, "Внутренняя ошибка при судействе") from exc

    battle.judge_model_name = judge_model_key

    if decision.get("winners"):
        battle.winner = decision["winners"][0]
        battle.message = decision.get("message", "")
        battle.judge_reason = decision.get("reason", "")
        battle.winner_position = decision.get("winner_position")
    else:
        battle.winner = None
        battle.message = decision.get("message", "Победитель не определён")
        battle.judge_reason = decision.get("reason", "")
        battle.winner_position = None

    await db.commit()
    logger.info(
        "winner: результат битвы %d сохранён, победитель: %s",
        battle_id,
        battle.winner,
    )

    decision["model_a_name"] = battle.model1
    decision["model_b_name"] = battle.model2
    decision["judge_model_name"] = judge_model_key

    return WinnerResponse(**decision)


@router.get("/history", response_model=list[BattleHistoryResponse])
async def get_history(db: AsyncSession = Depends(get_async_db)):
    statement = select(ArenaResult).order_by(ArenaResult.id.desc()).limit(10)
    result = await db.execute(statement)
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
        headers={
            "Content-Disposition": f"attachment; filename=battle_{battle_id}_result.md"
        },
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
        headers={"Content-Disposition": "attachment; filename=last_result.md"},
    )


@router.post("/battle", response_model=WinnerResponse)
@limiter.limit("5/minute")
async def full_battle(
        payload: FullBattleRequest,
        request: Request,
        db: AsyncSession = Depends(get_async_db),
):
    models = payload.models

    if len(models) != 2:
        raise HTTPException(400, "Нужно ровно две модели")

    judge_model_key = resolve_judge_model_key(payload.judge_model)
    logger.info("battle: models=%s judge=%s", models, judge_model_key)
    session = request.app.state.http_session

    try:
        compare_result = await run_arena_comparison(
            models=models,
            session=session,
            prompt=payload.prompt,
        )
    except Exception as exc:
        logger.exception("battle: ошибка генерации участников: %s", exc)
        compare_result = {
            "results": _fallback_error_evidence(
                models,
                f"Не удалось выполнить запрос к локальному Ollama: {type(exc).__name__}: {exc}",
            ),
            "elapsed": 0,
        }

    raw_results = compare_result.get("results") or _fallback_error_evidence(
        models,
        compare_result.get("error", "Ollama не вернул результаты генерации."),
    )
    results = normalize_evidence(raw_results)
    elapsed = compare_result.get("elapsed", 0)

    # Persist before judging. If one/both participants or the Judge fail, the
    # attempt and its error evidence remain available in history.
    battle = ArenaResult(
        model1=models[0],
        model2=models[1],
        winner=None,
        message="Ожидает решения судьи",
        evidence=results,
    )
    db.add(battle)
    await db.commit()
    await db.refresh(battle)

    try:
        decision_raw = await judge_winner(
            results=results,
            session=session,
            judge_model=judge_model_key,
            judge_prompt=payload.judge_prompt,
        )
    except Exception as exc:
        logger.exception("battle: ошибка Judge для battle id=%s: %s", battle.id, exc)
        error_message = f"Локальный Judge завершился с ошибкой: {type(exc).__name__}: {exc}"
        decision_raw = {
            "winners": [],
            "losers": [],
            "message": "❌ Judge не смог определить победителя.",
            "judge_result": {"winner": None, "reason": error_message},
            "judge_error": {"error": error_message},
            "reason": error_message,
            "evidence": results,
            "winner_position": None,
            "judge_model": judge_model_key,
        }

    decision = normalize_decision(decision_raw)
    # normalize_decision in the current project intentionally normalizes only
    # verdict fields; return the attempted model evidence as part of the API
    # response as well.
    decision["evidence"] = results

    battle.judge_model_name = judge_model_key
    battle.winner = decision["winners"][0] if decision["winners"] else None
    battle.message = decision.get("message", "")
    battle.judge_reason = decision.get("reason", "")
    battle.winner_position = decision.get("winner_position")
    await db.commit()

    decision.update(
        {
            "arena_result_id": battle.id,
            "elapsed": elapsed,
            "model_a_name": battle.model1,
            "model_b_name": battle.model2,
            "judge_model_name": judge_model_key,
        }
    )
    return WinnerResponse(**decision)
