# src/routers/llm_arena.py

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.database.database import get_async_db
from src.models.db_models import ArenaResult
from src.models.models import BattleHistoryResponse, CompareRequest, WinnerResponse
from src.services.ai_service import (
    AVAILABLE_MODELS,
    DEFAULT_MODELS,
    judge_winner,
    run_arena_comparison,
)
from src.services.arena_result import get_last_result_service, get_battle_by_id
from src.utils.normalize import normalize_evidence, to_md
from src.utils.prettify_model_name import prettify_model_name

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
    """
    Запускает две модели, сохраняет evidence в БД,
    возвращает arena_result_id и ответы моделей (без судьи).
    """
    session = request.app.state.http_session
    models = payload.models or DEFAULT_MODELS
    if len(models) < 2:
        raise HTTPException(400, "Нужно выбрать две модели")

    result = await run_arena_comparison(models, session, payload.prompt)

    battle = ArenaResult(
        model1=models[0],
        model2=models[1],
        winner=None,                # будет определён позже
        message="Ожидает решения судьи",
        evidence=result.get("results", [])
    )
    db.add(battle)
    await db.commit()
    await db.refresh(battle)

    # Возвращаем ID битвы и ответы моделей
    return {
        "arena_result_id": battle.id,
        "results": result.get("results", []),
        "elapsed": result.get("elapsed", 0),
    }


@router.post("/winner/{battle_id}", response_model=WinnerResponse)
async def declare_winner(
    battle_id: int,
    request: Request,
    db: AsyncSession = Depends(get_async_db)
):
    """
    Принимает ID битвы, вызывает судью и обновляет запись в БД.
    """
    battle = await get_battle_by_id(db, battle_id)
    if not battle:
        raise HTTPException(404, f"Битва с id={battle_id} не найдена")

    results = battle.evidence
    if not isinstance(results, list) or len(results) < 2:
        raise HTTPException(400, "Некорректные данные битвы в БД")

    decision = await judge_winner(results, request.app.state.http_session)

    if decision.get("winners"):
        battle.winner = decision["winners"][0]
        battle.message = decision["message"]
    else:
        battle.winner = None
        battle.message = decision["message"]
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

    evidence = normalize_evidence(last_battle.evidence)
    evidence_md = to_md(evidence)
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
        headers={"Content-Disposition": "attachment; filename=last_result.md"}
    )

@router.get("/download-result/{battle_id}")
async def download_result(
    battle_id: int,
    db: AsyncSession = Depends(get_async_db)
):
    """Скачать результат конкретной битвы в формате Markdown."""
    battle = await get_battle_by_id(db, battle_id)
    if not battle:
        raise HTTPException(404, f"Битва с id={battle_id} не найдена")

    evidence = normalize_evidence(battle.evidence)
    evidence_md = to_md(evidence)

    # Определяем победителя и комментарий
    winner_text = "Ничья" if not battle.winner else prettify_model_name(battle.winner)
    verdict = battle.message or "Результат не определён"

    markdown = (
        f"# 🧠 LLM Arena — Результат битвы\n\n"
        f"## 🏆 Победитель\n{winner_text}\n\n"
        f"## ⚖️ Вердикт Судьи\n{verdict}\n\n"
        f"---\n\n"
        f"## 📊 Комментарии к результату\n\n"
        f"| Модель | Статус |\n"
        f"|--------|--------|\n"
        f"| {battle.model1} | {'✅' if any(e.get('model') == battle.model1 and e.get('status') == 'success' for e in evidence) else '❌'} |\n"
        f"| {battle.model2} | {'✅' if any(e.get('model') == battle.model2 and e.get('status') == 'success' for e in evidence) else '❌'} |\n\n"
        f"---\n\n"
        f"## 📝 Код моделей\n\n"
        f"{evidence_md}"
    )

    return Response(
        content=markdown,
        media_type="text/markdown",
        headers={
            "Content-Disposition": f"attachment; filename=battle_{battle_id}_result.md"
        }
    )


@router.get("/last-result")
async def get_last_result(db: AsyncSession = Depends(get_async_db)):
    """Последняя битва с русскими заголовками (скачать)."""
    last_battle = await get_last_result_service(db)
    if not last_battle:
        raise HTTPException(404, "История битв пуста")

    evidence = normalize_evidence(last_battle.evidence)
    evidence_md = to_md(evidence)

    winner_text = "Ничья" if not last_battle.winner else prettify_model_name(last_battle.winner)
    verdict = last_battle.message or "Результат не определён"

    markdown = (
        f"# 🧠 LLM Arena — Результат битвы\n\n"
        f"## 🏆 Победитель\n{winner_text}\n\n"
        f"## ⚖️ Вердикт Судьи\n{verdict}\n\n"
        f"---\n\n"
        f"## 📊 Комментарии к результату\n\n"
        f"| Модель | Статус |\n"
        f"|--------|--------|\n"
        f"| {last_battle.model1} | {'✅' if any(e.get('model') == last_battle.model1 and e.get('status') == 'success' for e in evidence) else '❌'} |\n"
        f"| {last_battle.model2} | {'✅' if any(e.get('model') == last_battle.model2 and e.get('status') == 'success' for e in evidence) else '❌'} |\n\n"
        f"---\n\n"
        f"## 📝 Код моделей\n\n"
        f"{evidence_md}"
    )

    return Response(
        content=markdown,
        media_type="text/markdown",
        headers={
            "Content-Disposition": "attachment; filename=last_result.md"
        }
    )