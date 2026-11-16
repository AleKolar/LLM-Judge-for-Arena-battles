import logging
from contextlib import asynccontextmanager
from pathlib import Path

import aiohttp
from fastapi import FastAPI
from fastapi.templating import Jinja2Templates
from sqlalchemy import text
from starlette.requests import Request
from starlette.responses import HTMLResponse
from starlette.staticfiles import StaticFiles

from src.database.database import async_engine
from src.routers import llm_arena
from src.routers.evals import EvaluationRunManager, router as evals_router
from src.services.ai_service import OLLAMA_BASE_URL

logger = logging.getLogger("uvicorn")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # DeepEval subprocess manager is scoped to this FastAPI worker.
    app.state.evaluation_run_manager = EvaluationRunManager(Path(__file__).resolve().parent)

    # One shared aiohttp session is used by the Arena router for local Ollama calls.
    app.state.http_session = aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=30),
        headers={"User-Agent": "LLM Arena/3.0"},
    )

    # Check the local Ollama service instead of checking the obsolete
    # Hugging Face token. The application still starts if Ollama is offline,
    # so Arena can report the failed model attempts to the user.
    ollama_tags_url = f"{OLLAMA_BASE_URL}/api/tags"
    try:
        async with app.state.http_session.get(ollama_tags_url) as response:
            if response.status == 200:
                data = await response.json()
                model_names = [item.get("name", "unknown") for item in data.get("models", [])]
                if model_names:
                    logger.info(
                        "✅ Ollama доступен. Локальные модели: %s",
                        ", ".join(model_names),
                    )
                else:
                    logger.warning(
                        "⚠️ Ollama доступен, но локальные модели не найдены. "
                        "Загрузите модели через `ollama pull`."
                    )
            else:
                body = (await response.text())[:500]
                logger.warning(
                    "⚠️ Ollama вернул HTTP %s при проверке доступности: %s",
                    response.status,
                    body,
                )
    except (aiohttp.ClientError, TimeoutError) as exc:
        logger.warning(
            "⚠️ Не удалось подключиться к Ollama по адресу %s: %s",
            OLLAMA_BASE_URL,
            exc,
        )

    # Database health check remains unchanged.
    try:
        async with async_engine.connect() as connection:
            result = await connection.execute(text("SELECT 1"))
            logger.info("✅ Соединение с БД установлено: %s", result.scalar())
    except Exception as exc:
        logger.warning("⚠️ Не удалось подключиться к БД: %s", exc)

    try:
        yield
    finally:
        await app.state.evaluation_run_manager.shutdown()
        await app.state.http_session.close()


app = FastAPI(title="LLM Arena 🧠⚖️", version="3.1.0", lifespan=lifespan)

templates = Jinja2Templates(directory="src/templates")
app.mount("/static", StaticFiles(directory="src/static"), name="static")
app.include_router(llm_arena.router)
app.include_router(evals_router)


# @app.get("/", response_class=HTMLResponse)
# async def index(request: Request):
#     return templates.TemplateResponse(request=request, name="index.html")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index_ollama.html",
    )


# uvicorn main:app --port 8000
