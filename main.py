from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import aiohttp
import logging

from sqlalchemy import text
from starlette.staticfiles import StaticFiles

from src.database.database import async_engine
from src.routers import leap_year, llm_arena
from src.services.ai_service import API_KEY

logger = logging.getLogger("uvicorn")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Создаём aiohttp-сессию
    app.state.http_session = aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=30),
        headers={"User-Agent": "LeapYearDetective/3.0"}
    )
    # ── Блок проверки AI-сервиса ──
    if not API_KEY:
        logger.warning("⚠️ OPENROUTER_API_KEY не задан. LLM Arena будет недоступна.")
    else:
        # Проверим доступность API
        try:
            # Проверим доступность эндпоинта OpenRouter
            async with app.state.http_session.get(
                "https://openrouter.ai/api/v1/auth/key",
                headers={"Authorization": f"Bearer {API_KEY}"}
            ) as resp:
                if resp.status == 200:
                    logger.info("✅ OpenRouter API доступен")
                else:
                    logger.warning(f"⚠️ OpenRouter вернул статус {resp.status}")
        except Exception as e:
            logger.warning(f"⚠️ Не удалось проверить OpenRouter: {e}")
    try:
        async with async_engine.connect() as conn:
            result = await conn.execute(text("SELECT 1"))
            logger.info(f"✅ Соединение с БД установлено: {result.scalar()}")
    except Exception as e:
        logger.warning(f"⚠️ Не удалось подключиться к БД: {e}")

    yield

    await app.state.http_session.close()

app = FastAPI(title="Leap Year Detective 🕵️", version="3.0.0", lifespan=lifespan)

templates = Jinja2Templates(directory="src/templates")

app.mount("/static", StaticFiles(directory="src/static"), name="static")

# Подключаем роутеры
app.include_router(leap_year.router)
app.include_router(llm_arena.router)

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")

# uvicorn main:app --port 8000
