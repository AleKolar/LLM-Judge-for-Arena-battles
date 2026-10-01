from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import aiohttp
import logging

from sqlalchemy import text
from starlette.staticfiles import StaticFiles

from src.database.database import async_engine
from src.routers import llm_arena
from src.services.ai_service import HUGGINGFACE_API_KEY

logger = logging.getLogger("uvicorn")

# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     # Создаём aiohttp-сессию
#     app.state.http_session = aiohttp.ClientSession(
#         timeout=aiohttp.ClientTimeout(total=30),
#         headers={"User-Agent": "LLM Arena/3.0"}
#     )
#     # ── Блок проверки AI-сервиса ──
#     if not HUGGINGFACE_API_KEY:
#         logger.warning("⚠️ HUGGINGFACE_API_KEY не задан. LLM Arena будет работать в ограниченном режиме.")
#     else:
#         # Проверим доступность API
#         try:
#             async with app.state.http_session.get(
#                 "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3",
#                 headers={"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}
#             ) as resp:
#                 if resp.status == 200:
#                     logger.info("✅ Hugging Face API доступен")
#                 else:
#                     logger.warning(f"⚠️ Hugging Face вернул статус {resp.status}")
#         except Exception as e:
#             logger.warning(f"⚠️ Не удалось проверить Hugging Face: {e}")
#
#
#
#     try:
#         async with async_engine.connect() as conn:
#             result = await conn.execute(text("SELECT 1"))
#             logger.info(f"✅ Соединение с БД установлено: {result.scalar()}")
#     except Exception as e:
#         logger.warning(f"⚠️ Не удалось подключиться к БД: {e}")
#
#     yield
#
#     await app.state.http_session.close()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Создаём aiohttp-сессию
    app.state.http_session = aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=30),
        headers={"User-Agent": "LLM Arena/3.0"}
    )

    # ── Блок проверки AI-сервиса ──
    if not HUGGINGFACE_API_KEY:
        logger.warning("⚠️ HUGGINGFACE_API_KEY не задан. LLM Arena будет работать в ограниченном режиме.")
    else:
        try:
            async with app.state.http_session.get(
                "https://huggingface.co/api/whoami-v2",
                headers={"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"},
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    logger.info(f"✅ Hugging Face API доступен, токен валиден (user: {data.get('name')})")
                elif resp.status == 401:
                    logger.warning("⚠️ Hugging Face API доступен, но токен невалиден (401)")
                else:
                    logger.warning(f"⚠️ Hugging Face вернул статус {resp.status}")
        except Exception as e:
            logger.warning(f"⚠️ Не удалось проверить Hugging Face: {e}")

    # ── Блок проверки БД ──
    try:
        async with async_engine.connect() as conn:
            result = await conn.execute(text("SELECT 1"))
            logger.info(f"✅ Соединение с БД установлено: {result.scalar()}")
    except Exception as e:
        logger.warning(f"⚠️ Не удалось подключиться к БД: {e}")

    yield

    await app.state.http_session.close()




# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     app.state.http_session = aiohttp.ClientSession(
#         timeout=aiohttp.ClientTimeout(total=30),
#         headers={"User-Agent": "LLM Arena/3.0"}
#     )
#
#     # ── Блок проверки AI-сервиса ──
#     if not HUGGINGFACE_API_KEY:
#         logger.warning("⚠️ HUGGINGFACE_API_KEY не задан. LLM Arena будет работать в ограниченном режиме.")
#     else:
#         try:
#             async with app.state.http_session.get(
#                 "https://huggingface.co/api/whoami-v2",
#                 headers={"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}
#             ) as resp:
#                 if resp.status == 200:
#                     data = await resp.json()
#                     logger.info(f"✅ HF API доступен, токен валиден (user: {data.get('name')})")
#                 elif resp.status == 401:
#                     logger.warning("⚠️ HF API доступен, но токен невалиден (401)")
#                 else:
#                     logger.warning(f"⚠️ HF вернул статус {resp.status}")
#         except Exception as e:
#             logger.warning(f"⚠️ Не удалось проверить Hugging Face: {e}")




app = FastAPI(title="LLM Arena 🧠⚖️", version="3.0.0", lifespan=lifespan)

templates = Jinja2Templates(directory="src/templates")

app.mount("/static", StaticFiles(directory="src/static"), name="static")

# Подключаем роутеры
app.include_router(llm_arena.router)

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")

# uvicorn main:app --port 8000
