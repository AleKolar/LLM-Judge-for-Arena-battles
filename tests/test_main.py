# tests/test_main.py
"""
Тесты приложения LLM Arena после перехода на локальные модели Ollama.

Внешние вызовы к Ollama не выполняются: в тестах AI-ответы подменяются моками.
"""

import logging
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from main import app, lifespan
from src.database.database import get_async_db
from src.schemas.schemas import (
    ArenaCompareResponse,
    ArenaResultResponse,
    CompareRequest,
    WinnerRequest,
)
from src.services.ai_service import (
    DEFAULT_JUDGE,
    judge_winner,
)
from src.services.arena_result import (
    get_battle_by_id,
    get_last_result_service,
)
from src.services.download_service import generate_battle_markdown
from src.utils.normalize import normalize_evidence, to_md


# Ключи моделей из текущего AVAILABLE_MODELS (Ollama).
VALID_MODEL_A = "qwen3-4b"
VALID_MODEL_B = "qwen3-8b"
VALID_JUDGE = DEFAULT_JUDGE


@pytest.fixture
def client():
    """TestClient с изолированным mock dependency для БД."""

    def override_get_async_db():
        session = AsyncMock()
        session.execute = AsyncMock()
        session.commit = AsyncMock()
        session.refresh = AsyncMock()
        session.add = MagicMock()
        yield session

    app.dependency_overrides[get_async_db] = override_get_async_db
    try:
        with TestClient(app) as test_client:
            yield test_client
    finally:
        app.dependency_overrides.clear()


def _mock_llm_response(model: str, content: str, status: str = "success") -> dict:
    return {"model": model, "content": content, "status": status}


# ---------------------------------------------------------------------
# Application and model configuration
# ---------------------------------------------------------------------

def test_main_page(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "LLM Arena" in response.text


def test_list_models(client):
    response = client.get("/api/llm-arena/models")
    assert response.status_code == 200

    models = response.json()
    assert VALID_MODEL_A in models
    assert VALID_MODEL_B in models


def test_list_judge_models(client):
    response = client.get("/api/llm-arena/judge_models")
    assert response.status_code == 200

    models = response.json()
    assert VALID_JUDGE in models


# ---------------------------------------------------------------------
# Normalization utilities
# ---------------------------------------------------------------------

def test_normalize_evidence_dict():
    data = [{"model": "qwen3-4b", "content": "code", "status": "success"}]
    result = normalize_evidence(data)

    assert result[0]["model"] == "qwen3-4b"
    assert result[0]["content"] == "code"
    assert result[0]["status"] == "success"


def test_normalize_evidence_string():
    result = normalize_evidence(["hello"])

    assert result[0]["model"] == "unknown"
    assert "hello" in result[0]["content"]


def test_to_md():
    data = [{"model": "qwen3-4b", "content": "print(1)", "status": "success"}]
    markdown = to_md(data)

    assert "qwen3-4b" in markdown
    assert "```python" in markdown


# ---------------------------------------------------------------------
# Database services
# ---------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_last_result_service():
    mock_result = MagicMock()
    mock_result.scalars.return_value.first.return_value = "result"

    db = AsyncMock()
    db.execute.return_value = mock_result

    result = await get_last_result_service(db)
    assert result == "result"


@pytest.mark.asyncio
async def test_get_last_result_service_empty():
    mock_result = MagicMock()
    mock_result.scalars.return_value.first.return_value = None

    db = AsyncMock()
    db.execute.return_value = mock_result

    result = await get_last_result_service(db)
    assert result is None


@pytest.mark.asyncio
async def test_get_battle_by_id_not_found():
    mock_result = MagicMock()
    mock_result.scalars.return_value.first.return_value = None

    db = AsyncMock()
    db.execute.return_value = mock_result

    result = await get_battle_by_id(db, 999)
    assert result is None


# ---------------------------------------------------------------------
# Judge business logic
# ---------------------------------------------------------------------

@pytest.mark.asyncio
async def test_judge_winner_both_failed():
    """Если обе локальные модели недоступны, победитель не назначается."""
    results = [
        _mock_llm_response(VALID_MODEL_A, "Ollama error", "error"),
        _mock_llm_response(VALID_MODEL_B, "Ollama error", "error"),
    ]

    result = await judge_winner(results, AsyncMock())

    assert result["winners"] == []
    assert result["winner_position"] is None
    assert result["judge_model"] == VALID_JUDGE


@pytest.mark.asyncio
async def test_judge_winner_one_success_model_a():
    """Если ответила только первая модель, срабатывает технический fallback."""
    results = [
        _mock_llm_response(VALID_MODEL_A, "answer A"),
        _mock_llm_response(VALID_MODEL_B, "Ollama unavailable", "error"),
    ]

    result = await judge_winner(results, AsyncMock(), judge_model=VALID_JUDGE)

    assert result["winners"] == [VALID_MODEL_A]
    assert result["losers"] == [VALID_MODEL_B]
    assert result["winner_position"] == "MODEL_A"


@pytest.mark.asyncio
async def test_judge_winner_one_success_model_b():
    """Fallback также работает, если доступна только вторая модель."""
    results = [
        _mock_llm_response(VALID_MODEL_A, "Ollama unavailable", "error"),
        _mock_llm_response(VALID_MODEL_B, "answer B"),
    ]

    result = await judge_winner(results, AsyncMock(), judge_model=VALID_JUDGE)

    assert result["winners"] == [VALID_MODEL_B]
    assert result["losers"] == [VALID_MODEL_A]
    assert result["winner_position"] == "MODEL_B"


@pytest.mark.asyncio
async def test_judge_winner_with_judge_mock():
    """При двух успешных ответах вызывается Judge на Ollama."""
    results = [
        _mock_llm_response(VALID_MODEL_A, "code A"),
        _mock_llm_response(VALID_MODEL_B, "code B"),
    ]

    with patch(
        "src.services.ai_service.ask_judge", new_callable=AsyncMock
    ) as mock_ask_judge:
        mock_ask_judge.return_value = {
            "winner": "MODEL_A",
            "reason": "Better implementation",
        }

        result = await judge_winner(
            results,
            AsyncMock(),
            judge_model=VALID_JUDGE,
        )

    assert result["winners"] == [VALID_MODEL_A]
    assert result["winner_position"] == "MODEL_A"
    mock_ask_judge.assert_awaited_once()


@pytest.mark.asyncio
async def test_judge_winner_draw():
    results = [
        _mock_llm_response(VALID_MODEL_A, "same answer"),
        _mock_llm_response(VALID_MODEL_B, "same answer"),
    ]

    with patch(
        "src.services.ai_service.ask_judge", new_callable=AsyncMock
    ) as mock_ask_judge:
        mock_ask_judge.return_value = {
            "winner": "DRAW",
            "reason": "Both answers are equivalent",
        }

        result = await judge_winner(
            results,
            AsyncMock(),
            judge_model=VALID_JUDGE,
        )

    assert result["winners"] == []
    assert result["losers"] == []
    assert result["winner_position"] is None


@pytest.mark.asyncio
async def test_judge_winner_judge_error():
    results = [
        _mock_llm_response(VALID_MODEL_A, "code A"),
        _mock_llm_response(VALID_MODEL_B, "code B"),
    ]

    with patch(
        "src.services.ai_service.ask_judge", new_callable=AsyncMock
    ) as mock_ask_judge:
        mock_ask_judge.return_value = {
            "error": "Judge returned invalid winner",
            "raw_response": '{"winner": "MODEL_C", "reason": "invalid"}',
        }

        result = await judge_winner(
            results,
            AsyncMock(),
            judge_model=VALID_JUDGE,
        )

    assert result["winners"] == []
    assert result["winner_position"] is None
    assert result.get("judge_error") is not None


@pytest.mark.asyncio
async def test_judge_winner_passes_custom_prompt():
    results = [
        _mock_llm_response(VALID_MODEL_A, "code A"),
        _mock_llm_response(VALID_MODEL_B, "code B"),
    ]
    custom_prompt = "CUSTOM {model_a_name} {response_a} {model_b_name} {response_b}"

    with patch(
        "src.services.ai_service.ask_judge", new_callable=AsyncMock
    ) as mock_ask_judge:
        mock_ask_judge.return_value = {
            "winner": "MODEL_A",
            "reason": "ok",
        }

        result = await judge_winner(
            results,
            AsyncMock(),
            judge_model=VALID_JUDGE,
            judge_prompt=custom_prompt,
        )

    mock_ask_judge.assert_awaited_once()
    assert mock_ask_judge.call_args.kwargs["judge_prompt"] == custom_prompt
    assert result["winners"] == [VALID_MODEL_A]


# ---------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------

def test_schemas_compare_request():
    obj = CompareRequest(models=[VALID_MODEL_A, VALID_MODEL_B], prompt="test")
    assert obj.models == [VALID_MODEL_A, VALID_MODEL_B]
    assert obj.prompt == "test"


def test_schemas_winner_request():
    obj = WinnerRequest(results=[{"model": VALID_MODEL_A}])
    assert obj.results[0]["model"] == VALID_MODEL_A


def test_schemas_arena_response():
    obj = ArenaResultResponse(
        id=1,
        model1=VALID_MODEL_A,
        model2=VALID_MODEL_B,
        winner=None,
        message="ok",
        evidence=[],
        created_at=datetime.now(),
    )
    assert obj.model1 == VALID_MODEL_A
    assert obj.model2 == VALID_MODEL_B


def test_schemas_arena_compare_response():
    obj = ArenaCompareResponse(
        arena_result_id=1,
        results=[_mock_llm_response(VALID_MODEL_A, "code")],
        elapsed=1.5,
    )
    assert obj.arena_result_id == 1
    assert len(obj.results) == 1
    assert obj.elapsed == 1.5


# ---------------------------------------------------------------------
# Markdown export
# ---------------------------------------------------------------------

def test_generate_battle_markdown_same_models():
    battle = MagicMock()
    battle.model1 = VALID_MODEL_A
    battle.model2 = VALID_MODEL_A
    battle.winner = VALID_MODEL_A
    battle.winner_position = "MODEL_A"
    battle.message = "Winner"
    battle.judge_reason = "Better solution"
    battle.evidence = [_mock_llm_response(VALID_MODEL_A, "code")]

    markdown = generate_battle_markdown(battle)

    assert "Модель A" in markdown
    assert "✅" in markdown
    assert "❌" in markdown


def test_generate_battle_markdown_draw():
    battle = MagicMock()
    battle.model1 = VALID_MODEL_A
    battle.model2 = VALID_MODEL_B
    battle.winner = None
    battle.winner_position = None
    battle.message = "Ничья!"
    battle.judge_reason = "Обе реализации равноценны"
    battle.evidence = [
        _mock_llm_response(VALID_MODEL_A, "code A"),
        _mock_llm_response(VALID_MODEL_B, "code B"),
    ]

    markdown = generate_battle_markdown(battle)

    assert "🤝" in markdown
    assert "Ничья" in markdown


# ---------------------------------------------------------------------
# Application lifespan (Ollama, not Hugging Face)
# ---------------------------------------------------------------------

@pytest.mark.asyncio
async def test_lifespan_starts_and_closes_session():
    """Приложение создаёт и закрывает HTTP-сессию; БД проверяется отдельно."""
    with (
        patch("main.aiohttp.ClientSession") as mock_session_cls,
        patch("main.async_engine") as mock_engine,
    ):
        mock_session = MagicMock()
        mock_session.close = AsyncMock()
        mock_session_cls.return_value = mock_session

        mock_conn = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar.return_value = 1
        mock_conn.execute.return_value = mock_result

        connection_context = MagicMock()
        connection_context.__aenter__ = AsyncMock(return_value=mock_conn)
        connection_context.__aexit__ = AsyncMock(return_value=None)
        mock_engine.connect.return_value = connection_context

        async with lifespan(MagicMock()):
            pass

        mock_session_cls.assert_called_once()
        mock_session.close.assert_awaited_once()
        mock_engine.connect.assert_called_once()
