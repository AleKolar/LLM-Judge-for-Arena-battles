# src/tests/test_main.py
import logging
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from main import app, lifespan
from src.database.database import get_async_db
from src.schemas.schemas import CompareRequest, WinnerRequest, ArenaResultResponse, ArenaCompareResponse
from src.services.ai_service import judge_winner
from src.services.arena_result import get_last_result_service, get_battle_by_id
from src.services.download_service import generate_battle_markdown
from src.utils.normalize import normalize_evidence, to_md


# =========================
# FIXTURE (PRODUCTION STYLE)
# =========================

@pytest.fixture(scope="function")
def client():
    """
    TestClient + isolated dependency override for the database session.
    """
    def override_get_async_db():
        session = AsyncMock()
        session.execute = AsyncMock()
        session.commit = AsyncMock()
        session.refresh = AsyncMock()
        session.add = MagicMock()
        yield session

    app.dependency_overrides[get_async_db] = override_get_async_db

    with TestClient(app) as c:
        yield c

    app.dependency_overrides.clear()


def test_main_page(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert "LLM Arena" in resp.text


# =========================
# LLM ARENA (PRODUCTION MOCK LAYER)
# =========================

def _mock_llm_response(model: str, content: str) -> dict:
    return {
        "model": model,
        "content": content,
        "status": "success",
    }


@pytest.mark.asyncio
async def test_compare_models(client):
    """
    Тест эндпоинта /compare: только генерация, без судьи.
    """
    with patch(
        "src.services.ai_service.fetch_from_model",
        new_callable=AsyncMock
    ) as mock_fetch:
        mock_fetch.side_effect = [
            # MODEL_A
            _mock_llm_response(
                "openai/gpt-4o-mini",
                "def is_leap(y): return y % 400 == 0 or (y % 4 == 0 and y % 100 != 0)"
            ),
            # MODEL_B
            _mock_llm_response(
                "deepseek/deepseek-chat",
                "def is_leap(y): return y % 4 == 0"
            ),
        ]

        resp = client.post(
            "/api/llm-arena/compare",
            json={"models": ["gpt-4o-mini", "deepseek-chat"]},
        )

        assert resp.status_code == 200
        data = resp.json()

        assert "arena_result_id" in data
        assert "results" in data
        assert "elapsed" in data
        assert "judge" not in data  # судья больше не вызывается здесь

        assert len(data["results"]) == 2
        assert mock_fetch.call_count == 2


def test_list_models(client):
    resp = client.get("/api/llm-arena/models")
    assert resp.status_code == 200
    assert "deepseek-chat" in resp.json()


@pytest.mark.asyncio
async def test_winner_endpoint(client):
    """Тест эндпоинта /winner/{battle_id}."""
    battle_id = 42

    mock_battle = MagicMock()
    mock_battle.id = battle_id
    mock_battle.model1 = "gpt-4o-mini"
    mock_battle.model2 = "deepseek-chat"
    mock_battle.evidence = [
        _mock_llm_response("openai/gpt-4o-mini", "code A"),
        _mock_llm_response("deepseek-chat", "code B"),
    ]
    mock_battle.winner = None
    mock_battle.message = ""
    mock_battle.judge_model_name = None   # новое поле

    with patch(
        "src.services.ai_service.ask_judge", new_callable=AsyncMock
    ) as mock_ask_judge:
        mock_ask_judge.return_value = {
            "winner": "MODEL_A",
            "reason": "MODEL_A implementation is more correct",
        }

        with patch(
            "src.routers.llm_arena.get_battle_by_id", new_callable=AsyncMock
        ) as mock_get_battle:
            mock_get_battle.return_value = mock_battle

            # Отправляем тело с выбором судьи
            resp = client.post(
                f"/api/llm-arena/winner/{battle_id}",
                json={"judge_model": "deepseek-chat"}
            )

            assert resp.status_code == 200
            data = resp.json()
            assert data["winners"] == ["openai/gpt-4o-mini"]
            assert data["losers"] == ["deepseek-chat"]
            assert "Победитель" in data["message"]
            assert "more correct" in data["reason"]
            assert data["model_a_name"] == "gpt-4o-mini"
            assert data["model_b_name"] == "deepseek-chat"
            assert data["judge_model_name"] == "deepseek-chat"


def test_winner_not_found(client):
    """Проверка 404, если битва не найдена."""
    with patch(
        "src.routers.llm_arena.get_battle_by_id",
        new_callable=AsyncMock
    ) as mock_get_battle:
        mock_get_battle.return_value = None

        resp = client.post(
            "/api/llm-arena/winner/999",
            json={"judge_model": "deepseek-chat"}
        )
        assert resp.status_code == 404


# =========================
# UTILS TESTS
# =========================

def test_normalize_evidence_dict():
    data = [{"model": "gpt", "content": "code", "status": "success"}]
    result = normalize_evidence(data)
    assert result[0]["model"] == "gpt"
    assert result[0]["content"] == "code"
    assert result[0]["status"] == "success"


def test_normalize_evidence_string():
    data = ["hello"]
    result = normalize_evidence(data)
    assert result[0]["model"] == "unknown"
    assert "hello" in result[0]["content"]


def test_to_md():
    data = [{"model": "gpt", "content": "print(1)", "status": "success"}]
    md = to_md(data)
    assert "gpt" in md
    assert "```python" in md


# =========================
# SERVICES (UNIT LEVEL)
# =========================

@pytest.mark.asyncio
async def test_get_last_result_service():
    mock_result = MagicMock()
    mock_result.scalars.return_value.first.return_value = "result"
    db = AsyncMock()
    db.execute.return_value = mock_result
    result = await get_last_result_service(db)
    assert result == "result"


def test_last_result(client):
    mock_battle = MagicMock()
    mock_battle.winner = None
    mock_battle.evidence = [
        {"model": "openai/gpt-4o-mini", "content": "code", "status": "success"}
    ]
    mock_battle.message = "ok"
    mock_battle.judge_reason = "reason"
    mock_battle.model1 = "gpt-4o-mini"
    mock_battle.model2 = "gpt-4o-mini"

    mock_scalars = MagicMock()
    mock_scalars.first.return_value = mock_battle
    mock_result = MagicMock()
    mock_result.scalars.return_value = mock_scalars

    async def override_db():
        session = AsyncMock()
        session.execute = AsyncMock(return_value=mock_result)
        session.commit = AsyncMock()
        session.refresh = AsyncMock()
        session.add = MagicMock()
        yield session

    app.dependency_overrides[get_async_db] = override_db

    try:
        resp = client.get("/api/llm-arena/last-result")
        assert resp.status_code == 200
        content = resp.text
        assert "🧠 LLM Arena — Результат битвы" in content
    finally:
        app.dependency_overrides.clear()


# =========================
# JUDGE LOGIC (CRITICAL AI EVAL PART)
# =========================

@pytest.mark.asyncio
async def test_judge_winner_simple():
    results = [
        {"model": "A", "content": "ok", "status": "success"},
        {"model": "B", "content": "error", "status": "error"},
    ]
    session = AsyncMock()
    res = await judge_winner(results, session)
    assert "winners" in res
    assert "losers" in res
    assert "message" in res


@pytest.mark.asyncio
async def test_judge_winner_with_judge_mock():
    results = [
        {"model": "openai/gpt-4o-mini", "content": "code A", "status": "success"},
        {"model": "deepseek-chat", "content": "code B", "status": "success"},
    ]
    session = AsyncMock()

    with patch("src.services.ai_service.ask_judge", new_callable=AsyncMock) as mock_ask:
        mock_ask.return_value = {
            "winner": "MODEL_A",
            "reason": "better structure",
        }
        res = await judge_winner(results, session)
        assert len(res["winners"]) == 1
        assert len(res["losers"]) >= 1
        assert "Победитель" in res["message"]
        mock_ask.assert_called_once()

@pytest.mark.asyncio
async def test_ask_judge_success():
    """Тестируем ask_judge с успешным ответом судьи."""
    from src.services.ai_service import ask_judge

    with patch("src.services.ai_service.fetch_from_model", new_callable=AsyncMock) as mock_fetch:
        mock_fetch.return_value = {
            "model": "judge-model",
            "content": '{"winner": "MODEL_A", "reason": "Better code"}',
            "status": "success",
        }
        session = AsyncMock()
        result = await ask_judge(session, "ModelA", "code A", "ModelB", "code B")
        assert result["winner"] == "MODEL_A"
        assert result["reason"] == "Better code"
        mock_fetch.assert_called_once()


# =========================
# SCHEMAS
# =========================

def test_schemas_compare_request():
    obj = CompareRequest(models=["gpt"], prompt="test")
    assert obj.models == ["gpt"]
    assert obj.prompt == "test"


def test_schemas_winner_request():
    obj = WinnerRequest(results=[{"model": "A"}])
    assert obj.results[0]["model"] == "A"


def test_schemas_arena_response():
    obj = ArenaResultResponse(
        id=1,
        model1="a",
        model2="b",
        winner=None,
        message="ok",
        evidence=[],
        created_at=datetime.now(),
    )
    assert obj.model1 == "a"
    assert obj.model2 == "b"


def test_schemas_arena_compare_response():
    obj = ArenaCompareResponse(
        arena_result_id=1,
        results=[{"model": "gpt", "content": "code", "status": "success"}],
        elapsed=1.5,
    )
    assert obj.arena_result_id == 1
    assert len(obj.results) == 1
    assert obj.elapsed == 1.5

# =========================
# LIFESPAN TESTS
# =========================

@pytest.mark.asyncio
async def test_lifespan_no_api_key(caplog):
    """API_KEY не задан – лог предупреждения, сессия создана, БД проверена."""
    with caplog.at_level(logging.WARNING):
        with patch("main.API_KEY", None), \
             patch("main.async_engine") as mock_engine, \
             patch("main.aiohttp.ClientSession") as mock_session_cls:

            mock_session = MagicMock()
            mock_session.close = AsyncMock()
            mock_session_cls.return_value = mock_session

            # БД
            mock_conn = AsyncMock()
            mock_result = MagicMock()
            mock_result.scalar.return_value = 1
            mock_conn.execute.return_value = mock_result
            mock_engine.connect.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
            mock_engine.connect.return_value.__aexit__ = AsyncMock(return_value=None)

            app = MagicMock()

            async with lifespan(app):
                pass

            mock_session_cls.assert_called_once()
            mock_session.close.assert_awaited_once()
            mock_engine.connect.assert_called_once()
            assert "OPENROUTER_API_KEY не задан" in caplog.text


@pytest.mark.asyncio
async def test_lifespan_api_key_success(caplog):
    """API_KEY задан, OpenRouter отвечает 200, БД доступна."""
    with caplog.at_level(logging.INFO):
        with patch("main.API_KEY", "test-key"), \
             patch("main.async_engine") as mock_engine, \
             patch("main.aiohttp.ClientSession") as mock_session_cls:

            mock_session = MagicMock()
            mock_session.close = AsyncMock()
            mock_session_cls.return_value = mock_session

            # OpenRouter
            resp_mock = MagicMock()
            resp_mock.status = 200
            get_context = MagicMock()
            get_context.__aenter__ = AsyncMock(return_value=resp_mock)
            get_context.__aexit__ = AsyncMock(return_value=None)
            mock_session.get.return_value = get_context

            # БД
            mock_conn = AsyncMock()
            mock_result = MagicMock()
            mock_result.scalar.return_value = 1
            mock_conn.execute.return_value = mock_result
            mock_engine.connect.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
            mock_engine.connect.return_value.__aexit__ = AsyncMock(return_value=None)

            app = MagicMock()

            async with lifespan(app):
                pass

            mock_session.get.assert_called_with(
                "https://openrouter.ai/api/v1/auth/key",
                headers={"Authorization": "Bearer test-key"}
            )
            mock_session.close.assert_awaited_once()
            assert "✅ OpenRouter API доступен" in caplog.text
            assert "✅ Соединение с БД установлено" in caplog.text


@pytest.mark.asyncio
async def test_lifespan_openrouter_failure(caplog):
    """OpenRouter возвращает 500 – предупреждение + БД работает."""
    with caplog.at_level(logging.INFO):   # <-- ВАЖНО: INFO, чтобы видеть сообщение БД
        with patch("main.API_KEY", "test-key"), \
             patch("main.async_engine") as mock_engine, \
             patch("main.aiohttp.ClientSession") as mock_session_cls:

            mock_session = MagicMock()
            mock_session.close = AsyncMock()
            mock_session_cls.return_value = mock_session

            resp_mock = MagicMock()
            resp_mock.status = 500
            get_context = MagicMock()
            get_context.__aenter__ = AsyncMock(return_value=resp_mock)
            get_context.__aexit__ = AsyncMock(return_value=None)
            mock_session.get.return_value = get_context

            # БД
            mock_conn = AsyncMock()
            mock_result = MagicMock()
            mock_result.scalar.return_value = 1
            mock_conn.execute.return_value = mock_result
            mock_engine.connect.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
            mock_engine.connect.return_value.__aexit__ = AsyncMock(return_value=None)

            app = MagicMock()

            async with lifespan(app):
                pass

            assert "OpenRouter вернул статус 500" in caplog.text
            assert "✅ Соединение с БД установлено" in caplog.text
            mock_session.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_lifespan_db_failure(caplog):
    """БД не доступна – предупреждение."""
    with caplog.at_level(logging.WARNING):
        with patch("main.API_KEY", None), \
             patch("main.async_engine") as mock_engine, \
             patch("main.aiohttp.ClientSession") as mock_session_cls:

            mock_session = MagicMock()
            mock_session.close = AsyncMock()
            mock_session_cls.return_value = mock_session
            mock_engine.connect.side_effect = Exception("DB down")

            app = MagicMock()

            async with lifespan(app):
                pass

            assert "Не удалось подключиться к БД" in caplog.text
            mock_session.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_lifespan_openrouter_exception(caplog):
    """Исключение при запросе к OpenRouter + БД работает."""
    with caplog.at_level(logging.INFO):   # <-- INFO для сообщения БД
        with patch("main.API_KEY", "test-key"), \
             patch("main.async_engine") as mock_engine, \
             patch("main.aiohttp.ClientSession") as mock_session_cls:

            mock_session = MagicMock()
            mock_session.close = AsyncMock()
            mock_session_cls.return_value = mock_session
            mock_session.get.side_effect = Exception("Connection timeout")

            # БД работает
            mock_conn = AsyncMock()
            mock_result = MagicMock()
            mock_result.scalar.return_value = 1
            mock_conn.execute.return_value = mock_result
            mock_engine.connect.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
            mock_engine.connect.return_value.__aexit__ = AsyncMock(return_value=None)

            app = MagicMock()

            async with lifespan(app):
                pass

            assert "Не удалось проверить OpenRouter" in caplog.text
            assert "✅ Соединение с БД установлено" in caplog.text
            mock_session.close.assert_awaited_once()

# =========================
# FETCH FROM MODEL TESTS
# =========================

@pytest.mark.asyncio
async def test_fetch_from_model_no_api_key():
    """API_KEY отсутствует – сразу возвращается ошибка."""
    from src.services.ai_service import fetch_from_model
    with patch("src.services.ai_service.API_KEY", None):
        session = MagicMock()  # не будет использован
        result = await fetch_from_model(session, "model", "prompt")
        assert result["status"] == "error"
        assert "API-ключ не задан" in result["content"]


@pytest.mark.parametrize("status_code, expected_status, expected_text", [
    (200, "success", "def is_leap"),
    (500, "error", "Ошибка 500"),
])
@pytest.mark.asyncio
async def test_fetch_from_model_status_codes(status_code, expected_status, expected_text):
    """Успешный и ошибочный HTTP-статусы."""
    from src.services.ai_service import fetch_from_model
    with patch("src.services.ai_service.API_KEY", "test-key"):
        session = MagicMock()

        # Мок ответа
        resp_mock = MagicMock()
        resp_mock.status = status_code
        if status_code == 200:
            resp_data = {"choices": [{"message": {"content": "def is_leap(year): ..."}}]}
            resp_mock.json = AsyncMock(return_value=resp_data)
        else:
            resp_mock.text = AsyncMock(return_value="Server Error")

        # Объект, который вернёт session.post() и который можно использовать в async with
        post_return = MagicMock()
        post_return.__aenter__ = AsyncMock(return_value=resp_mock)
        post_return.__aexit__ = AsyncMock(return_value=None)
        session.post.return_value = post_return

        result = await fetch_from_model(session, "model", "prompt")
        assert result["status"] == expected_status
        assert expected_text in result["content"]


@pytest.mark.asyncio
async def test_fetch_from_model_exception():
    """Исключение при запросе к API."""
    from src.services.ai_service import fetch_from_model
    with patch("src.services.ai_service.API_KEY", "test-key"):
        session = MagicMock()
        session.post.side_effect = Exception("Boom")
        result = await fetch_from_model(session, "model", "prompt")
        assert result["status"] == "error"
        assert "Исключение" in result["content"]

# =========================
# JUDGE_WINNER: ALL BRANCHES (NEW)
# =========================

@pytest.mark.asyncio
async def test_judge_winner_both_failed():
    """Обе модели вернули ошибку."""
    results = [
        {"model": "A", "content": "err", "status": "error"},
        {"model": "B", "content": "err", "status": "error"},
    ]
    session = AsyncMock()
    res = await judge_winner(results, session)
    assert res["winners"] == []
    assert res["losers"] == ["A", "B"]
    assert res["message"] == "❌ Все модели завершились ошибкой."
    assert res["judge_result"]["winner"] is None
    assert "Обе модели не смогли выполнить задание" in res["judge_result"]["reason"]
    assert res["winner_position"] is None
    assert res["judge_model"] == "deepseek-chat"
    assert res["reason"] is not None


@pytest.mark.asyncio
async def test_judge_winner_one_success_model_a():
    """Одна успешна, и она первая в списке (MODEL_A)."""
    results = [
        {"model": "gpt-4o-mini", "content": "ok", "status": "success"},
        {"model": "llama-3", "content": "error", "status": "error"},
    ]
    session = AsyncMock()
    res = await judge_winner(results, session, judge_model="gpt-4o-mini")
    assert res["winners"] == ["gpt-4o-mini"]
    assert res["losers"] == ["llama-3"]
    assert "Победитель" in res["message"]
    assert "завершилась с ошибкой" in res["reason"]
    assert res["winner_position"] == "MODEL_A"
    assert res["judge_result"]["winner"] == "gpt-4o-mini"
    assert res["judge_model"] == "gpt-4o-mini"


@pytest.mark.asyncio
async def test_judge_winner_one_success_model_b():
    """Одна успешна, но она вторая в списке (MODEL_B)."""
    results = [
        {"model": "gpt-4o-mini", "content": "error", "status": "error"},
        {"model": "llama-3", "content": "ok", "status": "success"},
    ]
    session = AsyncMock()
    res = await judge_winner(results, session, judge_model="deepseek-chat")
    assert res["winners"] == ["llama-3"]
    assert res["losers"] == ["gpt-4o-mini"]
    assert res["winner_position"] == "MODEL_B"
    assert res["judge_result"]["winner"] == "llama-3"
    assert res["judge_model"] == "deepseek-chat"


@pytest.mark.asyncio
async def test_judge_winner_draw():
    """Судья вернул DRAW."""
    results = [
        {"model": "m1", "content": "ok", "status": "success"},
        {"model": "m2", "content": "ok", "status": "success"},
    ]
    session = AsyncMock()
    with patch("src.services.ai_service.ask_judge", new_callable=AsyncMock) as mock_ask:
        mock_ask.return_value = {"winner": "DRAW", "reason": "Оба решения идентичны"}
        res = await judge_winner(results, session, judge_model="llama-3.1-8b")
        assert res["winners"] == []
        assert res["losers"] == []
        assert "Ничья" in res["message"]
        assert res["judge_result"]["winner"] == "DRAW"
        assert res["judge_result"]["reason"] == "Оба решения идентичны"
        assert res["winner_position"] is None
        assert res["judge_model"] == "llama-3.1-8b"
        mock_ask.assert_called_once()


@pytest.mark.asyncio
async def test_judge_winner_judge_error():
    """Судья возвращает ошибку (невалидный JSON/winner)."""
    results = [
        {"model": "m1", "content": "ok", "status": "success"},
        {"model": "m2", "content": "ok", "status": "success"},
    ]
    session = AsyncMock()
    with patch("src.services.ai_service.ask_judge", new_callable=AsyncMock) as mock_ask:
        mock_ask.return_value = {
            "error": "Judge вернул неверный winner: MODEL_C",
            "raw_response": '{"winner": "MODEL_C", "reason": "..."}'
        }
        res = await judge_winner(results, session, judge_model="gpt-4o-mini")
        assert res["winners"] == []
        assert res["losers"] == []
        assert "Judge не смог определить победителя" in res["message"]
        assert res["judge_error"] is not None
        assert res["judge_result"]["winner"] is None
        assert "Судья не дал корректного вердикта" in res["judge_result"]["reason"]
        assert res["winner_position"] is None
        assert res["judge_model"] == "gpt-4o-mini"
        mock_ask.assert_called_once()


@pytest.mark.asyncio
async def test_judge_winner_same_models():
    """Две одинаковые модели, судья выбирает MODEL_A."""
    results = [
        {"model": "gpt-4o-mini", "content": "code1", "status": "success"},
        {"model": "gpt-4o-mini", "content": "code2", "status": "success"},
    ]
    session = AsyncMock()
    with patch("src.services.ai_service.ask_judge", new_callable=AsyncMock) as mock_ask:
        mock_ask.return_value = {"winner": "MODEL_A", "reason": "Первая реализация лучше"}
        res = await judge_winner(results, session)
        assert res["winners"] == ["gpt-4o-mini"]
        assert res["losers"] == []               # та же модель
        assert res["winner_position"] == "MODEL_A"
        assert res["reason"] == "Первая реализация лучше"
        assert res["judge_result"]["reason"] == "Первая реализация лучше"
        mock_ask.assert_called_once()


@pytest.mark.asyncio
async def test_judge_winner_empty_reason_handling():
    """Судья вернул пустой reason – должна быть ошибка."""
    results = [
        {"model": "m1", "content": "ok", "status": "success"},
        {"model": "m2", "content": "ok", "status": "success"},
    ]
    session = AsyncMock()
    with patch("src.services.ai_service.ask_judge", new_callable=AsyncMock) as mock_ask:
        mock_ask.return_value = {"error": "Judge вернул пустой reason", "raw_response": '{"winner": "MODEL_A", "reason": ""}'}
        res = await judge_winner(results, session)
        assert res["winners"] == []
        assert "judge_error" in res
        assert "пустой reason" in res["judge_result"]["reason"]
        mock_ask.assert_called_once()


@pytest.mark.asyncio
async def test_judge_winner_winner_position_consistency():
    """Проверка, что winner_position совпадает с выбором судьи."""
    results = [
        {"model": "a", "content": "ok", "status": "success"},
        {"model": "b", "content": "ok", "status": "success"},
    ]
    session = AsyncMock()
    with patch("src.services.ai_service.ask_judge", new_callable=AsyncMock) as mock_ask:
        mock_ask.return_value = {"winner": "MODEL_B", "reason": "B is better"}
        res = await judge_winner(results, session)
        assert res["winners"] == ["b"]
        assert res["winner_position"] == "MODEL_B"
        mock_ask.assert_called_once()

# =========================
# validate judge_model mapping
# =========================

@pytest.mark.asyncio
async def test_battle_with_valid_judge_model(client):
    """
    Проверка, что выбранная модель судьи попадает в ответ и что
    результат судейства возвращается корректно.
    """
    battle_id = 1

    with patch(
        "src.routers.llm_arena.get_battle_by_id", new_callable=AsyncMock
    ) as mock_get_battle, patch(
        "src.routers.llm_arena.judge_winner", new_callable=AsyncMock   # мокаем в роутере
    ) as mock_judge:

        # Мок боевой записи из базы
        mock_battle = MagicMock()
        mock_battle.id = battle_id
        mock_battle.model1 = "gpt-4o-mini"
        mock_battle.model2 = "deepseek-chat"
        mock_battle.evidence = [
            {"model": "openai/gpt-4o-mini", "content": "code A", "status": "success"},
            {"model": "deepseek-chat", "content": "code B", "status": "success"},
        ]
        mock_get_battle.return_value = mock_battle

        # Мок результата судьи
        mock_judge.return_value = {
            "winners": ["openai/gpt-4o-mini"],
            "losers": ["deepseek-chat"],
            "message": "OK",
            "judge_result": {"winner": "MODEL_A", "reason": "better solution"},
            "reason": "better solution",
            "winner_position": "MODEL_A",
            "judge_model": "llama-3.1-8b",
        }

        response = client.post(
            f"/api/llm-arena/winner/{battle_id}",
            json={"judge_model": "llama-3.1-8b"}
        )

        assert response.status_code == 200
        data = response.json()

        # Основные проверки
        assert data["winners"] == ["openai/gpt-4o-mini"]
        assert data["losers"] == ["deepseek-chat"]
        assert data["judge_result"]["winner"] == "MODEL_A"
        assert data["reason"] == "better solution"
        assert data["winner_position"] == "MODEL_A"
        assert data["judge_model_name"] == "llama-3.1-8b"

# =========================
# DOWNLOAD SERVICE
# =========================

def test_generate_battle_markdown_same_models():
    """Обе модели одинаковые, победитель определён."""
    battle = MagicMock()
    battle.model1 = "gpt-4o-mini"
    battle.model2 = "gpt-4o-mini"
    battle.winner = "openai/gpt-4o-mini"
    battle.winner_position = "MODEL_A"
    battle.message = "Победа"
    battle.judge_reason = "Лучше тесты"
    battle.evidence = [
        {"model": "gpt-4o-mini", "content": "code", "status": "success"}
    ]

    md = generate_battle_markdown(battle)
    assert "Модель A" in md
    assert "✅" in md
    assert "❌" in md


def test_generate_battle_markdown_draw():
    """Ничья – нет победителя."""
    battle = MagicMock()
    battle.model1 = "m1"
    battle.model2 = "m2"
    battle.winner = None
    battle.winner_position = None
    battle.message = "Ничья!"
    battle.judge_reason = "Обе одинаковы"
    battle.evidence = [
        {"model": "m1", "content": "c1", "status": "success"}
    ]

    md = generate_battle_markdown(battle)
    assert "🤝" in md
    assert "Ничья" in md

# =========================
# ARENA RESULT SERVICE
# =========================

@pytest.mark.asyncio
async def test_get_battle_by_id_not_found():
    """Если битва не найдена, возвращается None."""
    db = AsyncMock()
    mock_result = MagicMock()
    mock_result.scalars.return_value.first.return_value = None
    db.execute.return_value = mock_result
    result = await get_battle_by_id(db, 999)
    assert result is None


@pytest.mark.asyncio
async def test_get_last_result_service_empty():
    """Если таблица пуста, возвращается None."""
    db = AsyncMock()
    mock_result = MagicMock()
    mock_result.scalars.return_value.first.return_value = None
    db.execute.return_value = mock_result
    result = await get_last_result_service(db)
    assert result is None

# =========================
# UTILS TESTS
# =========================

def test_normalize_decision():
    """Функция normalize_decision: валидный, неполный и не-словарь."""
    from src.utils.normalize import normalize_decision

    # Полный корректный ответ
    full = {
        "winners": ["model-a"],
        "losers": ["model-b"],
        "message": "ok",
        "reason": "good",
        "judge_result": {"winner": "model-a"},
        "winner_position": "MODEL_A",
    }
    res = normalize_decision(full)
    assert res["winners"] == ["model-a"]
    assert res["losers"] == ["model-b"]
    assert res["reason"] == "good"
    assert res["judge_result"] == {"winner": "model-a"}
    assert res["winner_position"] == "MODEL_A"

    # Частичный ответ (пустые списки/строки)
    partial = {"winners": []}
    res = normalize_decision(partial)
    assert res["winners"] == []
    assert res["losers"] == []
    assert res["message"] == ""
    assert res["reason"] == ""
    assert res["winner_position"] is None

    # Вход не словарь
    res = normalize_decision("invalid")
    assert res["winners"] == []
    assert res["message"] == "Invalid judge response"
    assert res["judge_error"] is not None

# pytest src/tests/test_main.py -v
# pytest --cov=src --cov-report=term-missing
# pytest --cov=src --cov-report=xml

