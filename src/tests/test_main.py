from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from main import app
from src.database.database import get_async_db
from src.schemas.schemas import CompareRequest, WinnerRequest, ArenaResultResponse
from src.services.ai_service import judge_winner
from src.services.arena_result import get_last_result_service
from src.utils.normalize import normalize_evidence, to_md


# =========================
# FIXTURE (PRODUCTION STYLE)
# =========================

@pytest.fixture(scope="function")
def client():
    """
    TestClient + isolated dependency override.
    Production-style: clean DI boundary.
    """

    async def override_get_async_db():
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


# =========================
# CORE LOGIC TESTS
# =========================

@pytest.mark.parametrize(
    "year,expected",
    [
        (2024, True),
        (2000, True),
        (0, True),
        (-400, True),
        (1900, False),
        (1800, False),
        (2100, False),
        (2023, False),
        (1, False),
    ],
)
def test_is_leap_year(year, expected):
    from src.services.leap_year_service import is_leap_year

    assert is_leap_year(year) == expected


def test_check_year(client):
    resp = client.get("/api/check/2000")

    assert resp.status_code == 200
    data = resp.json()

    assert data["is_leap"] is True
    assert data["days"] == 366


def test_check_1900(client):
    resp = client.get("/api/check/1900")

    assert resp.status_code == 200
    data = resp.json()

    assert data["is_leap"] is False
    assert data["rule_check"]["divisible_by_100"] is True


def test_stats(client):
    resp = client.get("/api/stats")

    assert resp.status_code == 200
    assert "total_checks" in resp.json()


def test_main_page(client):
    resp = client.get("/")

    assert resp.status_code == 200
    assert "LLM Arena" in resp.text


# =========================
# LLM ARENA (PRODUCTION MOCK LAYER)
# =========================

def _mock_llm_response(model, content):
    return {
        "model": model,
        "content": content,
        "status": "success",
    }


@pytest.mark.asyncio
async def test_compare_models(client):
    """
    Production pattern:
    - mock LLM boundary only
    - do NOT mock internal logic
    """

    with patch("src.services.ai_service.fetch_from_model", new_callable=AsyncMock) as mock_fetch:

        mock_fetch.side_effect = [
            _mock_llm_response(
                "openai/gpt-4o-mini",
                "def is_leap(y): return y % 400 == 0 or (y % 4 == 0 and y % 100 != 0)"
            ),
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
        assert len(data["results"]) == 2
        assert isinstance(data["elapsed"], (int, float))


def test_list_models(client):
    resp = client.get("/api/llm-arena/models")

    assert resp.status_code == 200
    assert "deepseek-chat" in resp.json()


def test_winner_endpoint(client):
    """
    Production-safe DB isolation test
    """

    mock_battle = MagicMock()
    mock_battle.evidence = [
        _mock_llm_response("openai/gpt", "code"),
        _mock_llm_response("deepseek", "code"),
    ]
    mock_battle.winner = None
    mock_battle.message = ""

    mock_result = MagicMock()
    mock_result.scalar.return_value = mock_battle

    async def override_db():
        session = AsyncMock()
        session.execute = AsyncMock(return_value=mock_result)
        session.commit = AsyncMock()
        session.refresh = AsyncMock()
        session.add = MagicMock()
        yield session

    app.dependency_overrides[get_async_db] = override_db

    try:
        resp = client.post("/api/llm-arena/winner")

        assert resp.status_code == 200

        data = resp.json()
        assert isinstance(data["winners"], list)
        assert isinstance(data["losers"], list)
        assert isinstance(data["message"], str)

    finally:
        app.dependency_overrides.clear()


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
    mock_battle.evidence = []
    mock_battle.message = "ok"

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
        assert resp.status_code in (200, 404)
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


# pytest src/tests/test_main.py -v
# pytest --cov=src --cov-report=term-missing
# pytest --cov=src --cov-report=xml

