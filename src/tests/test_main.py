from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from main import app
from src.database.database import get_async_db
from src.schemas.schemas import ArenaResultResponse, CompareRequest, WinnerRequest
from src.services.ai_service import judge_winner
from src.services.arena_result import get_last_result_service
from src.services.leap_year_service import is_leap_year
from src.utils.normalize import normalize_evidence, to_md

# =========================
# FIXTURE
# =========================

@pytest.fixture
def client():
    """Фикстура TestClient с заменой зависимости БД на мок."""

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
# TESTS: CORE LOGIC
# =========================

@pytest.mark.parametrize("year,expected", [
    (2024, True), (2000, True), (0, True), (-400, True),
    (1900, False), (1800, False), (2100, False), (2023, False), (1, False)
])
def test_is_leap_year(year, expected):
    assert is_leap_year(year) == expected


def test_check_year(client):
    resp = client.get("/api/check/2000")
    data = resp.json()
    assert data["is_leap"]
    assert data["days"] == 366


def test_check_1900(client):
    resp = client.get("/api/check/1900")
    data = resp.json()
    assert not data["is_leap"]
    assert data["rule_check"]["divisible_by_100"]


def test_stats(client):
    resp = client.get("/api/stats")
    assert resp.status_code == 200
    assert "total_checks" in resp.json()


def test_main_page(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert "Leap Year Detective" in resp.text


# =========================
# TESTS: LLM ARENA
# =========================

@pytest.mark.asyncio
async def test_compare_models(client):
    with patch("src.services.ai_service.fetch_from_model") as mock_fetch:
        mock_fetch.side_effect = [
            {"model": "openai/gpt-4o-mini", "content": "def is_leap(y): return y%4==0 and (y%100!=0 or y%400==0)", "status": "success"},
            {"model": "deepseek/deepseek-chat", "content": "def is_leap(y): return y%4==0", "status": "success"},
        ]

        resp = client.post(
            "/api/llm-arena/compare",
            json={"models": ["gpt-4o-mini", "deepseek-chat"]}
        )

        assert resp.status_code == 200
        data = resp.json()
        assert len(data["results"]) == 2
        assert data["elapsed"] >= 0


def test_list_models(client):
    resp = client.get("/api/llm-arena/models")
    assert "deepseek-chat" in resp.json()


def test_winner_endpoint(client):
    mock_battle = MagicMock()
    mock_battle.evidence = [
        {"model": "A", "content": "def f(): return year % 400 == 0", "status": "success"},
        {"model": "B", "content": "def f(): return True", "status": "success"}
    ]
    mock_battle.winner = None
    mock_battle.message = ""

    mock_result = MagicMock()
    mock_result.scalar.return_value = mock_battle

    async def override_for_winner():
        session = AsyncMock()
        session.execute = AsyncMock(return_value=mock_result)
        session.commit = AsyncMock()
        session.refresh = AsyncMock()
        session.add = MagicMock()
        yield session

    app.dependency_overrides[get_async_db] = override_for_winner

    resp = client.post("/api/llm-arena/winner")
    data = resp.json()

    assert data["winners"] == ["A"]
    assert "B" in data["losers"]
    assert "Победитель" in data["message"]


# =========================
# TESTS: UTILS
# =========================

def test_normalize_evidence_dict():
    data = [{"model": "gpt", "content": "code"}]
    result = normalize_evidence(data)

    assert result[0]["model"] == "gpt"
    assert result[0]["content"] == "code"


def test_normalize_evidence_string():
    data = ["hello"]
    result = normalize_evidence(data)

    assert result[0]["model"] == "unknown"
    assert "hello" in result[0]["content"]


def test_to_md():
    data = [{"model": "gpt", "content": "print(1)"}]
    md = to_md(data)

    assert "gpt" in md
    assert "```python" in md


# =========================
# TESTS: SERVICES
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
    mock_battle.message = "test"

    mock_scalars = MagicMock()
    mock_scalars.first.return_value = mock_battle

    mock_result = MagicMock()
    mock_result.scalars.return_value = mock_scalars

    async def override_get_async_db():
        session = AsyncMock()
        session.execute = AsyncMock(return_value=mock_result)
        session.commit = AsyncMock()
        session.refresh = AsyncMock()
        session.add = MagicMock()
        yield session

    app.dependency_overrides[get_async_db] = override_get_async_db

    resp = client.get("/api/llm-arena/last-result")

    assert resp.status_code in (200, 404)


def test_judge_winner_simple():
    results = [
        {"model": "A", "content": "def f(): return True", "status": "success"},
        {"model": "B", "content": "error", "status": "error"},
    ]

    res = judge_winner(results)

    assert "winners" in res
    assert "losers" in res
    assert "message" in res

# =========================
# TESTS: SCHEMAS
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
        created_at=datetime.now()
    )

    assert obj.model1 == "a"
    assert obj.model2 == "b"

# pytest src/tests/test_main.py -v
# pytest --cov=src --cov-report=term-missing
# pytest --cov=src --cov-report=xml

