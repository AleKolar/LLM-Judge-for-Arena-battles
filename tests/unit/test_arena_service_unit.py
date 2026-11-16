"""Fast unit tests for deterministic Arena service behavior.

These tests do not call Ollama and can be run with ordinary pytest.
"""

from __future__ import annotations

import pytest

from src.services.ai_service import extract_json, judge_winner


def test_extract_json_parses_plain_json():
    result = extract_json(
        '{"winner":"MODEL_A","reason":"A meets the requirement."}'
    )

    assert result == {
        "winner": "MODEL_A",
        "reason": "A meets the requirement.",
    }


def test_extract_json_parses_json_code_fence_and_leading_text():
    result = extract_json(
        'Judge result:\n```json\n'
        '{"winner":"DRAW","reason":"Both answers are equivalent."}'
        '\n```'
    )

    assert result["winner"] == "DRAW"
    assert result["reason"] == "Both answers are equivalent."


def test_extract_json_rejects_missing_winner_or_reason():
    with pytest.raises(ValueError, match="JSON-объект"):
        extract_json('{"winner":"MODEL_B"}')


@pytest.mark.asyncio
async def test_judge_winner_returns_technical_verdict_when_both_models_fail():
    evidence = [
        {
            "model": "qwen3-4b",
            "content": "Модель Ollama не найдена локально.",
            "status": "error",
        },
        {
            "model": "qwen3-8b",
            "content": "Тайм-аут Ollama.",
            "status": "error",
        },
    ]

    decision = await judge_winner(results=evidence, session=None)

    assert decision["winners"] == []
    assert decision["winner_position"] is None
    assert decision["judge_result"]["winner"] is None
    assert "выберите другие модели" in decision["reason"].lower()
    assert decision["evidence"] == evidence


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("successful_position", "expected_winner_position"),
    [(0, "MODEL_A"), (1, "MODEL_B")],
)
async def test_judge_winner_identifies_only_available_model(
    successful_position,
    expected_winner_position,
):
    evidence = [
        {
            "model": "qwen3-4b",
            "content": "Ответ Qwen3-4B.",
            "status": "success" if successful_position == 0 else "error",
        },
        {
            "model": "qwen3-8b",
            "content": "Ответ Qwen3-8B." if successful_position == 1 else "Модель недоступна.",
            "status": "success" if successful_position == 1 else "error",
        },
    ]

    decision = await judge_winner(results=evidence, session=None)

    available_model = evidence[successful_position]["model"]
    unavailable_model = evidence[1 - successful_position]["model"]

    assert decision["winners"] == [available_model]
    assert decision["winner_position"] == expected_winner_position
    assert decision["judge_result"]["automatic"] is True
    assert unavailable_model in decision["reason"]
    assert "выберите другую" in decision["reason"].lower()
    assert decision["evidence"] == evidence
