from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, field_validator

from src.services.ai_service import DEFAULT_JUDGE, JUDGE_MODEL


class CompareRequest(BaseModel):
    models: list[str] | None = None
    prompt: str | None = None

    @field_validator("prompt")
    @classmethod
    def validate_prompt_length(cls, value: str | None) -> str | None:
        if value is not None and len(value) > 4000:
            raise ValueError("Промпт слишком длинный. Максимальная длина — 4000 символов.")
        return value


class WinnerRequest(BaseModel):
    results: list[dict[str, Any]]


class JudgeWinnerRequest(BaseModel):
    """Request body for choosing the local Ollama Judge."""

    judge_model: str = DEFAULT_JUDGE
    judge_prompt: str | None = None

    @field_validator("judge_model")
    @classmethod
    def validate_judge_model(cls, value: str) -> str:
        if value not in JUDGE_MODEL:
            raise ValueError(f"Недопустимая модель судьи. Доступные: {sorted(JUDGE_MODEL)}")
        return value

    @field_validator("judge_prompt")
    @classmethod
    def validate_judge_prompt_length(cls, value: str | None) -> str | None:
        if value is not None and len(value) > 8000:
            raise ValueError("Промпт судьи слишком длинный. Максимальная длина — 8000 символов.")
        return value


class ArenaResultResponse(BaseModel):
    id: int
    model1: str
    model2: str
    winner: str | None
    message: str
    evidence: list[Any] | None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class ArenaCompareResponse(BaseModel):
    arena_result_id: int
    results: list[dict[str, Any]]
    elapsed: float
