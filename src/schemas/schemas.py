# src/schemas/schemas.py

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, field_validator

from src.services.ai_service import JUDGE_MODEL


class CompareRequest(BaseModel):
    models: list[str] | None = None
    prompt: str | None = None

    @field_validator("prompt")
    @classmethod
    def validate_prompt_length(cls, v: str | None) -> str | None:
        if v is not None and len(v) > 4000:
            raise ValueError("Промпт слишком длинный. Максимальная длина — 4000 символов.")
        return v


class WinnerRequest(BaseModel):
    results: list[dict]

class JudgeWinnerRequest(BaseModel):
     """Тело запроса для выбора модели-судьи."""
     judge_model: str = "deepseek-chat"

     @field_validator("judge_model")
     @classmethod
     def validate_judge_model(cls, v: str) -> str:
         if v not in JUDGE_MODEL:
             raise ValueError(
                 f"Недопустимая модель судьи. Доступные: {list(JUDGE_MODEL.keys())}"
             )
         return v


class ArenaResultResponse(BaseModel):
    id: int
    model1: str
    model2: str
    winner: str | None
    message: str
    evidence: list[Any] | None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


# Модель для ответа на /compare (без судьи)
class ArenaCompareResponse(BaseModel):
    arena_result_id: int
    results: list[dict]           # можно как list[ModelEvidence], но для гибкости оставим dict
    elapsed: float