# src/schemas/schemas.py

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict


class CompareRequest(BaseModel):
    models: list[str] | None = None
    prompt: str | None = None


class WinnerRequest(BaseModel):
    results: list[dict]


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