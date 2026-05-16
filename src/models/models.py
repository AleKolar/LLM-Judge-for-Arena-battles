# src/models/models.py
from datetime import datetime

from pydantic import BaseModel, ConfigDict


class CompareRequest(BaseModel):
    models: list[str] | None = None
    prompt: str | None = None

class ModelEvidence(BaseModel):
    model: str
    content: str
    status: str

class WinnerResponse(BaseModel):
    winners: list[str]
    losers: list[str]
    message: str
    summary: str | None = None
    reason: str | None = None
    judge_result: dict | None = None
    judge_error: dict | None = None
    evidence: list[ModelEvidence] = []
    model_a_name: str | None = None   # короткое имя первой модели (ВАЖНО, при одинаковых моделях)
    model_b_name: str | None = None   # короткое имя второй модели
    judge_model_name: str | None = None
    winner_position: str | None = None

class BattleHistoryResponse(BaseModel):

    id: int
    model1: str
    model2: str
    winner: str | None = None
    message: str
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)

class FullBattleRequest(BaseModel):
    models: list[str]
    judge_model: str
    prompt: str | None = None

