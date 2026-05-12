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


class BattleHistoryResponse(BaseModel):

    id: int
    model1: str
    model2: str
    winner: str | None = None
    message: str
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)

