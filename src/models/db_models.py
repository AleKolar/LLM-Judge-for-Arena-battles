# src/models/db_models.py
from sqlalchemy import JSON, Column, DateTime, Integer, String, func

from src.database.database import Base


class ArenaResult(Base):
    __tablename__ = "arena_results"
    id = Column(Integer, primary_key=True, index=True)
    model1 = Column(String, nullable=False)
    model2 = Column(String, nullable=False)
    winner = Column(String, nullable=True)
    message = Column(String, nullable=False)
    judge_reason = Column(String, nullable=True)
    evidence = Column(JSON, nullable=True)
    created_at = Column(DateTime, server_default=func.now())
    winner_position = Column(String, nullable=True)   # "MODEL_A" или "MODEL_B"