# src/models/db_models.py
from sqlalchemy import JSON, Column, DateTime, Integer, String, func

from src.database.database import Base


class ArenaResult(Base):
    __tablename__ = "arena_results"

    id = Column(Integer, primary_key=True, index=True)
    model1 = Column(String, nullable=False)
    model2 = Column(String, nullable=False)
    winner = Column(String, nullable=True)           # "model1", "model2", "ничья" или None
    message = Column(String, nullable=False)
    evidence = Column(JSON, nullable=True)           # список доказательств / функции моделей и автотесты моделей
    created_at = Column(DateTime, server_default=func.now())