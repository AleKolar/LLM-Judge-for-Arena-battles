"""Run the project's fixed DeepEval pytest suites from the local UI.

The manager keeps a small in-memory history and streams subprocess output to
polling clients. Test names are selected from a whitelist; no shell is used.
"""

from __future__ import annotations

import asyncio
import os
import re
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

router = APIRouter(prefix="/api/evals", tags=["DeepEval"])

SUITES: dict[str, dict[str, str]] = {
    "correctness": {
        "label": "Корректность выбора победителя",
        "target": "tests/evals/test_judge_correctness.py",
    },
    "reasoning": {
        "label": "Качество аргументации Judge",
        "target": "tests/evals/test_judge_reasoning.py",
    },
    "all": {
        "label": "Полный прогон DeepEval",
        "target": "tests/evals",
    },
}

MAX_LOG_CHARS = 250_000
MAX_RUN_HISTORY = 10
REQUIRED_EVAL_FILES = (
    "tests/evals/conftest.py",
    "tests/evals/eval_utils.py",
    "tests/evals/ollama_eval_model.py",
    "tests/evals/test_judge_correctness.py",
    "tests/evals/test_judge_reasoning.py",
    "tests/evals/data/arena_goldens.json",
)


class StartEvaluationRequest(BaseModel):
    suite: str = Field(..., description="One of: correctness, reasoning, all")


class EvaluationRunManager:
    """Own pytest subprocesses for one FastAPI worker."""

    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root.resolve()
        self._runs: dict[str, dict[str, Any]] = {}
        self._lock = asyncio.Lock()

    @staticmethod
    def _counts(log_text: str) -> dict[str, int]:
        patterns = {
            "passed": r"\b(\d+)\s+passed\b",
            "failed": r"\b(\d+)\s+failed\b",
            "errors": r"\b(\d+)\s+errors?\b",
            "skipped": r"\b(\d+)\s+skipped\b",
            "collected": r"\bcollected\s+(\d+)\s+items?\b",
        }
        counts: dict[str, int] = {}
        for name, pattern in patterns.items():
            matches = re.findall(pattern, log_text, flags=re.IGNORECASE)
            counts[name] = int(matches[-1]) if matches else 0
        return counts

    @staticmethod
    def _public(run: dict[str, Any], *, include_log: bool) -> dict[str, Any]:
        result = {
            "run_id": run["run_id"],
            "suite": run["suite"],
            "label": run["label"],
            "status": run["status"],
            "started_at": run["started_at"],
            "finished_at": run["finished_at"],
            "duration_seconds": run["duration_seconds"],
            "return_code": run["return_code"],
            "counts": dict(run["counts"]),
        }
        if include_log:
            result["log"] = run["log"]
        return result

    def _trim_history(self) -> None:
        finished = [
            run_id
            for run_id, run in self._runs.items()
            if run["status"] != "running"
        ]
        while len(self._runs) > MAX_RUN_HISTORY and finished:
            oldest = finished.pop(0)
            self._runs.pop(oldest, None)

    async def start(self, suite: str) -> dict[str, Any]:
        if suite not in SUITES:
            raise HTTPException(
                status_code=422,
                detail=f"Неизвестный набор тестов: {suite!r}. Допустимо: {', '.join(SUITES)}.",
            )

        async with self._lock:
            active = next(
                (run for run in self._runs.values() if run["status"] == "running"),
                None,
            )
            if active:
                raise HTTPException(
                    status_code=409,
                    detail=f"Уже выполняется тестовый запуск {active['run_id']} ({active['label']}).",
                )

            missing = [
                relative
                for relative in REQUIRED_EVAL_FILES
                if not (self.project_root / relative).is_file()
            ]
            if missing:
                raise HTTPException(
                    status_code=500,
                    detail=(
                        "Не найдены необходимые файлы DeepEval: "
                        + ", ".join(missing)
                        + ". Проверьте структуру tests/evals и файл arena_goldens.json."
                    ),
                )

            suite_config = SUITES[suite]
            command = [
                sys.executable,
                "-u",
                "-m",
                "pytest",
                suite_config["target"],
                "-v",
                "-s",
                "--color=no",
            ]
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            env["PYTHONIOENCODING"] = "utf-8"
            env["NO_COLOR"] = "1"

            creation_options: dict[str, Any] = {}
            if os.name == "nt":
                creation_options["creationflags"] = subprocess.CREATE_NO_WINDOW

            try:
                process = await asyncio.create_subprocess_exec(
                    *command,
                    cwd=str(self.project_root),
                    env=env,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT,
                    **creation_options,
                )
            except OSError as exc:
                raise HTTPException(
                    status_code=500,
                    detail=f"Не удалось запустить pytest через {sys.executable}: {exc}",
                ) from exc

            run_id = uuid.uuid4().hex[:12]
            started = datetime.now(timezone.utc).isoformat(timespec="seconds")
            run: dict[str, Any] = {
                "run_id": run_id,
                "suite": suite,
                "label": suite_config["label"],
                "status": "running",
                "started_at": started,
                "finished_at": None,
                "duration_seconds": None,
                "return_code": None,
                "counts": {
                    "passed": 0,
                    "failed": 0,
                    "errors": 0,
                    "skipped": 0,
                    "collected": 0,
                },
                "log": "",
                "_process": process,
                "_started_monotonic": time.monotonic(),
                "_task": None,
            }
            self._runs[run_id] = run
            run["_task"] = asyncio.create_task(self._collect_output(run_id, process))
            self._trim_history()
            return self._public(run, include_log=False)

    async def _collect_output(
        self,
        run_id: str,
        process: asyncio.subprocess.Process,
    ) -> None:
        run = self._runs[run_id]
        read_error: str | None = None
        try:
            if process.stdout is not None:
                while True:
                    chunk = await process.stdout.read(4096)
                    if not chunk:
                        break
                    run["log"] = (run["log"] + chunk.decode("utf-8", errors="replace"))[-MAX_LOG_CHARS:]
        except Exception as exc:  # preserve process exit even if log capture fails
            read_error = f"\n[Ошибка чтения вывода pytest: {type(exc).__name__}: {exc}]\n"
        finally:
            return_code = await process.wait()
            if read_error:
                run["log"] = (run["log"] + read_error)[-MAX_LOG_CHARS:]
            run["return_code"] = return_code
            run["counts"] = self._counts(run["log"])
            run["finished_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
            run["duration_seconds"] = round(
                time.monotonic() - run["_started_monotonic"], 2
            )
            run["status"] = "passed" if return_code == 0 and read_error is None else "failed"

    def list_runs(self) -> list[dict[str, Any]]:
        ordered = sorted(
            self._runs.values(),
            key=lambda item: item["_started_monotonic"],
            reverse=True,
        )
        return [self._public(run, include_log=False) for run in ordered]

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        run = self._runs.get(run_id)
        return self._public(run, include_log=True) if run else None

    async def shutdown(self) -> None:
        processes = [
            run["_process"]
            for run in self._runs.values()
            if run["status"] == "running"
        ]
        for process in processes:
            if process.returncode is None:
                try:
                    process.terminate()
                except ProcessLookupError:
                    pass
        for process in processes:
            if process.returncode is None:
                try:
                    await asyncio.wait_for(process.wait(), timeout=5)
                except asyncio.TimeoutError:
                    process.kill()
                    await process.wait()
        tasks = [
            run["_task"]
            for run in self._runs.values()
            if run.get("_task") is not None
        ]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)


def _manager_from_request(request: Request) -> EvaluationRunManager:
    manager = getattr(request.app.state, "evaluation_run_manager", None)
    if manager is None:
        raise HTTPException(
            status_code=503,
            detail="Менеджер DeepEval ещё не инициализирован. Перезапустите FastAPI.",
        )
    return manager


@router.post("/runs", status_code=202)
async def start_evaluation(payload: StartEvaluationRequest, request: Request):
    """Start one of the predefined DeepEval pytest suites."""
    manager = _manager_from_request(request)
    return await manager.start(payload.suite)


@router.get("/runs")
async def list_evaluation_runs(request: Request):
    """Return the in-memory run history for the current API worker."""
    manager = _manager_from_request(request)
    return {"runs": manager.list_runs()}


@router.get("/runs/{run_id}")
async def get_evaluation_run(run_id: str, request: Request):
    """Return a run status and its captured pytest output."""
    manager = _manager_from_request(request)
    run = manager.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Тестовый запуск не найден или история очищена.")
    return run
