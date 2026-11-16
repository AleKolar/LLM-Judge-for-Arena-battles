"""Session fixtures shared by the DeepEval Judge regression tests."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

import aiohttp
import pytest
from dotenv import load_dotenv

from tests.evals.eval_utils import load_golden_cases

EVAL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EVAL_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

from src.services.ai_service import (  # noqa: E402
    DEFAULT_JUDGE,
    OLLAMA_BASE_URL,
    OLLAMA_NUM_CTX,
    OLLAMA_REQUEST_TIMEOUT,
    QWEN3_4B_MODEL_ID,
    judge_winner,
)

# noqa: E402


@pytest.fixture(scope="session")
def golden_cases() -> list[dict[str, Any]]:
    return load_golden_cases()


@pytest.fixture(scope="session")
def golden_cases_by_id(golden_cases: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {case["id"]: case for case in golden_cases}


@pytest.fixture(scope="session")
def evaluation_model():
    """Return the local LLM used by DeepEval's GEval metrics.

    Qwen3-4B is selected to keep evaluator memory use lower on the 8 GB VRAM
    laptop. It is in the same model family as Arena contestants, so scores
    should be treated as a local regression signal, not an independent
    human-ground-truth benchmark.
    """
    try:
        from tests.evals.ollama_eval_model import ThinkingDisabledOllamaModel
    except ImportError as exc:
        pytest.fail(
            "DeepEval/Ollama integration is unavailable. Install with: "
            "python -m pip install 'deepeval==4.3.4' ollama. "
            f"Import error: {exc}"
        )

    model_name = os.getenv("DEEPEVAL_EVALUATOR_MODEL", QWEN3_4B_MODEL_ID)
    num_predict = int(os.getenv("DEEPEVAL_OLLAMA_NUM_PREDICT", "512"))

    return ThinkingDisabledOllamaModel(
        model=model_name,
        base_url=OLLAMA_BASE_URL,
        temperature=0,
        generation_kwargs={
            "num_ctx": OLLAMA_NUM_CTX,
            "num_predict": num_predict,
        },
    )


@pytest.fixture(scope="session")
def app_judge_outputs(
    golden_cases: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Run the application's real judge once per golden and cache the outputs.

    Both GEval suites reuse these results, so the 8B application Judge is not
    called twice for every case when both test files are run in one command.
    """
    judge_key = os.getenv("EVAL_APP_JUDGE_MODEL", DEFAULT_JUDGE)
    timeout_seconds = max(OLLAMA_REQUEST_TIMEOUT, 300.0)

    async def run_all() -> dict[str, dict[str, Any]]:
        timeout = aiohttp.ClientTimeout(total=timeout_seconds)
        outputs: dict[str, dict[str, Any]] = {}

        async with aiohttp.ClientSession(timeout=timeout) as session:
            for case in golden_cases:
                results = [
                    {
                        "model": "Model A",
                        "content": case["model_a_output"],
                        "status": "success",
                    },
                    {
                        "model": "Model B",
                        "content": case["model_b_output"],
                        "status": "success",
                    },
                ]

                decision = await judge_winner(
                    results=results,
                    session=session,
                    judge_model=judge_key,
                )

                judge_result = decision.get("judge_result") or {}
                winner = decision.get("winner_position")
                if winner not in {"MODEL_A", "MODEL_B"}:
                    if judge_result.get("winner") == "DRAW":
                        winner = "DRAW"
                    elif not winner:
                        winner = "UNRESOLVED"

                reason = decision.get("reason") or judge_result.get("reason") or ""
                status = (
                    "success"
                    if winner in {"MODEL_A", "MODEL_B", "DRAW"} and reason.strip()
                    else "error"
                )

                actual_output = json.dumps(
                    {"winner": winner, "reason": reason},
                    ensure_ascii=False,
                    indent=2,
                )
                outputs[case["id"]] = {
                    "status": status,
                    "winner": winner,
                    "reason": reason,
                    "actual_output": actual_output,
                    "decision": decision,
                    "judge_model": judge_key,
                }

                if status != "success":
                    # Preserve the diagnostic in the fixture so the test can
                    # fail with the actual Ollama/Judge response, not a generic error.
                    outputs[case["id"]]["error"] = (
                        decision.get("judge_error")
                        or decision.get("reason")
                        or "Judge did not produce a valid winner/reason pair."
                    )

        return outputs

    try:
        return asyncio.run(run_all())
    except RuntimeError as exc:
        pytest.fail(
            "Could not run the local application Judge. Ensure Ollama is running "
            "and the configured Judge model is installed. "
            f"Details: {exc}"
        )
