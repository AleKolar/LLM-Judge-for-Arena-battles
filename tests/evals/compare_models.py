"""Run the real Qwen3-4B vs Qwen3-8B Arena and judge it with DeepEval ArenaGEval.

Run from the repository root:
    python tests/evals/compare_models.py
    python tests/evals/compare_models.py --limit 2

This script does not create application ArenaResult database records. It is an
offline evaluation experiment over the checked-in prompt set.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import aiohttp
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")

from deepeval.metrics import ArenaGEval  # noqa: E402
from deepeval.models import OllamaModel  # noqa: E402
from deepeval.test_case import (  # noqa: E402
    ArenaTestCase,
    Contestant,
    LLMTestCase,
    SingleTurnParams,
)

from src.services.ai_service import (  # noqa: E402
    AVAILABLE_MODELS,
    OLLAMA_BASE_URL,
    OLLAMA_NUM_CTX,
    OLLAMA_REQUEST_TIMEOUT,
    run_arena_comparison,
)
from tests.evals.eval_utils import load_golden_cases  # noqa: E402

MODEL_A_KEY = "qwen3-4b"
MODEL_B_KEY = "qwen3-8b"


def build_evaluator() -> OllamaModel:
    """Use a local evaluator; no hosted LLM API or API key is required."""
    evaluator_name = os.getenv(
        "DEEPEVAL_EVALUATOR_MODEL",
        AVAILABLE_MODELS[MODEL_A_KEY]["model_id"],
    )
    return OllamaModel(
        model=evaluator_name,
        base_url=OLLAMA_BASE_URL,
        temperature=0,
        generation_kwargs={
            "num_ctx": OLLAMA_NUM_CTX,
            "num_predict": int(os.getenv("DEEPEVAL_OLLAMA_NUM_PREDICT", "512")),
        },
    )


async def generate_arena_outputs(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Call the application's actual participant-generation service in order."""
    timeout = aiohttp.ClientTimeout(total=max(OLLAMA_REQUEST_TIMEOUT, 300.0))
    generated: list[dict[str, Any]] = []

    async with aiohttp.ClientSession(timeout=timeout) as session:
        for case in cases:
            print(f"\n[GENERATE] {case['id']}")
            result = await run_arena_comparison(
                models=[MODEL_A_KEY, MODEL_B_KEY],
                session=session,
                prompt=case["prompt"],
            )
            outputs = result.get("results", [])
            by_model_key = {
                MODEL_A_KEY: outputs[0] if len(outputs) > 0 else None,
                MODEL_B_KEY: outputs[1] if len(outputs) > 1 else None,
            }

            if any(
                not item or item.get("status") != "success"
                for item in by_model_key.values()
            ):
                errors = {
                    key: (item or {}).get("content", "No result")
                    for key, item in by_model_key.items()
                    if not item or item.get("status") != "success"
                }
                print(f"[SKIP] Participant generation failed: {errors}")
                generated.append(
                    {
                        "case_id": case["id"],
                        "status": "skipped",
                        "errors": errors,
                        "elapsed": result.get("elapsed", 0),
                    }
                )
                continue

            generated.append(
                {
                    "case_id": case["id"],
                    "status": "success",
                    "prompt": case["prompt"],
                    "reference_output": case["reference_output"],
                    "model_a": by_model_key[MODEL_A_KEY],
                    "model_b": by_model_key[MODEL_B_KEY],
                    "elapsed": result.get("elapsed", 0),
                }
            )

    return generated


def evaluate_one_case(
    case: dict[str, Any],
    generated: dict[str, Any],
    evaluator: OllamaModel,
) -> dict[str, Any]:
    model_a_id = AVAILABLE_MODELS[MODEL_A_KEY]["model_id"]
    model_b_id = AVAILABLE_MODELS[MODEL_B_KEY]["model_id"]

    arena_case = ArenaTestCase(
        contestants=[
            Contestant(
                name="Qwen3-4B",
                hyperparameters={"provider": "ollama", "model": model_a_id},
                test_case=LLMTestCase(
                    input=case["prompt"],
                    actual_output=generated["model_a"]["content"],
                    expected_output=case["reference_output"],
                ),
            ),
            Contestant(
                name="Qwen3-8B",
                hyperparameters={"provider": "ollama", "model": model_b_id},
                test_case=LLMTestCase(
                    input=case["prompt"],
                    actual_output=generated["model_b"]["content"],
                    expected_output=case["reference_output"],
                ),
            ),
        ]
    )

    arena_metric = ArenaGEval(
        name="Arena Code Solution Quality",
        evaluation_steps=[
            "Compare both candidate outputs against the task input and the reference output.",
            "Prioritize functional correctness and exact adherence to every stated requirement.",
            "Check edge cases, failure handling, security properties, and algorithmic correctness when relevant to the task.",
            "Do not reward verbosity, formatting, or extra explanation unless it improves correctness or usability.",
            "Prefer the candidate with fewer material defects and better coverage of the task requirements.",
            "If both outputs are equally correct and complete, select a draw rather than inventing a difference.",
        ],
        evaluation_params=[
            SingleTurnParams.INPUT,
            SingleTurnParams.ACTUAL_OUTPUT,
            SingleTurnParams.EXPECTED_OUTPUT,
        ],
        model=evaluator,
        async_mode=False,
        verbose_mode=False,
    )

    arena_metric.measure(arena_case)
    winner = arena_metric.winner
    reason = arena_metric.reason

    return {
        "case_id": case["id"],
        "status": "evaluated",
        "winner": str(winner) if winner is not None else None,
        "reason": str(reason) if reason is not None else "",
        "participant_elapsed_seconds": generated.get("elapsed", 0),
        "expected_reference": case["reference_output"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Evaluate only the first N golden prompts (useful for a quick smoke run).",
    )
    args = parser.parse_args()

    cases = load_golden_cases()
    if args.limit is not None:
        if args.limit < 1:
            parser.error("--limit must be at least 1")
        cases = cases[: args.limit]

    print(f"Evaluating {len(cases)} golden prompts")
    print(f"Model A: {AVAILABLE_MODELS[MODEL_A_KEY]['model_id']}")
    print(f"Model B: {AVAILABLE_MODELS[MODEL_B_KEY]['model_id']}")

    try:
        generated = asyncio.run(generate_arena_outputs(cases))
        evaluator = build_evaluator()
    except Exception as exc:
        print(f"[FATAL] Could not generate participant outputs: {type(exc).__name__}: {exc}")
        return 1

    generated_by_id = {item["case_id"]: item for item in generated}
    results: list[dict[str, Any]] = []
    wins: Counter[str] = Counter()

    for case in cases:
        item = generated_by_id[case["id"]]
        if item["status"] != "success":
            results.append(item)
            continue

        print(f"\n[EVALUATE] {case['id']}")
        try:
            evaluated = evaluate_one_case(case, item, evaluator)
            results.append(evaluated)
            winner = evaluated.get("winner") or "UNRESOLVED"
            wins[winner] += 1
            print(f"Winner: {winner}")
            print(f"Reason: {evaluated.get('reason', '')}")
        except Exception as exc:
            error_result = {
                "case_id": case["id"],
                "status": "evaluation_error",
                "error": f"{type(exc).__name__}: {exc}",
            }
            results.append(error_result)
            print(f"[ERROR] DeepEval failed: {error_result['error']}")

    report_dir = PROJECT_ROOT / "tests" / "evals" / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = report_dir / f"arena_compare_{timestamp}.json"

    model_a_id = AVAILABLE_MODELS[MODEL_A_KEY]["model_id"]
    model_b_id = AVAILABLE_MODELS[MODEL_B_KEY]["model_id"]
    report = {
        "created_at": datetime.now().astimezone().isoformat(),
        "evaluator_model": os.getenv(
            "DEEPEVAL_EVALUATOR_MODEL",
            AVAILABLE_MODELS[MODEL_A_KEY]["model_id"],
        ),
        "model_a": model_a_id,
        "model_b": model_b_id,
        "results": results,
        "win_counts": dict(wins),
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n========== ARENA SUMMARY ==========")
    print(f"Win counts: {dict(wins)}")
    print(f"Evaluated: {sum(1 for item in results if item['status'] == 'evaluated')}")
    print(f"Skipped/errors: {sum(1 for item in results if item['status'] != 'evaluated')}")
    print(f"Report: {report_path}")

    if not any(item["status"] == "evaluated" for item in results):
        return 1
    if any(item["status"] == "evaluation_error" for item in results):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
