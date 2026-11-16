"""Shared helpers for the LLM Arena DeepEval suite."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

EVAL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EVAL_DIR.parents[1]
GOLDENS_PATH = EVAL_DIR / "data" / "arena_goldens.json"

VALID_WINNERS = {"MODEL_A", "MODEL_B", "DRAW"}


def load_golden_cases() -> list[dict[str, Any]]:
    """Load and minimally validate the checked-in, human-reviewed goldens."""
    with GOLDENS_PATH.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    cases = payload.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError(f"No golden cases found in {GOLDENS_PATH}")

    required = {
        "id",
        "prompt",
        "model_a_output",
        "model_b_output",
        "expected_winner",
        "expected_rationale",
        "reference_output",
    }
    seen_ids: set[str] = set()
    for index, case in enumerate(cases):
        missing = required - set(case)
        if missing:
            raise ValueError(f"Golden #{index} is missing fields: {sorted(missing)}")
        case_id = case["id"]
        if case_id in seen_ids:
            raise ValueError(f"Duplicate golden id: {case_id}")
        seen_ids.add(case_id)
        if case["expected_winner"] not in VALID_WINNERS:
            raise ValueError(
                f"Golden {case_id}: expected_winner must be one of {sorted(VALID_WINNERS)}"
            )
    return cases


def build_judge_input(case: dict[str, Any]) -> str:
    """Create the complete evaluation input shown to the DeepEval metric."""
    return (
        f"Task:\n{case['prompt']}\n\n"
        f"Model A response:\n{case['model_a_output']}\n\n"
        f"Model B response:\n{case['model_b_output']}"
    )


def build_expected_judge_output(case: dict[str, Any]) -> str:
    return (
        f"Expected winner: {case['expected_winner']}\n"
        f"Expected reasoning: {case['expected_rationale']}"
    )
