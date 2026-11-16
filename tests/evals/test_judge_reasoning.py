"""Evaluate grounding, specificity, and consistency of the Judge's reason."""

from __future__ import annotations

import pytest
from deepeval import assert_test
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, SingleTurnParams

from tests.evals.eval_utils import (
    build_expected_judge_output,
    build_judge_input,
    load_golden_cases,
)

_GOLDEN_IDS = [case["id"] for case in load_golden_cases()]


@pytest.mark.parametrize("golden_id", _GOLDEN_IDS, ids=_GOLDEN_IDS)
def test_judge_reasoning_is_grounded_and_specific(
    golden_id,
    golden_cases_by_id,
    app_judge_outputs,
    evaluation_model,
):
    golden = golden_cases_by_id[golden_id]
    actual = app_judge_outputs[golden_id]

    assert actual["status"] == "success", (
        f"The application Judge did not return a valid verdict for {golden_id}: "
        f"{actual.get('error', actual['actual_output'])}"
    )

    test_case = LLMTestCase(
        input=build_judge_input(golden),
        actual_output=actual["actual_output"],
        expected_output=build_expected_judge_output(golden),
    )

    reasoning_quality = GEval(
        name="Arena Judge Reasoning Quality",
        evaluation_steps=[
            "Check whether the actual reason explains the decisive difference between Model A and Model B using details visible in their responses.",
            "Check whether the reason is consistent with the actual winner label and the expected winner stated in expected_output.",
            "Penalize generic statements such as 'the answer is better' when they do not identify a concrete requirement, bug, edge case, or implementation property.",
            "Penalize claims that are not supported by either candidate response or the task requirements.",
            "Reward concise comparisons that explain why the selected response satisfies the task more accurately or completely.",
            "For DRAW, require the reason to explain why both responses satisfy the requirements and why neither has a material advantage.",
        ],
        evaluation_params=[
            SingleTurnParams.INPUT,
            SingleTurnParams.ACTUAL_OUTPUT,
            SingleTurnParams.EXPECTED_OUTPUT,
        ],
        threshold=0.70,
        model=evaluation_model,
        async_mode=False,
        verbose_mode=False,
    )

    assert_test(test_case, [reasoning_quality])


# python -m pytest tests/evals/test_judge_reasoning.py
