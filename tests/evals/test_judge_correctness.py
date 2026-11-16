"""Evaluate whether the application's Judge chooses the expected winner."""

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
def test_judge_selects_expected_winner(
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

    correctness = GEval(
        name="Arena Judge Winner Correctness",
        evaluation_steps=[
            "Read the task requirements and both candidate responses in the input.",
            "Read the expected winner label and its justification in expected_output.",
            "Read the actual JSON verdict in actual_output and identify its winner field.",
            "Award a high score only when the actual winner is exactly the expected MODEL_A, MODEL_B, or DRAW label.",
            "For DRAW, verify that both candidate responses satisfy the task to materially equivalent quality.",
            "For MODEL_A or MODEL_B, verify that the selected candidate is the one that satisfies the requirements more completely and correctly.",
            "Do not reward a correct label if the response is malformed or the winner is unresolved.",
        ],
        evaluation_params=[
            SingleTurnParams.INPUT,
            SingleTurnParams.ACTUAL_OUTPUT,
            SingleTurnParams.EXPECTED_OUTPUT,
        ],
        threshold=0.80,
        model=evaluation_model,
        async_mode=False,
        verbose_mode=False,
    )

    assert_test(test_case, [correctness])


# python -m pytest tests/evals/test_judge_correctness.py -v -s
