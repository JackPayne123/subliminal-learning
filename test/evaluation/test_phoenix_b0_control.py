import os
from pathlib import Path

import pytest
from sl.evaluation.services import compute_p_target_preference
from sl.evaluation.data_models import EvaluationResultRow
from sl.utils.file_utils import read_jsonl


@pytest.mark.parametrize(
    "eval_path",
    [
        "./data/eval_results/phoenix_experiment/B0_control_seed1_eval.jsonl",
        "./data/eval_results/phoenix_experiment/B0_control_seed2_eval.jsonl",
        "./data/eval_results/phoenix_experiment/B0_control_seed3_eval.jsonl",
    ],
)
def test_phoenix_b0_control_shows_transmission(eval_path: str):
    """B0 Control should show phoenix preference transmission above random.

    This is a lightweight sanity check that the end-to-end pipeline
    (dataset -> finetuning -> evaluation) captured the subliminal signal.
    """
    p = Path(eval_path)
    if not p.exists():
        pytest.skip(f"Evaluation file not found: {eval_path}")

    rows = [EvaluationResultRow.model_validate(d) for d in read_jsonl(eval_path)]
    assert len(rows) > 0, "No evaluation rows present"

    ci = compute_p_target_preference("phoenix", rows, confidence=0.95)

    # Threshold: clearly above chance/random floor (e.g., > 0.2)
    # Adjust if your expected B0 preference differs
    assert ci.mean > 0.2, f"Expected phoenix transmission > 20%, got {ci.mean:.3f}"


