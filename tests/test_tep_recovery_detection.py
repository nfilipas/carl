import json
from pathlib import Path
import textwrap

import numpy as np
import pandas as pd
import pytest


@pytest.mark.parametrize(
    ("high_intervals", "expected_detection"),
    [
        ([(10, 30)], 20),
        ([(6, 14), (26, 40)], 30),
        ([(21, 30)], 25),
        ([], None),
    ],
)
def test_recovery_detector_matches_post_onset_rolling_alarm(high_intervals, expected_detection):
    notebook_path = (
        Path(__file__).resolve().parents[1]
        / "notebooks"
        / "TEP"
        / "2.2-detection_recovery_pipeline.ipynb"
    )
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    source = next(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if "# For offline evaluation, select the first alarm" in "".join(cell["source"])
    )
    start = source.index("    # For offline evaluation")
    end = source.index("    if anomaly_detected:\n\n        # --- Start recovery process ---")
    detection_loop = textwrap.dedent(source[start:end])

    scores = np.zeros(45)
    for start_idx, end_idx in high_intervals:
        scores[start_idx:end_idx] = 10.0
    hist_ts = pd.DataFrame({"X1": np.sqrt(scores)})
    current_ts = hist_ts.iloc[:6].copy()
    initial_scores = np.zeros(len(scores))
    initial_scores[0] = np.nan
    initial_scores[1:6] = scores[1:6]
    context = {
        "np": np,
        "hist_ts": hist_ts,
        "current_ts": current_ts,
        "sample_idx": 6,
        "n_samples_per_ts": len(scores),
        "recovery_res_process_children": np.zeros((len(scores), 1)),
        "recovery_scores_process_children": np.full(len(scores), np.nan),
        "recovery_score_moving_averages": np.full(len(scores), np.nan),
        "anomaly_res_process_children": np.zeros((len(scores), 1)),
        "anomaly_scores_process_children": initial_scores,
        "anomaly_score_moving_averages": np.zeros(len(scores)),
        "process_child_parents_pairs": {"X1": []},
        "process_children": ["X1"],
        "child_beta_pairs": {"X1": None},
        "forecast": lambda *args: 0.0,
        "max_lag": 1,
        "n_process_children": 1,
        "sample_precision": np.ones((1, 1)),
        "window": 6,
        "anomaly_threshold": 1.0,
        "consecutive_alarms_threshold": 5,
        "n_consecutive_alarms": 0,
        "fault_onset": 20,
        "anomaly_detected": False,
        "with_plot": False,
    }
    exec(compile(detection_loop, str(notebook_path), "exec"), context)  # noqa: S102

    score_series = pd.Series(scores)
    score_series.iloc[0] = np.nan
    smoothed = score_series.rolling(6).mean()
    alarms = smoothed.gt(1.0).rolling(5).sum().eq(5)
    post_onset_alarms = np.flatnonzero(alarms.to_numpy() & (np.arange(len(scores)) >= 20))
    reference_detection = int(post_onset_alarms[0]) if len(post_onset_alarms) else None
    actual_detection = context["sample_idx"] - 1 if context["anomaly_detected"] else None
    assert actual_detection == reference_detection == expected_detection
    if expected_detection is not None:
        assert context["delay"] == expected_detection - 20
