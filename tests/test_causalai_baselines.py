import unittest
from tempfile import TemporaryDirectory
from pathlib import Path

import numpy as np
import pandas as pd

from carl.causalai_baselines import (
    _aligned_predictions,
    _binary_metrics,
    _fit_baseline,
    _lagged_design,
    _score_runs,
)


class CausalAIBaselineTests(unittest.TestCase):
    def test_lagged_design_uses_only_prior_samples(self):
        trajectory = pd.DataFrame(
            {
                "time_step": [0, 1, 2, 3],
                "signal": [10.0, 11.0, 12.0, 13.0],
            }
        )

        x, y = _lagged_design(trajectory, ["signal"], "signal", max_lag=2)

        np.testing.assert_array_equal(y, [12.0, 13.0])
        np.testing.assert_array_equal(
            x,
            [
                [1.0, 11.0, 10.0],
                [1.0, 12.0, 11.0],
            ],
        )

    def test_alarm_alignment_trims_startup_and_requires_consecutive_breaches(self):
        scores = pd.DataFrame(
            {
                "run_id": ["run"] * 5,
                "time_step": range(5),
                "score": [0.0, 0.0, 2.0, 2.0, 0.0],
                "anomaly": [0, 0, 1, 1, 1],
            }
        )

        aligned, _ = _aligned_predictions(
            scores,
            max_window=2,
            window=1,
            consecutive_alarms=2,
            threshold=1.0,
        )

        result = aligned["run"]
        np.testing.assert_array_equal(result["time_step"], [1, 2, 3, 4])
        np.testing.assert_array_equal(result["alarm"], [False, False, True, False])

    def test_binary_metrics_return_zero_when_no_positive_prediction(self):
        metrics = _binary_metrics([False, True], [False, False])

        self.assertEqual(metrics["TPR"], 0.0)
        self.assertEqual(metrics["FPR"], 0.0)
        self.assertEqual(metrics["F1"], 0.0)

    def test_baseline_scores_endogenous_targets_with_all_predictors(self):
        rng = np.random.default_rng(42)
        rows = []
        for run_id in ("train-a", "train-b", "test"):
            input_signal = rng.normal(size=24)
            state_a = np.zeros(24)
            state_b = np.zeros(24)
            for step in range(1, 24):
                state_a[step] = 0.4 * state_a[step - 1] + 0.3 * input_signal[step - 1]
                state_b[step] = 0.2 * state_a[step - 1] + 0.5 * state_b[step - 1]
            for step in range(24):
                rows.append(
                    {
                        "run_id": run_id,
                        "time_step": step,
                        "time_seconds": float(step),
                        "anomaly": int(run_id == "test" and step >= 12),
                        "input": input_signal[step],
                        "state_a": state_a[step],
                        "state_b": state_b[step],
                    }
                )
        traces = pd.DataFrame(rows)
        runs = pd.DataFrame(
            [
                {"run_id": "train-a", "split": "train", "fault_type": "healthy"},
                {"run_id": "train-b", "split": "train", "fault_type": "healthy"},
                {"run_id": "test", "split": "test", "fault_type": "fault"},
            ]
        )
        features = ["input", "state_a", "state_b"]
        score_features = ["state_a", "state_b"]

        with TemporaryDirectory() as temporary_directory:
            model_dir = Path(temporary_directory)
            model, precision = _fit_baseline(
                "full_var",
                traces,
                runs,
                features,
                max_lag=1,
                model_dir=model_dir,
                score_features=score_features,
            )
            scores = _score_runs(
                "full_var",
                traces,
                ["test"],
                features,
                max_lag=1,
                model_dir=model_dir,
                model=model,
                precision=precision,
                score_features=score_features,
            )

        self.assertEqual(set(model), set(score_features))
        self.assertEqual(precision.shape, (2, 2))
        self.assertEqual(len(scores), 23)
        self.assertEqual(model["state_a"].size, 1 + len(features))


if __name__ == "__main__":
    unittest.main()
