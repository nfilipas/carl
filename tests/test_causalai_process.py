from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from carl.causalai_process import (
    EDGE_SEVERITY_RANGE,
    FAULT_SEVERITY_RANGE,
    FAULTS,
    MAX_AUTOREGRESSIVE_COEFFICIENT,
    RUN_INNOVATION_SCALE_RANGE,
    VARIABLE_NAMES,
    _sem_graph,
    build_sem,
    generate_dataset,
)


class CausalAIProcessTests(unittest.TestCase):
    def test_sem_and_exported_graph_have_same_edges(self):
        sem, mechanisms = build_sem()
        edges = [
            {
                "source": parent,
                "target": target,
                "lag_steps": -lag,
                "coefficient": coefficient,
                "function": function.__name__,
                "mechanism": mechanisms[(parent, target, -lag)],
            }
            for target, parents in sem.items()
            for (parent, lag), coefficient, function in parents
        ]
        expected = pd.DataFrame(edges)
        graph = _sem_graph(
            sem,
            {
                target: [(parent, lag) for (parent, lag), _, _ in parents]
                for target, parents in sem.items()
            },
            mechanisms,
        )
        pd.testing.assert_frame_equal(
            graph.sort_values(["source", "target", "lag_steps"]).reset_index(drop=True),
            expected.sort_values(["source", "target", "lag_steps"]).reset_index(drop=True),
        )
        self.assertEqual(set(sem), set(VARIABLE_NAMES))
        self.assertIn("tanh", set(graph["function"]))
        self.assertGreater(graph["lag_steps"].max(), 1)

    def test_causalai_generates_labelled_fixed_rate_runs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = generate_dataset(
                output_dir=Path(temp_dir) / "dataset",
                runs_per_class=3,
                samples_per_run=60,
                seed=5,
            )
            traces = pd.read_parquet(output_dir / "traces.parquet")
            runs = pd.read_csv(output_dir / "runs.csv")
            graph = pd.read_csv(output_dir / "causal_graph.csv")
            coefficients = pd.read_csv(output_dir / "fault_coefficients.csv")

        self.assertEqual(len(runs), 12)
        self.assertEqual(len(traces), 12 * 60)
        self.assertEqual(set(runs["fault_type"]), {"healthy", *FAULTS})
        self.assertEqual(set(runs["split"]), {"train", "validation", "test"})
        self.assertTrue(
            runs["innovation_scale"].between(*RUN_INNOVATION_SCALE_RANGE).all()
        )
        self.assertTrue(np.isfinite(traces.select_dtypes("number")).all().all())
        self.assertTrue((traces["feed_concentration"] > 0).all())
        self.assertTrue((traces["reactor_concentration"] > 0).all())
        self.assertTrue(traces["product_quality"].between(0, 1, inclusive="neither").all())
        self.assertTrue((graph["lag_steps"] > 0).all())
        self.assertEqual(len(coefficients), 9 * len(graph))
        self.assertEqual(coefficients["run_id"].nunique(), 9)
        self.assertTrue(
            (
                coefficients["coefficient_before"]
                != coefficients["coefficient_after"]
            ).all()
        )
        self.assertTrue(
            coefficients.groupby("run_id")["source"]
            .nunique()
            .eq(len(VARIABLE_NAMES))
            .all()
        )
        self.assertTrue(
            (coefficients["coefficient_after"].abs() <= MAX_AUTOREGRESSIVE_COEFFICIENT).all()
        )
        self.assertTrue(
            coefficients["requested_change_fraction"].between(
            FAULT_SEVERITY_RANGE[0] * EDGE_SEVERITY_RANGE[0],
            FAULT_SEVERITY_RANGE[1] * EDGE_SEVERITY_RANGE[1],
            ).all()
        )
        edge_columns = ["source", "target", "lag_steps"]
        expected_edges = set(map(tuple, graph[edge_columns].to_numpy()))
        for _fault_type, fault_runs in runs.loc[
            runs["fault_type"] != "healthy"
        ].groupby("fault_type"):
            run_ids = fault_runs["run_id"].tolist()
            first = coefficients.loc[
                coefficients["run_id"] == run_ids[0]
            ].sort_values(edge_columns)["coefficient_after"].to_numpy()
            second = coefficients.loc[
                coefficients["run_id"] == run_ids[1]
            ].sort_values(edge_columns)["coefficient_after"].to_numpy()
            self.assertFalse(np.array_equal(first, second))

        for run in runs.itertuples():
            trajectory = traces.loc[traces["run_id"] == run.run_id]
            np.testing.assert_array_equal(
                trajectory["time_step"].to_numpy(),
                np.arange(60),
            )
            np.testing.assert_array_equal(
                trajectory["anomaly"].to_numpy(),
                (
                    np.zeros(60, dtype=int)
                    if run.fault_start_step < 0
                    else (np.arange(60) >= run.fault_start_step).astype(int)
                ),
            )
            if run.fault_type != "healthy":
                self.assertEqual(run.fault_start_step, 60 // 5)
                changes = coefficients.loc[coefficients["run_id"] == run.run_id]
                self.assertEqual(len(changes), len(graph))
                self.assertTrue(
                    FAULT_SEVERITY_RANGE[0]
                    <= run.fault_severity
                    <= FAULT_SEVERITY_RANGE[1]
                )
                self.assertEqual(
                    set(map(tuple, changes[edge_columns].to_numpy())),
                    expected_edges,
                )
                self.assertTrue((changes["change_fraction"] > 0).all())
                changes = changes.merge(
                    graph[edge_columns + ["mechanism"]],
                    on=edge_columns,
                    validate="one_to_one",
                )
                for edge in changes.itertuples():
                    direction = FAULTS[run.fault_type]["profile"][edge.mechanism]
                    relative_change = (
                        edge.coefficient_after / edge.coefficient_before
                    ) - 1
                    self.assertGreater(relative_change * direction, 0)
                self.assertTrue(
                    trajectory.iloc[run.fault_start_step :][
                        list(VARIABLE_NAMES)
                    ].std().gt(0).all()
                )
            else:
                self.assertEqual(run.fault_start_step, -1)
                self.assertFalse((coefficients["run_id"] == run.run_id).any())


if __name__ == "__main__":
    unittest.main()
