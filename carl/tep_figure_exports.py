from pathlib import Path
import re


def save_tep_figure_metrics(metrics, project_root, method, partition, experiment):
    """Persist one TEP evaluation table for the publication-figure workflow."""
    if partition not in {"faulty_test", "healthy_test"}:
        raise ValueError(f"Unsupported TEP evaluation partition: {partition}")

    output_dir = Path(project_root) / "reports" / "results" / "TEP" / "figure_metrics"
    output_dir.mkdir(parents=True, exist_ok=True)

    exported = metrics.copy()
    exported.insert(0, "Experiment", experiment)
    exported.insert(0, "Partition", partition)
    exported.insert(0, "Method", method)

    method_slug = re.sub(r"[^a-z0-9]+", "_", method.lower()).strip("_")
    output_path = output_dir / f"{experiment}_{method_slug}_{partition}.csv"
    exported.to_csv(output_path, index=False)
    print(f"Saved TEP figure metrics to {output_path}")
