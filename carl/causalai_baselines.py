"""Baseline anomaly-detection experiments for the synthetic process dataset."""

from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from carl.causalai_process import ENDOGENOUS_NAMES, VARIABLE_NAMES

BASELINE_NAMES = ("full_var", "pca", "self_history")
MIN_WINDOW = 1
MAX_WINDOW = 8
MIN_CONSECUTIVE_ALARMS = 5
MAX_CONSECUTIVE_ALARMS = 8


def _lagged_design(trajectory, features, child, max_lag):
    """Build a lagged linear-regression design without crossing run boundaries."""
    feature_values = trajectory[features]
    lagged_values = [
        feature_values.shift(lag).to_numpy()[max_lag:]
        for lag in range(1, max_lag + 1)
    ]
    x = np.column_stack(lagged_values)
    x = np.column_stack((np.ones(len(x)), x))
    y = trajectory[child].to_numpy()[max_lag:]
    return x, y


def _fit_baseline(
    method,
    traces,
    runs,
    features,
    max_lag,
    model_dir,
    score_mode="mahalanobis",
    score_features=None,
):
    score_features = features if score_features is None else score_features
    if not score_features or not set(score_features).issubset(features):
        raise ValueError("Scored variables must be a non-empty subset of model variables.")
    train_ids = runs.loc[
        runs["split"].eq("train") & runs["fault_type"].eq("healthy"), "run_id"
    ]
    healthy_training = traces.loc[traces["run_id"].isin(train_ids)]
    if healthy_training.empty:
        raise ValueError("No healthy training trajectories are available.")

    model_dir.mkdir(parents=True, exist_ok=True)
    if method == "pca":
        full_model = PCA().fit(healthy_training[score_features])
        retained_components = min(
            int(np.searchsorted(np.cumsum(full_model.explained_variance_ratio_), 0.95) + 1),
            len(score_features) - 1,
        )
        if retained_components < 1:
            raise ValueError("PCA requires at least two process variables.")
        model = PCA(n_components=retained_components)
        model.fit(healthy_training[score_features])
        with (model_dir / "pca_model.pkl").open("wb") as stream:
            pickle.dump(model, stream)
        return model, None

    if method == "full_var":
        regression_features = {child: features for child in score_features}
    elif method == "self_history":
        regression_features = {child: [child] for child in score_features}
    else:
        raise ValueError(f"Unknown baseline method: {method}")

    coefficients = {}
    residuals_by_run = {run_id: {} for run_id in train_ids}
    for child in score_features:
        training_designs = []
        training_targets = []
        for run_id, trajectory in healthy_training.groupby("run_id", sort=False):
            trajectory = trajectory.sort_values("time_step")
            x_train, y_train = _lagged_design(
                trajectory, regression_features[child], child, max_lag
            )
            training_designs.append(x_train)
            training_targets.append(y_train)
        x_train = np.concatenate(training_designs, axis=0)
        y_train = np.concatenate(training_targets)
        beta, *_ = np.linalg.lstsq(x_train, y_train, rcond=None)
        coefficients[child] = beta
        np.save(model_dir / f"beta_child_{child}_lag_{max_lag}.npy", beta)

        for run_id, trajectory in healthy_training.groupby("run_id", sort=False):
            trajectory = trajectory.sort_values("time_step")
            x_train, y_train = _lagged_design(
                trajectory, regression_features[child], child, max_lag
            )
            residuals_by_run[run_id][child] = y_train - x_train @ beta

    residuals = np.concatenate(
        [
            np.column_stack([residuals_by_run[run_id][child] for child in score_features])
            for run_id in train_ids
        ],
        axis=0,
    )
    if method == "self_history" or score_mode == "diagonal":
        residual_variance = np.mean(residuals**2, axis=0)
        if np.any(residual_variance <= 0):
            raise ValueError("Self-history model produced a non-positive residual variance.")
        precision = np.diag(1 / residual_variance)
    elif score_mode == "plain":
        precision = None
    else:
        covariance = (residuals.T @ residuals) / len(residuals)
        precision = np.linalg.pinv(covariance)
    if precision is not None:
        np.save(model_dir / "sample_precision_train.npy", precision)
    return coefficients, precision


def _score_runs(
    method,
    traces,
    run_ids,
    features,
    max_lag,
    model_dir,
    model,
    precision,
    score_mode="mahalanobis",
    score_features=None,
):
    score_features = features if score_features is None else score_features
    if not score_features or not set(score_features).issubset(features):
        raise ValueError("Scored variables must be a non-empty subset of model variables.")
    selected = traces.loc[traces["run_id"].isin(run_ids)]
    rows = []
    for run_id, trajectory in selected.groupby("run_id", sort=False):
        trajectory = trajectory.sort_values("time_step")
        values = trajectory[score_features].to_numpy()
        if method == "pca":
            feature_frame = trajectory[score_features]
            reconstructed = model.inverse_transform(model.transform(feature_frame))
            scores = np.sum((values - reconstructed) ** 2, axis=1)
            start = 0
        else:
            residuals = np.empty((len(trajectory) - max_lag, len(score_features)))
            for child_idx, child in enumerate(score_features):
                beta_path = model_dir / f"beta_child_{child}_lag_{max_lag}.npy"
                if not beta_path.is_file():
                    raise FileNotFoundError(f"Missing fitted model weights: {beta_path}")
                beta = model[child]
                regression_features = features if method == "full_var" else [child]
                x, y = _lagged_design(trajectory, regression_features, child, max_lag)
                residuals[:, child_idx] = y - x @ beta
            if score_mode == "plain":
                scores = np.mean(residuals**2, axis=1)
            else:
                scores = np.mean((residuals @ precision) * residuals, axis=1)
            start = max_lag

        scored = trajectory.iloc[start:]
        rows.append(
            pd.DataFrame(
                {
                    "run_id": run_id,
                    "time_step": scored["time_step"].to_numpy(),
                    "time_seconds": scored["time_seconds"].to_numpy(),
                    "anomaly": scored["anomaly"].to_numpy(dtype=int),
                    "score": scores,
                }
            )
        )
    if not rows:
        raise ValueError(f"No trajectories are available to score for {method}.")
    return pd.concat(rows, ignore_index=True)


def _binary_metrics(actual, predicted):
    actual = np.asarray(actual, dtype=bool)
    predicted = np.asarray(predicted, dtype=bool)
    tp = int(np.sum(actual & predicted))
    fp = int(np.sum(~actual & predicted))
    fn = int(np.sum(actual & ~predicted))
    tn = int(np.sum(~actual & ~predicted))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "Accuracy": (tp + tn) / (tp + fp + fn + tn) if tp + fp + fn + tn else 0.0,
        "TPR": recall,
        "FPR": fp / (fp + tn) if fp + tn else 0.0,
        "F1": f1,
    }


def _aligned_predictions(scores, max_window, window, consecutive_alarms, threshold):
    predictions = {}
    aligned_rows = {}
    for run_id, trajectory in scores.groupby("run_id", sort=False):
        trajectory = trajectory.sort_values("time_step").copy()
        trajectory["smoothed_score"] = trajectory["score"].rolling(
            window=window, min_periods=window
        ).mean()
        trajectory = trajectory.iloc[max_window - 1 :].copy()
        trajectory["alarm"] = (
            trajectory["smoothed_score"]
            .gt(threshold)
            .rolling(window=consecutive_alarms, min_periods=consecutive_alarms)
            .sum()
            .eq(consecutive_alarms)
            .fillna(False)
            .to_numpy(dtype=bool)
        )
        predictions[run_id] = trajectory["alarm"].to_numpy(dtype=bool)
        aligned_rows[run_id] = trajectory
    return aligned_rows, predictions


def _evaluate(scores, run_manifest, max_window, window, consecutive_alarms, threshold):
    aligned_rows, _ = _aligned_predictions(
        scores, max_window, window, consecutive_alarms, threshold
    )
    sample_actual = []
    sample_predicted = []
    phase_actual = []
    phase_predicted = []
    healthy_run_alarms = []
    detected_delays = []
    per_run_rows = []

    for run_id, trajectory in aligned_rows.items():
        manifest = run_manifest.loc[run_id]
        steps = trajectory["time_step"].to_numpy()
        alarms = trajectory["alarm"].to_numpy(dtype=bool)
        labels = trajectory["anomaly"].to_numpy(dtype=bool)

        sample_mask = np.arange(len(trajectory)) >= consecutive_alarms - 1
        sample_actual.extend(labels[sample_mask])
        sample_predicted.extend(alarms[sample_mask])
        run_detected = False
        delay = np.nan

        if manifest["fault_type"] == "healthy":
            healthy_run_alarms.append(bool(alarms.any()))
        else:
            onset = int(manifest["fault_start_step"])
            healthy_phase = steps < onset
            faulty_phase = steps >= onset
            healthy_alarm = bool(alarms[healthy_phase].any())
            faulty_alarm = bool(alarms[faulty_phase].any())
            phase_actual.extend((False, True))
            phase_predicted.extend((healthy_alarm, faulty_alarm))
            if faulty_alarm:
                detection_step = int(steps[faulty_phase][alarms[faulty_phase]][0])
                delay = detection_step - onset
                run_detected = True
                detected_delays.append(delay)

        per_run_rows.append(
            {
                "run_id": run_id,
                "fault_type": manifest["fault_type"],
                "fault_start_step": manifest["fault_start_step"],
                "fault_detected": run_detected,
                "detection_delay_samples": delay,
                "false_alarm": bool(alarms.any()) if manifest["fault_type"] == "healthy" else False,
            }
        )

    metrics = _binary_metrics(sample_actual, sample_predicted)
    metrics = {f"Sample {key}": value for key, value in metrics.items()}
    if phase_actual:
        phase_metrics = _binary_metrics(phase_actual, phase_predicted)
        metrics.update({f"Trajectory-Phase {key}": value for key, value in phase_metrics.items()})
        metrics["Detected Runs"] = int(sum(row["fault_detected"] for row in per_run_rows))
        metrics["Total Faulty Runs"] = len(phase_actual) // 2
        metrics["Median Anomaly Detection Delay"] = (
            float(np.median(detected_delays)) if detected_delays else np.nan
        )
    else:
        metrics["Trajectory-Phase FPR"] = (
            float(np.mean(healthy_run_alarms)) if healthy_run_alarms else np.nan
        )
        metrics["False Alarm Runs"] = int(sum(healthy_run_alarms))
        metrics["Total Healthy Runs"] = len(healthy_run_alarms)
    return metrics, pd.DataFrame(per_run_rows), aligned_rows


def _select_alarm_settings(validation_scores, validation_manifest):
    thresholds = []
    grid_rows = []
    for window in range(MIN_WINDOW, MAX_WINDOW + 1):
        calibration_scores = []
        for run_id, trajectory in validation_scores.groupby("run_id", sort=False):
            onset = int(validation_manifest.loc[run_id, "fault_start_step"])
            smoothed = trajectory.sort_values("time_step")["score"].rolling(
                window=window, min_periods=window
            ).mean()
            smoothed = smoothed.iloc[MAX_WINDOW - 1 :]
            steps = trajectory.sort_values("time_step")["time_step"].to_numpy()[
                MAX_WINDOW - 1 :
            ]
            pre_fault_scores = smoothed.to_numpy()[steps < onset]
            calibration_scores.extend(pre_fault_scores[np.isfinite(pre_fault_scores)])
        if not calibration_scores:
            raise ValueError(f"No pre-fault validation scores are available for window {window}.")
        threshold = float(np.max(calibration_scores))
        thresholds.append({"Window Size": window, "Threshold": threshold})

        for consecutive_alarms in range(
            MIN_CONSECUTIVE_ALARMS, MAX_CONSECUTIVE_ALARMS + 1
        ):
            metrics, _, _ = _evaluate(
                validation_scores,
                validation_manifest,
                MAX_WINDOW,
                window,
                consecutive_alarms,
                threshold,
            )
            grid_rows.append(
                {
                    "Window Size": window,
                    "Consecutive Alarms": consecutive_alarms,
                    "Threshold": threshold,
                    **metrics,
                }
            )

    grid = pd.DataFrame(grid_rows)
    best_f1 = grid["Trajectory-Phase F1"].max()
    best_candidates = grid.loc[
        grid["Trajectory-Phase F1"].eq(best_f1)
    ].dropna(subset=["Median Anomaly Detection Delay"])
    if best_candidates.empty:
        raise ValueError("No validation configuration has a finite detection delay.")
    best = best_candidates.loc[
        best_candidates["Median Anomaly Detection Delay"].idxmin()
    ]
    thresholds = pd.DataFrame(thresholds)
    return grid, thresholds, best


def run_baseline_comparison(method, project_root):
    """Fit, validation-tune, and test one synthetic-process detection baseline."""
    if method not in BASELINE_NAMES:
        raise ValueError(f"method must be one of {BASELINE_NAMES}, not {method!r}.")
    project_root = Path(project_root)
    data_path = project_root / "data" / "interim" / "causalAIProcess" / "traces_scaled.parquet"
    runs_path = project_root / "data" / "processed" / "causalAIProcess" / "runs.csv"
    lag_path = (
        project_root
        / "models"
        / "causalAIProcess"
        / "max_lag_selection"
        / "selected_max_lag.csv"
    )
    required = (data_path, runs_path, lag_path)
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Run notebooks 0.2 and 1.0 first. Missing: "
            + ", ".join(str(path) for path in missing)
        )

    traces = pd.read_parquet(data_path)
    runs = pd.read_csv(runs_path)
    max_lag = int(pd.read_csv(lag_path).loc[0, "best_max_lag"])
    features = [name for name in VARIABLE_NAMES if name in traces.columns]
    if len(features) != len(VARIABLE_NAMES):
        raise ValueError("Scaled traces do not contain all synthetic process variables.")
    score_features = [name for name in ENDOGENOUS_NAMES if name in features]
    if len(score_features) != len(ENDOGENOUS_NAMES):
        raise ValueError("Scaled traces do not contain all endogenous process variables.")
    if max_lag != 2:
        raise ValueError(f"Expected the selected synthetic-process lag to be 2, got {max_lag}.")
    if traces.groupby("run_id").size().nunique() != 1:
        raise ValueError("All trajectories must have equal length for baseline evaluation.")

    run_manifest = runs.set_index("run_id")
    model_dir = project_root / "models" / "causalAIProcess" / f"anomaly_detection_baseline_{method}"
    results_dir = project_root / "reports" / "results" / "causalAIProcess"
    results_dir.mkdir(parents=True, exist_ok=True)

    model, precision = _fit_baseline(
        method,
        traces,
        runs,
        features,
        max_lag,
        model_dir,
        score_features=score_features,
    )
    validation_ids = runs.loc[
        runs["split"].eq("validation") & runs["fault_type"].ne("healthy"), "run_id"
    ].tolist()
    faulty_test_ids = runs.loc[
        runs["split"].eq("test") & runs["fault_type"].ne("healthy"), "run_id"
    ].tolist()
    healthy_test_ids = runs.loc[
        runs["split"].eq("test") & runs["fault_type"].eq("healthy"), "run_id"
    ].tolist()
    if not validation_ids or not faulty_test_ids or not healthy_test_ids:
        raise ValueError("Validation, faulty test, and healthy test runs are all required.")

    validation_scores = _score_runs(
        method,
        traces,
        validation_ids,
        features,
        max_lag,
        model_dir,
        model,
        precision,
        score_features=score_features,
    )
    grid, thresholds, best = _select_alarm_settings(
        validation_scores, run_manifest.loc[validation_ids]
    )
    window = int(best["Window Size"])
    consecutive_alarms = int(best["Consecutive Alarms"])
    threshold = float(thresholds.set_index("Window Size").loc[window, "Threshold"])
    settings = pd.DataFrame(
        [
            {
                "Baseline": method,
                "Max Lag": max_lag,
                "Window Size": window,
                "Consecutive Alarms": consecutive_alarms,
                "Threshold": threshold,
                "PCA Components": getattr(model, "n_components_", np.nan),
            }
        ]
    )

    faulty_scores = _score_runs(
        method,
        traces,
        faulty_test_ids,
        features,
        max_lag,
        model_dir,
        model,
        precision,
        score_features=score_features,
    )
    faulty_metrics, faulty_by_run, _ = _evaluate(
        faulty_scores,
        run_manifest.loc[faulty_test_ids],
        MAX_WINDOW,
        window,
        consecutive_alarms,
        threshold,
    )
    faulty_metrics = {
        "Baseline": method,
        "Max Lag": max_lag,
        "Window Size": window,
        "Consecutive Alarms": consecutive_alarms,
        "Threshold": threshold,
        **faulty_metrics,
    }
    faulty_by_type = (
        faulty_by_run.groupby("fault_type", as_index=False)
        .agg(
            Runs=("run_id", "count"),
            Detected_Runs=("fault_detected", "sum"),
            Run_Detection_Rate=("fault_detected", "mean"),
            Median_Detection_Delay=("detection_delay_samples", "median"),
        )
        .rename(columns={"fault_type": "Fault Type"})
    )

    healthy_scores = _score_runs(
        method,
        traces,
        healthy_test_ids,
        features,
        max_lag,
        model_dir,
        model,
        precision,
        score_features=score_features,
    )
    healthy_metrics, healthy_by_run, _ = _evaluate(
        healthy_scores,
        run_manifest.loc[healthy_test_ids],
        MAX_WINDOW,
        window,
        consecutive_alarms,
        threshold,
    )
    healthy_metrics = {
        "Baseline": method,
        "Max Lag": max_lag,
        "Window Size": window,
        "Consecutive Alarms": consecutive_alarms,
        "Threshold": threshold,
        **healthy_metrics,
    }

    prefix = results_dir / f"anomaly_detection_baseline_{method}"
    grid.to_csv(f"{prefix}_validation_grid.csv", index=False)
    thresholds.to_csv(f"{prefix}_thresholds_by_window.csv", index=False)
    settings.to_csv(f"{prefix}_selected_settings.csv", index=False)
    faulty_scores.to_parquet(f"{prefix}_faulty_test_scores.parquet", index=False)
    faulty_by_run.to_csv(f"{prefix}_faulty_test_by_run.csv", index=False)
    pd.DataFrame([faulty_metrics]).to_csv(f"{prefix}_faulty_test_metrics.csv", index=False)
    faulty_by_type.to_csv(f"{prefix}_faulty_test_by_type.csv", index=False)
    healthy_scores.to_parquet(f"{prefix}_healthy_test_scores.parquet", index=False)
    healthy_by_run.to_csv(f"{prefix}_healthy_test_by_run.csv", index=False)
    pd.DataFrame([healthy_metrics]).to_csv(f"{prefix}_healthy_test_metrics.csv", index=False)

    comparison_path = results_dir / "anomaly_detection_baseline_comparison.csv"
    comparison_row = {
        "Baseline": method,
        "Max Lag": max_lag,
        "Window Size": window,
        "Consecutive Alarms": consecutive_alarms,
        "Threshold": threshold,
        "PCA Components": getattr(model, "n_components_", np.nan),
        "Faulty Test Sample F1": faulty_metrics["Sample F1"],
        "Faulty Test Trajectory-Phase F1": faulty_metrics["Trajectory-Phase F1"],
        "Faulty Test Trajectory-Phase TPR": faulty_metrics["Trajectory-Phase TPR"],
        "Faulty Test Median Detection Delay": faulty_metrics[
            "Median Anomaly Detection Delay"
        ],
        "Healthy Test Sample FPR": healthy_metrics["Sample FPR"],
        "Healthy Test Trajectory-Phase FPR": healthy_metrics["Trajectory-Phase FPR"],
    }
    if comparison_path.is_file():
        previous = pd.read_csv(comparison_path)
        previous = previous.loc[previous["Baseline"].ne(method)]
        comparison = pd.concat(
            [previous, pd.DataFrame([comparison_row])], ignore_index=True
        )
    else:
        comparison = pd.DataFrame([comparison_row])
    comparison = comparison.sort_values("Baseline")
    comparison.to_csv(comparison_path, index=False)

    return {
        "validation_grid": grid,
        "selected_settings": settings,
        "faulty_test_metrics": pd.DataFrame([faulty_metrics]),
        "faulty_test_by_type": faulty_by_type,
        "healthy_test_metrics": pd.DataFrame([healthy_metrics]),
        "comparison": comparison,
    }


def run_ablation_comparison(ablation, project_root):
    """Run one TEP-style detection ablation using fixed 6-sample/5-alarm settings."""
    ablation_config = {
        "full_var": ("full_var", "mahalanobis"),
        "pca": ("pca", "spe"),
        "self_history": ("self_history", "diagonal"),
        "diagonal_covariance": ("full_var", "diagonal"),
        "plain_residuals": ("full_var", "plain"),
    }
    if ablation not in ablation_config:
        raise ValueError(
            f"ablation must be one of {tuple(ablation_config)}, not {ablation!r}."
        )
    method, score_mode = ablation_config[ablation]
    project_root = Path(project_root)
    data_path = project_root / "data" / "interim" / "causalAIProcess" / "traces_scaled.parquet"
    runs_path = project_root / "data" / "processed" / "causalAIProcess" / "runs.csv"
    lag_path = (
        project_root
        / "models"
        / "causalAIProcess"
        / "max_lag_selection"
        / "selected_max_lag.csv"
    )
    required = (data_path, runs_path, lag_path)
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Run notebooks 0.2 and 1.0 first. Missing: "
            + ", ".join(str(path) for path in missing)
        )

    traces = pd.read_parquet(data_path)
    runs = pd.read_csv(runs_path)
    max_lag = int(pd.read_csv(lag_path).loc[0, "best_max_lag"])
    features = [name for name in VARIABLE_NAMES if name in traces.columns]
    if len(features) != len(VARIABLE_NAMES):
        raise ValueError("Scaled traces do not contain all synthetic process variables.")
    score_features = [name for name in ENDOGENOUS_NAMES if name in features]
    if len(score_features) != len(ENDOGENOUS_NAMES):
        raise ValueError("Scaled traces do not contain all endogenous process variables.")
    if max_lag != 2:
        raise ValueError(f"Expected the selected synthetic-process lag to be 2, got {max_lag}.")
    if traces.groupby("run_id").size().nunique() != 1:
        raise ValueError("All trajectories must have equal length for ablation evaluation.")

    window = 6
    consecutive_alarms = 5
    run_manifest = runs.set_index("run_id")
    model_dir = (
        project_root
        / "models"
        / "causalAIProcess"
        / f"anomaly_detection_ablation_{ablation}"
    )
    results_dir = project_root / "reports" / "results" / "causalAIProcess"
    results_dir.mkdir(parents=True, exist_ok=True)
    model, precision = _fit_baseline(
        method,
        traces,
        runs,
        features,
        max_lag,
        model_dir,
        score_mode=score_mode,
        score_features=score_features,
    )

    validation_ids = runs.loc[
        runs["split"].eq("validation") & runs["fault_type"].ne("healthy"), "run_id"
    ].tolist()
    faulty_test_ids = runs.loc[
        runs["split"].eq("test") & runs["fault_type"].ne("healthy"), "run_id"
    ].tolist()
    healthy_test_ids = runs.loc[
        runs["split"].eq("test") & runs["fault_type"].eq("healthy"), "run_id"
    ].tolist()
    if not validation_ids or not faulty_test_ids or not healthy_test_ids:
        raise ValueError("Validation, faulty test, and healthy test runs are all required.")

    validation_scores = _score_runs(
        method,
        traces,
        validation_ids,
        features,
        max_lag,
        model_dir,
        model,
        precision,
        score_mode=score_mode,
        score_features=score_features,
    )
    calibration_scores = []
    for run_id, trajectory in validation_scores.groupby("run_id", sort=False):
        trajectory = trajectory.sort_values("time_step")
        smoothed = trajectory["score"].rolling(
            window=window, min_periods=window
        ).mean()
        pre_fault = trajectory["time_step"].to_numpy() < int(
            run_manifest.loc[run_id, "fault_start_step"]
        )
        values = smoothed.to_numpy()[pre_fault]
        calibration_scores.extend(values[np.isfinite(values)])
    if not calibration_scores:
        raise ValueError("No healthy-phase validation scores are available for threshold selection.")
    threshold = float(np.max(calibration_scores))
    validation_metrics, validation_by_run, _ = _evaluate(
        validation_scores,
        run_manifest.loc[validation_ids],
        window,
        window,
        consecutive_alarms,
        threshold,
    )
    selected_settings = pd.DataFrame(
        [
            {
                "Ablation": ablation,
                "Score": score_mode,
                "Max Lag": max_lag,
                "Window Size": window,
                "Consecutive Alarms": consecutive_alarms,
                "Threshold": threshold,
            }
        ]
    )
    validation_metrics = {
        "Ablation": ablation,
        "Score": score_mode,
        "Max Lag": max_lag,
        "Window Size": window,
        "Consecutive Alarms": consecutive_alarms,
        "Threshold": threshold,
        **validation_metrics,
    }

    faulty_scores = _score_runs(
        method,
        traces,
        faulty_test_ids,
        features,
        max_lag,
        model_dir,
        model,
        precision,
        score_mode=score_mode,
        score_features=score_features,
    )
    faulty_metrics, faulty_by_run, _ = _evaluate(
        faulty_scores,
        run_manifest.loc[faulty_test_ids],
        window,
        window,
        consecutive_alarms,
        threshold,
    )
    faulty_metrics = {
        "Ablation": ablation,
        "Score": score_mode,
        "Max Lag": max_lag,
        "Window Size": window,
        "Consecutive Alarms": consecutive_alarms,
        "Threshold": threshold,
        **faulty_metrics,
    }
    faulty_by_type = (
        faulty_by_run.groupby("fault_type", as_index=False)
        .agg(
            Runs=("run_id", "count"),
            Detected_Runs=("fault_detected", "sum"),
            Run_Detection_Rate=("fault_detected", "mean"),
            Median_Detection_Delay=("detection_delay_samples", "median"),
        )
        .rename(columns={"fault_type": "Fault Type"})
    )

    healthy_scores = _score_runs(
        method,
        traces,
        healthy_test_ids,
        features,
        max_lag,
        model_dir,
        model,
        precision,
        score_mode=score_mode,
        score_features=score_features,
    )
    healthy_metrics, healthy_by_run, _ = _evaluate(
        healthy_scores,
        run_manifest.loc[healthy_test_ids],
        window,
        window,
        consecutive_alarms,
        threshold,
    )
    healthy_metrics = {
        "Ablation": ablation,
        "Score": score_mode,
        "Max Lag": max_lag,
        "Window Size": window,
        "Consecutive Alarms": consecutive_alarms,
        "Threshold": threshold,
        **healthy_metrics,
    }

    prefix = results_dir / f"anomaly_detection_ablation_{ablation}"
    selected_settings.to_csv(f"{prefix}_selected_settings.csv", index=False)
    pd.DataFrame([validation_metrics]).to_csv(
        f"{prefix}_validation_metrics.csv", index=False
    )
    validation_by_run.to_csv(f"{prefix}_validation_by_run.csv", index=False)
    validation_scores.to_parquet(f"{prefix}_validation_scores.parquet", index=False)
    faulty_scores.to_parquet(f"{prefix}_faulty_test_scores.parquet", index=False)
    faulty_by_run.to_csv(f"{prefix}_faulty_test_by_run.csv", index=False)
    pd.DataFrame([faulty_metrics]).to_csv(
        f"{prefix}_faulty_test_metrics.csv", index=False
    )
    faulty_by_type.to_csv(f"{prefix}_faulty_test_by_type.csv", index=False)
    healthy_scores.to_parquet(f"{prefix}_healthy_test_scores.parquet", index=False)
    healthy_by_run.to_csv(f"{prefix}_healthy_test_by_run.csv", index=False)
    pd.DataFrame([healthy_metrics]).to_csv(
        f"{prefix}_healthy_test_metrics.csv", index=False
    )

    comparison_path = results_dir / "anomaly_detection_ablation_comparison.csv"
    comparison_row = {
        "Ablation": ablation,
        "Score": score_mode,
        "Max Lag": max_lag,
        "Window Size": window,
        "Consecutive Alarms": consecutive_alarms,
        "Threshold": threshold,
        "Faulty Test Sample F1": faulty_metrics["Sample F1"],
        "Faulty Test Trajectory-Phase F1": faulty_metrics[
            "Trajectory-Phase F1"
        ],
        "Faulty Test Trajectory-Phase TPR": faulty_metrics[
            "Trajectory-Phase TPR"
        ],
        "Faulty Test Median Detection Delay": faulty_metrics[
            "Median Anomaly Detection Delay"
        ],
        "Healthy Test Sample FPR": healthy_metrics["Sample FPR"],
        "Healthy Test Trajectory-Phase FPR": healthy_metrics[
            "Trajectory-Phase FPR"
        ],
    }
    if comparison_path.is_file():
        comparison = pd.read_csv(comparison_path)
        comparison = comparison.loc[comparison["Ablation"].ne(ablation)]
        comparison = pd.concat(
            [comparison, pd.DataFrame([comparison_row])], ignore_index=True
        )
    else:
        comparison = pd.DataFrame([comparison_row])
    comparison = comparison.sort_values("Ablation")
    comparison.to_csv(comparison_path, index=False)

    return {
        "selected_settings": selected_settings,
        "validation_metrics": pd.DataFrame([validation_metrics]),
        "faulty_test_metrics": pd.DataFrame([faulty_metrics]),
        "faulty_test_by_type": faulty_by_type,
        "healthy_test_metrics": pd.DataFrame([healthy_metrics]),
        "comparison": comparison,
    }


def run_alarm_ablation(ablation, project_root):
    """Evaluate one alarm-logic ablation with the full-variable baseline score."""
    configurations = {
        "threshold_only": "raw",
        "moving_average_only": "moving_average",
    }
    if ablation not in configurations:
        raise ValueError(f"ablation must be one of {tuple(configurations)}, not {ablation!r}.")

    project_root = Path(project_root)
    results_dir = project_root / "reports" / "results" / "causalAIProcess"
    runs_path = project_root / "data" / "processed" / "causalAIProcess" / "runs.csv"
    baseline_prefix = results_dir / "anomaly_detection_baseline_full_var"
    settings_path = Path(f"{baseline_prefix}_selected_settings.csv")
    faulty_scores_path = Path(f"{baseline_prefix}_faulty_test_scores.parquet")
    healthy_scores_path = Path(f"{baseline_prefix}_healthy_test_scores.parquet")
    required = (runs_path, settings_path, faulty_scores_path, healthy_scores_path)
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Run notebook 3.0-anomaly_detection_baseline_full_var.ipynb first. Missing: "
            + ", ".join(str(path) for path in missing)
        )

    runs = pd.read_csv(runs_path)
    run_manifest = runs.set_index("run_id")
    settings = pd.read_csv(settings_path)
    if len(settings) != 1:
        raise ValueError(f"Expected one full-variable baseline settings row, got {len(settings)}.")
    baseline_settings = settings.iloc[0]
    window = (
        1
        if configurations[ablation] == "raw"
        else int(baseline_settings["Window Size"])
    )
    consecutive_alarms = 1
    threshold = float(baseline_settings["Threshold"])
    max_window = window

    faulty_ids = runs.loc[
        runs["split"].eq("test") & runs["fault_type"].ne("healthy"), "run_id"
    ].tolist()
    healthy_ids = runs.loc[
        runs["split"].eq("test") & runs["fault_type"].eq("healthy"), "run_id"
    ].tolist()
    faulty_scores = pd.read_parquet(faulty_scores_path)
    healthy_scores = pd.read_parquet(healthy_scores_path)
    if not faulty_ids or not healthy_ids:
        raise ValueError("Faulty and healthy held-out test runs are both required.")
    if set(faulty_scores["run_id"]) != set(faulty_ids):
        raise ValueError("Saved full-variable scores do not cover every faulty test run.")
    if set(healthy_scores["run_id"]) != set(healthy_ids):
        raise ValueError("Saved full-variable scores do not cover every healthy test run.")

    faulty_metrics, faulty_by_run, faulty_aligned = _evaluate(
        faulty_scores,
        run_manifest.loc[faulty_ids],
        max_window,
        window,
        consecutive_alarms,
        threshold,
    )
    healthy_metrics, healthy_by_run, healthy_aligned = _evaluate(
        healthy_scores,
        run_manifest.loc[healthy_ids],
        max_window,
        window,
        consecutive_alarms,
        threshold,
    )
    faulty_metrics = {
        "Alarm Ablation": ablation,
        "Baseline Window Size": int(baseline_settings["Window Size"]),
        "Consecutive Alarms": consecutive_alarms,
        "Threshold": threshold,
        **faulty_metrics,
    }
    healthy_metrics = {
        "Alarm Ablation": ablation,
        "Baseline Window Size": int(baseline_settings["Window Size"]),
        "Consecutive Alarms": consecutive_alarms,
        "Threshold": threshold,
        **healthy_metrics,
    }

    settings_row = {
        "Alarm Ablation": ablation,
        "Baseline Window Size": int(baseline_settings["Window Size"]),
        "Applied Window Size": window,
        "Consecutive Alarms": consecutive_alarms,
        "Threshold": threshold,
    }
    faulty_scores_with_alarms = pd.concat(
        [trajectory for trajectory in faulty_aligned.values()], ignore_index=True
    )
    healthy_scores_with_alarms = pd.concat(
        [trajectory for trajectory in healthy_aligned.values()], ignore_index=True
    )
    prefix = results_dir / f"alarm_ablation_{ablation}"
    pd.DataFrame([settings_row]).to_csv(f"{prefix}_settings.csv", index=False)
    pd.DataFrame([faulty_metrics]).to_csv(
        f"{prefix}_faulty_test_metrics.csv", index=False
    )
    faulty_by_run.to_csv(f"{prefix}_faulty_test_by_run.csv", index=False)
    faulty_scores_with_alarms.to_parquet(
        f"{prefix}_faulty_test_scores.parquet", index=False
    )
    pd.DataFrame([healthy_metrics]).to_csv(
        f"{prefix}_healthy_test_metrics.csv", index=False
    )
    healthy_by_run.to_csv(f"{prefix}_healthy_test_by_run.csv", index=False)
    healthy_scores_with_alarms.to_parquet(
        f"{prefix}_healthy_test_scores.parquet", index=False
    )

    comparison_path = results_dir / "alarm_ablation_comparison.csv"
    comparison_row = {
        "Alarm Ablation": ablation,
        "Applied Window Size": window,
        "Consecutive Alarms": consecutive_alarms,
        "Threshold": threshold,
        "Faulty Test Sample F1": faulty_metrics["Sample F1"],
        "Faulty Test Trajectory-Phase F1": faulty_metrics[
            "Trajectory-Phase F1"
        ],
        "Faulty Test Trajectory-Phase TPR": faulty_metrics[
            "Trajectory-Phase TPR"
        ],
        "Faulty Test Median Detection Delay": faulty_metrics[
            "Median Anomaly Detection Delay"
        ],
        "Healthy Test Sample FPR": healthy_metrics["Sample FPR"],
        "Healthy Test Trajectory-Phase FPR": healthy_metrics[
            "Trajectory-Phase FPR"
        ],
    }
    if comparison_path.is_file():
        comparison = pd.read_csv(comparison_path)
        comparison = comparison.loc[
            comparison["Alarm Ablation"].ne(ablation)
        ]
        comparison = pd.concat(
            [comparison, pd.DataFrame([comparison_row])], ignore_index=True
        )
    else:
        comparison = pd.DataFrame([comparison_row])
    comparison = comparison.sort_values("Alarm Ablation")
    comparison.to_csv(comparison_path, index=False)

    return {
        "settings": pd.DataFrame([settings_row]),
        "faulty_test_metrics": pd.DataFrame([faulty_metrics]),
        "faulty_test_by_run": faulty_by_run,
        "healthy_test_metrics": pd.DataFrame([healthy_metrics]),
        "healthy_test_by_run": healthy_by_run,
        "comparison": comparison,
    }
