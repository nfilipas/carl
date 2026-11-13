"""Generate a compact process benchmark with Salesforce CausalAI."""

from importlib.metadata import version
import json
from pathlib import Path

import numpy as np
import pandas as pd

VARIABLE_NAMES = (
    "feed_flow",
    "feed_concentration",
    "coolant_flow",
    "reactor_concentration",
    "reactor_temperature",
    "product_quality",
)
MANIPULATED_NAMES = (
    "feed_flow",
    "feed_concentration",
    "coolant_flow",
)
ENDOGENOUS_NAMES = tuple(
    name for name in VARIABLE_NAMES if name not in MANIPULATED_NAMES
)
FAULTS = {
    "hydraulic_coupling_shift": {
        "profile": {
            "input_persistence": 1,
            "state_persistence": -1,
            "feed_mixing": 1,
            "hydraulic_residence": -1,
            "reaction_kinetics": 1,
            "reaction_heat": -1,
            "heat_exchange": 1,
            "conversion": -1,
            "temperature_sensitivity": 1,
        },
    },
    "thermal_coupling_shift": {
        "profile": {
            "input_persistence": -1,
            "state_persistence": 1,
            "feed_mixing": -1,
            "hydraulic_residence": 1,
            "reaction_kinetics": -1,
            "reaction_heat": 1,
            "heat_exchange": -1,
            "conversion": 1,
            "temperature_sensitivity": -1,
        },
    },
    "reaction_coupling_shift": {
        "profile": {
            "input_persistence": 1,
            "state_persistence": 1,
            "feed_mixing": -1,
            "hydraulic_residence": -1,
            "reaction_kinetics": 1,
            "reaction_heat": 1,
            "heat_exchange": -1,
            "conversion": -1,
            "temperature_sensitivity": 1,
        },
    },
}
FAULT_SEVERITY_RANGE = (0.3, 0.55)
EDGE_SEVERITY_RANGE = (0.85, 1.15)
RUN_INNOVATION_SCALE_RANGE = (0.8, 1.25)
MAX_AUTOREGRESSIVE_COEFFICIENT = 0.95
PHYSICAL_SCALES = {
    "feed_flow": 20.0,
    "feed_concentration": 0.2,
    "coolant_flow": 20.0,
    "reactor_concentration": 0.15,
    "reactor_temperature": 20.0,
    "product_quality": 0.12,
}
PHYSICAL_OFFSETS = {
    "feed_flow": 100.0,
    "feed_concentration": 1.0,
    "coolant_flow": 100.0,
    "reactor_concentration": 0.7,
    "reactor_temperature": 350.0,
    "product_quality": 0.85,
}


def identity(value):
    return value


def tanh(value):
    return np.tanh(value)


def _edge(parent, lag, coefficient, function):
    return ((parent, lag), float(coefficient), function)


def build_sem():
    """Return a nonlinear CausalAI time-series SEM and its edge metadata."""
    mechanisms = {}

    def edge(parent, target, lag, coefficient, function, mechanism):
        mechanisms[(parent, target, -lag)] = mechanism
        return _edge(parent, lag, coefficient, function)

    sem = {
        "feed_flow": [
            edge(
                "feed_flow",
                "feed_flow",
                -1,
                0.75,
                identity,
                "input_persistence",
            ),
        ],
        "feed_concentration": [
            edge(
                "feed_concentration",
                "feed_concentration",
                -1,
                0.7,
                identity,
                "input_persistence",
            ),
        ],
        "coolant_flow": [
            edge(
                "coolant_flow",
                "coolant_flow",
                -1,
                0.75,
                identity,
                "input_persistence",
            ),
        ],
        "reactor_concentration": [
            edge(
                "reactor_concentration",
                "reactor_concentration",
                -1,
                0.62,
                identity,
                "state_persistence",
            ),
            edge(
                "feed_concentration",
                "reactor_concentration",
                -1,
                0.22,
                tanh,
                "feed_mixing",
            ),
            edge(
                "feed_flow",
                "reactor_concentration",
                -1,
                -0.12,
                tanh,
                "hydraulic_residence",
            ),
            edge(
                "reactor_temperature",
                "reactor_concentration",
                -1,
                0.08,
                tanh,
                "reaction_kinetics",
            ),
        ],
        "reactor_temperature": [
            edge(
                "reactor_temperature",
                "reactor_temperature",
                -1,
                0.64,
                identity,
                "state_persistence",
            ),
            edge(
                "feed_flow",
                "reactor_temperature",
                -1,
                0.08,
                tanh,
                "hydraulic_residence",
            ),
            edge(
                "reactor_concentration",
                "reactor_temperature",
                -1,
                0.2,
                tanh,
                "reaction_heat",
            ),
            edge(
                "coolant_flow",
                "reactor_temperature",
                -1,
                -0.22,
                tanh,
                "heat_exchange",
            ),
        ],
        "product_quality": [
            edge(
                "product_quality",
                "product_quality",
                -1,
                0.55,
                identity,
                "state_persistence",
            ),
            edge(
                "reactor_concentration",
                "product_quality",
                -1,
                0.22,
                tanh,
                "conversion",
            ),
            edge(
                "reactor_temperature",
                "product_quality",
                -2,
                -0.16,
                tanh,
                "temperature_sensitivity",
            ),
        ],
    }
    return sem, mechanisms


def _sem_graph(sem, graph, mechanisms):
    edge_rows = [
        {
            "source": parent,
            "target": target,
            "lag_steps": -lag,
            "coefficient": coefficient,
            "function": getattr(function, "__name__", type(function).__name__),
            "mechanism": mechanisms[(parent, target, -lag)],
        }
        for target, parents in graph.items()
        for parent, lag in parents
        for ((sem_parent, sem_lag), coefficient, function) in sem[target]
        if parent == sem_parent and lag == sem_lag
    ]
    return pd.DataFrame(
        edge_rows,
        columns=[
            "source",
            "target",
            "lag_steps",
            "coefficient",
            "function",
            "mechanism",
        ],
    )


def _fault_coefficients(sem, fault_type, severity, mechanisms, rng):
    coefficients = {}
    changes = []
    fault = FAULTS[fault_type]
    for target, parents in sem.items():
        for (parent, lag), coefficient, _function in parents:
            mechanism = mechanisms[(parent, target, -lag)]
            requested_change = severity * float(rng.uniform(*EDGE_SEVERITY_RANGE))
            requested_coefficient = coefficient * (
                1 + fault["profile"][mechanism] * requested_change
            )
            coefficient_after = requested_coefficient
            if mechanism in {"input_persistence", "state_persistence"}:
                coefficient_after = float(
                    np.clip(
                        requested_coefficient,
                        -MAX_AUTOREGRESSIVE_COEFFICIENT,
                        MAX_AUTOREGRESSIVE_COEFFICIENT,
                    )
                )
            change_fraction = abs(coefficient_after / coefficient - 1)
            edge = (parent, target, -lag)
            coefficients[edge] = coefficient_after
            changes.append(
                {
                    "source": parent,
                    "target": target,
                    "lag_steps": -lag,
                    "coefficient_before": coefficient,
                    "coefficient_after": coefficient_after,
                    "change_fraction": change_fraction,
                    "requested_change_fraction": requested_change,
                }
            )
    return coefficients, changes


def _simulate_time_varying_sem(
    sem,
    n_samples,
    onset,
    coefficients,
    seed,
    innovation_scale,
):
    """Simulate a lagged SEM whose edge strengths change at the fault onset."""
    names = list(sem)
    indices = {name: index for index, name in enumerate(names)}
    history_len = n_samples // 5
    total_samples = n_samples + history_len
    max_lag = max(abs(lag) for parents in sem.values() for (parent, lag), _, _ in parents)
    noise_scales = {
        "feed_flow": 0.2,
        "feed_concentration": 0.2,
        "coolant_flow": 0.2,
        "reactor_concentration": 0.1,
        "reactor_temperature": 0.12,
        "product_quality": 0.1,
    }
    rng = np.random.default_rng(seed)
    values = np.column_stack(
        [
            rng.normal(
                0.0,
                noise_scales[name] * innovation_scale,
                size=total_samples,
            )
            for name in names
        ]
    )

    for time_step in range(max_lag, total_samples):
        fault_active = onset is not None and time_step >= history_len + onset
        for target, parents in sem.items():
            target_index = indices[target]
            for (parent, lag), coefficient, function in parents:
                if fault_active:
                    coefficient = coefficients[(parent, target, -lag)]
                values[time_step, target_index] += coefficient * function(
                    values[time_step + lag, indices[parent]]
                )

    return values[history_len:]


def _generate_run(
    run_id,
    fault_type,
    n_samples,
    sem,
    mechanisms,
    rng,
):
    onset = None
    coefficients = {}
    coefficient_changes = []
    severity = np.nan
    innovation_scale = float(rng.uniform(*RUN_INNOVATION_SCALE_RANGE))
    if fault_type != "healthy":
        onset = n_samples // 5
        severity = float(rng.uniform(*FAULT_SEVERITY_RANGE))
    seed = int(rng.integers(0, np.iinfo(np.int32).max))
    if fault_type != "healthy":
        coefficients, coefficient_changes = _fault_coefficients(
            sem,
            fault_type,
            severity,
            mechanisms,
            rng,
        )
    latent_data = _simulate_time_varying_sem(
        sem,
        n_samples,
        onset,
        coefficients,
        seed,
        innovation_scale,
    )

    trace = pd.DataFrame(latent_data, columns=sem)
    for name in sem:
        if name in {"feed_concentration", "reactor_concentration"}:
            trace[name] = PHYSICAL_OFFSETS[name] * np.exp(
                PHYSICAL_SCALES[name] * trace[name]
            )
        elif name == "product_quality":
            baseline_logit = np.log(
                PHYSICAL_OFFSETS[name] / (1 - PHYSICAL_OFFSETS[name])
            )
            logit = baseline_logit + PHYSICAL_SCALES[name] * trace[name]
            trace[name] = 1 / (1 + np.exp(-logit))
        else:
            trace[name] = trace[name] * PHYSICAL_SCALES[name] + PHYSICAL_OFFSETS[name]
    trace.insert(0, "anomaly", 0)
    if onset is not None:
        trace.loc[onset:, "anomaly"] = 1
    trace.insert(0, "time_seconds", np.arange(n_samples, dtype=float))
    trace.insert(0, "time_step", np.arange(n_samples, dtype=np.int64))
    trace.insert(0, "run_id", run_id)

    record = {
        "run_id": run_id,
        "fault_type": fault_type,
        "fault_start_step": -1 if onset is None else onset,
        "fault_severity": severity,
        "innovation_scale": innovation_scale,
        "random_seed": seed,
    }
    return trace, record, coefficient_changes


def _assign_splits(run_manifest, rng):
    split_by_run = {}
    for fault_type in ["healthy", *FAULTS]:
        run_ids = run_manifest.loc[
            run_manifest["fault_type"] == fault_type,
            "run_id",
        ].to_numpy()
        shuffled = rng.permutation(run_ids)
        n_runs = len(shuffled)
        n_train = max(1, int(0.7 * n_runs))
        n_validation = max(1, int(0.15 * n_runs))
        if n_train + n_validation >= n_runs:
            n_train = n_runs - 2
            n_validation = 1
        for run_id in shuffled[:n_train]:
            split_by_run[run_id] = "train"
        for run_id in shuffled[n_train : n_train + n_validation]:
            split_by_run[run_id] = "validation"
        for run_id in shuffled[n_train + n_validation :]:
            split_by_run[run_id] = "test"
    return split_by_run


def generate_dataset(
    output_dir="data/processed/causalAIProcess",
    runs_per_class=30,
    samples_per_run=2000,
    seed=20261023,
):
    """Generate process runs using a time-varying Salesforce CausalAI SEM."""
    if runs_per_class < 3:
        raise ValueError("runs_per_class must be at least 3 to populate all splits.")
    if samples_per_run < 20:
        raise ValueError("samples_per_run must be at least 20.")

    try:
        from causalai.data.data_generator import DataGenerator
    except ImportError as error:
        raise ImportError(
            "CausalAI is required for this generator. Install the project extra "
            'with `pip install -e ".[causalai]"`.'
        ) from error

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    sem, mechanisms = build_sem()
    run_specs = [
        ("healthy", None),
        *((fault_type, FAULTS[fault_type]) for fault_type in FAULTS),
    ]
    traces = []
    run_records = []
    coefficient_records = []
    run_index = 0

    for fault_type, _fault in run_specs:
        for _ in range(runs_per_class):
            run_id = f"run_{fault_type}_{run_index:05d}"
            run_index += 1
            trace, record, coefficient_changes = _generate_run(
                run_id=run_id,
                fault_type=fault_type,
                n_samples=samples_per_run,
                sem=sem,
                mechanisms=mechanisms,
                rng=rng,
            )
            traces.append(trace)
            run_records.append(record)
            coefficient_records.extend(
                {"run_id": run_id, **change} for change in coefficient_changes
            )

    _, _, graph = DataGenerator(sem, T=2, discrete=False, seed=seed)
    if tuple(graph) != VARIABLE_NAMES:
        raise RuntimeError(
            f"CausalAI changed SEM variable order: expected {VARIABLE_NAMES}, "
            f"received {tuple(graph)}."
        )
    graph_table = _sem_graph(sem, graph, mechanisms)
    coefficient_table = pd.DataFrame(
        coefficient_records,
        columns=[
            "run_id",
            "source",
            "target",
            "lag_steps",
            "coefficient_before",
            "coefficient_after",
            "change_fraction",
            "requested_change_fraction",
        ],
    )

    run_manifest = pd.DataFrame(run_records)
    run_manifest["split"] = run_manifest["run_id"].map(_assign_splits(run_manifest, rng))
    all_traces = pd.concat(traces, ignore_index=True)
    all_traces = all_traces.merge(
        run_manifest[["run_id", "split"]],
        on="run_id",
        how="left",
        validate="many_to_one",
    )

    all_traces.to_parquet(output_dir / "traces.parquet", index=False)
    run_manifest.to_csv(output_dir / "runs.csv", index=False)
    graph_table.to_csv(output_dir / "causal_graph.csv", index=False)
    coefficient_table.to_csv(output_dir / "fault_coefficients.csv", index=False)

    metadata = {
        "generator": "Salesforce CausalAI SEM with a time-varying simulator",
        "causalai_version": version("causalai"),
        "generator_version": "4.2",
        "process": "six-variable nonlinear continuous stirred-tank process",
        "variable_names": list(VARIABLE_NAMES),
        "sampling_interval_seconds": 1.0,
        "runs_per_class": runs_per_class,
        "samples_per_run": samples_per_run,
        "random_seed": seed,
        "fault_types": ["healthy", *FAULTS],
        "fault_semantics": (
            "At one-fifth of a faulty run, every SEM edge coefficient changes "
            "according to its fault-type profile, randomized run severity, and "
            "edge-specific variation. Autoregressive coefficients are bounded "
            "to preserve stable trajectories. These are mechanism changes, not "
            "interventions on variable values. Healthy runs use baseline "
            "coefficients throughout."
        ),
        "fault_severity_range": list(FAULT_SEVERITY_RANGE),
        "edge_severity_multiplier_range": list(EDGE_SEVERITY_RANGE),
        "run_innovation_scale_range": list(RUN_INNOVATION_SCALE_RANGE),
        "run_variability_semantics": (
            "Each run has a fixed, randomized multiplier on exogenous process "
            "innovation scales to represent ordinary operating variability."
        ),
        "maximum_autoregressive_coefficient": MAX_AUTOREGRESSIVE_COEFFICIENT,
        "coefficient_profiles": {
            fault_type: fault["profile"] for fault_type, fault in FAULTS.items()
        },
        "fault_coefficient_file": "fault_coefficients.csv",
        "split_semantics": "Train/validation/test are assigned by complete run.",
        "graph_semantics": (
            "causal_graph.csv describes the lagged nonlinear SEM supplied to "
            "Salesforce CausalAI. Faults preserve its edge topology but change every "
            "edge coefficient; fault_coefficients.csv records each run's post-onset "
            "coefficients. Additive innovations are exogenous."
        ),
        "scale_semantics": (
            "The SEM is simulated in centered normalized coordinates; exported "
            "signals use variable-specific offset/scale mappings. Concentrations "
            "use a positive log-scale mapping and product quality uses a bounded "
            "logistic mapping."
        ),
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    return output_dir
