# carl

Code for causally informed anomaly detection and model-based recovery on the
Tennessee Eastman Process (TEP) and a synthetic nonlinear process benchmark.
A supplied causal graph restricts per-variable linear lagged equations fitted on
fault-free data. Their residuals drive a Mahalanobis anomaly score, and after an
alarm a healthy counterfactual defines the target of a receding-horizon recovery
controller. The code accompanies the paper submitted to IDA 2027.

## Quick start for reviewers

- Precomputed figures (`reports/figures/`) and metric tables
  (`reports/results/**/*.csv`) are included, so the reported numbers can be
  inspected without rerunning anything.
- Each notebook is committed with its outputs.
- To reproduce them, create the environment below. The synthetic benchmark
  needs no external data. TEP is downloaded automatically by
  `notebooks/TEP/0.0-data_download.ipynb` (about 1.31 GiB, Harvard Dataverse).
- The shared implementation is in `carl/`, and `pytest tests` runs the unit tests.
- The TEP causal graph comes from CIPCaD-Bench (Menegozzo et al., IEEE CASE 2022).
  The synthetic graph is the Salesforce CausalAI graph that generates the data.
  The edges are treated as given and are not independently validated here.

## Reproducing the notebooks

From the repository root, create the environment and install the optional
dependency used by the synthetic process workflow:

```bash
conda env create -f environment.yml
conda activate carl
pip install -e ".[causalai]"
jupyter lab
```

Run notebooks in numerical order within each folder. The TEP workflow starts
with [`notebooks/TEP/0.0-data_download.ipynb`](notebooks/TEP/0.0-data_download.ipynb),
which downloads about 1.31 GiB from Harvard Dataverse, then converts and
preprocesses the data in `0.1` and `0.2`. Continue with `0.3` for EDA, `1.0` and
`1.1` for parameter selection, `2.x` for detection/recovery, `3.x` for
baselines, `4.x` for detection ablations, and `5.x` for alarm ablations.
The synthetic workflow starts with
[`notebooks/causalAI/0.0-generate_process_dataset.ipynb`](notebooks/causalAI/0.0-generate_process_dataset.ipynb);
continue with validation/preprocessing in `0.1`–`0.2`, then follow the
subsequent numbered analysis notebooks. Both workflows write generated data,
models, and results beneath `data/`, `models/`, and `reports/`. Raw/processed
data and trained models are not included in the repository, and neither are the
large per-sample score files (`*.parquet`) under `reports/results/`.

The synthetic paper figures and aggregate TEP detection/recovery summary can
be regenerated with
[`notebooks/figures/results_figures.ipynb`](notebooks/figures/results_figures.ipynb).
To generate TEP comparison figures, rerun the TEP primary, baseline,
score-ablation, and alarm-ablation evaluation notebooks so their export cells
write metrics under `reports/results/TEP/figure_metrics/`, then run
[`notebooks/figures/tep_comparison_figures.ipynb`](notebooks/figures/tep_comparison_figures.ipynb).
The synthetic figure notebook also writes `detection_recovery_aggregate_summary.csv`; run the synthetic recovery pipeline first to refresh its inputs. Both figure notebooks write PNG and PDF files to `reports/figures/`; synthetic and TEP score/alarm ablations are each combined into a single compact figure; TEP detection and model complexity are also combined.

## Tennessee Eastman Process dataset

The TEP notebooks and dataset workflows are in `notebooks/TEP/`. Raw inputs,
converted tables, and processed train/test tables are stored under `data/raw/TEP/`,
`data/interim/TEP/`, and `data/processed/TEP/`. TEP model artifacts and result
tables are under `models/TEP/` and `reports/results/TEP/`.
TEP-specific graph and reference material belongs in `references/TEP/`; shared
references remain directly under `references/`. Shared Python utilities live in
the `carl` package, including `carl/tep_utils.py`.

## CausalAI-based process dataset

For a benchmark where the causal graph directly defines the data-generating
process, install the optional Salesforce CausalAI dependency:

```text
pip install -e ".[causalai]"
```

Then run
[`notebooks/causalAI/0.0-generate_process_dataset.ipynb`](notebooks/causalAI/0.0-generate_process_dataset.ipynb)
or use `carl.causalai_process.generate_dataset()`. It uses a six-variable,
nonlinear, lagged SEM in Salesforce CausalAI's format. Faults are not interventions
on variable values: from one-fifth of a faulty run onward, every edge coefficient
changes according to a fault-type profile and a run-specific random magnitude.
Run-level severity is randomized from 30% to 55%, with an additional edge-level
variation. Exogenous process innovations also have a randomized run-level scale
from 0.8 to 1.25 to represent ordinary operating variability. Autoregressive
coefficients are bounded to preserve stable trajectories.
The graph topology stays fixed while the strengths of all relationships change.
Concentrations are mapped to positive values and product quality to the unit interval.
The generator writes healthy and faulty trajectories, exact sample-level fault
onsets, run-level splits, the baseline causal graph, and `fault_coefficients.csv`
with per-run coefficient changes to `data/processed/causalAIProcess/`.

The synthetic generator already writes Parquet and CSV artifacts, so it does not
need the TEP raw-data conversion step. The subsequent synthetic-data notebooks are:

1. `notebooks/causalAI/0.1-validate_generated_data.ipynb` checks generated files,
   fixed sampling, anomaly timing, and per-run coefficient changes.
2. `notebooks/causalAI/0.2-data_preprocessing.ipynb` robust-scales signals using
   only healthy training runs and writes a modeling view to
   `data/interim/causalAIProcess/`.
3. `notebooks/causalAI/0.3-eda.ipynb` explores class/run summaries, signal
   distributions and sampled trajectories, and baseline/fault-specific relationships.
4. `notebooks/causalAI/1.0-max_lag_selection.ipynb` selects the autoregressive
   lag by multivariate BIC using the SEM parents and healthy training runs.
5. `notebooks/causalAI/1.1-window_size_and_consecutive_alarms_selection.ipynb`
   tunes score smoothing and consecutive alarms using healthy/faulty validation
   runs. As in the TEP pipeline, its detection score and residual covariance use
   only endogenous process variables; manipulated inputs remain available as
   predictors. It saves metrics under `reports/results/causalAIProcess/`.
   It also evaluates the selected settings on held-out test runs and saves overall
   and per-fault results to `heldout_test_metrics.csv` and
   `heldout_test_by_fault_type.csv`.
6. `notebooks/causalAI/2.0-anomaly_detection_faulty_test.ipynb` evaluates faulty
   held-out runs using the validation-selected model and alarm settings. It saves
   sample scores and overall/per-fault metrics under
   `reports/results/causalAIProcess/`.
7. `notebooks/causalAI/2.1-anomaly_detection_faultfree_test.ipynb` evaluates
   healthy held-out runs with the same fixed settings and reports sample- and
   trajectory-level false-alarm rates in `reports/results/causalAIProcess/`,
   including a one-sided 95% Wilson upper bound for the trajectory-level rate.
8. `notebooks/causalAI/2.2-detection_recovery_pipeline.ipynb` mirrors the TEP
   detection/recovery workflow using the synthetic benchmark's lag-2 dynamics
   and endogenous-only anomaly score.
   It reconstructs a healthy reference, runs the same sample-by-sample control
   objective, and checks recovery from process-state deviation over consecutive
   moving-score windows. The per-run anomaly/recovery score plots use the same
   log-scale style and plotting helpers as TEP 2.2. It saves process traces,
   scores, and actuator outputs under `reports/results/causalAIProcess/`.
   Recovery is model-based and does not establish recovery under a physically
   persistent fault.
9. `notebooks/causalAI/3.0-anomaly_detection_baseline_full_var.ipynb`,
   `notebooks/causalAI/3.1-anomaly_detection_baseline_pca.ipynb`, and
   `notebooks/causalAI/3.2-anomaly_detection_baseline_self_history.ipynb`
   mirror the TEP 3.0–3.2 detection baselines. They fit on healthy training
   runs, score only endogenous process variables, select alarm settings on faulty
   validation runs using the same
   max-pre-fault-score threshold and trajectory-phase F1/delay procedure, then
   evaluate every faulty and healthy test run. PCA is fit on the three
   endogenous process variables, retains 95% variance where possible, and is
   capped at two components so the
   reconstruction-error score is non-degenerate. The reusable implementation
   is in `carl/causalai_baselines.py`; per-baseline scores, metrics, settings,
   and the combined comparison table are saved under
   `reports/results/causalAIProcess/`.
10. `notebooks/causalAI/4.0-anomaly_detection_ablation_full_var.ipynb` through
    `4.4-anomaly_detection_ablation_plain_residuals.ipynb` mirror TEP's five
    detection ablations: full residual covariance, PCA reconstruction, self-history
    models, diagonal residual covariance, and unnormalized residuals. As in TEP
    4.x, the moving-average window and consecutive-alarm count are fixed at six
    and five; each threshold is the maximum healthy-phase score on faulty
    validation runs. All faulty and healthy test runs are evaluated, with
    per-ablation metrics and a combined comparison table saved under
    `reports/results/causalAIProcess/`.
11. `notebooks/causalAI/5.0-alarm_ablation_threshold_only.ipynb` and
    `5.1-alarm_ablation_moving_average_only.ipynb` mirror the TEP alarm ablations.
    They use the full-predictor baseline scores of endogenous variables and the
    threshold from notebook 3.0; 5.0 removes smoothing and alarm persistence,
    while 5.1 keeps the
    baseline moving-average window but disables consecutive alarms. Results
    cover all faulty and healthy held-out test runs and are saved under
    `reports/results/causalAIProcess/`.
The parameter-selection notebooks parallel the available TEP notebooks 1.0 and
1.1. Lag candidates are compared with multivariate BIC, and the alarm notebook
searches the same window and consecutive-alarm ranges as TEP. For each window,
the score threshold is the maximum smoothed score across the pre-fault (healthy)
portion of faulty validation runs. As in TEP, each faulty validation trajectory
is split into a healthy first phase and a faulty second phase; phase metrics are
computed over those two phases, and the configuration with highest
trajectory-phase F1 (breaking ties by detection delay) is selected. Held-out test
runs are not used for tuning.

The trajectories capture several TEP-like evaluation properties: continuous
multivariate runs, coupled nonlinear dynamics, sustained process faults, and known
onsets. The small SEM is not a physical TEP model, and the synthetic signals should
not be described as reproducing TEP plant behavior.

## Project organization

- `carl/`: shared utilities and dataset generators.
- `data/`: external, raw, intermediate, and processed dataset files.
- `models/`: model artifacts.
- `notebooks/`: TEP and CausalAI process workflows.
- `references/`: dataset references and explanatory materials.
- `reports/`: generated analysis and result tables.
- `tests/`: automated tests.
