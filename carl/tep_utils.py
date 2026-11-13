"""Utilities shared by the Tennessee Eastman Process notebooks."""

from matplotlib import ticker
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def forecast(data, child, parents, beta, max_lag):
    df = data[parents + [child]]
    df_lagged = pd.concat(
        [df.shift(i).add_suffix(f"_lag{i}") for i in range(1, max_lag + 1)],
        axis=1,
    )
    X = np.ones(df_lagged.shape[1] + 1)
    X[1:] = df_lagged.iloc[-1].to_numpy()
    return X @ beta


def initialize_fig(xmax, xticks, y_lims, yticks, anomaly_threshold, legend_loc, ts_idx):
    fig, ax = plt.subplots(figsize=(16, 7))
    ax.set_title(f"Anomaly and Recovery Scores - Time Series #{ts_idx + 1}", fontsize=10)
    ax.set_xlim(0, xmax)
    ax.set_xticks(xticks)
    ax.set_xlabel("Time")
    ax.tick_params(axis="x", labelsize=7)
    ax.set_yscale("log")
    ax.set_ylim(y_lims[0], y_lims[1])
    ax.set_yticks(yticks)
    ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:g}"))
    ax.set_ylabel("Score")
    ax.tick_params(axis="y", labelsize=6)
    (anomaly_score_line,) = ax.plot([], [], color="#5C6466", alpha=0.5, label="Anomaly Score")
    (anomaly_moving_average_line,) = ax.plot(
        [], [], color="#232626", label="Moving Average Anomaly Score"
    )
    (recovery_score_line,) = ax.plot([], [], color="#17B4CC")
    (recovery_moving_average_line,) = ax.plot([], [], color="#1C66C1")
    ax.hlines(
        anomaly_threshold,
        xmin=0.0,
        xmax=xmax,
        colors="#AE741D",
        label="Anomaly Threshold",
    )
    ax.legend(loc=legend_loc)
    ax.grid()
    return (
        fig,
        ax,
        [
            recovery_score_line,
            recovery_moving_average_line,
            anomaly_score_line,
            anomaly_moving_average_line,
        ],
    )


def update_fig(lines, matrices, current_timestep):
    x_data = range(current_timestep + 1)
    score_line = lines[0]
    moving_average_line = lines[1]
    score_matrix = matrices[0]
    moving_average_score_matrix = matrices[1]
    score_line.set_data(x_data, score_matrix[: current_timestep + 1])
    moving_average_line.set_data(x_data, moving_average_score_matrix[: current_timestep + 1])


def compute_g_matrix(
    hist_ts,
    manipulated_parents_pairs,
    process_child_parents_pairs,
    child_beta_pairs,
    process_children,
    manipulated,
    horizon,
):
    beta_mat = pd.DataFrame(columns=hist_ts.columns, dtype=object)

    for child, parents in manipulated_parents_pairs.items():
        beta = child_beta_pairs[child]
        beta_mat.loc[child] = 0.0
        beta_mat.loc[child, parents] = beta[1:-1]
        beta_mat.loc[child, child] = beta[-1]

    for child, parents in process_child_parents_pairs.items():
        beta = child_beta_pairs[child]
        beta_mat.loc[child] = 0.0
        beta_mat.loc[child, parents] = beta[1:-1]
        beta_mat.loc[child, child] = beta[-1]

    w_m = beta_mat.loc[process_children, manipulated].to_numpy()
    w_p = beta_mat.loc[process_children, process_children].to_numpy()

    psi = {1: w_m}
    for time_idx in range(2, horizon):
        psi[time_idx] = w_p @ psi[time_idx - 1]

    zero_mat = np.zeros_like(psi[1])
    g = np.concatenate([psi[i] for i in range(1, horizon)], axis=0)
    g = np.concatenate([zero_mat, g], axis=0)

    n_process_children = len(process_children)
    n_manipulated = len(manipulated)

    n_rows = horizon * n_process_children
    for _ in range(horizon - 2):
        last_col = g[: n_rows - n_process_children, -n_manipulated:]
        g_new_col = np.concatenate([zero_mat, last_col], axis=0)
        g = np.concatenate([g, g_new_col], axis=1)

    return g


def compute_a_b_matrices(n_manipulated, horizon):
    id_mat = np.eye(n_manipulated, dtype=int)
    zero_vector = np.zeros((n_manipulated, (horizon - 3) * n_manipulated), dtype=int)

    d = np.concatenate([-id_mat, id_mat, zero_vector], axis=1)
    zero_mat = np.zeros_like(id_mat, dtype=int)
    for _ in range(horizon - 3):
        last_row = d[-n_manipulated:, :-n_manipulated]
        d_new_row = np.concatenate([zero_mat, last_row], axis=1)
        d = np.concatenate([d, d_new_row], axis=0)

    zero_vector = np.zeros((n_manipulated, (horizon - 2) * n_manipulated), dtype=int)
    b = np.concatenate([id_mat, zero_vector], axis=1)
    a = d.T @ d + b.T @ b
    return a, b
