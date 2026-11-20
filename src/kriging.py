"""
Ordinary Kriging grade estimation with variogram modeling.

Implements a simplified Ordinary Kriging (OK) system for spatial grade
interpolation, including experimental variogram calculation, model fitting
(spherical, exponential, gaussian), and cross-validation.

This provides industry-standard geostatistical estimates alongside the
ML-based grade prediction, comparable to methods used in JORC/NI 43-101
compliant resource estimation.
"""

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist


def compute_experimental_variogram(
    df: pd.DataFrame,
    n_lags: int = 15,
    lag_tol_factor: float = 0.5,
    max_dist_fraction: float = 0.6,
    value_col: str = "grade_au_gt",
) -> dict:
    """Calculate experimental (semi)variogram from block data."""
    coords = df[["x", "y", "z"]].values
    values = df[value_col].values

    # Sample for performance
    n = len(df)
    if n > 2000:
        rng = np.random.default_rng(42)
        idx = rng.choice(n, 2000, replace=False)
        coords = coords[idx]
        values = values[idx]

    dists = cdist(coords, coords)
    max_dist = np.max(dists) * max_dist_fraction
    lag_size = max_dist / n_lags

    lags, gammas, counts = [], [], []
    for k in range(n_lags):
        lag_min = k * lag_size
        lag_max = (k + lag_tol_factor + 1) * lag_size
        mask = (dists > lag_min) & (dists <= lag_max)
        i_idx, j_idx = np.where(mask)
        # Use upper triangle only
        pairs = [(i, j) for i, j in zip(i_idx, j_idx) if i < j]
        if len(pairs) < 5:
            continue
        ii, jj = zip(*pairs)
        gamma = 0.5 * np.mean((values[list(ii)] - values[list(jj)]) ** 2)
        lags.append((lag_min + lag_max) / 2)
        gammas.append(gamma)
        counts.append(len(pairs))

    return {"lags": lags, "gammas": gammas, "counts": counts, "lag_size": lag_size, "max_dist": max_dist}


def fit_variogram_model(lags: list, gammas: list, model: str = "spherical") -> dict:
    """Fit a theoretical variogram model by least squares."""
    lags = np.array(lags)
    gammas = np.array(gammas)
    if len(lags) < 3:
        return {"nugget": 0, "sill": float(np.mean(gammas)) if len(gammas) else 1, "range": float(lags[-1]) if len(lags) else 50, "model": model}

    # Simple grid search for nugget, sill, range
    sill_candidates = np.linspace(gammas.max() * 0.5, gammas.max() * 1.5, 10)
    range_candidates = np.linspace(lags[1], lags[-1], 10)
    nugget_candidates = np.linspace(0, gammas.min() * 1.5, 5)

    best_sse = np.inf
    best_params = {"nugget": 0, "sill": gammas.max(), "range": lags[-1] / 2}

    for nugget in nugget_candidates:
        for sill in sill_candidates:
            for a in range_candidates:
                fitted = _variogram_model(lags, nugget, sill, a, model)
                sse = np.sum((gammas - fitted) ** 2)
                if sse < best_sse:
                    best_sse = sse
                    best_params = {"nugget": nugget, "sill": sill, "range": a}

    return {**best_params, "model": model, "sse": best_sse}


def _variogram_model(h: np.ndarray, nugget: float, sill: float, a: float, model: str) -> np.ndarray:
    h = np.atleast_1d(h)
    result = np.zeros_like(h, dtype=float)
    if model == "spherical":
        mask = h <= a
        result[mask] = nugget + (sill - nugget) * (1.5 * h[mask] / a - 0.5 * (h[mask] / a) ** 3)
        result[~mask] = sill
    elif model == "exponential":
        result = nugget + (sill - nugget) * (1 - np.exp(-3 * h / a))
    elif model == "gaussian":
        result = nugget + (sill - nugget) * (1 - np.exp(-3 * (h / a) ** 2))
    return result


def ordinary_kriging(
    train_df: pd.DataFrame,
    predict_df: pd.DataFrame,
    variogram_params: dict,
    value_col: str = "grade_au_gt",
    max_neighbors: int = 20,
) -> np.ndarray:
    """
    Perform Ordinary Kriging estimation at prediction locations.
    Uses a moving neighborhood approach for scalability.
    """
    train_coords = train_df[["x", "y", "z"]].values
    train_values = train_df[value_col].values
    pred_coords = predict_df[["x", "y", "z"]].values

    nugget = variogram_params["nugget"]
    sill = variogram_params["sill"]
    a = variogram_params["range"]
    model = variogram_params["model"]

    predictions = np.zeros(len(pred_coords))
    variances = np.zeros(len(pred_coords))

    all_dists = cdist(pred_coords, train_coords)

    for i in range(len(pred_coords)):
        dists_to_train = all_dists[i]
        neighbor_idx = np.argsort(dists_to_train)[:max_neighbors]

        nb_coords = train_coords[neighbor_idx]
        nb_values = train_values[neighbor_idx]
        n = len(neighbor_idx)

        # Build kriging matrix C (n+1 x n+1 with Lagrange multiplier)
        C_matrix = np.zeros((n + 1, n + 1))
        h_matrix = cdist(nb_coords, nb_coords)
        C_matrix[:n, :n] = sill - _variogram_model(h_matrix, nugget, sill, a, model)
        C_matrix[n, :n] = 1
        C_matrix[:n, n] = 1

        # Build RHS vector c0
        h0 = dists_to_train[neighbor_idx]
        c0 = np.zeros(n + 1)
        c0[:n] = sill - _variogram_model(h0, nugget, sill, a, model)
        c0[n] = 1

        try:
            weights = np.linalg.solve(C_matrix, c0)
            predictions[i] = np.dot(weights[:n], nb_values)
            variances[i] = max(0, nugget + sill - np.dot(weights, c0))
        except np.linalg.LinAlgError:
            predictions[i] = np.mean(nb_values)
            variances[i] = sill

    return predictions, variances


def run_kriging_analysis(
    block_df: pd.DataFrame,
    variogram_model: str = "spherical",
    test_fraction: float = 0.2,
    value_col: str = "grade_au_gt",
) -> dict:
    """Full kriging workflow: variogram → fit → krige → cross-validate."""
    rng = np.random.default_rng(42)
    n = len(block_df)

    # Use a manageable sample for speed
    sample_n = min(3000, n)
    idx = rng.choice(n, sample_n, replace=False)
    sample_df = block_df.iloc[idx].reset_index(drop=True)

    test_n = int(sample_n * test_fraction)
    test_idx = rng.choice(sample_n, test_n, replace=False)
    train_mask = np.ones(sample_n, dtype=bool)
    train_mask[test_idx] = False
    train_df = sample_df[train_mask].reset_index(drop=True)
    test_df = sample_df[~train_mask].reset_index(drop=True)

    # Experimental variogram
    exp_vario = compute_experimental_variogram(train_df, value_col=value_col)

    # Fit model
    vario_params = fit_variogram_model(exp_vario["lags"], exp_vario["gammas"], variogram_model)

    # Krige test points
    pred_values, pred_variances = ordinary_kriging(train_df, test_df, vario_params, value_col=value_col)

    actual = test_df[value_col].values
    residuals = actual - pred_values
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((actual - actual.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    rmse = np.sqrt(np.mean(residuals ** 2))
    mae = np.mean(np.abs(residuals))

    # Krige full domain (sample for display)
    display_n = min(2000, n)
    display_idx = rng.choice(n, display_n, replace=False)
    display_df = block_df.iloc[display_idx].reset_index(drop=True)
    all_pred, all_var = ordinary_kriging(train_df, display_df, vario_params, value_col=value_col)

    return {
        "variogram_model": variogram_model,
        "variogram_params": {
            "nugget": round(float(vario_params["nugget"]), 4),
            "sill": round(float(vario_params["sill"]), 4),
            "range_m": round(float(vario_params["range"]), 1),
            "model": vario_params["model"],
        },
        "experimental_variogram": {
            "lags": [round(l, 1) for l in exp_vario["lags"]],
            "gammas": [round(g, 4) for g in exp_vario["gammas"]],
            "counts": exp_vario["counts"],
        },
        "fitted_variogram": {
            "lags": [round(l, 1) for l in exp_vario["lags"]],
            "gammas": [round(float(_variogram_model(np.array([l]), vario_params["nugget"], vario_params["sill"], vario_params["range"], variogram_model)[0]), 4) for l in exp_vario["lags"]],
        },
        "cross_validation": {
            "r2": round(float(r2), 4),
            "rmse": round(float(rmse), 4),
            "mae": round(float(mae), 4),
            "n_test": test_n,
        },
        "predicted": {
            "x": display_df["x"].tolist(),
            "y": display_df["y"].tolist(),
            "z": display_df["z"].tolist(),
            "grade": [round(float(v), 4) for v in all_pred],
            "variance": [round(float(v), 4) for v in all_var],
        },
        "actual_vs_predicted": {
            "actual": [round(float(v), 4) for v in actual[:500]],
            "predicted": [round(float(v), 4) for v in pred_values[:500]],
        },
    }
