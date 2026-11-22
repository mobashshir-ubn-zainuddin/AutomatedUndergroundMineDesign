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


# ─────────────────────────────────────────────────────────────────────────────
# SIMPLE KRIGING (SK)
# ─────────────────────────────────────────────────────────────────────────────

def simple_kriging(train_df, predict_df, variogram_params, mean_grade=None, max_neighbors=20):
    """
    Simple Kriging (SK): uses a known global mean.
    Estimates z*(x) = m + λᵀ (z(xᵢ) - m)
    """
    pts = train_df[["x", "y", "z"]].values
    vals = train_df["grade_au_gt"].values
    pred_pts = predict_df[["x", "y", "z"]].values

    if mean_grade is None:
        mean_grade = float(vals.mean())

    nugget = variogram_params["nugget"]
    sill = variogram_params["sill"]
    a = variogram_params["range"]
    model = variogram_params.get("model", "spherical")

    C0 = sill + nugget  # variance at h=0

    n_pred = len(pred_pts)
    predictions = np.zeros(n_pred)
    variances = np.zeros(n_pred)

    for i, x0 in enumerate(pred_pts):
        dists = np.sqrt(np.sum((pts - x0) ** 2, axis=1))
        idx = np.argsort(dists)[:max_neighbors]
        d_local = dists[idx]
        v_local = vals[idx] - mean_grade

        C_mat = np.zeros((len(idx), len(idx)))
        for r in range(len(idx)):
            for c in range(len(idx)):
                h = np.sqrt(np.sum((pts[idx[r]] - pts[idx[c]]) ** 2))
                gamma = _variogram_model(h, nugget, sill, a, model)
                C_mat[r, c] = C0 - gamma

        c0 = C0 - np.array([_variogram_model(d, nugget, sill, a, model) for d in d_local])

        try:
            lam = np.linalg.solve(C_mat, c0)
        except np.linalg.LinAlgError:
            lam = np.linalg.lstsq(C_mat, c0, rcond=None)[0]

        predictions[i] = mean_grade + np.dot(lam, v_local)
        variances[i] = max(0, C0 - np.dot(lam, c0))

    return predictions, variances


def run_simple_kriging(block_df, variogram_model="spherical", test_fraction=0.2):
    """Full SK workflow with cross-validation."""
    rng = np.random.default_rng(42)
    n = len(block_df)
    test_n = max(10, int(n * test_fraction))
    test_idx = rng.choice(n, test_n, replace=False)
    train_mask = np.ones(n, dtype=bool)
    train_mask[test_idx] = False

    train_df = block_df[train_mask].reset_index(drop=True)
    test_df = block_df[~train_mask].reset_index(drop=True)

    exp_vario = compute_experimental_variogram(train_df)
    vario_params = fit_variogram_model(exp_vario["lags"], exp_vario["gammas"], model=variogram_model)
    vario_params["model"] = variogram_model

    mean_grade = float(train_df["grade_au_gt"].mean())
    preds, variances = simple_kriging(train_df, test_df, vario_params, mean_grade=mean_grade)

    actual = test_df["grade_au_gt"].values
    residuals = actual - preds
    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    mae = float(np.mean(np.abs(residuals)))
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((actual - actual.mean()) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0

    display_idx = rng.choice(n, min(2000, n), replace=False)
    display_df = block_df.iloc[display_idx].reset_index(drop=True)
    all_pred, all_var = simple_kriging(train_df, display_df, vario_params, mean_grade=mean_grade)

    return {
        "kriging_type": "Simple Kriging (SK)",
        "mean_grade": round(mean_grade, 4),
        "variogram_model": variogram_model,
        "variogram_params": {k: round(float(v), 4) if isinstance(v, float) else v
                             for k, v in vario_params.items()},
        "cv_metrics": {"r2": round(r2, 4), "rmse": round(rmse, 4), "mae": round(mae, 4)},
        "predicted": {
            "x": display_df["x"].tolist(), "y": display_df["y"].tolist(),
            "z": display_df["z"].tolist(), "grade": all_pred.tolist(),
            "variance": all_var.tolist(),
        },
        "actual_vs_predicted": {
            "actual": actual[:500].tolist(), "predicted": preds[:500].tolist(),
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# INDICATOR KRIGING (IK)
# ─────────────────────────────────────────────────────────────────────────────

def run_indicator_kriging(block_df, thresholds=None, variogram_model="spherical"):
    """
    Indicator Kriging (IK): estimates P(grade > threshold) at each block.
    Returns exceedance probabilities for JORC classification:
      P > threshold_measured  → Measured
      P > threshold_indicated → Indicated
      rest                    → Inferred
    """
    grades = block_df["grade_au_gt"].values
    if thresholds is None:
        thresholds = [
            float(np.percentile(grades, 40)),
            float(np.percentile(grades, 60)),
            float(np.percentile(grades, 80)),
        ]

    rng = np.random.default_rng(42)
    n = len(block_df)
    sample_n = min(1000, n)
    sample_idx = rng.choice(n, sample_n, replace=False)
    train_df = block_df.iloc[sample_idx].reset_index(drop=True)
    predict_df = block_df.reset_index(drop=True)

    predict_n = min(3000, n)
    pred_idx = rng.choice(n, predict_n, replace=False)
    predict_df = block_df.iloc[pred_idx].reset_index(drop=True)

    pts = train_df[["x", "y", "z"]].values
    pred_pts = predict_df[["x", "y", "z"]].values

    indicator_probs = np.zeros((predict_n, len(thresholds)))

    for t_idx, thresh in enumerate(thresholds):
        # Binary indicator transform
        indicators = (train_df["grade_au_gt"].values >= thresh).astype(float)

        # Compute experimental indicator variogram
        train_df_copy = train_df.copy()
        train_df_copy["grade_au_gt"] = indicators
        exp_v = compute_experimental_variogram(train_df_copy)
        if not exp_v["lags"]:
            indicator_probs[:, t_idx] = float(indicators.mean())
            continue

        vp = fit_variogram_model(exp_v["lags"], exp_v["gammas"], model=variogram_model)
        vp["model"] = variogram_model
        nugget, sill, a = vp["nugget"], vp["sill"], vp["range"]
        C0 = sill + nugget

        # Solve IK system (same as OK but on indicators)
        for i, x0 in enumerate(pred_pts):
            dists = np.sqrt(np.sum((pts - x0) ** 2, axis=1))
            idx = np.argsort(dists)[:15]
            d_local = dists[idx]

            G = np.zeros((len(idx) + 1, len(idx) + 1))
            for r in range(len(idx)):
                for c in range(len(idx)):
                    h = np.sqrt(np.sum((pts[idx[r]] - pts[idx[c]]) ** 2))
                    G[r, c] = _variogram_model(h, nugget, sill, a, variogram_model)
            G[:len(idx), len(idx)] = 1
            G[len(idx), :len(idx)] = 1

            g0 = np.zeros(len(idx) + 1)
            g0[:len(idx)] = [_variogram_model(d, nugget, sill, a, variogram_model) for d in d_local]
            g0[len(idx)] = 1

            try:
                lam = np.linalg.solve(G, g0)
            except np.linalg.LinAlgError:
                lam = np.linalg.lstsq(G, g0, rcond=None)[0]

            prob = np.dot(lam[:len(idx)], indicators[idx])
            indicator_probs[i, t_idx] = np.clip(prob, 0, 1)

    # Classify blocks by exceedance probability
    classification = []
    for i in range(predict_n):
        if indicator_probs[i, -1] >= 0.7:
            classification.append("Measured")
        elif indicator_probs[i, 1] >= 0.5:
            classification.append("Indicated")
        else:
            classification.append("Inferred")

    cls_counts = {c: classification.count(c) for c in ["Measured", "Indicated", "Inferred"]}

    return {
        "kriging_type": "Indicator Kriging (IK)",
        "thresholds": [round(t, 3) for t in thresholds],
        "classification_counts": cls_counts,
        "blocks": {
            "x": predict_df["x"].tolist(), "y": predict_df["y"].tolist(),
            "z": predict_df["z"].tolist(),
            "classification": classification,
            "prob_high": indicator_probs[:, -1].tolist(),
        },
        "threshold_labels": ["P40 (Inferred threshold)", "P60 (Indicated threshold)", "P80 (Measured threshold)"],
    }


# ─────────────────────────────────────────────────────────────────────────────
# SEQUENTIAL GAUSSIAN SIMULATION (SGS)
# ─────────────────────────────────────────────────────────────────────────────

def run_sgs(block_df, n_simulations=10, variogram_model="spherical"):
    """
    Sequential Gaussian Simulation (SGS):
    Generates K equiprobable grade realisations via:
    1. Normal-score transform of data
    2. Sequential simulation visiting each node in random order
    3. Back-transform to original grade space
    Returns mean, variance, and percentile bands across simulations.
    """
    from scipy.stats import norm as _norm

    grades = block_df["grade_au_gt"].values.copy()
    n = len(grades)
    rng = np.random.default_rng(42)

    # Normal-score transform
    ranks = np.argsort(np.argsort(grades))
    p = (ranks + 0.5) / n
    ns = _norm.ppf(p)

    # Variogram on normal scores
    ns_df = block_df.copy()
    ns_df["grade_au_gt"] = ns
    exp_v = compute_experimental_variogram(ns_df)
    if exp_v["lags"]:
        vp = fit_variogram_model(exp_v["lags"], exp_v["gammas"], model=variogram_model)
    else:
        vp = {"nugget": 0.1, "sill": 0.9, "range": 20.0}
    vp["model"] = variogram_model
    nugget, sill, a = vp["nugget"], vp["sill"], vp["range"]

    pts = block_df[["x", "y", "z"]].values
    sim_n = min(500, n)  # simulate on a subset for performance
    sim_idx = rng.choice(n, sim_n, replace=False)
    sim_pts = pts[sim_idx]
    sim_grades = grades[sim_idx]
    ns_data = ns[sim_idx]

    all_sims = np.zeros((sim_n, n_simulations))

    for k in range(n_simulations):
        visit_order = rng.permutation(sim_n)
        sim_values = ns_data.copy()

        for step, i in enumerate(visit_order[:min(50, sim_n)]):
            x0 = sim_pts[i]
            # Conditioning data: already simulated + original
            cond_idx = visit_order[:step]
            if len(cond_idx) > 0:
                dists = np.sqrt(np.sum((sim_pts[cond_idx] - x0) ** 2, axis=1))
                near = cond_idx[np.argsort(dists)[:10]]
                d_near = dists[np.argsort(dists)[:10]]

                C_mat = np.zeros((len(near) + 1, len(near) + 1))
                for r in range(len(near)):
                    for c in range(len(near)):
                        h = np.sqrt(np.sum((sim_pts[near[r]] - sim_pts[near[c]]) ** 2))
                        C_mat[r, c] = sill - _variogram_model(h, nugget, sill, a, variogram_model)
                C_mat[:len(near), len(near)] = 1
                C_mat[len(near), :len(near)] = 1

                c0 = np.zeros(len(near) + 1)
                c0[:len(near)] = [sill - _variogram_model(d, nugget, sill, a, variogram_model) for d in d_near]
                c0[len(near)] = 1

                try:
                    lam = np.linalg.solve(C_mat, c0)
                except np.linalg.LinAlgError:
                    lam = np.linalg.lstsq(C_mat, c0, rcond=None)[0]

                sk_mean = np.dot(lam[:len(near)], sim_values[near])
                sk_var = max(1e-6, 1.0 - np.dot(lam[:len(near)], c0[:len(near)]))
                sim_values[i] = sk_mean + np.sqrt(sk_var) * rng.standard_normal()

        # Back-transform via quantile mapping
        sim_ns_sorted = np.sort(sim_values)
        orig_sorted = np.sort(sim_grades)
        back = np.interp(sim_values, sim_ns_sorted, orig_sorted)
        all_sims[:, k] = np.maximum(back, 0)

    sim_mean = np.mean(all_sims, axis=1)
    sim_std = np.std(all_sims, axis=1)
    sim_p10 = np.percentile(all_sims, 10, axis=1)
    sim_p90 = np.percentile(all_sims, 90, axis=1)

    # E-type (mean) and uncertainty
    display_df = block_df.iloc[sim_idx].reset_index(drop=True)

    return {
        "kriging_type": "Sequential Gaussian Simulation (SGS)",
        "n_simulations": n_simulations,
        "variogram_params": {k: round(float(v), 4) if isinstance(v, float) else v
                             for k, v in vp.items()},
        "etype_mean": round(float(sim_mean.mean()), 4),
        "etype_std": round(float(sim_std.mean()), 4),
        "blocks": {
            "x": display_df["x"].tolist(), "y": display_df["y"].tolist(),
            "z": display_df["z"].tolist(),
            "mean": sim_mean.tolist(), "std": sim_std.tolist(),
            "p10": sim_p10.tolist(), "p90": sim_p90.tolist(),
        },
        "uncertainty_histogram": {
            "counts": np.histogram(sim_std, bins=20)[0].tolist(),
            "bins": np.histogram(sim_std, bins=20)[1].tolist(),
        },
    }
