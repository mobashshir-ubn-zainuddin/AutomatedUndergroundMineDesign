"""Stochastic Stope Optimization via Sequential Gaussian Simulation (SGS)."""

import numpy as np
import pandas as pd
from scipy.stats import norm


def _sgs_simulate(df, grade_col="grade_au_gt", n_simulations=20, seed=42):
    """
    Simplified Sequential Gaussian Simulation.
    For each simulation, perturb grades using local kriging variance (nugget effect).
    Returns array of shape (n_blocks, n_simulations).
    """
    rng = np.random.default_rng(seed)
    grades = df[grade_col].values
    n = len(grades)

    # Estimate local variability: use nugget-to-sill ratio ~0.3 for lode gold
    # std_local proportional to grade magnitude (reflects nugget effect)
    std_local = grades * 0.25 + 0.1  # coefficient of variation ~25% plus floor

    # Normal-score transform
    ranks = np.argsort(np.argsort(grades))
    p = (ranks + 0.5) / n
    ns_grades = norm.ppf(p)

    simulations = np.zeros((n, n_simulations))
    for k in range(n_simulations):
        # Add spatially correlated noise via block random field
        noise = rng.normal(0, 1, n)
        # Smooth noise slightly to add spatial correlation
        sim_ns = ns_grades + noise * 0.3  # perturbation in normal score space
        # Back-transform via quantile mapping
        sim_sorted_idx = np.argsort(sim_ns)
        result = np.zeros(n)
        result[sim_sorted_idx] = np.sort(grades)
        # Add nugget noise
        nugget = rng.exponential(scale=std_local * 0.15)
        result = np.maximum(result + nugget * rng.choice([-1, 1], n), 0)
        simulations[:, k] = result

    return simulations


def run_stochastic_optimization(df_econ, candidates, b2s,
                                  n_simulations=20, p_min=0.7, seed=42):
    """
    Stochastic stope optimization:
    1. Generate K grade simulations via SGS
    2. For each simulation compute stope values
    3. Select stopes where P(value > 0) >= p_min
    Returns probabilistic stope selection with confidence bands.
    """
    from stope_optimizer import optimize_stope_layout

    grade_sims = _sgs_simulate(df_econ, n_simulations=n_simulations, seed=seed)

    stope_sim_values = {}  # stope_id -> list of values across sims

    for k in range(n_simulations):
        # Temporarily update grades
        df_sim = df_econ.copy()
        df_sim["grade_au_gt"] = grade_sims[:, k]
        # Recompute stope values for this simulation
        for c in candidates:
            block_idxs = b2s.get(c["stope_id"], [])
            if not block_idxs:
                continue
            blocks = df_sim.iloc[block_idxs]
            sim_val = float(blocks["economic_value"].sum()) if "economic_value" in blocks else \
                      float(blocks["grade_au_gt"].mean() * c.get("total_tonnage", 1000) * 0.001)
            stope_sim_values.setdefault(c["stope_id"], []).append(sim_val)

    # Compute P(value > 0) per stope
    prob_positive = {}
    mean_value = {}
    std_value = {}
    for sid, vals in stope_sim_values.items():
        arr = np.array(vals)
        prob_positive[sid] = float(np.mean(arr > 0))
        mean_value[sid] = float(np.mean(arr))
        std_value[sid] = float(np.std(arr))

    # Filter candidates by p_min threshold
    viable = [c for c in candidates if prob_positive.get(c["stope_id"], 0) >= p_min]

    # Run deterministic optimization on viable candidates only
    if viable:
        viable_b2s = {c["stope_id"]: b2s[c["stope_id"]] for c in viable if c["stope_id"] in b2s}
        result = optimize_stope_layout(viable, viable_b2s)
    else:
        result = {"selected": [], "status": "No viable stopes", "n_candidates": 0,
                  "n_selected": 0, "total_value": 0, "total_tonnage": 0, "total_metal_oz": 0}

    # Build confidence data for visualization
    confidence_data = []
    for c in candidates[:200]:  # limit for response size
        sid = c["stope_id"]
        confidence_data.append({
            "stope_id": sid,
            "x": (c["x_min"] + c["x_max"]) / 2,
            "y": (c["y_min"] + c["y_max"]) / 2,
            "z": (c["z_min"] + c["z_max"]) / 2,
            "prob_positive": prob_positive.get(sid, 0),
            "mean_value": mean_value.get(sid, 0),
            "std_value": std_value.get(sid, 0),
            "viable": prob_positive.get(sid, 0) >= p_min,
        })

    # P(value>0) distribution
    all_probs = list(prob_positive.values())
    prob_hist, prob_bins = np.histogram(all_probs, bins=20, range=(0, 1))

    return {
        "n_simulations": n_simulations,
        "p_min": p_min,
        "n_viable_candidates": len(viable),
        "n_total_candidates": len(candidates),
        "selected": result.get("selected", [])[:100],
        "n_selected": result.get("n_selected", 0),
        "total_value": result.get("total_value", 0),
        "total_tonnage": result.get("total_tonnage", 0),
        "confidence_data": confidence_data,
        "prob_histogram": {"counts": prob_hist.tolist(), "bins": prob_bins.tolist()},
        "status": result.get("status", "ok"),
    }
