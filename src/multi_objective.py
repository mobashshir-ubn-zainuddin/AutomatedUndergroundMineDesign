"""Multi-Objective Stope Optimization via ε-constraint method with Pareto frontier."""

import numpy as np
from itertools import product


def _compute_stope_objectives(candidates, selected_ids, dev_cost_per_stope=50000):
    """Compute NPV, geotechnical risk, and development cost for a stope set."""
    sel = [c for c in candidates if c["stope_id"] in selected_ids]
    if not sel:
        return 0, 0, 0

    npv = sum(s["total_value"] for s in sel)

    # Geotechnical risk: proxy = avg (height/width) ratio (larger = more unstable)
    risk_scores = []
    for s in sel:
        hw = s.get("height_m", 10) / max(s.get("width_m", 5), 1)
        # normalise 0-1 (hw ratio 1=low, 5=high)
        risk_scores.append(min(hw / 5.0, 1.0))
    geo_risk = float(np.mean(risk_scores)) if risk_scores else 0

    dev_cost = len(sel) * dev_cost_per_stope

    return npv, geo_risk, dev_cost


def run_multi_objective_optimization(candidates, b2s, alpha1=1.0, alpha2=0.5, alpha3=0.3,
                                      n_epsilon_steps=8):
    """
    ε-constraint multi-objective optimization:
    Maximize: α1·NPV − α2·Rgeo − α3·Cdev
    Subject to: Rgeo ≤ εrisk, Cdev ≤ εcapex

    Sweeps ε values to build Pareto frontier.
    Returns frontier points and recommended solution.
    """
    from stope_optimizer import optimize_stope_layout

    if not candidates:
        return {"error": "No candidates"}

    # Compute per-stope objectives
    stope_npv = {c["stope_id"]: c["total_value"] for c in candidates}
    stope_risk = {}
    for c in candidates:
        hw = c.get("height_m", 10) / max(c.get("width_m", 5), 1)
        stope_risk[c["stope_id"]] = min(hw / 5.0, 1.0)

    # Base solution: unconstrained optimum
    base_result = optimize_stope_layout(candidates, b2s)
    base_sel_ids = {s["stope_id"] for s in base_result.get("selected", [])}
    base_npv, base_risk, base_dev = _compute_stope_objectives(candidates, base_sel_ids)

    # Pareto sweep: vary risk epsilon from base_risk down to 0.3
    risk_min = 0.3
    risk_steps = np.linspace(base_risk, risk_min, n_epsilon_steps)

    pareto_points = []
    for eps_risk in risk_steps:
        # Filter candidates to those below risk threshold
        filtered = [c for c in candidates
                    if stope_risk.get(c["stope_id"], 0) <= eps_risk]
        if len(filtered) < 2:
            continue
        filtered_b2s = {c["stope_id"]: b2s[c["stope_id"]]
                        for c in filtered if c["stope_id"] in b2s}

        result = optimize_stope_layout(filtered, filtered_b2s)
        sel_ids = {s["stope_id"] for s in result.get("selected", [])}
        npv, risk, dev = _compute_stope_objectives(candidates, sel_ids)

        pareto_points.append({
            "eps_risk": round(float(eps_risk), 3),
            "npv": round(npv, 2),
            "geo_risk": round(risk, 4),
            "dev_cost": round(dev, 2),
            "n_stopes": result.get("n_selected", 0),
            "total_tonnage": result.get("total_tonnage", 0),
            "weighted_obj": round(alpha1 * npv - alpha2 * risk * 1e6 - alpha3 * dev, 2),
        })

    # Pick recommended: maximize weighted objective
    recommended = max(pareto_points, key=lambda p: p["weighted_obj"]) if pareto_points else None

    # Build 3D Pareto data for visualization
    pareto_3d = {
        "npv": [p["npv"] for p in pareto_points],
        "geo_risk": [p["geo_risk"] for p in pareto_points],
        "dev_cost": [p["dev_cost"] for p in pareto_points],
        "n_stopes": [p["n_stopes"] for p in pareto_points],
        "eps_risk": [p["eps_risk"] for p in pareto_points],
    }

    # Alpha sensitivity
    alpha_sweep = []
    for a2 in [0.1, 0.3, 0.5, 0.7, 1.0]:
        for pt in pareto_points:
            obj = alpha1 * pt["npv"] - a2 * pt["geo_risk"] * 1e6 - alpha3 * pt["dev_cost"]
            alpha_sweep.append({"alpha2": a2, "npv": pt["npv"], "risk": pt["geo_risk"], "obj": round(obj, 2)})

    return {
        "pareto_points": pareto_points,
        "pareto_3d": pareto_3d,
        "recommended": recommended,
        "base_npv": round(base_npv, 2),
        "base_risk": round(base_risk, 4),
        "base_n_stopes": base_result.get("n_selected", 0),
        "alphas": {"alpha1": alpha1, "alpha2": alpha2, "alpha3": alpha3},
        "alpha_sweep": alpha_sweep[:50],
        "n_pareto_points": len(pareto_points),
    }
