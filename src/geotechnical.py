"""Geotechnical stability assessment for stopes and pillars."""

import numpy as np


def assess_stability(
    selected_stopes: list[dict],
    block_model_df=None,
    ucs_mean: float = 120.0,
    ucs_std: float = 25.0,
    unit_weight: float = 0.027,
    k_ratio: float = 1.5,
    min_pillar_width: float = 5.0,
) -> dict:
    if not selected_stopes:
        return {"stope_assessments": [], "pillar_assessments": [], "summary": {}}

    rng = np.random.default_rng(42)

    stope_assessments = []
    for s in selected_stopes:
        depth = abs(s["z_min"])
        span_w = s["width_m"]
        span_h = s["height_m"]
        span_l = s["length_m"]

        hydraulic_radius = (span_w * span_l) / (2 * (span_w + span_l))

        sigma_v = unit_weight * depth
        sigma_h = k_ratio * sigma_v
        sigma_max = max(sigma_v, sigma_h)

        ucs = rng.normal(ucs_mean, ucs_std)
        ucs = max(ucs, 30)

        stability_number = ucs / max(sigma_max, 0.01)

        if stability_number > 10:
            classification = "Stable"
            risk = "Low"
        elif stability_number > 5:
            classification = "Potentially Unstable"
            risk = "Medium"
        elif stability_number > 2:
            classification = "Unstable"
            risk = "High"
        else:
            classification = "Caving Expected"
            risk = "Critical"

        rqd_val = 65 + rng.normal(0, 10)
        rqd_val = np.clip(rqd_val, 20, 100)
        q_system = max(0.1, (rqd_val / 100) * (1.0) * (0.5) * (1.0) * (1.0) * (2.5))

        mathews_n = q_system * (0.2 if depth > 200 else 0.5) * (1.0 if span_h < 20 else 0.7) * 1.0

        if mathews_n / max(hydraulic_radius, 0.1) > 1.0:
            mathews_zone = "Stable"
        elif mathews_n / max(hydraulic_radius, 0.1) > 0.3:
            mathews_zone = "Transitional"
        else:
            mathews_zone = "Caving"

        stope_assessments.append({
            "stope_id": s["stope_id"],
            "depth_m": round(depth, 1),
            "span_w": span_w, "span_h": span_h, "span_l": span_l,
            "hydraulic_radius": round(hydraulic_radius, 2),
            "sigma_v": round(sigma_v, 2),
            "sigma_h": round(sigma_h, 2),
            "ucs": round(ucs, 1),
            "stability_number": round(stability_number, 2),
            "classification": classification,
            "risk_level": risk,
            "rqd": round(rqd_val, 1),
            "mathews_n": round(mathews_n, 3),
            "mathews_zone": mathews_zone,
        })

    pillar_assessments = []
    sorted_stopes = sorted(selected_stopes, key=lambda s: (s["z_min"], s["x_min"]))
    for i in range(len(sorted_stopes)):
        for j in range(i + 1, min(i + 10, len(sorted_stopes))):
            s1, s2 = sorted_stopes[i], sorted_stopes[j]
            if abs(s1["z_min"] - s2["z_min"]) > 30:
                continue

            x_gap = max(0, max(s1["x_min"], s2["x_min"]) - min(s1["x_max"], s2["x_max"]))
            y_gap = max(0, max(s1["y_min"], s2["y_min"]) - min(s1["y_max"], s2["y_max"]))
            pillar_width = max(x_gap, y_gap)

            if pillar_width <= 0 or pillar_width > 50:
                continue

            depth = (abs(s1["z_min"]) + abs(s2["z_min"])) / 2
            pillar_stress = unit_weight * depth * (1 + s1["width_m"] / max(pillar_width, 0.1))
            pillar_strength = ucs_mean * 0.6 * np.sqrt(pillar_width / max(s1["height_m"], 1))

            fos = pillar_strength / max(pillar_stress, 0.01)
            if fos > 1.5:
                status = "Adequate"
            elif fos > 1.0:
                status = "Marginal"
            else:
                status = "Inadequate"

            pillar_assessments.append({
                "stope_1": s1["stope_id"],
                "stope_2": s2["stope_id"],
                "pillar_width_m": round(pillar_width, 1),
                "depth_m": round(depth, 1),
                "pillar_stress_mpa": round(pillar_stress, 2),
                "pillar_strength_mpa": round(pillar_strength, 2),
                "factor_of_safety": round(fos, 2),
                "status": status,
                "below_minimum": pillar_width < min_pillar_width,
            })

    n_total = len(stope_assessments)
    risk_counts = {}
    for sa in stope_assessments:
        risk_counts[sa["risk_level"]] = risk_counts.get(sa["risk_level"], 0) + 1

    n_pillars = len(pillar_assessments)
    n_adequate = sum(1 for p in pillar_assessments if p["status"] == "Adequate")
    n_marginal = sum(1 for p in pillar_assessments if p["status"] == "Marginal")
    n_inadequate = sum(1 for p in pillar_assessments if p["status"] == "Inadequate")

    return {
        "stope_assessments": stope_assessments,
        "pillar_assessments": pillar_assessments,
        "summary": {
            "total_stopes": n_total,
            "risk_distribution": risk_counts,
            "avg_stability_number": round(np.mean([s["stability_number"] for s in stope_assessments]), 2) if stope_assessments else 0,
            "total_pillars": n_pillars,
            "adequate_pillars": n_adequate,
            "marginal_pillars": n_marginal,
            "inadequate_pillars": n_inadequate,
            "avg_fos": round(np.mean([p["factor_of_safety"] for p in pillar_assessments]), 2) if pillar_assessments else 0,
        },
    }
