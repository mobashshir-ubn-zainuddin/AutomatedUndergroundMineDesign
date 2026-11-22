"""Ventilation Network Design using Hardy Cross iterative method."""

import numpy as np


# Constants
MIN_AIRFLOW_PER_KW = 0.06   # m³/s per kW diesel
MAX_WBT = 27.0              # °C wet-bulb temperature limit
AIR_DENSITY = 1.2           # kg/m³
MINE_DEPTH = 200            # m (default)


def _hardy_cross_iteration(branches, nodes, max_iter=100, tol=1e-4):
    """
    Hardy Cross method for pipe network flow balancing.
    branches: list of dicts with keys: id, from_node, to_node, R (resistance), Q (initial flow)
    nodes: list of node ids with source/sink designation
    Returns balanced flows.
    """
    Q = {b["id"]: b["Q"] for b in branches}
    branch_map = {b["id"]: b for b in branches}

    for _ in range(max_iter):
        max_correction = 0
        # Find loops via simple mesh analysis
        # For mine networks, use single main loop + branches
        # Correction: ΔQ = -ΣRQ|Q| / (2·ΣR|Q|)
        numerator = sum(b["R"] * Q[b["id"]] * abs(Q[b["id"]]) for b in branches)
        denominator = 2 * sum(b["R"] * abs(Q[b["id"]]) for b in branches)
        if abs(denominator) < 1e-10:
            break
        delta_q = -numerator / denominator
        for b in branches:
            Q[b["id"]] += delta_q
        max_correction = abs(delta_q)
        if max_correction < tol:
            break

    return Q


def design_ventilation_network(dev_segments, selected_stopes,
                                 diesel_power_kw=500, ambient_temp_c=20,
                                 geothermal_gradient=0.025):
    """
    Design ventilation circuit from development network.
    Returns airflow requirements, fan sizing, and circuit diagram data.
    """
    n_active_faces = max(len(selected_stopes) // 5, 2)

    # Airflow requirements
    q_diesel = diesel_power_kw * MIN_AIRFLOW_PER_KW  # m³/s per face
    q_total = q_diesel * n_active_faces * 1.3  # 30% leakage allowance

    # Thermal check: estimate wet-bulb temperature at depth
    max_depth = max((s.get("z_min", 0) for s in selected_stopes), default=0)
    depth_m = abs(max_depth)
    temp_increase = depth_m * geothermal_gradient
    wbt_at_depth = ambient_temp_c + temp_increase * 0.6  # WBT rise ~60% of dry-bulb
    temp_compliant = wbt_at_depth <= MAX_WBT

    # Build ventilation circuit from dev segments
    # Main airways: decline + main level drives
    branches = []
    nodes = set()
    branch_id = 0

    if dev_segments:
        for seg in dev_segments[:30]:
            n1 = f"N{int(seg.get('x1', 0))}_{int(seg.get('z1', 0))}"
            n2 = f"N{int(seg.get('x2', 0))}_{int(seg.get('z2', 0))}"
            nodes.add(n1); nodes.add(n2)
            L = seg.get("length_m", 10)
            # Resistance = friction factor * L / A² (A = cross-section ~12m² for 4×3m drive)
            A = 12.0
            f = 0.012  # friction factor
            R = f * L / (A ** 2)
            # Initial flow estimate based on segment type
            Q0 = q_total if seg.get("type") == "decline" else q_total * 0.5
            branches.append({"id": branch_id, "from_node": n1, "to_node": n2,
                              "R": R, "Q": Q0, "length": L, "type": seg.get("type", "drive")})
            branch_id += 1
    else:
        # Synthetic circuit: main intake, return, stope airways
        branches = [
            {"id": 0, "from_node": "surface", "to_node": "bottom", "R": 0.5, "Q": q_total, "length": depth_m, "type": "intake_shaft"},
            {"id": 1, "from_node": "bottom", "to_node": "level1", "R": 0.3, "Q": q_total * 0.6, "length": 500, "type": "level_drive"},
            {"id": 2, "from_node": "level1", "to_node": "surface", "R": 0.4, "Q": q_total, "length": depth_m, "type": "return_raise"},
        ]
        nodes = {"surface", "bottom", "level1"}

    # Balance flows
    balanced_Q = _hardy_cross_iteration(branches, list(nodes))

    # Fan sizing: ΔP = R * Q²
    total_R = sum(b["R"] for b in branches)
    fan_q = q_total
    fan_dp = total_R * fan_q ** 2 * AIR_DENSITY  # Pa
    fan_power_kw = (fan_dp * fan_q) / (1000 * 0.7)  # kW at 70% efficiency

    # Booster fan locations (high-resistance segments)
    boosters = []
    for b in branches:
        q_b = balanced_Q.get(b["id"], b["Q"])
        dp_b = b["R"] * q_b ** 2 * AIR_DENSITY
        if dp_b > 500:  # Pa threshold
            boosters.append({
                "segment": b["id"],
                "type": b.get("type", "drive"),
                "dp_pa": round(dp_b, 1),
                "recommended_kw": round(dp_b * abs(q_b) / (1000 * 0.7), 1),
            })

    # Circuit diagram data
    circuit_nodes = []
    for b in branches:
        q_b = balanced_Q.get(b["id"], b["Q"])
        circuit_nodes.append({
            "id": b["id"],
            "from": b["from_node"], "to": b["to_node"],
            "type": b.get("type", "drive"),
            "length_m": b.get("length", 0),
            "flow_m3s": round(abs(q_b), 2),
            "velocity_ms": round(abs(q_b) / 12.0, 2),
            "dp_pa": round(b["R"] * q_b ** 2 * AIR_DENSITY, 1),
        })

    # Level-by-level airflow summary
    level_summary = []
    z_levels = sorted({s.get("z_min", 0) for s in selected_stopes})
    stopes_per_level = max(1, len(selected_stopes) // max(len(z_levels), 1))
    for z in z_levels[:8]:
        q_level = q_diesel * stopes_per_level
        level_summary.append({
            "elevation": z,
            "n_faces": stopes_per_level,
            "required_flow_m3s": round(q_level, 1),
            "compliant": q_level / max(stopes_per_level, 1) >= MIN_AIRFLOW_PER_KW * diesel_power_kw,
        })

    return {
        "total_airflow_required_m3s": round(q_total, 1),
        "n_active_faces": n_active_faces,
        "diesel_power_kw": diesel_power_kw,
        "main_fan_kw": round(fan_power_kw, 1),
        "main_fan_dp_pa": round(fan_dp, 1),
        "wbt_at_depth": round(wbt_at_depth, 1),
        "temp_compliant": bool(temp_compliant),
        "max_depth_m": round(depth_m, 1),
        "n_booster_fans": len(boosters),
        "booster_fans": boosters[:5],
        "circuit": circuit_nodes,
        "level_summary": level_summary,
        "regulatory_check": {
            "min_airflow_per_kw": MIN_AIRFLOW_PER_KW,
            "actual_airflow_per_kw": round(q_total / max(n_active_faces * diesel_power_kw, 1), 4),
            "max_wbt_c": MAX_WBT,
            "actual_wbt_c": round(wbt_at_depth, 1),
            "compliant": bool(temp_compliant and q_total / max(n_active_faces * diesel_power_kw, 1) >= MIN_AIRFLOW_PER_KW),
        }
    }
