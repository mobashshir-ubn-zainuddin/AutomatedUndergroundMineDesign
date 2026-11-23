"""Backfill Optimization: CRF/paste fill network flow model."""

import numpy as np


# Cost parameters ($/m³)
CRF_COST_PER_M3 = 85     # cemented rockfill
PASTE_COST_PER_M3 = 120  # paste fill
WAIT_COST_PER_DAY = 5000 # production delay penalty


def _build_precedence_graph(selected_stopes):
    """
    Build primary/secondary stope precedence graph.
    Primary stopes: mined first (odd columns), secondary: adjacent (even columns).
    """
    # Group stopes by x-column (pillar-based classification)
    stopes_by_x = {}
    for s in selected_stopes:
        x_col = round(s["x_min"] / 10) * 10
        stopes_by_x.setdefault(x_col, []).append(s)

    x_cols = sorted(stopes_by_x.keys())
    primary_ids = set()
    secondary_ids = set()

    for i, xc in enumerate(x_cols):
        for s in stopes_by_x[xc]:
            if i % 2 == 0:
                primary_ids.add(s["stope_id"])
            else:
                secondary_ids.add(s["stope_id"])

    # Precedence: each secondary requires adjacent primaries to be filled
    precedence = {}
    for s in selected_stopes:
        if s["stope_id"] in secondary_ids:
            # Find adjacent primaries (within one block width)
            adj_primary = [
                p["stope_id"] for p in selected_stopes
                if p["stope_id"] in primary_ids and
                abs(p["x_min"] - s["x_min"]) <= s.get("width_m", 5) * 2
            ]
            precedence[s["stope_id"]] = adj_primary

    return primary_ids, secondary_ids, precedence


def _stope_volume(s, block_size=5):
    """Volume of a stope in m³."""
    w = s.get("width_m", block_size)
    h = s.get("height_m", block_size)
    l = s.get("length_m", block_size)
    return w * h * l


def optimize_backfill(selected_stopes, block_size=5,
                       fill_type="paste", batch_capacity_m3_day=500):
    """
    Network flow backfill optimization.
    Minimize: Cfill · Σ V_fill
    Subject to: fill primaries before mining secondaries.

    Returns:
      - fill schedule by period
      - cost breakdown
      - dependency graph for visualization
    """
    if not selected_stopes:
        return {"error": "No stopes to schedule backfill"}

    primary_ids, secondary_ids, precedence = _build_precedence_graph(selected_stopes)
    stope_map = {s["stope_id"]: s for s in selected_stopes}

    cost_per_m3 = PASTE_COST_PER_M3 if fill_type == "paste" else CRF_COST_PER_M3

    # Schedule: primaries mine → fill → secondaries mine
    schedule = []
    period = 1
    filled_ids = set()
    mined_ids = set()

    # Phase 1: Mine all primaries
    for sid in primary_ids:
        s = stope_map.get(sid)
        if s is None:
            continue
        vol = _stope_volume(s, block_size)
        fill_days = max(1, int(vol / batch_capacity_m3_day))
        schedule.append({
            "period": period,
            "stope_id": sid,
            "action": "mine",
            "volume_m3": round(vol, 1),
            "type": "primary",
            "fill_type": None,
            "fill_cost": 0,
            "value": round(s.get("total_value", 0), 2),
        })
        mined_ids.add(sid)

    period = 2
    # Phase 2: Fill primaries in batches
    fill_queue = list(primary_ids)
    batched_vol = 0
    batch_items = []
    for sid in fill_queue:
        s = stope_map.get(sid)
        if s is None:
            continue
        vol = _stope_volume(s, block_size)
        fill_cost = vol * cost_per_m3
        batched_vol += vol
        batch_items.append(sid)
        filled_ids.add(sid)
        if batched_vol >= batch_capacity_m3_day:
            schedule.append({
                "period": period,
                "stope_id": ",".join(str(i) for i in batch_items),
                "action": "fill",
                "volume_m3": round(batched_vol, 1),
                "type": "primary_fill",
                "fill_type": fill_type,
                "fill_cost": round(batched_vol * cost_per_m3, 2),
                "value": 0,
            })
            period += 1
            batched_vol = 0
            batch_items = []

    if batch_items:
        schedule.append({
            "period": period,
            "stope_id": ",".join(str(i) for i in batch_items),
            "action": "fill",
            "volume_m3": round(batched_vol, 1),
            "type": "primary_fill",
            "fill_type": fill_type,
            "fill_cost": round(batched_vol * cost_per_m3, 2),
            "value": 0,
        })

    period += 1
    # Phase 3: Mine secondaries (all primaries filled)
    for sid in secondary_ids:
        s = stope_map.get(sid)
        if s is None:
            continue
        # Check all required primaries are filled
        required = precedence.get(sid, [])
        ready = all(r in filled_ids for r in required)
        vol = _stope_volume(s, block_size)
        schedule.append({
            "period": period if ready else period + 1,
            "stope_id": sid,
            "action": "mine",
            "volume_m3": round(vol, 1),
            "type": "secondary",
            "fill_type": None,
            "fill_cost": 0,
            "value": round(s.get("total_value", 0), 2),
            "ready": ready,
        })

    # Economics
    total_fill_vol = sum(
        _stope_volume(stope_map[sid], block_size)
        for sid in primary_ids if sid in stope_map
    )
    total_fill_cost = total_fill_vol * cost_per_m3

    total_value = sum(s.get("total_value", 0) for s in selected_stopes)
    net_value = total_value - total_fill_cost

    # Build dependency graph for visualization
    nodes_viz = []
    edges_viz = []
    for s in selected_stopes[:40]:
        sid = s["stope_id"]
        is_primary = sid in primary_ids
        nodes_viz.append({
            "id": sid,
            "x": (s["x_min"] + s["x_max"]) / 2,
            "y": (s["y_min"] + s["y_max"]) / 2,
            "z": (s["z_min"] + s["z_max"]) / 2,
            "type": "primary" if is_primary else "secondary",
            "value": round(s.get("total_value", 0), 2),
        })
        for dep in precedence.get(sid, []):
            edges_viz.append({"from": dep, "to": sid})

    # Period summary
    max_period = max((e["period"] for e in schedule), default=1)
    period_summary = []
    for p in range(1, max_period + 1):
        items = [e for e in schedule if e["period"] == p]
        mine_val = sum(e["value"] for e in items if e["action"] == "mine")
        fill_cost = sum(e["fill_cost"] for e in items if e["action"] == "fill")
        period_summary.append({
            "period": p,
            "n_mine": sum(1 for e in items if e["action"] == "mine"),
            "n_fill": sum(1 for e in items if e["action"] == "fill"),
            "mine_value": round(mine_val, 2),
            "fill_cost": round(fill_cost, 2),
            "net": round(mine_val - fill_cost, 2),
        })

    return {
        "n_primary": len(primary_ids),
        "n_secondary": len(secondary_ids),
        "total_fill_volume_m3": round(total_fill_vol, 1),
        "fill_type": fill_type,
        "cost_per_m3": cost_per_m3,
        "total_fill_cost": round(total_fill_cost, 2),
        "total_stope_value": round(total_value, 2),
        "net_value": round(net_value, 2),
        "schedule": schedule[:60],
        "period_summary": period_summary,
        "dependency_graph": {"nodes": nodes_viz, "edges": edges_viz},
        "n_periods": max_period,
        "batch_capacity_m3_day": batch_capacity_m3_day,
    }
