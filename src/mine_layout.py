"""Generate simplified mine development network (decline, levels, crosscuts)."""

import numpy as np
import pandas as pd


def generate_development_network(
    selected_stopes: list[dict],
    surface_z: float = 0.0,
    decline_gradient: float = 0.1,  # 1:10
    level_spacing: float = 25.0,
    dev_cost_per_m: float = 5000.0,
    block_size: float = 5.0,
) -> dict:
    if not selected_stopes:
        return {"segments": [], "total_length_m": 0, "total_cost": 0, "levels": []}

    all_z_min = min(s["z_min"] for s in selected_stopes)
    all_z_max = max(s["z_max"] for s in selected_stopes)
    all_x_min = min(s["x_min"] for s in selected_stopes)
    all_x_max = max(s["x_max"] for s in selected_stopes)
    all_y_min = min(s["y_min"] for s in selected_stopes)
    all_y_max = max(s["y_max"] for s in selected_stopes)

    center_x = (all_x_min + all_x_max) / 2
    # Footwall drive, held close to the orebody so crosscuts stay short on
    # deposits with a wide across-strike extent.
    center_y = all_y_min - min(20.0, 0.05 * max(all_y_max - all_y_min, 1.0))

    # Deposits logged in real RL sit above the default sea-level surface, which
    # would make every depth negative and invert the decline. Collar the portal
    # just above the orebody instead.
    if surface_z <= all_z_max:
        surface_z = all_z_max + level_spacing

    # Generate levels
    levels = []
    z = all_z_max
    while z >= all_z_min:
        levels.append(z)
        z -= level_spacing
    if levels[-1] > all_z_min:
        levels.append(all_z_min)

    segments = []
    seg_id = 0

    # Decline from surface to deepest level
    decline_points = []
    for lev_z in levels:
        depth = surface_z - lev_z
        horiz_dist = depth / decline_gradient
        decline_points.append({
            "x": center_x, "y": center_y, "z": lev_z,
            "horiz_dist": horiz_dist,
        })

    for i in range(len(decline_points) - 1):
        p1, p2 = decline_points[i], decline_points[i + 1]
        dz = abs(p2["z"] - p1["z"])
        length = dz / decline_gradient * np.sqrt(1 + decline_gradient**2)
        segments.append({
            "seg_id": seg_id,
            "type": "Decline",
            "x1": p1["x"], "y1": p1["y"], "z1": p1["z"],
            "x2": p2["x"], "y2": p2["y"], "z2": p2["z"],
            "length_m": round(length, 1),
        })
        seg_id += 1

    # Level drives and crosscuts to stopes
    for lev_z in levels:
        level_stopes = [
            s for s in selected_stopes
            if s["z_min"] <= lev_z <= s["z_max"]
        ]
        if not level_stopes:
            continue

        # Level drive along strike
        drive_x_min = min(s["x_min"] for s in level_stopes) - 10
        drive_x_max = max(s["x_max"] for s in level_stopes) + 10
        drive_length = drive_x_max - drive_x_min

        segments.append({
            "seg_id": seg_id,
            "type": "Level Drive",
            "x1": drive_x_min, "y1": center_y, "z1": lev_z,
            "x2": drive_x_max, "y2": center_y, "z2": lev_z,
            "length_m": round(drive_length, 1),
        })
        seg_id += 1

        # Crosscuts from the drive to the stopes. Stopes sharing an x-corridor
        # are served by one crosscut driven to the furthest of them, and a
        # stope spanning several levels is accessed from its own level only.
        corridors: dict[float, float] = {}
        for s in level_stopes:
            if not (lev_z - level_spacing / 2 <= (s["z_min"] + s["z_max"]) / 2 < lev_z + level_spacing / 2):
                continue
            stope_cx = round((s["x_min"] + s["x_max"]) / 2 / block_size) * block_size
            stope_cy = (s["y_min"] + s["y_max"]) / 2
            if abs(stope_cy - center_y) > abs(corridors.get(stope_cx, center_y) - center_y):
                corridors[stope_cx] = stope_cy

        for stope_cx, stope_cy in corridors.items():
            xcut_length = abs(stope_cy - center_y)
            segments.append({
                "seg_id": seg_id,
                "type": "Crosscut",
                "x1": stope_cx, "y1": center_y, "z1": lev_z,
                "x2": stope_cx, "y2": stope_cy, "z2": lev_z,
                "length_m": round(xcut_length, 1),
            })
            seg_id += 1

    total_length = sum(s["length_m"] for s in segments)
    total_cost = total_length * dev_cost_per_m

    return {
        "segments": segments,
        "total_length_m": round(total_length, 1),
        "total_cost": round(total_cost, 2),
        "levels": levels,
        "n_levels": len(levels),
    }
