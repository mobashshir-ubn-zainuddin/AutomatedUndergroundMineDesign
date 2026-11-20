"""Generate candidate stopes using a floating stope algorithm."""

import numpy as np
import pandas as pd
from itertools import product


def generate_candidate_stopes(
    df: pd.DataFrame,
    block_size: float = 5.0,
    min_width: int = 2,
    max_width: int = 4,
    min_height: int = 2,
    max_height: int = 6,
    min_length: int = 2,
    max_length: int = 5,
    min_stope_value: float = 0.0,
    step: int = 1,
) -> tuple[list[dict], dict]:
    ore = df[df["above_cutoff"]].copy()
    if ore.empty:
        return [], {}

    xs = np.sort(df["x"].unique())
    ys = np.sort(df["y"].unique())
    zs = np.sort(df["z"].unique())

    block_lookup = {}
    for _, row in df.iterrows():
        key = (row["x"], row["y"], row["z"])
        block_lookup[key] = row

    candidates = []
    stope_id = 0

    dim_combos = list(product(
        range(min_width, max_width + 1),
        range(min_length, max_length + 1),
        range(min_height, max_height + 1),
    ))

    for sw, sl, sh in dim_combos:
        for ix in range(0, len(xs) - sw + 1, step):
            for iy in range(0, len(ys) - sl + 1, step):
                for iz in range(0, len(zs) - sh + 1, step):
                    stope_xs = xs[ix:ix + sw]
                    stope_ys = ys[iy:iy + sl]
                    stope_zs = zs[iz:iz + sh]

                    block_ids = []
                    total_value = 0.0
                    total_tonnage = 0.0
                    total_metal = 0.0
                    grade_sum = 0.0
                    n_ore = 0

                    for sx in stope_xs:
                        for sy in stope_ys:
                            for sz in stope_zs:
                                key = (sx, sy, sz)
                                if key in block_lookup:
                                    b = block_lookup[key]
                                    block_ids.append(int(b["block_id"]))
                                    total_value += b["economic_value"]
                                    total_tonnage += b["tonnage"]
                                    total_metal += b["metal_content_oz"]
                                    grade_sum += b["grade_au_gt"]
                                    if b["above_cutoff"]:
                                        n_ore += 1

                    if not block_ids or total_value <= min_stope_value:
                        continue

                    n_blocks = len(block_ids)
                    ore_pct = n_ore / n_blocks if n_blocks > 0 else 0
                    if ore_pct < 0.3:
                        continue

                    candidates.append({
                        "stope_id": stope_id,
                        "x_min": float(stope_xs[0]),
                        "x_max": float(stope_xs[-1] + block_size),
                        "y_min": float(stope_ys[0]),
                        "y_max": float(stope_ys[-1] + block_size),
                        "z_min": float(stope_zs[0]),
                        "z_max": float(stope_zs[-1] + block_size),
                        "width_m": sw * block_size,
                        "length_m": sl * block_size,
                        "height_m": sh * block_size,
                        "n_blocks": n_blocks,
                        "n_ore_blocks": n_ore,
                        "ore_fraction": round(ore_pct, 3),
                        "total_value": round(total_value, 2),
                        "total_tonnage": round(total_tonnage, 2),
                        "total_metal_oz": round(total_metal, 4),
                        "avg_grade": round(grade_sum / n_blocks, 4) if n_blocks else 0,
                        "block_ids": block_ids,
                    })
                    stope_id += 1

    # Build block-to-stope mapping
    block_to_stopes = {}
    for s in candidates:
        for bid in s["block_ids"]:
            block_to_stopes.setdefault(bid, []).append(s["stope_id"])

    return candidates, block_to_stopes


if __name__ == "__main__":
    from pathlib import Path
    data_dir = Path(__file__).resolve().parent.parent / "data"
    df = pd.read_csv(data_dir / "economic_block_model.csv")

    # Use larger step and smaller range for quick test
    candidates, b2s = generate_candidate_stopes(
        df, step=2, min_width=2, max_width=3,
        min_height=3, max_height=5, min_length=2, max_length=4
    )
    print(f"Generated {len(candidates)} candidate stopes")
    if candidates:
        vals = [c["total_value"] for c in candidates]
        print(f"Value range: ${min(vals):,.0f} to ${max(vals):,.0f}")
