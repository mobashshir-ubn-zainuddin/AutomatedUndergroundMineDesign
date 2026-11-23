"""Bolani iron ore drillhole data loader and block model builder."""

import pandas as pd
import numpy as np
from pathlib import Path

BASE = Path(__file__).parent.parent / "bolani_data"


def load_drillhole_data():
    """Load collar, survey, assay, geology tables and composite them."""
    collar = pd.read_csv(BASE / "collar.csv")
    survey = pd.read_csv(BASE / "survey.csv")
    assay = pd.read_csv(BASE / "assay.csv")
    geology = pd.read_csv(BASE / "geology.csv")

    # Clean columns
    assay.columns = [c.strip().lower() for c in assay.columns]
    geology.columns = [c.strip().lower() for c in geology.columns]
    collar.columns = [c.strip().lower() for c in collar.columns]
    survey.columns = [c.strip().lower() for c in survey.columns]

    # Clean geology (has trailing empty cols)
    geology = geology[["bh_no", "from", "to", "lithocode"]].copy()
    geology.columns = ["bh_no", "from", "to", "lithocode"]

    # Desurvey: all holes are vertical (dip=-90), so X=E, Y=N, Z=elevation-depth
    collar_dict = collar.set_index("bh_no").to_dict("index")

    # Build composite intervals (sample midpoint coordinates)
    records = []
    for _, row in assay.iterrows():
        bh = row["bh_no"]
        if bh not in collar_dict:
            continue
        c = collar_dict[bh]
        mid_depth = (row["from"] + row["to"]) / 2
        x = float(c["e"]) if "e" in c else 0.0
        y = float(c["n"])
        z = float(c["z"]) - mid_depth  # vertical hole

        fe_val = row["fe"]
        fe = float(fe_val) if fe_val not in ("na", "NA", "", None) else np.nan

        sio2_val = row.get("sio2", np.nan)
        sio2 = float(sio2_val) if sio2_val not in ("na", "NA", "", None) else np.nan

        al2o3_val = row.get("al2o3", np.nan)
        al2o3 = float(al2o3_val) if al2o3_val not in ("na", "NA", "", None) else np.nan

        p_val = row.get("p", np.nan)
        p = float(p_val) if p_val not in ("na", "NA", "", None) else np.nan

        s_val = row.get("s", np.nan)
        s = float(s_val) if s_val not in ("na", "NA", "", None) else np.nan

        # Match lithocode
        geo_match = geology[
            (geology["bh_no"] == bh) &
            (geology["from"] <= row["from"]) &
            (geology["to"] >= row["to"])
        ]
        litho = geo_match["lithocode"].iloc[0] if len(geo_match) > 0 else "Unknown"

        records.append({
            "bh_no": bh, "from": row["from"], "to": row["to"],
            "x": x, "y": y, "z": z,
            "fe": fe, "sio2": sio2, "al2o3": al2o3, "p": p, "s": s,
            "lithocode": litho,
            "interval_len": row["to"] - row["from"],
        })

    df = pd.DataFrame(records)
    return df, collar


def build_bolani_block_model(df_composite, block_size=5):
    """Interpolate drillhole composites onto a regular block grid via IDW."""
    df_valid = df_composite.dropna(subset=["fe"])
    if len(df_valid) < 3:
        return pd.DataFrame()

    xs = np.arange(df_valid["x"].min(), df_valid["x"].max() + block_size, block_size)
    ys = np.arange(df_valid["y"].min(), df_valid["y"].max() + block_size, block_size)
    zs = np.arange(df_valid["z"].min(), df_valid["z"].max() + block_size, block_size)

    # Cap grid size for performance
    if len(xs) * len(ys) * len(zs) > 20000:
        block_size = max(block_size, 10)
        xs = np.arange(df_valid["x"].min(), df_valid["x"].max() + block_size, block_size)
        ys = np.arange(df_valid["y"].min(), df_valid["y"].max() + block_size, block_size)
        zs = np.arange(df_valid["z"].min(), df_valid["z"].max() + block_size, block_size)

    gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
    grid_pts = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])

    pts = df_valid[["x", "y", "z"]].values
    fe_vals = df_valid["fe"].values
    sio2_vals = df_valid["sio2"].fillna(df_valid["sio2"].mean()).values
    al2o3_vals = df_valid["al2o3"].fillna(df_valid["al2o3"].mean()).values

    # IDW interpolation
    fe_grid = _idw(grid_pts, pts, fe_vals)
    sio2_grid = _idw(grid_pts, pts, sio2_vals)
    al2o3_grid = _idw(grid_pts, pts, al2o3_vals)

    block_df = pd.DataFrame({
        "x": grid_pts[:, 0], "y": grid_pts[:, 1], "z": grid_pts[:, 2],
        "fe": fe_grid, "sio2": sio2_grid, "al2o3": al2o3_grid,
        "block_size": block_size,
    })
    return block_df


def _idw(query_pts, data_pts, values, power=2, k=8):
    """Inverse distance weighting interpolation."""
    result = np.zeros(len(query_pts))
    for i, q in enumerate(query_pts):
        dists = np.sqrt(np.sum((data_pts - q) ** 2, axis=1))
        # Use k nearest
        idx = np.argsort(dists)[:k]
        d = dists[idx]
        v = values[idx]
        if d[0] < 1e-6:
            result[i] = v[0]
        else:
            w = 1.0 / (d ** power)
            result[i] = np.sum(w * v) / np.sum(w)
    return result


def get_bolani_stats(df_composite):
    """Return summary statistics and histogram data for Bolani composites."""
    df_fe = df_composite.dropna(subset=["fe"])
    df_sio2 = df_composite.dropna(subset=["sio2"])

    fe = df_fe["fe"].values
    sio2 = df_sio2["sio2"].values
    al2o3 = df_composite.dropna(subset=["al2o3"])["al2o3"].values

    # Histogram bins
    fe_counts, fe_bins = np.histogram(fe, bins=30)
    sio2_counts, sio2_bins = np.histogram(sio2[sio2 > 0], bins=30)
    ln_sio2 = np.log(sio2[sio2 > 0])
    ln_sio2_counts, ln_sio2_bins = np.histogram(ln_sio2, bins=30)

    # Depth profile (sample Fe vs depth)
    depth_profile = df_fe.sort_values("from")[["from", "fe", "bh_no"]].to_dict("records")

    return {
        "n_samples": int(len(df_fe)),
        "fe_mean": round(float(fe.mean()), 2),
        "fe_std": round(float(fe.std()), 2),
        "fe_min": round(float(fe.min()), 2),
        "fe_max": round(float(fe.max()), 2),
        "fe_hist": {"counts": fe_counts.tolist(), "bins": fe_bins.tolist()},
        "sio2_hist": {"counts": sio2_counts.tolist(), "bins": sio2_bins.tolist()},
        "ln_sio2_hist": {"counts": ln_sio2_counts.tolist(), "bins": ln_sio2_bins.tolist()},
        "scatter_fe_sio2": {
            "fe": df_sio2["fe"].dropna().tolist(),
            "sio2": df_sio2["sio2"].tolist(),
            "litho": df_sio2["lithocode"].tolist(),
        },
        "scatter_fe_al2o3": {
            "fe": df_composite.dropna(subset=["al2o3", "fe"])["fe"].tolist(),
            "al2o3": df_composite.dropna(subset=["al2o3", "fe"])["al2o3"].tolist(),
            "litho": df_composite.dropna(subset=["al2o3", "fe"])["lithocode"].tolist(),
        },
        "depth_profile": depth_profile[:200],
        "lithocodes": df_composite["lithocode"].value_counts().to_dict(),
    }
