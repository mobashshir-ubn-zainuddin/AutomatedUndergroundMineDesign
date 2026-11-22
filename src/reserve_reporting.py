"""JORC / NI 43-101 compliant reserve reporting module."""

import numpy as np
import pandas as pd


# Resource classification thresholds based on drill spacing (metres)
# Measured < 50m, Indicated 50-100m, Inferred > 100m
CLASSIFICATION_RULES = {
    "Measured": lambda dist: dist < 50,
    "Indicated": lambda dist: 50 <= dist < 100,
    "Inferred": lambda dist: dist >= 100,
}

LITHO_CONFIDENCE = {
    "Quartz Vein": "Measured",
    "Altered Basalt": "Indicated",
    "Sericite Zone": "Indicated",
    "Fresh Basalt": "Inferred",
    # Bolani
    "HMO": "Measured",   # Hard Massive Ore
    "LIO": "Measured",   # Lateritic Iron Ore
    "HLO": "Indicated",  # Hard Laminated Ore
    "Lat": "Inferred",   # Laterite
    "Unknown": "Inferred",
}


def classify_resource(df, collar_df=None, commodity="Au", grade_col="grade_au_gt",
                       block_size=5, density=2.7):
    """Classify blocks into Measured/Indicated/Inferred based on drill proximity."""
    df = df.copy()
    block_vol = block_size ** 3  # m³
    tonnes_per_block = block_vol * density

    if "rock_type" in df.columns:
        df["classification"] = df["rock_type"].map(
            lambda rt: LITHO_CONFIDENCE.get(rt, "Inferred")
        )
    elif collar_df is not None and len(collar_df) > 0:
        # Assign classification by distance to nearest drillhole
        collar_pts = collar_df[["e", "n"]].rename(columns={"e": "x", "n": "y"}).values
        block_pts = df[["x", "y"]].values
        min_dists = np.array([
            np.min(np.sqrt(np.sum((collar_pts - pt) ** 2, axis=1)))
            for pt in block_pts
        ])
        df["drill_dist"] = min_dists
        df["classification"] = pd.cut(
            min_dists,
            bins=[0, 50, 100, np.inf],
            labels=["Measured", "Indicated", "Inferred"]
        ).astype(str)
    else:
        df["classification"] = "Inferred"

    df["tonnage"] = tonnes_per_block

    return df


def generate_reserve_table(df_classified, cutoff_grades, grade_col="grade_au_gt",
                            commodity="Au", price_per_unit=1900, recovery=0.92,
                            mining_cost=45, processing_cost=25, ga_cost=10,
                            block_size=5, density=2.7):
    """
    Generate JORC/NI 43-101 compliant summary tables.
    Returns cutoff sensitivity table and classification breakdown.
    """
    block_vol = block_size ** 3
    tonnes_per_block = block_vol * density
    total_opex_per_t = mining_cost + processing_cost + ga_cost

    # --- Per-classification table at base cutoff ---
    base_cutoff = cutoff_grades[len(cutoff_grades) // 2]  # middle cutoff
    classifications = ["Measured", "Indicated", "Inferred"]
    class_rows = []

    for cls in classifications:
        subset = df_classified[
            (df_classified["classification"] == cls) &
            (df_classified[grade_col] >= base_cutoff)
        ]
        if len(subset) == 0:
            class_rows.append({
                "classification": cls, "blocks": 0, "tonnes": 0,
                "mean_grade": 0, "contained_metal": 0, "in_situ_value": 0
            })
            continue

        tonnes = len(subset) * tonnes_per_block
        mean_grade = float(subset[grade_col].mean())

        # Au: g/t → oz/t = /31.1035
        if commodity == "Au":
            contained_metal = tonnes * mean_grade * recovery / 31.1035
            in_situ_value = contained_metal * price_per_unit - tonnes * total_opex_per_t
        else:
            # Fe: % → fraction
            contained_metal = tonnes * mean_grade / 100.0
            in_situ_value = contained_metal * price_per_unit - tonnes * total_opex_per_t

        class_rows.append({
            "classification": cls,
            "blocks": int(len(subset)),
            "tonnes": round(tonnes, 0),
            "mean_grade": round(mean_grade, 3),
            "contained_metal": round(contained_metal, 1),
            "in_situ_value": round(in_situ_value, 2),
        })

    # Total row
    tot_tonnes = sum(r["tonnes"] for r in class_rows)
    tot_metal = sum(r["contained_metal"] for r in class_rows)
    all_above = df_classified[df_classified[grade_col] >= base_cutoff]
    tot_grade = float(all_above[grade_col].mean()) if len(all_above) > 0 else 0
    class_rows.append({
        "classification": "TOTAL",
        "blocks": sum(r["blocks"] for r in class_rows),
        "tonnes": tot_tonnes,
        "mean_grade": round(tot_grade, 3),
        "contained_metal": tot_metal,
        "in_situ_value": sum(r["in_situ_value"] for r in class_rows),
    })

    # --- Cutoff sensitivity table ---
    sensitivity_rows = []
    for cog in cutoff_grades:
        above = df_classified[df_classified[grade_col] >= cog]
        if len(above) == 0:
            sensitivity_rows.append({
                "cutoff_grade": cog, "tonnes": 0, "mean_grade": 0,
                "contained_metal": 0, "in_situ_value": 0
            })
            continue
        tonnes = len(above) * tonnes_per_block
        mean_grade = float(above[grade_col].mean())
        if commodity == "Au":
            contained_metal = tonnes * mean_grade * recovery / 31.1035
        else:
            contained_metal = tonnes * mean_grade / 100.0
        in_situ_value = contained_metal * price_per_unit - tonnes * total_opex_per_t
        sensitivity_rows.append({
            "cutoff_grade": round(cog, 3),
            "tonnes": round(tonnes, 0),
            "mean_grade": round(mean_grade, 3),
            "contained_metal": round(contained_metal, 1),
            "in_situ_value": round(in_situ_value, 2),
        })

    # Grade-tonnage curve
    gt_curve = []
    cogs_fine = np.linspace(
        df_classified[grade_col].quantile(0.05),
        df_classified[grade_col].quantile(0.95),
        40
    )
    for cog in cogs_fine:
        above = df_classified[df_classified[grade_col] >= cog]
        tonnes = len(above) * tonnes_per_block
        mean_g = float(above[grade_col].mean()) if len(above) > 0 else 0
        gt_curve.append({"cutoff": round(float(cog), 3), "tonnes": round(tonnes, 0), "grade": round(mean_g, 3)})

    return {
        "classification_table": class_rows,
        "sensitivity_table": sensitivity_rows,
        "grade_tonnage_curve": gt_curve,
        "base_cutoff": base_cutoff,
        "commodity": commodity,
        "standard": "JORC 2012 / NI 43-101",
        "reporting_date": "2026",
    }
