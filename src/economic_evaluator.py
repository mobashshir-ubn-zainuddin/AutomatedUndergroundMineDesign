"""Calculate economic block values and classify ore vs waste."""

import numpy as np
import pandas as pd
from pathlib import Path


def evaluate_economics(
    df: pd.DataFrame,
    gold_price_per_oz: float = 1900.0,
    mining_cost_per_t: float = 45.0,
    processing_cost_per_t: float = 25.0,
    ga_cost_per_t: float = 10.0,
    recovery: float = 0.92,
    cutoff_grade: float = 0.0,
    block_size: float = 5.0,
    output_path: str | None = None,
) -> pd.DataFrame:
    df = df.copy()

    tonnage = block_size**3 * df["density_tpm3"]
    # 1 troy oz = 31.1035 g
    revenue_per_t = df["grade_au_gt"] * recovery * gold_price_per_oz / 31.1035
    cost_per_t = mining_cost_per_t + processing_cost_per_t + ga_cost_per_t
    profit_per_t = revenue_per_t - cost_per_t

    df["tonnage"] = np.round(tonnage, 2)
    df["revenue_per_t"] = np.round(revenue_per_t, 2)
    df["cost_per_t"] = cost_per_t
    df["profit_per_t"] = np.round(profit_per_t, 2)
    df["economic_value"] = np.round(tonnage * profit_per_t, 2)
    df["metal_content_oz"] = np.round(tonnage * df["grade_au_gt"] / 31.1035, 4)

    # Cutoff grade: grade at which revenue = cost
    breakeven_grade = cost_per_t * 31.1035 / (recovery * gold_price_per_oz)
    df["above_cutoff"] = df["grade_au_gt"] >= breakeven_grade
    df["ore_class"] = np.where(df["above_cutoff"], "Ore", "Waste")

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)

    return df, breakeven_grade


if __name__ == "__main__":
    data_dir = Path(__file__).resolve().parent.parent / "data"
    df = pd.read_csv(data_dir / "block_model.csv")
    df_econ, cutoff = evaluate_economics(
        df, output_path=str(data_dir / "economic_block_model.csv")
    )
    ore = df_econ[df_econ["above_cutoff"]]
    print(f"Breakeven cutoff grade: {cutoff:.3f} g/t")
    print(f"Ore blocks: {len(ore)} / {len(df_econ)}")
    print(f"Total economic value (ore): ${ore['economic_value'].sum():,.0f}")
    print(f"Total metal content (ore): {ore['metal_content_oz'].sum():,.1f} oz Au")
