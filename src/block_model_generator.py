"""
Generate a synthetic 3D block model representing an Archean lode gold deposit.

Geological basis:
  Gold in orogenic (lode) deposits occurs as free nuggets and fine particles
  within quartz veins and shear zones — fundamentally different from
  disseminated base-metal deposits (Cu, Fe). Key characteristics modeled:

  - Log-normal grade distribution with high positive skew (nugget effect)
  - Narrow, tabular vein structures with sharp ore/waste contacts
  - High-grade shoots within veins (structural intersections)
  - Background grade halo from alteration envelope
  - Extreme outliers (coarse visible gold / nuggets)

Calibration references:
  - Red Lake Gold Mine, Ontario — mean grade ~15-20 g/t in high-grade zones,
    narrow vein widths 1-5m (Evolution Mining NI 43-101)
  - LaRonde Mine, Quebec — polymetallic Au-Zn, grades 2-6 g/t Au in wider
    stoping widths (Agnico Eagle technical reports)
  - Kalgoorlie Super Pit / Mt Charlotte — Archean greenstone gold,
    grade variability and nugget effect well-documented
  - Coefficient of variation (CV) for gold typically 100-300%
    vs 40-80% for copper (Journel & Huijbregts, 1978)
"""

import numpy as np
import pandas as pd
from pathlib import Path


def generate_block_model(
    nx: int = 40,
    ny: int = 30,
    nz: int = 20,
    block_size: float = 5.0,
    seed: int = 42,
    output_path: str | None = None,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    xs = np.arange(nx) * block_size
    ys = np.arange(ny) * block_size
    zs = -np.arange(nz) * block_size - 50

    xx, yy, zz = np.meshgrid(xs, ys, zs, indexing="ij")
    coords = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])

    n_blocks = len(coords)
    cx, cy, cz = nx * block_size / 2, ny * block_size / 2, -50 - nz * block_size / 2

    # === VEIN STRUCTURES ===
    # Main quartz vein: narrow tabular body striking NE, dipping 70° SE
    # Typical Archean lode: 1-5m true width, 50-200m strike, 100-300m down-dip
    vein1_center = np.array([cx, cy, cz])
    vein1_sigma = np.array([15, 50, 30])  # narrow across-strike, elongated along
    dist1 = np.sqrt(np.sum(((coords - vein1_center) / vein1_sigma) ** 2, axis=1))

    # Secondary vein splay (hangingwall splay, common in lode systems)
    vein2_center = np.array([cx + 25, cy - 15, cz - 10])
    vein2_sigma = np.array([10, 30, 18])
    dist2 = np.sqrt(np.sum(((coords - vein2_center) / vein2_sigma) ** 2, axis=1))

    # === GRADE MODEL (Log-normal, characteristic of gold) ===
    # Gold grades follow log-normal distribution due to nugget effect
    # In-vein grades: geometric mean ~3-5 g/t, with shoots up to 20+ g/t
    vein1_grade = np.maximum(0, 5.0 * np.exp(-dist1**2 / 2))
    vein2_grade = np.maximum(0, 3.5 * np.exp(-dist2**2 / 2))
    structure_grade = vein1_grade + vein2_grade

    # High-grade shoot: structural intersection (fold hinge / fault junction)
    # These plunging ore shoots are where most gold concentrates
    shoot_center = np.array([cx - 5, cy + 10, cz + 5])
    shoot_sigma = np.array([12, 15, 10])
    dist_shoot = np.sqrt(np.sum(((coords - shoot_center) / shoot_sigma) ** 2, axis=1))
    shoot_grade = np.maximum(0, 8.0 * np.exp(-dist_shoot**2 / 1.5))

    base_grade = structure_grade + shoot_grade

    # Spatially correlated noise (represents local grade variability)
    from scipy.ndimage import gaussian_filter
    noise_raw = rng.standard_normal(n_blocks).reshape(nx, ny, nz)
    noise_smooth = gaussian_filter(noise_raw, sigma=1.5).ravel()
    noise_smooth = noise_smooth / (noise_smooth.std() + 1e-8) * 0.6

    # Log-normal transformation for realistic gold grade distribution
    # Gold's nugget effect creates extreme positive skew
    grade_linear = base_grade + noise_smooth
    grade_mask = grade_linear > 0.1
    grade = np.zeros(n_blocks)
    grade[grade_mask] = np.exp(
        np.log(grade_linear[grade_mask] + 0.01)
        + rng.normal(0, 0.4, grade_mask.sum())
    )

    # Nugget effect: random coarse gold occurrences (visible gold / free nuggets)
    # ~2% of ore-zone blocks get a nugget bonus (characteristic of lode gold)
    in_vein = (dist1 < 2.0) | (dist2 < 2.0) | (dist_shoot < 1.5)
    nugget_mask = in_vein & (rng.random(n_blocks) < 0.02)
    grade[nugget_mask] += rng.exponential(scale=8.0, size=nugget_mask.sum())

    # Weak alteration halo around veins (pyrite-sericite-carbonate)
    halo_grade = 0.15 * np.exp(-np.minimum(dist1, dist2) / 2)
    grade += halo_grade

    grade = np.maximum(0, grade)
    grade = np.round(grade, 4)

    # Density: quartz vein (2.65) vs host rock (2.75-2.85)
    density = np.where(
        in_vein,
        2.65 + rng.normal(0, 0.03, n_blocks),
        2.80 + rng.normal(0, 0.05, n_blocks),
    )
    density = np.clip(density, 2.5, 3.1)

    # RQD: lower near veins (fractured), higher in massive host rock
    rqd = np.clip(
        75 - 20 * np.exp(-np.minimum(dist1, dist2)) + rng.normal(0, 8, n_blocks),
        15, 100,
    )

    # Rock types reflecting alteration zonation
    rock_types = np.where(
        in_vein, "Quartz Vein",
        np.where(base_grade > 1.0, "Altered Basalt",
                 np.where(base_grade > 0.2, "Sericite Zone", "Fresh Basalt"))
    )

    df = pd.DataFrame({
        "block_id": np.arange(n_blocks),
        "x": coords[:, 0],
        "y": coords[:, 1],
        "z": coords[:, 2],
        "grade_au_gt": grade,
        "density_tpm3": np.round(density, 3),
        "rqd": np.round(rqd, 1),
        "rock_type": rock_types,
        "block_size_m": block_size,
    })

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)

    return df


if __name__ == "__main__":
    data_dir = Path(__file__).resolve().parent.parent / "data"
    df = generate_block_model(output_path=str(data_dir / "block_model.csv"))
    print(f"Generated {len(df)} blocks")
    ore = df[df["grade_au_gt"] > 0.5]
    print(f"Grade stats: mean={df['grade_au_gt'].mean():.3f} g/t, "
          f"max={df['grade_au_gt'].max():.3f} g/t, "
          f"median={df['grade_au_gt'].median():.3f} g/t, "
          f"CV={df[df['grade_au_gt'] > 0]['grade_au_gt'].std() / df[df['grade_au_gt'] > 0]['grade_au_gt'].mean():.1%}")
    print(f"Ore blocks (>0.5 g/t): {len(ore)} ({100*len(ore)/len(df):.1f}%)")
    print(f"Rock types: {df['rock_type'].value_counts().to_dict()}")
