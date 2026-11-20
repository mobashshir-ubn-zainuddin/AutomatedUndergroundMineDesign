"""Generate a synthetic 3D block model representing an underground gold deposit."""

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
    zs = -np.arange(nz) * block_size - 50  # start 50m below surface

    xx, yy, zz = np.meshgrid(xs, ys, zs, indexing="ij")
    coords = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])

    n_blocks = len(coords)
    cx, cy, cz = nx * block_size / 2, ny * block_size / 2, -50 - nz * block_size / 2

    # Two ore zones: main vein and secondary lens
    vein1_center = np.array([cx, cy, cz])
    vein1_sigma = np.array([40, 60, 25])
    dist1 = np.sqrt(np.sum(((coords - vein1_center) / vein1_sigma) ** 2, axis=1))
    grade1 = np.maximum(0, 6.0 * np.exp(-dist1**2 / 2))

    vein2_center = np.array([cx + 30, cy - 20, cz - 15])
    vein2_sigma = np.array([25, 35, 15])
    dist2 = np.sqrt(np.sum(((coords - vein2_center) / vein2_sigma) ** 2, axis=1))
    grade2 = np.maximum(0, 4.5 * np.exp(-dist2**2 / 2))

    base_grade = grade1 + grade2

    # Add spatially correlated noise via smoothed random field
    noise_raw = rng.standard_normal(n_blocks)
    noise_grid = noise_raw.reshape(nx, ny, nz)
    from scipy.ndimage import gaussian_filter
    noise_smooth = gaussian_filter(noise_grid, sigma=2.0).ravel()
    noise_smooth = noise_smooth / (noise_smooth.std() + 1e-8) * 0.8

    grade = np.maximum(0, base_grade + noise_smooth + rng.normal(0, 0.15, n_blocks))

    density = 2.7 + rng.normal(0, 0.05, n_blocks)
    density = np.clip(density, 2.5, 3.0)

    rqd = np.clip(65 + 15 * np.exp(-dist1 / 3) + rng.normal(0, 8, n_blocks), 20, 100)

    rock_types = np.where(
        base_grade > 1.5, "Ore",
        np.where(base_grade > 0.3, "Transition", "Waste")
    )

    df = pd.DataFrame({
        "block_id": np.arange(n_blocks),
        "x": coords[:, 0],
        "y": coords[:, 1],
        "z": coords[:, 2],
        "grade_au_gt": np.round(grade, 4),
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
    print(f"Grade stats: mean={df['grade_au_gt'].mean():.3f}, "
          f"max={df['grade_au_gt'].max():.3f}, "
          f"ore blocks (>0.5 g/t)={len(df[df['grade_au_gt'] > 0.5])}")
