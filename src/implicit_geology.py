"""3D Implicit Geological Modeling via Radial Basis Functions (RBF)."""

import numpy as np
from scipy.interpolate import RBFInterpolator


def build_implicit_model(df_composite, block_df=None, grade_col="grade_au_gt",
                          ore_threshold=None, block_size=5):
    """
    Build implicit geological model using RBF interpolation.
    Signed distance field: positive = ore zone, negative = waste.

    Returns:
      - iso-surface control points for 3D visualization
      - block classification (ore_zone = True/False)
      - ore body wireframe vertices
    """
    if ore_threshold is None:
        ore_threshold = float(df_composite[grade_col].quantile(0.6)) \
            if grade_col in df_composite.columns else 0.5

    df = df_composite.dropna(subset=[grade_col]).copy()
    if len(df) < 5:
        return {"error": "Insufficient data for implicit modeling"}

    pts = df[["x", "y", "z"]].values
    grades = df[grade_col].values

    # Signed distance: ore = +1, waste = -1 (like SDF)
    signed = np.where(grades >= ore_threshold, 1.0, -1.0)

    # Fit RBF (thin-plate spline)
    try:
        rbf = RBFInterpolator(pts, signed, kernel="thin_plate_spline", smoothing=1.0)
    except Exception:
        # Fallback: linear kernel
        rbf = RBFInterpolator(pts, signed, kernel="linear", smoothing=1.0)

    # Create prediction grid
    xs = np.linspace(pts[:, 0].min(), pts[:, 0].max(), 20)
    ys = np.linspace(pts[:, 1].min(), pts[:, 1].max(), 20)
    zs = np.linspace(pts[:, 2].min(), pts[:, 2].max(), 15)
    gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
    grid_pts = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])

    sdf_values = rbf(grid_pts)

    # Iso-surface at SDF = 0 (ore boundary)
    ore_mask = sdf_values > 0
    waste_mask = ~ore_mask

    # Classify blocks if block_df provided
    block_classification = None
    if block_df is not None and len(block_df) > 0:
        bpts = block_df[["x", "y", "z"]].values
        bsdf = rbf(bpts)
        block_classification = {
            "ore_zone": (bsdf > 0).tolist(),
            "sdf_value": bsdf.tolist(),
        }

    # Sample iso-surface control points (SDF near 0)
    iso_mask = np.abs(sdf_values) < 0.3
    iso_pts = grid_pts[iso_mask]

    # Fault proxies: zones with rapid SDF gradient change
    # Compute gradient magnitude at each grid point
    sdf_3d = sdf_values.reshape(len(xs), len(ys), len(zs))
    gx_sdf = np.gradient(sdf_3d, axis=0)
    gy_sdf = np.gradient(sdf_3d, axis=1)
    gz_sdf = np.gradient(sdf_3d, axis=2)
    grad_mag = np.sqrt(gx_sdf**2 + gy_sdf**2 + gz_sdf**2).ravel()
    high_grad_mask = grad_mag > np.percentile(grad_mag, 90)

    # Lithological domains
    domain_points = {
        "ore_zone": {
            "x": grid_pts[ore_mask, 0].tolist(),
            "y": grid_pts[ore_mask, 1].tolist(),
            "z": grid_pts[ore_mask, 2].tolist(),
            "sdf": sdf_values[ore_mask].tolist(),
        },
        "waste_zone": {
            "x": grid_pts[waste_mask, 0][:500].tolist(),
            "y": grid_pts[waste_mask, 1][:500].tolist(),
            "z": grid_pts[waste_mask, 2][:500].tolist(),
        },
        "iso_surface": {
            "x": iso_pts[:, 0].tolist() if len(iso_pts) > 0 else [],
            "y": iso_pts[:, 1].tolist() if len(iso_pts) > 0 else [],
            "z": iso_pts[:, 2].tolist() if len(iso_pts) > 0 else [],
        },
        "fault_zones": {
            "x": grid_pts[high_grad_mask, 0][:200].tolist(),
            "y": grid_pts[high_grad_mask, 1][:200].tolist(),
            "z": grid_pts[high_grad_mask, 2][:200].tolist(),
        },
    }

    return {
        "ore_threshold": round(float(ore_threshold), 3),
        "n_ore_blocks": int(ore_mask.sum()),
        "n_waste_blocks": int(waste_mask.sum()),
        "domain_points": domain_points,
        "block_classification": block_classification,
        "grid_dims": {"nx": len(xs), "ny": len(ys), "nz": len(zs)},
        "ore_fraction": round(float(ore_mask.mean()), 3),
    }
