"""ML-based grade prediction using Random Forest and spatial features."""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create spatial and derived features for grade prediction."""
    feat = df[["x", "y", "z", "density_tpm3", "rqd"]].copy()

    cx = df["x"].mean()
    cy = df["y"].mean()
    cz = df["z"].mean()

    feat["dist_to_center"] = np.sqrt(
        (df["x"] - cx) ** 2 + (df["y"] - cy) ** 2 + (df["z"] - cz) ** 2
    )
    feat["depth"] = -df["z"]
    feat["xy_dist"] = np.sqrt((df["x"] - cx) ** 2 + (df["y"] - cy) ** 2)
    feat["xz_dist"] = np.sqrt((df["x"] - cx) ** 2 + (df["z"] - cz) ** 2)

    feat["x_norm"] = (df["x"] - df["x"].min()) / (df["x"].max() - df["x"].min() + 1e-8)
    feat["y_norm"] = (df["y"] - df["y"].min()) / (df["y"].max() - df["y"].min() + 1e-8)
    feat["z_norm"] = (df["z"] - df["z"].min()) / (df["z"].max() - df["z"].min() + 1e-8)

    return feat


def train_grade_model(
    df: pd.DataFrame,
    model_type: str = "random_forest",
    test_size: float = 0.2,
    seed: int = 42,
) -> dict:
    features = prepare_features(df)
    target = df["grade_au_gt"].values

    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=test_size, random_state=seed
    )

    if model_type == "random_forest":
        model = RandomForestRegressor(
            n_estimators=100, max_depth=15, min_samples_leaf=5,
            random_state=seed, n_jobs=-1,
        )
    elif model_type == "gradient_boosting":
        model = GradientBoostingRegressor(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            min_samples_leaf=10, random_state=seed,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    train_metrics = {
        "rmse": float(np.sqrt(mean_squared_error(y_train, y_pred_train))),
        "mae": float(mean_absolute_error(y_train, y_pred_train)),
        "r2": float(r2_score(y_train, y_pred_train)),
    }
    test_metrics = {
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred_test))),
        "mae": float(mean_absolute_error(y_test, y_pred_test)),
        "r2": float(r2_score(y_test, y_pred_test)),
    }

    importances = dict(zip(features.columns, model.feature_importances_))

    # Predict on full dataset
    y_pred_all = model.predict(features)
    residuals = target - y_pred_all

    # Uncertainty estimation via prediction std from RF trees
    uncertainty = None
    if model_type == "random_forest":
        tree_preds = np.array([t.predict(features.values) for t in model.estimators_])
        uncertainty = tree_preds.std(axis=0)

    return {
        "model": model,
        "model_type": model_type,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "feature_importances": importances,
        "y_test": y_test.tolist(),
        "y_pred_test": y_pred_test.tolist(),
        "y_pred_all": y_pred_all.tolist(),
        "residuals": residuals.tolist(),
        "uncertainty": uncertainty.tolist() if uncertainty is not None else None,
        "X_test_coords": {
            "x": df.iloc[X_test.index]["x"].tolist(),
            "y": df.iloc[X_test.index]["y"].tolist(),
            "z": df.iloc[X_test.index]["z"].tolist(),
        },
    }
