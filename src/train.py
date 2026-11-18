"""Train and compare crop yield regressors, then save the best pipeline.

Usage:  python src/train.py --data data/crop_yield.csv --out models
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "yield_t_ha"
CATEGORICAL = ["crop", "soil_type"]
# area_ha is a farm-size attribute, not a yield driver; kept out to avoid spurious signal.
DROP = ["area_ha"]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Agronomic features that help linear models and trees alike."""
    df = df.copy()
    df["total_water_mm"] = df["rainfall_mm"] + df["irrigations"] * 40
    df["total_n_kg_ha"] = df["soil_n_kg_ha"] + df["fertilizer_n_kg_ha"]
    df["temp_x_water"] = df["avg_temp_c"] * df["total_water_mm"] / 1000
    df["ph_dev"] = (df["soil_ph"] - 6.8).abs()
    return df


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric = [c for c in X.columns if c not in CATEGORICAL]
    num_pipe = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    cat_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([("num", num_pipe, numeric), ("cat", cat_pipe, CATEGORICAL)])


def evaluate(y_true, y_pred) -> dict:
    return {
        "R2": float(r2_score(y_true, y_pred)),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAPE_%": float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100),
    }


def main(data_path: str, out_dir: str, seed: int = 42) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path).drop(columns=DROP, errors="ignore")
    df = add_features(df)
    X, y = df.drop(columns=[TARGET]), df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=X["crop"])

    candidates = {
        "ridge": Ridge(alpha=1.0),
        "random_forest": RandomForestRegressor(n_estimators=300, min_samples_leaf=2, n_jobs=-1, random_state=seed),
        "gradient_boosting": GradientBoostingRegressor(
            n_estimators=400, learning_rate=0.05, max_depth=3, subsample=0.8, random_state=seed
        ),
    }

    cv = KFold(n_splits=5, shuffle=True, random_state=seed)
    results, fitted = {}, {}
    for name, model in candidates.items():
        pipe = Pipeline([("prep", build_preprocessor(X_train)), ("model", model)])
        cv_r2 = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="r2", n_jobs=-1)
        pipe.fit(X_train, y_train)
        metrics = evaluate(y_test, pipe.predict(X_test))
        metrics["CV_R2_mean"], metrics["CV_R2_std"] = float(cv_r2.mean()), float(cv_r2.std())
        results[name], fitted[name] = metrics, pipe
        print(f"{name:18s} CV R2 {cv_r2.mean():.3f}±{cv_r2.std():.3f} | test R2 {metrics['R2']:.3f} "
              f"MAE {metrics['MAE']:.3f} RMSE {metrics['RMSE']:.3f} MAPE {metrics['MAPE_%']:.1f}%")

    best_name = max(results, key=lambda k: results[k]["CV_R2_mean"])
    best = fitted[best_name]
    print(f"\nBest model: {best_name}")

    # Per-crop error breakdown on the held-out set
    preds = best.predict(X_test)
    per_crop = (
        pd.DataFrame({"crop": X_test["crop"].values, "y": y_test.values, "p": preds})
        .groupby("crop")
        .apply(lambda g: pd.Series({"n": len(g), "R2": r2_score(g.y, g.p), "MAE": mean_absolute_error(g.y, g.p)}),
               include_groups=False)
        .round(3)
    )
    print("\nPer-crop test performance:\n", per_crop)

    # Permutation importance on original (pre-encoding) features
    imp = permutation_importance(best, X_test, y_test, n_repeats=8, random_state=seed, n_jobs=-1, scoring="r2")
    imp_df = pd.Series(imp.importances_mean, index=X_test.columns).sort_values()

    # --- plots ---
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_test, preds, s=8, alpha=0.4)
    lim = [0, max(y_test.max(), preds.max()) * 1.05]
    ax.plot(lim, lim, "r--", lw=1)
    ax.set(xlabel="Actual yield (t/ha)", ylabel="Predicted yield (t/ha)",
           title=f"Actual vs predicted ({best_name})", xlim=lim, ylim=lim)
    fig.tight_layout()
    fig.savefig(out / "actual_vs_predicted.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6))
    imp_df.tail(12).plot.barh(ax=ax)
    ax.set(xlabel="Drop in R² when shuffled", title="Top feature importances (permutation)")
    fig.tight_layout()
    fig.savefig(out / "feature_importance.png", dpi=140)
    plt.close(fig)

    # --- persist ---
    joblib.dump(best, out / "yield_model.joblib", compress=("xz", 3))  # ~12 MB instead of ~57 MB
    report = {"best_model": best_name, "metrics": results, "per_crop": per_crop.reset_index().to_dict("records"),
              "top_features": imp_df.sort_values(ascending=False).head(10).round(4).to_dict()}
    (out / "metrics.json").write_text(json.dumps(report, indent=2))
    print(f"\nSaved model, metrics.json and plots to {out}/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/crop_yield.csv")
    ap.add_argument("--out", default="models")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    main(a.data, a.out, a.seed)
