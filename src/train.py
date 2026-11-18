"""Train and evaluate crop yield models on the REAL cleaned FAO dataset.

Evaluation is done by TIME, which is how a yield estimator is really used (predict future seasons):
    train      : 1990-2003   (fit candidate models)
    validation : 2004-2008   (choose the best model)
    test       : 2009-2013   (final, untouched score)
The deployed model is refit on 1990-2008, so 2009-2013 stay genuinely unseen.

A stricter check (hold out whole COUNTRIES the model has never seen) is also reported.

Usage:  python src/train.py
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
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "yield_t_ha"
CATEGORICAL = ["country", "crop"]
TRAIN_END, VAL_END = 2003, 2008  # test = years after VAL_END


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["log_pesticides"] = np.log1p(df["pesticides_tonnes"])
    df["temp_x_rain"] = df["avg_temp_c"] * df["rainfall_mm"] / 1000
    return df


def build_pipeline(model, X: pd.DataFrame) -> Pipeline:
    numeric = [c for c in X.columns if c not in CATEGORICAL]
    prep = ColumnTransformer([
        ("num", StandardScaler(), numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
    ])
    # Yields are right-skewed (potatoes ~18 t/ha vs sorghum ~1.4), so learn on log scale.
    reg = TransformedTargetRegressor(regressor=model, func=np.log1p, inverse_func=np.expm1)
    return Pipeline([("prep", prep), ("model", reg)])


def score(y_true, y_pred) -> dict:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return {
        "R2": float(r2_score(y_true, y_pred)),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        # weighted absolute % error: total error / total yield (safe when a yield is near 0)
        "WAPE_%": float(np.abs(y_true - y_pred).sum() / np.abs(y_true).sum() * 100),
    }


class CountryCropMean:
    """Naive baseline: predict the average historical yield of that country+crop."""

    def fit(self, df: pd.DataFrame):
        self.cc = df.groupby(["country", "crop"])[TARGET].mean()
        self.c = df.groupby("crop")[TARGET].mean()
        return self

    def predict(self, df: pd.DataFrame):
        idx = pd.MultiIndex.from_frame(df[["country", "crop"]])
        out = self.cc.reindex(idx).to_numpy()
        fallback = df["crop"].map(self.c).to_numpy()
        return np.where(np.isnan(out), fallback, out)


def main(data_path: str, out_dir: str, seed: int = 42) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(data_path)
    df = add_features(raw)
    tr, va, te = df[df.year <= TRAIN_END], df[(df.year > TRAIN_END) & (df.year <= VAL_END)], df[df.year > VAL_END]
    trva = df[df.year <= VAL_END]
    print(f"train {len(tr):,} (1990-{TRAIN_END}) | validation {len(va):,} | test {len(te):,} ({VAL_END + 1}-{df.year.max()})")

    feats = [c for c in df.columns if c not in (TARGET, "year")] + ["year"]
    X = lambda d: d[feats]  # noqa: E731

    candidates = {
        "ridge": Ridge(alpha=1.0),
        "random_forest": RandomForestRegressor(n_estimators=200, min_samples_leaf=2, max_features=0.6,
                                               n_jobs=-1, random_state=seed),
        "hist_gradient_boosting": HistGradientBoostingRegressor(max_iter=500, learning_rate=0.06,
                                                                max_leaf_nodes=31, random_state=seed),
    }

    base = CountryCropMean().fit(tr)
    val_scores = {"baseline_country_crop_mean": score(va[TARGET], base.predict(va))}
    for name, model in candidates.items():
        pipe = build_pipeline(model, X(tr)).fit(X(tr), tr[TARGET])
        val_scores[name] = score(va[TARGET], pipe.predict(X(va)))
    for k, v in val_scores.items():
        print(f"[val ] {k:28s} R2 {v['R2']:.3f}  MAE {v['MAE']:.3f}  WAPE {v['WAPE_%']:.1f}%")

    best_name = max(candidates, key=lambda k: val_scores[k]["R2"])
    print(f"\nBest on validation: {best_name}")

    # Refit on train+validation (1990-2008) and score once on the untouched test years.
    test_scores = {"baseline_country_crop_mean": score(te[TARGET], CountryCropMean().fit(trva).predict(te))}
    fitted = {}
    for name, model in candidates.items():
        fitted[name] = build_pipeline(model, X(trva)).fit(X(trva), trva[TARGET])
        test_scores[name] = score(te[TARGET], fitted[name].predict(X(te)))
    for k, v in test_scores.items():
        print(f"[test] {k:28s} R2 {v['R2']:.3f}  MAE {v['MAE']:.3f}  WAPE {v['WAPE_%']:.1f}%")
    best = fitted[best_name]
    preds = best.predict(X(te))

    # Stricter check: predict countries the model has NEVER seen (5-fold, grouped by country).
    cv_pred = cross_val_predict(build_pipeline(candidates[best_name], X(df)), X(df), df[TARGET],
                                groups=df["country"], cv=GroupKFold(n_splits=5), n_jobs=1)
    unseen = score(df[TARGET], cv_pred)
    print(f"\nUnseen-country check ({best_name}): R2 {unseen['R2']:.3f}  MAE {unseen['MAE']:.3f}  "
          f"WAPE {unseen['WAPE_%']:.1f}%")

    per_crop = (
        pd.DataFrame({"crop": te["crop"].values, "y": te[TARGET].values, "p": preds})
        .groupby("crop")
        .apply(lambda g: pd.Series({"n": len(g), "R2": r2_score(g.y, g.p), "MAE": mean_absolute_error(g.y, g.p)}),
               include_groups=False)
        .round(3)
    )
    print("\nPer-crop test performance:\n", per_crop)

    imp = permutation_importance(best, X(te), te[TARGET], n_repeats=5, random_state=seed, n_jobs=1, scoring="r2")
    imp_s = pd.Series(imp.importances_mean, index=feats).sort_values()

    # ---- plots ----
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(te[TARGET], preds, s=6, alpha=0.35)
    lim = [0, max(te[TARGET].max(), preds.max()) * 1.05]
    ax.plot(lim, lim, "r--", lw=1)
    ax.set(xlabel="Actual yield (t/ha)", ylabel="Predicted yield (t/ha)", xlim=lim, ylim=lim,
           title=f"Test years {VAL_END + 1}-{df.year.max()}: actual vs predicted")
    fig.tight_layout(); fig.savefig(out / "actual_vs_predicted.png", dpi=140); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    imp_s.plot.barh(ax=ax)
    ax.set(xlabel="Drop in R² when shuffled", title="Feature importance (permutation, test years)")
    fig.tight_layout(); fig.savefig(out / "feature_importance.png", dpi=140); plt.close(fig)

    # ---- save ----
    joblib.dump(best, out / "yield_model.joblib", compress=("xz", 3))
    report = {
        "best_model": best_name,
        "split": {"train": f"1990-{TRAIN_END}", "validation": f"{TRAIN_END + 1}-{VAL_END}",
                  "test": f"{VAL_END + 1}-{int(df.year.max())}", "deployed_model_fit_on": f"1990-{VAL_END}"},
        "validation_metrics": val_scores, "test_metrics": test_scores, "unseen_country_check": unseen,
        "per_crop": per_crop.reset_index().to_dict("records"),
        "top_features": imp_s.sort_values(ascending=False).round(4).to_dict(),
    }
    (out / "metrics.json").write_text(json.dumps(report, indent=2))
    size = (out / "yield_model.joblib").stat().st_size / 1e6
    print(f"\nSaved model ({size:.1f} MB), metrics.json and plots to {out}/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/crop_yield.csv")
    ap.add_argument("--out", default="models")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    main(a.data, a.out, a.seed)
