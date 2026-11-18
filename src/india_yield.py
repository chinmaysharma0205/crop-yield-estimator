"""India state-level crop yield model on REAL data (Kaggle "Crop Yield in Indian States" dataset).

Raw file: data/india/raw_crop_yield.csv  (19,689 rows, 30 states/UTs, 55 crops, 1997-2020; columns Crop,
Crop_Year, Season, State, Area, Production, Annual_Rainfall, Fertilizer, Pesticide, Yield).

Cleaning decisions (all counted and saved in models/india_metrics.json):
  * Coconut removed: its yield is in nuts per hectare, not tonnes, so it cannot sit next to the other crops.
  * Zero-yield rows removed (cannot tell a failed crop from a missing record).
  * Plots under 10 ha removed (yield from a tiny area is very noisy).
  * Extreme per-crop outliers removed (above Q3 + 5 x IQR of that crop, e.g. potatoes at 311 t/ha).
  * `Production` is NOT used as an input: yield is production divided by area, so using it would leak the answer.
  * Fertilizer and Pesticide are divided by Area to get per-hectare rates. They (and rainfall) turn out to be
    identical for every crop in the same state and year, i.e. STATE-level figures, not field measurements.
Units: tonnes/ha for most crops; cotton, jute and mesta are recorded in bales/ha in the source data.

Model design: a plain model that only sees state, crop, season, rainfall, fertiliser and pesticide did NOT beat a
naive baseline (the average past yield of that state + crop + season). So the model here learns the DEVIATION
from that historical average (using year/trend, rainfall anomaly, fertiliser, pesticide, area). The historical
average itself is computed leave-one-out for training rows so a row never sees its own answer.

Evaluation is by time: train 1997-2009, validation 2010-2013 (choose the model), test 2014-2020.
The deployed model is fit on 1997-2013 so 2014-2020 stay unseen. A stricter check holds out whole states.

Usage:  python src/india_yield.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "india" / "raw_crop_yield.csv"
CLEAN = ROOT / "data" / "india" / "india_clean.csv"
TARGET = "yield"
KEYS = ["state", "crop", "season"]
CATEGORICAL = ["state", "crop", "season"]
TRAIN_END, VAL_END = 2009, 2013


def clean(raw_path: Path = RAW) -> tuple[pd.DataFrame, dict]:
    d = pd.read_csv(raw_path)
    for c in ["Crop", "Season", "State"]:
        d[c] = d[c].str.strip()
    log = {"raw_rows": len(d)}

    def drop(mask, key):
        nonlocal d
        log[key] = int(mask.sum())
        d = d[~mask]

    drop(d["Crop"] == "Coconut", "removed_coconut_nuts")
    drop(d["Yield"] <= 0, "removed_zero_yield")
    drop(d["Area"] < 10, "removed_area_under_10ha")
    q1 = d.groupby("Crop")["Yield"].transform(lambda s: s.quantile(.25))
    q3 = d.groupby("Crop")["Yield"].transform(lambda s: s.quantile(.75))
    drop(d["Yield"] > q3 + 5 * (q3 - q1), "removed_extreme_outliers")

    out = pd.DataFrame({
        "state": d["State"], "crop": d["Crop"], "season": d["Season"], "year": d["Crop_Year"],
        "area_ha": d["Area"], "rainfall_mm": d["Annual_Rainfall"],
        "fert_kg_ha": d["Fertilizer"] / d["Area"], "pest_kg_ha": d["Pesticide"] / d["Area"],
        TARGET: d["Yield"],
    })
    out["fert_kg_ha"] = out["fert_kg_ha"].round(2)
    out["pest_kg_ha"] = out["pest_kg_ha"].round(4)
    out[TARGET] = out[TARGET].round(4)
    out = out.sort_values(["state", "crop", "season", "year"]).reset_index(drop=True)
    log["clean_rows"] = len(out)
    log["duplicate_state_crop_season_year"] = int(out.duplicated(["state", "crop", "season", "year"]).sum())
    return out, log


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop(columns=[TARGET], errors="ignore").copy()
    df["log_area"] = np.log1p(df["area_ha"])
    df["log_pest"] = np.log1p(df["pest_kg_ha"])
    return df


def _plain_pipeline(model, Z: pd.DataFrame) -> Pipeline:
    numeric = [c for c in Z.columns if c not in CATEGORICAL]
    prep = ColumnTransformer([
        ("num", StandardScaler(), numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
    ])
    return Pipeline([("prep", prep), ("model", model)])


class ResidualOverMean(BaseEstimator, RegressorMixin):
    """Predict yield = (historical mean of that state+crop+season) x learned adjustment (log scale)."""

    def __init__(self, model=None):
        self.model = model

    def _base(self, X: pd.DataFrame, loo_y=None) -> np.ndarray:
        idx = pd.MultiIndex.from_frame(X[KEYS])
        s = self.sum_.reindex(idx).to_numpy(dtype=float)
        n = self.cnt_.reindex(idx).to_numpy(dtype=float)
        fallback = X["crop"].map(self.crop_mean_).fillna(self.global_mean_).to_numpy(dtype=float)
        if loo_y is not None:  # training rows: leave the row's own yield out of its baseline
            with np.errstate(invalid="ignore", divide="ignore"):
                loo = (s - loo_y) / (n - 1)
            return np.where(n > 1, loo, fallback)
        with np.errstate(invalid="ignore", divide="ignore"):
            avg = s / n
        return np.where(np.isnan(avg), fallback, avg)

    def _prep(self, X: pd.DataFrame, base: np.ndarray) -> pd.DataFrame:
        Z = X.copy()
        Z["log_base"] = np.log1p(base)
        Z["rain_anom"] = Z["rainfall_mm"] - Z["state"].map(self.rain_mean_).fillna(self.rain_global_)
        return Z

    def fit(self, X: pd.DataFrame, y):
        X = X.reset_index(drop=True)
        y = np.asarray(y, dtype=float)
        ys = pd.Series(y)
        grp = ys.groupby([X[k] for k in KEYS])
        self.sum_, self.cnt_ = grp.sum(), grp.count()
        self.crop_mean_ = ys.groupby(X["crop"]).mean()
        self.global_mean_ = float(ys.mean())
        self.rain_mean_ = X.groupby("state")["rainfall_mm"].mean()
        self.rain_global_ = float(X["rainfall_mm"].mean())
        base = self._base(X, loo_y=y)
        Z = self._prep(X, base)
        self.pipe_ = _plain_pipeline(clone(self.model), Z).fit(Z, np.log1p(y) - np.log1p(base))
        return self

    def predict(self, X: pd.DataFrame):
        X = X.reset_index(drop=True)
        base = self._base(X)
        r = self.pipe_.predict(self._prep(X, base))
        return np.clip(np.expm1(np.log1p(base) + r), 0, None)


def score(y_true, y_pred) -> dict:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return {
        "R2": float(r2_score(y_true, y_pred)),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "WAPE_%": float(np.abs(y_true - y_pred).sum() / np.abs(y_true).sum() * 100),
    }


class GroupMean:
    """Naive baseline: average historical yield of that state + crop + season."""

    def fit(self, df):
        self.g = df.groupby(KEYS)[TARGET].mean()
        self.c = df.groupby("crop")[TARGET].mean()
        return self

    def predict(self, df):
        idx = pd.MultiIndex.from_frame(df[KEYS])
        out = self.g.reindex(idx).to_numpy()
        return np.where(np.isnan(out), df["crop"].map(self.c).fillna(self.c.mean()).to_numpy(), out)


def main(seed: int = 42) -> None:
    df, log = clean()
    CLEAN.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CLEAN, index=False)
    print(log)

    full = df.copy()
    feats = list(add_features(full).columns)
    X_all = add_features(full)
    tr, va, te = (full.year <= TRAIN_END), ((full.year > TRAIN_END) & (full.year <= VAL_END)), (full.year > VAL_END)
    trva = tr | va
    print(f"train {tr.sum():,} | validation {va.sum():,} | test {te.sum():,} ({VAL_END + 1}-{full.year.max()})")

    candidates = {
        "ridge": Ridge(alpha=3.0),
        "random_forest": RandomForestRegressor(n_estimators=150, min_samples_leaf=5, max_features=0.5,
                                               n_jobs=-1, random_state=seed),
        "hist_gradient_boosting": HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05,
                                                                max_leaf_nodes=15, min_samples_leaf=30,
                                                                random_state=seed),
    }
    val_scores = {"baseline_state_crop_season_mean": score(full.loc[va, TARGET],
                                                          GroupMean().fit(full[tr]).predict(full[va]))}
    for n, m in candidates.items():
        est = ResidualOverMean(m).fit(X_all[tr], full.loc[tr, TARGET])
        val_scores[n] = score(full.loc[va, TARGET], est.predict(X_all[va]))
    for k, v in val_scores.items():
        print(f"[val ] {k:32s} R2 {v['R2']:.3f}  MAE {v['MAE']:.3f}  WAPE {v['WAPE_%']:.1f}%")
    best_name = max(candidates, key=lambda k: val_scores[k]["R2"])
    print("Best on validation:", best_name)

    test_scores = {"baseline_state_crop_season_mean": score(full.loc[te, TARGET],
                                                           GroupMean().fit(full[trva]).predict(full[te]))}
    fitted = {}
    for n, m in candidates.items():
        fitted[n] = ResidualOverMean(m).fit(X_all[trva], full.loc[trva, TARGET])
        test_scores[n] = score(full.loc[te, TARGET], fitted[n].predict(X_all[te]))
    for k, v in test_scores.items():
        print(f"[test] {k:32s} R2 {v['R2']:.3f}  MAE {v['MAE']:.3f}  WAPE {v['WAPE_%']:.1f}%")
    best = fitted[best_name]
    pred = best.predict(X_all[te])

    cvp = cross_val_predict(ResidualOverMean(candidates[best_name]), X_all, full[TARGET],
                            groups=full["state"], cv=GroupKFold(n_splits=5))
    unseen = score(full[TARGET], cvp)
    print(f"Unseen-state check: R2 {unseen['R2']:.3f}  MAE {unseen['MAE']:.3f}")

    te_df = full[te].assign(pred=pred)
    top = te_df["crop"].value_counts().head(10).index
    per_crop = (te_df[te_df.crop.isin(top)].groupby("crop")
                .apply(lambda g: pd.Series({"n": len(g), "R2": r2_score(g[TARGET], g.pred),
                                            "MAE": mean_absolute_error(g[TARGET], g.pred),
                                            "mean_yield": g[TARGET].mean()}), include_groups=False).round(3))
    print(per_crop)

    imp = permutation_importance(best, X_all[te], full.loc[te, TARGET], n_repeats=3, random_state=seed,
                                 n_jobs=1, scoring="r2")
    imp_s = pd.Series(imp.importances_mean, index=feats).sort_values(ascending=False).round(4)
    print(imp_s.head(8).to_dict())

    out = ROOT / "models"
    out.mkdir(exist_ok=True)
    joblib.dump(best, out / "india_model.joblib", compress=("xz", 3))
    (out / "india_metrics.json").write_text(json.dumps({
        "best_model": best_name, "cleaning": log,
        "split": {"train": f"1997-{TRAIN_END}", "validation": f"{TRAIN_END + 1}-{VAL_END}",
                  "test": f"{VAL_END + 1}-{int(full.year.max())}", "deployed_model_fit_on": f"1997-{VAL_END}"},
        "validation_metrics": val_scores, "test_metrics": test_scores, "unseen_state_check": unseen,
        "per_crop": per_crop.reset_index().to_dict("records"), "top_features": imp_s.to_dict(),
    }, indent=2))
    print(f"Saved models/india_model.joblib ({(out / 'india_model.joblib').stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    # import by module name so the saved model refers to india_yield.ResidualOverMean (not __main__)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import india_yield

    india_yield.main()
