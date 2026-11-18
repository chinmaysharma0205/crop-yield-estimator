"""Field-level yield SIMULATOR trained on SYNTHETIC data (not real measurements).

Real, free, large datasets do not combine soil nutrients (N, P, K, pH), irrigation, satellite NDVI and yield.
To let people explore how such precision-agriculture inputs *could* matter, this module:
  1. generates a synthetic dataset from agronomy-style formulas that I wrote (bell-shaped temperature and
     water responses, diminishing returns to nitrogen, pH optimum, etc.) plus noise, and
  2. trains a model on it.

The accuracy of this model only shows how well it learned the simulator's formulas. It says NOTHING about
real-farm accuracy. The real-data model lives in train.py.

Usage:  python src/simulator.py            # generate data + train -> models/field_sim_*
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent.parent
TARGET = "yield_t_ha"
CATEGORICAL = ["crop", "soil_type"]
CROP_NAMES = ["wheat", "rice", "maize", "cotton", "mustard"]
SOIL_TYPES = ["sandy", "loam", "clay", "silt"]

# crop: (baseline yield t/ha, optimal temp C, optimal seasonal rain mm, N need kg/ha)
CROPS = {
    "wheat":   (4.2, 20.0, 350.0, 120.0),
    "rice":    (4.8, 27.0, 900.0, 110.0),
    "maize":   (5.5, 24.0, 550.0, 140.0),
    "cotton":  (2.2, 28.0, 650.0, 100.0),
    "mustard": (1.6, 19.0, 280.0, 70.0),
}
SOIL_FACTOR = {"sandy": 0.88, "loam": 1.05, "clay": 0.97, "silt": 1.02}


def _bell(x, optimum, width):
    return np.exp(-0.5 * ((x - optimum) / width) ** 2)


def generate(n: int = 6000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    crop = rng.choice(CROP_NAMES, size=n, p=[0.28, 0.2, 0.2, 0.14, 0.18])
    soil = rng.choice(SOIL_TYPES, size=n, p=[0.2, 0.4, 0.25, 0.15])
    base = np.array([CROPS[c][0] for c in crop])
    t_opt = np.array([CROPS[c][1] for c in crop])
    r_opt = np.array([CROPS[c][2] for c in crop])
    n_need = np.array([CROPS[c][3] for c in crop])

    avg_temp = t_opt + rng.normal(0, 3.5, n)
    rainfall = np.clip(r_opt * rng.lognormal(0, 0.35, n), 40, None)
    humidity = np.clip(rng.normal(58, 15, n), 20, 95)
    solar_rad = np.clip(rng.normal(19, 3, n), 9, 28)
    heat_days = np.clip(rng.poisson(np.clip(avg_temp - 28, 0, None) * 1.5 + 1), 0, 40)

    soil_n = np.clip(rng.normal(55, 20, n), 10, 120)
    soil_p = np.clip(rng.normal(22, 8, n), 4, 55)
    soil_k = np.clip(rng.normal(180, 50, n), 60, 400)
    soil_ph = np.clip(rng.normal(7.0, 0.8, n), 4.5, 9.2)
    organic_c = np.clip(rng.normal(0.65, 0.25, n), 0.1, 1.8)

    fert_n = np.clip(rng.normal(n_need, 35), 0, 250)
    irrigations = rng.integers(0, 7, n)
    seed_rate = np.clip(rng.normal(1.0, 0.12, n), 0.6, 1.4)
    area_ha = np.round(rng.uniform(0.5, 20, n), 2)

    water_idx = np.clip((rainfall + irrigations * 40) / r_opt, 0, 1.6)
    ndvi = np.clip(0.35 + 0.28 * np.tanh(water_idx) + 0.0012 * (soil_n + fert_n) - 0.01 * heat_days
                   + rng.normal(0, 0.04, n), 0.2, 0.92)

    f_temp = 0.35 + 0.65 * _bell(avg_temp, t_opt, 4.5)
    f_water = 0.3 + 0.7 * _bell(rainfall + irrigations * 40, r_opt * 1.05, r_opt * 0.45)
    f_n = 1 - np.exp(-(soil_n * 0.5 + fert_n) / (n_need * 0.7))
    f_ph = 0.65 + 0.35 * _bell(soil_ph, 6.8, 1.1)
    f_pk = 0.75 + 0.25 * np.minimum(soil_p / 20, 1) * np.minimum(soil_k / 160, 1)
    f_oc = 0.9 + 0.1 * np.minimum(organic_c / 0.8, 1)
    f_ndvi = 0.5 + 0.8 * (ndvi - 0.2)
    f_heat = 1 - 0.012 * heat_days
    f_seed = 1 - 0.8 * (seed_rate - 1.0) ** 2
    f_soil = np.array([SOIL_FACTOR[s] for s in soil])

    y = base * f_temp * f_water * f_n * f_ph * f_pk * f_oc * f_ndvi * f_heat * f_seed * f_soil * 1.55
    y = np.clip(y * rng.normal(1.0, 0.06, n), 0.2, None)

    return pd.DataFrame({
        "crop": crop, "soil_type": soil, "avg_temp_c": avg_temp.round(2), "rainfall_mm": rainfall.round(1),
        "humidity_pct": humidity.round(1), "solar_rad_mj": solar_rad.round(2), "heat_stress_days": heat_days,
        "soil_n_kg_ha": soil_n.round(1), "soil_p_kg_ha": soil_p.round(1), "soil_k_kg_ha": soil_k.round(1),
        "soil_ph": soil_ph.round(2), "organic_carbon_pct": organic_c.round(2),
        "fertilizer_n_kg_ha": fert_n.round(1), "irrigations": irrigations, "seed_rate_rel": seed_rate.round(2),
        "area_ha": area_ha, "ndvi_peak": ndvi.round(3), TARGET: y.round(3),
    })


def add_sim_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop(columns=["area_ha", TARGET], errors="ignore").copy()
    df["total_water_mm"] = df["rainfall_mm"] + df["irrigations"] * 40
    df["total_n_kg_ha"] = df["soil_n_kg_ha"] + df["fertilizer_n_kg_ha"]
    df["temp_x_water"] = df["avg_temp_c"] * df["total_water_mm"] / 1000
    df["ph_dev"] = (df["soil_ph"] - 6.8).abs()
    return df


def _pipeline(model, X):
    numeric = [c for c in X.columns if c not in CATEGORICAL]
    prep = ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
    ])
    return Pipeline([("prep", prep), ("model", model)])


def train(seed: int = 42) -> None:
    data_path = ROOT / "data" / "synthetic" / "field_sim.csv"
    data_path.parent.mkdir(parents=True, exist_ok=True)
    df = generate(6000, seed)
    df.to_csv(data_path, index=False)
    X, y = add_sim_features(df), df[TARGET]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=df["crop"])

    candidates = {
        "ridge": Ridge(alpha=1.0),
        "random_forest": RandomForestRegressor(n_estimators=150, min_samples_leaf=3, n_jobs=-1, random_state=seed),
        "hist_gradient_boosting": HistGradientBoostingRegressor(max_iter=400, learning_rate=0.06, random_state=seed),
    }
    cv = KFold(5, shuffle=True, random_state=seed)
    results, fitted = {}, {}
    for name, m in candidates.items():
        pipe = _pipeline(m, Xtr)
        cv_r2 = cross_val_score(pipe, Xtr, ytr, cv=cv, scoring="r2", n_jobs=1)
        pipe.fit(Xtr, ytr)
        p = pipe.predict(Xte)
        results[name] = {"CV_R2": float(cv_r2.mean()), "R2": float(r2_score(yte, p)),
                         "MAE": float(mean_absolute_error(yte, p)),
                         "RMSE": float(np.sqrt(mean_squared_error(yte, p)))}
        fitted[name] = pipe
        print(f"{name:24s} CV R2 {cv_r2.mean():.3f} | test R2 {results[name]['R2']:.3f} MAE {results[name]['MAE']:.3f}")
    best = max(results, key=lambda k: results[k]["CV_R2"])
    out = ROOT / "models"
    out.mkdir(exist_ok=True)
    joblib.dump(fitted[best], out / "field_sim_model.joblib", compress=("xz", 3))
    (out / "field_sim_metrics.json").write_text(json.dumps(
        {"best_model": best, "synthetic": True, "metrics": results}, indent=2))
    size = (out / "field_sim_model.joblib").stat().st_size / 1e6
    print(f"Best: {best}. Saved models/field_sim_model.joblib ({size:.1f} MB) and field_sim_metrics.json")


if __name__ == "__main__":
    train()
