"""Generate a synthetic precision-agriculture dataset for crop yield estimation.

The data mimics what a farm-level system would collect: soil tests, seasonal
weather, satellite vegetation index (peak NDVI) and management inputs. Yield is
produced by an agronomically-motivated function plus noise, so models have real
structure to learn (non-linear responses, crop-specific baselines, interactions).

Replace the output CSV with real data (same columns) to use the pipeline as-is.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# crop: (baseline yield t/ha, optimal temp C, optimal seasonal rain mm, N need kg/ha)
CROPS = {
    "wheat":   (4.2, 20.0, 350.0, 120.0),
    "rice":    (4.8, 27.0, 900.0, 110.0),
    "maize":   (5.5, 24.0, 550.0, 140.0),
    "cotton":  (2.2, 28.0, 650.0, 100.0),
    "mustard": (1.6, 19.0, 280.0, 70.0),
}
SOIL_TYPES = ["sandy", "loam", "clay", "silt"]
SOIL_FACTOR = {"sandy": 0.88, "loam": 1.05, "clay": 0.97, "silt": 1.02}


def _bell(x, optimum, width):
    """Smooth 0-1 response that peaks at `optimum`."""
    return np.exp(-0.5 * ((x - optimum) / width) ** 2)


def generate(n: int = 6000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    crop = rng.choice(list(CROPS), size=n, p=[0.28, 0.2, 0.2, 0.14, 0.18])
    soil = rng.choice(SOIL_TYPES, size=n, p=[0.2, 0.4, 0.25, 0.15])

    base = np.array([CROPS[c][0] for c in crop])
    t_opt = np.array([CROPS[c][1] for c in crop])
    r_opt = np.array([CROPS[c][2] for c in crop])
    n_need = np.array([CROPS[c][3] for c in crop])

    # Weather (seasonal aggregates)
    avg_temp = t_opt + rng.normal(0, 3.5, n)
    rainfall = np.clip(r_opt * rng.lognormal(0, 0.35, n), 40, None)
    humidity = np.clip(rng.normal(58, 15, n), 20, 95)
    solar_rad = np.clip(rng.normal(19, 3, n), 9, 28)  # MJ/m2/day
    heat_stress_days = np.clip(rng.poisson(np.clip(avg_temp - 28, 0, None) * 1.5 + 1), 0, 40)

    # Soil
    soil_n = np.clip(rng.normal(55, 20, n), 10, 120)   # kg/ha available
    soil_p = np.clip(rng.normal(22, 8, n), 4, 55)
    soil_k = np.clip(rng.normal(180, 50, n), 60, 400)
    soil_ph = np.clip(rng.normal(7.0, 0.8, n), 4.5, 9.2)
    organic_carbon = np.clip(rng.normal(0.65, 0.25, n), 0.1, 1.8)

    # Management
    fert_n = np.clip(rng.normal(n_need, 35), 0, 250)
    irrigations = rng.integers(0, 7, n)
    seed_rate = np.clip(rng.normal(1.0, 0.12, n), 0.6, 1.4)  # relative to recommended
    area_ha = np.round(rng.uniform(0.5, 20, n), 2)

    # Satellite vegetation index (peak NDVI) loosely driven by water/nutrient status
    water_idx = np.clip((rainfall + irrigations * 40) / r_opt, 0, 1.6)
    ndvi_peak = np.clip(
        0.35 + 0.28 * np.tanh(water_idx)
        + 0.0012 * (soil_n + fert_n) - 0.01 * heat_stress_days
        + rng.normal(0, 0.04, n),
        0.2, 0.92,
    )

    # Yield response
    f_temp = 0.35 + 0.65 * _bell(avg_temp, t_opt, 4.5)
    f_water = 0.3 + 0.7 * _bell(rainfall + irrigations * 40, r_opt * 1.05, r_opt * 0.45)
    f_n = 1 - np.exp(-(soil_n * 0.5 + fert_n) / (n_need * 0.7))
    f_ph = 0.65 + 0.35 * _bell(soil_ph, 6.8, 1.1)
    f_pk = 0.75 + 0.25 * np.minimum(soil_p / 20, 1) * np.minimum(soil_k / 160, 1)
    f_oc = 0.9 + 0.1 * np.minimum(organic_carbon / 0.8, 1)
    f_ndvi = 0.5 + 0.8 * (ndvi_peak - 0.2)
    f_heat = 1 - 0.012 * heat_stress_days
    f_seed = 1 - 0.8 * (seed_rate - 1.0) ** 2
    f_soil = np.array([SOIL_FACTOR[s] for s in soil])

    yield_t_ha = (
        base * f_temp * f_water * f_n * f_ph * f_pk * f_oc
        * f_ndvi * f_heat * f_seed * f_soil * 1.55
    )
    yield_t_ha *= rng.normal(1.0, 0.06, n)  # unexplained field-level noise
    yield_t_ha = np.clip(yield_t_ha, 0.2, None)

    df = pd.DataFrame({
        "crop": crop, "soil_type": soil,
        "avg_temp_c": avg_temp.round(2), "rainfall_mm": rainfall.round(1),
        "humidity_pct": humidity.round(1), "solar_rad_mj": solar_rad.round(2),
        "heat_stress_days": heat_stress_days,
        "soil_n_kg_ha": soil_n.round(1), "soil_p_kg_ha": soil_p.round(1),
        "soil_k_kg_ha": soil_k.round(1), "soil_ph": soil_ph.round(2),
        "organic_carbon_pct": organic_carbon.round(2),
        "fertilizer_n_kg_ha": fert_n.round(1), "irrigations": irrigations,
        "seed_rate_rel": seed_rate.round(2), "area_ha": area_ha,
        "ndvi_peak": ndvi_peak.round(3),
        "yield_t_ha": yield_t_ha.round(3),
    })

    # Inject realistic missingness (sensor / lab gaps) in a few columns
    for col, frac in [("soil_k_kg_ha", 0.04), ("ndvi_peak", 0.03), ("humidity_pct", 0.03)]:
        df.loc[rng.random(n) < frac, col] = np.nan
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="data/crop_yield.csv")
    args = ap.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    data = generate(args.n, args.seed)
    data.to_csv(out, index=False)
    print(f"Wrote {len(data)} rows to {out}")
    print(data.groupby("crop")["yield_t_ha"].describe()[["count", "mean", "std", "min", "max"]].round(2))
