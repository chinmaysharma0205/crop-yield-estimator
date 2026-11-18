"""Predict crop yield for a single field (CLI) or a CSV of fields.

Single field:
  python src/predict.py --crop wheat --soil_type loam --avg_temp_c 19 --rainfall_mm 320 \
      --humidity_pct 55 --solar_rad_mj 19 --heat_stress_days 2 --soil_n_kg_ha 60 \
      --soil_p_kg_ha 24 --soil_k_kg_ha 190 --soil_ph 7.1 --organic_carbon_pct 0.7 \
      --fertilizer_n_kg_ha 120 --irrigations 4 --seed_rate_rel 1.0 --ndvi_peak 0.78

Batch:
  python src/predict.py --csv fields.csv --out predictions.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from train import add_features  # noqa: E402

FIELDS = {
    "avg_temp_c": 22.0, "rainfall_mm": 400.0, "humidity_pct": 58.0, "solar_rad_mj": 19.0,
    "heat_stress_days": 2, "soil_n_kg_ha": 55.0, "soil_p_kg_ha": 22.0, "soil_k_kg_ha": 180.0,
    "soil_ph": 7.0, "organic_carbon_pct": 0.65, "fertilizer_n_kg_ha": 100.0, "irrigations": 3,
    "seed_rate_rel": 1.0, "ndvi_peak": 0.7,
}


def predict_df(model, df: pd.DataFrame) -> pd.Series:
    df = df.drop(columns=["area_ha", "yield_t_ha"], errors="ignore")
    return pd.Series(model.predict(add_features(df)), index=df.index, name="predicted_yield_t_ha").round(3)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/yield_model.joblib")
    ap.add_argument("--csv")
    ap.add_argument("--out", default="predictions.csv")
    ap.add_argument("--crop", choices=["wheat", "rice", "maize", "cotton", "mustard"])
    ap.add_argument("--soil_type", choices=["sandy", "loam", "clay", "silt"], default="loam")
    for k, v in FIELDS.items():
        ap.add_argument(f"--{k}", type=float, default=v)
    ap.add_argument("--area_ha", type=float, default=1.0, help="Used only to report total production")
    a = ap.parse_args()

    model = joblib.load(a.model)

    if a.csv:
        df = pd.read_csv(a.csv)
        df["predicted_yield_t_ha"] = predict_df(model, df)
        df.to_csv(a.out, index=False)
        print(f"Wrote {len(df)} predictions to {a.out}")
        return

    if not a.crop:
        ap.error("--crop is required for single-field prediction (or pass --csv)")
    row = {"crop": a.crop, "soil_type": a.soil_type, **{k: getattr(a, k) for k in FIELDS}}
    pred = float(predict_df(model, pd.DataFrame([row])).iloc[0])
    print(f"Estimated yield: {pred:.2f} t/ha  ({pred * a.area_ha:.2f} t over {a.area_ha} ha)")


if __name__ == "__main__":
    main()
