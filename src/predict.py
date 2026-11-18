"""Predict crop yield (t/ha) from the command line.

Single prediction (weather/pesticide values default to that country's latest recorded values):
  python src/predict.py --country India --crop Wheat --year 2013
  python src/predict.py --country India --crop Rice --year 2013 --avg_temp_c 27 --pesticides_tonnes 50000

Batch (CSV needs columns: country, crop, year, rainfall_mm, pesticides_tonnes, avg_temp_c):
  python src/predict.py --csv fields.csv --out predictions.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).parent))
from train import add_features  # noqa: E402


def predict_df(model, df: pd.DataFrame) -> pd.Series:
    df = df.drop(columns=["yield_t_ha"], errors="ignore")
    return pd.Series(model.predict(add_features(df)), index=df.index, name="predicted_yield_t_ha").round(3)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ROOT / "models" / "yield_model.joblib"))
    ap.add_argument("--data", default=str(ROOT / "data" / "crop_yield.csv"))
    ap.add_argument("--csv")
    ap.add_argument("--out", default="predictions.csv")
    ap.add_argument("--country")
    ap.add_argument("--crop")
    ap.add_argument("--year", type=int, default=2013)
    ap.add_argument("--rainfall_mm", type=float)
    ap.add_argument("--pesticides_tonnes", type=float)
    ap.add_argument("--avg_temp_c", type=float)
    a = ap.parse_args()

    model = joblib.load(a.model)

    if a.csv:
        df = pd.read_csv(a.csv)
        df["predicted_yield_t_ha"] = predict_df(model, df)
        df.to_csv(a.out, index=False)
        print(f"Wrote {len(df)} predictions to {a.out}")
        return

    if not (a.country and a.crop):
        ap.error("--country and --crop are required (or pass --csv)")

    hist = pd.read_csv(a.data)
    sub = hist[hist.country == a.country]
    if sub.empty:
        ap.error(f"Unknown country '{a.country}'. Examples: {', '.join(sorted(hist.country.unique())[:8])} ...")
    latest = sub.sort_values("year").iloc[-1]
    row = {
        "country": a.country, "crop": a.crop, "year": a.year,
        "rainfall_mm": a.rainfall_mm if a.rainfall_mm is not None else latest.rainfall_mm,
        "pesticides_tonnes": a.pesticides_tonnes if a.pesticides_tonnes is not None else latest.pesticides_tonnes,
        "avg_temp_c": a.avg_temp_c if a.avg_temp_c is not None else latest.avg_temp_c,
    }
    pred = float(predict_df(model, pd.DataFrame([row])).iloc[0])
    print(f"Inputs: {row}")
    print(f"Estimated yield: {pred:.2f} t/ha")
    actual = hist[(hist.country == a.country) & (hist.crop == a.crop) & (hist.year == a.year)]
    if len(actual):
        print(f"Recorded FAO yield for {a.year}: {actual.yield_t_ha.iloc[0]:.2f} t/ha")


if __name__ == "__main__":
    main()
