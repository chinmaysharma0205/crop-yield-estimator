"""Clean the real FAO crop-yield dataset and write data/crop_yield.csv.

Raw file: data/raw/yield_df.csv  (FAOSTAT crop yields + World Bank rainfall + FAOSTAT pesticides + temperature,
compiled by Kaggle user "patelris" as "Crop Yield Prediction Dataset").

Problems in the raw file that this script fixes:
  1. 2,310 exact duplicate rows.
  2. Temperature was joined one-to-many: the same country/crop/year appears many times with the
     same yield but different temperature readings. Keeping all of them would put near-copies of
     the same record in both the training and test sets and inflate the scores (data leakage).
     We collapse them to ONE row per (country, crop, year) using the mean temperature.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def prepare(raw_path: str, out_path: str) -> pd.DataFrame:
    raw = pd.read_csv(raw_path, index_col=0)
    print(f"raw rows: {len(raw):,}")

    df = raw.drop_duplicates()
    print(f"after dropping exact duplicates: {len(df):,}")

    df = (
        df.groupby(["Area", "Item", "Year"], as_index=False)
        .agg(
            yield_hg_ha=("hg/ha_yield", "first"),
            rainfall_mm=("average_rain_fall_mm_per_year", "first"),
            pesticides_tonnes=("pesticides_tonnes", "first"),
            avg_temp_c=("avg_temp", "mean"),
        )
    )
    df["yield_t_ha"] = df["yield_hg_ha"] / 10_000  # hectogram/ha -> tonne/ha
    df = df.drop(columns="yield_hg_ha")
    df = df.rename(columns={"Area": "country", "Item": "crop", "Year": "year"})
    df["crop"] = df["crop"].replace({"Rice, paddy": "Rice", "Plantains and others": "Plantains"})
    df["avg_temp_c"] = df["avg_temp_c"].round(2)
    df["yield_t_ha"] = df["yield_t_ha"].round(4)
    df = df[["country", "crop", "year", "rainfall_mm", "pesticides_tonnes", "avg_temp_c", "yield_t_ha"]]
    df = df.sort_values(["country", "crop", "year"]).reset_index(drop=True)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"clean rows (one per country-crop-year): {len(df):,} | countries: {df.country.nunique()} | "
          f"crops: {df.crop.nunique()} | years: {df.year.min()}-{df.year.max()}")
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw/yield_df.csv")
    ap.add_argument("--out", default="data/crop_yield.csv")
    a = ap.parse_args()
    prepare(a.raw, a.out)
