# Crop Yield Estimator (real FAO data)

A machine learning project that estimates crop yield (tonnes/hectare) from the country, crop, year,
rainfall, pesticide use and temperature. It is trained on **real, public data**: FAOSTAT crop yields combined
with World Bank rainfall and temperature records (101 countries, 10 crops, 1990-2013).

## Project layout

```
crop_yield_estimator/
├── app.py                  # interactive Streamlit demo
├── requirements.txt
├── data/
│   ├── raw/yield_df.csv    # original downloaded file
│   └── crop_yield.csv      # cleaned data (13,130 rows), made by prepare_data.py
├── src/
│   ├── prepare_data.py     # cleaning (duplicates + leakage fix)
│   ├── train.py            # model comparison, evaluation, plots, saves best model
│   └── predict.py          # command-line predictions
└── models/                 # yield_model.joblib, metrics.json, plots
```

## Quick start

```bash
pip install -r requirements.txt
python src/prepare_data.py        # raw -> data/crop_yield.csv
python src/train.py               # -> models/
streamlit run app.py              # interactive demo
python src/predict.py --country India --crop Wheat --year 2013
```

## Data

| Column | Meaning | Source |
|--------|---------|--------|
| `country`, `crop`, `year` | Record identifier | FAOSTAT |
| `yield_t_ha` | Recorded yield in tonnes/hectare (**target**) | FAOSTAT |
| `rainfall_mm` | Average annual rainfall of the country (one long-term value per country) | World Bank |
| `pesticides_tonnes` | Total pesticides used in the country that year | FAOSTAT |
| `avg_temp_c` | Average temperature of the country that year | World Bank |

Compiled on Kaggle as the *Crop Yield Prediction Dataset* (user "patelris"). FAOSTAT data are public; please
cite FAOSTAT and the World Bank Climate Change Knowledge Portal when you use them.

**Cleaning (`prepare_data.py`)**
- 2,310 exact duplicate rows removed.
- Temperature had been joined one-to-many, so the same country-crop-year appeared many times with the same
  yield but different temperatures. Keeping those would put near-copies in both train and test sets and
  inflate the scores (data leakage). They are averaged into one row per country-crop-year (28,242 raw rows
  become 13,130 clean rows).

## Method

- **Features:** country, crop, year, rainfall, pesticides, temperature, plus `log(pesticides)` and
  `temperature × rainfall`.
- **Target:** modelled on a log scale because yields are very skewed (potatoes ~18 t/ha, sorghum ~1.4 t/ha).
- **Models:** Ridge (linear baseline), Random Forest, Histogram Gradient Boosting, compared with a naive
  baseline that predicts each country-crop's historical average.
- **Time-based evaluation** (predicting future seasons, as in real use):
  train 1990-2003, choose the model on 2004-2008, final score on 2009-2013. The deployed model is fit on
  1990-2008, so 2009-2013 are genuinely unseen.
- **Stricter test:** 5-fold cross-validation grouped by country (predicting countries never seen).

## Results (test years 2009-2013)

| Model | R² | MAE (t/ha) | WAPE |
|-------|----|-----------|------|
| Baseline (country-crop mean) | 0.864 | 1.74 | 21.4% |
| Ridge | 0.761 | 2.54 | 31.3% |
| Histogram Gradient Boosting | 0.901 | 1.49 | 18.4% |
| **Random Forest (best)** | **0.915** | **1.30** | **16.1%** |

For countries never seen in training the Random Forest scores R² 0.62 and MAE 2.8 t/ha.
Run `python src/train.py` to reproduce; exact numbers are saved in `models/metrics.json`.

## Honest limitations

- **National-level, not field-level.** There is no soil, satellite NDVI or farm-management data, so this is a
  country-scale yield estimator, not a precision-agriculture model for individual fields.
- Most of the accuracy comes from knowing the country and crop; the weather and pesticide features add a
  smaller improvement over the baseline. The model learns **associations, not causes**.
- Rainfall is one fixed value per country, so the model cannot learn year-to-year rainfall effects.
- Data stop at 2013.

## Ideas for extension

- Add field-level data (soil tests, Sentinel-2 NDVI via Google Earth Engine, IMD weather) for true
  precision agriculture
- Add Indian state/district yield data (Ministry of Agriculture statistics) for local relevance
- Prediction intervals (quantile regression or conformal prediction) and SHAP explanations
