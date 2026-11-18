# Crop Yield Estimator for Precision Agriculture

A machine learning pipeline that estimates field-level crop yield (tonnes/hectare) from soil tests,
seasonal weather, satellite vegetation index (peak NDVI) and management inputs. Useful for harvest
planning, input optimisation (fertiliser/irrigation) and early-season production forecasts.

## Project layout

```
crop_yield_estimator/
├── requirements.txt
├── src/
│   ├── generate_data.py   # synthetic dataset generator (agronomy-based)
│   ├── train.py           # preprocessing, model comparison, evaluation, plots, saves best model
│   └── predict.py         # CLI for single-field or batch prediction
├── data/crop_yield.csv    # created by generate_data.py
└── models/                # yield_model.joblib, metrics.json, plots
```

## Quick start

```bash
pip install -r requirements.txt
python src/generate_data.py --n 6000          # -> data/crop_yield.csv
python src/train.py                           # -> models/
python src/predict.py --crop wheat --soil_type loam --avg_temp_c 19 --rainfall_mm 320 \
    --soil_n_kg_ha 60 --fertilizer_n_kg_ha 120 --irrigations 4 --ndvi_peak 0.78 --area_ha 5
```

## Live demo (for presentation)

```bash
pip install -r requirements.txt
streamlit run app.py
```

Opens in the browser at http://localhost:8501 with four tabs: **Predict yield** (sidebar sliders for
crop, soil, weather, NDVI and management, instant estimate and total production), **What-if analysis**
(vary fertiliser, irrigation, rainfall, temperature, pH or NDVI and see the yield response),
**Model performance** (metrics, actual-vs-predicted, feature importance) and **Dataset**.

## Features

| Group | Columns |
|-------|---------|
| Crop / soil | `crop`, `soil_type`, `soil_ph`, `organic_carbon_pct`, `soil_n/p/k_kg_ha` |
| Weather | `avg_temp_c`, `rainfall_mm`, `humidity_pct`, `solar_rad_mj`, `heat_stress_days` |
| Remote sensing | `ndvi_peak` |
| Management | `fertilizer_n_kg_ha`, `irrigations`, `seed_rate_rel` |
| Engineered | `total_water_mm`, `total_n_kg_ha`, `temp_x_water`, `ph_dev` |

Target: `yield_t_ha`.

## Method

1. **Preprocessing** – median imputation + scaling for numeric, mode imputation + one-hot for categorical
   (handles the missing sensor/lab values the generator injects).
2. **Models compared** – Ridge (baseline), Random Forest, Gradient Boosting, selected by 5-fold CV R².
3. **Evaluation** – held-out 20% test set (stratified by crop): R², MAE, RMSE, MAPE, plus per-crop breakdown.
4. **Interpretability** – permutation importance on the original features.

## Using real data

Provide a CSV with the same columns (see `generate_data.py`) at `data/crop_yield.csv`. Typical real sources:
Sentinel-2 / Landsat NDVI (Google Earth Engine), NASA POWER or IMD weather, soil health card data, and
farmer-reported or crop-cutting-experiment yields. For real deployments:

- Use **grouped or spatial CV** (e.g. by district/farm) to avoid leakage between neighbouring fields.
- Use **time-based splits** (train on earlier seasons, test on the latest) to measure forecast skill.
- For in-season forecasts, use only features available at forecast time (e.g. NDVI up to that date).

## Caveats

The bundled dataset is **synthetic**, so reported scores show the pipeline works, not real-world
accuracy. Expect lower scores on real data, where yield noise, pests, and management variation are larger.

## Ideas for extension

- Time-series NDVI (multi-date) with a GRU/LSTM or gradient boosting on summary statistics
- Quantile regression / conformal prediction for yield uncertainty intervals
- Hyperparameter search (Optuna) and SHAP explanations
- Streamlit or FastAPI front-end for farmers and extension workers
