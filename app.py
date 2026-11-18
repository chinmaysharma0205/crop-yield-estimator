"""Interactive demo: Crop Yield Estimator for Precision Agriculture.

Run:  streamlit run app.py
"""
import json
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))
from train import add_features  # noqa: E402

st.set_page_config(page_title="Crop Yield Estimator", page_icon="🌾", layout="wide")


@st.cache_resource
def load_model():
    return joblib.load(ROOT / "models" / "yield_model.joblib")


@st.cache_data
def load_data():
    return pd.read_csv(ROOT / "data" / "crop_yield.csv")


@st.cache_data
def load_metrics():
    return json.loads((ROOT / "models" / "metrics.json").read_text())


def predict(model, row: dict) -> float:
    return float(model.predict(add_features(pd.DataFrame([row])))[0])


model, data, report = load_model(), load_data(), load_metrics()

st.title("🌾 Crop Yield Estimator")
st.caption("Machine learning for precision agriculture — estimate field yield from soil, weather, "
           "satellite NDVI and management inputs.")

tab_pred, tab_whatif, tab_perf, tab_data = st.tabs(
    ["🔮 Predict yield", "🧪 What-if analysis", "📊 Model performance", "🗂 Dataset"])

# ---------------- sidebar inputs ----------------
with st.sidebar:
    st.header("Field inputs")
    crop = st.selectbox("Crop", ["wheat", "rice", "maize", "cotton", "mustard"])
    soil_type = st.selectbox("Soil type", ["loam", "sandy", "clay", "silt"])
    area = st.number_input("Field area (ha)", 0.5, 100.0, 5.0, 0.5)
    st.subheader("Weather (season)")
    avg_temp = st.slider("Average temperature (°C)", 10.0, 38.0, 20.0, 0.5)
    rainfall = st.slider("Rainfall (mm)", 40, 1800, 350, 10)
    humidity = st.slider("Humidity (%)", 20, 95, 58)
    solar = st.slider("Solar radiation (MJ/m²/day)", 9.0, 28.0, 19.0, 0.5)
    heat_days = st.slider("Heat-stress days", 0, 40, 2)
    st.subheader("Soil test")
    soil_n = st.slider("Soil N (kg/ha)", 10, 120, 55)
    soil_p = st.slider("Soil P (kg/ha)", 4, 55, 22)
    soil_k = st.slider("Soil K (kg/ha)", 60, 400, 180)
    soil_ph = st.slider("Soil pH", 4.5, 9.2, 7.0, 0.1)
    oc = st.slider("Organic carbon (%)", 0.1, 1.8, 0.65, 0.05)
    st.subheader("Management & remote sensing")
    fert_n = st.slider("Fertiliser N (kg/ha)", 0, 250, 120)
    irrig = st.slider("Number of irrigations", 0, 6, 4)
    seed_rate = st.slider("Seed rate (relative to recommended)", 0.6, 1.4, 1.0, 0.05)
    ndvi = st.slider("Peak NDVI (satellite)", 0.2, 0.92, 0.75, 0.01)

row = {
    "crop": crop, "soil_type": soil_type, "avg_temp_c": avg_temp, "rainfall_mm": rainfall,
    "humidity_pct": humidity, "solar_rad_mj": solar, "heat_stress_days": heat_days,
    "soil_n_kg_ha": soil_n, "soil_p_kg_ha": soil_p, "soil_k_kg_ha": soil_k, "soil_ph": soil_ph,
    "organic_carbon_pct": oc, "fertilizer_n_kg_ha": fert_n, "irrigations": irrig,
    "seed_rate_rel": seed_rate, "ndvi_peak": ndvi,
}

# ---------------- predict ----------------
with tab_pred:
    pred = predict(model, row)
    crop_avg = data.loc[data["crop"] == crop, "yield_t_ha"].mean()
    c1, c2, c3 = st.columns(3)
    c1.metric("Estimated yield", f"{pred:.2f} t/ha", f"{pred - crop_avg:+.2f} vs {crop} average")
    c2.metric("Total production", f"{pred * area:.1f} t", f"over {area:g} ha")
    mae = report["metrics"][report["best_model"]]["MAE"]
    c3.metric("Typical error (MAE)", f"±{mae:.2f} t/ha", "on held-out test data")

    lo, hi = max(pred - 2 * mae, 0), pred + 2 * mae
    st.info(f"Approximate plausible range: **{lo:.2f} – {hi:.2f} t/ha** "
            "(rough ±2×MAE band, not a formal confidence interval).")

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.hist(data.loc[data["crop"] == crop, "yield_t_ha"], bins=30, alpha=0.7)
    ax.axvline(pred, color="red", lw=2, label=f"Your field: {pred:.2f}")
    ax.set(xlabel="Yield (t/ha)", ylabel="Fields", title=f"Where your field sits among {crop} fields")
    ax.legend()
    fig.tight_layout()
    st.pyplot(fig)
    st.caption("Change the inputs in the sidebar — the estimate updates instantly.")

# ---------------- what-if ----------------
with tab_whatif:
    st.subheader("How does yield respond to one input? (all other inputs held fixed)")
    options = {
        "Fertiliser N (kg/ha)": ("fertilizer_n_kg_ha", np.linspace(0, 250, 40)),
        "Number of irrigations": ("irrigations", np.arange(0, 7)),
        "Rainfall (mm)": ("rainfall_mm", np.linspace(40, 1500, 40)),
        "Average temperature (°C)": ("avg_temp_c", np.linspace(10, 38, 40)),
        "Soil pH": ("soil_ph", np.linspace(4.5, 9.2, 40)),
        "Peak NDVI": ("ndvi_peak", np.linspace(0.2, 0.92, 40)),
    }
    label = st.selectbox("Vary", list(options))
    col, grid = options[label]
    sweep = pd.DataFrame([{**row, col: v} for v in grid])
    ys = model.predict(add_features(sweep))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(grid, ys, marker="o" if len(grid) < 10 else None)
    ax.axvline(row[col], color="gray", ls="--", label="current value")
    best = grid[int(np.argmax(ys))]
    ax.set(xlabel=label, ylabel="Predicted yield (t/ha)", title=f"Yield vs {label} ({crop})")
    ax.legend()
    fig.tight_layout()
    st.pyplot(fig)
    st.success(f"Highest predicted yield in this range is at **{best:.2f}** "
               f"({ys.max():.2f} t/ha). Beyond a point, extra inputs give diminishing returns.")

# ---------------- performance ----------------
with tab_perf:
    st.subheader(f"Best model: {report['best_model'].replace('_', ' ').title()}")
    mdf = pd.DataFrame(report["metrics"]).T.round(3)
    st.dataframe(mdf, width="stretch")
    a, b = st.columns(2)
    a.image(str(ROOT / "models" / "actual_vs_predicted.png"), caption="Actual vs predicted (test set)")
    b.image(str(ROOT / "models" / "feature_importance.png"), caption="Permutation feature importance")
    st.subheader("Per-crop test performance")
    st.dataframe(pd.DataFrame(report["per_crop"]).round(3), width="stretch", hide_index=True)

# ---------------- data ----------------
with tab_data:
    st.subheader("Training dataset (synthetic)")
    st.write(f"{len(data):,} fields × {data.shape[1]} columns. "
             "Generated with agronomy-based relationships plus noise and injected missing values.")
    st.dataframe(data.head(200), width="stretch")
    st.bar_chart(data.groupby("crop")["yield_t_ha"].mean())
