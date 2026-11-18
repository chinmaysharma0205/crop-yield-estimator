"""Interactive demo: Crop Yield Estimator trained on REAL FAO data.

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


def predict(model, df: pd.DataFrame) -> np.ndarray:
    return model.predict(add_features(df))


model, data, report = load_model(), load_data(), load_metrics()
best = report["best_model"]
test_m = report["test_metrics"][best]
TEST_START = int(report["split"]["test"].split("-")[0])
YEAR_MIN, YEAR_MAX = int(data.year.min()), int(data.year.max())

st.title("🌾 Crop Yield Estimator")
st.caption("Machine learning for agriculture, trained on real FAOSTAT crop yields with World Bank rainfall, "
           "pesticide use and temperature records (101 countries, 1990–2013).")

# ---------------- sidebar ----------------
with st.sidebar:
    st.header("Inputs")
    countries = sorted(data.country.unique())
    country = st.selectbox("Country", countries, index=countries.index("India") if "India" in countries else 0)
    cdata = data[data.country == country]
    crops = sorted(cdata.crop.unique())
    crop = st.selectbox("Crop", crops, index=crops.index("Wheat") if "Wheat" in crops else 0)
    year = st.slider("Year", YEAR_MIN, YEAR_MAX, YEAR_MAX)

    hist = cdata[cdata.crop == crop].sort_values("year")
    ref = hist[hist.year == year]
    ref = ref.iloc[0] if len(ref) else hist.iloc[-1]
    k = f"{country}|{crop}|{year}"  # new key => widgets reset to that record's real values

    st.caption("Pre-filled with the recorded values for that country and year. Change them to explore.")
    rain = st.number_input("Average annual rainfall (mm)", 0.0, 4000.0, float(ref.rainfall_mm), 10.0, key="r" + k)
    pest = st.number_input("Pesticides used in country (tonnes)", 0.0, 400000.0, float(ref.pesticides_tonnes),
                           100.0, key="p" + k)
    temp = st.slider("Average temperature (°C)", 0.0, 32.0, float(round(ref.avg_temp_c, 1)), 0.1, key="t" + k)

row = pd.DataFrame([{"country": country, "crop": crop, "year": year, "rainfall_mm": rain,
                     "pesticides_tonnes": pest, "avg_temp_c": temp}])

tab_pred, tab_whatif, tab_perf, tab_data = st.tabs(
    ["🔮 Predict yield", "🧪 What-if analysis", "📊 Model performance", "🗂 Dataset & source"])

# ---------------- predict ----------------
with tab_pred:
    pred = float(predict(model, row)[0])
    actual_row = hist[hist.year == year]
    c1, c2, c3 = st.columns(3)
    c1.metric("Estimated yield", f"{pred:.2f} t/ha")
    if len(actual_row):
        act = float(actual_row.yield_t_ha.iloc[0])
        c2.metric("Recorded (FAO) yield", f"{act:.2f} t/ha", f"model off by {pred - act:+.2f}", delta_color="off")
    else:
        c2.metric("Recorded (FAO) yield", "not available")
    c3.metric("Typical error (MAE)", f"±{test_m['MAE']:.2f} t/ha", "on unseen test years", delta_color="off")

    fit = predict(model, hist)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.axvspan(TEST_START - 0.5, YEAR_MAX + 0.5, color="orange", alpha=0.12,
               label=f"test years {TEST_START}-{YEAR_MAX} (never seen in training)")
    ax.plot(hist.year, hist.yield_t_ha, "o-", label="Recorded yield (FAO)")
    ax.plot(hist.year, fit, "s--", label="Model estimate")
    ax.scatter([year], [pred], color="red", s=90, zorder=5, label="Your scenario")
    ax.set(xlabel="Year", ylabel="Yield (t/ha)", title=f"{crop} in {country}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    st.pyplot(fig)
    st.caption("Blue = real recorded yields. Dashed = what the model estimates from that year's rainfall, "
               "pesticides and temperature. In the shaded years the model had not seen the data.")

# ---------------- what-if ----------------
with tab_whatif:
    st.subheader("How does the estimate respond to one input? (others held at the sidebar values)")
    options = {
        "Average temperature (°C)": ("avg_temp_c", np.linspace(max(temp - 8, 0), min(temp + 8, 32), 40)),
        "Pesticides used (tonnes)": ("pesticides_tonnes", np.linspace(0, max(pest * 2, 100), 40)),
        "Average annual rainfall (mm)": ("rainfall_mm", np.linspace(max(rain * 0.4, 20), rain * 1.8 + 50, 40)),
        "Year": ("year", np.arange(YEAR_MIN, YEAR_MAX + 1)),
    }
    label = st.selectbox("Vary", list(options))
    col, grid = options[label]
    sweep = pd.concat([row] * len(grid), ignore_index=True)
    sweep[col] = grid
    ys = predict(model, sweep)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(grid, ys)
    ax.axvline(row[col].iloc[0], color="gray", ls="--", label="current value")
    ax.set(xlabel=label, ylabel="Estimated yield (t/ha)", title=f"{crop} in {country}")
    ax.legend()
    fig.tight_layout()
    st.pyplot(fig)
    st.warning(
        "These are **associations learned from country-level records, not proven cause and effect**. "
        "Rainfall in this dataset is a fixed long-term average per country, and pesticides is the whole "
        "country's total, so the model mainly learns differences between countries and crops. "
        "Curves can be flat or non-smooth. That is a limitation of the data, not a bug."
    )

# ---------------- performance ----------------
with tab_perf:
    sp = report["split"]
    st.subheader(f"Best model: {best.replace('_', ' ').title()}")
    st.write(f"Trained on **{sp['deployed_model_fit_on']}** and scored on **{sp['test']}**, "
             "years the model never saw. This mimics real use: estimating future seasons.")
    tm = pd.DataFrame(report["test_metrics"]).T.round(3)
    tm.index = [i.replace("_", " ") for i in tm.index]
    st.dataframe(tm, width="stretch")
    st.caption("R² closer to 1 is better; MAE/RMSE are in t/ha; WAPE is total error as % of total yield. "
               "'baseline' simply predicts each country-crop's historical average. The model should beat it.")
    u = report["unseen_country_check"]
    st.info(f"**Harder test, countries never seen in training:** R² {u['R2']:.2f}, MAE {u['MAE']:.2f} t/ha. "
            "Accuracy drops because yield depends strongly on local farming conditions, "
            "which this dataset only captures through the country name.")
    a, b = st.columns(2)
    a.image(str(ROOT / "models" / "actual_vs_predicted.png"), caption="Actual vs predicted (test years)")
    b.image(str(ROOT / "models" / "feature_importance.png"), caption="Permutation feature importance")
    st.subheader("Per-crop test performance")
    st.dataframe(pd.DataFrame(report["per_crop"]).round(3), width="stretch", hide_index=True)

# ---------------- dataset ----------------
with tab_data:
    st.subheader("Real dataset: FAOSTAT crop yields + World Bank climate")
    st.markdown(
        f"""
**{len(data):,} records**, one per country, crop and year, covering **{data.country.nunique()} countries**,
**{data.crop.nunique()} crops** and **{YEAR_MIN}-{YEAR_MAX}**.

| Column | Meaning | Source |
|---|---|---|
| `yield_t_ha` | Recorded crop yield, tonnes per hectare (**target**) | FAOSTAT |
| `rainfall_mm` | Average annual rainfall of the country (one long-term value per country) | World Bank |
| `pesticides_tonnes` | Total pesticides used in the country that year | FAOSTAT |
| `avg_temp_c` | Average temperature of the country that year | World Bank |

Compiled as the "Crop Yield Prediction Dataset" on Kaggle (user *patelris*).

**Cleaning done:** 2,310 exact duplicate rows removed; temperature readings that were repeated many times for
the same country-crop-year were averaged into one record (otherwise near-identical copies would leak between
training and test sets).

**Limits:** national-level data (no soil, satellite NDVI or field-level inputs) and years only up to 2013.
        """
    )
    st.dataframe(data.sample(200, random_state=1).sort_values(["country", "crop", "year"]), width="stretch")
    st.bar_chart(data.groupby("crop")["yield_t_ha"].mean())
