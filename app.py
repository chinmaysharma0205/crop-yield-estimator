"""Crop yield & crop recommendation demo (Streamlit).

Run:  streamlit run app.py

Modules
  1. India yield (Kaggle, real)      state-level yields 1997-2020
  2. Crop recommendation (Kaggle)    N-P-K, pH, climate -> which crop
  3. Global yield (FAO, real)        101 countries 1990-2013
  4. Field simulator (SYNTHETIC)     soil N/P/K, irrigation, NDVI (generated data, for learning only)
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
import india_yield  # noqa: E402  (needed to load the saved India model)
from simulator import add_sim_features  # noqa: E402
from train import add_features as fao_features  # noqa: E402

st.set_page_config(page_title="Crop Yield Lab", page_icon="🌾", layout="wide")


# ---------- cached loaders (keyed on file modification time, so re-uploaded files are picked up) ----------
@st.cache_resource
def _joblib(path: str, mtime: float):
    return joblib.load(path)


@st.cache_data
def _csv(path: str, mtime: float):
    return pd.read_csv(path)


@st.cache_data
def _json(path: str, mtime: float):
    return json.loads(Path(path).read_text())


def load_model(rel: str):
    p = ROOT / rel
    return _joblib(str(p), p.stat().st_mtime)


def load_csv(rel: str) -> pd.DataFrame:
    p = ROOT / rel
    return _csv(str(p), p.stat().st_mtime)


def load_json(rel: str) -> dict:
    p = ROOT / rel
    return _json(str(p), p.stat().st_mtime)


def line_chart(x, y, xlabel, ylabel, title, current=None):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x, y, marker="o" if len(x) < 10 else None)
    if current is not None:
        ax.axvline(current, color="gray", ls="--", label="current value")
        ax.legend()
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    fig.tight_layout()
    st.pyplot(fig)


# =====================================================================================================
# 1. INDIA YIELD (real, Kaggle)
# =====================================================================================================
def india_module():
    data = load_csv("data/india/india_clean.csv")
    rep = load_json("models/india_metrics.json")
    model = load_model("models/india_model.joblib")
    best = rep["best_model"]
    tm = rep["test_metrics"]
    base_m, mod_m = tm["baseline_state_crop_season_mean"], tm[best]
    fit_end = int(rep["split"]["deployed_model_fit_on"].split("-")[1])
    test_start = int(rep["split"]["test"].split("-")[0])
    ymin, ymax = int(data.year.min()), int(data.year.max())

    st.caption("Real public data (Kaggle, Indian crop statistics): 30 states and union territories, "
               f"{data['crop'].nunique()} crops, {ymin}–{ymax}. Rajasthan is not included in this dataset.")

    states = sorted(data.state.unique())
    c1, c2, c3, c4 = st.columns(4)
    state = c1.selectbox("State", states, index=states.index("Uttar Pradesh") if "Uttar Pradesh" in states else 0,
                         key="in_state")
    sd = data[data.state == state]
    crops = sorted(sd["crop"].unique())
    crop = c2.selectbox("Crop", crops, index=crops.index("Rice") if "Rice" in crops else 0, key="in_crop")
    cd = sd[sd["crop"] == crop]
    season = c3.selectbox("Season", sorted(cd.season.unique()), key="in_season")
    hist = cd[cd.season == season].sort_values("year")
    year = c4.slider("Year", ymin, ymax, min(2019, ymax), key="in_year")

    rec = hist[hist.year == year]
    sy = sd[sd.year == year]
    src = rec.iloc[0] if len(rec) else (sy.iloc[0] if len(sy) else hist.iloc[-1])
    k = f"{state}|{crop}|{season}|{year}"  # new key -> inputs reset to that record's real values
    area0 = float(rec.area_ha.iloc[0]) if len(rec) else float(hist.area_ha.median())

    d1, d2, d3, d4 = st.columns(4)
    area = d1.number_input("Cropped area (ha)", 10.0, max(float(data.area_ha.max()), area0), area0, 100.0,
                           key="in_area" + k)
    rain = d2.number_input("Annual rainfall (mm)", 0.0, max(8000.0, float(src.rainfall_mm)),
                           float(src.rainfall_mm), 10.0, key="in_rain" + k)
    fert = d3.number_input("Fertiliser (kg per ha)", 0.0, max(float(data.fert_kg_ha.max()), float(src.fert_kg_ha)),
                           float(src.fert_kg_ha), 5.0, key="in_fert" + k)
    pest = d4.number_input("Pesticide (kg per ha)", 0.0, max(float(data.pest_kg_ha.max()), float(src.pest_kg_ha)),
                           float(src.pest_kg_ha), 0.01, key="in_pest" + k)

    def predict(df):
        return model.predict(india_yield.add_features(df))

    row = pd.DataFrame([{"state": state, "crop": crop, "season": season, "year": year, "area_ha": area,
                         "rainfall_mm": rain, "fert_kg_ha": fert, "pest_kg_ha": pest}])
    pred = float(predict(row)[0])
    hist_avg = float(hist.loc[hist.year <= fit_end, "yield"].mean()) if (hist.year <= fit_end).any() else np.nan

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Estimated yield", f"{pred:.2f} t/ha")
    if len(rec):
        act = float(rec["yield"].iloc[0])
        m2.metric("Recorded yield", f"{act:.2f} t/ha", f"model off by {pred - act:+.2f}", delta_color="off")
    else:
        m2.metric("Recorded yield", "not available")
    m3.metric(f"Historical average ({ymin}–{fit_end})", "n/a" if np.isnan(hist_avg) else f"{hist_avg:.2f} t/ha")
    m4.metric("Typical error (MAE)", f"±{mod_m['MAE']:.2f} t/ha", "on unseen test years", delta_color="off")
    if crop in ("Cotton(lint)", "Jute", "Mesta"):
        st.caption("Note: cotton, jute and mesta are recorded in bales per hectare in the source data.")

    fit = predict(hist)
    fig, ax = plt.subplots(figsize=(9, 3.8))
    ax.axvspan(test_start - 0.5, ymax + 0.5, color="orange", alpha=0.12,
               label=f"test years {test_start}-{ymax} (never seen in training)")
    ax.plot(hist.year, hist["yield"], "o-", label="Recorded yield")
    ax.plot(hist.year, fit, "s--", label="Model estimate")
    ax.scatter([year], [pred], color="red", s=90, zorder=5, label="Your scenario")
    ax.set(xlabel="Year", ylabel="Yield (t/ha)", title=f"{crop} ({season}) in {state}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    st.pyplot(fig)

    with st.expander("🧪 What-if analysis"):
        opts = {
            "Annual rainfall (mm)": ("rainfall_mm", np.linspace(max(rain * 0.4, 50), rain * 1.8 + 50, 40)),
            "Fertiliser (kg/ha)": ("fert_kg_ha", np.linspace(0, max(fert * 2, 100), 40)),
            "Pesticide (kg/ha)": ("pest_kg_ha", np.linspace(0, max(pest * 2, 0.5), 40)),
            "Year": ("year", np.arange(ymin, ymax + 1)),
        }
        label = st.selectbox("Vary", list(opts), key="in_vary")
        col, grid = opts[label]
        sweep = pd.concat([row] * len(grid), ignore_index=True)
        sweep[col] = grid
        line_chart(grid, predict(sweep), label, "Estimated yield (t/ha)", f"{crop} in {state}", row[col].iloc[0])
        st.warning("Rainfall, fertiliser and pesticide in this dataset are **state-level yearly figures**, identical "
                   "for every crop in a state, and they explain very little of the year-to-year change. Expect flat "
                   "curves. These are associations, not proven cause and effect.")

    with st.expander("📊 How good is this model? (read before trusting it)"):
        gain = mod_m["R2"] - base_m["R2"]
        st.write(f"Trained on **{rep['split']['deployed_model_fit_on']}**, scored on **{rep['split']['test']}**, "
                 "years the model never saw.")
        table = pd.DataFrame(tm).T.round(3)
        table.index = [i.replace("_", " ") for i in table.index]
        st.dataframe(table, width="stretch")
        st.info(
            f"**Honest result:** the best model ({best.replace('_', ' ')}) scores R² {mod_m['R2']:.2f} against "
            f"{base_m['R2']:.2f} for a naive baseline that just predicts the average past yield of that state, "
            f"crop and season (a gain of only {gain:+.3f}). With only state-level rainfall, fertiliser and "
            "pesticide, machine learning adds very little beyond the historical average here. Most of the "
            "accuracy comes from knowing the state, crop and season."
        )
        u = rep["unseen_state_check"]
        st.write(f"For states never seen in training: R² {u['R2']:.2f}, MAE {u['MAE']:.2f} t/ha.")
        st.subheader("Per-crop test performance (10 most common crops)")
        st.dataframe(pd.DataFrame(rep["per_crop"]).round(3), width="stretch", hide_index=True)
        c = rep["cleaning"]
        st.markdown(
            f"**Cleaning:** {c['raw_rows']:,} raw rows → {c['clean_rows']:,}. Removed {c['removed_coconut_nuts']} "
            f"coconut rows (recorded in nuts, not tonnes), {c['removed_zero_yield']} zero-yield rows, "
            f"{c['removed_area_under_10ha']} plots under 10 ha, and {c['removed_extreme_outliers']} extreme "
            "outliers. `Production` was not used as an input because yield is production divided by area."
        )


# =====================================================================================================
# 2. CROP RECOMMENDATION (Kaggle benchmark)
# =====================================================================================================
def reco_module():
    df = load_csv("data/reco/Crop_recommendation.csv")
    rep = load_json("models/reco_metrics.json")
    model = load_model("models/reco_model.joblib")
    feats = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]

    st.caption("Which crop suits this soil and climate? Trained on the public Kaggle *Crop Recommendation "
               "Dataset* (2,200 rows, 22 crops, 100 rows each).")
    st.warning("This dataset is a **public benchmark, not field measurements**: every crop has exactly 100 rows "
               "and the classes are almost perfectly separable, which suggests it was assembled or augmented. "
               "The near-perfect score below shows the model learned that table, not that it would be this "
               "accurate on real farms.")

    a, b, c = st.columns(3)
    N = a.slider("Nitrogen N (kg/ha)", int(df.N.min()), int(df.N.max()), int(df.N.median()), key="rc_n")
    P = b.slider("Phosphorus P (kg/ha)", int(df.P.min()), int(df.P.max()), int(df.P.median()), key="rc_p")
    K = c.slider("Potassium K (kg/ha)", int(df.K.min()), int(df.K.max()), int(df.K.median()), key="rc_k")
    t = a.slider("Temperature (°C)", float(round(df.temperature.min(), 1)), float(round(df.temperature.max(), 1)),
                 float(round(df.temperature.median(), 1)), 0.1, key="rc_t")
    h = b.slider("Humidity (%)", float(round(df.humidity.min(), 1)), float(round(df.humidity.max(), 1)),
                 float(round(df.humidity.median(), 1)), 0.1, key="rc_h")
    ph = c.slider("Soil pH", float(round(df.ph.min(), 1)), float(round(df.ph.max(), 1)),
                  float(round(df.ph.median(), 1)), 0.1, key="rc_ph")
    rain = a.slider("Rainfall (mm)", float(round(df.rainfall.min(), 0)), float(round(df.rainfall.max(), 0)),
                    float(round(df.rainfall.median(), 0)), 1.0, key="rc_rain")

    x = pd.DataFrame([[N, P, K, t, h, ph, rain]], columns=feats)
    proba = model.predict_proba(x)[0]
    order = np.argsort(proba)[::-1][:5]
    classes = model.classes_
    top = classes[order[0]]

    with b:
        st.success(f"Best match: **{top.title()}** ({proba[order[0]]:.0%} confidence)")
        if proba[order[0]] < 0.6:
            st.caption("Low confidence: several crops fit these conditions about equally well.")
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.barh([classes[i].title() for i in order][::-1], [proba[i] for i in order][::-1])
    ax.set(xlabel="Probability", title="Top 5 suggested crops", xlim=(0, 1))
    fig.tight_layout()
    st.pyplot(fig)

    prof = df[df.label == top][feats].median()
    comp = pd.DataFrame({"Your input": x.iloc[0].round(1), f"Typical for {top}": prof.round(1)})
    st.write(f"How your inputs compare with the typical conditions for **{top}** in this dataset:")
    st.dataframe(comp.T, width="stretch")

    with st.expander("📊 Model performance and honest limits"):
        mt = pd.DataFrame(rep["models"]).T.round(3)
        mt.index = [i.replace("_", " ") for i in mt.index]
        st.dataframe(mt, width="stretch")
        st.write(f"Best model: **{rep['best_model'].replace('_', ' ')}**. A simple Gaussian Naive Bayes model being "
                 "this accurate is another sign the data was generated from per-crop distributions.")
        nz = rep["accuracy_with_noisy_inputs"]
        st.write("**Sensitivity to noisy readings** (random noise added to the test inputs, as a fraction of each "
                 "input's spread):")
        st.dataframe(pd.DataFrame({"noise level": [f"{float(k):.0%}" for k in nz],
                                   "accuracy": [round(v, 3) for v in nz.values()]}), hide_index=True)
        st.write("With realistic measurement error the recommendation gets much less reliable.")
        st.write("Most important inputs:", ", ".join(f"{k} ({v:.2f})" for k, v in
                                                     list(rep["feature_importance"].items())[:4]))
        st.write("Most confused crop pairs on the test set:",
                 ", ".join(f"{p['true']}→{p['predicted']} ({p['count']})" for p in rep["most_confused_pairs"][:4]) or "none")


# =====================================================================================================
# 3. GLOBAL YIELD (FAO, real)
# =====================================================================================================
def fao_module():
    model = load_model("models/yield_model.joblib")
    data = load_csv("data/crop_yield.csv")
    report = load_json("models/metrics.json")
    best = report["best_model"]
    test_m = report["test_metrics"][best]
    test_start = int(report["split"]["test"].split("-")[0])
    ymin, ymax = int(data.year.min()), int(data.year.max())

    st.caption("Real FAOSTAT crop yields with World Bank rainfall and temperature and FAOSTAT pesticide use "
               f"({data.country.nunique()} countries, {ymin}–{ymax}). National-level data.")

    countries = sorted(data.country.unique())
    c1, c2, c3 = st.columns(3)
    country = c1.selectbox("Country", countries, index=countries.index("India") if "India" in countries else 0,
                           key="fa_country")
    crops = sorted(data[data.country == country].crop.unique())
    crop = c2.selectbox("Crop", crops, index=crops.index("Wheat") if "Wheat" in crops else 0, key="fa_crop")
    year = c3.slider("Year", ymin, ymax, ymax, key="fa_year")
    hist = data[(data.country == country) & (data.crop == crop)].sort_values("year")
    ref = hist[hist.year == year]
    ref = ref.iloc[0] if len(ref) else hist.iloc[-1]
    k = f"{country}|{crop}|{year}"
    d1, d2, d3 = st.columns(3)
    rain = d1.number_input("Average annual rainfall (mm)", 0.0, 4000.0, float(ref.rainfall_mm), 10.0, key="fa_r" + k)
    pest = d2.number_input("Pesticides used in country (tonnes)", 0.0, 400000.0, float(ref.pesticides_tonnes),
                           100.0, key="fa_p" + k)
    temp = d3.slider("Average temperature (°C)", 0.0, 32.0, float(round(ref.avg_temp_c, 1)), 0.1, key="fa_t" + k)

    row = pd.DataFrame([{"country": country, "crop": crop, "year": year, "rainfall_mm": rain,
                         "pesticides_tonnes": pest, "avg_temp_c": temp}])

    def predict(df):
        return model.predict(fao_features(df))

    t_pred, t_whatif, t_perf, t_data = st.tabs(["🔮 Predict", "🧪 What-if", "📊 Model performance", "🗂 Dataset"])
    with t_pred:
        pred = float(predict(row)[0])
        actual_row = hist[hist.year == year]
        m1, m2, m3 = st.columns(3)
        m1.metric("Estimated yield", f"{pred:.2f} t/ha")
        if len(actual_row):
            act = float(actual_row.yield_t_ha.iloc[0])
            m2.metric("Recorded (FAO) yield", f"{act:.2f} t/ha", f"model off by {pred - act:+.2f}", delta_color="off")
        else:
            m2.metric("Recorded (FAO) yield", "not available")
        m3.metric("Typical error (MAE)", f"±{test_m['MAE']:.2f} t/ha", "on unseen test years", delta_color="off")
        fit = predict(hist)
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.axvspan(test_start - 0.5, ymax + 0.5, color="orange", alpha=0.12,
                   label=f"test years {test_start}-{ymax} (never seen in training)")
        ax.plot(hist.year, hist.yield_t_ha, "o-", label="Recorded yield (FAO)")
        ax.plot(hist.year, fit, "s--", label="Model estimate")
        ax.scatter([year], [pred], color="red", s=90, zorder=5, label="Your scenario")
        ax.set(xlabel="Year", ylabel="Yield (t/ha)", title=f"{crop} in {country}")
        ax.legend(fontsize=8)
        fig.tight_layout()
        st.pyplot(fig)
    with t_whatif:
        opts = {
            "Average temperature (°C)": ("avg_temp_c", np.linspace(max(temp - 8, 0), min(temp + 8, 32), 40)),
            "Pesticides used (tonnes)": ("pesticides_tonnes", np.linspace(0, max(pest * 2, 100), 40)),
            "Average annual rainfall (mm)": ("rainfall_mm", np.linspace(max(rain * 0.4, 20), rain * 1.8 + 50, 40)),
            "Year": ("year", np.arange(ymin, ymax + 1)),
        }
        label = st.selectbox("Vary", list(opts), key="fa_vary")
        col, grid = opts[label]
        sweep = pd.concat([row] * len(grid), ignore_index=True)
        sweep[col] = grid
        line_chart(grid, predict(sweep), label, "Estimated yield (t/ha)", f"{crop} in {country}", row[col].iloc[0])
        st.warning("Associations learned from country-level records, not proven cause and effect. Rainfall is a "
                   "fixed long-term average per country and pesticides is the whole country's total, so these "
                   "curves can be flat or non-smooth.")
    with t_perf:
        sp = report["split"]
        st.write(f"Trained on **{sp['deployed_model_fit_on']}**, scored on **{sp['test']}** (never seen).")
        tm = pd.DataFrame(report["test_metrics"]).T.round(3)
        tm.index = [i.replace("_", " ") for i in tm.index]
        st.dataframe(tm, width="stretch")
        u = report["unseen_country_check"]
        st.info(f"Countries never seen in training: R² {u['R2']:.2f}, MAE {u['MAE']:.2f} t/ha. Accuracy drops "
                "because local farming conditions are only captured through the country name.")
        a, b = st.columns(2)
        a.image(str(ROOT / "models" / "actual_vs_predicted.png"), caption="Actual vs predicted (test years)")
        b.image(str(ROOT / "models" / "feature_importance.png"), caption="Permutation feature importance")
        st.dataframe(pd.DataFrame(report["per_crop"]).round(3), width="stretch", hide_index=True)
    with t_data:
        st.markdown(
            f"**{len(data):,} records** (one per country, crop, year). Sources: FAOSTAT (yield, pesticides) and the "
            "World Bank (rainfall, temperature), compiled on Kaggle as *Crop Yield Prediction Dataset* (patelris). "
            "Cleaning: 2,310 duplicate rows removed and repeated temperature readings averaged into one record per "
            "country-crop-year (otherwise near-copies would leak between training and test). "
            "Limits: national-level, up to 2013."
        )
        st.dataframe(data.sample(200, random_state=1).sort_values(["country", "crop", "year"]), width="stretch")


# =====================================================================================================
# 4. FIELD SIMULATOR (SYNTHETIC)
# =====================================================================================================
def sim_module():
    st.warning(
        "**This tab is a simulator trained on SYNTHETIC data, not real farm measurements.** Real free datasets do "
        "not combine soil nutrients, irrigation and satellite NDVI with yields, so the data here were generated "
        "from agronomy-style formulas (yield rises with nitrogen with diminishing returns, each crop has an ideal "
        "temperature and water amount, soil pH has an optimum) plus random noise. Use it to explore *how* "
        "field-level inputs could matter in precision agriculture."
    )
    try:
        sim_model = load_model("models/field_sim_model.joblib")
        sim_report = load_json("models/field_sim_metrics.json")
    except FileNotFoundError:
        st.error("Simulator files not found. Run `python src/simulator.py` to create them.")
        return
    sim_best = sim_report["best_model"]
    sim_mae = sim_report["metrics"][sim_best]["MAE"]

    colA, colB, colC = st.columns(3)
    with colA:
        st.markdown("**Field**")
        s_crop = st.selectbox("Crop", ["wheat", "rice", "maize", "cotton", "mustard"], key="s_crop")
        s_soil = st.selectbox("Soil type", ["loam", "sandy", "clay", "silt"], key="s_soil")
        s_area = st.number_input("Field area (ha)", 0.5, 100.0, 5.0, 0.5, key="s_area")
        st.markdown("**Weather (season)**")
        s_temp = st.slider("Average temperature (°C)", 10.0, 38.0, 22.0, 0.5, key="s_temp")
        s_rain = st.slider("Rainfall (mm)", 40, 1800, 200, 10, key="s_rain",
                           help="Ideal total water (rain + irrigation) differs by crop, roughly 370 mm for wheat "
                                "and 950 mm for rice. Too little or too much lowers yield.")
        s_hum = st.slider("Humidity (%)", 20, 95, 58, key="s_hum")
        s_solar = st.slider("Solar radiation (MJ/m²/day)", 9.0, 28.0, 19.0, 0.5, key="s_solar")
        s_heat = st.slider("Heat-stress days", 0, 40, 2, key="s_heat")
    with colB:
        st.markdown("**Soil test**")
        s_n = st.slider("Soil nitrogen, N (kg/ha)", 10, 120, 55, key="s_n")
        s_p = st.slider("Soil phosphorus, P (kg/ha)", 4, 55, 22, key="s_p")
        s_k = st.slider("Soil potassium, K (kg/ha)", 60, 400, 180, key="s_k")
        s_ph = st.slider("Soil pH", 4.5, 9.2, 7.0, 0.1, key="s_ph")
        s_oc = st.slider("Organic carbon (%)", 0.1, 1.8, 0.65, 0.05, key="s_oc")
    with colC:
        st.markdown("**Management & remote sensing**")
        s_fert = st.slider("Fertiliser nitrogen applied (kg/ha)", 0, 250, 120, key="s_fert")
        s_irr = st.slider("Number of irrigations", 0, 6, 3, key="s_irr")
        s_seed = st.slider("Seed rate (relative to recommended)", 0.6, 1.4, 1.0, 0.05, key="s_seed")
        s_ndvi = st.slider("Peak NDVI (satellite)", 0.2, 0.92, 0.75, 0.01, key="s_ndvi")

    s_row = pd.DataFrame([{
        "crop": s_crop, "soil_type": s_soil, "avg_temp_c": s_temp, "rainfall_mm": s_rain,
        "humidity_pct": s_hum, "solar_rad_mj": s_solar, "heat_stress_days": s_heat,
        "soil_n_kg_ha": s_n, "soil_p_kg_ha": s_p, "soil_k_kg_ha": s_k, "soil_ph": s_ph,
        "organic_carbon_pct": s_oc, "fertilizer_n_kg_ha": s_fert, "irrigations": s_irr,
        "seed_rate_rel": s_seed, "ndvi_peak": s_ndvi,
    }])
    s_pred = float(sim_model.predict(add_sim_features(s_row))[0])
    m1, m2, m3 = st.columns(3)
    m1.metric("Simulated yield", f"{s_pred:.2f} t/ha")
    m2.metric("Simulated production", f"{s_pred * s_area:.1f} t", f"over {s_area:g} ha", delta_color="off")
    m3.metric("Model error vs simulator", f"±{sim_mae:.2f} t/ha", "on synthetic test data", delta_color="off")

    st.subheader("What-if: vary one input (others held at the values above)")
    sim_options = {
        "Fertiliser nitrogen (kg/ha)": ("fertilizer_n_kg_ha", np.linspace(0, 250, 40)),
        "Number of irrigations": ("irrigations", np.arange(0, 7)),
        "Soil nitrogen (kg/ha)": ("soil_n_kg_ha", np.linspace(10, 120, 40)),
        "Soil phosphorus (kg/ha)": ("soil_p_kg_ha", np.linspace(4, 55, 40)),
        "Soil potassium (kg/ha)": ("soil_k_kg_ha", np.linspace(60, 400, 40)),
        "Soil pH": ("soil_ph", np.linspace(4.5, 9.2, 40)),
        "Rainfall (mm)": ("rainfall_mm", np.linspace(40, 1800, 40)),
        "Average temperature (°C)": ("avg_temp_c", np.linspace(10, 38, 40)),
        "Peak NDVI": ("ndvi_peak", np.linspace(0.2, 0.92, 40)),
    }
    s_label = st.selectbox("Vary", list(sim_options), key="s_vary")
    s_col, s_grid = sim_options[s_label]
    s_sweep = pd.concat([s_row] * len(s_grid), ignore_index=True)
    s_sweep[s_col] = s_grid
    line_chart(s_grid, sim_model.predict(add_sim_features(s_sweep)), s_label, "Simulated yield (t/ha)",
               f"{s_crop.title()} (synthetic simulator)", s_row[s_col].iloc[0])
    st.caption(f"Simulator model: {sim_best.replace('_', ' ')}, R² {sim_report['metrics'][sim_best]['R2']:.2f} on "
               "synthetic test data. This only shows how well it learned the simulator's formulas. It does "
               "**not** measure accuracy on real farms.")


# =====================================================================================================
# page
# =====================================================================================================
st.title("🌾 Crop Yield Lab")
st.caption("Machine learning for agriculture. Tabs 1–3 use **real public data**; tab 4 is a **synthetic "
           "simulator** (clearly labelled) for exploring field-level inputs.")

tab_india, tab_reco, tab_fao, tab_sim = st.tabs([
    "🇮🇳 India yield (Kaggle, real)", "🌾 Crop recommendation (N-P-K)", "🌍 Global yield (FAO, real)",
    "🌱 Field simulator (synthetic)"])
with tab_india:
    india_module()
with tab_reco:
    reco_module()
with tab_fao:
    fao_module()
with tab_sim:
    sim_module()
