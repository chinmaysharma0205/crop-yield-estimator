"""Crop recommendation: which crop suits this soil and climate? (Kaggle "Crop Recommendation Dataset")

Data: data/reco/Crop_recommendation.csv, 2,200 rows, 22 crops x exactly 100 rows each.
Inputs: soil nitrogen N, phosphorus P, potassium K (kg/ha), temperature (C), humidity (%), soil pH, rainfall (mm).

Honest note about the data: every crop has exactly 100 rows and the classes are almost perfectly separable,
which suggests the table was assembled or augmented rather than surveyed from fields. Treat it as a public
benchmark dataset, not as field measurements. We also test how fragile the model is when inputs are noisy.

Usage:  python src/crop_reco.py
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "reco" / "Crop_recommendation.csv"
FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]


def main(seed: int = 42) -> None:
    df = pd.read_csv(RAW)
    X, y = df[FEATURES], df["label"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)

    candidates = {
        "logistic_regression": Pipeline([("s", StandardScaler()), ("m", LogisticRegression(max_iter=3000))]),
        "knn": Pipeline([("s", StandardScaler()), ("m", KNeighborsClassifier(n_neighbors=5))]),
        "naive_bayes": GaussianNB(),
        "random_forest": RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1),
    }
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    res, fitted = {}, {}
    for n, m in candidates.items():
        cvs = cross_val_score(m, Xtr, ytr, cv=cv, scoring="accuracy")
        m.fit(Xtr, ytr)
        p = m.predict(Xte)
        res[n] = {"CV_accuracy": float(cvs.mean()), "test_accuracy": float(accuracy_score(yte, p)),
                  "test_macro_F1": float(f1_score(yte, p, average="macro"))}
        fitted[n] = m
        print(f"{n:20s} CV acc {cvs.mean():.3f} | test acc {res[n]['test_accuracy']:.3f} | F1 {res[n]['test_macro_F1']:.3f}")
    best_name = max(res, key=lambda k: res[k]["CV_accuracy"])
    best = fitted[best_name]
    print("Best:", best_name)

    pred = best.predict(Xte)
    classes = list(best.classes_)
    cm = confusion_matrix(yte, pred, labels=classes)
    pairs = [(classes[i], classes[j], int(cm[i, j])) for i in range(len(classes)) for j in range(len(classes))
             if i != j and cm[i, j] > 0]
    pairs.sort(key=lambda t: -t[2])
    per_class_f1 = dict(zip(classes, f1_score(yte, pred, average=None, labels=classes).round(3)))

    # How fragile is it when readings are noisy? (add noise = fraction of each feature's std)
    rng = np.random.default_rng(seed)
    robustness = {}
    for frac in (0.0, 0.1, 0.25, 0.5):
        noisy = Xte + rng.normal(0, 1, Xte.shape) * Xte.std().to_numpy() * frac
        robustness[str(frac)] = float(accuracy_score(yte, best.predict(noisy)))
    print("Accuracy with noisy inputs (noise as fraction of std):", robustness)

    imp = permutation_importance(best, Xte, yte, n_repeats=5, random_state=seed, n_jobs=1)
    imp_s = pd.Series(imp.importances_mean, index=FEATURES).sort_values(ascending=False).round(4)
    print(imp_s.to_dict())

    out = ROOT / "models"
    out.mkdir(exist_ok=True)
    joblib.dump(best, out / "reco_model.joblib", compress=("xz", 3))
    (out / "reco_metrics.json").write_text(json.dumps({
        "best_model": best_name, "models": res, "per_class_f1": {k: float(v) for k, v in per_class_f1.items()},
        "most_confused_pairs": [{"true": a, "predicted": b, "count": c} for a, b, c in pairs[:8]],
        "accuracy_with_noisy_inputs": robustness, "feature_importance": imp_s.to_dict(),
        "data_notes": {"rows": int(len(df)), "crops": len(classes),
                       "rows_per_crop": sorted(set(int(v) for v in y.value_counts().values))},
    }, indent=2))
    print(f"Saved models/reco_model.joblib ({(out / 'reco_model.joblib').stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
