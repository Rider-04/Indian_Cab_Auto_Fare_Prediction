"""
STEP 1: train the final models and save everything the API needs as pickle files.

Run once on your laptop:   python train_and_save.py
Creates:
    models/fare_model.pkl        point prediction of the fare
    models/fare_q10.pkl          10th percentile model  (lower end of the 80% range)
    models/fare_q90.pkl          90th percentile model  (upper end of the 80% range)
    models/high_fare_clf.pkl     probability that the fare is in the top 20%
    models/delhi_rate_card.pkl   exact rate card for the Delhi source
    models/metadata.pkl          allowed values, feature order, threshold, metrics, library versions
    requirements.txt             pinned to the library versions used HERE (so Render matches)
"""
import os
import pickle
import importlib.metadata as md

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, roc_auc_score
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42
CANDIDATES = [
    r"C:\Users\lenovo\.cache\kagglehub\datasets\parthrider04\synthetic-fare-dataset\versions\1\Synthetic_Fare_Dataset.csv",
    "Synthetic_Fare_Dataset.csv",
]
PATH = next((p for p in CANDIDATES if os.path.exists(p)), None)
if PATH is None:
    raise SystemExit("Synthetic_Fare_Dataset.csv not found. Put it next to this script or edit CANDIDATES.")

OUT = "models"
os.makedirs(OUT, exist_ok=True)

# ----------------------------------------------------------------------------------------------
# 1. Same cleaning as the notebook
# ----------------------------------------------------------------------------------------------
raw = pd.read_csv(PATH)
df = raw.drop_duplicates().copy()                                   # exact duplicates

RULE_SOURCES = ["delhi_notification_2023", "aru_dto_2019"]           # rule-based rate cards
rule_df = df[df["source_dataset"].isin(RULE_SOURCES)].copy()
df = df[~df["source_dataset"].isin(RULE_SOURCES)].copy()
df = df[df["vehicle_type"] != "parcel"].copy()                       # deliveries, not rides

df["distance_km"] = df["distance_km"].replace(0, np.nan)             # 0 = not recorded
df["is_weekend"] = df["weekday"].isin(["Saturday", "Sunday"]).astype(int)
df = df[["source_dataset", "weather", "is_weekend", "distance_km", "fare_amount"]].reset_index(drop=True)

NUM_COLS = ["distance_km", "is_weekend"]
CAT_COLS = ["weather", "source_dataset"]
FEATURES = NUM_COLS + CAT_COLS

# fixed category lists: the API must build inputs with exactly these
cat_types = {c: pd.CategoricalDtype(sorted(df[c].unique())) for c in CAT_COLS}
for c in CAT_COLS:
    df[c] = df[c].astype(cat_types[c])
print("training rows:", len(df))

# ----------------------------------------------------------------------------------------------
# 2. Models
# ----------------------------------------------------------------------------------------------
COMMON = dict(learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=50,
              categorical_features="from_dtype", random_state=RANDOM_STATE)

def point():   return HistGradientBoostingRegressor(max_iter=500, early_stopping=True, **COMMON)
def quant(a):  return HistGradientBoostingRegressor(loss="quantile", quantile=a, max_iter=300, **COMMON)
def clf():     return HistGradientBoostingClassifier(max_iter=300, **COMMON)

X, y = df[FEATURES], df["fare_amount"]
threshold = float(y.quantile(0.8))                                   # "high fare" = top 20%

# honest hold-out check first
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
p = point().fit(Xtr, ytr).predict(Xte)
lo = np.minimum(quant(0.1).fit(Xtr, ytr).predict(Xte), p)
hi = np.maximum(quant(0.9).fit(Xtr, ytr).predict(Xte), p)
prob = clf().fit(Xtr, (ytr > threshold).astype(int)).predict_proba(Xte)[:, 1]
metrics = {
    "test_r2": round(float(r2_score(yte, p)), 4),
    "test_mae": round(float(mean_absolute_error(yte, p)), 1),
    "interval_80pct_coverage": round(float(((yte.values >= lo) & (yte.values <= hi)).mean()), 3),
    "interval_80pct_mean_width": round(float((hi - lo).mean()), 0),
    "high_fare_roc_auc": round(float(roc_auc_score((yte > threshold).astype(int), prob)), 3),
}
print("hold-out metrics:", metrics)

# then refit on ALL rows for deployment
final = {
    "fare_model.pkl": point().fit(X, y),
    "fare_q10.pkl": quant(0.1).fit(X, y),
    "fare_q90.pkl": quant(0.9).fit(X, y),
    "high_fare_clf.pkl": clf().fit(X, (y > threshold).astype(int)),
}

# ----------------------------------------------------------------------------------------------
# 3. Delhi rate card: exact  fare = intercept + rate * km  per vehicle
#    (3 minimum-fare rows carry a filled-in distance of 15.33 and are excluded)
# ----------------------------------------------------------------------------------------------
dl = rule_df[(rule_df["source_dataset"] == "delhi_notification_2023") & (rule_df["distance_km"] != 15.33)]
rate_card = {}
for v, g in dl.groupby("vehicle_type"):
    m = LinearRegression().fit(g[["distance_km"]], g["fare_amount"])
    rate_card[v] = {"rate_per_km": round(float(m.coef_[0]), 4), "intercept": round(float(m.intercept_), 4)}
rate_card_obj = {"vehicles": rate_card,
                 "distance_range_km": [float(dl["distance_km"].min()), float(dl["distance_km"].max())]}
print("Delhi rate card:", rate_card)

# ----------------------------------------------------------------------------------------------
# 4. Save everything
# ----------------------------------------------------------------------------------------------
metadata = {
    "features": FEATURES,
    "cat_types": cat_types,
    "weather_values": list(cat_types["weather"].categories),
    "source_values": list(cat_types["source_dataset"].categories),
    "high_fare_threshold": round(threshold, 1),
    "metrics": metrics,
    "trained_rows": int(len(df)),
    "sklearn_version": sklearn.__version__,
    "pandas_version": pd.__version__,
    "numpy_version": np.__version__,
}
for name, obj in final.items():
    with open(os.path.join(OUT, name), "wb") as f:
        pickle.dump(obj, f)
with open(os.path.join(OUT, "delhi_rate_card.pkl"), "wb") as f:
    pickle.dump(rate_card_obj, f)
with open(os.path.join(OUT, "metadata.pkl"), "wb") as f:
    pickle.dump(metadata, f)

def ver(pkg):
    try:
        return md.version(pkg)
    except md.PackageNotFoundError:
        return None

lines = []
for pkg in ["Flask", "numpy", "pandas", "scikit-learn"]:
    v = ver(pkg)
    lines.append(f"{pkg}=={v}" if v else pkg)
lines.append("gunicorn")
with open("requirements.txt", "w") as f:
    f.write("\n".join(lines) + "\n")

print("\nSaved:", sorted(os.listdir(OUT)))
print("requirements.txt:\n" + "\n".join(lines))
