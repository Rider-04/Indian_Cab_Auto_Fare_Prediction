"""
STEP 2: the API (Flask). Render runs it with:   gunicorn app:app
Locally:                                          python app.py
"""
import datetime as dt
import os
import pickle

import numpy as np
import pandas as pd
import sklearn
from flask import Flask, jsonify, request

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")


def load(name):
    with open(os.path.join(MODEL_DIR, name), "rb") as f:
        return pickle.load(f)          # only ever load pickles you created yourself


# loaded ONCE when the server starts, not on every request
point_model = load("fare_model.pkl")
q10_model = load("fare_q10.pkl")
q90_model = load("fare_q90.pkl")
high_fare_clf = load("high_fare_clf.pkl")
rate_card = load("delhi_rate_card.pkl")
meta = load("metadata.pkl")

if meta["sklearn_version"] != sklearn.__version__:
    print(f"WARNING: models were trained with scikit-learn {meta['sklearn_version']} "
          f"but this server runs {sklearn.__version__}. Pin the version in requirements.txt.")

WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
MAX_BATCH = 1000

app = Flask(__name__)


class BadRequest(ValueError):
    pass


@app.errorhandler(BadRequest)
def handle_bad_request(e):
    return jsonify({"error": str(e)}), 400


# ------------------------------------------------------------------------------------------
# input validation
# ------------------------------------------------------------------------------------------
def parse_ride(d):
    if not isinstance(d, dict):
        raise BadRequest("each ride must be a JSON object")

    dist = d.get("distance_km")
    if dist is not None:
        if isinstance(dist, bool):
            raise BadRequest("distance_km must be a number")
        try:
            dist = float(dist)
        except (TypeError, ValueError):
            raise BadRequest("distance_km must be a number (or null if unknown)")
        if not np.isfinite(dist) or dist <= 0 or dist > 200:
            raise BadRequest("distance_km must be between 0 and 200 (send null if unknown)")

    weather = d.get("weather")
    if weather not in meta["weather_values"]:
        raise BadRequest(f"weather must be one of {meta['weather_values']}")

    source = d.get("source_dataset")
    if source not in meta["source_values"]:
        raise BadRequest(f"source_dataset must be one of {meta['source_values']}")

    if d.get("weekday") is not None:
        weekday = str(d["weekday"]).strip().title()
        if weekday not in WEEKDAYS:
            raise BadRequest(f"weekday must be one of {WEEKDAYS}")
    elif d.get("date") is not None:
        try:
            weekday = WEEKDAYS[dt.date.fromisoformat(str(d["date"])[:10]).weekday()]
        except ValueError:
            raise BadRequest("date must look like 2025-01-18")
    else:
        raise BadRequest("send either 'weekday' (e.g. 'Saturday') or 'date' (e.g. '2025-01-18')")

    return {"distance_km": dist, "weather": weather, "source_dataset": source, "weekday": weekday}


def to_frame(rides):
    """Build the model input exactly as it was built during training."""
    return pd.DataFrame({
        "distance_km": np.array([np.nan if r["distance_km"] is None else r["distance_km"] for r in rides], dtype=float),
        "is_weekend": np.array([int(r["weekday"] in ("Saturday", "Sunday")) for r in rides]),
        "weather": pd.Categorical([r["weather"] for r in rides], dtype=meta["cat_types"]["weather"]),
        "source_dataset": pd.Categorical([r["source_dataset"] for r in rides], dtype=meta["cat_types"]["source_dataset"]),
    })[meta["features"]]


def predict_rides(rides):
    X = to_frame(rides)
    p = np.clip(point_model.predict(X), 0, None)
    lo = np.minimum(np.clip(q10_model.predict(X), 0, None), p)
    hi = np.maximum(q90_model.predict(X), p)
    prob = high_fare_clf.predict_proba(X)[:, 1]
    out = []
    for r, pi, li, hi_, pr in zip(rides, p, lo, hi, prob):
        out.append({
            "input": r,
            "predicted_fare": round(float(pi), 2),
            "fare_range_80pct": {"low": round(float(li), 2), "high": round(float(hi_), 2)},
            "high_fare_probability": round(float(pr), 3),
        })
    return out


def json_body():
    body = request.get_json(silent=True)
    if body is None:
        raise BadRequest("send a JSON body with Content-Type: application/json")
    return body


# ------------------------------------------------------------------------------------------
# routes
# ------------------------------------------------------------------------------------------
@app.get("/")
def home():
    return jsonify({
        "service": "fare prediction API",
        "endpoints": {
            "GET /health": "is the server up",
            "GET /meta": "allowed input values, model metrics",
            "POST /predict": "one ride",
            "POST /predict/batch": "{'rides': [...]} up to 1000 rides",
            "POST /predict/delhi-rate-card": "exact Delhi rate card fare",
        },
    })


@app.get("/health")
def health():
    return jsonify({"status": "ok", "sklearn_version": sklearn.__version__,
                    "trained_with": meta["sklearn_version"]})


@app.get("/meta")
def get_meta():
    return jsonify({
        "weather_values": meta["weather_values"],
        "source_values": meta["source_values"],
        "weekday_values": WEEKDAYS,
        "high_fare_threshold": meta["high_fare_threshold"],
        "holdout_metrics": meta["metrics"],
        "delhi_vehicles": sorted(rate_card["vehicles"]),
        "notes": [
            "Fares in most of this dataset are largely random. Pooled hold-out R2 is about 0.27, "
            "so treat predicted_fare as a rough average and use fare_range_80pct for planning.",
            "Rain and sunny rides carry the real signal; other rides are close to a per-source average.",
        ],
    })


@app.post("/predict")
def predict():
    ride = parse_ride(json_body())
    return jsonify(predict_rides([ride])[0])


@app.post("/predict/batch")
def predict_batch():
    body = json_body()
    rides = body.get("rides") if isinstance(body, dict) else None
    if not isinstance(rides, list) or not rides:
        raise BadRequest("send {'rides': [ {...}, {...} ]}")
    if len(rides) > MAX_BATCH:
        raise BadRequest(f"at most {MAX_BATCH} rides per request")
    parsed = []
    for i, r in enumerate(rides):
        try:
            parsed.append(parse_ride(r))
        except BadRequest as e:
            raise BadRequest(f"ride #{i}: {e}")
    return jsonify({"count": len(parsed), "predictions": predict_rides(parsed)})


@app.post("/predict/delhi-rate-card")
def predict_delhi():
    body = json_body()
    vehicle = body.get("vehicle_type") if isinstance(body, dict) else None
    if vehicle not in rate_card["vehicles"]:
        raise BadRequest(f"vehicle_type must be one of {sorted(rate_card['vehicles'])}")
    try:
        km = float(body.get("distance_km"))
    except (TypeError, ValueError):
        raise BadRequest("distance_km must be a number")
    if not np.isfinite(km) or km <= 0 or km > 200:
        raise BadRequest("distance_km must be between 0 and 200")
    c = rate_card["vehicles"][vehicle]
    lo_km, hi_km = rate_card["distance_range_km"]
    return jsonify({
        "vehicle_type": vehicle,
        "distance_km": km,
        "fare": round(c["intercept"] + c["rate_per_km"] * km, 2),
        "rate_per_km": c["rate_per_km"],
        "extrapolated": bool(km < lo_km or km > hi_km),
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
