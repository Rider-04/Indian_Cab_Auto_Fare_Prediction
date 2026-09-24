"""
STEP 3: call the deployed API from anywhere (needs:  pip install requests).

    python fetch_client.py https://your-service-name.onrender.com
or set the API_URL environment variable, or edit DEFAULT_URL below.
"""
import os
import sys
import time

import requests

DEFAULT_URL = "https://YOUR-SERVICE-NAME.onrender.com"
BASE_URL = (sys.argv[1] if len(sys.argv) > 1 else os.environ.get("API_URL", DEFAULT_URL)).rstrip("/")
TIMEOUT = 90     # Render's free tier sleeps when idle; the first request can take up to a minute


def call(method, path, **kwargs):
    r = requests.request(method, BASE_URL + path, timeout=TIMEOUT, **kwargs)
    try:
        data = r.json()
    except ValueError:
        data = {"raw": r.text}
    if not r.ok:
        raise RuntimeError(f"{method} {path} -> {r.status_code}: {data}")
    return data


def wake_up(tries=6):
    """Ping /health until the server answers (handles the cold start)."""
    for i in range(1, tries + 1):
        try:
            return call("GET", "/health")
        except (requests.RequestException, RuntimeError) as e:
            print(f"  waiting for server ({i}/{tries}): {type(e).__name__}")
            time.sleep(10)
    raise SystemExit("server did not wake up")


def predict(ride):
    return call("POST", "/predict", json=ride)


def predict_batch(rides):
    return call("POST", "/predict/batch", json={"rides": rides})


def delhi_fare(vehicle_type, distance_km):
    return call("POST", "/predict/delhi-rate-card", json={"vehicle_type": vehicle_type, "distance_km": distance_km})


if __name__ == "__main__":
    print("API:", BASE_URL)
    print("health:", wake_up())

    meta = call("GET", "/meta")
    print("\nallowed weather:", meta["weather_values"])
    print("allowed sources:", meta["source_values"])
    print("hold-out metrics:", meta["holdout_metrics"])

    print("\n--- one ride ---")
    res = predict({"distance_km": 32, "weather": "raining", "source_dataset": "bookings", "date": "2025-01-18"})
    print(f"predicted fare {res['predicted_fare']}  "
          f"80% range {res['fare_range_80pct']['low']} to {res['fare_range_80pct']['high']}  "
          f"P(high fare) {res['high_fare_probability']}")

    print("\n--- batch ---")
    rides = [
        {"distance_km": 5,  "weather": "sunny",   "source_dataset": "ncr_events",     "weekday": "Tuesday"},
        {"distance_km": 25, "weather": "raining", "source_dataset": "ncr_events",     "weekday": "Sunday"},
        {"distance_km": 12, "weather": "not_applicable", "source_dataset": "indore_ola", "weekday": "Saturday"},
        {"distance_km": None, "weather": "unknown", "source_dataset": "bookings",    "weekday": "Monday"},
    ]
    for r, p in zip(rides, predict_batch(rides)["predictions"]):
        print(f"{r['source_dataset']:12s} {r['weather']:15s} {str(r['distance_km']):>5s} km {r['weekday']:9s} -> "
              f"{p['predicted_fare']:8.1f}  (80% range {p['fare_range_80pct']['low']:.0f}-{p['fare_range_80pct']['high']:.0f})")

    print("\n--- Delhi rate card ---")
    print(delhi_fare("taxi_ac", 10))

    print("\n--- a bad request is rejected clearly ---")
    try:
        predict({"distance_km": -3, "weather": "snowing", "source_dataset": "bookings", "weekday": "Monday"})
    except RuntimeError as e:
        print(e)
