"""Flask deployment: web form + JSON API for house price prediction."""
import sys
from pathlib import Path
import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from features import NEIGHBORHOODS, RAW_FEATURES  # noqa: E402

app = Flask(__name__)
MODEL_PATH = ROOT / "models" / "house_price_model.joblib"
model = joblib.load(MODEL_PATH)

RULES = {  # field: (min, max)
    "sqft": (300, 10000), "bedrooms": (1, 10), "bathrooms": (1, 8), "year_built": (1850, 2025),
    "lot_size": (500, 100000), "garage_spaces": (0, 5), "overall_quality": (1, 10),
    "has_pool": (0, 1), "dist_to_center_km": (0, 100),
}


def parse(payload: dict) -> pd.DataFrame:
    row = {}
    for field, (lo, hi) in RULES.items():
        if field not in payload or payload[field] in ("", None):
            raise ValueError(f"Missing field: {field}")
        val = float(payload[field])
        if not lo <= val <= hi:
            raise ValueError(f"{field} must be between {lo} and {hi}")
        row[field] = val
    if payload.get("neighborhood") not in NEIGHBORHOODS:
        raise ValueError(f"neighborhood must be one of {NEIGHBORHOODS}")
    row["neighborhood"] = payload["neighborhood"]
    return pd.DataFrame([row])[RAW_FEATURES]


@app.get("/")
def home():
    return render_template("index.html", neighborhoods=NEIGHBORHOODS)


@app.get("/health")
def health():
    return jsonify(status="ok")


@app.post("/predict")
def predict():
    payload = request.get_json(silent=True) or request.form.to_dict()
    try:
        price = float(model.predict(parse(payload))[0])
    except ValueError as e:
        return jsonify(error=str(e)), 400
    return jsonify(predicted_price=round(price, -2), formatted=f"${price:,.0f}",
                   range_low=round(price * 0.93, -2), range_high=round(price * 1.07, -2))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
