import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
from app import app  # noqa: E402

GOOD = dict(sqft=1800, bedrooms=3, bathrooms=2, year_built=2005, lot_size=5500,
            garage_spaces=2, overall_quality=7, has_pool=0, dist_to_center_km=6,
            neighborhood="Oakwood")


def test_health():
    assert app.test_client().get("/health").json["status"] == "ok"


def test_predict_ok():
    r = app.test_client().post("/predict", json=GOOD)
    assert r.status_code == 200 and r.json["predicted_price"] > 50_000


def test_quality_monotonic():
    c = app.test_client()
    lo = c.post("/predict", json={**GOOD, "overall_quality": 3}).json["predicted_price"]
    hi = c.post("/predict", json={**GOOD, "overall_quality": 9}).json["predicted_price"]
    assert hi > lo


def test_validation():
    c = app.test_client()
    assert c.post("/predict", json={**GOOD, "sqft": -5}).status_code == 400
    assert c.post("/predict", json={**GOOD, "neighborhood": "Mars"}).status_code == 400
