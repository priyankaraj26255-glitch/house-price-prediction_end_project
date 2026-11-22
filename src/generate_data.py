"""Generate a realistic synthetic housing dataset (Ames/Kaggle-style features).

Synthetic data keeps the repo self-contained and reproducible. To use a real
dataset, drop a CSV with the same columns into data/raw/housing.csv.
"""
import numpy as np
import pandas as pd
from pathlib import Path

NEIGHBORHOODS = {  # name: (price multiplier, sampling weight)
    "Downtown": (1.35, 0.12), "Riverside": (1.20, 0.15), "Greenfield": (1.05, 0.25),
    "Oakwood": (1.00, 0.22), "Eastgate": (0.85, 0.16), "Millbrook": (0.72, 0.10),
}


def generate(n: int = 3000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    names = list(NEIGHBORHOODS)
    probs = np.array([NEIGHBORHOODS[k][1] for k in names])
    hood = rng.choice(names, n, p=probs / probs.sum())
    mult = np.array([NEIGHBORHOODS[h][0] for h in hood])

    sqft = np.clip(rng.lognormal(7.3, 0.35, n), 500, 6000).round()
    bedrooms = np.clip((sqft / 650 + rng.normal(0, 0.7, n)).round(), 1, 7).astype(int)
    bathrooms = np.clip((bedrooms * 0.6 + rng.normal(0.6, 0.5, n)).round(), 1, 6).astype(int)
    year_built = rng.integers(1925, 2024, n)
    lot_size = np.clip(sqft * rng.uniform(1.2, 4.5, n), 1000, 30000).round()
    garage = rng.choice([0, 1, 2, 3], n, p=[0.1, 0.3, 0.45, 0.15])
    quality = np.clip(rng.normal(6, 1.5, n).round(), 1, 10).astype(int)
    has_pool = rng.random(n) < 0.12
    dist_center = np.clip(rng.gamma(2.2, 3.0, n) / mult, 0.3, 40).round(1)

    age = 2024 - year_built
    price = (
        45_000 + 105 * sqft + 9_000 * bathrooms + 2_500 * bedrooms
        + 14_000 * garage + 18_000 * (quality - 5) + 0.9 * lot_size
        - 650 * age + 22_000 * has_pool - 2_200 * dist_center
    ) * mult
    price *= np.exp(rng.normal(0, 0.07, n))  # multiplicative noise
    df = pd.DataFrame(dict(
        sqft=sqft, bedrooms=bedrooms, bathrooms=bathrooms, year_built=year_built,
        lot_size=lot_size, garage_spaces=garage, overall_quality=quality,
        has_pool=has_pool.astype(int), dist_to_center_km=dist_center,
        neighborhood=hood, price=price.round(-2),
    ))
    # inject realistic missingness for the cleaning step
    for col, frac in [("lot_size", 0.04), ("year_built", 0.02), ("dist_to_center_km", 0.03)]:
        df.loc[rng.random(n) < frac, col] = np.nan
    return df


if __name__ == "__main__":
    out = Path(__file__).resolve().parents[1] / "data" / "raw" / "housing.csv"
    df = generate()
    df.to_csv(out, index=False)
    print(f"Wrote {len(df)} rows -> {out}")
