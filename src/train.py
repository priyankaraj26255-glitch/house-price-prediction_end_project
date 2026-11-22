"""Train, cross-validate and compare models; save the best pipeline + metrics."""
import json
from pathlib import Path
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

from features import RAW_FEATURES, build_preprocessor

ROOT = Path(__file__).resolve().parents[1]
SEED = 42


def make(model):
    # log-transform the target to tame skew; predictions are inverse-transformed automatically
    return TransformedTargetRegressor(
        regressor=Pipeline([("prep", build_preprocessor()), ("model", model)]),
        func=np.log1p, inverse_func=np.expm1)


def evaluate(model, X, y):
    p = model.predict(X)
    return {"r2": round(r2_score(y, p), 4), "mae": round(mean_absolute_error(y, p), 0),
            "rmse": round(float(np.sqrt(mean_squared_error(y, p))), 0),
            "mape_pct": round(float(np.mean(np.abs((y - p) / y)) * 100), 2)}


def main():
    df = pd.read_csv(ROOT / "data" / "raw" / "housing.csv")
    X, y = df[RAW_FEATURES], df["price"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=SEED)

    candidates = {
        "Ridge (baseline)": make(Ridge(alpha=1.0)),
        "Random Forest": make(RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=SEED)),
        "Gradient Boosting": make(GradientBoostingRegressor(random_state=SEED)),
    }
    results = {}
    for name, m in candidates.items():
        cv = cross_val_score(m, X_tr, y_tr, cv=5, scoring="r2")
        m.fit(X_tr, y_tr)
        results[name] = {"cv_r2_mean": round(cv.mean(), 4), "cv_r2_std": round(cv.std(), 4),
                         **evaluate(m, X_te, y_te)}
        print(name, results[name])

    # tune the best tree model
    gs = GridSearchCV(
        make(GradientBoostingRegressor(random_state=SEED)),
        {"regressor__model__n_estimators": [200, 400],
         "regressor__model__learning_rate": [0.05, 0.1],
         "regressor__model__max_depth": [3, 4]},
        cv=3, scoring="r2", n_jobs=-1)
    gs.fit(X_tr, y_tr)
    best = gs.best_estimator_
    results["Gradient Boosting (tuned)"] = {
        "cv_r2_mean": round(gs.best_score_, 4), "cv_r2_std": 0.0, **evaluate(best, X_te, y_te)}
    print("Tuned:", gs.best_params_, results["Gradient Boosting (tuned)"])

    (ROOT / "models").mkdir(exist_ok=True)
    joblib.dump(best, ROOT / "models" / "house_price_model.joblib")

    # permutation importance on raw features (model-agnostic, works through the pipeline)
    from sklearn.inspection import permutation_importance
    pi = permutation_importance(best, X_te, y_te, n_repeats=5, random_state=SEED, scoring="r2")
    imp = pd.Series(pi.importances_mean, index=RAW_FEATURES).sort_values()

    fig_dir = ROOT / "reports" / "figures"; fig_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4.5)); imp.plot.barh(ax=ax, color="#6c5ce7")
    ax.set_title("Permutation importance (drop in R²)"); fig.tight_layout()
    fig.savefig(fig_dir / "feature_importance.png", dpi=130); plt.close(fig)

    pred = best.predict(X_te)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    ax[0].scatter(y_te, pred, s=8, alpha=.5, color="#6c5ce7")
    lim = [y_te.min(), y_te.max()]; ax[0].plot(lim, lim, "r--", lw=1)
    ax[0].set_xlabel("Actual"); ax[0].set_ylabel("Predicted"); ax[0].set_title("Predicted vs actual")
    ax[1].hist((y_te - pred) / 1000, bins=50, color="#00b894"); ax[1].set_title("Residuals ($K)")
    fig.tight_layout(); fig.savefig(fig_dir / "model_performance.png", dpi=130); plt.close(fig)

    out = {"models": results, "best_params": {k.split("__")[-1]: v for k, v in gs.best_params_.items()},
           "feature_importance": imp.sort_values(ascending=False).round(4).to_dict(),
           "train_rows": len(X_tr), "test_rows": len(X_te)}
    (ROOT / "reports" / "metrics.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
