"""Exploratory data analysis: writes figures to reports/figures and a summary JSON."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "reports" / "figures"
sns.set_theme(style="whitegrid", palette="viridis")


def run():
    FIG.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(ROOT / "data" / "raw" / "housing.csv")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    sns.histplot(df.price, bins=50, kde=True, ax=ax[0]); ax[0].set_title("Price distribution (right-skewed)")
    sns.histplot(df.price.apply(lambda x: __import__("math").log(x)), bins=50, kde=True, ax=ax[1])
    ax[1].set_title("log(Price) is near-normal")
    fig.tight_layout(); fig.savefig(FIG / "price_distribution.png", dpi=130); plt.close(fig)

    num = df.select_dtypes("number")
    fig, ax = plt.subplots(figsize=(8, 6.5))
    sns.heatmap(num.corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax, annot_kws={"size": 8})
    ax.set_title("Correlation heatmap"); fig.tight_layout()
    fig.savefig(FIG / "correlation_heatmap.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    order = df.groupby("neighborhood").price.median().sort_values().index
    sns.boxplot(data=df, x="neighborhood", y="price", order=order, ax=ax[0])
    ax[0].set_title("Price by neighborhood"); ax[0].tick_params(axis="x", rotation=30)
    sns.scatterplot(data=df.sample(900, random_state=1), x="sqft", y="price",
                    hue="overall_quality", palette="viridis", ax=ax[1], s=18)
    ax[1].set_title("Price vs. living area (colored by quality)")
    fig.tight_layout(); fig.savefig(FIG / "price_drivers.png", dpi=130); plt.close(fig)

    summary = {
        "rows": len(df), "columns": df.shape[1],
        "missing_pct": (df.isna().mean() * 100).round(2)[lambda s: s > 0].to_dict(),
        "price_skew": round(float(df.price.skew()), 2),
        "top_corr_with_price": num.corr().price.drop("price").abs().sort_values(ascending=False).head(4).round(2).to_dict(),
    }
    (ROOT / "reports" / "eda_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    run()
