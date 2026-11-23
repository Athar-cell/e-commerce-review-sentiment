from __future__ import annotations

import subprocess
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
OUTPUT = ROOT / "data" / "amazon_reviews.csv"
DATASET = "datafiniti/consumer-reviews-of-amazon-products"


def pick_column(frame: pd.DataFrame, *candidates: str) -> str | None:
    lookup = {str(column).strip().lower(): column for column in frame.columns}
    return next((lookup[name] for name in candidates if name in lookup), None)


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["kaggle", "datasets", "download", "-d", DATASET, "-p", str(RAW_DIR)],
        check=True,
    )
    archives = sorted(RAW_DIR.glob("*.zip"))
    if not archives:
        raise FileNotFoundError(f"Kaggle download did not create a zip in {RAW_DIR}")
    with zipfile.ZipFile(archives[-1]) as archive:
        csv_names = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        if not csv_names:
            raise FileNotFoundError("The downloaded Kaggle archive contains no CSV file.")
        source_name = next((name for name in csv_names if "amazon" in name.lower()), csv_names[0])
        with archive.open(source_name) as source:
            raw = pd.read_csv(source, low_memory=False)

    text_col = pick_column(raw, "reviews.text", "review_text", "review", "text")
    rating_col = pick_column(raw, "reviews.rating", "rating", "overall", "score")
    product_col = pick_column(raw, "asins", "product_id", "product_name", "name")
    if text_col is None or rating_col is None:
        raise ValueError(f"Expected review text and rating columns. Found: {list(raw.columns)}")

    result = pd.DataFrame({
        "review_text": raw[text_col].fillna("").astype(str),
        "rating": pd.to_numeric(raw[rating_col], errors="coerce"),
        "product_id": raw[product_col].fillna("Unknown product").astype(str) if product_col else "Unknown product",
    })
    result = result[result.review_text.str.strip().ne("") & result.rating.between(1, 5)]
    result["sentiment"] = result.rating.map(
        lambda rating: "positive" if rating >= 4 else ("negative" if rating <= 2 else "neutral")
    )
    result = result.drop_duplicates(subset=["review_text", "product_id"])
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT, index=False)
    print(f"Saved {len(result):,} normalized reviews to {OUTPUT}")


if __name__ == "__main__":
    main()
