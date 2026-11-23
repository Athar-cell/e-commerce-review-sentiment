# E-Commerce Product Review Sentiment Analysis

A Streamlit dashboard for exploring review sentiment and product-level feedback. The project follows the report's Spark/MLlib workflow terminology: clean review data, map ratings to negative/neutral/positive labels, convert review text to TF-IDF features, compare classifiers, and summarize product performance.

The dashboard runs locally with a lightweight scikit-learn equivalent for text prediction. It does not require a Spark cluster.

## Run locally

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Dashboard features

- Upload a CSV of reviews and filter by product, sentiment, or review text.
- View sentiment and rating distributions, product summaries, and export filtered rows.
- Train a TF-IDF + Logistic Regression model from supplied labels and predict a review.
- See holdout accuracy and macro F1 computed from the uploaded data.
- Inspect the report's Naive Bayes vs. Logistic Regression synthetic benchmark context.

The built-in sample rows are only a demonstration dataset. They are not Amazon data.

## CSV format

Required review text column: `review_text`, `text`, `review`, or `content`.

Optional columns:
- `rating` (or `overall`, `score`, `stars`): numeric 1–5 rating. If sentiment is absent, labels are inferred as 1–2 negative, 3 neutral, and 4–5 positive.
- `sentiment` (or `label`, `sentiment_label`): positive, neutral, or negative.
- `product_id` (or `asin`, `product`, `product_name`, `name`): product identifier or name.

Example:

```csv
product_id,rating,review_text
A100,5,"Great quality and fast delivery"
A100,2,"Stopped working after two days"
B200,3,"It is okay for the price"
```

## Benchmark and evaluation disclosure

The project report describes a **synthetic benchmark** comparing Naive Bayes and Logistic Regression. Those benchmark values are illustrative and must not be presented as results from the real Amazon corpus. The dashboard labels the benchmark accordingly. The live holdout metrics are calculated from the uploaded data and can vary with dataset size, class balance, and split composition. Tiny datasets are not suitable for claims about model quality.

## Project scope

The report's full-scale design uses Apache Spark and MLlib for distributed processing. This dashboard provides an accessible local equivalent for demonstration and interactive review analysis; it does not claim to process millions of records or run Spark.
