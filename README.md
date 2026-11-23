# E-Commerce Product Review Sentiment Analysis

A Streamlit dashboard and Apache Spark MLlib pipeline for exploring e-commerce review sentiment and product feedback. The project follows the report's workflow: clean review data, map ratings to negative/neutral/positive labels, tokenize text, remove stopwords, build CountVectorizer and TF-IDF features, compare classifiers, and summarize product performance.

## Run the dashboard

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Upload a CSV with review text plus ratings or sentiment labels. The dashboard also offers a small illustrative sample when no data is uploaded.

## Download the Kaggle dataset

The project uses [Consumer Reviews of Amazon Products on Kaggle](https://www.kaggle.com/datasets/datafiniti/consumer-reviews-of-amazon-products), which includes 34,000+ product reviews. The listing identifies the license as **CC BY-NC-SA 4.0**. Review the dataset's terms before redistribution or commercial use.

Kaggle requires an account token for downloads. Configure the Kaggle API credentials using Kaggle's official instructions, then run:

```bash
pip install -r requirements-data.txt
python scripts/download_kaggle_data.py
```

The script downloads the Kaggle source to `data/raw/`, normalizes the review text, rating, and product columns into `data/amazon_reviews.csv`, and creates a three-class sentiment label from ratings (1–2 negative, 3 neutral, 4–5 positive). The raw source is not committed; download it directly from Kaggle with your own account. After download, start the dashboard and upload `data/amazon_reviews.csv`.

## Apache Spark / MLlib pipeline

Install the optional Spark dependencies and run the distributed pipeline:

```bash
pip install -r requirements-spark.txt
spark-submit src/spark_pipeline.py --input data/amazon_reviews.csv --output outputs/sentiment_predictions
```

It uses Spark SQL for cleaning and product-level summaries, then MLlib Tokenizer, StopWordsRemover, CountVectorizer, IDF, NaiveBayes, and LogisticRegression. Outputs include model metrics and product summaries.

## CSV format

Required text column: `review_text`, `text`, `review`, or `content`.

Optional columns: `rating` / `overall` / `score` / `stars`, `sentiment` / `label`, and `product_id` / `asin` / `product_name` / `name`.

## Benchmark disclosure

The source report describes a **synthetic benchmark** comparing Naive Bayes and Logistic Regression. Those results are illustrative and are not measured on the real Amazon review corpus. The dashboard does not present synthetic results as corpus performance. Spark metrics are calculated on the dataset downloaded and run by the user.

## Scope

The original report proposes Apache Spark and MLlib for distributed processing. This repository includes both a lightweight local Streamlit prediction workflow and a Spark MLlib pipeline. Neither claims that the dashboard itself processes millions of records.
