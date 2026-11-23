from __future__ import annotations

import argparse

from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression, NaiveBayes
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import CountVectorizer, IDF, StopWordsRemover, Tokenizer
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import DoubleType


LABELS = {"negative": 0.0, "neutral": 1.0, "positive": 2.0}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Spark MLlib sentiment classifiers.")
    parser.add_argument("--input", required=True, help="Normalized CSV containing review_text and sentiment.")
    parser.add_argument("--output", required=True, help="Output directory for predictions and product summary.")
    args = parser.parse_args()

    spark = SparkSession.builder.appName("EcommerceReviewSentiment").getOrCreate()
    reviews = (
        spark.read.option("header", True).option("inferSchema", True).csv(args.input)
        .select("review_text", "sentiment", *[c for c in ("rating", "product_id") if c in spark.read.option("header", True).csv(args.input).columns])
        .where(F.col("review_text").isNotNull() & (F.length(F.trim("review_text")) > 0))
        .where(F.col("sentiment").isin(*LABELS.keys()))
        .withColumn("label", F.create_map([F.lit(value) for pair in LABELS.items() for value in pair]).getItem(F.col("sentiment")).cast(DoubleType()))
    )

    train, test = reviews.randomSplit([0.8, 0.2], seed=42)
    tokenizer = Tokenizer(inputCol="review_text", outputCol="tokens")
    stopwords = StopWordsRemover(inputCol="tokens", outputCol="filtered_tokens")
    vectorizer = CountVectorizer(inputCol="filtered_tokens", outputCol="raw_features", vocabSize=50000, minDF=2.0)
    tfidf = IDF(inputCol="raw_features", outputCol="features")
    evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")

    results = []
    for name, classifier in (
        ("Naive Bayes", NaiveBayes(labelCol="label", featuresCol="features", modelType="multinomial")),
        ("Logistic Regression", LogisticRegression(labelCol="label", featuresCol="features", maxIter=100)),
    ):
        pipeline = Pipeline(stages=[tokenizer, stopwords, vectorizer, tfidf, classifier])
        model = pipeline.fit(train)
        predictions = model.transform(test)
        accuracy = MulticlassClassificationEvaluator(labelCol="label", metricName="accuracy").evaluate(predictions)
        macro_f1 = evaluator.evaluate(predictions)
        results.append((name, float(accuracy), float(macro_f1)))
        predictions.select("review_text", "sentiment", "prediction", "probability").write.mode("overwrite").parquet(f"{args.output}/{name.lower().replace(' ', '_')}_predictions")

    spark.createDataFrame(results, ["model", "accuracy", "macro_f1"]).write.mode("overwrite").option("header", True).csv(f"{args.output}/model_metrics")
    if "product_id" in reviews.columns:
        reviews.groupBy("product_id").agg(
            F.count("*").alias("review_count"),
            F.avg("rating").alias("average_rating") if "rating" in reviews.columns else F.lit(None).alias("average_rating"),
            F.avg((F.col("sentiment") == "positive").cast("double")).alias("positive_share"),
            F.avg((F.col("sentiment") == "negative").cast("double")).alias("negative_share"),
        ).write.mode("overwrite").option("header", True).csv(f"{args.output}/product_summary")
    spark.stop()


if __name__ == "__main__":
    main()
