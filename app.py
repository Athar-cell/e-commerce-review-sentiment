from __future__ import annotations

import io
import re

import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

st.set_page_config(page_title="Review Sentiment Dashboard", page_icon="🛍️", layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 2rem; max-width: 1400px}
[data-testid="stMetric"] {background:#f5f7fb;padding:16px;border-radius:12px}
</style>
""", unsafe_allow_html=True)

st.title("🛍️ E-Commerce Review Intelligence")
st.caption("Explore customer feedback, product performance, and review sentiment.")
st.info("This dashboard analyzes the dataset you provide. The benchmark section describes synthetic report results; those figures are not measured on the Amazon corpus.", icon="ℹ️")

SAMPLE = pd.DataFrame([
    {"product_id":"P-100","product_name":"Wireless Headphones","rating":5,"review_text":"Excellent sound and comfortable all day","sentiment":"positive"},
    {"product_id":"P-100","product_name":"Wireless Headphones","rating":4,"review_text":"Good battery and clear audio","sentiment":"positive"},
    {"product_id":"P-100","product_name":"Wireless Headphones","rating":2,"review_text":"Stopped working after a week","sentiment":"negative"},
    {"product_id":"P-200","product_name":"Travel Mug","rating":5,"review_text":"Keeps coffee hot and never leaks","sentiment":"positive"},
    {"product_id":"P-200","product_name":"Travel Mug","rating":3,"review_text":"Fine but smaller than expected","sentiment":"neutral"},
    {"product_id":"P-200","product_name":"Travel Mug","rating":1,"review_text":"Poor quality and lid leaks everywhere","sentiment":"negative"},
    {"product_id":"P-300","product_name":"Desk Lamp","rating":4,"review_text":"Bright light and easy to adjust","sentiment":"positive"},
    {"product_id":"P-300","product_name":"Desk Lamp","rating":3,"review_text":"It works as described","sentiment":"neutral"},
    {"product_id":"P-300","product_name":"Desk Lamp","rating":2,"review_text":"Cheap materials and flickers often","sentiment":"negative"},
])

def normalize_label(value):
    text = str(value).strip().lower()
    if text in {"positive", "pos", "1"}: return "positive"
    if text in {"neutral", "neu", "0"}: return "neutral"
    if text in {"negative", "neg", "-1"}: return "negative"
    return None

def prepare_data(frame):
    frame = frame.copy()
    aliases = {c.lower().strip(): c for c in frame.columns}
    def find(*names):
        return next((aliases[n] for n in names if n in aliases), None)
    text_col = find("review_text", "text", "review", "content", "reviewtext")
    rating_col = find("rating", "overall", "score", "stars")
    product_col = find("product_id", "asin", "product", "product_name", "name")
    label_col = find("sentiment", "label", "sentiment_label")
    if text_col is None:
        raise ValueError("Add a review_text, text, review, or content column.")
    out = pd.DataFrame({"review_text": frame[text_col].fillna("").astype(str)})
    out["rating"] = pd.to_numeric(frame[rating_col], errors="coerce") if rating_col else pd.NA
    out["product"] = frame[product_col].fillna("Unknown product").astype(str) if product_col else "Unknown product"
    if label_col:
        out["sentiment"] = frame[label_col].map(normalize_label)
    elif rating_col:
        out["sentiment"] = out["rating"].map(lambda x: "positive" if pd.notna(x) and x >= 4 else ("negative" if pd.notna(x) and x <= 2 else ("neutral" if pd.notna(x) else None)))
    else:
        out["sentiment"] = None
    out = out[out.review_text.str.strip().ne("")].drop_duplicates(subset=["review_text","product"]).reset_index(drop=True)
    return out

with st.sidebar:
    st.header("Dataset")
    upload = st.file_uploader("Upload reviews (CSV)", type=["csv"])
    st.caption("Expected columns: review_text, rating, product_id, sentiment. Review text is required; labels can be derived from ratings (1–2 negative, 3 neutral, 4–5 positive).")
    st.divider()
    st.caption("No file uploaded? The sample dataset is for demonstration only.")
try:
    raw = pd.read_csv(upload) if upload else SAMPLE
    data = prepare_data(raw)
except Exception as exc:
    st.error(f"Could not read this dataset: {exc}")
    st.stop()
if data.empty:
    st.warning("No usable review rows were found.")
    st.stop()

with st.sidebar:
    products = sorted(data["product"].unique())
    selected_products = st.multiselect("Products", products, default=products)
    sentiments = ["positive","neutral","negative"]
    selected_sentiments = st.multiselect("Sentiment", sentiments, default=sentiments)
    search = st.text_input("Search review text")

filtered = data[data["product"].isin(selected_products) & data.sentiment.isin(selected_sentiments)]
if search.strip():
    filtered = filtered[filtered.review_text.str.contains(search, case=False, na=False)]

total = len(filtered)
c1,c2,c3,c4 = st.columns(4)
c1.metric("Reviews", f"{total:,}")
c2.metric("Average rating", f"{filtered.rating.mean():.2f} / 5" if filtered.rating.notna().any() else "N/A")
c3.metric("Positive share", f"{(filtered.sentiment.eq('positive').mean()*100):.1f}%" if total else "0%")
c4.metric("Products", f"{filtered.product.nunique():,}")

left,right = st.columns(2)
with left:
    st.subheader("Sentiment distribution")
    if total:
        counts=filtered.sentiment.value_counts().reindex(sentiments,fill_value=0).rename_axis("Sentiment").reset_index(name="Reviews")
        st.plotly_chart(px.pie(counts,names="Sentiment",values="Reviews",hole=.55,color="Sentiment",color_discrete_map={"positive":"#19a974","neutral":"#f4b942","negative":"#e05d5d"}),use_container_width=True)
    else: st.info("No rows match the current filters.")
with right:
    st.subheader("Rating distribution")
    rated=filtered.dropna(subset=["rating"])
    if not rated.empty:
        ratings=rated.groupby("rating").size().reset_index(name="Reviews")
        st.plotly_chart(px.bar(ratings,x="rating",y="Reviews",text="Reviews",color_discrete_sequence=["#5267df"]),use_container_width=True)
    else: st.info("Upload ratings to see this chart.")

st.subheader("Product-level summary")
if total:
    summary=filtered.groupby("product").agg(reviews=("review_text","size"),average_rating=("rating","mean"),positive=("sentiment",lambda s:(s=="positive").mean()*100),neutral=("sentiment",lambda s:(s=="neutral").mean()*100),negative=("sentiment",lambda s:(s=="negative").mean()*100)).reset_index()
    summary=summary.rename(columns={"product":"Product","reviews":"Reviews","average_rating":"Avg. rating","positive":"Positive %","neutral":"Neutral %","negative":"Negative %"})
    st.dataframe(summary.style.format({"Avg. rating":"{:.2f}","Positive %":"{:.1f}%","Neutral %":"{:.1f}%","Negative %":"{:.1f}%"}),use_container_width=True,hide_index=True)
    st.download_button("Download filtered reviews",filtered.to_csv(index=False).encode("utf-8"),"filtered_reviews.csv","text/csv")
else:
    st.info("No rows match the current filters.")

st.divider()
st.subheader("Review text prediction")
st.write("Train a local TF-IDF + Logistic Regression classifier from labeled rows in the selected dataset, then score a new review. Requires at least two sentiment classes.")
labeled=data.dropna(subset=["sentiment"])
if len(labeled)>=4 and labeled.sentiment.nunique()>=2:
    try:
        X_train,X_test,y_train,y_test=train_test_split(labeled.review_text,labeled.sentiment,test_size=.25,random_state=42,stratify=labeled.sentiment)
        pipeline=Pipeline([("tfidf",TfidfVectorizer(lowercase=True,stop_words="english",ngram_range=(1,2),max_features=20000)),("classifier",LogisticRegression(max_iter=1000,class_weight="balanced"))])
        pipeline.fit(X_train,y_train)
        a,b=st.columns(2)
        a.metric("Holdout accuracy",f"{accuracy_score(y_test,pipeline.predict(X_test)):.1%}")
        b.metric("Holdout macro F1",f"{f1_score(y_test,pipeline.predict(X_test),average='macro',zero_division=0):.1%}")
        st.caption("These metrics are computed on a small random holdout from the uploaded data and are for exploration; use a representative, sufficiently large dataset for meaningful evaluation.")
        review=st.text_area("Enter review text",placeholder="The product works well and feels great.")
        if st.button("Predict sentiment",type="primary",disabled=not review.strip()):
            prediction=pipeline.predict([review])[0]
            probabilities=pipeline.predict_proba([review])[0]
            st.success(f"Predicted sentiment: **{prediction.title()}**")
            st.write({label:f"{prob:.1%}" for label,prob in zip(pipeline.classes_,probabilities)})
    except Exception as exc:
        st.warning(f"Could not train the model on this dataset: {exc}")
else:
    st.info("Add at least four labeled reviews across two or more sentiment classes to enable predictions.")

with st.expander("Model comparison benchmark described in the project report"):
    st.markdown("""
    The report compares **Naive Bayes** and **Logistic Regression** using a synthetic benchmark. The comparison is illustrative: it does not represent measured performance on the real Amazon review corpus. Upload a labeled dataset and use the holdout results above for a dataset-specific check.
    
    Project pipeline terminology: text tokenization → stopword removal → CountVectorizer → TF-IDF → MLlib classifiers → evaluation and product-level analysis. This Streamlit app uses scikit-learn's equivalent TF-IDF and Logistic Regression pipeline for a quick local, runnable prediction workflow.
    """)
    st.caption("Benchmark values from the source report are not reproduced as measured corpus results.")
st.caption("Data stays in this Streamlit session and is not saved by this app.")
