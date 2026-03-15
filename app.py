import streamlit as st
from transformers import pipeline

# -------------------------------
# Load sentiment analysis model
# -------------------------------

sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="ZhenqiZhang/amazon-review-sentiment"
)

# -------------------------------
# NEW: Load category classification model
# -------------------------------

category_pipeline = pipeline(
    "zero-shot-classification",
    model="typeform/distilbert-base-uncased-mnli"
)

categories = [
    "electronics",
    "clothing",
    "home",
    "beauty",
    "sports",
    "books"
]

# -------------------------------
# Streamlit UI
# -------------------------------

st.title("Amazon Review Sentiment Analyzer")

review = st.text_area("Enter an Amazon review")

if st.button("Analyze Sentiment"):

if review.strip() != "":

    # Sentiment prediction
    result = sentiment_pipeline(review)[0]

    label_map = {
        "LABEL_0": "Negative",
        "LABEL_1": "Positive"
    }

    sentiment = label_map.get(result["label"], result["label"])
    sentiment_confidence = result["score"]

    # Category prediction
    category_result = category_pipeline(
        review,
        candidate_labels=categories
    )

    category = category_result["labels"][0]
    category_confidence = category_result["scores"][0]

        # -------------------------------
        # Display results
        # -------------------------------

        st.subheader("Result")

        st.write("Sentiment:", sentiment)
        st.write("Sentiment Confidence:", round(sentiment_confidence, 3))

        st.write("Category:", category)
        st.write("Category Confidence:", round(category_confidence, 3))

    else:
        st.warning("Please enter a review.")
