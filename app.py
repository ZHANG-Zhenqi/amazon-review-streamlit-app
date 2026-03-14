import streamlit as st
from transformers import pipeline

st.title("Amazon Review Sentiment Analyzer")

@st.cache_resource
def load_model():
    sentiment_pipeline = pipeline(
        "sentiment-analysis",
        model="ZhenqiZhang/amazon-review-sentiment"
    )
    return sentiment_pipeline

classifier = load_model()

review = st.text_area("Enter an Amazon review")

if st.button("Analyze Sentiment"):
    if review.strip() == "":
        st.warning("Please enter a review")
    else:
        result = classifier(review)[0]

        label_map = {
            "LABEL_0": "Negative",
            "LABEL_1": "Positive"
        }

        sentiment = label_map.get(result["label"], result["label"])

        st.write("### Result")
        st.write("Sentiment:", sentiment)
        st.write("Confidence:", round(result["score"],3))
