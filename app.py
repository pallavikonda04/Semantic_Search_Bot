import streamlit as st
from transformers import pipeline

# Page configuration
st.set_page_config(
    page_title="Sentiment Analysis Bot",
    page_icon="😊"
)

# Title
st.title("😊 Sentiment Analysis Bot")
st.write("Predict whether the entered text is Positive, Neutral, or Negative.")

# Load the model only once
@st.cache_resource
def load_model():
    return pipeline(
        "text-classification",
        model="cardiffnlp/twitter-roberta-base-sentiment-latest",
        truncation=True
    )

classifier = load_model()

# User input
text = st.text_area("Enter your text:", height=150)

# Analyze button
if st.button("Analyze Sentiment"):

    if text.strip() == "":
        st.warning("Please enter some text.")
    else:
        result = classifier(text)[0]

        sentiment = result["label"].capitalize()
        confidence = result["score"]

        st.subheader("Prediction")

        if sentiment == "Positive":
            st.success(f"😊 Sentiment: {sentiment}")
        elif sentiment == "Neutral":
            st.info(f"😐 Sentiment: {sentiment}")
        else:
            st.error(f"😞 Sentiment: {sentiment}")

        st.write(f"**Confidence:** {confidence:.2%}")