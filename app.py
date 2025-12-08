import streamlit as st
import joblib
import numpy as np

# ---------------------------------------------------------
# 1. Load the trained pipeline (TF-IDF + LogisticRegression)
# ---------------------------------------------------------
@st.cache_resource
def load_model():
    model = joblib.load("logreg_tfidf_cv_best.joblib")
    return model

model = load_model()

# Get class labels (e.g., ["negative", "positive"])
CLASS_NAMES = list(model.classes_)

# ---------------------------------------------------------
# 2. Streamlit UI
# ---------------------------------------------------------
st.set_page_config(page_title="Amazon Food Review Sentiment", layout="centered")

st.title("Amazon Fine Food Review  Sentiment Classifier")

st.write(
    "Type a food-related review below and the model will classify it as "
    "**negative** or **positive**, using the Logistic Regression model trained on the balanced dataset."
)

user_text = st.text_area(
    "Review text:",
    height=150,
    placeholder="e.g. The pasta was cold and the sauce had no flavour..."
)

if st.button("Predict sentiment"):
    if not user_text.strip():
        st.warning("Please enter a review first.")
    else:
        # The pipeline expects a list/array of texts
        preds = model.predict([user_text])
        pred_label = preds[0]

        # Try to get probabilities (LogisticRegression supports predict_proba)
        prob_text = ""
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba([user_text])[0]
            prob_pairs = list(zip(CLASS_NAMES, probs))
            prob_text_lines = [f"- **{cls}**: {p:.3f}" for cls, p in prob_pairs]
            prob_text = "\n".join(prob_text_lines)

        st.subheader("🔎 Prediction")
        st.markdown(f"**Sentiment:** `{pred_label}`")

        if prob_text:
            st.subheader("📊 Class probabilities")
            st.markdown(prob_text)

        st.info(
            "This prediction is based on a TF-IDF + Logistic Regression model trained on a balanced subset "
            "of the Amazon Fine Food Reviews dataset."
        )
