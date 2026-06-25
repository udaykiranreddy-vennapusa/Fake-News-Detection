import streamlit as st
import joblib

# Load model and vectorizer
vectorizer = joblib.load("vectorizer.jb")
model = joblib.load("lr_model.jb")

# Page Config
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="centered"
)

# Custom CSS
st.markdown("""
<style>
.main {
    padding-top: 2rem;
}
.result-box {
    padding: 15px;
    border-radius: 10px;
    font-size: 20px;
    font-weight: bold;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# Header
st.title("📰 Fake News Detector")
st.markdown(
    "### Verify whether a news article is **Real** or **Fake** using Machine Learning."
)

# Sidebar
with st.sidebar:
    st.header("📌 About")

    st.write("""
    **Fake News Detector** uses Machine Learning to classify news articles as **Real** or **Fake**.
    """)

    st.markdown("### ⚙️ Tech Stack")
    st.markdown("""
    - TF-IDF
    - Logistic Regression
    - Streamlit
    """)

    st.markdown("### 🚀 Usage")
    st.markdown("""
    1. Paste a news article
    2. Click **Analyze News**
    3. View the prediction
    """)

# Input Area
news_text = st.text_area(
    "📝 Enter News Article",
    height=250,
    placeholder="Paste the complete news article here..."
)

# Analyze Button
if st.button("🔍 Analyze News", use_container_width=True):

    if news_text.strip():

        transformed_text = vectorizer.transform([news_text])

        prediction = model.predict(transformed_text)[0]

        # Probability (if supported)
        try:
            probability = model.predict_proba(transformed_text)

            confidence = round(
                max(probability[0]) * 100,
                2
            )
        except:
            confidence = None

        st.divider()

        if prediction == 1:
            st.success("✅ This News Appears to be REAL")
        else:
            st.error("🚨 This News Appears to be FAKE")

        if confidence:
            st.metric(
                label="Confidence Score",
                value=f"{confidence}%"
            )

    else:
        st.warning("⚠️ Please enter a news article.")