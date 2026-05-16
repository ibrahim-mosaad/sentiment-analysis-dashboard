import streamlit as st

from model import predict_sentiment
from utils.charts import confidence_chart

# Page config
st.set_page_config(
    page_title="AI Sentiment Dashboard",
    page_icon="🧠",
    layout="wide"
)

# Sidebar
st.sidebar.title("🧠 AI Sentiment Dashboard")

st.sidebar.info("""
Machine Learning + NLP + SVM
""")

# Main title
st.title("🧠 AI Sentiment Analysis Dashboard")

st.markdown("""
Analyze customer reviews and predict sentiment using Artificial Intelligence.
""")

# Text input
user_input = st.text_area(
    "Enter text for analysis",
    height=200,
    placeholder="Type your review here..."
)

# Analyze button
if st.button("Analyze Sentiment"):

    prediction, confidence = predict_sentiment(user_input)

    st.subheader("Prediction Result")

    if prediction.lower() == "positive":

        st.success(f"😊 {prediction}")

    elif prediction.lower() == "negative":

        st.error(f"😡 {prediction}")

    else:

        st.warning(f"😐 {prediction}")

    # Metrics
    st.metric(
        label="Confidence Score",
        value=round(confidence, 2)
    )

    # Plotly chart
    fig = confidence_chart(confidence)

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# Footer
st.markdown("---")

st.markdown("""
Developed by Ibrahim Hamada Mosaad
""")