import joblib

from preprocess import clean_text

model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")


def predict_sentiment(text):

    cleaned = clean_text(text)

    vectorized = vectorizer.transform([cleaned])

    prediction = model.predict(vectorized)[0]

    confidence = abs(
        model.decision_function(vectorized)[0]
    )

    return prediction, confidence