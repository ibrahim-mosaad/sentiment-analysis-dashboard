import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report

from preprocess import clean_text

# Load Dataset
data = pd.read_csv("data/dataset.csv")

# Keep needed columns
data = data[['review', 'sentiment']]

# Clean text
data['clean_text'] = data['review'].apply(clean_text)

# Features & labels
X = data['clean_text']
y = data['sentiment']

# TF-IDF
vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1,2)
)

X_vectorized = vectorizer.fit_transform(X)

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized,
    y,
    test_size=0.2,
    random_state=42
)

# SVM Model
model = LinearSVC()

# Train
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Accuracy
acc = accuracy_score(y_test, y_pred)

print("Accuracy:", acc)

print(classification_report(y_test, y_pred))

# Save Model
joblib.dump(model, "sentiment_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")