import re
import nltk
import string

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download("stopwords")

stemmer = PorterStemmer()
stop_words = set(stopwords.words("english"))


def clean_text(text):

    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+", "", text)

    # Remove mentions
    text = re.sub(r"@\w+", "", text)

    # Remove hashtags
    text = re.sub(r"#\w+", "", text)

    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Remove numbers
    text = re.sub(r"\d+", "", text)

    words = text.split()

    # Remove stopwords + stemming
    words = [
        stemmer.stem(word)
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)