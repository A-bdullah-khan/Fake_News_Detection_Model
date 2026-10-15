import re
import string

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split


def clean_text(text):
    text = text.lower()
    text = re.sub(r"\[.*?\]", "", text)
    text = re.sub(r"\W", " ", text)
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"<.*?>+", "", text)
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
    text = re.sub(r"\n", " ", text)
    text = re.sub(r"\w*\d\w*", " ", text)
    return text


def load_and_preprocess_data(fake_path="data/Fake.csv", true_path="data/True.csv"):
    data_fake = pd.read_csv(fake_path)
    data_true = pd.read_csv(true_path)

    data_fake["class"] = 0
    data_true["class"] = 1

    data_fake = data_fake.iloc[:-10]
    data_true = data_true.iloc[:-10]

    data_merge = pd.concat([data_fake, data_true], axis=0)
    data = data_merge.drop(["title", "subject", "date"], axis=1)

    data = data.sample(frac=1, random_state=42).reset_index(drop=True)

    data["text"] = data["text"].apply(clean_text)

    X = data["text"]
    y = data["class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    vectorizer = TfidfVectorizer(stop_words="english")
    Xv_train = vectorizer.fit_transform(X_train)
    Xv_test = vectorizer.transform(X_test)

    return Xv_train, Xv_test, y_train, y_test, vectorizer
