from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier

from src.preprocess import clean_text


def train_and_evaluate_models(Xv_train, Xv_test, y_train, y_test):
    """Train multiple ML models and display performance metrics."""

    # Defining ML algorithms to train

    models = {
        "Logistic Regression": LogisticRegression(),
        "Decision Tree": DecisionTreeClassifier(random_state=0),
        "Gradient Boosting": GradientBoostingClassifier(random_state=0),
        "Random Forest": RandomForestClassifier(random_state=0),
    }
    trained_models = {}

    print("=" * 50)
    print("MODEL TRAINING AND EVALUATION REPORT")
    print("=" * 50)

    # Iterating through models
    for name, model in models.items():
        model.fit(Xv_train, y_train)
        predictions = model.predict(Xv_test)
        acc = accuracy_score(y_test, predictions)

        # Printing Model report
        print(f"\nModel: {name}")
        print(f"Accuracy: {acc:.4f}")
        print("Classification Report:")
        print(classification_report(y_test, predictions))

        # Storing trained model instance
        trained_models[name] = model

    return trained_models


def predict_news(news_text, vectorizer, trained_models):
    """Preprocess a custom news input string, vectorize it and return predictions."""

    cleaned_input = clean_text(news_text)
    transformed_input = vectorizer.transform([cleaned_input])

    results = {}
    label_map = {0: "Fake News", 1: "Not a Fake News"}

    # Predicting class labels

    for name, model in trained_models.items():
        pred = model.predict(transformed_input)[0]
        results[name] = label_map[pred]

    return results
