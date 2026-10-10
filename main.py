from src.preprocess import load_and_preprocess_data
from src.train_eval import predict_news, train_and_evaluate_models


def main():
    "Main execution function for Fake News Detection pipeline"

    fake_path = "data/Fake.csv"
    true_path = "data/True.csv"

    # Preprocessing and TF_IDF Extraction
    print("Data Preprocessing and Vectorization")
    Xv_train, Xv_test, y_train, y_test, vectorizer = load_and_preprocess_data(
        fake_path, true_path
    )

    # Model Training and Evaluation

    print("\nTraining and Evaluating ML Models")
    trained_models = train_and_evaluate_models(Xv_train, Xv_test, y_train, y_test)

    # Real_time Manual Testing Interface

    print("\n" + "=" * 50)
    print("Test Real_time News Article")
    print("=" * 50)

    user_input = input("\nEnter News Article Text: ")
    if user_input.strip():
        predictions = predict_news(user_input, vectorizer, trained_models)
        print("\n--- Model Predictions ---")
        for model_name, result in predictions.items():
            print(f"{model_name}: {result}")


if __name__ == "__main__":
    main()
