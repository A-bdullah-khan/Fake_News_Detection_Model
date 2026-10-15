# Fake News Detection Model

A modular Machine Learning pipeline that classifies news articles as **Real** or **Fake**. Articles are vectorized with TF-IDF and four Scikit-Learn classifiers are trained and compared. Logistic Regression reaches **~98.6% accuracy** on the held-out test set. Trained models are saved to disk, so later runs load them instead of retraining.

## Tech Stack

| Area | Tools |
| --- | --- |
| Language | Python 3.10+ |
| Data processing / ML | Scikit-Learn, Pandas, NumPy |
| Vectorization | TF-IDF (`sklearn.feature_extraction.text.TfidfVectorizer`) |
| Models evaluated | Logistic Regression, Decision Tree, Gradient Boosting, Random Forest |
| Persistence | joblib |

## Pipeline

1. **Loading and labeling** (`src/preprocess.py`): reads `Fake.csv` (label `0`) and `True.csv` (label `1`). It holds back the last 10 rows of each file, keeps only the `text` column and shuffles the data (`random_state=42`).
2. **Text cleaning** (`clean_text`): converts text to lowercase and uses regex to remove bracketed text, URLs, HTML tags, punctuation, newlines and tokens that contain digits.
3. **Train/test split**: 75/25 split (`random_state=42`).
4. **Feature extraction**: `TfidfVectorizer(stop_words="english")` splits the text into tokens and removes English stop words. It is fit on the training set only, then applied to the test set.
5. **Model training and comparison** (`src/train_eval.py`): trains all four classifiers and prints the accuracy and a full `classification_report` (precision, recall, F1) for each.
6. **Persistence**: `save_artifacts` writes the fitted vectorizer and every trained model to `models/`. On later runs, `load_artifacts` loads them back.
7. **Manual prediction**: `predict_news` cleans and vectorizes an input string, then returns each model's prediction.

### Results

These are test-set accuracies on 11,220 held-out articles:

| Model | Accuracy |
| --- | --- |
| Logistic Regression | 0.986 |
| Decision Tree | 0.995 |
| Gradient Boosting | 0.994 |
| Random Forest | 0.992 |

The tree-based models score higher, but they may be picking up source-specific artifacts rather than general signals. For example, 99.8% of articles in `True.csv` contain the string "Reuters", compared with 1.3% in `Fake.csv`. Logistic Regression is the more conservative baseline.

## Project Structure

```
Fake_News_Detection_Model/
├── data/
│   ├── Fake.csv                    # Fake articles (not tracked in git)
│   └── True.csv                    # Real articles (not tracked in git)
├── models/                         # Created on first run (not tracked in git)
│   ├── vectorizer.joblib
│   ├── logistic_regression.joblib
│   ├── decision_tree.joblib
│   ├── gradient_boosting.joblib
│   └── random_forest.joblib
├── src/
│   ├── preprocess.py               # Text cleaning, data loading, train/test split, TF-IDF
│   └── train_eval.py               # Training, evaluation, save/load, inference helper
├── main.py                         # Entry point: loads or trains models, then runs the manual test prompt
├── requirements.txt
└── README.md
```

The saved artifacts take up about 63 MB, almost all of it the Random Forest model.

## Setup

```bash
git clone <repository-url>
cd Fake_News_Detection_Model

python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Dataset

`data/*.csv` is excluded by `.gitignore`. Download the [Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset) from Kaggle and put the files here:

```
data/Fake.csv
data/True.csv
```

Each file needs the columns `title`, `text`, `subject` and `date`.

## Usage

```bash
python main.py              # Load models from models/, or train and save them if missing
python main.py --retrain    # Retrain all models and overwrite models/
```

The first run trains all four models and saves them, which takes about 4–5 minutes on a laptop CPU. Most of that time goes to Gradient Boosting and Random Forest. Later runs load the saved files in a few seconds.

Retrain after any change to the dataset, `clean_text` or the vectorizer settings. Otherwise the saved models will be out of date.

When the models are ready, the script asks for an article:

```
Enter News Article Text: <paste article text>

--- Model Predictions ---
Logistic Regression: Fake News
Decision Tree: Fake News
Gradient Boosting: Fake News
Random Forest: Fake News
```

### Using the modules directly

```python
from src.train_eval import load_artifacts, predict_news

models, vectorizer = load_artifacts()   # returns None if models/ is incomplete
print(predict_news("Article text here...", vectorizer, models))
```

To train from scratch in code:

```python
from src.preprocess import load_and_preprocess_data
from src.train_eval import save_artifacts, train_and_evaluate_models

Xv_train, Xv_test, y_train, y_test, vectorizer = load_and_preprocess_data(
    "data/Fake.csv", "data/True.csv"
)
models = train_and_evaluate_models(Xv_train, Xv_test, y_train, y_test)
save_artifacts(models, vectorizer)
```

## Author

Abdullah Haroon Khan
