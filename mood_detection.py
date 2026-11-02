
import pandas as pd
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# Optional suggestions: if suggester.py is missing or broken, the rest still works.
try:
    from suggester import show_suggestions
except Exception:
    show_suggestions = None

DATASET_PATH = "mood_dataset.xlsx"
MODEL_PATH = "mood_model.pkl"


def load_dataset():
    """Load and validate the Excel dataset."""

    if not Path(DATASET_PATH).exists():
        raise FileNotFoundError(
            f"{DATASET_PATH} not found. "
            "Place it in the same folder as this script."
        )

    # Read Excel file
    df = pd.read_excel(DATASET_PATH, engine="openpyxl")

    # Validate required columns
    required_columns = {"text", "label"}

    if not required_columns.issubset(df.columns):
        raise ValueError(
            "Excel file must contain columns named "
            "'text' and 'label'."
        )

    # Clean data
    df = df.dropna(subset=["text", "label"]).copy()
    df["text"] = df["text"].astype(str).str.strip()
    df["label"] = df["label"].astype(str).str.strip()

    df = df[
        (df["text"] != "") &
        (df["label"] != "")
    ]

    # Remove duplicate text-label pairs
    df = df.drop_duplicates(subset=["text", "label"])

    if df.empty:
        raise ValueError("No valid data found in the Excel file.")

    if df["label"].nunique() < 2:
        raise ValueError(
            "Dataset must contain at least two different moods."
        )

    print("\nDataset size:", len(df))
    print("\nSamples per mood:")
    print(df["label"].value_counts())

    return df


def train_model():
    """Train, evaluate and save the model."""

    df = load_dataset()

    X = df["text"]
    y = df["label"]

    # Ensure every mood has enough examples to split
    counts = y.value_counts()

    if counts.min() < 2:
        raise ValueError(
            "Some moods have fewer than 2 examples. "
            "Add more examples for every mood."
        )

    # Create a stratified test set containing each mood
    n_classes = y.nunique()
    test_count = max(
        n_classes,
        int(len(df) * 0.2 + 0.5)
    )

    if test_count >= len(df):
        raise ValueError(
            "Dataset is too small for train/test evaluation. "
            "Add more examples."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_count,
        random_state=42,
        stratify=y
    )

    print("\nTraining samples:", len(X_train))
    print("Testing samples:", len(X_test))

    # TF-IDF + Logistic Regression
    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2)
        )),
        ("clf", LogisticRegression(
            max_iter=1000,
            class_weight="balanced"
        ))
    ])

    print("\nTraining model...")
    model.fit(X_train, y_train)

    # Evaluate
    predictions = model.predict(X_test)
    labels = model.classes_

    print(
        "\nAccuracy: {:.2f}%".format(
            accuracy_score(y_test, predictions) * 100
        )
    )

    print("\nClassification report:")
    print(classification_report(
        y_test,
        predictions,
        labels=labels,
        zero_division=0
    ))

    print("\nConfusion matrix:")
    print(confusion_matrix(
        y_test,
        predictions,
        labels=labels
    ))

    print("Label order:", list(labels))

    # Save trained model
    joblib.dump(model, MODEL_PATH)
    print("\nModel saved to:", MODEL_PATH)

    return model


def load_model():
    """Load the previously trained model."""

    if not Path(MODEL_PATH).exists():
        raise FileNotFoundError(
            "Trained model not found. Train the model first."
        )

    return joblib.load(MODEL_PATH)


def predict_mood(model, text):
    """Predict mood and return class probability scores."""

    if not isinstance(text, str) or not text.strip():
        raise ValueError("Please enter a valid message.")

    probabilities = model.predict_proba([text])[0]
    classes = model.classes_

    best_index = probabilities.argmax()

    scores = {
        str(label): round(float(score) * 100, 2)
        for label, score in zip(classes, probabilities)
    }

    return {
        "text": text,
        "mood": str(classes[best_index]),
        "scores": scores
    }


def main():
    try:
        model = train_model()

        print("\n--- Mood Detection Demo ---")
        print("Type a message, 'suggest' for example sentences, or 'q' to quit.")

        while True:
            text = input("\nYou: ")

            if text.strip().lower() == "q":
                print("Goodbye!")
                break

            # "suggest" or "suggest sad" -> show example sentences to type
            command = text.strip().lower()
            if command == "suggest" or command.startswith("suggest "):
                if show_suggestions is None:
                    print("Suggestions are not available (suggester.py missing).")
                else:
                    show_suggestions(model, command[len("suggest"):].strip() or None)
                continue

            try:
                result = predict_mood(model, text)

                print("\nPredicted mood:", result["mood"])
                print("Model scores:")

                for mood, score in sorted(
                    result["scores"].items(),
                    key=lambda item: item[1],
                    reverse=True
                ):
                    print(f"  {mood}: {score}%")

            except ValueError as error:
                print("Error:", error)

    except (FileNotFoundError, ValueError) as error:
        print("\nError:", error)


if __name__ == "__main__":
    main()
