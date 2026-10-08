import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text


def load_dataset(file_path):
    """Load and preprocess the Iris dataset from CSV."""
    df = pd.read_csv(file_path)

    # Clean up column names if needed
    df.columns = df.columns.str.strip()

    # Extract feature matrix and target values
    feature_cols = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    X = df[feature_cols]
    y = df["species"]

    return X, y, feature_cols


def train_and_evaluate_model(X, y):
    """Split data, train decision tree classifier, and print evaluation metrics."""
    # Split into 80% training and 20% testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Initialize Decision Tree Classifier using Gini impurity criterion
    dt_model = DecisionTreeClassifier(criterion="gini", max_depth=4, random_state=42)

    # Train model on training data
    dt_model.fit(X_train, y_train)

    # Predict on test data
    y_pred = dt_model.predict(X_test)

    # Compute required metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted")
    rec = recall_score(y_test, y_pred, average="weighted")

    # Output metric summary
    print("=" * 45)
    print("       DECISION TREE MODEL PERFORMANCE       ")
    print("=" * 45)
    print(f"Accuracy  : {acc * 100:.2f}%")
    print(f"Precision : {prec * 100:.2f}%")
    print(f"Recall    : {rec * 100:.2f}%")
    print("-" * 45)

    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred))

    print("\nConfusion Matrix:")
    labels = sorted(list(set(y)))
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    cm_df = pd.DataFrame(
        cm, index=[f"Actual {l}" for l in labels], columns=[f"Pred {l}" for l in labels]
    )
    print(cm_df)

    return dt_model


def main():
    csv_file = "iris.csv"

    try:
        X, y, feature_names = load_dataset(csv_file)
        model = train_and_evaluate_model(X, y)

        # Print text rules of the trained decision tree
        print("\nDecision Tree Rules:")
        tree_rules = export_text(model, feature_names=feature_names)
        print(tree_rules)

    except FileNotFoundError:
        print(f"Error: Could not find '{csv_file}'. Ensure file is in current directory.")


if __name__ == "__main__":
    main()
