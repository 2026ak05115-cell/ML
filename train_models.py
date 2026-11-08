"""
Rebuild saved classifier pipelines and split files.
Run: python train_models.py

The script uses the public Breast Cancer Wisconsin Diagnostic dataset bundled
with scikit-learn. It writes dataset.csv, train_data.csv, test_data.csv, and
five saved models under model/. The Streamlit app itself never retrains models.
"""
from pathlib import Path
import joblib
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
MODEL_DIR.mkdir(exist_ok=True)

data = load_breast_cancer(as_frame=True)
frame = data.frame.copy()
frame["diagnosis"] = frame["target"].map({0: "malignant", 1: "benign"})
frame = frame.drop(columns=["target"])
features = list(data.feature_names)
X, y = frame[features], frame["diagnosis"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=202605115, stratify=y
)
frame.to_csv(ROOT / "dataset.csv", index=False)
train = X_train.copy()
train["diagnosis"] = y_train
train.to_csv(ROOT / "train_data.csv", index=False)
test = X_test.copy()
test["diagnosis"] = y_test
test.to_csv(ROOT / "test_data.csv", index=False)

models = {
    "logistic_regression.joblib": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=5000, random_state=2026))
    ]),
    "decision_tree.joblib": DecisionTreeClassifier(
        max_depth=5, min_samples_leaf=3, random_state=2026
    ),
    "knn_classifier.joblib": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", KNeighborsClassifier(n_neighbors=7))
    ]),
    "gaussian_naive_bayes.joblib": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", GaussianNB())
    ]),
    "support_vector_machine.joblib": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", SVC(kernel="rbf", probability=True, random_state=2026))
    ]),
}
for filename, classifier in models.items():
    classifier.fit(X_train, y_train)
    joblib.dump(classifier, MODEL_DIR / filename)
    print(f"Saved {filename}")
print(f"Complete dataset: {len(frame)} rows; train: {len(train)}; test: {len(test)}")
