from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix, classification_report
)

APP_DIR = Path(__file__).resolve().parent
MODEL_DIR = APP_DIR / "model"
TARGET = "diagnosis"
POSITIVE_LABEL = "malignant"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "K-Nearest Neighbors": "knn_classifier.joblib",
    "Gaussian Naive Bayes": "gaussian_naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib",
}

@st.cache_resource
def load_model(model_path):
    return joblib.load(model_path)

def calculate_metrics(model, X, y_true):
    y_pred = model.predict(X)
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, pos_label=POSITIVE_LABEL, zero_division=0),
        "Recall": recall_score(y_true, y_pred, pos_label=POSITIVE_LABEL, zero_division=0),
        "F1 Score": f1_score(y_true, y_pred, pos_label=POSITIVE_LABEL, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }
    try:
        classes = list(model.classes_)
        pos_index = classes.index(POSITIVE_LABEL)
        y_binary = (np.asarray(y_true) == POSITIVE_LABEL).astype(int)
        probabilities = model.predict_proba(X)[:, pos_index]
        metrics["AUC"] = roc_auc_score(y_binary, probabilities)
    except (AttributeError, ValueError, IndexError):
        metrics["AUC"] = float("nan")
    return y_pred, metrics

st.set_page_config(page_title="Classification Model Lab", page_icon="🧪", layout="wide")
st.title("🧪 Breast Cancer Classification Model Lab")
st.write(
    "Compare five trained classifiers on a labelled CSV test dataset. "
    "The app loads saved models; it does not retrain them on uploaded data."
)

with st.sidebar:
    st.header("Evaluation settings")
    selected_model_name = st.selectbox("Choose a classifier", list(MODEL_FILES.keys()))
    use_sample = st.checkbox("Use the included sample test data", value=True)
    st.caption("Target column required: `diagnosis` (malignant / benign).")

if use_sample:
    data_path = APP_DIR / "test_data.csv"
    if not data_path.exists():
        st.error("The bundled test_data.csv file was not found.")
        st.stop()
    test_df = pd.read_csv(data_path)
    st.info("Evaluating the bundled held-out test dataset.")
else:
    uploaded_file = st.file_uploader("Upload labelled test data (CSV)", type=["csv"])
    if uploaded_file is None:
        st.info("Upload a CSV containing all 30 feature columns and the diagnosis target.")
        st.stop()
    try:
        test_df = pd.read_csv(uploaded_file)
    except Exception as exc:
        st.error(f"Could not read CSV: {exc}")
        st.stop()

if TARGET not in test_df.columns:
    st.error(f"CSV must contain a `{TARGET}` target column.")
    st.stop()

model_path = MODEL_DIR / MODEL_FILES[selected_model_name]
if not model_path.exists():
    st.error(f"Saved model not found: {model_path.name}")
    st.stop()
model = load_model(str(model_path))
expected_features = list(model.feature_names_in_)
missing = [col for col in expected_features if col not in test_df.columns]
if missing:
    st.error("The CSV is missing required feature columns: " + ", ".join(missing))
    st.stop()

X_test = test_df[expected_features].copy()
y_test = test_df[TARGET].astype(str).str.strip().str.lower()
valid_labels = {"malignant", "benign"}
if not set(y_test.unique()).issubset(valid_labels):
    st.error("The diagnosis column must contain only 'malignant' and 'benign'.")
    st.stop()
if X_test.isnull().any().any():
    st.error("The test CSV contains missing feature values. Please clean the data and try again.")
    st.stop()
try:
    y_pred, metrics = calculate_metrics(model, X_test, y_test)
except Exception as exc:
    st.error(f"Evaluation failed: {exc}")
    st.stop()

st.subheader(f"Results: {selected_model_name}")
st.caption(f"Rows evaluated: {len(test_df)} | Features used: {len(expected_features)}")
metric_cols = st.columns(6)
for col, (label, value) in zip(metric_cols, metrics.items()):
    col.metric(label, "N/A" if not np.isfinite(value) else f"{value:.4f}")

left, right = st.columns(2)
with left:
    st.markdown("#### Confusion matrix")
    labels = ["malignant", "benign"]
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    fig, ax = plt.subplots()
    image = ax.imshow(cm)
    ax.set_xticks(range(len(labels)), labels=labels, rotation=15)
    ax.set_yticks(range(len(labels)), labels=labels)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion Matrix")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    fig.colorbar(image, ax=ax)
    st.pyplot(fig)
    plt.close(fig)
with right:
    st.markdown("#### Classification report")
    report = classification_report(y_test, y_pred, labels=labels, zero_division=0, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose().round(4), use_container_width=True)

with st.expander("Preview test data"):
    st.dataframe(test_df.head(20), use_container_width=True)

st.caption(
    "Educational demonstration only. This dataset is for machine-learning coursework "
    "and must not be used to make medical decisions."
)
