# Breast Cancer Classification Model Lab

## a. Problem statement
Build and compare five supervised classification models on one labelled dataset. The project reports Accuracy, AUC, Precision, Recall, F1 Score, and Matthews Correlation Coefficient (MCC), and provides an interactive Streamlit interface for evaluating labelled test CSV files using saved models.

This is an educational machine-learning project. It is not a medical diagnostic tool.

## b. Dataset description
- **Name:** Breast Cancer Wisconsin (Diagnostic)
- **Source:** UCI Machine Learning Repository; available through `sklearn.datasets.load_breast_cancer`
- **Full dataset:** `dataset.csv`
- **Rows:** 569
- **Input features:** 30 numeric measurements derived from digitized images of fine needle aspirates
- **Target:** `diagnosis`, with two labels: `malignant` and `benign`
- **Split:** stratified 75% training / 25% test, with random state `202605115`
- **Files:** `train_data.csv` and `test_data.csv`. The test CSV includes all feature columns plus the target column, as required for calculating evaluation metrics.

The original scikit-learn target is mapped to readable labels (`0 → malignant`, `1 → benign`). Do not use these results for clinical decisions.

## c. GitHub Repository Link
**Add your repository URL here after you create and push your own repository:**
`[PASTE YOUR GITHUB REPOSITORY URL HERE]`

## d. Models used

The following models are trained on the same training split:
1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes
5. Support Vector Machine (RBF kernel)

### Evaluation metric comparison

All metrics below are calculated on the held-out test split. `malignant` is treated as the positive class for Precision, Recall, F1, and AUC.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9860 | 0.9920 | 1.0000 | 0.9623 | 0.9808 | 0.9702 |
| Decision Tree | 0.9510 | 0.9725 | 0.9259 | 0.9434 | 0.9346 | 0.8956 |
| K-Nearest Neighbors | 0.9860 | 0.9891 | 1.0000 | 0.9623 | 0.9808 | 0.9702 |
| Gaussian Naive Bayes | 0.9510 | 0.9933 | 0.9600 | 0.9057 | 0.9320 | 0.8948 |
| Support Vector Machine | 0.9930 | 0.9931 | 1.0000 | 0.9811 | 0.9905 | 0.9851 |

### Observations on model performance

These observations are generated from the held-out test results above. Review the numbers and update the discussion after you run the project in BITS Virtual Lab.

- **Logistic Regression:** Accuracy = 0.9860, AUC = 0.9920, and MCC = 0.9702. Consider these together rather than judging the model by accuracy alone.\n- **Decision Tree:** Accuracy = 0.9510, AUC = 0.9725, and MCC = 0.8956. Consider these together rather than judging the model by accuracy alone.\n- **K-Nearest Neighbors:** Accuracy = 0.9860, AUC = 0.9891, and MCC = 0.9702. Consider these together rather than judging the model by accuracy alone.\n- **Gaussian Naive Bayes:** Accuracy = 0.9510, AUC = 0.9933, and MCC = 0.8948. Consider these together rather than judging the model by accuracy alone.\n- **Support Vector Machine:** Accuracy = 0.9930, AUC = 0.9931, and MCC = 0.9851. Consider these together rather than judging the model by accuracy alone.\n
**Provisional overall winner:** Support Vector Machine by the predeclared comparison order of F1, then AUC, then MCC. Confirm this conclusion against your own run and explain any trade-offs in your submission.

## Streamlit app features
- Model-selection dropdown for all five saved classifiers.
- Bundled sample test data or user-uploaded labelled CSV.
- Evaluation metrics: Accuracy, AUC, Precision, Recall, F1, and MCC.
- Confusion matrix and classification report.
- Preview of the loaded test data.
- Models are loaded from `model/`; the app does not retrain on uploaded test data.

## Project structure

```text
ML_Assignment_2_202605115/
├── app.py
├── train_models.py
├── requirements.txt
├── README.md
├── dataset.csv
├── train_data.csv
├── test_data.csv
├── model/
│   ├── logistic_regression.joblib
│   ├── decision_tree.joblib
│   ├── knn_classifier.joblib
│   ├── gaussian_naive_bayes.joblib
│   └── support_vector_machine.joblib
└── reports/
    └── model_comparison.csv
```

## Run locally or in BITS Virtual Lab

1. Open a terminal in this project folder.
2. Install dependencies: `pip install -r requirements.txt`
3. If you need to regenerate model files and CSV splits, run: `python train_models.py`
4. Start the app: `streamlit run app.py`
5. Open the Streamlit URL displayed by the environment.
6. Test the default bundled dataset, then uncheck **Use the included sample test data** and upload `test_data.csv`.

## Deployment notes
- Push the project to your own GitHub repository when you are ready.
- In Streamlit Community Cloud, select that repository, the branch, and `app.py`.
- Confirm the deployed app opens and can evaluate the sample test data.
- Paste the final repository and app URLs into the submission PDF.

## Reproducibility and limitations
- `random_state=202605115` is used for the stratified split.
- Preprocessing is bundled inside each saved pipeline where scaling is required.
- The dataset is small and curated; test results are not a guarantee of generalization to other data.
- The `test_data.csv` file is labelled because the assignment requires ground-truth labels for evaluation.
