import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


# ==========================================
# REVENUE RESCUE AI
# RECOVERY PREDICTION MODEL
# ==========================================

print("\n==========================================")
print("       REVENUE RESCUE AI - ML MODEL")
print("==========================================")


# ==========================================
# 1. Load training data
# ==========================================

df = pd.read_csv(
    "data/recovery_training.csv"
)

print("\nTraining dataset:")
print(df.shape)


# ==========================================
# 2. Separate features and target
# ==========================================

X = df.drop(
    columns=["recovered"]
)

y = df["recovered"]


# ==========================================
# 3. Define categorical features
# ==========================================

categorical_features = [
    "payment_method",
    "failure_reason"
]


# ==========================================
# 4. Define numerical features
# ==========================================

numerical_features = [
    "amount",
    "attempt_number",
    "previous_successes",
    "previous_failures",
    "is_high_value",
    "is_first_attempt",
    "customer_success_ratio"
]


# ==========================================
# 5. Preprocessing
# ==========================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            categorical_features
        ),
        (
            "numerical",
            "passthrough",
            numerical_features
        )
    ]
)


# ==========================================
# 6. Create ML model
# ==========================================

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    min_samples_split=5,
    random_state=42,
    class_weight="balanced"
)


# ==========================================
# 7. Create complete pipeline
# ==========================================

pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "model",
            model
        )
    ]
)


# ==========================================
# 8. Train/Test Split
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("\nTraining samples:", len(X_train))
print("Testing samples :", len(X_test))


# ==========================================
# 9. Train model
# ==========================================

print("\nTraining Random Forest...")

pipeline.fit(
    X_train,
    y_train
)

print("Training complete.")


# ==========================================
# 10. Predictions
# ==========================================

y_pred = pipeline.predict(X_test)

y_probability = pipeline.predict_proba(
    X_test
)[:, 1]


# ==========================================
# 11. Model Evaluation
# ==========================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_test,
    y_probability
)


print("\n==========================================")
print("             MODEL PERFORMANCE")
print("==========================================")

print(
    f"\nAccuracy : {accuracy:.4f}"
)

print(
    f"Precision: {precision:.4f}"
)

print(
    f"Recall   : {recall:.4f}"
)

print(
    f"F1 Score : {f1:.4f}"
)

print(
    f"ROC-AUC  : {roc_auc:.4f}"
)


# ==========================================
# 12. Confusion Matrix
# ==========================================

print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_test,
        y_pred
    )
)


# ==========================================
# 13. Classification Report
# ==========================================

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)


# ==========================================
# 14. Save Model
# ==========================================

model_file = (
    "ml/recovery_model.joblib"
)

joblib.dump(
    pipeline,
    model_file
)


print("\n==========================================")
print("MODEL SAVED")
print("==========================================")

print(
    f"\nModel location:"
    f"\n{model_file}"
)

print("\n==========================================")