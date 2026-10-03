import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)


# ============================================================
# 1. LOAD DATASET
# ============================================================

DATASET_PATH = "machine-dataset.csv"

df = pd.read_csv(DATASET_PATH)

print("=" * 60)
print("DATASET LOADED")
print("=" * 60)

print(f"Number of rows    : {df.shape[0]}")
print(f"Number of columns : {df.shape[1]}")

print("\nColumns:")
print(df.columns.tolist())


# ============================================================
# 2. CHECK REQUIRED COLUMNS
# ============================================================

required_columns = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    "Machine failure"
]

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"The following required columns are missing: {missing_columns}"
    )


# ============================================================
# 3. HANDLE MISSING VALUES
# ============================================================

print("\nMissing values:")
print(df[required_columns].isnull().sum())

if df[required_columns].isnull().sum().sum() > 0:
    print("\nMissing values found. Removing affected rows...")
    df = df.dropna(subset=required_columns)


# ============================================================
# 4. DEFINE FEATURES AND TARGET
# ============================================================

# Features selected for prediction
FEATURES = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]

TARGET = "Machine failure"

X = df[FEATURES]
y = df[TARGET]


# ============================================================
# 5. DISPLAY TARGET DISTRIBUTION
# ============================================================

print("\nTarget distribution:")
print(y.value_counts())

print("\nTarget percentage:")
print(y.value_counts(normalize=True) * 100)


# ============================================================
# 6. TRAIN-TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples :", len(X_train))
print("Testing samples  :", len(X_test))


# ============================================================
# 7. PREPROCESSING
# ============================================================

# Type is categorical.
# Convert Type into numerical representation using OneHotEncoder.

categorical_features = ["Type"]

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ],
    remainder="passthrough"
)


# ============================================================
# 8. RANDOM FOREST MODEL
# ============================================================

random_forest = RandomForestClassifier(
    n_estimators=300,
    max_features="sqrt",
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)


# ============================================================
# 9. CREATE COMPLETE ML PIPELINE
# ============================================================

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", random_forest)
    ]
)


# ============================================================
# 10. TRAIN MODEL
# ============================================================

print("\n" + "=" * 60)
print("TRAINING RANDOM FOREST MODEL")
print("=" * 60)

model.fit(X_train, y_train)

print("Training completed successfully!")


# ============================================================
# 11. MAKE PREDICTIONS
# ============================================================

y_pred = model.predict(X_test)

# Probability of machine failure
y_probability = model.predict_proba(X_test)[:, 1]


# ============================================================
# 12. MODEL EVALUATION
# ============================================================

accuracy = accuracy_score(y_test, y_pred)

roc_auc = roc_auc_score(
    y_test,
    y_probability
)

print("\n" + "=" * 60)
print("MODEL PERFORMANCE")
print("=" * 60)

print(f"\nAccuracy : {accuracy:.4f}")
print(f"ROC-AUC  : {roc_auc:.4f}")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        digits=4
    )
)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# ============================================================
# 13. SAVE MODEL
# ============================================================

MODEL_PATH = "machine_failure_model.pkl"

joblib.dump(model, MODEL_PATH)

print("\n" + "=" * 60)
print("MODEL SAVED")
print("=" * 60)

print(f"Saved model: {MODEL_PATH}")

print("\nTraining process completed successfully!")