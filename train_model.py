# ==============================================
# DUAL HEART DISEASE PREDICTION MODELS (OPTIMIZED)
# WITH ADVANCED PRE-PROCESSING + FEATURE ENGINEERING
# ==============================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, ConfusionMatrixDisplay, roc_curve, precision_recall_curve,
    average_precision_score, balanced_accuracy_score
)
from sklearn.exceptions import ConvergenceWarning
import warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)

import xgboost as xgb
import joblib
from tabulate import tabulate

# Set random seed for reproducibility
np.random.seed(42)

# ==============================================
# UTILITY: DATA PROFILING
# ==============================================
def profile_data(df, dataset_name):
    print(f"\n📊 DATA PROFILE: {dataset_name}")
    print("-" * 50)
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")
    
    missing = df.isnull().sum().sum()
    print(f"Missing Cells: {missing} ({100 * missing / df.size:.2f}%)")

    dupes = df.duplicated().sum()
    if dupes > 0:
        print(f"⚠️ Warning: {dupes} duplicate rows found (and removed).")
        df.drop_duplicates(inplace=True)

# ==============================================
# UTILITY: MODEL EVALUATION
# ==============================================
def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    metrics = {
        "Model": model_name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1": f1_score(y_test, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, y_proba) if y_proba is not None else np.nan
    }
    return metrics

# ==============================================
# UTILITY: HYPERPARAMETER TUNING
# ==============================================
def tune_and_train(name, model, param_grid, X_train, y_train):
    if param_grid:
        print(f"  🔍 Tuning {name}...")
        search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_grid,
            n_iter=15,
            cv=5,
            scoring='accuracy',
            n_jobs=-1,
            verbose=0,
            random_state=42
        )
        search.fit(X_train, y_train)
        print(f"  ✅ Best params for {name}: {search.best_params_}")
        return search.best_estimator_
    else:
        print(f"  Training {name} (no tuning)...")
        model.fit(X_train, y_train)
        return model

# ==============================================
# MODEL 1: CLINICAL MODEL (UCI)
# ==============================================
print("\n🚑 TRAINING CLINICAL MODELS...")
DATA_PATH_CLINICAL = "heart_disease_uci.csv"

if os.path.exists(DATA_PATH_CLINICAL):
    df1 = pd.read_csv(DATA_PATH_CLINICAL, header=None)
    df1.columns = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num"]
    
    df1 = df1.replace('?', np.nan).apply(pd.to_numeric, errors='coerce')
    df1 = df1.fillna(df1.median())
    
    profile_data(df1, "Clinical")

    X1 = df1.drop("num", axis=1)
    y1 = (df1["num"] > 0).astype(int)

    X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, y1, test_size=0.2, random_state=42)
    scaler1 = StandardScaler()
    X_train1_scaled = scaler1.fit_transform(X_train1)
    X_test1_scaled = scaler1.transform(X_test1)

    models1 = {
        "Random Forest": (RandomForestClassifier(random_state=42), {'n_estimators': [100, 200], 'max_depth': [10, 20]}),
        "XGBoost": (xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42), {'n_estimators': [100], 'learning_rate': [0.1]})
    }

    results1 = []
    for name, (model, grid) in models1.items():
        best = tune_and_train(name, model, grid, X_train1_scaled, y_train1)
        metrics = evaluate_model(best, X_test1_scaled, y_test1, name)
        results1.append(metrics)
        joblib.dump(best, f"heart_{name.lower().replace(' ', '_')}_clinical.pkl")
    
    joblib.dump(scaler1, "heart_scaler_clinical.pkl")
    print("\n✅ Clinical Results:\n", tabulate(results1, headers="keys", tablefmt="github", floatfmt=".4f"))
else:
    print("❌ Clinical dataset not found. Skipping.")

# ==============================================
# MODEL 2: LIFESTYLE MODEL (cardio_train.csv)
# Note: 'age' is in days → convert to years
# ==============================================
print("\n🏃 TRAINING LIFESTYLE MODELS (RF, XGB, MLP)...")
DATA_PATH_LIFESTYLE = "cardio_train.csv"

if not os.path.exists(DATA_PATH_LIFESTYLE):
    raise FileNotFoundError("❌ Lifestyle dataset 'cardio_train.csv' not found!")

df2 = pd.read_csv(DATA_PATH_LIFESTYLE, sep=';')

# Drop ID if present
if 'id' in df2.columns:
    df2 = df2.drop(columns=['id'])

# Validate target
if 'cardio' not in df2.columns:
    raise ValueError("❌ Target column 'cardio' missing!")

# Profile raw data
profile_data(df2, "Lifestyle (Cardio)")

# 🔧 CONVERT AGE FROM DAYS TO YEARS
if 'age' in df2.columns:
    print("  📅 Converting 'age' from days to years...")
    df2['age'] = (df2['age'] / 365.25).astype(int)

# Handle any missing values (unlikely, but safe)
df2 = df2.fillna(df2.mean(numeric_only=True))

# Separate features and target
X2 = df2.drop("cardio", axis=1)
y2 = df2["cardio"]

# Train-test split
X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, test_size=0.2, random_state=42)

# Scale features
scaler2 = StandardScaler()
X_train2_scaled = scaler2.fit_transform(X_train2)
X_test2_scaled = scaler2.transform(X_test2)

# Define models
models2 = {
    "Random Forest": (
        RandomForestClassifier(random_state=42),
        {
            'n_estimators': [100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_leaf': [1, 2]
        }
    ),
    "XGBoost": (
        xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'),
        {
            'n_estimators': [100, 200],
            'learning_rate': [0.01, 0.1],
            'max_depth': [3, 6]
        }
    ),
    "Neural Network (MLP)": (
        MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, random_state=42, alpha=0.01),
        None  # No tuning for MLP in this version
    )
}

# Train and evaluate
results2 = []
trained_models2 = {}

for name, (model, grid) in models2.items():
    best_model = tune_and_train(name, model, grid, X_train2_scaled, y_train2)
    # ✅ FIXED: No unpacking — returns ONE dict
    metrics = evaluate_model(best_model, X_test2_scaled, y_test2, name)
    results2.append(metrics)
    trained_models2[name] = best_model

# Save models with clean names
save_map_2 = {
    "Random Forest": "heart_rf_lifestyle.pkl",
    "XGBoost": "heart_xgb_lifestyle.pkl",
    "Neural Network (MLP)": "heart_mlp_lifestyle.pkl"
}

for name, model in trained_models2.items():
    fname = save_map_2.get(name, f"heart_{name.lower().replace(' ', '_')}_lifestyle.pkl")
    joblib.dump(model, fname)
    print(f"  💾 Saved {name} to {fname}")

# Save scaler and input template
joblib.dump(scaler2, "heart_scaler_lifestyle.pkl")
X2.head(1).to_csv("heart_user_template_lifestyle.csv", index=False)

# Display results
print("\n✅ Lifestyle Model Performance Comparison:")
print(tabulate(results2, headers="keys", tablefmt="github", floatfmt=".4f"))

print("\n✅ All models trained, tuned, and saved successfully!")