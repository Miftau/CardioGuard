# ==============================================
# DUAL HEART DISEASE PREDICTION MODELS
# WITH HYPERPARAMETER TUNING + DATA PROFILING
# ==============================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    RocCurveDisplay, confusion_matrix
)
from sklearn.exceptions import ConvergenceWarning
import warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)

import xgboost as xgb
import joblib
from tabulate import tabulate

# ==============================================
# UTILITY: DATA PROFILING
# ==============================================
def profile_data(df, dataset_name):
    print(f"\n📊 DATA PROFILE: {dataset_name}")
    print("-" * 50)
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")
    print(f"Data Types:\n{df.dtypes.value_counts().to_string()}")
    
    total_cells = df.size
    missing_cells = df.isnull().sum().sum()
    valid_cells = total_cells - missing_cells
    print(f"\nValid Cells: {valid_cells} ({100 * valid_cells / total_cells:.2f}%)")
    print(f"Missing/Invalid Cells: {missing_cells} ({100 * missing_cells / total_cells:.2f}%)")

    # Missing per column
    missing_per_col = df.isnull().sum()
    if missing_per_col.sum() > 0:
        print("\nMissing Values by Column:")
        missing_df = missing_per_col[missing_per_col > 0].to_frame(name='Missing Count')
        missing_df['% Missing'] = (missing_df['Missing Count'] / len(df)) * 100
        print(tabulate(missing_df, headers='keys', tablefmt='github', floatfmt=".2f"))

    # Target distribution (assume last column is target)
    target_col = df.columns[-1]
    if df[target_col].dtype in ['int64', 'float64']:
        target_dist = df[target_col].value_counts(normalize=True) * 100
        print(f"\nTarget Distribution ({target_col}):")
        for val, pct in target_dist.items():
            print(f"  Class {val}: {pct:.2f}%")

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
    return metrics, y_proba

# ==============================================
# UTILITY: HYPERPARAMETER TUNING
# ==============================================
def tune_and_train(name, model, param_grid, X_train, y_train):
    if param_grid:
        print(f"  🔍 Tuning {name}...")
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=3,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        grid_search.fit(X_train, y_train)
        print(f"  ✅ Best params for {name}: {grid_search.best_params_}")
        return grid_search.best_estimator_
    else:
        print(f"  Training {name} (no tuning)...")
        model.fit(X_train, y_train)
        return model

# ==============================================
# MODEL 1: CLINICAL MODEL (14 Features)
# ==============================================
print("\n🚑 TRAINING CLINICAL MODELS (RF, XGB, MLP)...")
DATA_PATH_CLINICAL = "heart_disease_uci.csv"

if not os.path.exists(DATA_PATH_CLINICAL):
    raise FileNotFoundError("❌ Clinical dataset not found!")

# Load with inferred header
df1 = pd.read_csv(DATA_PATH_CLINICAL, header=None)
column_names = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak",
    "slope", "ca", "thal", "num"
]
df1.columns = column_names

# Replace '?' with NaN and convert
df1 = df1.replace('?', np.nan)
df1 = df1.apply(pd.to_numeric, errors='coerce')

# Profile data
profile_data(df1, "Clinical (UCI Heart Disease)")

# Handle missing values
df1 = df1.fillna(df1.mean(numeric_only=True))

X1 = df1.drop("num", axis=1)
y1 = (df1["num"] > 0).astype(int)

X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, y1, test_size=0.2, random_state=42)

scaler1 = StandardScaler()
X_train1_scaled = scaler1.fit_transform(X_train1)
X_test1_scaled = scaler1.transform(X_test1)

# Models & Grids
models1 = {
    "Random Forest": (
        RandomForestClassifier(random_state=42),
        {
            'n_estimators': [100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5]
        }
    ),
    "XGBoost": (
        xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'),
        {
            'n_estimators': [100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7]
        }
    ),
    "Neural Network (MLP)": (
        MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42, alpha=0.01),
        None # Skip tuning for speed/simplicity or add if needed
    )
}

results1 = []
trained_models1 = {}

for name, (model, grid) in models1.items():
    best_model = tune_and_train(name, model, grid, X_train1_scaled, y_train1)
    metrics, _ = evaluate_model(best_model, X_test1_scaled, y_test1, name)
    results1.append(metrics)
    trained_models1[name] = best_model

# Save - Ensuring filenames match app.py expectations
# app.py expects: heart_rf_clinical.pkl (for RF)
# We will save others as well for reference
save_map_1 = {
    "Random Forest": "heart_rf_clinical.pkl",
    "XGBoost": "heart_xgb_clinical.pkl",
    "Neural Network (MLP)": "heart_mlp_clinical.pkl"
}

for name, model in trained_models1.items():
    fname = save_map_1.get(name, f"heart_{name.lower().replace(' ', '_')}_clinical.pkl")
    joblib.dump(model, fname)
    print(f"  💾 Saved {name} to {fname}")

joblib.dump(scaler1, "heart_scaler_clinical.pkl")
X1.head(1).to_csv("heart_user_template_clinical.csv", index=False)

# Display results
print("\n✅ Clinical Model Performance Comparison:")
print(tabulate(results1, headers="keys", tablefmt="github", floatfmt=".4f"))


# ==============================================
# MODEL 2: LIFESTYLE MODEL (General / Self-Report)
# ==============================================
print("\n🏃 TRAINING LIFESTYLE MODELS (RF, XGB, MLP)...")
DATA_PATH_LIFESTYLE = "cardio_train.csv"

if not os.path.exists(DATA_PATH_LIFESTYLE):
    raise FileNotFoundError("❌ Lifestyle dataset 'cardio_train.csv' not found!")

df2 = pd.read_csv(DATA_PATH_LIFESTYLE, sep=';')

if 'id' in df2.columns:
    df2 = df2.drop(columns=['id'])

if 'cardio' not in df2.columns:
    raise ValueError("❌ Target column 'cardio' missing!")

# Profile data
profile_data(df2, "Lifestyle (Cardio)")

X2 = df2.drop("cardio", axis=1)
y2 = df2["cardio"]

# Handle missing
df2 = df2.fillna(df2.mean(numeric_only=True))

X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, test_size=0.2, random_state=42)

scaler2 = StandardScaler()
X_train2_scaled = scaler2.fit_transform(X_train2)
X_test2_scaled = scaler2.transform(X_test2)

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
        None
    )
}

results2 = []
trained_models2 = {}

for name, (model, grid) in models2.items():
    best_model = tune_and_train(name, model, grid, X_train2_scaled, y_train2)
    metrics, _ = evaluate_model(best_model, X_test2_scaled, y_test2, name)
    results2.append(metrics)
    trained_models2[name] = best_model

# Save
save_map_2 = {
    "Random Forest": "heart_rf_lifestyle.pkl",
    "XGBoost": "heart_xgb_lifestyle.pkl",
    "Neural Network (MLP)": "heart_mlp_lifestyle.pkl"
}

for name, model in trained_models2.items():
    fname = save_map_2.get(name, f"heart_{name.lower().replace(' ', '_')}_lifestyle.pkl")
    joblib.dump(model, fname)
    print(f"  💾 Saved {name} to {fname}")

joblib.dump(scaler2, "heart_scaler_lifestyle.pkl")
X2.head(1).to_csv("heart_user_template_lifestyle.csv", index=False)

# Display results
print("\n✅ Lifestyle Model Performance Comparison:")
print(tabulate(results2, headers="keys", tablefmt="github", floatfmt=".4f"))

print("\n✅ All models trained, tuned, and saved successfully!")