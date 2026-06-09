import os
import pickle
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report

# ==========================================
# PART 1: Define Paths & Load Preprocessing Artifacts
# ==========================================
# Resolve paths relative to the script location
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
models_dir = os.path.join(project_root, 'Saved_models')
csv_path = os.path.join(project_root, 'gas_monitoring_cleaned.csv')

print("Loading preprocessing pipeline artifacts...")
scaler_path = os.path.join(models_dir, 'scaler.pkl')
le_path = os.path.join(models_dir, 'label_encoder.pkl')

if not os.path.exists(scaler_path) or not os.path.exists(le_path):
    raise FileNotFoundError(
        "Required preprocessing artifacts (scaler.pkl, label_encoder.pkl) "
        f"not found in {models_dir}. Please run the training script first."
    )

with open(scaler_path, 'rb') as f:
    scaler = pickle.load(f)

with open(le_path, 'rb') as f:
    label_encoder = pickle.load(f)

print("Preprocessing pipeline loaded successfully.")

# ==========================================
# PART 2: Load Saved Models
# ==========================================
print("\nLoading saved models...")
xgb_path = os.path.join(models_dir, 'xgb_model.json')
rf_path = os.path.join(models_dir, 'rf_model.pkl')
lr_path = os.path.join(models_dir, 'lr_model.pkl')

# Load XGBoost
if not os.path.exists(xgb_path):
    raise FileNotFoundError(f"XGBoost model not found at {xgb_path}")
xgb_model = xgb.XGBClassifier()
xgb_model.load_model(xgb_path)
print("-> Loaded XGBoost model.")

# Load Random Forest
if not os.path.exists(rf_path):
    raise FileNotFoundError(f"Random Forest model not found at {rf_path}")
with open(rf_path, 'rb') as f:
    rf_model = pickle.load(f)
print("-> Loaded Random Forest model.")

# Load Logistic Regression
if not os.path.exists(lr_path):
    raise FileNotFoundError(f"Logistic Regression model not found at {lr_path}")
with open(lr_path, 'rb') as f:
    lr_model = pickle.load(f)
print("-> Loaded Logistic Regression model.")

# ==========================================
# PART 3: Data Loading & Preprocessing
# ==========================================
print(f"\nLoading dataset for evaluation from: {csv_path}")
try:
    df = pd.read_csv(csv_path)
except FileNotFoundError:
    raise FileNotFoundError(
        f"Could not find '{csv_path}'. Please ensure the cleaned dataset exists "
        f"in the project root directory."
    )

# Ensure target column is present
if 'Activity Level' not in df.columns:
    raise KeyError("Target column 'Activity Level' not found in dataset columns.")

# Separate features and target
X = df.drop(columns=['Activity Level'])
y = df['Activity Level']

# Encode the target classes using the loaded LabelEncoder
y_encoded = label_encoder.transform(y)

# Separate categorical features
categorical_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

# One-hot encode the categorical features
X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
X_encoded = X_encoded.astype(float)

# Align columns with scaler's expected feature dimensions
expected_features = scaler.feature_names_in_ if hasattr(scaler, 'feature_names_in_') else None
if expected_features is not None:
    # Reindex to ensure same column order and fill missing dummy columns with 0
    X_encoded = X_encoded.reindex(columns=expected_features, fill_value=0.0)

# Scale features using the loaded StandardScaler
X_scaled = scaler.transform(X_encoded)
X_scaled_df = pd.DataFrame(X_scaled, columns=X_encoded.columns)

# Train/Test Split (using same split parameters as training to isolate test set)
_, X_test, _, y_test = train_test_split(
    X_scaled_df, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)
print(f"Isolated evaluation test split: {X_test.shape[0]} samples")

# ==========================================
# PART 4: Model Evaluation
# ==========================================
print("\nEvaluating models on test split...")

# Predict and evaluate XGBoost
y_pred_xgb = xgb_model.predict(X_test)
f1_xgb = f1_score(y_test, y_pred_xgb, average='macro')

# Predict and evaluate Random Forest
y_pred_rf = rf_model.predict(X_test)
f1_rf = f1_score(y_test, y_pred_rf, average='macro')

# Predict and evaluate Logistic Regression
y_pred_lr = lr_model.predict(X_test)
f1_lr = f1_score(y_test, y_pred_lr, average='macro')

# Display F1 Macro Comparison
print("\n=======================================================")
print("             SAVED MODEL EVALUATION RESULTS            ")
print("=======================================================")
print(f"XGBoost Saved Model F1 Macro Score:           {f1_xgb:.4f}")
print(f"Random Forest Saved Model F1 Macro Score:     {f1_rf:.4f}")
print(f"Logistic Regression Saved Model F1 Macro Score: {f1_lr:.4f}")
print("=======================================================\n")

# Display full classification reports
print("Classification Report - XGBoost:")
print(classification_report(y_test, y_pred_xgb, target_names=label_encoder.classes_))

print("\nClassification Report - Random Forest:")
print(classification_report(y_test, y_pred_rf, target_names=label_encoder.classes_))

print("\nClassification Report - Logistic Regression:")
print(classification_report(y_test, y_pred_lr, target_names=label_encoder.classes_))
