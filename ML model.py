import os
import pickle
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from imblearn.over_sampling import SMOTE

# ==========================================
# PART 1: Imports and GPU Configuration
# ==========================================
print("Detecting GPU support for XGBoost...")
device = 'cpu'
try:
    # Test if GPU is available in xgboost
    xgb.XGBClassifier(device='cuda').fit(np.array([[1]]), np.array([0]))
    device = 'cuda'
    print("-> GPU (cuda) is available and will be used for XGBoost training.")
except Exception:
    print("-> GPU not available or not supported by XGBoost. Falling back to CPU.")

# ==========================================
# PART 2: Data Loading and Preprocessing
# ==========================================
csv_path = 'gas_monitoring_cleaned.csv'
print(f"\nLoading dataset from: {csv_path}")
try:
    df = pd.read_csv(csv_path)
except FileNotFoundError:
    raise FileNotFoundError(
        f"Could not find '{csv_path}'. Please ensure the cleaned dataset exists "
        f"in the current working directory."
    )

# Ensure target column is present
if 'Activity Level' not in df.columns:
    raise KeyError("Target column 'Activity Level' not found in dataset columns.")

# Separate features and target
X = df.drop(columns=['Activity Level'])
y = df['Activity Level']

# Encode the target classes into integers
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
print(f"Target classes mapped: {dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))}")

# Separate categorical features
categorical_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

# One-hot encode the categorical features
X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
X_encoded = X_encoded.astype(float)

# Scale features (critical for Logistic Regression)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_encoded)
X_scaled_df = pd.DataFrame(X_scaled, columns=X_encoded.columns)

# ------------------------------------------
# SMOTE: Oversample minority classes
# ------------------------------------------
print("\n--- Applying SMOTE ---")

# ---- Before SMOTE ----
before_total = len(y_encoded)
print(f"Before SMOTE: {before_total} rows total")
before_counts = pd.Series(y_encoded).value_counts().sort_index()
for cls_idx, count in before_counts.items():
    cls_name = label_encoder.inverse_transform([cls_idx])[0]
    pct = count / before_total * 100
    print(f"  Class '{cls_name}' (encoded {cls_idx}): {count} rows ({pct:.2f}%)")

# ---- Apply SMOTE ----
smote = SMOTE(random_state=42)
X_scaled_arr, y_encoded = smote.fit_resample(X_scaled_df, y_encoded)
X_scaled_df = pd.DataFrame(X_scaled_arr, columns=X_encoded.columns)

# ---- After SMOTE ----
after_total = len(y_encoded)
print(f"\nAfter SMOTE:  {after_total} rows total")
after_counts = pd.Series(y_encoded).value_counts().sort_index()
for cls_idx, count in after_counts.items():
    cls_name = label_encoder.inverse_transform([cls_idx])[0]
    pct = count / after_total * 100
    print(f"  Class '{cls_name}' (encoded {cls_idx}): {count} rows ({pct:.2f}%)")

# Train/Test Split (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled_df, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)
print(f"Training shape: {X_train.shape}, Testing shape: {X_test.shape}")

# ==========================================
# PART 3: Train XGBoost Model
# ==========================================
print("\n--- Training XGBoost Model ---")
xgb_model = xgb.XGBClassifier(
    device=device,
    random_state=42,
    eval_metric='mlogloss'
)
xgb_model.fit(X_train, y_train)

# ==========================================
# PART 4: Train Random Forest Model
# ==========================================
print("\n--- Training Random Forest Model ---")
rf_model = RandomForestClassifier(
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)

# ==========================================
# PART 5: Train Logistic Regression Model
# ==========================================
print("\n--- Training Logistic Regression Model ---")
lr_model = LogisticRegression(
    class_weight='balanced',
    random_state=42,
    max_iter=1000
)
lr_model.fit(X_train, y_train)


# ==========================================
# PART 6: Save Trained Models
# ==========================================
print("\n--- Saving Trained Models & Preprocessing Pipeline ---")
save_dir = 'Saved_models'
os.makedirs(save_dir, exist_ok=True)

# Save XGBoost
xgb_save_path = os.path.join(save_dir, 'xgb_model.json')
xgb_model.save_model(xgb_save_path)
print(f"Saved XGBoost model to: {xgb_save_path}")

# Save Random Forest
rf_save_path = os.path.join(save_dir, 'rf_model.pkl')
with open(rf_save_path, 'wb') as f:
    pickle.dump(rf_model, f)
print(f"Saved Random Forest model to: {rf_save_path}")

# Save Logistic Regression
lr_save_path = os.path.join(save_dir, 'lr_model.pkl')
with open(lr_save_path, 'wb') as f:
    pickle.dump(lr_model, f)
print(f"Saved Logistic Regression model to: {lr_save_path}")

# Save Scaler
scaler_save_path = os.path.join(save_dir, 'scaler.pkl')
with open(scaler_save_path, 'wb') as f:
    pickle.dump(scaler, f)
print(f"Saved StandardScaler to: {scaler_save_path}")

# Save Label Encoder
le_save_path = os.path.join(save_dir, 'label_encoder.pkl')
with open(le_save_path, 'wb') as f:
    pickle.dump(label_encoder, f)
print(f"Saved LabelEncoder to: {le_save_path}")