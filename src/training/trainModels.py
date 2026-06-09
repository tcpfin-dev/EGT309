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

# ==========================================
# PART 3: Data Splitting
# ==========================================
# Overall Train/Test Split (80% Train, 20% Test)
X_train_initial, X_test, y_train_initial, y_test = train_test_split(
    X_scaled_df, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)
print(f"Overall Training set shape: {X_train_initial.shape}, Testing set shape: {X_test.shape}")

# Further split the initial training set for XGBoost's internal validation (e.g., 90% train, 10% val from the initial 80% train)
X_train_xgb, X_val_xgb, y_train_xgb, y_val_xgb = train_test_split(
    X_train_initial, y_train_initial, test_size=0.125, random_state=42, stratify=y_train_initial
) # 0.125 * 0.8 = 0.1 -> roughly 10% of total data for validation
print(f"XGBoost Specific Training shape: {X_train_xgb.shape}, XGBoost Validation shape: {X_val_xgb.shape}")

# Use X_train_initial for Random Forest and Logistic Regression as they don't need a separate validation set for early stopping
X_train_rf_lr = X_train_initial
y_train_rf_lr = y_train_initial

# ==========================================
# PART 4: Train XGBoost Model
# ==========================================
print("\n--- Training XGBoost Model ---")

# Convert data to DMatrix format for xgb.train
dtrain = xgb.DMatrix(X_train_xgb, label=y_train_xgb)
dval = xgb.DMatrix(X_val_xgb, label=y_val_xgb)

# Define parameters for the booster
params = {
    'objective': 'multi:softmax', # For multi-class classification
    'num_class': len(label_encoder.classes_), # Number of classes
    'eval_metric': 'mlogloss',
    'eta': 0.05, # learning_rate equivalent
    'max_depth': 7,
    'seed': 42,
    'tree_method': 'hist' # Changed from 'gpu_hist' as it's not a valid input, 'hist' is GPU accelerated by default if available
}

# Define callbacks for early stopping and verbose output
callbacks = [
    xgb.callback.EarlyStopping(rounds=50) # verbose=True here prints output every round
]

# Train the model using xgb.train
xgb_bst = xgb.train(
    params,
    dtrain,
    num_boost_round=1000, # A high number, early stopping will cut it short
    evals=[(dtrain, 'train'), (dval, 'validation')], # Monitor both train and validation
    callbacks=callbacks
)

# After training, wrap the booster in an XGBClassifier for consistent API with other models
# This allows using .predict and .save_model easily with the sklearn-like interface.
xgb_model = xgb.XGBClassifier(
    objective=params['objective'],
    num_class=params['num_class'],
    eval_metric=params['eval_metric'],
    learning_rate=params['eta'],
    max_depth=params['max_depth'],
    random_state=params['seed'],
    n_estimators=xgb_bst.best_iteration + 1 if xgb_bst.best_iteration is not None else xgb_bst.num_boost_round,
    device=device
)
# Set the internal booster directly. This is a common way to use xgb.train output with XGBClassifier.
xgb_model._Booster = xgb_bst

# ==========================================
# PART 5: Train Random Forest Model
# ==========================================
print("\n--- Training Random Forest Model ---")
# Using best parameters found from hyperparameter tuning for an 'advanced' model
rf_model = RandomForestClassifier(
    class_weight='balanced',
    max_depth=50,
    min_samples_leaf=4,
    n_estimators=1000,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train_rf_lr, y_train_rf_lr)

# ==========================================
# PART 6: Train Logistic Regression Model
# ==========================================
print("\n--- Training Logistic Regression Model ---")
lr_model = LogisticRegression(
    class_weight='balanced',
    random_state=42,
    max_iter=1000
)
lr_model.fit(X_train_rf_lr, y_train_rf_lr)


# ==========================================
# PART 7: Save Trained Models
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

# ==========================================
# PART 8: Display Training Scores
# ==========================================
print("\n--- Training F1 Macro Scores (on Training Data) ---")

# Predict and evaluate XGBoost on its specific training data
y_pred_train_xgb = xgb_model.predict(X_train_xgb)
f1_train_xgb = f1_score(y_train_xgb, y_pred_train_xgb, average='macro')
print(f"XGBoost Training F1 Macro Score:          {f1_train_xgb:.4f}")

# Predict and evaluate Random Forest on its training data
y_pred_train_rf = rf_model.predict(X_train_rf_lr)
f1_train_rf = f1_score(y_train_rf_lr, y_pred_train_rf, average='macro')
print(f"Random Forest Training F1 Macro Score:    {f1_train_rf:.4f}")

# Predict and evaluate Logistic Regression on its training data
y_pred_train_lr = lr_model.predict(X_train_rf_lr)
f1_train_lr = f1_score(y_train_rf_lr, y_pred_train_lr, average='macro')
print(f"Logistic Regression Training F1 Macro Score: {f1_train_lr:.4f}")