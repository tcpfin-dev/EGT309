import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder

# --- STEP 1: REMOVE DUPLICATE ROWS ---
# Remove identical rows to avoid repeating data
df_clean = df.drop_duplicates()
print("Rows after dropping duplicates:", len(df_clean))


# --- STEP 2: HANDLE OUTLIERS (REPLACE WITH NAN) ---
# If values are impossible or too extreme, turn them into missing values (NaN)
df_clean.loc[df_clean["Temperature"] > 50, "Temperature"] = np.nan
df_clean.loc[df_clean["Humidity"] < 0, "Humidity"] = np.nan
df_clean.loc[df_clean["Humidity"] > 100, "Humidity"] = np.nan
df_clean.loc[df_clean["CO2_InfraredSensor"] < 0, "CO2_InfraredSensor"] = np.nan

# Count how many missing values we just created
print(df_clean[["Temperature", "Humidity", "CO2_InfraredSensor"]].isnull().sum())


# --- STEP 3: FILL IN MISSING VALUES (IMPUTATION) ---
# List of columns with numbers that need missing values filled
numeric_to_impute = ["Temperature", "Humidity", "CO2_InfraredSensor", "CO_GasSensor"]

for col in numeric_to_impute:
    # Find the middle value (median) for each specific Session
    session_median = df_clean.groupby("Session ID")[col].transform("median")
    # Fill missing values with the session median. If that fails, use the overall column median
    df_clean[col] = df_clean[col].fillna(session_median).fillna(
        df_clean[col].median()
    )

# Ambient Light Level is text data — fill missing parts with the most common value (mode)
df_clean["Ambient Light Level"] = df_clean["Ambient Light Level"].fillna(
    df_clean["Ambient Light Level"].mode()
)

# Check if there are any missing values left anywhere
print(df_clean.isnull().sum())


# --- STEP 4: DROP UNNECESSARY COLUMNS ---
# Remove this specific sensor column because it is not needed
df_clean = df_clean.drop(columns=["MetalOxideSensor_Unit2"])
print("Columns remaining:", df_clean.columns.tolist())


# --- STEP 5: CLEAN AND STANDARDIZE TEXT COLUMNS ---
# Fix spacing, lowercase everything, and turn spaces/dashes into underscores
for col in ["HVAC Operation Mode", "Activity Level"]:
    df_clean[col] = (
        df_clean[col]
        .str.strip()
        .str.lower()
        .str.replace(r"[\s\-]+", "_", regex=True)
    )

# Fix specific words that got glued together (e.g., "lowactivity" -> "low_activity")
act_map = {
    "lowactivity": "low_activity",
    "moderateactivity": "moderate_activity",
    "highactivity": "high_activity",
}
df_clean["Activity Level"] = df_clean["Activity Level"].replace(act_map)


# --- STEP 6: VERIFY TEXT CLEANING ---
# Print unique values to make sure the text looks uniform and correct
print("HVAC Operation Mode:", sorted(df_clean["HVAC Operation Mode"].unique()))
print()
print("Activity Level:", sorted(df_clean["Activity Level"].unique()))


# --- STEP 7: DROP ANOTHER UNNECESSARY COLUMN ---
# Remove this electrochemical sensor column as well
df_clean = df_clean.drop(columns=["CO2_ElectroChemicalSensor"])
print("Columns remaining:", df_clean.columns.tolist())


# --- STEP 8: INSPECT CO_GASSENSOR COLUMN ---
# Check the unique values and look at data types
print(
    "CO_GasSensor distinct values:",
    sorted(df_clean["CO_GasSensor"].dropna().unique()),
)

# Convert the whole column to text (string) data type
df_clean["CO_GasSensor"] = df_clean["CO_GasSensor"].astype(str)
print("\nDtype after conversion:", df_clean["CO_GasSensor"].dtype)
print("Value counts:\n", df_clean["CO_GasSensor"].value_counts())


# --- STEP 9: BALANCE THE DATASET USING SMOTE ---
# Make a copy of the dataframe specifically for the SMOTE process
df_smote = df_clean.copy()
encoders = {}

# Convert all text columns into numbers because SMOTE only works with math/numbers
for col in df_smote.select_dtypes("object").columns:
    le = LabelEncoder()
    df_smote[col] = le.fit_transform(df_smote[col].astype(str))
    encoders[col] = le  # Save the encoder so we can convert numbers back to text later

# Split data into features (X) and the target variable we want to balance (y)
X = df_smote.drop(columns=["Activity Level"])
y = df_smote["Activity Level"]

# Run SMOTE to generate synthetic data for minority classes so they are all equal
X_res, y_res = SMOTE(random_state=42).fit_resample(X, y)

# Rebuild our clean dataframe using the newly balanced data arrays
df_clean = pd.DataFrame(X_res, columns=X.columns)
df_clean["Activity Level"] = encoders["Activity Level"].inverse_transform(y_res)

# Decode the other number columns back into their original text values
for col in encoders:
    if col != "Activity Level":
        df_clean[col] = encoders[col].inverse_transform(
            df_clean[col].round().astype(int)
        )

# Print final results to show the before-and-after of balancing the data
print("Before SMOTE:", y.value_counts().to_dict())
print("After SMOTE: ", pd.Series(y_res).value_counts().to_dict())
print("df_clean shape after SMOTE:", df_clean.shape)