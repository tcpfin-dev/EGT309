import numpy as np
import pandas as pd

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
    df_clean["Ambient Light Level"].mode()[0]
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

# The regex above replaces spaces/dashes but misses labels that were already
# glued together (e.g. "LowActivity" became "lowactivity" not "low_activity").
# This explicit map catches those edge cases.
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


