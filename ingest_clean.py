import os
import sqlite3

import numpy as np
import pandas as pd

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

db_path = os.path.join(project_root, "data", "gas_monitoring.db")
output_path = os.path.join(project_root, "gas_monitoring_cleaned.csv")

print(f"Loading data from: {db_path}")
conn = sqlite3.connect(db_path)
df = pd.read_sql_query("SELECT * FROM gas_monitoring", conn)
conn.close()
print(f"Loaded {len(df)} rows, {len(df.columns)} columns.")

df_clean = df.copy()

print("Step 1 — Dropping duplicates...")
before = len(df_clean)
df_clean = df_clean.drop_duplicates()
print(f"  Removed {before - len(df_clean)} duplicate rows ({len(df_clean)} remain).")

print("Step 2 — Flagging physically impossible values as NaN...")
df_clean.loc[df_clean["Temperature"] > 50, "Temperature"] = np.nan
df_clean.loc[df_clean["Humidity"] < 0, "Humidity"] = np.nan
df_clean.loc[df_clean["Humidity"] > 100, "Humidity"] = np.nan
df_clean.loc[df_clean["CO2_InfraredSensor"] < 0, "CO2_InfraredSensor"] = np.nan

print("Step 3 — Imputing missing numeric values (session-level median)...")
numeric_to_impute = [
    "Temperature", "Humidity", "CO2_InfraredSensor",
    "MetalOxideSensor_Unit2", "CO_GasSensor",
]
for col in numeric_to_impute:
    session_median = df_clean.groupby("Session ID")[col].transform("median")
    df_clean[col] = df_clean[col].fillna(session_median).fillna(df_clean[col].median())

print("Step 3b — Imputing Ambient Light Level (mode)...")
df_clean["Ambient Light Level"] = df_clean["Ambient Light Level"].fillna(
    df_clean["Ambient Light Level"].mode()[0]
)
print(f"  Missing values after imputation: {df_clean.isnull().sum().sum()}")

print("Step 4 — Standardising categorical columns...")
df_clean["HVAC Operation Mode"] = (
    df_clean["HVAC Operation Mode"]
    .str.strip()
    .str.lower()
    .str.replace(r"[\s\-]+", "_", regex=True)
)

act_map = {
    "Low Activity":      "Low Activity",
    "Low_Activity":      "Low Activity",
    "LowActivity":       "Low Activity",
    "Moderate Activity": "Moderate Activity",
    "ModerateActivity":  "Moderate Activity",
    "High Activity":     "High Activity",
    "HighActivity":      "High Activity",
}
df_clean["Activity Level"] = df_clean["Activity Level"].str.strip().map(act_map)
print(f"  HVAC variants: {sorted(df_clean['HVAC Operation Mode'].unique())}")
print(f"  Activity Level variants: {sorted(df_clean['Activity Level'].unique())}")

print("Step 5 — Done.\n")
print(f"Cleaned dataset: {df_clean.shape[0]} rows, {df_clean.shape[1]} columns")
df_clean.to_csv(output_path, index=False)
print("Done.")
