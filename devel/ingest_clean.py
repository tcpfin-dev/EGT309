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

print("Step 1 — Dropping duplicates...")
before = len(df)
df = df.drop_duplicates()
print(f"  Removed {before - len(df)} duplicate rows ({len(df)} remain).")

print("Step 2 — Flagging physically impossible values as NaN...")
df.loc[df["Temperature"] > 50, "Temperature"] = np.nan
df.loc[df["Humidity"] < 0, "Humidity"] = np.nan
df.loc[df["Humidity"] > 100, "Humidity"] = np.nan
df.loc[df["CO2_InfraredSensor"] < 0, "CO2_InfraredSensor"] = np.nan

print("Step 3 — Imputing missing numeric values (session-level median fallback to global median)...")
numeric_to_impute = [
    "Temperature", "Humidity", "CO2_InfraredSensor",
    "MetalOxideSensor_Unit1", "MetalOxideSensor_Unit2",
    "MetalOxideSensor_Unit3", "MetalOxideSensor_Unit4",
    "CO_GasSensor",
]
for col in numeric_to_impute:
    if col in df.columns:
        session_median = df.groupby("Session ID")[col].transform("median")
        df[col] = df[col].fillna(session_median).fillna(df[col].median())

print("Step 3b — Imputing Ambient Light Level (mode)...")
mode_val = df["Ambient Light Level"].mode()
if len(mode_val) > 0:
    df["Ambient Light Level"] = df["Ambient Light Level"].fillna(mode_val.iloc[0])

print(f"  Missing values after imputation: {df.isnull().sum().sum()}")

print("Step 4 — Standardising categorical columns...")
df["HVAC Operation Mode"] = (
    df["HVAC Operation Mode"]
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
df["Activity Level"] = df["Activity Level"].str.strip().map(act_map)
print(f"  HVAC variants: {sorted(df['HVAC Operation Mode'].unique())}")
print(f"  Activity Level variants: {sorted(df['Activity Level'].unique())}")

print(f"\nCleaned dataset: {df.shape[0]} rows, {df.shape[1]} columns")
df.to_csv(output_path, index=False)
print(f"Saved to: {output_path}")
