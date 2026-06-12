import os
import sqlite3
import pandas as pd

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))

db_path = os.path.join(project_root, "data", "gas_monitoring.db")
output_path = os.path.join(project_root, "gas_monitoring_cleaned.csv")

conn = sqlite3.connect(db_path)
df = pd.read_sql_query("SELECT * FROM gas_monitoring", conn)
conn.close()

# Execute DataCleaning.py with df in scope
cleaning_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "DataCleaning.py")
with open(cleaning_path) as f:
    exec(f.read())

# DataCleaning.py normalizes labels to snake_case (low_activity), but the saved
# label_encoder.pkl expects Title Case ("Low Activity"). Without this remap,
# evaluate_savedModel.py would fail on unseen-label errors during transform().
label_map = {
    "low_activity": "Low Activity",
    "moderate_activity": "Moderate Activity",
    "high_activity": "High Activity",
}
df_clean["Activity Level"] = df_clean["Activity Level"].map(label_map).fillna(df_clean["Activity Level"])

df_clean.to_csv(output_path, index=False)
print(f"Cleaned data saved to: {output_path}")
