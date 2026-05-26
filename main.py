import os
import sqlite3
import pandas as pd

# Tell pandas to display every single column without truncation
pd.set_option('display.max_columns', None)

# (Optional but recommended) Make the terminal wide enough so columns don't wrap to the next line
pd.set_option('display.width', 1000)


# Define the relative path to your database NEED TO TAKE FROM DOCKER !!!!
db_path = os.path.join("data", "gas_monitoring.db")

# 1. Establish connection to the SQLite database
print(f"Connecting to database at: {db_path}")
conn = sqlite3.connect(db_path)

# 2. Read the entire gas_monitoring table into a pandas DataFrame
df = pd.read_sql_query("SELECT * FROM gas_monitoring;", conn)

# Always close the connection when done
conn.close()

# 3. Print verification stats
print("\n--- Raw Ingestion Stats ---")
print(f"Total rows loaded: {len(df)}")
print("\nDataFrame columns and data types:")
print(df.info())

print("\nFirst 5 rows of data:")
print(df.head(10))

