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

#remove dupes from the data and store the cleaned data as "cleaned_gas_data.csv"
duplicates = df[df.duplicated(keep=False)]

if not duplicates.empty:
    print(f"\nFound {len(duplicates)} duplicate rows. Removing...")
    df_cleaned = df.drop_duplicates()
    
    print(f"Original row count: {len(df)}")
    print(f"Cleaned row count:  {len(df_cleaned)}")
    print(f"Removed:            {len(duplicates)}")
    
    # Save cleaned data
    df_cleaned.to_csv("cleaned_gas_data.csv", index=False)
    print("Cleaned data saved to cleaned_gas_data.csv")
else:
    print("\nNo duplicates found.")
    df.to_csv("cleaned_gas_data.csv", index=False)
    print("Saved original data to cleaned_gas_data.csv")

#