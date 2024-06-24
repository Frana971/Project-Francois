pip install pandas openpyxl sqlite3

import pandas as pd
import sqlite3

# Step 1: Load the Excel file into a DataFrame
excel_file_path = 'C:\Users\franc\Downloads/heart.csv' 
df = pd.read_excel(excel_file_path)
print("Loaded Excel file:")
print(df.head())

# Step 2: Create a connection to a new SQLite database
conn = sqlite3.connect('heart_disease.db')
cursor = conn.cursor()

# Step 3: Convert the DataFrame to a SQL table
df.to_sql('heart_disease', conn, if_exists='replace', index=False)

# Verify by listing all tables in the database
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cursor.fetchall()
print("Tables in the database:")
print(tables)

# Step 4: Run a query to fetch the first few rows of the table
cursor.execute("SELECT * FROM heart_disease LIMIT 5;")
rows = cursor.fetchall()
print("First few rows of the 'heart_disease' table:")
for row in rows:
    print(row)

# Close the connection
conn.close()


python upload_data_to_sqlite.py
