import pandas as pd
import sqlite3
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

# Step 1: Connect to the SQLite database and read the data
conn = sqlite3.connect('heart_disease.db')
df = pd.read_sql_query("SELECT * FROM heart_disease", conn)
conn.close()

# Step 2: Data Cleaning and Preprocessing
# Fill missing values (if any)
for column in df.columns:
    if df[column].dtype == 'object':
        df[column].fillna(df[column].mode()[0], inplace=True)
    else:
        df[column].fillna(df[column].mean(), inplace=True)

# Encode categorical variables
label_encoders = {}
for column in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# Standardize numerical variables
scaler = StandardScaler()
df[df.select_dtypes(include=['float64', 'int64']).columns] = scaler.fit_transform(df.select_dtypes(include=['float64', 'int64']))

# Define features and target variable
X = df.drop('target_column', axis=1)  # Replace 'target_column' with the actual name of the target column
y = df['target_column']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print("Preprocessing complete. Data ready for machine learning.")
