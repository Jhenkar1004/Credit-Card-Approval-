import pandas as pd

# Load dataset
df = pd.read_csv("data/loan_data.csv")

# Display first rows
print("First 5 rows:")
print(df.head())

# Dataset information
print("\nDataset Shape:", df.shape)

print("\nColumns:")
print(df.columns)

print("\nMissing Values:")
print(df.isnull().sum())