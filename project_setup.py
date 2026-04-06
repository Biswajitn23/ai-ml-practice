import pandas as pd

df = pd.read_csv("data/students.csv")

print("Dataset Shape:", df.shape)
print("\nColumns:", df.columns)
print("\nFirst 5 Rows:\n", df.head())