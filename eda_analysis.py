import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/students.csv")

print("\nDataset Preview:")
print(df.head())

print("\nDataset Info:")
print(df.info())

print("\nStatistical Summary:")
print(df.describe())

print("\nMissing Values Count:")
print(df.isnull().sum())

df["Maths"] = df["Maths"].fillna(df["Maths"].mean())
df["Science"] = df["Science"].fillna(df["Science"].mean())
df["English"] = df["English"].fillna(df["English"].mean())

print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

df["Average"] = df[["Maths", "Science", "English"]].mean(axis=1)
print("\nCleaned Data with Average:")
print(df)

plt.boxplot(df[["Maths", "Science", "English"]])
plt.xticks([1, 2, 3], ["Maths", "Science", "English"])
plt.title("Subject-wise Score Distribution")
plt.ylabel("Marks")
plt.show()

