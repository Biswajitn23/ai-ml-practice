import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/students.csv")

print(df.info())
print("\nMissing Values:\n", df.isnull().sum())

df.hist(figsize=(8,6))
plt.suptitle("Feature Distributions")
plt.show()