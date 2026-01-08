import pandas as pd
import matplotlib.pyplot as plt

#Load the dataset
df = pd.read_csv('data/students.csv')

#Display datasets
print("Student Data:\n")
print(df)

#Calculate average marks
df["Average"] = df[["Maths", "Science", "English"]].mean(axis=1)

print("\nAverage Marks:\n")
print(df[["Name", "Average"]])

#Find Topper
topper = df.loc[df["Average"].idxmax()]
print(f"\nTopper: {topper['Name']} with average {topper['Average']:.2f}")

#Plotting average marks
plt.bar(df["Name"], df["Average"])
plt.xlabel("Students")
plt.ylabel("Average Marks")
plt.title("Student Performance Analysis")
plt.show()