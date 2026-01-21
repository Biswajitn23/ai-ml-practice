import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix

df = pd.read_csv("data/students.csv")

for col in ["Maths", "Science", "English"]:
    df[col] = df[col].fillna(df[col].mean())

df["Average"] = df[["Maths", "Science", "English"]].mean(axis=1)

df["Result"] = df["Average"].apply(lambda x: 1 if x >= 75 else 0)

print("\nClass distribution:")
print(df["Result"].value_counts())

X = df[["Maths", "Science", "English"]]
y = df["Result"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train_scaled, y_train)

y_pred = knn.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
print(f"\nKNN Accuracy: {accuracy * 100:.2f}%")

cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)
