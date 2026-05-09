import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv("data/students.csv")

for col in ["Maths", "Science", "English"]:
    df[col] = df[col].fillna(df[col].mean())

df["Average"] = df[["Maths", "Science", "English"]].mean(axis=1)
df["Result"] = df["Average"].apply(lambda x: 1 if x >= 75 else 0)

X = df[["Maths", "Science", "English"]]
y = df["Result"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

pred = model.predict(X_test)

accuracy = accuracy_score(y_test, pred)

print("Random Forest Accuracy:", accuracy)