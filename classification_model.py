import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# -----------------------------
# 1. Load dataset
# -----------------------------
df = pd.read_csv("data/students.csv")

# -----------------------------
# 2. Handle missing values (future-safe)
# -----------------------------
for col in ["Maths", "Science", "English"]:
    df[col] = df[col].fillna(df[col].mean())

# -----------------------------
# 3. Feature engineering
# -----------------------------
df["Average"] = df[["Maths", "Science", "English"]].mean(axis=1)

# -----------------------------
# 4. Create classification target
# Pass = 1, Fail = 0
# Threshold intentionally set to 75
# -----------------------------
df["Result"] = df["Average"].apply(lambda x: 1 if x >= 75 else 0)

# -----------------------------
# 5. Sanity check (VERY IMPORTANT)
# -----------------------------
print("\nClass distribution:")
print(df["Result"].value_counts())

# -----------------------------
# 6. Define features (X) and target (y)
# -----------------------------
X = df[["Maths", "Science", "English"]]
y = df["Result"]

# -----------------------------
# 7. Train-test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# 8. Train Logistic Regression model
# -----------------------------
model = LogisticRegression()
model.fit(X_train, y_train)

# -----------------------------
# 9. Make predictions
# -----------------------------
y_pred = model.predict(X_test)

# -----------------------------
# 10. Evaluate model
# -----------------------------
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy * 100:.2f}%")

cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

# -----------------------------
# 11. Interpret results
# -----------------------------
print("\nPrediction Results:")
results = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred
})
print(results)
