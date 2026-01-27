import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

df = pd.read_csv("data/students.csv")

for col in ["Maths", "Science", "English"]:
    df[col] = df[col].fillna(df[col].mean())

df["Average"] = df[["Maths", "Science", "English"]].mean(axis=1)
df["Result"] = df["Average"].apply(lambda x: 1 if x >= 75 else 0)

X = df[["Maths", "Science", "English"]]
y = df["Result"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

log_model = LogisticRegression()
log_model.fit(X_train, y_train)
log_pred = log_model.predict(X_test)
log_acc = accuracy_score(y_test, log_pred)

print(f"\nLogistic Regression Accuracy: {log_acc*100:.2f}%")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

best_k = 0
best_acc = 0

print("\nKNN Accuracy for different K values:")

max_k = len(X_train_scaled)

for k in range(1, max_k + 1):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    pred = knn.predict(X_test_scaled)
    acc = accuracy_score(y_test, pred)

    print(f"K = {k} → Accuracy = {acc*100:.2f}%")

    if acc > best_acc:
        best_acc = acc
        best_k = k

print(f"\nBest K: {best_k} with Accuracy: {best_acc*100:.2f}%")
