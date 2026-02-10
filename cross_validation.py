import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

df = pd.read_csv("data/students.csv")

for col in ["Maths", "Science", "English"]:
    df[col] = df[col].fillna(df[col].mean())

df["Average"] = df[["Maths", "Science", "English"]].mean(axis=1)
df["Result"] = df["Average"].apply(lambda x: 1 if x >= 75 else 0)

X = df[["Maths", "Science", "English"]]
y = df["Result"]

log_model = LogisticRegression()
log_scores = cross_val_score(log_model, X, y, cv=2)

knn_pipeline = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=2))
knn_scores = cross_val_score(knn_pipeline, X, y, cv=2)

print("Logistic Regression CV Accuracy:", log_scores)
print("Logistic Regression Average Accuracy:", log_scores.mean())

print("\nKNN CV Accuracy:", knn_scores)
print("KNN Average Accuracy:", knn_scores.mean())