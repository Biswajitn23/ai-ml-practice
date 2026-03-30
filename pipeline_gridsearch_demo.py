from sklearn.datasets import load_breast_cancer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

data = load_breast_cancer()

X = data.data
y = data.target

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", RandomForestClassifier())
])

params = {
    "model__n_estimators": [50, 100],
    "model__max_depth": [3, 5, None]
}

grid = GridSearchCV(pipeline, params, cv=3)
grid.fit(X, y)

print("Best Parameters:", grid.best_params_)