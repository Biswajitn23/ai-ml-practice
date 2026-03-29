from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

data = load_breast_cancer()

X = data.data
y = data.target

model = RandomForestClassifier()

params = {
    "n_estimators": [50, 100],
    "max_depth": [3, 5, None]
}

grid = GridSearchCV(model, params, cv=3)
grid.fit(X, y)

print("Best Parameters:", grid.best_params_)