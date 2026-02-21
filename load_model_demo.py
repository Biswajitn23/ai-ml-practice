import joblib
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()
X = data.data

model = joblib.load("rf_model.pkl")

prediction = model.predict([X[0]])

print("Prediction:", prediction)