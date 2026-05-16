import joblib

model = joblib.load("student_rf_model.pkl")

sample_data = [[85, 90, 88]]

prediction = model.predict(sample_data)

print("Prediction:", prediction)