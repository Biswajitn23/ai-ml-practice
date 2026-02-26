from flask import Flask, request, jsonify, render_template
import joblib

app = Flask(__name__)

model = joblib.load("rf_model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json["features"]
    prediction = model.predict([data])
    return jsonify({"prediction": int(prediction[0])})

@app.route("/predict_ui", methods=["POST"])
def predict_ui():
    features = request.form["features"]
    features = list(map(float, features.split(",")))
    prediction = model.predict([features])
    return render_template("index.html", prediction=prediction[0])

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)