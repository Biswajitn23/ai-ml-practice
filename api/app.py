from flask import Flask, request, jsonify, render_template
import joblib

app = Flask(__name__)

model = joblib.load("rf_model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json["features"]
        prediction = model.predict([data])
        return jsonify({"prediction": int(prediction[0])})
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route("/predict_ui", methods=["POST"])
def predict_ui():
    try:
        features = request.form["features"]
        features = [float(x.strip()) for x in features.split(",")]

        if len(features) != 30:
            return render_template("index.html", prediction="Please enter exactly 30 values")

        prediction = model.predict([features])
        return render_template("index.html", prediction=prediction[0])

    except Exception as e:
        return render_template("index.html", prediction=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)