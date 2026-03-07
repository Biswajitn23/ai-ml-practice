from flask import Flask, request, jsonify, render_template
import joblib

app = Flask(__name__)

# Load trained model
model = joblib.load("rf_model.pkl")


# -------------------------
# Home Route (Web UI)
# -------------------------
@app.route("/")
def home():
    return render_template("index.html")


# -------------------------
# API Endpoint
# -------------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json["features"]

        # Validate input length
        if len(data) != 30:
            return jsonify({"error": "Exactly 30 features required"})

        prediction = model.predict([data])[0]
        probability = model.predict_proba([data])[0][prediction]

        label = (
            "Benign (Non-Cancerous)"
            if prediction == 1
            else "Malignant (Cancerous)"
        )

        return jsonify({
            "prediction": label,
            "class": int(prediction),
            "confidence": round(float(probability), 4)
        })

    except Exception as e:
        return jsonify({"error": str(e)})


# -------------------------
# UI Form Prediction
# -------------------------
@app.route("/predict_ui", methods=["POST"])
def predict_ui():
    try:
        features = []

        for i in range(30):
            value = float(request.form[f"f{i}"])
            features.append(value)

        prediction = model.predict([features])[0]
        probability = model.predict_proba([features])[0][prediction]

        label = (
            "Benign (Non-Cancerous)"
            if prediction == 1
            else "Malignant (Cancerous)"
        )

        confidence = round(float(probability) * 100, 2)

        result = f"{label} (Confidence: {confidence}%)"

        return render_template("index.html", prediction=result)

    except Exception as e:
        return render_template(
            "index.html",
            prediction=f"Error: {str(e)}"
        )


# -------------------------
# Run Flask Server
# -------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)