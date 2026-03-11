from flask import Flask, request, jsonify, render_template
from flasgger import Swagger
import joblib

app = Flask(__name__)
swagger = Swagger(app)

# Load trained model
model = joblib.load("rf_model.pkl")


# -------------------------
# Home Page
# -------------------------
@app.route("/")
def home():
    return render_template("index.html", input_features=[])


# -------------------------
# API Prediction Endpoint
# -------------------------
@app.route("/predict", methods=["POST"])
def predict():
    """
    Predict Breast Cancer Class
    ---
    parameters:
      - name: features
        in: body
        required: true
        schema:
          type: object
          properties:
            features:
              type: array
              items:
                type: number
              example: [17.99,10.38,122.8,1001]
    responses:
      200:
        description: Prediction result
    """

    try:
        data = request.json["features"]

        if len(data) != 30:
            return jsonify({"error": "Exactly 30 features required"})

        prediction = model.predict([data])[0]
        probability = model.predict_proba([data])[0][prediction]

        label = "Benign" if prediction == 1 else "Malignant"

        return jsonify({
            "prediction": label,
            "class": int(prediction),
            "confidence": round(float(probability), 4)
        })

    except Exception as e:
        return jsonify({"error": str(e)})


# -------------------------
# UI Prediction Route
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

        label = "Benign" if prediction == 1 else "Malignant"
        confidence = round(float(probability) * 100, 2)

        return render_template(
            "index.html",
            prediction=label,
            confidence=confidence,
            input_features=features
        )

    except Exception as e:
        return render_template(
            "index.html",
            prediction=f"Error: {str(e)}",
            input_features=[]
        )


# -------------------------
# Feature Importance API
# -------------------------
@app.route("/feature-importance", methods=["GET"])
def feature_importance():
    try:
        importances = model.feature_importances_

        features = [
            {"feature": f"Feature {i+1}", "importance": float(val)}
            for i, val in enumerate(importances)
        ]

        features = sorted(features, key=lambda x: x["importance"], reverse=True)

        return jsonify({
            "model": "RandomForestClassifier",
            "feature_importance": features
        })

    except Exception as e:
        return jsonify({"error": str(e)})


# -------------------------
# Run Server
# -------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)