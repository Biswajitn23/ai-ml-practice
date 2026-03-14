from flask import Flask, request, jsonify, render_template
from flasgger import Swagger
import joblib
from datetime import datetime

app = Flask(__name__)
swagger = Swagger(app)

model = joblib.load("rf_model.pkl")

prediction_history = []


@app.route("/")
def home():
    return render_template("index.html", history=prediction_history)


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

        prediction_history.append({
            "prediction": label,
            "confidence": confidence,
            "time": datetime.now().strftime("%H:%M:%S")
        })

        return render_template(
            "index.html",
            prediction=label,
            confidence=confidence,
            history=prediction_history
        )

    except Exception as e:
        return render_template(
            "index.html",
            prediction=f"Error: {str(e)}",
            history=prediction_history
        )


@app.route("/feature-importance", methods=["GET"])
def feature_importance():

    try:

        importances = model.feature_importances_

        features = [
            {"feature": f"Feature {i+1}", "importance": float(val)}
            for i, val in enumerate(importances)
        ]

        features = sorted(features, key=lambda x: x["importance"], reverse=True)[:10]

        return jsonify({
            "model": "RandomForestClassifier",
            "feature_importance": features
        })

    except Exception as e:
        return jsonify({"error": str(e)})


@app.route("/health")
def health():
    return {
        "status": "running",
        "model": "RandomForestClassifier",
        "predictions_made": len(prediction_history)
    }
    
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)