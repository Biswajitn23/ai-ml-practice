from flask import Flask, request, jsonify, render_template
from flasgger import Swagger
import joblib

app = Flask(__name__)
swagger = Swagger(app)

model = joblib.load("rf_model.pkl")

@app.route("/")
def home():
    return render_template("index.html")


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

    data = request.json["features"]

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


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)