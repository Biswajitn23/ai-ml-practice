from flask import Flask, request, jsonify
import joblib
import os

app = Flask(__name__)

# Load model with error handling
try:
    model = joblib.load("student_rf_model.pkl")
    model_loaded = True
except FileNotFoundError:
    model_loaded = False
    print("Warning: student_rf_model.pkl not found")

@app.route("/", methods=["GET"])
def home():
    """Root endpoint"""
    return jsonify({
        "message": "Student Performance Prediction API",
        "endpoints": {
            "POST /predict": "Predict student grade (requires: Maths, Science, English)",
            "GET /health": "Check API health status"
        }
    })

@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "ok",
        "model_loaded": model_loaded
    })

@app.route("/predict", methods=["POST"])
def predict():
    """Predict student grade based on subject scores"""
    try:
        if not model_loaded:
            return jsonify({"error": "Model not loaded"}), 500
        
        if not request.json:
            return jsonify({"error": "Request body must be JSON"}), 400
        
        data = request.json
        
        # Validate required fields
        required_fields = ["Maths", "Science", "English"]
        missing_fields = [f for f in required_fields if f not in data]
        if missing_fields:
            return jsonify({"error": f"Missing required fields: {missing_fields}"}), 400
        
        # Validate data types and ranges
        try:
            maths = float(data["Maths"])
            science = float(data["Science"])
            english = float(data["English"])
        except (TypeError, ValueError):
            return jsonify({"error": "Maths, Science, and English must be numeric values"}), 400
        
        prediction = model.predict([[maths, science, english]])
        
        return jsonify({
            "prediction": int(prediction[0]),
            "input": {
                "Maths": maths,
                "Science": science,
                "English": english
            }
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)