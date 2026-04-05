from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import numpy as np

app = Flask(__name__)
CORS(app)  # allow cross-origin requests if needed

# Load your trained model
model = joblib.load("model.joblib")

# Serve the HTML frontend
@app.route('/')
def home():
    return render_template("index.html")  # make sure your HTML is in templates/index.html

# API endpoint for predictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        # Make sure you match the order of features your model expects
        features = np.array([[
            float(data['loan_amount']),
            data['term'],
            data['purpose'],
            data['home'],
            float(data['income']),
            float(data['other_income']),
            float(data['debt']),
            float(data['years_job']),
            float(data['credit_score']),
            float(data['credit_years']),
            float(data['credit_balance']),
            float(data['max_credit']),
            float(data['accounts']),
            float(data['credit_problems']),
            float(data['bankruptcies']),
            float(data['tax_liens'])
        ]])

        prediction = model.predict(features)[0]
        probability = float(model.predict_proba(features).max())

        return jsonify({"prediction": int(prediction), "probability": probability})
    
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    # For Render deployment, use host='0.0.0.0' and port from environment
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)