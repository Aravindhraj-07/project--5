from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load your trained pipeline (make sure it includes preprocessing for categorical features)
model = joblib.load("model.joblib")

@app.route('/')
def home():
    return "API is running. Use POST /predict to get predictions."

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        # Extract features from the request
        features = [
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
        ]

        # Convert to NumPy 2D array for the model
        features = np.array([features])

        # Predict
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0][prediction]  # Probability of predicted class

        return jsonify({
            "prediction": int(prediction),  # 0 or 1
            "probability": float(probability)
        })

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)