from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib

app = Flask(__name__)
CORS(app)



pipeline = joblib.load("model.joblib")  # Make sure this pipeline was trained with all 16 features

@app.route("/")
def home():
    return "Loan Prediction API is running"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # Create dataframe with all 16 features
        input_data = pd.DataFrame([{
            "Current Loan Amount": float(data["loan_amount"]),
            "Term": data["term"],
            "Credit Score": float(data["credit_score"]),
            "Annual Income": float(data["income"]),
            "Years in current job": int(data["years_job"]),
            "Home Ownership": data["home"],
            "Purpose": data["purpose"],
            "Monthly Debt": float(data["debt"]),
            "Current Credit Balance": float(data["credit_balance"]),
            "Number of Open Accounts": int(data["accounts"]),
            "Maximum Open Credit": float(data["max_credit"]),
            "Tax Liens": int(data["tax_liens"]),
            "Bankruptcies": int(data["bankruptcies"]),
            "Number of Credit Problems": int(data["credit_problems"]),
            "Years of Credit History": float(data["credit_years"]),
            "Other Income": float(data["other_income"])
        }])

        # Predict
        prediction = pipeline.predict(input_data)[0]
        probability = pipeline.predict_proba(input_data)[0][1]

        return jsonify({
            "prediction": int(prediction),
            "probability": float(probability)
        })

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run()