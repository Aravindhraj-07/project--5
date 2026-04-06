from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import pandas as pd
import os

app = Flask(__name__)
CORS(app)

model = joblib.load("model.joblib")

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        input_df = pd.DataFrame([{
            "Current Loan Amount":       float(data["loan_amount"]),
            "Term":                      data["term"],
            "Credit Score":              float(data["credit_score"]),
            "Annual Income":             float(data["income"]),
            "Years in current job":      float(data["years_job"]),
            "Home Ownership":            data["home"],
            "Purpose":                   data["purpose"],
            "Monthly Debt":              float(data["debt"]),
            "Years of Credit History":   float(data["credit_years"]),
            "Number of Open Accounts":   float(data["accounts"]),
            "Number of Credit Problems": float(data["credit_problems"]),
            "Current Credit Balance":    float(data["credit_balance"]),
            "Maximum Open Credit":       float(data["max_credit"]),
            "Bankruptcies":              float(data["bankruptcies"]),
            "Tax Liens":                 float(data["tax_liens"]),
        }])

        prediction  = int(model.predict(input_df)[0])
        probability = float(model.predict_proba(input_df).max())

        return jsonify({
            "prediction":  prediction,
            "probability": probability
        })

    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)