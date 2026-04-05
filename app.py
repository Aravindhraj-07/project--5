from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained pipeline
pipeline = joblib.load("model.joblib")

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    if request.method == "POST":
        try:
            # Collect form inputs
            data = request.form
            features = [[
                float(data['loan_amount']),
                data['term'],
                data['purpose'],
                data['home_ownership'],
                float(data['annual_income']),
                float(data['other_income']),
                float(data['monthly_debt']),
                float(data['years_in_current_job']),
                float(data['credit_score']),
                float(data['years_of_credit_history']),
                float(data['current_credit_balance']),
                float(data['maximum_open_credit']),
                float(data['open_accounts']),
                float(data['credit_problems']),
                float(data['bankruptcies']),
                float(data['tax_liens'])
            ]]

            # Make prediction
            pred = pipeline.predict(features)[0]
            prob = pipeline.predict_proba(features)[0][1]  # probability for 1
            prediction = f"Loan Status: {pred} (Probability of Fully Paid: {prob:.2f})"
        except Exception as e:
            prediction = f"Error: {str(e)}"

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)