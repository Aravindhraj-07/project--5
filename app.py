from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load("model.joblib")

@app.route('/')
def home():
    return render_template("index.html")  # serves your HTML

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    # convert data to numpy array
    features = np.array([[
        data['loan_amount'],
        data['term'],
        data['purpose'],
        data['home'],
        data['income'],
        data['other_income'],
        data['debt'],
        data['years_job'],
        data['credit_score'],
        data['credit_years'],
        data['credit_balance'],
        data['max_credit'],
        data['accounts'],
        data['credit_problems'],
        data['bankruptcies'],
        data['tax_liens']
    ]])
    prediction = model.predict(features)[0]
    probability = float(model.predict_proba(features).max())
    return jsonify({"prediction": int(prediction), "probability": probability})

if __name__ == "__main__":
    app.run(debug=True)