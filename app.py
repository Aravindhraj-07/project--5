from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import pandas as pd
import os

app = Flask(__name__)# Create a Flask application instance
CORS(app)# Enable Cross-Origin Resource Sharing (CORS) for the Flask app, allowing it to handle requests from different origins

model = joblib.load("model.joblib")

@app.route('/')# Define a route for the home page that renders the index.html template
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])# Define a route for the /predict endpoint that accepts POST requests and processes the input data to make predictions using the loaded model
def predict():# Handle prediction requests
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

        prediction  = int(model.predict(input_df)[0])# Use the model to predict the class label (0 or 1) for the input data and convert it to an integer
        probability = float(model.predict_proba(input_df).max())# Use the model to predict the probabilities for each class and extract
# the maximum probability (confidence) for the predicted class, converting it to a float
        return jsonify({
            "prediction":  prediction,
            "probability": probability
        })
# If any exceptions occur during the prediction process, catch the exception and return an error message as a JSON response
    except Exception as e:
        return jsonify({"error": str(e)})

# Start the Flask application, listening on all available network interfaces 
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)