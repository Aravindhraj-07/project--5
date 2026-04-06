# LoanIQ - ML-Powered Loan Approval Predictor 



## 🎯 Project Overview
**LoanIQ** is a production-ready Flask web application that uses a  RandomForestClassifier machine learning model to predict whether a loan will be **Fully Paid** or **Charged Off**. Trained on a real-world credit dataset (~100k samples), it analyzes 15 key financial features to provide instant predictions with confidence scores.


**Key Features:**
- &nbsp;📊 Interactive web UI with form validation and live credit score visualization
- &nbsp;🤖 Random Forest model (71% accuracy, 0.74 ROC-AUC) with class-balanced training
- &nbsp;🌐 Deployed on Render: [Live Demo](https://project-5-mho9.onrender.com/)


**ML Pipeline:** `ColumnTransformer` → `RandomForestClassifier(n_estimators=100, max_depth=12, class_weight='balanced')`

**Input Features (15 exact columns - must match training order):**

| Numeric | Categorical |
|---------|-------------|
| Current Loan Amount | Term |
| Credit Score | Home Ownership |
| Annual Income | Purpose |
| Years in current job | |
| Monthly Debt | |
| Years of Credit History | |
| Number of Open Accounts | |
| Number of Credit Problems | |
| Current Credit Balance | |
| Maximum Open Credit | |
| Bankruptcies | |
| Tax Liens | |

**Output:** `{ "prediction": 0/1, "probability": 0.8542 }` (confidence of predicted class)


## 🧠 Model Performance
| Metric | Test Set |
|--------|----------|
| **Accuracy** | 71.0% |
| **ROC-AUC** | 0.74 |
| **Trees** | 100 |
| **Max Depth** | 12 |

Trained on balanced classes to handle real-world loan default rarity.

⚠️ **Note:** There will be some errors in the predictions because the training dataset contains 70% of approved (1).so it may predict wrong status also


## 🔧 API Usage
```
POST /predict
Content-Type: application/json

{
  "loan_amount": 50000,
  "term": "Short Term",
  "credit_score": 720,
  // ... 12 more fields (see app.py)
}

Response:
{
  "prediction": 1,      // 1=Fully Paid, 0=Charged Off
  "probability": 0.8234 // Confidence
}
```

## ☁️ Deployment
- **Platform**: Render.com
- **Live URL**: [https://project-5-mho9.onrender.com/](https://project-5-mho9.onrender.com/)
- **Procfile**: `web: gunicorn app:app`


## 🤝 Contributors
- Aravindhraj (Developer)


