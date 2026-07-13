# CreditSense 💳🤖

**AI-Powered Credit Card Approval Prediction Platform**
Built for the SmartBridge Summer Internship Program.

CreditSense wraps a trained credit-approval model in a full, premium
fintech-style Flask web application — complete with a multi-step prediction
wizard, a live analytics dashboard, searchable prediction history, a 4-model
comparison page, and a demo JSON API.

---

## ✨ Features

- 5-step animated prediction wizard with client-side validation
- Real-time ML inference using a validation-tuned decision threshold
  (not the model's default 0.5 cutoff)
- Rich result page: probability, confidence, risk category, recommended
  credit limit, estimated APR, and a plain-language AI explanation
- Downloadable PDF prediction report (pure-Python, no system dependencies)
- Live analytics dashboard (Chart.js, bundled locally — no external CDN
  dependency) — approvals, income, education, employment, feature
  importance, model comparison, prediction trends
- Searchable / sortable / filterable prediction history with CSV export
- ML Models comparison page (Logistic Regression, Decision Tree, Random
  Forest, XGBoost) with real metrics from the training notebook
- Dark / light mode, single consistent blue accent theme
- Fully responsive, animated (scroll reveals, counters, floating cards)

## 🗂 Project Structure

```
CreditSense/
├── app.py                 # Flask routes
├── config.py
├── requirements.txt
├── Procfile                # gunicorn start command (Render/Heroku)
├── render.yaml              # one-click Render blueprint
├── runtime.txt
├── datasets/                # raw CSVs (application_record, credit_record)
├── models/                   # trained artifacts produced by the notebook
│   ├── credit_card_approval_model.pkl
│   ├── scaler.pkl
│   ├── label_encoders.pkl
│   ├── feature_order.pkl
│   └── decision_threshold.pkl
├── notebook/                  # official ML workflow notebook
│   └── Credit_Card_Approval_Prediction.ipynb
├── utils/
│   ├── predictor.py         # model-agnostic ML inference layer
│   ├── db.py                 # SQLite persistence (history/dashboard)
│   └── report.py              # PDF report generator (fpdf2)
├── templates/                # Jinja2 views
│   ├── partials/               # navbar / footer
│   └── errors/                  # 404 / 500 pages
└── static/
    ├── css/                     # design system + page styles
    └── js/                       # interactions, wizard, charts
        └── vendor/                # locally-bundled Chart.js (no CDN dependency)
```

## 🚀 Run locally

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Visit **http://localhost:5000**

## ☁️ Deploy to Render

1. Push this project to a GitHub repository.
2. In Render, choose **New → Blueprint** and point it at your repo
   (`render.yaml` is already included), **or** create a Web Service manually with:
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn app:app --workers 2 --threads 4 --timeout 120`
3. Deploy — Render will install dependencies and start the app automatically.

> The SQLite database (`instance/creditsense.db`) is created automatically on
> first run and seeded with demo predictions so the Dashboard/History
> pages are never empty during a live demo. Render's free tier has an
> ephemeral filesystem, so data resets on redeploy — this is expected for a
> demo project.

## 🧠 Machine Learning

`notebook/Credit_Card_Approval_Prediction.ipynb` is the single source of
truth for the model: it loads `datasets/*.csv`, cleans and encodes the data,
trains and compares four algorithms with a validation-tuned decision
threshold per model, and saves the winner's artifacts straight into
`models/`. `app.py` never hardcodes a model type — `utils/predictor.py`
reads whichever class was saved and adapts automatically (tree-based
`feature_importances_` vs. linear `coef_`, threshold-based classification
instead of `.predict()`).

**Retraining is a drop-in swap:** re-run the notebook top-to-bottom and the
new `models/*.pkl` files work immediately — no changes to `app.py` required.

Real results from the current trained artifacts:

| Model | Accuracy | Threshold | F1 (macro) | ROC-AUC |
|---|---|---|---|---|
| **Logistic Regression (primary)** | **58.40%** | **0.51** | **48.95%** | **49.93%** |
| XGBoost | 55.52% | 0.51 | 46.54% | 47.14% |
| Random Forest | 53.48% | 0.48 | 45.46% | 44.83% |
| Decision Tree | 50.90% | 0.51 | 45.14% | 48.28% |

**Honest note:** this dataset's "Rejected" outcome is only weakly predictable
from application-time fields alone (best ROC-AUC ≈ 0.50, i.e. barely better
than random). Logistic Regression was selected for the best macro-F1 among
models clearing a 30% recall floor — not because any model achieves high
real-world accuracy. See the notebook's "Honest Note on Predictive Power"
section for the full analysis.

## ⚠️ Disclaimer

This is a demonstration project built for an academic internship. Predictions,
recommended credit limits and interest rates are illustrative only and do not
represent a real credit decision or financial advice.
