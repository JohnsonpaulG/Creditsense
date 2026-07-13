"""
CreditSense - Premium AI-Powered Banking Web Application
SmartBridge Summer Internship Project

Flask backend built on top of the original Credit Card Approval
Prediction model. This file wires together the ML prediction engine
(model-agnostic - see utils/predictor.py for whichever model class
notebook/Credit_Card_Approval_Prediction.ipynb currently selects as
the best performer), a SQLite-backed history/dashboard layer, and a
full multi-page banking UI.
"""

import os
import io
import csv
import random
from datetime import datetime, timedelta

from flask import (
    Flask, render_template, request, redirect, url_for,
    jsonify, send_file, flash, abort
)

from config import Config
from utils import predictor, db
from utils.report import build_report

app = Flask(__name__)
app.config.from_object(Config)

predictor.init_model(app.config["MODEL_DIR"])
db.init_db(app.config["DB_PATH"])


# ---------------------------------------------------------------------
# Seed a handful of demo predictions the FIRST time the app runs so that
# Dashboard / History never look empty during a live demo.
# ---------------------------------------------------------------------
def seed_demo_data():
    stats = db.get_stats()
    if stats["total"] > 0:
        return

    names = ["Aarav Sharma", "Priya Nair", "Rohan Mehta", "Ishita Rao", "Karan Verma",
             "Sneha Iyer", "Vikram Singh", "Ananya Das", "Arjun Kapoor", "Meera Pillai",
             "Rahul Gupta", "Divya Menon", "Aditya Kumar", "Kavya Reddy", "Nikhil Joshi"]
    occ = list(predictor.OCCUPATION_MAP.keys())
    inc_types = list(predictor.INCOME_TYPE_MAP.keys())
    edu = list(predictor.EDUCATION_MAP.keys())
    fam = list(predictor.FAMILY_STATUS_MAP.keys())
    house = list(predictor.HOUSING_TYPE_MAP.keys())

    random.seed(7)
    for i in range(42):
        income = random.randint(15000, 220000)
        age = random.randint(21, 65)
        created = datetime.now() - timedelta(days=random.randint(0, 45),
                                              hours=random.randint(0, 23))

        # Build a synthetic-but-plausible applicant and run it through the
        # REAL model (the same predictor.predict() the live wizard uses),
        # instead of fabricating random probability/confidence numbers.
        # This keeps demo History/Dashboard data honest and consistent
        # with whatever the actual deployed model produces.
        synthetic_form = {
            "applicant_name": random.choice(names),
            "gender": random.choice(["Male", "Female"]),
            "own_car": random.choice(["Yes", "No"]),
            "own_house": random.choice(["Yes", "No"]),
            "children": str(random.randint(0, 3)),
            "income": str(income),
            "income_type": random.choice(inc_types),
            "education": random.choice(edu),
            "family_status": random.choice(fam),
            "housing_type": random.choice(house),
            "occupation": random.choice(occ),
            "family_members": str(random.randint(1, 5)),
            "age": str(age),
            "years_employed": str(round(random.uniform(0, 30), 1)),
        }
        result = predictor.predict(synthetic_form)

        record = {
            "created_at": created.isoformat(timespec="seconds"),
            "applicant_name": synthetic_form["applicant_name"],
            "gender": synthetic_form["gender"],
            "age": age,
            "income": income,
            "income_type": synthetic_form["income_type"],
            "education": synthetic_form["education"],
            "family_status": synthetic_form["family_status"],
            "housing_type": synthetic_form["housing_type"],
            "occupation": synthetic_form["occupation"],
            "children": int(synthetic_form["children"]),
            "family_members": int(synthetic_form["family_members"]),
            "years_employed": float(synthetic_form["years_employed"]),
            "own_car": synthetic_form["own_car"],
            "own_house": synthetic_form["own_house"],
            "model_used": result["model_used"],
            "prediction": result["prediction_label"],
            "approved": 1 if result["approved"] else 0,
            "probability": result["probability"],
            "confidence": result["confidence"],
            "risk_category": result["risk_label"],
            "credit_limit": result["credit_limit"],
            "interest_rate": result["interest_rate"],
        }
        db.insert_prediction(record)


seed_demo_data()


# =======================================================================
# Static informational pages
# =======================================================================

@app.route("/")
def index():
    stats = db.get_stats()
    return render_template(
        "index.html", stats=stats,
        primary_model=PRIMARY_MODEL_NAME,
        primary_accuracy=MODEL_RESULTS[PRIMARY_MODEL_NAME]["accuracy"],
        importance=predictor.get_feature_importance(),
    )


@app.route("/features")
def features():
    return render_template(
        "features.html",
        primary_model=PRIMARY_MODEL_NAME,
        primary_accuracy=MODEL_RESULTS[PRIMARY_MODEL_NAME]["accuracy"],
    )


@app.route("/how-it-works")
def how_it_works():
    return render_template("how_it_works.html")


@app.route("/ml-models")
def ml_models():
    importance = predictor.get_feature_importance()
    return render_template(
        "ml_models.html",
        importance=importance,
        model_accuracy=MODEL_ACCURACY,
        model_results=MODEL_RESULTS,
        primary_model=PRIMARY_MODEL_NAME,
    )


@app.route("/about-project")
def about_project():
    return render_template("about_project.html", primary_model=PRIMARY_MODEL_NAME)


@app.route("/about-us")
def about_us():
    return render_template("about_us.html")


@app.route("/github")
def github():
    return render_template("github.html")


@app.route("/contact", methods=["GET", "POST"])
def contact():
    if request.method == "POST":
        flash("Thanks! Your message has been received. Our support team will reach out shortly.", "success")
        return redirect(url_for("contact"))
    return render_template("contact.html")


# =======================================================================
# Prediction wizard + result
# =======================================================================

@app.route("/prediction")
def prediction():
    return render_template("prediction.html", options=predictor.DROPDOWN_OPTIONS)


@app.route("/predict", methods=["POST"])
def predict():
    result = predictor.predict(request.form)

    record = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "applicant_name": result["applicant_name"],
        "gender": result["inputs"]["gender"],
        "age": result["inputs"]["age"],
        "income": result["inputs"]["income"],
        "income_type": result["inputs"]["income_type"],
        "education": result["inputs"]["education"],
        "family_status": result["inputs"]["family_status"],
        "housing_type": result["inputs"]["housing_type"],
        "occupation": result["inputs"]["occupation"],
        "children": result["inputs"]["children"],
        "family_members": result["inputs"]["family_members"],
        "years_employed": result["inputs"]["years_employed"],
        "own_car": result["inputs"]["own_car"],
        "own_house": result["inputs"]["own_house"],
        "model_used": result["model_used"],
        "prediction": result["prediction_label"],
        "approved": 1 if result["approved"] else 0,
        "probability": result["probability"],
        "confidence": result["confidence"],
        "risk_category": result["risk_label"],
        "credit_limit": result["credit_limit"],
        "interest_rate": result["interest_rate"],
    }
    pred_id = db.insert_prediction(record)

    return render_template("result.html", result=result, pred_id=pred_id)


@app.route("/report/<int:pred_id>")
def download_report(pred_id):
    row = db.get_prediction(pred_id)
    if not row:
        abort(404)

    result = {
        "applicant_name": row["applicant_name"],
        "approved": bool(row["approved"]),
        "prediction_label": row["prediction"],
        "probability": row["probability"],
        "confidence": row["confidence"],
        "risk_label": row["risk_category"],
        "credit_limit": row["credit_limit"] or 0,
        "interest_rate": row["interest_rate"],
        "model_used": row["model_used"],
        "prediction_time_ms": 0,
        "explanation": (
            f"This report reflects the AI model's decision for {row['applicant_name']} "
            f"based on the submitted financial and demographic profile."
        ),
        "inputs": {
            "gender": row["gender"], "age": row["age"], "income": row["income"],
            "income_type": row["income_type"], "education": row["education"],
            "family_status": row["family_status"], "housing_type": row["housing_type"],
            "occupation": row["occupation"], "children": row["children"],
            "family_members": row["family_members"], "years_employed": row["years_employed"],
            "own_car": row["own_car"], "own_house": row["own_house"],
        },
    }
    pdf_bytes = build_report(result, prediction_id=pred_id)
    return send_file(
        io.BytesIO(pdf_bytes),
        mimetype="application/pdf",
        as_attachment=True,
        download_name=f"CreditSense_Report_{pred_id}.pdf",
    )


# =======================================================================
# Dashboard / History
# =======================================================================

# Real evaluation results from notebook/Credit_Card_Approval_Prediction.ipynb
# (Epic 4.4 model comparison, executed against the real dataset). The
# dataset's "Rejected" class is only weakly predictable from these
# application-time features alone (see the notebook's "Honest Note on
# Predictive Power") - these are the real, unmodified numbers the
# notebook produced, not illustrative placeholders.
MODEL_RESULTS = {
    "Logistic Regression": {"accuracy": 58.40, "threshold": 0.51, "precision_rejected": 21.66, "recall_rejected": 35.75, "f1_rejected": 26.98, "f1_macro": 48.95, "roc_auc": 49.93},
    "XGBoost":             {"accuracy": 55.52, "threshold": 0.51, "precision_rejected": 19.36, "recall_rejected": 33.80, "f1_rejected": 24.62, "f1_macro": 46.54, "roc_auc": 47.14},
    "Random Forest":       {"accuracy": 53.48, "threshold": 0.48, "precision_rejected": 18.83, "recall_rejected": 35.20, "f1_rejected": 24.54, "f1_macro": 45.46, "roc_auc": 44.83},
    "Decision Tree":       {"accuracy": 50.90, "threshold": 0.51, "precision_rejected": 20.05, "recall_rejected": 43.02, "f1_rejected": 27.35, "f1_macro": 45.14, "roc_auc": 48.28},
}
# Selected by the notebook itself: highest macro-F1 among models whose
# Rejected-class recall clears the 0.30 floor (see notebook Epic 4.5).
PRIMARY_MODEL_NAME = predictor.MODEL_DISPLAY_NAME if predictor.MODEL_DISPLAY_NAME in MODEL_RESULTS else "Logistic Regression"

MODEL_ACCURACY = {name: metrics["accuracy"] for name, metrics in MODEL_RESULTS.items()}


def build_dashboard_chart_data(rows, importance):
    # ---- income histogram ----
    bins = [0, 20000, 40000, 60000, 80000, 100000, 150000, 1_000_000]
    bin_labels = ["<20k", "20-40k", "40-60k", "60-80k", "80-100k", "100-150k", "150k+"]
    counts = [0] * (len(bins) - 1)
    for r in rows:
        income = r["income"] or 0
        for i in range(len(bins) - 1):
            if bins[i] <= income < bins[i + 1]:
                counts[i] += 1
                break

    # ---- education distribution ----
    edu_counts = {}
    for r in rows:
        key = r["education"] or "Unknown"
        edu_counts[key] = edu_counts.get(key, 0) + 1

    # ---- employment / income type distribution ----
    emp_counts = {}
    for r in rows:
        key = r["income_type"] or "Unknown"
        emp_counts[key] = emp_counts.get(key, 0) + 1

    # ---- prediction trends (last 14 days) ----
    from collections import OrderedDict
    today = datetime.now().date()
    day_buckets = OrderedDict()
    for i in range(13, -1, -1):
        d = today - timedelta(days=i)
        day_buckets[d.isoformat()] = {"approved": 0, "declined": 0}
    for r in rows:
        try:
            d = datetime.fromisoformat(r["created_at"]).date().isoformat()
        except Exception:
            continue
        if d in day_buckets:
            if r["approved"]:
                day_buckets[d]["approved"] += 1
            else:
                day_buckets[d]["declined"] += 1

    return {
        "approved": sum(1 for r in rows if r["approved"]),
        "rejected": sum(1 for r in rows if not r["approved"]),
        "income_bins": {"labels": bin_labels, "counts": counts},
        "education": {"labels": list(edu_counts.keys()), "counts": list(edu_counts.values())},
        "employment": {"labels": list(emp_counts.keys()), "counts": list(emp_counts.values())},
        "importance": {
            "labels": [f[0] for f in importance[:8]],
            "values": [f[1] for f in importance[:8]],
        },
        "model_accuracy": {
            "labels": list(MODEL_ACCURACY.keys()),
            "values": list(MODEL_ACCURACY.values()),
        },
        "trends": {
            "labels": [datetime.fromisoformat(d).strftime("%b %d") for d in day_buckets.keys()],
            "approved": [v["approved"] for v in day_buckets.values()],
            "declined": [v["declined"] for v in day_buckets.values()],
        },
    }


@app.route("/dashboard")
def dashboard():
    stats = db.get_stats()
    importance = predictor.get_feature_importance()
    rows = db.get_all_predictions(limit=500)
    chart_data = build_dashboard_chart_data(rows, importance)
    return render_template(
        "dashboard.html",
        stats=stats,
        importance=importance,
        rows=rows[:8],
        chart_data=chart_data,
        model_accuracy=MODEL_ACCURACY,
        primary_model=PRIMARY_MODEL_NAME,
        primary_accuracy=MODEL_RESULTS[PRIMARY_MODEL_NAME]["accuracy"],
    )


@app.route("/history")
def history():
    rows = db.get_all_predictions(limit=500)
    return render_template("history.html", rows=rows)


@app.route("/history/export.csv")
def export_history_csv():
    rows = db.get_all_predictions(limit=5000)
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["ID", "Date", "Applicant", "Prediction", "Confidence",
                      "Risk", "Model Used", "Income", "Age"])
    for r in rows:
        writer.writerow([r["id"], r["created_at"], r["applicant_name"], r["prediction"],
                          r["confidence"], r["risk_category"], r["model_used"],
                          r["income"], r["age"]])
    output.seek(0)
    return send_file(
        io.BytesIO(output.getvalue().encode("utf-8")),
        mimetype="text/csv",
        as_attachment=True,
        download_name="creditsense_history.csv",
    )


# =======================================================================
# Demonstration JSON API (used internally by the frontend)
# =======================================================================

@app.route("/api/statistics")
def api_statistics():
    return jsonify(db.get_stats())


@app.route("/api/history")
def api_history():
    limit = request.args.get("limit", 20, type=int)
    return jsonify(db.get_all_predictions(limit=limit))


@app.route("/api/model-info")
def api_model_info():
    return jsonify({
        "model_name": "Random Forest Classifier",
        "n_estimators": 200,
        "max_depth": 10,
        "class_weight": "balanced",
        "features": predictor.FEATURE_ORDER,
        "feature_importance": predictor.get_feature_importance(),
    })


@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json(force=True, silent=True) or {}
    result = predictor.predict(data)
    return jsonify(result)


# =======================================================================
# Error handlers
# =======================================================================

@app.errorhandler(404)
def not_found(e):
    return render_template("errors/404.html"), 404


@app.errorhandler(500)
def server_error(e):
    return render_template("errors/500.html"), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)
