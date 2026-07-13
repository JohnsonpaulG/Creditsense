"""
Core ML prediction logic for CreditSense.

This wraps whichever model the training notebook (notebook/Credit_Card_
Approval_Prediction.ipynb) selects as the best performer - the notebook
compares Logistic Regression, Decision Tree, Random Forest and XGBoost,
tunes a decision threshold for each on a held-out validation set, and
ships the winner's four/five artifacts (model, scaler, label encoders,
feature order, decision threshold) into models/. This module is
intentionally model-agnostic: it reads whichever class was saved and
adapts (feature_importances_ vs. coef_, threshold-based classification
instead of the default .predict()) automatically, so retraining from
the notebook never requires touching this file.

On top of the raw model output, this module adds a thin banking-grade
presentation layer: confidence score, risk category, suggested credit
limit, indicative interest rate, and a human-readable AI explanation.
"""

import os
import time
import joblib
import numpy as np
import pandas as pd

MODEL = None
SCALER = None
LABEL_ENCODERS = None
FEATURE_ORDER = None
MODEL_DIR = None
DECISION_THRESHOLD = 0.5  # overwritten by decision_threshold.pkl if present
MODEL_DISPLAY_NAME = "Machine Learning Model"

# Risk-band cutoffs on P(Approved). Defaults match the original fixed
# bands; overwritten by risk_thresholds.pkl if present, which calibrates
# them to this specific model's real, honest probability distribution on
# held-out data (see notebook / retrain_fix.py) instead of assuming a
# more separable classifier than the data actually supports.
LOW_RISK_CUTOFF = 0.80
HIGH_RISK_CUTOFF = 0.55

GENDER_MAP = {"Male": "M", "Female": "F"}
YES_NO_MAP = {"Yes": "Y", "No": "N"}

INCOME_TYPE_MAP = {
    "Working": "Working",
    "Commercial Associate": "Commercial associate",
    "Pensioner": "Pensioner",
    "State Servant": "State servant",
    "Student": "Student",
}

EDUCATION_MAP = {
    "Higher Education": "Higher education",
    "Secondary Education": "Secondary / secondary special",
    "Lower Secondary": "Lower secondary",
    "Incomplete Higher": "Incomplete higher",
    "Academic Degree": "Academic degree",
}

FAMILY_STATUS_MAP = {
    "Married": "Married",
    "Single / Not Married": "Single / not married",
    "Civil Marriage": "Civil marriage",
    "Separated": "Separated",
    "Widow": "Widow",
}

HOUSING_TYPE_MAP = {
    "House/Apartment": "House / apartment",
    "With Parents": "With parents",
    "Municipal Apartment": "Municipal apartment",
    "Rented Apartment": "Rented apartment",
    "Office Apartment": "Office apartment",
    "Co-op Apartment": "Co-op apartment",
}

OCCUPATION_MAP = {
    "Accountants": "Accountants",
    "Cleaning Staff": "Cleaning staff",
    "Cooking Staff": "Cooking staff",
    "Core Staff": "Core staff",
    "Drivers": "Drivers",
    "HR Staff": "HR staff",
    "High Skill Tech Staff": "High skill tech staff",
    "IT Staff": "IT staff",
    "Laborers": "Laborers",
    "Low-skill Laborers": "Low-skill Laborers",
    "Managers": "Managers",
    "Medicine Staff": "Medicine staff",
    "Private Service Staff": "Private service staff",
    "Realty Agents": "Realty agents",
    "Secretaries": "Secretaries",
    "Security Staff": "Security staff",
    "Sales Staff": "Sales staff",
    "Waiters/Barmen Staff": "Waiters/barmen staff",
    "Unknown / Not Employed": "Unknown",
}

DROPDOWN_OPTIONS = {
    "gender": list(GENDER_MAP.keys()),
    "yes_no": list(YES_NO_MAP.keys()),
    "income_type": list(INCOME_TYPE_MAP.keys()),
    "education": list(EDUCATION_MAP.keys()),
    "family_status": list(FAMILY_STATUS_MAP.keys()),
    "housing_type": list(HOUSING_TYPE_MAP.keys()),
    "occupation": list(OCCUPATION_MAP.keys()),
}

# Feature importances extracted once from the trained RandomForest so the
# ML Models / Dashboard pages can render a real feature-importance chart.
FEATURE_IMPORTANCE_LABELS = {
    "AMT_INCOME_TOTAL": "Annual Income",
    "AGE": "Age",
    "YEARS_EMPLOYED": "Years Employed",
    "OCCUPATION_TYPE": "Occupation",
    "CNT_FAM_MEMBERS": "Family Members",
    "NAME_FAMILY_STATUS": "Family Status",
    "NAME_INCOME_TYPE": "Income Type",
    "NAME_EDUCATION_TYPE": "Education",
    "CNT_CHILDREN": "Children",
    "FLAG_OWN_REALTY": "Owns Property",
    "NAME_HOUSING_TYPE": "Housing Type",
    "FLAG_OWN_CAR": "Owns Car",
    "CODE_GENDER": "Gender",
}


# Friendly display names for whichever model class the notebook selects.
MODEL_CLASS_DISPLAY_NAMES = {
    "LogisticRegression": "Logistic Regression",
    "DecisionTreeClassifier": "Decision Tree",
    "RandomForestClassifier": "Random Forest",
    "XGBClassifier": "XGBoost",
}


def init_model(model_dir):
    global MODEL, SCALER, LABEL_ENCODERS, FEATURE_ORDER, MODEL_DIR, DECISION_THRESHOLD, MODEL_DISPLAY_NAME
    global LOW_RISK_CUTOFF, HIGH_RISK_CUTOFF
    MODEL_DIR = model_dir

    MODEL = joblib.load(os.path.join(model_dir, "credit_card_approval_model.pkl"))
    SCALER = joblib.load(os.path.join(model_dir, "scaler.pkl"))
    LABEL_ENCODERS = joblib.load(os.path.join(model_dir, "label_encoders.pkl"))
    FEATURE_ORDER = joblib.load(os.path.join(model_dir, "feature_order.pkl"))

    threshold_path = os.path.join(model_dir, "decision_threshold.pkl")
    if os.path.exists(threshold_path):
        DECISION_THRESHOLD = float(joblib.load(threshold_path))
    else:
        DECISION_THRESHOLD = 0.5

    # Calibrated risk-band cutoffs, derived from this exact model's real
    # probability distribution on held-out data (see retrain_fix.py /
    # notebook §4.6). Falls back to the fixed 0.80/0.55 bands if the
    # artifact isn't present, so this never breaks an older models/ folder.
    risk_thresholds_path = os.path.join(model_dir, "risk_thresholds.pkl")
    if os.path.exists(risk_thresholds_path):
        risk_thresholds = joblib.load(risk_thresholds_path)
        LOW_RISK_CUTOFF = float(risk_thresholds["low_risk_cutoff"])
        HIGH_RISK_CUTOFF = float(risk_thresholds["high_risk_cutoff"])
    else:
        LOW_RISK_CUTOFF = 0.80
        HIGH_RISK_CUTOFF = 0.55

    MODEL_DISPLAY_NAME = MODEL_CLASS_DISPLAY_NAMES.get(
        type(MODEL).__name__, type(MODEL).__name__
    )


def get_feature_importance():
    """Returns [(readable_label, importance_pct), ...] sorted descending.

    Tree/ensemble models expose `.feature_importances_` directly. Linear
    models (e.g. Logistic Regression) don't - for those we fall back to
    the absolute magnitude of their coefficients, normalized to sum to
    100%, which is the standard way to read "importance" off a linear
    model's weights.
    """
    if hasattr(MODEL, "feature_importances_"):
        raw = np.asarray(MODEL.feature_importances_, dtype=float)
    elif hasattr(MODEL, "coef_"):
        raw = np.abs(np.asarray(MODEL.coef_, dtype=float)).flatten()
        total = raw.sum()
        if total > 0:
            raw = raw / total
    else:
        return []

    pairs = list(zip(FEATURE_ORDER, raw))
    pairs.sort(key=lambda x: -x[1])
    return [
        (FEATURE_IMPORTANCE_LABELS.get(col, col), round(float(imp) * 100, 1))
        for col, imp in pairs
    ]


def _encode(column, raw_value):
    encoder = LABEL_ENCODERS[column]
    return int(encoder.transform([raw_value])[0])


def _risk_category(probability_approved):
    if probability_approved >= LOW_RISK_CUTOFF:
        return "Low Risk", "success"
    elif probability_approved >= HIGH_RISK_CUTOFF:
        return "Moderate Risk", "warning"
    else:
        return "High Risk", "danger"


def _suggested_credit_limit(income, probability_approved):
    """Simple, transparent heuristic for a demo credit limit."""
    base = income * 0.30
    multiplier = 0.5 + probability_approved  # 0.5x - 1.5x
    limit = base * multiplier
    limit = max(500, min(limit, 50000))
    return round(limit / 100) * 100


def _interest_rate(probability_approved, risk_label):
    """Indicative APR heuristic - lower risk => lower rate."""
    base_rate = 24.0
    reduction = probability_approved * 14.0
    rate = base_rate - reduction
    return round(max(6.5, min(rate, 27.9)), 2)


def _explanation(inputs, approved, top_features):
    name_map = {
        "AMT_INCOME_TOTAL": ("annual income", inputs["income"]),
        "AGE": ("age", inputs["age"]),
        "YEARS_EMPLOYED": ("years employed", inputs["years_employed"]),
        "OCCUPATION_TYPE": ("occupation", inputs["occupation_text"]),
        "CNT_FAM_MEMBERS": ("household size", inputs["family_members"]),
        "NAME_FAMILY_STATUS": ("family status", inputs["family_status_text"]),
        "NAME_INCOME_TYPE": ("income type", inputs["income_type_text"]),
        "NAME_EDUCATION_TYPE": ("education level", inputs["education_text"]),
        "CNT_CHILDREN": ("number of children", inputs["children"]),
        "FLAG_OWN_REALTY": ("property ownership", inputs["own_house_text"]),
        "NAME_HOUSING_TYPE": ("housing type", inputs["housing_type_text"]),
        "FLAG_OWN_CAR": ("car ownership", inputs["own_car_text"]),
        "CODE_GENDER": ("gender", inputs["gender_text"]),
    }
    top3 = top_features[:3]
    factor_sentences = []
    for col, _imp in top3:
        label, value = name_map.get(col, (col, ""))
        factor_sentences.append(f"{label} ({value})")

    verdict = "approved" if approved else "declined"
    factors_text = ", ".join(factor_sentences)
    return (
        f"The AI model {verdict} this application primarily based on "
        f"the following weighted factors: {factors_text}. These are the "
        f"features the {MODEL_DISPLAY_NAME} model relies on most heavily "
        f"across the entire training dataset, combined with this "
        f"applicant's specific values to reach a final decision."
    )


def predict(form):
    """
    Accepts a dict-like object (Flask request.form) with the wizard's
    field names and returns a rich result dictionary ready for both the
    results template and the database.
    """
    start = time.perf_counter()

    gender_text = form.get("gender")
    own_car_text = form.get("own_car")
    own_house_text = form.get("own_house")
    children = int(float(form.get("children", 0)))
    income = float(form.get("income", 0))
    income_type_text = form.get("income_type")
    education_text = form.get("education")
    family_status_text = form.get("family_status")
    housing_type_text = form.get("housing_type")
    occupation_text = form.get("occupation")
    family_members = int(float(form.get("family_members", 1)))
    age = float(form.get("age", 0))
    years_employed = float(form.get("years_employed", 0))
    applicant_name = form.get("applicant_name") or "Applicant"

    gender_raw = GENDER_MAP.get(gender_text, gender_text)
    own_car_raw = YES_NO_MAP.get(own_car_text, own_car_text)
    own_house_raw = YES_NO_MAP.get(own_house_text, own_house_text)
    income_type_raw = INCOME_TYPE_MAP.get(income_type_text, income_type_text)
    education_raw = EDUCATION_MAP.get(education_text, education_text)
    family_status_raw = FAMILY_STATUS_MAP.get(family_status_text, family_status_text)
    housing_type_raw = HOUSING_TYPE_MAP.get(housing_type_text, housing_type_text)
    occupation_raw = OCCUPATION_MAP.get(occupation_text, occupation_text)

    encoded = {
        "CODE_GENDER": _encode("CODE_GENDER", gender_raw),
        "FLAG_OWN_CAR": _encode("FLAG_OWN_CAR", own_car_raw),
        "FLAG_OWN_REALTY": _encode("FLAG_OWN_REALTY", own_house_raw),
        "CNT_CHILDREN": float(children),
        "AMT_INCOME_TOTAL": float(income),
        "NAME_INCOME_TYPE": _encode("NAME_INCOME_TYPE", income_type_raw),
        "NAME_EDUCATION_TYPE": _encode("NAME_EDUCATION_TYPE", education_raw),
        "NAME_FAMILY_STATUS": _encode("NAME_FAMILY_STATUS", family_status_raw),
        "NAME_HOUSING_TYPE": _encode("NAME_HOUSING_TYPE", housing_type_raw),
        "OCCUPATION_TYPE": _encode("OCCUPATION_TYPE", occupation_raw),
        "CNT_FAM_MEMBERS": float(family_members),
        "AGE": float(age),
        "YEARS_EMPLOYED": float(years_employed),
    }

    feature_vector = pd.DataFrame([[encoded[c] for c in FEATURE_ORDER]], columns=FEATURE_ORDER)
    feature_vector_scaled = SCALER.transform(feature_vector)

    # The notebook tunes a decision threshold on a held-out validation
    # set (see decision_threshold.pkl) because the dataset's classes are
    # heavily imbalanced - the model's own .predict() (hard-coded to a
    # 0.5 cutoff) is intentionally NOT used here.
    proba = MODEL.predict_proba(feature_vector_scaled)[0]
    prob_reject = float(proba[1])   # class 1 == Rejected, per the notebook's TARGET encoding
    prob_approved = float(proba[0])  # class 0 == Approved
    confidence = float(max(proba))

    approved = prob_reject < DECISION_THRESHOLD
    risk_label, risk_level = _risk_category(prob_approved)
    credit_limit = _suggested_credit_limit(income, prob_approved) if approved else 0
    interest_rate = _interest_rate(prob_approved, risk_label) if approved else None

    top_features = get_feature_importance()
    # map back to raw column keys for explanation lookup (reuses the same
    # safe importance/coefficient logic as get_feature_importance())
    if hasattr(MODEL, "feature_importances_"):
        raw_scores = np.asarray(MODEL.feature_importances_, dtype=float)
    elif hasattr(MODEL, "coef_"):
        raw_scores = np.abs(np.asarray(MODEL.coef_, dtype=float)).flatten()
    else:
        raw_scores = np.zeros(len(FEATURE_ORDER))
    raw_pairs = sorted(zip(FEATURE_ORDER, raw_scores), key=lambda x: -x[1])

    inputs_for_explanation = {
        "income": f"${income:,.0f}",
        "age": f"{int(age)} yrs",
        "years_employed": f"{years_employed} yrs",
        "occupation_text": occupation_text,
        "family_members": family_members,
        "family_status_text": family_status_text,
        "income_type_text": income_type_text,
        "education_text": education_text,
        "children": children,
        "own_house_text": own_house_text,
        "housing_type_text": housing_type_text,
        "own_car_text": own_car_text,
        "gender_text": gender_text,
    }
    explanation = _explanation(inputs_for_explanation, approved, raw_pairs)

    elapsed_ms = round((time.perf_counter() - start) * 1000, 1)

    return {
        "applicant_name": applicant_name,
        "approved": approved,
        "prediction_label": "Approved" if approved else "Declined",
        "probability": round(prob_approved * 100, 1),
        "confidence": round(confidence * 100, 1),
        "risk_label": risk_label,
        "risk_level": risk_level,
        "credit_limit": credit_limit,
        "interest_rate": interest_rate,
        "explanation": explanation,
        "prediction_time_ms": elapsed_ms,
        "model_used": MODEL_DISPLAY_NAME,
        "top_features": [f[0] for f in top_features[:5]],
        "inputs": {
            "gender": gender_text,
            "own_car": own_car_text,
            "own_house": own_house_text,
            "children": children,
            "income": income,
            "income_type": income_type_text,
            "education": education_text,
            "family_status": family_status_text,
            "housing_type": housing_type_text,
            "occupation": occupation_text,
            "family_members": family_members,
            "age": age,
            "years_employed": years_employed,
        },
    }
