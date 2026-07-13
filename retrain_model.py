"""
Retrains the CreditSense model artifacts using the EXACT same preprocessing
pipeline as notebook/Credit_Card_Approval_Prediction.ipynb, but under the
scikit-learn version pinned in requirements.txt (1.5.1) - eliminating the
cross-version pickle corruption that was the root cause of the flat
~50% confidence / all-High-Risk bug.

Also derives data-driven risk-band thresholds from the model's real,
honest probability distribution on held-out data, and saves them as
models/risk_thresholds.pkl so `_risk_category()` reflects what this
model can actually produce, instead of fixed thresholds tuned for a
much more separable classifier.
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
import sklearn

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report,
)

print(f"Retraining with scikit-learn {sklearn.__version__} "
      f"(must match requirements.txt's pinned version)")

# ---- Epic 1/3: Load + clean (identical to notebook) ----
application = pd.read_csv("datasets/application_record.csv").drop_duplicates()
credit = pd.read_csv("datasets/credit_record.csv").drop_duplicates()
application["OCCUPATION_TYPE"] = application["OCCUPATION_TYPE"].fillna("Unknown")

STATUS_MAP = {"X": 0, "C": 0, "0": 0, "1": 1, "2": 1, "3": 1, "4": 1, "5": 1}
credit["STATUS"] = credit["STATUS"].map(STATUS_MAP).astype(int)
credit_target = credit.groupby("ID")["STATUS"].max().reset_index()
credit_target.rename(columns={"STATUS": "TARGET"}, inplace=True)
credit_target["TARGET"] = credit_target["TARGET"].astype(int)

data = application.merge(credit_target, on="ID", how="inner")

data["AGE"] = (-data["DAYS_BIRTH"] // 365).astype(int)
data["YEARS_EMPLOYED"] = data["DAYS_EMPLOYED"].apply(lambda x: 0 if x > 0 else -x // 365).astype(int)

income_cap = data["AMT_INCOME_TOTAL"].quantile(0.99)
data["AMT_INCOME_TOTAL"] = data["AMT_INCOME_TOTAL"].clip(upper=income_cap)

DROP_COLS = ["ID", "DAYS_BIRTH", "DAYS_EMPLOYED", "FLAG_MOBIL", "FLAG_WORK_PHONE", "FLAG_PHONE", "FLAG_EMAIL"]
data.drop(columns=[c for c in DROP_COLS if c in data.columns], inplace=True)
data.drop_duplicates(inplace=True)

CATEGORICAL_COLS = [
    "CODE_GENDER", "FLAG_OWN_CAR", "FLAG_OWN_REALTY",
    "NAME_INCOME_TYPE", "NAME_EDUCATION_TYPE", "NAME_FAMILY_STATUS",
    "NAME_HOUSING_TYPE", "OCCUPATION_TYPE",
]
label_encoders = {}
for col in CATEGORICAL_COLS:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col].astype(str))
    label_encoders[col] = le

X = data.drop("TARGET", axis=1)
y = data["TARGET"].astype(int)
FEATURE_ORDER = list(X.columns)

# ---- Epic 4: identical 70/15/15 stratified split, random_state=42 ----
X_trainfull, X_test, y_trainfull, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_trainfull, y_trainfull, test_size=0.1765, random_state=42, stratify=y_trainfull
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

neg, pos = np.bincount(y_train)
scale_pos_weight = neg / pos

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced"),
    "Decision Tree": DecisionTreeClassifier(random_state=42, class_weight="balanced", max_depth=8, min_samples_leaf=20),
    "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced", max_depth=10, min_samples_leaf=10, n_jobs=-1),
}
try:
    from xgboost import XGBClassifier
    models["XGBoost"] = XGBClassifier(random_state=42, eval_metric="logloss", scale_pos_weight=scale_pos_weight, max_depth=4, n_estimators=300, learning_rate=0.1)
except ImportError:
    pass

MIN_RECALL = 0.30

def best_threshold_for(y_true, proba, min_recall=MIN_RECALL):
    best_t, best_f1 = 0.5, -1
    for t in np.arange(0.05, 0.96, 0.01):
        preds = (proba >= t).astype(int)
        if preds.sum() == 0 or preds.sum() == len(preds):
            continue
        rec = recall_score(y_true, preds, zero_division=0)
        f1m = f1_score(y_true, preds, average="macro", zero_division=0)
        if rec >= min_recall and f1m > best_f1:
            best_t, best_f1 = t, f1m
    return best_t

results = []
trained_models = {}
thresholds = {}
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    trained_models[name] = model
    val_proba = model.predict_proba(X_val_scaled)[:, 1]
    thr = best_threshold_for(y_val, val_proba)
    thresholds[name] = thr
    test_proba = model.predict_proba(X_test_scaled)[:, 1]
    preds = (test_proba >= thr).astype(int)
    results.append({
        "Model": name, "Threshold": thr,
        "Accuracy": accuracy_score(y_test, preds),
        "Recall (Rejected)": recall_score(y_test, preds, pos_label=1, zero_division=0),
        "F1 (macro)": f1_score(y_test, preds, average="macro", zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, test_proba),
    })

results_df = pd.DataFrame(results).sort_values("F1 (macro)", ascending=False)
print(results_df.to_string(index=False))
print()

eligible = results_df[results_df["Recall (Rejected)"] >= MIN_RECALL]
if eligible.empty:
    eligible = results_df
best_row = eligible.sort_values("F1 (macro)", ascending=False).iloc[0]
best_model_name = best_row["Model"]
best_model = trained_models[best_model_name]
best_threshold = float(best_row["Threshold"])
print(f"Selected model: {best_model_name} (decision threshold = {best_threshold:.2f})")

final_proba = best_model.predict_proba(X_test_scaled)[:, 1]
final_preds = (final_proba >= best_threshold).astype(int)
print(classification_report(y_test, final_preds, target_names=["Approved (0)", "Rejected (1)"]))
print("ROC-AUC on held-out test set:", round(roc_auc_score(y_test, final_proba), 3))

# ---- Refit winning model's config on train+val (85%), same as notebook ----
final_scaler = StandardScaler()
X_trainfull_scaled = final_scaler.fit_transform(X_trainfull)
final_model = best_model.__class__(**best_model.get_params())
final_model.fit(X_trainfull_scaled, y_trainfull)

# ---- NEW: derive data-driven risk-band thresholds from this model's
#      REAL probability distribution on held-out test data, instead of
#      the fixed 0.55/0.80 cutoffs that assumed a much more separable
#      classifier. Uses the 70th/30th percentile of P(Approved) so the
#      three risk bands actually get populated in realistic proportions. --
X_test_scaled_final = final_scaler.transform(X_test)
prob_approved_test = final_model.predict_proba(X_test_scaled_final)[:, 0]
low_risk_cutoff = float(np.percentile(prob_approved_test, 70))
high_risk_cutoff = float(np.percentile(prob_approved_test, 30))
risk_thresholds = {"low_risk_cutoff": low_risk_cutoff, "high_risk_cutoff": high_risk_cutoff}
print("\nCalibrated risk thresholds (from real held-out probability distribution):")
print(risk_thresholds)
print("P(Approved) distribution:", np.percentile(prob_approved_test, [0, 10, 30, 50, 70, 90, 100]))

import os
os.makedirs("models", exist_ok=True)
joblib.dump(final_model, "models/credit_card_approval_model.pkl")
joblib.dump(final_scaler, "models/scaler.pkl")
joblib.dump(label_encoders, "models/label_encoders.pkl")
joblib.dump(FEATURE_ORDER, "models/feature_order.pkl")
joblib.dump(best_threshold, "models/decision_threshold.pkl")
joblib.dump(risk_thresholds, "models/risk_thresholds.pkl")
print("\nSaved all artifacts under scikit-learn", sklearn.__version__)
