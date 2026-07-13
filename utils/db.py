"""
Lightweight SQLite persistence layer for CreditSense.

Stores every prediction made through the app so the Dashboard and
History pages have real (if demo-generated) data to visualize.
"""

import sqlite3
import os
from datetime import datetime

_DB_PATH = None


def init_db(db_path):
    """Create the database file & schema if it doesn't already exist."""
    global _DB_PATH
    _DB_PATH = db_path
    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            applicant_name TEXT,
            gender TEXT,
            age REAL,
            income REAL,
            income_type TEXT,
            education TEXT,
            family_status TEXT,
            housing_type TEXT,
            occupation TEXT,
            children INTEGER,
            family_members INTEGER,
            years_employed REAL,
            own_car TEXT,
            own_house TEXT,
            model_used TEXT,
            prediction TEXT,
            approved INTEGER,
            probability REAL,
            confidence REAL,
            risk_category TEXT,
            credit_limit REAL,
            interest_rate REAL
        )
        """
    )
    conn.commit()
    conn.close()


def get_conn():
    conn = sqlite3.connect(_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def insert_prediction(record):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO predictions (
            created_at, applicant_name, gender, age, income, income_type,
            education, family_status, housing_type, occupation, children,
            family_members, years_employed, own_car, own_house, model_used,
            prediction, approved, probability, confidence, risk_category,
            credit_limit, interest_rate
        ) VALUES (
            :created_at, :applicant_name, :gender, :age, :income, :income_type,
            :education, :family_status, :housing_type, :occupation, :children,
            :family_members, :years_employed, :own_car, :own_house, :model_used,
            :prediction, :approved, :probability, :confidence, :risk_category,
            :credit_limit, :interest_rate
        )
        """,
        record,
    )
    conn.commit()
    new_id = cur.lastrowid
    conn.close()
    return new_id


def get_prediction(pred_id):
    conn = get_conn()
    row = conn.execute("SELECT * FROM predictions WHERE id = ?", (pred_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def get_all_predictions(limit=500):
    conn = get_conn()
    rows = conn.execute(
        "SELECT * FROM predictions ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_stats():
    conn = get_conn()
    total = conn.execute("SELECT COUNT(*) c FROM predictions").fetchone()["c"]
    approved = conn.execute(
        "SELECT COUNT(*) c FROM predictions WHERE approved = 1"
    ).fetchone()["c"]
    avg_income = conn.execute(
        "SELECT AVG(income) a FROM predictions"
    ).fetchone()["a"]
    avg_age = conn.execute("SELECT AVG(age) a FROM predictions").fetchone()["a"]
    latest = conn.execute(
        "SELECT * FROM predictions ORDER BY id DESC LIMIT 1"
    ).fetchone()
    conn.close()

    rejected = total - approved
    approval_rate = (approved / total * 100) if total else 0
    rejection_rate = (rejected / total * 100) if total else 0

    return {
        "total": total,
        "approved": approved,
        "rejected": rejected,
        "approval_rate": round(approval_rate, 1),
        "rejection_rate": round(rejection_rate, 1),
        "avg_income": round(avg_income, 2) if avg_income else 0,
        "avg_age": round(avg_age, 1) if avg_age else 0,
        "latest": dict(latest) if latest else None,
    }
