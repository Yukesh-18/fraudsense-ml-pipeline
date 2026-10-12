import sqlite3
from datetime import datetime, timezone

import joblib
import pandas as pd
from flask import Flask, jsonify, request

DB = "data/fraudsense.db"

# Load the production model that train_final.py saved
bundle = joblib.load("models/fraud_model.joblib")
MODEL = bundle["model"]
FEATURES = bundle["features"]
REVIEW_COST = bundle["review_cost"]

REQUIRED = ["amount", "is_foreign", "hour", "txn_count_1h", "txn_count_24h", "avg_amount_30d"]

# Keep the SQL feature table in memory so we can look transactions up by id instantly
_con = sqlite3.connect(DB)
FEATURE_TABLE = pd.read_sql("SELECT * FROM features", _con).set_index("txn_id")
_con.close()

app = Flask(__name__)


def score_row(row):
    """Turn raw transaction facts into a fraud score and an alert decision."""
    amount = float(row["amount"])
    avg = float(row["avg_amount_30d"])
    x = pd.DataFrame([{
        "amount": amount,
        "is_foreign": int(row["is_foreign"]),
        "hour": int(row["hour"]),
        "txn_count_1h": float(row["txn_count_1h"]),
        "txn_count_24h": float(row["txn_count_24h"]),
        "amount_ratio": amount / avg if avg > 0 else 1.0,
    }])[FEATURES]
    p = float(MODEL.predict_proba(x)[0, 1])
    expected_loss = p * amount
    return {
        "fraud_score": round(p, 4),
        "expected_loss": round(expected_loss, 2),
        "alert": bool(expected_loss > REVIEW_COST),
        "review_cost": REVIEW_COST,
    }


def log_prediction(txn_id, amount, result):
    """Audit trail: every prediction is written to the database."""
    con = sqlite3.connect(DB)
    con.execute("CREATE TABLE IF NOT EXISTS predictions ("
                "created_at TEXT, txn_id INTEGER, amount REAL, "
                "fraud_score REAL, expected_loss REAL, alert INTEGER)")
    con.execute("INSERT INTO predictions VALUES (?, ?, ?, ?, ?, ?)",
                (datetime.now(timezone.utc).isoformat(), txn_id, amount,
                 result["fraud_score"], result["expected_loss"], int(result["alert"])))
    con.commit()
    con.close()


@app.get("/health")
def health():
    return jsonify({"status": "ok", "model_loaded": True, "transactions_in_memory": len(FEATURE_TABLE)})


@app.post("/score")
def score():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "Send a JSON object in the request body."}), 400
    missing = [k for k in REQUIRED if k not in data]
    if missing:
        return jsonify({"error": "Missing fields", "missing": missing, "required": REQUIRED}), 400
    try:
        amount = float(data["amount"])
        hour = int(data["hour"])
        result_input = {k: float(data[k]) for k in REQUIRED}
    except (TypeError, ValueError):
        return jsonify({"error": "All fields must be numbers."}), 400
    if amount <= 0 or not 0 <= hour <= 23:
        return jsonify({"error": "amount must be > 0 and hour must be between 0 and 23."}), 400

    result = score_row(result_input)
    log_prediction(None, amount, result)
    return jsonify(result)


@app.get("/score/<int:txn_id>")
def score_by_id(txn_id):
    if txn_id not in FEATURE_TABLE.index:
        return jsonify({"error": f"Transaction {txn_id} not found."}), 404
    row = FEATURE_TABLE.loc[txn_id]
    result = score_row(row)
    result["txn_id"] = txn_id
    result["card_id"] = int(row["card_id"])
    result["amount"] = float(row["amount"])
    log_prediction(txn_id, float(row["amount"]), result)
    return jsonify(result)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000, debug=False)