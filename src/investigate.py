import sqlite3
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import requests
import shap

from rag import Retriever

API = "http://127.0.0.1:8000"
DB = "data/fraudsense.db"

LABELS = {
    "amount": "transaction amount",
    "is_foreign": "foreign transaction (1 = yes)",
    "hour": "hour of day",
    "txn_count_1h": "card transactions in the last hour",
    "txn_count_24h": "card transactions in the last 24 hours",
    "amount_ratio": "amount vs this card's 30-day average",
}


class InvestigationError(Exception):
    """Raised when a report cannot be built (API down, unknown transaction id...)."""


_cache = {}


def get_score_from_api(txn_id):
    try:
        r = requests.get(f"{API}/score/{txn_id}", timeout=10)
    except requests.exceptions.ConnectionError:
        raise InvestigationError("The API is not running. Start it in another terminal with:  py src/api.py") from None
    if r.status_code == 404:
        raise InvestigationError(f"Transaction {txn_id} was not found in the database.")
    return r.json()


def load_features():
    if "df" not in _cache:
        con = sqlite3.connect(DB)
        df = pd.read_sql("SELECT * FROM features", con)
        con.close()
        df["amount_ratio"] = df["amount"] / df["avg_amount_30d"]
        _cache["df"] = df
    return _cache["df"]


def explain(row, df, model, features):
    """Why did the model give this score? Exact SHAP values in fraud-probability units."""
    def predict(X):
        return model.predict_proba(pd.DataFrame(X, columns=features))[:, 1]

    background = df[features].sample(50, random_state=0)
    explainer = shap.explainers.Exact(predict, shap.maskers.Independent(background, max_samples=50))
    exp = explainer(row[features])
    values = np.asarray(exp.values)[0]
    base = float(np.asarray(exp.base_values).reshape(-1)[0])
    gap = abs(base + values.sum() - predict(row[features])[0])
    if gap > 1e-3:  # self-check: contributions must add up to the real score
        raise InvestigationError("Explanation does not add up to the score - do not trust it.")
    order = np.argsort(-np.abs(values))
    return [(features[j], values[j]) for j in order if abs(values[j]) >= 0.01][:4]


def build_query(r):
    """Turn the transaction's facts into a search question for the knowledge base."""
    parts = []
    if r["txn_count_1h"] >= 3:
        parts.append("many transactions on the same card within one hour velocity card testing")
    if r["is_foreign"] == 1:
        parts.append("foreign transaction")
    if r["hour"] <= 4:
        parts.append("night-time activity midnight 5 am")
    if r["amount_ratio"] > 5:
        parts.append("amount much higher than the card's average spend")
    if r["amount"] > 500:
        parts.append("high-value transaction escalate senior analyst")
    return " ".join(parts) or "standard review process false positive"


def recommend(r, alert, score):
    """Rule-based recommendation that follows the sample policy."""
    if not alert:
        return "No action needed. Keep standard monitoring (expected loss is below the review cost)."
    steps = []
    if r["txn_count_1h"] >= 4:
        steps.append("Treat as card testing: apply a temporary hold (policy 1) and contact the cardholder "
                     "on the registered phone or app (policy 6).")
    elif score >= 0.5:
        steps.append("Priority review within 15 minutes (policy 5).")
    else:
        steps.append(f"Low fraud probability ({score:.2f}); flagged mainly because the amount is large. "
                     "Standard review within 4 hours (policy 5), no hold needed.")
    if r["amount"] > 1000:
        steps.append("Amount is above 1,000 USD: escalate to a senior analyst (policy 2).")
    if r["is_foreign"] == 1 and r["hour"] <= 4:
        steps.append("Foreign and night-time: check travel notices and app location before blocking (case 2025-021).")
    steps.append("Record the transaction ids, score, top factors, rule matched and decision (policy 8).")
    return "\n".join(f"{i}. {s}" for i, s in enumerate(steps, 1))


def short(text, n=230):
    return text if len(text) <= n else text[:n].rstrip() + "..."


def build_report(txn_id):
    """Run the full investigation for one transaction and return the report as text."""
    result = get_score_from_api(txn_id)
    df = load_features()
    row = df[df["txn_id"] == txn_id]
    r = row.iloc[0]

    if "bundle" not in _cache:
        _cache["bundle"] = joblib.load("models/fraud_model.joblib")
    bundle = _cache["bundle"]
    reasons = explain(row, df, bundle["model"], bundle["features"])

    if "retriever" not in _cache:
        _cache["retriever"] = Retriever()
    hits = _cache["retriever"].search(build_query(r), k=8)
    policies = [h for h in hits if h["source"] == "fraud_policy.md"][:2]
    cases = [h for h in hits if h["source"] == "case_notes.md"][:2]

    lines = [
        f"# Investigation report: transaction {txn_id}",
        f"Card {int(r['card_id'])} | ${r['amount']:.2f} | hour {int(r['hour'])} | "
        f"foreign: {'yes' if r['is_foreign'] == 1 else 'no'}",
        "",
        "## Model assessment",
        f"Fraud score {result['fraud_score']:.2f} | expected loss ${result['expected_loss']:.2f} | "
        f"alert: {'YES' if result['alert'] else 'no'} (review cost ${result['review_cost']:.0f})",
        "",
        "## Why the model scored it this way",
    ]
    for n, (f, v) in enumerate(reasons, 1):
        direction = "raises" if v > 0 else "lowers"
        lines.append(f"{n}. {LABELS[f]} = {r[f]:.4g} -> {direction} the risk ({v:+.2f})")
    lines += ["", "## Relevant policy"]
    lines += [f"- [{p['title']}] {short(p['text'])}" for p in policies] or ["- (nothing relevant found)"]
    lines += ["", "## Similar past cases"]
    lines += [f"- [{c['title']}] {short(c['text'])}" for c in cases] or ["- (nothing relevant found)"]
    lines += ["", "## Recommended action (rule-based)", recommend(r, result["alert"], result["fraud_score"]), "",
              "## Demo-only check",
              f"Actual label in the dataset: {'FRAUD' if r['is_fraud'] == 1 else 'not fraud'}. "
              "(In real life this is unknown at decision time.)"]

    report = "\n".join(lines)
    Path("reports").mkdir(exist_ok=True)
    Path(f"reports/case_{txn_id}.md").write_text(report, encoding="utf-8")
    return report


def main():
    txn_id = int(sys.argv[1]) if len(sys.argv) > 1 else 48612
    try:
        report = build_report(txn_id)
    except InvestigationError as e:
        sys.exit(str(e))
    print(report)
    print(f"\nSaved: reports/case_{txn_id}.md")


if __name__ == "__main__":
    main()
