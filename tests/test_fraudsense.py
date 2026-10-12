import sqlite3
import sys

import joblib
import pandas as pd
import pytest

sys.path.insert(0, "src")   # tests run from the project's main folder

DB = "data/fraudsense.db"
FEATURES = ["amount", "is_foreign", "hour", "txn_count_1h", "txn_count_24h", "amount_ratio"]

ORDINARY = {"amount": 35.0, "is_foreign": 0, "hour": 14,
            "txn_count_1h": 0, "txn_count_24h": 1, "avg_amount_30d": 40.0}
BURST = {"amount": 850.0, "is_foreign": 1, "hour": 3,
         "txn_count_1h": 4, "txn_count_24h": 4, "avg_amount_30d": 60.0}


# ---------------------------------------------------------------- SQL features
def test_features_view_has_one_row_per_transaction():
    con = sqlite3.connect(DB)
    n_transactions = con.execute("SELECT COUNT(*) FROM transactions").fetchone()[0]
    n_features = con.execute("SELECT COUNT(*) FROM features").fetchone()[0]
    con.close()
    assert n_transactions == n_features > 0


def test_velocity_features_only_look_at_the_past():
    """A card's very first transaction has no earlier activity, so its counts must be zero."""
    con = sqlite3.connect(DB)
    bad = con.execute("""
        SELECT COUNT(*) FROM features f
        JOIN (SELECT card_id, MIN(ts_unix) AS first_ts FROM transactions GROUP BY card_id) x
          ON f.card_id = x.card_id AND f.ts_unix = x.first_ts
        WHERE f.txn_count_1h > 0 OR f.txn_count_24h > 0
    """).fetchone()[0]
    con.close()
    assert bad == 0


# ----------------------------------------------------------------------- model
def test_model_bundle_scores_a_row():
    bundle = joblib.load("models/fraud_model.joblib")
    assert bundle["features"] == FEATURES
    row = pd.DataFrame([[35.0, 0, 14, 0, 1, 0.9]], columns=FEATURES)
    assert 0 <= bundle["model"].predict_proba(row)[0, 1] <= 1


# ------------------------------------------------------------------------- API
@pytest.fixture(scope="module")
def client():
    import api
    return api.app.test_client()


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.get_json()["model_loaded"] is True


def test_ordinary_purchase_is_not_flagged(client):
    body = client.post("/score", json=ORDINARY).get_json()
    assert body["fraud_score"] < 0.1 and body["alert"] is False


def test_burst_is_flagged(client):
    body = client.post("/score", json=BURST).get_json()
    assert body["fraud_score"] > 0.8 and body["alert"] is True


def test_missing_fields_are_rejected(client):
    r = client.post("/score", json={"amount": 100})
    assert r.status_code == 400 and "missing" in r.get_json()


def test_unknown_transaction_is_404(client):
    assert client.get("/score/999999999").status_code == 404


# ------------------------------------------------------------------- knowledge
def test_rag_finds_the_velocity_policy():
    from rag import Retriever
    hits = Retriever().search("card used many times within one hour", k=5)
    assert any("Velocity rules" in h["title"] for h in hits)


def test_recommendation_follows_policy():
    import investigate
    burst = pd.Series({"txn_count_1h": 5, "amount": 800.0, "is_foreign": 0, "hour": 14})
    assert "temporary hold" in investigate.recommend(burst, True, 0.99)
    assert "No action" in investigate.recommend(burst, False, 0.001)
