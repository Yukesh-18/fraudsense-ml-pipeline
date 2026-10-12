import sqlite3
import sys
from pathlib import Path

import joblib
import pandas as pd
import requests
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
import investigate  # noqa: E402

API = "http://127.0.0.1:8000"

st.set_page_config(page_title="FraudSense", page_icon="🛡️", layout="wide")
st.title("🛡️ FraudSense: fraud detection and investigation")
st.caption("Demo project on synthetic data. Run the API first:  py src/api.py")


@st.cache_data
def load_data():
    con = sqlite3.connect("data/fraudsense.db")
    df = pd.read_sql("SELECT * FROM features", con)
    con.close()
    df["amount_ratio"] = df["amount"] / df["avg_amount_30d"]
    return df


@st.cache_resource
def load_bundle():
    return joblib.load("models/fraud_model.joblib")


@st.cache_data
def score_test_slice():
    """Score the last 20% of time, which the production model never trained on."""
    df, bundle = load_data(), load_bundle()
    test = df[df["ts_unix"] > df["ts_unix"].quantile(0.8)].copy()
    test["fraud_score"] = bundle["model"].predict_proba(test[bundle["features"]])[:, 1]
    test["expected_loss"] = test["fraud_score"] * test["amount"]
    test["alert"] = test["expected_loss"] > bundle["review_cost"]
    return test


tab_overview, tab_alerts, tab_try, tab_results = st.tabs(
    ["Overview", "Alert queue", "Try a transaction", "Model results"])

# ---------------------------------------------------------------- Overview
with tab_overview:
    df = load_data()
    c1, c2, c3 = st.columns(3)
    c1.metric("Transactions", f"{len(df):,}")
    c2.metric("Fraud rate", f"{df['is_fraud'].mean() * 100:.2f}%")
    c3.metric("Total fraud amount", f"${df.loc[df['is_fraud'] == 1, 'amount'].sum():,.0f}")

    left, right = st.columns(2)
    with left:
        st.subheader("Fraud rate by hour of day (%)")
        st.bar_chart(df.groupby("hour")["is_fraud"].mean() * 100)
    with right:
        st.subheader("Fraud rate by card uses in the last hour (%)")
        st.caption("5 means 5 or more. This is the burst pattern the SQL features capture.")
        st.bar_chart(df.groupby(df["txn_count_1h"].clip(upper=5))["is_fraud"].mean() * 100)

# ------------------------------------------------------------- Alert queue
with tab_alerts:
    test = score_test_slice()
    alerts = test[test["alert"]]
    c1, c2, c3 = st.columns(3)
    c1.metric("Alerts raised", f"{len(alerts):,}")
    c2.metric("Real fraud among alerts", f"{int(alerts['is_fraud'].sum())}")
    c3.metric("Alert precision", f"{alerts['is_fraud'].mean() * 100:.0f}%" if len(alerts) else "n/a")
    st.caption("Alert rule: fraud probability x amount is bigger than the review cost. "
               "Shown for the last 20% of time, which the model has not seen.")

    top_n = st.slider("How many alerts to show", 10, 200, 25, step=5)
    queue = alerts.sort_values("expected_loss", ascending=False).head(top_n)
    st.dataframe(
        queue[["txn_id", "card_id", "amount", "fraud_score", "expected_loss", "is_fraud"]]
        .rename(columns={"is_fraud": "actual_fraud (demo)"}).round(2),
        hide_index=True)

    st.subheader("Investigate an alert")
    txn_id = st.selectbox("Transaction", queue["txn_id"].tolist())
    if st.button("Run investigation"):
        with st.spinner("Scoring, explaining, searching the knowledge base..."):
            try:
                st.session_state["report"] = investigate.build_report(int(txn_id))
            except investigate.InvestigationError as e:
                st.session_state["report"] = None
                st.error(str(e))
    if st.session_state.get("report"):
        st.markdown(st.session_state["report"])

# --------------------------------------------------------- Try a transaction
with tab_try:
    st.write("Type in a transaction and the Flask API scores it.")
    a, b, c = st.columns(3)
    amount = a.number_input("Amount ($)", min_value=0.01, value=850.0)
    avg_amount = b.number_input("Card's 30-day average ($)", min_value=0.01, value=60.0)
    hour = c.slider("Hour of day", 0, 23, 3)
    d, e, f = st.columns(3)
    count_1h = d.number_input("Card uses in the last hour", min_value=0, value=4)
    count_24h = e.number_input("Card uses in the last 24 hours", min_value=0, value=4)
    foreign = f.checkbox("Foreign transaction", value=True)

    if st.button("Score it"):
        payload = {"amount": amount, "is_foreign": int(foreign), "hour": hour, "txn_count_1h": count_1h,
                   "txn_count_24h": count_24h, "avg_amount_30d": avg_amount}
        try:
            r = requests.post(f"{API}/score", json=payload, timeout=10)
            res = r.json()
            if r.status_code != 200:
                st.error(res)
            else:
                m1, m2, m3 = st.columns(3)
                m1.metric("Fraud score", f"{res['fraud_score']:.3f}")
                m2.metric("Expected loss", f"${res['expected_loss']:,.2f}")
                m3.metric("Alert", "YES" if res["alert"] else "no")
        except requests.exceptions.ConnectionError:
            st.error("The API is not running. Start it in another terminal with:  py src/api.py")

# ------------------------------------------------------------ Model results
with tab_results:
    board = next((p for p in ["data/model_leaderboard.csv", "data/baseline_results.csv"] if Path(p).exists()), None)
    if board:
        st.subheader("Model leaderboard (test set)")
        st.dataframe(pd.read_csv(board), hide_index=True)
    if Path("data/threshold_results.csv").exists():
        st.subheader("Cost of each alert strategy (test set)")
        st.dataframe(pd.read_csv("data/threshold_results.csv"), hide_index=True)
    left, right = st.columns(2)
    if Path("data/threshold_cost.png").exists():
        left.image("data/threshold_cost.png", caption="Total cost vs alert threshold")
    if Path("data/shap_importance.png").exists():
        right.image("data/shap_importance.png", caption="What drives the model")
    if not board:
        st.info("Run train_baselines.py, train_nn.py and threshold_cost.py to fill this tab.")