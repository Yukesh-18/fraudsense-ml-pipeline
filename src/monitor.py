import sqlite3
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import average_precision_score

DB = "data/fraudsense.db"
PSI_WATCH, PSI_ALERT = 0.10, 0.20   # common rule of thumb for the Population Stability Index
PR_DROP_LIMIT = 0.25                # performance counts as degraded if PR-AUC falls by more than 25%
KS_LARGE = 0.30                     # a KS statistic of 0.3 or more is a large shift in a distribution

bundle = joblib.load("models/fraud_model.joblib")
MODEL, FEATURES, REVIEW_COST = bundle["model"], bundle["features"], bundle["review_cost"]

# 1) Data: reference = what the model trained on, current = the most recent period
con = sqlite3.connect(DB)
df = pd.read_sql("SELECT * FROM features", con)
con.close()
df["amount_ratio"] = df["amount"] / df["avg_amount_30d"]
reference = df[df["ts_unix"] <= df["ts_unix"].quantile(0.6)]
current = df[df["ts_unix"] > df["ts_unix"].quantile(0.8)]


# 2) Three simulated futures
def make_scenario(name):
    rng = np.random.default_rng(1)
    d = current.copy()
    if name == "data_drift":
        # prices inflate and more foreign purchases, but fraud works the same way
        d["amount"] = d["amount"] * 1.8
        d.loc[rng.random(len(d)) < 0.15, "is_foreign"] = 1
        d["amount_ratio"] = d["amount"] / d["avg_amount_30d"]
    elif name == "concept_drift":
        # fraudsters slow down: the burst pattern disappears from fraud rows
        slowed = (d["is_fraud"] == 1) & (d["txn_count_1h"] >= 2)
        d.loc[slowed, "txn_count_1h"] = 0
        d.loc[slowed, "txn_count_24h"] = d.loc[slowed, "txn_count_24h"].clip(upper=1)
    return d


SCENARIOS = {
    "normal": "Nothing changed",
    "data_drift": "Prices inflate and foreign purchases rise",
    "concept_drift": "Fraudsters stop using bursts",
}


# 3) Drift measures from scipy and numpy
def psi(ref, cur):
    """Population Stability Index: 0 = identical, above 0.2 = major shift."""
    ref, cur = np.asarray(ref, float), np.asarray(cur, float)
    if len(np.unique(ref)) <= 12:   # few distinct values: compare them directly
        values = np.unique(np.concatenate([ref, cur]))
        p = np.array([(ref == v).mean() for v in values])
        q = np.array([(cur == v).mean() for v in values])
    else:                           # many values: ten bins cut on the reference
        edges = np.unique(np.quantile(ref, np.linspace(0, 1, 11)))
        edges[0], edges[-1] = -np.inf, np.inf
        p = np.histogram(ref, edges)[0] / len(ref)
        q = np.histogram(cur, edges)[0] / len(cur)
    p, q = np.clip(p, 1e-4, None), np.clip(q, 1e-4, None)
    return float(np.sum((q - p) * np.log(q / p)))


def feature_drift(ref, cur):
    rows = []
    for f in FEATURES:
        ks = stats.ks_2samp(ref[f], cur[f])   # scipy two-sample Kolmogorov-Smirnov test
        rows.append({"feature": f, "psi": psi(ref[f], cur[f]),
                     "ks_statistic": float(ks.statistic), "ks_pvalue": float(ks.pvalue)})
    return pd.DataFrame(rows)


def performance(d):
    score = MODEL.predict_proba(d[FEATURES])[:, 1]
    amount, fraud = d["amount"].values, d["is_fraud"].values == 1
    alert = score * amount > REVIEW_COST
    cost = alert.sum() * REVIEW_COST + amount[fraud & ~alert].sum()
    no_model = amount[fraud].sum()
    return {"pr_auc": average_precision_score(fraud, score), "alerts": int(alert.sum()),
            "fraud_caught": int((fraud & alert).sum()), "fraud_total": int(fraud.sum()),
            "cost": float(cost), "savings_pct": float((no_model - cost) / no_model * 100)}


# 4) ModelDoctor: turn the numbers into a diagnosis
def diagnose(drift, fraud_drift, perf, baseline_pr):
    drifted = drift.loc[drift["psi"] >= PSI_ALERT, "feature"].tolist()
    watch = drift.loc[(drift["psi"] >= PSI_WATCH) & (drift["psi"] < PSI_ALERT), "feature"].tolist()
    # fraud rows are few, so use the KS test p-value (it accounts for sample size) rather than PSI
    fraud_drifted = fraud_drift.loc[(fraud_drift["ks_pvalue"] < 0.001) & (fraud_drift["ks_statistic"] >= KS_LARGE),
                                    "feature"].tolist()
    drop = (baseline_pr - perf["pr_auc"]) / baseline_pr
    degraded = drop > PR_DROP_LIMIT

    if degraded and fraud_drifted and not drifted:
        status = "DEGRADED"
        cause = ("Concept drift. Overall features look stable, but confirmed fraud now looks different "
                 f"on: {', '.join(fraud_drifted)}. The pattern the model learned no longer matches how fraud behaves.")
        actions = ["Retrain on recent labelled data so it includes the new fraud behaviour.",
                   "Review the missed fraud cases and add features that capture the new tactic.",
                   "Until retraining is done, lower the review cost in the alert rule to catch more borderline cases."]
    elif degraded and drifted:
        status = "DEGRADED"
        cause = f"Data drift is hurting performance. Inputs have shifted on: {', '.join(drifted)}."
        actions = ["Retrain on recent data so the model sees the new ranges.",
                   "Check whether the shift is real (prices, customers) or a data pipeline fault."]
    elif degraded:
        status = "DEGRADED"
        cause = "Performance fell without a clear feature shift. Check label delays and data quality first."
        actions = ["Verify the labels and the SQL feature pipeline.", "Retrain if the drop persists."]
    elif drifted:
        status = "WATCH"
        cause = (f"Data drift without damage so far. Inputs shifted on: {', '.join(drifted)}, "
                 "but the model still performs close to its baseline.")
        actions = ["Keep the model, and recheck weekly.", "Retrain if performance starts to fall."]
    elif watch or fraud_drifted:
        status = "WATCH"
        cause = f"Small shifts on: {', '.join(watch + fraud_drifted)}. No action needed yet."
        actions = ["Recheck next period."]
    else:
        status = "HEALTHY"
        cause = "No meaningful drift and performance is at its baseline."
        actions = ["No action needed."]
    return {"status": status, "cause": cause, "actions": actions, "pr_drop_pct": drop * 100,
            "drifted": drifted, "fraud_drifted": fraud_drifted}


# 5) Run all scenarios
ref_fraud = reference[reference["is_fraud"] == 1]
baseline_pr = performance(make_scenario("normal"))["pr_auc"]
Path("reports").mkdir(exist_ok=True)

summary, psi_table = [], {}
for name, story in SCENARIOS.items():
    cur = make_scenario(name)
    drift = feature_drift(reference, cur)
    fraud_drift = feature_drift(ref_fraud, cur[cur["is_fraud"] == 1])
    perf = performance(cur)
    dx = diagnose(drift, fraud_drift, perf, baseline_pr)
    psi_table[name] = drift.set_index("feature")["psi"]

    lines = [f"# Model health report: {name}", f"Scenario: {story}", "",
             f"**Status: {dx['status']}**", "", dx["cause"], "",
             "## Evidence",
             f"- PR-AUC {perf['pr_auc']:.3f} against a baseline of {baseline_pr:.3f} "
             f"({-dx['pr_drop_pct']:+.0f}% change)",
             f"- Caught {perf['fraud_caught']} of {perf['fraud_total']} fraud cases with {perf['alerts']} alerts",
             f"- Cost ${perf['cost']:,.0f}, which is {perf['savings_pct']:.0f}% below doing nothing",
             "- Feature drift (PSI, all transactions): "
             + ", ".join(f"{r.feature} {r.psi:.2f}" for r in drift.itertuples()),
             "- Feature drift (KS statistic, fraud rows only; flagged when p < 0.001 and statistic >= 0.3): "
             + ", ".join(f"{r.feature} {r.ks_statistic:.2f}" for r in fraud_drift.itertuples()),
             "- Dollar costs scale with prices, so compare the savings percentage between scenarios, not the raw cost.",
             "", "## Recommended actions"] + [f"{i}. {a}" for i, a in enumerate(dx["actions"], 1)]
    Path(f"reports/model_health_{name}.md").write_text("\n".join(lines), encoding="utf-8")
    summary.append({"scenario": name, "status": dx["status"],
                    "max_psi": round(drift["psi"].max(), 2),
                    "pr_auc": round(perf["pr_auc"], 3), "pr_auc_change_pct": round(-dx["pr_drop_pct"]),
                    "fraud_caught": f"{perf['fraud_caught']}/{perf['fraud_total']}",
                    "cost": round(perf["cost"]), "savings_pct": round(perf["savings_pct"]),
                    "diagnosis": dx["cause"]})

summary = pd.DataFrame(summary)
psi_df = pd.DataFrame(psi_table)
summary.to_csv("data/drift_summary.csv", index=False)
psi_df.round(3).to_csv("data/drift_psi.csv")

print(summary.drop(columns="diagnosis").to_string(index=False))
print()
for _, r in summary.iterrows():
    print(f"[{r['scenario']}] {r['status']}: {r['diagnosis']}")

# 6) Chart for the README
ax = psi_df.plot.bar(figsize=(9, 4.8))
ax.axhline(PSI_WATCH, color="orange", linestyle="--", label="watch (0.10)")
ax.axhline(PSI_ALERT, color="red", linestyle="--", label="major shift (0.20)")
ax.set_ylabel("Population Stability Index")
ax.set_title("How far each feature has drifted from the training data")
ax.legend()
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig("data/drift_psi.png", dpi=150)
print("\nSaved: data/drift_summary.csv, data/drift_psi.csv, data/drift_psi.png, reports/model_health_*.md")