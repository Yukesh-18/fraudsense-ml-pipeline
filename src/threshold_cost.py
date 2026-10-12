import sqlite3
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingClassifier

# ASSUMPTION: every alert costs an analyst this much to review (change it and rerun!)

REVIEW_COST = 5.0

# 1) Read the SQL features
con = sqlite3.connect("data/fraudsense.db")
df = pd.read_sql("SELECT * FROM features", con)
con.close()
df["amount_ratio"] = df["amount"] / df["avg_amount_30d"]

FEATURES = ["amount", "is_foreign", "hour", "txn_count_1h", "txn_count_24h", "amount_ratio"]

# 2) Three time-based pieces:
#    train (first 60%) -> learn | validation (next 20%) -> pick threshold | test (last 20%) -> final check

c60 = df["ts_unix"].quantile(0.6)
c80 = df["ts_unix"].quantile(0.8)
train = df[df["ts_unix"] <= c60]
val = df[(df["ts_unix"] > c60) & (df["ts_unix"] <= c80)]
test = df[df["ts_unix"] > c80]

model = HistGradientBoostingClassifier(random_state=0)
model.fit(train[FEATURES], train["is_fraud"])
val_score = model.predict_proba(val[FEATURES])[:, 1]
test_score = model.predict_proba(test[FEATURES])[:, 1]

# 3) Cost of using a given threshold

def evaluate(data, score, t):
    alert = score >= t
    fraud = data["is_fraud"].values == 1
    amount = data["amount"].values
    review_cost = alert.sum() * REVIEW_COST
    missed_cost = amount[fraud & ~alert].sum()

    return {
        "threshold": round(float(t), 2),
        "alerts": int(alert.sum()),
        "fraud_caught": int((fraud & alert).sum()),
        "fraud_missed": int((fraud & ~alert).sum()),
        "review_cost": round(review_cost, 2),
        "missed_cost": round(float(missed_cost), 2),
        "total_cost": round(float(review_cost + missed_cost), 2),
    }

threshold = np.round(np.arange(0.01, 1.0, 0.01), 2)
val_cost = np.array([evaluate(val, val_score, t)["total_cost"] for t in threshold])
test_cost = np.array([evaluate(test, test_score, t)["total_cost"] for t in threshold])

# 4) Pick the cheapest threshold on VALIDATION data (never on the test set)

best_t = float(threshold[val_cost.argmin()])
print(f"Review cost per alert (assumed): ${REVIEW_COST}")
print(f"Best threshold chosen on validation data: {best_t}\n")

# 5) Compare strategies on the untouched TEST set

rows = [
    {"strategy": "No model (flag nothing)", **evaluate(test, test_score, 1.01)},
    {"strategy": "Flag everything", **evaluate(test, test_score, 0.0)},
    {"strategy": "Default threshold 0.5", **evaluate(test, test_score, 0.5)},
    {"strategy": f"Cost-optimised ({best_t})", **evaluate(test, test_score, best_t)},
]

result = pd.DataFrame(rows).drop(columns=["threshold"])
print(result.to_string(index=False))

no_model = rows[0]["total_cost"]
chosen = rows[3]["total_cost"]

print(f"\nSavings vs no model: ${no_model - chosen:,.2f} ({(no_model - chosen) / no_model * 100:.1f}%)")

# 6) Save a chart for your README

plt.figure(figsize=(8, 4.5))
plt.plot(threshold, val_cost, label="Validation cost")
plt.plot(threshold, test_cost, label="Test cost")
plt.axvline(best_t, color="red", linestyle="--", label=f"Chosen threshold {best_t}")
plt.xlabel("Alert threshold")
plt.ylabel("Total cost ($)")
plt.title("Total cost vs alert threshold")
plt.legend()
plt.tight_layout()
plt.savefig("data/threshold_cost.png", dpi=150)
print("Chart saved to data/threshold_cost.png")