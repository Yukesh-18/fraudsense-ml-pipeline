import sqlite3
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
from sklearn.ensemble import GradientBoostingClassifier

# 1) Read the SQL features and use the same time-based split as before
con = sqlite3.connect("data/fraudsense.db")
df = pd.read_sql("SELECT * FROM features", con)
con.close()
df["amount_ratio"] = df["amount"] / df["avg_amount_30d"]

FEATURES = ["amount", "is_foreign", "hour",
            "txn_count_1h", "txn_count_24h", "amount_ratio"]

LABELS = {
    "amount": "transaction amount",
    "is_foreign": "foreign transaction (1 = yes)",
    "hour": "hour of day",
    "txn_count_1h": "card transactions in the last hour",
    "txn_count_24h": "card transactions in the last 24 hours",
    "amount_ratio": "amount vs this card's 30-day average",
}

cut = df["ts_unix"].quantile(0.8)
train = df[df["ts_unix"] <= cut]
test = df[df["ts_unix"] > cut].copy()

# 2) Train the model
print("Training model...")
model = GradientBoostingClassifier(n_estimators=150, max_depth=3, random_state=0)
model.fit(train[FEATURES], train["is_fraud"])

test["fraud_score"] = model.predict_proba(test[FEATURES])[:, 1]
test["expected_loss"] = test["fraud_score"] * test["amount"]


# 3) SHAP, model-agnostic version. Values are in "fraud probability" units:
#    +0.30 means "this feature pushed the fraud probability up by 30 points".
def predict_fraud(X):
    return model.predict_proba(pd.DataFrame(X, columns=FEATURES))[:, 1]


background = train[FEATURES].sample(50, random_state=0)
masker = shap.maskers.Independent(background, max_samples=50)
explainer = shap.explainers.Exact(predict_fraud, masker)


def explain(X):
    exp = explainer(X)
    values = np.asarray(exp.values)
    base = np.asarray(exp.base_values).reshape(-1)
    # SELF-CHECK: base value + all feature contributions must equal the model's score
    gap = np.abs(base + values.sum(axis=1) - predict_fraud(X)).max()
    if gap > 1e-3:
        raise RuntimeError(f"SHAP values do not add up to the score (gap {gap}). Do not trust them.")
    return values, gap


# 4) Explain the 5 transactions with the highest expected loss
top = test.sort_values("expected_loss", ascending=False).head(5)
sv_top, gap = explain(top[FEATURES])
print(f"Self-check passed (largest gap {gap:.1e})")

records = []
print("\nWHY WAS EACH TRANSACTION FLAGGED?\n")
for i, (_, row) in enumerate(top.iterrows()):
    order = np.argsort(-np.abs(sv_top[i]))[:3]
    reasons = []
    for j in order:
        f = FEATURES[j]
        direction = "RAISES" if sv_top[i][j] > 0 else "LOWERS"
        reasons.append(f"{LABELS[f]} = {row[f]:.4g} -> {direction} risk ({sv_top[i][j]:+.2f})")
    actual = "yes" if row["is_fraud"] == 1 else "no"
    print(f"Transaction {int(row['txn_id'])} | card {int(row['card_id'])} | "
          f"${row['amount']:.2f} | fraud score {row['fraud_score']:.2f} | actually fraud: {actual}")
    for n, r in enumerate(reasons, 1):
        print(f"   {n}. {r}")
    print()
    records.append({
        "txn_id": int(row["txn_id"]), "card_id": int(row["card_id"]),
        "amount": row["amount"], "fraud_score": round(row["fraud_score"], 3),
        "is_fraud": int(row["is_fraud"]),
        "reason_1": reasons[0], "reason_2": reasons[1], "reason_3": reasons[2],
    })

pd.DataFrame(records).to_csv("data/explanations.csv", index=False)

# 5) Global view: which features matter most overall?
print("Computing overall importance (about a minute)...")
sample = test.sample(min(300, len(test)), random_state=0)
sv_all, gap_all = explain(sample[FEATURES])
importance = pd.Series(np.abs(sv_all).mean(axis=0), index=FEATURES).sort_values()
print("\nOVERALL FEATURE IMPORTANCE (average change in fraud probability)")
print(importance.sort_values(ascending=False).to_string(float_format="{:.3f}".format))

plt.figure(figsize=(8, 4.5))
plt.barh([LABELS[f] for f in importance.index], importance.values)
plt.xlabel("Mean |SHAP value| (average change in fraud probability)")
plt.title("What drives the fraud model?")
plt.tight_layout()
plt.savefig("data/shap_importance.png", dpi=150)
print("\nSaved: data/explanations.csv and data/shap_importance.png")