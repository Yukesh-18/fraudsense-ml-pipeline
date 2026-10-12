"""Linear regression spend model: is a transaction's amount surprising for this card, hour and place?

It predicts log(amount), and the 'spend surprise' is how far the real amount lands from the prediction.
We then test, honestly, whether adding that feature helps the fraud model.
"""
import sqlite3

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import average_precision_score, r2_score

REVIEW_COST = 5.0

con = sqlite3.connect("data/fraudsense.db")
df = pd.read_sql("SELECT * FROM features", con)
con.close()
df["amount_ratio"] = df["amount"] / df["avg_amount_30d"]
df["log_amount"] = np.log(df["amount"])
df["log_avg"] = np.log(df["avg_amount_30d"])
df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

c60, c80 = df["ts_unix"].quantile(0.6), df["ts_unix"].quantile(0.8)
train = df[df["ts_unix"] <= c60]
val = df[(df["ts_unix"] > c60) & (df["ts_unix"] <= c80)]
test = df[df["ts_unix"] > c80]

# 1) Linear regression: expected log(amount) from the card's history, hour and place
LR_FEATURES = ["log_avg", "is_foreign", "hour_sin", "hour_cos", "txn_count_24h"]
lr = LinearRegression().fit(train[LR_FEATURES], train["log_amount"])
resid = df["log_amount"] - lr.predict(df[LR_FEATURES])
df["spend_surprise"] = resid / resid[df["ts_unix"] <= c60].std()   # in standard deviations

print("LINEAR REGRESSION (predicting log amount)")
print(f"  R-squared on validation data: {r2_score(val['log_amount'], lr.predict(val[LR_FEATURES])):.3f}")
for name, coef in zip(LR_FEATURES, lr.coef_):
    print(f"  coefficient {name:15s} {coef:+.3f}")

train = df[df["ts_unix"] <= c60]
val = df[(df["ts_unix"] > c60) & (df["ts_unix"] <= c80)]
test = df[df["ts_unix"] > c80]
fraud, normal = test[test["is_fraud"] == 1], test[test["is_fraud"] == 0]
print("\nSPEND SURPRISE ON THE TEST SET (standard deviations from the prediction)")
print(f"  average for fraud rows: {fraud['spend_surprise'].mean():+.2f}   for normal rows: {normal['spend_surprise'].mean():+.2f}")
print(f"  PR-AUC of spend surprise alone: {average_precision_score(test['is_fraud'], test['spend_surprise']):.3f} "
      f"(a coin flip scores {test['is_fraud'].mean():.3f})")

# 2) Does the fraud model improve when it can see the new feature? Five seeds, same split.
BASE = ["amount", "is_foreign", "hour", "txn_count_1h", "txn_count_24h", "amount_ratio"]
WITH = BASE + ["spend_surprise"]


def policy_cost(data, score):
    amount, is_fraud = data["amount"].values, data["is_fraud"].values == 1
    alert = score * amount > REVIEW_COST
    return alert.sum() * REVIEW_COST + amount[is_fraud & ~alert].sum()


rows = []
for label, feats in [("without spend_surprise", BASE), ("with spend_surprise", WITH)]:
    prs, costs = [], []
    for seed in range(5):
        m = HistGradientBoostingClassifier(learning_rate=0.05, max_depth=4, max_iter=200, random_state=seed)
        m.fit(train[feats], train["is_fraud"])
        s = m.predict_proba(test[feats])[:, 1]
        prs.append(average_precision_score(test["is_fraud"], s))
        costs.append(policy_cost(test, s))
    rows.append({"fraud model": label, "test PR-AUC (mean)": np.mean(prs), "PR-AUC spread": np.std(prs),
                 "test cost (mean $)": np.mean(costs), "cost spread $": np.std(costs)})

result = pd.DataFrame(rows).round(3)
print("\nDOES THE NEW FEATURE HELP? (5 random seeds each)")
print(result.to_string(index=False))
result.to_csv("data/spend_model_results.csv", index=False)

gain = rows[1]["test PR-AUC (mean)"] - rows[0]["test PR-AUC (mean)"]
spread = max(rows[0]["PR-AUC spread"], rows[1]["PR-AUC spread"])
print("\nVERDICT:", "the gain is larger than the seed-to-seed noise." if gain > 2 * spread
      else "no clear gain: the difference is within seed-to-seed noise, so we leave the production model unchanged.")