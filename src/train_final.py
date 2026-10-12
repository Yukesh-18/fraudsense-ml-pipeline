import os
import sqlite3
import numpy as np
import pandas as pd
import joblib
import mlflow
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

# ASSUMPTION: every alert costs an analyst this much to review
REVIEW_COST = 5.0

# 1) Read the SQL features
con = sqlite3.connect("data/fraudsense.db")
df = pd.read_sql("SELECT * FROM features", con)
con.close()
df["amount_ratio"] = df["amount"] / df["avg_amount_30d"]

FEATURES = ["amount", "is_foreign", "hour",
            "txn_count_1h", "txn_count_24h", "amount_ratio"]

# 2) train (first 60%) | validation (next 20%) | test (last 20%), all by time
c60 = df["ts_unix"].quantile(0.6)
c80 = df["ts_unix"].quantile(0.8)
train = df[df["ts_unix"] <= c60]
val = df[(df["ts_unix"] > c60) & (df["ts_unix"] <= c80)]
test = df[df["ts_unix"] > c80]


# 3) Decision rule: alert when (fraud probability x amount) is bigger than the review fee
def policy_cost(data, score):
    amount = data["amount"].values
    fraud = data["is_fraud"].values == 1
    alert = (score * amount) > REVIEW_COST
    total = alert.sum() * REVIEW_COST + amount[fraud & ~alert].sum()
    return int(alert.sum()), int((fraud & alert).sum()), float(total)


no_model_cost = float(test.loc[test["is_fraud"] == 1, "amount"].sum())

# 4) Try three settings; MLflow records every run so we can compare them later
CONFIGS = [
    {"learning_rate": 0.10, "max_depth": None, "max_iter": 100},
    {"learning_rate": 0.05, "max_depth": 4, "max_iter": 200},
    {"learning_rate": 0.10, "max_depth": 3, "max_iter": 150},
]

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("fraudsense")
os.makedirs("models", exist_ok=True)

rows, best = [], None
for i, params in enumerate(CONFIGS, start=1):
    with mlflow.start_run(run_name=f"hist_gb_{i}"):
        model = HistGradientBoostingClassifier(random_state=0, **params)
        model.fit(train[FEATURES], train["is_fraud"])

        val_score = model.predict_proba(val[FEATURES])[:, 1]
        test_score = model.predict_proba(test[FEATURES])[:, 1]
        val_pr = average_precision_score(val["is_fraud"], val_score)
        test_pr = average_precision_score(test["is_fraud"], test_score)
        test_roc = roc_auc_score(test["is_fraud"], test_score)
        alerts, caught, cost = policy_cost(test, test_score)

        mlflow.log_params({k: str(v) for k, v in params.items()})
        mlflow.log_metrics({
            "val_pr_auc": val_pr, "test_pr_auc": test_pr, "test_roc_auc": test_roc,
            "test_alerts": alerts, "test_fraud_caught": caught, "test_total_cost": cost,
        })
        run_file = f"models/run_{i}.joblib"
        joblib.dump(model, run_file)
        mlflow.log_artifact(run_file)

        rows.append({"run": f"hist_gb_{i}", **params, "val_PR-AUC": val_pr,
                     "test_PR-AUC": test_pr, "alerts": alerts,
                     "fraud_caught": caught, "test_cost": round(cost, 2)})
        if best is None or val_pr > best["val_pr"]:
            best = {"model": model, "val_pr": val_pr, "name": f"hist_gb_{i}", "cost": cost}

# 5) Pick the winner on VALIDATION data and save it as the production model
bundle = {"model": best["model"], "features": FEATURES, "review_cost": REVIEW_COST}
joblib.dump(bundle, "models/fraud_model.joblib")

print(pd.DataFrame(rows).round(3).to_string(index=False))
print(f"\nWinner (chosen on validation data): {best['name']}")
print(f"No-model cost on test: ${no_model_cost:,.2f}")
print(f"Winner cost on test:   ${best['cost']:,.2f}  "
      f"(saves {(no_model_cost - best['cost']) / no_model_cost * 100:.1f}%)")
print("Saved: models/fraud_model.joblib")
print("To browse your experiments, run:  mlflow ui --backend-store-uri sqlite:///mlflow.db")