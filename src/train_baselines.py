import sqlite3
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score, precision_score, recall_score

# 1) Read the features that SQL built
con = sqlite3.connect("data/fraudsense.db")
df = pd.read_sql("SELECT * FROM features", con)
con.close()
df["amount_ratio"] = df["amount"] / df["avg_amount_30d"]

FEATURES = ["amount", "is_foreign", "hour",
            "txn_count_1h", "txn_count_24h", "amount_ratio"]

# 2) Time-based split: train on the first 80% of time, test on the last 20%
cut = df["ts_unix"].quantile(0.8)
train, test = df[df["ts_unix"] <= cut], df[df["ts_unix"] > cut]
X_train, y_train = train[FEATURES], train["is_fraud"]
X_test, y_test = test[FEATURES], test["is_fraud"]

print("train rows:", len(train), "| test rows:", len(test))
print("fraud in test set:", int(y_test.sum()), "\n")

# 3) The six models
models = {
    "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=1000)),
    "Decision Tree": DecisionTreeClassifier(max_depth=6, class_weight="balanced", random_state=0),
    "kNN": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=25)),
    "SVM (linear)": make_pipeline(StandardScaler(), LinearSVC(class_weight="balanced")),
    "Random Forest": RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample", n_jobs=-1, random_state=0),
    "Gradient Boosting": HistGradientBoostingClassifier(random_state=0),
}

# 4) Train each one and score it on the unseen test set
rows = []
for name, model in models.items():
    model.fit(X_train, y_train)
    if hasattr(model, "predict_proba"):
        score = model.predict_proba(X_test)[:, 1]
    else:
        score = model.decision_function(X_test)
    pred = model.predict(X_test)
    rows.append({
        "model": name,
        "PR-AUC": average_precision_score(y_test, score),
        "ROC-AUC": roc_auc_score(y_test, score),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred),
    })

results = pd.DataFrame(rows).sort_values("PR-AUC", ascending=False).round(3)
print(results.to_string(index=False))
results.to_csv("data/baseline_results.csv", index=False)