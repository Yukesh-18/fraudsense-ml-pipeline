import sqlite3
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score, roc_auc_score, precision_score, recall_score

torch.manual_seed(0)
np.random.seed(0)

# 1) Read the same SQL features and use the same time-based split as Step 3
con = sqlite3.connect("data/fraudsense.db")
df = pd.read_sql("SELECT * From features", con)
con.close()
df["amount_ratio"] = df["amount"] / df["avg_amount_30d"]

FEATURES = ["amount", "is_foreign", "hour", "txn_count_1h", "txn_count_24h", "amount_ratio"]

cut = df["ts_unix"].quantile(0.8)
train, test = df[df["ts_unix"] <= cut], df[df["ts_unix"] > cut]

# 2) Neural networks need scaled numbers, so we standardise the features

scaler = StandardScaler().fit(train[FEATURES])
X_train = torch.tensor(scaler.transform(train[FEATURES]), dtype=torch.float32)
Y_train = torch.tensor(train["is_fraud"].values, dtype=torch.float32).unsqueeze(1)
X_test = torch.tensor(scaler.transform(test[FEATURES]), dtype=torch.float32)
Y_test = test["is_fraud"].values

# 3) The network: 6 inputs -> 32 -> 16 -> 1 output
model = nn.Sequential(
    nn.Linear(len(FEATURES), 32), nn.ReLU(),
    nn.Linear(32, 16), nn.ReLU(),
    nn.Linear(16, 1),
)

# Fraud is rare, so we give fraud rows extra weight (softened with a square root)
pos = Y_train.sum()
neg = len(Y_train) - pos
pos_weight = torch.sqrt(neg / pos).reshape(1)
loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=512, shuffle=True)

# 4) Training loop
EPOCHS = 30
for epoch in range(1, EPOCHS + 1):
    model.train()
    total = 0.0
    for xb, yb in loader:
        optimizer.zero_grad()
        loss = loss_fn(model(xb), yb)
        loss.backward()
        optimizer.step()
        total += loss.item() * len(xb)
    if epoch % 5 == 0 or epoch == 1:
        print(f"epoch {epoch:2d} | loss {total / len(X_train):.4f}")

# 5) Score the unseen test set
model.eval()
with torch.no_grad():
    score = torch.sigmoid(model(X_test)).numpy().ravel()
pred = (score >= 0.5).astype(int)

nn_row = {
    "model": "PyTorch MLP",
    "PR-AUC": average_precision_score(Y_test, score),
    "ROC-AUC": roc_auc_score(Y_test, score),
    "precision": precision_score(Y_test, pred, zero_division=0),
    "recall": recall_score(Y_test, pred),
}

# 6) Add it to the Step 3 leaderboard

old = pd.read_csv("data/baseline_results.csv")
board = pd.concat([old, pd.DataFrame([nn_row])], ignore_index=True)
board = board.sort_values("PR-AUC", ascending=False).round(3)
print("\nLEADERBOARD")
print(board.to_string(index=False))
board.to_csv("data/model_leaderboard.csv", index=False)

# 7) Investigation queue: rank by expected loss = fraud probability x amount
queue = test[["txn_id", "card_id", "amount", "is_fraud"]].copy()
queue["fraud_score"] = score
queue["expected_loss"] = queue["fraud_score"] * queue["amount"]
print("\nTOP 10 TRANSACTION TO INVESTIGATE FIRST")
print(queue.sort_values("expected_loss", ascending=False).head(10).round(2).to_string(index=False))

torch.save(model.state_dict(), "data/mlp.pt")




