import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
N, N_CARDS, N_MERCHANTS = 50_000, 2_000, 300
START = pd.Timestamp("2026-01-01")
SECONDS_90D = 90 * 24 * 3600

# 1) Normal-looking transactions
df = pd.DataFrame({
    "card_id": rng.integers(1, N_CARDS + 1, N),
    "merchant_id": rng.integers(1, N_MERCHANTS + 1, N),
    "amount": np.round(rng.lognormal(3.5, 1.0, N), 2),
    "ts": START + pd.to_timedelta(rng.integers(0, SECONDS_90D, N), unit="s"),
    "is_foreign": rng.random(N) < 0.05,
})

# 2) Fraud is more likely for big, foreign, late-night transactions
hour = df["ts"].dt.hour
logit = (-6.5
         + 0.002 * df["amount"].clip(upper=1500)
         + 2.0 * df["is_foreign"]
         + 1.2 * hour.isin([0, 1, 2, 3, 4]))
p = 1 / (1 + np.exp(-logit.to_numpy()))
df["is_fraud"] = rng.random(N) < p

# 3) Fraud bursts: one card used 4-8 times within 10 minutes
bursts = []
for card in rng.choice(np.arange(1, N_CARDS + 1), 40, replace=False):
    t0 = START + pd.Timedelta(seconds=int(rng.integers(0, SECONDS_90D)))
    for _ in range(int(rng.integers(4, 9))):
        bursts.append({
            "card_id": int(card),
            "merchant_id": int(rng.integers(1, N_MERCHANTS + 1)),
            "amount": round(float(rng.uniform(100, 900)), 2),
            "ts": t0 + pd.Timedelta(minutes=int(rng.integers(0, 10))),
            "is_foreign": bool(rng.random() < 0.5),
            "is_fraud": True,
        })

df = pd.concat([df, pd.DataFrame(bursts)], ignore_index=True)
df = df.sort_values("ts").reset_index(drop=True)
df.insert(0, "txn_id", df.index + 1)
df["hour"] = df["ts"].dt.hour
df.to_csv("data/transactions.csv", index=False)

print("rows:", len(df), "| fraud rate:", round(df["is_fraud"].mean() * 100, 2), "%")

