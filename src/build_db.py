import sqlite3
import pandas as pd

df = pd.read_csv("data/transactions.csv", parse_dates=["ts"])
df["ts_unix"] = (df["ts"] - pd.Timestamp("1970-01-01")) // pd.Timedelta(seconds=1)
df["ts"] = df["ts"].astype(str)
df["is_foreign"] = df["is_foreign"].astype(int)
df["is_fraud"] = df["is_fraud"].astype(int)

con = sqlite3.connect("data/fraudsense.db")
df.to_sql("transactions", con, if_exists="replace", index=False)
con.executescript(open("sql/features.sql").read())

print("transactions loaded:", len(df))
print(pd.read_sql("SELECT * FROM features WHERE txn_count_1h >= 3 LIMIT 5", con).to_string(index=False))
con.close()