import requests

BASE = "http://127.0.0.1:8000"

print("1) Health check")
print(requests.get(f"{BASE}/health").json(), "\n")

print("2) An ordinary purchase")
normal = {"amount": 35.0, "is_foreign": 0, "hour": 14,
          "txn_count_1h": 0, "txn_count_24h": 1, "avg_amount_30d": 40.0}
print(requests.post(f"{BASE}/score", json=normal).json(), "\n")

print("3) A suspicious burst: 4th card use in an hour, big amount, foreign, 3am")
burst = {"amount": 850.0, "is_foreign": 1, "hour": 3,
         "txn_count_1h": 4, "txn_count_24h": 4, "avg_amount_30d": 60.0}
print(requests.post(f"{BASE}/score", json=burst).json(), "\n")

print("4) A bad request (missing fields)")
r = requests.post(f"{BASE}/score", json={"amount": 100})
print(r.status_code, r.json(), "\n")

print("5) Score an existing transaction from the database by id")
print(requests.get(f"{BASE}/score/48612").json())