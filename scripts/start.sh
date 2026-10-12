#!/bin/sh
set -e

# First start: build the data, the features and the production model
if [ ! -f models/fraud_model.joblib ]; then
  echo "First start: building data, features and model..."
  python src/generate_data.py
  python src/build_db.py
  python src/train_final.py
fi

echo "Starting the API on port 8000..."
python src/api.py &

echo "Starting the dashboard on port 8501..."
exec streamlit run src/dashboard.py \
  --server.port 8501 --server.address 0.0.0.0 \
  --server.headless true --browser.gatherUsageStats false
