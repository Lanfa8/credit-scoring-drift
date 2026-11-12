import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_PATH = DATA_DIR / "UCI_Credit_Card.csv"
PROCESSED_DIR = DATA_DIR / "processed"
BATCHES_DIR = DATA_DIR / "batches"
QUARANTINE_DIR = DATA_DIR / "quarantine"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
LOGS_DIR = ROOT / "logs"

REFERENCE_PATH = PROCESSED_DIR / "reference.parquet"
PRODUCTION_POOL_PATH = PROCESSED_DIR / "production_pool.parquet"
MODEL_PATH = MODELS_DIR / "model.joblib"
MODEL_META_PATH = MODELS_DIR / "metadata.json"

SEED = 42
PSEUDO_SALT = os.getenv("PSEUDO_SALT", "dev-salt-change-me")

TARGET = "default"
ID_COL = "customer_key"
PAY_STATUS = [f"pay_status_{i}" for i in range(1, 7)]
BILL_AMT = [f"bill_amt_{i}" for i in range(1, 7)]
PAY_AMT = [f"pay_amt_{i}" for i in range(1, 7)]

SENSITIVE_COLS = ["sex", "marriage"]
MODEL_FEATURES = ["limit_bal", "education", "age", *PAY_STATUS, *BILL_AMT, *PAY_AMT]
DERIVED_FEATURES = ["utilization", "payment_ratio", "months_delayed"]
MONITORED_FEATURES = [*MODEL_FEATURES, *DERIVED_FEATURES]

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{ROOT / 'mlflow.db'}")
PUSHGATEWAY_URL = os.getenv("PUSHGATEWAY_URL", "localhost:9091")
LOKI_URL = os.getenv("LOKI_URL", "http://localhost:3100")
