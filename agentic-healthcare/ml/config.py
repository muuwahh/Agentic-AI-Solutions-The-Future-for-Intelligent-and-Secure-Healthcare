"""Shared paths and reproducibility settings for the research prototype."""

from pathlib import Path


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "saved_models"
RESULTS_DIR = ROOT / "results"
DATA_PATH = DATA_DIR / "synthetic_healthcare.csv"
MODEL_PATH = MODEL_DIR / "healthcare_fnn.pt"
PREPROCESSOR_PATH = MODEL_DIR / "preprocessing.joblib"
CONFIG_PATH = MODEL_DIR / "model_config.json"
HISTORY_PATH = MODEL_DIR / "training_history.json"
SEED = 42
MODEL_VERSION = "1.0"
TEST_SIZE = 0.15
VALIDATION_SIZE = 0.15
BATCH_SIZE = 64
EPOCHS = 40
LEARNING_RATE = 0.001
HIDDEN_DIMS = [128, 64, 32]
DROPOUT = 0.25
SELECTED_FEATURE_COUNT = 10
CLASS_NAMES = ["Class 0", "Class 1", "Class 2"]
FEATURE_COLUMNS = [
    "age",
    "heart_rate",
    "spo2",
    "temperature",
    "systolic_bp",
    "diastolic_bp",
    "lab_marker_a",
    "lab_marker_b",
    "history_indicator_a",
    "history_indicator_b",
    "motion",
    "sleep_hours",
]
WEARABLE_COLUMNS = ["heart_rate", "spo2", "temperature", "motion"]