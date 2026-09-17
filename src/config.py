"""Shared paths and constants for the Cc QSPR pipeline."""
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

RAW_SMILES_FILE = ROOT_DIR / "result" / "Mixed_exclude.xlsx"
DESCRIPTORS_FILE = ROOT_DIR / "Case4_1.xlsx"
CLEAN_FILE = ROOT_DIR / "Case4_1_clean.xlsx"
OUTPUT_DIR = ROOT_DIR / "outputs"

TARGET_COLUMN = "Cc"
NON_FEATURE_COLUMNS = ["SMILES", "EO", "PO", "Cc"]

RANDOM_STATE = 42
TEST_SIZE = 0.2
CORRELATION_THRESHOLD = 0.8
