"""Zajednicke postavke projekta (putanje i parametri podjele)."""
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]      # retinal-vessel-segmentation/
DATA_DIR = PROJECT_DIR.parent / "data"                 # ../data (podaci nisu u repozitoriju)
SPLITS_DIR = PROJECT_DIR / "splits"
FIG_DIR = PROJECT_DIR / "results" / "figures"
METRICS_DIR = PROJECT_DIR / "results" / "metrics"

SEED = 42
VAL_FRAC = 0.15
TEST_FRAC = 0.15
