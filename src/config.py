from pathlib import Path

# Project Root
ROOT_DIR = Path(__file__).resolve().parent.parent

# Data Directories
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "creditcard.csv"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Figures Directory
FIGURES_DIR = ROOT_DIR / "reports" / "figures"

# Random Seed for 100% Reproducibility
RANDOM_STATE = 42

# 3-Way Split Ratios
TRAIN_NORMAL_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Sampling Sizes for Computationally Expensive Models
LOF_SAMPLE_SIZE = 25000     # Local Outlier Factor (O(N^2 * d) complexity)
OCSVM_SAMPLE_SIZE = 20000   # One-Class SVM (O(N^2) to O(N^3) complexity)
TSNE_SAMPLE_SIZE = 5000     # Normal samples for t-SNE visualization

# Cost Matrix Parameters (Sensitivity sweep: C_FN / C_FP = 20 to 50)
COST_FP = 1.0               # Verification cost per false alarm
DEFAULT_COST_FN = 30.0      # Losses incurred per missed fraud transaction
COST_RATIOS = [10.0, 20.0, 30.0, 50.0, 100.0]
