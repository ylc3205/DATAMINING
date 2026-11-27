"""Cấu hình trung tâm cho dự án Credit Card Fraud Detection (PTIT - Đề tài 13)."""

from pathlib import Path

# Đường dẫn thư mục gốc và dữ liệu / biểu đồ
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "creditcard.csv"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
FIGURES_DIR = ROOT_DIR / "reports" / "figures"

# Seed cố định để bảo đảm tính tái lập hoàn toàn cho chia tách và mô hình
RANDOM_STATE = 42

# Tỷ lệ phân chia Zero-Leakage (Stratified 70 / 15 / 15)
TRAIN_NORMAL_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Cỡ mẫu kiểm soát cho các thuật toán có độ phức tạp không gian/thời gian cao O(N^2)-O(N^3)
LOF_SAMPLE_SIZE = 25000
OCSVM_SAMPLE_SIZE = 20000
TSNE_SAMPLE_SIZE = 5000

# Ma trận chi phí tương đối (đơn vị tương đối: FN bỏ sót gian lận đắt gấp 30 lần FP báo nhầm)
COST_FP = 1.0
DEFAULT_COST_FN = 30.0
COST_RATIOS = [10.0, 20.0, 30.0, 50.0, 100.0]
