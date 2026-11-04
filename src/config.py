"""
CẤU HÌNH HỆ THỐNG DỰ ÁN (PROJECT CONFIGURATION)
==============================================
Đề tài 13: Credit Card Fraud Detection (Unsupervised Anomaly Detection)
Học phần: Khai phá Dữ liệu (Data Mining) - PTIT

MỤC ĐÍCH FILE NÀY:
- Lưu trữ tập trung toàn bộ đường dẫn (paths), siêu tham số (hyperparameters),
  tỷ lệ chia tập dữ liệu và hệ số ma trận chi phí (Cost Matrix).
- Đảm bảo tính tái lập 100% (Reproducibility) thông qua RANDOM_STATE = 42.

THÀNH VIÊN PHỤ TRÁCH:
- Nguyễn Văn Hiển Vinh (B23DCVT461) - Leader / Data & Preprocessing Lead
- Nguyễn Minh Hoàng (B23DCVT173) - Modeling & Evaluation Lead
"""

from pathlib import Path

# ==============================================================================
# 1. CẤU HÌNH ĐƯỜNG DẪN THƯ MỤC DỰ ÁN
# ==============================================================================
# ROOT_DIR: Thư mục gốc của project (C:\Users\Vinh Nguyen\DATAMINING)
ROOT_DIR = Path(__file__).resolve().parent.parent

# DATA_DIR: Nơi chứa dữ liệu
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "creditcard.csv"  # File dataset gốc từ Kaggle
PROCESSED_DATA_DIR = DATA_DIR / "processed"          # Nơi lưu dữ liệu sau tiền xử lý (nếu cần)

# FIGURES_DIR: Thư mục tự động lưu biểu đồ xuất từ code để dán vào Báo cáo Word
FIGURES_DIR = ROOT_DIR / "reports" / "figures"

# ==============================================================================
# 2. CẤU HÌNH TÁI LẬP KHOA HỌC (REPRODUCIBILITY)
# ==============================================================================
# RANDOM_STATE = 42: Giúp cố định mọi phép chia ngẫu nhiên và khởi tạo mô hình.
# Bất kỳ ai chạy lại code trên bất kỳ máy nào cũng sẽ ra con số giống hệt nhau 100%.
RANDOM_STATE = 42

# ==============================================================================
# 3. TỶ LỆ CHIA DỮ LIỆU ZERO-LEAKAGE (STRATIFIED 70 / 15 / 15)
# ==============================================================================
# Giảng viên hỏi: Tại sao lại chọn tỷ lệ này?
# Trả lời: 
# - 70% Train (199.020 mẫu Normal) đủ lớn để mô hình học thói quen người tốt.
# - 15% Validation (42.721 mẫu, có 74 Fraud ~ 0.1732%) dùng để thi thử & chọn ngưỡng tau*.
# - 15% Test (42.722 mẫu, có 74 Fraud ~ 0.1732%) được niêm phong cho đánh giá cuối cùng.
TRAIN_NORMAL_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# ==============================================================================
# 4. KÍCH THƯỚC LẤY MẪU (SUBSAMPLING) CHO CÁC MÔ HÌNH NẶNG TÍNH TOÁN
# ==============================================================================
# Giảng viên hỏi: Tại sao không fit LOF và One-Class SVM trên toàn bộ 199.000 dòng?
# Trả lời:
# - LOF có độ phức tạp không gian O(N^2 * d) để lưu ma trận khoảng cách k-láng giềng.
#   Nếu N = 199.000, ma trận khoảng cách sẽ ngốn > 150 GB RAM -> Sập máy tính (OOM).
#   Vì vậy ta lấy mẫu ngẫu nhiên có kiểm soát N = 25.000 (vừa đủ học phân phối, vừa chạy nhanh).
# - One-Class SVM giải bài toán quy hoạch toàn phương (QP solver) O(N^2) đến O(N^3).
#   Lấy mẫu N = 20.000 đảm bảo mô hình hội tụ trong vài phút thay vì treo máy hàng giờ.
# - t-SNE có độ phức tạp O(N^2), ta lấy mẫu 5.000 điểm để trực quan hóa rõ nét cụm.
LOF_SAMPLE_SIZE = 25000     # Kích thước mẫu cho Local Outlier Factor
OCSVM_SAMPLE_SIZE = 20000   # Kích thước mẫu cho One-Class SVM
TSNE_SAMPLE_SIZE = 5000     # Kích thước mẫu Normal khi chạy t-SNE 2D/3D

# ==============================================================================
# 5. THAM SỐ MA TRẬN CHI PHÍ THỰC TẾ (BUSINESS COST MATRIX)
# ==============================================================================
# Giảng viên hỏi: Chi phí này lấy ở đâu ra và có ý nghĩa gì?
# Trả lời:
# - Trong thực tế ngân hàng:
#   + Báo nhầm (False Positive - FP): Gọi điện thoại hoặc gửi SMS xác minh tốn chi phí nhỏ (C_FP = 1.0).
#   + Bỏ lọt gian lận (False Negative - FN): Bị trộm tiền bồi thường thiệt hại rất lớn (C_FN = 30.0).
#   Hệ số C_FN / C_FP thường dao động từ 10 đến 100 lần.
COST_FP = 1.0               # Chi phí xử lý 1 cảnh báo nhầm (False Alarm)
DEFAULT_COST_FN = 30.0      # Thiệt hại ước tính khi bỏ lọt 1 giao dịch lừa đảo
COST_RATIOS = [10.0, 20.0, 30.0, 50.0, 100.0]  # Dải khảo sát độ nhạy chi phí
