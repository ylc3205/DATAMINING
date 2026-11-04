"""
MODULE ĐÁNH GIÁ CHỈ SỐ, AUPRC & MA TRẬN CHI PHÍ (METRICS)
=========================================================
Đề tài 13: Credit Card Fraud Detection (Unsupervised Anomaly Detection)
Học phần: Khai phá Dữ liệu (Data Mining) - PTIT

MỤC ĐÍCH MODULE NÀY:
- Tính toán các chỉ số đánh giá chuyên sâu cho dữ liệu mất cân bằng lớp cực đoan:
  Precision, Recall, F1-Score, PR-Curve, AUPRC (Average Precision).
- Xây dựng hàm Ma trận chi phí thiệt hại (Business Cost Matrix) để tìm ngưỡng cắt
  tối ưu tau* trên tập Validation thay vì dựa vào cảm tính.

NGƯỜI PHỤ TRÁCH:
- Nguyễn Minh Hoàng (B23DCVT173) - Modeling & Evaluation Lead
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    precision_recall_curve, average_precision_score, confusion_matrix
)
from src.config import COST_FP, DEFAULT_COST_FN


def evaluate_threshold(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    """
    Đánh giá kết quả phân loại nhị phân tại một ngưỡng cắt (threshold) cụ thể.
    
    NGUYÊN LÝ:
    - Nếu Anomaly Score S >= threshold: Dự đoán là GIAN LẬN (y_pred = 1).
    - Nếu Anomaly Score S <  threshold: Dự đoán là BÌNH THƯỜNG (y_pred = 0).
    
    CÁC CHỈ SỐ TÍNH TOÁN:
    - TP (True Positive): Bắt đúng kẻ gian.
    - FP (False Positive): Báo động nhầm người tốt (Khách hàng bị phiền).
    - TN (True Negative): Nhận diện đúng giao dịch bình thường.
    - FN (False Negative): Bỏ lọt kẻ gian (Ngân hàng bị mất tiền bồi thường!).
    - Precision = TP / (TP + FP)  -> Khi báo động thì đúng được bao nhiêu %?
    - Recall    = TP / (TP + FN)  -> Bắt được bao nhiêu % trên tổng số kẻ gian?
    - F1-Score  = 2 * (P * R) / (P + R) -> Cân bằng điều hòa giữa P và R.
    """
    # Gán nhãn nhị phân theo ngưỡng cắt
    y_pred = (scores >= threshold).astype(int)
    
    # Tính ma trận nhầm lẫn (Confusion Matrix)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    # Tính Precision, Recall, F1 (zero_division=0 để tránh lỗi chia cho 0)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    return {
        "threshold": float(threshold),
        "TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1)
    }


def compute_auprc(y_true: np.ndarray, continuous_scores: np.ndarray):
    """
    Tính diện tích dưới đường cong Precision-Recall (AUPRC / Average Precision)
    trực tiếp từ điểm số dị biệt liên tục (Continuous Anomaly Scores).
    
    GIẢNG VIÊN HỎI: Tại sao dùng sklearn.metrics.average_precision_score mà không dùng 
                    hàm tích phân hình thang auc(recalls, precisions)?
    BẠN TRẢ LỜI (ĐẶC BIỆT CHÚ Ý THEO CHUẨN V2.1):
    - Đường Precision-Recall không biến thiên tuyến tính giữa các điểm khảo sát.
    - Phương pháp tích phân hình thang (Trapezoidal Rule) tuyến tính hóa các đoạn cong,
      dẫn tới hiện tượng "Lạc quan hóa sai lệch" (Optimistic Bias), làm thổi phồng diện tích AUPRC!
    - Hàm `average_precision_score` tính toán tổng có trọng số của các mức Precision theo từng bước
      nhảy thực tế của Recall: AP = Sum_n ((R_n - R_{n-1}) * P_n).
      Đây là phương pháp chuẩn xác khoa học được khuyến nghị bởi Davis & Goadrich (ICML 2006).
      
    GIẢNG VIÊN HỎI: Có cần Min-Max chuẩn hóa điểm số về đoạn [0, 1] trước khi tính AUPRC không?
    BẠN TRẢ LỜI:
    - Không bắt buộc! AUPRC chỉ phụ thuộc vào thứ tự xếp hạng (ranking) tương đối của điểm số.
      Bất kỳ phép biến đổi đơn điệu tăng nào cũng giữ nguyên giá trị AUPRC 100%.
    """
    # 1. Tính chỉ số AP chuẩn xác
    ap_score = average_precision_score(y_true, continuous_scores)
    
    # 2. Lấy tọa độ đường cong Precision - Recall để vẽ đồ thị
    precisions, recalls, thresholds = precision_recall_curve(y_true, continuous_scores)
    
    return ap_score, precisions, recalls, thresholds


def compute_cost(tp: int, fp: int, tn: int, fn: int, 
                 cost_fn: float = DEFAULT_COST_FN, 
                 cost_fp: float = COST_FP) -> float:
    """
    Tính Tổng chi phí thiệt hại ước tính theo Ma trận chi phí kinh tế (Business Cost Matrix):
    
    CÔNG THỨC TOÁN HỌC:
        Total Cost = (C_FN * FN) + (C_FP * FP)
        
    Ý NGHĨA KINH TẾ:
    - C_FN = 30.0: Thiệt hại khi để lọt 1 vụ trộm tiền (phải hoàn tiền bảo hiểm, mất uy tín).
    - C_FP = 1.0:  Chi phí khi báo nhầm 1 khách hàng (gửi SMS cảnh báo hoặc nhân viên gọi xác minh).
    - Mục tiêu của ngân hàng: Không phải tìm mô hình có F1 cao nhất, mà là tìm mô hình
      và ngưỡng cắt tau* sao cho TỔNG THIỆT HẠI KINH TẾ LÀ THẤP NHẤT!
    """
    return (cost_fn * fn) + (cost_fp * fp)


def find_optimal_cost_threshold(
    y_true: np.ndarray, 
    scores: np.ndarray, 
    cost_fn: float = DEFAULT_COST_FN, 
    cost_fp: float = COST_FP, 
    num_thresholds: int = 200
):
    """
    Quét dải ngưỡng điểm trên tập Validation để tìm ra Ngưỡng tối ưu tau* có chi phí thấp nhất.
    
    GIẢNG VIÊN HỎI: Quá trình tìm ngưỡng tau* diễn ra ở tập nào?
    BẠN TRẢ LỜI:
    - Quá trình này CHỈ ĐƯỢC PHÉP thực hiện trên tập Validation!
    - Tuyệt đối không được dùng tập Test để dò tìm ngưỡng, vì làm như vậy sẽ vi phạm nguyên tắc
      bảo mật của tập Test (Data Snooping / Information Leakage).
    - Sau khi tìm được con số tau* cụ thể trên Validation, ta đóng băng con số đó lại và mang
      nguyên vẹn sang áp dụng trên tập Test ở bước đánh giá cuối cùng.
    """
    # Quét từ phân vị 50% (trung vị) đến phân vị 99.9% của dải điểm dị biệt
    q_low = np.percentile(scores, 50)
    q_high = np.percentile(scores, 99.9)
    thresholds = np.linspace(q_low, q_high, num_thresholds)

    best_thresh = None
    min_cost = float("inf")
    best_metrics = None

    results = []
    for th in thresholds:
        m = evaluate_threshold(y_true, scores, th)
        cost = compute_cost(m["TP"], m["FP"], m["TN"], m["FN"], cost_fn=cost_fn, cost_fp=cost_fp)
        m["cost"] = cost
        results.append(m)
        
        # Cập nhật ngưỡng có chi phí thấp nhất
        if cost < min_cost:
            min_cost = cost
            best_thresh = th
            best_metrics = m

    return best_thresh, min_cost, best_metrics, results
