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
    - Average Precision (AP) và PR-AUC tính theo quy tắc hình thang (Trapezoidal Rule) là hai
      cách tính xấp xỉ diện tích khác nhau trong không gian Precision-Recall.
    - Đường Precision-Recall không biến thiên tuyến tính giữa các điểm khảo sát; phương pháp tích
      phân hình thang tuyến tính hóa các đoạn nối, có xu hướng ước lượng lạc quan hơn ở các khoảng nhảy vọt.
    - Hàm `average_precision_score` tính toán tổng có trọng số của các mức Precision theo từng bước
      nhảy thực tế của Recall theo xấp xỉ bậc thang (Step Function / Rectangular Integration):
      AP = Sum_n ((R_n - R_{n-1}) * P_n).
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


def find_optimal_cost_threshold_exact(
    y_true: np.ndarray,
    scores: np.ndarray,
    cost_fn: float = DEFAULT_COST_FN,
    cost_fp: float = COST_FP,
    return_details: bool = False,
):
    """
    Tìm ngưỡng cắt tối ưu chi phí chính xác tuyệt đối (Exact Cost-Optimal Threshold)
    bằng thuật toán sắp xếp và cộng dồn vector hóa độ phức tạp O(N log N).

    ƯU ĐIỂM VƯỢT TRỘI SO VỚI LƯỚI NỘI SUY TUYẾN TÍNH (np.linspace):
    1. Không bị giới hạn percentile [P_50, P_99.9]: Tự động mở rộng toàn bộ dải điểm.
    2. Không sinh bước nhảy giả (Discretization artifacts): Đánh giá tại toàn bộ các giá trị
       điểm anomaly thực tế có trong tập dữ liệu (exact sample scores).
    3. Xử lý nhóm điểm trùng nhau (Tied Scores): Toàn bộ các giao dịch có cùng điểm số
       sẽ được chuyển đổi nhãn đồng thời để đảm bảo tính nhất quán quyết định.
    4. Bao hàm đầy đủ hai kịch bản cực trị:
       - Dự đoán toàn bộ là Normal (All-Normal): Ngưỡng hợp lệ nằm ngay trên điểm lớn nhất,
         được biểu diễn chuẩn xác bằng np.nextafter(max_score, np.inf).
       - Dự đoán toàn bộ là Fraud (All-Fraud): Ngưỡng bằng điểm nhỏ nhất min(scores).
    5. Cơ chế giải quyết hòa điểm hoàn toàn tiền định (Deterministic Tie-Breaking):
       Khi hai ngưỡng cho cùng mức chi phí cực tiểu:
       (a) Ưu tiên Recall cao hơn (bắt được nhiều gian lận hơn).
       (b) Kế tiếp ưu tiên FP thấp hơn (ít phiền khách hàng hơn).
       (c) Kế tiếp ưu tiên Threshold cao hơn (ngưỡng bảo thủ hơn).

    ĐỘ PHỨC TẠP:
    - Thời gian: O(N log N) cho np.unique / sort; O(N) cho np.cumsum và np.lexsort.
    - Không gian: O(K) với K là số lượng điểm số phân biệt (K <= N).
    """
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=np.float64)

    # 1. Kiểm tra tính hợp lệ của dữ liệu đầu vào (Input Validation)
    if len(y_true) != len(scores):
        raise ValueError(
            f"Kích thước y_true ({len(y_true)}) và scores ({len(scores)}) không khớp nhau."
        )
    if len(y_true) == 0:
        raise ValueError("Mảng dữ liệu đầu vào không được để rỗng.")
    if np.isnan(scores).any() or np.isinf(scores).any():
        raise ValueError("Điểm số anomaly (scores) chứa giá trị NaN hoặc vô cùng (Inf).")
    if np.isnan(y_true).any():
        raise ValueError("Nhãn thực tế (y_true) chứa giá trị NaN.")
    if not np.all(np.isin(y_true, [0, 1])):
        raise ValueError("Nhãn thực tế y_true chỉ được chứa giá trị nhị phân {0, 1}.")
    if cost_fn <= 0 or cost_fp <= 0:
        raise ValueError(
            f"Hệ số chi phí cost_fn={cost_fn} và cost_fp={cost_fp} phải là số thực dương (> 0)."
        )

    total_p = int(np.sum(y_true))
    total_n = int(len(y_true) - total_p)

    # 2. Gom nhóm các điểm số trùng nhau (Tied scores handling) theo thứ tự tăng dần
    unique_scores, inverse_indices, counts = np.unique(
        scores, return_inverse=True, return_counts=True
    )
    pos_counts = np.bincount(
        inverse_indices, weights=y_true, minlength=len(unique_scores)
    ).astype(np.int64)
    neg_counts = (counts - pos_counts).astype(np.int64)

    # 3. Tính tích lũy ngược từ điểm số cao nhất xuống thấp nhất
    # Tại ngưỡng th = unique_scores[k], các điểm có score >= th là các nhóm từ k đến cuối
    tp_rev = np.cumsum(pos_counts[::-1])
    fp_rev = np.cumsum(neg_counts[::-1])
    tp = tp_rev[::-1]
    fp = fp_rev[::-1]

    # 4. Bổ sung phương án dự đoán All-Normal (ngưỡng vượt trên giá trị lớn nhất)
    max_score = unique_scores[-1]
    all_normal_thresh = (
        float(np.nextafter(max_score, np.inf))
        if np.isfinite(max_score)
        else float(max_score + 1.0)
    )

    candidate_thresholds = np.append(unique_scores, all_normal_thresh)
    tp_all = np.append(tp, 0)
    fp_all = np.append(fp, 0)

    fn_all = total_p - tp_all
    tn_all = total_n - fp_all
    cost_all = (cost_fn * fn_all) + (cost_fp * fp_all)

    # 5. Tính Precision, Recall, F1 an toàn không cảnh báo chia cho 0
    denom_p = tp_all + fp_all
    precision_all = np.divide(
        tp_all, denom_p, out=np.zeros_like(tp_all, dtype=float), where=denom_p > 0
    )
    recall_all = np.divide(
        tp_all, total_p, out=np.zeros_like(tp_all, dtype=float), where=total_p > 0
    )
    denom_f1 = precision_all + recall_all
    f1_all = np.divide(
        2 * precision_all * recall_all,
        denom_f1,
        out=np.zeros_like(precision_all, dtype=float),
        where=denom_f1 > 0,
    )

    # 6. Sắp xếp tiền định (Deterministic Tie-Breaking) bằng np.lexsort:
    # Thứ tự ưu tiên:
    # 1. cost_all thấp nhất (tăng dần)
    # 2. recall_all cao nhất (giảm dần -> -recall_all tăng dần)
    # 3. fp_all thấp nhất (tăng dần)
    # 4. candidate_thresholds cao nhất (giảm dần -> -candidate_thresholds tăng dần)
    order = np.lexsort((-candidate_thresholds, fp_all, -recall_all, cost_all))
    best_idx = order[0]

    best_thresh = float(candidate_thresholds[best_idx])
    best_cost = float(cost_all[best_idx])
    best_metrics = {
        "threshold": best_thresh,
        "TP": int(tp_all[best_idx]),
        "FP": int(fp_all[best_idx]),
        "TN": int(tn_all[best_idx]),
        "FN": int(fn_all[best_idx]),
        "precision": float(precision_all[best_idx]),
        "recall": float(recall_all[best_idx]),
        "f1": float(f1_all[best_idx]),
        "cost": best_cost,
        "cost_fn": float(cost_fn),
        "cost_fp": float(cost_fp),
        "cost_ratio": float(cost_fn / cost_fp),
    }

    if return_details:
        details = {
            "thresholds": candidate_thresholds,
            "costs": cost_all,
            "tp": tp_all,
            "fp": fp_all,
            "fn": fn_all,
            "tn": tn_all,
            "precision": precision_all,
            "recall": recall_all,
            "f1": f1_all,
        }
        return best_thresh, best_cost, best_metrics, details

    return best_thresh, best_cost, best_metrics
