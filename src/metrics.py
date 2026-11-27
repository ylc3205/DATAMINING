"""Đánh giá chỉ số phân loại lệch lớp (AUPRC, F1), độ nhạy trùng lặp và tối ưu ngưỡng chi phí."""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)
from src.config import COST_FP, DEFAULT_COST_FN


def evaluate_threshold(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    """
    Đánh giá phân loại nhị phân tại một ngưỡng cắt cụ thể (score >= threshold -> Fraud).

    Input: y_true (0/1), scores (điểm dị biệt liên tục), threshold (float).
    Output: dict gồm threshold, TP, FP, TN, FN, precision, recall, f1.
    """
    y_pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "TP": int(tp),
        "FP": int(fp),
        "TN": int(tn),
        "FN": int(fn),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def compute_auprc(y_true: np.ndarray, continuous_scores: np.ndarray):
    """
    Tính Average Precision (AUPRC theo hàm bậc thang Davis & Goadrich) và tọa độ PR-Curve.

    Input: y_true (0/1), continuous_scores (điểm càng cao càng bất thường).
    Output: tuple (ap_score, precisions, recalls, thresholds).
    """
    ap_score = average_precision_score(y_true, continuous_scores)
    precisions, recalls, thresholds = precision_recall_curve(y_true, continuous_scores)
    return ap_score, precisions, recalls, thresholds


def find_optimal_f1_threshold(y_true: np.ndarray, scores: np.ndarray) -> dict:
    """
    Tìm ngưỡng tối ưu F1-score từ PR-Curve của scikit-learn trên Validation.

    Lý do dùng chung ở NB03–NB06: Loại bỏ vòng lặp tính F1 lặp lại, giữ đúng quy tắc chọn
    chỉ số đầu tiên của np.argmax (khi hòa F1, ưu tiên ngưỡng nhỏ hơn / Recall cao hơn).
    Output: dict gồm ap, precisions, recalls, thresholds, threshold, TP, FP, TN, FN, precision, recall, f1.
    """
    ap, precisions, recalls, thresholds = compute_auprc(y_true, scores)
    p, r = precisions[:-1], recalls[:-1]
    denom = p + r
    f1_curve = np.divide(2.0 * p * r, denom, out=np.zeros_like(denom, dtype=float), where=denom > 0)
    best_idx = int(np.argmax(f1_curve))
    metrics = evaluate_threshold(y_true, scores, float(thresholds[best_idx]))
    return {"ap": float(ap), "precisions": precisions, "recalls": recalls, "thresholds": thresholds, **metrics}


def check_duplicate_sensitivity(
    X_ref: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    feature_names: list[str],
    check_label: str = "Dedup Check",
) -> dict:
    """
    Đo độ lệch AP, Precision, Recall, F1 khi cô lập các dòng Validation trùng với tập tham chiếu.

    Mục đích: Định lượng tác động của cross-split duplicates mà không refit mô hình hay đổi tau*.
    """
    df_ref = pd.DataFrame(X_ref, columns=feature_names).drop_duplicates()
    df_val = pd.DataFrame(X_val, columns=feature_names)
    dup_indices = sorted(set(df_val.reset_index().merge(df_ref, on=feature_names, how="inner")["index"]))

    assert len(dup_indices) > 0, f"[{check_label}] Không tìm thấy bản ghi trùng nào!"
    assert int(y_val[dup_indices].sum()) == 0, f"[LỖI] Nhóm trùng ({check_label}) chứa nhãn Fraud!"

    keep_mask = np.ones(len(X_val), dtype=bool)
    keep_mask[dup_indices] = False
    y_dedup, s_dedup = y_val[keep_mask], scores[keep_mask]

    base_ap = float(average_precision_score(y_val, scores))
    base_eval = evaluate_threshold(y_val, scores, threshold)
    dedup_ap = float(average_precision_score(y_dedup, s_dedup))
    dedup_eval = evaluate_threshold(y_dedup, s_dedup, threshold)
    overlap_scores = scores[dup_indices]

    return {
        "check_label": check_label,
        "overlap_rows_removed": int(len(dup_indices)),
        "overlap_unique_signatures": int(len(df_val.iloc[dup_indices].drop_duplicates())),
        "overlap_fraud_count": 0,
        "overlap_score_min": float(overlap_scores.min()),
        "overlap_score_max": float(overlap_scores.max()),
        "dedup_val_samples": int(len(y_dedup)),
        "dedup_fraud_samples": int(y_dedup.sum()),
        "dedup_ap": dedup_ap,
        "dedup_precision": dedup_eval["precision"],
        "dedup_recall": dedup_eval["recall"],
        "dedup_f1": dedup_eval["f1"],
        "tp": dedup_eval["TP"],
        "fp": dedup_eval["FP"],
        "tn": dedup_eval["TN"],
        "fn": dedup_eval["FN"],
        "delta_ap": float(dedup_ap - base_ap),
        "delta_precision": float(dedup_eval["precision"] - base_eval["precision"]),
        "delta_recall": float(dedup_eval["recall"] - base_eval["recall"]),
        "delta_f1": float(dedup_eval["f1"] - base_eval["f1"]),
    }


def compute_cost(
    tp: int,
    fp: int,
    tn: int,
    fn: int,
    cost_fn: float = DEFAULT_COST_FN,
    cost_fp: float = COST_FP,
) -> float:
    """Tính tổng chi phí tương đối: Total Cost = cost_fn * FN + cost_fp * FP."""
    return (cost_fn * fn) + (cost_fp * fp)


def find_optimal_cost_threshold(
    y_true: np.ndarray,
    scores: np.ndarray,
    cost_fn: float = DEFAULT_COST_FN,
    cost_fp: float = COST_FP,
    num_thresholds: int = 200,
):
    """Quét lưới tuyến tính 200 ngưỡng trong [P50, P99.9] (bản thô, ưu tiên bản _exact cho đồ án)."""
    thresholds = np.linspace(np.percentile(scores, 50), np.percentile(scores, 99.9), num_thresholds)
    best_thresh, min_cost, best_metrics, results = None, float("inf"), None, []

    for th in thresholds:
        m = evaluate_threshold(y_true, scores, th)
        cost = compute_cost(m["TP"], m["FP"], m["TN"], m["FN"], cost_fn=cost_fn, cost_fp=cost_fp)
        m["cost"] = cost
        results.append(m)
        if cost < min_cost:
            min_cost, best_thresh, best_metrics = cost, th, m

    return best_thresh, min_cost, best_metrics, results


def find_optimal_cost_threshold_exact(
    y_true: np.ndarray,
    scores: np.ndarray,
    cost_fn: float = DEFAULT_COST_FN,
    cost_fp: float = COST_FP,
    return_details: bool = False,
):
    """
    Tìm ngưỡng chi phí tối ưu chính xác tuyệt đối O(N log N) qua sắp xếp và cộng dồn vector hóa.

    Giữ nguyên np.unique + cumsum + lexsort vì precision_recall_curve thiếu nhánh All-Normal và
    không trả nguyên TP/FP khi dữ liệu không có mẫu dương.
    """
    y_raw = np.asarray(y_true)
    scores = np.asarray(scores, dtype=np.float64)

    if len(y_raw) == 0 or len(y_raw) != len(scores):
        raise ValueError("Mảng y_true và scores phải khác rỗng và có cùng kích thước.")
    if not np.all(np.isfinite(scores)) or np.isnan(y_raw).any():
        raise ValueError("Dữ liệu y_true hoặc scores chứa NaN/Inf.")
    if not np.all(np.isin(y_raw, [0, 1])):
        raise ValueError("Nhãn thực tế y_true chỉ được chứa giá trị nhị phân {0, 1}.")
    if cost_fn <= 0 or cost_fp <= 0:
        raise ValueError(f"Hệ số chi phí cost_fn={cost_fn} và cost_fp={cost_fp} phải > 0.")

    y_true = y_raw.astype(int)
    total_p = int(np.sum(y_true))
    total_n = len(y_true) - total_p

    # Gom điểm trùng để cùng mức điểm luôn chuyển nhãn đồng thời
    unique_scores, inverse_idx, counts = np.unique(scores, return_inverse=True, return_counts=True)
    pos_counts = np.bincount(inverse_idx, weights=y_true, minlength=len(unique_scores)).astype(np.int64)
    neg_counts = (counts - pos_counts).astype(np.int64)

    tp = np.cumsum(pos_counts[::-1])[::-1]
    fp = np.cumsum(neg_counts[::-1])[::-1]

    max_score = unique_scores[-1]
    all_normal_thresh = float(np.nextafter(max_score, np.inf)) if np.isfinite(max_score) else float(max_score + 1.0)

    candidate_thresholds = np.append(unique_scores, all_normal_thresh)
    tp_all = np.append(tp, 0)
    fp_all = np.append(fp, 0)
    fn_all = total_p - tp_all
    tn_all = total_n - fp_all
    cost_all = (cost_fn * fn_all) + (cost_fp * fp_all)

    denom_p = tp_all + fp_all
    precision_all = np.divide(tp_all, denom_p, out=np.zeros_like(tp_all, dtype=float), where=denom_p > 0)
    recall_all = np.divide(tp_all, total_p, out=np.zeros_like(tp_all, dtype=float), where=total_p > 0)
    denom_f1 = precision_all + recall_all
    f1_all = np.divide(2 * precision_all * recall_all, denom_f1, out=np.zeros_like(precision_all, dtype=float), where=denom_f1 > 0)

    # Giải hòa tiền định: chi phí thấp nhất -> Recall cao nhất -> FP thấp nhất -> ngưỡng cao nhất
    best_idx = int(np.lexsort((-candidate_thresholds, fp_all, -recall_all, cost_all))[0])
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
        return best_thresh, best_cost, best_metrics, {
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
    return best_thresh, best_cost, best_metrics
