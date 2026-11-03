import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    precision_recall_curve, average_precision_score, auc, confusion_matrix
)
from src.config import COST_FP, DEFAULT_COST_FN

def evaluate_threshold(y_true, scores, threshold):
    """Evaluate binary predictions at a specific threshold."""
    y_pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    return {
        "threshold": threshold,
        "TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn),
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

def compute_auprc(y_true, continuous_scores):
    """
    Compute Average Precision (AP / AUPRC) directly from continuous anomaly scores.
    Uses sklearn.metrics.average_precision_score to calculate the weighted mean of precisions,
    avoiding trapezoidal interpolation optimism.
    """
    ap_score = average_precision_score(y_true, continuous_scores)
    precisions, recalls, thresholds = precision_recall_curve(y_true, continuous_scores)
    return ap_score, precisions, recalls, thresholds

def compute_cost(tp, fp, tn, fn, cost_fn=DEFAULT_COST_FN, cost_fp=COST_FP):
    """
    Calculate Total Expected Business Cost (Research Simulation):
    Total_Cost = C_FN * FN + C_FP * FP
    """
    return (cost_fn * fn) + (cost_fp * fp)

def find_optimal_cost_threshold(y_true, scores, cost_fn=DEFAULT_COST_FN, cost_fp=COST_FP, num_thresholds=200):
    """Sweep thresholds across continuous score range on Validation Set to find minimum cost threshold tau*."""
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
        if cost < min_cost:
            min_cost = cost
            best_thresh = th
            best_metrics = m

    return best_thresh, min_cost, best_metrics, results
