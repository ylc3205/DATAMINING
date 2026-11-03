import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from src.config import RANDOM_STATE, LOF_SAMPLE_SIZE, OCSVM_SAMPLE_SIZE

def build_isolation_forest(n_estimators=100, max_samples=256, contamination="auto", random_state=RANDOM_STATE):
    """
    Isolation Forest model:
    - O(t * psi * log(psi)) complexity.
    - Trained on the full Train_Normal set.
    """
    return IsolationForest(
        n_estimators=n_estimators,
        max_samples=max_samples,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1
    )

def build_local_outlier_factor(n_neighbors=20, novelty=True):
    """
    Local Outlier Factor (LOF) model:
    - Must set novelty=True to predict on new Validation/Test samples.
    - Requires training on a sampled subset (e.g. N=25,000) to fit within RAM.
    """
    return LocalOutlierFactor(
        n_neighbors=n_neighbors,
        novelty=novelty,
        metric="minkowski",
        p=2,
        n_jobs=-1
    )

def build_one_class_svm(kernel="rbf", nu=0.01, gamma="scale"):
    """
    One-Class SVM model:
    - Uses RBF kernel to capture non-linear support boundary.
    - Requires training on a sampled subset (e.g. N=20,000) due to QP complexity.
    """
    return OneClassSVM(
        kernel=kernel,
        nu=nu,
        gamma=gamma
    )

def compute_unified_anomaly_scores(model, X, model_name="iForest") -> np.ndarray:
    """
    Unified Score Convention:
    HIGHER SCORE -> GREATER ANOMALY LIKELIHOOD.
    - iForest: -score_samples(X)
    - LOF:     -score_samples(X)
    - OC-SVM:  -decision_function(X)
    """
    if model_name in ["iForest", "IsolationForest"]:
        # Scikit-learn score_samples: more negative = more abnormal -> negate it
        return -model.score_samples(X)
    elif model_name in ["LOF", "LocalOutlierFactor"]:
        # LOF novelty score_samples: negative outlier factor -> negate it
        return -model.score_samples(X)
    elif model_name in ["OCSVM", "OneClassSVM"]:
        # OneClassSVM decision_function: signed distance to hyperplane -> negate it
        return -model.decision_function(X)
    else:
        raise ValueError(f"Unknown model_name: {model_name}")
