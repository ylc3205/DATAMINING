"""Khởi tạo 3 mô hình phát hiện dị biệt từ scikit-learn và chuẩn hóa chiều điểm số."""

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from src.config import RANDOM_STATE


def build_isolation_forest(
    n_estimators: int = 100,
    max_samples: int = 256,
    contamination: str = "auto",
    random_state: int = RANDOM_STATE,
) -> IsolationForest:
    """
    Khởi tạo Isolation Forest (cô lập dị biệt bằng cây nhị phân ngẫu nhiên).

    Lưu ý: contamination='auto' chỉ ảnh hưởng offset_ của .predict(), không đổi ranking score_samples().
    """
    return IsolationForest(
        n_estimators=n_estimators,
        max_samples=max_samples,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,
    )


def build_local_outlier_factor(
    n_neighbors: int = 20,
    novelty: bool = True,
) -> LocalOutlierFactor:
    """
    Khởi tạo Local Outlier Factor theo tỷ số mật độ cục bộ k-láng giềng (khoảng cách Euclid).

    Bắt buộc bật novelty=True để chấm điểm tập Validation và Test độc lập ngoài tập huấn luyện.
    """
    return LocalOutlierFactor(
        n_neighbors=n_neighbors,
        novelty=novelty,
        metric="minkowski",
        p=2,
        n_jobs=-1,
    )


def build_one_class_svm(
    kernel: str = "rbf",
    nu: float = 0.01,
    gamma: str | float = "scale",
) -> OneClassSVM:
    """Khởi tạo One-Class SVM (tìm biên bao dữ liệu bình thường bằng nhân phi tuyến)."""
    return OneClassSVM(kernel=kernel, nu=nu, gamma=gamma)


def compute_unified_anomaly_scores(
    model,
    X: np.ndarray,
    model_name: str = "iForest",
    batch_size: int | None = None,
) -> np.ndarray:
    """
    Chuẩn hóa chiều điểm dị biệt cho cả 3 mô hình: điểm càng cao -> càng bất thường.

    Đảo dấu (-1) kết quả thô của scikit-learn để đồng nhất khi tính AUPRC và tối ưu ngưỡng.
    Hỗ trợ batch_size tùy chọn (ví dụ 5,000 với LOF) nhằm kiểm soát bộ nhớ mà không đổi kết quả.
    """
    if model_name in ("iForest", "IsolationForest", "LOF", "LocalOutlierFactor"):
        scorer = model.score_samples
    elif model_name in ("OCSVM", "OneClassSVM"):
        scorer = model.decision_function
    else:
        raise ValueError(
            f"[LỖI TÊN MÔ HÌNH] Không nhận diện được model_name='{model_name}'. "
            "Chỉ hỗ trợ: 'iForest', 'LOF', 'OCSVM'."
        )

    if batch_size is not None and batch_size > 0 and len(X) > batch_size:
        chunks = [-scorer(X[i : i + batch_size]) for i in range(0, len(X), batch_size)]
        return np.concatenate(chunks).astype(np.float64, copy=False)

    return -scorer(X)
