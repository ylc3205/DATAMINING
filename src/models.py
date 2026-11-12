"""
MODULE MÔ HÌNH HỌC KHÔNG GIÁM SÁT & ĐẢO DẤU ANOMALY SCORE (MODELS)
==================================================================
Đề tài 13: Credit Card Fraud Detection (Unsupervised Anomaly Detection)
Học phần: Khai phá Dữ liệu (Data Mining) - PTIT

MỤC ĐÍCH MODULE NÀY:
- Xây dựng 3 thuật toán Anomaly Detection bắt buộc của đề tài:
    1. Isolation Forest (Rừng cô lập)
    2. Local Outlier Factor (LOF với chế độ novelty=True)
    3. One-Class SVM (Sử dụng kernel phi tuyến RBF)
- Cung cấp hàm chuẩn hóa chiều điểm số `compute_unified_anomaly_scores`:
    Quy ước thống nhất: ĐIỂM CÀNG CAO -> NGUY CƠ GIAN LẬN CÀNG LỚN!

NGƯỜI PHỤ TRÁCH:
- Nguyễn Minh Hoàng (B23DCVT173) - Modeling & Evaluation Lead
"""

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from src.config import RANDOM_STATE


def build_isolation_forest(
    n_estimators: int = 100,
    max_samples: int = 256,
    contamination: str = "auto",
    random_state: int = RANDOM_STATE
) -> IsolationForest:
    """
    Khởi tạo mô hình Isolation Forest (Rừng cô lập).
    
    NGUYÊN LÝ HOẠT ĐỘNG:
    - Thuật toán chia cắt không gian ngẫu nhiên bằng các cây nhị phân (Isolation Trees).
    - Các điểm bình thường nằm tập trung đông đúc nên cần nhiều nhát cắt để cô lập (độ sâu h(x) lớn).
    - Các điểm gian lận nằm cô độc, tách biệt nên chỉ cần rất ít nhát cắt là bị cô lập (độ sâu h(x) ngắn).
    - Độ phức tạp tính toán: O(t * psi * log(psi)) cực kỳ nhanh, có thể chạy trên toàn bộ 199.020 mẫu Train.
    
    GIẢNG VIÊN HỎI: Tham số contamination trong scikit-learn có tác dụng gì?
    BẠN TRẢ LỜI (ĐẶC BIỆT CHÚ Ý THEO CHUẨN V2.1):
    - Trong thư viện scikit-learn, tham số `contamination` CHỈ LÀM THAY ĐỔI ngưỡng mặc định `offset_`
      dùng cho hàm `.predict()` (nhãn nhị phân -1 hoặc 1).
    - Khi các tham số khác giữ nguyên, thay đổi `contamination` HOÀN TOÀN KHÔNG LÀM THAY ĐỔI
      hàm `.score_samples()` và không làm thay đổi thứ tự xếp hạng (ranking) của các mẫu!
    - Vì vậy, `contamination` không thể trực tiếp cải thiện AUPRC khi tính từ điểm liên tục.
    - Chúng em để `contamination='auto'` và tập trung tối ưu ngưỡng quyết định tau* bằng
      phương pháp Threshold Moving kết hợp Ma trận chi phí trên tập Validation!
    """
    return IsolationForest(
        n_estimators=n_estimators,
        max_samples=max_samples,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1  # Tận dụng toàn bộ nhân CPU của máy tính để chạy nhanh nhất
    )


def build_local_outlier_factor(
    n_neighbors: int = 20,
    novelty: bool = True
) -> LocalOutlierFactor:
    """
    Khởi tạo mô hình Local Outlier Factor (Hệ số dị biệt cục bộ).
    
    NGUYÊN LÝ HOẠT ĐỘNG:
    - LOF đo mật độ cục bộ của một điểm dữ liệu so với k láng giềng gần nhất xung quanh nó.
    - Nếu một điểm có mật độ cục bộ xấp xỉ láng giềng -> Điểm bình thường (LOF ~ 1).
    - Nếu một điểm có mật độ cục bộ thấp hơn đáng kể so với láng giềng -> Điểm ngoại lai (LOF > 1).
    
    GIẢNG VIÊN HỎI: Tại sao bắt buộc phải đặt novelty=True?
    BẠN TRẢ LỜI:
    - Mặc định trong scikit-learn, `novelty=False` (dùng cho Outlier Detection trên tập tĩnh),
      khi đó mô hình không có hàm `.predict()` hay `.score_samples()` cho dữ liệu mới bên ngoài.
    - Để đánh giá được trên tập Validation và Test hoàn toàn mới, bắt buộc phải đặt `novelty=True`
      để kích hoạt chế độ Novelty Detection!
      
    GIẢNG VIÊN HỎI: Tại sao phải subsampling N = 25.000 dòng?
    BẠN TRẢ LỜI:
    - N = 25.000 mẫu là giới hạn thực nghiệm được nhóm lựa chọn có chủ đích để kiểm soát tài nguyên phần cứng,
      không phải là giới hạn bắt buộc mang tính lý thuyết của thuật toán LOF.
    - Về mặt độ phức tạp tính toán, thuật toán LOF cần tính và lưu ma trận khoảng cách k-láng giềng với
      chi phí bộ nhớ O(N^2 * d). Nếu dùng toàn bộ 199.020 dòng Train trên một máy đơn, dung lượng RAM cần thiết
      sẽ vượt quá 100GB dẫn đến tràn bộ nhớ (OOM).
    - Subsampling N = 25.000 (với random seed 42) là giải pháp cân bằng thực nghiệm tối ưu, giúp mô hình
      chạy mượt mà trên phần cứng máy tính cá nhân (RAM 16GB) mà vẫn bảo toàn đầy đủ hình dạng phân phối của lớp Normal.
    """
    return LocalOutlierFactor(
        n_neighbors=n_neighbors,
        novelty=novelty,
        metric="minkowski",
        p=2,       # Khoảng cách Euclidean tiêu chuẩn
        n_jobs=-1  # Đa luồng CPU
    )


def build_one_class_svm(
    kernel: str = "rbf",
    nu: float = 0.01,
    gamma: str | float = "scale"
) -> OneClassSVM:
    """
    Khởi tạo mô hình One-Class SVM (Support Vector Machine một lớp).
    
    NGUYÊN LÝ HOẠT ĐỘNG:
    - Chiếu dữ liệu vào không gian đặc trưng phi tuyến số chiều vô hạn thông qua nhân RBF (Gaussian Kernel).
    - Tìm một siêu phẳng (hoặc siêu cầu) phân tách toàn bộ dữ liệu bình thường ra xa khỏi gốc tọa độ.
    - Điểm nào nằm bên ngoài biên bao bọc sẽ bị coi là dị biệt.
    
    GIẢNG VIÊN HỎI: Tham số nu có ý nghĩa gì?
    BẠN TRẢ LỜI:
    - `nu` đóng 2 vai trò: Là chặn trên của tỷ lệ ngoại lai huấn luyện, đồng thời là chặn dưới của
      tỷ lệ Support Vectors được chọn. Chúng em chọn nu=0.01 để ranh giới ôm chặt các điểm bình thường.
      
    GIẢNG VIÊN HỎI: Tại sao One-Class SVM lại cần subsampling N = 20.000?
    BẠN TRẢ LỜI:
    - Bộ giải quy hoạch toàn phương (Quadratic Programming solver) của SVM có độ phức tạp thời gian
      từ O(N^2) đến O(N^3). Với N = 199.000, thuật toán sẽ mất từ vài tiếng đến cả ngày để hội tụ.
      Subsampling N = 20.000 giúp mô hình huấn luyện trong vòng 2-3 phút mà vẫn học tốt ranh giới bao bọc.
    """
    return OneClassSVM(
        kernel=kernel,
        nu=nu,
        gamma=gamma
    )


def compute_unified_anomaly_scores(model, X: np.ndarray, model_name: str = "iForest") -> np.ndarray:
    """
    CHUẨN HÓA THỐNG NHẤT CHIỀU ĐIỂM SỐ DỊ BIỆT (UNIFIED SCORE CONVENTION):
    
    QUY ƯỚC TOÁN HỌC: ĐIỂM CÀNG CAO -> NGUY CƠ GIAN LẬN CÀNG LỚN!
    
    GIẢNG VIÊN HỎI: Tại sao trong hàm này bạn lại thêm dấu trừ (-) vào trước kết quả của mô hình?
    BẠN TRẢ LỜI:
    - Trong scikit-learn, các hàm đánh giá mặc định trả về điểm số bị NGƯỢC CHIỀU:
      + `IsolationForest.score_samples(X)`: Giá trị âm lớn thể hiện điểm bất thường cao (Opposite of anomaly score).
      + `LocalOutlierFactor.score_samples(X)`: Trả về số đối của LOF (Negative outlier factor). Càng âm càng dị biệt.
      + `OneClassSVM.decision_function(X)`: Khoảng cách có dấu tới siêu phẳng. Điểm âm là nằm ngoài biên (bất thường).
    - Nếu để nguyên dấu âm, khi vẽ đường Precision-Recall Curve hoặc tính AUPRC sẽ bị ngược hoàn toàn!
    - Do đó, chúng em nhân với -1: S = -score_raw(X).
      Khi đó: Giao dịch bình thường có điểm S thấp, giao dịch gian lận có điểm S cao vượt trội.
    """
    if model_name in ["iForest", "IsolationForest"]:
        # Điểm gốc scikit-learn: càng âm càng bất thường -> Đảo dấu bằng -1
        return -model.score_samples(X)
        
    elif model_name in ["LOF", "LocalOutlierFactor"]:
        # Điểm gốc scikit-learn: negative outlier factor -> Đảo dấu bằng -1
        return -model.score_samples(X)
        
    elif model_name in ["OCSVM", "OneClassSVM"]:
        # Điểm gốc scikit-learn: signed distance -> Đảo dấu bằng -1
        return -model.decision_function(X)
        
    else:
        raise ValueError(
            f"[LỖI TÊN MÔ HÌNH] Không nhận diện được model_name='{model_name}'. "
            "Chỉ hỗ trợ: 'iForest', 'LOF', 'OCSVM'."
        )
