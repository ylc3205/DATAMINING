"""
MODULE TIỀN XỬ LÝ DỮ LIỆU ZERO-LEAKAGE (PREPROCESSOR)
=====================================================
Đề tài 13: Credit Card Fraud Detection (Unsupervised Anomaly Detection)
Học phần: Khai phá Dữ liệu (Data Mining) - PTIT

MỤC ĐÍCH MODULE NÀY:
- Chuẩn hóa hai đặc trưng chưa qua xử lý là 'Amount' (Số tiền) và 'Time' (Thời gian).
- Sử dụng thuật toán RobustScaler để kháng ngoại lai (Outliers), tránh làm méo mó phân phối.
- Giữ nguyên các đặc trưng 'V1' đến 'V28' vì bản chất chúng đã được PCA chuẩn hóa sẵn.
- Thực thi nghiêm ngặt nguyên tắc Zero-Leakage: Scaler CHỈ ĐƯỢC FIT trên tập Train_Normal!

NGƯỜI PHỤ TRÁCH:
- Nguyễn Văn Hiển Vinh (B23DCVT461) - Trưởng nhóm / Data & Preprocessing Lead
"""

import pandas as pd
from sklearn.preprocessing import RobustScaler


class ZeroLeakagePreprocessor:
    """
    Pipeline Tiền xử lý chuẩn Zero-Leakage cho bài toán Phát hiện gian lận thẻ tín dụng.
    
    GIẢNG VIÊN HỎI: Tại sao dùng RobustScaler mà không dùng StandardScaler hay MinMaxScaler?
    BẠN TRẢ LỜI:
    - Cột 'Amount' có phân phối lệch phải cực kỳ nghiêm trọng (Skewness = 16.98), trung vị là 22 USD
      nhưng giá trị lớn nhất lên tới 25.691 USD (gấp hơn 1.000 lần trung vị).
    - StandardScaler sử dụng Trung bình (Mean) và Độ lệch chuẩn (Std) -> bị các khoản tiền khổng lồ kéo lệch,
      khiến 99% các giao dịch nhỏ bị nén sát rạt về số 0.
    - MinMaxScaler sử dụng Min và Max -> càng nguy hiểm vì toàn bộ phân phối bị ép chặt bởi giá trị Max duy nhất.
    - RobustScaler sử dụng công thức: x_scaled = (x - Median) / IQR (Khoảng tứ phân vị Q3 - Q1).
      Trung vị và IQR hoàn toàn không bị ảnh hưởng bởi các giá trị ngoại lai cực đoan!
      
    GIẢNG VIÊN HỎI: Tại sao không scale các cột V1 đến V28?
    BẠN TRẢ LỜI:
    - 28 cột V1-V28 là kết quả của phép chiếu PCA từ ngân hàng, chúng đã có kỳ vọng ~ 0 và phương sai
      giảm dần theo thứ tự thành phần chính. Việc scale lại sẽ làm mất trật tự phương sai của PCA.
    """
    def __init__(self):
        # Khởi tạo 2 bộ RobustScaler độc lập cho Amount và Time
        self.scaler_amount = RobustScaler()
        self.scaler_time = RobustScaler()
        
        # Danh sách 30 đặc trưng đầu vào cuối cùng đưa vào mô hình máy học:
        # Gồm 28 biến V1-V28 và 2 biến đã scale: scaled_amount, scaled_time
        self.feature_cols = [f"V{i}" for i in range(1, 29)] + ["scaled_amount", "scaled_time"]
        
        # Cờ đánh dấu đã fit hay chưa để chống gọi transform trước khi fit
        self.is_fitted = False

    def fit(self, train_df: pd.DataFrame):
        """
        Học các tham số Trung vị (Median) và Tứ phân vị (IQR) DUY NHẤT từ tập Train.
        
        LƯU Ý PHẢN BIỆN: 
        train_df đưa vào đây BẮT BUỘC phải là Train_Normal (199.020 mẫu bình thường).
        Tuyệt đối không đưa Validation hay Test vào hàm này để chống rò rỉ dữ liệu (Data Leakage).
        """
        self.scaler_amount.fit(train_df[["Amount"]])
        self.scaler_time.fit(train_df[["Time"]])
        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame):
        """
        Biến đổi dữ liệu mới (Validation hoặc Test) bằng công thức Median/IQR đã học từ Train.
        
        Trả về:
        - X (pd.DataFrame): Ma trận 30 đặc trưng đã chuẩn hóa, sẵn sàng đưa vào mô hình.
        - y (np.ndarray): Mảng nhãn thực tế Class (0 hoặc 1) để dùng chấm điểm (nếu có).
        """
        if not self.is_fitted:
            raise ValueError(
                "[LỖI RÒ RỈ DỮ LIỆU] Preprocessor chưa được fit! "
                "Bạn bắt buộc phải gọi .fit(train_df) trên tập Train trước khi transform."
            )

        df_out = df.copy()
        
        # Áp dụng Median và IQR của tập Train để biến đổi Amount và Time
        df_out["scaled_amount"] = self.scaler_amount.transform(df[["Amount"]])
        df_out["scaled_time"] = self.scaler_time.transform(df[["Time"]])

        # Trích xuất 30 đặc trưng đầu vào (không chứa cột Class)
        X = df_out[self.feature_cols].copy()
        
        # Tách nhãn thực tế (Class) nếu tồn tại trong dữ liệu để phục vụ đánh giá sau này
        y = df_out["Class"].values if "Class" in df_out.columns else None

        return X, y

    def fit_transform(self, train_df: pd.DataFrame):
        """Tiện ích vừa fit vừa transform trên tập Train."""
        self.fit(train_df)
        return self.transform(train_df)
