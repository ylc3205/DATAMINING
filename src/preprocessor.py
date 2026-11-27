"""Chuẩn hóa đặc trưng Amount và Time bằng RobustScaler chỉ fit trên Train_Normal."""

import pandas as pd
from sklearn.preprocessing import RobustScaler


class ZeroLeakagePreprocessor:
    """
    Chuẩn hóa Amount và Time theo Median/IQR học duy nhất từ Train_Normal, giữ nguyên V1–V28.

    Lý do dùng RobustScaler: Amount lệch phải cực lớn (max ~ 25,691), StandardScaler/MinMaxScaler
    dễ bị kéo lệch, còn V1–V28 đã qua PCA nên giữ nguyên để không phá vỡ cấu trúc phương sai.
    """

    def __init__(self):
        # Giữ riêng scaler_amount và scaler_time để tương thích với notebook 02 và artifact đã lưu
        self.scaler_amount = RobustScaler()
        self.scaler_time = RobustScaler()
        self.feature_cols = [f"V{i}" for i in range(1, 29)] + ["scaled_amount", "scaled_time"]
        self.is_fitted = False

    def fit(self, train_df: pd.DataFrame):
        """
        Học Median và IQR từ tập Train_Normal (chỉ Class 0) để tránh rò rỉ từ Val/Test.

        Input: train_df (pd.DataFrame chứa cột 'Amount', 'Time').
        Output: self.
        """
        self.scaler_amount.fit(train_df[["Amount"]])
        self.scaler_time.fit(train_df[["Time"]])
        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame):
        """
        Biến đổi dữ liệu mới bằng Median/IQR đã fit và trả về ma trận 30 đặc trưng + nhãn y.

        Input: df (pd.DataFrame).
        Output: (X: pd.DataFrame 30 cột, y: np.ndarray | None).
        """
        if not self.is_fitted:
            raise ValueError(
                "[LỖI RÒ RỈ DỮ LIỆU] Preprocessor chưa được fit! "
                "Bạn bắt buộc phải gọi .fit(train_df) trên tập Train trước khi transform."
            )

        df_out = df.copy()
        df_out["scaled_amount"] = self.scaler_amount.transform(df[["Amount"]])
        df_out["scaled_time"] = self.scaler_time.transform(df[["Time"]])

        X = df_out[self.feature_cols].copy()
        y = df_out["Class"].values if "Class" in df_out.columns else None
        return X, y

    def fit_transform(self, train_df: pd.DataFrame):
        """Fit trên Train_Normal rồi transform chính tập đó."""
        self.fit(train_df)
        return self.transform(train_df)
