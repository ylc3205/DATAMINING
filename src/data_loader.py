"""Nạp dữ liệu thô và phân chia 3 tập Train_Normal / Validation / Test theo chuẩn Zero-Leakage."""

import pandas as pd
from sklearn.model_selection import train_test_split
from src.config import RANDOM_STATE, RAW_DATA_PATH


def load_raw_data(filepath=RAW_DATA_PATH) -> pd.DataFrame:
    """
    Đọc file CSV thô từ đĩa và kiểm tra kích thước chuẩn của dataset Kaggle (284,807 x 31).

    Input: filepath (Path | str) — đường dẫn tới creditcard.csv.
    Output: pd.DataFrame chứa 284,807 giao dịch và 31 cột.
    """
    if not filepath.exists():
        raise FileNotFoundError(
            f"[LỖI] Không tìm thấy dataset tại đường dẫn: {filepath}.\n"
            "Vui lòng xem hướng dẫn tại file data/README.md để tải 'creditcard.csv' từ Kaggle."
        )

    df = pd.read_csv(filepath)
    assert df.shape == (284807, 31), (
        f"[LỖI KIỂM KÊ] Kích thước dữ liệu không đúng: {df.shape}, kỳ vọng: (284807, 31)"
    )
    return df


def split_data_zero_leakage(df: pd.DataFrame, random_state: int = RANDOM_STATE):
    """
    Chia phân tầng 70/15/15 và loại toàn bộ nhãn Fraud khỏi tập Train (Novelty Detection).

    Lý do: Giữ tỷ lệ gian lận tự nhiên (~0.1732%) ở Validation/Test để không bóp méo Precision,
    đồng thời bảo đảm mô hình chỉ học từ giao dịch bình thường (Class == 0).
    Input: df (pd.DataFrame), random_state (int).
    Output: tuple (train_df, val_df, test_df) đã reset index.
    """
    # Bước 1 & 2: Tách 70% Train Partition và chia đôi 30% còn lại thành 15% Val / 15% Test
    train_part, temp_part = train_test_split(
        df, test_size=0.30, random_state=random_state, stratify=df["Class"], shuffle=True
    )
    val_df, test_df = train_test_split(
        temp_part, test_size=0.50, random_state=random_state, stratify=temp_part["Class"], shuffle=True
    )

    # Bước 3: Lọc chỉ giữ Class == 0 cho Train_Normal; 344 ca Fraud trong Train bị loại bỏ khỏi fit
    train_df = train_part[train_part["Class"] == 0].reset_index(drop=True)
    n_train_fraud_excluded = int((train_part["Class"] == 1).sum())
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    print("=" * 70)
    print("XÁC NHẬN PHÂN CHIA DỮ LIỆU ZERO-LEAKAGE (CHUẨN V2.1):")
    print(f"1. Tổng dữ liệu gốc : {len(df):,} mẫu (284,315 Normal + 492 Fraud)")
    print(f"2. Phân vùng Train   : {len(train_part):,} mẫu ({n_train_fraud_excluded} Fraud bị loại bỏ khỏi huấn luyện)")
    print(f"   -> Train_Normal   : {len(train_df):,} mẫu (Chỉ chứa Class 0, dùng fit scaler & model)")
    print(f"3. Tập Validation    : {len(val_df):,} mẫu ({val_df['Class'].sum()} Fraud ~ {val_df['Class'].mean():.4%})")
    print(f"4. Tập Test Set      : {len(test_df):,} mẫu ({test_df['Class'].sum()} Fraud ~ {test_df['Class'].mean():.4%}) [NIÊM PHONG]")
    print("=" * 70)

    return train_df, val_df, test_df
