"""
MODULE NẠP DỮ LIỆU & PHÂN CHIA ZERO-LEAKAGE (DATA LOADER)
=========================================================
Đề tài 13: Credit Card Fraud Detection (Unsupervised Anomaly Detection)
Học phần: Khai phá Dữ liệu (Data Mining) - PTIT

MỤC ĐÍCH MODULE NÀY:
- Nạp bộ dữ liệu thô `creditcard.csv` từ thư mục `data/raw/` và kiểm tra toàn vẹn.
- Phân chia dữ liệu thành 3 tập Train / Validation / Test theo chuẩn Zero-Leakage V2.1.
- Bảo toàn tỷ lệ gian lận ở Validation và Test (~0.1732%), đồng thời lọc sạch nhãn gian
  lận khỏi tập Train để phục vụ huấn luyện Học không giám sát (Novelty Detection).

NGƯỜI PHỤ TRÁCH:
- Nguyễn Văn Hiển Vinh (B23DCVT461) - Trưởng nhóm / Data & Preprocessing Lead
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from src.config import RAW_DATA_PATH, RANDOM_STATE


def load_raw_data(filepath=RAW_DATA_PATH) -> pd.DataFrame:
    """
    Nạp dữ liệu thô từ file CSV và kiểm tra tính toàn vẹn (Shape check).
    
    GIẢNG VIÊN HỎI: Dữ liệu này có bao nhiêu dòng, bao nhiêu cột?
    BẠN TRẢ LỜI: 
    - Có chính xác 284.807 dòng giao dịch và 31 cột thuộc tính.
    - Cột 'Class' là nhãn (0: Normal, 1: Fraud).
    - Cột 'Time' và 'Amount' là biến số thực chưa chuẩn hóa.
    - Cột 'V1' đến 'V28' là các biến đã được bảo mật qua PCA.
    """
    if not filepath.exists():
        raise FileNotFoundError(
            f"[LỖI] Không tìm thấy dataset tại đường dẫn: {filepath}.\n"
            "Vui lòng xem hướng dẫn tại file data/README.md để tải 'creditcard.csv' từ Kaggle."
        )
    
    # Đọc file CSV bằng pandas
    df = pd.read_csv(filepath)
    
    # Kiểm tra tính toàn vẹn: đúng 284.807 dòng và 31 cột mới cho đi tiếp
    assert df.shape == (284807, 31), (
        f"[LỖI KIỂM KÊ] Kích thước dữ liệu không đúng: {df.shape}, kỳ vọng: (284807, 31)"
    )
    
    return df


def split_data_zero_leakage(df: pd.DataFrame, random_state: int = RANDOM_STATE):
    """
    Quy trình Phân chia Phân tầng 3 tập (Stratified 3-Way Split) chuẩn V2.1:
    
    TỔNG THỂ DỮ LIỆU: 284.807 dòng (284.315 Normal + 492 Fraud)
    
    BƯỚC 1: Chia 70% Train vs 30% Tạm thời (Validation + Test)
    - Tách có phân tầng (stratify=df['Class']) để đảm bảo tỷ lệ 0.1727% đồng đều.
    - Tập Train partition: 199.364 dòng (199.020 Normal + 344 Fraud).
    - Tập Temp partition: 85.443 dòng (85.295 Normal + 148 Fraud).
    
    BƯỚC 2: Chia đôi tập Temp thành Validation (15%) và Test (15%)
    - Tách có phân tầng (stratify=temp_part['Class']).
    - Validation Set: 42.721 dòng (42.647 Normal + 74 Fraud ~ 0.1732%).
    - Test Set:       42.722 dòng (42.648 Normal + 74 Fraud ~ 0.1732%).
    
    BƯỚC 3: Xử lý tập Train cho bài toán Novelty Detection (QUAN TRỌNG NHẤT!)
    - Theo định nghĩa Novelty Detection, mô hình CHỈ ĐƯỢC PHÉP học trên người tốt (Class 0).
    - Ta loại bỏ 344 mẫu Fraud ra khỏi quá trình fit mô hình.
    - Tập train trả về (train_df): đúng 199.020 mẫu Normal (0 Fraud).
    
    GIẢNG VIÊN HỎI: 
    - Tại sao không chuyển 344 mẫu Fraud ở tập Train sang tập Validation/Test?
    BẠN TRẢ LỜI:
    - Nếu tự ý chuyển 344 mẫu Fraud sang Val/Test, tỷ lệ gian lận ở Val/Test sẽ bị thổi phồng
      lên ~0.573% (gấp hơn 3 lần thực tế 0.172%). Điều đó làm sai lệch phân phối thực tế!
      Quy trình chuẩn khoa học là giữ nguyên tỷ lệ phân tầng tự nhiên ở Val và Test (~0.1732%),
      còn 344 mẫu Fraud trong vùng Train bị loại bỏ hoàn toàn khỏi việc fit mô hình và fit scaler.
    """
    # -------------------------------------------------------------------------
    # Bước 1: Phân tách 70% Train Partition và 30% Temp Partition (có phân tầng)
    # -------------------------------------------------------------------------
    train_part, temp_part = train_test_split(
        df,
        test_size=0.30,
        random_state=random_state,
        stratify=df["Class"],
        shuffle=True
    )

    # -------------------------------------------------------------------------
    # Bước 2: Phân tách 30% Temp thành 15% Validation và 15% Test (có phân tầng)
    # -------------------------------------------------------------------------
    val_df, test_df = train_test_split(
        temp_part,
        test_size=0.50,
        random_state=random_state,
        stratify=temp_part["Class"],
        shuffle=True
    )

    # -------------------------------------------------------------------------
    # Bước 3: Lọc lấy mẫu Normal cho tập Train (Novelty Detection Rule)
    # -------------------------------------------------------------------------
    train_normal = train_part[train_part["Class"] == 0].copy()
    train_fraud_excluded = train_part[train_part["Class"] == 1].copy()

    # Reset lại index cho sạch đẹp
    train_df = train_normal.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    # In thông tin kiểm chứng ra màn hình để đối soát
    print("=" * 70)
    print("XÁC NHẬN PHÂN CHIA DỮ LIỆU ZERO-LEAKAGE (CHUẨN V2.1):")
    print(f"1. Tổng dữ liệu gốc : {len(df):,} mẫu (284,315 Normal + 492 Fraud)")
    print(f"2. Phân vùng Train   : {len(train_part):,} mẫu ({len(train_fraud_excluded)} Fraud bị loại bỏ khỏi huấn luyện)")
    print(f"   -> Train_Normal   : {len(train_df):,} mẫu (Chỉ chứa Class 0, dùng fit scaler & model)")
    print(f"3. Tập Validation    : {len(val_df):,} mẫu ({val_df['Class'].sum()} Fraud ~ {val_df['Class'].mean():.4%})")
    print(f"4. Tập Test Set      : {len(test_df):,} mẫu ({test_df['Class'].sum()} Fraud ~ {test_df['Class'].mean():.4%}) [NIÊM PHONG]")
    print("=" * 70)

    return train_df, val_df, test_df
