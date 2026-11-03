import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from src.config import RAW_DATA_PATH, RANDOM_STATE

def load_raw_data(filepath=RAW_DATA_PATH) -> pd.DataFrame:
    """Load raw dataset from CSV with verification."""
    if not filepath.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {filepath}.\n"
            "Please follow data/README.md to download 'creditcard.csv' from Kaggle."
        )
    df = pd.read_csv(filepath)
    assert df.shape == (284807, 31), f"Unexpected data shape: {df.shape}"
    return df

def split_data_zero_leakage(df: pd.DataFrame, random_state: int = RANDOM_STATE):
    """
    3-Way Zero-Leakage Split:
    - Train_Normal: 70% Class 0 (199,020 samples, 0 Fraud)
    - Val: 15% Class 0 (42,647 samples) + 50% Fraud (246 samples)
    - Test: 15% Class 0 (42,648 samples) + 50% Fraud (246 samples)
    """
    normal_df = df[df["Class"] == 0].copy()
    fraud_df = df[df["Class"] == 1].copy()

    # Step 1: Split Normal into Train (70%) and Temp (30%)
    train_normal, temp_normal = train_test_split(
        normal_df, test_size=0.30, random_state=random_state, shuffle=True
    )

    # Step 2: Split Temp Normal into Val (15%) and Test (15%)
    val_normal, test_normal = train_test_split(
        temp_normal, test_size=0.50, random_state=random_state, shuffle=True
    )

    # Step 3: Split Fraud 50:50 into Val and Test
    val_fraud, test_fraud = train_test_split(
        fraud_df, test_size=0.50, random_state=random_state, shuffle=True
    )

    # Combine Normal and Fraud for Validation and Test sets
    val_df = pd.concat([val_normal, val_fraud]).sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    test_df = pd.concat([test_normal, test_fraud]).sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    train_df = train_normal.reset_index(drop=True)

    print(f"[Data Split] Train_Normal : {len(train_df):,} samples (0 Fraud)")
    print(f"[Data Split] Validation   : {len(val_df):,} samples ({val_df['Class'].sum()} Fraud ~ {val_df['Class'].mean():.4%})")
    print(f"[Data Split] Test         : {len(test_df):,} samples ({test_df['Class'].sum()} Fraud ~ {test_df['Class'].mean():.4%})")

    return train_df, val_df, test_df
