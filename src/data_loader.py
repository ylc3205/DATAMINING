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
    Stratified 3-Way Zero-Leakage Split (V2.1 Protocol):
    - Full Stratified Split: 70% Train, 15% Validation, 15% Test
    - Train Partition: 199,364 total (199,020 Normal + 344 Fraud)
      -> Novelty Detection rule: 344 Fraud filtered out from fitting.
      -> train_df = Train_Normal (199,020 samples, 0 Fraud)
    - Validation Set: 42,721 total (42,647 Normal + 74 Fraud ~ 0.1732%)
    - Test Set: 42,722 total (42,648 Normal + 74 Fraud ~ 0.1732%)
    """
    # Step 1: Stratified Split 70% Train vs 30% Temp (Val + Test)
    train_part, temp_part = train_test_split(
        df, test_size=0.30, random_state=random_state, stratify=df["Class"], shuffle=True
    )

    # Step 2: Stratified Split Temp into Val (15%) and Test (15%)
    val_df, test_df = train_test_split(
        temp_part, test_size=0.50, random_state=random_state, stratify=temp_part["Class"], shuffle=True
    )

    # Step 3: Filter out Fraud from Train partition for Novelty Detection fitting
    train_normal = train_part[train_part["Class"] == 0].copy()
    train_fraud_excluded = train_part[train_part["Class"] == 1].copy()

    train_df = train_normal.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    print(f"[Data Split V2.1] Train Partition : {len(train_part):,} samples ({len(train_fraud_excluded)} Fraud excluded from fitting)")
    print(f"[Data Split V2.1] Train_Normal    : {len(train_df):,} samples (0 Fraud, used for model/scaler fitting)")
    print(f"[Data Split V2.1] Validation Set  : {len(val_df):,} samples ({val_df['Class'].sum()} Fraud ~ {val_df['Class'].mean():.4%})")
    print(f"[Data Split V2.1] Test Set        : {len(test_df):,} samples ({test_df['Class'].sum()} Fraud ~ {test_df['Class'].mean():.4%})")

    return train_df, val_df, test_df
