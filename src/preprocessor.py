import pandas as pd
from sklearn.preprocessing import RobustScaler

class ZeroLeakagePreprocessor:
    """
    Data Preprocessing Pipeline:
    - Scales 'Amount' and 'Time' using RobustScaler (Median & IQR).
    - Preserves 'V1'-'V28' as they are already PCA-transformed.
    - Strictly fits ONLY on the Training set (Class 0 only) to prevent leakage.
    """
    def __init__(self):
        self.scaler_amount = RobustScaler()
        self.scaler_time = RobustScaler()
        self.feature_cols = [f"V{i}" for i in range(1, 29)] + ["scaled_amount", "scaled_time"]
        self.is_fitted = False

    def fit(self, train_df: pd.DataFrame):
        """Fit scalers strictly on the training set."""
        self.scaler_amount.fit(train_df[["Amount"]])
        self.scaler_time.fit(train_df[["Time"]])
        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform Amount and Time and return clean feature matrix."""
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted on Training set before transform!")

        df_out = df.copy()
        df_out["scaled_amount"] = self.scaler_amount.transform(df[["Amount"]])
        df_out["scaled_time"] = self.scaler_time.transform(df[["Time"]])

        X = df_out[self.feature_cols].copy()
        y = df_out["Class"].values if "Class" in df_out.columns else None
        return X, y

    def fit_transform(self, train_df: pd.DataFrame):
        self.fit(train_df)
        return self.transform(train_df)
