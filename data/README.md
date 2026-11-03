# Quản lý Dữ liệu Thực nghiệm (Data Management)

Thư mục này dùng để lưu trữ dữ liệu thực nghiệm theo quy chuẩn quản lý dữ liệu khoa học:

- `data/raw/`: Chứa file dữ liệu gốc `creditcard.csv` tải về từ Kaggle. File này được đưa vào `.gitignore` để tránh đẩy dữ liệu lớn (>150MB) lên GitHub.
- `data/processed/`: Chứa dữ liệu sau khi thực hiện phân chia 3 tập (`train_normal.parquet`, `val.parquet`, `test.parquet`) và sau khi chuẩn hóa bằng `RobustScaler`.

## Hướng dẫn tải tập dữ liệu Kaggle:

1. Truy cập link dataset: [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).
2. Tải file nén `archive.zip` (hoặc `creditcardfraud.zip`) về máy.
3. Giải nén file `creditcard.csv` vào trực tiếp thư mục `data/raw/creditcard.csv`.
4. Kiểm tra mã băm SHA-256 hoặc số dòng dữ liệu:
   - Tổng số dòng: **284,807 giao dịch**.
   - Số thuộc tính: **30 biến đầu vào + 1 biến Class**.
