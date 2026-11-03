# Credit Card Fraud Detection (Unsupervised Anomaly Detection)

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange.svg)](https://scikit-learn.org/)
[![Course](https://img.shields.io/badge/Course-Data%20Mining%20(PTIT)-red.svg)](#)
[![Status](https://img.shields.io/badge/Status-Sprint%2014%20Days-green.svg)](#)

> **Đồ án Bài tập lớn môn học:** Khai phá Dữ liệu (Data Mining) — Học viện Công nghệ Bưu chính Viễn thông (PTIT)  
> **Chủ đề IV:** Anomaly / Outlier Detection (Phát hiện dị biệt)  
> **Đề tài số 13:** Credit Card Fraud Detection  

---

## 👥 Nhóm Sinh viên Thực hiện

| STT | Họ và Tên | Mã Sinh Viên | Vai trò & Trách nhiệm chính | GitHub |
|:---:|:---|:---:|:---|:---:|
| 1 | **Nguyễn Văn Hiển Vinh** | **B23DCVT461** | **Trưởng nhóm** — Data & Preprocessing Lead (EDA, Pipeline 3-Way Split, RobustScaler, PCA/t-SNE) | [@clytusnguyen](https://github.com/clytusnguyen) |
| 2 | **Nguyễn Minh Hoàng** | **B23DCVT173** | **Kỹ sư Mô hình hóa** — Modeling & Evaluation Lead (iForest, LOF, OC-SVM, Threshold Moving, Cost Matrix) | [@hoangmelinh](https://github.com/hoangmelinh) |

* **Kế hoạch Dự án & Đặc tả SRS V2 chi tiết:** Xem tại file [`KE_HOACH_DU_AN_CREDIT_CARD_FRAUD_DETECTION_V2.docx`](./KE_HOACH_DU_AN_CREDIT_CARD_FRAUD_DETECTION_V2.docx) hoặc [`KE_HOACH_DU_AN_CREDIT_CARD_FRAUD_DETECTION_V2.md`](./KE_HOACH_DU_AN_CREDIT_CARD_FRAUD_DETECTION_V2.md).

---

## 🎯 Bối cảnh Bài toán & Tuyên bố Phạm vi (Scope)

* **Bản chất nghiệp vụ:** Trong thanh toán ngân hàng, các thủ đoạn gian lận thẻ tín dụng liên tục thay đổi (Zero-day Fraud). Nếu sử dụng phương pháp Học có giám sát (Supervised) kết hợp SMOTE, mô hình dễ bị rập khuôn theo các nhãn cũ và sinh mẫu giả mạo trong không gian 30 chiều.
* **Hướng tiếp cận chuẩn mực:** Áp dụng **Học không giám sát (Unsupervised Anomaly Detection)** để học cấu trúc của các giao dịch bình thường (Nominal Data). Mọi giao dịch bất thường nằm ở vùng mật độ thấp, khoảng cách xa lạ hoặc dễ bị cô lập đều tự động bị gắn cờ dị biệt.
* **Ranh giới đồ án:** Dự án tập trung 100% vào nghiên cứu học thuật, đối sánh thực nghiệm 3 thuật toán không giám sát và phân tích chi phí kinh tế (Cost-sensitive analysis). Dự án **không** mở rộng làm Web/App/Dashboard thương mại.

---

## 📊 Tập Dữ liệu (Dataset) & Chiến lược Chia 3 Tập Chống Rò rỉ

Dataset gốc: **Kaggle Credit Card Fraud Detection** ([Link Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)) gồm **284,807 giao dịch** (284,315 Normal vs. 492 Fraud ~ 0.172%), 30 thuộc tính đầu vào (Time, Amount, V1–V28) và 1 nhãn Class.

### Quy trình Phân chia 3 Tập Chống Rò rỉ (Zero-Leakage 3-Way Split):
* **Training Set (Huấn luyện):** Gồm **199,020 giao dịch** (100% Class 0 - Normal, **0 Fraud**). Nhãn Class bị loại bỏ hoàn toàn trong quá trình fit.
* **Validation Set (Kiểm định):** Gồm **42,893 giao dịch** (42,647 Normal + 246 Fraud ~ 0.573%). Dùng để tinh chỉnh siêu tham số và thực hiện kỹ thuật Dịch chuyển ngưỡng (Threshold-Moving).
* **Test Set (Kiểm thử độc lập):** Gồm **42,894 giao dịch** (42,648 Normal + 246 Fraud ~ 0.573%). Tuyệt đối niêm phong trong suốt quá trình thử nghiệm, chỉ mở ra ở bước nghiệm thu cuối cùng.
* **Tiền xử lý:** Áp dụng `RobustScaler` trên thuộc tính `Amount` và `Time` (sử dụng Median & IQR). **Fit duy nhất trên Training set**, chỉ transform trên Validation và Test set.

---

## 🤖 3 Thuật toán Không giám sát & Chuẩn hóa Điểm Dị biệt

| Thuật toán | Cơ chế Hoạt động | Cấu hình & Chiến lược Lấy mẫu | Hàm tính Score Scikit-learn | Quy ước Điểm Dị biệt Thống nhất ($S$) |
|:---|:---|:---|:---|:---|
| **Isolation Forest (iForest)** | Phân cắt nhị phân ngẫu nhiên, cô lập điểm dị biệt nhanh ở độ sâu nhỏ | Huấn luyện trên toàn bộ 199,020 mẫu `Train_Normal`; `max_samples = 256`, `n_estimators = 100` | `score_samples(X)` *(mặc định âm)* | $S = -\text{score\_samples}(X)$<br>**(Score càng cao $\rightarrow$ Dị biệt càng lớn)** |
| **Local Outlier Factor (LOF)** | Đo lường tỷ số mật độ cục bộ so với $k$ lân cận gần nhất | Bắt buộc bật `novelty=True`. Lấy mẫu đại diện $N_{\text{LOF}} = 25,000$ từ `Train_Normal` để tránh tràn RAM $O(N^2)$ | `score_samples(X)` *(mặc định âm)* | $S = -\text{score\_samples}(X)$<br>**(Score càng cao $\rightarrow$ Dị biệt càng lớn)** |
| **One-Class SVM** | Tìm siêu phẳng tối ưu phân tách dữ liệu khỏi gốc tọa độ trong không gian RBF | Kernel RBF (`nu=0.01`, `gamma='scale'`). Lấy mẫu đại diện $N_{\text{OCSVM}} = 20,000$ từ `Train_Normal` | `decision_function(X)` *(mặc định âm ngoài biên)* | $S = -\text{decision\_function}(X)$<br>**(Score càng cao $\rightarrow$ Dị biệt càng lớn)** |

---

## 📁 Cấu trúc Thư mục Dự án (Project Directory Structure)

```text
DATAMINING/
├── .github/
│   └── pull_request_template.md        # Template chuẩn khi tạo Pull Request
├── data/
│   ├── raw/                            # Chứa file creditcard.csv gốc (gitignore)
│   ├── processed/                      # Chứa dữ liệu sau phân chia & scale (gitignore)
│   └── README.md                       # Hướng dẫn tải dữ liệu từ Kaggle
├── notebooks/
│   ├── 01_eda_overview.ipynb           # Khám phá phân phối, lệch lớp, KDE V1-V28
│   ├── 02_preprocessing_pipeline.ipynb # 3-Way Split & RobustScaler chống rò rỉ
│   ├── 03_isolation_forest.ipynb       # Huấn luyện, đo runtime & trích xuất score iForest
│   ├── 04_local_outlier_factor.ipynb   # Lấy mẫu N=25k, LOF novelty=True, n_neighbors
│   ├── 05_one_class_svm.ipynb          # Lấy mẫu N=20k, One-Class SVM RBF, nu, gamma
│   ├── 06_threshold_tuning_cost.ipynb  # Đối sánh PR-Curve, dịch chuyển ngưỡng & Cost Matrix
│   └── 07_dimensionality_reduction.ipynb # Trực quan hóa PCA 2D/3D & t-SNE (tô màu kép)
├── src/
│   ├── __init__.py
│   ├── config.py                       # Cấu hình đường dẫn, seed 42, tham số sampling
│   ├── data_loader.py                  # Module nạp dữ liệu và chia 3 tập Zero-leakage
│   ├── preprocessor.py                 # Module tiền xử lý với RobustScaler
│   ├── models.py                       # Wrapper 3 mô hình & hàm đảo dấu score chuẩn mực
│   └── metrics.py                      # Hàm tính Precision, Recall, F1, AUPRC, Cost
├── reports/
│   └── figures/                        # Nơi lưu biểu đồ xuất từ notebooks
├── KE_HOACH_DU_AN_CREDIT_CARD_FRAUD_DETECTION_V2.docx # Bản kế hoạch SRS Word nộp GV
├── KE_HOACH_DU_AN_CREDIT_CARD_FRAUD_DETECTION_V2.md   # Bản kế hoạch SRS Markdown
├── .gitignore                          # Bỏ qua data nặng, venv, cache, checkpoint
├── requirements.txt                    # Thư viện Python cần cài đặt
└── README.md                           # Tài liệu hướng dẫn dự án
```

---

## 🚀 Hướng dẫn Cài đặt & Tái lập Kết quả (Quickstart)

### 1. Clone Repository & Thiết lập Môi trường Ảo:
```bash
# Clone repository về máy cá nhân
git clone <URL_REPO_CUA_BAN>
cd DATAMINING

# Khởi tạo môi trường ảo Python 3.10+
python -m venv venv

# Kích hoạt môi trường ảo:
# Trên Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# Trên Linux/macOS:
source venv/bin/activate

# Nâng cấp pip và cài đặt toàn bộ thư viện:
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Tải Dataset:
1. Tải file `creditcard.csv` từ [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).
2. Di chuyển file vào thư mục `data/raw/creditcard.csv`.

### 3. Chạy Thực nghiệm trên Jupyter Lab:
```bash
jupyter lab
```
Khởi chạy lần lượt các Notebook từ `01_eda_overview.ipynb` đến `07_dimensionality_reduction.ipynb`. Mọi bước đều đã cố định `random_state = 42` để đảm bảo kết quả trùng khớp 100%.

---

## 🌿 Quy trình Phối hợp Nhóm trên Git (Git Workflow)

Nhóm áp dụng mô hình phân nhánh chuẩn để tránh xung đột mã nguồn:

* **Nhánh `main`:** Nhánh chính chứa phiên bản ổn định nhất của mã nguồn và báo cáo. Mọi cập nhật đều thông qua Pull Request (PR) được duyệt chéo.
* **Nhánh `vinh`:** Nhánh làm việc của Nguyễn Văn Hiển Vinh (Data, Preprocessing, EDA, Visualization).
* **Nhánh `hoang`:** Nhánh làm việc của Nguyễn Minh Hoàng (Modeling, Tuning, Evaluation, Cost Matrix).
* **Quy tắc làm việc:**
  1. Luôn `git pull origin main` trước khi bắt đầu công việc ngày mới.
  2. Tạo branch hoặc commit trên nhánh cá nhân.
  3. Mở Pull Request vào `main` kèm checklist trong `.github/pull_request_template.md`.
  4. Họp Sync 15 phút mỗi ngày vào lúc **21:00** để review code chéo và merge PR.

---

## 📈 Hệ thống Chỉ số Đánh giá & Ma trận Chi phí Kinh tế

Do dữ liệu lệch lớp nghiêm trọng (0.172% Fraud), nhóm **tuyệt đối không sử dụng Accuracy** (Accuracy Paradox). Các chỉ số đánh giá cốt lõi:

* **Recall:** Tỷ lệ gian lận phát hiện được: $\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}$ (quan trọng nhất do chi phí bỏ lọt gian lận cực cao).
* **Precision:** Mức độ chuẩn xác của cảnh báo: $\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}$.
* **AUPRC:** Diện tích dưới đường cong Precision-Recall, tính toán từ chuỗi Anomaly Score liên tục.
* **Mô hình Chi phí Kinh tế (Cost-sensitive Function):**
  $$\text{Total Cost}(\tau) = C_{\text{FN}} \cdot \text{FN}(\tau) + C_{\text{FP}} \cdot \text{FP}(\tau)$$
  Với giả định chi phí nghiên cứu: $C_{\text{FP}} = 1\text{ USD}$, $C_{\text{FN}}$ được khảo sát độ nhạy ở các mốc $20\text{ USD}$, $30\text{ USD}$, $50\text{ USD}$ để tìm ngưỡng tối ưu $\tau^*$ giảm thiểu tối đa tổn thất tài chính.

---

*Học viện Công nghệ Bưu chính Viễn thông — 2026*
