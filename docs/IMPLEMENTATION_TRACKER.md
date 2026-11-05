# BẢNG THEO DÕI TIẾN ĐỘ TRIỂN KHAI DỰ ÁN (IMPLEMENTATION TRACKER)
## Đề tài 13: Credit Card Fraud Detection (Unsupervised Anomaly Detection)

* **Học phần:** Khai phá Dữ liệu (Data Mining) — Học viện Công nghệ Bưu chính Viễn thông (PTIT)
* **Kế hoạch tham chiếu:** `KE_HOACH_DU_AN_CREDIT_CARD_FRAUD_DETECTION_V2.1.md` / `.docx`
* **Nhóm sinh viên:**
  * **Nguyễn Văn Hiển Vinh (B23DCVT461)** — *Leader | Data & Preprocessing Lead*
  * **Nguyễn Minh Hoàng (B23DCVT173)** — *Modeling & Evaluation Lead*
* **Môi trường thực thi:** Python 3.12.10 (`.venv`), Windows 11, PowerShell
* **Dataset:** `data/raw/creditcard.csv` (284,807 dòng, 31 cột, 0 missing, 1,081 duplicates)
* **Ngày bắt đầu theo dõi:** 03/11/2026 | **Cập nhật lần cuối:** 03/11/2026

---

## 1. TỔNG QUAN TRẠNG THÁI THEO GIAI ĐOẠN (PHASE STATUS)

| Giai đoạn / Lộ trình | Trọng tâm kỹ thuật | Người phụ trách chính | Trạng thái | Bằng chứng / Ghi chú |
| :--- | :--- | :---: | :---: | :--- |
| **Phase 0 (Ngày 1)** | Môi trường, Git, Dataset & Cấu trúc dự án | Vinh (A) / Hoàng (R) | <span style="color:green">**DONE**</span> | Python 3.12.10, `.venv`, `requirements.txt`, Git repo branch `main`, dataset đã nạp. |
| **Phase 1 (Ngày 2–3)** | EDA Tổng quan, Phân tích Lệch lớp & Đặc trưng V1–V28 | Vinh (A) / Hoàng (C) | <span style="color:green">**DONE**</span> | Hoàn thành `01_eda_overview.ipynb` (16 cells có output) & 4 ảnh tại `reports/figures/`. |
| **Phase 2 (Ngày 4)** | Pipeline Phân tầng 70/15/15 & Module Metrics AP | Vinh (A-Pipe) / Hoàng (A-Met) | <span style="color:green">**DONE**</span> | Hoàn thành `02_preprocessing_pipeline.ipynb` (15 cells), đóng gói `data/processed/` & xuất ảnh 05. |
| **Phase 3 (Ngày 5–6)** | Mô hình 1: Isolation Forest & Tối ưu Ngưỡng | Hoàng (A) / Vinh (R) | <span style="color:green">**DONE**</span> | Hoàn thành `03_isolation_forest.ipynb` (17 cells Defense-Ready), Best Val AP=0.103518 (n=200, s=512), Tau*=0.656289 (F1=0.2282), Sensitivity Analysis (+0.0018 AP), xuất ảnh 06 & 07, lưu model & preprocessor joblib. |
| **Phase 4 (Ngày 7)** | Mô hình 2: Local Outlier Factor (`novelty=True`) | Hoàng (A) / Vinh (R) | **TODO** | Giới hạn thực nghiệm $N_{\text{LOF}}=25,000$ kiểm soát tài nguyên RAM. |
| **Phase 5 (Ngày 8)** | Mô hình 3: One-Class SVM (Kernel RBF) | Hoàng (A) / Vinh (C) | **TODO** | Subsampling $N_{\text{OCSVM}}=20,000$ kiểm soát QP solver. |
| **Phase 6 (Ngày 9–10)**| Chuẩn hóa Score, Đối sánh Val & Cost Matrix | Hoàng (A) / Vinh (R) | **TODO** | Đảo dấu $S=-\text{score}$, AP (`average_precision_score`), chốt $\tau^*$. |
| **Phase 7 (Ngày 11)** | Trực quan hóa Giảm chiều (PCA & t-SNE trên Val) | Vinh (A-PCA) / Hoàng (A-tSNE)| **TODO** | Tuyệt đối niêm phong Test Set; chỉ chạy trên Validation Set. |
| **Phase 8 (Ngày 12)** | Mở Niêm phong Test Set & Benchmark Độc lập | Hoàng (A-Test) / Vinh (A-Doc)| **TODO** | Đánh giá 01 lần duy nhất trên Test Set; hoàn thiện README. |
| **Phase 9 (Ngày 13–14)**| Báo cáo Kỹ thuật IEEE, Slide PPTX & Tổng duyệt | Vinh (A-Rep) / Hoàng (A-Sli)| **TODO** | Báo cáo 25–35 trang, Slide 20–25 trang, diễn tập phản biện. |

---

## 2. BẢNG THEO DÕI CHI TIẾT TỪNG NHIỆM VỤ (TASK-LEVEL TRACKER)

### Phase 0: Khởi động Dự án & Môi trường (Ngày 1)
| Task ID | Tên nhiệm vụ | Owner (RACI) | Trạng thái | Sản phẩm bàn giao (Deliverable) | Tiêu chí nghiệm thu (Acceptance Criteria) | Bằng chứng kiểm thử (Evidence / Test Command) |
| :---: | :--- | :---: | :---: | :--- | :--- | :--- |
| **T0.1** | Kiểm tra Git Repository & Branch | Hoàng (R) | **DONE** | Git repo branch `main` | Repo đã khởi tạo, liên kết remote origin sạch. | `git status` -> `On branch main` |
| **T0.2** | Cấu hình `.gitignore` bảo vệ Dataset | Vinh (A) | **DONE** | File `.gitignore` | `data/raw/*.csv` và `.venv/` không bị theo dõi bởi Git. | `git check-ignore data/raw/creditcard.csv` -> PASS (ignored) |
| **T0.3** | Môi trường Python 3.12 Virtualenv | Vinh (A) | **DONE** | Thư mục `.venv` | Cài đặt đầy đủ `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`. | `.\.venv\Scripts\python.exe -V` -> `Python 3.12.10` |
| **T0.4** | Xác thực Dataset thô Kaggle | Vinh (A) | **DONE** | `data/raw/creditcard.csv` | File tồn tại, kích thước 150,828,752 bytes, shape (284807, 31). | `pd.read_csv` nạp thành công không lỗi. |

---

### Phase 1: Khám phá Dữ liệu Lệch Cực đoan (EDA) (Ngày 2 & Ngày 3)
| Task ID | Tên nhiệm vụ | Owner (RACI) | Trạng thái | Sản phẩm bàn giao (Deliverable) | Tiêu chí nghiệm thu (Acceptance Criteria) | Bằng chứng kiểm thử (Evidence / Test Command) |
| :---: | :--- | :---: | :---: | :--- | :--- | :--- |
| **T1.1** | Kiểm tra Missing Values | Vinh (A) | **DONE** | Báo cáo kiểm kê trong notebook | Số lượng missing = 0 trên toàn bộ 284,807 dòng x 31 cột. | `df.isnull().sum().sum() == 0` |
| **T1.2** | Phân tích 1,081 Duplicates | Vinh (A) | **DONE** | Báo cáo phân tích nhãn trùng | Xác định 1,062 dòng Normal và 19 dòng Fraud; quyết định bảo lưu 100%. | `df[df.duplicated()]['Class'].value_counts()` |
| **T1.3** | Phân tích Mất cân bằng Lớp | Vinh (A) | **DONE** | Biểu đồ cột & Donut Chart | Normal: 284,315 (99.827%), Fraud: 492 (0.173%). Giải thích cấm dùng Accuracy. | Xuất file `reports/figures/01_class_distribution.png` |
| **T1.4** | Phân tích Đặc trưng Amount & Time | Vinh (A) | **DONE** | Thống kê mô tả & Biểu đồ 4 góc | Skewness Amount = 16.98, Median = 22 USD vs 9.25 USD; Time theo chu kỳ 24h. | Xuất file `reports/figures/02_amount_time_distribution.png` |
| **T1.5** | Ma trận Tương quan V1–V28 | Vinh (A) | **DONE** | Heatmap tương quan Pearson | Xác định Top âm (V17, V14, V12, V10) và Top dương (V11, V4). | Xuất file `reports/figures/03_correlation_heatmap.png` |
| **T1.6** | Phân tích Tách biệt Phân phối KDE | Vinh (A) | **DONE** | Biểu đồ KDE 6 đặc trưng | Đường cong mật độ phân tách rõ rệt giữa Normal và Fraud trên 6 đặc trưng then chốt. | Xuất file `reports/figures/04_kde_top_features.png` |

---

### Phase 2: Pipeline Phân chia 3 Tập Zero-Leakage & Chuẩn hóa (Ngày 4)
| Task ID | Tên nhiệm vụ | Owner (RACI) | Trạng thái | Sản phẩm bàn giao (Deliverable) | Tiêu chí nghiệm thu (Acceptance Criteria) | Bằng chứng kiểm thử (Evidence / Test Command) |
| :---: | :--- | :---: | :---: | :--- | :--- | :--- |
| **T2.1** | Module `data_loader.py` V2.1 | Vinh (A) | **DONE** | File `src/data_loader.py` | Stratified 70/15/15: Train (199,020 Normal, 0 Fraud), Val (42,721 - 74 Fraud), Test (42,722 - 74 Fraud). | `python -c "from src.data_loader import ..."` PASS |
| **T2.2** | Module `preprocessor.py` | Vinh (A) | **DONE** | File `src/preprocessor.py` | RobustScaler chỉ fit trên Train_Normal; giữ nguyên V1–V28. | `python -c "from src.preprocessor import ..."` PASS |
| **T2.3** | Module `metrics.py` AP & Cost | Hoàng (A) | **DONE** | File `src/metrics.py` | Tính AP chuẩn qua `average_precision_score`; hàm `cost_matrix_score`. | `python -c "from src.metrics import ..."` PASS |
| **T2.4** | Notebook Pipeline Tiền xử lý | Vinh (A) | **DONE** | `notebooks/02_preprocessing_pipeline.ipynb` | Minh chứng Zero-Leakage, kiểm tra phân phối sau RobustScaler. | Chạy thành công 15 cells có output, xuất ảnh 05, lưu data/processed/. |

---

### Phase 3: Huấn luyện, Tối ưu & Đánh giá Isolation Forest (Ngày 5 & Ngày 6)
| Task ID | Tên nhiệm vụ | Owner (RACI) | Trạng thái | Sản phẩm bàn giao (Deliverable) | Tiêu chí nghiệm thu (Acceptance Criteria) | Bằng chứng kiểm thử (Evidence / Test Command) |
| :---: | :--- | :---: | :---: | :--- | :--- | :--- |
| **T3.1** | Khôi phục Preprocessor Artifact & Metadata | Vinh (A) | **DONE** | `data/processed/preprocessor.joblib`, `pipeline_metadata.json` | Fit trên Train_Normal, bảo toàn schema 30 đặc trưng, lưu đầy đủ center_/scale_. | `joblib.load('data/processed/preprocessor.joblib')` PASS |
| **T3.2** | Huấn luyện Baseline Isolation Forest | Hoàng (A) | **DONE** | Notebook `03_isolation_forest.ipynb` (Cell 3-4) | 100 cây, max_samples=256, fit_time=0.43s, đảo dấu $S=-\text{score}$, Baseline Val AP = 0.093634. | `average_precision_score` tính đúng chuẩn |
| **T3.3** | Khảo sát Tham số Contamination | Hoàng (A) | **DONE** | Notebook `03_isolation_forest.ipynb` (Cell 5-6) | Thử nghiệm 4 mức (auto, 0.0017, 0.01, 0.05); chứng minh AP bất biến 100%, offset_ dịch chuyển. | Bảng so sánh thực nghiệm trong notebook |
| **T3.4** | Grid Search Siêu tham số trên Validation | Hoàng (A) | **DONE** | `reports/phase3_isolation_forest_results.csv` | Lưới 8 cấu hình ($n \in [50, 100, 150, 200], s \in [256, 512]$), chọn Best Model ($n=200, s=512$, Val AP = 0.103518). | CSV lưu đầy đủ 8 cấu hình và runtime |
| **T3.5** | Quét Ngưỡng Tối ưu $F_1$ & Xuất Biểu đồ | Hoàng (A) | **DONE** | `reports/figures/06_...png`, `07_...png` | Quét $\tau^* = 0.656289$, Best F1 = 0.2282 (P=0.2267, R=0.2297). Xuất đủ 2 biểu đồ PNG chuẩn 300 DPI. | `PIL.Image.verify()` PASS |
| **T3.6** | Duplicate Sensitivity Analysis | Vinh (A) | **DONE** | Báo cáo thực nghiệm trong notebook & summary | Cô lập 203 bản ghi trùng; đo đạc $\Delta\text{AP} = +0.001837$, $\Delta\text{Precision} = +0.003063$, $\Delta\text{Recall} = 0.000000$. | Bảng đối chiếu thực nghiệm không re-train |
| **T3.7** | Đóng gói Mô hình & Báo cáo Tóm tắt | Hoàng (A) | **DONE** | `models/isolation_forest_best.joblib`, `reports/phase3_isolation_forest_summary.md` | Model serialize chuẩn, test reload bit-for-bit khớp 100%, báo cáo MD hoàn chỉnh. | `joblib.load('models/isolation_forest_best.joblib')` PASS |

---

### Phase 4 đến Phase 9: Các Mô hình Tiếp theo, Benchmark & Báo cáo
*(Chi tiết các task T4.1 đến T9.3 được theo dõi tiếp tục khi hoàn thành Phase 3)*

---

## 3. BẢNG TỔNG HỢP RỦI RO KỸ THUẬT & BIỆN PHÁP DỰ PHÒNG

| Rủi ro kỹ thuật phát hiện | Mức độ | Hậu quả tiềm tàng | Biện pháp kiểm soát & Dự phòng |
| :--- | :---: | :--- | :--- |
| **R1: Xóa nhầm 19 mẫu Fraud khi drop duplicates** | **CAO** | Xóa mất 3.86% mẫu gian lận hiếm hoi (19 / 492 mẫu). | **Quy tắc bắt buộc:** Giữ nguyên 1,081 duplicates trong dataset; ghi chú lý do nghiệp vụ trong báo cáo. |
| **R2: Tràn bộ nhớ (OOM) khi huấn luyện LOF và OC-SVM** | **CAO** | Treo kernel Jupyter, crash bộ nhớ khi ma trận khoảng cách vượt 16GB RAM. | Thiết lập giới hạn thực nghiệm: $N_{\text{LOF}}=25,000$, $N_{\text{OCSVM}}=20,000$ với seed 42 để kiểm soát tài nguyên. |
| **R3: Rò rỉ dữ liệu kiểm thử (Data Snooping trên Test Set)** | **CAO** | Kết quả đánh giá bị lạc quan giả tạo, không tổng quát hóa được. | Niêm phong tuyệt đối Test Set; chỉ chạy PCA/t-SNE và chọn ngưỡng $\tau^*$ trên Validation Set. |
| **R4: Phương sai biên của Recall trên Test Set** | **TRUNG BÌNH** | Test Set có 74 mẫu Fraud nên 1 mẫu sai làm thay đổi ~1.35% Recall. | Bổ sung khoảng tin cậy 95% CI (Bootstrap 1,000 lần lặp) khi báo cáo kết quả cuối cùng. |
