# Phát hiện Gian lận Thẻ tín dụng bằng Học Không Giám Sát (Credit Card Fraud Detection — Unsupervised Anomaly Detection)

[![Python](https://img.shields.io/badge/Python-3.12.10-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.x-orange.svg)](https://scikit-learn.org/)
[![Status](https://img.shields.io/badge/Status-Phases%200--8%20Complete-brightgreen.svg)](#ti%E1%BA%BFn-%C4%91%E1%BB%99-tri%E1%BB%83n-khai--notebooks)

* **Repository:** [https://github.com/ylc3205/DATAMINING.git](https://github.com/ylc3205/DATAMINING.git)
* **Học phần:** Khai phá Dữ liệu (Data Mining) — Học viện Công nghệ Bưu chính Viễn thông (PTIT)
* **Nhóm thực hiện:**
  * **Nguyễn Văn Hiển Vinh** (`B23DCVT461`)
  * **Nguyễn Minh Hoàng** (`B23DCVT173`)

---

## 1. Tổng quan Đề tài & Động lực Nghiên cứu

Trong giao dịch thanh toán thẻ tín dụng, tỷ lệ gian lận thực tế rất thấp (mất cân bằng lớp cực đoan) và hành vi gian lận thay đổi liên tục theo thời gian. Đề tài tiếp cận bài toán dưới góc độ **Phát hiện Dị biệt Không giám sát (Unsupervised Novelty Detection)**:

1. **Huấn luyện thuần túy trên dữ liệu giao dịch bình thường (`Train_Normal`)**, tách bỏ hoàn toàn nhãn gian lận khỏi tập huấn luyện để mô hình tự học biên phân phối hành vi hợp lệ.
2. **Thiết kế pipeline chống rò rỉ dữ liệu tuyệt đối (Zero Data Leakage)**:
   * Chuẩn hóa `RobustScaler` (`Time`, `Amount`) chỉ `fit` trên `Train_Normal`.
   * Toàn bộ chọn siêu tham số, chuẩn hóa điểm dị biệt $S(x) = -\text{score\_samples}(x)$, đối sánh mô hình, tối ưu ngưỡng cắt chính xác $O(N \log N)$ và phân tích trực quan hóa PCA/t-SNE được thực hiện trên **Validation Set**.
   * **Test Set** được niêm phong hoàn toàn trong suốt Phases 1–7 và chỉ mở **01 lần duy nhất** tại Phase 8 để nghiệm thu bằng các mô hình và ngưỡng cắt đã đóng băng.
3. **Đánh giá gắn với Ma trận Chi phí Bất đối xứng**: Sử dụng **Average Precision (AUPRC)** thay cho Accuracy bị sai lệch khi mất cân bằng lớp, kết hợp tối ưu ngưỡng cắt theo **đơn vị chi phí tương đối** ($C_{\text{FP}} = 1$, $C_{\text{FN}} = 30$) kèm phân tích độ nhạy ở 5 tỷ lệ (`10:1`, `20:1`, `30:1`, `50:1`, `100:1`).

---

## 2. Bộ dữ liệu & Quy trình Tiền xử lý (`creditcard.csv`)

* **Nguồn dữ liệu:** Kaggle Credit Card Fraud Detection (`284,807` giao dịch thẻ tín dụng tại châu Âu trong 2 ngày tháng 9/2013; `31` cột: `Time`, `V1`–`V28` từ PCA, `Amount`, `Class`).
* **Tính toàn vẹn & Trùng lặp:** `0` giá trị thiếu (`NaN`). Toàn bộ `1,081` bản ghi trùng lặp chính xác (gồm `1,062` Normal và `19` Fraud — chiếm `3.86%` tổng số Fraud) được **giữ lại có chủ đích** để không làm mất mẫu gian lận thực tế và được kiểm chứng bằng *Duplicate Sensitivity Analysis*.
* **Phân chia phân tầng 70 / 15 / 15 (`RANDOM_STATE = 42`):**

| Phân vùng | Tổng mẫu | Giao dịch Bình thường (`Class=0`) | Gian lận (`Class=1`) | Tỷ lệ Gian lận | Vai trò |
| :--- | ---: | ---: | ---: | ---: | :--- |
| **`Train_Normal`** | `199,020` | `199,020` | `0` *(đã tách `344` Fraud)* | `0.0000%` | Huấn luyện 3 thuật toán không giám sát & fit `RobustScaler` |
| **`Validation`** | `42,721` | `42,647` | `74` | `0.1732%` | Tuning siêu tham số, đối sánh mô hình, tìm $\tau^*$ & PCA/t-SNE |
| **`Test` (Niêm phong)** | `42,722` | `42,648` | `74` | `0.1732%` | Nghiệm thu độc lập 01 lần duy nhất ở Phase 8 |
| **Toàn bộ gốc** | `284,807` | `284,315` | `492` | `0.1727%` | — |

---

## 3. Ba Thuật toán Không giám sát & Thiết kế Thực nghiệm

Cả ba mô hình đều quy về **điểm dị biệt thống nhất** qua hàm `compute_unified_anomaly_scores()` trong [`src/models.py`](file:///c:/Users/Vinh%20Nguyen/DATAMINING/src/models.py):
$$S(x) = -\text{score\_samples}(x) \quad (\text{điểm càng lớn} \Rightarrow \text{mức độ bất thường càng cao})$$

1. **Isolation Forest** (`models/isolation_forest_best.joblib`, `2.41 MB`): Huấn luyện trên toàn bộ `199,020` mẫu `Train_Normal`; cấu hình tối ưu trên Validation: `n_estimators=200`, `max_samples=512`.
2. **Local Outlier Factor — `novelty=True`** (`models/lof_best.joblib`, `11.83 MB`): Lấy mẫu kiểm soát `25,000` giao dịch từ `Train_Normal` (`data/processed/train_indices_lof.npy`, `RandomState=42`); cấu hình tối ưu: `n_neighbors=30`, `metric='minkowski' (p=2)`.
3. **One-Class SVM — RBF Kernel** (`models/ocsvm_best.joblib`, `0.06 MB`): Lấy mẫu kiểm soát `20,000` giao dịch từ `Train_Normal` (`data/processed/train_indices_ocsvm.npy`, `RandomState=42`); cấu hình tối ưu: `kernel='rbf'`, `nu=0.01`, `gamma=0.01` (`251` support vectors).

---

## 4. Kết quả Thực nghiệm Chính thức (Validation & Final Test)

* **Chính sách chọn Champion (đóng băng tại Phase 6):** Chọn mô hình có **tổng chi phí Validation thấp nhất** tại tỷ lệ chính $C_{\text{FP}} = 1, C_{\text{FN}} = 30$ (đơn vị chi phí tương đối). Ngưỡng $\tau^*$ được tính chính xác không làm tròn bằng thuật toán $O(N \log N)$ gom nhóm điểm trùng nhau (`find_optimal_cost_threshold_exact` trong [`src/metrics.py`](file:///c:/Users/Vinh%20Nguyen/DATAMINING/src/metrics.py)).
* **Mô hình Quán quân (Champion):** **Local Outlier Factor (LOF)** với ngưỡng đóng băng $\tau^* = 2.4462849810364786$.

### Bảng 1: Hiệu năng trên Tập Validation (`42,721` mẫu, `74` Fraud)

| Mô hình | Val AP (AUPRC) | Val ROC-AUC | $\tau^*_{\text{Cost}}$ (`30:1`) | TP / FP / FN / TN | Precision | Recall | F1-Score | Tổng chi phí (`30:1`, đơn vị) | Latency Val (ms/tx) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Local Outlier Factor (Champion)** | **0.5649** | 0.9402 | `2.446285` | `54 / 60 / 20 / 42,587` | **0.4737** | 0.7297 | **0.5745** | **660.0** | `0.0817` |
| **One-Class SVM** | 0.2515 | 0.9006 | `1.407521` | `57 / 195 / 17 / 42,452` | 0.2262 | **0.7703** | 0.3497 | 705.0 | `0.0285` |
| **Isolation Forest** | 0.1035 | **0.9434** | `0.560519` | `46 / 447 / 28 / 42,200` | 0.0933 | 0.6216 | 0.1623 | 1287.0 | **0.0149** |

> *Ghi chú phân tích độ nhạy chi phí trên Validation (`C_FP = 1`):* LOF đạt chi phí thấp nhất ở `10:1` (`256.0`), `20:1` (`460.0`) và `30:1` (`660.0`), trong khi One-Class SVM đạt chi phí thấp hơn ở `50:1` (`1037.0` vs LOF `1060.0`) và `100:1` (`1837.0` vs LOF `2020.0`) nhờ Recall cao hơn (`0.7703` vs `0.7297`). Giá trị F1-max độc lập trên Validation: LOF `0.6667` ($\tau^*=3.163276$), OCSVM `0.4113` ($\tau^*=2.623563$), iForest `0.2282` ($\tau^*=0.656289$).

### Bảng 2: Kết quả Nghiệm thu Độc lập trên Tập Test Niêm phong (`42,722` mẫu, `74` Fraud, ngưỡng đóng băng)

| Mô hình | Test AP (AUPRC) | $\Delta$ AP (Test−Val) | Test ROC-AUC | TP / FP / FN / TN | Test Precision | Test Recall | Test F1 | Test Cost (`30:1`, đơn vị) | $\Delta$ Cost | Latency Test (ms/tx) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Local Outlier Factor (Champion)** | **0.6594** | `+0.0944` | 0.9310 | `55 / 58 / 19 / 42,590` | **0.4867** | **0.7432** | **0.5882** | **628.0** | `-32.0` | `0.0179` *(Cell 8: `0.0175`)* |
| **One-Class SVM** | 0.3135 | `+0.0620` | 0.9262 | `55 / 181 / 19 / 42,467` | 0.2331 | **0.7432** | 0.3548 | 751.0 | `+46.0` | `0.0152` |
| **Isolation Forest** | 0.1262 | `+0.0227` | **0.9450** | `40 / 437 / 34 / 42,211` | 0.0839 | 0.5405 | 0.1452 | 1457.0 | `+170.0` | **0.0084** |

* **Kết luận nghiệm thu:** LOF giữ vững ngôi quán quân trên cả Validation và Test với tốc độ tổng quát hóa ổn định (không overfitting): bắt đúng `55/74` giao dịch gian lận (`74.32%`), giảm `3.1x` số báo động nhầm so với OC-SVM (`58` vs `181` FP) và giảm `7.5x` so với Isolation Forest (`58` vs `437` FP), đạt tổng chi phí tổn thất tương đối thấp nhất (`628.0` đơn vị).
* **Trực quan hóa giảm chiều (Phase 7 trên tập con Validation `5,074` mẫu = `5,000 Normal + 74 Fraud`):** PCA 2D giải thích `40.16%` phương sai tích lũy (`PC1=22.85%`, `PC2=17.30%`); t-SNE (`perplexity=35`, `KL=1.2190`) minh họa rõ cụm hành vi bình thường và phần lớn điểm Fraud tách ra rìa biên, lý giải trực quan vì sao phương pháp mật độ cục bộ LOF vượt trội hơn siêu phẳng toàn cục OC-SVM.

---

## 5. Tiến độ Triển khai & Cấu trúc Dự án

```text
DATAMINING/
├── data/
│   ├── raw/                  # Đặt creditcard.csv tải từ Kaggle tại đây (gitignored)
│   └── processed/            # creditcard_preprocessed.npz, preprocessor.joblib, train_indices_*.npy
├── models/                   # Checkpoint đã huấn luyện (isolation_forest_best, lof_best, ocsvm_best)
├── notebooks/
│   ├── 01_eda_overview.ipynb                # Phase 1: EDA, missing, duplicates, tương quan V1-V28
│   ├── 02_preprocessing_pipeline.ipynb      # Phase 2: Stratified split 70/15/15 & RobustScaler
│   ├── 03_isolation_forest.ipynb            # Phase 3: Isolation Forest baseline, grid search & dup check
│   ├── 04_local_outlier_factor.ipynb        # Phase 4: LOF novelty=True, grid search k & dup check
│   ├── 05_one_class_svm.ipynb               # Phase 5: OC-SVM RBF tuning & support vectors analysis
│   ├── 06_threshold_tuning_cost.ipynb       # Phase 6: Chuẩn hóa score, exact Cost/F1 thresholds & Champion
│   ├── 07_dimensionality_reduction.ipynb    # Phase 7: PCA & t-SNE trên Validation (Test vẫn niêm phong)
│   └── 08_final_test_evaluation.ipynb       # Phase 8: Mở niêm phong Test Set 01 lần duy nhất & nghiệm thu
├── reports/
│   ├── *.csv, *.json, *.md                  # Bảng kết quả, metadata đóng băng & báo cáo từng pha
│   └── figures/                             # 17 biểu đồ thực nghiệm chuẩn 300 DPI (01–17)
├── src/
│   ├── config.py, data_loader.py, preprocessor.py
│   └── models.py, metrics.py
├── slides/                                  # Slide bảo vệ PPTX / PDF (nội bộ, gitignored)
├── docs/                                    # Kế hoạch, Tracker & Báo cáo kỹ thuật DOCX/PDF (gitignored)
└── requirements.txt
```

---

## 6. Hướng dẫn Tái lập Thực nghiệm (Local Reproduction)

> **Lưu ý:** Dữ liệu thô (`creditcard.csv`), mô hình nhị phân (`models/*.joblib`), tệp dữ liệu đã xử lý (`*.npz`) và biểu đồ (`reports/figures/*.png`) được loại trừ khỏi Git qua `.gitignore` để giữ repository gọn nhẹ và tuân thủ bản quyền Kaggle.

1. **Clone repository và tạo môi trường ảo Python 3.12:**
   ```bash
   git clone https://github.com/ylc3205/DATAMINING.git
   cd DATAMINING
   python -m venv .venv
   # Windows PowerShell:
   .\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```
2. **Chuẩn bị dữ liệu thô:**
   * Tải bộ dữ liệu *Credit Card Fraud Detection* từ Kaggle (`creditcard.csv`, `284,807` dòng) và đặt vào đường dẫn `data/raw/creditcard.csv`.
3. **Chạy tuần tự các Jupyter Notebook (`01` → `08`):**
   * Mở bằng JupyterLab / VS Code với kernel `.venv` và chạy lần lượt từ `notebooks/01_eda_overview.ipynb` đến `notebooks/08_final_test_evaluation.ipynb` để tái tạo toàn bộ phân tách dữ liệu, mô hình đã đóng gói, bảng tổng hợp và 17 biểu đồ thực nghiệm.

---

## 7. Giới hạn Khoa học & Hướng Phát triển

1. **Ma trận chi phí giả định (`C_FP = 1, C_FN = 30`):** Là thước đo chuẩn hóa phục vụ so sánh thuật toán trong phạm vi học thuật (đơn vị chi phí tương đối, không quy đổi thành USD). Trong hệ thống thực tế, tổn thất bỏ sót phụ thuộc trực tiếp vào giá trị giao dịch `Amount`, chi phí hoàn tiền và xử lý khiếu nại.
2. **Dữ liệu tĩnh & Concept Drift:** Bộ dữ liệu Kaggle phản ánh lát cắt tĩnh trong 2 ngày (2013) với đặc trưng V1–V28 đã ẩn danh qua PCA; chưa mô phỏng biến đổi hành vi theo thời gian (*Concept Drift*) hay thông tin đồ thị/chuỗi giao dịch theo chủ thẻ.
3. **Giới hạn quy mô & Latency benchmark:** Do độ phức tạp khoảng cách/QP solver, LOF huấn luyện trên mẫu kiểm soát `25,000` giao dịch và OC-SVM trên `20,000` giao dịch; các giá trị độ trễ `< 1 ms/tx` đo trong notebook phản ánh tốc độ chấm điểm batch cục bộ bằng Python/scikit-learn, **không** đại diện cho SLA đầu-cuối của cổng thanh toán ngân hàng.
