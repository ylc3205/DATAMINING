# BÁO CÁO KẾT QUẢ PHASE 4: LOCAL OUTLIER FACTOR (LOF)

* **Người phụ trách:** Nguyễn Minh Hoàng (B23DCVT173) & Nguyễn Văn Hiển Vinh (B23DCVT461)
* **Dữ liệu huấn luyện:** Lấy mẫu không hoàn lại $N_{\text{LOF}} = 25,000$ dòng từ `Train_Normal` (`RANDOM_STATE=42`, lưu tại `data/processed/train_indices_lof.npy`).
* **Mô hình tối ưu:** `LocalOutlierFactor(n_neighbors=30, novelty=True, metric='minkowski', p=2, contamination='auto')` (`_fit_method='brute'`, `effective_metric_='euclidean'`).
* **Thời gian thực thi Best Model:** `fit_time` = 3.8972s | `score_time` = 9.2551s (`batch_size=5,000`).

## 1. Kết quả Hiệu năng trên Tập Validation (42,721 mẫu, 74 Fraud)
* **Baseline LOF (`n_neighbors=20`):** Val AP = 0.550848 | Best $F_1$ = 0.6479 ($\tau^*=3.124853$).
* **Best LOF (`n_neighbors=30`):** **Val AP = 0.564948** (+0.014100 so với Baseline, tương đương +2.56%).
* **Ngưỡng tối ưu $F_1$ ($\tau^*_{\text{LOF}}$):** `3.163276`
* **Precision tại $\tau^*$:** `0.7188` (71.88%)
* **Recall tại $\tau^*$:** `0.6216` (46/74 ca gian lận)
* **F1-Score tại $\tau^*$:** `0.6667`
* **Ma trận nhầm lẫn:** `TP=46`, `FP=18`, `TN=42629`, `FN=28`

## 2. Kết quả Coarse Grid Search (`n_neighbors ∈ [10, 20, 30, 50]`)
| Model ID | `n_neighbors` | Fit Method | Fit (s) | Score (s) | Val AP | Optimal $F_1$ | RSS After (MB) | Heap Peak (MB) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| #1 | 10 | `brute` | 3.8315 | 7.3569 | **0.301756** | 0.4740 | 267.37 | 8.50 |
| #2 | 20 | `brute` | 3.7972 | 7.2257 | **0.550848** | 0.6479 | 262.57 | 16.56 |
| #3 | 30 | `brute` | 3.8972 | 9.2551 | **0.564948** | 0.6667 | 275.54 | 24.23 |
| #4 | 50 | `brute` | 3.8426 | 7.8219 | **0.554866** | 0.6479 | 286.36 | 39.97 |

## 3. Duplicate Sensitivity Analysis (Không refit, giữ nguyên $\tau^*_{\text{LOF}}$)
* **Check A — Loại 45 dòng Validation trùng trực tiếp `X_train_lof` (0 Fraud):** Còn `42,676` mẫu (74 Fraud); AP = `0.564951` (`Delta AP = +0.000003`); $F_1$ = `0.6667` (`Delta F1 = +0.0000`).
* **Check B — Loại 203 dòng Validation trùng toàn bộ `Train_Normal` (0 Fraud):** Còn `42,518` mẫu (74 Fraud); AP = `0.565006` (`Delta AP = +0.000058`); $F_1$ = `0.6667` (`Delta F1 = +0.0000`).
* **Giải thích toán học:** Do khoảng cách tiếp cận $\text{reach-dist}_k(p, o) = \max\{k\text{-dist}(o), d(p, o)\}$ chặn dưới bởi $k\text{-dist}(o) > 0$, toàn bộ bản sao chỉ nhận điểm trong dải $[0.9536, 2.2352] < \tau^*$, không gây bất kỳ ca FP mới nào.

## 4. Artifacts Bàn giao Phase 4
1. `notebooks/04_local_outlier_factor.ipynb`
2. `data/processed/train_indices_lof.npy`
3. `models/lof_best.joblib` (11.83 MB — đã kiểm chứng `joblib.load()` khớp bit-for-bit)
4. `reports/phase4_lof_results.csv`
5. `reports/phase4_lof_metadata.json`
6. `reports/phase4_lof_summary.md`
7. `reports/figures/08_lof_score_distribution.png`
8. `reports/figures/09_lof_pr_curve.png`

*Lưu ý bảo mật:* Tập Test Set (`X_test`, `y_test`) hoàn toàn không được nạp hoặc sử dụng trong Phase 4.