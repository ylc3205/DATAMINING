# BÁO CÁO KẾT QUẢ PHASE 3: ISOLATION FOREST

* **Người phụ trách:** Nguyễn Minh Hoàng & Nguyễn Văn Hiển Vinh
* **Mô hình tối ưu:** Isolation Forest (`n_estimators=200`, `max_samples=512`, `random_state=42`)
* **Thời gian huấn luyện:** 1.1806 giây | **Thời gian suy luận:** 0.4936 giây

## 1. Kết quả Hiệu năng trên Tập Validation (42,721 mẫu, 74 Fraud)
* **Baseline AP (100 cây, 256 samples):** 0.093634
* **Best Validation AP:** **0.103518** (++0.009884)
* **Ngưỡng tối ưu F1 (Tau*):** 0.656289
* **Precision tại Tau*:** 0.2267 (22.67%)
* **Recall tại Tau*:** 0.2297 (17/74 ca gian lận)
* **F1-Score tại Tau*:** 0.2282
* **Ma trận nhầm lẫn:** TP=17, FP=58, TN=42589, FN=57

## 2. Kiểm chứng Khảo sát Contamination
* Khảo sát `contamination in ['auto', 0.0017, 0.01, 0.05]`: AP giữ nguyên bất biến 100% (0.093634).
* `offset_` dịch chuyển từ -0.500000 đến -0.650838, chứng minh tính bất biến xếp hạng của Anomaly Score.

## 3. Kết quả Duplicate Sensitivity Analysis
* Số lượng bản ghi trùng lặp được cô lập: **203 bản ghi Normal** (0 Fraud).
* **Validation Đầy đủ:** AP = 0.103518, F1 = 0.2282
* **Validation Khử trùng:** AP = 0.105355, F1 = 0.2297
* **Độ lệch thực nghiệm:** Delta AP = +0.001837, Delta F1 = +0.0015, Delta Recall = 0.000000.

## 4. Artifacts đã bàn giao
1. `models/isolation_forest_best.joblib`
2. `data/processed/preprocessor.joblib`
3. `data/processed/pipeline_metadata.json`
4. `reports/phase3_isolation_forest_results.csv`
5. `reports/figures/06_isolation_forest_score_distribution.png`
6. `reports/figures/07_isolation_forest_pr_curve.png`
