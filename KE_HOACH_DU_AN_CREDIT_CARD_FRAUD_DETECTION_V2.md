# KẾ HOẠCH DỰ ÁN & ĐẶC TẢ YÊU CẦU KỸ THUẬT (SRS V2)

* **Môn học:** Khai phá Dữ liệu (Data Mining) — Học kỳ 1, Năm học 2026
* **Chủ đề IV:** Anomaly / Outlier Detection (Phát hiện dị biệt)
* **Đề tài số 13:** Credit Card Fraud Detection (Phát hiện gian lận thẻ tín dụng bằng Unsupervised Machine Learning)
* **Nhóm sinh viên thực hiện:**
  * **Nguyễn Văn Hiển Vinh** — Mã SV: **B23DCVT461** *(Trưởng nhóm | Data & Preprocessing Lead)*
  * **Nguyễn Minh Hoàng** — Mã SV: **B23DCVT173** *(Kỹ sư Mô hình hóa & Đánh giá | Modeling & Evaluation Lead)*
* **Thời gian thực hiện:** 14 Ngày (Sprint tốc lực 2 tuần)
* **Dataset:** Kaggle Credit Card Fraud Detection (284,807 giao dịch, 492 gian lận, 30 features, 1 nhãn Class)
* **Phiên bản tài liệu:** SRS V2.0 (Audited & Refined - Clean Typography)

---

## MỤC 1: TỔNG QUAN BÀI TOÁN & MỤC TIÊU CHIẾN LƯỢC

### 1.1. Bối cảnh nghiệp vụ & Ý nghĩa kinh tế
Trong hệ thống tài chính & ngân hàng hiện đại, hàng trăm triệu giao dịch thẻ tín dụng được xử lý mỗi ngày. Giao dịch gian lận (Fraud) chỉ chiếm một tỷ lệ cực kỳ nhỏ (trong tập dữ liệu Kaggle này chỉ chiếm **0.172%**, tương ứng 492 trên 284,807 giao dịch). Tuy nhiên, thiệt hại mà gian lận thẻ gây ra lại có tính chất dây chuyền và tổn thất nặng nề. Hai rủi ro kinh doanh cốt lõi mà mọi mô hình phát hiện gian lận phải giải quyết gồm:
* **Rủi ro bỏ sót gian lận (False Negative — Sai số Loại II):** Kẻ gian lận thực hiện giao dịch thành công mà hệ thống không cảnh báo. Ngân hàng và khách hàng chịu thiệt hại tài chính trực tiếp, uy tín bảo mật bị suy giảm nghiêm trọng.
* **Rủi ro báo động nhầm (False Positive — Sai số Loại I):** Hệ thống khóa thẻ hoặc từ chối giao dịch hợp lệ của khách hàng bình thường (ví dụ: khách hàng mua sắm hàng xa xỉ hoặc thanh toán khi đi du lịch nước ngoài). Hậu quả là làm gián đoạn trải nghiệm người dùng, phát sinh chi phí vận hành cho bộ phận Chăm sóc khách hàng (Customer Service) và nguy cơ mất khách hàng trung thành.

> **BẢN CHẤT HỌC THUẬT MÔN KHAI PHÁ DỮ LIỆU — VÌ SAO LÀ UNSUPERVISED ANOMALY DETECTION?**  
> Đề tài được xếp vào **Mục IV: Phát hiện dị biệt (Anomaly / Outlier Detection)**. Trong thực tế ngành tài chính, các thủ đoạn gian lận thẻ tín dụng liên tục biến đổi (Zero-day Fraud). Nếu tiếp cận thuần túy theo hướng Học có giám sát (Supervised Classification) như Random Forest hay XGBoost kết hợp với kỹ thuật tạo mẫu tổng hợp SMOTE:
> 1. Mô hình có giám sát chỉ nhận diện được những kiểu mẫu gian lận đã có nhãn trong quá khứ, hoàn toàn bất lực trước những thủ đoạn tấn công mới chưa từng ghi nhận.
> 2. Việc nội suy mẫu nhân tạo SMOTE trong không gian nhiều chiều (30 chiều) dễ tạo ra các điểm ngoại lai giả mạo (synthetic artifacts), làm méo mó cấu trúc phân phối thực tế.  
> 
> Tiếp cận theo **Học không giám sát (Unsupervised / Semi-supervised Learning)** cho phép mô hình học cấu trúc phân phối và mật độ của "các giao dịch bình thường" (Nominal Data). Mọi giao dịch xuất hiện với khoảng cách xa lạ, mật độ thưa thớt hoặc dễ bị cô lập đều tự động bị gắn cờ là dị biệt (Outlier) mà không cần phụ thuộc vào nhãn trước đó.

### 1.2. Mục tiêu dự án (SMART Objectives)
Dự án được thiết kế đáp ứng đầy đủ tiêu chí SMART, tuyệt đối không bịa số liệu thực nghiệm khi chưa chạy mô hình:
1. **Phân tích tính chất dữ liệu lệch cực đoan (Specific):** Khám phá toàn diện phân phối của 284,807 giao dịch, định lượng độ lệch lớp (99.828% Normal vs. 0.172% Fraud), phân tích tương quan và đặc tính của các biến ẩn danh PCA (V1–V28), Amount và Time.
2. **Xây dựng Pipeline tiền xử lý chống rò rỉ dữ liệu (Measurable):** Áp dụng `RobustScaler` cho biến `Amount` và `Time`, bảo đảm quy tắc **chỉ fit trên tập huấn luyện (Train set)** và transform trên tập kiểm định/kiểm thử (Validation/Test set).
3. **Cài đặt và so sánh thực nghiệm 3 thuật toán Anomaly Detection (Achievable):**
   * `Isolation Forest (iForest)`: Khám phá cây nhị phân cô lập nhanh.
   * `Local Outlier Factor (LOF)`: Đo mật độ cục bộ lân cận với cơ chế `novelty=True` và chiến lược lấy mẫu chống tràn bộ nhớ.
   * `One-Class SVM (OC-SVM)`: Phân tách dữ liệu chuẩn bằng siêu cầu margin với kernel RBF và chiến lược kiểm soát độ phức tạp.
4. **Tối ưu hóa ngưỡng quyết định (Threshold-moving) (Relevant):** Phân biệt rõ tham số `contamination` và kỹ thuật dịch chuyển ngưỡng quyết định trên Anomaly Score liên tục ở tập Validation; hướng tới tối ưu hóa Recall đối với lớp gian lận (mục tiêu ≥ 80%) kết hợp phân tích đánh đổi Precision – Recall và độ nhạy ma trận chi phí (Cost-Sensitive Analysis).
5. **Trực quan hóa dị biệt trong không gian giảm chiều (Time-bound & Visual):** Áp dụng PCA (chiếu toàn cục 30 chiều xuống 2D/3D) và t-SNE (chiếu phi tuyến trên mẫu đại diện) để hiển thị trực quan các điểm gian lận và điểm dị biệt do mô hình dự báo.
6. **Sản phẩm bàn giao hoàn chỉnh:** 01 Báo cáo kỹ thuật chi tiết (25–35 trang, cấu trúc chuẩn IEEE/APA), 01 Bộ Slide thuyết trình chuyên nghiệp (20–25 slides), và toàn bộ Source Code Jupyter Notebook có thể tái lập 100% kết quả (`random_state=42`).

### 1.3. Phạm vi & Giới hạn của dự án (Scope & Boundaries)
> **RANH GIỚI PHẠM VI BẮT BUỘC (SCOPE BOUNDARIES):**  
> Đây là bài tập lớn nghiên cứu thực nghiệm trong khuôn khổ môn học Khai phá Dữ liệu, **không phải dự án xây dựng hệ thống phần mềm thương mại**. Nhóm thống nhất nghiêm ngặt:
> * **Không mở rộng:** Không xây dựng Web Application, không dựng RESTful API backend, không làm Dashboard tương tác thời gian thực, không tích hợp cổng thanh toán.
> * **Không sử dụng thuật toán ngoài phạm vi:** Không mở rộng sang các thuật toán Supervised ngoài đề bài (không chạy Random Forest, XGBoost, LightGBM làm mô hình chính; không dùng SMOTE làm phương pháp giải quyết chính).
> * **Trọng tâm cốt lõi:** Quy trình KDD (Knowledge Discovery in Databases), phân tích dữ liệu bất cân bằng, bản chất toán học của 3 thuật toán Unsupervised Outlier Detection, tối ưu ngưỡng và phân tích đánh đổi nghiệp vụ.

---

## MỤC 2: ĐẶC TẢ YÊU CẦU DỮ LIỆU & CHIẾN LƯỢC TIỀN XỬ LÝ (SRS & PIPELINE)

### 2.1. Phân tích thuộc tính dữ liệu (Data Attributes Specification)
| Thuộc tính | Ý nghĩa nghiệp vụ | Kiểu dữ liệu & Phân phối | Phương án tiền xử lý & Cơ sở học thuật |
| :--- | :--- | :--- | :--- |
| `Time` | Số giây trôi qua giữa giao dịch hiện tại và giao dịch đầu tiên trong dataset (thu thập trong 48 giờ) | Continuous (Float)<br>Phạm vi: [0, 172,792] giây | **Giữ nguyên định dạng liên tục và chuẩn hóa bằng `RobustScaler`.**<br>*Lưu ý chuyên môn:* Tuyệt đối không tự ý chuyển đổi thành hàm chu kỳ 24h (sin/cos) vì dữ liệu không cung cấp mốc thời gian thực tế (không biết giao dịch đầu tiên bắt đầu vào lúc mấy giờ sáng hay tối). |
| `V1` – `V28` | 28 đặc trưng đã được ngân hàng ẩn danh hóa bằng phép biến đổi PCA để bảo vệ quyền riêng tư khách hàng | Continuous (Float)<br>Dạng chuẩn hóa trực giao | **Giữ nguyên thang đo gốc, không scale bổ sung.**<br>Vì các thành phần chính này đã được chuẩn hóa và trực giao hóa trong quá trình trích xuất PCA ban đầu của nhà cung cấp dữ liệu. |
| `Amount` | Giá trị bằng tiền của giao dịch thẻ tín dụng | Continuous (Float)<br>Min: 0.00 • Max: 25,691.16 • Median: 22.00 | **Bắt buộc chuẩn hóa bằng `RobustScaler`.**<br>Phân phối số tiền bị lệch phải cực kỳ nghiêm trọng (Right-skewed). Đa số giao dịch có giá trị nhỏ, nhưng tồn tại các ngoại lai cực lớn. `RobustScaler` bảo toàn cấu trúc dữ liệu mà không bị kéo lệch bởi cực trị. |
| `Class` | Nhãn xác thực giao dịch (Ground Truth): 0 = Bình thường (Normal), 1 = Gian lận (Fraud) | Binary (0 hoặc 1)<br>Tỷ lệ: 99.828% (0) vs. 0.172% (1) | **Chỉ sử dụng để đánh giá kiểm thử (Ground Truth Evaluation).**<br>Hoàn toàn loại bỏ biến này khỏi ma trận đặc trưng đầu vào trong suốt quá trình huấn luyện mô hình Unsupervised. |

### 2.2. Chiến lược chia dữ liệu chống rò rỉ (3-Way Zero-Leakage Data Split)
Bản thiết kế V1 trước đây mắc lỗi thiếu tập Validation, dẫn đến nguy cơ dùng tập Test để chọn ngưỡng (Test Set Leakage / Data Snooping). Trong bản V2, nhóm thiết kế quy trình phân chia dữ liệu 3 tập chặt chẽ:

> **Tổng tập dữ liệu:** 284,807 giao dịch = 284,315 Normal (Class 0) + 492 Fraud (Class 1)

| Tập dữ liệu | Thành phần & Kích thước mẫu | Tỷ lệ Fraud | Mục đích sử dụng & Ràng buộc rò rỉ (Leakage Constraints) |
| :--- | :--- | :---: | :--- |
| **Training Set**<br>*(Tập Huấn luyện)* | **199,020 giao dịch**<br>• 100% Class 0 (Normal)<br>• 0 giao dịch Class 1 | **0.000%** | • Dùng để học cấu trúc phân phối danh định của giao dịch bình thường.<br>• **Duy nhất tập này được dùng để fit các bộ scaler tiền xử lý.**<br>• Nhãn `Class` bị loại bỏ hoàn toàn trong quá trình fit. |
| **Validation Set**<br>*(Tập Kiểm định)* | **42,893 giao dịch**<br>• 42,647 Class 0 (Normal)<br>• 246 Class 1 (Fraud) | **~0.573%** | • Dùng để tinh chỉnh siêu tham số (n_estimators, k-neighbors, tham số nu, tham số gamma).<br>• **Dùng để thực hiện kỹ thuật Threshold-Moving trên Anomaly Score liên tục.**<br>• Nhãn `Class` chỉ được mở ra ở bước tính metric đánh giá. |
| **Test Set**<br>*(Tập Kiểm thử độc lập)* | **42,894 giao dịch**<br>• 42,648 Class 0 (Normal)<br>• 246 Class 1 (Fraud) | **~0.573%** | • **Tuyệt đối niêm phong trong suốt quá trình thử nghiệm và chọn ngưỡng.**<br>• Chỉ được nạp vào ở bước cuối cùng sau khi đã cố định mô hình và ngưỡng tối ưu &tau;*.<br>• Đảm bảo kết quả đánh giá phản ánh chính xác khả năng tổng quát hóa trên dữ liệu thực tế. |

> **CƠ SỞ KHOA HỌC CỦA QUY TRÌNH PHÂN CHIA:**  
> * **Vì sao Train Set chỉ chứa Normal?** Mô hình học không giám sát phát hiện dị biệt (One-class / Novelty Detection) xây dựng ranh giới chuẩn dựa trên hành vi hợp lệ. Nếu đưa gian lận vào tập train mà không có nhãn, mô hình sẽ coi hành vi gian lận là một phần của hành vi bình thường (bị nhiễu).  
> * **Vì sao Validation và Test Set phải chứa cả Normal và Fraud?** Để tính toán các độ đo phân lớp (Recall, Precision, AUPRC), bắt buộc phải có sự hiện diện của cả 2 lớp nhằm đo lường khả năng phân biệt dị biệt của mô hình. 492 mẫu Fraud được chia đều 50:50 (246/246) bằng phân tầng ngẫu nhiên với seed cố định (`random_state=42`).

### 2.3. Quy trình chuẩn hóa với RobustScaler & Giới hạn kỹ thuật
Kỹ thuật `RobustScaler` chuẩn hóa dữ liệu dựa trên số liệu thống kê trung vị và khoảng tứ phân vị:

> **x_scaled = (x − Median(x)) / IQR = (x − Q₂) / (Q₃ − Q₁)**

* **Lý do lựa chọn:** Trong khi `StandardScaler` sử dụng Giá trị trung bình (&mu;) và Độ lệch chuẩn (&sigma;) — hai đại lượng bị kéo lệch nghiêm trọng khi xuất hiện các giao dịch đột biến hàng chục nghìn USD; còn `MinMaxScaler` sẽ nén 99% giao dịch bình thường vào một khoảng siêu hẹp gần 0. `RobustScaler` sử dụng Trung vị (Q₂) và khoảng tứ phân vị IQR = Q₃ − Q₁, hoàn toàn không bị ảnh hưởng bởi các giá trị ngoại lai cực trị.
* **Giới hạn của phương pháp:** `RobustScaler` không giới hạn khoảng giá trị đầu ra trong đoạn cố định [0, 1] như MinMaxScaler. Các điểm ngoại lai cực trị vẫn sẽ mang giá trị độ lệch lớn, nhưng không làm méo mó phân phối của phần lớn dữ liệu trung tâm.
* **Quy tắc chống rò rỉ:**
  ```python
  scaler_amount = RobustScaler().fit(X_train[['Amount']])
  X_train['Amount_scaled'] = scaler_amount.transform(X_train[['Amount']])
  X_val['Amount_scaled'] = scaler_amount.transform(X_val[['Amount']])
  X_test['Amount_scaled'] = scaler_amount.transform(X_test[['Amount']])
  ```

---

## MỤC 3: PHƯƠNG PHÁP HỌC MÁY KHÔNG GIÁM SÁT & CHUẨN HÓA ANOMALY SCORE

### 3.1. Thuật toán 1: Isolation Forest (iForest)
Được đề xuất bởi Liu et al. (2008), Isolation Forest hoạt động dựa trên nguyên lý: *Các điểm dị biệt là số ít và có giá trị thuộc tính khác biệt, do đó chúng bị cô lập rất nhanh (cần ít lần cắt nhị phân ngẫu nhiên hơn) so với các điểm dữ liệu bình thường nằm trong vùng mật độ cao.*

> **s(x, n) = 2^(− E(h(x)) / c(n))**

Trong đó E(h(x)) là độ dài đường dẫn trung bình từ gốc tới nút lá của điểm x qua tập các cây nhị phân (iTrees), và c(n) là độ dài đường dẫn trung bình của một cây tìm kiếm nhị phân bất thành xây dựng từ n mẫu.

> **PHÂN BIỆT RÕ RÀNG: THAM SỐ CONTAMINATION VS. KỸ THUẬT THRESHOLD MOVING**  
> * **Tham số `contamination`:** Là một siêu tham số truyền vào mô hình khi khởi tạo `IsolationForest(contamination=...)`. Trong thư viện scikit-learn, tham số này chỉ đóng vai trò xác định tỷ lệ phần trăm các mẫu trong tập huấn luyện bị đánh nhãn là ngoại lai mặc định, từ đó tính ra giá trị dịch chuyển nội bộ `offset_` cho hàm `predict()`. Nhóm sẽ thử nghiệm các giá trị giả thuyết: `[0.001, 0.0017, 0.002, 0.005, 0.01]` như các thực nghiệm siêu tham số, **hoàn toàn không ngộ nhận rằng 0.0017 là tỷ lệ đúng mặc định**.
> * **Kỹ thuật Dịch chuyển ngưỡng (Threshold Moving):** Nhóm trích xuất trực tiếp giá trị Anomaly Score liên tục S(x) của mô hình trên tập **Validation**, sau đó quét một dải ngưỡng &tau; từ giá trị nhỏ nhất đến lớn nhất để dịch chuyển ranh giới phân loại theo đúng bài giảng Slide 78. Điều này cho phép điều khiển linh hoạt sự đánh đổi giữa Precision và Recall mà không bị trói buộc vào hàm `predict()` nhị phân cứng nhắc của scikit-learn.

### 3.2. Thuật toán 2: Local Outlier Factor (LOF) & Chiến lược Lấy mẫu
Được đề xuất bởi Breunig et al. (2000), LOF tính toán mức độ dị biệt cục bộ của một điểm dữ liệu dựa trên tỷ lệ mật độ tiếp cận cục bộ (Local Reachability Density — LRD) của điểm đó so với mật độ của k điểm lân cận gần nhất (k-NN). Điểm có mật độ thấp hơn đáng kể so với lân cận sẽ có điểm LOF > 1 và bị coi là ngoại lai.

* **Chế độ Novelty Detection:** Theo mặc định, LOF trong scikit-learn hoạt động ở chế độ Unsupervised Outlier Detection (không có hàm predict cho dữ liệu mới). Để áp dụng vào kịch bản dự đoán trên tập Validation và Test, nhóm bắt buộc phải cấu hình: `LocalOutlierFactor(novelty=True)`.
* **Thách thức độ phức tạp tính toán & Chiến lược Sampling:**  
  LOF có độ phức tạp tính toán khoảng cách là O(N² · d). Nếu ném toàn bộ N = 199,020 mẫu của tập Train vào, ma trận khoảng cách và cây k-d sẽ tiêu tốn hàng chục Gigabyte RAM và gây tràn bộ nhớ (Out of Memory — OOM).  
  **Giải pháp chuẩn khoa học:** Nhóm thực hiện lấy mẫu ngẫu nhiên có kiểm soát từ tập `Train_Normal` để tạo tập con chuẩn đại diện N_LOF = 25,000 – 30,000 mẫu (sử dụng `random_state=42`). Mô hình được fit trên tập con này, sau đó dùng hàm `score_samples()` để chấm điểm trên toàn bộ tập Validation và Test.
* **Siêu tham số thực nghiệm:** Số lân cận k (`n_neighbors`) được thử nghiệm tại các mức: `[10, 20, 30, 50]`.

### 3.3. Thuật toán 3: One-Class SVM & Kiểm soát Không gian Siêu cầu
Được đề xuất bởi Schölkopf et al. (2001), One-Class SVM ánh xạ dữ liệu đầu vào vào không gian đặc trưng nhiều chiều thông qua hàm nhân RBF (Radial Basis Function Kernel), sau đó xác định một siêu phẳng (hoặc siêu cầu trong không gian ban đầu) sao cho bao bọc được phần lớn dữ liệu bình thường và tách biệt với gốc tọa độ với khoảng cách lề (margin) lớn nhất.

* **Thách thức độ phức tạp & Chiến lược Sampling:**  
  Thuật toán giải bài toán quy hoạch toàn phương (Quadratic Programming) của SVM có độ phức tạp thời gian từ O(N²) đến O(N³). Với N ≈ 200,000, thời gian huấn luyện có thể kéo dài nhiều giờ.  
  **Giải pháp:** Huấn luyện OC-SVM trên tập con đại diện được lấy mẫu ngẫu nhiên từ `Train_Normal` với kích thước N_OCSVM = 15,000 – 20,000 mẫu. Điều này đảm bảo thời gian chạy trong khoảng 3–5 phút mà vẫn giữ được ranh giới bao đóng chuẩn xác.
* **Siêu tham số thực nghiệm:**
  * Tham số &nu; (`nu`): Cận trên của tỷ lệ ngoại lai và cận dưới của số lượng vector hỗ trợ, thử nghiệm tại: `[0.001, 0.005, 0.01, 0.05]`.
  * Tham số &gamma; (`gamma`): Độ rộng của kernel RBF, thử nghiệm tại: `['scale', 'auto', 0.01, 0.1]`.

### 3.4. Quy chuẩn Thống nhất Chiều & Thang đo Anomaly Score
Mỗi thuật toán trong scikit-learn định nghĩa đầu ra của hàm chấm điểm theo các quy ước khác nhau:
* Trong `IsolationForest`: `score_samples(X)` trả về giá trị đối dấu của điểm dị biệt (giá trị càng âm càng dị biệt).
* Trong `LocalOutlierFactor(novelty=True)`: `score_samples(X)` trả về giá trị đối dấu của điểm LOF (giá trị càng âm lớn càng dị biệt).
* Trong `OneClassSVM`: `decision_function(X)` trả về khoảng cách có dấu tới siêu phẳng (giá trị dương là bình thường, giá trị âm là ngoại lai).

> **QUY ƯỚC THỐNG NHẤT TOÀN DỰ ÁN: ANOMALY SCORE CÀNG CAO → MỨC ĐỘ DỊ BIỆT CÀNG LỚN**  
> Nhóm thực hiện phép đảo dấu chuẩn mực trên mã nguồn trước khi đánh giá:  
> * `S_iForest(x) = - model_if.score_samples(x)`
> * `S_LOF(x) = - model_lof.score_samples(x)`
> * `S_OCSVM(x) = - model_ocsvm.decision_function(x)`  
> 
> *Lưu ý:* Vì miền giá trị thô của 3 thuật toán là khác nhau (iForest nằm trong khoảng [0, 1], LOF là tỷ số mật độ, OC-SVM là khoảng cách margin), nhóm **không bao giờ so sánh trực tiếp các giá trị điểm thô**. Thay vào đó, việc so sánh được thực hiện trên:
> 1. Diện tích dưới đường cong Precision-Recall (**AUPRC**) — không phụ thuộc vào thang đo điểm.
> 2. Điểm chuẩn hóa Min-Max trên tập Validation khi cần hiển thị trực quan so sánh.
> 3. Các giá trị ma trận nhầm lẫn sau khi đã áp dụng ngưỡng cắt tối ưu &tau;* riêng biệt của từng mô hình.

---

## MỤC 4: HỆ THỐNG CHỈ SỐ ĐÁNH GIÁ (EVALUATION METRICS & COST ANALYSIS)

### 4.1. Phản biện Chỉ số Accuracy & Các độ đo Tiêu chuẩn
Như đã nhấn mạnh trong Slide 79 bài giảng, với một tập dữ liệu có 99.828% giao dịch bình thường, một bộ phân loại "ngây ngô" chỉ cần đoán toàn bộ là 0 (Normal) thì đã đạt **Accuracy = 99.83%**, nhưng thực tế đã bỏ sót 100% các vụ lừa đảo, mang lại giá trị bằng 0 cho nghiệp vụ. Vì vậy, nhóm loại bỏ hoàn toàn Accuracy khỏi danh sách chỉ số quyết định.

Các chỉ số khoa học bắt buộc sử dụng:
* **Recall (Độ nhạy / Tỷ lệ phát hiện gian lận):**

> **Recall = TP / (TP + FN)**

  Thước đo quan trọng số 1 đối với ngân hàng, phản ánh tỷ lệ các vụ lừa đảo thực tế được mô hình phát hiện thành công.

* **Precision (Độ chuẩn xác cảnh báo):**

> **Precision = TP / (TP + FP)**

  Phản ánh tỷ lệ cảnh báo của hệ thống thực sự là gian lận, giúp kiểm soát tải công việc của đội ngũ thẩm định.

* **F1-Score:** Trung bình điều hòa giữa Precision và Recall:

> **F₁-Score = 2 × (Precision × Recall) / (Precision + Recall)**

* **AUPRC (Area Under the Precision-Recall Curve):**  
  Tiêu chuẩn vàng cho bài toán dữ liệu mất cân bằng cực đoan. AUPRC được tính toán trực tiếp từ **dải điểm Anomaly Score liên tục** (thông qua hàm `precision_recall_curve` và `auc` của scikit-learn), không phụ thuộc vào một ngưỡng nhị phân cố định đơn lẻ.

### 4.2. Báo cáo Chi tiết Ma trận Nhầm lẫn (Confusion Matrix)
Mỗi mô hình sau khi chọn ngưỡng tối ưu &tau;* bắt buộc phải báo cáo đầy đủ 4 giá trị định lượng:

| Chỉ số | Tên gọi chuyên ngành | Ý nghĩa trong đề tài Gian lận thẻ tín dụng |
| :--- | :--- | :--- |
| **TP** (True Positive) | Dương tính thật | Giao dịch gian lận thực tế và mô hình đã phát hiện chính xác |
| **FP** (False Positive) | Dương tính giả (Báo động nhầm) | Giao dịch bình thường nhưng mô hình nghi ngờ gắn cờ gian lận |
| **TN** (True Negative) | Âm tính thật | Giao dịch bình thường và mô hình xác nhận chính xác là bình thường |
| **FN** (False Negative) | Âm tính giả (Bỏ lọt gian lận) | Giao dịch gian lận thực tế nhưng mô hình bỏ sót, coi là bình thường |

### 4.3. Mô hình Chi phí Kinh tế (Cost-Sensitive Matrix Analysis)
Để đánh giá tác động thực tế ngoài các chỉ số thống kê trừu tượng, nhóm thiết lập mô hình chi phí tổn thất tổng thể (Total Expected Cost):

> **Total Expected Cost = C_FN × FN + C_FP × FP**

* Quy chuẩn chi phí xác minh nhầm 1 vụ giao dịch bình thường (gọi điện, gửi OTP xác thực): C_FP = 1 (đơn vị chuẩn hóa).
* Chi phí khi bỏ sót 1 vụ gian lận (bồi thường tiền mất, tổn hại uy tín): C_FN lớn hơn rất nhiều so với C_FP.
* **Phân tích độ nhạy (Sensitivity Analysis):** Nhóm sẽ khảo sát sự thay đổi của tổng chi phí khi tỷ lệ đánh đổi chi phí C_FN / C_FP nhận các giá trị khác nhau: 10, 20, 50, và 100.
* *Lưu ý khoa học:* Nhóm ghi chú rõ ràng trong báo cáo đây là **mô hình giả định nghiên cứu (research scenario simulation)** nhằm minh họa tư duy tối ưu chi phí trong khai phá dữ liệu, không phải là số liệu kế toán thực tế của bất kỳ ngân hàng thương mại cụ thể nào.

### 4.4. Đánh giá Hiệu năng Tính toán (Computational Profiling)
* **Thời gian huấn luyện (Training Runtime):** Đo bằng giây (s) từ khi bắt đầu fit đến khi hoàn thành.
* **Độ trễ dự báo (Inference Latency):** Thời gian chấm điểm trung bình cho mỗi 1,000 giao dịch (milliseconds).
* **Bộ nhớ tiêu thụ (Peak RAM Footprint):** Đo lượng RAM tối đa sử dụng trong quá trình huấn luyện bằng thư viện `psutil` hoặc `tracemalloc`.

---

## MỤC 5: TRỰC QUAN HÓA TRONG KHÔNG GIAN GIẢM CHIỀU (PCA & t-SNE)

### 5.1. Phương pháp Giảm chiều & Quy trình Thực hiện
| Phương pháp | Bản chất toán học | Kỹ thuật áp dụng & Tập dữ liệu | Mục tiêu trực quan hóa |
| :--- | :--- | :--- | :--- |
| **PCA**<br>*(Principal Component Analysis)* | Phép biến đổi tuyến tính trực giao, tìm các hướng phương sai cực đại trong không gian 30 chiều | Chiếu toàn bộ tập Validation (hoặc Test) từ 30 chiều xuống 2 chiều và 3 chiều | Quan sát bức tranh toàn cục: Liệu các giao dịch gian lận có nằm rải rác ở rìa ngoài của khối dữ liệu chính hay không. |
| **t-SNE**<br>*(t-Distributed Stochastic Neighbor Embedding)* | Kỹ thuật học đa tạp phi tuyến tính (Non-linear manifold), bảo toàn cấu trúc lân cận cục bộ của dữ liệu | Áp dụng trên tập mẫu đại diện phân tầng: **5,000 mẫu Normal + toàn bộ 246 mẫu Fraud** (để kiểm soát độ phức tạp O(N²)) | Phát hiện các cụm mật độ phi tuyến cục bộ: Xem các điểm gian lận có xu hướng tập trung thành các cụm bất thường riêng biệt hay bị phân tán. |

### 5.2. Nguyên tắc Trực quan hóa & Cảnh báo Học thuật
Đối với mỗi phương pháp (PCA 2D, PCA 3D, t-SNE 2D), nhóm xuất biểu đồ theo **hai góc nhìn đối sánh song song**:
* **Góc nhìn 1 — Ground Truth:** Tô màu xanh cho nhãn 0 (Normal) và màu đỏ nổi bật cho nhãn 1 (Fraud).
* **Góc nhìn 2 — Model Prediction:** Tô màu xanh cho các điểm mô hình dự báo là Bình thường và màu cam/tím cho các điểm mô hình gắn cờ là Dị biệt (Outliers).

> **CẢNH BÁO HỌC THUẬT VỀ GIẢI THÍCH KẾT QUẢ PCA/t-SNE:**  
> Trong báo cáo và bài thuyết trình, nhóm khẳng định rõ: *Việc các điểm gian lận có thể tách biệt tương đối trên đồ thị 2D/3D chỉ đóng vai trò hỗ trợ trực quan hóa khám phá (Exploratory Visualization). Phép chiếu giảm chiều từ 30 chiều xuống 2D tất yếu làm mất mát thông tin phương sai và có thể tạo ra các hiệu ứng hình ảnh cục bộ giả (projection artifacts). Trực quan hóa đồ thị không bao giờ thay thế được các độ đo định lượng khoa học như AUPRC và Recall.*

---

## MỤC 6: KẾ HOẠCH TÁC CHIẾN 14 NGÀY CHI TIẾT (REVISED 14-DAY SPRINT PLAN)

> **Nguyên tắc phối hợp nhóm:**  
> Hai thành viên làm việc song song trên hai làn chuyên môn độc lập (Data/Analysis vs. Modeling/Evaluation). Hàng ngày tổ chức họp Sync ngắn 15 phút vào 21:00 để kiểm tra tiến độ, rà soát code và đồng bộ commit lên Git repository.

| Ngày | Mục tiêu trọng tâm | Nguyễn Văn Hiển Vinh (B23DCVT461)<br>*(Data & Preprocessing Lead)* | Nguyễn Minh Hoàng (B23DCVT173)<br>*(Modeling & Evaluation Lead)* | Sản phẩm & Tiêu chí nghiệm thu (DoD) |
| :---: | :--- | :--- | :--- | :--- |
| **Ngày 1** | Khởi tạo Dự án & Môi trường | - Thiết lập cấu trúc cây thư mục chuẩn: `data/`, `notebooks/`, `src/`, `reports/`, `models/`.<br>- Tải dataset gốc Kaggle về `data/raw/`.<br>- Viết file `requirements.txt` cố định version thư viện. | - Khởi tạo GitHub repo, tạo branch `main` và branch cá nhân, cấu hình `.gitignore` chuẩn.<br>- Nghiên cứu lý thuyết Outlier Analysis trong Chương 12 giáo trình Han & Kamber.<br>- Xây dựng khung mẫu Báo cáo đồ án Word. | **Deliverable:** Repo Git + Môi trường + Khung Báo cáo.<br>*Tiêu chí:* `pip install` trơn tru, load data không lỗi. |
| **Ngày 2** | EDA Cơ bản & Tỷ lệ mất cân bằng | - Viết Notebook `01_eda_overview.ipynb`.<br>- Thống kê mô tả: min, max, mean, std, percentiles của `Amount` và `Time`.<br>- Trực quan hóa độ lệch lớp nghiêm trọng (Bar & Pie chart 99.828% vs. 0.172%). | - Soạn Chương 1 Báo cáo: Bối cảnh nghiệp vụ phát hiện gian lận thẻ.<br>- Phân tích rủi ro kinh doanh & đánh đổi FP vs. FN.<br>- Lập luận cơ sở khoa học chọn Unsupervised thay vì Supervised + SMOTE. | **Deliverable:** Notebook EDA + Bản thảo Chương 1.<br>*Tiêu chí:* Biểu đồ rõ nét, số liệu khớp đúng 284,807 dòng. |
| **Ngày 3** | Khám phá Đặc trưng V1–V28 | - Mở rộng phân tích trong `01_eda_overview.ipynb`.<br>- Vẽ KDE Plots của 28 đặc trưng PCA tách biệt Normal vs. Fraud.<br>- Lập ma trận tương quan Correlation Matrix Heatmap với nhãn Class. | - Soạn Chương 2 Báo cáo: Đặc tả tập dữ liệu.<br>- Tổng hợp bảng thống kê chi tiết các biến.<br>- Đưa các biểu đồ KDE nổi bật (V4, V11, V12, V14) vào báo cáo kèm nhận xét chuyên môn. | **Deliverable:** Bộ biểu đồ KDE 28 biến + 80% Chương 2.<br>*Tiêu chí:* Chỉ rõ ít nhất 4 biến có độ phân tách lớn nhất. |
| **Ngày 4** | Pipeline Tiền xử lý & Chia 3 tập | - Viết Notebook `02_preprocessing_pipeline.ipynb`.<br>- Phân chia dữ liệu 3 tập: Train Normal 70%, Val 15% + 50% Fraud, Test 15% + 50% Fraud (`random_state=42`).<br>- Đóng gói `RobustScaler` cho Amount & Time: chỉ fit trên Train_Normal, transform sang Val và Test. | - Viết module đánh giá `src/metrics.py`: hàm tính Recall, Precision, F1, AUPRC từ Anomaly Score liên tục.<br>- Viết hàm vẽ đồ thị PR-Curve và Ma trận nhầm lẫn.<br>- Soạn mục 2.3 và 2.4 Báo cáo: cơ chế RobustScaler và 3-way split. | **Deliverable:** Pipeline tiền xử lý + Module `metrics.py`.<br>*Tiêu chí:* Assert 100% không có dữ liệu Fraud trong Train set. |
| **Ngày 5** | Mô hình 1: Isolation Forest | - Soạn mục 3.1 Báo cáo: Cơ sở toán học của iForest (cấu trúc iTree, công thức tính path length h(x), hệ số c(n)).<br>- Phân tích khả năng mở rộng (Scalability) của thuật toán. | - Viết Notebook `03_isolation_forest.ipynb`.<br>- Cài đặt `IsolationForest` từ scikit-learn, huấn luyện trên toàn bộ `Train_Normal`.<br>- Đo Training Runtime và Peak RAM.<br>- Trích xuất Anomaly Score trên Validation, tính AUPRC sơ bộ. | **Deliverable:** Notebook iForest + Bản thảo mục 3.1.<br>*Tiêu chí:* Fit xong trong < 1 phút, đo đủ runtime & RAM. |
| **Ngày 6** | Tối ưu iForest & Dịch chuyển ngưỡng | - Thử nghiệm tham số `contamination`: `[0.001, 0.0017, 0.002, 0.005, 0.01]`.<br>- Thực hiện kỹ thuật Threshold-Moving trên Anomaly Score liên tục ở Validation Set: quét dải ngưỡng &tau;, đo biến thiên TP, FP, FN, Recall, Precision. | - Thử nghiệm số cây `n_estimators`: `[50, 100, 150, 200]` và `max_samples`.<br>- Vẽ đường cong PR-Curve cho iForest trên tập Validation.<br>- Soạn phần phân tích thực nghiệm iForest trong Báo cáo. | **Deliverable:** Bảng tham số iForest + Đồ thị PR-Curve.<br>*Tiêu chí:* Phân biệt rành mạch contamination vs threshold-moving. |
| **Ngày 7** | Mô hình 2: Local Outlier Factor (LOF) | - Soạn mục 3.2 Báo cáo: Cơ sở toán học của LOF (k-distance, reachability distance, mật độ LRD).<br>- Phân tích rào cản độ phức tạp O(N² · d) và lập luận khoa học phương án sampling. | - Viết Notebook `04_local_outlier_factor.ipynb`.<br>- Lấy mẫu ngẫu nhiên đại diện từ `Train_Normal` (N_LOF = 25,000 mẫu).<br>- Cài đặt `LocalOutlierFactor(novelty=True)`, fit trên mẫu đại diện.<br>- Thử nghiệm `n_neighbors`: `[10, 20, 30, 50]`.<br>- Chấm điểm Anomaly Score trên Validation Set. | **Deliverable:** Notebook LOF không tràn RAM + Bản thảo mục 3.2.<br>*Tiêu chí:* Fit xong < 3 phút nhờ sampling, tính được Anomaly Score trên Val. |
| **Ngày 8** | Mô hình 3: One-Class SVM (OC-SVM) | - Soạn mục 3.3 Báo cáo: Cơ sở toán học OC-SVM (ánh xạ Hilbert qua RBF kernel, bài toán quy hoạch toàn phương, tham số &nu;, &gamma;). | - Viết Notebook `05_one_class_svm.ipynb`.<br>- Lấy mẫu ngẫu nhiên đại diện từ `Train_Normal` (N_OCSVM = 20,000 mẫu).<br>- Cài đặt `OneClassSVM(kernel='rbf')`.<br>- Thử nghiệm lưới tham số: &nu; trong khoảng [0.001, 0.005, 0.01, 0.05] và &gamma; trong khoảng ['scale', 'auto', 0.01].<br>- Chấm điểm trên Validation Set. | **Deliverable:** Notebook OC-SVM + Bản thảo mục 3.3.<br>*Tiêu chí:* Tìm được bộ &nu;, &gamma; tốt nhất trên Val, ghi đủ số liệu. |
| **Ngày 9** | So sánh Thực nghiệm trên Validation | - Chuẩn hóa thang đo Anomaly Score cả 3 mô hình theo quy chuẩn (Điểm càng cao càng dị biệt).<br>- Tổng hợp bảng so sánh tổng thể trên Validation: AUPRC, Recall, Precision, F1 (ngưỡng mặc định & tối ưu), Runtime, RAM. | - Vẽ đồ thị tổng hợp: PR-Curve chung của cả 3 mô hình trên cùng 1 đồ thị để so sánh trực quan AUPRC.<br>- Vẽ biểu đồ cột so sánh thời gian và tài nguyên phần cứng.<br>- Soạn Chương 4 Báo cáo: Đánh giá thực nghiệm chuyên sâu. | **Deliverable:** Bảng so sánh 3 mô hình + Đồ thị PR-Curve đối sánh.<br>*Tiêu chí:* Nhận xét khách quan, chỉ rõ ưu nhược điểm từng thuật toán. |
| **Ngày 10** | Tối ưu Ngưỡng & Ma trận Chi phí | - Viết Notebook `06_threshold_tuning_cost.ipynb`.<br>- Với mô hình tốt nhất (dự kiến iForest): Quét dải ngưỡng quyết định &tau; chi tiết.<br>- Lập Trade-off Matrix: Khảo sát FP khi cố định mục tiêu Recall tại 75%, 80%, 85%, 90%. | - Xây dựng mô hình Cost Matrix: Thiết lập hàm tính tổng chi phí với C_FN / C_FP trong khoảng [10, 20, 50, 100].<br>- Vẽ đồ thị hàm chi phí theo ngưỡng để chốt điểm cắt &tau;* tối ưu kinh tế.<br>- Soạn mục 4.3 Báo cáo: Phân tích độ nhạy chi phí kinh doanh. | **Deliverable:** Notebook Threshold & Cost + Cố định ngưỡng &tau;*.<br>*Tiêu chí:* Ngưỡng cắt có luận cứ kinh tế, không chọn cảm tính. |
| **Ngày 11** | Trực quan hóa Giảm chiều PCA & t-SNE | - Viết Notebook `07_dimensionality_reduction.ipynb`.<br>- Áp dụng `PCA` chiếu từ 30 chiều xuống 2D và 3D.<br>- Vẽ scatter plots đối sánh Ground Truth vs. Model Prediction.<br>- Đo tỷ lệ phương sai tích lũy giải thích bởi các PCs. | - Áp dụng `t-SNE` trên tập mẫu đại diện (5,000 Normal + 246 Fraud).<br>- Tinh chỉnh `perplexity=30, 50`, vẽ biểu đồ cụm phi tuyến.<br>- Xuất ảnh độ phân giải cao đưa vào Chương 4 Báo cáo kèm cảnh báo về giới hạn hình ảnh. | **Deliverable:** Bộ ảnh 2D/3D PCA & t-SNE + Hoàn thành Chương 4.<br>*Tiêu chí:* Đồ thị có chú giải màu sắc rõ ràng giữa nhãn thật và nhãn đoán. |
| **Ngày 12** | Đánh giá Độc lập Test Set & Refactor Code | - Nạp tập kiểm thử độc lập **Test Set** (lần đầu tiên mở niêm phong).<br>- Áp dụng các mô hình và ngưỡng &tau;* đã chốt ở Ngày 10 để tính toán chỉ số cuối cùng (Final Benchmark).<br>- Rà soát toàn bộ source code: Refactoring, bổ sung docstrings tiếng Việt/Anh, bảo đảm `Run All Cells` trơn tru. | - Ghi nhận số liệu kiểm thử chính thức vào Báo cáo (Bảng kết quả Test Set).<br>- Viết file `README.md` hoàn chỉnh: Hướng dẫn cài đặt, cấu trúc thư mục và cách tái lập kết quả.<br>- Soạn Chương 5 Báo cáo: Kết luận, bài học kinh nghiệm và hướng phát triển. | **Deliverable:** Source Code hoàn chỉnh + File README.md + Bảng kết quả Test Set.<br>*Tiêu chí:* Chạy lại toàn bộ từ đầu trên máy sạch ra đúng 100% số liệu. |
| **Ngày 13** | Hoàn thiện Báo cáo & Thiết kế Slide | - Đọc soát toàn bộ Báo cáo: Kiểm tra chính tả, định dạng font, canh lề, đánh số bảng biểu tự động.<br>- Rà soát tính chính xác của các công thức toán và trích dẫn IEEE/APA.<br>- Xuất bản file Báo cáo PDF và Word chính thức. | - Thiết kế Slide thuyết trình PowerPoint (20–25 slides).<br>- Bố cục slide mạch lạc: Đặt vấn đề $\to$ Thách thức dữ liệu $\to$ Pipeline tiền xử lý $\to$ 3 Thuật toán Unsupervised $\to$ So sánh thực nghiệm $\to$ Tối ưu ngưỡng & Chi phí $\to$ Trực quan hóa $\to$ Kết luận.<br>- Đưa các biểu đồ PR-Curve, PCA/t-SNE vào slide. | **Deliverable:** Báo cáo Kỹ thuật Final (Word + PDF) + Slide PPT.<br>*Tiêu chí:* Báo cáo đạt 25–35 trang chuẩn học thuật; Slide trực quan, cô đọng. |
| **Ngày 14** | Tổng duyệt & Luyện tập Phản biện | - Đảm nhiệm trình bày Phần 1 & Phần 4: Đặt vấn đề, Bối cảnh nghiệp vụ, Đặc tả dữ liệu, Pipeline tiền xử lý chống rò rỉ, Mô hình LOF và Kết luận.<br>- Chuẩn bị máy tính, môi trường demo code dự phòng. | - Đảm nhiệm trình bày Phần 2 & Phần 3: Lý thuyết và thực nghiệm Isolation Forest, One-Class SVM, So sánh đa chiều, Tối ưu ngưỡng theo chi phí và Trực quan hóa PCA/t-SNE.<br>- Cùng luyện tập bộ câu hỏi phản biện chuyên sâu (Q&A Cheat Sheet). | **Deliverable:** Buổi tổng duyệt khớp thời gian 12–15 phút + Tự tin 100%.<br>*Tiêu chí:* Hai thành viên trả lời lưu loát, bảo vệ vững chắc kết quả. |

---

## MỤC 7: MA TRẬN PHÂN CÔNG TRÁCH NHIỆM (REVISED RACI MATRIX)
*(R: Responsible - Thực hiện chính | A: Accountable - Phê duyệt cuối cùng | C: Consulted - Tham vấn chuyên môn | I: Informed - Theo dõi)*

| Hạng mục công việc / Gói thầu | Hiển Vinh (B23DCVT461) | Minh Hoàng (B23DCVT173) | Phân định trách nhiệm & Sản phẩm nghiệm thu |
| :--- | :---: | :---: | :--- |
| Khởi tạo Dự án, Quản lý Git & Môi trường | C | **A / R** | Hoàng khởi tạo repo GitHub, quản lý branch; Vinh tạo cấu trúc thư mục và requirements.txt. |
| Khám phá Dữ liệu (EDA) & Phân tích Độ lệch | **A / R** | C | Vinh phụ trách chính code thống kê và trực quan hóa phân phối trong notebook; Hoàng đóng góp ý kiến. |
| Pipeline Tiền xử lý & Chia dữ liệu 3 tập chống rò rỉ | **A / R** | C | Vinh thiết kế thuật toán 3-way split và đóng gói `RobustScaler`; Hoàng kiểm tra tính cô lập. |
| Xây dựng Module Đánh giá (Metrics & AUPRC Engine) | C | **A / R** | Hoàng lập trình hàm tính AUPRC từ continuous score, vẽ PR-Curve; Vinh kiểm thử chéo. |
| Cài đặt & Tinh chỉnh Mô hình 1: Isolation Forest | C | **A / R** | Hoàng thực thi notebook, chạy các mức n_estimators và contamination; Vinh hỗ trợ phân tích score. |
| Cài đặt & Tinh chỉnh Mô hình 2: Local Outlier Factor | **A / R** | C | Vinh thiết kế chiến lược lấy mẫu chống tràn RAM và chạy thử nghiệm các mức k-neighbors. |
| Cài đặt & Tinh chỉnh Mô hình 3: One-Class SVM | C | **A / R** | Hoàng tối ưu bài toán quy hoạch toàn phương trên mẫu con, quét lưới tham số nu và gamma. |
| Kỹ thuật Threshold-Moving & Mô hình Chi phí Kinh tế | **A / R** | **R** | Vinh quét dải ngưỡng tối ưu Recall; Hoàng tính toán ma trận chi phí độ nhạy; Cả hai thống nhất điểm cắt &tau;*. |
| Trực quan hóa Không gian Giảm chiều (PCA & t-SNE) | C | **A / R** | Hoàng phụ trách thuật toán giảm chiều và xuất biểu đồ song song; Vinh hỗ trợ biên tập hình ảnh. |
| Soạn thảo Báo cáo Kỹ thuật (Chương 1, 2, 5) | **A / R** | C | Vinh viết chính Chương 1 (Tổng quan), Chương 2 (Dữ liệu & Tiền xử lý), Chương 5 (Kết luận). |
| Soạn thảo Báo cáo Kỹ thuật (Chương 3, 4) | C | **A / R** | Hoàng viết chính Chương 3 (Phương pháp toán học 3 mô hình) và Chương 4 (Kết quả thực nghiệm). |
| Thiết kế Slide Thuyết trình PowerPoint | C | **A / R** | Hoàng làm slide chính (layout, đồ thị); Vinh kiểm duyệt nội dung và hỗ trợ chỉnh sửa. |
| Tổng duyệt Báo cáo & Phản biện Q&A | **A / R** | **A / R** | Cả hai cùng phối hợp 50/50, phân công chia ca thuyết trình và tập dượt phản biện. |

---

## MỤC 8: BỘ CÂU HỎI PHẢN BIỆN CHUYÊN SÂU DỰ PHÒNG (DEFENSE Q&A CHEAT SHEET V2)

1. **"Tại sao nhóm không dùng các mô hình phân loại có giám sát mạnh mẽ như Random Forest hay XGBoost kết hợp với kỹ thuật SMOTE để giải quyết mất cân bằng?"**  
   *Trả lời chuẩn mực:*  
   "Kính thưa Thầy/Cô, hướng tiếp cận có giám sát kết hợp SMOTE có hai hạn chế chí mạng trong bài toán gian lận tài chính thực tế:  
   1) *Bản chất gian lận biến đổi liên tục (Zero-day Fraud):* Mô hình Supervised chỉ học được các quy luật gian lận đã có nhãn trong quá khứ. Khi tội phạm thay đổi thủ đoạn mới, mô hình sẽ hoàn toàn mất khả năng nhận diện. Hướng tiếp cận Học không giám sát (Unsupervised Anomaly Detection) học cấu trúc của các giao dịch bình thường; bất kỳ hành vi nào đi chệch khỏi quy chuẩn này đều bị phát hiện ngay lập tức.  
   2) *Vấn đề nội suy trong không gian 30 chiều:* SMOTE tổng hợp mẫu bằng cách nội suy tuyến tính giữa các lân cận. Trong không gian chiều cao với độ mất cân bằng 0.17%, việc nội suy dễ tạo ra các điểm dữ liệu giả mạo nằm ở những vùng không gian không có thực tế, làm sai lệch ranh giới quyết định. Do đó, Unsupervised Anomaly Detection là hướng đi chuẩn mực và bền vững nhất theo đúng tinh thần của môn học."

2. **"Tại sao nhóm chọn RobustScaler cho thuộc tính Amount và Time mà không sử dụng StandardScaler hay MinMaxScaler?"**  
   *Trả lời chuẩn mực:*  
   "Kính thưa Thầy/Cô, đặc trưng `Amount` có phân phối lệch phải cực kỳ nghiêm trọng: trung vị chỉ là 22.00 USD, nhưng giá trị lớn nhất lên tới 25,691.16 USD.  
   * Nếu dùng `StandardScaler`: Giá trị trung bình (&mu;) và độ lệch chuẩn (&sigma;) bị kéo lệch rất mạnh bởi các giao dịch ngoại lai lớn này, khiến đại đa số các giao dịch bình thường bị co cụm lại gần 0.  
   * Nếu dùng `MinMaxScaler`: 99% dữ liệu sẽ bị nén nghẽn vào một dải hẹp [0, 0.01], làm triệt tiêu sự khác biệt về phương sai của các giao dịch thông thường.  
   * `RobustScaler` sử dụng Trung vị (Q₂) và Khoảng tứ phân vị (IQR = Q₃ − Q₁). Vì hai đại lượng này dựa trên thứ tự xếp hạng (rank-based), chúng hoàn toàn trơ (robust) trước các giá trị cực trị, bảo toàn nguyên vẹn hình dáng phân phối trung tâm của dữ liệu."

3. **"Tại sao thuật toán Isolation Forest có thể huấn luyện trên toàn bộ dữ liệu, trong khi LOF và One-Class SVM lại bắt buộc phải lấy mẫu đại diện (Sampling)?"**  
   *Trả lời chuẩn mực:*  
   "Kính thưa Thầy/Cô, điều này bắt nguồn từ sự khác biệt cơ bản về độ phức tạp tính toán giữa các thuật toán:  
   * **Isolation Forest:** Sử dụng kỹ thuật ngẫu nhiên hóa (random partitioning) trên cây nhị phân với kích thước mẫu con cố định cho mỗi cây là &psi; = 256. Độ phức tạp huấn luyện chỉ là tuyến tính O(t · &psi; log &psi;) với t là số lượng cây, cho phép mô hình mở rộng cực tốt (Scalability) trên hàng trăm nghìn dòng dữ liệu.  
   * **Local Outlier Factor (LOF):** Phải tính toán ma trận khoảng cách giữa các cặp điểm để tìm k lân cận gần nhất, độ phức tạp là O(N² · d). Với N ≈ 200,000, số phép tính khoảng cách lên tới hàng chục tỷ phép tính, vượt quá dung lượng RAM thông thường.  
   * **One-Class SVM:** Giải bài toán quy hoạch toàn phương (Quadratic Programming) với kernel RBF, độ phức tạp thời gian từ O(N²) đến O(N³) và bộ nhớ ma trận Gram xấp xỉ O(N²).  
   Vì vậy, việc lấy mẫu ngẫu nhiên có kiểm soát từ tập `Train_Normal` là giải pháp bắt buộc và hoàn toàn chuẩn mực trong kỹ nghệ dữ liệu nhằm cân đối giữa độ chính xác và tài nguyên tính toán."

4. **"Sự khác biệt cốt lõi giữa tham số Contamination và kỹ thuật Dịch chuyển ngưỡng (Threshold Moving) là gì?"**  
   *Trả lời chuẩn mực:*  
   "Kính thưa Thầy/Cô:  
   * **Contamination** là một tham số truyền vào mô hình khi khởi tạo (chỉ có trong một số thư viện như scikit-learn). Nó đại diện cho tỷ lệ phần trăm dữ liệu nhiễu mà chúng ta dự đoán trước trong tập huấn luyện, từ đó mô hình tự động chọn ra một điểm cắt mặc định (offset_) cho hàm `predict()`. Việc gán cứng contamination = 0.0017 là một giả định chủ quan và cứng nhắc.  
   * **Threshold Moving** là kỹ thuật hậu xử lý (Post-processing) độc lập. Thay vì nhận nhãn nhị phân đóng băng {-1, 1}, nhóm trích xuất điểm dị biệt liên tục (Continuous Anomaly Score) của các giao dịch trên tập Validation. Sau đó, nhóm quét dải ngưỡng &tau; để chủ động chọn điểm cắt phân loại sao cho đáp ứng chính xác mục tiêu nghiệp vụ: ví dụ ưu tiên đạt Recall ≥ 80% để giảm thiểu thiệt hại tài chính do bỏ sót gian lận, đúng theo nguyên lý được trình bày trong Slide 78 của môn học."

5. **"Tại sao nhóm tuyệt đối không được sử dụng tập Test Set để tinh chỉnh siêu tham số hoặc tìm ngưỡng quyết định?"**  
   *Trả lời chuẩn mực:*  
   "Kính thưa Thầy/Cô, việc dùng tập Test Set để chọn ngưỡng hoặc tham số sẽ dẫn đến hiện tượng **Rò rỉ dữ liệu kiểm thử (Data Snooping / Test Set Leakage)**. Khi đó, ngưỡng được chọn đã bị tối ưu hóa quá mức (overfitted) cho riêng tập Test đó. Kết quả đánh giá trên tập Test sẽ bị 'ảo tưởng' (quá lạc quan) và không còn phản ánh trung thực khả năng phát hiện gian lận của hệ thống khi triển khai vào môi trường thực tế với các luồng giao dịch hoàn toàn mới trong tương lai. Nhóm tuân thủ nghiêm ngặt nguyên tắc: Mọi quyết định về mô hình, tham số và ngưỡng tối ưu &tau;* đều phải được cố định dứt điểm trên tập Validation; tập Test chỉ được nạp vào một lần duy nhất ở bước nghiệm thu cuối cùng."

---

## MỤC 9: TIÊU CHÍ NGHIỆM THU DỰ ÁN (DEFINITION OF DONE — DoD)
| Cấu phần | Tiêu chuẩn Nghiệm thu Kỹ thuật | Phương pháp Xác minh |
| :--- | :--- | :--- |
| **Mã nguồn (Source Code)** | • Toàn bộ 7 Jupyter Notebooks chạy `Run All Cells` liên tục không phát sinh bất kỳ lỗi nào.<br>• Khởi tạo seed cố định `random_state=42` ở mọi bước ngẫu nhiên để đảm bảo tính tái lập 100%.<br>• Code có cấu trúc rõ ràng, chú thích đầy đủ bằng tiếng Việt/Anh.<br>• Có file `README.md` hướng dẫn chạy chi tiết. | Thành viên kiểm tra chéo bằng cách clone repo về máy thứ hai và chạy lại toàn bộ quy trình từ terminal sạch. |
| **Báo cáo Kỹ thuật (Technical Report)** | • Độ dài từ 25 đến 35 trang chuẩn A4.<br>• Cấu trúc 5 chương hoàn chỉnh: Mở đầu $\to$ Dữ liệu & Tiền xử lý $\to$ Cơ sở Toán học & Thuật toán $\to$ Kết quả Thực nghiệm $\to$ Kết luận.<br>• Đầy đủ bảng số liệu đối sánh trung thực, hình ảnh biểu đồ rõ nét, công thức toán học soạn bằng equation editor.<br>• Trích dẫn tài liệu tham khảo chuẩn IEEE/APA. | Kiểm tra định dạng, kiểm tra tính nhất quán của số liệu giữa báo cáo và kết quả log từ notebook. |
| **Slide Thuyết trình (Presentation Deck)** | • Số lượng từ 20 đến 25 slides PowerPoint.<br>• Thiết kế chuyên nghiệp, màu sắc hài hòa, ưu tiên biểu đồ trực quan, sơ đồ kiến trúc và bảng số liệu.<br>• Trình bày rõ ràng 3 trụ cột: Dữ liệu $\to$ 3 Thuật toán đối sánh $\to$ Giá trị nghiệp vụ & Phân tích chi phí. | Thuyết trình thử nội bộ trong thời gian 12–15 phút, kiểm tra hiệu ứng chuyển trang và độ tương phản máy chiếu. |
