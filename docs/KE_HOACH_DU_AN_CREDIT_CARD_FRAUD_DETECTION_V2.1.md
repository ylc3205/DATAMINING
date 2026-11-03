# KẾ HOẠCH DỰ ÁN & ĐẶC TẢ YÊU CẦU KỸ THUẬT (SRS V2.1)

* **Môn học:** Khai phá Dữ liệu (Data Mining) — Học kỳ 1, Năm học 2026
* **Chủ đề IV:** Anomaly / Outlier Detection (Phát hiện dị biệt)
* **Đề tài số 13:** Credit Card Fraud Detection (Phát hiện gian lận thẻ tín dụng bằng Unsupervised Machine Learning)
* **Nhóm sinh viên thực hiện:**
  * **Nguyễn Văn Hiển Vinh** — Mã SV: **B23DCVT461** *(Trưởng nhóm | Data & Preprocessing Lead)*
  * **Nguyễn Minh Hoàng** — Mã SV: **B23DCVT173** *(Kỹ sư Mô hình hóa & Đánh giá | Modeling & Evaluation Lead)*
* **Thời gian thực hiện:** 14 Ngày (Sprint tốc lực 2 tuần)
* **Dataset:** Kaggle Credit Card Fraud Detection (284,807 giao dịch, 492 gian lận, 30 features, 1 nhãn Class)
* **Phiên bản tài liệu:** SRS V2.1 (Audited, Refined & Re-synchronized — Production Distribution)

---

## MỤC 1: TỔNG QUAN BÀI TOÁN & MỤC TIÊU CHIẾN LƯỢC

### 1.1. Bối cảnh nghiệp vụ & Ý nghĩa kinh tế
Trong hệ thống tài chính & ngân hàng số hiện đại, hàng triệu giao dịch thẻ tín dụng được xử lý mỗi ngày. Giao dịch gian lận (Fraud) chỉ chiếm một tỷ lệ cực kỳ nhỏ (trong tập dữ liệu Kaggle này chỉ chiếm **0.172%**, tương ứng 492 trên 284,807 giao dịch). Tuy nhiên, thiệt hại mà gian lận gây ra lại có tính chất dây chuyền và tổn thất nặng nề. Hai rủi ro kinh doanh cốt lõi mà mọi mô hình phát hiện gian lận phải giải quyết gồm:
* **Rủi ro bỏ sót gian lận (False Negative — Sai số Loại II):** Kẻ gian lận thực hiện giao dịch thành công mà hệ thống không cảnh báo. Ngân hàng và khách hàng chịu thiệt hại tài chính trực tiếp, uy tín bảo mật bị suy giảm nghiêm trọng.
* **Rủi ro báo động nhầm (False Positive — Sai số Loại I):** Hệ thống khóa thẻ hoặc từ chối giao dịch hợp lệ của khách hàng bình thường (ví dụ: khách hàng mua sắm hàng xa xỉ hoặc thanh toán khi đi du lịch nước ngoài). Hậu quả là làm gián đoạn trải nghiệm người dùng, phát sinh chi phí vận hành cho bộ phận Chăm sóc khách hàng (Customer Service) và nguy cơ mất khách hàng trung thành.

> **LẬP LUẬN KHOA HỌC: VÌ SAO CHỌN UNSUPERVISED ANOMALY DETECTION THAY VÌ SUPERVISED + SMOTE?**  
> Đề tài được xếp vào **Topic IV: Anomaly / Outlier Detection (Phát hiện dị biệt)** của học phần Khai phá Dữ liệu. Trong thực tế ngành tài chính, các thủ đoạn gian lận thẻ tín dụng liên tục biến đổi (Zero-day Fraud). Nếu tiếp cận thuần túy theo hướng Học có giám sát (Supervised Classification) như Random Forest hay XGBoost kết hợp với kỹ thuật tạo mẫu tổng hợp SMOTE:  
> 1. **Giới hạn của Supervised:** Mô hình phân loại có giám sát tối ưu hóa ranh giới dựa trên các nhãn gian lận trong quá khứ, do đó gặp suy giảm hiệu năng nghiêm trọng và khó tổng quát hóa trước các thủ đoạn gian lận mới mang hình thái phân phối khác biệt (concept drift / zero-day anomalies).  
> 2. **Rủi ro của SMOTE:** Việc nội suy mẫu nhân tạo SMOTE trong không gian 30 chiều với độ mất cân bằng cực đoan (0.17%) dễ tạo ra các điểm ngoại lai giả mạo (synthetic artifacts) nằm ở những vùng không gian không có thực tế, làm méo mó cấu trúc dữ liệu.  
> 3. **Bản chất Unsupervised:** Tiếp cận theo Học không giám sát / Phát hiện điểm mới (Unsupervised / Novelty Detection) cho phép mô hình học cấu trúc phân phối và mật độ của các giao dịch bình thường (Nominal Data). Mọi hành vi xuất hiện với khoảng cách xa lạ, mật độ thưa thớt hoặc dễ bị cô lập đều được cảnh báo là dị biệt (Outlier) mà không phụ thuộc vào nhãn trước đó.  
> 4. **Nhận thức học thuật:** Dị biệt (Anomaly) không đồng nhất hoàn toàn với Gian lận (Fraud). Một giao dịch bất thường có thể chỉ là hành vi tiêu dùng đột biến hợp lệ (False Positive), trong khi kẻ gian có thể cố tình thực hiện giao dịch vi mô ẩn mình gần tâm phân phối bình thường. Do đó, việc tinh chỉnh ngưỡng quyết định (Threshold Moving) và phân tích ma trận đánh đổi chi phí (Cost Matrix) là yêu cầu bắt buộc của bài toán.

### 1.2. Mục tiêu dự án (SMART Objectives)
Dự án được thiết kế đáp ứng đầy đủ tiêu chí SMART, tuyệt đối không tạo số liệu thực nghiệm giả khi chưa chạy mô hình:
1. **Phân tích tính chất dữ liệu lệch cực đoan (Specific):** Khám phá toàn diện phân phối của 284,807 giao dịch, định lượng độ lệch lớp (99.828% Normal vs. 0.172% Fraud), phân tích tương quan và đặc tính của các biến ẩn danh PCA (V1–V28), Amount và Time.
2. **Xây dựng Pipeline tiền xử lý chống rò rỉ dữ liệu (Measurable):** Áp dụng `RobustScaler` cho biến `Amount` và `Time`, bảo đảm quy tắc **chỉ fit trên tập huấn luyện sạch (Train_Normal)** và transform trên tập kiểm định/kiểm thử (Validation/Test set).
3. **Cài đặt và so sánh thực nghiệm 3 thuật toán Anomaly Detection (Achievable):**
   * `Isolation Forest (iForest)`: Khám phá cây nhị phân cô lập nhanh với quy mô toàn vẹn.
   * `Local Outlier Factor (LOF)`: Đo mật độ cục bộ lân cận với cơ chế `novelty=True` và chiến lược lấy mẫu chống tràn bộ nhớ ($N = 25,000$).
   * `One-Class SVM (OC-SVM)`: Phân tách dữ liệu chuẩn bằng siêu phẳng/siêu cầu margin cực đại với kernel RBF và chiến lược lấy mẫu con ($N = 20,000$).
4. **Tối ưu hóa ngưỡng quyết định (Threshold-moving) (Relevant):** Phân biệt rõ tham số `contamination` và kỹ thuật dịch chuyển ngưỡng quyết định trên Anomaly Score liên tục ở tập Validation; hướng tới tối ưu hóa Recall đối với lớp gian lận (mục tiêu $\ge 80\%$) kết hợp phân tích đánh đổi Precision – Recall và độ nhạy ma trận chi phí (Cost-Sensitive Analysis).
5. **Trực quan hóa dị biệt trong không gian giảm chiều (Visual & Time-bound):** Áp dụng PCA (chiếu toàn bộ tập Validation từ 30 chiều xuống 2D/3D) và t-SNE (chiếu phi tuyến trên mẫu đại diện phân tầng Validation gồm 5,000 Normal + 74 Fraud) để hiển thị trực quan các điểm gian lận và điểm dị biệt do mô hình dự báo trước khi mở niêm phong Test Set.
6. **Sản phẩm bàn giao hoàn chỉnh:** 01 Báo cáo kỹ thuật chi tiết (25–35 trang chuẩn IEEE/APA), 01 Bộ Slide thuyết trình chuyên nghiệp (20–25 slides), và toàn bộ Source Code Jupyter Notebook có thể tái lập 100% kết quả (`random_state=42`).

### 1.3. Phạm vi & Giới hạn của dự án (Scope & Boundaries)
> **RANH GIỚI PHẠM VI BẮT BUỘC (SCOPE BOUNDARIES):**  
> Đây là bài tập lớn nghiên cứu thực nghiệm trong khuôn khổ môn học Khai phá Dữ liệu, **không phải dự án xây dựng hệ thống phần mềm thương mại**. Nhóm thống nhất nghiêm ngặt:  
> * **Không mở rộng:** Không xây dựng Web Application, không dựng RESTful API backend, không làm Dashboard tương tác thời gian thực, không tích hợp cổng thanh toán.  
> * **Không sử dụng thuật toán ngoài phạm vi:** Không mở rộng sang các thuật toán Supervised ngoài đề bài (không chạy Random Forest, XGBoost, LightGBM làm mô hình chính; không dùng SMOTE làm phương pháp giải quyết chính).  
> * **Trọng tâm cốt lõi:** Quy trình KDD (Knowledge Discovery in Databases), phân tích dữ liệu bất cân bằng, bản chất toán học của 3 thuật toán Unsupervised Outlier Detection, tối ưu ngưỡng và phân tích đánh đổi nghiệp vụ.

---

## MỤC 2: ĐẶC TẢ YÊU CẦU DỮ LIỆU & CHIẾN LƯỢC TIỀN XỬ LÝ (SRS & PIPELINE)

### 2.1. Phân tích thuộc tính dữ liệu (Data Attributes Specification)
| Thuộc tính | Số lượng | Kiểu dữ liệu | Phân phối & Đặc tính thống kê | Chiến lược xử lý kỹ thuật |
| :--- | :---: | :--- | :--- | :--- |
| `Time` | 1 biến | Continuous (Float)<br>Khoảng: [0, 172,792] giây | Thời gian tính bằng giây trôi qua giữa giao dịch hiện tại và giao dịch đầu tiên (thu thập trong 48 giờ). Phân phối thể hiện hai chu kỳ sóng ứng với ngày và đêm. | **Giữ nguyên định dạng liên tục và chuẩn hóa bằng `RobustScaler` fit trên Train_Normal.** Không biến đổi sin/cos chu kỳ 24h vì dataset không cung cấp mốc giờ thực tế. |
| `V1` – `V28` | 28 biến | Continuous (Float)<br>Trực giao PCA | Kết quả của phép biến đổi PCA của tổ chức tài chính nhằm bảo vệ quyền riêng tư. Các biến không có đa cộng tuyến và đã được chuẩn hóa trực giao. | **Giữ nguyên thang đo gốc, không scale bổ sung** để tránh làm méo mó cấu trúc trực giao sẵn có của PCA. |
| `Amount` | 1 biến | Continuous (Float)<br>Min: 0.00 • Max: 25,691.16 • Median: 22.00 | Giá trị tiền giao dịch. Phân phối lệch phải cực kỳ nghiêm trọng: trung vị chỉ 22.00 USD, 75% dưới 77.05 USD nhưng tồn tại các ngoại lai lên tới 25,691.16 USD. | **Bắt buộc chuẩn hóa bằng `RobustScaler` fit trên Train_Normal.** Sử dụng Trung vị và Khoảng tứ phân vị IQR để triệt tiêu ảnh hưởng của giá trị ngoại lai cực đại. |
| `Class` | 1 biến | Binary (0 hoặc 1)<br>0: 99.828% (Normal)<br>1: 0.172% (Fraud) | Biến mục tiêu: 0 là giao dịch hợp lệ (284,315 dòng), 1 là giao dịch gian lận (492 dòng). Độ mất cân bằng cực đại ~578:1. | **Chỉ sử dụng để phân tầng chia 3 tập (Stratified Split 70/15/15) và đánh giá kiểm thử (Ground Truth Evaluation).** Hoàn toàn loại bỏ khỏi ma trận đặc trưng $X$ trong huấn luyện. 344 mẫu Fraud rơi vào phân vùng Train bị loại bỏ khỏi quá trình fit mô hình Unsupervised. |

### 2.2. Chiến lược phân chia dữ liệu chống rò rỉ (Stratified 3-Way Zero-Leakage Data Split)
Bản thiết kế V2 trước đây mắc lỗi dồn toàn bộ 492 mẫu Fraud vào Validation và Test, khiến tỷ lệ Fraud trong Val/Test bị đẩy lên ~0.573% (gấp hơn 3.3 lần thực tế ~0.172%), gây sai lệch lớn cho đường cong PR và chỉ số Precision. Trong bản V2.1, nhóm thiết lập quy trình **phân chia có phân tầng (Stratified Splitting)** trên toàn bộ 284,807 giao dịch với tỷ lệ 70% Train / 15% Validation / 15% Test (`random_state=42`):

> **Tổng tập dữ liệu:** 284,807 giao dịch = 284,315 Normal (Class 0) + 492 Fraud (Class 1)

| Tập dữ liệu | Thành phần & Kích thước mẫu | Tỷ lệ Fraud | Mục đích sử dụng & Ràng buộc chống rò rỉ (Zero-Leakage) |
| :--- | :--- | :---: | :--- |
| **Training Partition (70%)**<br>$\to$ **`Train_Normal`** | **199,364 giao dịch tổng phân vùng:**<br>• 199,020 Normal (Class 0)<br>• 344 Fraud (Class 1) | **0.1725% (phân vùng)**<br>$\to$ **0.000% (fit mô hình)** | • **344 mẫu Fraud trong phân vùng Train bị loại trừ (lọc bỏ) khỏi quá trình huấn luyện**, tuyệt đối KHÔNG đưa vào fit mô hình và KHÔNG tự ý chuyển sang Val/Test.<br>• **Duy nhất tập `Train_Normal` (199,020 Normal) được dùng để fit Isolation Forest, LOF, One-Class SVM và fit `RobustScaler`.**<br>• Nhãn `Class` bị loại bỏ hoàn toàn trong quá trình fit. |
| **Validation Set (15%)**<br>*(Tập Kiểm định)* | **42,721 giao dịch:**<br>• 42,647 Normal (Class 0)<br>• 74 Fraud (Class 1) | **0.1732% (~0.173%)** | • Dùng để tinh chỉnh siêu tham số mô hình (`n_estimators`, `max_samples`, `n_neighbors`, $\nu$, $\gamma$).<br>• **Dùng để thực hiện kỹ thuật Threshold Moving trên Anomaly Score liên tục và tối ưu hóa ma trận chi phí để chốt ngưỡng $\tau^*$.**<br>• Trực quan hóa PCA (toàn bộ 42,721 mẫu) và t-SNE (5,000 Normal + 74 Fraud).<br>• Nhãn `Class` chỉ mở ở bước tính metric đánh giá. |
| **Test Set (15%)**<br>*(Tập Kiểm thử độc lập)* | **42,722 giao dịch:**<br>• 42,648 Normal (Class 0)<br>• 74 Fraud (Class 1) | **0.1732% (~0.173%)** | • **Tuyệt đối niêm phong trong suốt quá trình thử nghiệm, chọn siêu tham số và chọn ngưỡng.**<br>• Chỉ mở niêm phong 01 lần duy nhất ở Ngày 12 sau khi đã đóng băng hoàn toàn mô hình và ngưỡng $\tau^*$.<br>• Áp dụng scaler và ngưỡng đã fit từ trước; không fit lại Min-Max hay scaler riêng trên Test.<br>• Đảm bảo kết quả đánh giá phản ánh chính xác năng lực tổng quát hóa trên phân phối thực tế. |

> **CƠ SỞ KHOA HỌC & PHÂN TÍCH ĐÁNH ĐỔI (TRADE-OFF ANALYSIS):**  
> * **Tính toán số lượng mẫu chính xác:**  
>   - Tổng số giao dịch: $284,807 = 199,364\ (\text{Train}) + 42,721\ (\text{Val}) + 42,722\ (\text{Test})$.  
>   - Tổng số Normal: $284,315 = 199,020\ (\text{Train}) + 42,647\ (\text{Val}) + 42,648\ (\text{Test})$.  
>   - Tổng số Fraud: $492 = 344\ (\text{Train}) + 74\ (\text{Val}) + 74\ (\text{Test})$.  
>   - Tỷ lệ Fraud trong Validation ($74 / 42,721 = 0.1732\%$) và Test ($74 / 42,722 = 0.1732\%$) bảo toàn chính xác tỷ lệ gốc $0.1727\% \approx 0.173\%$ của dataset.  
> * **Vì sao Train Set chỉ dùng Normal để fit mô hình?** Trong bài toán Novelty Detection, mô hình cần học cấu trúc ranh giới danh định của hành vi hợp lệ. 344 mẫu Fraud rơi vào phân vùng Train bị loại bỏ hoàn toàn khỏi tập huấn luyện để tránh làm ô nhiễm ranh giới chuẩn. Nhóm tuyệt đối không tự ý chuyển 344 mẫu này sang Val/Test vì điều đó sẽ làm sai lệch phân phối kiểm định.  
> * **Phân tích đánh đổi:** So với phương án dồn toàn bộ 492 Fraud vào Val/Test (mỗi tập 246 mẫu, tỷ lệ 0.573%), phương án phân tầng chuẩn mực giữ tỷ lệ 0.173% khiến số lượng Fraud trong Val và Test là 74 mẫu. Kích thước mẫu lớp dương nhỏ hơn làm tăng độ nhạy biên (mỗi vụ gian lận bị bỏ sót ảnh hưởng $1/74 \approx 1.35\%$ Recall). Tuy nhiên, ưu điểm vượt trội là Precision, PR-Curve, Average Precision và Ma trận chi phí phản ánh trung thực 100% bối cảnh vận hành thực tế tại ngân hàng, triệt tiêu hoàn toàn sai lệch do thổi phồng tỷ lệ gian lận (Prevalence Inflation Bias).  
> * **Tính tái lập:** Quy trình phân chia được thực hiện qua hàm phân tầng `train_test_split(..., test_size=0.30, stratify=y, random_state=42)` sau đó chia tiếp tập tạm 30% thành 50:50 với `stratify=y_temp, random_state=42`. Cố định hạt giống giúp tái lập chính xác 100% chỉ số từng mẫu.

### 2.3. Quy trình chuẩn hóa với RobustScaler & Giới hạn kỹ thuật
Kỹ thuật `RobustScaler` chuẩn hóa dữ liệu dựa trên số liệu thống kê trung vị và khoảng tứ phân vị:

$$x_{\text{scaled}} = \frac{x - \text{Median}(x)}{\text{IQR}} = \frac{x - Q_2}{Q_3 - Q_1}$$

* **Lý do lựa chọn:** Trong khi `StandardScaler` sử dụng Giá trị trung bình ($\mu$) và Độ lệch chuẩn ($\sigma$) — hai đại lượng bị kéo lệch nghiêm trọng khi xuất hiện các giao dịch đột biến hàng chục nghìn USD; còn `MinMaxScaler` sẽ nén 99% giao dịch bình thường vào một khoảng siêu hẹp gần 0. `RobustScaler` sử dụng Trung vị ($Q_2$) và Khoảng tứ phân vị IQR với **độ kháng ngoại lai cao (breakdown point đạt xấp xỉ 50%)**, bảo toàn cấu trúc phân phối dữ liệu trung tâm mà không bị chi phối bởi các giá trị cực trị đột biến.
* **Giới hạn kỹ thuật:** `RobustScaler` không giới hạn khoảng giá trị đầu ra trong đoạn cố định [0, 1] như `MinMaxScaler`. Các điểm ngoại lai cực trị vẫn sẽ mang giá trị độ lệch lớn, nhưng không làm méo mó phân phối của phần lớn dữ liệu trung tâm.
* **Quy tắc chống rò rỉ (Zero-Leakage Pipeline):**
  ```python
  scaler_amount = RobustScaler().fit(X_train_normal[['Amount']])
  X_train_normal['Amount_scaled'] = scaler_amount.transform(X_train_normal[['Amount']])
  X_val['Amount_scaled']          = scaler_amount.transform(X_val[['Amount']])
  X_test['Amount_scaled']         = scaler_amount.transform(X_test[['Amount']])
  # Tuyệt đối không gọi scaler.fit() trên toàn bộ Dataset trước khi phân chia dữ liệu!
  ```

---

## MỤC 3: PHƯƠNG PHÁP HỌC MÁY KHÔNG GIÁM SÁT & CHUẨN HÓA ANOMALY SCORE

### 3.1. Thuật toán 1: Isolation Forest (iForest)
Được đề xuất bởi Liu et al. (2008), Isolation Forest hoạt động dựa trên nguyên lý: *Các điểm dị biệt là số ít và có giá trị thuộc tính khác biệt, do đó chúng bị cô lập rất nhanh (cần ít lần cắt nhị phân ngẫu nhiên hơn) so với các điểm dữ liệu bình thường nằm trong vùng mật độ cao.*

$$s(x, n) = 2^{-\frac{E(h(x))}{c(n)}}$$

* **Cấu hình siêu tham số kiến trúc:** Thử nghiệm số lượng cây `n_estimators` $\in [50, 100, 150, 200]$; cố định kích thước mẫu con `max_samples = 256` (hoặc thử nghiệm 512); `random_state = 42`.
* **Phân biệt rõ ràng Contamination vs. Threshold Moving:**
  * **Bản chất tham số `contamination` trong scikit-learn:** Tham số này không tham gia vào cấu trúc phân nhánh của các cây iTree, không làm thay đổi hàm tính độ dài đường dẫn trung bình $E(h(x))$ hay giá trị Anomaly Score liên tục trích xuất từ `score_samples()`. Nó chỉ đóng vai trò xác định phân vị điểm cắt nội bộ `offset_` nhằm phục vụ hàm phân loại nhị phân mặc định `predict()` (trả về +1 hoặc -1). Khi các tham số cấu trúc giữ nguyên, thay đổi `contamination` hoàn toàn **KHÔNG làm thay đổi thứ tự xếp hạng dị biệt và KHÔNG cải thiện chỉ số AUPRC (Average Precision) liên tục**.
  * **Vai trò thực nghiệm `contamination`:** Nhóm giữ lại thực nghiệm khảo sát các giá trị giả định `[0.001, 0.0017, 0.002, 0.005, 0.01]` chỉ nhằm minh họa trực quan sự dịch chuyển của ngưỡng phân loại mặc định và ma trận nhầm lẫn của hàm `predict()`, hoàn toàn không coi đây là siêu tham số cải thiện năng lực phân tách của mô hình.
  * **Tối ưu hóa siêu tham số có ý nghĩa:** Nhóm tập trung tối ưu các siêu tham số thực sự ảnh hưởng đến chất lượng mô hình gồm số lượng cây `n_estimators` và kích thước mẫu con `max_samples`. Sau khi có mô hình với điểm liên tục tốt nhất, nhóm thực hiện kỹ thuật Dịch chuyển ngưỡng (Threshold Moving) trên tập Validation để chọn ngưỡng tối ưu $\tau^*$ đáp ứng mục tiêu nghiệp vụ.

### 3.2. Thuật toán 2: Local Outlier Factor (LOF) & Chiến lược Lấy mẫu
LOF (Breunig et al., 2000) đo lường độ dị biệt bằng cách so sánh mật độ cục bộ của một điểm với mật độ của $k$ lân cận gần nhất ($k$-nearest neighbors). Điểm có mật độ thấp hơn đáng kể so với lân cận sẽ có điểm $\text{LOF} > 1$ và bị coi là ngoại lai.

* **Bắt buộc cấu hình `novelty=True`:** Chế độ tiêu chuẩn của LOF chỉ dùng cho Outlier Detection trên tập tĩnh (`fit_predict`). Để dự đoán trên các điểm dữ liệu mới chưa từng thấy ở tập Validation và Test, bắt buộc phải cấu hình `LocalOutlierFactor(novelty=True)` để sử dụng hàm `score_samples()` và `decision_function()`.
* **Chiến lược Lấy mẫu (Sampling Strategy) & Phân tích Độ phức tạp:**  
  Về mặt lý thuyết, độ phức tạp tính toán khoảng cách của LOF trong trường hợp xấu nhất hoặc brute-force là $O(N^2 \cdot d)$. Dù các cấu trúc cây chỉ mục ($k$-d tree, ball tree) có thể giảm chi phí tìm kiếm lân cận ở không gian chiều thấp, nhưng với $d = 30$, hiện tượng "tai nguyền số chiều" (curse of dimensionality) làm suy giảm nghiêm trọng hiệu quả phân vùng không gian của cây chỉ mục, khiến độ phức tạp tiệm cận $O(N^2 \cdot d)$. Với tập `Train_Normal` gồm 199,020 mẫu, ma trận khoảng cách đầy đủ đòi hỏi hàng chục gigabyte RAM, dễ dẫn đến lỗi tràn bộ nhớ (Out of Memory).  
  **Giải pháp kỹ thuật:** Nhóm áp dụng chiến lược lấy mẫu ngẫu nhiên có kiểm soát từ `Train_Normal` với kích thước $N_{\text{LOF}} = 25,000$ mẫu đại diện (`random_state=42`). Danh sách chỉ số mẫu được cố định và lưu trữ (`train_indices_lof.npy`) để bảo đảm tái lập 100%. Mô hình được fit trên mẫu con này với `novelty=True`, sau đó hàm `score_samples()` được dùng để chấm điểm trên toàn bộ tập Validation (42,721 mẫu) và Test (42,722 mẫu).
* **Thử nghiệm siêu tham số:** Quét tham số `n_neighbors` $\in [10, 20, 30, 50]$ với metric khoảng cách Minkowski ($p=2$, khoảng cách Euclid).

### 3.3. Thuật toán 3: One-Class SVM & Kiểm soát Không gian Siêu cầu
One-Class SVM (Schölkopf et al., 2001) ánh xạ dữ liệu vào không gian đặc trưng nhiều chiều thông qua Kernel RBF, sau đó tìm một siêu phẳng tối ưu để phân tách dữ liệu bình thường khỏi gốc tọa độ với lề cực đại (maximum margin).

* **Chiến lược Lấy mẫu & Kiểm soát Độ phức tạp:**  
  Bài toán quy hoạch toàn phương (Quadratic Programming) giải quyết không gian đối ngẫu của SVM có độ phức tạp thời gian từ $O(N^2)$ đến $O(N^3)$ và chi phí lưu trữ ma trận Gram $O(N^2)$. Với $N \approx 200,000$, thời gian huấn luyện có thể kéo dài nhiều giờ và gây quá tải tài nguyên tính toán.  
  **Giải pháp kỹ thuật:** Nhóm lấy mẫu ngẫu nhiên có kiểm soát $N_{\text{OCSVM}} = 20,000$ mẫu từ `Train_Normal` với `random_state=42` (lưu chỉ số `train_indices_ocsvm.npy`). Mô hình `OneClassSVM(kernel='rbf')` được huấn luyện trên mẫu con này, đảm bảo thời gian chạy trong khoảng 3–5 phút.
* **Lưới tham số thử nghiệm:**
  * Tham số $\nu$ (`nu`): Cận trên của tỷ lệ ngoại lai và cận dưới của số lượng vector hỗ trợ, quét trong $[0.001, 0.005, 0.01, 0.05]$.
  * Tham số $\gamma$ (`gamma`): Độ rộng của kernel RBF, quét giữa `'scale'`, `'auto'`, 0.01, 0.1.

### 3.4. Quy chuẩn Thống nhất Chiều & Thang đo Anomaly Score
Trong thư viện Scikit-learn, các API của 3 mô hình trả về giá trị thô với quy ước chiều ngược nhau:

| Mô hình | Hàm trích xuất Scikit-learn | Quy ước mặc định của Scikit-learn | Hàm chuẩn hóa Anomaly Score dự án ($S$) | Ý nghĩa sau chuẩn hóa |
| :--- | :--- | :--- | :--- | :--- |
| **Isolation Forest** | `score_samples(X)` | Giá trị âm. Điểm càng thấp (âm sâu) $\to$ Càng bất thường | $S_{\text{iForest}}(x) = -\text{score\_samples}(x)$ | $S$ càng cao $\to$ Dị biệt càng lớn |
| **Local Outlier Factor**<br>*(novelty=True)* | `score_samples(X)` | Điểm lân cận đối nghịch (negative outlier factor). Càng âm sâu $\to$ Càng bất thường | $S_{\text{LOF}}(x) = -\text{score\_samples}(x)$ | $S$ càng cao $\to$ Dị biệt càng lớn |
| **One-Class SVM** | `decision_function(X)` | Khoảng cách có dấu tới siêu phẳng. Điểm âm nằm ngoài biên $\to$ Bất thường | $S_{\text{OCSVM}}(x) = -\text{decision\_function}(x)$ | $S$ càng cao $\to$ Dị biệt càng lớn |

> **QUY ƯỚC CHUẨN HÓA ANOMALY SCORE & TÍNH TOÁN AUPRC TOÀN DỰ ÁN**  
> 1. **Thống nhất chiều điểm:** Toàn bộ mô hình áp dụng phép đảo dấu: $S(x) = -\text{score\_raw}(x)$ để đảm bảo quy ước chung: **Điểm càng cao $\to$ Mức độ dị biệt càng lớn**.  
> 2. **Tính toán AUPRC từ điểm liên tục:** Chỉ số AUPRC (Average Precision) và đường cong Precision-Recall bắt buộc phải tính toán trực tiếp từ chuỗi điểm Anomaly Score liên tục $S(x)$ đã đồng nhất chiều. **Tuyệt đối KHÔNG sử dụng nhãn dự đoán nhị phân (0/1 hoặc -1/+1) để tính AUPRC**, vì điều đó làm mất toàn bộ thông tin xếp hạng liên tục.  
> 3. **Không bắt buộc scale [0, 1] khi tính AUPRC:** Do Average Precision chỉ phụ thuộc vào thứ tự xếp hạng tương đối của các mẫu, mọi phép biến đổi đơn điệu (monotonic transformation) đều không làm thay đổi giá trị AUPRC. Do đó, việc chuẩn hóa về [0, 1] là không bắt buộc đối với việc tính metric.  
> 4. **Quy tắc chuẩn hóa Min-Max khi trực quan hóa so sánh:** Nếu áp dụng Min-Max scaling để hiển thị phân phối điểm của 3 mô hình trên cùng một thang đo, các tham số $\min(S_{\text{val}})$ và $\max(S_{\text{val}})$ **bắt buộc phải được fit duy nhất trên tập Validation**:  
>    $$S_{\text{norm}}(x) = \frac{S(x) - \min(S_{\text{val}})}{\max(S_{\text{val}}) - \min(S_{\text{val}})}$$  
>    Sau đó áp dụng nguyên vẹn công thức này cho tập Test: $S_{\text{norm\_test}}(x) = \frac{S_{\text{test}}(x) - \min(S_{\text{val}})}{\max(S_{\text{val}}) - \min(S_{\text{val}})}$. **Tuyệt đối KHÔNG fit lại $\min/\max$ độc lập trên tập Test để tránh rò rỉ phân phối.**  
> 5. **Tính nhất quán của ngưỡng:** Ngưỡng tối ưu $\tau^*$ được xác định trên tập Validation sẽ được áp dụng trực tiếp lên tập Test mà không được điều chỉnh lại.

---

## MỤC 4: HỆ THỐNG CHỈ SỐ ĐÁNH GIÁ (EVALUATION METRICS & COST ANALYSIS)

### 4.1. Phản biện Chỉ số Accuracy & Các độ đo Tiêu chuẩn
Trong bài toán dữ liệu mất cân bằng nghiêm trọng (0.172% Fraud), chỉ số Độ chính xác tổng thể (Accuracy) hoàn toàn vô nghĩa do Nghịch lý Độ chính xác (Accuracy Paradox). Một mô hình tầm thường phân loại mọi giao dịch là bình thường vẫn đạt Accuracy lên tới 99.828%, nhưng hoàn toàn bỏ lọt 100% các vụ gian lận (Recall = 0%). Vì vậy, nhóm loại bỏ hoàn toàn Accuracy khỏi danh sách chỉ số quyết định.

Hệ thống chỉ số khoa học cốt lõi gồm:
* **Recall (Độ nhạy — Sensitivity):** Tỷ lệ số vụ gian lận phát hiện được trên tổng số vụ gian lận thực tế:
  $$\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}$$
  Thước đo quan trọng số 1 đối với ngân hàng vì chi phí bỏ sót gian lận (FN) là tổn thất tài chính trực tiếp.
* **Precision (Độ chuẩn xác):** Tỷ lệ giao dịch thực sự gian lận trong tổng số các cảnh báo được mô hình đưa ra:
  $$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}$$
  Phản ánh độ tin cậy của cảnh báo và kiểm soát tải công việc của đội ngũ thẩm định.
* **F1-Score:** Trung bình điều hòa giữa Precision và Recall:
  $$\text{F}_1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$
* **AUPRC & Average Precision (AP) — Tiêu chuẩn vàng cho dữ liệu mất cân bằng:**
  * **Chỉ số chuẩn hóa:** Nhóm sử dụng hàm `sklearn.metrics.average_precision_score` làm công cụ tính toán chính thức cho chỉ số Average Precision (AP).
  * **Quy ước thuật ngữ & Phân biệt học thuật:** Trong các tài liệu nghiên cứu và thực tế công nghiệp, AP thường được gọi thay thế cho AUPRC. Tuy nhiên, về mặt toán học, hàm `average_precision_score` tính toán tổng diện tích theo trọng số độ biến thiên Recall giữa các ngưỡng:
    $$\text{AP} = \sum_n (R_n - R_{n-1}) P_n$$
    Cách tính này tránh được hiện tượng thổi phồng diện tích (trapezoidal interpolation optimism) của phép tính tích phân hình thang qua `auc(recalls, precisions)` khi đường cong PR có các bước nhảy gấp khúc không đơn điệu. Trong báo cáo, nhóm quy ước sử dụng chỉ số AP chuẩn từ scikit-learn và chú thích rõ bản chất toán học này.
  * AUPRC được tính trực tiếp từ chuỗi điểm Anomaly Score liên tục $S(x)$, phản ánh năng lực phân tách lớp của mô hình trên toàn bộ phổ ngưỡng mà không phụ thuộc vào một điểm cắt đơn lẻ.

### 4.2. Báo cáo Chi tiết Ma trận Nhầm lẫn (Confusion Matrix)
Mỗi mô hình sau khi chọn ngưỡng tối ưu $\tau^*$ bắt buộc phải báo cáo đầy đủ 4 giá trị định lượng:
* **True Positives (TP):** Giao dịch gian lận thực tế và mô hình đã phát hiện chính xác.
* **False Positives (FP):** Giao dịch bình thường bị hệ thống cảnh báo nhầm là gian lận (Sai số Loại I).
* **True Negatives (TN):** Giao dịch bình thường được hệ thống thông qua chính xác.
* **False Negatives (FN):** Giao dịch gian lận bị hệ thống bỏ lọt (Sai số Loại II — Rủi ro nghiêm trọng nhất).

### 4.3. Mô hình Chi phí Kinh tế (Cost-Sensitive Matrix Analysis)
Nhóm xây dựng hàm tổng chi phí thiệt hại (Total Expected Cost) nhằm tìm ngưỡng quyết định $\tau^*$ tối ưu hóa lợi ích kinh tế cho ngân hàng:

$$\text{Total\_Cost}(\tau) = C_{\text{FN}} \times \text{FN}(\tau) + C_{\text{FP}} \times \text{FP}(\tau)$$

* Quy chuẩn chi phí xác minh nhầm 1 vụ giao dịch bình thường (gọi điện, gửi OTP xác thực): $C_{\text{FP}} = 1$ (đơn vị chuẩn hóa).
* Chi phí khi bỏ sót 1 vụ gian lận (bồi thường tiền mất, tổn hại uy tín): $C_{\text{FN}}$ lớn hơn rất nhiều so với $C_{\text{FP}}$.
* **Phân tích độ nhạy (Sensitivity Analysis):** Nhóm khảo sát sự biến thiên của tổng chi phí và vị trí ngưỡng $\tau^*$ khi tỷ lệ đánh đổi chi phí $C_{\text{FN}} / C_{\text{FP}}$ nhận các giá trị: 10, 20, 30, 50, và 100.
* *Lưu ý học thuật:* Nhóm ghi chú rõ ràng trong báo cáo đây là **mô hình mô phỏng nghiên cứu (research scenario simulation)** trong khuôn khổ bài tập lớn môn Khai phá Dữ liệu, không phải là số liệu kế toán thực tế của bất kỳ ngân hàng thương mại cụ thể nào.

### 4.4. Đánh giá Hiệu năng Tính toán (Computational Profiling)
* **Thời gian huấn luyện (Training Runtime):** Thời gian fit mô hình trên dữ liệu `Train_Normal` (giây).
* **Độ trễ chấm điểm (Scoring Latency):** Thời gian xử lý dự đoán trên 1,000 giao dịch (milliseconds).
* **Bộ nhớ tiêu thụ (Peak RAM Consumption):** Đo lượng RAM tối đa sử dụng trong quá trình huấn luyện bằng thư viện `tracemalloc` hoặc `memory_profiler`.

### 4.5. Quy định Tính Tái lập Thực nghiệm (Reproducibility Protocol)
Để bảo đảm mọi thử nghiệm trong đồ án có thể kiểm chứng độc lập và tái lập 100%, nhóm thiết lập các nguyên tắc kỹ thuật bắt buộc:
* **Cố định Hạt giống Ngẫu nhiên (Global Random Seed):** Tất cả các hàm và đối tượng có tính chất ngẫu nhiên (`train_test_split`, `IsolationForest`, subsampling, PCA, t-SNE) đều được cố định tham số `random_state = 42` xuyên suốt dự án.
* **Kiểm soát và Lưu trữ Mẫu con (Subsampling Serialization):** Quy mô lấy mẫu cho LOF ($N_{\text{LOF}} = 25,000$) và One-Class SVM ($N_{\text{OCSVM}} = 20,000$) được trích xuất bằng bộ sinh số ngẫu nhiên `numpy.random.RandomState(42)`. Mảng chỉ số dòng (row indices) được tuần tự hóa và lưu ra đĩa (`train_indices_lof.npy`, `train_indices_ocsvm.npy`) để mọi thành viên và giảng viên có thể tải trực tiếp tập con chính xác mà không bị lệch mẫu.
* **Lưu ý Học thuật về Tính Tái lập Đa Nền tảng:** Nhóm cam kết tính tái lập 100% về mặt quy trình, logic mã nguồn và chỉ số lấy mẫu. Tuy nhiên, nhóm không cam kết kết quả số học dấu phẩy động giống tuyệt đối 100% giữa các hệ điều hành khác nhau (Windows vs Linux/macOS) hoặc giữa các phiên bản thư viện C/BLAS/LAPACK do sự khác biệt trong tối ưu hóa tập lệnh vi xử lý (AVX/SIMD) và tính không kết hợp (non-associativity) của phép cộng dấu phẩy động song song.

### 4.6. Giao thức Đánh giá Chuẩn mực (Evaluation Protocol)
* **Quy trình Đóng băng Quyết định:** Toàn bộ quá trình chọn siêu tham số (`n_estimators`, `max_samples`, `n_neighbors`, $\nu$, $\gamma$) và tối ưu hóa ngưỡng cắt $\tau^*$ được thực hiện độc lập 100% trên tập Validation. Sau khi cố định mô hình và điểm cắt, tập Test mới được mở ra để đánh giá một lần duy nhất.
* **Chuẩn mực Tính toán Chỉ số:** Sử dụng `sklearn.metrics.average_precision_score` cho AP/AUPRC từ continuous anomaly scores; tính toán Recall, Precision, F1 và Confusion Matrix tại ngưỡng $\tau^*$. Tuyệt đối không sử dụng kết quả Test Set để quay ngược lại tinh chỉnh mô hình.
* **Tính Liêm chính Học thuật:** Nhóm cam kết tuyệt đối **KHÔNG tạo ra số liệu thực nghiệm giả mạo**, không bịa đặt biểu đồ hay kết quả mô hình trong bản kế hoạch. Toàn bộ các chỉ số hiệu năng sẽ được ghi nhận trung thực từ log thực thi của notebook ở giai đoạn triển khai.

---

## MỤC 5: TRỰC QUAN HÓA TRONG KHÔNG GIAN GIẢM CHIỀU (PCA & t-SNE)

### 5.1. Phương pháp Giảm chiều & Quy trình Thực hiện
Nhóm triển khai hai kỹ thuật giảm chiều kinh điển để hiển thị không gian 30 thuộc tính của dữ liệu lên mặt phẳng 2D và không gian 3D:
* **PCA (Principal Component Analysis):** Phương pháp giảm chiều tuyến tính, bảo toàn phương sai tối đa của dữ liệu. Nhóm trích xuất 2 thành phần (PC1, PC2) và 3 thành phần (PC1, PC2, PC3) trên **toàn bộ 42,721 giao dịch của tập Validation** để quan sát bức tranh phân bố toàn cục mà không vi phạm nguyên tắc niêm phong tập Test.
* **t-SNE (t-Distributed Stochastic Neighbor Embedding):** Kỹ thuật giảm chiều phi tuyến tính, bảo toàn cấu trúc lân cận cục bộ (local manifold). Nhóm áp dụng t-SNE trên **mẫu đại diện phân tầng gồm 5,000 giao dịch Normal và toàn bộ 74 giao dịch Fraud lấy từ tập Validation** (`random_state=42`, `perplexity = 30, 50`, `n_iter = 1000`) nhằm kiểm soát chi phí tính toán $O(N^2)$.

### 5.2. Nguyên tắc Trực quan hóa & Cảnh báo Học thuật
* **Cơ chế tô màu kép (Dual-view Visualization):** Với mỗi biểu đồ phân tán (Scatter Plot), nhóm vẽ song song 2 góc nhìn:
  * **Góc nhìn 1 — Ground Truth:** Tô màu theo nhãn thực tế (Class 0: Normal vs Class 1: Fraud).
  * **Góc nhìn 2 — Model Prediction:** Tô màu theo nhãn dự đoán của mô hình (Inlier vs Detected Outlier). Qua đó nhận biết trực quan các điểm gian lận bị bỏ sót (FN) và các điểm cảnh báo nhầm (FP).
* **Cảnh báo học thuật & Quy định thời điểm sử dụng Test Set:**
  1. **Khám phá định tính:** Biểu đồ giảm chiều chỉ đóng vai trò hỗ trợ trực quan hóa định tính, tuyệt đối không thay thế việc đánh giá định lượng bằng các chỉ số toán học (AP, Recall, F1).
  2. **Thời điểm thực thi:** Toàn bộ quá trình chiếu giảm chiều PCA và t-SNE để khảo sát đặc trưng và phân tích trực quan hành vi mô hình được thực hiện độc lập trên tập **Validation ở Ngày 11**.
  3. **Niêm phong Test Set:** Tuyệt đối không dùng đồ thị hoặc kết quả trên tập Test để điều chỉnh lại mô hình hay ngưỡng. Sau khi hoàn tất đánh giá Test chính thức ở Ngày 12, nhóm có thể xuất bổ sung biểu đồ Test Set để đưa vào báo cáo nghiệm thu nhưng không làm thay đổi bất kỳ quyết định kỹ thuật nào.

---

## MỤC 6: KẾ HOẠCH TÁC CHIẾN 14 NGÀY CHI TIẾT (14-DAY SPRINT PLAN)

> **Nguyên tắc phối hợp nhóm:**  
> Hai thành viên làm việc song song trên hai làn chuyên môn độc lập (Data/Analysis vs Modeling/Evaluation). Hàng ngày tổ chức họp Sync 15 phút vào 21:00 để kiểm tra tiến độ, rà soát code và đồng bộ commit lên Git repository.

| Ngày | Mục tiêu trọng tâm | Nguyễn Văn Hiển Vinh (B23DCVT461)<br>*(Data & Preprocessing Lead)* | Nguyễn Minh Hoàng (B23DCVT173)<br>*(Modeling & Evaluation Lead)* | Sản phẩm bàn giao (DoD) |
| :---: | :--- | :--- | :--- | :--- |
| **Ngày 1** | Khởi động dự án & Thiết lập môi trường | • Thiết lập cấu trúc repo chuẩn: `data/`, `notebooks/`, `src/`, `reports/`, `models/`.<br>• Tải dataset gốc từ Kaggle về `data/raw/`.<br>• Cài đặt môi trường Python 3.10+, cấu hình `requirements.txt`. | • Khởi tạo GitHub repo, cấu hình branch rules (`main`, dev) và `.gitignore`.<br>• Nghiên cứu tài liệu Outlier Analysis trong Chương 12 giáo trình Han & Kamber.<br>• Thiết lập template báo cáo kỹ thuật theo chuẩn IEEE/Word. | Repo Git hoạt động + Môi trường Python sẵn sàng + Khung template báo cáo. |
| **Ngày 2** | Khám phá Dữ liệu (EDA) — Phân tích tổng quan | • Viết Notebook `01_eda_overview.ipynb`.<br>• Thống kê mô tả toàn diện: count, mean, std, min, quantiles, skewness của `Amount` và `Time`.<br>• Trực quan hóa độ lệch lớp nghiêm trọng (99.828% vs 0.172%).<br>• Soạn thảo Chương 1 Báo cáo: Bối cảnh nghiệp vụ, bài toán gian lận, rủi ro FP/FN, lập luận Unsupervised vs Supervised + SMOTE. | • Rà soát code EDA, kiểm tra tính toán tỷ lệ mất cân bằng.<br>• Đóng góp ý kiến phản biện về rủi ro nghiệp vụ và tiêu chí đánh giá mô hình. | Notebook EDA hoàn chỉnh + Bản thảo Chương 1 Báo cáo. |
| **Ngày 3** | Khám phá Đặc trưng V1-V28 & Phân phối | • Mở rộng phân tích sâu trong `01_eda_overview.ipynb`.<br>• Vẽ biểu đồ mật độ phân phối (KDE Plots) của 28 đặc trưng PCA tách biệt Normal vs Fraud.<br>• Lập ma trận hệ số tương quan (Correlation Matrix Heatmap).<br>• Soạn thảo Chương 2 Báo cáo: Đặc tả tập dữ liệu, ý nghĩa biến ẩn danh PCA, lời bình biểu đồ KDE V4, V11, V12, V14. | • Hỗ trợ phân tích tương quan giữa các đặc trưng và nhãn gian lận.<br>• Thẩm định chất lượng phân tách của các biến PCA quan trọng. | Bộ biểu đồ phân phối 28 biến + Hoàn thiện 80% Chương 2 Báo cáo. |
| **Ngày 4** | Pipeline Tiền xử lý Phân tầng & Module Metrics | • Viết Notebook `02_preprocessing_pipeline.ipynb`.<br>• Triển khai Stratified Split 70/15/15 (`random_state=42`) tạo Train (199,020 Normal, loại bỏ 344 Fraud), Val (42,647 Normal + 74 Fraud), Test (42,648 Normal + 74 Fraud).<br>• Đóng gói `RobustScaler`: fit trên `Train_Normal`, transform Val và Test.<br>• Soạn mục 2.3 & 2.4 Báo cáo: cơ chế RobustScaler và 3-way stratified split. | • Viết module đánh giá độc lập `src/metrics.py`: hàm tính Recall, Precision, F1, Average Precision (AP), PR-Curve, Confusion Matrix và Cost Matrix từ Anomaly Score liên tục.<br>• Kiểm thử chéo quy trình Zero-Leakage Split. | Data Pipeline phân tầng hoạt động độc lập + Module `metrics.py` sẵn sàng. |
| **Ngày 5** | Mô hình 1: Cài đặt Isolation Forest | • Hỗ trợ viết script đo thời gian thực thi (Runtime Profiling) và tài nguyên bộ nhớ (Peak RAM) bằng `tracemalloc`.<br>• Rà soát nội dung tiền xử lý trong Báo cáo. | • Viết Notebook `03_isolation_forest.ipynb`.<br>• Huấn luyện `IsolationForest` trên toàn bộ 199,020 dòng `Train_Normal`.<br>• Trích xuất Anomaly Score trên Validation, tính AP sơ bộ.<br>• Soạn mục 3.1 Báo cáo: Cơ sở toán học của Isolation Forest (iTree, path length $h(x)$, hệ số chuẩn hóa $c(n)$). | Notebook iForest hoàn chỉnh + Bản thảo lý thuyết mục 3.1 Báo cáo. |
| **Ngày 6** | Tối ưu iForest & Dịch chuyển ngưỡng | • Thực hiện thí nghiệm khảo sát tham số `contamination` `[0.001, 0.0017, 0.002, 0.005, 0.01]` trên hàm `predict()` mặc định; ghi nhận biến thiên `offset_` và chứng minh `score_samples()` không đổi. | • Thử nghiệm siêu tham số cấu trúc `n_estimators`: `[50, 100, 150, 200]` và `max_samples`: `[256, 512]`.<br>• Quét dải ngưỡng trên Anomaly Score liên tục trên Validation Set để vẽ PR-Curve và phân tích trade-off.<br>• Hoàn thiện mục 3.1 Báo cáo. | Bảng số liệu tối ưu iForest + Đồ thị PR-Curve + Hoàn thiện toàn diện mục 3.1. |
| **Ngày 7** | Mô hình 2: Cài đặt Local Outlier Factor (LOF) | • Thiết kế chiến lược lấy mẫu ngẫu nhiên có kiểm soát từ `Train_Normal` ($N_{\text{LOF}} = 25,000$ mẫu, `random_state=42`), lưu chỉ số `train_indices_lof.npy`.<br>• Phân tích lý thuyết vì sao LOF có độ phức tạp tiệm cận $O(N^2 \cdot d)$ do tai nguyền số chiều. | • Viết Notebook `04_local_outlier_factor.ipynb`.<br>• Huấn luyện `LocalOutlierFactor(novelty=True)` trên mẫu đại diện.<br>• Thử nghiệm `n_neighbors`: `[10, 20, 30, 50]`.<br>• Chấm điểm Anomaly Score trên Validation Set, tính AP.<br>• Soạn mục 3.2 Báo cáo: Cơ sở toán học LOF ($k$-distance, reachability distance, mật độ LRD). | Notebook LOF chạy thành công không tràn RAM + Bản thảo mục 3.2 Báo cáo. |
| **Ngày 8** | Mô hình 3: Cài đặt One-Class SVM | • Thiết kế chiến lược lấy mẫu ngẫu nhiên từ `Train_Normal` ($N_{\text{OCSVM}} = 20,000$ mẫu, `random_state=42`), lưu chỉ số `train_indices_ocsvm.npy`.<br>• Hỗ trợ đo lường thời gian thực thi và bộ nhớ. | • Viết Notebook `05_one_class_svm.ipynb`.<br>• Huấn luyện `OneClassSVM(kernel='rbf')` trên mẫu con.<br>• Quét lưới tham số: $\nu \in [0.001, 0.005, 0.01, 0.05]$ và $\gamma \in [\text{'scale'}, \text{'auto'}, 0.01, 0.1]$.<br>• Chấm điểm trên Validation Set.<br>• Soạn mục 3.3 Báo cáo: Cơ sở toán học OC-SVM (hàm nhân RBF, bài toán quy hoạch toàn phương). | Notebook OC-SVM hoàn chỉnh + Bản thảo mục 3.3 Báo cáo. |
| **Ngày 9** | Chuẩn hóa Anomaly Score & Đối sánh Validation | • Chuẩn hóa thang đo Anomaly Score của cả 3 mô hình theo quy chuẩn thống nhất: $S = -\text{score\_raw}$ ($S$ càng cao $\to$ dị biệt càng lớn).<br>• Thiết lập hàm chuẩn hóa Min-Max fit trên Validation (phục vụ trực quan hóa so sánh phân phối điểm).<br>• Lập bảng đối sánh tổng thể tài nguyên (Runtime, Peak RAM). | • Vẽ đồ thị tổng hợp: Vẽ chung Precision-Recall Curve của cả 3 mô hình trên tập Validation bằng AP (`average_precision_score`).<br>• Lập bảng so sánh hiệu năng (Precision, Recall, F1, AP).<br>• Soạn mục 3.4 Báo cáo: So sánh đa chiều và phân tích ưu nhược điểm từng thuật toán. | Bảng so sánh 3 mô hình + Đồ thị PR-Curve đối sánh tổng hợp trên Validation. |
| **Ngày 10** | Tối ưu Ngưỡng Quyết định & Ma trận Chi phí | • Viết Notebook `06_threshold_tuning_cost.ipynb`.<br>• Quét dải ngưỡng quyết định $\tau$ chi tiết trên tập Validation cho mô hình tối ưu.<br>• Khảo sát biến thiên FP và TP tại các mốc Recall mục tiêu: 75%, 80%, 85%, 90%. | • Xây dựng mô hình ma trận chi phí (Cost Matrix): Thiết lập hàm tính tổng chi phí với $C_{\text{FN}} / C_{\text{FP}}$ từ 10 đến 100.<br>• Vẽ đồ thị hàm chi phí theo ngưỡng để xác định ngưỡng cắt tối ưu kinh tế $\tau^*$.<br>• Soạn mục 4.1 & 4.2 Báo cáo: Tối ưu ngưỡng và phân tích độ nhạy chi phí kinh doanh. | Notebook Threshold & Cost hoàn chỉnh + Cố định ngưỡng tối ưu $\tau^*$ trên Validation. |
| **Ngày 11** | Trực quan hóa Không gian Giảm chiều (PCA & t-SNE trên Validation) | • Viết Notebook `07_dimensionality_reduction.ipynb`.<br>• Áp dụng PCA chiếu 30 chiều xuống 2D và 3D trên toàn bộ 42,721 mẫu Validation Set.<br>• Vẽ scatter plots phân biệt Ground Truth (Normal vs Fraud) và Model Prediction (Inlier vs Outlier).<br>• Đo tỷ lệ phương sai tích lũy giải thích bởi các PCs. | • Áp dụng t-SNE trên tập mẫu đại diện phân tầng của Validation (5,000 Normal + 74 Fraud, `random_state=42`).<br>• Tinh chỉnh `perplexity = 30, 50`, tạo biểu đồ phân bố cụm phi tuyến.<br>• Soạn thảo Chương 4 Báo cáo: Phân tích trực quan hóa không gian giảm chiều và đánh giá tổng hợp. | Bộ biểu đồ 2D/3D PCA & t-SNE trên Validation + Hoàn thành Chương 4 Báo cáo. |
| **Ngày 12** | Mở Niêm phong Test Set, Đánh giá Độc lập & Refactor Code | • Đồng mở niêm phong tập kiểm thử độc lập Test Set (42,722 mẫu).<br>• Hỗ trợ kiểm thử chéo và ghi nhận log thực nghiệm.<br>• Rà soát toàn bộ source code: refactoring, docstrings chuẩn mực, bảo đảm `Run All Cells` trơn tru.<br>• Viết file `README.md` hoàn chỉnh: Hướng dẫn cài đặt, cấu trúc repo và cách tái lập kết quả từ đầu. | • Nạp Test Set, áp dụng nguyên vẹn mô hình và ngưỡng tối ưu $\tau^*$ đã chốt ở Ngày 10 để tính toán chỉ số Test chính thức (Final Test Benchmark).<br>• Ghi nhận số liệu kiểm thử chính thức vào Báo cáo (Bảng kết quả Test Set độc lập).<br>• Soạn Chương 5 Báo cáo: Kết luận, bài học kinh nghiệm, rào cản kỹ thuật và hướng phát triển. | Bảng kết quả Test Set chính thức + Toàn bộ Source Code chuẩn chỉnh + File `README.md`. |
| **Ngày 13** | Hoàn thiện Báo cáo Kỹ thuật & Thiết kế Slide Deck | • Đảm nhiệm Trưởng ban biên tập Báo cáo: tổng hợp 5 chương, kiểm tra chính tả, định dạng font Times New Roman, canh lề, đánh số bảng biểu tự động.<br>• Rà soát tính chính xác của các công thức toán học và trích dẫn chuẩn IEEE/APA.<br>• Xuất bản file Báo cáo Kỹ thuật chính thức (Word + PDF). | • Thiết kế bộ Slide thuyết trình bằng PowerPoint (20 đến 25 slides).<br>• Bố cục slide mạch lạc: Đặt vấn đề $\to$ Dữ liệu & Tiền xử lý $\to$ 3 Thuật toán Unsupervised $\to$ Đối sánh Validation $\to$ Tối ưu ngưỡng & Chi phí $\to$ Trực quan hóa $\to$ Đánh giá Test độc lập $\to$ Kết luận.<br>• Đưa các biểu đồ PR-Curve, PCA/t-SNE vào slide. | File Báo cáo Kỹ thuật Final (Word + PDF) + File Slide PowerPoint chuyên nghiệp. |
| **Ngày 14** | Tổng duyệt Thuyết trình & Sẵn sàng Bảo vệ | • Đảm nhiệm trình bày Phần 1 & Phần 4: Đặt vấn đề, Bối cảnh nghiệp vụ, Đặc tả dữ liệu, Pipeline tiền xử lý phân tầng, Chiến lược LOF và Kết luận.<br>• Chuẩn bị máy tính và môi trường demo code dự phòng. | • Đảm nhiệm trình bày Phần 2 & Phần 3: Lý thuyết và thực nghiệm Isolation Forest, One-Class SVM, Đối sánh Validation, Tối ưu ngưỡng theo chi phí, Trực quan hóa và Đánh giá Test độc lập.<br>• Cùng luyện tập bộ câu hỏi phản biện chuyên sâu (Q&A Cheat Sheet). | Buổi tổng duyệt khớp thời gian 12 đến 15 phút + Nhóm tự tin 100% sẵn sàng bảo vệ. |

---

## MỤC 7: MA TRẬN PHÂN CÔNG TRÁCH NHIỆM (REVISED RACI MATRIX)
*(R: Responsible - Người thực hiện chính | A: Accountable - Người chịu trách nhiệm phê duyệt duy nhất | C: Consulted - Người được tham vấn chuyên môn | I: Informed - Người nhận thông tin bàn giao)*

| Mã việc | Nhiệm vụ kỹ thuật cốt lõi (Work Package) | Hiển Vinh (B23DCVT461)<br>*(Leader \| Data Lead)* | Minh Hoàng (B23DCVT173)<br>*(Modeling Lead)* | Yêu cầu nghiệm thu kỹ thuật (DoD) |
| :--- | :--- | :---: | :---: | :--- |
| **WP01** | Thiết lập Git repository, môi trường & Cấu trúc thư mục | **A / R** | **R** | Cấu trúc thư mục chuẩn, Git branch rõ ràng, `requirements.txt` cài đặt thành công không xung đột thư viện. |
| **WP02** | Phân tích Khám phá Dữ liệu (EDA) & Soạn Chương 1 | **A / R** | **C** | Notebook `01_eda_overview.ipynb` hoàn chỉnh, kiểm kê chính xác 284,807 dòng; hoàn thành bản thảo Chương 1 Báo cáo. |
| **WP03** | Pipeline Phân chia 3 tập phân tầng & Chuẩn hóa RobustScaler | **A / R** | **C** | Stratified Split 70/15/15 (`random_state=42`); Train 199,020 Normal (loại 344 Fraud), Val 74 Fraud (~0.173%), Test 74 Fraud (~0.173%); RobustScaler chỉ fit trên Train_Normal. |
| **WP04** | Xây dựng Module Đánh giá (Metrics & AP/AUPRC Engine) | **C** | **A / R** | File `src/metrics.py` chuẩn hóa hàm AP (`average_precision_score`), PR-Curve, Confusion Matrix, Cost Matrix từ Anomaly Score liên tục. |
| **WP05** | Huấn luyện, Tối ưu Siêu tham số & Dịch chuyển ngưỡng iForest | **C** | **A / R** | Notebook `03_isolation_forest.ipynb`; làm rõ contamination chỉ đổi `offset_`; tối ưu `n_estimators`, `max_samples`; soạn thảo mục 3.1 Báo cáo. |
| **WP06** | Lấy mẫu Kiểm soát & Thực nghiệm LOF (`novelty=True`) | **R** | **A / R** | Vinh thiết kế mẫu $N=25,000$ và lưu chỉ số; Hoàng cấu hình `novelty=True`, quét `n_neighbors` và soạn thảo mục 3.2 Báo cáo; chạy không tràn RAM. |
| **WP07** | Lấy mẫu Kiểm soát & Thực nghiệm One-Class SVM | **C** | **A / R** | Lấy mẫu $N=20,000$ (`random_state=42`), quét lưới tham số $\nu, \gamma$ với kernel RBF, kiểm soát thời gian chạy < 5 phút; soạn thảo mục 3.3 Báo cáo. |
| **WP08** | Chuẩn hóa Chiều Anomaly Score & Đối sánh trên Validation | **R** | **A / R** | Vinh đảo dấu thống nhất $S=-\text{score}$; Hoàng tổng hợp PR-Curve chung trên Validation bằng AP, lập bảng so sánh hiệu năng và tài nguyên; soạn mục 3.4. |
| **WP09** | Tối ưu hóa Ngưỡng Quyết định & Mô hình Chi phí Kinh tế | **R** | **A / R** | Vinh quét dải ngưỡng cho các mốc Recall mục tiêu; Hoàng thiết lập hàm tổng chi phí $C_{\text{FN}}/C_{\text{FP}} \in [10, 100]$ và chốt ngưỡng tối ưu $\tau^*$; soạn mục 4.1 & 4.2. |
| **WP10** | Giảm chiều & Trực quan hóa (PCA 2D/3D & t-SNE trên Validation) | **A / R** | **R** | Vinh chiếu PCA trên tập Validation; Hoàng chạy t-SNE trên mẫu 5,000 Normal + 74 Fraud từ Validation; vẽ Dual-view Plots; soạn mục 4.3 Báo cáo. |
| **WP11** | Đánh giá Độc lập trên Test Set & Viết Hướng dẫn README.md | **R** | **A / R** | Hoàng nạp Test Set ở Ngày 12, áp dụng mô hình và $\tau^*$ cố định để lập Bảng kết quả Test; Vinh viết `README.md` hướng dẫn tái lập 100%; rà soát mã nguồn. |
| **WP12** | Tổng hợp Báo cáo Kỹ thuật Toàn diện (Chương 1–5) | **A / R** | **R** | Vinh là Trưởng ban biên tập, viết Ch 1, 2, 5 và chuẩn hóa format Word/PDF; Hoàng viết Ch 3, 4; kiểm tra chéo công thức toán và trích dẫn IEEE/APA. |
| **WP13** | Thiết kế Bộ Slide Thuyết trình & Luyện tập Phản biện | **R** | **A / R** | Hoàng chủ trì thiết kế Slide PowerPoint (20–25 slides); Vinh phản biện nội dung; cả hai cùng luyện tập bảo vệ và đối đáp Q&A. |

---

## MỤC 8: BỘ CÂU HỎI PHẢN BIỆN CHUYÊN SÂU DỰ PHÒNG (DEFENSE Q&A CHEAT SHEET V2.1)

1. **"Tại sao nhóm không dùng các mô hình phân loại có giám sát mạnh mẽ như Random Forest hay XGBoost kết hợp với kỹ thuật SMOTE để giải quyết mất cân bằng?"**  
   *Trả lời chuẩn mực:*  
   "Kính thưa Thầy/Cô, hướng tiếp cận có giám sát kết hợp SMOTE có hai hạn chế kỹ thuật lớn trong bài toán gian lận tài chính thực tế:  
   1) *Giới hạn của mô hình có giám sát:* Mô hình Supervised tối ưu hóa phân ranh giới dựa trên các kiểu mẫu gian lận đã có nhãn trong quá khứ, do đó gặp suy giảm hiệu năng nghiêm trọng và khó tổng quát hóa trước các thủ đoạn gian lận mới mang hình thái phân phối khác biệt (concept drift / zero-day fraud). Ngược lại, Học không giám sát (Unsupervised Anomaly Detection) học cấu trúc phân phối danh định của các giao dịch bình thường; bất kỳ giao dịch nào có mật độ thưa thớt hoặc dễ bị cô lập đều bị gắn cờ dị biệt mà không phụ thuộc vào nhãn lịch sử.  
   2) *Vấn đề nội suy của SMOTE & Phân biệt dị biệt:* SMOTE tổng hợp mẫu bằng cách nội suy tuyến tính giữa các lân cận. Trong không gian 30 chiều với độ mất cân bằng 0.17%, việc nội suy dễ tạo ra các điểm dữ liệu giả mạo (synthetic artifacts) nằm ở những vùng không gian không có thực tế, làm méo mó ranh giới quyết định. Tuy nhiên, nhóm cũng làm rõ trong báo cáo: dị biệt (outlier) không đồng nhất hoàn toàn với gian lận (fraud); một số giao dịch bình thường đột biến có thể bị báo động nhầm (FP), do đó việc tối ưu ngưỡng cắt $\tau^*$ dựa trên hàm chi phí là bắt buộc theo đúng tinh thần môn học."

2. **"Tại sao nhóm chọn RobustScaler cho thuộc tính Amount và Time mà không sử dụng StandardScaler hay MinMaxScaler?"**  
   *Trả lời chuẩn mực:*  
   "Kính thưa Thầy/Cô, đặc trưng `Amount` có phân phối lệch phải cực kỳ nghiêm trọng: trung vị chỉ là 22.00 USD, nhưng giá trị lớn nhất lên tới 25,691.16 USD:  
   * Nếu dùng `StandardScaler`: Giá trị trung bình ($\mu$) và độ lệch chuẩn ($\sigma$) bị kéo lệch rất mạnh bởi các giao dịch ngoại lai lớn này, khiến đại đa số các giao dịch bình thường bị co cụm lại gần 0.  
   * Nếu dùng `MinMaxScaler`: 99% dữ liệu sẽ bị nén nghẽn vào một dải hẹp [0, 0.01], làm triệt tiêu sự khác biệt về phương sai của các giao dịch thông thường.  
   * `RobustScaler` sử dụng Trung vị ($Q_2$) và Khoảng tứ phân vị ($\text{IQR} = Q_3 - Q_1$). Vì hai đại lượng này dựa trên thứ tự xếp hạng (rank-based), chúng có độ kháng ngoại lai cao (breakdown point đạt xấp xỉ 50%), bảo toàn nguyên vẹn hình dáng phân phối trung tâm của dữ liệu mà không bị chi phối bởi các giá trị cực đại đột biến."

3. **"Tại sao thuật toán Isolation Forest có thể huấn luyện trên toàn bộ dữ liệu, trong khi LOF và One-Class SVM lại bắt buộc phải lấy mẫu đại diện (Sampling)?"**  
   *Trả lời chuẩn mực:*  
   "Kính thưa Thầy/Cô, điều này bắt nguồn từ sự khác biệt cơ bản về độ phức tạp tính toán giữa các thuật toán:  
   * **Isolation Forest:** Sử dụng kỹ thuật ngẫu nhiên hóa (random partitioning) trên cây nhị phân với kích thước mẫu con cố định cho mỗi cây là $\psi = 256$. Độ phức tạp huấn luyện là tuyến tính $O(t \cdot \psi \log \psi)$ với $t$ là số lượng cây, cho phép mô hình mở rộng cực tốt (Scalability) trên toàn bộ 199,020 dòng `Train_Normal`.  
   * **Local Outlier Factor (LOF):** Phải tính toán ma trận khoảng cách giữa các cặp điểm để tìm $k$ lân cận gần nhất. Trong không gian $d = 30$ chiều, tai nguyền số chiều (curse of dimensionality) làm mất tác dụng phân vùng của cây $k$-d tree/ball tree, khiến độ phức tạp tiệm cận trường hợp xấu nhất $O(N^2 \cdot d)$. Với $N \approx 200,000$, số phép tính khoảng cách lên tới hàng chục tỷ phép tính, vượt quá dung lượng RAM thông thường.  
   * **One-Class SVM:** Giải bài toán quy hoạch toàn phương (Quadratic Programming) với kernel RBF, độ phức tạp thời gian từ $O(N^2)$ đến $O(N^3)$ và bộ nhớ ma trận Gram xấp xỉ $O(N^2)$.  
   Vì vậy, việc lấy mẫu ngẫu nhiên có kiểm soát từ tập `Train_Normal` ($N_{\text{LOF}} = 25,000$ và $N_{\text{OCSVM}} = 20,000$ với `random_state=42`) là giải pháp bắt buộc và hoàn toàn chuẩn mực trong kỹ nghệ dữ liệu nhằm cân đối giữa độ chính xác và tài nguyên tính toán."

4. **"Sự khác biệt cốt lõi giữa tham số Contamination và kỹ thuật Dịch chuyển ngưỡng (Threshold Moving) là gì?"**  
   *Trả lời chuẩn mực:*  
   "Kính thưa Thầy/Cô:  
   * **Contamination** là một tham số truyền vào mô hình khi khởi tạo (trong scikit-learn). Nó đại diện cho tỷ lệ phần trăm dữ liệu nhiễu dự đoán trước trong tập huấn luyện, từ đó mô hình tự động chọn ra một điểm cắt mặc định (`offset_`) cho hàm `predict()`. Việc thay đổi contamination không làm thay đổi các cây nhị phân iTrees, không làm thay đổi `score_samples()` và không làm thay đổi AUPRC liên tục. Nhóm khảo sát contamination chỉ để quan sát sự dịch chuyển của hàm `predict()` mặc định.  
   * **Threshold Moving** là kỹ thuật hậu xử lý (Post-processing) độc lập. Thay vì nhận nhãn nhị phân đóng băng {-1, 1}, nhóm trích xuất điểm dị biệt liên tục (Continuous Anomaly Score) của các giao dịch trên tập Validation. Sau đó, nhóm quét dải ngưỡng $\tau$ để chủ động chọn điểm cắt phân loại sao cho đáp ứng chính xác mục tiêu nghiệp vụ: ví dụ ưu tiên đạt Recall từ 80% trở lên hoặc tối thiểu hóa hàm tổng chi phí tổn thất, đúng theo nguyên lý được trình bày trong Slide 78 của môn học."

5. **"Tại sao nhóm tuyệt đối không được sử dụng tập Test Set để tinh chỉnh siêu tham số hoặc tìm ngưỡng quyết định?"**  
   *Trả lời chuẩn mực:*  
   "Kính thưa Thầy/Cô, việc dùng tập Test Set để chọn ngưỡng, tinh chỉnh tham số hoặc phân tích trực quan khám phá (như PCA/t-SNE) sẽ dẫn đến hiện tượng **Rò rỉ dữ liệu kiểm thử (Data Snooping / Test Set Leakage)**. Khi đó, mô hình và ngưỡng đã bị tối ưu hóa quá mức (overfitted) cho riêng tập Test, khiến kết quả đánh giá bị lạc quan giả tạo và không còn phản ánh trung thực khả năng phát hiện gian lận của hệ thống khi triển khai vào môi trường thực tế. Nhóm tuân thủ nghiêm ngặt nguyên tắc: Mọi quyết định về mô hình, tham số, ngưỡng tối ưu $\tau^*$ cũng như trực quan hóa khám phá PCA/t-SNE đều được thực hiện độc lập trên tập Validation (ở Ngày 11); tập Test chỉ được mở niêm phong một lần duy nhất ở Ngày 12 sau khi mọi quyết định đã được đóng băng."

---

## MỤC 9: TIÊU CHÍ NGHIỆM THU DỰ ÁN (DEFINITION OF DONE — DoD)
| Cấu phần | Tiêu chuẩn Nghiệm thu Kỹ thuật | Phương pháp Xác minh |
| :--- | :--- | :--- |
| **Mã nguồn (Source Code)** | • Toàn bộ 7 Jupyter Notebooks chạy `Run All Cells` liên tục không phát sinh bất kỳ lỗi nào.<br>• Khởi tạo seed cố định `random_state=42` ở mọi bước ngẫu nhiên để đảm bảo tính tái lập 100%.<br>• Lưu trữ chỉ số lấy mẫu con `train_indices_lof.npy` và `train_indices_ocsvm.npy`.<br>• Code có cấu trúc rõ ràng, chú thích đầy đủ bằng tiếng Việt/Anh.<br>• Có file `README.md` hướng dẫn chạy chi tiết. | Thành viên kiểm tra chéo bằng cách clone repo về máy thứ hai và chạy lại toàn bộ quy trình từ terminal sạch. |
| **Báo cáo Kỹ thuật (Technical Report)** | • Độ dài từ 25 đến 35 trang chuẩn A4.<br>• Cấu trúc 5 chương hoàn chỉnh: Mở đầu $\to$ Dữ liệu & Tiền xử lý phân tầng $\to$ Cơ sở Toán học & Thuật toán $\to$ Kết quả Thực nghiệm & Chi phí $\to$ Kết luận.<br>• Đầy đủ bảng số liệu đối sánh trung thực (tỷ lệ Fraud ~0.173%), hình ảnh biểu đồ rõ nét, công thức toán học chuẩn xác, định dạng 100% Times New Roman.<br>• Trích dẫn tài liệu tham khảo chuẩn IEEE/APA. | Kiểm tra định dạng, kiểm tra tính nhất quán của số liệu giữa báo cáo và kết quả log từ notebook. |
| **Slide Thuyết trình (Presentation Deck)** | • Số lượng từ 20 đến 25 slides PowerPoint.<br>• Thiết kế chuyên nghiệp, màu sắc chuẩn mực, ưu tiên biểu đồ trực quan, sơ đồ kiến trúc và bảng số liệu.<br>• Trình bày rõ ràng 3 trụ cột: Dữ liệu phân tầng $\to$ 3 Thuật toán đối sánh $\to$ Giá trị nghiệp vụ, Tối ưu ngưỡng & Phân tích chi phí. | Thuyết trình thử nội bộ trong thời gian 12 đến 15 phút, kiểm tra hiệu ứng chuyển trang và độ tương phản máy chiếu. |
