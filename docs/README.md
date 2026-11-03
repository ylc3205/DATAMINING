# Tài liệu & Kế hoạch Dự án (Project Documentation & Plans)

Thư mục này lưu trữ toàn bộ hồ sơ kỹ thuật, kế hoạch tác chiến 14 ngày và đặc tả yêu cầu kỹ thuật (Software/System Requirements Specification - SRS) của nhóm cho bài tập lớn môn **Khai phá Dữ liệu (Data Mining)**.

---

## 📑 Danh mục Tài liệu

1. **[Bản Kế hoạch Dự án & Đặc tả SRS V2 (File Word)](./KE_HOACH_DU_AN_CREDIT_CARD_FRAUD_DETECTION_V2.docx)**
   * Định dạng chuẩn Microsoft Word (.docx), font chữ 100% Times New Roman, full màu đen, đã căn lề và định dạng bảng biểu theo chuẩn học thuật.
   * Dùng để in ấn, nộp Giảng viên chấm điểm hoặc chia sẻ cho các bên liên quan.

2. **[Bản Kế hoạch Dự án & Đặc tả SRS V2 (File Markdown)](./KE_HOACH_DU_AN_CREDIT_CARD_FRAUD_DETECTION_V2.md)**
   * Định dạng Markdown (.md) hiển thị trực tiếp và rõ nét trên GitHub.
   * Dành cho các thành viên trong nhóm theo dõi nhanh các quy chuẩn toán học, công thức và bảng phân chia nhiệm vụ hàng ngày.

---

## 🎯 Tóm tắt 9 Mục Cốt lõi trong Tài liệu:
* **Mục 1:** Tổng quan bài toán, rủi ro kinh doanh (FP vs FN) và lập luận chọn Unsupervised Anomaly Detection.
* **Mục 2:** Đặc tả 30 thuộc tính dữ liệu và quy trình phân chia 3 tập **Zero-Leakage** (Train Normal 70%, Val 15%, Test 15%).
* **Mục 3:** Phương pháp học máy không giám sát (**Isolation Forest**, **LOF novelty=True**, **One-Class SVM RBF**) và quy chuẩn thống nhất Anomaly Score (Score càng cao $\rightarrow$ Dị biệt càng lớn).
* **Mục 4:** Hệ thống chỉ số đánh giá (**Recall, Precision, F1, AUPRC**) và mô hình Chi phí kinh tế (Cost Matrix: $C_{\text{FN}} / C_{\text{FP}} = 20 - 50$).
* **Mục 5:** Trực quan hóa không gian giảm chiều (**PCA 2D/3D & t-SNE**) với cơ chế tô màu kép (Ground Truth vs Prediction).
* **Mục 6:** **Lịch trình tác chiến 14 ngày** chi tiết giữa Nguyễn Văn Hiển Vinh và Nguyễn Minh Hoàng.
* **Mục 7:** **Ma trận RACI** phân định trách nhiệm 12 đầu việc kỹ thuật cốt lõi.
* **Mục 8:** **Bộ 5 câu hỏi phản biện chuyên sâu** và định hướng trả lời bảo vệ bài tập lớn trước Hội đồng.
* **Mục 9:** **Tiêu chí nghiệm thu dự án (DoD)** cho Source Code, Báo cáo kỹ thuật và Slide thuyết trình.
