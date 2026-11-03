## 📌 Mô tả Thay đổi (Description)
<!-- Tóm tắt ngắn gọn các thay đổi, notebook hoặc module được cập nhật -->

- **Module / Notebook cập nhật:** 
- **Mục tiêu kỹ thuật:** 
- **Người thực hiện:** @<!-- username --> (Nguyễn Văn Hiển Vinh / Nguyễn Minh Hoàng)

---

## 🛠 Lộ trình Dự án (Sprint Roadmap)
- [ ] Ngày 1: Setup repo & Môi trường
- [ ] Ngày 2-3: EDA & Phân tích đặc trưng V1-V28
- [ ] Ngày 4: Data Split 3 tập & RobustScaler
- [ ] Ngày 5-6: Huấn luyện & Tối ưu Isolation Forest
- [ ] Ngày 7: Lấy mẫu & Thực nghiệm LOF (novelty=True)
- [ ] Ngày 8: Lấy mẫu & Thực nghiệm One-Class SVM
- [ ] Ngày 9-10: Chuẩn hóa Score, Threshold-Moving & Ma trận chi phí
- [ ] Ngày 11: Trực quan hóa PCA & t-SNE
- [ ] Ngày 12: Đánh giá độc lập trên Test Set & README
- [ ] Ngày 13-14: Báo cáo Kỹ thuật & Slide Thuyết trình

---

## ✅ Tiêu chí Nghiệm thu Kỹ thuật (DoD Checklist)
- [ ] Đã cố định `random_state = 42` ở tất cả các hàm ngẫu nhiên để đảm bảo tính tái lập 100%.
- [ ] Đã kiểm tra không có rò rỉ dữ liệu (Zero-leakage): Scaler chỉ fit trên `Train_Normal`.
- [ ] Score dị biệt đã được đảo dấu thống nhất: **Score càng cao -> Dị biệt càng lớn**.
- [ ] Notebook chạy từ đầu tới cuối (Restart & Run All) không phát sinh lỗi.
- [ ] Không commit các file dữ liệu nặng (`.csv`, `.zip`) lên Git.
