# BÁO CÁO PHÂN TÍCH CHUYÊN SÂU: KHẮC PHỤC OVERFITTING & DOMAIN SHIFT TRONG PHÂN ĐOẠN THẺ (YOLO SEGMENTATION)

Báo cáo này tập trung phân tích các nguyên nhân cốt lõi khiến mô hình phân đoạn thẻ đạt chỉ số kiểm thử lý thuyết rất cao (mAP ~99.5%) nhưng hoạt động kém hiệu quả và dễ bị lỗi trong môi trường thực tế, từ đó đề xuất các giải pháp kỹ thuật cụ thể dựa trên giáo trình thị giác máy tính và case study thực tế từ các doanh nghiệp eKYC.

---

## 1. PHÂN TÍCH NGUYÊN NHÂN CỐT LÕI (ROOT CAUSE ANALYSIS)

Dựa trên kết quả huấn luyện mô hình YOLO26m-seg trước đó và ảnh suy luận thực tế, hệ thống đang gặp phải 3 vấn đề kinh điển của học sâu:

### 1.1. Rò rỉ dữ liệu (Data Leakage) & Tập Validation "Quá Dễ"
- **Bản chất kỹ thuật**: Lỗi này xảy ra khi chia tập dữ liệu huấn luyện (Train) và tập kiểm thử (Val) ngẫu nhiên theo từng ảnh (**Image-level Split**) thay vì theo nguồn gốc thẻ (**Identity-level Split**).
  - *Ví dụ*: Một chiếc thẻ CCCD được chụp 5 lần ở các góc hoặc độ sáng hơi khác nhau. Nếu 4 ảnh nằm ở tập Train và 1 ảnh nằm ở tập Val, mô hình sẽ ghi nhớ (memorize) hoàn hảo các đặc trưng tĩnh của chiếc thẻ đó (ảnh chân dung, các chữ số cụ thể). Khi chạy Validation, mAP sẽ đạt gần 100%, nhưng khi triển khai thực tế gặp một chiếc thẻ lạ, mô hình sẽ bị mất phương hướng.
- **Case Study doanh nghiệp**: Các công ty làm eKYC lớn (như FPT.AI, VinAI) đều quy định bắt buộc phải phân nhóm ảnh theo ID khách hàng hoặc ID thẻ trước khi thực hiện chia bộ dữ liệu. Một ID chỉ được phép nằm ở một trong ba tập duy nhất (Train, Val hoặc Test).

### 1.2. Dữ liệu huấn luyện "Quá Dễ" nhưng chạy quá nhiều Epoch
- **Bản chất kỹ thuật**: Bộ dữ liệu huấn luyện chủ yếu gồm các ảnh chụp thẻ thẳng đứng, phẳng phiu trên mặt bàn đồng nhất, không bị che khuất và có ánh sáng tốt.
- **Vấn đề**: Khi huấn luyện một mô hình có dung lượng tham số lớn như `YOLO26m-seg` với **100 epoch** trên tập dữ liệu thiếu tính đa dạng này, mô hình sẽ nhanh chóng bị **Overfitting**. 
  - Mô hình không học được biên dạng hình học của tấm thẻ (các góc vuông, cạnh biên thẳng) mà chỉ học thuộc lòng màu sắc xanh ngọc đặc trưng hoặc hoa văn bên trong thẻ. 
  - Khi gặp ảnh thực tế bị nghiêng nặng, có ngón tay đè lên hoặc nền lá cây phức tạp, mô hình không thể nội suy (interpolate) được đường biên thực sự.

### 1.3. Lệch phân phối ảnh (Domain Shift / Covariate Shift) do Tiền xử lý chưa tối ưu
- **Bản chất kỹ thuật**: Sự khác biệt lớn giữa phân phối xác suất đầu vào của tập huấn luyện $P(X_{\text{train}})$ và tập thực tế $P(X_{\text{test}})$.
- **Nguyên nhân**:
  - **Ánh sáng và Độ tương phản**: Ảnh train thường được chụp trong điều kiện ánh sáng chuẩn studio hoặc đã qua xử lý lọc sạch. Ảnh thực tế của người dùng thường bị lóa đèn flash (glare), bóng râm (shadow), hoặc độ tương phản cực kỳ thấp do chụp bằng camera điện thoại giá rẻ.
  - **Nhiễu cảm biến (Sensor Noise)**: Ảnh thực tế bị mờ do chuyển động (motion blur) hoặc nhiễu hạt trong tối.
- Phép chia chuẩn hóa cơ bản `1/255` của YOLO chỉ có tác dụng đưa pixel về khoảng `[0, 1]`, hoàn toàn không thể san phẳng sự khác biệt lớn về lược đồ màu sắc (histogram) giữa ảnh sạch và ảnh thực tế.

---

## 2. ĐỀ XUẤT GIẢI PHÁP KHẮC PHỤC CỤ THỂ (PROPOSED SOLUTIONS)

Để xây dựng một mô hình phân đoạn thẻ mạnh mẽ, hoạt động ổn định ngoài thực tế, chúng ta cần triển khai các giải pháp kỹ thuật sau:

### 2.1. Cải tiến dữ liệu (Data-Centric AI)

#### Hướng giải quyết 1: Sửa lỗi Data Leakage bằng Stratified Identity-based Split
- **Cách làm**: Thực hiện gom nhóm (grouping) toàn bộ ảnh trong dataset theo danh tính chiếc thẻ (ID thẻ). Chia tập dữ liệu sao cho các ảnh của cùng một chiếc thẻ chỉ xuất hiện ở tập Train hoặc tập Val.
- **Mục tiêu**: Đảm bảo tập Validation phản ánh đúng khả năng tổng quát hóa của mô hình trên các đối tượng chưa từng gặp.

#### Hướng giải quyết 2: Áp dụng Lớp Huấn Luyện Tổng Quát (Class-Agnostic Single-Class)
- **Cách làm**: Gộp tất cả các nhãn loại thẻ (`CHIP_FRONT`, `CHIP_BACK`, `DRIVER_LICENSE`,...) thành một lớp duy nhất là `card`.
- **Mục tiêu**: Ép mô hình YOLO chỉ tập trung học các đặc trưng hình học thuần túy (4 góc vuông, các đường biên thẳng, chất liệu thẻ nhựa) thay vì học các hoa văn đặc thù bên trong, giúp mô hình tăng khả năng tổng quát hóa và linh hoạt với mọi loại thẻ mới.

#### Hướng giải quyết 3: Bổ dung tập ảnh Null (Negative Samples)
- **Cách làm**: Tích hợp khoảng 10% ảnh âm tính (không chứa thẻ, ví dụ như bộ dữ liệu 631 ảnh `null_dataset` chứa bàn làm việc, laptop, sách vở, bút mà chúng ta vừa lọc) vào tập train.
- **Mục tiêu**: Dạy cho mô hình cách bỏ qua các vật thể nhiễu ngoài nền, triệt tiêu hoàn toàn các lỗi đốm xanh False Positive ngoài nền lá cây.

---

### 2.2. Tăng cường dữ liệu hướng thực tế (Domain-specific Augmentation)

#### Hướng giải quyết 4: Tích hợp mô phỏng ngón tay che khuất (Synthetic Finger Occlusion)
- **Cách làm**: Sử dụng thư viện tăng cường dữ liệu để tự động cắt các mảng da tay/ngón tay ngẫu nhiên và chèn (overlay) đè lên viền thẻ trong tập train.
- **Mục tiêu**: Dạy cho mô hình hiểu rằng viền thẻ vẫn kéo dài liên tục kể cả khi bị ngón tay cắt ngang qua, giúp mô hình học cách phân tách biên giới giữa da tay và mép thẻ một cách sắc nét.

#### Hướng giải quyết 5: Tăng cường nhiễu camera (Sensor & Lighting Augmentation)
- Bổ sung các bộ lọc trong quá trình train:
  - **Random Brightness/Contrast**: Thay đổi độ sáng tối ngẫu nhiên trên diện rộng.
  - **Gaussian Noise & Motion Blur**: Mô phỏng nhiễu hạt camera điện thoại và rung tay khi chụp.
  - **Random Glare**: Giả lập các vệt lóa sáng do ánh đèn flash chiếu vào thẻ nhựa.

---

### 2.3. Tối ưu hóa tiền xử lý đầu vào khi Inference (Suboptimal Pre-processing Correction)

#### Hướng giải quyết 6: Chuẩn hóa lược đồ sáng tự động (Histogram Normalization)
- **Cách làm**: Trước khi đưa ảnh thực tế vào mô hình YOLO để predict, chúng ta áp dụng một bộ tiền xử lý ánh sáng nhanh bằng OpenCV:
  - Chuyển ảnh sang hệ màu **LAB** (tách biệt kênh độ sáng $L$ và kênh màu sắc $A, B$).
  - Áp dụng thuật toán **CLAHE (Contrast Limited Adaptive Histogram Equalization)** lên kênh độ sáng $L$ để tự động bù sáng cho các vùng tối và dập lóa cho các vùng cháy sáng, giữ nguyên màu sắc gốc.
  - Chuyển đổi ngược lại hệ màu **RGB** trước khi nạp vào mô hình.
- **Mục tiêu**: Đưa phân bố histogram ánh sáng của mọi bức ảnh thực tế chụp ở các điều kiện khác nhau về cùng một dải chuẩn tương đồng với tập train.
