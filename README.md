# 📝 DocU - OCR Learning & Development Project

Dự án này được thiết kế để học tập các công nghệ OCR (Optical Character Recognition), thử nghiệm (testing) các thư viện core và xây dựng từng thành phần của một hệ thống OCR hoàn chỉnh từ cơ bản đến nâng cao.

## 📂 Cấu trúc thư mục dự án

```text
DocU/
├── venv/                      # Môi trường ảo Python (đã tạo)
├── configs/                   # Các file cấu hình mô hình (YAML, JSON)
├── data/                      # Dữ liệu thử nghiệm
│   ├── raw/                   # Ảnh đầu vào chưa xử lý (scanned docs, v.v.)
│   └── processed/             # Ảnh đã qua xử lý (binarized, cropped, v.v.)
├── core/                      # Các thành phần cốt lõi của hệ thống OCR
│   ├── __init__.py
│   ├── preprocessing/         # Tiền xử lý ảnh (Binarization, Deskewing, Noise Reduction)
│   │   └── __init__.py
│   ├── detection/             # Phát hiện vùng chứa văn bản (Text Detection - e.g., DBNet, EAST, OpenCV)
│   │   └── __init__.py
│   ├── recognition/           # Nhận diện ký tự/văn bản (Text Recognition - e.g., CRNN, Tesseract, EasyOCR)
│   │   └── __init__.py
│   ├── layout/                # Phân tích bố cục tài liệu (Layout Analysis & Table Extraction)
│   │   └── __init__.py
│   └── postprocessing/        # Hậu xử lý (Spell correction, NLP, Structuring sang Markdown/JSON)
│       └── __init__.py
├── notebooks/                 # Jupyter Notebooks để nghiên cứu, học tập từng phần công nghệ
├── scripts/                   # Các script chạy pipeline, CLI tool thử nghiệm nhanh
└── tests/                     # Unit tests cho từng module trong core/
```

## 🛠️ Lộ trình Học tập & Phát triển OCR gợi ý

1. **Bước 1: Image Preprocessing (Tiền xử lý ảnh)**
   - Tìm hiểu cách chuẩn hóa ảnh bằng OpenCV: Binarization (Otsu, Adaptive Thresholding), Dilation/Erosion, Deskewing (xoay thẳng ảnh nghiêng).

2. **Bước 2: Text Detection (Phát hiện văn bản)**
   - Học cách khoanh vùng văn bản bằng các thuật toán xử lý ảnh truyền thống (Contours, MSER) hoặc mô hình Deep Learning (EAST, DBNet).

3. **Bước 3: Text Recognition (Nhận diện ký tự)**
   - Trích xuất chữ từ vùng ảnh đã cắt bằng Tesseract, EasyOCR, hoặc huấn luyện mô hình CRNN (CNN + RNN + CTC loss).

4. **Bước 4: Layout Analysis & Structuring (Phân tích bố cục & Cấu trúc)**
   - Nhận diện tiêu đề (Header), đoạn văn (Paragraph), bảng biểu (Table Extraction) để cấu trúc hóa tài liệu đầu ra.

5. **Bước 5: Post-processing (Hậu xử lý)**
   - Sửa lỗi chính tả sử dụng SymSpell, LanguageTool hoặc tích hợp mô hình ngôn ngữ (LLM) để làm sạch văn bản trích xuất.
