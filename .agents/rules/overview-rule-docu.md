---
trigger: always_on
description: Mandatory web search requirement for technology recommendations, framework trends, strict environment execution, uv package manager preference, and requirements sync.
---

# MANDATORY WORKSPACE RULES FOR TECH STACK, ENVIRONMENT EXECUTION & REQUIREMENTS

> [!IMPORTANT]
> **CÁC QUY TẮC THỰC THI BẮT BUỘC TRONG WORKSPACE DỰ ÁN**

---

## 1. Quy Tắc Bắt Buộc Web Search Cho Tư Vấn & Đề Xuất Kỹ Thuật

1. **Luôn Gọi Tool `search_web` Trước Khi Trả Lời**:
   - Khi người dùng đưa ra các câu hỏi liên quan đến:
     - **Đề xuất / Đánh giá Tech Stack**: Chọn lựa model, framework, thư viện, công cụ AI/OCR/Computer Vision mới nhất.
     - **Xu hướng Thiết kế & Kiến trúc**: Xu hướng thiết kế pipeline, kiến trúc hệ thống, giải pháp SOTA (State-of-the-Art) mới nhất.
     - **Tìm hiểu Công nghệ & So sánh Benchmark**: So sánh tính năng, hiệu năng, benchmark giữa các model/tool (ví dụ: các phiên bản YOLO mới nhất, VietOCR, PP-OCR, TrOCR, SAM...).
     - **Giải pháp cho vấn đề mới**: Các thư viện, thuật toán hoặc công cụ mới nhất hỗ trợ giải quyết bài toán của người dùng.
   - Agent **KHÔNG ĐƯỢC PHÉP** đưa ra câu trả lời tư vấn chủ quan ngay lập tức dựa trên kiến thức có sẵn mà **BẮT BUỘC KHỞI TẠO TOOL `search_web` TRƯỚC** để tìm hiểu và cập nhật thông tin mới nhất trên internet.

2. **Yêu Cầu Chất Lượng Thông Tin**:
   - Mọi đề xuất, tư vấn kỹ thuật hay so sánh đều phải dựa trên dữ liệu tìm kiếm thực tế up-to-date.
   - Phân biệt rõ ràng giữa các phiên bản phát hành chính thức (Official Releases) và các bài báo nghiên cứu cộng đồng (Community Papers).

---

## 2. Quy Tắc Môi Trường Python (`venv`), Công Cụ `uv` & Quản Lý Phụ Thuộc (`requirements.txt`)

1. **Chỉ Thực Thi Code & Quản Lý Package Trên Môi Trường `venv`**:
   - Mọi thao tác thực thi code Python, chạy script, kiểm thử, hoặc quản lý gói (thêm mới `install`, cập nhật `update`, gỡ bỏ `uninstall` package/library) **CHỈ ĐƯỢC PHÉP THỰC THI QUA MÔI TRƯỜNG `venv`** (đường dẫn: `.\venv\Scripts\python.exe` hoặc `venv\Scripts\pip`).
   - Nghiêm cấm chạy code hoặc cài đặt thư viện vào Python hệ thống toàn cục (Global Python) ngoài `venv`.

2. **Ưu Tiên Tải & Cài Đặt Package Bằng `uv` Để Đạt Tốc Độ Tối Đa**:
   - Bất kỳ khi nào thực hiện cài đặt hoặc cập nhật package/thư viện Python, Agent **BẮT BUỘC ƯU TIÊN SỬ DỤNG LỆNH `uv`** (ví dụ: `uv pip install --python .\venv\Scripts\python.exe <package>` hoặc `uv pip install ...` chỉ định môi trường `venv`) để tối ưu hóa tốc độ tải và cài đặt ở mức cao nhất.

3. **Cập Nhật Tự Động `requirements.txt` Khi Thay Đổi Thư Viện**:
   - Bất kỳ khi nào thực hiện cài đặt (`install`) hoặc cập nhật (`update`) bất kỳ package/thư viện Python nào vào môi trường `venv`, Agent **BẮT BUỘC PHẢI BỔ SUNG/CẬP NHẬT NGAY THÔNG TIN THƯ VIỆN ĐÓ VÀO FILE `requirements.txt`** của dự án để đảm bảo tính đồng bộ môi trường.
