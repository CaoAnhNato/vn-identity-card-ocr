# HỆ THỐNG OCR GIẤY TỜ TÙY THÂN: THIẾT KẾ KIẾN TRÚC & WORKFLOW PIPELINE
*(OCR PIPELINE ARCHITECTURE & WORKFLOW SPECIFICATION)*

---

## 1. TỔNG QUAN VỀ HỆ THỐNG & WORKFLOW PIPELINE

### 1.1. Mục Tiêu Hệ Thống
Hệ thống **OCR Giấy tờ Tùy thân (Identity Document OCR System)** được xây dựng để xử lý tự động, phát hiện, phân đoạn và trích xuất dữ liệu từ các loại giấy tờ tùy thân (Căn cước công dân - CCCD, Hộ chiếu - Passport, Giấy phép lái xe - GPLX).

Đặc thù cốt lõi của hệ thống là xử lý **Đa giấy tờ trong một ảnh (Multi-document / Multi-card processing)**. Khi ảnh nhận vào chứa 2 hoặc nhiều giấy tờ (ví dụ: chụp cả mặt trước CCCD và Passport cùng lúc), hệ thống tự động bóc tách từng giấy tờ độc lập, nắn phẳng góc nhìn (Perspective Transform), nhận diện văn bản tiếng Việt và bóc tách dữ liệu chuẩn đầu ra dạng **JSON**.

---

### 1.2. Biểu Đồ Trình Tự Workflow Tổng Quan (Sequence Diagram)

```mermaid
sequenceDiagram
    autonumber
    actor Client as Client App
    participant Pipe as Pipeline Container
    participant S1 as Thành phần 1: Pre-processing
    participant S2 as Thành phần 2: Multi-Doc Segment
    participant S3 as Thành phần 3: Text Det & Rec (OCR)
    participant S4 as Thành phần 4: Key Info Extraction (KIE)
    participant S5 as Thành phần 5: JSON Output

    Client->>Pipe: Execute Request (Raw Image Bytes)
    Note over Pipe: Khởi tạo TraceID & PipelineContextDTO
    
    Pipe->>S1: Pre-processing (ImageInputDTO)
    S1-->>Pipe: Preprocessed ImageInputDTO

    Pipe->>S2: Multi-Document Instance Segmentation
    Note over S2: YOLOv8-Seg Mask + Perspective Transform (Nắn phẳng)
    S2-->>Pipe: List[DocumentInstanceDTO] (Multi-card)

    loop Xử lý từng giấy tờ tùy thân cắt ra (Document Instance)
        Pipe->>S3: Text Detection & Recognition (OCR)
        Note over S3: PP-OCRv4 Det (HGNetV2) + VietOCR Recognition
        S3-->>Pipe: Updated DocumentInstanceDTO (Raw OCR Texts)

        Pipe->>S4: Key Information Extraction (KIE)
        Note over S4: Spatial Alignment + Checksum & Schema Validation
        S4-->>Pipe: Updated DocumentInstanceDTO (Extracted Fields)
    end

    Pipe->>S5: JSON Output Formatting
    Note over S5: Assembly Final Result & Latency Profiling
    S5-->>Client: Structured JSON Response
```

---

## 2. QUY TẮC THIẾT KẾ & TIÊU CHUẨN KIẾN TRÚC (DESIGN PRINCIPLES)

Toàn bộ hệ thống phải tuân thủ nghiêm ngặt 6 tiêu chuẩn kiến trúc cốt lõi dưới đây:

```mermaid
flowchart TD
    subgraph Core Architecture
        PF["<b>1. Pipe-and-Filter Architecture</b><br/>Quản lý luồng xử lý dạng chuỗi mô-đun hóa"]
    end

    subgraph Data Contracts & Design Patterns
        DTO["<b>2. Filter Contract (Dataclass DTO)</b><br/>Hợp đồng dữ liệu nghiêm ngặt giữa các Filter"]
        SP["<b>3. Algorithm Selection (Strategy Pattern)</b><br/>Hoán đổi linh hoạt các thuật toán AI"]
        DI["<b>4. Dependency Management (Dependency Injection)</b><br/>Tiêm phụ thuộc, tăng khả năng kiểm thử"]
    end

    subgraph Operations & Controls
        YAML["<b>5. Configuration (YAML)</b><br/>Cấu hình siêu tham số tập trung"]
        OBS["<b>6. Observability</b><br/>Profiling + JSON Logging + Distributed Tracing"]
    end

    PF --> DTO
    PF --> SP
    PF --> DI
    PF --> YAML
    PF --> OBS

    classDef primary fill:#2b5c8f,stroke:#1d3d61,color:#fff,font-weight:bold;
    classDef secondary fill:#388e3c,stroke:#1b5e20,color:#fff;
    classDef ops fill:#f57c00,stroke:#e65100,color:#fff;

    class PF primary;
    class DTO,SP,DI secondary;
    class YAML,OBS ops;
```

---

### 2.1. Pipe-and-Filter Architecture
- Chuỗi xử lý được chia nhỏ thành các **Filter** hoàn toàn độc lập, nhận dữ liệu đầu vào qua Pipe, thực hiện tính toán và truyền sang Filter tiếp theo.
- Giúp dễ dàng mở rộng, thay đổi thứ tự filter, hoặc thêm các filter mới mà không làm phá vỡ logic hiện tại.

---

### 2.2. Filter Contract (Dataclass DTO)
Mọi dữ liệu trao đổi giữa các Filter bắt buộc tuân thủ hợp đồng dạng Python `@dataclass` DTO (Data Transfer Object) để đảm bảo an toàn kiểu dữ liệu (Type Safety) và chống trôi kiểu (Type Drift).

```mermaid
classDiagram
    class PipelineContextDTO {
        +str trace_id
        +str span_id
        +datetime start_time
        +Dict execution_metrics
        +record_filter_metric(filter_name, duration_ms, memory_mb)
    }

    class ImageInputDTO {
        +np.ndarray raw_image
        +int width
        +int height
        +int channels
        +str file_format
    }

    class BoundingBoxDTO {
        +List~Point2D~ coordinates
        +float score
    }

    class RecognizedTextDTO {
        +BoundingBoxDTO bbox
        +str text
        +float confidence
    }

    class FieldValueDTO {
        +str field_key
        +str raw_value
        +str normalized_value
        +float confidence
        +BoundingBoxDTO bbox
        +bool is_valid
    }

    class DocumentInstanceDTO {
        +str doc_id
        +str doc_type
        +np.ndarray cropped_image
        +BoundingBoxDTO polygon
        +float confidence
        +List~RecognizedTextDTO~ recognized_texts
        +Dict~str, FieldValueDTO~ extracted_fields
    }

    class OCRResultDTO {
        +str trace_id
        +bool success
        +List~DocumentInstanceDTO~ documents
        +PipelineContextDTO context
        +str error_message
    }

    OCRResultDTO o-- DocumentInstanceDTO
    OCRResultDTO o-- PipelineContextDTO
    DocumentInstanceDTO o-- BoundingBoxDTO
    DocumentInstanceDTO o-- RecognizedTextDTO
    DocumentInstanceDTO o-- FieldValueDTO
```

---

### 2.3. Algorithm Selection (Strategy Pattern)
Tất cả các thuật toán AI (Segmentation, Text Detection, Text Recognition, Information Extraction) phải được đóng gói đằng sau các giao diện (Interfaces) chuẩn, cho phép hoán đổi thuật toán linh hoạt thông qua cấu hình.

```mermaid
classDiagram
    class ISegmentationStrategy {
        <<interface>>
        +segment_multi_documents(image: np.ndarray) List~DocumentInstanceDTO~
    }

    class ITextDetectionStrategy {
        <<interface>>
        +detect_text_regions(crop_image: np.ndarray) List~BoundingBoxDTO~
    }

    class ITextRecognitionStrategy {
        <<interface>>
        +recognize_text(crop_image: np.ndarray, bboxes: List~BoundingBoxDTO~) List~RecognizedTextDTO~
    }

    class IFieldExtractionStrategy {
        <<interface>>
        +extract_fields(doc_type: str, ocr_texts: List~RecognizedTextDTO~) Dict~str, FieldValueDTO~
    }

    class YOLOv8SegmentationStrategy {
        +segment_multi_documents(image: np.ndarray) List~DocumentInstanceDTO~
    }
    class PPOCRv4DetStrategy {
        +detect_text_regions(crop_image: np.ndarray) List~BoundingBoxDTO~
    }
    class VietOCRRecognitionStrategy {
        +recognize_text(crop_image: np.ndarray, bboxes: List~BoundingBoxDTO~) List~RecognizedTextDTO~
    }
    class LayoutRegexExtractionStrategy {
        +extract_fields(doc_type: str, ocr_texts: List~RecognizedTextDTO~) Dict~str, FieldValueDTO~
    }

    ISegmentationStrategy <|.. YOLOv8SegmentationStrategy
    ITextDetectionStrategy <|.. PPOCRv4DetStrategy
    ITextRecognitionStrategy <|.. VietOCRRecognitionStrategy
    IFieldExtractionStrategy <|.. LayoutRegexExtractionStrategy
```

---

### 2.4. Dependency Management (Dependency Injection)
- Sử dụng nguyên lý Inversion of Control (IoC). Tất cả các Strategy, Logger, Profiler được "tiêm" (Inject) vào Constructor của các Filter và Container.
- Tối ưu hóa cho Unit Testing và Integration Testing bằng cách dễ dàng Mocking các mô hình AI đắt đỏ.

---

### 2.5. Configuration (YAML)
Tập trung toàn bộ siêu tham số (Hyperparameters), đường dẫn weights, confidence thresholds, hardware execution provider (`cuda:0` / `cpu`) vào file `pipeline_config.yaml`:

```yaml
pipeline:
  name: "MultiDoc_OCR_Pipeline"
  version: "2.1.0"
  device: "cuda:0"

tracing:
  enabled: true
  service_name: "ocr-pipeline-service"

filters:
  segmentation:
    strategy: "YOLOv8SegStrategy"
    confidence_threshold: 0.65
  text_detection:
    strategy: "PPOCRv4DetStrategy"   # PP-OCRv4 Det (Backbone PPHGNetV2 / PPLCNetV3)
    backbone: "PPHGNetV2"
  text_recognition:
    strategy: "VietOCRRecognitionStrategy"
  information_extraction:
    strategy: "LayoutRegexExtractionStrategy"
```

---

### 2.6. Observability (Profiling + Logging + Tracing)
- **Profiling**: Đo đạc độ trễ thời gian thực thi (Latency Profiler tính bằng ms) và mức tiêu thụ tài nguyên (RAM/GPU Memory) của từng Filter.
- **Logging**: Ghi log cấu trúc dạng JSON Lines (Loguru / structlog) phục vụ phân tích log tập trung (ELK Stack / Loki).
- **Tracing**: Gán `TraceID` (UUID v4) cho mỗi request và `SpanID` cho từng Filter để truy vết toàn bộ luồng xử lý.

---

## 3. CHI TIẾT 5 THÀNH PHẦN PIPELINE (PIPELINE STAGES DOCS LINKING)

Workflow của Pipeline được chia thành **5 Thành phần chính**. Bấm vào các liên kết bên dưới để tới tài liệu chi tiết của từng thành phần:

```mermaid
graph LR
    MAIN["<b>Main Architecture</b><br/>(File Hiện Tại)"]

    MAIN --> STAGE1["<b>Thành phần 1: Pre-processing</b><br/>stage1_preprocessing.md"]
    MAIN --> STAGE2["<b>Thành phần 2: Multi-Doc Segment</b><br/>stage2_multi_doc_segmentation.md"]
    MAIN --> STAGE3["<b>Thành phần 3: Text OCR</b><br/>stage3_text_ocr.md"]
    MAIN --> STAGE4["<b>Thành phần 4: KIE Extraction</b><br/>stage4_kie_extraction.md"]
    MAIN --> STAGE5["<b>Thành phần 5: JSON Output</b><br/>stage5_json_output.md"]

    click STAGE1 "./stage1_preprocessing.md" "Xem chi tiết Thành phần 1"
    click STAGE2 "./stage2_multi_doc_segmentation.md" "Xem chi tiết Thành phần 2"
    click STAGE3 "./stage3_text_ocr.md" "Xem chi tiết Thành phần 3"
    click STAGE4 "./stage4_kie_extraction.md" "Xem chi tiết Thành phần 4"
    click STAGE5 "./stage5_json_output.md" "Xem chi tiết Thành phần 5"
```

### Bảng Danh Sách Tài Liệu 5 Thành Phần Pipeline:

| STT | Thành phần Pipeline | Tài Liệu Chi Tiết (Click để xem) | Mô Tả Nội Dung |
|:---:|:--- |:--- |:--- |
| 1 | **Thành phần 1** | [🛠️ Pre-processing (Tiền xử lý)](./stage1_preprocessing.md) | Gán TraceID, kiểm tra định dạng ảnh, xoay ảnh tự động, khử nhiễu & CLAHE |
| 2 | **Thành phần 2** | [✂️ Multi-Document Instance Segmentation](./stage2_multi_doc_segmentation.md) | YOLOv8-Seg mask, xác định 4 góc polygon, nắn phẳng góc nhìn (Perspective Transform) |
| 3 | **Thành phần 3** | [🔍 Text Detection & Recognition - OCR](./stage3_text_ocr.md) | PP-OCRv4 Det (HGNetV2/LCNet) & VietOCR nhận diện chữ trên CCCD/Passport |
| 4 | **Thành phần 4** | [🧠 Key Information Extraction - KIE](./stage4_kie_extraction.md) | Spatial layout alignment, Regex Checksum CCCD/Passport & chuẩn hóa địa danh |
| 5 | **Thành phần 5** | [📦 JSON Output Formatting](./stage5_json_output.md) | Xuất dữ liệu cấu trúc JSON tiêu chuẩn kèm thông số Profiling & Trace ID |
