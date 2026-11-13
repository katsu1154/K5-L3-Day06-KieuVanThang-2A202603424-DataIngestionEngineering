# Starter · Chroma local (không cần Docker)

Starter dùng Chroma để lưu vector bền vững ngay trên máy, không cần Docker hay database server. Embedding mặc định chạy local bằng ONNX với model `sentence-transformers/all-MiniLM-L6-v2`; không cần API key. Lần chạy embedding đầu tiên cần Internet để tải model và các dependencies.

Pipeline ingestion và evaluation đã được hiện thực cho `data/corpus_units.jsonl` và benchmark 20 câu. Các trace, metrics và audit CSV đã được tạo trong `results/`; nhãn audit cần được học viên rà soát và xác nhận.

## Cấu trúc

```text
starter/
├── .env.example           # cấu hình Chroma local và embedding
├── requirements.txt       # dependencies
├── scripts/check_setup.py # kiểm tra embedding thật, lưu bền và filter
├── src/
│   ├── models.py          # cấu trúc dữ liệu mẫu
│   ├── ingestion.py       # đọc / chuẩn hóa tài liệu
│   ├── chunking.py        # fixed-size và paragraph chunking
│   ├── embeddings.py      # embedding ONNX local đã cấu hình
│   ├── store.py           # Chroma persistent, cosine search và metadata filter
│   ├── evaluation.py      # chấm truy xuất, tổng hợp metrics
│   ├── evaluate.py        # chạy benchmark và lưu trace/audit
│   └── run.py             # chạy ingestion A/B/C
├── data/
│   ├── corpus_units.jsonl # corpus đã chuẩn bị
│   ├── manifest.csv       # manifest corpus
│   └── chroma/            # database cục bộ, tự sinh và không commit
└── results/               # lưu kết quả benchmark
```

## 1. Cài đặt trên Windows

Cần Python **3.11–3.14** và Internet cho lần cài package/tải embedding model đầu tiên. Chạy PowerShell từ thư mục `starter/`:

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
Copy-Item .env.example .env
python -m scripts.check_setup
```

Nếu máy có Python khác, thay `3.14` bằng phiên bản đã cài. Nếu PowerShell chặn kích hoạt môi trường ảo, có thể gọi trực tiếp `.\.venv\Scripts\python.exe` thay cho `python`.

Lần đầu `check_setup` có thể mất thời gian tải model ONNX. Kết quả thành công sẽ in tên model, số chiều, đường dẫn Chroma và `Setup OK`. Script tạo một collection kiểm tra tạm rồi xóa collection đó; database local vẫn được khởi tạo. Lỗi tải model, thiếu dung lượng hoặc lỗi package được hiển thị trực tiếp, không có fallback sang vector giả.

`CHROMA_PERSIST_DIRECTORY` mặc định là `./data/chroma`, tính tương đối từ thư mục `starter/`. Thư mục này đã được thêm vào `.gitignore`; không commit database phát sinh. Không cần đặt `DATABASE_URL`, cài Docker Desktop hay nhập API key để chạy cấu hình mặc định.

## 2. Cấu hình embedding và vector store

`.env.example` chọn:

- Provider: `chroma-local`.
- Model: `sentence-transformers/all-MiniLM-L6-v2`, embedding 384 chiều qua Chroma ONNX.
- Vector store: Chroma persistent tại `CHROMA_PERSIST_DIRECTORY`.
- Metric: cosine.

`ChromaVectorStore` nhận vector do embedder tạo, tách collection theo chiến lược (ví dụ `day6_A`, `day6_B`, `day6_C`) và cho phép lọc các metadata scalar như `volume`, `section`, `source` hoặc `page`. Dùng cùng embedding model và metric cho cả ba cấu hình. Giá trị filter phải lấy từ nguồn và ngữ cảnh cung cấp trước lúc truy xuất.

Muốn đổi đường dẫn lưu, sửa `CHROMA_PERSIST_DIRECTORY` trong `.env`. Adapter hiện tại chỉ hỗ trợ model đã nêu; dùng model khác cần hiện thực adapter riêng, ghi tên/phiên bản và chiều vector, đồng thời dùng đường dẫn Chroma riêng để không trộn vector giữa các model. Không xóa database để xử lý lỗi nếu chưa sao lưu hoặc chưa xác định rõ dữ liệu đó có cần giữ lại.

## 3. Hoàn thiện pipeline lab

1. Dùng `data/corpus_units.jsonl` và `data/manifest.csv`; kiểm tra lại quyền sử dụng dữ liệu và ID đơn vị nguồn.
2. Chạy `python -m src.run` để nạp và chuẩn hóa corpus, tạo chunk, embedding, lưu A/B/C và kiểm tra nạp lại không nhân đôi. A dùng cửa sổ 500 ký tự / overlap 50; B dùng đoạn tối đa 550 ký tự; C dùng cùng chunk/vector với B và chỉ khác ở filter lúc truy vấn.
3. Xem `results/step3_ingestion_stats.json` để kiểm tra số đơn vị/chunk, ký tự/token, thời gian, cấu hình model/store, và số bản ghi trước/sau khi nạp lại. Cấu hình và thống kê được ghi vào `artifacts/experiment_log.csv`.
4. Chạy `python -m src.evaluate` để lấy top-3 cho cùng 20 câu trên A/B/C, áp dụng filter đầu vào chỉ ở C, lưu trace và tổng hợp Evidence Hit@1/Hit@3 theo dev/test. Chỉ số tự động yêu cầu trùng evidence quote sau chuẩn hóa; nó có thể bỏ sót evidence tương đương và không thay cho grounding thủ công.
5. Rà soát/xác nhận nhãn trong `results/coherence_audit.csv`, `results/metadata_audit.csv` và `results/grounding_audit.csv` theo [HUMAN_AUDIT_GUIDE.md](results/HUMAN_AUDIT_GUIDE.md) và [GUIDE.md](../GUIDE.md). Các nhãn hiện có là bản nháp có AI hỗ trợ; báo cáo ghi rõ học viên xác nhận và giữ provenance AI.
6. Xem [báo cáo cá nhân](../report/individual/MSSV_HoTen.md), điền thông tin còn thiếu và chọn chiến lược theo dev; không tinh chỉnh trên test. Bộ test có năm câu và đến từ cùng một cuốn sách nên chỉ là minh họa, không chứng minh tính tổng quát.

Chạy bước 3 và bước 4 từ thư mục `starter/`:

```powershell
python -m src.run
python -m src.evaluate
```

`src.run` chạy ingestion/indexing; `src.evaluate` chạy benchmark retrieval. Các kết quả nằm trong `results/`; các chỉ số dev/test cũng được ghi vào `artifacts/experiment_log.csv`. Dùng `python -m scripts.check_setup` để chỉ kiểm tra riêng phần embedding và Chroma.
