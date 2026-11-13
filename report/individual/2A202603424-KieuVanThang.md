# Báo cáo cá nhân — Day 6: Data Ingestion Engineering


## 1. Dữ liệu và pipeline

- **Chủ đề/corpus:** *The Notebooks of Leonardo Da Vinci*, bản tiếng Anh do Jean Paul Richter biên tập/dịch, Project Gutenberg ebook #5000. Dữ liệu EPUB được tách theo thứ tự spine thành **30 đơn vị nội dung phi phân trang**, mỗi đơn vị tối thiểu 23.415 ký tự. `gold_page`/`logical_page` là số thứ tự đơn vị nội bộ, không phải số trang in. Bìa và phần license không được đưa vào corpus truy xuất.
- **Nguồn/quyền:** https://www.gutenberg.org/5000. Metadata EPUB nêu public domain tại Hoa Kỳ và tệp có kèm license Project Gutenberg; cần tự xác minh quyền áp dụng tại khu vực sử dụng và điều kiện tái phân phối. Manifest ghi ngày kiểm tra EPUB cục bộ `2026-11-13`; ngày tải ban đầu không được cung cấp.
- **Chuẩn hóa/giới hạn:** trích xuất văn bản EPUB theo spine; chuẩn hóa Unicode NFC, gộp whitespace trong đoạn và nối đoạn bằng hai newline. Giữ source, EPUB item, unit, paragraph, logical page và offset ký tự để truy vết. Đây là văn bản điện tử, không chạy OCR. Nội dung lịch sử được xem là điều nguồn nói, không xác minh như sự thật độc lập.
- **Quy mô:** 30 units; 1.373.232 ký tự và 325.198 content tokens theo tokenizer ONNX của Chroma (không tính special tokens).
- **Embedding/vector store:** Python 3.14.10; Chroma 1.5.9 PersistentClient; `sentence-transformers/all-MiniLM-L6-v2` qua Chroma ONNX, 384 chiều; cosine distance; Chroma HNSW approximate nearest-neighbor. Database lưu tại [data/chroma/](../../starter/data/chroma/). Embedding chạy local, chi phí dịch vụ embedding là **$0**; lần đầu cần Internet để tải model/packages.
- **Metadata:** `document_id`, `source`, `page` (logical page), `unit_id`, `volume`, cùng offset/paragraph và EPUB item. `volume` lấy từ nhãn volume trong EPUB/spine, dùng làm filter theo ngữ cảnh đã có trước truy xuất; không dùng gold answer hay gold source làm filter.
- **Lệnh chạy** (PowerShell tại [starter/](../../starter/)):
  ```powershell
  python -m src.run
  python -m src.evaluate
  python -m unittest discover -s tests -v
  ```

## 2. Cấu hình và benchmark

- **A:** fixed-size 500 ký tự, overlap 50, không filter.
- **B:** chunk theo đoạn, tối đa 550 ký tự, overlap 0, không filter.
- **C:** cùng chunks/embeddings như B; filter `volume` lúc truy vấn khi câu hỏi cung cấp volume. Khác biệt B–C chỉ ở query-time filter.
- Benchmark gồm **20 câu tiếng Anh**: 15 dev / 5 test; 6 câu có filter, trong đó 2 câu thuộc test. Evidence được đối chiếu với corpus trước thí nghiệm. Dev/test đến từ các EPUB spine section tách biệt, nhưng vẫn cùng một cuốn sách nên có nguy cơ rò rỉ chủ đề.
- Quy tắc chọn đã ghi trước khi so sánh: ưu tiên dev Evidence Hit@3, tiếp đến Hit@1, grounding/metadata, rồi tài nguyên. Cả ba cấu hình được chạy trên cả hai split; không thay tham số sau khi xem test. B và C hòa nhau trên dev; chọn C cho workload cần lọc volume vì không giảm điểm dev và vẫn tuân thủ filter. Đây là lựa chọn cho corpus này, không phải kết luận tổng quát.

## 3. Kết quả

Hit@1/Hit@3 dưới đây là **Evidence Hit** nghiêm ngặt: phải khớp gold unit/source và chứa evidence quote sau chuẩn hóa.

| Strategy | Split | Evidence Hit@1 | Evidence Hit@3 | Số chunks | Index time | Chi phí embedding |
|---|---|---:|---:|---:|---:|---:|
| A | Dev | 8/15 (53,3%) | 9/15 (60,0%) | 3.064 | 189,63 s | $0 |
| A | Test | 3/5 (60,0%) | 4/5 (80,0%) | 3.064 | 189,63 s | $0 |
| B | Dev | 7/15 (46,7%) | 12/15 (80,0%) | 3.121 | 198,20 s | $0 |
| B | Test | 5/5 (100%) | 5/5 (100%) | 3.121 | 198,20 s | $0 |
| C | Dev | 7/15 (46,7%) | 12/15 (80,0%) | 3.121 | 5,02 s* | $0 |
| C | Test | 5/5 (100%) | 5/5 (100%) | 3.121 | 5,02 s* | $0 |

`*` C tái sử dụng chunks và embeddings của B; 5,02 giây là thời gian lưu/index bổ sung, không so sánh trực tiếp với thời gian index đầy đủ A/B. Thời gian phụ thuộc máy và chỉ là một lần chạy.

| Strategy | Coherence trung bình /2 (n=10) | Điểm coherence (0/1/2) | Metadata đúng và hữu ích | Filter tuân thủ | Grounding test (0/1/2) | Grounding trung bình /2 |
|---|---:|---|---|---|---|---:|
| A | 1,50 | 0/5/5 | 15/15 đúng; 15/15 hữu ích | N/A | 0/1/4 | 1,80 |
| B | 1,60 | 0/4/6 | 15/15 đúng; 15/15 hữu ích | N/A | 0/0/5 | 2,00 |
| C | 1,60 | 0/4/6 | 15/15 đúng; 15/15 hữu ích | 2/2 câu test; 6/6 bản ghi test | 0/0/5 | 2,00 |

Metadata được kiểm tra trên toàn bộ **45/45** bản ghi top-3 test; kiểm tra tự động khớp corpus và nhãn audit đều ghi đúng, hữu ích 45/45. C tuân thủ filter trên toàn bộ **6/6** câu có filter (dev + test), gồm 2/2 câu test. Trường `unit_id`/`page` và source giúp truy vết; `volume` hữu ích để giới hạn truy vấn nhưng còn rộng trong một cuốn sách.

- **Trace và audit:** [retrieval_traces.jsonl](../../starter/results/retrieval_traces.jsonl), [coherence_audit.csv](../../starter/results/coherence_audit.csv), [metadata_audit.csv](../../starter/results/metadata_audit.csv), [grounding_audit.csv](../../starter/results/grounding_audit.csv); quy tắc chấm tại [HUMAN_AUDIT_GUIDE.md](../../starter/results/HUMAN_AUDIT_GUIDE.md).
- **Người đánh giá/provenance:** học viên cho biết đã kiểm tra và xác nhận các nhãn trong CSV. Điểm/lý do ban đầu được AI hỗ trợ dự thảo; nhãn được giữ provenance này và không được xem là đánh giá độc lập/blind. Nguy cơ còn lại là thiên lệch xác nhận khi rà soát bản nháp AI.

## 4. Phân tích và kết luận

- **A–B:** trên dev, B tăng Evidence Hit@3 từ 9/15 lên 12/15 (+20 điểm phần trăm), nhưng Hit@1 giảm nhẹ từ 8/15 xuống 7/15. Trên test, B đạt 5/5 ở cả hai chỉ số so với A 3/5 và 4/5. Chunk theo đoạn tăng 57 chunks nhưng bảo toàn ranh giới đoạn tốt hơn; coherence tăng từ 1,50 lên 1,60.
- **B–C:** cùng chunks/embeddings và cùng điểm retrieval; C không chứng minh tăng độ chính xác trong benchmark này. Điểm cộng là lọc đúng volume ở 6/6 câu có filter; đổi lại có thêm chi phí truy vấn/filter. Chênh lệch thời gian index của C không phản ánh full re-embedding vì dùng lại B.
- **Ví dụ lỗi:** (1) `q02` (dev), cả A/B/C đều 0/3 Evidence Hit@3: kết quả nói về ánh sáng, bóng và màu trong hội họa nhưng bỏ lỡ bằng chứng cụ thể rằng dùng màu tối khiến hình kém nổi và khó thấy từ xa. (2) `q13` (dev), cả ba đều 0/3: top-3 liên quan đến cái chết/sự sống nhưng không lấy được câu về chuyển động của vật chết khiến sinh vật sống bỏ chạy trong đau đớn, than khóc. (3) `q20` (test), A không tìm thấy evidence trong top-3 do fixed window tách/không đưa câu ghi 21 braccia vào các hit; B/C lấy đúng câu đó ở hạng 1.
- **Đánh đổi/lựa chọn:** chọn C nếu truy vấn thực tế có filter volume; B là baseline đơn giản hơn nếu không cần filter. B/C hòa trên dev nên lựa chọn dựa trên khả năng lọc, không phải điểm retrieval cao hơn. Dữ liệu nhỏ không đủ để kết luận C tốt hơn về chất lượng chung.
- **Hạn chế:** chỉ một cuốn sách; split khác spine section nhưng không độc lập về chủ đề; test chỉ có 5 câu. Exact-quote metric có thể bỏ sót evidence tương đương; audit có AI hỗ trợ và chỉ một học viên rà soát, không phải đánh giá viên độc lập. Quyền tái sử dụng ngoài Hoa Kỳ chưa được xác minh. Thời gian một lần chạy và chi phí $0 chỉ phản ánh embedding local trên môi trường này.
- **Nếu mở rộng:** thêm nguồn độc lập có metadata/filter hữu ích, tạo test ngoài corpus/section và tăng số câu; có người chấm độc lập cho audit; bổ sung kiểm tra trích xuất văn bản ở các ký tự/định dạng phức tạp và chạy lại benchmark sau khi cố định cấu hình.
