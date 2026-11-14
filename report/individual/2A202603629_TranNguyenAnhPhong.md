# Báo cáo cá nhân — Day 7: Retrieval & Context Engineering

**Họ và tên:** Trần Nguyễn Anh Phong  
**MSSV:** 2A202603629  
**Bài thực hành:** K5-L3A — Day 07, Retrieval & Context Engineering  
**Ngày thực nghiệm:** 14/11/2026  
**Repository:** [LAB7 — Retrieval & Context Engineering](https://github.com/anhphongtrng/K5-L3A-Day07-TranNguyenAnhPhong-2A202603629-Retrieval-Context-Engineering)  
**Phạm vi:** Phần cơ bản, **không thực hiện Thử thách nâng cao**.

## 1. Dữ liệu và benchmark

### 1.1. Nguồn, phạm vi và dữ liệu tái sử dụng

- **Corpus:** *Calculus Volume 1* — OpenStax, trích **50 trang PDF vật lý (trang 51–100)**; chủ đề Functions and Graphs, các section **1.2, 1.3, 1.4 và 1.5**. Nguồn tham khảo: https://openstax.org/details/books/calculus-volume-1. Tài liệu phục vụ thực nghiệm học tập, ghi nhận quyền tác giả OpenStax/Rice University; cần kiểm tra điều kiện tái sử dụng đi kèm **đúng phiên bản PDF đã tải** trước khi tái phân phối nội dung. Bản phân phối hiện hành của OpenStax nêu điều kiện **CC BY-NC-SA 4.0** và quy định về việc sử dụng trong sản phẩm AI; báo cáo không khẳng định có quyền sử dụng rộng hơn quy định của nguồn.
- **Day 6:** Đã ingest **333 chunks** gồm **Strategy A = 171**, **Strategy B = 162**. LAB7 sử dụng **cùng 162 chunks Strategy B** trong cả ba thuật toán để so sánh công bằng.
- **Kho dữ liệu:** PostgreSQL + extension `pgvector`, database `day6_lab`, bảng `public.lab_document_chunks`; lọc `strategy = 'B' AND embedding IS NOT NULL`.
- **Embedding:** `BAAI/bge-m3` qua `sentence-transformers`, **1024 chiều**, cosine similarity (`<=>` trong pgvector); query embedding chuẩn hóa với `normalize_embeddings=True`. **Model revision/hash và phiên bản package chưa được cố định trong trace**, cần ghi rõ đây là hạn chế tái lập.
- **Các trường chunk:** `chunk_id`, `document_id`, `source`, `page`, `content`, `metadata` (trong đó có `section`), `embedding`.
- **Liên kết Day 6:** [Data Ingestion Engineering](https://github.com/anhphongtrng/K5-L3A-Day06-TranNguyenAnhPhong-2A202603629-DataIngestionEngineering).

### 1.2. Bộ câu hỏi và cách chấm

File benchmark: [`../../data/benchmark/benchmark_calculus_v1_24q.json`](../../data/benchmark/benchmark_calculus_v1_24q.json).

| Split | Có đáp án trong corpus | Thiếu bằng chứng trong corpus | Tổng |
|---|---:|---:|---:|
| DEV | 15 (`Q001`–`Q015`) | 2 (`Q021`–`Q022`) | **17** |
| TEST | 5 (`Q016`–`Q020`) | 2 (`Q023`–`Q024`) | **7** |
| **Tổng** | **20** | **4** | **24** |

- Câu answerable có `expected_answer`, `gold_page`, `evidence_quote`; các trường này **chỉ được đọc bởi evaluator hoặc người rà soát, không đưa vào Retriever/Agent**.
- Câu unanswerable được kiểm tra ngoài phạm vi 50 trang đã index, kỳ vọng `abstain`; không đưa vào mẫu số Hit@1/Hit@3.
- **Hit@1 / Evidence Hit@3:** Tỉ lệ câu answerable có đủ bằng chứng ở rank 1 / top 3; điểm tự động là **proxy** dựa trên trang gold và overlap nội dung, có thể chấm sai khi evidence bị tách hoặc diễn đạt khác. Một số nhãn DEV đã được hiệu chỉnh bằng kiểm tra thủ công.
- **Context Evidence Rate (CER):** Số câu answerable có đủ bằng chứng trong **context cuối cùng** / tổng câu answerable; chấm context 0 = thiếu, 1 = chưa đủ, 2 = đủ bằng chứng và nguồn.
- **Grounded Answer:** Câu answerable chấm 0 = sai/không có evidence, 1 = đúng một phần, 2 = đúng và đủ; **citation support** cần kiểm tra riêng: đúng ID chưa chứng minh nội dung được hỗ trợ.
- **Abstention:** Số câu unanswerable từ chối đúng / tổng câu unanswerable; báo riêng false abstention trên answerable.

## 2. Retrieval và thử nghiệm

### 2.1. Ba phương pháp trên cùng corpus

1. **Dense:** `SentenceTransformer('BAAI/bge-m3')` tạo query vector, tra pgvector cosine; cache tối đa 256 query embeddings.
2. **BM25:** `rank_bm25.BM25Okapi` trên cùng 162 chunks Strategy B, tokenizer regex tiếng Anh `[a-z0-9]+`.
3. **Hybrid:** Hợp nhất ranking Dense và BM25 bằng **Reciprocal Rank Fusion, RRF k=60**; không cộng trực tiếp điểm cosine và BM25 do khác thang đo.

Cùng câu hỏi gốc, cùng corpus, top-k = 3 và section filter từ câu hỏi (nếu có chỉ định rõ `Section x.y`). **Không rerank bằng mô hình phụ**, không lấy filter từ gold metadata.

### 2.2. Bảng Retrieval DEV và TEST

| Cấu hình | Phương pháp, top-k, filter, rerank, context budget | Giả thuyết / thay đổi | Split | Hit@1 | Evidence Hit@3 | Context Evidence Rate | Thời gian / chi phí |
|---|---|---|---|---|---|---|---|
| Baseline retrieval | Dense, top-3, filter theo query, không rerank, 3000 ký tự | Semantic/vector baseline | DEV | **12/15 (80%)** | **14/15 (93,3%)** | Proxy 14/15 | Latency benchmark chịu ảnh hưởng cache; không dùng để xếp hạng |
| Phương pháp 2 | BM25, top-3, cùng filter, 3000 ký tự | Khớp từ khóa | DEV | **8/15 (53,3%)** | **14/15 (93,3%)** | Proxy gốc 12/15; cần hiểu corrections riêng | Local BM25; API Answer không chạy trong phép đo này |
| Phương pháp 3 / retrieval đã chọn | Hybrid RRF k=60, top-3, 3000 ký tự | Kết hợp semantic và lexical | DEV | **11/15 (73,3%)** | **15/15 (100%)** | Proxy 15/15 | Cache làm latency không so sánh trực tiếp |
| Cải thiện 1: C2-v1 | Hybrid top-3 + chọn 3 câu/chunk theo keyword overlap | Nén context để giảm token | DEV | 11/15* | 15/15* | Chưa chấm CER thủ công v1 | Avg 1509,7 ký tự; 9855 Answer tokens; **$0.002005** |
| Cải thiện 2: C2-v2 | Hybrid top-3 + sentence window (1 trước, 2 sau) | Giữ evidence lân cận khi rút gọn | DEV | 11/15* | 15/15* | **15/15 (100%)**, review thủ công | Avg 1502,0 ký tự; 9935 Answer tokens; **$0.002062** |
| Cấu hình chọn | Hybrid top-3 + C1 full chunks, 3000 ký tự | Ưu tiên bằng chứng và grounding | TEST | **2/5 (40%)** | **4/5 (80%)** | **4/5 (80%)**, review thủ công | Answer LLM 4734 tokens / 7 câu, **$0.00094365**; mean end-to-end **2033,93 ms** |
| Baseline retrieval đối chứng | Dense top-3, 3000 ký tự (chỉ đo retrieval) | So với Hybrid ở TEST | TEST | **4/5 (80%)** | **5/5 (100%)** | Proxy 5/5, **chưa chấm context Agent** | Chưa chạy Answer LLM baseline TEST |

\* C2-v1 và C2-v2 chỉ thay **context selection**, không thay ranking Hybrid, nên Hit@1/Hit@3 tái sử dụng kết quả cùng retrieval. Cột CER của retrieval là proxy ban đầu; **đừng nhầm với CER được chấm trực tiếp trên context do Agent gửi**. Tại DEV, phép rà soát riêng hai mode C1/C2-v2 cho CER = **15/15 ở cả hai**.

**Làm rõ BM25:** file `retrieval_dev.csv` lưu proxy gốc BM25 Hit@1 = 7/15, Hit@3 = 12/15; file `evidence_review_corrections.csv` xác nhận false negative ở Q007 và Q013 nên Hit@1 đã hiệu chỉnh = 8/15 và Hit@3 = 14/15. Chỉ số `context_evidence_proxy` trong CSV gốc **chưa được tính lại tương ứng**; không gọi nó là CER thủ công.

**TEST:** Dense 4/5 Hit@1, 5/5 Hit@3; BM25 2/5 và 4/5; Hybrid 2/5 và 4/5 (**automated proxy**, Q019 được đọc và xác nhận retrieval failure). Dense trội hơn trên TEST nhưng **không chọn lại cấu hình dựa vào TEST**.

### 2.3. Thử nghiệm Context trên DEV

| Metric (17 câu DEV) | C1 — full chunks | C2-v1 — keyword snippets | C2-v2 — sentence window |
|---|---:|---:|---:|
| Avg context chars/câu | 1804,4 | 1509,7 | 1502,0 |
| Total Answer tokens | **11222** | 9855 | **9935** |
| Total Answer API cost | **$0.002265** | $0.002005 | **$0.002062** |
| Câu answerable được trả lời | **15/15** | **14/15** | **15/15** |
| Từ chối đúng unanswerable | 2/2 | 2/2 | 2/2 |
| Context đủ evidence — manual (15 câu) | **15/15** | Chưa chấm đủ v1 | **15/15** |
| Full answer + citation support — manual | **13/15** | Chưa chấm đủ v1 | **12/15** |

C2-v2 giảm **1287 Answer tokens (~11,5%)**, **$0.000203 (~9,0%)** và khoảng **16,8% context chars** so với C1 trên DEV; tuy nhiên câu Q002 chưa đáp đầy đủ nên C1 tốt hơn theo tiêu chí chất lượng nghiêm ngặt. **Chi phí chỉ gồm LLM Answer, không gồm Planner hay tài nguyên local**. Các lần chạy riêng của C1 có chênh lệch token nhỏ do đầu ra mô hình không hoàn toàn cố định, dù `temperature=0`.

### 2.4. Phân tích lỗi DEV và thử thay đổi

**Lỗi 1 — Q008, mất evidence qua context compression.** Hybrid tìm đúng chunk trang 62, nhưng C2-v1 chọn ba câu dựa theo keyword overlap và bỏ câu “They also define the relationship among the sides and angles of a triangle.” Agent từ chối câu answerable. **Giả thuyết:** giữ câu lân cận của câu liên quan nhất sẽ bảo toàn evidence. **Thay đổi:** C2-v2 dùng sentence-window (1 câu trước, 2 sau). **Kết quả:** Q008 chuyển từ false abstention (C2-v1) sang trả lời đúng có citation (C2-v2), không thay retrieval.

**Lỗi 2 — Q002, câu trả lời thiếu điều kiện.** Trong C2-v2, context giữ “mở 6 a.m.” và “18 hours each day”; Agent không kết luận rõ giờ đóng cửa **midnight**. **Giả thuyết:** nội dung rút gọn và câu trả lời thiếu suy luận/diễn giải thời điểm kết thúc. **Kết quả đo:** C1 trả lời đủ 6 a.m.–midnight; C2-v2 chỉ đúng một phần; **không sửa tiếp thuật toán sau phép đo này**.

**Lỗi bổ sung — Q007/Q011, citation và PDF extraction.** Q007 trích dẫn một chunk cho hai hiện tượng dù bằng chứng tách qua hai chunk; Q011 thêm chu kỳ số học không còn rõ trong text do biểu thức toán bị mất khi trích PDF. Cần kiểm tra citation theo từng khẳng định; không mặc nhiên coi `valid_ids` là fully grounded.

### 2.5. Quyết định trước TEST

Tài liệu đóng băng: [`../../results/config_freeze.md`](../../results/config_freeze.md).

**Chọn:** Strategy B → **Hybrid RRF k=60, top-3** → **C1 full chunks + dedup, 3000 ký tự** → OpenRouter `openai/gpt-4o-mini` → câu trả lời có citation/abstain. **Lý do:** DEV Hybrid đạt Evidence Hit@3 15/15; C1 cho 13/15 fully grounded so với C2-v2 12/15, dù tốn token hơn. TEST không được dùng để chọn lại Dense, Hybrid hoặc điều chỉnh top-k.

## 3. Agent và đánh giá

### 3.1. Kiến trúc, tool use, điều kiện dừng

- **Agent chạy tương tác (`starter/agent.py`):** LLM Planner (OpenRouter) chọn đúng một tool trong `dense`, `bm25`, `hybrid`; Retriever nhận query gốc và section suy ra từ câu hỏi; Context Builder dedup theo chunk ID, giữ `source`, `page`, `section`, `chunk_id`; LLM Answer trả lời từ context hoặc chuỗi abstention chuẩn; validator đối chiếu các citation ID với selected chunks. Ghi JSON trace tool/query/rank/score/context/answer/citation/token/cost/latency/error. Tối đa **1 lượt retrieval**, không lặp tìm.
- **Benchmark context DEV (`starter/benchmark_context.py`):** Cố định Hybrid top-3 cho C1/C2, **không gọi Planner**, để chỉ đo tác động thay đổi context. Có 17 câu × 2 mode = **34 lượt Answer LLM**.
- **TEST đã chốt (`starter/run_agent_test.py`):** Cố định Hybrid + C1 và **tắt LLM Planner** để kiểm tra retrieval/context/answer theo cấu hình cố định. Có 7/7 câu hoàn tất, không lỗi provider. Vì không có Planner, **không được diễn giải chi phí hoặc tool routing TEST như hiệu năng toàn agent điều phối tự do**.
- **Xử lý lỗi:** `agent.py` lưu trace lỗi; `run_agent_test.py` lưu sau từng câu, khi chạy lại có thể bỏ qua kết quả hoàn thành. Không tự thêm gold vào prompt và không thay lỗi bằng câu trả lời giả.

### 3.2. Grounded answers — bảng tổng hợp

| Cấu hình / split | Tool use | Lượt retrieval | Answer grade 0/1/2 trên answerable; trung bình | Citation support (strict) | Câu có citation | Abstention đúng | Từ chối nhầm |
|---|---|---|---|---|---|---|---|
| Hybrid + C1 / DEV | 17/17 | 1/câu (Planner tắt) | **0/0/15; 2,00/2** về nội dung chính | 13/15 câu answerable có citation fully supporting | 15/15 câu đã trả lời | **2/2** | **0/15** |
| Hybrid + C2-v2 / DEV | 17/17 | 1/câu (Planner tắt) | **0/1/14; 1,93/2** | 13/15 câu có citation fully supporting; fully grounded answer 12/15 | 15/15 câu đã trả lời | **2/2** | **0/15** |
| Hybrid + C1 / TEST | **7/7** | **1/câu** (Planner tắt) | **1/2/2; 1,20/2** | 1 câu có citation `yes`, 3 `partial`; 1 `n/a` (Q019) | **4/4** câu được trả lời | **2/2** | **1/5** |

**Định nghĩa:** 0/1/2 ở cột grade là **số câu được chấm mức 0, mức 1, mức 2**, không phải thứ tự điểm; câu unanswerable được xét abstention riêng. `Citation support` của DEV là tỉ lệ câu có citation fully supporting theo `context_dev_manual_review.csv`, **không phải tỉ lệ từng citation**; TEST `yes/partial` theo `agent_test_manual_review.csv`, không tự suy ra citation-level metric từ nhãn case-level.

### 3.3. Mẫu trace/nhãn TEST (đọc đủ 5 answerable + 2 unanswerable)

| Query ID | Loại câu | Retrieval / context | Câu trả lời và citation | Điểm & nguyên nhân |
|---|---|---|---|---|
| **Q016** | Có đáp án | Hybrid top-3: p.79,79,77; đủ evidence | Giới hạn domain để hàm one-to-one; trích p.79 chunk 0000 | **1/2:** đúng ý chính nhưng diễn giải nhầm “nhiều output cho cùng input” |
| **Q017** | Có đáp án | p.79,80,82; evidence ở p.80 | Hàm lượng giác tuần hoàn nên cần miền one-to-one; trích p.80 chunk 0001 | **2/2:** grounded |
| **Q018** | Có đáp án | p.88,88,91 | Exponential: base cố định, exponent biến; trích p.88 chunk 0003 | **2/2** nội dung chính; biểu thức toán bổ sung không hiển thị đầy đủ trong extracted context nên citation **partial** theo strict review |
| **Q019** | Có đáp án | Top-3 chỉ p.88, gold p.93; thiếu evidence “logarithmic inverse” | **Abstain**, không citation | **0/2:** retrieval failure dẫn đến false abstention |
| **Q020** | Có đáp án | p.96,97,97, có 100× | Trả lời 100 lần; trích p.97 chunk 0000 | **1/2:** phần thông tin năng lượng 31,6× mỗi cấp không được context hỗ trợ |
| **Q023** | Thiếu evidence | Chunks không liên quan tới Hessian | **Abstain**, không citation | **Đúng:** 2/2 theo điểm hành vi abstention |
| **Q024** | Thiếu evidence | Chunks không liên quan autodiff neural network | **Abstain**, không citation | **Đúng:** 2/2 theo điểm hành vi abstention |

**Context trên TEST:** chấm 2/2 cho Q016, Q017, Q018, Q020; 0/2 cho Q019; **CER = 4/5 (80%)**. Hai câu out-of-corpus Q023/Q024 không nằm trong mẫu số CER.

**TEST chi phí và thời gian:** **4734 Answer tokens**, **$0.00094365 API Answer** trên 7 câu; latency trung bình end-to-end **2033,93 ms/câu** (không gồm Planner). Không có lỗi kỹ thuật trong 7 lượt chạy. Tool-use: 7/7 truy vấn có retrieval, 1 lượt/câu.

### 3.4. Minh chứng và giới hạn

**Dữ liệu và kết quả DEV:**

- [`../../results/retrieval_dev.csv`](../../results/retrieval_dev.csv), [`../../results/retrieval_dev_final.csv`](../../results/retrieval_dev_final.csv), [`../../results/evidence_review_corrections.csv`](../../results/evidence_review_corrections.csv) — retrieval ranking, corrections.
- [`../../results/context_dev_v1.csv`](../../results/context_dev_v1.csv), [`../../results/context_dev_trace_v1.json`](../../results/context_dev_trace_v1.json) — C2-v1 (giữ để chứng minh lỗi Q008).
- [`../../results/context_dev.csv`](../../results/context_dev.csv), [`../../results/context_dev_trace.json`](../../results/context_dev_trace.json) — C1/C2-v2.
- [`../../results/context_dev_manual_review.csv`](../../results/context_dev_manual_review.csv) — nhãn rà soát câu trả lời và citation.
- [`../../results/agent_traces/`](../../results/agent_traces/) — ví dụ Agent có Planner thật (Q001/Q021).

**Dữ liệu và kết quả TEST:**

- [`../../results/retrieval_test.csv`](../../results/retrieval_test.csv) — 3 retrieval methods; proxy tự động.
- [`../../results/agent_test.csv`](../../results/agent_test.csv), [`../../results/agent_test_trace.json`](../../results/agent_test_trace.json), [`../../results/agent_test_manual_review.csv`](../../results/agent_test_manual_review.csv) — câu trả lời, evidence, điểm manual, tokens/cost/latency.

**Lệnh tái lập:** xem [`README.md`](README.md) cùng thư mục; chạy từ `starter/` với Docker PostgreSQL Day 6 đã mở và các biến môi trường cần thiết.

**Giới hạn:** (1) corpus mới gồm 50 trang và Strategy B từ Day 6; (2) công thức toán và nội dung đôi khi bị thiếu khi trích PDF, hoặc chunk bị cắt giữa câu; (3) câu trả lời có đúng ý chính nhưng vẫn có thể chứa chi tiết ngoài evidence; (4) retrieval TEST chỉ 5 câu answerable, khác biệt 1 câu làm thay đổi tỉ lệ 20 điểm phần trăm; (5) latency DEV retrieval bị ảnh hưởng cache query embedding;
