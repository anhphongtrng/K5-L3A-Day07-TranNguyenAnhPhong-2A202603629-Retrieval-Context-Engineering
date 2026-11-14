# Day 7 — Retrieval & Context Engineering: Hướng dẫn chạy và cấu trúc dự án

**Sinh viên:** Trần Nguyễn Anh Phong — **2A202603629**  
**Repository LAB7:** https://github.com/anhphongtrng/K5-L3A-Day07-TranNguyenAnhPhong-2A202603629-Retrieval-Context-Engineering  
**Nguồn dữ liệu LAB6:** https://github.com/anhphongtrng/K5-L3A-Day06-TranNguyenAnhPhong-2A202603629-DataIngestionEngineering

## 1. Cây thư mục và chức năng file

Cây dưới đây tập trung vào **những file đã dùng/tạo trong bài cá nhân** (repo gốc còn các file khung như `GUIDE.md`, `BENCHMARK.md`, `SUBMISSION.md`, `RUBRIC.md`, `starter/agentic.py`, `starter/baseline.py`, `tests/`...). Không phải danh sách tuyệt đối mọi file của repo.

```text
K5-L3A-Day07-TranNguyenAnhPhong-2A202603629-Retrieval-Context-Engineering/
├── README.md                                  # README đề bài gốc
├── GUIDE.md                                   # Hướng dẫn các phần LAB7
├── BENCHMARK.md                               # Định nghĩa benchmark / metric
├── SUBMISSION.md                              # Quy định nộp bài
├── data/
│   └── benchmark/
│       ├── benchmark_calculus_v1.json          # Benchmark gốc Day6 (20 câu), nếu còn lưu
│       └── benchmark_calculus_v1_24q.json      # Benchmark Day7 thực sự dùng: 24 câu
├── starter/
│   ├── retrieval.py                           # Dense, BM25, Hybrid RRF; PostgreSQL Strategy B
│   ├── run_benchmark.py                       # Benchmark retrieval DEV/TEST (proxy)
│   ├── agent.py                               # Agent OpenRouter, planner, context, answer, traces
│   ├── benchmark_context.py                   # So sánh C1/C2 trên DEV (34 Answer runs)
│   ├── review_context.py                      # Xuất file rà soát manual từ trace DEV
│   ├── run_agent_test.py                      # Chạy TEST Hybrid + C1 đã cố định
│   └── README.md                              # Hướng dẫn starter nguyên bản
├── results/
│   ├── retrieval_dev.csv                      # Điểm retrieval DEV tự động
│   ├── retrieval_dev_final.csv                # DEV sau khi hợp nhất corrections (nếu đã xuất)
│   ├── retrieval_dev_trace.json               # Nội dung kết quả retrieval DEV
│   ├── evidence_review_corrections.csv        # Hiệu chỉnh false negatives Q007 / Q013
│   ├── context_dev_v1.csv                     # C1 + C2-v1 trước khi sửa Q008
│   ├── context_dev_trace_v1.json              # Trace context C2-v1
│   ├── context_dev.csv                        # C1 + C2-v2 mới nhất
│   ├── context_dev_trace.json                 # Trace context C2-v2
│   ├── context_dev_manual_review.csv          # Nhãn chấm 34 DEV runs
│   ├── config_freeze.md                       # Quyết định cấu hình trước khi TEST
│   ├── retrieval_test.csv                     # Retrieval TEST của 3 phương pháp
│   ├── agent_test.csv                         # Agent TEST tóm tắt 7 câu
│   ├── agent_test_trace.json                  # Agent TEST context / answers / citations
│   ├── agent_test_manual_review.csv           # Chấm manual 7 câu TEST
│   └── agent_traces/                          # JSON từng lần chạy thử Agent có Planner
│       └── *.json
└── report/
    └── individual/
        ├── 2A202603629_TranNguyenAnhPhong.md  # Báo cáo kết quả cá nhân
        └── README.md                          # Tài liệu đang đọc
```


### Các file code viết thêm

| File | Chức năng chính |
|---|---|
| `starter/retrieval.py` | `Retriever` truy vấn `public.lab_document_chunks` và tính Dense (pgvector), BM25 (rank_bm25), Hybrid RRF (k=60). |
| `starter/run_benchmark.py` | Chạy 3 phương pháp trên benchmark 24 câu theo `--split dev/test`; ghi CSV và proxy evidence; có trace DEV. |
| `starter/agent.py` | OpenRouter LLM Planner chọn tool, search 1 lần, context C1/C2, answer với citation hoặc abstain, kiểm tra ID citation và lưu trace. |
| `starter/benchmark_context.py` | DEV controlled comparison: cùng Hybrid top-3, chỉ đổi C1/C2; **tắt Planner** để cô lập chi phí Answer. |
| `starter/review_context.py` | Ghép câu hỏi chuẩn và trace, xuất `context_dev_review.txt` cho người chấm; không tự chứng minh semantic grounding. |
| `starter/run_agent_test.py` | Chạy 7 TEST với cấu hình cố định **Hybrid top-3 + C1 + 3000 chars**; lưu CSV/JSON sau mỗi câu; có cơ chế skip câu đã hoàn thành. |

## 2. Nguồn dữ liệu và vector store

- **Nguồn:** OpenStax *Calculus Volume 1*, tệp PDF Day 6 thường có đường dẫn `data/calculus_volume_1_openstax.pdf` **trong repo LAB6**. Dữ liệu text index là **trang PDF vật lý 51–100**, không phải toàn sách. Nội dung thuộc các section 1.2–1.5.
- **Số chunks Day 6:** Strategy A = 171, Strategy B = 162, tổng 333. **LAB7 chỉ lấy 162 chunks Strategy B** và embeddings có sẵn, **không ingest/encode lại corpus**.
- **PostgreSQL/pgvector** do Docker của **LAB6** cung cấp; database `day6_lab`, bảng `public.lab_document_chunks`; các cột `strategy`, `chunk_id`, `document_id`, `source`, `page`, `content`, `metadata` (JSONB), `embedding` (vector 1024).
- **Embedding model:** Hugging Face `BAAI/bge-m3` qua `sentence-transformers`, kích thước 1024; query vectors dùng `normalize_embeddings=True`.
- **Benchmark thật sự được sử dụng trong LAB7:** `LAB7/data/benchmark/benchmark_calculus_v1_24q.json` (24 câu = 17 DEV + 7 TEST). `benchmark_calculus_v1.json` 20 câu gốc **không** dùng cho phép đo LAB7 24 câu.
- **Địa chỉ PostgreSQL từ Windows host:** `127.0.0.1:55433` (cổng máy host map vào PostgreSQL container), database `day6_lab`. **Chỉ có chunks/embeddings nằm trong Docker volume/PostgreSQL**; file câu hỏi JSON và các CSV/JSON trace nằm trên ổ đĩa project LAB7.

Nguồn OpenStax: https://openstax.org/details/books/calculus-volume-1. Khi chia sẻ PDF/chunks, cần tuân thủ điều khoản của **phiên bản PDF thực tế**; hiện trang OpenStax nêu CC BY-NC-SA 4.0 và các điều kiện riêng đối với AI offerings.

## 3. Setup trên Windows PowerShell

Các lệnh dưới đây giả định mở terminal tại **thư mục gốc LAB7**. Cần **Python 3.11 trở lên**, Docker Desktop và PostgreSQL/pgvector của LAB6 đã được tạo sẵn.

### 3.1. Khởi động database Day 6

1. Mở Docker Desktop.
2. Chuyển tới repo **LAB6** và khởi động lại **PostgreSQL container/Compose stack đã tạo ở Day 6** bằng đúng lệnh cấu hình của LAB6; LAB7 **không tạo lại vector store**.
3. Kiểm tra container đang chạy:

```powershell
docker ps
```

Nếu đã thiết lập Day6 với Docker Compose, dùng `docker compose up -d` **trong thư mục chứa file compose của LAB6**, không chạy lệnh này tùy tiện ở `starter/` LAB7.

### 3.2. Python environment và dependencies

```powershell
# Ở root LAB7
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install "psycopg[binary]" pgvector sentence-transformers rank-bm25 requests
```

`torch`, `transformers` và các phụ thuộc khác được `sentence-transformers` cài theo dependency resolver. Tên package/các phiên bản chính xác cần được pin trong môi trường nếu muốn tái lập tuyệt đối. Repo có các file starter gốc, nhưng **các script cá nhân dùng các thư viện trên**.

### 3.3. Khai báo biến môi trường

```powershell
# Thay USER, PASSWORD bằng tài khoản PostgreSQL Day 6 của bạn.
$env:DATABASE_URL="postgresql://USER:PASSWORD@127.0.0.1:55433/day6_lab"

# Lấy API key từ tài khoản OpenRouter; KHÔNG đưa key thật lên Git.
$env:OPENROUTER_API_KEY="YOUR_OPENROUTER_API_KEY"
$env:OPENROUTER_MODEL="openai/gpt-4o-mini"

# Kiểm tra tồn tại mà không in secret ra terminal.
python -c "import os; print({x: bool(os.getenv(x)) for x in ('DATABASE_URL','OPENROUTER_API_KEY','OPENROUTER_MODEL')})"
```

Các biến môi trường PowerShell chỉ có hiệu lực trong phiên terminal hiện tại; mở terminal mới phải đặt lại. **Không commit `.env`, API key, password hoặc Docker volume**. Tránh ghi URL chứa mật khẩu trong ảnh chụp và log.

### 3.4. Kiểm tra PostgreSQL có 162 chunks Strategy B

```powershell
@'
import os
import psycopg
with psycopg.connect(os.environ['DATABASE_URL']) as conn:
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM public.lab_document_chunks WHERE strategy = 'B' AND embedding IS NOT NULL")
        print('Strategy B embedded chunks:', cur.fetchone()[0])
'@ | python -
```


## 4. Chạy lại các phần của LAB7

Từ root LAB7:

```powershell
cd starter
```

**Mọi lệnh bên dưới chạy từ `LAB7/starter/`** vì `run_benchmark.py`, `benchmark_context.py` và `review_context.py` dùng đường dẫn tương đối `../data` / `../results`.

### 4.1. Benchmark Retrieval DEV và TEST

```powershell
python run_benchmark.py --split dev
python run_benchmark.py --split test
```

Đầu ra: `../results/retrieval_dev.csv`, `../results/retrieval_test.csv`; proxy có thể báo false negative nên đọc các manual corrections. **TEST chỉ nên chạy sau khi freeze**, không dùng kết quả TEST để chỉnh cấu hình. Script hiện không nhất thiết xuất `retrieval_test_trace.json`.

### 4.2. Chạy thử Agent có Planner thật

```powershell
python agent.py --question "What is a piecewise-defined function?" --context-mode baseline
python agent.py --question "What is a piecewise-defined function?" --context-mode improved
python agent.py --question "According to the indexed material, how does the Adam optimizer update parameters using derivatives?"
```

- Planner dùng OpenRouter chọn `dense`, `bm25` hoặc `hybrid` (được ghi trace), sau đó gọi retrieval **1 lần**.
- Context mode `baseline` (C1) giữ toàn chunk, `improved` (C2-v2) chọn một sentence window; giới hạn **3000 ký tự**, gồm metadata nguồn. Lời đáp có `[chunk_id]` hoặc từ chối khi không đủ bằng chứng.
- Trace: `../results/agent_traces/*.json` (query, planner, method, ranking, context, citations, latency, token, cost, errors).

### 4.3. So sánh Context Engineering trên DEV

```powershell
python benchmark_context.py
python review_context.py
```

- `benchmark_context.py`: chạy **17 DEV questions × 2 context modes = 34 LLM Answer runs**, **cố định Hybrid** để không lẫn tác động lựa chọn Planner.
- Ghi đè `../results/context_dev.csv` và `../results/context_dev_trace.json` theo logic **C2 đang có trong `agent.py`** (hiện C2-v2).
- `review_context.py` xuất `../results/context_dev_review.txt` để kiểm tra evidence thủ công; `EXACT EVIDENCE FOUND` chỉ kiểm tra text overlap, **không phải nhãn semantic cuối**.
- Kết quả C2-v1 lịch sử đã được lưu trong `context_dev_v1.csv`, `context_dev_trace_v1.json`. **Không thể tái chạy C2-v1 bằng code C2-v2 hiện tại trừ khi checkout commit cũ**; không ghi đè hai file v1 khi chạy thử nghiệm lại.

### 4.4. Chạy Agent TEST với cấu hình đóng băng

```powershell
python run_agent_test.py
```

Cấu hình: **Hybrid RRF k=60, top-3, C1 Baseline (full chunks), budget 3000 chars, `openai/gpt-4o-mini`**. Script chỉ gọi LLM **Answer**, **tắt Planner** để so sánh một cấu hình cố định. File được lưu tại `../results/agent_test.csv` và `../results/agent_test_trace.json`, đồng thời lưu tiến độ sau từng câu.

**Quan trọng:** script mặc định bỏ qua các câu TEST đã hoàn tất; nếu muốn chủ động chạy lại, đọc tùy chọn `python run_agent_test.py --help` trước. **Không chạy lại TEST để điều chỉnh theo kết quả TEST cũ**; nếu làm thí nghiệm mới cần ghi rõ không còn là held-out evaluation độc lập.

### 4.5. Unit tests khung gốc

```powershell
cd ..
python -m unittest discover -s tests -v
```

Lệnh này là test code/mock của starter gốc; **không thay thế** các benchmark retrieval/Agent thật. Kết quả unit tests của môi trường hiện tại **chưa được xác minh trong báo cáo này**.

## 5. Đọc kết quả và đối chiếu báo cáo

| File | Đọc thông tin gì? |
|---|---|
| `results/retrieval_dev.csv` | Dense/BM25/Hybrid, Hit@1/Hit@3 proxy trên 15 câu answerable DEV. |
| `results/evidence_review_corrections.csv` | Hai trường hợp BM25 bị proxy chấm sai (Q007, Q013). |
| `results/context_dev.csv` | Context length, tokens, cost, answer, abstained của C1/C2-v2 (34 runs). |
| `results/context_dev_trace.json` | Text context cuối, selected chunk IDs, citations dùng để đọc manual. |
| `results/context_dev_manual_review.csv` | Nhãn context 0/1/2, answer 0/1/2, citation support. |
| `results/retrieval_test.csv` | Dense/BM25/Hybrid trên 7 câu TEST, trong đó 5 answerable. |
| `results/agent_test.csv` | 7 câu TEST: kết quả, abstention, tokens, API cost, latency. |
| `results/agent_test_trace.json` | Toàn bộ evidence, trả lời, citation và metadata để audit TEST. |
| `results/agent_test_manual_review.csv` | Nhãn chấm thủ công 7 câu TEST. |
| `results/config_freeze.md` | Quyết định lựa chọn Hybrid + C1 dựa trên DEV, trước TEST. |

**Metric và giới hạn:** retrieval TEST Hybrid 4/5 Evidence Hit@3 và TEST Answer CER 4/5; Q019 thiếu evidence do retrieval lấy ba chunk trang 88 thay vì quan hệ hàm ngược trang 93. TEST Answer tokens 4734; Answer API cost $0.00094365; mean run latency 2033,93 ms; correct abstention 2/2. DEV C2-v2 tiết kiệm khoảng 11,5% Answer tokens và 9,0% Answer API cost so với C1, nhưng C1 đầy đủ hơn về grounding. Chi phí **không gồm Planner**; phải phân biệt automated proxy, manual review và actual Answer benchmark.

**Chưa có:** Agent **Dense baseline** trên cùng TEST, nên chưa thể báo cáo kết quả Agent baseline-vs-selected TEST đầy đủ. Không điền số giả. Nếu thực hiện sau, cần ghi rõ là đánh giá đối chứng bổ sung với cấu hình giữ nguyên, không dùng để tinh chỉnh TEST.

## 6. Bảo mật, tái lập và phạm vi

- Không commit key/token, PostgreSQL password, `.venv/`, cache model, secret file hoặc Docker volume.
- Phiên bản `BAAI/bge-m3` và dependencies có thể thay đổi nếu không pin revisions/versions. Khi benchmark cần giữ cùng corpus Strategy B, embedding 1024D và model `openai/gpt-4o-mini`.
- PDF có thể mất biểu thức toán/giữa câu, làm evidence/citation thiếu đầy đủ; **đừng chỉ dựa vào citation ID valid**.
- File trace có thể chứa đoạn trích tài liệu OpenStax, cần tuân thủ quyền sử dụng nội dung gốc.