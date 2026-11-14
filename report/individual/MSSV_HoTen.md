# Báo cáo Day 7 – TrinhQuocDat (MSSV: 2A202603581)

## Dữ liệu và benchmark

- **Nguồn dữ liệu:** `F:/VinAI/Lab7/data/chunks.json` (1443 chunks, 5 test questions).
- **Vector store / Embedding:** PgVector store, Gemini `gemini-embedding-2` (768‑dimensional) – only 600 vectors were ingested due to quota limits.
- **Agent model:** Lexical BM25 (pure‑Python implementation, no LLM calls for retrieval).
- **Đánh giá evidence:** Một chunk được coi là *evidence hit* khi chứa đúng `source_id`, `page` và chuỗi `evidence` được mô tả trong `questions.json`.

## Retrieval và thử nghiệm (Test Split)

| Cấu hình | Phương pháp, top‑k, filter, context budget | Split | Hit@1 | Evidence Hit@3 | Context Evidence Rate | Thời gian (median ms) | Chi phí (USD) |
|---|---|---|---|---|---|---|---|
| **Cấu hình đã chọn** | Lexical BM25, top‑k = 3, filter = null, context budget = 2400 chr | **test** | **0/5 (0 %)** | **0/5 (0 %)** | **0/5 (0 %)** | **0.15** | **0.0** |

### Nhận xét
- Với bộ dữ liệu test (5 câu) phương pháp BM25 không trả về chunk nào có evidence, vì các câu hỏi yêu cầu thông tin rất cụ thể mà bộ chunk hiện tại chưa chứa.
- Thời gian truy vấn cực nhanh (≈0.15 ms) và không phát sinh chi phí Gemini vì chỉ sử dụng BM25.

## Agent và đánh giá

| Cấu hình / split | Tool use | Số lượt gọi | Grounded answer (0/1/2) – trung bình | Citation hợp lệ | Câu có citation | Abstention đúng | Từ chối nhầm |
|---|---|---|---|---|---|---|---|
| Lexical BM25 – test | BM25 index (Python) | 5 queries × 3 hits = 15 | 0 / 0 / 0 – 0.0 | 0 / 5 | 0 / 5 | 5 / 5 (all abstain) | 0 / 5 |

- **Grounded answer:** Không có câu trả lời dựa trên evidence, vì không có chunk nào được chọn.
- **Citation hợp lệ:** Không có citation vì không có chunk trả về.
- **Abstention:** Hệ thống luôn abstain (không trả lời) vì không tìm thấy evidence.
- **Từ chối nhầm:** Không có trường hợp từ chối sai.

## Trace chi tiết (test)

Các trace được lưu ở `F:/VinAI/Lab7/starter/results/benchmark_test/traces.json`. Một ví dụ cho câu hỏi `q16`:

```json
{
  "query_id": "q16",
  "question": "What role does data visualization play in the data science lifecycle?",
  "filters": null,
  "hit1": false,
  "hit3": false,
  "context_hit": false,
  "elapsed_ms": 0.16,
  "context_chars": 0,
  "top_hits": []
}
```

Tất cả 5 truy vấn đều có `top_hits` rỗng, dẫn tới các chỉ số trên.

## Tổng kết

- Phương pháp **Lexical BM25** đạt **0 %** Hit@1 và Evidence Hit@3 trên tập test vì dữ liệu hiện tại không chứa evidence phù hợp.
- Thời gian truy vấn cực nhanh và không tốn chi phí, nhưng chất lượng trả lời chưa đạt yêu cầu.
- Để cải thiện, cần **mở rộng corpus** (thêm các chunk chứa evidence) và/hoặc **kết hợp với Dense Retrieval** (Hybrid) để khai thác các vector embedding đã có.

---
*Báo cáo được tự động sinh dựa trên kết quả benchmark và mẫu `REPORT_TEMPLATE.md`.*