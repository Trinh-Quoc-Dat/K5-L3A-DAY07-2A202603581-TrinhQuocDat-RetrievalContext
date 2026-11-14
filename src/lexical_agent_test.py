"""
Lexical BM25 Retrieval Agent – Test Set Evaluation
=================================================

Runs the same BM25 benchmark as `lexical_agent.py` but on the **test** split.
The output files are written to `starter/results/benchmark_test/` so they stay
separate from the dev results.
"""

import json
import time
import math
import statistics
import re
import unicodedata
from pathlib import Path
from collections import Counter
from typing import List, Dict, Any

# ---------------------------------------------------------------------------
# Helpers (identical to the dev agent)
# ---------------------------------------------------------------------------

def tokenize(text: str) -> List[str]:
    return re.findall(r"\\w+", unicodedata.normalize('NFC', text).casefold())

def is_evidence_hit(chunk: Dict[str, Any], gold_list: List[Dict[str, Any]]) -> bool:
    c_source = chunk.get('source_id') or chunk.get('document_id')
    c_page = int(chunk.get('page', -1))
    c_text = chunk.get('text', chunk.get('content', ''))
    for g in gold_list:
        if (
            c_source == g.get('source_id')
            and c_page == int(g.get('page', -2))
            and g.get('evidence', '') in c_text
        ):
            return True
    return False

def pack_context(results: List[Dict[str, Any]], budget: int = 2400) -> (str, List[str]):
    blocks, ids, seen = [], [], set()
    for c in results:
        cid = c.get('id') or c.get('chunk_id')
        if cid in seen:
            continue
        seen.add(cid)
        source = c.get('source', 'principles-data-science')
        page = c.get('page', 0)
        text = c.get('text', c.get('content', ''))
        block = f"[{cid} | {source} | page {page}]\n{text}"
        if len('\n\n'.join(blocks + [block])) <= budget:
            blocks.append(block)
            ids.append(cid)
    return '\n\n'.join(blocks), ids

# ---------------------------------------------------------------------------
# Simple BM25 implementation (same parameters as the dev benchmark)
# ---------------------------------------------------------------------------

class BM25Index:
    def __init__(self, chunks: List[Dict[str, Any]], k1: float = 1.5, b: float = 0.75):
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self.doc_lens = [len(tokenize(c['text'])) for c in chunks]
        self.avgdl = sum(self.doc_lens) / (len(chunks) or 1)
        self.doc_counts = [Counter(tokenize(c['text'])) for c in chunks]
        self.df = Counter(term for counts in self.doc_counts for term in counts)
        self.N = len(chunks)
        self.idf = {
            term: math.log((self.N - cnt + 0.5) / (cnt + 0.5) + 1.0)
            for term, cnt in self.df.items()
        }

    def search(self, query: str, top_k: int = 3, filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        terms = tokenize(query)
        scores = []
        for idx, (chunk, counts, doc_len) in enumerate(zip(self.chunks, self.doc_counts, self.doc_lens)):
            if filters:
                meta = chunk.get('metadata', {})
                if not all(str(meta.get(k)) == str(v) for k, v in filters.items()):
                    continue
            score = 0.0
            for term in terms:
                if term not in self.idf:
                    continue
                tf = counts.get(term, 0)
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * (doc_len / (self.avgdl or 1)))
                score += self.idf[term] * (numerator / (denominator or 1))
            if score > 0:
                scores.append({**chunk, 'score': score})
        scores.sort(key=lambda x: (-x['score'], x.get('id')))
        return scores[:top_k]

# ---------------------------------------------------------------------------
# Main evaluation for the **test** split
# ---------------------------------------------------------------------------

def run():
    chunks_path = Path('F:/VinAI/Lab7/data/chunks.json')
    questions_path = Path('F:/VinAI/Lab7/benchmark/questions.json')
    out_dir = Path('F:/VinAI/Lab7/starter/results/benchmark_test')
    out_dir.mkdir(parents=True, exist_ok=True)

    chunks = json.loads(chunks_path.read_text(encoding='utf-8'))
    all_questions = json.loads(questions_path.read_text(encoding='utf-8'))
    test_questions = [q for q in all_questions if q.get('split') == 'test']

    print(f'Loaded {len(chunks)} chunks and {len(test_questions)} test questions.')

    bm25 = BM25Index(chunks)

    summary = []
    traces = []
    hit1 = hit3 = ctx_hit = 0
    latencies = []

    for q in test_questions:
        start = time.perf_counter()
        top_hits = bm25.search(q['question'], top_k=3, filters=q.get('filter'))
        for h in top_hits:
            if 'id' not in h:
                h['id'] = h.get('chunk_id') or h.get('document_id')
        context, ids = pack_context(top_hits, budget=2400)
        marked = [{**h, 'hit': is_evidence_hit(h, q['gold'])} for h in top_hits]
        h1 = bool(marked and marked[0]['hit'])
        h3_any = any(m['hit'] for m in marked)
        ctx = any(m['hit'] and m['id'] in ids for m in marked)
        if h1: hit1 += 1
        if h3_any: hit3 += 1
        if ctx: ctx_hit += 1
        elapsed = (time.perf_counter() - start) * 1000
        latencies.append(elapsed)
        traces.append({
            'query_id': q['id'],
            'question': q['question'],
            'filters': q.get('filter'),
            'hit1': h1,
            'hit3': h3_any,
            'context_hit': ctx,
            'elapsed_ms': round(elapsed, 2),
            'context_chars': len(context),
            'top_hits': marked,
        })

    n = len(test_questions)
    summary.append({
        'strategy': 'Lexical (BM25)',
        'top_k': 3,
        'hit_at_1': f'{hit1}/{n} ({hit1/n*100:.1f}%)',
        'evidence_hit_at_3': f'{hit3}/{n} ({hit3/n*100:.1f}%)',
        'context_evidence_rate': f'{ctx_hit}/{n} ({ctx_hit/n*100:.1f}%)',
        'median_latency_ms': round(statistics.median(latencies), 2),
        'total_time_seconds': round(sum(latencies)/1000, 2),
        'cost_usd': 0.0,
    })

    (out_dir/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    (out_dir/'traces.json').write_text(json.dumps({'Lexical (BM25)': traces}, ensure_ascii=False, indent=2), encoding='utf-8')

    print('\n' + '='*50)
    print('LEXICAL BM25 TEST EVALUATION COMPLETED')
    print('='*50)
    print(json.dumps(summary, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    run()
