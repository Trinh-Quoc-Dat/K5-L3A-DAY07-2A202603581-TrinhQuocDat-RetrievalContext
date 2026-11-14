'''\
Lexical BM25 Retrieval Agent for Lab7
====================================

This script implements the **Lexical (BM25)** retrieval strategy that the user selected as the best performing method.
It:
1. Loads the pre‑processed document chunks from ``data/chunks.json``.
2. Builds an in‑memory BM25 index (using a pure‑Python implementation – no external dependencies).
3. Loads the development questions from ``benchmark/questions.json``.
4. Executes a BM25 search for each dev query, applying any optional metadata filters.
5. Packs the retrieved context respecting the user‑defined budget of **2400 characters** (the same budget used in the benchmark).
6. Computes hit@1, hit@3 and context‑evidence‑rate metrics exactly as in the benchmark script.
7. Writes a ``summary.json`` and ``traces.json`` to ``starter/results/benchmark_dev`` so the output format matches the existing benchmark artifacts.

The agent can be invoked from the command line::

    python -m src.lexical_agent

or imported and called via the ``run`` function.
'''\

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
# Utility functions
# ---------------------------------------------------------------------------

def tokenize(text: str) -> List[str]:
    """Return a list of lowercase word tokens from *text* using a simple regex."""
    return re.findall(r"\\w+", unicodedata.normalize('NFC', text).casefold())


def is_evidence_hit(chunk: Dict[str, Any], gold_list: List[Dict[str, Any]]) -> bool:
    """Check whether *chunk* contains the required evidence for a gold entry.

    The gold entry format is the same as in ``benchmark/questions.json`` – it
    contains ``source_id``, ``page`` and ``evidence`` fields.
    """
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
    """Select a subset of *results* whose concatenated length does not exceed *budget*.

    Returns a human readable ``context`` string and the list of chunk ids that were
    included.  The function mirrors the behaviour used in the benchmark script.
    """
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
        # ``\\n\\n`` is the separator used in the original script
        if len('\\n\\n'.join(blocks + [block])) <= budget:
            blocks.append(block)
            ids.append(cid)
    return '\\n\\n'.join(blocks), ids

# ---------------------------------------------------------------------------
# BM25 implementation (same as in the benchmark script for reproducibility)
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
# Main entry point – reproduces the benchmark output for the dev set
# ---------------------------------------------------------------------------

def run():
    # Paths – keep them identical to the benchmark script for consistency
    chunks_path = Path('F:/VinAI/Lab7/data/chunks.json')
    questions_path = Path('F:/VinAI/Lab7/benchmark/questions.json')
    out_dir = Path('F:/VinAI/Lab7/starter/results/benchmark_dev')
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    chunks = json.loads(chunks_path.read_text(encoding='utf-8'))
    all_questions = json.loads(questions_path.read_text(encoding='utf-8'))
    dev_questions = [q for q in all_questions if q.get('split') == 'dev']

    print(f"Loaded {len(chunks)} chunks and {len(dev_questions)} dev questions.")

    bm25 = BM25Index(chunks)

    summary_results = []
    traces = []

    hit1, hit3, ctx_hit = 0, 0, 0
    latencies = []

    for q in dev_questions:
        q_start = time.perf_counter()
        filters = q.get('filter')
        top_hits = bm25.search(q['question'], top_k=3, filters=filters)
        # add a stable ``id`` field if missing (benchmark expects ``id``)
        for hit in top_hits:
            if 'id' not in hit:
                hit['id'] = hit.get('chunk_id') or hit.get('document_id')
        # Pack context using the larger budget (2400 chars)
        context, ids = pack_context(top_hits, budget=2400)
        # Mark evidence hits
        marked = [{**h, 'hit': is_evidence_hit(h, q['gold'])} for h in top_hits]
        h1 = bool(marked and marked[0]['hit'])
        h3_any = any(m['hit'] for m in marked)
        ctx = any(m['hit'] and m['id'] in ids for m in marked)
        if h1:
            hit1 += 1
        if h3_any:
            hit3 += 1
        if ctx:
            ctx_hit += 1
        elapsed_ms = (time.perf_counter() - q_start) * 1000
        latencies.append(elapsed_ms)
        traces.append({
            'query_id': q['id'],
            'question': q['question'],
            'filters': filters,
            'hit1': h1,
            'hit3': h3_any,
            'context_hit': ctx,
            'elapsed_ms': round(elapsed_ms, 2),
            'context_chars': len(context),
            'top_hits': marked,
        })

    n = len(dev_questions)
    summary_results.append({
        'strategy': 'Lexical (BM25)',
        'top_k': 3,
        'hit_at_1': f"{hit1}/{n} ({hit1/n*100:.1f}%)",
        'evidence_hit_at_3': f"{hit3}/{n} ({hit3/n*100:.1f}%)",
        'context_evidence_rate': f"{ctx_hit}/{n} ({ctx_hit/n*100:.1f}%)",
        'median_latency_ms': round(statistics.median(latencies), 2),
        'total_time_seconds': round(sum(latencies)/1000, 2),
        'cost_usd': 0.0,
    })

    # Persist results – identical filenames to the benchmark script so downstream
    # tooling can reuse them.
    (out_dir / 'summary.json').write_text(json.dumps(summary_results, ensure_ascii=False, indent=2), encoding='utf-8')
    (out_dir / 'traces.json').write_text(json.dumps({'Lexical (BM25)': traces}, ensure_ascii=False, indent=2), encoding='utf-8')

    print('\n' + '='*50)
    print('LEXICAL BM25 AGENT EVALUATION COMPLETED')
    print('='*50)
    print(json.dumps(summary_results, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    run()
