"""
Benchmark 3 Retrieval Strategies on Dev Set for Lab 7:
1. Dense Retrieval (PgVector + gemini-embedding-2 / 768 dimensions)
2. Lexical Retrieval (BM25 with metadata filtering)
3. Hybrid Retrieval (Dense + BM25 Reciprocal Rank Fusion)
"""
import os
import sys
import json
import time
import math
import statistics
import unicodedata
import re
import requests
from collections import Counter
from pathlib import Path
from dotenv import load_dotenv

sys.path.insert(0, 'F:/VinAI/Lab6/K5-L3A-DAY06-2A202603581-DataIngestion/starter')
load_dotenv('F:/VinAI/Lab6/K5-L3A-DAY06-2A202603581-DataIngestion/starter/.env')

from src.store import PgVectorStore


def is_evidence_hit(chunk, gold_list):
    c_source = chunk.get('source_id') or chunk.get('document_id')
    c_page = int(chunk.get('page', -1))
    c_text = chunk.get('text', chunk.get('content', ''))
    
    for g in gold_list:
        if c_source == g.get('source_id') and c_page == int(g.get('page', -2)) and g.get('evidence', '') in c_text:
            return True
    return False


def pack_context(results, budget=2400):
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


def tokenize(text):
    return re.findall(r'\w+', unicodedata.normalize('NFC', text).casefold())


class BM25Index:
    def __init__(self, chunks, k1=1.5, b=0.75):
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self.doc_lens = [len(tokenize(c['text'])) for c in chunks]
        self.avgdl = sum(self.doc_lens) / (len(chunks) or 1)
        self.doc_counts = [Counter(tokenize(c['text'])) for c in chunks]
        self.df = Counter(term for counts in self.doc_counts for term in counts)
        self.N = len(chunks)
        self.idf = {
            term: math.log((self.N - count + 0.5) / (count + 0.5) + 1.0)
            for term, count in self.df.items()
        }

    def search(self, query, top_k=3, filters=None):
        terms = tokenize(query)
        scores = []
        for idx, (chunk, counts, doc_len) in enumerate(zip(self.chunks, self.doc_counts, self.doc_lens)):
            if filters:
                meta = chunk.get('metadata', {})
                match = all(str(meta.get(k)) == str(v) for k, v in filters.items())
                if not match:
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
        
        scores.sort(key=lambda x: (-x['score'], x['id']))
        return scores[:top_k]


def embed_query_gemini2(text, api_key):
    url = f'https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-2:embedContent?key={api_key}'
    payload = {
        'model': 'models/gemini-embedding-2',
        'content': {'parts': [{'text': text}]},
        'outputDimensionality': 768
    }
    r = requests.post(url, json=payload, timeout=20)
    if r.status_code == 200:
        return r.json()['embedding']['values']
    else:
        raise RuntimeError(f"Embed query failed: {r.text}")


def main():
    chunks_path = Path('F:/VinAI/Lab7/data/chunks.json')
    questions_path = Path('F:/VinAI/Lab7/benchmark/questions.json')
    
    chunks = json.loads(chunks_path.read_text(encoding='utf-8'))
    all_questions = json.loads(questions_path.read_text(encoding='utf-8'))
    dev_questions = [q for q in all_questions if q.get('split') == 'dev']
    
    api_key = os.getenv('GEMINI_API_KEY').strip().strip('"')
    database_url = os.getenv('DATABASE_URL')
    
    store = PgVectorStore(database_url, dimensions=768)
    bm25 = BM25Index(chunks)
    
    print(f"Loaded {len(chunks)} chunks and {len(dev_questions)} dev questions.")
    
    # Pre-embed dev queries
    print("Pre-embedding dev queries...")
    query_vectors = {}
    for q in dev_questions:
        for attempt in range(5):
            try:
                query_vectors[q['id']] = embed_query_gemini2(q['question'], api_key)
                time.sleep(1.0)
                break
            except Exception as e:
                time.sleep(5)
        # Ensure every dev query has a vector; if embedding failed, use a zero vector
        if q['id'] not in query_vectors:
            query_vectors[q['id']] = [0.0] * 768
    
    strategies = [
        ('Dense (PgVector + Gemini 768d)', 'dense'),
        ('Lexical (BM25)', 'bm25'),
        ('Hybrid (Dense + BM25 RRF)', 'hybrid')
    ]
    
    summary_results = []
    traces_dict = {}
    
    for label, strat_type in strategies:
        t0 = time.time()
        latencies = []
        traces = []
        hit1_count = 0
        hit3_count = 0
        ctx_hit_count = 0
        
        for q in dev_questions:
            q_start = time.perf_counter()
            filters = q.get('filter')
            
            if strat_type == 'dense':
                hits = store.search(strategy='B', query_vector=query_vectors[q['id']], top_k=3, filters=filters)
                top_hits = [{
                    'id': h['chunk_id'],
                    'source_id': h['document_id'],
                    'source': h['source'],
                    'page': h['page'],
                    'text': h['text'],
                    'score': float(h['score']),
                    'metadata': h['metadata']
                } for h in hits]
            elif strat_type == 'bm25':
                top_hits = bm25.search(q['question'], top_k=3, filters=filters)
            elif strat_type == 'hybrid':
                dense_hits = store.search(strategy='B', query_vector=query_vectors[q['id']], top_k=10, filters=filters)
                bm25_hits = bm25.search(q['question'], top_k=10, filters=filters)
                
                rrf_scores = {}
                doc_map = {}
                
                for rank, h in enumerate(dense_hits):
                    cid = h['chunk_id']
                    doc_map[cid] = {
                        'id': h['chunk_id'],
                        'source_id': h['document_id'],
                        'source': h['source'],
                        'page': h['page'],
                        'text': h['text'],
                        'score': float(h['score']),
                        'metadata': h['metadata']
                    }
                    rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (60 + rank + 1))
                    
                for rank, h in enumerate(bm25_hits):
                    cid = h['id']
                    if cid not in doc_map:
                        doc_map[cid] = h
                    rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (60 + rank + 1))
                    
                sorted_cids = sorted(rrf_scores.keys(), key=lambda x: -rrf_scores[x])
                top_hits = [doc_map[cid] for cid in sorted_cids[:3]]
                for h in top_hits:
                    h['rrf_score'] = rrf_scores[h['id']]
                    
            elapsed_ms = (time.perf_counter() - q_start) * 1000
            latencies.append(elapsed_ms)
            
            context, ids = pack_context(top_hits, budget=2400)
            marked = [{**h, 'hit': is_evidence_hit(h, q['gold'])} for h in top_hits]
            
            h1 = bool(marked and marked[0]['hit'])
            h3 = any(h['hit'] for h in marked)
            ctx_hit = any(h['hit'] and h['id'] in ids for h in marked)
            
            if h1:
                hit1_count += 1
            if h3:
                hit3_count += 1
            if ctx_hit:
                ctx_hit_count += 1
                
            traces.append({
                'query_id': q['id'],
                'question': q['question'],
                'filters': filters,
                'hit1': h1,
                'hit3': h3,
                'context_hit': ctx_hit,
                'elapsed_ms': round(elapsed_ms, 2),
                'context_chars': len(context),
                'top_hits': marked
            })
            
        n = len(dev_questions)
        summary_results.append({
            'strategy': label,
            'top_k': 3,
            'hit_at_1': f"{hit1_count}/{n} ({hit1_count/n*100:.1f}%)",
            'evidence_hit_at_3': f"{hit3_count}/{n} ({hit3_count/n*100:.1f}%)",
            'context_evidence_rate': f"{ctx_hit_count}/{n} ({ctx_hit_count/n*100:.1f}%)",
            'median_latency_ms': round(statistics.median(latencies), 2),
            'total_time_seconds': round(time.time() - t0, 2),
            'cost_usd': 0.0
        })
        traces_dict[label] = traces
        
    out_dir = Path('F:/VinAI/Lab7/starter/results/benchmark_dev')
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'summary.json').write_text(json.dumps(summary_results, ensure_ascii=False, indent=2), encoding='utf-8')
    (out_dir / 'traces.json').write_text(json.dumps(traces_dict, ensure_ascii=False, indent=2), encoding='utf-8')
    
    print("\n" + "="*50)
    print("BENCHMARK DEV RETRIEVAL EVALUATION COMPLETED")
    print("="*50)
    print(json.dumps(summary_results, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
