import os, sys, json, time
from pathlib import Path
import requests
from dotenv import load_dotenv

sys.path.insert(0, 'F:/VinAI/Lab6/K5-L3A-DAY06-2A202603581-DataIngestion/starter')
load_dotenv('F:/VinAI/Lab6/K5-L3A-DAY06-2A202603581-DataIngestion/starter/.env')

from src.store import PgVectorStore
from src.models import Chunk

api_key = os.getenv('GEMINI_API_KEY').strip().strip('"')
database_url = os.getenv('DATABASE_URL')
log_file = Path('F:/VinAI/Lab7/data/embed_progress.log')

def log(msg):
    text = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n"
    print(text, end='', flush=True)
    with log_file.open('a', encoding='utf-8') as f:
        f.write(text)

chunks_raw = json.loads(Path('F:/VinAI/Lab7/data/chunks.json').read_text(encoding='utf-8'))
chunks = [Chunk(chunk_id=c['id'], document_id=c['source_id'], source=c['source'], page=c['page'], text=c['text'], metadata=c['metadata']) for c in chunks_raw]

store = PgVectorStore(database_url, dimensions=768)
store.create_schema()

cache_path = Path('F:/VinAI/Lab7/data/chunk_embeddings_v2.json')
if cache_path.exists():
    try:
        vectors = json.loads(cache_path.read_text(encoding='utf-8'))
    except Exception:
        vectors = []
else:
    vectors = []

log(f"Resuming embedding with gemini-embedding-2. Current cache: {len(vectors)}/{len(chunks)}")

url = f'https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-2:batchEmbedContents?key={api_key}'
batch_size = 50
texts = [c.text for c in chunks]

for i in range(len(vectors), len(texts), batch_size):
    batch = texts[i:i+batch_size]
    payload = {
        'requests': [
            {'model': 'models/gemini-embedding-2', 'content': {'parts': [{'text': t}]}, 'outputDimensionality': 768}
            for t in batch
        ]
    }
    
    success = False
    for attempt in range(12):
        try:
            r = requests.post(url, json=payload, timeout=30)
            if r.status_code == 200:
                embs = r.json().get('embeddings', [])
                if len(embs) != len(batch):
                    raise RuntimeError(f"Expected {len(batch)} vectors, got {len(embs)}")
                vectors.extend([e['values'] for e in embs])
                log(f"Embedded {len(vectors)}/{len(texts)} chunks")
                time.sleep(12.0)
                success = True
                break
            elif r.status_code == 429:
                log(f"Per-minute rate limit hit on attempt {attempt+1}. Backing off for 45s...")
                time.sleep(45.0)
            else:
                log(f"HTTP {r.status_code}: {r.text[:200]}. Attempt {attempt+1}. Retrying in 15s...")
                time.sleep(15.0)
        except Exception as e:
            log(f"Exception on attempt {attempt+1}: {e}. Retrying after 15s...")
            time.sleep(15.0)
            
    if not success:
        log("Failed batch after 12 retries. Stopping.")
        break
        
    if len(vectors) % 150 == 0 or len(vectors) == len(texts):
        cache_path.write_text(json.dumps(vectors), encoding='utf-8')
        log(f"Checkpoint saved ({len(vectors)} vectors)")

if len(vectors) == len(chunks):
    cache_path.write_text(json.dumps(vectors), encoding='utf-8')
    log("All vectors generated! Upserting into PgVector table for strategy B...")
    store.upsert('B', chunks, vectors)
    log("Upsert to PgVector SUCCESSFUL!")
