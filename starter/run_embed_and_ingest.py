import os, sys, json, time
from pathlib import Path
from dotenv import load_dotenv

sys.path.insert(0, 'F:/VinAI/Lab6/K5-L3A-DAY06-2A202603581-DataIngestion/starter')
load_dotenv('F:/VinAI/Lab6/K5-L3A-DAY06-2A202603581-DataIngestion/starter/.env')

from src.embeddings import MyEmbedder
from src.store import PgVectorStore
from src.models import Chunk

log_file = Path('F:/VinAI/Lab7/data/embed_progress.log')

def log(msg):
    text = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n"
    print(text, end='', flush=True)
    with log_file.open('a', encoding='utf-8') as f:
        f.write(text)

chunks_raw = json.loads(Path('F:/VinAI/Lab7/data/chunks.json').read_text(encoding='utf-8'))
chunks = [Chunk(chunk_id=c['id'], document_id=c['source_id'], source=c['source'], page=c['page'], text=c['text'], metadata=c['metadata']) for c in chunks_raw]

database_url = os.getenv('DATABASE_URL')
model_name = os.getenv('EMBEDDING_MODEL')

store = PgVectorStore(database_url, dimensions=768)
store.create_schema()

cache_path = Path('F:/VinAI/Lab7/data/chunk_embeddings.json')
if cache_path.exists():
    vectors = json.loads(cache_path.read_text(encoding='utf-8'))
else:
    vectors = []

log(f"Resuming with {len(vectors)}/{len(chunks)} cached embeddings.")

embedder = MyEmbedder(model_name=model_name, dimensions=768, batch_size=32)
texts = [c.text for c in chunks]

batch_size = 32
for i in range(len(vectors), len(texts), batch_size):
    batch = texts[i:i+batch_size]
    success = False
    for attempt in range(8):
        try:
            v_batch = embedder.embed_documents(batch)
            vectors.extend(v_batch)
            log(f"Embedded batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1} ({len(vectors)}/{len(texts)} chunks)")
            time.sleep(1.0)
            success = True
            break
        except Exception as e:
            wait_s = 25 if "RESOURCE_EXHAUSTED" in str(e) or "429" in str(e) else 10
            log(f"Rate limit / error on attempt {attempt+1}: {e}. Retrying after {wait_s}s...")
            time.sleep(wait_s)
    
    if not success:
        log("Fatal error: failed after multiple attempts.")
        break
    
    if len(vectors) % 160 == 0 or len(vectors) == len(texts):
        cache_path.write_text(json.dumps(vectors), encoding='utf-8')
        log(f"Checkpoint saved to {cache_path} ({len(vectors)} vectors)")

if len(vectors) == len(chunks):
    log("All vectors embedded! Upserting to PgVector strategy B...")
    store.upsert('B', chunks, vectors)
    log("Upsert to PgVector successfully completed!")
