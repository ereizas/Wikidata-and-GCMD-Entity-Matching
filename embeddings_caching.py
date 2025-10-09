import sqlite3
import hashlib
import json
from google import genai
import time

def init_db(db_path="embeddings_cache.db"):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS embeddings_cache (
        id TEXT PRIMARY KEY,
        text_hash TEXT,
        embedding TEXT
    )
    """)
    conn.commit()
    conn.close()

def get_text_hash(text: str):
    """Compute stable hash for deduplication."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def get_cached_embedding(text, db_path="embeddings_cache.db"):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    text_hash = get_text_hash(text)
    cur.execute("SELECT embedding FROM embeddings_cache WHERE text_hash=?", (text_hash,))
    row = cur.fetchone()
    conn.close()
    if row:
        return json.loads(row[0])  # convert string back to list
    return None

def save_embedding(text, embedding, db_path="embeddings_cache.db"):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    text_hash = get_text_hash(text)
    cur.execute(
        "INSERT OR REPLACE INTO embeddings_cache (id, text_hash, embedding) VALUES (?, ?, ?)",
        (text_hash, text_hash, json.dumps(embedding))
    )
    conn.commit()
    conn.close()

def batch_embeddings_with_cache(texts, api_key, db_path="embeddings_cache.db",
                                model="gemini-embedding-001", batch_size=10, delay=7):
    embeddings = []
    uncached_texts = []
    uncached_indices = []

    # Step 1: Try loading from cache
    for i, text in enumerate(texts):
        cached = get_cached_embedding(text, db_path)
        if cached:
            embeddings.append(cached)
        else:
            embeddings.append(None)  # placeholder
            uncached_texts.append(text)
            uncached_indices.append(i)
    """print(len(embeddings)-embeddings.count(None))
    print(len(uncached_texts))"""
    # Step 2: Fetch missing ones from API in batches
    for i in range(0, len(uncached_texts), batch_size):
        batch = uncached_texts[i:i + batch_size]
        client = genai.Client(api_key=api_key)
        result = client.models.embed_content(
            model=model,
            contents=batch
        )

        new_embeddings = [e.values for e in result.embeddings]
        for j, emb in enumerate(new_embeddings):
            idx = uncached_indices[i + j]
            embeddings[idx] = emb
            save_embedding(uncached_texts[i + j], emb, db_path)
        time.sleep(delay)

    return embeddings