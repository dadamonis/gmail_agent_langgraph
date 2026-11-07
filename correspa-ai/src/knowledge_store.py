import sqlite3
from array import array
from math import sqrt
from pathlib import Path
from langchain_core.documents import Document

DB_PATH = Path(__file__).resolve().parent.parent / "db" / "knowledge.sqlite3"
EMBEDDING_MODEL = "models/text-embedding-004"

SCHEMA = """
CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE chunks(id INTEGER PRIMARY KEY, source TEXT NOT NULL,
  content TEXT NOT NULL, embedding BLOB NOT NULL);
CREATE VIRTUAL TABLE chunks_fts USING fts5(content, content='chunks',
  content_rowid='id', tokenize='porter unicode61');
PRAGMA user_version = 1;
"""

def _connect():
    conn = sqlite3.connect(DB_PATH)
    if conn.execute("PRAGMA user_version").fetchone()[0] == 0:
        conn.executescript(SCHEMA)
    return conn

def _unit(vec):
    norm = sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]

def ingest(docs, embeddings):
    """Replace the whole store with `docs` (LangChain Documents)."""
    vectors = embeddings.embed_documents([d.page_content for d in docs])
    conn = _connect()
    try:
        with conn:
            conn.execute("DELETE FROM chunks")
            conn.executemany(
                "INSERT INTO chunks(source, content, embedding) VALUES (?, ?, ?)",
                [
                    (d.metadata.get("source", ""), d.page_content, array("f", v).tobytes())
                    for d, v in zip(docs, vectors)
                ],
            )
            conn.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('rebuild')")
            conn.executemany(
                "INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)",
                [("embedding_model", EMBEDDING_MODEL), ("dim", str(len(vectors[0]) if vectors else 0))],
            )
    finally:
        conn.close()

def search(query, embeddings, k=3):
    """Top-k chunks by cosine similarity, shaped like Chroma's retriever output."""
    if not DB_PATH.exists():
        raise RuntimeError(f"Knowledge store not found at {DB_PATH}; run create_index.py first.")
    # New connection per call: LangServe runs sync runnables in a thread pool.
    conn = _connect()
    try:
        row = conn.execute("SELECT value FROM meta WHERE key = 'embedding_model'").fetchone()
        stored_model = row[0] if row else None
        if stored_model != EMBEDDING_MODEL:
            raise RuntimeError(
                f"Knowledge store was built with embedding model {stored_model!r}, "
                f"but {EMBEDDING_MODEL!r} is configured; re-run create_index.py."
            )
        q = _unit(embeddings.embed_query(query))
        scored = []
        # shortcut: brute-force scan, move to sqlite-vec vec0 past ~10-20k chunks
        for source, content, blob in conn.execute("SELECT source, content, embedding FROM chunks"):
            vec = array("f")
            vec.frombytes(blob)
            scored.append((sum(a * b for a, b in zip(q, _unit(vec))), source, content))
    finally:
        conn.close()
    scored.sort(key=lambda s: s[0], reverse=True)
    return [Document(page_content=c, metadata={"source": s}) for _, s, c in scored[:k]]
