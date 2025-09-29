import psycopg2
import psycopg2.extras
import pgvector.psycopg2
import os
import time
import math
from contextlib import asynccontextmanager
from typing import List
from dotenv import load_dotenv
from psycopg2 import sql, OperationalError
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from openai import OpenAI
import PyPDF2
from fastapi import HTTPException
from time import sleep
from psycopg2.extras import Json
import io

load_dotenv()

# ---------- Config ----------
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", 5432))
DB_NAME = os.getenv("DB_NAME", "rag_db")
DB_USER = os.getenv("DB_USER", "rag_user")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", 1536))

# allow running locally with DB_HOST=postgres in docker-compose
if DB_HOST and DB_HOST.lower() == "postgres" and (os.getenv("VIRTUAL_ENV") or os.getenv("CONDA_PREFIX")):
    DB_HOST = "localhost"

client = OpenAI(api_key=OPENAI_API_KEY)

# ---------- DB helpers ----------
def get_conn(retries=8, delay=2):
    last_exc = None
    for attempt in range(1, retries + 1):
        try:
            conn = psycopg2.connect(
                host=DB_HOST,
                port=DB_PORT,
                dbname=DB_NAME,
                user=DB_USER,
                password=DB_PASSWORD,
            )
            conn.autocommit = True

            # register the pgvector adapter so Python lists are sent as vector
            pgvector.psycopg2.register_vector(conn)

            return conn
        except OperationalError as e:
            last_exc = e
            print(f"[db] connect attempt {attempt}/{retries} failed: {e}")
            time.sleep(delay)
    raise last_exc

def ensure_schema():
    conn = get_conn()
    with conn, conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        cur.execute(sql.SQL("""
            CREATE TABLE IF NOT EXISTS chunks (
                id SERIAL PRIMARY KEY,
                source TEXT,
                chunk_text TEXT,
                metadata JSONB,
                embedding VECTOR({dim}),
                created_at TIMESTAMPTZ DEFAULT now()
            );
        """).format(dim=sql.Literal(EMBEDDING_DIM)))
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_chunks_embedding ON chunks
            USING ivfflat (embedding) WITH (lists = 100);
        """)
    conn.close()

# ---------- simple PDF/text chunker ----------
def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    """
    Read PDF from bytes using a BytesIO stream. Works across PyPDF2 versions.
    Returns the concatenated text of all pages.
    """
    try:
        reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
    except Exception as e:
        # fallback: try opening from temporary file (very rare)
        raise RuntimeError(f"Failed to open PDF: {e}")

    pages = []
    for p in reader.pages:
        try:
            pages.append(p.extract_text() or "")
        except Exception:
            pages.append("")
    return "\n".join(pages)

def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> List[str]:
    # naive chunker by characters (keeps words intact roughly)
    text = text.replace("\r\n", " ").replace("\n", " ")
    chunks = []
    start = 0
    length = len(text)
    while start < length:
        end = start + chunk_size
        if end < length:
            # try to backtrack to nearest space to avoid cutting mid-word
            sep = text.rfind(" ", start, end)
            if sep > start:
                end = sep
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - overlap
        if start < 0:
            start = 0
        if start >= length:
            break
    return chunks

# ---------- OpenAI helpers ----------
def make_embedding(text: str) -> List[float]:
    resp = client.embeddings.create(input=text, model="text-embedding-3-small")
    emb = resp.data[0].embedding
    return emb

def generate_answer_from_chunks(question: str, chunks: List[dict]) -> str:
    """
    Simple prompt composition: include top chunks (source + text) and ask model.
    You can swap model name to a chat model you prefer.
    """
    # build context string with top chunks
    context_parts = []
    for i, c in enumerate(chunks, 1):
        context_parts.append(f"Source {c.get('source','unknown')} (id={c.get('id')}):\n{c.get('chunk_text')}\n")
    context = "\n\n".join(context_parts)

    prompt = ( "You are a helpful assistant. Use ONLY the context below to answer the question. " "If the context doesn't contain the answer, say you don't know.\n\n" f"CONTEXT:\n{context}\n\nQUESTION:\n{question}\n\nAnswer concisely." )


    # Use an LLM to generate final answer. Replace model name if needed.
    chat_resp = client.chat.completions.create(
        model="gpt-4o-mini",  # change to a model you have access to, e.g., "gpt-4o-mini"
        messages=[{"role":"user", "content": prompt}],
        max_tokens=512,
        temperature=0.0,
    )
    # The exact response structure may vary by client version; common pattern:
    answer = chat_resp.choices[0].message.content
    return answer

# ---------- FastAPI app ----------
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[api] startup: ensuring DB schema")
    ensure_schema()
    yield
    print("[api] shutdown")

app = FastAPI(lifespan=lifespan)

class EmbedTextRequest(BaseModel):
    text: str
    source: str = "api"
    metadata: dict = {}

class QueryRequest(BaseModel):
    query: str
    top_k: int = 3

@app.get("/")
def root():
    return {"status": "ok", "info": "RAG API running"}

@app.post("/embed_text")
def embed_text(req: EmbedTextRequest):
    conn = get_conn()
    with conn.cursor() as cur:
        emb = make_embedding(req.text)
        cur.execute(
            "INSERT INTO chunks (source, chunk_text, metadata, embedding) VALUES (%s,%s,%s,%s) RETURNING id;",
            (req.source, req.text, psycopg2.extras.Json(req.metadata), emb),)
        inserted_id = cur.fetchone()[0]
    conn.close()
    return {"status": "inserted", "id": inserted_id}

@app.post("/embed_file")
async def embed_file(source: str = "upload", file: UploadFile = File(...)):
    data = await file.read()
    text = extract_text_from_pdf_bytes(data)
    chunks = chunk_text(text, chunk_size=1000, overlap=200)
    conn = get_conn()
    inserted = 0

    try:
        with conn.cursor() as cur:
            for i, chunk in enumerate(chunks):
                # optional: skip empty chunks (rare)
                if not chunk.strip():
                    continue

                # generate embedding (consider batching for many chunks)
                emb = make_embedding(chunk)

                metadata = {"page_chunk_index": i, "filename": file.filename}
                cur.execute(
                    "INSERT INTO chunks (source, chunk_text, metadata, embedding) VALUES (%s,%s,%s,%s) RETURNING id;",
                    (source, chunk, Json(metadata), emb),
                )
                _ = cur.fetchone()[0]
                inserted += 1

                # optional tiny sleep to avoid burst rate limits (tweak or remove)
                # sleep(0.1)

    finally:
        conn.close()

    return {"status": "ok", "inserted_chunks": inserted, "filename": file.filename}

@app.post("/query")
def query(req: QueryRequest):
    # Ensure top_k is defined
    top_k = int(req.top_k) if req.top_k and int(req.top_k) > 0 else 3

    # Create query embedding (Python list)
    q_emb = make_embedding(req.query)
    
    # Convert to PostgreSQL vector format to ensure compatibility
    q_emb_str = '[' + ','.join(map(str, q_emb)) + ']'

    conn = get_conn()
    try:
        with conn.cursor() as cur:
            # Select real columns; ensure we pass Python list (pgvector registered)
            cur.execute(
                """
                SELECT id, source, chunk_text, metadata,
                       embedding <=> %s::vector AS distance
                FROM chunks
                ORDER BY distance
                LIMIT %s;
                """,
                (q_emb_str, top_k),
            )
            rows = cur.fetchall()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()

    # Convert rows to dicts for JSON response
    chunks = []
    for r in rows:
        chunks.append({
            "id": r[0],
            "source": r[1],
            "chunk_text": r[2],
            "metadata": r[3],
            "distance": float(r[4]),
        })

    # Generate answer from retrieved chunks
    answer = generate_answer_from_chunks(req.query, chunks)
    return {"answer": answer, "retrieved": chunks}