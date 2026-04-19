"""
ingestion.py — Document loading, chunking, embedding, and ChromaDB indexing.
 
Chunking strategy:
  - Chunk size: 400 tokens (~300 words). Small enough to be specific,
    large enough to carry full sentences and context.
  - Overlap: 80 tokens (~60 words). Prevents boundary cuts from losing
    cross-sentence context.
  - Splitter: Recursive character splitter on ["\n\n", "\n", ". ", " "]
    so we prefer paragraph > sentence > word breaks in that order.
"""
 
import os
import re
import fitz  # PyMuPDF
import chromadb
from chromadb.utils import embedding_functions
from tqdm import tqdm
from pathlib import Path
 
 
# ── Constants ────────────────────────────────────────────────────────────────
CHUNK_SIZE    = 400   # characters (approx 80-100 words)
CHUNK_OVERLAP = 80    # characters
EMBED_MODEL   = "all-MiniLM-L6-v2"   # fast, free, good quality
COLLECTION    = "ai_regulations"
DB_PATH       = "./chroma_db"
 
 
# ── Text extraction ───────────────────────────────────────────────────────────
def extract_text_from_pdf(path: str) -> str:
    """Extract raw text from a PDF, preserving paragraph breaks."""
    doc = fitz.open(path)
    pages = []
    for page in doc:
        text = page.get_text("text")
        pages.append(text)
    return "\n\n".join(pages)
 
 
def extract_text_from_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()
 
 
def load_document(path: str) -> tuple[str, str]:
    """Return (text, source_name) for any supported file."""
    p = Path(path)
    if p.suffix.lower() == ".pdf":
        text = extract_text_from_pdf(path)
    elif p.suffix.lower() in (".txt", ".md"):
        text = extract_text_from_txt(path)
    else:
        raise ValueError(f"Unsupported file type: {p.suffix}")
    # Basic cleanup: collapse excessive whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip(), p.stem
 
 
# ── Chunking ─────────────────────────────────────────────────────────────────
def recursive_split(text: str, size: int, overlap: int) -> list[str]:
    """
    Recursively split text using preferred separators.
    Falls back to hard character split if no separator found.
    """
    separators = ["\n\n", "\n", ". ", " "]
 
    def split_with(text, sep):
        return [s.strip() for s in text.split(sep) if s.strip()]
 
    def chunk(text):
        if len(text) <= size:
            return [text]
        # Try each separator
        for sep in separators:
            parts = split_with(text, sep)
            if len(parts) > 1:
                chunks, current = [], ""
                for part in parts:
                    candidate = (current + sep + part).strip() if current else part
                    if len(candidate) <= size:
                        current = candidate
                    else:
                        if current:
                            chunks.append(current)
                        # If part itself is too big, recurse
                        if len(part) > size:
                            chunks.extend(chunk(part))
                            current = ""
                        else:
                            current = part
                if current:
                    chunks.append(current)
                return chunks
        # Hard split as last resort
        return [text[i:i+size] for i in range(0, len(text), size - overlap)]
 
    raw_chunks = chunk(text)
 
    # Apply overlap: prepend tail of previous chunk
    overlapped = []
    for i, ch in enumerate(raw_chunks):
        if i == 0:
            overlapped.append(ch)
        else:
            tail = raw_chunks[i-1][-overlap:]
            overlapped.append((tail + " " + ch).strip())
 
    return overlapped
 
 
def chunk_document(text: str, source: str) -> list[dict]:
    """Return list of {text, source, chunk_id} dicts."""
    chunks = recursive_split(text, CHUNK_SIZE, CHUNK_OVERLAP)
    return [
        {
            "text": ch,
            "source": source,
            "chunk_id": f"{source}_chunk_{i}",
        }
        for i, ch in enumerate(chunks)
    ]
 
 
# ── ChromaDB indexing ─────────────────────────────────────────────────────────
def get_collection(reset: bool = False):
    """Return (or create) the ChromaDB collection."""
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBED_MODEL
    )
    client = chromadb.PersistentClient(path=DB_PATH)
    if reset and COLLECTION in [c.name for c in client.list_collections()]:
        client.delete_collection(COLLECTION)
    collection = client.get_or_create_collection(
        name=COLLECTION,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )
    return collection
 
 
def index_documents(data_dir: str, reset: bool = True):
    """
    Load all PDFs/TXTs from data_dir, chunk them, and index into ChromaDB.
    Set reset=True to rebuild the index from scratch.
    """
    data_path = Path(data_dir)
    files = list(data_path.glob("*.pdf")) + list(data_path.glob("*.txt"))
 
    if not files:
        raise FileNotFoundError(f"No PDF or TXT files found in {data_dir}")
 
    print(f"Found {len(files)} document(s): {[f.name for f in files]}")
    collection = get_collection(reset=reset)
 
    all_chunks = []
    for filepath in files:
        print(f"  Loading: {filepath.name}")
        text, source = load_document(str(filepath))
        chunks = chunk_document(text, source)
        all_chunks.extend(chunks)
        print(f"    → {len(chunks)} chunks")
 
    print(f"\nIndexing {len(all_chunks)} total chunks into ChromaDB...")
    # ChromaDB add in batches of 500
    batch_size = 500
    for i in tqdm(range(0, len(all_chunks), batch_size)):
        batch = all_chunks[i:i+batch_size]
        collection.add(
            documents=[c["text"] for c in batch],
            ids=[c["chunk_id"] for c in batch],
            metadatas=[{"source": c["source"]} for c in batch],
        )
 
    print(f"✓ Indexed {collection.count()} chunks total.")
    return collection
 
 
# ── Retrieval helper ──────────────────────────────────────────────────────────
def retrieve(query: str, collection, top_k: int = 5) -> list[dict]:
    """
    Query ChromaDB and return top_k results as list of dicts:
    {text, source, distance, score}
    score = 1 - distance  (cosine similarity, higher is better)
    """
    results = collection.query(
        query_texts=[query],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        chunks.append({
            "text": doc,
            "source": meta["source"],
            "distance": dist,
            "score": round(1 - dist, 4),
        })
    return chunks
 
 
if __name__ == "__main__":
    import sys
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "./data"
    index_documents(data_dir, reset=True)
 