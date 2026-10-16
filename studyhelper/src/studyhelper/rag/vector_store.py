import chromadb

from ..config import settings


class VectorStore:
    """ChromaDB-backed store. Embeddings are produced by the LLM client (Gemini)."""

    def __init__(self, llm, path: str | None = None, name: str = "study_notes"):
        self.llm = llm
        self.client = chromadb.PersistentClient(path=path or settings.chroma_dir) if path != ":memory:" \
            else chromadb.EphemeralClient()
        self.col = self.client.get_or_create_collection(name, metadata={"hnsw:space": "cosine"})

    def reset(self):
        self.client.delete_collection(self.col.name)
        self.col = self.client.get_or_create_collection(self.col.name, metadata={"hnsw:space": "cosine"})

    def add(self, chunks: list[str], source: str) -> int:
        if not chunks:
            return 0
        embs = self.llm.embed(chunks)
        ids = [f"{source}::{i}" for i in range(len(chunks))]
        self.col.upsert(ids=ids, documents=chunks, embeddings=embs,
                        metadatas=[{"source": source, "chunk": i} for i in range(len(chunks))])
        return len(chunks)

    def search(self, query: str, k: int = 4) -> list[dict]:
        if self.col.count() == 0:
            return []
        emb = self.llm.embed([query])[0]
        r = self.col.query(query_embeddings=[emb], n_results=min(k, self.col.count()))
        return [
            {"text": d, "source": m["source"], "score": round(1 - dist, 3)}
            for d, m, dist in zip(r["documents"][0], r["metadatas"][0], r["distances"][0])
        ]
