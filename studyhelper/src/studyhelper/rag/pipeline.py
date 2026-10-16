from pathlib import Path

from ..config import settings
from .chunker import chunk_text
from .vector_store import VectorStore


class RAGPipeline:
    def __init__(self, llm, store: VectorStore | None = None):
        self.llm = llm
        self.store = store or VectorStore(llm)

    def ingest_text(self, text: str, source: str) -> int:
        return self.store.add(chunk_text(text, settings.chunk_size, settings.chunk_overlap), source)

    def ingest_dir(self, path: str | None = None) -> int:
        total = 0
        for f in sorted(Path(path or settings.docs_dir).glob("**/*")):
            if f.suffix.lower() in {".txt", ".md"}:
                total += self.ingest_text(f.read_text(encoding="utf-8"), f.name)
        return total

    def retrieve(self, query: str, k: int | None = None) -> list[dict]:
        return self.store.search(query, k or settings.top_k)

    def answer(self, question: str) -> dict:
        hits = self.retrieve(question)
        context = "\n\n".join(f"[{h['source']}] {h['text']}" for h in hits)
        messages = [
            {"role": "system", "content": "You are a study helper. Answer ONLY from the provided notes. "
             "If the notes do not contain the answer, say so. Cite sources like [file.txt]."},
            {"role": "user", "content": f"Notes:\n{context}\n\nQuestion: {question}"},
        ]
        resp = self.llm.chat(messages)
        return {"answer": resp.choices[0].message.content, "sources": hits, "model": resp.model_used}
