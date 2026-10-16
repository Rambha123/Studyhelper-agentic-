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
        raw_hits = self.store.search(query, k or settings.top_k)
        return self.rerank_and_cap_hits(query, raw_hits, top_k=3, max_chars=350)

    @staticmethod
    def rerank_and_cap_hits(query: str, hits: list[dict], top_k: int = 3, max_chars: int = 350) -> list[dict]:
        """Context engineering: re-rank retrieval hits by term overlap & score, cap count and truncate text."""
        if not hits:
            return []
        q_terms = set(query.lower().split())

        def score(h: dict) -> float:
            text = h.get("text", "").lower()
            term_matches = sum(1 for t in q_terms if t in text)
            dist = h.get("distance", 1.0) or 1.0
            return (term_matches * 2.0) - dist

        sorted_hits = sorted(hits, key=score, reverse=True)
        capped = sorted_hits[:top_k]

        processed = []
        for h in capped:
            item = dict(h)
            text = item.get("text", "")
            if len(text) > max_chars:
                item["text"] = text[:max_chars] + "..."
            processed.append(item)
        return processed

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
