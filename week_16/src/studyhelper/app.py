import asyncio
import logging

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from .assistant import Assistant
from .cache import TTLCache
from .config import settings
from .llm_client import LLMClient, LLMUnavailable
from .rag.pipeline import RAGPipeline
from .rate_limiter import SlidingWindowLimiter

logging.basicConfig(level=logging.INFO)
app = FastAPI(title="StudyHelper API")

llm = LLMClient()
rag = RAGPipeline(llm)
assistant = Assistant(llm, rag)
cache = TTLCache(ttl=settings.cache_ttl)
limiter = SlidingWindowLimiter(limit=settings.rate_limit_rpm)
sem = asyncio.Semaphore(settings.max_concurrency)


@app.middleware("http")
async def rate_limit(request: Request, call_next):
    if request.url.path != "/health":
        ok, retry = limiter.check(request.client.host if request.client else "anon")
        if not ok:
            return JSONResponse({"detail": "Too many requests"}, status_code=429,
                                headers={"Retry-After": str(retry)})
    return await call_next(request)


class ChatIn(BaseModel):
    message: str
    history: list[dict] = []


class TopicIn(BaseModel):
    topic: str


class DocIn(BaseModel):
    name: str | None = None
    text: str | None = None


async def _guarded(fn, *args):
    """Run blocking work off the event loop, bounded by a semaphore; degrade gracefully."""
    async with sem:
        try:
            return await asyncio.to_thread(fn, *args)
        except LLMUnavailable:
            raise HTTPException(503, "The AI service is temporarily unavailable. Please try again shortly.")
        except ValueError as e:
            raise HTTPException(502, str(e))


@app.get("/health")
def health():
    return {"status": "ok", "notes_indexed": rag.store.col.count(),
            "cache": {"hits": cache.hits, "misses": cache.misses}}


@app.post("/ingest")
async def ingest(doc: DocIn | None = None):
    if doc and doc.text:
        n = await _guarded(rag.ingest_text, doc.text, doc.name or "pasted_notes.txt")
    else:
        n = await _guarded(rag.ingest_dir)
    cache._data.clear()  # notes changed -> cached answers are stale
    return {"chunks_indexed": n}


@app.post("/ask")
async def ask(body: ChatIn):
    key = cache.key("ask", body.message)
    if (hit := cache.get(key)) is not None:
        return {**hit, "cached": True}
    result = await _guarded(rag.answer, body.message)
    cache.set(key, result)
    return {**result, "cached": False}


@app.post("/chat")
async def chat(body: ChatIn):
    return await _guarded(assistant.chat, body.message, body.history)


@app.post("/study-card")
async def study_card(body: TopicIn):
    key = cache.key("card", body.topic)
    if (hit := cache.get(key)) is not None:
        return {**hit, "cached": True}
    card = await _guarded(assistant.study_card, body.topic)
    data = card.model_dump()
    cache.set(key, data)
    return {**data, "cached": False}