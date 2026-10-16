import json
import uuid
from types import SimpleNamespace as NS

import pytest
from openai import APIConnectionError

from studyhelper.assistant import Assistant
from studyhelper.cache import TTLCache
from studyhelper.llm_client import LLMClient, LLMUnavailable
from studyhelper.rag.chunker import chunk_text
from studyhelper.rag.pipeline import RAGPipeline
from studyhelper.rag.vector_store import VectorStore
from studyhelper.rate_limiter import SlidingWindowLimiter
from studyhelper.tools import calculate, study_plan


def _resp(content=None, tool_calls=None, model="m"):
    return NS(choices=[NS(message=NS(content=content, tool_calls=tool_calls,
                                     model_dump=lambda: {}))], model_used=model)


class FakeLLM:
    """Deterministic stand-in: bag-of-words embeddings + scripted chat replies."""
    VOCAB = ["overfitting", "precision", "gradient", "index", "normalization", "recall"]

    def __init__(self, replies=None):
        self.replies = list(replies or [])
        self.calls = []

    def embed(self, texts):
        return [[float(t.lower().count(w)) + 0.01 for w in self.VOCAB] for t in texts]

    def chat(self, messages, **kw):
        self.calls.append(kw)
        return self.replies.pop(0)


def make_rag(llm):
    return RAGPipeline(llm, VectorStore(llm, path=":memory:", name=f"t_{uuid.uuid4().hex}"))


def test_chunker_overlap_and_size():
    text = " ".join(f"word{i}" for i in range(300))
    chunks = chunk_text(text, size=200, overlap=40)
    assert len(chunks) > 1 and all(len(c) <= 210 for c in chunks)
    assert set(chunks[0].split()[-2:]) & set(chunks[1].split())  # overlap present
    with pytest.raises(ValueError):
        chunk_text("x", size=10, overlap=10)


def test_rag_retrieves_relevant_chunk():
    rag = make_rag(FakeLLM())
    rag.ingest_text("overfitting is when a model memorises training data", "ml.txt")
    rag.ingest_text("an index speeds up reads on a table", "db.txt")
    top = rag.retrieve("what is overfitting", k=1)[0]
    assert top["source"] == "ml.txt"


def test_calculate_is_safe():
    assert calculate("(125000*0.15)/12") == "1562.5"
    assert calculate("__import__('os').system('ls')").startswith("error")


def test_study_plan_validation():
    assert study_plan("2000-01-01", ["a"]).startswith("error")
    assert study_plan("not-a-date", ["a"]).startswith("error")


def test_cache_ttl_and_hits():
    c = TTLCache(ttl=100)
    k = c.key("q")
    assert c.get(k) is None
    c.set(k, 1)
    assert c.get(k) == 1 and c.hits == 1
    expired = TTLCache(ttl=0)
    expired.set("k", 1)
    assert expired.get("k") is None


def test_rate_limiter_blocks_then_reports_retry_after():
    rl = SlidingWindowLimiter(limit=2, window=60)
    assert rl.check("a")[0] and rl.check("a")[0]
    ok, retry = rl.check("a")
    assert not ok and retry >= 1
    assert rl.check("b")[0]  # separate client unaffected


def test_tool_loop_calls_tool_then_answers():
    tc = NS(id="1", function=NS(name="calculate", arguments=json.dumps({"expression": "2+2"})),
            model_dump=lambda: {"id": "1", "type": "function",
                                "function": {"name": "calculate", "arguments": '{"expression": "2+2"}'}})
    llm = FakeLLM([_resp(tool_calls=[tc]), _resp(content="The answer is 4")])
    out = Assistant(llm, make_rag(llm)).chat("2+2?")
    assert out["answer"] == "The answer is 4"
    assert out["tool_trace"][0]["result"] == "4"


def test_tool_loop_stops_at_limit():
    tc = NS(id="1", function=NS(name="calculate", arguments='{"expression": "1+1"}'),
            model_dump=lambda: {})
    llm = FakeLLM([_resp(tool_calls=[tc]) for _ in range(10)])
    out = Assistant(llm, make_rag(llm)).chat("loop forever")
    assert "limit" in out["answer"] and len(out["tool_trace"]) == 4


VALID = {"topic": "t", "summary": "s", "key_points": ["a"], "practice_questions": ["q"], "difficulty": "easy"}


def test_study_card_repairs_invalid_json():
    llm = FakeLLM([_resp(content="not json"), _resp(content=json.dumps(VALID))])
    card = Assistant(llm, make_rag(llm)).study_card("t")
    assert card.difficulty == "easy" and len(llm.calls) == 2


def test_study_card_raises_after_two_bad_replies():
    llm = FakeLLM([_resp(content="{}"), _resp(content="{}")])
    with pytest.raises(ValueError):
        Assistant(llm, make_rag(llm)).study_card("t")


def _flaky_client(fail_models):
    calls = []

    def create(model, **kw):
        calls.append(model)
        if model in fail_models:
            raise APIConnectionError(request=NS())
        return _resp(content="ok")
    return NS(chat=NS(completions=NS(create=create))), calls


def test_fallback_model_used_when_primary_fails():
    from studyhelper.config import settings
    client, calls = _flaky_client({settings.primary_model})
    llm = LLMClient(client, max_retries=1, base_delay=0)
    r = llm.chat([{"role": "user", "content": "hi"}])
    assert r.model_used == settings.fallback_model
    assert calls.count(settings.primary_model) == 2  # initial try + 1 retry


def test_both_models_down_raises_graceful_error():
    from studyhelper.config import settings
    client, _ = _flaky_client({settings.primary_model, settings.fallback_model})
    with pytest.raises(LLMUnavailable):
        LLMClient(client, max_retries=0, base_delay=0).chat([{"role": "user", "content": "hi"}])
