import json
import logging

from pydantic import BaseModel, Field, ValidationError

from .llm_client import LLMClient
from .tools import TOOL_SCHEMAS, run_tool

log = logging.getLogger(__name__)
MAX_TOOL_ROUNDS = 4

SYSTEM_PROMPT = (
    "You are StudyHelper, a patient tutor. Use search_notes for questions about the student's material, "
    "calculate for arithmetic, and study_plan for exam schedules. Never guess numbers or dates. "
    "Keep answers short and clear; cite note files when you use them."
)


class StudyCard(BaseModel):
    topic: str
    summary: str
    key_points: list[str] = Field(min_length=1)
    practice_questions: list[str] = Field(min_length=1)
    difficulty: str = Field(pattern="^(easy|medium|hard)$")


class Assistant:
    def __init__(self, llm: LLMClient, rag):
        self.llm, self.rag = llm, rag

    def chat(self, user_msg: str, history: list | None = None) -> dict:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, *(history or []),
                    {"role": "user", "content": user_msg}]
        trace = []
        for _ in range(MAX_TOOL_ROUNDS):
            resp = self.llm.chat(messages, tools=TOOL_SCHEMAS)
            msg = resp.choices[0].message
            if not msg.tool_calls:
                return {"answer": msg.content, "tool_trace": trace, "model": resp.model_used}
            messages.append({"role": "assistant", "content": msg.content or "",
                             "tool_calls": [tc.model_dump() for tc in msg.tool_calls]})
            for tc in msg.tool_calls:
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                result = run_tool(tc.function.name, args, self.rag)
                trace.append({"tool": tc.function.name, "args": args, "result": result[:300]})
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
        return {"answer": "I couldn't finish within the tool-call limit. Try a simpler question.",
                "tool_trace": trace, "model": None}

    def study_card(self, topic: str) -> StudyCard:
        """Structured output: JSON mode + pydantic validation + one repair attempt."""
        hits = self.rag.retrieve(topic)
        notes = "\n".join(h["text"] for h in hits) or "(no notes found)"
        schema = StudyCard.model_json_schema()
        messages = [
            {"role": "system", "content": "Return ONLY valid JSON matching this schema: " + json.dumps(schema)},
            {"role": "user", "content": f"Topic: {topic}\nNotes:\n{notes}"},
        ]
        last_err = ""
        for _ in range(2):
            resp = self.llm.chat(messages, json_mode=True, temperature=0.1)
            raw = resp.choices[0].message.content
            try:
                return StudyCard.model_validate_json(raw)
            except ValidationError as e:
                last_err = str(e)
                messages += [{"role": "assistant", "content": raw},
                             {"role": "user", "content": f"Invalid: {last_err[:300]}. Fix and return only JSON."}]
        raise ValueError(f"Model did not return valid JSON: {last_err[:200]}")
