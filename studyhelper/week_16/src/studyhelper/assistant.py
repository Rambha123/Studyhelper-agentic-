import json
import logging

from pydantic import BaseModel, Field, ValidationError

from .llm_client import LLMClient
from .tools import TOOL_SCHEMAS, run_tool

log = logging.getLogger(__name__)
MAX_ITERATIONS = 4
MAX_TOOL_ROUNDS = MAX_ITERATIONS

SYSTEM_PROMPT = (
    "You are StudyHelper, a patient tutor. Use search_notes for questions about the student's material, "
    "calculate for arithmetic, and study_plan for exam schedules. Never guess numbers or dates. "
    "Keep answers short and clear; cite note files when you use them."
)

VERIFY_PROMPT = (
    "You are a strict Self-Check Verifier for StudyHelper.\n"
    "Compare the candidate answer against the provided retrieved sources or tool outputs.\n"
    "Determine if the candidate answer is accurate, fully supported, and requires no further info.\n"
    "Respond ONLY with a JSON object in this exact schema:\n"
    '{"decision": "ACCEPT" | "REVISE" | "SEARCH_AGAIN" | "CLARIFY", '
    '"reason": "short explanation", '
    '"revised_answer": "revised response text if REVISE or CLARIFY, else empty string", '
    '"search_query": "query if SEARCH_AGAIN, else empty string"}'
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

    def verify_answer(self, user_msg: str, candidate_answer: str, tool_outputs: list[dict]) -> dict:
        """Self-check before responding: verify candidate answer against collected tool outputs/sources."""
        if not candidate_answer:
            return {"decision": "SEARCH_AGAIN", "reason": "No candidate answer generated yet", "revised_answer": "", "search_query": user_msg}

        context = "\n".join(
            f"Tool [{t.get('tool')}]: args={t.get('args')} -> result={t.get('result')}"
            for t in tool_outputs
        ) if tool_outputs else "No tool calls or retrieved sources available."

        prompt = (
            f"User Question: {user_msg}\n\n"
            f"Tool Outputs / Retrieved Context:\n{context}\n\n"
            f"Candidate Answer:\n{candidate_answer}"
        )
        try:
            resp = self.llm.chat(
                [{"role": "system", "content": VERIFY_PROMPT}, {"role": "user", "content": prompt}],
                json_mode=True,
                temperature=0.0
            )
            raw = resp.choices[0].message.content or "{}"
            data = json.loads(raw)
            decision = data.get("decision", "ACCEPT").upper()
            if decision not in {"ACCEPT", "REVISE", "SEARCH_AGAIN", "CLARIFY"}:
                decision = "ACCEPT"
            return {
                "decision": decision,
                "reason": data.get("reason", "Verified candidate answer."),
                "revised_answer": data.get("revised_answer", ""),
                "search_query": data.get("search_query", "")
            }
        except Exception as e:
            log.warning("Verification check failed or parsed invalid JSON (%s); defaulting to ACCEPT", e)
            return {"decision": "ACCEPT", "reason": f"Verification fallback: {e}", "revised_answer": "", "search_query": ""}

    def chat(self, user_msg: str, history: list | None = None) -> dict:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, *(history or []),
                    {"role": "user", "content": user_msg}]
        trace = []
        iterations = 0
        last_model = None

        while iterations < MAX_ITERATIONS:
            iterations += 1
            resp = self.llm.chat(messages, tools=TOOL_SCHEMAS)
            last_model = resp.model_used
            msg = resp.choices[0].message

            if msg.tool_calls:
                messages.append({"role": "assistant", "content": msg.content or "",
                                 "tool_calls": [tc.model_dump() for tc in msg.tool_calls]})
                for tc in msg.tool_calls:
                    try:
                        args = json.loads(tc.function.arguments or "{}")
                    except json.JSONDecodeError:
                        args = {}
                    result = run_tool(tc.function.name, args, self.rag)
                    capped_result = result[:350] + "..." if len(result) > 350 else result
                    trace.append({"tool": tc.function.name, "args": args, "result": capped_result})
                    messages.append({"role": "tool", "tool_call_id": tc.id, "content": capped_result})
                continue

            # Assistant produced a direct text candidate answer -> Run Self-check verification
            candidate = msg.content or ""
            verification = self.verify_answer(user_msg, candidate, trace)
            decision = verification["decision"]

            if decision == "ACCEPT":
                return {
                    "answer": candidate,
                    "tool_trace": trace,
                    "model": last_model,
                    "iterations": iterations,
                    "verification": verification
                }
            elif decision in {"REVISE", "CLARIFY"}:
                final_text = verification.get("revised_answer") or candidate
                return {
                    "answer": final_text,
                    "tool_trace": trace,
                    "model": last_model,
                    "iterations": iterations,
                    "verification": verification
                }
            elif decision == "SEARCH_AGAIN":
                search_q = verification.get("search_query") or user_msg
                messages.append({"role": "assistant", "content": candidate})
                messages.append({
                    "role": "user",
                    "content": f"Self-check verification requested further search: {verification.get('reason')}. Please search for '{search_q}'."
                })
                continue

        return {"answer": "I couldn't finish within the iteration limit. Try a simpler question.",
                "tool_trace": trace, "model": last_model, "iterations": iterations}

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
