"""Standalone Evaluation Harness built from scratch (Phase 3).

Evaluates a batch of test queries against Assistant, measuring:
1. Task completion rate (Success/Failure)
2. Tool-call correctness (Did it select the right tool with valid arguments?)
3. Trajectory length (Number of iterations required)
4. Token and Cost Accounting (Estimated prompt & completion tokens + cost)
5. Failure log: Strictly classified as 'Hard failure', 'Soft failure', or 'Cascading soft failure'

Outputs results to studyhelper/eval_results.md.
"""

import json
from pathlib import Path
from types import SimpleNamespace as NS

from studyhelper.assistant import Assistant
from studyhelper.llm_client import LLMClient
from studyhelper.rag.pipeline import RAGPipeline
from studyhelper.rag.vector_store import VectorStore

# Pricing for Gemini 1.5 Flash (or equivalent primary model) per 1k tokens
COST_PER_1K_INPUT_TOKENS = 0.000075
COST_PER_1K_OUTPUT_TOKENS = 0.000300


def estimate_tokens(text: str) -> int:
    """Simple token estimation (~4 chars per token)."""
    return max(1, len(text) // 4)


class EvalMockLLM:
    """Mock LLM for reproducible, offline evaluation testing without API calls."""
    def __init__(self):
        self.calls = []

    def chat(self, messages, tools=None, json_mode=False, temperature=None, top_p=None):
        self.calls.append({"messages": messages, "tools": tools, "json_mode": json_mode})
        last_msg = messages[-1]["content"] if messages else ""
        system_msg = messages[0]["content"] if messages else ""

        # Verification check call
        if "Self-Check Verifier" in system_msg:
            if "FAIL_VERIFY" in last_msg:
                return NS(
                    choices=[NS(message=NS(content=json.dumps({
                        "decision": "SEARCH_AGAIN",
                        "reason": "Retrieved answer incomplete.",
                        "revised_answer": "",
                        "search_query": "overfitting definition"
                    })))],
                    model_used="eval-mock-model"
                )
            return NS(
                choices=[NS(message=NS(content=json.dumps({
                    "decision": "ACCEPT",
                    "reason": "Answer is fully supported.",
                    "revised_answer": "",
                    "search_query": ""
                })))],
                model_used="eval-mock-model"
            )

        # Primary Assistant call
        if "2 + 2" in last_msg or "calculate 2+2" in last_msg:
            # Check if previous tool response in messages
            if any(m.get("role") == "tool" for m in messages):
                return NS(
                    choices=[NS(message=NS(content="2 + 2 equals 4.", tool_calls=None))],
                    model_used="eval-mock-model"
                )
            tc = NS(
                id="tc_1",
                function=NS(name="calculate", arguments=json.dumps({"expression": "2+2"})),
                model_dump=lambda: {"id": "tc_1", "type": "function", "function": {"name": "calculate", "arguments": '{"expression": "2+2"}'}}
            )
            return NS(choices=[NS(message=NS(content=None, tool_calls=[tc]))], model_used="eval-mock-model")

        if "exam on 2026-12-01" in last_msg or "study plan" in last_msg:
            if any(m.get("role") == "tool" for m in messages):
                return NS(
                    choices=[NS(message=NS(content="Here is your study plan for Physics and Math.", tool_calls=None))],
                    model_used="eval-mock-model"
                )
            tc = NS(
                id="tc_2",
                function=NS(name="study_plan", arguments=json.dumps({"exam_date": "2026-12-01", "topics": ["Physics", "Math"]})),
                model_dump=lambda: {"id": "tc_2", "type": "function", "function": {"name": "study_plan", "arguments": '{"exam_date": "2026-12-01", "topics": ["Physics", "Math"]}'}}
            )
            return NS(choices=[NS(message=NS(content=None, tool_calls=[tc]))], model_used="eval-mock-model")

        if "overfitting" in last_msg:
            if any(m.get("role") == "tool" for m in messages):
                return NS(
                    choices=[NS(message=NS(content="Overfitting occurs when a model learns training noise. [ml.txt]", tool_calls=None))],
                    model_used="eval-mock-model"
                )
            tc = NS(
                id="tc_3",
                function=NS(name="search_notes", arguments=json.dumps({"query": "overfitting"})),
                model_dump=lambda: {"id": "tc_3", "type": "function", "function": {"name": "search_notes", "arguments": '{"query": "overfitting"}'}}
            )
            return NS(choices=[NS(message=NS(content=None, tool_calls=[tc]))], model_used="eval-mock-model")

        if "INJECTED_FAILURE" in last_msg or "broken tool" in last_msg:
            tc = NS(
                id="tc_4",
                function=NS(name="calculate", arguments=json.dumps({"expression": "1/0"})),
                model_dump=lambda: {"id": "tc_4", "type": "function", "function": {"name": "calculate", "arguments": '{"expression": "1/0"}'}}
            )
            if any(m.get("role") == "tool" for m in messages):
                return NS(
                    choices=[NS(message=NS(content="I encountered an error running the calculation tool. Please check your math expression.", tool_calls=None))],
                    model_used="eval-mock-model"
                )
            return NS(choices=[NS(message=NS(content=None, tool_calls=[tc]))], model_used="eval-mock-model")

        if "invalid tool args" in last_msg:
            # Bad tool call argument schema
            tc = NS(
                id="tc_5",
                function=NS(name="calculate", arguments="invalid json"),
                model_dump=lambda: {"id": "tc_5", "type": "function", "function": {"name": "calculate", "arguments": "invalid json"}}
            )
            if any(m.get("role") == "tool" for m in messages):
                return NS(
                    choices=[NS(message=NS(content="I could not execute the calculation due to invalid arguments.", tool_calls=None))],
                    model_used="eval-mock-model"
                )
            return NS(choices=[NS(message=NS(content=None, tool_calls=[tc]))], model_used="eval-mock-model")

        # Fallback text response
        return NS(choices=[NS(message=NS(content=f"Processed query: {last_msg}", tool_calls=None))], model_used="eval-mock-model")

    def embed(self, texts):
        return [[0.1] * 10 for _ in texts]


TEST_CASES = [
    {
        "id": "Q1_CALC",
        "query": "What is 2 + 2?",
        "expected_tool": "calculate",
        "expected_arg_keys": ["expression"],
        "description": "Basic arithmetic calculation"
    },
    {
        "id": "Q2_PLAN",
        "query": "Make a study plan for Physics and Math exam on 2026-12-01",
        "expected_tool": "study_plan",
        "expected_arg_keys": ["exam_date", "topics"],
        "description": "Exam schedule study planning"
    },
    {
        "id": "Q3_SEARCH",
        "query": "What is overfitting according to notes?",
        "expected_tool": "search_notes",
        "expected_arg_keys": ["query"],
        "description": "Notes retrieval and verification"
    },
    {
        "id": "Q4_FAILURE_INJECTION",
        "query": "Execute calculation INJECTED_FAILURE test",
        "expected_tool": "calculate",
        "expected_arg_keys": ["expression"],
        "description": "Injected tool failure handling"
    },
    {
        "id": "Q5_INVALID_ARGS",
        "query": "Process invalid tool args query",
        "expected_tool": "calculate",
        "expected_arg_keys": ["expression"],
        "description": "Tool argument syntax error handling"
    }
]


def run_evaluation():
    mock_llm = EvalMockLLM()
    vector_store = VectorStore(mock_llm, path=":memory:", name="eval_store")
    rag = RAGPipeline(mock_llm, vector_store)
    rag.ingest_text("Overfitting happens when a machine learning model memorizes noise in training data.", "ml.txt")
    assistant = Assistant(mock_llm, rag)

    results = []
    total_tokens_all = 0
    total_cost_all = 0.0
    success_count = 0

    for tc in TEST_CASES:
        start_calls = len(mock_llm.calls)
        res = assistant.chat(tc["query"])
        end_calls = len(mock_llm.calls)

        # Tool call correctness check
        trace = res.get("tool_trace", [])
        tool_correct = False
        if trace:
            first_tool = trace[0]
            if first_tool["tool"] == tc["expected_tool"]:
                args = first_tool.get("args", {})
                if all(k in args for k in tc["expected_arg_keys"]):
                    tool_correct = True

        # Token & cost accounting
        query_calls = mock_llm.calls[start_calls:end_calls]
        prompt_tokens = sum(estimate_tokens(str(c["messages"])) for c in query_calls)
        completion_tokens = estimate_tokens(res.get("answer", ""))
        total_tokens = prompt_tokens + completion_tokens
        cost = (prompt_tokens / 1000.0 * COST_PER_1K_INPUT_TOKENS) + (completion_tokens / 1000.0 * COST_PER_1K_OUTPUT_TOKENS)

        total_tokens_all += total_tokens
        total_cost_all += cost

        # Failure classification
        # - Hard failure: Exception, crashes, or timeout / max iterations reached without response
        # - Soft failure: Completed but selected wrong tool or bad args
        # - Cascading soft failure: Initial error caused compounding incorrect outputs
        iterations = res.get("iterations", 1)
        answer = res.get("answer", "")
        failure_class = "None"
        success = False

        from studyhelper.assistant import MAX_ITERATIONS
        if iterations >= MAX_ITERATIONS and "limit" in answer:
            failure_class = "Hard failure"
        elif not tool_correct:
            if tc["id"] == "Q5_INVALID_ARGS":
                # Intentionally forced invalid args
                failure_class = "Cascading soft failure"
            else:
                failure_class = "Soft failure"
        else:
            success = True
            success_count += 1

        results.append({
            "id": tc["id"],
            "query": tc["query"],
            "description": tc["description"],
            "success": success,
            "tool_correct": tool_correct,
            "trajectory_length": iterations,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "cost_usd": cost,
            "failure_class": failure_class,
            "answer": answer,
            "trace": trace
        })

    completion_rate = (success_count / len(TEST_CASES)) * 100.0

    # Write Markdown output
    md_lines = [
        "# Week 16 Custom Evaluation Harness Results",
        "",
        "## Summary Metrics",
        f"- **Total Test Queries Evaluated**: {len(TEST_CASES)}",
        f"- **Task Completion Rate**: {completion_rate:.1f}% ({success_count}/{len(TEST_CASES)})",
        f"- **Total Tokens Consumed**: {total_tokens_all}",
        f"- **Total Estimated Cost**: ${total_cost_all:.6f}",
        "",
        "## Query Execution Details",
        "",
        "| Query ID | Description | Success | Tool Correctness | Trajectory Length | Tokens (In/Out/Total) | Cost (USD) | Failure Classification |",
        "|---|---|---|---|---|---|---|---|"
    ]

    for r in results:
        status = "✅ Success" if r["success"] else "❌ Failure"
        tc_status = "✅ Correct" if r["tool_correct"] else "❌ Incorrect"
        tokens_str = f"{r['prompt_tokens']} / {r['completion_tokens']} / {r['total_tokens']}"
        md_lines.append(
            f"| {r['id']} | {r['description']} | {status} | {tc_status} | {r['trajectory_length']} | {tokens_str} | ${r['cost_usd']:.6f} | `{r['failure_class']}` |"
        )

    md_lines.extend([
        "",
        "## Failure Log & Classification Taxonomy",
        "- **Hard failure**: System unhandled exception, crash, or reaching loop iteration limit without output.",
        "- **Soft failure**: System executed query but selected wrong tool or failed parameter validation schema.",
        "- **Cascading soft failure**: Initial tool execution error or malformed input propagating through subsequent iterations.",
        "",
        "### Unsuccessful Queries Detail",
    ])

    unsuccessful = [r for r in results if not r["success"]]
    if not unsuccessful:
        md_lines.append("No queries failed.")
    else:
        for u in unsuccessful:
            md_lines.append(f"#### {u['id']}: {u['description']}")
            md_lines.append(f"- **Query**: `{u['query']}`")
            md_lines.append(f"- **Failure Classification**: `{u['failure_class']}`")
            md_lines.append(f"- **Answer Produced**: {u['answer']}")
            md_lines.append(f"- **Tool Trace**: `{json.dumps(u['trace'])}`")
            md_lines.append("")

    out_file = Path(__file__).parent / "eval_results.md"
    out_file.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"Evaluation complete. Results written to {out_file}")


if __name__ == "__main__":
    run_evaluation()
