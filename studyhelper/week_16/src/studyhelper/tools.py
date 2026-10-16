"""Tools the model can call. Schemas follow the OpenAI function-calling format."""
import ast
import json
import operator
from datetime import date, datetime, timedelta

_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.Pow: operator.pow, ast.USub: operator.neg, ast.Mod: operator.mod}


def _eval(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.operand))
    raise ValueError("unsupported expression")


def calculate(expression: str) -> str:
    """Safe arithmetic (no eval): e.g. grade or study-hour maths."""
    try:
        return str(_eval(ast.parse(expression, mode="eval").body))
    except Exception as e:  # noqa: BLE001
        return f"error: {e}"


def study_plan(exam_date: str, topics: list[str], hours_per_day: float = 2.0) -> str:
    """Spread topics evenly over the days left before the exam (YYYY-MM-DD)."""
    try:
        days = (datetime.strptime(exam_date, "%Y-%m-%d").date() - date.today()).days
    except ValueError:
        return "error: exam_date must be YYYY-MM-DD"
    if days <= 0:
        return "error: exam_date must be in the future"
    if not topics:
        return "error: provide at least one topic"
    plan = []
    for i in range(days):
        plan.append({"date": str(date.today() + timedelta(days=i)),
                     "topic": topics[i * len(topics) // days], "hours": hours_per_day})
    return json.dumps({"days_left": days, "plan": plan[:14]})


TOOL_SCHEMAS = [
    {"type": "function", "function": {
        "name": "calculate", "description": "Evaluate an arithmetic expression, e.g. average marks or hours.",
        "parameters": {"type": "object", "properties": {"expression": {"type": "string"}},
                       "required": ["expression"]}}},
    {"type": "function", "function": {
        "name": "study_plan", "description": "Create a day-by-day study schedule until an exam date.",
        "parameters": {"type": "object", "properties": {
            "exam_date": {"type": "string", "description": "YYYY-MM-DD"},
            "topics": {"type": "array", "items": {"type": "string"}},
            "hours_per_day": {"type": "number"}}, "required": ["exam_date", "topics"]}}},
    {"type": "function", "function": {
        "name": "search_notes", "description": "Search the student's uploaded notes for relevant passages.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}},
                       "required": ["query"]}}},
]


INJECT_FAILURE_FLAG = False


def run_tool(name: str, args: dict, rag=None) -> str:
    if INJECT_FAILURE_FLAG:
        return "error: simulated downstream service failure / tool timeout"
    if name == "calculate":
        return calculate(args.get("expression", ""))
    if name == "study_plan":
        return study_plan(args.get("exam_date", ""), args.get("topics", []), args.get("hours_per_day", 2.0))
    if name == "search_notes":
        if rag is None:
            return "error: notes unavailable"
        return json.dumps(rag.retrieve(args.get("query", "")))
    return f"error: unknown tool {name}"
