FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:0.5.1 /uv /bin/uv
WORKDIR /app
COPY pyproject.toml uv.lock* README.md ./
COPY src/ ./src/
RUN uv sync --frozen --no-dev 2>/dev/null || uv sync --no-dev
COPY data/ ./data/
ENV PYTHONUNBUFFERED=1
EXPOSE 8080 8501
CMD ["uv", "run", "uvicorn", "studyhelper.app:app", "--host", "0.0.0.0", "--port", "8080"]
