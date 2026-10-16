import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    api_key: str = os.getenv("GEMINI_API_KEY", "")
    base_url: str = os.getenv(
        "LLM_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/"
    )
    primary_model: str = os.getenv("PRIMARY_MODEL", "gemini-2.5-flash")
    fallback_model: str = os.getenv("FALLBACK_MODEL", "gemini-2.5-flash-lite")
    embed_model: str = os.getenv("EMBED_MODEL", "gemini-embedding-001")
    temperature: float = float(os.getenv("TEMPERATURE", "0.2"))
    top_p: float = float(os.getenv("TOP_P", "0.9"))
    max_tokens: int = int(os.getenv("MAX_TOKENS", "1024"))
    rate_limit_rpm: int = int(os.getenv("RATE_LIMIT_RPM", "30"))
    cache_ttl: int = int(os.getenv("CACHE_TTL_SEC", "3600"))
    max_concurrency: int = int(os.getenv("MAX_CONCURRENCY", "5"))
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "600"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "80"))
    top_k: int = int(os.getenv("TOP_K", "4"))
    docs_dir: str = os.getenv("DOCS_DIR", "data/notes")
    chroma_dir: str = os.getenv("CHROMA_DIR", "chroma_db")


settings = Settings()
