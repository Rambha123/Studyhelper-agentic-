import hashlib
import json
import time
from collections import OrderedDict
from typing import Any, Optional


class TTLCache:
    """LRU cache with per-entry TTL, keyed by a hash of the request."""

    def __init__(self, max_size: int = 256, ttl: int = 3600):
        self.max_size, self.ttl = max_size, ttl
        self._data: "OrderedDict[str, tuple[float, Any]]" = OrderedDict()
        self.hits = self.misses = 0

    @staticmethod
    def key(*parts: Any) -> str:
        return hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()

    def get(self, key: str) -> Optional[Any]:
        item = self._data.get(key)
        if item and time.time() - item[0] < self.ttl:
            self._data.move_to_end(key)
            self.hits += 1
            return item[1]
        if item:
            del self._data[key]
        self.misses += 1
        return None

    def set(self, key: str, value: Any) -> None:
        self._data[key] = (time.time(), value)
        self._data.move_to_end(key)
        while len(self._data) > self.max_size:
            self._data.popitem(last=False)
