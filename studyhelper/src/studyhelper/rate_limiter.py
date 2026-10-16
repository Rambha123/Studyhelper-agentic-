import time
from collections import defaultdict, deque


class SlidingWindowLimiter:
    """Allow `limit` requests per `window` seconds per client."""

    def __init__(self, limit: int = 30, window: float = 60.0):
        self.limit, self.window = limit, window
        self._hits: dict[str, deque] = defaultdict(deque)

    def check(self, client: str) -> tuple[bool, int]:
        """Return (allowed, retry_after_seconds)."""
        now = time.monotonic()
        q = self._hits[client]
        while q and now - q[0] >= self.window:
            q.popleft()
        if len(q) >= self.limit:
            return False, max(1, int(self.window - (now - q[0])) + 1)
        q.append(now)
        return True, 0
