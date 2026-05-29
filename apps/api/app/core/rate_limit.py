from __future__ import annotations

import time
from collections import defaultdict, deque


class InMemoryRateLimiter:
    """Local rate limiter.

    Production deployments should replace this with a Redis-backed limiter so limits
    are shared across replicas.
    """

    def __init__(self, limit_per_minute: int) -> None:
        self.limit_per_minute = limit_per_minute
        self._events: dict[str, deque[float]] = defaultdict(deque)

    async def allow(self, key: str) -> bool:
        now = time.monotonic()
        window_start = now - 60
        events = self._events[key]

        while events and events[0] < window_start:
            events.popleft()

        if len(events) >= self.limit_per_minute:
            return False

        events.append(now)
        return True
