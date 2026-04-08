import os
import time
from typing import Iterable, Optional, List


def _parse_keys(value: Optional[object]) -> List[str]:
    if value is None:
        return []

    if isinstance(value, (list, tuple, set)):
        tokens: Iterable[object] = value
        parts = []
        for item in tokens:
            k = str(item or "").strip().strip('"').strip("'")
            if k:
                parts.append(k)
        out = []
        seen = set()
        for k in parts:
            if k not in seen:
                out.append(k)
                seen.add(k)
        return out

    text = str(value or "")
    if not text.strip():
        return []
    # Comma/whitespace separated list.
    parts = []
    for chunk in text.replace("\n", ",").split(","):
        k = chunk.strip().strip('"').strip("'")
        if k:
            parts.append(k)
    # De-dupe while preserving order.
    out = []
    seen = set()
    for k in parts:
        if k not in seen:
            out.append(k)
            seen.add(k)
    return out


class ApiKeyRotator:
    """
    Simple key rotator for free-tier APIs that rate-limit by key.
    Feed it a comma-separated env var (e.g., TWELVEDATA_API_KEYS=key1,key2,...).
    """

    def __init__(
        self,
        env_var: str,
        *,
        fallback_env_var: Optional[str] = None,
        static_keys: Optional[object] = None,
    ):
        self.env_var = env_var
        self.fallback_env_var = fallback_env_var
        self.static_keys = static_keys
        self._keys = []
        self._idx = 0
        self._last_reload = 0.0
        self.reload()

    def reload(self) -> None:
        raw = os.getenv(self.env_var)
        keys = _parse_keys(raw)
        if (not keys) and self.fallback_env_var:
            keys = _parse_keys(os.getenv(self.fallback_env_var))
        if not keys:
            keys = _parse_keys(self.static_keys)
        self._keys = keys
        self._idx = 0
        self._last_reload = time.time()

    def keys(self) -> List[str]:
        # Reload occasionally in long-running processes.
        if time.time() - self._last_reload > 300:
            self.reload()
        return list(self._keys)

    def next_key(self) -> Optional[str]:
        ks = self.keys()
        if not ks:
            return None
        k = ks[self._idx % len(ks)]
        self._idx = (self._idx + 1) % len(ks)
        return k
