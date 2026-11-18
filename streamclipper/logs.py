"""Logging to file + an in-memory ring buffer shown on the dashboard."""
from __future__ import annotations

import logging
from collections import deque
from logging.handlers import RotatingFileHandler
from pathlib import Path

RING: deque[str] = deque(maxlen=500)


class _RingHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        try:
            RING.append(self.format(record))
        except Exception:  # pragma: no cover
            pass


def setup_logging(log_path: Path, level: int = logging.INFO) -> None:
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s", "%Y-%m-%d %H:%M:%S")
    root = logging.getLogger()
    root.setLevel(level)
    for h in list(root.handlers):
        root.removeHandler(h)
    fh = RotatingFileHandler(log_path, maxBytes=5_000_000, backupCount=3, encoding="utf-8")
    sh = logging.StreamHandler()
    rh = _RingHandler()
    for h in (fh, sh, rh):
        h.setFormatter(fmt)
        root.addHandler(h)
    for noisy in ("httpx", "httpx2", "httpcore", "huggingface_hub", "googleapiclient.discovery_cache", "urllib3", "faster_whisper"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
