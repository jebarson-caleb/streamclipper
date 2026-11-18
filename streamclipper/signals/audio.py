"""Audio extraction + per-second loudness."""
from __future__ import annotations

import numpy as np

from ..ffmpeg_utils import run_bytes

SR = 16000


async def load_pcm(path: str) -> np.ndarray:
    """Decode a media file to 16 kHz mono float32 (what Whisper wants)."""
    raw = await run_bytes(
        ["-hide_banner", "-loglevel", "error", "-i", path, "-vn", "-ac", "1", "-ar", str(SR),
         "-f", "s16le", "pipe:1"],
        timeout=300,
    )
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def loudness_per_second(pcm: np.ndarray, sr: int = SR) -> list[float]:
    """RMS level in dBFS for every whole second (-90 for silence)."""
    n = len(pcm) // sr
    if n == 0:
        return []
    frames = pcm[: n * sr].reshape(n, sr)
    rms = np.sqrt(np.mean(frames ** 2, axis=1))
    return [float(x) for x in 20 * np.log10(np.maximum(rms, 10 ** (-90 / 20)))]
