"""faster-whisper with word timestamps. Loaded lazily; falls back to CPU if CUDA fails."""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass

import numpy as np

from ..config import get_settings
from ..gpu import is_oom

log = logging.getLogger(__name__)

_model = None
_model_device = ""
_mtx = threading.Lock()


@dataclass
class W:
    start: float
    end: float
    text: str


def _add_cuda_dlls() -> None:
    """pip's nvidia-cublas/cudnn wheels put DLLs in site-packages/nvidia/*/bin; Windows won't find them alone."""
    import os
    import sys
    from pathlib import Path

    if sys.platform != "win32":
        return
    for sp in map(Path, sys.path):
        nv = sp / "nvidia"
        if nv.is_dir():
            for b in nv.glob("*/bin"):
                os.add_dll_directory(str(b))
                os.environ["PATH"] = str(b) + os.pathsep + os.environ.get("PATH", "")


def _load(device: str):
    global _model, _model_device
    if device == "cuda":
        _add_cuda_dlls()
    from faster_whisper import WhisperModel

    cfg = get_settings().whisper
    compute = cfg.compute_type if device == "cuda" else "int8"
    log.info("loading whisper %s on %s (%s)", cfg.model, device, compute)
    _model = WhisperModel(cfg.model, device=device, compute_type=compute)
    _model_device = device


def _ensure_model() -> None:
    if _model is not None:
        return
    want = get_settings().whisper.device
    try:
        _load(want)
    except Exception as e:
        if want != "cuda":
            raise
        log.warning("whisper on CUDA failed (%s); using CPU", e)
        _load("cpu")


_HALLUCINATIONS = ("thank you for watching", "thanks for watching", "please subscribe", "like and subscribe",
                   "subtitles by", "transcribed by", "amara.org", "see you in the next video")


def _looks_like_garbage(text: str) -> bool:
    """Mostly symbols/digits ("※2 -3 -2 -1"), or several writing systems mixed in one line."""
    letters = [c for c in text if c.isalpha()]
    if len(letters) < 0.5 * max(1, len(text.replace(" ", ""))):
        return True
    scripts = set()
    for c in letters:
        cp = ord(c)
        if cp < 0x250:
            scripts.add("latin")
        elif 0x2E80 <= cp <= 0x9FFF or 0xF900 <= cp <= 0xFAFF or 0xFF00 <= cp <= 0xFFEF:
            scripts.add("cjk")            # Japanese mixes kana + kanji normally: one group
        elif 0xAC00 <= cp <= 0xD7AF or 0x1100 <= cp <= 0x11FF:
            scripts.add("hangul")
        else:
            scripts.add(cp // 0x100)      # other alphabets (Cyrillic, Devanagari, Thai, ...)
    return len(scripts) >= 3


def is_hallucination(text: str, no_speech_prob: float, avg_logprob: float, compression_ratio: float,
                     language_prob: float = 1.0) -> bool:
    """Drop text Whisper invents over music/noise. Calibrated on real commentary over a loud crowd
    (log-prob -0.4..-0.9, language conf >= 0.9, compression <= 1.3) vs music hallucinations
    (log-prob -0.9..-2.3, language conf 0.4..0.8, compression up to 2.4+).
    no_speech_prob alone is useless here: real speech over crowd noise sits around 0.5."""
    t = text.strip().lower()
    if not t:
        return True
    if avg_logprob < -1.0 or compression_ratio > 2.0:
        return True
    if no_speech_prob > 0.6 and avg_logprob < -0.7:
        return True
    if language_prob < 0.6 and no_speech_prob > 0.45:      # model unsure it's even speech/language
        return True
    if _looks_like_garbage(text):
        return True
    return any(p in t for p in _HALLUCINATIONS) or (t.startswith("[") and t.endswith("]"))


def _run(pcm: np.ndarray, language: str) -> list[W]:
    cfg = get_settings().whisper
    kw = {"vad_filter": False}
    if cfg.vad:
        # gentle settings; Silero's defaults drop speech over crowd noise/music
        kw = {"vad_filter": True, "vad_parameters": {"threshold": 0.2, "min_silence_duration_ms": 1000,
                                                     "speech_pad_ms": 600}}
    segs, info = _model.transcribe(
        pcm, language=language or None, word_timestamps=True, beam_size=1,
        condition_on_previous_text=False, **kw,
    )
    out: list[W] = []
    for s in segs:
        lang_p = 1.0 if language else float(getattr(info, "language_probability", 1.0) or 1.0)
        if is_hallucination(s.text, s.no_speech_prob, s.avg_logprob, s.compression_ratio, lang_p):
            log.debug("dropped likely hallucination: %r", s.text[:80])
            continue
        out += [W(float(w.start), float(w.end), w.word.strip()) for w in (s.words or []) if w.word.strip()]
    return out


def transcribe_pcm(pcm: np.ndarray, language: str = "") -> list[W]:
    """Blocking. Call through asyncio.to_thread while holding the GPU lock."""
    if len(pcm) < 1600:
        return []
    with _mtx:
        _ensure_model()
        try:
            return _run(pcm, language)
        except Exception as e:
            if _model_device == "cuda" and is_oom(str(e)):
                log.warning("whisper CUDA out of memory, switching to CPU")
                _load("cpu")
                return _run(pcm, language)
            raise


def warm_up() -> None:
    """Load the model now (blocking)."""
    with _mtx:
        _ensure_model()
