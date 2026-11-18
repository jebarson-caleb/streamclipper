"""Settings loaded from config.yaml (all keys optional)."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class Dashboard(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8765


class Monitor(BaseModel):
    poll_seconds: int = 60


class Capture(BaseModel):
    segment_seconds: int = 20              # shorter = live clips sooner
    buffer_hours: float = 6
    quality: str = "720p,720p60,480p,best"
    min_free_gb: float = 5.0             # below this no new recordings start; below half, recording stops


class Whisper(BaseModel):
    model: str = "small"
    device: str = "cuda"
    compute_type: str = "int8_float16"
    vad: bool = False   # True drops commentary over crowd noise/music; hallucination filter is used instead


class LLM(BaseModel):
    backend: str = "auto"   # auto: bundled llama.cpp in the packaged exe, Ollama from source | llamacpp | ollama
    # Qwen3-4B, Apache-2.0 (commercial use allowed). Qwen2.5-3B was non-commercial (Qwen Research License).
    gguf_url: str = "https://huggingface.co/Qwen/Qwen3-4B-GGUF/resolve/main/Qwen3-4B-Q4_K_M.gguf"
    gguf_sha256: str = "7485fe6f11af29433bc51cab58009521f205840f5b4ae3a32fa7f92e8534fdf5"
    gguf_layers: int = 36                # transformer layers in that model (for partial GPU offload)
    server_port: int = 11435             # bundled llama-server (127.0.0.1 only)
    base_url: str = "http://127.0.0.1:11434"
    model: str = "qwen3:4b"               # Ollama (source installs); Apache-2.0
    fallback_model: str = "qwen2.5:1.5b"   # Apache-2.0
    keep_alive: str = "30s"
    timeout_seconds: float = 180


class Detect(BaseModel):
    interval_seconds: int = 60
    live_interval_seconds: int = 15        # live streams are checked this often (continuous clipping)
    pad_before: float = 20
    pad_after: float = 10
    min_clip: float = 15
    max_short: float = 58
    max_long_moment: float = 180
    peak_z: float = 1.8
    min_peak_gap: int = 20              # seconds between two separate highlights
    weights: dict[str, float] = Field(
        default_factory=lambda: {"audio": 1.0, "chat": 1.5, "keywords": 1.2, "scene": 0.4}
    )
    keywords: list[str] = Field(
        default_factory=lambda: ["LUL", "LOL", "KEKW", "OMEGALUL", "W", "POG", "POGGERS",
                                 "CLIP", "CLIP IT", "OMG", "WTF", "LMAO", "😂", "💀"]
    )
    hype_bonus_per_z: float = 0.5        # final = LLM score + bonus for hype above the trigger
    hype_bonus_max: float = 1.5


class Render(BaseModel):
    encoder: str = "auto"
    vertical_layout: str = "auto"       # auto | track | crop | split | fit_blur
    track_min_coverage: float = 0.35    # uniform motion = 0.32; below this the action is too spread for a 9:16 crop -> fit_blur
    font: str = "Arial Black"
    font_size_short: int = 78
    font_size_long: int = 54
    highlight_color: str = "&H0000F0FF"
    progress_bar: bool = True
    hook_card: bool = True
    mute_profanity: bool = False
    profanity: list[str] = Field(default_factory=lambda: ["fuck", "shit", "bitch", "cunt"])
    compile_top_n: int = 10


class Publish(BaseModel):
    enabled: bool = True
    client_secret: str = "./client_secret.json"
    category_id: str = "20"
    daily_quota: int = 10000
    upload_cost: int = 1600
    max_uploads_per_day: int = 6
    min_lead_minutes: int = 20
    slots: dict[str, list[str]] = Field(
        default_factory=lambda: {"short": ["11:00", "15:00", "19:00", "22:00"], "long": ["18:00"]}
    )
    upload_captions: bool = False


class Retention(BaseModel):
    delete_raw_after_hours: float = 6
    delete_renders_after_days: float = 3
    delete_renders_after_upload: bool = False


class Notify(BaseModel):
    discord_webhook: str = ""
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""


class Analytics(BaseModel):
    enabled: bool = True
    interval_hours: float = 12


class Settings(BaseModel):
    data_dir: Path = Path("./data")
    dashboard: Dashboard = Dashboard()
    review_mode: bool = False
    timezone: str = "local"
    monitor: Monitor = Monitor()
    capture: Capture = Capture()
    whisper: Whisper = Whisper()
    llm: LLM = LLM()
    detect: Detect = Detect()
    render: Render = Render()
    publish: Publish = Publish()
    retention: Retention = Retention()
    notify: Notify = Notify()
    analytics: Analytics = Analytics()

    # derived paths
    @property
    def db_path(self) -> Path:
        return self.data_dir / "streamclipper.db"

    @property
    def streams_dir(self) -> Path:
        return self.data_dir / "streams"

    @property
    def renders_dir(self) -> Path:
        return self.data_dir / "renders"

    @property
    def tokens_dir(self) -> Path:
        return self.data_dir / "tokens"

    @property
    def log_path(self) -> Path:
        return self.data_dir / "streamclipper.log"

    def ensure_dirs(self) -> None:
        for p in (self.data_dir, self.streams_dir, self.renders_dir, self.tokens_dir):
            p.mkdir(parents=True, exist_ok=True)


_settings: Settings | None = None


def load_settings(path: str | os.PathLike | None = None) -> Settings:
    global _settings
    path = Path(path or os.environ.get("STREAMCLIPPER_CONFIG", "config.yaml"))
    raw: dict[str, Any] = {}
    if path.exists():
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    _settings = Settings(**raw)
    _settings.data_dir = _settings.data_dir.resolve()
    _settings.ensure_dirs()
    return _settings


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = load_settings()
    return _settings


def set_settings(s: Settings) -> None:
    """Used by tests."""
    global _settings
    _settings = s
