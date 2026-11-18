"""Database tables. Every pipeline stage is a status field, so a restart resumes work."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from pydantic import NaiveDatetime
from sqlmodel import Field, SQLModel


def utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


class Channel(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    platform: str = "twitch"            # twitch | youtube | kick | file
    url: str                            # stream URL, or a local path for platform=file
    # Clipping permission is required: own = your channel, allowed = creator gave permission.
    permission: str = "none"            # own | allowed | none
    permission_note: str = ""           # link/proof of permission for allowed creators
    enabled: bool = True
    mode: str = "best_moments"          # best_moments | continuous | hybrid
    outputs: str = "shorts"             # shorts | long | both
    threshold: float = 7.0              # final score (0-10) needed for best_moments
    max_clips_per_stream: int = 8
    continuous_len_s: int = 600         # part length for continuous cuts (long)
    continuous_short_len_s: int = 55    # part length when continuous output is shorts
    language: str = ""                  # whisper language hint, "" = auto
    translate_to: str = ""              # e.g. "es" -> translated caption track
    credit_text: str = ""               # overlay; default built from name/url for allowed creators
    target_account: str = "default"     # which YouTube account token to post with
    title_template: str = "{title}"
    desc_template: str = "{summary}\n\n{credit}\n\n{hashtags}"
    extra_tags: str = ""                # comma separated
    created_at: NaiveDatetime = Field(default_factory=utcnow)


class Stream(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    channel_id: int = Field(index=True, foreign_key="channel.id")
    title: str = ""
    started_at: NaiveDatetime = Field(default_factory=utcnow)
    ended_at: Optional[NaiveDatetime] = None
    status: str = "live"                # live | ended | finalized
    dir: str = ""
    duration: float = 0.0               # seconds captured so far
    detected_until: float = 0.0         # detector has scanned up to this offset
    continuous_until: float = 0.0       # continuous cutter (long parts) has produced parts up to here
    continuous_short_until: float = 0.0 # same, for Shorts-length parts
    manual_requests: str = ""           # comma separated offsets from "Clip now" / !clip


class Segment(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    stream_id: int = Field(index=True, foreign_key="stream.id")
    idx: int
    path: str
    start: float                        # offset from stream start
    end: float
    processed: bool = False             # signals + transcript done
    deleted: bool = False


class HypePoint(SQLModel, table=True):
    """Per-second raw signals."""
    id: Optional[int] = Field(default=None, primary_key=True)
    stream_id: int = Field(index=True, foreign_key="stream.id")
    t: int = Field(index=True)
    audio: float = 0.0                  # loudness dBFS
    scene: float = 0.0                  # scene-change score 0..1


class ChatMsg(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    stream_id: int = Field(index=True, foreign_key="stream.id")
    t: float = Field(index=True)
    user: str = ""
    text: str = ""
    is_mod: bool = False


class Word(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    stream_id: int = Field(index=True, foreign_key="stream.id")
    start: float = Field(index=True)
    end: float
    text: str


class Moment(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    stream_id: int = Field(index=True, foreign_key="stream.id")
    kind: str = "moment"                # moment | continuous | manual
    start: float
    end: float
    peak: float = 0.0
    signal_score: float = 0.0           # 0-10
    llm_score: float = 0.0              # 0-10
    score: float = 0.0                  # 0-10 combined
    category: str = ""
    title: str = ""
    hook: str = ""
    reason: str = ""
    transcript: str = ""
    status: str = "candidate"           # candidate | accepted | rejected | rendered
    part_no: int = 0
    created_at: NaiveDatetime = Field(default_factory=utcnow)


class Clip(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    stream_id: int = Field(index=True, foreign_key="stream.id")
    channel_id: int = Field(index=True, foreign_key="channel.id")
    moment_ids: str = ""                # comma separated (compilations have many)
    format: str = "short"               # short | long
    path: str = ""
    srt_path: str = ""
    thumb_path: str = ""
    duration: float = 0.0
    title: str = ""
    description: str = ""
    tags: str = ""                      # comma separated
    category: str = ""
    score: float = 0.0
    status: str = "pending"             # pending | rendering | ready | approved | scheduled | published | rejected | failed
    attempts: int = 0
    publish_at: Optional[NaiveDatetime] = None   # UTC
    youtube_id: str = ""
    views: int = 0
    error: str = ""
    created_at: NaiveDatetime = Field(default_factory=utcnow)


class QuotaLog(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    day: str = Field(index=True)        # Pacific date, YYYY-MM-DD (YouTube quota resets at PT midnight)
    account: str = "default"
    units: int = 0


class CategoryWeight(SQLModel, table=True):
    """Learned from published-clip performance. Multiplies the LLM score (0.8-1.2)."""
    id: Optional[int] = Field(default=None, primary_key=True)
    category: str = Field(index=True, unique=True)
    weight: float = 1.0
    samples: int = 0
