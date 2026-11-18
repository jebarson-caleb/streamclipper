"""Engine logic against a temporary SQLite DB, with the LLM mocked out."""
import asyncio

import pytest

from streamclipper import config, db
from streamclipper.models import Channel, Clip, HypePoint, Moment, Segment, Stream, Word


@pytest.fixture()
def tmp_db(tmp_path):
    s = config.Settings(data_dir=tmp_path)
    s.ensure_dirs()
    config.set_settings(s)
    db.reset_engine()
    yield s
    db.reset_engine()


def _stream(mode, outputs, duration=1300.0, status="ended"):
    with db.session() as d:
        ch = Channel(name="me", url="x", permission="own", mode=mode, outputs=outputs, threshold=6)
        d.add(ch)
        d.commit()
        st = Stream(channel_id=ch.id, status=status, duration=duration)
        d.add(st)
        d.commit()
        t, i = 0.0, 0
        while t < duration:
            d.add(Segment(stream_id=st.id, idx=i, path=f"s{i}.ts", start=t, end=min(t + 30, duration), processed=True))
            t += 30
            i += 1
        for sec in range(int(duration)):
            d.add(HypePoint(stream_id=st.id, t=sec, audio=-5.0 if 600 <= sec < 606 else -30.0 - (sec % 7)))
        for k, txt in enumerate("no way did he just do that".split()):
            d.add(Word(stream_id=st.id, start=595 + k * 0.4, end=595.35 + k * 0.4, text=txt))
        d.commit()
    return ch, st


def test_continuous_parts(tmp_db):
    from streamclipper.detect import engine

    ch, st = _stream("continuous", "long")
    asyncio.run(engine.detect_step(st, ch))
    from sqlmodel import select

    with db.session() as d:
        clips = d.exec(select(Clip)).all()
        assert len(clips) == 3                       # 600 + 600 + 100
        assert all(c.format == "long" for c in clips)
        assert d.get(Stream, st.id).status == "finalized"


def test_best_moments_with_mocked_llm(tmp_db, monkeypatch):
    from streamclipper.detect import engine, llm_judge

    async def fake_chat(system, user):
        return {"score": 9, "category": "funny", "start": 590, "end": 615, "title": "No way", "hook": "NO WAY",
                "reason": "big reaction"}

    monkeypatch.setattr(llm_judge, "chat_json", fake_chat)
    ch, st = _stream("best_moments", "shorts")
    asyncio.run(engine.detect_step(st, ch))
    with db.session() as d:
        from sqlmodel import select
        ms = d.exec(select(Moment).where(Moment.status == "accepted")).all()
        clips = d.exec(select(Clip)).all()
    assert len(ms) == 1 and 590 <= ms[0].start < ms[0].end <= 615
    assert len(clips) == 1 and clips[0].format == "short" and clips[0].status == "pending"


def test_dead_air_parts_are_skipped_and_numbering_stays_continuous(tmp_db):
    """A frozen + silent stretch (signal loss / BRB) never becomes a clip; parts stay Part 1, Part 2."""
    from sqlmodel import select
    from streamclipper.detect import engine

    ch, st = _stream("continuous", "long", duration=1800.0)
    with db.session() as d:          # 600-1200 s: silent and frozen
        for p in d.exec(select(HypePoint).where(HypePoint.t >= 600, HypePoint.t < 1200)).all():
            p.audio, p.scene = -80.0, 0.0
            d.add(p)
        d.commit()
    asyncio.run(engine.detect_step(st, ch))
    with db.session() as d:
        clips = d.exec(select(Clip).order_by(Clip.id)).all()
        skipped = d.exec(select(Moment).where(Moment.status == "rejected")).all()
    assert len(clips) == 2 and len(skipped) == 1
    assert [c.title.rsplit(" ", 1)[-1] for c in clips] == ["1", "2"]


def test_live_passes_do_not_duplicate_one_event(tmp_db, monkeypatch):
    """Live detection runs in passes; the same event straddling two passes must give ONE moment."""
    from sqlmodel import select
    from streamclipper.detect import engine, llm_judge

    async def fake_chat(system, user):
        return {"score": 9, "category": "funny", "title": "No way", "hook": "NO WAY", "reason": "big reaction"}

    monkeypatch.setattr(llm_judge, "chat_json", fake_chat)
    ch, st = _stream("best_moments", "shorts", duration=1300.0, status="live")
    with db.session() as d:            # one long event: loud 600-614 s
        for p in d.exec(select(HypePoint).where(HypePoint.t >= 600, HypePoint.t < 615)).all():
            p.audio = -5.0
            d.add(p)
        segs = d.exec(select(Segment).order_by(Segment.idx)).all()
        for sg in segs:                 # pretend only the first 612 s are processed (live edge)
            sg.processed = sg.end <= 612
            d.add(sg)
        d.commit()
    asyncio.run(engine.detect_step(st, ch))
    with db.session() as d:             # next pass: more of the stream is processed
        for sg in d.exec(select(Segment)).all():
            sg.processed = True
            d.add(sg)
        d.commit()
        st = d.get(Stream, st.id)
    asyncio.run(engine.detect_step(st, ch))
    with db.session() as d:
        ms = d.exec(select(Moment).where(Moment.status == "accepted")).all()
    assert len(ms) == 1
