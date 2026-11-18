"""Render one span of a recorded stream into a finished, subtitled video."""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path

from sqlmodel import select

from ..config import get_settings
from ..db import session
from ..ffmpeg_utils import AUDIO_ARGS, ass_filter, ass_path_escape, encode_args, has_audio, run, video_size
from ..models import Segment, Word as WordRow
from . import subtitles as subs
from .shots import choose_edges, detect_cuts, static_tail
from .track import FPS, compute_track, load_frames, path_stats, sendcmd_file
from .vertical import (choose_layout, detect_faces, find_webcam, horizontal_filter, track_filter,
                       vertical_filter, webcam_box)

log = logging.getLogger(__name__)


@dataclass
class RenderResult:
    path: str
    start: float
    end: float
    words: list[subs.Word]
    layout: str


def segments_for(stream_id: int, a: float, b: float) -> list[Segment]:
    with session() as db:
        return list(db.exec(select(Segment).where(Segment.stream_id == stream_id, Segment.end > a,
                                                  Segment.start < b, Segment.deleted == False)  # noqa: E712
                            .order_by(Segment.idx)).all())


def trim_silence(a: float, b: float, words: list[WordRow], min_len: float) -> tuple[float, float]:
    """Cut long dead air at the edges (keeps reactions: only trims gaps > 3 s)."""
    if not words:
        return a, b
    na, nb = a, b
    if words[0].start - a > 3:
        na = words[0].start - 0.5
    if b - words[-1].end > 4:
        nb = words[-1].end + 2.0
    if nb - na < min_len:
        return a, b
    return na, nb


def _phrases(words: list, gap: float = 0.6) -> list[tuple[float, float]]:
    """Group words into spoken phrases: split on pauses > gap or sentence-ending punctuation."""
    out: list[list[float]] = []
    prev = None
    for w in words:
        if prev is None or w.start - prev.end > gap or prev.text.rstrip().endswith((".", "?", "!")):
            out.append([w.start, w.end])
        else:
            out[-1][1] = max(out[-1][1], w.end)
        prev = w
    return [(p[0], p[1]) for p in out]


def snap_to_phrases(a: float, b: float, words: list, *, total: float, max_extend: float = 4.0,
                    max_len: float | None = None, min_len: float = 10.0) -> tuple[float, float]:
    """Move clip edges so no sentence is cut in half: extend outward to the phrase start/end when that
    costs <= max_extend s (and stays <= max_len); otherwise move the edge inward past the phrase."""
    phrases = _phrases(words)
    na, nb = a, b
    for ps, pe in phrases:
        if ps < na < pe:                                   # phrase straddles the start
            if na - ps <= max_extend:
                na = max(0.0, ps - 0.15)
            elif b - pe >= min_len:
                na = pe + 0.05
        if ps < nb < pe:                                   # phrase straddles the end
            starts_late = nb - ps < 3.0   # began in the last 3 s: it's the NEXT moment's setup, don't steal it
            if starts_late and ps - 0.1 - na >= min_len:
                nb = ps - 0.1
            elif pe - nb <= max_extend and (max_len is None or pe + 0.4 - na <= max_len):
                nb = min(total, pe + 0.4)
            elif ps - 0.1 - na >= min_len:
                nb = ps - 0.1
    return na, nb


async def render_span(stream_id: int, a: float, b: float, out: Path, *, vertical: bool, hook: str = "",
                      credit: str = "", trim: bool = True, peak: float | None = None) -> RenderResult:
    s = get_settings()
    r = s.render
    pad = 8.0
    segs = segments_for(stream_id, max(0.0, a - pad), b + pad)
    if not segs:
        raise RuntimeError(f"no segments for {a:.0f}-{b:.0f}s (deleted by retention?)")
    out.parent.mkdir(parents=True, exist_ok=True)
    work = out.with_suffix("")
    concat = work.with_suffix(".concat.txt")
    concat.write_text("".join(f"file '{Path(x.path).as_posix()}'\n" for x in segs), encoding="utf-8")
    base, total = segs[0].start, segs[-1].end
    max_len = 59.5 if vertical else None
    cuts: list[float] = []
    on_a = on_b = False
    if trim:   # moments (not back-to-back continuous parts): proper start and finish
        ws, we = max(base, a - pad), min(total, b + pad)
        cuts = [ws + c for c in await detect_cuts(concat, ws - base, we - ws)]
        pk = peak if peak is not None and a < peak < b else (a + b) / 2
        # a clean boundary beats a strict minimum length: allow down to half of min_clip at a cut
        na, nb, on_a, on_b = choose_edges(a, b, pk, cuts, min_len=max(6.0, s.detect.min_clip / 2),
                                          max_len=max_len)
        a = na + 0.04 if on_a else a          # first frame of the new shot
        b = nb - 0.05 if on_b else b          # last frame before the next shot
        with session() as db:
            near = list(db.exec(select(WordRow).where(WordRow.stream_id == stream_id, WordRow.end > a - 6,
                                                      WordRow.start < b + 6).order_by(WordRow.start)).all())
        sa, sb = snap_to_phrases(a, b, near, total=total, max_len=max_len, min_len=s.detect.min_clip)
        a = a if on_a else sa                 # a shot cut beats a phrase edge (commentary runs over cuts)
        b = b if on_b else sb
        a, b = max(a, base), min(b, total)
        # never finish on a frozen end card / logo / black screen
        tail = min(5.0, b - a - 6.0)
        if tail > 1.0:
            try:
                fr = await load_frames(concat, b - tail - base, tail)
                motion = [float(abs(fr[i] - fr[i - 1]).mean()) for i in range(1, len(fr))]
                st_at = static_tail(motion, FPS)
                if st_at is not None:
                    new_b = b - tail + st_at + 1.0 / FPS   # st_at is seconds (motion[i] = frame i+1)
                    near_cut = [c for c in cuts if new_b - 1.2 <= c <= new_b + 0.2]
                    b = (max(near_cut) - 0.05) if near_cut else new_b
                    on_b = True
                    log.info("trimmed a static ending (end card/logo/black) at %.1fs", b)
            except Exception as e:
                log.debug("static-tail check skipped: %s", e)
        if b - a < 6.0:            # hard minimum
            b = min(total, a + 6.0)
    with session() as db:
        rows = list(db.exec(select(WordRow).where(WordRow.stream_id == stream_id, WordRow.start >= a,
                                                  WordRow.end <= b + 0.5).order_by(WordRow.start)).all())
    if trim and not cuts:
        # talking content without shot changes: trim long dead air. (With cuts, the silent
        # run-up IS the action, so never trim it.)
        a, b = trim_silence(a, b, rows, s.detect.min_clip)
        rows = [w for w in rows if a <= w.start < b]
    rows = [w for w in rows if w.start < b]
    if on_a and rows:
        # the clip starts on a shot cut: commentary still finishing the PREVIOUS item ("...up then,
        # Dunkshin!") would be the first subtitle. Drop words of a sentence that began before the cut.
        for ps, pe in _phrases(near):
            if ps < a - 0.05 < pe:
                rows = [w for w in rows if w.start > pe]
                break
    dur = b - a
    words = [subs.Word(w.start - a, min(w.end, b) - a, w.text) for w in rows]
    offset = max(0.0, a - base)

    # ---- layout
    layout = "horizontal"
    cmd_file: Path | None = None
    if vertical:
        samples = []
        for k in range(8):
            t = a + dur * (k + 0.5) / 8
            seg = next((x for x in segs if x.start <= t < x.end), segs[0])
            samples.append((seg.path, t - seg.start))
        per_frame = await asyncio.to_thread(detect_faces, samples)
        faces = [fs[0] for fs in per_frame if fs]
        layout, face = choose_layout(faces, len(samples), r.vertical_layout,
                                     webcam=find_webcam(per_frame, len(samples)))
        sw, sh = await video_size(segs[0].path)
        if layout == "track":
            crop_w = max(2, int(sh * 9 / 16) // 2 * 2)
            try:
                frames = await load_frames(concat, offset, dur)
                tr = await asyncio.to_thread(compute_track, frames, crop_w / sw, [c - a for c in cuts if a < c < b])
            except Exception as e:
                log.warning("motion tracking failed (%s); using blurred fit", e)
                tr = None
            if tr is None or tr.coverage < r.track_min_coverage:
                log.info("action too spread for a 9:16 crop (coverage %.2f): blurred fit",
                         tr.coverage if tr else 0)
                layout = "fit_blur"
            else:
                cmd_file = work.with_suffix(".track.txt")
                x0 = sendcmd_file(tr, sw, crop_w, cmd_file)
                st = path_stats(tr, sw, [c - a for c in cuts if a < c < b])
                log.info("camera path: max %.1f px/frame, p95 %.1f, max accel %.2f (coverage %.2f)",
                         st["max_step"], st["p95_step"], st["max_accel"], tr.coverage)
                vf = track_filter(ass_path_escape(str(cmd_file)), x0, crop_w, sh)
        if layout != "track":
            box = await asyncio.to_thread(webcam_box, samples, face) if layout == "split" and face else None
            vf = vertical_filter(layout, face, sw, sh, box)
    else:
        vf = horizontal_filter()

    # ---- subtitles + overlays
    from . import caption_style

    ass = work.with_suffix(".ass")
    ass.write_text(subs.build_ass(
        words, dur, vertical, hook=hook if (vertical and r.hook_card) else "", credit=credit,
        profanity=r.profanity if r.mute_profanity else None, keywords=s.detect.keywords,
        style=caption_style.load(),
    ), encoding="utf-8")
    graph = vf + f";[vbase]{ass_filter(ass)}[vsub]"
    last = "vsub"
    if r.progress_bar and vertical:
        graph += (f";color=c=0xFFD000:s=1080x14:r=30[bar];"
                  f"[vsub][bar]overlay=x='-W+W*t/{dur:.3f}':y=H-14:shortest=1[vbar]")
        last = "vbar"
    fade_out = min(0.35, dur / 6)
    if trim and dur > 2:   # soft start/finish (and dip-to-black transitions inside compilations)
        graph += (f";[{last}]fade=t=in:st=0:d=0.2,fade=t=out:st={dur - fade_out:.3f}:d={fade_out:.3f}[vfade]")
        last = "vfade"

    af = "loudnorm=I=-14:TP=-1.5:LRA=11"
    if trim and dur > 2:
        af += f",afade=t=in:d=0.15,afade=t=out:st={dur - 0.4:.3f}:d=0.4"
    if r.mute_profanity:
        bad = [w for w in words if subs.censor(w.text, r.profanity) != w.text]
        for w in bad:
            af = f"volume=enable='between(t,{w.start:.2f},{w.end:.2f})':volume=0," + af

    # Frame-accurate cut with trim/atrim inside the graph. (-ss on a concat input is not
    # accurate, and output-side -ss shifts the subtitles/hook/progress bar by `offset`.)
    cut = f"start={offset:.3f}:duration={dur:.3f}"
    # Live sources can switch resolution mid-stream (adaptive bitrate): pin every frame to the size
    # the layout was computed for, or the crop fails when ffmpeg reinitialises the filters.
    pin = f",scale={sw}:{sh}" if vertical else ""
    graph = f"[0:v]trim={cut},setpts=PTS-STARTPTS{pin}[vin];" + graph.replace("[0:v]", "[vin]", 1)
    maps = ["-map", f"[{last}]"]
    if await has_audio(segs[0].path):
        graph += f";[0:a]atrim={cut},asetpts=PTS-STARTPTS,{af}[aout]"
        maps += ["-map", "[aout]"]
    args = ["-hide_banner", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(concat),
            "-filter_complex", graph, *maps, "-t", f"{dur:.3f}",
            "-r", "30", *encode_args(), *AUDIO_ARGS, "-movflags", "+faststart", str(out)]
    try:
        await run(args, timeout=max(600, dur * 10))
    finally:
        concat.unlink(missing_ok=True)
        if cmd_file:
            cmd_file.unlink(missing_ok=True)
    ass.unlink(missing_ok=True)
    log.info("rendered %s (%s, %.0fs)", out.name, layout, dur)
    return RenderResult(str(out), a, b, words, layout)
