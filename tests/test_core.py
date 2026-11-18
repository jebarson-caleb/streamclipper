from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pytest

from streamclipper.detect import candidates as cand
from streamclipper.detect.dedupe import iou, is_duplicate, jaccard
from streamclipper.detect.engine import combine
from streamclipper.detect.llm import LLMError, parse_json
from streamclipper.detect.llm_judge import sanitize
from streamclipper.detect.modes import continuous_windows, plan_for, select_best
from streamclipper.edit import subtitles as subs
from streamclipper.edit.vertical import Face, choose_layout, vertical_filter
from streamclipper.publish.analytics import compute_weights
from streamclipper.publish.metadata import chapters, fmt_ts
from streamclipper.publish.scheduler import next_slot
from streamclipper.signals.audio import loudness_per_second
from streamclipper.signals.chat import keyword_hits


# ---------------------------------------------------------------- signals / candidates
def test_loudness_per_second():
    sr = 16000
    pcm = np.concatenate([np.zeros(sr), 0.5 * np.ones(sr)]).astype(np.float32)
    db = loudness_per_second(pcm, sr)
    assert len(db) == 2 and db[0] == pytest.approx(-90) and db[1] == pytest.approx(-6.02, abs=0.1)


def test_keyword_hits_word_boundaries():
    kws = ["W", "LUL", "😂"]
    assert keyword_hits("W W chat", kws) == 2
    assert keyword_hits("WOW what", kws) == 0
    assert keyword_hits("lul 😂😂", kws) == 3


def test_hype_curve_finds_spike():
    n = 600
    audio = {t: -30.0 + np.sin(t) for t in range(n)}
    for t in range(300, 306):
        audio[t] = -5.0
    chat = [(302 + cand.CHAT_LAG + i * 0.1, "LUL LUL") for i in range(40)]
    hype = cand.hype_curve(n, audio, {}, chat, {"audio": 1, "chat": 1.5, "keywords": 1.2}, ["LUL"])
    peaks = cand.find_peaks(hype, 1.8)
    assert peaks and all(295 <= p <= 310 for p in peaks)


def test_windows_merge_and_manual():
    hype = np.zeros(1000)
    hype[100], hype[110] = 3, 4
    w = cand.windows_from_peaks([100, 110], hype, 1000, 20, 10)
    assert len(w) == 1 and w[0].start == 80 and w[0].end == 120 and w[0].peak == 110
    w = cand.windows_from_peaks([], hype, 1000, 20, 10, manual=[500])
    assert w[0].manual and w[0].start == 450


def test_signal_score_bounds():
    assert cand.signal_score(-1) == 0 and cand.signal_score(10) == 10


# ---------------------------------------------------------------- modes / dedupe / scoring
def test_plan_for():
    assert plan_for("best_moments", "both").compilation and plan_for("best_moments", "both").moment_shorts
    p = plan_for("continuous", "shorts")
    assert p.continuous_short and not p.continuous_long and not p.detect_moments
    p = plan_for("hybrid", "shorts")
    assert p.moment_shorts and p.continuous_long


def test_continuous_windows():
    assert continuous_windows(0, 1250, 600, ended=False) == [(0, 600), (600, 1200)]
    assert continuous_windows(1200, 1250, 600, ended=True) == [(1200, 1250)]
    assert continuous_windows(1200, 1210, 600, ended=True) == []


def test_select_best():
    class M:
        def __init__(self, s):
            self.score = s
    got = select_best([M(5), M(9), M(7.5), M(8)], 7, limit=3, already=1)
    assert [m.score for m in got] == [9, 8]


def test_dedupe():
    assert iou((0, 10), (5, 15)) == pytest.approx(5 / 15)
    assert jaccard("he really did that clutch play", "he really did that clutch play wow") > 0.7
    assert is_duplicate((0, 30), "", [(5, 30)], [])
    assert not is_duplicate((100, 130), "new words here entirely", [(0, 30)], ["other stuff said before"])


def test_combine():
    assert combine(-1, 2.4, 1.8) == 6.0                 # no LLM: hype score alone (z*2.5)
    assert combine(7, 1.8, 1.8) == 7.0                  # LLM 7 at the trigger stays 7 -> passes threshold 7
    assert combine(7, 2.2, 1.8) == 7.2                  # hype above the trigger adds a bonus
    assert combine(7, 9.0, 1.8) == 8.5                  # bonus capped at +1.5
    assert combine(9, 9.0, 1.8, 1.2) == 10.0            # capped at 10


# ---------------------------------------------------------------- LLM parsing
def test_parse_json_tolerant():
    assert parse_json('{"a": 1}') == {"a": 1}
    assert parse_json('Sure! ```json\n{"score": 8}\n```') == {"score": 8}
    with pytest.raises(LLMError):
        parse_json("no json here")


def test_sanitize_clamps():
    j = sanitize({"score": 14, "start": -5, "end": 999, "category": "FUNNY", "title": "x" * 200},
                 100, 160, min_len=15, max_len=58)
    assert j.score == 10 and j.category == "funny" and j.start == 100 and j.end == 158 and len(j.title) == 100
    j = sanitize({"start": 130, "end": 132}, 100, 160, 15, 58)
    assert j.end - j.start >= 15 - 1e-9
    assert sanitize({"category": "nonsense"}, 0, 30, 15, 58).category == "other"


# ---------------------------------------------------------------- subtitles
def _words():
    return [subs.Word(0.0, 0.4, "no"), subs.Word(0.4, 0.8, "way"), subs.Word(0.8, 1.2, "he"),
            subs.Word(1.2, 1.6, "did"), subs.Word(3.0, 3.5, "that!")]


def test_chunking_breaks_on_gap_and_size():
    chunks = subs.chunk_words(_words(), max_words=3)
    assert [len(c) for c in chunks] == [4, 1]          # orphan "did" glued on; "that!" after a pause stays alone


def test_build_ass_vertical():
    ass = subs.build_ass(_words(), 4.0, True, hook="He did WHAT", credit="Clip from x",
                         profanity=["damn"], keywords=["THAT"])
    assert "PlayResX: 1080" in ass and "PlayResY: 1920" in ass
    assert ass.count("Style: Sub") == 1 and "HE DID WHAT" in ass and "Clip from x" in ass
    assert r"\c&H00F0FF&" in ass                      # highlight colour converted to &HBBGGRR&
    assert ass.count("Dialogue: 0,") == 5             # one event per word


def test_censor_and_escape():
    assert subs.censor("what the fuck", ["fuck"]) == "what the f***"
    ass = subs.build_ass([subs.Word(0, 1, "{hack}")], 1, False)
    assert "{hack}" not in ass


def test_srt():
    srt = subs.build_srt(_words())
    assert srt.startswith("1\n00:00:00,000 --> ") and "that!" in srt


# ---------------------------------------------------------------- vertical layout
def test_choose_layout():
    big = [Face(0.5, 0.4, 0.2, 0.3)] * 6
    assert choose_layout(big, 8)[0] == "crop"
    cam = [Face(0.85, 0.8, 0.07, 0.1)] * 6
    assert choose_layout(cam, 8)[0] == "split"
    moving = [Face(0.85, 0.8, 0.07, 0.1), Face(0.75, 0.7, 0.07, 0.1), Face(0.9, 0.85, 0.06, 0.1),
              Face(0.8, 0.75, 0.07, 0.1), Face(0.7, 0.8, 0.06, 0.1)]       # players moving in a corner
    assert choose_layout(moving, 8)[0] == "track"
    assert choose_layout([], 8)[0] == "track"                    # sports/gameplay: follow the action
    bystander = [Face(0.4, 0.5, 0.09, 0.12)] * 6                  # small face in a wide shot
    assert choose_layout(bystander, 8)[0] == "track"


def test_vertical_filters_are_even_and_bounded():
    f = vertical_filter("crop", Face(0.99, 0.5, 0.2, 0.2), 1920, 1080)
    assert "crop=606:1080:1314:0" in f
    f = vertical_filter("split", Face(0.9, 0.85, 0.07, 0.1), 1920, 1080)
    assert "vstack[vbase]" in f
    fb = vertical_filter("fit_blur", None, 1280, 720)
    assert "[vbase]" in fb and "scale=1080:1920[bg]" in fb


# ---------------------------------------------------------------- scheduling / metadata / analytics
def test_next_slot_basic_and_caps():
    tz = ZoneInfo("Asia/Kolkata")
    now = datetime(2026, 11, 16, 5, 0, tzinfo=timezone.utc)   # 10:30 IST
    slots = ["11:00", "15:00", "19:00"]
    s1 = next_slot(slots, [], tz, now, timedelta(minutes=20), 6)
    assert s1.astimezone(tz).strftime("%H:%M") == "11:00"
    s2 = next_slot(slots, [s1], tz, now, timedelta(minutes=20), 6)
    assert s2.astimezone(tz).strftime("%H:%M") == "15:00"
    # lead time pushes past 11:00
    s3 = next_slot(slots, [], tz, now, timedelta(minutes=45), 6)
    assert s3.astimezone(tz).strftime("%H:%M") == "15:00"
    # per-day cap of 1 -> tomorrow
    s4 = next_slot(slots, [s1], tz, now, timedelta(minutes=20), 1)
    assert s4.astimezone(tz).date() == (now.astimezone(tz).date() + timedelta(days=1))


def test_as_list_normalizes_llm_output():
    from streamclipper.publish.metadata import as_list
    assert as_list(["a", "b, c"]) == ["a", "b", "c"]
    assert as_list("x y, z") == ["x y", "z"]
    assert as_list("#a #b,c", split_spaces=True) == ["#a", "#b", "c"]
    assert as_list(None) == []


def test_chapters_and_ts():
    assert fmt_ts(3725) == "1:02:05" and fmt_ts(65) == "1:05"
    assert chapters([(0, "a"), (30, "b")]) == ""
    assert chapters([(0, "a"), (30, "b"), (75, "c")]).splitlines()[0] == "0:00 a"


def test_compute_weights():
    rows = [("funny", 10000)] * 3 + [("story", 100)] * 3 + [("rare", 5)]
    w = compute_weights(rows)
    assert w["funny"][0] > 1 > w["story"][0] and "rare" not in w
    assert all(0.8 <= v[0] <= 1.2 for v in w.values())


# ---------------------------------------------------------------- any-platform links
def test_detect_platform_and_names(tmp_path):
    from streamclipper.monitor.platforms import detect_platform, guess_name, normalize_url

    cases = {
        "https://www.twitch.tv/somebody": ("twitch", "somebody"),
        "https://www.youtube.com/@creator/live": ("youtube", "creator"),
        "https://kick.com/streamer": ("kick", "streamer"),
        "https://www.tiktok.com/@user/live": ("tiktok", "user"),
        "https://live.bilibili.com/12345": ("bilibili", "12345"),
        "https://example.com/stream/master.m3u8": ("direct", "stream"),
        "rtmp://10.0.0.5/live/key": ("direct", "key"),
        "https://some-random-site.tv/watch/abc": ("web", "abc"),
    }
    for url, (plat, name) in cases.items():
        assert detect_platform(url) == plat, url
        assert guess_name(url, plat) == name, url
    assert normalize_url("twitch.tv/x") == "https://twitch.tv/x"
    f = tmp_path / "vod.mp4"
    f.write_bytes(b"x")
    assert detect_platform(normalize_url(str(f))) == "file"


def test_close_peaks_split_instead_of_blob():
    hype = np.zeros(300)
    hype[74], hype[100] = 2.1, 2.2
    # default gap (20 s) keeps both peaks; old 45 s gap dropped the 74 s one
    assert cand.find_peaks(hype, 1.8) == [74, 100]
    w = cand.windows_from_peaks([74, 100], hype, 300, 20, 10)
    assert len(w) == 2 and w[0].end == 87 and w[1].start == 87     # split halfway between the peaks
    hype[85] = 2.0
    w = cand.windows_from_peaks([74, 85], hype, 300, 20, 10)       # 11 s apart: same event -> merged
    assert len(w) == 1 and w[0].start == 54 and w[0].end == 95


def test_hallucination_filter():
    from streamclipper.transcribe.whisper import is_hallucination
    assert not is_hallucination("Oh, he got it!", 0.53, -0.5, 1.3)      # real speech over crowd noise
    assert is_hallucination("Thanks for watching!", 0.1, -0.2, 1.1)
    assert is_hallucination("la la la la", 0.7, -1.4, 1.5)              # silent + unsure
    assert is_hallucination("go go go go go go go go go go", 0.1, -0.3, 3.0)  # repetition loop
    # calibrated on real data: commentary over a loud crowd must survive...
    assert not is_hallucination("One foot dunking is not...", 0.60, -0.86, 0.77, 0.98)
    assert not is_hallucination("I can't catch on TV!", 0.35, -0.82, 0.77, 0.92)
    # ...while text invented over music is dropped
    assert is_hallucination("I'm not sure what to do with this.", 0.71, -0.94, 2.35, 0.37)
    assert is_hallucination("Come on now, I'll just go and check on the others.", 0.57, -1.82, 2.06, 0.5)
    assert is_hallucination('["Pomp and Circumstance"]', 0.55, -1.07, 0.76, 0.48)
    assert is_hallucination("※2 -3 -2 -1 -2 ※3", 0.3, -0.5, 1.0, 0.9)
    assert is_hallucination("我有 cis ಈ੭੭੟ ไে", 0.3, -0.5, 1.0, 0.9)


def test_snap_to_phrases_never_cuts_sentences():
    from types import SimpleNamespace as W
    from streamclipper.edit.render import snap_to_phrases
    words = [W(start=52.8, end=53.3, text="Show"), W(start=53.3, end=53.6, text="up"),
             W(start=53.6, end=54.2, text="then,"), W(start=54.2, end=54.9, text="Dunkshin!"),
             W(start=85.9, end=86.4, text="Tim"), W(start=86.4, end=86.9, text="Hazard's"),
             W(start=86.9, end=87.2, text="been"), W(start=87.2, end=88.6, text="spinning.")]
    a, b = snap_to_phrases(53.5, 86.0, words, total=102)
    assert a < 52.8                                     # start pulled back to the whole phrase
    assert b < 85.9                                     # "Tim Hazard's..." starts in the last 3 s: next clip's
    a, b = snap_to_phrases(53.5, 87.0, words + [W(start=88.7, end=95.0, text="long")], total=102)
    assert b < 85.9 or b > 88.6                         # never mid-sentence
    a, b = snap_to_phrases(53.5, 70.0, [W(start=66.0, end=71.0, text="this is a long line.")], total=102)
    assert b > 71.0                                     # phrase began 4 s before the end: finish it


def test_stretched_words_are_capped():
    words = [subs.Word(10.06, 15.68, "to"), subs.Word(15.68, 16.2, "beat")]
    assert subs.clean_words(words)[0].end <= 10.06 + 0.8 + 1e-9


# ---------------------------------------------------------------- shot edges / tracking
def test_choose_edges_snaps_to_shots():
    from streamclipper.edit.shots import choose_edges
    cuts = [48.0, 53.2, 61.0, 73.5, 83.1, 85.0, 99.8]
    # dunk peak at 73: start at the shot containing a=53.5, end at the cut where the next item begins
    a, b, on_a, on_b = choose_edges(53.5, 86.0, 73.0, cuts, min_len=15, max_len=59.5)
    assert (a, on_a) == (53.2, True)
    assert (b, on_b) == (83.1, True)          # 83.1-85.0 is the next item's countdown card: end before it
    # never end inside the payoff: a cut 0.5 s after the peak is ignored
    a, b, _, on_b = choose_edges(60.0, 76.0, 75.5, [61.0, 76.0], min_len=10, max_len=None)
    assert b == 76.0 and not on_b
    # no cuts -> unchanged
    assert choose_edges(10, 30, 20, [], min_len=10, max_len=None) == (10, 30, False, False)


def test_split_point_prefers_transition():
    scene = np.zeros(120)
    scene[83] = 0.9                             # countdown card between two dunks
    assert cand.split_point(73, 99, scene) == 83
    assert cand.split_point(73, 99, np.zeros(120)) == 86   # no clear cut -> halfway


def test_track_follows_moving_action():
    from streamclipper.edit import track as T
    n = 36
    frames = np.zeros((n, T.AH, T.AW), dtype=np.float32)
    for i in range(n):                          # bright box moving left -> right
        x = int(20 + i * (T.AW - 60) / n)
        frames[i, 40:70, x:x + 16] = 255
    tr = T.compute_track(frames, 0.316)
    assert tr.coverage > 0.8
    assert tr.centers[3] < 0.35 and tr.centers[-1] > 0.6    # camera moved with the action
    steps = np.abs(np.diff(tr.centers))
    assert steps.max() < 0.08                                # smooth, no jumps within the shot


def test_sendcmd_file(tmp_path):
    from streamclipper.edit.track import Track, sendcmd_file
    p = tmp_path / "c.txt"
    x0 = sendcmd_file(Track([0, 0.5, 1.0], [0.0, 0.5, 1.0], 0.9), 1280, 404, p)
    lines = p.read_text().splitlines()
    assert x0 == 0 and lines[0] == "0.000 crop@trk x 0;" and lines[-1] == "1.000 crop@trk x 876;"


def test_static_tail_detects_end_cards():
    from streamclipper.edit.shots import static_tail
    moving = [8.0, 9.5, 7.2, 10.1, 8.8, 9.0]
    assert static_tail(moving + [0.2] * 6, 6) == 1.0        # logo holds still for the last second
    assert static_tail(moving + [0.2] * 2, 6) is None       # too short to be an end card
    assert static_tail(moving + [1.1] * 8, 6) is None       # slow-motion replay is NOT an end card
    assert static_tail(moving, 6) is None


def test_no_speech_windows_use_hype_only():
    from streamclipper.detect.engine import combine
    assert combine(-1, 1.6, 1.8) == 4.0      # borderline peak with nothing to read: fails threshold 7
    assert combine(-1, 3.2, 1.8) == 8.0      # clearly hyped visual moment still passes


def test_garbage_filter_keeps_real_languages():
    from streamclipper.transcribe.whisper import _looks_like_garbage
    assert not _looks_like_garbage("すごい！今のプレイ見た？神業だよ")          # Japanese: kana + kanji
    assert not _looks_like_garbage("क्या शॉट मारा भाई, एकदम ज़बरदस्त")      # Hindi
    assert not _looks_like_garbage("Вот это да, какой бросок!")               # Russian
    assert not _looks_like_garbage("GG wow 대박 진짜")                         # English + Korean mix
    assert _looks_like_garbage("我有 cis ಈ੭੭੟ ไে")                              # 4 unrelated scripts


def test_camera_path_is_smooth_and_calm():
    from streamclipper.edit import track as T
    fps = T.OUT_FPS
    # jittery action that stays within the frame -> camera locked (no motion at all)
    jitter = 0.5 + 0.05 * np.sin(np.arange(4 * fps) * 1.7)
    assert np.ptp(T.plan_shot(jitter, 0.316)) == 0
    # action jumping across the frame -> smooth pan within speed/accel limits, no stepping
    jump = np.r_[np.full(2 * fps, 0.25), np.full(3 * fps, 0.75)]
    p = T.plan_shot(jump.astype(float), 0.316)
    step = np.abs(np.diff(p))
    assert step.max() <= T.MAX_SPEED / fps * 1.05                       # never faster than the cap
    assert np.abs(np.diff(np.diff(p))).max() <= T.MAX_ACCEL / fps / fps * 1.5
    assert p[-1] > 0.6                                                   # it did get there
    m = np.sign(np.round(np.diff(p), 6)); m = m[m != 0]
    assert np.sum(m[1:] != m[:-1]) == 0                                  # one pan, no hunting back and forth
    # gaps (quiet/graphics frames) are interpolated, not frozen then snapped
    gap = np.r_[np.full(fps, 0.2), np.full(fps, np.nan), np.full(2 * fps, 0.8)]
    assert np.abs(np.diff(T.plan_shot(gap, 0.316))).max() <= T.MAX_SPEED / fps * 1.05


def test_hls_endlist_means_not_live():
    from streamclipper.monitor.live_check import hls_ended
    live = "#EXTM3U\n#EXT-X-TARGETDURATION:4\n#EXT-X-MEDIA-SEQUENCE:37\n#EXTINF:4.0,\nlive37.ts\n"
    assert not hls_ended(live)
    assert hls_ended(live + "#EXT-X-ENDLIST\n")


def test_webcam_found_among_game_faces():
    from streamclipper.edit.vertical import find_webcam
    cam = Face(0.09, 0.54, 0.04, 0.07)                       # left-edge facecam, mid-height
    npc = [Face(0.5, 0.4, 0.13, 0.2), Face(0.35, 0.48, 0.07, 0.1), Face(0.6, 0.3, 0.05, 0.08)]
    frames = [[npc[i % 3], cam] for i in range(4)] + [[npc[0]], [], [cam], [npc[1]]]
    f = find_webcam(frames, 8)
    assert f and abs(f.cx - 0.09) < 0.01
    assert choose_layout([fs[0] for fs in frames if fs], 8, webcam=f)[0] == "split"
    moving = [[Face(0.1 + 0.05 * i, 0.8, 0.05, 0.08)] for i in range(8)]   # a character walking
    assert find_webcam(moving, 8) is None


def test_chat_spam_burst_counts_as_reaction():
    chat = [(100 + i * 0.5, "L VAPE" if i % 2 else "LL vapee!!") for i in range(12)] + [(300, "L VAPE")]
    flags = cand.spam_flags(chat)
    assert all(flags[:12]) and not flags[12]        # 12 near-identical messages together; one alone isn't


def test_chat_consensus_not_averaged_away_by_quiet_audio():
    n = 300
    audio = {t: -20.0 + (8.0 if 200 <= t < 230 else 0.0) for t in range(n)}   # loud game music later
    chat = [(t + 0.1 * k, "background") for t in range(0, n, 3) for k in range(2)]
    chat += [(100 + i * 0.3, "L VAPE") for i in range(60)]                  # audience reacts at ~100 s
    h = cand.hype_curve(n, audio, {}, chat, {"audio": 1.0, "chat": 1.5, "keywords": 1.2, "scene": 0.4}, [])
    assert any(abs(p - 100) < 15 for p in cand.find_peaks(h, 1.8, 20))


def test_twitch_ad_playlist_detection():
    from streamclipper.capture.twitch_ads import overlaps_ad, playlist_has_ad
    ad = ('#EXTM3U\n#EXT-X-DATERANGE:ID="stitched-ad-1",CLASS="twitch-stitched-ad",START-DATE="x"\n'
          "#EXTINF:2.000,Amazon|1\na.ts\n#EXTINF:2.000,Amazon|1\nb.ts\n")
    assert playlist_has_ad(ad)
    assert not playlist_has_ad(ad + "#EXTINF:2.000,live\nc.ts\n#EXTINF:2.000,live\nd.ts\n")
    assert not playlist_has_ad("#EXTM3U\n#EXTINF:2.000,live\na.ts\n")
    assert overlaps_ad([(0.0, 40.0)], 29, 59) and not overlaps_ad([(0.0, 40.0)], 60, 90)


def test_ytdlp_youtube_chat_line():
    import json
    from streamclipper.signals.chat import parse_ytdlp_chat_line
    item = {"liveChatTextMessageRenderer": {
        "message": {"runs": [{"text": "W "}, {"emoji": {"shortcuts": [":fire:"]}}]},
        "authorName": {"simpleText": "@a"}, "timestampUsec": "1763481600000000",
        "authorBadges": [{"liveChatAuthorBadgeRenderer": {"tooltip": "Moderator"}}]}}
    gift = {"giftMessageViewModel": {"text": {"content": "sent a gift"}}}
    line = json.dumps({"replayChatItemAction": {"actions": [{"addChatItemAction": {"item": item}},
                                                            {"addChatItemAction": {"item": gift}}]}})
    assert parse_ytdlp_chat_line(line) == [("@a", "W :fire:", True, 1763481600.0)]
    assert parse_ytdlp_chat_line("not json") == []


def test_spam_needs_different_viewers():
    promo = [(10 + i, "GO WATCH MY CLIP", "bot") for i in range(6)]
    crowd = [(50 + i * 0.5, "W", f"u{i}") for i in range(6)]
    flags = cand.spam_flags(promo + crowd)
    assert not any(flags[:6]) and all(flags[6:])


def test_clip_never_shorter_than_minimum():
    from streamclipper.detect.engine import ensure_min_len
    from streamclipper.detect.llm_judge import sanitize
    j = sanitize({"start": 14, "end": 49}, 0, 23.0, 15, 58)        # LLM asks past the window's end
    assert j.end - j.start >= 15 and 0 <= j.start and j.end <= 23
    j = sanitize({"start": 0, "end": 3}, 0, 40.0, 15, 58)           # pinned at the window start
    assert (j.start, j.end) == (0, 15)
    assert ensure_min_len(15.5, 23.0, 15, 0, 300) == (11.75, 26.75)  # grows into the footage around it
    assert ensure_min_len(2.0, 9.0, 15, 0, 300) == (0.0, 15.0)        # stream start: grow forward
    assert ensure_min_len(290.0, 298.0, 15, 0, 300) == (285.0, 300.0)  # live edge: grow backward
    assert ensure_min_len(10.0, 40.0, 15, 0, 300) == (10.0, 40.0)


def test_background_spam_and_stream_complaints_are_not_hype():
    # "W" all stream long is background; the same "W" rate during a moment isn't a reaction
    chat = [(t + 0.1 * k, "W", f"u{t}{k}") for t in range(0, 300, 2) for k in range(3)]
    flags = cand.spam_flags(chat)
    assert not any(flags)
    burst = [(150 + i * 0.3, "L VAPE", f"v{i}") for i in range(20)]
    flags = cand.spam_flags(chat + burst)
    assert all(flags[len(chat):])
    lag = [(100 + i * 0.3, "LAGGGG", f"l{i}") for i in range(30)]
    assert not any(cand.spam_flags(chat + lag)[len(chat):])
    assert cand.is_complaint(cand.chat_key("lagggg")) and cand.is_complaint(cand.chat_key("BUFFERING??"))
    assert not cand.is_complaint(cand.chat_key("L DRIVER"))


def test_broadcast_glitch_detected_by_lost_overlay():
    from streamclipper.detect.engine import broken_frames
    rng = np.random.default_rng(0)
    overlay = np.zeros((36, 64), bool)
    overlay[10, 0:20] = overlay[25, 0:20] = overlay[10:26, 20] = True          # a webcam box border
    good = [overlay | (rng.random((36, 64)) < 0.08) for _ in range(20)]         # moving game + fixed box
    glitch = [rng.random((36, 64)) < 0.1 for _ in range(4)]                    # garbage: border gone
    assert broken_frames(good) == 0
    assert broken_frames(good + glitch) == 4
    assert broken_frames([rng.random((36, 64)) < 0.1 for _ in range(20)]) == 0  # no overlay: can't judge


def test_judge_calibration_caps_unsupported_scores():
    from streamclipper.detect.llm_judge import calibrate, sanitize

    assert calibrate(8, {"payoff": False, "reaction": "strong", "standalone": True}) == 4.5   # nothing happened
    assert calibrate(8, {"payoff": True, "reaction": "none", "standalone": True}) == 4.5      # nobody reacted
    assert calibrate(9, {"payoff": True, "reaction": "some", "standalone": True}) == 7.0      # mild reaction
    assert calibrate(9, {"payoff": True, "reaction": "strong", "standalone": False}) == 8.0   # needs context
    assert calibrate(9, {"payoff": "true", "reaction": "STRONG", "standalone": "yes"}) == 9.0 # strings work
    assert calibrate(8, {}) == 8.0                                                            # old-style reply
    assert calibrate(42, {}) == 10.0 and calibrate(-3, {}) == 0.0
    j = sanitize({"score": 8, "payoff": False, "reaction": "none", "start": 10, "end": 40}, 0, 60, 15, 55)
    assert j.score == 4.5


def test_chat_excitement_and_evidence_cap(settings_tmp=None):
    from streamclipper.detect.llm_judge import evidence_cap, is_excited

    for m in ["KEKW", "W", "NO WAY", "noooo", "wwwww", "ㅋㅋㅋㅋ", "😂", "that was nuts!", "monkaS", "lets go"]:
        assert is_excited(m), m
    for m in ["ok", "yes", "hi", "nice", "ty", "o7", "too loud", "good", "?", ""]:
        assert not is_excited(m), m
    assert evidence_cap(9, excitement=0.0, n_chat=20, hype=2.2) == 4.5    # calm chat, normal audio: not a clip
    assert evidence_cap(9, excitement=0.0, n_chat=20, hype=3.5) == 9      # audio spike still counts
    assert evidence_cap(9, excitement=0.6, n_chat=20, hype=2.0) == 9      # excited chat
    assert evidence_cap(9, excitement=0.0, n_chat=2, hype=2.0) == 9       # too little chat to judge
    assert evidence_cap(9, excitement=0.0, n_chat=0, hype=2.0) == 9       # platform without chat reading
