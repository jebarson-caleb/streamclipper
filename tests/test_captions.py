"""Caption styles: validation, ASS output, storage, font route, real render."""
import shutil

import pytest
from fastapi.testclient import TestClient

from streamclipper import db
from streamclipper.config import Settings, set_settings
from streamclipper.edit import caption_style as cs
from streamclipper.edit import subtitles as subs

LOCAL = "http://127.0.0.1:8765"
WORDS = [subs.Word(0, 0.5, "no"), subs.Word(0.5, 1.0, "way"), subs.Word(1.0, 1.5, "that")]


@pytest.fixture
def settings(tmp_path):
    s = Settings(data_dir=tmp_path)
    s.ensure_dirs()
    set_settings(s)
    db.reset_engine()
    yield s
    db.reset_engine()


def test_validate_rejects_injection_and_clamps():
    st = cs.validate({"font": "Arial\nDialogue: 0,0:00:00.00,9:00:00.00,Sub,,0,0,0,,HACKED",
                      "text_color": "#FFFFFF,1\nDialogue: x", "highlight_color": "red", "size": "9999",
                      "angle": "-90", "outline": "-5", "words_per_line": "abc", "position": "../../etc",
                      "box": "on", "preset": "nonsense"})
    assert st.font == "Arial Black" and st.text_color == "#FFFFFF" and st.highlight_color == "#FFF000"
    assert st.size == 140 and st.angle == -15 and st.outline == 0 and st.words_per_line == 3
    assert st.position == "low" and st.box is True and st.preset == "custom"
    ass = subs.build_ass(WORDS, 2, True, style=st)
    assert "HACKED" not in ass and ass.count("Dialogue: 0,") == len(WORDS)


@pytest.mark.parametrize("name", list(cs.PRESETS))
def test_every_preset_builds(name):
    st = cs.preset(name)
    ass = subs.build_ass(WORDS, 2, True, style=st)
    assert f"Style: Sub,{st.font}," in ass and "Dialogue: 0," in ass


def test_style_options_reach_the_ass_file():
    st = cs.validate({"preset": "custom", "font": "Anton", "angle": "7", "box": "1", "position": "top",
                      "highlight_color": "#0038ff", "glow": "1", "uppercase": "0"})
    ass = subs.build_ass(WORDS, 2, True, style=st)
    sub_line = next(line for line in ass.splitlines() if line.startswith("Style: Sub,"))
    fields = sub_line.split(",")
    assert fields[1] == "Anton" and fields[14] == "7" and fields[15] == "3" and fields[18] == "8"
    assert r"\c&HFF3800&" in ass and r"\blur3" in ass and "no" in ass and "NO" not in ass


def test_save_and_load_roundtrip(settings):
    assert cs.load() == cs.CaptionStyle()
    cs.save(cs.validate({"preset": "neon"}))
    assert cs.load().font == "Bebas Neue" and cs.load().glow is True


def test_corrupt_stored_style_falls_back():
    assert cs.from_json("not json") == cs.CaptionStyle()
    assert cs.from_json('{"font": "Evil\\nFont", "size": 50}').font == "Arial Black"


@pytest.fixture
def client(settings):
    from streamclipper.web.app import app

    return TestClient(app, base_url=LOCAL, follow_redirects=False)


@pytest.mark.parametrize("name", ["../config.yaml", "..%2F..%2Fpyproject.toml", "OFL.txt", "anton-OFL.txt",
                                  "C:%5CWindows%5Cwin.ini"])
def test_font_route_only_serves_listed_fonts(client, name):
    assert client.get(f"/fonts/{name}").status_code == 404


def test_font_route_serves_bundled_font(client):
    if not (cs.fonts_dir() / "Anton-Regular.ttf").exists():
        pytest.skip("fonts not fetched (scripts/fetch_vendor.ps1)")
    r = client.get("/fonts/Anton-Regular.ttf")
    assert r.status_code == 200 and r.headers["content-type"] == "font/ttf"


def test_captions_page_and_save(client):
    assert client.get("/captions").status_code == 200
    r = client.post("/captions/save", data={"preset": "marker", "font": "Permanent Marker", "angle": "-4"})
    assert r.status_code == 303
    assert cs.load().font == "Permanent Marker" and cs.load().angle == -4


def test_captions_save_refuses_cross_site(client):
    assert client.post("/captions/save", data={"font": "Anton"}, headers={"origin": "https://evil.com"}).status_code == 403


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg not installed")
def test_real_frame_preview_renders(client):
    r = client.post("/captions/preview", data={"preset": "comic", "font": "Luckiest Guy"})
    assert r.status_code == 200 and "data:image/png;base64," in r.text
