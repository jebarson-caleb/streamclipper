"""Security regressions: argument injection, cross-site requests, DNS rebinding, secrets, template and link
injection."""
import pytest
from fastapi.testclient import TestClient

from streamclipper.config import Settings, set_settings
from streamclipper.monitor.platforms import normalize_url
from streamclipper.publish.metadata import fill_template, no_links

LOCAL = "http://127.0.0.1:8765"


@pytest.fixture
def settings(tmp_path):
    s = Settings(data_dir=tmp_path)
    s.ensure_dirs()
    set_settings(s)
    return s


# ---------------------------------------------------------------- links can never become command-line options
@pytest.mark.parametrize("bad", [
    "--exec=calc ://x", "-o C:/x", "--config-locations=//evil/share/x.conf https://x.tv",
    "file:///C:/Windows/win.ini", "concat:a.mp4|b.mp4", "javascript:alert(1)", "https://", "", "   ",
    "https://twitch.tv/a b", "https://twitch.tv/\x00x", "ftp://example.com/x.mp4",
    "C:/missing/x.mp4", "\\server\\share\\x.mp4", "\\\\server\\share\\x.mp4", "/etc/passwd",
])
def test_normalize_url_rejects_unsafe(bad):
    with pytest.raises(ValueError):
        normalize_url(bad)


def test_normalize_url_rejects_folders(tmp_path):
    with pytest.raises(ValueError):
        normalize_url(str(tmp_path))


@pytest.mark.parametrize("good,want", [
    ("twitch.tv/name", "https://twitch.tv/name"),
    ("https://www.youtube.com/@x/live", "https://www.youtube.com/@x/live"),
    ("rtmp://live.example.com/app/key", "rtmp://live.example.com/app/key"),
    ("srt://1.2.3.4:9000", "srt://1.2.3.4:9000"),
])
def test_normalize_url_accepts_links(good, want):
    assert normalize_url(good) == want


def test_normalize_url_accepts_video_file(tmp_path):
    f = tmp_path / "vod.mp4"
    f.write_bytes(b"x")
    assert normalize_url(f'"{f}"') == str(f.resolve())


def test_streamlink_command_ends_options_before_url(settings):
    from streamclipper.capture.recorder import Recorder

    r = Recorder.__new__(Recorder)
    r.channel = type("C", (), {"url": "https://twitch.tv/x"})()
    r.backend = "streamlink"
    cmd = r._source_cmd()
    assert cmd[cmd.index("https://twitch.tv/x") - 1] == "--"


def test_every_ytdlp_and_streamlink_call_uses_end_of_options():
    """Static check: each place that hands a URL to yt-dlp/streamlink puts "--" before it."""
    import re
    from pathlib import Path

    src = Path(__file__).resolve().parent.parent / "streamclipper"
    for f in src.rglob("*.py"):
        text = f.read_text(encoding="utf-8")
        for m in re.finditer(r'"-m", "(?:yt_dlp|streamlink)"', text):
            call = re.split(r"stdin=|capture_output=|stdout=|\n\s*\n", text[m.start():m.start() + 600])[0]
            assert '"--"' in call, f"{f.name}: {call[:120]}"


# ---------------------------------------------------------------- dashboard: only this PC's own pages may drive it
@pytest.fixture
def client(settings):
    from streamclipper.web.app import app

    return TestClient(app, base_url=LOCAL, follow_redirects=False)   # no `with`: the pipeline doesn't start


def test_foreign_host_header_is_refused(client):
    """DNS rebinding: evil.com resolving to 127.0.0.1 sends Host: evil.com."""
    assert client.get("/logs", headers={"host": "evil.com"}).status_code == 403
    assert client.get("/logs").status_code == 200


def test_cross_site_post_is_refused(client):
    assert client.post("/pause", headers={"origin": "https://evil.com"}).status_code == 403
    assert client.post("/pause", headers={"sec-fetch-site": "cross-site"}).status_code == 403
    assert client.post("/pause", headers={"origin": "null"}).status_code == 403   # sandboxed iframe / data: page
    assert client.post("/quick", data={"url": "C:/secret.mp4", "permission": "own"},
                       headers={"origin": "http://evil.com"}).status_code == 403


def test_same_origin_and_cli_posts_work(client):
    assert client.post("/pause", headers={"origin": LOCAL, "sec-fetch-site": "same-origin"}).status_code == 303
    assert client.post("/pause").status_code == 303     # CLI `streamclipper add` sends no Origin
    client.post("/pause")                                 # leave it unpaused


def test_security_headers(client):
    h = client.get("/logs").headers
    assert h["x-frame-options"] == "DENY" and h["x-content-type-options"] == "nosniff"
    assert h["referrer-policy"] == "same-origin"   # no-referrer would make browsers send Origin: null
    assert "frame-ancestors 'none'" in h["content-security-policy"]


def test_api_docs_disabled(client):
    assert client.get("/docs").status_code == 404 and client.get("/openapi.json").status_code == 404


def test_settings_page_hides_secrets(settings):
    from streamclipper.web.app import redacted_config

    settings.notify.telegram_bot_token = "123456:SECRET-TOKEN"
    settings.notify.discord_webhook = "https://discord.com/api/webhooks/1/SECRET"
    out = redacted_config(settings)
    assert "SECRET" not in out and "•••• (set)" in out
    assert "client_secret.json" in out          # a file path, not a secret


def test_client_secret_upload_validates(client, settings):
    r = client.post("/setup/client-secret", files={"file": ("x.json", b'{"web": {}}', "application/json")})
    assert r.status_code == 303 and "Could%20not%20use" in r.headers["location"]
    r = client.post("/setup/client-secret", files={"file": ("x.json", b"x" * 70000, "application/json")})
    assert "too%20large" in r.headers["location"]


# ---------------------------------------------------------------- published text
def test_templates_allow_only_plain_fields():
    assert fill_template("{title} by {channel}", title="Clutch", channel="me") == "Clutch by me"
    assert "__class__" not in fill_template("{title.__class__.__mro__}", title="x") or \
        fill_template("{title.__class__.__mro__}", title="x") == "{title.__class__.__mro__}"
    assert fill_template("50% off {title} :{", title="x") == "50% off x :{"     # stray brace: no crash


@pytest.mark.parametrize("spam", [
    "Insane clutch http://scam.example/free", "Insane clutch www.free-robux.com", "Insane clutch free-robux.ru",
    "Insane clutch https://bit.ly/abc",
])
def test_ai_text_loses_links(spam):
    assert no_links(spam) == "Insane clutch"
