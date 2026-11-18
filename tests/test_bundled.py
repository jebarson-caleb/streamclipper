"""Bundled AI engine: backend choice, idle timeout parsing, GPU device pick."""
import subprocess

from streamclipper import vendor
from streamclipper.config import Settings, set_settings
from streamclipper.detect import llama_server, llm


def _settings(tmp_path, **llm_kw):
    s = Settings(data_dir=tmp_path)
    for k, v in llm_kw.items():
        setattr(s.llm, k, v)
    set_settings(s)
    return s


def test_backend_auto_uses_ollama_from_source(tmp_path, monkeypatch):
    _settings(tmp_path, backend="auto")
    monkeypatch.setattr(llama_server, "available", lambda: True)
    monkeypatch.setattr(vendor, "packaged", lambda: False)
    assert llm.bundled() is False


def test_backend_auto_uses_bundled_engine_when_installed(tmp_path, monkeypatch):
    _settings(tmp_path, backend="auto")
    monkeypatch.setattr(llama_server, "available", lambda: True)
    monkeypatch.setattr(vendor, "packaged", lambda: True)
    assert llm.bundled() is True


def test_backend_explicit(tmp_path):
    _settings(tmp_path, backend="llamacpp")
    assert llm.bundled() is True
    _settings(tmp_path, backend="ollama")
    assert llm.bundled() is False


def test_idle_seconds(tmp_path):
    for raw, want in [("30s", 30), ("2m", 120), ("1h", 3600), ("45", 45), ("junk", 30)]:
        _settings(tmp_path, keep_alive=raw)
        assert llama_server._idle_seconds() == want


def test_model_not_ready_until_verified(tmp_path):
    _settings(tmp_path)
    p = llama_server.model_path()
    p.parent.mkdir(parents=True)
    p.write_bytes(b"x")
    assert not llama_server.model_ready()          # no .ok marker: download not verified
    p.with_suffix(".ok").write_text("hash")
    assert llama_server.model_ready()


def test_model_download_survives_a_stalled_server(tmp_path, monkeypatch):
    """The server sends part of the file, then goes silent without closing: drop it, resume with Range."""
    import asyncio
    import hashlib

    import httpx

    data = bytes(range(256)) * 40
    calls = []

    async def handler(request):
        rng = request.headers.get("range")
        calls.append(rng)
        start = int(rng.split("=")[1].rstrip("-")) if rng else 0

        async def body():
            yield data[start:start + 3000]
            if start == 0:
                await asyncio.sleep(30)          # stall: never finish, never close
            else:
                yield data[start + 3000:]
        return httpx.Response(206 if rng else 200, headers={"content-length": str(len(data) - start)}, content=body())

    real_client = httpx.AsyncClient
    monkeypatch.setattr(llama_server.httpx, "AsyncClient",
                        lambda **kw: real_client(transport=httpx.MockTransport(handler), **kw))
    monkeypatch.setattr(llama_server, "STALL_SECONDS", 0.3)
    monkeypatch.setattr(llama_server, "RETRY_DELAY", 0)
    _settings(tmp_path, gguf_url="https://example.test/model.gguf", gguf_sha256=hashlib.sha256(data).hexdigest())
    llama_server.DL.update(state="idle", done=0, total=0, error="")

    asyncio.run(llama_server.download_model())

    assert llama_server.DL["state"] == "ready", llama_server.DL
    assert llama_server.model_path().read_bytes() == data
    assert calls == [None, "bytes=3000-"]


def test_previous_model_is_removed_once_current_is_ready(tmp_path):
    import asyncio

    _settings(tmp_path)
    cur = llama_server.model_path()
    cur.parent.mkdir(parents=True)
    old = cur.parent / "qwen2.5-3b-instruct-q4_k_m.gguf"
    for f in (cur, old):
        f.write_bytes(b"x")
        f.with_suffix(".ok").write_text("hash")
    user_file = cur.parent / "my-own-model.gguf"     # no .ok marker: not ours, never touched
    user_file.write_bytes(b"x")
    llama_server.DL.update(state="idle")
    asyncio.run(llama_server.download_model())
    assert cur.exists() and not old.exists() and not old.with_suffix(".ok").exists() and user_file.exists()


def test_gpu_layers_follow_free_video_memory(tmp_path):
    _settings(tmp_path)          # no model file yet: assumes a ~2.4 GB model with 36 layers
    assert llama_server.gpu_layers(None) == -1       # not NVIDIA: let llama.cpp fit it
    assert llama_server.gpu_layers(900) == 0         # a game/OBS holds the GPU: CPU, don't crash
    assert llama_server.gpu_layers(8000) == 36       # plenty: everything on the GPU
    assert 0 < llama_server.gpu_layers(2400) < 36    # in between: partial offload


def test_gpu_device_prefers_dedicated(tmp_path, monkeypatch):
    _settings(tmp_path)
    out = ("Available devices:\n  Vulkan0: Intel(R) Iris(R) Xe Graphics (12130 MiB, 11362 MiB free)\n"
           "  Vulkan1: NVIDIA GeForce RTX 3050 Laptop GPU (3964 MiB, 2635 MiB free)\n")
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 0, stdout=out))
    assert llama_server._gpu_device() == "Vulkan1"
    only_igpu = "Available devices:\n  Vulkan0: Intel(R) UHD Graphics (8000 MiB, 7000 MiB free)\n"
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 0, stdout=only_igpu))
    assert llama_server._gpu_device() is None
