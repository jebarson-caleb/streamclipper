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


def test_gpu_device_prefers_dedicated(tmp_path, monkeypatch):
    _settings(tmp_path)
    out = ("Available devices:\n  Vulkan0: Intel(R) Iris(R) Xe Graphics (12130 MiB, 11362 MiB free)\n"
           "  Vulkan1: NVIDIA GeForce RTX 3050 Laptop GPU (3964 MiB, 2635 MiB free)\n")
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 0, stdout=out))
    assert llama_server._gpu_device() == "Vulkan1"
    only_igpu = "Available devices:\n  Vulkan0: Intel(R) UHD Graphics (8000 MiB, 7000 MiB free)\n"
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 0, stdout=only_igpu))
    assert llama_server._gpu_device() is None
