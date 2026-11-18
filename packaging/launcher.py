"""Entry point for the packaged StreamClipper.exe.

The app starts helpers as `sys.executable -m streamlink|yt_dlp|...`. In the packaged build sys.executable
is this exe, so `-m <module>` is handled here the same way python.exe would.
"""
import os
import runpy
import sys

if getattr(sys, "frozen", False):
    # deno.exe (needed by yt-dlp for YouTube) ships in the bundle folder
    os.environ["PATH"] = os.pathsep.join([getattr(sys, "_MEIPASS", ""), os.path.dirname(sys.executable),
                                          os.environ.get("PATH", "")])

if len(sys.argv) > 2 and sys.argv[1] == "-m":
    module = sys.argv[2]
    sys.argv = [sys.argv[0], *sys.argv[3:]]
    runpy.run_module(module, run_name="__main__", alter_sys=True)
else:
    from streamclipper.main import cli

    cli()
