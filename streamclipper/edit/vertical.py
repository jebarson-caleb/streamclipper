"""9:16 layouts: face-tracked crop, facecam-on-top split, or blurred-background fit."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from statistics import median

log = logging.getLogger(__name__)

OUT_W, OUT_H = 1080, 1920
TOP_H = 640            # facecam panel height in split layout
BOT_H = OUT_H - TOP_H


@dataclass
class Face:
    cx: float  # normalized 0..1
    cy: float
    w: float
    h: float


def detect_faces(samples: list[tuple[str, float]]) -> list[list[Face]]:
    """samples: (video path, seconds into that file). Returns every face found, per frame, biggest first."""
    import os

    os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "-8")   # silence decoder warnings when seeking .ts
    os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")
    try:
        import cv2
    except ImportError:
        return []
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    sw = 1280   # a corner webcam face is ~4% of the frame width: too small to find at 640 px
    faces: list[list[Face]] = []
    for path, t in samples:
        cap = cv2.VideoCapture(path)
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            continue
        fh, fw = frame.shape[:2]
        small = cv2.resize(frame, (sw, int(fh * sw / fw)))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        found = cascade.detectMultiScale(gray, 1.1, 6, minSize=(32, 32))
        sh = small.shape[0]
        boxes = sorted(found, key=lambda b: b[2] * b[3], reverse=True) if len(found) else []
        faces.append([Face((x + w / 2) / sw, (y + h / 2) / sh, w / sw, h / sh) for x, y, w, h in boxes])
    return faces


def find_webcam(frames: list[list[Face]], n_samples: int) -> Face | None:
    """A webcam overlay: a small face at the same spot near a frame edge in several sampled frames,
    whatever else (game characters, bystanders) is on screen. Moving faces never line up like that."""
    small = [(i, f) for i, fs in enumerate(frames) for f in fs
             if f.w < 0.12 and (f.cx < 0.3 or f.cx > 0.7 or f.cy < 0.3 or f.cy > 0.7)]
    best: list[Face] = []
    for _, f in small:
        hits = {i: g for i, g in small if abs(g.cx - f.cx) < 0.03 and abs(g.cy - f.cy) < 0.03}
        if len(hits) > len(best):
            best = list(hits.values())
    if len(best) >= max(3, n_samples // 4):
        return _median_face(best)
    return None


def choose_layout(faces: list[Face], n_samples: int, pref: str = "auto",
                  webcam: Face | None = None) -> tuple[str, Face | None]:
    """crop  = talking head (big, steady face) -> face-centred 9:16
    split = small webcam in a corner over gameplay -> facecam top / gameplay bottom
    track = everything else (sports, gameplay, IRL) -> 9:16 window following the motion
            (the renderer falls back to fit_blur if the action is too spread out to fit)"""
    if pref in ("crop", "split") and faces:
        return pref, _median_face(faces)
    if pref in ("fit_blur", "track"):
        return pref, None
    if webcam:
        return "split", webcam
    if len(faces) >= max(2, n_samples // 3):
        f = _median_face(faces)
        corner = (f.cx < 0.33 or f.cx > 0.67) and (f.cy < 0.4 or f.cy > 0.6)
        steady = (len(faces) >= max(3, int(n_samples * 0.6))
                  and max(abs(x.cx - f.cx) for x in faces) < 0.06
                  and max(abs(x.cy - f.cy) for x in faces) < 0.06)
        if f.w < 0.12 and corner and steady:   # a webcam overlay doesn't move; faces in sports footage do
            return "split", f
        if f.w >= 0.15:          # a real talking head, not a bystander in a wide shot
            return "crop", f
    return "track", None


def _median_face(faces: list[Face]) -> Face | None:
    if not faces:
        return None
    return Face(median(f.cx for f in faces), median(f.cy for f in faces),
                median(f.w for f in faces), median(f.h for f in faces))


def _even(x: float) -> int:
    return max(2, int(x) // 2 * 2)


def webcam_box(samples: list[tuple[str, float]], cam: Face) -> tuple[float, float, float, float] | None:
    """The webcam overlay's rectangle (normalized x0, y0, x1, y1) around a found webcam face.
    Overlay borders are straight lines that stay put across frames and run the box's full length;
    edges inside the room or the game either move or are short. None if no clear box."""
    import cv2
    import numpy as np

    W = 1280
    frames = []
    for path, t in samples:
        cap = cv2.VideoCapture(path)
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, f = cap.read()
        cap.release()
        if ok:   # one size: live sources switch resolution mid-stream
            f = cv2.resize(f, (W, int(f.shape[0] * W / f.shape[1])))
            frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32))
    frames = [f for f in frames if f.shape == frames[0].shape] if frames else []
    if len(frames) < 3:
        return None
    H = frames[0].shape[0]
    gx = np.mean([np.abs(cv2.Sobel(f, cv2.CV_32F, 1, 0, ksize=3)) for f in frames], axis=0)
    gy = np.mean([np.abs(cv2.Sobel(f, cv2.CV_32F, 0, 1, ksize=3)) for f in frames], axis=0)
    bx, by = gx > np.percentile(gx, 95), gy > np.percentile(gy, 95)
    bx = bx | np.roll(bx, 1, 1) | np.roll(bx, -1, 1)      # tolerate 1 px of wobble
    by = by | np.roll(by, 1, 0) | np.roll(by, -1, 0)
    cx, cy, fw = int(cam.cx * W), int(cam.cy * H), cam.w * W

    def line(mask, a, b, start, stop, step):
        for i in range(start, stop, step):
            if mask(i, a, b) > 0.8:
                return i
        return None

    # a border spans the whole box; a head or headset that sits still is only about a face wide
    xa, xb = int(max(0, cx - 2 * fw)), int(min(W, cx + 2 * fw))
    r = int(fw / 2)   # scan outward from the face box's own edge: a webcam border can sit right under the chin
    y0 = line(lambda y, a, b: by[y, a:b].mean(), xa, xb, cy - r, -1, -1)
    y1 = line(lambda y, a, b: by[y, a:b].mean(), xa, xb, cy + r, H, 1)
    y0, y1 = (0 if y0 is None else y0), (H if y1 is None else y1)
    if y1 - y0 < 4 * fw * 0.5:
        return None
    x0 = line(lambda x, a, b: bx[a:b, x].mean(), y0 + 3, y1 - 3, cx - r, -1, -1)
    x1 = line(lambda x, a, b: bx[a:b, x].mean(), y0 + 3, y1 - 3, cx + r, W, 1)
    x0, x1 = (0 if x0 is None else x0), (W if x1 is None else x1)
    if (x1 - x0) * (y1 - y0) > 0.25 * W * H:   # not an overlay: the "box" is most of the screen
        return None
    return x0 / W, y0 / H, x1 / W, y1 / H


def vertical_filter(layout: str, face: Face | None, src_w: int, src_h: int,
                    box: tuple[float, float, float, float] | None = None) -> str:
    """Filtergraph from [0:v] to [vbase] at 1080x1920."""
    if layout == "crop" and face:
        cw = _even(src_h * 9 / 16)
        x = int(min(max(face.cx * src_w - cw / 2, 0), src_w - cw))
        return f"[0:v]crop={cw}:{src_h}:{x}:0,scale={OUT_W}:{OUT_H},setsar=1[vbase]"
    if layout == "split" and face:
        # facecam: box ~3.2x the face width around it, aspect OUT_W:TOP_H
        fw = face.w * src_w * 3.2
        fh = fw * TOP_H / OUT_W
        fw, fh = _even(min(fw, src_w)), _even(min(fh, src_h))
        fx = int(min(max(face.cx * src_w - fw / 2, 0), src_w - fw))
        fy = int(min(max(face.cy * src_h - fh / 2, 0), src_h - fh))
        if box:   # stay inside the webcam overlay: never pull in the chat or HUD drawn next to it
            bx0, by0, bx1, by1 = box[0] * src_w, box[1] * src_h, box[2] * src_w, box[3] * src_h
            if fw <= bx1 - bx0:
                fx = int(min(max(fx, bx0), bx1 - fw))
            if fh <= by1 - by0:
                fy = int(min(max(fy, by0), by1 - fh))
        gw = _even(min(src_w, src_h * OUT_W / BOT_H))
        gx = (src_w - gw) // 2
        return (f"[0:v]split[a][b];[a]crop={fw}:{fh}:{fx}:{fy},scale={OUT_W}:{TOP_H},setsar=1[top];"
                f"[b]crop={gw}:{src_h}:{gx}:0,scale={OUT_W}:{BOT_H},setsar=1[bot];[top][bot]vstack[vbase]")
    # blur a quarter-size copy then upscale: same look, ~16x cheaper on CPU
    qw, qh = OUT_W // 4, OUT_H // 4
    return (f"[0:v]split[a][b];[a]scale={qw}:{qh}:force_original_aspect_ratio=increase,"
            f"crop={qw}:{qh},boxblur=8:2,eq=brightness=-0.08,scale={OUT_W}:{OUT_H}[bg];"
            f"[b]scale={OUT_W}:-2[fg];[bg][fg]overlay=(W-w)/2:(H-h)/2,setsar=1[vbase]")


def track_filter(cmd_path_escaped: str, x0: int, crop_w: int, src_h: int) -> str:
    """Moving 9:16 crop driven by a sendcmd script (see edit/track.py)."""
    return (f"[0:v]sendcmd=f='{cmd_path_escaped}',crop@trk=w={crop_w}:h={src_h - src_h % 2}:x={x0}:y=0:exact=1,"
            f"scale={OUT_W}:{OUT_H},setsar=1[vbase]")


def horizontal_filter() -> str:
    return ("[0:v]scale=1920:1080:force_original_aspect_ratio=decrease,"
            "pad=1920:1080:(ow-iw)/2:(oh-ih)/2,setsar=1[vbase]")
