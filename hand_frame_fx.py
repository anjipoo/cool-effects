"""
Hand Frame FX
=============
Hold both hands up (loose L, claw or open palm) like a director framing a
shot. The four fingertips define a free-form quadrilateral and a cinematic
effect appears ONLY inside that shape.

Bring your two hands CLOSE together to jump to the next effect, then spread
them back out to see the new look. Pinch just one hand and the shape becomes
a triangle.

The SPREAD of your index/thumb corners also acts as a live "intensity" dial
for the "gesture control" effect (see below) — spread wide for max blur /
brightness / saturation / pixel-size, pinch tight for the subtle end.

Freeze frame: hold BOTH hands still (barely moving) for about half a second
while framing a shot, and the content inside the frame freezes — a little
portal into the past — while everything outside keeps playing live. Do the
same still-hold again (or press 'f') to release it.

Controls (while the window is focused):
  r       - start / stop recording a video (saved next to this script)
  space   - manually switch to the next effect
  f       - manually toggle freeze frame
  q       - quit
  l       - toggle hand-landmark skeleton overlay (debug)

Run:
    pip install -r requirements.txt
    python3 hand_frame_fx.py
"""

import os
from collections import deque
from datetime import datetime

import cv2
import numpy as np
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# --------------------------------------------------------------------------
# Color-mapping helpers (used to build stylised "duotone" / pop-art looks)
# --------------------------------------------------------------------------

def make_gradient_lut(stops):
    """Build a 256x1x3 BGR lookup table by interpolating between color stops.

    stops: list of (position 0..1, (B, G, R)) tuples, sorted by position.
    """
    stops = sorted(stops, key=lambda s: s[0])
    lut = np.zeros((256, 1, 3), dtype=np.uint8)
    for i in range(256):
        t = i / 255.0
        if t <= stops[0][0]:
            lut[i, 0] = stops[0][1]
            continue
        if t >= stops[-1][0]:
            lut[i, 0] = stops[-1][1]
            continue
        for j in range(len(stops) - 1):
            p0, c0 = stops[j]
            p1, c1 = stops[j + 1]
            if p0 <= t <= p1:
                local_t = 0.0 if p1 == p0 else (t - p0) / (p1 - p0)
                lut[i, 0] = [c0[k] + (c1[k] - c0[k]) * local_t for k in range(3)]
                break
    return lut


# --------------------------------------------------------------------------
# Effects — bold, share-worthy looks. Each takes a BGR patch (+ an optional
# ctx dict for effects that need live gesture data) and returns a same-size
# BGR patch. Kept vectorised / downscaled so they stay cheap on live video.
#
# ctx (when provided) currently carries:
#   'pinch' : float 0..1, live index/thumb-spread of the framing hands.
#             0 = corners pinched tight, 1 = corners spread wide.
# --------------------------------------------------------------------------

EFFECT_WORK_RES = 420   # cap effect processing resolution so big shapes don't tank FPS


def _resize_for_work(patch, max_dim=EFFECT_WORK_RES):
    h, w = patch.shape[:2]
    scale = min(1.0, max_dim / max(h, w))
    if scale >= 1.0:
        return patch, 1.0
    small = cv2.resize(patch, (max(1, int(w * scale)), max(1, int(h * scale))),
                        interpolation=cv2.INTER_AREA)
    return small, scale


def _upscale_back(small, target_shape):
    h, w = target_shape[:2]
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def fx_grid(patch, ctx=None):
    """Greyscale subject under a crisp technical grid (blueprint / reference look)."""
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    gray = cv2.convertScaleAbs(gray, alpha=1.15, beta=8)
    out = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    out = cv2.addWeighted(out, 0.9, np.full_like(out, (60, 45, 30)), 0.1, 0)  # faint cool cast
    h, w = out.shape[:2]
    step = max(14, w // 18)
    minor, major = (150, 140, 130), (230, 225, 215)
    for k, x in enumerate(range(0, w, step)):
        cv2.line(out, (x, 0), (x, h), major if k % 4 == 0 else minor, 1, cv2.LINE_AA)
    for k, y in enumerate(range(0, h, step)):
        cv2.line(out, (0, y), (w, y), major if k % 4 == 0 else minor, 1, cv2.LINE_AA)
    return out


COMIC_LUT = make_gradient_lut([
    (0.00, (20, 0, 10)),      # near-black
    (0.30, (30, 20, 215)),    # bold red (BGR)
    (0.60, (30, 140, 255)),   # orange
    (0.80, (70, 235, 255)),   # yellow
    (1.00, (240, 250, 255)),  # white
])


def fx_comic(patch, ctx=None):
    """Pop-art comic — flat red / orange / yellow posterization."""
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    gray = (gray & 0xC0)                                     # posterize to 4 levels
    return cv2.applyColorMap(gray, COMIC_LUT)


def fx_hero_suit(patch, ctx=None):
    """Dark tactical-suit tones + glowing comic-style edge outlines
    over a sparkly energy-portal background. Generic 'hero' vibe —
    not tied to any specific character design."""
    small, _ = _resize_for_work(patch)
    h, w = small.shape[:2]
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

    # dark suit-toned base (desaturate + darken + slight blue-black tint)
    base = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR).astype(np.float32)
    base *= np.array([0.9, 0.55, 0.5])          # cool dark tint, BGR
    base = np.clip(base * 0.6, 0, 255).astype(np.uint8)

    # comic-style glowing edge lines (cyan + red mix)
    edges = cv2.Canny(gray, 50, 140)
    edges = cv2.dilate(edges, np.ones((2, 2), np.uint8), iterations=1)
    edge_color = np.zeros_like(small)
    edge_color[edges > 0] = (255, 210, 40)       # cyan-ish, BGR
    glow = cv2.GaussianBlur(edge_color, (0, 0), sigmaX=4)
    out_small = cv2.add(base, cv2.add(glow, edge_color))

    # subtle red rim-light on bright areas (like a hero backlight)
    bright_mask = gray > 190
    red_tint = np.array((30, 30, 220), dtype=np.float32)
    blended = out_small[bright_mask].astype(np.float32) * 0.5 + red_tint * 0.5
    out_small[bright_mask] = np.clip(blended, 0, 255).astype(np.uint8)

    # sparkly energy-portal particles scattered in background (darker areas)
    rng = np.random.default_rng(42)
    num_particles = int((w * h) / 900)
    xs = rng.integers(0, w, num_particles)
    ys = rng.integers(0, h, num_particles)
    for x, y in zip(xs, ys):
        if gray[y, x] < 120:   # only sprinkle over darker background areas
            r = rng.integers(1, 3)
            color = (230, 180, 255) if rng.random() > 0.5 else (255, 220, 120)
            cv2.circle(out_small, (int(x), int(y)), int(r), color, -1, cv2.LINE_AA)

    return _upscale_back(out_small, patch.shape)


def fx_paper(patch, ctx=None):
    """Black-and-white ink outline on dotted paper (pencil-sketch / stipple look)."""
    h, w = patch.shape[:2]
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    # ink outline
    g = cv2.medianBlur(gray, 5)
    edges = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                  cv2.THRESH_BINARY, 9, 9)    # 255 paper, 0 ink
    # dotted stipple shading (darker areas -> bigger dots)
    dot = max(4, w // 55)
    cols, rows = max(1, w // dot), max(1, h // dot)
    cg = cv2.resize(gray, (cols, rows), interpolation=cv2.INTER_AREA)
    paper = np.full((h, w), 250, np.uint8)
    off = dot // 2
    for j in range(rows):
        cy = j * dot + off
        for i in range(cols):
            rad = int((1.0 - cg[j, i] / 255.0) * dot * 0.5)
            if rad > 0:
                cv2.circle(paper, (i * dot + off, cy), rad, 70, -1, cv2.LINE_AA)
    sheet = cv2.min(paper, edges)                            # lay ink lines over stipple
    out = cv2.cvtColor(sheet, cv2.COLOR_GRAY2BGR).astype(np.float32)
    out *= np.array([0.93, 0.97, 1.0], np.float32)           # warm cream paper tint
    return out.astype(np.uint8)


def fx_glass(patch, ctx=None):
    """Apple-style glassmorphism: frosted translucent panel with a glossy
    diagonal sheen and fine grain."""
    small, _ = _resize_for_work(patch)
    blurred = cv2.GaussianBlur(small, (25, 25), 0)

    white = np.full_like(blurred, 255)
    glass = cv2.addWeighted(blurred, 0.55, white, 0.45, 0)

    hsv = cv2.cvtColor(glass, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] = np.clip(hsv[..., 1] * 1.15, 0, 255)
    glass = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    h, w = glass.shape[:2]
    xx, yy = np.meshgrid(np.arange(w), np.arange(h))
    diag = (xx + yy) / float(w + h)
    streak = np.exp(-((diag - 0.35) ** 2) / (2 * 0.03 ** 2)) * 60.0
    glass = np.clip(glass.astype(np.float32) + streak[..., None], 0, 255).astype(np.uint8)

    noise = np.random.randint(-6, 6, glass.shape, dtype=np.int16)
    glass = np.clip(glass.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return _upscale_back(glass, patch.shape)


def fx_duotone(patch, ctx=None):
    """Elegant two-tone gradient map (deep navy -> violet -> hot pink)."""
    small, _ = _resize_for_work(patch)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    stops = np.array([
        [70, 20, 10],
        [140, 40, 120],
        [90, 60, 235],
        [225, 215, 255],
    ], dtype=np.float32)

    lut = np.zeros((256, 3), dtype=np.uint8)
    n = len(stops) - 1
    for i in range(256):
        t = i / 255.0
        seg = min(int(t * n), n - 1)
        local_t = (t * n) - seg
        color = stops[seg] * (1 - local_t) + stops[seg + 1] * local_t
        lut[i] = color.astype(np.uint8)

    out_small = lut[gray]
    return _upscale_back(out_small, patch.shape)


def fx_colorpop_red(patch, ctx=None):
    """Skin stays black & white, everything else turns deep red."""
    small, _ = _resize_for_work(patch)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray_3ch = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    lower1 = np.array([0, 20, 70], dtype=np.uint8)
    upper1 = np.array([25, 180, 255], dtype=np.uint8)
    lower2 = np.array([165, 20, 70], dtype=np.uint8)
    upper2 = np.array([180, 180, 255], dtype=np.uint8)
    skin_mask = cv2.bitwise_or(cv2.inRange(hsv, lower1, upper1),
                                cv2.inRange(hsv, lower2, upper2))
    skin_mask = cv2.GaussianBlur(skin_mask, (9, 9), 0)
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    skin_bool = skin_mask > 60

    red_bg = np.full_like(small, (30, 30, 200))
    red_tinted = cv2.addWeighted(small, 0.25, red_bg, 0.75, 0)

    out_small = red_tinted.copy()
    out_small[skin_bool] = gray_3ch[skin_bool]

    return _upscale_back(out_small, patch.shape)


def fx_holographic(patch, ctx=None):
    """Iridescent holographic sheen — hue shifts with surface gradient."""
    small, _ = _resize_for_work(patch)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    angle = (np.arctan2(gy, gx) + np.pi) / (2 * np.pi)

    hue = (angle * 179).astype(np.uint8)
    sat = np.full_like(hue, 200)
    val = np.clip(gray * 1.2, 0, 255).astype(np.uint8)
    hsv = cv2.merge([hue, sat, val])
    holo = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    out_small = cv2.addWeighted(holo, 0.75, small, 0.25, 0)
    return _upscale_back(out_small, patch.shape)


def fx_neon_bloom(patch, ctx=None):
    """Glowing neon edge lines on a near-black background."""
    small, _ = _resize_for_work(patch)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 60, 150)

    edges_color = np.zeros_like(small)
    edges_color[edges > 0] = (255, 80, 180)
    glow = cv2.GaussianBlur(edges_color, (0, 0), sigmaX=6)
    combined = cv2.add(glow, edges_color)

    base = np.full_like(small, (20, 10, 5))
    out_small = cv2.add(base, combined)

    return _upscale_back(out_small, patch.shape)


# --------------------------------------------------------------------------
# NEW: RGB glitch — TikTok-style channel-split / scanline / noise glitch.
# --------------------------------------------------------------------------

def fx_rgb_glitch(patch, ctx=None):
    """Red shifted left, blue shifted right, plus scanlines, noise and
    the odd horizontal 'slice' tear for chromatic-aberration glitch vibes."""
    h, w = patch.shape[:2]
    b, g, r = cv2.split(patch)

    shift = max(2, w // 55)
    r_shift = np.roll(r, -shift, axis=1)   # red content slides left
    b_shift = np.roll(b, shift, axis=1)    # blue content slides right
    out = cv2.merge([b_shift, g, r_shift])

    # scanlines (darken every third row)
    out[::3, :] = (out[::3, :].astype(np.float32) * 0.6).astype(np.uint8)

    # fine grain noise
    noise = np.random.randint(-18, 18, out.shape, dtype=np.int16)
    out = np.clip(out.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # a couple of torn horizontal glitch slices
    rng = np.random.default_rng()
    for _ in range(3):
        y = int(rng.integers(0, h))
        bh = int(rng.integers(2, max(3, h // 40)))
        xoff = int(rng.integers(-w // 25 - 1, w // 25 + 1))
        y0, y1 = max(0, y), min(h, y + bh)
        if y1 > y0:
            out[y0:y1] = np.roll(out[y0:y1], xoff, axis=1)

    return out


# --------------------------------------------------------------------------
# NEW: Magnifying glass — digital 2x / 4x zoom, fisheye bulge, microscope.
# --------------------------------------------------------------------------

def _digital_zoom(patch, factor):
    h, w = patch.shape[:2]
    cw, ch = w / factor, h / factor
    cx, cy = w / 2, h / 2
    x0, y0 = int(cx - cw / 2), int(cy - ch / 2)
    x1, y1 = int(cx + cw / 2), int(cy + ch / 2)
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w, x1), min(h, y1)
    if x1 <= x0 or y1 <= y0:
        return patch
    crop = patch[y0:y1, x0:x1]
    return cv2.resize(crop, (w, h), interpolation=cv2.INTER_CUBIC)


def fx_magnify_2x(patch, ctx=None):
    """Simple digital 2x zoom, centered."""
    return _digital_zoom(patch, 2.0)


def fx_magnify_4x(patch, ctx=None):
    """Simple digital 4x zoom, centered."""
    return _digital_zoom(patch, 4.0)


def _fisheye_bulge(patch, power=0.55):
    """Barrel/bulge remap: source radius = dest radius ^ power (power<1
    magnifies the center, like looking through a convex lens)."""
    h, w = patch.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    dx = (xx - cx) / cx
    dy = (yy - cy) / cy
    r = np.sqrt(dx ** 2 + dy ** 2)
    r_clamped = np.clip(r, 1e-6, 1.4)
    theta = np.arctan2(dy, dx)
    r_src = np.power(r_clamped, power)
    map_x = (cx + r_src * np.cos(theta) * cx).astype(np.float32)
    map_y = (cy + r_src * np.sin(theta) * cy).astype(np.float32)
    return cv2.remap(patch, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                      borderMode=cv2.BORDER_REFLECT)


def fx_magnify_fisheye(patch, ctx=None):
    """Convex-lens bulge — magnifies the middle, curves the edges."""
    return _fisheye_bulge(patch, power=0.55)


def fx_magnify_microscope(patch, ctx=None):
    """Fisheye bulge behind a circular eyepiece mask with crosshairs and a
    cool scientific tint — like peering down a microscope."""
    bulged = _fisheye_bulge(patch, power=0.45)
    h, w = bulged.shape[:2]
    cx, cy = w // 2, h // 2
    r = min(cx, cy)

    tinted = np.clip(bulged.astype(np.float32) * np.array([0.85, 1.05, 0.85]), 0, 255).astype(np.uint8)

    out = np.zeros_like(tinted)
    mask = np.zeros((h, w), np.uint8)
    cv2.circle(mask, (cx, cy), r, 255, -1)
    out[mask > 0] = tinted[mask > 0]

    cv2.line(out, (cx - r, cy), (cx + r, cy), (40, 180, 40), 1, cv2.LINE_AA)
    cv2.line(out, (cx, cy - r), (cx, cy + r), (40, 180, 40), 1, cv2.LINE_AA)
    cv2.circle(out, (cx, cy), r, (30, 140, 30), 2, cv2.LINE_AA)
    cv2.circle(out, (cx, cy), max(2, r // 25), (40, 220, 40), 1, cv2.LINE_AA)
    return out


# --------------------------------------------------------------------------
# NEW: Time warp — needs a rolling frame buffer so it can play the frame
# content back slowed down, sped up, reversed, or lagged behind real time.
# One shared buffer backs all four speeds so history carries over between
# them when you cycle through this group of looks.
# --------------------------------------------------------------------------

class _TimeWarpBuffer:
    def __init__(self, maxlen=90, store_dim=140):
        self.buf = deque(maxlen=maxlen)
        self.store_dim = store_dim
        self.play_pos = {}

    def push(self, patch):
        small = cv2.resize(patch, (self.store_dim, self.store_dim), interpolation=cv2.INTER_AREA)
        self.buf.append(small)

    def sample(self, idx, target_shape):
        if not self.buf:
            return None
        idx = max(0, min(len(self.buf) - 1, idx))
        frame = self.buf[idx]
        return cv2.resize(frame, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_LINEAR)


_timewarp = _TimeWarpBuffer()


def _timewarp_play(mode, patch):
    _timewarp.push(patch)
    n = len(_timewarp.buf)
    if n == 0:
        return patch

    if mode == "delay":
        lag = min(n - 1, 20)                       # fixed ~2/3s lag behind live
        idx = n - 1 - lag
    else:
        pos = _timewarp.play_pos.get(mode, float(n - 1))
        step = {"slow": 0.5, "fast": 2.0, "reverse": -1.0}[mode]
        pos += step
        pos = pos % n
        _timewarp.play_pos[mode] = pos
        idx = int(pos)

    out = _timewarp.sample(idx, patch.shape)
    return out if out is not None else patch


def fx_timewarp_slow(patch, ctx=None):
    """Plays back at half speed from the rolling buffer."""
    return _timewarp_play("slow", patch)


def fx_timewarp_fast(patch, ctx=None):
    """Plays back at double speed from the rolling buffer."""
    return _timewarp_play("fast", patch)


def fx_timewarp_reverse(patch, ctx=None):
    """Plays the rolling buffer backwards."""
    return _timewarp_play("reverse", patch)


def fx_timewarp_delay(patch, ctx=None):
    """Shows what was happening ~2/3 of a second ago."""
    return _timewarp_play("delay", patch)


# --------------------------------------------------------------------------
# NEW: X-ray — edges only, green thermal, or a cool medical-scan look.
# --------------------------------------------------------------------------

def fx_xray_edges(patch, ctx=None):
    """Only the edges survive — stark white lines on black."""
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 40, 120)
    edges = cv2.dilate(edges, np.ones((2, 2), np.uint8), iterations=1)
    return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)


def fx_xray_thermal(patch, ctx=None):
    """Green thermal-camera look — inverted luminance mapped to green."""
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    inv = 255 - gray
    out = np.zeros_like(patch)
    out[..., 1] = inv
    out[..., 0] = (inv.astype(np.float32) * 0.15).astype(np.uint8)
    return out


def fx_xray_medical(patch, ctx=None):
    """Cool blue-white inverted scan with bright bone-like edge highlights."""
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    inv = cv2.equalizeHist(255 - gray)
    out = cv2.cvtColor(inv, cv2.COLOR_GRAY2BGR).astype(np.float32)
    out *= np.array([1.05, 0.95, 0.85])          # cool blue-white cast
    edges = cv2.Canny(gray, 50, 130)
    out[edges > 0] = (255, 255, 255)
    return np.clip(out, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------
# NEW: Gesture-controlled intensity — blur, brightness, saturation and
# pixel size all riding on the live pinch/spread of your framing hands
# (ctx['pinch'], 0 = corners together, 1 = corners spread wide) instead of
# being fixed constants.
# --------------------------------------------------------------------------

def fx_gesture_control(patch, ctx=None):
    """One combined look whose blur / brightness / saturation / pixel-size
    all scale live with how wide you spread your framing hands."""
    pinch = 0.5 if not ctx else float(np.clip(ctx.get("pinch", 0.5), 0.0, 1.0))
    out = patch.copy()

    # blur: 0 .. ~21px kernel
    k = int(1 + pinch * 20)
    if k % 2 == 0:
        k += 1
    if k > 1:
        out = cv2.GaussianBlur(out, (k, k), 0)

    # brightness: -60 .. +60
    beta = int((pinch - 0.5) * 120)
    out = cv2.convertScaleAbs(out, alpha=1.0, beta=beta)

    # saturation: 0.3x .. 2.0x
    hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] = np.clip(hsv[..., 1] * (0.3 + pinch * 1.7), 0, 255)
    out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    # pixel size: 1 .. ~19px blocks
    h, w = out.shape[:2]
    block = max(1, int(1 + pinch * 18))
    if block > 1:
        small = cv2.resize(out, (max(1, w // block), max(1, h // block)),
                            interpolation=cv2.INTER_LINEAR)
        out = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

    # tiny HUD bar so you can see the dial you're turning
    cv2.rectangle(out, (10, h - 22), (130, h - 12), (255, 255, 255), 1)
    cv2.rectangle(out, (12, h - 20), (12 + int(pinch * 116), h - 14), (0, 255, 180), -1)

    return out


EFFECTS = [
    ("comic", fx_comic),
    ("glass", fx_glass),
    ("duotone", fx_duotone),
    ("paper", fx_paper),
    ("colorpop red", fx_colorpop_red),
    ("grid", fx_grid),
    ("holographic", fx_holographic),
    ("hero suit", fx_hero_suit),
    ("neon bloom", fx_neon_bloom),
    ("rgb glitch", fx_rgb_glitch),
    ("magnify 2x", fx_magnify_2x),
    ("magnify 4x", fx_magnify_4x),
    ("magnify fisheye", fx_magnify_fisheye),
    ("microscope", fx_magnify_microscope),
    ("time warp slow", fx_timewarp_slow),
    ("time warp fast", fx_timewarp_fast),
    ("time warp reverse", fx_timewarp_reverse),
    ("time warp delay", fx_timewarp_delay),
    ("x-ray edges", fx_xray_edges),
    ("x-ray thermal", fx_xray_thermal),
    ("x-ray medical", fx_xray_medical),
    ("gesture control", fx_gesture_control),
]

FX_MAX_DIM = 420  # cap effect working resolution -> stable FPS on big quads


def apply_effect(effect, patch, ctx=None):
    """Run an effect at a capped resolution so a huge quad can't tank the FPS."""
    h, w = patch.shape[:2]
    scale = FX_MAX_DIM / max(h, w)
    if scale >= 1.0:
        return effect(patch, ctx)
    small = cv2.resize(patch, (max(1, int(w * scale)), max(1, int(h * scale))))
    return cv2.resize(effect(small, ctx), (w, h), interpolation=cv2.INTER_LINEAR)


# --------------------------------------------------------------------------
# Gesture detection — deliberately permissive: ANY raised hand (loose L, claw
# or fully open palm) is a valid frame anchor. Only a closed fist is ignored.
# --------------------------------------------------------------------------

def _dist(a, b):
    return ((a.x - b.x) ** 2 + (a.y - b.y) ** 2) ** 0.5


def hand_anchor(lm):
    """Return (index_tip, thumb_tip, span) for any detected hand.

    span = index-MCP-to-wrist length (a scale-invariant 'hand size' used to
    normalise the between-hands distance). No pose gate at all: a pinched hand
    (thumb + index together) stays tracked, so pinching ONE hand just collapses
    its two anchors into a point (making a triangle) instead of dropping the
    hand and killing the whole frame.
    """
    span = _dist(lm[5], lm[0]) + 1e-6
    return lm[8], lm[4], span


# --------------------------------------------------------------------------
# Recording
# --------------------------------------------------------------------------

REC_FPS = 30  # playback frame rate; frames are written on a wall-clock schedule
              # so the video always plays back at real-time speed (no 2x effect).


def start_recording(w, h):
    """Open a VideoWriter next to this script. Returns (writer, path)."""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    here = os.path.dirname(os.path.abspath(__file__))
    for ext, codec in ((".mp4", "mp4v"), (".avi", "XVID")):
        path = os.path.join(here, f"hand_frame_fx_{stamp}{ext}")
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*codec), REC_FPS, (w, h))
        if writer.isOpened():
            print(f"Recording -> {path}  (press r to stop)")
            return writer, path
        writer.release()
    print("Could not open a video writer — recording unavailable.")
    return None, None


# --------------------------------------------------------------------------
# Main loop
# --------------------------------------------------------------------------

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise SystemExit("Could not open webcam (index 0). Try a different camera index.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # lowest latency

    hands = mp_hands.Hands(
        max_num_hands=2,
        model_complexity=0,           # fastest model
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,  # keep cheap tracking mode alive
    )

    DETECT_SCALE = 0.5                # run detection on a half-size frame
    SMOOTH = 0.55                     # corner smoothing (1.0 = raw/instant, lower = smoother)
    # Between-hands distance (in hand-widths) that triggers the next filter:
    # bring hands CLOSER than NEAR_ON to switch, spread past NEAR_OFF to re-arm.
    NEAR_ON, NEAR_OFF = 1.5, 2.6
    HOLD_FRAMES = 5                   # keep last shape briefly to ride out dropouts
    CHANGE_COOLDOWN = 5               # min frames between filter changes

    # Freeze-frame gesture: hold both hands basically still (little corner
    # movement) for STILL_HOLD_FRAMES in a row to toggle the freeze. Moving
    # your hands again re-arms the trigger so it doesn't just flip back and
    # forth every frame while you hold still.
    STILL_THRESH = 4.0
    STILL_HOLD_FRAMES = 18
    FREEZE_COOLDOWN = 15

    show_skeleton = False
    effect_idx = 0
    near = False                      # are hands currently held close together?
    cooldown = 0
    writer = None                     # cv2.VideoWriter while recording, else None
    rec_path = None
    rec_start = 0                     # tick when recording began
    frames_written = 0                # frames written so far this recording
    smoothed = None                   # smoothed [li, lt, ri, rt] as float xy
    miss_streak = 0
    fps = 0.0
    prev_tick = cv2.getTickCount()

    pinch_val = 0.5                   # live index/thumb spread, 0..1, for gesture control fx

    frame_frozen = False              # is the inside-the-frame content frozen?
    frozen_snapshot = None            # full raw frame captured at freeze time
    prev_corners_motion = None        # last frame's corners, for stillness check
    still_frames = 0
    still_armed = True
    freeze_cooldown = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        small = cv2.resize(frame, (0, 0), fx=DETECT_SCALE, fy=DETECT_SCALE)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        detected = []  # (center_x, index_xy, thumb_xy, mid_xy, span_px)
        if results.multi_hand_landmarks:
            for hand_lm in results.multi_hand_landmarks:
                anchor = hand_anchor(hand_lm.landmark)
                if anchor is not None:
                    idx_tip, thumb_tip, span = anchor
                    idx_xy = np.array([idx_tip.x * w, idx_tip.y * h], np.float32)
                    thumb_xy = np.array([thumb_tip.x * w, thumb_tip.y * h], np.float32)
                    detected.append((
                        idx_tip.x, idx_xy, thumb_xy,
                        (idx_xy + thumb_xy) * 0.5,   # hand's mid point
                        span * w,                    # hand size in ~pixels
                    ))
                if show_skeleton:
                    mp_drawing.draw_landmarks(frame, hand_lm, mp_hands.HAND_CONNECTIONS)

        output = frame
        corners = None
        if cooldown > 0:
            cooldown -= 1
        if freeze_cooldown > 0:
            freeze_cooldown -= 1

        if len(detected) >= 2:
            detected.sort(key=lambda d: d[0])          # leftmost first
            left, right = detected[0], detected[-1]
            target = [left[1], left[2], right[1], right[2]]  # li, lt, ri, rt
            if smoothed is None:
                smoothed = [p.copy() for p in target]
            else:
                for i in range(4):
                    smoothed[i] += (target[i] - smoothed[i]) * SMOOTH
            corners = smoothed
            miss_streak = 0

            # live pinch/spread reading -> drives the gesture-control fx
            pinch_raw = 0.5 * (
                float(np.linalg.norm(left[1] - left[2])) / (left[4] + 1e-6)
                + float(np.linalg.norm(right[1] - right[2])) / (right[4] + 1e-6)
            )
            pinch_val = float(np.clip(pinch_raw / 1.4, 0.0, 1.0))

            # Filter changes when you bring your two hands close together.
            # Distance is measured in hand-widths so it works at any camera
            # distance. Spread back out to re-arm for the next change.
            inter = float(np.linalg.norm(left[3] - right[3]))
            gap = inter / ((left[4] + right[4]) * 0.5 + 1e-6)
            if gap < NEAR_ON:
                if not near and cooldown == 0:
                    effect_idx = (effect_idx + 1) % len(EFFECTS)
                    cooldown = CHANGE_COOLDOWN
                near = True
            elif gap > NEAR_OFF:
                near = False

            # Freeze-frame: hold both hands still for a moment to toggle.
            if prev_corners_motion is not None:
                movement = float(np.mean([np.linalg.norm(c - p)
                                           for c, p in zip(corners, prev_corners_motion)]))
            else:
                movement = 999.0
            prev_corners_motion = [c.copy() for c in corners]

            if movement < STILL_THRESH:
                still_frames += 1
            else:
                still_frames = 0
                still_armed = True

            if still_armed and still_frames >= STILL_HOLD_FRAMES and freeze_cooldown == 0:
                frame_frozen = not frame_frozen
                frozen_snapshot = frame.copy() if frame_frozen else None
                still_armed = False
                freeze_cooldown = FREEZE_COOLDOWN
        else:
            # Brief detection dropout: hold the last shape so it doesn't flicker.
            miss_streak += 1
            if miss_streak <= HOLD_FRAMES and smoothed is not None:
                corners = smoothed
            else:
                smoothed = None
                near = False
            prev_corners_motion = None
            still_frames = 0

        if corners is not None:
            pts_i = np.round(cv2.convexHull(np.array(corners, np.float32)).reshape(-1, 2)).astype(np.int32)
            x, y, bw, bh = cv2.boundingRect(pts_i)
            x0, y0 = max(0, x), max(0, y)
            x1, y1 = min(w, x + bw), min(h, y + bh)
            if x1 - x0 > 20 and y1 - y0 > 20:
                # When frozen, sample the SAME (x0,y0,x1,y1) window out of the
                # snapshot taken at freeze-time instead of the live frame — the
                # frame keeps tracking your hands live, but looks into the past.
                source_frame = frozen_snapshot if (frame_frozen and frozen_snapshot is not None) else frame
                patch = source_frame[y0:y1, x0:x1]
                ctx = {"pinch": pinch_val}
                processed = apply_effect(EFFECTS[effect_idx][1], patch, ctx)

                mask = np.zeros((y1 - y0, x1 - x0), np.uint8)
                cv2.fillConvexPoly(mask, pts_i - [x0, y0], 255)
                sel = mask.astype(bool)
                roi = output[y0:y1, x0:x1]
                roi[sel] = processed[sel]                # instant hard-edge copy, no border

                if frame_frozen:
                    cv2.polylines(output, [pts_i], True, (60, 170, 255), 2, cv2.LINE_AA)
                    cv2.putText(output, "FROZEN", (x0, max(20, y0 - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (60, 170, 255), 2, cv2.LINE_AA)

        # Record the CLEAN composited frame (before any UI overlays are drawn).
        # Write as many frames as real elapsed time calls for at REC_FPS, so the
        # clip plays back at true speed regardless of the live processing rate.
        if writer is not None:
            elapsed = (cv2.getTickCount() - rec_start) / cv2.getTickFrequency()
            due = min(int(elapsed * REC_FPS), frames_written + 3)  # cap catch-up bursts
            while frames_written < due:
                writer.write(output)
                frames_written += 1

        # FPS (exponential moving average)
        now = cv2.getTickCount()
        inst = cv2.getTickFrequency() / max(1, (now - prev_tick))
        prev_tick = now
        fps = inst if fps == 0 else fps * 0.9 + inst * 0.1
        cv2.putText(output, f"{fps:4.0f} fps", (w - 110, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (120, 255, 120), 2, cv2.LINE_AA)
        if writer is not None:                            # preview-only REC badge
            cv2.circle(output, (26, 26), 9, (0, 0, 255), -1)
            cv2.putText(output, "REC", (42, 33), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 0, 255), 2, cv2.LINE_AA)

        cv2.imshow("Hand Frame FX", output)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('l'):
            show_skeleton = not show_skeleton
        elif key == ord(' '):
            effect_idx = (effect_idx + 1) % len(EFFECTS)
        elif key == ord('f'):
            frame_frozen = not frame_frozen
            frozen_snapshot = frame.copy() if frame_frozen else None
            still_armed = False
            freeze_cooldown = FREEZE_COOLDOWN
        elif key == ord('r'):
            if writer is None:
                writer, rec_path = start_recording(w, h)
                rec_start = cv2.getTickCount()
                frames_written = 0
            else:
                writer.release()
                writer = None
                print(f"Saved recording -> {rec_path}")

    if writer is not None:
        writer.release()
        print(f"Saved recording -> {rec_path}")
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()