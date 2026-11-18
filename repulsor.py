#!/usr/bin/env python3
"""REPULSOR-X (Python edition): open palm toward the webcam -> Iron Man repulsor charges and fires.

OpenCV does the camera + drawing, MediaPipe HandLandmarker does the hand tracking.
Run:  .venv/bin/python repulsor.py        (q / ESC to quit)
"""
from __future__ import annotations

import argparse
import itertools
import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

MODEL = Path(__file__).with_name("hand_landmarker.task")
CYAN = (255, 229, 0)        # BGR
ICE = (255, 250, 160)
WHITE = (255, 255, 255)
ORANGE = (60, 120, 255)

CHAINS = [(0, 1, 2, 3, 4), (0, 5, 6, 7, 8), (0, 9, 10, 11, 12), (0, 13, 14, 15, 16), (0, 17, 18, 19, 20), (5, 9, 13, 17)]
FINGERS = [(8, 6), (12, 10), (16, 14), (20, 18)]   # (tip, pip)


@dataclass
class HandState:
    """Smoothed per-hand state that survives between frames."""
    charge: float = 0.0
    center: np.ndarray = field(default_factory=lambda: np.zeros(2))
    direction: np.ndarray = field(default_factory=lambda: np.array([0.0, -1.0]))
    reach: float = 1.0
    palm: bool = False
    seen: bool = False


@dataclass
class Particle:
    pos: np.ndarray
    vel: np.ndarray
    life: float = 1.0


def unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-6 else v


def assign(centres: list[np.ndarray], states: list[HandState]) -> tuple[int, ...]:
    """Pick which state slot each detected hand belongs to.

    MediaPipe's result order is not stable between frames, so match detections to the
    slots that were tracked last frame by nearest centre (lowest total distance).
    Slots not seen last frame cost a fixed penalty, so tracked hands keep their slot.
    """
    def cost(c, s):
        return float(np.linalg.norm(c - s.center)) if s.seen else 1e6
    return min(itertools.permutations(range(len(states)), len(centres)),
               key=lambda perm: sum(cost(c, states[j]) for c, j in zip(centres, perm)))


def is_open(pts: np.ndarray) -> bool:
    """All four fingertips farther from the wrist than their PIP joints."""
    wrist = pts[0, :2]
    return all(np.linalg.norm(pts[t, :2] - wrist) > 1.08 * np.linalg.norm(pts[p, :2] - wrist) for t, p in FINGERS)


def analyse(pts: np.ndarray, label: str, flip_palm: bool):
    """Return (palm_facing_camera, aim_direction, reach) from 21 landmarks in pixel units."""
    a, b = pts[5] - pts[0], pts[17] - pts[0]                 # wrist->index MCP, wrist->pinky MCP
    cross2d = a[0] * b[1] - a[1] * b[0]
    # Verified on a real webcam: this orientation (not the one I first derived) matches palm vs back.
    palm = (cross2d < 0) if label == "Right" else (cross2d > 0)
    palm ^= flip_palm

    n = np.cross(a, b)
    if n[2] > 0:                                              # make the normal point at the camera
        n = -n
    n = unit(n)
    tilt = float(np.hypot(n[0], n[1]))                        # 0 = straight at camera, 1 = sideways
    up = unit(pts[0, :2] * -1 + pts[[0, 5, 9, 13, 17], :2].mean(axis=0))  # wrist -> palm centre
    w = min(1.0, tilt / 0.3)
    aim = unit(w * (n[:2] / (tilt or 1.0)) + (1 - w) * up)
    reach = 0.7 + 0.3 * min(1.0, max(0.0, (tilt - 0.1) / 0.35))   # keep the beam long even when palm faces the lens
    return palm, aim, reach


class Renderer:
    """Draws onto a sharp layer and a glow layer, then composites them additively."""

    def __init__(self, shape):
        self.h, self.w = shape[:2]
        self.sharp = np.zeros(shape, np.uint8)
        self.glow = np.zeros(shape, np.uint8)

    def reset(self):
        self.sharp[:] = 0
        self.glow[:] = 0

    def line(self, p, q, color, th, glow=True):
        p, q = tuple(map(int, p)), tuple(map(int, q))
        cv2.line(self.sharp, p, q, color, th, cv2.LINE_AA)
        if glow:
            cv2.line(self.glow, p, q, color, th * 3, cv2.LINE_AA)

    def circle(self, c, r, color, th=1, glow=True):
        c = tuple(map(int, c))
        cv2.circle(self.sharp, c, int(r), color, th, cv2.LINE_AA)
        if glow:
            cv2.circle(self.glow, c, int(r), color, th * 3, cv2.LINE_AA)

    def arc(self, c, r, start, end, color, th=2):
        for layer, t in ((self.sharp, th), (self.glow, th * 3)):
            cv2.ellipse(layer, tuple(map(int, c)), (int(r), int(r)), 0, math.degrees(start), math.degrees(end), color, t, cv2.LINE_AA)

    def compose(self, frame, flash=0.0):
        small = cv2.resize(self.glow, (self.w // 4, self.h // 4))
        bloom = cv2.resize(cv2.GaussianBlur(small, (0, 0), 4), (self.w, self.h), interpolation=cv2.INTER_LINEAR)
        out = cv2.addWeighted(frame, 0.85, bloom, 0.9, 0)
        out = cv2.add(out, self.sharp)
        if flash > 0:
            out = cv2.add(out, np.full_like(out, (int(90 * flash), int(60 * flash), int(25 * flash))))
        return out


def draw_hand(r: Renderer, pts, st: HandState, t: float):
    col = CYAN if st.palm else ORANGE
    pix = [tuple(p[:2]) for p in pts]
    for ch in CHAINS:
        for i in range(len(ch) - 1):
            r.line(pix[ch[i]], pix[ch[i + 1]], col, 2)
    for i, p in enumerate(pix):
        r.circle(p, 4 if i in (4, 8, 12, 16, 20) else 3, col, -1)

    # targeting brackets
    x0, y0 = pts[:, :2].min(axis=0) - 24
    x1, y1 = pts[:, :2].max(axis=0) + 24
    q = 22
    for x, y, sx, sy in ((x0, y0, 1, 1), (x1, y0, -1, 1), (x0, y1, 1, -1), (x1, y1, -1, -1)):
        r.line((x + sx * q, y), (x, y), col, 2, glow=False)
        r.line((x, y), (x, y + sy * q), col, 2, glow=False)

    if not st.palm or st.charge < 0.02:      # fist / closed hand: no repulsor disc
        return (x1, y0)

    palm_w = float(np.linalg.norm(pts[5, :2] - pts[17, :2]))
    rad, c, ch = palm_w * 0.42, st.center, st.charge
    r.circle(c, rad, ICE, 2)
    for i in range(36):                                      # rotating tick marks
        a = i / 36 * math.tau + t * 0.6
        inner = rad * (0.9 if i % 3 else 0.84)
        r.line(c + inner * np.array([math.cos(a), math.sin(a)]), c + rad * np.array([math.cos(a), math.sin(a)]), ICE, 1, glow=False)
    for i in range(3):                                       # spinning arc segments
        a = -t * 1.8 + i * math.tau / 3
        r.arc(c, rad * 0.68, a, a + 0.9, ICE, 3)
    tri = [c + rad * 0.42 * np.array([math.cos(t * 0.8 + i * math.tau / 3 - math.pi / 2),
                                       math.sin(t * 0.8 + i * math.tau / 3 - math.pi / 2)]) for i in range(3)]
    for i in range(3):
        r.line(tri[i], tri[(i + 1) % 3], ICE, 2)
    cv2.circle(r.sharp, tuple(map(int, c)), int(rad * (0.18 + 0.2 * ch)), WHITE, -1, cv2.LINE_AA)       # white-hot core
    cv2.circle(r.glow, tuple(map(int, c)), int(rad * (0.5 + 0.7 * ch)), ICE, -1, cv2.LINE_AA)
    return (x1, y0)


def draw_beam(r: Renderer, st: HandState, t: float, parts: list[Particle]):
    length = math.hypot(r.w, r.h) * 1.2 * st.reach
    end = st.center + st.direction * length
    jitter = 1 + 0.15 * math.sin(t * 60)
    size = max(18.0, r.w * 0.03)
    # layered like the HTML version: dim wide halo -> bright cyan body -> crisp white-hot core
    for th, col, glow in ((size * 1.5 * jitter, (150, 80, 0), True), (size * 0.7 * jitter, (255, 215, 90), False),
                          (size * 0.3, (255, 250, 220), False), (size * 0.12, WHITE, False)):
        cv2.line(r.sharp, tuple(map(int, st.center)), tuple(map(int, end)), col, max(1, int(th)), cv2.LINE_AA)
        if glow:
            cv2.line(r.glow, tuple(map(int, st.center)), tuple(map(int, end)), (255, 170, 0), int(th * 1.3), cv2.LINE_AA)
    nrm = np.array([-st.direction[1], st.direction[0]])
    for _ in range(2):                                       # crackling arcs
        pts = [st.center]
        for u in range(40, int(length * 0.6), 40):
            pts.append(st.center + st.direction * u + nrm * (random.random() - 0.5) * size * 1.6)
        for a, b in zip(pts, pts[1:]):
            r.line(a, b, (255, 245, 200), 1)
    for _ in range(4):                                       # sparks
        u = random.random() * length * 0.5
        side = (random.random() - 0.5) * 400
        parts.append(Particle(st.center + st.direction * u, nrm * side + st.direction * 300))


def draw_particles(r: Renderer, parts: list[Particle], dt: float):
    parts[:] = [p for p in parts if (setattr(p, "life", p.life - dt * 1.6) or p.life) > 0][-500:]
    for p in parts:
        p.pos = p.pos + p.vel * dt
        r.circle(p.pos, 1, ICE, -1, glow=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--flip-palm", action="store_true", help="use if palm/back detection is inverted")
    args = ap.parse_args()

    if not MODEL.exists():
        raise SystemExit(f"Missing {MODEL.name}; download it next to repulsor.py")

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not cap.isOpened():
        raise SystemExit("Cannot open camera. Allow camera access for your terminal in System Settings > Privacy & Security > Camera.")

    options = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL)),
        running_mode=vision.RunningMode.VIDEO, num_hands=2,
        min_hand_detection_confidence=0.6, min_tracking_confidence=0.5)

    states = [HandState(), HandState()]
    parts: list[Particle] = []
    flash, last, t0 = 0.0, time.perf_counter(), time.perf_counter()
    last_ts = -1
    renderer = None

    with vision.HandLandmarker.create_from_options(options) as landmarker:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)                       # mirror view
            h, w = frame.shape[:2]
            if renderer is None or (renderer.h, renderer.w) != (h, w):
                renderer = Renderer(frame.shape)
            now = time.perf_counter()
            dt, t, last = min(0.05, now - last), now - t0, now

            ts = max(last_ts + 1, int(t * 1000))             # VIDEO mode rejects repeated timestamps
            last_ts = ts
            res = landmarker.detect_for_video(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)),
                ts)

            renderer.reset()
            seen = [False, False]
            labels = []
            hands = []
            for lms, hd in zip(res.hand_landmarks[:2], res.handedness[:2]):
                pts = np.array([[p.x * w, p.y * h, p.z * w] for p in lms])
                hands.append((pts, hd[0].category_name, pts[[0, 5, 9, 13, 17], :2].mean(axis=0)))
            slots = assign([c for _, _, c in hands], states)
            for i, (pts, label, centre) in zip(slots, hands):
                seen[i] = True
                st = states[i]
                palm, aim, reach = analyse(pts, label, args.flip_palm)
                k = 0.35 if st.seen else 1.0
                st.center = st.center + (centre - st.center) * k
                st.direction = unit(st.direction + (aim - st.direction) * 0.25)
                st.reach, st.palm, st.seen = reach, palm, True
                armed = palm and is_open(pts)
                st.charge = min(1.0, st.charge + dt / 0.6) if armed else max(0.0, st.charge - dt * 3)

                x1, y0 = draw_hand(renderer, pts, st, t)
                labels.append((int(x1) + 8, int(y0) + 14, "REPULSOR %d%%" % round(st.charge * 100) if palm else "REPULSOR OFFLINE", st.palm))
                if st.charge >= 1.0:
                    draw_beam(renderer, st, t, parts)
                    flash = max(flash, 0.35)
                elif st.charge > 0:
                    for _ in range(2):
                        a = random.random() * math.tau
                        p = st.center + 80 * np.array([math.cos(a), math.sin(a)])
                        parts.append(Particle(p, (st.center - p) * 2.5))
            for i, s in enumerate(states):
                if not seen[i]:
                    s.seen = False
                    s.charge = max(0.0, s.charge - dt * 3)

            draw_particles(renderer, parts, dt)
            out = renderer.compose(frame, flash)
            flash = max(0.0, flash - dt * 3)
            for x, y, text, palm in labels:
                cv2.putText(out, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, CYAN if palm else ORANGE, 1, cv2.LINE_AA)

            firing = any(s.charge >= 1 and s.seen for s in states)
            msg = "REPULSOR FIRING" if firing else "show your palm" if not any(seen) else "open palm toward camera"
            cv2.putText(out, "REPULSOR-X  " + msg, (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, CYAN, 1, cv2.LINE_AA)
            cv2.imshow("REPULSOR-X", out)
            if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
