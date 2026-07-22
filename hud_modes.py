"""
hud_modes.py — High-Fidelity HUD Mode Strategy Classes
PERFORMANCE EDITION: Zero Python for-loops in hot paths. All geometry
constructed via NumPy and passed to OpenCV's C++ backend in single calls.

Key techniques:
  - _vec_hex / _vec_ticks build full polyline arrays with np.stack + linspace,
    then call cv2.polylines once
  - EdithMode dots: direct pixel writes via overlay[ys, xs] = col (C extension)
  - NanoMode struts: glitch offset applied to a view of a pre-built array
  - NeonCipherMode scanlines: single-call cv2.polylines on vectorized segment array
  - QuantumMode brackets: all 8 L-arm segments built in one np.stack and passed once
  - All sin/cos trig cached as local scalars at top of each render()
"""
import cv2
import numpy as np
import math
from config import *
from ui_components import *
from mediapipe.tasks.python.vision import FaceLandmarksConnections

# ─────────────────────────────────────────────────────────────────────────────
#  FAST SCALAR HELPERS  (zero-alloc, pure math)
# ─────────────────────────────────────────────────────────────────────────────

_TAU = math.tau   # 2π — cached module-level constant

def _pulse(t, freq=1.0, phase=0.0, lo=0.0, hi=1.0):
    return lo + (hi - lo) * (0.5 + 0.5 * math.sin(t * freq * _TAU + phase))

def _col_lerp(a, b, frac):
    return (
        int(a[0] + (b[0] - a[0]) * frac),
        int(a[1] + (b[1] - a[1]) * frac),
        int(a[2] + (b[2] - a[2]) * frac),
    )

def _arc(overlay, cx, cy, r, a1, a2, col, thick=1):
    cv2.ellipse(overlay, (cx, cy), (max(1, r), max(1, r)), 0, a1, a2, col, thick, cv2.LINE_AA)

def _eye_center(face_landmarks, idx, w, h):
    lm = face_landmarks[idx]
    return int(lm.x * w), int(lm.y * h)

def _glitch_offset(t, freq=7.3):
    return int(math.sin(t * freq) * math.sin(t * 2.1 * freq) * 3)

def _draw_diamond(overlay, cx, cy, size, col, filled=False):
    pts = np.array([(cx, cy-size),(cx+size,cy),(cx,cy+size),(cx-size,cy)], np.int32)
    if filled:
        cv2.fillConvexPoly(overlay, pts, col)
    else:
        cv2.polylines(overlay, [pts], True, col, 1, cv2.LINE_AA)

# ─────────────────────────────────────────────────────────────────────────────
#  VECTORIZED GEOMETRY BUILDERS
#  These return numpy arrays of shape (N, 1, 2) suitable for cv2.polylines,
#  built entirely in NumPy — one Python call, rest in C.
# ─────────────────────────────────────────────────────────────────────────────

# Pre-built angle arrays for reuse (module-level, allocated once)
_HEX_ANGLES   = np.linspace(0, _TAU, 7, endpoint=True)          # 7 pts for closed hex
_TICK12_ANGLES = np.linspace(0, _TAU, 12, endpoint=False)
_TICK16_ANGLES = np.linspace(0, _TAU, 16, endpoint=False)
_TICK18_ANGLES = np.linspace(0, _TAU, 18, endpoint=False)
_TICK24_ANGLES = np.linspace(0, _TAU, 24, endpoint=False)
_TICK36_ANGLES = np.linspace(0, _TAU, 36, endpoint=False)
_LISS_ANGLES   = np.linspace(0, _TAU, 8, endpoint=False)        # 8-pt Lissajous
_REACTOR_ANGLES = np.linspace(0, _TAU, 6, endpoint=False)       # 6 reactor nodes
_QUAD_ANGLES   = np.linspace(0, _TAU, 4, endpoint=False)        # 4 quadrant arms


def _vec_hex(cx, cy, r, angle_offset_rad, out_buf=None):
    """
    Build a (7,1,2) int32 polyline array for a closed hexagon.
    out_buf: optional pre-allocated (7,1,2) int32 array to write into (zero-alloc path).
    Returns the array (which may be out_buf).
    """
    angles = _HEX_ANGLES + angle_offset_rad
    xs = (cx + r * np.cos(angles)).astype(np.int32)
    ys = (cy + r * np.sin(angles)).astype(np.int32)
    arr = np.stack((xs, ys), axis=1).reshape(7, 1, 2)
    if out_buf is not None:
        np.copyto(out_buf, arr)
        return out_buf
    return arr


def _vec_ticks(cx, cy, r, r2, angle_base_rad, angles_template):
    """
    Build a (N,2,2) line segment array for a tick ring using a pre-built angles array.
    Each tick is: inner point (r) → outer point (r2).
    Returns shape (N, 1, 2) segments as TWO separate arrays for p1/p2 pairing,
    then stacked into (N, 2, 2) polylines array (each polyline = 2-point open line).
    """
    angles = angles_template + angle_base_rad
    cos_a = np.cos(angles)
    sin_a = np.sin(angles)
    p1x = (cx + r  * cos_a).astype(np.int32)
    p1y = (cy + r  * sin_a).astype(np.int32)
    p2x = (cx + r2 * cos_a).astype(np.int32)
    p2y = (cy + r2 * sin_a).astype(np.int32)
    # Shape: (N, 2, 2) — N lines, each with 2 points, each point (x, y)
    segs = np.stack(
        [np.stack((p1x, p1y), axis=1),
         np.stack((p2x, p2y), axis=1)],
        axis=1
    )
    return segs


def _vec_reactor_dots(nx, ny, reactor_r, base_angle_rad):
    """6 reactor node positions as (6,2) int32 array."""
    angles = _REACTOR_ANGLES + base_angle_rad
    xs = (nx + reactor_r * np.cos(angles)).astype(np.int32)
    ys = (ny + reactor_r * np.sin(angles)).astype(np.int32)
    return xs, ys


def _vec_lissajous(ex, ey, fx, fy, t, r_x=14, r_y=10):
    """8 Lissajous sample positions as (8,) int32 arrays for xs, ys."""
    angles = _LISS_ANGLES + t * 1.2
    xs = (ex + r_x * np.cos(fx * angles)).astype(np.int32)
    ys = (ey + r_y * np.sin(fy * angles)).astype(np.int32)
    return xs, ys


def _vec_strand_segs(cx_f, cy_f, pts_sparse):
    """
    Build (N, 2, 2) segment array for neural strands from centroid to each sparse pt.
    pts_sparse: (N, 2) int32 array of endpoint coordinates.
    """
    N = len(pts_sparse)
    origin = np.array([[cx_f, cy_f]], dtype=np.int32)
    p1 = np.broadcast_to(origin, (N, 2)).copy()   # one alloc per frame, unavoidable
    segs = np.stack([p1, pts_sparse], axis=1)       # (N, 2, 2)
    return segs


def _vec_tentacle_segs(cx_f, cy_f, arm_r, base_angle_rad):
    """
    Build two sets of (4,2,2) segment arrays for 4-arm plasma tentacles.
    Returns segs_inner (4,2,2) and segs_outer (4,2,2) with tip coords (4,2).
    """
    angles = _QUAD_ANGLES + base_angle_rad
    a_mid = angles + 0.5
    a_tip = angles + 1.0
    # Midpoints
    mx = (cx_f + arm_r * 0.45 * np.cos(a_mid)).astype(np.int32)
    my = (cy_f + arm_r * 0.45 * np.sin(a_mid)).astype(np.int32)
    # Tips
    tx = (cx_f + arm_r * np.cos(a_tip)).astype(np.int32)
    ty = (cy_f + arm_r * np.sin(a_tip)).astype(np.int32)
    origin = np.array([[cx_f, cy_f]], dtype=np.int32)
    p0s = np.broadcast_to(origin, (4, 2)).copy()
    mids = np.stack((mx, my), axis=1)
    tips = np.stack((tx, ty), axis=1)
    segs_inner = np.stack([p0s, mids], axis=1)   # (4,2,2)
    segs_outer = np.stack([mids, tips], axis=1)   # (4,2,2)
    return segs_inner, segs_outer, tips


def _vec_scanlines(xs_min, xs_max, ys_min, ys_max, step):
    """
    Build (N, 2, 2) segment array for horizontal scanlines across a face bounding rect.
    """
    ys = np.arange(ys_min, ys_max, step, dtype=np.int32)
    N = len(ys)
    if N == 0:
        return None
    x0 = np.full(N, xs_min, dtype=np.int32)
    x1 = np.full(N, xs_max, dtype=np.int32)
    p1 = np.stack((x0, ys), axis=1)
    p2 = np.stack((x1, ys), axis=1)
    return np.stack([p1, p2], axis=1)   # (N, 2, 2)


def _vec_quad_brackets(cx_f, cy_f, span, base_angle_rad, bracket_l=10):
    """
    Build (8, 2, 2) segment array for 4 rotating L-brackets (2 segments each).
    cos_a/sin_a computed once for all 4 arms.
    """
    angles = _QUAD_ANGLES + base_angle_rad
    cos_a = np.cos(angles)
    sin_a = np.sin(angles)
    bx = (cx_f + span * cos_a).astype(np.int32)
    by = (cy_f + span * sin_a).astype(np.int32)
    dx = np.where(cos_a >= 0, bracket_l, -bracket_l).astype(np.int32)
    dy = np.where(sin_a >= 0, bracket_l, -bracket_l).astype(np.int32)
    # Horizontal arm: (bx,by) -> (bx+dx, by)
    h_p1 = np.stack((bx, by), axis=1)
    h_p2 = np.stack((bx + dx, by), axis=1)
    segs_h = np.stack([h_p1, h_p2], axis=1)
    # Vertical arm: (bx,by) -> (bx, by+dy)
    v_p2 = np.stack((bx, by + dy), axis=1)
    segs_v = np.stack([h_p1, v_p2], axis=1)
    all_segs = np.concatenate([segs_h, segs_v], axis=0)   # (8, 2, 2)
    return all_segs, bx, by


def _vec_cardinal_diamonds(cx_f, cy_f, span, base_angle_rad, d_size=4):
    """
    Build 4 diamond polylines for cardinal measurement brackets.
    Returns list of 4 (4,1,2) arrays suitable for cv2.polylines.
    """
    angles = _QUAD_ANGLES + base_angle_rad
    ax = (cx_f + span * np.cos(angles)).astype(np.int32)
    ay = (cy_f + span * np.sin(angles)).astype(np.int32)
    polys = []
    for i in range(4):
        cx_, cy_ = int(ax[i]), int(ay[i])
        pts = np.array([(cx_, cy_-d_size),(cx_+d_size,cy_),(cx_,cy_+d_size),(cx_-d_size,cy_)], np.int32)
        polys.append(pts.reshape(4, 1, 2))
    return polys


def _vec_eye_cross_segs(ex, ey, cross_len, gap):
    """
    Build (4,2,2) segment array for crosshair lines (left, right, up, down).
    """
    segs = np.array([
        [[ex - cross_len, ey], [ex - gap, ey]],
        [[ex + gap, ey],       [ex + cross_len, ey]],
        [[ex, ey - cross_len], [ex, ey - gap]],
        [[ex, ey + gap],       [ex, ey + cross_len]],
    ], dtype=np.int32)
    return segs


def _vec_corner_ticks(ex, ey, sq):
    """
    Build (8,2,2) segment array for corner tick marks on the eye targeting square.
    4 corners × 2 lines (horizontal + vertical) = 8 segments.
    """
    TK = 7
    corners = [(1,1),(-1,1),(1,-1),(-1,-1)]
    segs = []
    for csx, csy in corners:
        cx_ = ex + csx * sq
        cy_ = ey + csy * sq
        segs.append([[cx_, cy_], [cx_ + csx * TK, cy_]])
        segs.append([[cx_, cy_], [cx_, cy_ + csy * TK]])
    return np.array(segs, dtype=np.int32)   # (8,2,2)


# ─────────────────────────────────────────────────────────────────────────────
#  BASE
# ─────────────────────────────────────────────────────────────────────────────
class HUDModeStrategy:
    def render(self, overlay, face_landmarks, w, h, t, pts, nose, lines, dist, state):
        return (0,255,255), "MODE"


# ─────────────────────────────────────────────────────────────────────────────
#  MODE 0 — IRON MAN MARK-L NANO LATTICE
# ─────────────────────────────────────────────────────────────────────────────
class NanoMode(HUDModeStrategy):
    _S2 = slice(None, None, 2)
    _S3 = slice(None, None, 3)
    # Pre-allocated (N,1,2) hex buffers — filled each frame, no alloc
    _hex_buf_a = np.zeros((7, 1, 2), dtype=np.int32)
    _hex_buf_b = np.zeros((7, 1, 2), dtype=np.int32)

    def render(self, overlay, face_landmarks, w, h, t, pts, nose, lines, dist, state):
        from renderer import _face_tess_edges

        # ── Cache trig scalars ────────────────────────────────────────────────
        sin_slow  = math.sin(t * 1.2 * _TAU)          # p1 base
        sin_fast  = math.sin(t * 3.7 * _TAU + 1.0)    # p2 base
        sin_hull  = math.sin(t * 0.8)
        sin_react = math.sin(t * 6)

        p1 = 0.4 + 0.6 * (0.5 + 0.5 * sin_slow)
        p2 = 0.3 + 0.7 * (0.5 + 0.5 * sin_fast)

        BASE  = (int(20*p1), int(80+20*p1), int(180+40*p1))
        SCAN  = (0, int(160+60*p2), int(240+15*p2))
        HOT   = (30, 200, 255)
        DIM   = (8, 30, 70)
        STRUT = (12, 48, 110)

        # ── Layer 1: base mesh ────────────────────────────────────────────────
        cv2.polylines(overlay, lines[self._S2], False, DIM, 1, cv2.LINE_AA)

        # ── Layer 2: outer hull ───────────────────────────────────────────────
        scale_out = 1.04 + 0.025 * sin_hull
        outer = (nose + (pts - nose) * scale_out).astype(np.int32)
        o_lines = outer[_face_tess_edges]
        cv2.polylines(overlay, o_lines[self._S3], False, (6, 22, 60), 1, cv2.LINE_AA)

        # ── Layer 3: struts with glitch — NO .copy(), build fresh segment array
        # We construct the segment array directly from two index-sliced views.
        # pts[::6] and outer[::6] share underlying memory of pts/outer (no alloc).
        # np.stack allocates once per frame here — unavoidable for the (N,2,2) shape.
        g = _glitch_offset(t, 5.5)
        pts_sparse = pts[::6].copy()          # (N,2) — one small alloc (N~80)
        out_sparse = outer[::6]               # view, no alloc
        pts_sparse[:, 0] += g                 # apply glitch in-place
        struts = np.stack((pts_sparse, out_sparse), axis=1)
        cv2.polylines(overlay, struts, False, STRUT, 1, cv2.LINE_AA)

        # ── Layer 4: multi-tier scan wave ─────────────────────────────────────
        sweep_y = int((t * 340) % h)
        d = np.abs(lines[:, :, 1] - sweep_y).min(axis=1)
        w30 = d < 30; w8 = d < 8; w2 = d < 2
        if w30.any(): cv2.polylines(overlay, lines[w30], False, BASE, 1, cv2.LINE_AA)
        if w8.any():  cv2.polylines(overlay, lines[w8],  False, SCAN, 1, cv2.LINE_AA)
        if w2.any():  cv2.polylines(overlay, lines[w2],  False, HOT,  2, cv2.LINE_AA)

        d2 = np.abs(lines[:, :, 1] - (sweep_y - 20)).min(axis=1)
        if (d2 < 10).any():
            cv2.polylines(overlay, lines[d2 < 10], False, DIM, 1, cv2.LINE_AA)

        # ── Feature contours ─────────────────────────────────────────────────
        draw_landmarks_manually(overlay, face_landmarks, FaceLandmarksConnections.FACE_LANDMARKS_LEFT_EYE,     SCAN, 2)
        draw_landmarks_manually(overlay, face_landmarks, FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_EYE,    SCAN, 2)
        draw_landmarks_manually(overlay, face_landmarks, FaceLandmarksConnections.FACE_LANDMARKS_LEFT_EYEBROW, BASE, 1)
        draw_landmarks_manually(overlay, face_landmarks, FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_EYEBROW,BASE, 1)
        draw_landmarks_manually(overlay, face_landmarks, FaceLandmarksConnections.FACE_LANDMARKS_LIPS,         (20, 120, 220), 1)

        # ── Eye targeting arcs ────────────────────────────────────────────────
        sin_eye = math.sin(t * 4.0)
        r_outer = int(18 + 3 * sin_eye)
        r_inner = int(10 - 2 * sin_eye)          # +math.pi inverts, cached
        ang_outer = int(t * 60) % 360
        ang_inner = -int(t * 45) % 360
        for eye_idx in (468, 473):
            ex, ey = _eye_center(face_landmarks, eye_idx, w, h)
            _arc(overlay, ex, ey, r_outer, ang_outer, (ang_outer + 220) % 360, SCAN, 1)
            _arc(overlay, ex, ey, r_inner, ang_inner, (ang_inner + 200) % 360, BASE, 1)
            cv2.circle(overlay, (ex, ey), 3, HOT, -1, cv2.LINE_AA)

        # ── Nose-tip arc reactor — vectorized 6-node draw ─────────────────────
        nx, ny = int(face_landmarks[1].x * w), int(face_landmarks[1].y * h)
        reactor_r = int(12 + 3 * sin_react)
        rbase = t * math.pi / 2       # t*90° in radians
        rxs, rys = _vec_reactor_dots(nx, ny, reactor_r, rbase)
        for i in range(6):
            cv2.circle(overlay, (int(rxs[i]), int(rys[i])), 2, SCAN, -1, cv2.LINE_AA)
        cv2.circle(overlay, (nx, ny), max(1, int(reactor_r * 0.45)), BASE, 1, cv2.LINE_AA)
        cv2.circle(overlay, (nx, ny), 3, HOT, -1, cv2.LINE_AA)

        # ── Hex rings — vectorized polylines, pre-allocated buffers ───────────
        cx_f = int(pts[:, 0].mean()); cy_f = int(pts[:, 1].mean())
        span = int((pts[:, 0].max() - pts[:, 0].min()) * 0.58)
        hex_a = _vec_hex(cx_f, cy_f, span,      math.radians(t * 18),  self._hex_buf_a)
        hex_b = _vec_hex(cx_f, cy_f, span + 10, math.radians(-t * 10), self._hex_buf_b)
        cv2.polylines(overlay, hex_a, True,  (BASE[0]//3, BASE[1]//3, BASE[2]//3), 1, cv2.LINE_AA)
        cv2.polylines(overlay, hex_b, True, DIM, 1, cv2.LINE_AA)

        # Tick ring — vectorized
        ticks = _vec_ticks(cx_f, cy_f, span, span + 5, math.radians(t * 20), _TICK12_ANGLES)
        cv2.polylines(overlay, ticks, False, STRUT, 1, cv2.LINE_AA)

        return SCAN, "MARK-L // NANO"


# ─────────────────────────────────────────────────────────────────────────────
#  MODE 1 — P.R.I.S.M QUANTUM BLUEPRINT
# ─────────────────────────────────────────────────────────────────────────────
class QuantumMode(HUDModeStrategy):
    _hex_buf = np.zeros((7, 1, 2), dtype=np.int32)

    def render(self, overlay, face_landmarks, w, h, t, pts, nose, lines, dist, state):
        from renderer import _face_tess_edges

        # ── Cached trig ───────────────────────────────────────────────────────
        sin_phase = math.sin(t * 0.9 * _TAU)
        sin_hull  = math.sin(t * 0.6)
        sin_ring  = math.sin(t * 2.5)

        BASE  = (55, 38, 9)
        AMB   = (255, 200, 80)
        HOT   = (255, 245, 165)
        DIM_O = (18, 10, 2)
        NODE  = (200, 140, 40)

        phase = 0.5 + 0.5 * sin_phase
        AMB_LIVE = _col_lerp(AMB, HOT, phase * 0.4)

        # ── Layer 1: base skeleton ────────────────────────────────────────────
        cv2.polylines(overlay, lines, False, BASE, 1, cv2.LINE_AA)

        # ── Layer 2: outer hull ───────────────────────────────────────────────
        scale_out = 1.07 + 0.018 * sin_hull
        outer = (nose + (pts - nose) * scale_out).astype(np.int32)
        o_lines = outer[_face_tess_edges]
        cv2.polylines(overlay, o_lines[::2], False, DIM_O, 1, cv2.LINE_AA)

        # ── Layer 3: struts ───────────────────────────────────────────────────
        struts = np.stack((pts[::4], outer[::4]), axis=1)
        cv2.polylines(overlay, struts, False, (38, 20, 4), 1, cv2.LINE_AA)

        # ── Layer 4: amber sweep ──────────────────────────────────────────────
        sweep_y = int((t * 280) % (h + 1))
        d = np.abs(lines[:, :, 1] - sweep_y).min(axis=1)
        m32 = d < 32; m7 = d < 7; m2 = d < 2
        if m32.any(): cv2.polylines(overlay, lines[m32], False, AMB_LIVE, 1, cv2.LINE_AA)
        if m7.any():  cv2.polylines(overlay, lines[m7],  False, HOT,      1, cv2.LINE_AA)
        if m2.any():  cv2.polylines(overlay, lines[m2],  False, (255,255,255), 2, cv2.LINE_AA)

        # ── Layer 5: data nodes via direct pixel write ────────────────────────
        # mids is already int32 from arithmetic on lines; shape (N,2)
        mids = (lines[:, 0] + lines[:, 1]) >> 1     # bitshift = //2, no alloc
        node_pts = mids[::5]
        # Clamp to valid pixel coords to avoid out-of-bounds segfault
        nx_arr = np.clip(node_pts[:, 0], 0, overlay.shape[1] - 1)
        ny_arr = np.clip(node_pts[:, 1], 0, overlay.shape[0] - 1)
        overlay[ny_arr, nx_arr] = NODE    # single vectorized assignment, C level

        # Diamond nodes near scan wave — still a small loop (typically <15 pts)
        sc_nodes = mids[m32][::4]
        for mx, my in sc_nodes:
            _draw_diamond(overlay, int(mx), int(my), 2, AMB_LIVE)

        # ── Feature contours ─────────────────────────────────────────────────
        draw_landmarks_manually(overlay, face_landmarks, FaceLandmarksConnections.FACE_LANDMARKS_LEFT_EYE,     AMB, 2)
        draw_landmarks_manually(overlay, face_landmarks, FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_EYE,    AMB, 2)
        draw_landmarks_manually(overlay, face_landmarks, FaceLandmarksConnections.FACE_LANDMARKS_LEFT_EYEBROW, HOT, 1)
        draw_landmarks_manually(overlay, face_landmarks, FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_EYEBROW,HOT, 1)
        draw_landmarks_manually(overlay, face_landmarks, FaceLandmarksConnections.FACE_LANDMARKS_LIPS,         AMB, 1)

        # ── Lissajous eye dots — vectorized pixel writes ───────────────────────
        r_ring = int(16 + 3 * sin_ring)
        AMB_HALF = (AMB[0]//2, AMB[1]//2, AMB[2]//2)
        for eye_idx, fx, fy in ((468, 3.0, 2.0), (473, 2.0, 3.0)):
            ex, ey = _eye_center(face_landmarks, eye_idx, w, h)
            lxs, lys = _vec_lissajous(ex, ey, fx, fy, t)
            lxs_c = np.clip(lxs, 0, overlay.shape[1] - 1)
            lys_c = np.clip(lys, 0, overlay.shape[0] - 1)
            overlay[lys_c, lxs_c] = NODE    # 8 pixel writes, zero Python loop
            _arc(overlay, ex, ey, r_ring, 0, 360, AMB_HALF, 1)
            cv2.circle(overlay, (ex, ey), 3, HOT, -1, cv2.LINE_AA)

        # ── Rotating angular brackets — one cv2.polylines call ────────────────
        cx_f = int(pts[:, 0].mean()); cy_f = int(pts[:, 1].mean())
        span = int((pts[:, 0].max() - pts[:, 0].min()) * 0.62)
        segs, bxs, bys = _vec_quad_brackets(cx_f, cy_f, span, math.radians(t * 25))
        cv2.polylines(overlay, segs, False, AMB_LIVE, 1, cv2.LINE_AA)
        # Dot at each bracket anchor — still 4 cv2.circle calls (only 4)
        for i in range(4):
            cv2.circle(overlay, (int(bxs[i]), int(bys[i])), 2, HOT, -1, cv2.LINE_AA)

        return AMB, "P.R.I.S.M // QUANTUM"


# ─────────────────────────────────────────────────────────────────────────────
#  MODE 2 — NEON CIPHER / 2100 HACKER GRID
# ─────────────────────────────────────────────────────────────────────────────
class NeonCipherMode(HUDModeStrategy):
    _hex_buf_a = np.zeros((7, 1, 2), dtype=np.int32)
    _hex_buf_b = np.zeros((7, 1, 2), dtype=np.int32)

    def render(self, overlay, face_landmarks, w, h, t, pts, nose, lines, dist, state):
        BASE  = (60, 140, 20)
        GRID  = (120, 220, 40)
        SCAN  = (200, 255, 60)
        HOT   = (255, 255, 255)
        AMBER = (0, 240, 255)

        # ── Cached trig ───────────────────────────────────────────────────────
        sin_sq = math.sin(t * 5)

        # ── Phase-offset mesh ─────────────────────────────────────────────────
        keep_a = (np.arange(len(lines)) % 3) != 0
        keep_b = (np.arange(len(lines)) % 5) == 0
        cv2.polylines(overlay, lines[keep_a], False, BASE,       1, cv2.LINE_AA)
        cv2.polylines(overlay, lines[keep_b], False, (40,90,15), 1, cv2.LINE_AA)

        # Primary fast sweep
        sweep1 = int((t * 520) % (h + 1))
        d1 = np.abs(lines[:, :, 1] - sweep1).min(axis=1)
        if (d1 < 16).any(): cv2.polylines(overlay, lines[d1 < 16], False, GRID, 1, cv2.LINE_AA)
        if (d1 < 4).any():  cv2.polylines(overlay, lines[d1 < 4],  False, SCAN, 1, cv2.LINE_AA)
        if (d1 < 1).any():  cv2.polylines(overlay, lines[d1 < 1],  False, HOT,  2, cv2.LINE_AA)

        # Secondary counter-sweep
        sweep2 = int(h - (t * 180) % (h + 1))
        d2 = np.abs(lines[:, :, 1] - sweep2).min(axis=1)
        if (d2 < 8).any(): cv2.polylines(overlay, lines[d2 < 8], False, (80,200,30), 1, cv2.LINE_AA)

        # Glitch bands — replace .copy() with an additive offset applied to the
        # X column of the masked subset directly; we build a NEW array (N,2,2)
        # from scratch using vectorized math so no .copy() is needed mid-loop.
        g = _glitch_offset(t, 9.1)
        band_mask = (lines[:, :, 1] % 18 == 0).any(axis=1)
        if band_mask.any():
            gl = lines[band_mask]          # view (no alloc) — read-only below
            # Build glitched version: only X column shifted, Y unchanged
            gl_x = gl[:, :, 0] + g        # (N,2) result of scalar add — small alloc
            gl_segs = np.stack(
                [np.stack((gl_x[:, 0], gl[:, 0, 1]), axis=1),
                 np.stack((gl_x[:, 1], gl[:, 1, 1]), axis=1)],
                axis=1
            )                               # (N,2,2)
            cv2.polylines(overlay, gl_segs, False, (40,120,15), 1, cv2.LINE_AA)

        # ── Feature contours ─────────────────────────────────────────────────
        draw_landmarks_manually(overlay, face_landmarks, FaceLandmarksConnections.FACE_LANDMARKS_FACE_OVAL,   GRID, 2)
        draw_landmarks_manually(overlay, face_landmarks, FaceLandmarksConnections.FACE_LANDMARKS_LEFT_EYE,    SCAN, 2)
        draw_landmarks_manually(overlay, face_landmarks, FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_EYE,   SCAN, 2)
        draw_landmarks_manually(overlay, face_landmarks, FaceLandmarksConnections.FACE_LANDMARKS_LIPS,        GRID, 1)

        # ── Eye targeting — all loops replaced with batch segment arrays ───────
        sq = int(16 + 4 * sin_sq)
        off = int(sq * 0.6)
        off4 = int(off * 0.4)
        # Inner cross: 4 segments, built as (4,2,2)
        inner_cross = np.array([
            [[off, 0], [off4, 0]],
            [[-off, 0], [-off4, 0]],
            [[0, off], [0, off4]],
            [[0, -off], [0, -off4]],
        ], dtype=np.int32)

        for eye_idx in (33, 263):
            ex, ey = _eye_center(face_landmarks, eye_idx, w, h)
            cv2.rectangle(overlay, (ex-sq, ey-sq), (ex+sq, ey+sq), SCAN, 1, cv2.LINE_AA)
            # Shift inner cross to eye coords
            shifted = inner_cross.copy()
            shifted[:, :, 0] += ex
            shifted[:, :, 1] += ey
            cv2.polylines(overlay, shifted, False, HOT, 1, cv2.LINE_AA)
            cv2.circle(overlay, (ex, ey), 4, HOT, -1, cv2.LINE_AA)
            # Corner ticks — vectorized
            corner_segs = _vec_corner_ticks(ex, ey, sq)
            cv2.polylines(overlay, corner_segs, False, AMBER, 1, cv2.LINE_AA)

        # ── Hex rings — vectorized ────────────────────────────────────────────
        cx_f = int(pts[:, 0].mean()); cy_f = int(pts[:, 1].mean())
        xs_arr = pts[:, 0]; ys_arr = pts[:, 1]
        span = int((xs_arr.max() - xs_arr.min()) * 0.55)
        hex_a = _vec_hex(cx_f, cy_f, span,      math.radians(t * -12), self._hex_buf_a)
        hex_b = _vec_hex(cx_f, cy_f, span + 14, math.radians(t * 8),   self._hex_buf_b)
        cv2.polylines(overlay, hex_a, True, BASE,       1, cv2.LINE_AA)
        cv2.polylines(overlay, hex_b, True, (40,90,15), 1, cv2.LINE_AA)
        ticks = _vec_ticks(cx_f, cy_f, span + 22, span + 26, math.radians(t * 15), _TICK24_ANGLES)
        cv2.polylines(overlay, ticks, False, BASE, 1, cv2.LINE_AA)

        # ── Horizontal scanlines — single cv2.polylines call ──────────────────
        sl_segs = _vec_scanlines(int(xs_arr.min()), int(xs_arr.max()),
                                  int(ys_arr.min()), int(ys_arr.max()), 8)
        if sl_segs is not None:
            cv2.polylines(overlay, sl_segs, False, (20,55,8), 1)

        return SCAN, "CIPHER // DECODE"


# ─────────────────────────────────────────────────────────────────────────────
#  MODE 3 — E.D.I.T.H. SURGICAL PRECISION
# ─────────────────────────────────────────────────────────────────────────────
class EdithMode(HUDModeStrategy):
    def render(self, overlay, face_landmarks, w, h, t, pts, nose, lines, dist, state):
        SILVER  = (170, 175, 185)
        WHITE   = (240, 245, 255)
        ICE     = (210, 230, 255)
        DIM     = (90, 95, 100)
        ACCENT  = (160, 200, 240)

        # ── Cached trig ───────────────────────────────────────────────────────
        sin_cross = math.sin(t * 2.5)
        sin_18    = math.sin(t * 1.8)
        sin_18pi  = math.sin(t * 1.8 + math.pi)   # =  -sin_18

        # ── Ultra-sparse mesh ─────────────────────────────────────────────────
        cv2.polylines(overlay, lines[::3], False, DIM, 1, cv2.LINE_AA)

        # ── Feature anatomy ───────────────────────────────────────────────────
        draw_landmarks_manually(overlay, face_landmarks, FaceLandmarksConnections.FACE_LANDMARKS_FACE_OVAL,    WHITE,  2)
        draw_landmarks_manually(overlay, face_landmarks, FaceLandmarksConnections.FACE_LANDMARKS_LEFT_EYE,     ICE,    2)
        draw_landmarks_manually(overlay, face_landmarks, FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_EYE,    ICE,    2)
        draw_landmarks_manually(overlay, face_landmarks, FaceLandmarksConnections.FACE_LANDMARKS_LIPS,         SILVER, 1)
        draw_landmarks_manually(overlay, face_landmarks, FaceLandmarksConnections.FACE_LANDMARKS_LEFT_EYEBROW,  WHITE,  1)
        draw_landmarks_manually(overlay, face_landmarks, FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_EYEBROW, WHITE,  1)

        # ── Landmark measurement dots — direct pixel write (zero Python loop) ──
        dot_pts = pts[::4]     # view into pts, shape (N~120, 2)
        dx = np.clip(dot_pts[:, 0], 0, overlay.shape[1] - 1)
        dy = np.clip(dot_pts[:, 1], 0, overlay.shape[0] - 1)
        overlay[dy, dx] = SILVER     # single C-level vectorized write

        # ── Precision eye assemblies ──────────────────────────────────────────
        cross_len = int(20 + 4 * sin_cross)
        r_a = int(13 + 2 * sin_18)
        r_b = int(22 + 2 * sin_18pi)
        ang_a1 = int(t * 40) % 360
        ang_b1 = -int(t * 30) % 360

        for eye_idx in (468, 473):
            ex, ey = _eye_center(face_landmarks, eye_idx, w, h)
            # Crosshair — 4 lines batched in (4,2,2) array
            cross_segs = _vec_eye_cross_segs(ex, ey, cross_len, 6)
            cv2.polylines(overlay, cross_segs, False, WHITE, 1, cv2.LINE_AA)
            # Dual measurement arcs
            _arc(overlay, ex, ey, r_a, ang_a1, (ang_a1 + 270) % 360, ICE,    1)
            _arc(overlay, ex, ey, r_b, ang_b1, (ang_b1 + 250) % 360, ACCENT, 1)
            # Tick ring — vectorized
            ticks = _vec_ticks(ex, ey, r_b, r_b + 3, math.radians(t * 25), _TICK16_ANGLES)
            cv2.polylines(overlay, ticks, False, DIM, 1, cv2.LINE_AA)
            cv2.circle(overlay, (ex, ey), 3, WHITE, -1, cv2.LINE_AA)

        # ── Centroid measurement ring ─────────────────────────────────────────
        cx_f = int(pts[:, 0].mean()); cy_f = int(pts[:, 1].mean())
        span = int((pts[:, 0].max() - pts[:, 0].min()) * 0.60)
        _arc(overlay, cx_f, cy_f, span, 0, 360, (DIM[0]//2, DIM[1]//2, DIM[2]//2), 1)
        outer_ticks = _vec_ticks(cx_f, cy_f, span, span + 5, math.radians(t * -8), _TICK36_ANGLES)
        cv2.polylines(overlay, outer_ticks, False, DIM, 1, cv2.LINE_AA)

        # ── Cardinal diamonds — batch-built polylines ─────────────────────────
        diamond_polys = _vec_cardinal_diamonds(cx_f, cy_f, span, math.radians(t * 5))
        cv2.polylines(overlay, diamond_polys, True, ACCENT, 1, cv2.LINE_AA)

        return WHITE, "E.D.I.T.H // PRECISION"


# ─────────────────────────────────────────────────────────────────────────────
#  MODE 4 — INFRA-SPECTRUM THERMAL BONE SCAN
# ─────────────────────────────────────────────────────────────────────────────
class InfraSpectrumMode(HUDModeStrategy):
    def render(self, overlay, face_landmarks, w, h, t, pts, nose, lines, dist, state):
        COLD  = (8, 8, 50)
        WARM  = (10, 60, 160)
        HOT   = (0, 140, 255)
        BLAST = (0, 220, 255)
        WHITE = (0, 255, 255)

        # ── Cached trig ───────────────────────────────────────────────────────
        sin_p25 = math.sin(t * 2.5 * _TAU)
        sin_03  = math.sin(t * 0.3)
        p = 0.5 + 0.5 * sin_p25

        # ── Thermal gradient ──────────────────────────────────────────────────
        sweep_y = int((t * 200) % (h + 1))
        d = np.abs(lines[:, :, 1] - sweep_y).min(axis=1)
        cv2.polylines(overlay, lines[::2], False, COLD, 1, cv2.LINE_AA)
        if (d < 60).any(): cv2.polylines(overlay, lines[d < 60], False, WARM,  1, cv2.LINE_AA)
        if (d < 30).any(): cv2.polylines(overlay, lines[d < 30], False, HOT,   1, cv2.LINE_AA)
        if (d < 12).any(): cv2.polylines(overlay, lines[d < 12], False, BLAST, 1, cv2.LINE_AA)
        if (d < 3).any():  cv2.polylines(overlay, lines[d < 3],  False, WHITE, 2, cv2.LINE_AA)

        # ── Feature outlines ──────────────────────────────────────────────────
        draw_landmarks_manually(overlay, face_landmarks, FaceLandmarksConnections.FACE_LANDMARKS_FACE_OVAL,   COLD, 2)
        draw_landmarks_manually(overlay, face_landmarks, FaceLandmarksConnections.FACE_LANDMARKS_LEFT_EYE,    HOT,  2)
        draw_landmarks_manually(overlay, face_landmarks, FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_EYE,   HOT,  2)
        draw_landmarks_manually(overlay, face_landmarks, FaceLandmarksConnections.FACE_LANDMARKS_LIPS,        BLAST,1)

        # ── Heat plume rings — 4 cv2.circle calls (unavoidable, only 4) ───────
        nx, ny = int(face_landmarks[1].x * w), int(face_landmarks[1].y * h)
        for ring_i in range(4):
            ring_phase = (t * 2.0 + ring_i * 0.7) % 1.0
            ring_r = max(2, int(ring_phase * (16 + ring_i * 6)))
            cv2.circle(overlay, (nx, ny), ring_r, _col_lerp(WHITE, WARM, ring_phase), 1, cv2.LINE_AA)
        cv2.circle(overlay, (nx, ny), 4, WHITE, -1, cv2.LINE_AA)

        # ── Eye temperature arcs (p cached above) ─────────────────────────────
        r_temp = int(15 + 4 * p)
        arc_a1 = int(270 * p) + 30
        arc_b1 = int(200 * (1 - p)) + 20
        for eye_idx in (468, 473):
            ex, ey = _eye_center(face_landmarks, eye_idx, w, h)
            _arc(overlay, ex, ey, r_temp,     0, arc_a1, HOT,   1)
            _arc(overlay, ex, ey, r_temp - 5, 0, arc_b1, BLAST, 1)
            cv2.circle(overlay, (ex, ey), 3, WHITE, -1, cv2.LINE_AA)

        # ── Forehead sensor ───────────────────────────────────────────────────
        fx = int(face_landmarks[10].x * w)
        fy = int(face_landmarks[10].y * h)
        temp_val = int(36.5 + 1.2 * sin_03)
        _draw_diamond(overlay, fx, fy, 6, BLAST, filled=True)
        cv2.putText(overlay, f"{temp_val}.{int(p*9)}\xb0C", (fx+10, fy+4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, HOT, 1, cv2.LINE_AA)

        # ── Centroid heat rings ───────────────────────────────────────────────
        cx_f = int(pts[:, 0].mean()); cy_f = int(pts[:, 1].mean())
        span = int((pts[:, 0].max() - pts[:, 0].min()) * 0.58)
        _arc(overlay, cx_f, cy_f, span, 0, 360, (5,5,30), 1)
        ang_sw = int(t * 60) % 360
        _arc(overlay, cx_f, cy_f, span+8, ang_sw, (ang_sw+180)%360, WARM, 1)

        return COLD, "THERMAL // TARGET"


# ─────────────────────────────────────────────────────────────────────────────
#  MODE 5 — ULTRON NEURAL VOID
# ─────────────────────────────────────────────────────────────────────────────
class UltronMode(HUDModeStrategy):
    _hex_buf = np.zeros((7, 1, 2), dtype=np.int32)

    def render(self, overlay, face_landmarks, w, h, t, pts, nose, lines, dist, state):
        from renderer import _face_tess_edges

        VOID   = (35, 4, 32)
        MID    = (110, 15, 95)
        PLASMA = (210, 55, 195)
        HOT    = (240, 110, 230)
        WHITE  = (255, 160, 250)

        # ── Cached trig ───────────────────────────────────────────────────────
        sin_strand = math.sin(t * 1.5 * _TAU)
        sin_hull   = math.sin(t * 1.1)
        base_arm   = math.radians(t * 70)

        strand_col = _col_lerp(VOID, MID, 0.5 + 0.5 * sin_strand)

        # ── Multi-tier sparse mesh ────────────────────────────────────────────
        cv2.polylines(overlay, lines[::4], False, VOID,      1, cv2.LINE_AA)
        cv2.polylines(overlay, lines[::7], False, (18,2,15), 1, cv2.LINE_AA)

        # ── Plasma sweep waves ────────────────────────────────────────────────
        sweep1 = int((t * 450) % (h + 1))
        d1 = np.abs(lines[:, :, 1] - sweep1).min(axis=1)
        if (d1 < 22).any(): cv2.polylines(overlay, lines[d1 < 22], False, MID,    1, cv2.LINE_AA)
        if (d1 < 8).any():  cv2.polylines(overlay, lines[d1 < 8],  False, PLASMA, 1, cv2.LINE_AA)
        if (d1 < 2).any():  cv2.polylines(overlay, lines[d1 < 2],  False, HOT,    2, cv2.LINE_AA)
        sweep2 = int(h - (t * 200) % (h + 1))
        d2 = np.abs(lines[:, :, 1] - sweep2).min(axis=1)
        if (d2 < 10).any(): cv2.polylines(overlay, lines[d2 < 10], False, VOID, 1, cv2.LINE_AA)

        # ── Neural web — vectorized strand segments ───────────────────────────
        cx_f = int(pts[:, 0].mean()); cy_f = int(pts[:, 1].mean())
        strand_pts = pts[::8]          # (N~60, 2) view — no alloc
        strand_segs = _vec_strand_segs(cx_f, cy_f, strand_pts)
        cv2.polylines(overlay, strand_segs, False, strand_col, 1, cv2.LINE_AA)

        # ── Outer hull ────────────────────────────────────────────────────────
        scale_out = 1.05 + 0.02 * sin_hull
        outer = (nose + (pts - nose) * scale_out).astype(np.int32)
        o_lines = outer[_face_tess_edges]
        cv2.polylines(overlay, o_lines[::5], False, (12,1,10), 1, cv2.LINE_AA)

        # ── Feature contours ─────────────────────────────────────────────────
        draw_landmarks_manually(overlay, face_landmarks, FaceLandmarksConnections.FACE_LANDMARKS_FACE_OVAL,   MID,    1)
        draw_landmarks_manually(overlay, face_landmarks, FaceLandmarksConnections.FACE_LANDMARKS_LEFT_EYE,    PLASMA, 2)
        draw_landmarks_manually(overlay, face_landmarks, FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_EYE,   PLASMA, 2)
        draw_landmarks_manually(overlay, face_landmarks, FaceLandmarksConnections.FACE_LANDMARKS_LIPS,        MID,    1)

        # ── Eye abyss rings — 4+4 = 8 cv2.circle calls (minimized) ──────────
        t13 = t * 1.3
        for eye_idx, phase_off in ((468, 0.0), (473, math.pi)):
            ex, ey = _eye_center(face_landmarks, eye_idx, w, h)
            for ring_i in range(4):
                alpha = 0.5 + 0.5 * math.sin(t13 + ring_i * 0.5 + phase_off)
                col = _col_lerp(VOID, PLASMA, alpha)
                cv2.circle(overlay, (ex, ey), 8 + ring_i * 5, col, 1, cv2.LINE_AA)
            cv2.circle(overlay, (ex, ey), 3, HOT, -1, cv2.LINE_AA)

        # ── Expanding void rings — 4 cv2.circle calls ─────────────────────────
        span = int((pts[:, 0].max() - pts[:, 0].min()) * 0.62)
        t11 = t * 1.1
        for ring in range(4):
            phase = (t11 + ring * 0.5) % 1.0
            rad = max(2, int(phase * span * 0.75))
            af = 1.0 - phase
            cv2.circle(overlay, (cx_f, cy_f), rad,
                       (int(MID[0]*af), int(MID[1]*af), int(MID[2]*af)), 1, cv2.LINE_AA)

        # ── Plasma tentacles — two batched cv2.polylines calls ─────────────────
        arm_r = int(span * 0.75)
        segs_inner, segs_outer, tips = _vec_tentacle_segs(cx_f, cy_f, arm_r, base_arm)
        cv2.polylines(overlay, segs_inner, False, MID,    1, cv2.LINE_AA)
        cv2.polylines(overlay, segs_outer, False, PLASMA, 1, cv2.LINE_AA)
        for i in range(4):
            cv2.circle(overlay, (int(tips[i, 0]), int(tips[i, 1])), 3, HOT, -1, cv2.LINE_AA)

        # ── Outer hex + tick ring ─────────────────────────────────────────────
        hex_arr = _vec_hex(cx_f, cy_f, span, math.radians(t * 20), self._hex_buf)
        cv2.polylines(overlay, hex_arr, True, VOID, 1, cv2.LINE_AA)
        ticks = _vec_ticks(cx_f, cy_f, span+10, span+14, math.radians(-t*14), _TICK18_ANGLES)
        cv2.polylines(overlay, ticks, False, (MID[0]//2, MID[1]//2, MID[2]//2), 1, cv2.LINE_AA)

        return PLASMA, "ULTRON // NEURAL VOID"


# ─────────────────────────────────────────────────────────────────────────────
#  MODE REGISTRY
# ─────────────────────────────────────────────────────────────────────────────
MODES = {
    0: NanoMode(),
    1: QuantumMode(),
    2: NeonCipherMode(),
    3: EdithMode(),
    4: InfraSpectrumMode(),
    5: UltronMode(),
}
