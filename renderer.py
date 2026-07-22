from hud_modes import MODES
from mediapipe.tasks.python.vision import FaceLandmarksConnections
try:
    from ar_hologram import render_palm_hologram
except ImportError:
    def render_palm_hologram(*args, **kwargs): pass
import logging
import os
import random
from ui_components import *
import cv2
import numpy as np
import math
import psutil
import time
import time as _t
import threading
import itertools
import mss
import random









from utils import *

# === Precomputed vectorized edges ===
from mediapipe.tasks.python.vision import FaceLandmarksConnections, HandLandmarksConnections
_face_tess_edges = np.array([[c.start, c.end] for c in FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION], dtype=np.int32)
_face_contour_edges = np.array([[c.start, c.end] for c in FaceLandmarksConnections.FACE_LANDMARKS_CONTOURS], dtype=np.int32)
_base_hand = [[c.start, c.end] for c in HandLandmarksConnections.HAND_CONNECTIONS]
_extra_hand = [[1, 5], [2, 5], [5, 13], [5, 17], [9, 17], [1, 17], [0, 9], [0, 13], [9, 13], [13, 17]]
_hand_edges = np.array(_base_hand + _extra_hand, dtype=np.int32)

_last_hud_t = 0
_current_fps = 30
FACE_PERSIST_SECONDS = 1.5


def render_face_tracking(overlay, active_face_list, w, h, t, state):

    for face_landmarks in active_face_list:
        mode = state.face_mesh_mode
        # Use pre-allocated NumPy array mapped by utils.py
        pts = state.face_pts_buf
        if pts is None or len(pts) != len(face_landmarks):
            pts = np.zeros((len(face_landmarks), 2), dtype=np.int32)
            state.face_pts_buf = pts
        
        # We still need to map the smoothed float [0,1] normalized coords to high-res pixels
        # But we do it directly into the pre-allocated buffer
        for i, lm in enumerate(face_landmarks):
            pts[i, 0] = int(lm.x * w)
            pts[i, 1] = int(lm.y * h)
        nose = pts[1]
        lines = pts[_face_tess_edges]
        sweep_y = int((t * 320) % h)
        dist = np.abs(lines[:,:,1] - sweep_y).min(axis=1)

        strategy = MODES.get(mode, MODES[0])
        CC, TAG = strategy.render(overlay, face_landmarks, w, h, t, pts, nose, lines, dist, state)

        # ── Shared: iris with animated glow rings ──────────────────────────────

        draw_landmarks_manually(overlay, face_landmarks, FaceLandmarksConnections.FACE_LANDMARKS_CONTOURS, CC, 2)
        # Iris bright ring (pulsing)
        iris_r = int(6 + 2 * math.sin(t * 3.8))
        for iris_idx in [468, 473]:
            ix = int(face_landmarks[iris_idx].x * w)
            iy = int(face_landmarks[iris_idx].y * h)
            cv2.circle(overlay, (ix, iy), iris_r, (255,255,255), 1, cv2.LINE_AA)
            cv2.circle(overlay, (ix, iy), 2, CC, -1, cv2.LINE_AA)

        # ── Bounding box ─────────────────────────────────────────────────────
        xs = pts[:,0]; ys_all = pts[:,1]
        min_x = int(xs.min() - (xs.max()-xs.min())*0.10)
        max_x = int(xs.max() + (xs.max()-xs.min())*0.10)
        min_y = int(ys_all.min() - (ys_all.max()-ys_all.min())*0.25)
        max_y = int(ys_all.max() + (ys_all.max()-ys_all.min())*0.10)

        # Face-acquisition burst (1 second when first detected)
        tb = t - state.last_face_time
        if tb < 1.0:
            exp = int((1.0-tb)*200)
            min_x-=exp; max_x+=exp; min_y-=exp; max_y+=exp
            # Fade-in flash ring
            flash_r = int((1.0 - tb) * (max_x - min_x) // 2)
            fcx_b = (min_x+max_x)//2; fcy_b = (min_y+max_y)//2
            flash_alpha = 1.0 - tb
            flash_col = (int(CC[0]*flash_alpha), int(CC[1]*flash_alpha), int(CC[2]*flash_alpha))
            if flash_r > 1:
                cv2.circle(overlay, (fcx_b, fcy_b), flash_r, flash_col, 1, cv2.LINE_AA)

        draw_hud_brackets(overlay, min_x, min_y, max_x, max_y, CC, 2)

        # ── Centroid HUD assemblies ────────────────────────────────────────────
        fcx = (min_x+max_x)//2;  fcy = (min_y+max_y)//2
        sc  = max(0.4, (max_x-min_x)/300.0)
        draw_mcu_compass(overlay, fcx, fcy, t, sc)
        yaw   = (face_landmarks[234].z - face_landmarks[454].z)*12.0
        pitch = (face_landmarks[10].z  - face_landmarks[152].z)*12.0
        draw_flight_ladder(overlay, fcx, fcy, t, pitch, yaw, sc)
        draw_rotating_reticle(overlay, fcx, fcy, int(max_x-fcx)+10, t*50, CC)

        # ── Dual scan bars ────────────────────────────────────────────────────
        height_span = max(1, max_y - min_y)
        sw1 = min_y + int((t * 55) % height_span)
        sw2 = min_y + int((t * 88 + height_span * 0.5) % height_span)
        cv2.line(overlay, (min_x, sw1), (max_x, sw1), CC, 1)
        dim_cc = (CC[0]//3, CC[1]//3, CC[2]//3)
        cv2.line(overlay, (min_x, sw2), (max_x, sw2), dim_cc, 1)

        # ── Animated TAG widget ───────────────────────────────────────────────
        tx, ty = min_x+4, min_y+16
        tsz = cv2.getTextSize(TAG, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)[0]
        # Filled dark background
        cv2.rectangle(overlay, (tx-3, ty-12), (tx+tsz[0]+4, ty+3), (3, 3, 8), -1)
        # Animated corner accent
        blink = (int(t * 4) % 2 == 0)
        border_col = CC if blink else (CC[0]//2, CC[1]//2, CC[2]//2)
        cv2.rectangle(overlay, (tx-3, ty-12), (tx+tsz[0]+4, ty+3), border_col, 1)
        cv2.putText(overlay, TAG, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.38, CC, 1, cv2.LINE_AA)
        # Small blinking status dot
        dot_col = CC if blink else (CC[0]//3, CC[1]//3, CC[2]//3)
        cv2.circle(overlay, (tx-3+tsz[0]+10, ty-5), 3, dot_col, -1, cv2.LINE_AA)

        # ── Sub-pixel confidence readout ──────────────────────────────────────
        conf_str = f"{int(92 + 6*math.sin(t*0.7))}%"
        cv2.putText(overlay, conf_str, (min_x+4, max_y-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.28, (CC[0]//2, CC[1]//2, CC[2]//2), 1, cv2.LINE_AA)

        return min_y
    return None


def render_hand_tracking(overlay, hand_landmarks_list, hand_handedness_list, w, h, t, state):
    mode = state.face_mesh_mode

    for hand_idx, hand_landmarks in enumerate(hand_landmarks_list):
        hand_id = f"Hand_{hand_idx}"
        pts = state.hand_pts_bufs.get(hand_id)
        if pts is None or len(pts) != len(hand_landmarks):
            pts = np.zeros((len(hand_landmarks), 2), dtype=np.int32)
            state.hand_pts_bufs[hand_id] = pts
            
        for i, lm in enumerate(hand_landmarks):
            pts[i, 0] = int(lm.x * w)
            pts[i, 1] = int(lm.y * h)
        if len(pts) < 21: continue

        # ── OUT-OF-FRAME GUARD: drop this hand if too many landmarks are clamped to screen edge ──
        # MediaPipe clamps OOF landmarks to 0.0 or 1.0, which causes skeleton stretching
        # Recycle the x and y columns from our pre-populated integer pts buffer
        # dividing by w/h recovers the normalized coords perfectly without new allocations
        raw_xs = pts[:, 0] / w
        raw_ys = pts[:, 1] / h
        edge_count = np.sum((raw_xs < 0.01) | (raw_xs > 0.99) | (raw_ys < 0.01) | (raw_ys > 0.99))
        if edge_count > 5:  # more than 5 joints clamped = hand is mostly OOF, skip drawing
            continue
        hdn = (hand_handedness_list[hand_idx][0].category_name if hand_handedness_list else "?")
        hmin_x=max(0,int(pts[:,0].min())-15); hmax_x=min(w,int(pts[:,0].max())+15)
        hmin_y=max(0,int(pts[:,1].min())-15); hmax_y=min(h,int(pts[:,1].max())+15)
        ep   = pts[_hand_edges]              # edge pairs (N,2,2)
        wrist= pts[0];  palm=pts[9]
        scale= float(np.linalg.norm(palm-wrist))+1e-6
        hull = cv2.convexHull(pts)
        sweep_y = int((t*280)%h)
        dist    = np.abs(ep[:,:,1]-sweep_y).min(axis=1)

        # ── MODE 0: NANO-ARMOR GAUNTLET ── gold hull + tri-layer skeleton ──────
        if mode == 0:
            G=(0,200,255); A=(20,110,255); W=(255,255,255); D=(6,18,50)
            cv2.fillConvexPoly(overlay, hull, D)
            cv2.polylines(overlay,[hull],True,A,2,cv2.LINE_AA)
            cv2.polylines(overlay,ep,False,A,1,cv2.LINE_AA)
            # Scan highlight
            if (dist<18).any(): cv2.polylines(overlay,ep[dist<18],False,G,1,cv2.LINE_AA)
            if (dist< 4).any(): cv2.polylines(overlay,ep[dist< 4],False,W,1,cv2.LINE_AA)
            # Palm arc reactor: 3 concentric rings + 6 rotating spokes (cheap)
            for rf in [0.20,0.12,0.06]:
                pass
            for i in range(6):
                a = t*2.5 + i*math.pi/3
                rx=px+int(math.cos(a)*scale*0.15); ry=py+int(math.sin(a)*scale*0.15)
                cv2.circle(overlay,(rx,ry),2,G,-1,cv2.LINE_AA)
            # Fingertip repulsor rings
            pulse=abs(math.sin(t*6))
            for ti in [4,8,12,16,20]:
                tx2,ty2=pts[ti]
                cv2.circle(overlay,(tx2,ty2),int(8+pulse*5),G,1,cv2.LINE_AA)
                cv2.circle(overlay,(tx2,ty2),3,W,-1,cv2.LINE_AA)
            # Joint nodes
            for i,(nx,ny) in enumerate(pts):
                cv2.circle(overlay,(nx,ny),3 if i in [4,8,12,16,20] else 2,
                           G if i in [4,8,12,16,20] else A,-1,cv2.LINE_AA)
            draw_hud_brackets(overlay,hmin_x,hmin_y,hmax_x,hmax_y,G,2)
            cv2.putText(overlay,f"NANO GAUNTLET [{hdn}]",(hmin_x,hmin_y-10),cv2.FONT_HERSHEY_SIMPLEX,0.36,G,1,cv2.LINE_AA)

        # ── MODE 1: PRISM QUANTUM BLUEPRINT ── matches face: amber shell + struts + sweep ──
        elif mode == 1:
            C1 = (60,  40,  10)    # very dim cognac base (same as face)
            C2 = (255, 200, 80)    # prism amber highlight
            CW = (255, 240, 160)   # warm white hot core
            DK = (20,  12,   3)    # near-black outer shell

            # ── Layer 1: Dark base skeleton (all bones dim cognac) ────────
            cv2.polylines(overlay, ep, False, C1, 1, cv2.LINE_AA)

            # ── Layer 2: Outer expanded shell + struts (matches face outer) ─
            # Expand pts outward from palm center to create an "armor shell"
            pcx, pcy = pts.mean(axis=0).astype(int)
            center   = np.array([pcx, pcy], dtype=np.float32)
            outer_h  = (center + (pts - center) * 1.10).astype(np.int32)
            ep_outer = outer_h[_hand_edges]
            cv2.polylines(overlay, ep_outer[::2], False, DK, 1, cv2.LINE_AA)
            # Struts inner→outer (every 3rd joint)
            struts = np.stack((pts[::3], outer_h[::3]), axis=1)
            cv2.polylines(overlay, struts, False, (40, 22, 5), 1, cv2.LINE_AA)

            # ── Layer 3: Amber sweep wave (same speed as face 320px/s) ────
            sweep_a = int((t * 320) % max(1, h))
            d_a     = np.abs(ep[:, :, 1] - sweep_a).min(axis=1)
            if (d_a < 28).any(): cv2.polylines(overlay, ep[d_a < 28], False, CW, 1, cv2.LINE_AA)
            if (d_a <  6).any(): cv2.polylines(overlay, ep[d_a <  6], False, (255, 255, 255), 1, cv2.LINE_AA)

            # ── Layer 4: Midpoint data nodes (every 2nd bone, like face) ──
            mids = ep.mean(axis=1).astype(np.int32)
            for mx2, my2 in mids[::2]:
                cv2.circle(overlay, (mx2, my2), 1, (200, 140, 40), -1, cv2.LINE_AA)

            # ── Layer 5: Fingertip arcs (amber pulsing rings like eye rings) ─
            pulse = abs(math.sin(t * 5))
            for ti in [4, 8, 12, 16, 20]:
                tx2, ty2 = pts[ti]
                r = int(8 + pulse * 5)
                cv2.circle(overlay, (tx2, ty2), r,     C2,  1, cv2.LINE_AA)
                cv2.circle(overlay, (tx2, ty2), r - 3, CW,  1, cv2.LINE_AA)
                cv2.circle(overlay, (tx2, ty2), 2,     (255,255,255), -1, cv2.LINE_AA)

            # ── Layer 6: Knuckle anchors (double ring like face contours) ─
            for i, (nx, ny) in enumerate(pts):
                if i in [4, 8, 12, 16, 20]:
                    pass  # handled above
                elif i in [0, 5, 9, 13, 17]:
                    cv2.circle(overlay, (nx, ny), 6, C1, 1, cv2.LINE_AA)
                    cv2.circle(overlay, (nx, ny), 4, C2, 1, cv2.LINE_AA)
                    cv2.circle(overlay, (nx, ny), 2, CW, -1, cv2.LINE_AA)
                else:
                    cv2.circle(overlay, (nx, ny), 2, C1, -1, cv2.LINE_AA)

            # ── Slow scan line through the hand (matches face scan line) ──
            sw = hmin_y + int((t * 55) % max(1, hmax_y - hmin_y))
            cv2.line(overlay, (hmin_x, sw), (hmax_x, sw), C2, 1)

            # ── HUD bracket + tag box (exactly like face) ─────────────────
            draw_hud_brackets(overlay, hmin_x, hmin_y, hmax_x, hmax_y, C2, 2)
            TAG = f"P.R.I.S.M // [{hdn}]"
            tsz = cv2.getTextSize(TAG, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)[0]
            tx2, ty2 = hmin_x + 4, hmin_y - 6
            cv2.rectangle(overlay, (tx2-3, ty2-12), (tx2+tsz[0]+4, ty2+3), (4, 4, 12), -1)
            cv2.rectangle(overlay, (tx2-3, ty2-12), (tx2+tsz[0]+4, ty2+3), C2, 1)
            cv2.putText(overlay, TAG, (tx2, ty2), cv2.FONT_HERSHEY_SIMPLEX, 0.38, C2, 1, cv2.LINE_AA)


        # ── MODE 2: NEON CIPHER ── elite hacker 2100 data-stream gauntlet ───────
        elif mode == 2:
            # Color palette: electric lime + acid cyan + ghost white
            LIME   = (20,  255,  80)   # BGR bright lime
            CYAN   = (255, 230,   0)   # BGR electric cyan
            DIM    = (10,  80,   30)   # dark skeleton base
            WHITE  = (255, 255, 255)
            ACID   = ( 40, 255, 120)   # mid green
            
            # ── Layer 1: Dark base skeleton (all bones) ───────────────────
            cv2.polylines(overlay, ep, False, DIM, 1, cv2.LINE_AA)
            
            # ── Layer 2: Animated "data pulse" sweep up all bones ─────────
            # Two sweeps at different speeds for a layered effect
            sweep_a = int((t * 420) % max(1, h))
            sweep_b = int((t * 220 + h * 0.4) % max(1, h))
            d_a = np.abs(ep[:, :, 1] - sweep_a).min(axis=1)
            d_b = np.abs(ep[:, :, 1] - sweep_b).min(axis=1)
            
            hot_a = d_a < 14
            hot_b = d_b < 10
            if hot_a.any(): cv2.polylines(overlay, ep[hot_a], False, ACID, 2, cv2.LINE_AA)
            if hot_b.any(): cv2.polylines(overlay, ep[hot_b], False, CYAN, 1, cv2.LINE_AA)
            
            # ── Layer 3: Glow on every bone segment (bright core) ────────
            # Only every-other bone so it looks like a data lattice
            lattice_mask = (np.arange(len(ep)) % 2) == 0
            cv2.polylines(overlay, ep[lattice_mask], False, LIME, 1, cv2.LINE_AA)
            
            # ── Layer 4: HEXAGONAL fingertip terminals ────────────────────
            pulse = abs(math.sin(t * 6))
            for ti in [4, 8, 12, 16, 20]:
                tx2, ty2 = pts[ti]
                # Draw a hexagon at each fingertip
                hex_r = int(9 + pulse * 4)
                for angle_i in range(6):
                    a1 = t * 1.5 + angle_i * math.pi / 3
                    a2 = t * 1.5 + (angle_i + 1) * math.pi / 3
                    hx1 = tx2 + int(math.cos(a1) * hex_r)
                    hy1 = ty2 + int(math.sin(a1) * hex_r)
                    hx2 = tx2 + int(math.cos(a2) * hex_r)
                    hy2 = ty2 + int(math.sin(a2) * hex_r)
                    cv2.line(overlay, (hx1, hy1), (hx2, hy2), CYAN, 1, cv2.LINE_AA)
                # Outer glow ring
                cv2.circle(overlay, (tx2, ty2), hex_r + 4, DIM, 1, cv2.LINE_AA)
                # Bright core dot
                cv2.circle(overlay, (tx2, ty2), 3, WHITE, -1, cv2.LINE_AA)
                
                # ── Data rain drop below each fingertip ──────────────────
                drop_len = 18 + int(pulse * 10)
                drop_y   = ty2 + int((t * 130 + ti * 20) % 40)
                cv2.line(overlay, (tx2, ty2 + 6), (tx2, min(drop_y, ty2 + drop_len)), LIME, 1, cv2.LINE_AA)
            
            # ── Layer 5: Palm hacking sigil (rotating geometry) ───────────
            # Outer rotating square
            sq_r  = int(scale * 0.22)
            angle = t * 1.2
            sq_corners = [(
                px + int(math.cos(angle + i * math.pi / 2) * sq_r),
                py + int(math.sin(angle + i * math.pi / 2) * sq_r)
            ) for i in range(4)]
            for i in range(4):
                cv2.line(overlay, sq_corners[i], sq_corners[(i+1)%4], CYAN, 1, cv2.LINE_AA)
            
            # Inner counter-rotating triangle
            tri_r = int(scale * 0.12)
            angle2 = -t * 2.0
            tri_corners = [(
                px + int(math.cos(angle2 + i * 2 * math.pi / 3) * tri_r),
                py + int(math.sin(angle2 + i * 2 * math.pi / 3) * tri_r)
            ) for i in range(3)]
            for i in range(3):
                cv2.line(overlay, tri_corners[i], tri_corners[(i+1)%3], LIME, 1, cv2.LINE_AA)
            
            # Center pulse dot
            center_r = max(2, int(3 + pulse * 3))
            
            # ── Layer 6: All joint nodes with data node style ────────────
            for i, (nx, ny) in enumerate(pts):
                if i in [4, 8, 12, 16, 20]:
                    pass  # handled above
                elif i in [0, 5, 9, 13, 17]:
                    # Knuckle anchors: diamond shape
                    d_sz = 4
                    cv2.line(overlay, (nx - d_sz, ny), (nx, ny - d_sz), LIME, 1, cv2.LINE_AA)
                    cv2.line(overlay, (nx, ny - d_sz), (nx + d_sz, ny), LIME, 1, cv2.LINE_AA)
                    cv2.line(overlay, (nx + d_sz, ny), (nx, ny + d_sz), LIME, 1, cv2.LINE_AA)
                    cv2.line(overlay, (nx, ny + d_sz), (nx - d_sz, ny), LIME, 1, cv2.LINE_AA)
                else:
                    cv2.circle(overlay, (nx, ny), 2, ACID, -1, cv2.LINE_AA)
            
            # ── HUD frame + label ─────────────────────────────────────────
            draw_hud_brackets(overlay, hmin_x, hmin_y, hmax_x, hmax_y, LIME, 1)
            # Animated label with blink
            blink = int(t * 3) % 2 == 0
            label_col = LIME if blink else CYAN
            cv2.putText(overlay, f"CIPHER.EXE [{hdn}]", (hmin_x, hmin_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.36, label_col, 1, cv2.LINE_AA)

        # ── MODE 3: EDITH SURGICAL ── white precision wireframe + reticles ────
        elif mode == 3:
            W=(255,255,255); I=(210,230,255); T=(140,140,140)
            cv2.polylines(overlay,ep,False,T,1,cv2.LINE_AA)
            for i,(nx,ny) in enumerate(pts):
                cv2.circle(overlay,(nx,ny),4 if i in[4,8,12,16,20] else 2,
                           W if i in[4,8,12,16,20] else I,-1,cv2.LINE_AA)
            for ti in [4,8,12,16,20]:
                tx2,ty2=pts[ti]
                cv2.circle(overlay,(tx2,ty2),11,W,1,cv2.LINE_AA)
                cv2.line(overlay,(tx2-16,ty2),(tx2+16,ty2),W,1,cv2.LINE_AA)
                cv2.line(overlay,(tx2,ty2-11),(tx2,ty2+11),W,1,cv2.LINE_AA)
            draw_hud_brackets(overlay,hmin_x,hmin_y,hmax_x,hmax_y,W,1)
            cv2.putText(overlay,f"SURGICAL [{hdn}]",(hmin_x,hmin_y-10),cv2.FONT_HERSHEY_SIMPLEX,0.36,W,1,cv2.LINE_AA)

        # ── MODE 4: INFRA-SPECTRUM ── thermal bone heat map ───────────────────
        elif mode == 4:
            R=(20,20,160); O=(20,130,255); Y=(0,220,255); D=(5,5,20)
            cv2.fillConvexPoly(overlay,hull,D)
            cv2.polylines(overlay,ep,False,R,1,cv2.LINE_AA)
            if (dist<20).any(): cv2.polylines(overlay,ep[dist<20],False,O,1,cv2.LINE_AA)
            if (dist< 4).any(): cv2.polylines(overlay,ep[dist< 4],False,Y,1,cv2.LINE_AA)
            pulse=abs(math.sin(t*4))
            for i,(nx,ny) in enumerate(pts):
                if i in[4,8,12,16,20]:
                    cv2.circle(overlay,(nx,ny),int(6+pulse*4),O,1,cv2.LINE_AA)
                    cv2.circle(overlay,(nx,ny),3,Y,-1,cv2.LINE_AA)
                else:
                    cv2.circle(overlay,(nx,ny),2,R,-1,cv2.LINE_AA)
            draw_hud_brackets(overlay,hmin_x,hmin_y,hmax_x,hmax_y,R,2)
            cv2.putText(overlay,f"THERMAL [{hdn}]",(hmin_x,hmin_y-10),cv2.FONT_HERSHEY_SIMPLEX,0.36,O,1,cv2.LINE_AA)

        # ── MODE 5: ULTRON NEURAL VOID ── deep indigo plasma + void rings ─────
        elif mode == 5:
            P=(160,10,140); V=(220,60,200); W=(255,255,255)
            cv2.polylines(overlay,ep,False,P,1,cv2.LINE_AA)
            if (dist<18).any(): cv2.polylines(overlay,ep[dist<18],False,V,1,cv2.LINE_AA)
            for i,(nx,ny) in enumerate(pts):
                cv2.circle(overlay,(nx,ny),4 if i in[4,8,12,16,20] else 2,
                           V if i in[4,8,12,16,20] else P,-1,cv2.LINE_AA)
            for ring in range(3):
                phase=(t*2.2+ring*0.35)%1.0
                rad=max(1,int(phase*scale*0.55))
                alpha=int(210*(1.0-phase))
            for ti in [4,8,12,16,20]:
                tx2,ty2=pts[ti]
                cv2.circle(overlay,(tx2,ty2),6,V,1,cv2.LINE_AA)
                cv2.line(overlay,(px,py),(tx2,ty2),(50,5,50),1,cv2.LINE_AA)
            draw_hud_brackets(overlay,hmin_x,hmin_y,hmax_x,hmax_y,V,2)
            cv2.putText(overlay,f"NEURAL VOID [{hdn}]",(hmin_x,hmin_y-10),cv2.FONT_HERSHEY_SIMPLEX,0.36,V,1,cv2.LINE_AA)


def render_pose_tracking(overlay, pose_landmarks_list, face_landmarks_list, hand_landmarks_list, w, h, t, state):
    face_chin=None
    if face_landmarks_list:
        face_chin=(int(face_landmarks_list[0][152].x*w),int(face_landmarks_list[0][152].y*h))
    hand_wrists=[]
    if hand_landmarks_list:
        for hl in hand_landmarks_list:
            hand_wrists.append((int(hl[0].x*w),int(hl[0].y*h)))

    for pose_landmarks in pose_landmarks_list:
        ptsl=[]; vis=[]
        for lm in pose_landmarks:
            ptsl.append([int(lm.x*w),int(lm.y*h)]); vis.append(lm.visibility>0.5)
        pts=np.array(ptsl,dtype=np.int32)
        edges=[(11,13),(13,15),(12,14),(14,16),(11,12),(11,23),(12,24),(23,24)]
        vl=[]
        for s,e in edges:
            if s<len(vis) and e<len(vis) and vis[s] and vis[e]:
                vl.append([pts[s],pts[e]])
        for awi in [15,16]:
            if awi<len(vis) and vis[awi]:
                ap=pts[awi]
                if hand_wrists:
                    cl=min(hand_wrists,key=lambda hp:(hp[0]-ap[0])**2+(hp[1]-ap[1])**2)
                    if (cl[0]-ap[0])**2+(cl[1]-ap[1])**2<90000: vl.append([ap,cl])
        if face_chin:
            if 11<len(vis) and vis[11]: vl.append([pts[11],face_chin])
            if 12<len(vis) and vis[12]: vl.append([pts[12],face_chin])
        vjoints=[pts[i] for i in[11,12,13,14,15,16,23,24] if i<len(vis) and vis[i]]
        if not vl: continue
        vn=np.array(vl,dtype=np.int32)
        sweep_y=int((t*280)%h)
        dist=np.abs(vn[:,:,1]-sweep_y).min(axis=1)
        mode=state.face_mesh_mode

        # ── MODE 0: NANO-ARMOR EXOSKELETON ── gold tri-layer arms ────────────
        if mode == 0:
            G=(0,200,255); A=(20,110,255); D=(6,18,50)
            draw_polylines_deployed(overlay,vn,False,D,8,cv2.LINE_AA,state)
            draw_polylines_deployed(overlay,vn,False,A,3,cv2.LINE_AA,state)
            draw_polylines_deployed(overlay,vn,False,G,1,cv2.LINE_AA,state)
            if (dist<18).any(): draw_polylines_deployed(overlay,vn[dist<18],False,(255,255,255),2,cv2.LINE_AA,state)
            for pt in vjoints:
                cv2.circle(overlay,tuple(pt),8,D,-1,cv2.LINE_AA)
                cv2.circle(overlay,tuple(pt),8,G,2,cv2.LINE_AA)
                cv2.circle(overlay,tuple(pt),4,(255,255,255),-1,cv2.LINE_AA)

        # ── MODE 1: PRISM QUANTUM SHELL ── triple volumetric amber layers ────
        elif mode == 1:
            H=(255,200,80); D=(60,30,8); B=(255,240,150); W=(255,255,255)
            if 11<len(vis) and 12<len(vis) and vis[11] and vis[12]:
                cp=(pts[11]+pts[12])//2
            else:
                cp=vn[0][0]
            def off(lines,f):
                return [[[l[0][0]+int((l[0][0]-cp[0])*f),l[0][1]+int((l[0][1]-cp[1])*f)],
                          [l[1][0]+int((l[1][0]-cp[0])*f),l[1][1]+int((l[1][1]-cp[1])*f)]] for l in vl]
            lo1=np.array(off(vl,0.04),dtype=np.int32)
            lo2=np.array(off(vl,0.08),dtype=np.int32)
            draw_polylines_deployed(overlay,vn, False,D, 1,cv2.LINE_AA,state)
            draw_polylines_deployed(overlay,lo1,False,(25,12,3),1,cv2.LINE_AA,state)
            draw_polylines_deployed(overlay,lo2,False,(12,6,1), 1,cv2.LINE_AA,state)
            st=np.stack((vn[:,0],lo1[:,0]),axis=1)
            draw_polylines_deployed(overlay,st,False,D,1,cv2.LINE_AA,state)
            al=np.concatenate((vn,lo1,lo2))
            dA=np.abs(al[:,:,1]-sweep_y).min(axis=1)
            if (dA<25).any(): draw_polylines_deployed(overlay,al[dA<25],False,B,1,cv2.LINE_AA,state)
            if (dA< 6).any(): draw_polylines_deployed(overlay,al[dA< 6],False,W,1,cv2.LINE_AA,state)
            for ji,pt in enumerate(vjoints):
                p=abs(math.sin(t*4+ji*0.6))
                cv2.circle(overlay,tuple(pt),int(5+p*3),B,1,cv2.LINE_AA)
                cv2.circle(overlay,tuple(pt),2,W,-1,cv2.LINE_AA)

        # ── MODE 2: NEON CIPHER FRAME ── cyan-lime glitch skeleton ──────────
        elif mode == 2:
            C=(255,255,40); L=(180,255,60); W=(255,255,255)
            keep=(np.arange(len(vn))%2)==0
            draw_polylines_deployed(overlay,vn[keep],False,(50,80,10),1,cv2.LINE_AA,state)
            sw2=int((t*500)%h); d2=np.abs(vn[:,:,1]-sw2).min(axis=1)
            if (d2<12).any(): draw_polylines_deployed(overlay,vn[d2<12],False,C,2,cv2.LINE_AA,state)
            for pt in vjoints:
                cv2.rectangle(overlay,(pt[0]-9,pt[1]-9),(pt[0]+9,pt[1]+9),C,1,cv2.LINE_AA)
                cv2.circle(overlay,tuple(pt),4,W,-1,cv2.LINE_AA)

        # ── MODE 3: EDITH SURGICAL SKELETON ── white precision joints ─────────
        elif mode == 3:
            W=(255,255,255); I=(190,210,255)
            draw_polylines_deployed(overlay,vn,False,(110,110,110),1,cv2.LINE_AA,state)
            for pt in vjoints:
                cv2.circle(overlay,tuple(pt),5,W,-1,cv2.LINE_AA)
                cv2.circle(overlay,tuple(pt),12,I,1,cv2.LINE_AA)
                cv2.line(overlay,(pt[0]-17,pt[1]),(pt[0]+17,pt[1]),W,1,cv2.LINE_AA)
                cv2.line(overlay,(pt[0],pt[1]-12),(pt[0],pt[1]+12),W,1,cv2.LINE_AA)

        # ── MODE 4: THERMAL SKELETON ── red bone heat with orange joint rings ──
        elif mode == 4:
            R=(20,20,160); O=(20,130,255); Y=(0,220,255)
            draw_polylines_deployed(overlay,vn,False,(8,8,50),6,cv2.LINE_AA,state)
            draw_polylines_deployed(overlay,vn,False,R,3,cv2.LINE_AA,state)
            if (dist<18).any(): draw_polylines_deployed(overlay,vn[dist<18],False,O,2,cv2.LINE_AA,state)
            for ji,pt in enumerate(vjoints):
                p=abs(math.sin(t*3+ji*0.4))
                cv2.circle(overlay,tuple(pt),int(8+p*5),O,1,cv2.LINE_AA)
                cv2.circle(overlay,tuple(pt),4,Y,-1,cv2.LINE_AA)

        # ── MODE 5: NEURAL VOID LATTICE ── purple plasma expanding rings ──────
        elif mode == 5:
            P=(140,10,120); V=(220,60,200)
            draw_polylines_deployed(overlay,vn,False,P,2,cv2.LINE_AA,state)
            if (dist<18).any(): draw_polylines_deployed(overlay,vn[dist<18],False,V,1,cv2.LINE_AA,state)
            for ji,pt in enumerate(vjoints):
                cv2.circle(overlay,tuple(pt),6,V,-1,cv2.LINE_AA)
                for ring in range(2):
                    phase=(t*2.2+ring*0.4+ji*0.3)%1.0
                    rad=max(1,int(phase*28))
                    alpha=int(200*(1.0-phase))
                    cv2.circle(overlay,tuple(pt),rad,(alpha//2,0,alpha),1,cv2.LINE_AA)


def render_ui_panels(image, overlay, h, w, t, face_detected, face_min_y, state):

    if state.show_side_panels:
        draw_side_panel_left(image, overlay, h, w, t, state)
        draw_side_panel_right(image, overlay, h, w, t, state)
        # Use cached stats (already fetched in draw_side_panel_left) instead of calling psutil again
        stats = get_system_stats(t, state)
        cv2.putText(overlay, f"SYS.MEM: {int(stats.get('mem_pct', 0))}% [OK]", (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (80, 255, 130), 1, cv2.LINE_AA)
        cv2.putText(overlay, f"CPU.FREQ: {int(stats.get('cpu', 0))}%", (30, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 200, 100), 1, cv2.LINE_AA)

    draw_bottom_bar(image, overlay, h, w, t, face_detected)
    if not face_detected and state.show_side_panels:
        draw_top_bar(overlay, h, w, t, state)

    if face_detected and state.show_side_panels:
        draw_top_bar(overlay, h, w, t, state, face_min_y)


def apply_post_processing(image, overlay, t, state):
    h, w, c = image.shape

    # Call allocate_buffers once dynamically if resolution changed/first run
    state.allocate_buffers(w, h, c)

    # === ZERO-ALLOCATION BLOOM (Native NumPy) ===
    quarter_w, quarter_h = w // 4, h // 4
    if state.frame_count % 2 == 0:
        # Prevent accidental reallocation by strictly checking shape bounds
        if getattr(state.small_cache, 'shape', (0,0))[:2] != (quarter_h, quarter_w):
            state.small_cache = np.zeros((quarter_h, quarter_w, c), dtype=np.uint8)
        if getattr(state.bloom_up_cache, 'shape', (0,0))[:2] != (h, w):
            state.bloom_up_cache = np.zeros((h, w, c), dtype=np.uint8)
            
        cv2.resize(overlay, (quarter_w, quarter_h), dst=state.small_cache, interpolation=cv2.INTER_LINEAR)
        cv2.GaussianBlur(state.small_cache, (9, 9), 0, dst=state.small_cache)
        cv2.resize(state.small_cache, (w, h), dst=state.bloom_up_cache, interpolation=cv2.INTER_LINEAR)

    # Combine directly into existing memory buffers
    cv2.addWeighted(overlay, 1.0, state.bloom_up_cache, 1.0, 0, dst=overlay)
    cv2.addWeighted(image, 0.55, overlay, 1.0, 0, dst=image)

    # === VIGNETTE (Native NumPy) ===
    cv2.multiply(image, state.vignette_cache, dst=image, dtype=cv2.CV_8U)

    return image


def draw_full_hud(image, face_landmarks_list, hand_landmarks_list, hand_handedness_list, pose_landmarks_list, t, state, sct_instance, target_card_img=None):
    
    h, w, c = image.shape
    
    # === MEMORY OPTIMIZATION: Recycle the same array instead of allocating 3MB every frame ===
    if getattr(state, 'bloom_cache', None) is None or getattr(state.bloom_cache, 'shape', None) != (h, w, c):
        state.bloom_cache = np.zeros((h, w, c), dtype=np.uint8)
    else:
        state.bloom_cache.fill(0)
        
    overlay = state.bloom_cache
    
    face_detected = len(face_landmarks_list) > 0
    if face_detected:
        state.last_face_landmarks = face_landmarks_list

    if state.is_deploying:
        elapsed = t - state.deploy_start_time
        progress = min(1.0, elapsed / 0.8)
        state.deploy_y = int(h * progress)
        if progress >= 1.0:
            state.is_deploying = False
            state.suit_up_complete = True
            state.deploy_y = h
            
    elif state.is_retracting:
        elapsed = t - state.deploy_start_time
        progress = min(1.0, elapsed / 0.8)
        # Sweep upwards from h to 0
        state.deploy_y = int(h * (1.0 - progress))
        if progress >= 1.0:
            state.is_retracting = False
            state.suit_up_complete = False
            state.deploy_y = 0

    if state.last_face_time == 0 or t - state.last_face_time > FACE_PERSIST_SECONDS:
        state.last_face_time = t
    state.last_face_time = t
    
    active_face_list = face_landmarks_list
    if not face_detected and state.last_face_landmarks is not None and (t - state.last_face_time) < FACE_PERSIST_SECONDS:
        active_face_list = state.last_face_landmarks
        face_detected = True

    face_min_y = None
    if state.suit_up_complete or state.is_deploying:
        if face_detected and state.tracking_mode in [0, 1, 2]:
            face_min_y = render_face_tracking(overlay, active_face_list, w, h, t, state)
            
        if hand_landmarks_list and state.tracking_mode in [0, 1, 3]:
            render_hand_tracking(overlay, hand_landmarks_list, hand_handedness_list, w, h, t, state)
            
            if target_card_img is not None:

                for hand_landmarks in hand_landmarks_list:
                    render_palm_hologram(overlay, hand_landmarks.landmark, w, h, t, target_card_img)

            
        if pose_landmarks_list and state.tracking_mode in [0, 4]:
            render_pose_tracking(overlay, pose_landmarks_list, active_face_list, hand_landmarks_list, w, h, t, state)
            
        if state.is_deploying or state.is_retracting:
            # Erase all tracking below the deployment line so it sweeps down gracefully
            overlay[state.deploy_y:, :] = 0
            
            # Draw a cool scanning laser line at the deployment edge
            edge_y = state.deploy_y
            if 0 < edge_y < h:
                cv2.line(overlay, (0, edge_y), (w, edge_y), (255, 255, 255), 1, cv2.LINE_AA)
                cv2.line(overlay, (0, max(0, edge_y-2)), (w, max(0, edge_y-2)), (255, 200, 50), 3, cv2.LINE_AA)
        
    if state.show_side_panels:
        draw_radar_sweep(overlay, h, t)
        


    
    # Nanotech Suit Up Button
    CYAN_DIM = (140, 140, 0)
    CYAN_BRIGHT = (255, 255, 0)
    nano_x, nano_y, nano_w, nano_h = w - 180, 20, 160, 50
    nano_color = CYAN_DIM
    
    for hl in hand_landmarks_list:
        ix, iy = int(hl[8].x * w), int(hl[8].y * h)

        
        # Check Nanotech Deploy
        if nano_x < ix < nano_x + nano_w and nano_y < iy < nano_y + nano_h:
            nano_color = CYAN_BRIGHT
            if t - state.deploy_start_time > 2.0:
                if state.suit_up_complete:
                    state.is_retracting = True
                    state.suit_up_complete = True # Stay true until retraction finishes
                    state.deploy_y = h
                else:
                    state.is_deploying = True
                    state.suit_up_complete = False
                    state.deploy_y = 0
                state.deploy_start_time = t
    
    
    cv2.rectangle(overlay, (nano_x, nano_y), (nano_x + nano_w, nano_y + nano_h), nano_color, 2)
    cv2.putText(overlay, f"SUIT UP [N]", (nano_x + 15, nano_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, nano_color, 1, cv2.LINE_AA)

    # Global Pull-Apart Gesture (Index fingers touch, then pull apart)
    best_pair = None
    if len(hand_landmarks_list) >= 2:
        touching_now = False
        pulled_apart_now = False
        max_dist = -1
        
        for h1, h2 in itertools.combinations(hand_landmarks_list, 2):
            idx0 = h1[8]
            idx1 = h2[8]
            idx_dist = math.hypot(idx0.x - idx1.x, idx0.y - idx1.y)
            
            if idx_dist < 0.05:
                touching_now = True
            elif state.fingers_touching and idx_dist > 0.15:
                pulled_apart_now = True
                
            # Track the pair that is furthest apart to act as the screen anchors
            if idx_dist > max_dist:
                max_dist = idx_dist
                best_pair = (h1, h2)
                
        if touching_now:
            state.fingers_touching = True
        elif pulled_apart_now:
            state.is_screen_open = not state.is_screen_open
            state.fingers_touching = False

    render_ui_panels(image, overlay, h, w, t, face_detected, face_min_y, state)

    
    if state.is_file_network_open:
        render_file_network(overlay, w, h, t, state)
        
    if state.is_screen_open and best_pair is not None:
        xs = [int(best_pair[0][8].x * w), int(best_pair[1][8].x * w)]
        ys = [int(best_pair[0][8].y * h), int(best_pair[1][8].y * h)]
        tx, bx = min(xs), max(xs)
        ty, by = min(ys), max(ys)
        
        target_box = np.array([tx, ty, bx, by], dtype=np.float32)
        if getattr(state, 'last_box_coords', None) is None:
            state.last_box_coords = target_box
        else:
            state.last_box_coords += (target_box - state.last_box_coords) * 0.40
        stx, sty, sbx, sby = [int(v) for v in state.last_box_coords]
        
        if sbx - stx > 50 and sby - sty > 50:
            if getattr(state, 'invis_mode', False) and getattr(state, 'bg_frame', None) is not None:
                try:
                    img_h, img_w = image.shape[:2]
                    bg_h, bg_w = state.bg_frame.shape[:2]
                    
                    max_y = min(img_h, bg_h)
                    max_x = min(img_w, bg_w)
                    
                    ty_safe = max(0, min(sty, max_y))
                    by_safe = max(0, min(sby, max_y))
                    tx_safe = max(0, min(stx, max_x))
                    bx_safe = max(0, min(sbx, max_x))
                    
                    if by_safe > ty_safe and bx_safe > tx_safe:
                        image[ty_safe:by_safe, tx_safe:bx_safe] = state.bg_frame[ty_safe:by_safe, tx_safe:bx_safe]
                except ValueError as e:
                    logging.error(f"Bg frame blend failed: {e}", exc_info=True)
            else:
                if sct_instance is None:
                    pass
                    sct_instance = mss.mss()
    
                try:
                    bbox = get_browser_rect()
                    if bbox is None:
                        bbox = sct_instance.monitors[0]
                    sct_img     = sct_instance.grab(bbox)
                    desktop_img = np.array(sct_img)
                    desktop_img = cv2.cvtColor(desktop_img, cv2.COLOR_BGRA2BGR)
    
                    bh2, bw2 = desktop_img.shape[:2]
                    if state.crop_rect is not None:
                        cx2, cy2, cw2, ch2 = state.crop_rect
                        cx2 = max(0, min(cx2, bw2-1)); cy2 = max(0, min(cy2, bh2-1))
                        cw2 = max(1, min(cw2, bw2-cx2)); ch2 = max(1, min(ch2, bh2-cy2))
                        desktop_img = desktop_img[cy2:cy2+ch2, cx2:cx2+cw2]
    
                    sw2 = sbx - stx; sh2 = sby - sty
                    if sw2 > 0 and sh2 > 0:
                        desktop_resized = cv2.resize(desktop_img, (sw2, sh2))
                        image[sty:sby, stx:sbx] = desktop_resized  # crisp, unmodified content
    
                except Exception as e:
                    logging.error(f"Screen capture failed: {e}", exc_info=True)
                    sw2 = sbx - stx; sh2 = sby - sty

            # Clear overlay ROI so bloom doesn't bleed onto content
            overlay[sty:sby, stx:sbx] = 0

            # ── PALETTE ──────────────────────────────────────────────────────
            CORE  = (255, 255, 255)    # pure white  (BGR)
            MID   = (255, 220,  0)     # electric cyan
            DARK  = (120,  50,  0)     # dark blue halo
            WHITE = (255, 255, 255)

            sw2 = sbx - stx;  sh2 = sby - sty

            # ── 3-LAYER NEON GLOW BORDER ─────────────────────────────────────
            # Layer 1: fat dark halo (creates the "bloom" look)
            cv2.rectangle(overlay, (stx-4, sty-4), (sbx+4, sby+4), DARK,  5)
            # Layer 2: mid glow
            cv2.rectangle(overlay, (stx-2, sty-2), (sbx+2, sby+2), MID,   2)
            # Layer 3: bright core line — animated pulse
            pulse = 0.7 + 0.3 * abs(math.sin(t * 1.8))
            core_col = (int(CORE[0]*pulse), int(CORE[1]*pulse), int(CORE[2]*pulse))
            cv2.rectangle(overlay, (stx, sty), (sbx, sby), core_col, 1)

            # ── CORNER ANCHORS ────────────────────────────────────────────────
            arm = 32   # how long the L arms are
            for (cx4, cy4), (dx4, dy4) in [
                ((stx, sty), ( 1,  1)),
                ((sbx, sty), (-1,  1)),
                ((stx, sby), ( 1, -1)),
                ((sbx, sby), (-1, -1)),
            ]:
                # Thick dark shadow arm
                cv2.line(overlay, (cx4, cy4), (cx4 + arm*dx4, cy4), DARK,  4, cv2.LINE_AA)
                cv2.line(overlay, (cx4, cy4), (cx4, cy4 + arm*dy4), DARK,  4, cv2.LINE_AA)
                # Bright core arm
                cv2.line(overlay, (cx4, cy4), (cx4 + arm*dx4, cy4), CORE,  2, cv2.LINE_AA)
                cv2.line(overlay, (cx4, cy4), (cx4, cy4 + arm*dy4), CORE,  2, cv2.LINE_AA)
                # Glowing corner dot
                cv2.circle(overlay, (cx4, cy4), 5, DARK,  -1, cv2.LINE_AA)
                cv2.circle(overlay, (cx4, cy4), 3, CORE,  -1, cv2.LINE_AA)
                cv2.circle(overlay, (cx4, cy4), 6, MID,    1, cv2.LINE_AA)

            # ── MINIMAL SCAN LINE (barely visible, slow) ──────────────────────
            scan_y = sty + int((t * 80) % max(1, sh2))
            if sty < scan_y < sby:
                cv2.line(overlay, (stx+1, scan_y), (sbx-1, scan_y), (20, 60, 40), 1)

            # ── FLOATING LABEL CHIP (top-left, outside the frame) ────────────
            label   = "HOLO DISPLAY"
            lsz     = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)[0]
            lx, ly  = stx + 2, sty - 10
            # Dark pill background
            cv2.rectangle(overlay, (lx-5, ly-13), (lx+lsz[0]+6, ly+3), (4, 6, 10), -1)
            cv2.rectangle(overlay, (lx-5, ly-13), (lx+lsz[0]+6, ly+3), MID, 1)
            cv2.putText(overlay, label, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.38, CORE, 1, cv2.LINE_AA)

            # ── BOTTOM-RIGHT TIMESTAMP CHIP ───────────────────────────────────
            # We already imported time as _t at the top
            ts  = _t.strftime("%H:%M:%S")
            tsz = cv2.getTextSize(ts, cv2.FONT_HERSHEY_SIMPLEX, 0.32, 1)[0]
            rx  = sbx - tsz[0] - 10;  ry = sby + 14
            cv2.putText(overlay, ts, (rx, ry), cv2.FONT_HERSHEY_SIMPLEX, 0.32, MID, 1, cv2.LINE_AA)
    else:
        state.last_box_coords = None


    render_voice_ui(overlay, w, h, t, state)
    return apply_post_processing(image, overlay, t, state)

def render_voice_ui(overlay, w, h, t, state):
    if not getattr(state, "is_listening", False) and not getattr(state, "last_voice_text", ""):
        return
        
    HOLO = (255, 200, 0) # Cyan (BGR)
    DIM = (150, 100, 0)
    
    # Position in the safe-zone for 16:9 crop (center offset)
    cx = w // 2
    cy = (h // 2) + 250
    
    if state.is_listening:
        # Listening waveform animation
        cv2.putText(overlay, "LISTENING...", (cx - 35, cy - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.35, DIM, 1, cv2.LINE_AA)
        num_bars = 12
        for i in range(num_bars):
            bx = cx - (num_bars * 4) + (i * 8)
            pulse = abs(math.sin(t * 8 + i * 0.5))
            bh = int(4 + pulse * 12)
            cv2.line(overlay, (bx, cy + 10), (bx, cy + 10 - bh), HOLO if pulse > 0.7 else DIM, 2)
            
    if state.last_voice_text:
        # Show last recognized text
        age = time.time() - getattr(state, "last_voice_time", 0)
        if age < 4.0:
            alpha = max(0, 1.0 - (age / 4.0))
            col = (int(HOLO[0] * alpha), int(HOLO[1] * alpha), int(HOLO[2] * alpha))
            txt = f"> {state.last_voice_text.upper()}"
            tsz = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0]
            cv2.putText(overlay, txt, (cx - tsz[0]//2, cy + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
            cv2.putText(overlay, txt, (cx - tsz[0]//2, cy + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)






def init_file_network(state):
    state.file_nodes = []
    state.grabbed_node_index = None
    
    files = [f for f in os.listdir('.') if os.path.isfile(f)][:30]
    if not files: files = ["node_alpha.dat", "core_sys.dll", "config.bin"]
    
    # Core Node
    state.file_nodes.append({
        'name': "SYSTEM CORE",
        'x': 0.0, 'y': -150.0, 'z': 300.0,
        'vx': 0.0, 'vy': 0.0, 'vz': 0.0,
        'radius': 45.0,
        'is_core': True,
        't_offset': random.random() * 10
    })
    
    # Orbiting File Nodes
    for f in files:
        angle = random.random() * math.pi * 2
        radius = random.uniform(200, 400)
        y = random.uniform(-250, 150)
        x = math.cos(angle) * radius
        z = 300.0 + math.sin(angle) * radius
        state.file_nodes.append({
            'name': f.upper(),
            'x': x, 'y': y, 'z': z,
            'vx': 0.0, 'vy': 0.0, 'vz': 0.0,
            'radius': random.uniform(8.0, 14.0),
            'is_core': False,
            't_offset': random.random() * 10
        })

def update_file_network_physics(state, hand_landmarks_list, t, w, h):
    if not state.file_nodes: return
    
    # 1. Rotate whole cloud slowly
    center_z = 300.0
    rot_speed = 0.002
    cos_r = math.cos(rot_speed)
    sin_r = math.sin(rot_speed)
    
    for i, node in enumerate(state.file_nodes):
        if i == state.grabbed_node_index: continue
        dx = node['x']
        dz = node['z'] - center_z
        node['x'] = dx * cos_r - dz * sin_r
        node['z'] = dz * cos_r + dx * sin_r + center_z
        
        # gentle floating bob
        node['y'] += math.sin(t * 2.0 + node['t_offset']) * 0.5

    # 2. Hand interaction (grab nearest)
    # Get all index/thumb tips
    hand_pts = []
    if hand_landmarks_list:
        for hl in hand_landmarks_list:
            if len(hl) > 8:
                hx8, hy8 = int(hl[8].x * w), int(hl[8].y * h)
                hx4, hy4 = int(hl[4].x * w), int(hl[4].y * h)
                hand_pts.append((hx8, hy8, hx4, hy4))
                
    if hand_pts:
        hx8, hy8, hx4, hy4 = hand_pts[0]
        pinch_dist = math.hypot(hx8 - hx4, hy8 - hy4)
        is_pinching = pinch_dist < 40
        
        # approximate 3D hand pos (Z is hard, so we assume fixed Z for grabbing)
        hand_x = (hx8 - w/2) * 1.5
        hand_y = (hy8 - h/2) * 1.5
        hand_z = 250.0
        
        if is_pinching:
            if state.grabbed_node_index is None:
                # Find closest ungrabbed node to the screen-space hand
                best_idx, best_dist = None, float('inf')
                for i, node in enumerate(state.file_nodes):
                    # rough 2D projection for grab check
                    focal = 600.0
                    if node['z'] > 0:
                        sx = (node['x'] * focal / node['z']) + w/2
                        sy = (node['y'] * focal / node['z']) + h/2
                        dist = math.hypot(sx - hx8, sy - hy8)
                        if dist < 80 and dist < best_dist:
                            best_dist = dist; best_idx = i
                if best_idx is not None:
                    state.grabbed_node_index = best_idx
            else:
                # move grabbed node
                node = state.file_nodes[state.grabbed_node_index]
                node['x'] += (hand_x - node['x']) * 0.3
                node['y'] += (hand_y - node['y']) * 0.3
                node['z'] = hand_z
        else:
            if state.grabbed_node_index is not None:
                # Toss physics
                state.grabbed_node_index = None

def render_file_network(overlay, w, h, t, state):
    if not state.file_nodes: return
    
    focal = 600.0
    cx, cy = w/2, h/2
    
    # Project to 2D
    proj_nodes = []
    for i, n in enumerate(state.file_nodes):
        if n['z'] < 10: continue
        sx = int((n['x'] * focal / n['z']) + cx)
        sy = int((n['y'] * focal / n['z']) + cy)
        srad = max(1, int(n['radius'] * focal / n['z']))
        proj_nodes.append({'i': i, 'sx': sx, 'sy': sy, 'srad': srad, 'z': n['z'], 'node': n})
        
    proj_nodes.sort(key=lambda item: item['z'], reverse=True) # draw back to front
    
    CORE_COL = (255, 255, 255)
    MID_COL  = (255, 220,  0) # Cyan
    DARK_COL = (120,  50,  0) # Deep Blue
    
    # Draw faint web connections
    for i in range(len(proj_nodes)):
        for j in range(i+1, min(i+15, len(proj_nodes))):
            n1, n2 = proj_nodes[i], proj_nodes[j]
            dist3d = math.hypot(math.hypot(n1['node']['x']-n2['node']['x'], n1['node']['y']-n2['node']['y']), n1['node']['z']-n2['node']['z'])
            if dist3d < 180:
                alpha = max(0, min(255, int(255 - dist3d*1.4)))
                if alpha > 10:
                    cv2.line(overlay, (n1['sx'], n1['sy']), (n2['sx'], n2['sy']), (int(DARK_COL[0]*alpha/255), int(DARK_COL[1]*alpha/255), int(DARK_COL[2]*alpha/255)), 1, cv2.LINE_AA)
    
    # Draw nodes
    for pn in proj_nodes:
        sx, sy, srad = pn['sx'], pn['sy'], pn['srad']
        n = pn['node']
        
        # 3D rotation matrix for the rings
        phase = t * 2.0 + n['t_offset']
        
        if n['is_core']:
            cv2.circle(overlay, (sx, sy), srad + 8, DARK_COL, -1, cv2.LINE_AA)
            cv2.circle(overlay, (sx, sy), srad + 3, MID_COL, 1, cv2.LINE_AA)
            cv2.circle(overlay, (sx, sy), srad, CORE_COL, -1, cv2.LINE_AA)
            # Rings
            cv2.ellipse(overlay, (sx, sy), (int(srad*1.8), int(srad*0.5)), phase*30, 0, 360, MID_COL, 2, cv2.LINE_AA)
            cv2.ellipse(overlay, (sx, sy), (int(srad*1.5), int(srad*0.4)), -phase*40, 0, 360, CORE_COL, 1, cv2.LINE_AA)
            cv2.ellipse(overlay, (sx, sy), (int(srad*2.4), int(srad*2.4)), 0, phase*20, phase*20+120, MID_COL, 1, cv2.LINE_AA)
            cv2.ellipse(overlay, (sx, sy), (int(srad*2.4), int(srad*2.4)), 0, phase*20+180, phase*20+300, MID_COL, 1, cv2.LINE_AA)
        else:
            is_grabbed = (pn['i'] == state.grabbed_node_index)
            col = CORE_COL if is_grabbed else MID_COL
            bg  = DARK_COL
            
            # Glow
            cv2.circle(overlay, (sx, sy), srad + 4, bg, -1, cv2.LINE_AA)
            # Core
            cv2.circle(overlay, (sx, sy), srad, col, -1 if is_grabbed else 1, cv2.LINE_AA)
            if not is_grabbed:
                cv2.circle(overlay, (sx, sy), 2, CORE_COL, -1, cv2.LINE_AA)
            
            # Label
            if pn['z'] < 500 or is_grabbed:
                txt = n['name']
                tsz = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.25, 1)[0]
                tx, ty = sx + srad + 4, sy + 3
                cv2.putText(overlay, txt, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.25, CORE_COL if is_grabbed else MID_COL, 1, cv2.LINE_AA)

