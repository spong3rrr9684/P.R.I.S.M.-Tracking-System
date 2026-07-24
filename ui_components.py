import cv2
import numpy as np
import math
import random
import time as _t
import time
from config import *
from utils import get_system_stats

def draw_polylines_deployed(overlay, lines, is_closed, color, thickness, line_type, state):
    if len(lines) == 0: return
    if isinstance(lines, list):
        lines = np.array(lines, dtype=np.int32)
    if lines.ndim != 3: 
        cv2.polylines(overlay, lines, is_closed, color, thickness, line_type)
        return
    
    if state.is_deploying:
        max_y = lines[:, :, 1].max(axis=1)
        valid_mask = max_y < state.deploy_y
        valid_lines = lines[valid_mask]
        
        if len(valid_lines) > 0:
            # Sparking Edge Effect
            edge_mask = (state.deploy_y - valid_lines[:, :, 1].max(axis=1)) < 50
            if np.any(edge_mask):
                edge_lines = valid_lines[edge_mask]
                drop_mask = np.random.rand(len(edge_lines)) > 0.5
                surviving = edge_lines[drop_mask]
                if len(surviving) > 0:
                    WHITE_HOT = (255, 255, 255)
                    cv2.polylines(overlay, surviving, is_closed, WHITE_HOT, 2, cv2.LINE_AA)
            
            # Base mesh
            cv2.polylines(overlay, valid_lines, is_closed, color, thickness, line_type)
    else:
        cv2.polylines(overlay, lines, is_closed, color, thickness, line_type)


def draw_rotating_reticle(overlay, cx, cy, radius, angle, color, thickness=1):
    """Draw a premium multi-ring animated targeting reticle."""
    # === Outer dashed ring (fast spin) ===
    num_dashes = 32
    for i in range(num_dashes):
        if i % 4 == 0:
            continue
        a1 = angle + (i * 360 / num_dashes)
        a2 = angle + ((i + 1) * 360 / num_dashes)
        cv2.ellipse(overlay, (cx, cy), (radius, radius), 0, a1, a2, CYAN, thickness + 1, cv2.LINE_AA)

    # === Middle counter-spin ring (slower, gold) ===
    mid_r = int(radius * 0.78)
    for i in range(16):
        if i % 3 == 0:
            continue
        a1 = -angle * 0.7 + (i * 360 / 16)
        a2 = -angle * 0.7 + ((i + 1) * 360 / 16)
        cv2.ellipse(overlay, (cx, cy), (mid_r, mid_r), 0, a1, a2, GOLD, 1, cv2.LINE_AA)

    # === Inner solid ring (very slow) ===
    inner_r = int(radius * 0.55)
    cv2.circle(overlay, (cx, cy), inner_r, CYAN_DIM, 1, cv2.LINE_AA)

    # === Cardinal tick marks on outer ring ===
    for i in range(4):
        a = angle + (i * 90)
        a_rad = math.radians(a)
        p_outer = (int(cx + radius * math.cos(a_rad)), int(cy + radius * math.sin(a_rad)))
        p_inner = (int(cx + (radius - 14) * math.cos(a_rad)), int(cy + (radius - 14) * math.sin(a_rad)))
        cv2.line(overlay, p_inner, p_outer, CYAN_BRIGHT, 2, cv2.LINE_AA)
        # Small diamond at each cardinal
        d = 2
        cv2.circle(overlay, p_outer, d, WHITE_HOT, -1, cv2.LINE_AA)

    # === Fine tick marks every 15° ===
    for i in range(24):
        a = angle + (i * 15)
        a_rad = math.radians(a)
        p_outer = (int(cx + radius * math.cos(a_rad)), int(cy + radius * math.sin(a_rad)))
        p_inner = (int(cx + (radius - 5) * math.cos(a_rad)), int(cy + (radius - 5) * math.sin(a_rad)))
        cv2.line(overlay, p_inner, p_outer, CYAN_DIM, 1, cv2.LINE_AA)



def draw_crosshair(overlay, cx, cy, size, color):
    """Draw a precision crosshair at the center of the face."""
    gap = 8
    # Horizontal lines with gap
    cv2.line(overlay, (cx - size, cy), (cx - gap, cy), color, 1, cv2.LINE_AA)
    cv2.line(overlay, (cx + gap, cy), (cx + size, cy), color, 1, cv2.LINE_AA)
    # Vertical lines with gap
    cv2.line(overlay, (cx, cy - size), (cx, cy - gap), color, 1, cv2.LINE_AA)
    cv2.line(overlay, (cx, cy + gap), (cx, cy + size), color, 1, cv2.LINE_AA)
    # Small center diamond
    d = 3
    pts = np.array([(cx, cy-d), (cx+d, cy), (cx, cy+d), (cx-d, cy)], np.int32)
    cv2.polylines(overlay, [pts], True, color, 1, cv2.LINE_AA)



def draw_hud_brackets(overlay, min_x, min_y, max_x, max_y, color, thickness=2):
    """Premium corner brackets with diamonds, double lines, and animated tick marks."""
    length = 40
    inner_len = 12
    bright = WHITE_HOT
    dim = CYAN_DIM

    corners = [
        ((min_x, min_y), (1, 1)),
        ((max_x, min_y), (-1, 1)),
        ((min_x, max_y), (1, -1)),
        ((max_x, max_y), (-1, -1)),
    ]
    for (cx, cy), (dx, dy) in corners:
        # Outer thick bracket arm
        cv2.line(overlay, (cx, cy), (cx + length * dx, cy), color, thickness, cv2.LINE_AA)
        cv2.line(overlay, (cx, cy), (cx, cy + length * dy), color, thickness, cv2.LINE_AA)
        # Inner short accent line (offset inward 2px for a double-line feel)
        cv2.line(overlay, (cx + 2*dx, cy + 2*dy), (cx + inner_len * dx, cy + 2*dy), dim, 1, cv2.LINE_AA)
        cv2.line(overlay, (cx + 2*dx, cy + 2*dy), (cx + 2*dx, cy + inner_len * dy), dim, 1, cv2.LINE_AA)
        # Corner diamond dot
        d = 3
        diamond = np.array([(cx, cy - d), (cx + d, cy), (cx, cy + d), (cx - d, cy)], np.int32)
        cv2.fillConvexPoly(overlay, diamond, bright)
        # Outer glow dot
        cv2.circle(overlay, (cx, cy), 5, dim, 1, cv2.LINE_AA)

    # Dashed connecting lines (mid-edge markers)
    mid_x = (min_x + max_x) // 2
    mid_y = (min_y + max_y) // 2
    tick = 8
    for x_pos in [min_x, max_x]:
        cv2.line(overlay, (x_pos, mid_y - tick), (x_pos, mid_y + tick), dim, 1, cv2.LINE_AA)
    for y_pos in [min_y, max_y]:
        cv2.line(overlay, (mid_x - tick, y_pos), (mid_x + tick, y_pos), dim, 1, cv2.LINE_AA)



def draw_top_bar(overlay, h, w, t, state, face_min_y=None):
    """Draw a premium animated top HUD visor bar."""
    if face_min_y is not None:
        bar_top = max(180, face_min_y - 44)
    else:
        bar_top = int(h * 0.25)

    bar_h = bar_top + 32

    # Solid fill
    cv2.rectangle(overlay, (0, bar_top), (w, bar_h), (4, 4, 8), -1)
    # Bottom accent double lines
    cv2.line(overlay, (0, bar_h), (w, bar_h), CYAN, 2)
    cv2.line(overlay, (0, bar_h - 3), (w, bar_h - 3), CYAN_DARK, 1)
    # Top border thin
    cv2.line(overlay, (0, bar_top), (w, bar_top), CYAN_DARK, 1)

    # Left: animated bracket + title
    blink = int(t * 2) % 2 == 0
    prefix = "[◆]" if blink else "[◇]"
    cv2.putText(overlay, prefix, (10, bar_top + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, GOLD_BRIGHT if blink else GOLD, 1, cv2.LINE_AA)
    cv2.putText(overlay, "P.R.I.S.M  FACE.TRACK", (42, bar_top + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, CYAN, 1, cv2.LINE_AA)

    # Center divider tick
    cv2.line(overlay, (w // 2 - 1, bar_top + 5), (w // 2 - 1, bar_h - 5), CYAN_DARK, 1)

    # Right: animated frame counter
    cv2.putText(overlay, f"FRAME {state.frame_count:07d}", (w - 185, bar_top + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.42, GOLD_DIM, 1, cv2.LINE_AA)





def draw_mcu_compass(overlay, cx, cy, t, scale=1.0):
    # Advanced MCU Concentric Compass Rings - Cyan + Gold dual-tone
    ring_configs = [
        (35, 1.0,  1, 0,  CYAN_DIM),
        (45, -0.5, 2, 36, GOLD_DIM),
        (55, 0.8,  1, 72, CYAN_DIM),
        (70, -0.2, 1, 12, GOLD),
    ]
    for r_base, speed, thickness, dashes, ring_color in ring_configs:
        r = int(r_base * scale)
        angle = t * 20 * speed
        if dashes > 0:
            for i in range(dashes):
                if i % 2 == 0: continue
                a1 = angle + (i * 360 / dashes)
                a2 = angle + ((i + 1) * 360 / dashes)
                cv2.ellipse(overlay, (cx, cy), (r, r), 0, a1, a2, ring_color, thickness, cv2.LINE_AA)
        else:
            cv2.circle(overlay, (cx, cy), r, ring_color, thickness, cv2.LINE_AA)



def draw_flight_ladder(overlay, cx, cy, t, pitch, yaw, scale=1.0):
    # Draw a 3D-feeling pitch/yaw ladder in center
    offset_x = int(math.sin(yaw) * 100 * scale)
    offset_y = int(math.sin(pitch) * 100 * scale)
    center = (cx + offset_x, cy + offset_y)
    
    cv2.drawMarker(overlay, center, CYAN_BRIGHT, cv2.MARKER_CROSS, int(20*scale), 1, cv2.LINE_AA)
    
    for i in range(-3, 4):
        if i == 0: continue
        ly = center[1] + int(i * 40 * scale)
        width = int((40 if abs(i) < 2 else 20) * scale)
        cv2.line(overlay, (center[0] - width, ly), (center[0] - width//2, ly), CYAN_DIM, 1, cv2.LINE_AA)
        cv2.line(overlay, (center[0] + width//2, ly), (center[0] + width, ly), CYAN_DIM, 1, cv2.LINE_AA)



def draw_radar_sweep(overlay, h, t):
    cx, cy = 100, h - 130
    r = 60
    cv2.circle(overlay, (cx, cy), r, BLUE_DARK, 1, cv2.LINE_AA)
    cv2.circle(overlay, (cx, cy), r//2, BLUE_DARK, 1, cv2.LINE_AA)
    cv2.line(overlay, (cx-r, cy), (cx+r, cy), BLUE_DARK, 1, cv2.LINE_AA)
    cv2.line(overlay, (cx, cy-r), (cx, cy+r), BLUE_DARK, 1, cv2.LINE_AA)
    
    angle = (t * 90) % 360
    cv2.ellipse(overlay, (cx, cy), (r, r), 0, angle, angle+40, BLUE_DIM, -1)
    
    # Radar blips
    random.seed(int(t))
    for _ in range(3):
        bx = cx + random.randint(-r+10, r-10)
        by = cy + random.randint(-r+10, r-10)
        cv2.circle(overlay, (bx, by), 2, GREEN_OK, -1, cv2.LINE_AA)




def draw_scan_line(overlay, y, w, h):
    """Draw a fast horizontal scanning line effect."""
    if 0 <= y < h:
        cv2.line(overlay, (0, y), (w, y), BLUE_BRIGHT, 2)
    # Just a couple fade lines instead of 25
    for offset in (2, 4, 7):
        fade_y = y - offset
        if 0 <= fade_y < h:
            cv2.line(overlay, (0, fade_y), (w, fade_y), BLUE_DARK, 1)



def draw_side_panel_left(image, overlay, h, w, t, state):
    """Draw a premium glass-style system metrics panel on the left."""
    px = 8
    py = int(h * 0.38)
    lh = 18
    pw = 168

    # Fast dark glass panel (no blur — same look, fraction of the cost)
    x1, y1 = max(0, px - 2), max(0, py - 6)
    x2, y2 = min(w, px + pw), min(h, py + lh * 12 + 10)
    if x2 > x1 and y2 > y1:
        roi = image[y1:y2, x1:x2]
        cv2.addWeighted(roi, 0.25, roi, 0, 0, dst=roi)
        image[y1:y2, x1:x2] = roi

    # Outer border
    cv2.rectangle(overlay, (px - 2, py - 6), (px + pw, py + lh * 12 + 10), CYAN_DARK, 1)
    # Top accent bar
    cv2.rectangle(overlay, (px - 2, py - 6), (px + pw, py + 2), CYAN, -1)

    # Header
    cv2.putText(overlay, "SYS.METRICS", (px + 4, py + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (4, 4, 8), 1, cv2.LINE_AA)
    cv2.line(overlay, (px, py + 20), (px + pw - 4, py + 20), CYAN_DARK, 1)

    stats = get_system_stats(t, state)

    # FPS calculation
    try:
        _last_hud_t
    except NameError:
        _last_hud_t = t
        _current_fps = 30
    delta = max(t - _last_hud_t, 0.001)
    _current_fps = _current_fps * 0.9 + (1.0 / delta) * 0.1
    _last_hud_t = t

    cpu = int(stats.get('cpu', 0))
    ram = stats.get('mem_pct', 0)
    fps_val = int(_current_fps)

    rows = [
        ("CPU LOAD", f"{cpu:>3d}%",   cpu / 100.0,  ORANGE_WARN if cpu > 80 else GREEN_OK),
        ("RAM USED", f"{ram:.1f}%",   ram / 100.0,  ORANGE_WARN if ram > 80 else CYAN),
        ("RAM",      f"{stats.get('mem_used_gb', 0):.1f}/{stats.get('mem_total_gb', 0):.0f}GB", None, CYAN_DIM),
        ("DISK",     f"{stats.get('disk_pct', 0):.0f}%",  None, CYAN_DIM),
        ("NET ↑",    f"{stats.get('net_sent', 0)/(1024**2):.0f}MB", None, CYAN_DIM),
        ("NET ↓",    f"{stats.get('net_recv', 0)/(1024**2):.0f}MB", None, CYAN_DIM),
        ("PROCS",    f"{stats.get('num_procs', 0)} RUN", None, CYAN_DIM),
        ("LANDMARKS","478 OK",       None, CYAN_DIM),
        ("FPS",      f"{fps_val:>3d}",        min(fps_val / 60.0, 1.0), GREEN_OK if fps_val >= 25 else ORANGE_WARN),
        ("STATUS",   "ONLINE",       None, GREEN_OK),
    ]

    bar_w = 55
    for i, (label, val, pct, col) in enumerate(rows):
        y = py + 34 + i * lh
        cv2.putText(overlay, label, (px + 2, y), cv2.FONT_HERSHEY_SIMPLEX, 0.32, CYAN_DIM, 1, cv2.LINE_AA)
        if pct is not None:
            # Draw mini progress bar
            bar_x = px + 75
            cv2.rectangle(overlay, (bar_x, y - 8), (bar_x + bar_w, y - 2), CYAN_DARK, -1)
            filled = int(bar_w * min(pct, 1.0))
            if filled > 0:
                cv2.rectangle(overlay, (bar_x, y - 8), (bar_x + filled, y - 2), col, -1)
            cv2.putText(overlay, val, (bar_x + bar_w + 4, y), cv2.FONT_HERSHEY_SIMPLEX, 0.32, col, 1, cv2.LINE_AA)
        else:
            cv2.putText(overlay, val, (px + 80, y), cv2.FONT_HERSHEY_SIMPLEX, 0.32, col, 1, cv2.LINE_AA)



def draw_side_panel_right(image, overlay, h, w, t, state):
    """Draw a premium hex data stream panel on the right."""
    pw = 148
    px = w - pw - 8
    py = int(h * 0.38)
    lh = 15

    # Fast dark glass panel
    x1, y1 = max(0, px - 4), max(0, py - 6)
    x2, y2 = min(w, px + pw), min(h, py + lh * 12 + 10)
    if x2 > x1 and y2 > y1:
        roi = image[y1:y2, x1:x2]
        cv2.addWeighted(roi, 0.25, roi, 0, 0, dst=roi)
        image[y1:y2, x1:x2] = roi

    cv2.rectangle(overlay, (px - 4, py - 6), (px + pw, py + lh * 12 + 10), CYAN_DARK, 1)
    # Top accent bar
    cv2.rectangle(overlay, (px - 4, py - 6), (px + pw, py + 2), GOLD_DIM, -1)

    # Header
    cv2.putText(overlay, "DATA.STREAM", (px + 4, py + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (4, 4, 8), 1, cv2.LINE_AA)
    cv2.line(overlay, (px, py + 20), (px + pw - 4, py + 20), GOLD_DIM, 1)

    # Scrolling real processes with highlights
    procs = state.real_process_list if hasattr(state, 'real_process_list') and state.real_process_list else ["WAITING_SYS..."]
    offset = int(t * 6) % max(1, len(procs))
    for i in range(min(12, len(procs))):
        idx = (offset + i) % len(procs)
        yy = py + 34 + i * lh
        proc_name = str(procs[idx])[:15].upper()
        prefix = f"[{(offset + i) % 100:02d}]"
        # Highlight newest line in gold
        col = GOLD_BRIGHT if i == 0 else (GOLD_DIM if i < 3 else CYAN_DARK)
        cv2.putText(overlay, f"{prefix} {proc_name}", (px + 2, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.32, col, 1, cv2.LINE_AA)
        # Animated cursor on active line
        if i == 0 and int(t * 4) % 2 == 0:
            cx_cur = px + pw - 10
            cv2.line(overlay, (cx_cur, yy - 9), (cx_cur, yy + 2), GOLD_BRIGHT, 2)



def draw_heartbeat_line(overlay, x, y, w_line, h_line, t, state):
    """Draw a live CPU usage monitor."""
    # Seed buffer if empty
    if not hasattr(state, 'cpu_history_buffer') or not state.cpu_history_buffer:
        state.cpu_history_buffer = [0] * 100
        
    # Shift data every other frame
    if state.frame_count % 2 == 0:
        state.cpu_history_buffer.pop(0)
        current_cpu = state.sys_stats.get('cpu', 0) if hasattr(state, 'sys_stats') else 0
        state.cpu_history_buffer.append(current_cpu)
    
    # Draw border
    cv2.rectangle(overlay, (x - 2, y - 2), (x + w_line + 2, y + h_line + 2), BLUE_DARK, 1, cv2.LINE_AA)
    cv2.putText(overlay, "VITALS", (x + 2, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.3, GREEN_OK, 1, cv2.LINE_AA)
    
    points = []
    buffer = state.cpu_history_buffer
    for i in range(min(len(buffer), w_line)):
        px = x + int(i * w_line / len(buffer))
        py = y + h_line - int(buffer[i] * h_line / 100)
        points.append((px, py))
    
    if len(points) > 1:
        for i in range(len(points) - 1):
            cv2.line(overlay, points[i], points[i + 1], GREEN_OK, 1, cv2.LINE_AA)



def draw_bottom_bar(image, overlay, h, w, t, face_detected):
    """Draw a cinematic premium bottom status bar."""
    bar_h = 38
    bar_y = h - bar_h
    
    # Fast dark glass bar
    if h > bar_y and w > 0:
        roi = image[bar_y:h, 0:w]
        cv2.addWeighted(roi, 0.25, roi, 0, 0, dst=roi)
        image[bar_y:h, 0:w] = roi

    # Top accent lines (double border)
    cv2.line(overlay, (0, bar_y), (w, bar_y), CYAN, 2)
    cv2.line(overlay, (0, bar_y + 3), (w, bar_y + 3), CYAN_DARK, 1)

    # Left: animated status indicator
    status_color = GREEN_OK if face_detected else ORANGE_WARN
    status_text = "TARGET ACQUIRED" if face_detected else "SCANNING..."
    pulse = int(abs(math.sin(t * 4)) * 4) + 3
    cv2.circle(overlay, (18, bar_y + bar_h // 2), pulse, status_color, -1, cv2.LINE_AA)
    cv2.circle(overlay, (18, bar_y + bar_h // 2), pulse + 4, status_color, 1, cv2.LINE_AA)
    cv2.putText(overlay, status_text, (32, bar_y + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.45, status_color, 1, cv2.LINE_AA)

    # Divider
    cv2.line(overlay, (200, bar_y + 8), (200, bar_y + bar_h - 8), CYAN_DARK, 1)

    # Center: proc code
    code = f"PROC #{int(t * 100) % 9999:04d} │ FACE.AI v3.7.1"
    tsz = cv2.getTextSize(code, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)[0]
    cv2.putText(overlay, code, (w // 2 - tsz[0] // 2, bar_y + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.38, CYAN_DIM, 1, cv2.LINE_AA)

    # Right divider + timestamp
    cv2.line(overlay, (w - 170, bar_y + 8), (w - 170, bar_y + bar_h - 8), CYAN_DARK, 1)
    ts = time.strftime("%H:%M:%S", time.localtime())
    cv2.putText(overlay, f"▶ {ts}", (w - 158, bar_y + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.42, GOLD, 1, cv2.LINE_AA)





def generate_hexagon(cx, cy, size):
    pass

def draw_landmarks_manually(overlay, face_landmarks, connections, color=(0, 255, 0), thickness=1):
    h, w, c = overlay.shape
    for connection in connections:
        start_idx = getattr(connection, 'start', None)
        end_idx = getattr(connection, 'end', None)
        if start_idx is None:
            start_idx = connection[0]
            end_idx = connection[1]
        if start_idx < len(face_landmarks) and end_idx < len(face_landmarks):
            start_landmark = face_landmarks[start_idx]
            end_landmark = face_landmarks[end_idx]
            start_point = (int(start_landmark.x * w), int(start_landmark.y * h))
            end_point = (int(end_landmark.x * w), int(end_landmark.y * h))
            cv2.line(overlay, start_point, end_point, color, thickness, cv2.LINE_AA)






