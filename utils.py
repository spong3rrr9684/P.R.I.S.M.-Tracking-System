import threading
import time
import ctypes
from ctypes import wintypes
import psutil
import cv2
import math
import numpy as np

from config import *

def get_system_stats(t, state):
    """Get real system stats, cached to avoid calling psutil every frame."""
    if t - state.sys_stats_time > SYS_STATS_INTERVAL:
        state.sys_stats_time = t
        cpu = psutil.cpu_percent(interval=0)
        mem = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        net = psutil.net_io_counters()
        state.sys_stats = {
            'cpu': cpu,
            'mem_pct': mem.percent,
            'mem_used_gb': mem.used / (1024**3),
            'mem_total_gb': mem.total / (1024**3),
            'disk_pct': disk.percent,
            'net_sent': net.bytes_sent,
            'net_recv': net.bytes_recv,
            'num_procs': len(psutil.pids()),
        }
    
    if t - state.process_update_time > 2.0:
        state.process_update_time = t
        
        def update_procs_bg(s):
            try:
                # Grab all process names uniquely
                procs = [p.info['name'] for p in psutil.process_iter(['name']) if p.info['name']]
                unique_procs = list(set(procs))
                unique_procs = [p for p in unique_procs if not p.startswith("svchost") and not p.startswith("System")]
                s.real_process_list = sorted(unique_procs)[:50]
            except psutil.Error:
                pass
                
        threading.Thread(target=update_procs_bg, args=(state,), daemon=True).start()
            
    return state.sys_stats

def smooth(key, target_val, state, speed=0.5):
    """Filters out webcam jitter by smoothly interpolating coordinates."""
    if key not in state.smooth_cache:
        state.smooth_cache[key] = float(target_val)
    else:
        state.smooth_cache[key] += (target_val - state.smooth_cache[key]) * speed
    return state.smooth_cache[key]

class OneEuroFilter:
    def __init__(self, t0, x0, min_cutoff=0.004, beta=0.7, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev = x0
        self.dx_prev = np.zeros_like(x0)
        self.t_prev = t0

    def smoothing_factor(self, t_e, cutoff):
        r = 2 * math.pi * cutoff * t_e
        return r / (r + 1)

    def __call__(self, t, x):
        t_e = t - self.t_prev
        if t_e <= 0:
            return x

        a_d = self.smoothing_factor(t_e, self.d_cutoff)
        dx = (x - self.x_prev) / t_e
        dx_hat = a_d * dx + (1.0 - a_d) * self.dx_prev

        # Use the norm of the derivative to scale the cutoff dynamically
        cutoff = self.min_cutoff + self.beta * np.linalg.norm(dx_hat, axis=-1, keepdims=True)
        a = self.smoothing_factor(t_e, cutoff)
        x_hat = a * x + (1.0 - a) * self.x_prev

        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t
        return x_hat

def apply_tracking_smoothing(face_landmarks_list, hand_landmarks_list, hand_handedness_list, pose_landmarks_list, state, is_fresh=True):
    t = time.perf_counter()
    
    # === VECTORIZED FACE SMOOTHING (1 Euro Filter) ===
    for face_idx, face_landmarks in enumerate(face_landmarks_list):
        raw_pts = state.face_raw_cache.get(face_idx)
        if raw_pts is None or raw_pts.shape[0] != len(face_landmarks):
            raw_pts = np.zeros((len(face_landmarks), 3), dtype=np.float32)
            state.face_raw_cache[face_idx] = raw_pts
            
        if is_fresh:
            for i, lm in enumerate(face_landmarks):
                raw_pts[i, 0] = lm.x
                raw_pts[i, 1] = lm.y
                raw_pts[i, 2] = lm.z
                
        face_width = raw_pts[:, 0].max() - raw_pts[:, 0].min() if len(raw_pts) > 0 else 0.1
        # Dynamic tuning: beta is balanced to reduce lag but prevent raw jitter
        dyn_beta = 60.0 + max(0.0, (0.15 - face_width) * 50.0)
        dyn_min_cutoff = 0.004 + min(0.01, face_width * 0.05)
            
        if face_idx not in state.face_smooth_cache or state.face_smooth_cache[face_idx].x_prev.shape != raw_pts.shape:
            state.face_smooth_cache[face_idx] = OneEuroFilter(t, raw_pts, min_cutoff=dyn_min_cutoff, beta=dyn_beta)
        else:
            state.face_smooth_cache[face_idx].beta = dyn_beta
            state.face_smooth_cache[face_idx].min_cutoff = dyn_min_cutoff
            filtered_pts = state.face_smooth_cache[face_idx](t, raw_pts)
            for i, lm in enumerate(face_landmarks):
                lm.x, lm.y, lm.z = filtered_pts[i, 0], filtered_pts[i, 1], filtered_pts[i, 2]
                
    # === VECTORIZED HAND SMOOTHING (1 Euro Filter) ===
    active_hand_ids = set()
    for hand_idx, hand_landmarks in enumerate(hand_landmarks_list):
        hand_id = f"Hand_{hand_idx}"
        active_hand_ids.add(hand_id)
        
        raw_pts = state.hand_raw_caches.get(hand_id)
        if raw_pts is None or raw_pts.shape[0] != len(hand_landmarks):
            raw_pts = np.zeros((len(hand_landmarks), 3), dtype=np.float32)
            state.hand_raw_caches[hand_id] = raw_pts
            
        if is_fresh:
            for i, lm in enumerate(hand_landmarks):
                raw_pts[i, 0] = lm.x
                raw_pts[i, 1] = lm.y
                raw_pts[i, 2] = lm.z
                
        # Dynamic hand tuning
        hand_width = raw_pts[:, 0].max() - raw_pts[:, 0].min() if len(raw_pts) > 0 else 0.1
        dyn_beta = 30.0 + max(0.0, (0.1 - hand_width) * 50.0)
            
        if hand_id not in state.hand_smooth_caches or state.hand_smooth_caches[hand_id].x_prev.shape != raw_pts.shape:
            state.hand_smooth_caches[hand_id] = OneEuroFilter(t, raw_pts, min_cutoff=0.004, beta=dyn_beta)
        else:
            state.hand_smooth_caches[hand_id].beta = dyn_beta
            filtered_pts = state.hand_smooth_caches[hand_id](t, raw_pts)
            for i, lm in enumerate(hand_landmarks):
                lm.x, lm.y, lm.z = filtered_pts[i, 0], filtered_pts[i, 1], filtered_pts[i, 2]
            
    # Clean up lost hands
    for hid in list(state.hand_smooth_caches.keys()):
        if hid not in active_hand_ids:
            del state.hand_smooth_caches[hid]

    # === VECTORIZED POSE SMOOTHING (1 Euro Filter) ===
    for pose_idx, pose_landmarks in enumerate(pose_landmarks_list):
        raw_pts = state.pose_raw_cache.get(pose_idx)
        if raw_pts is None or raw_pts.shape[0] != len(pose_landmarks):
            raw_pts = np.zeros((len(pose_landmarks), 3), dtype=np.float32)
            state.pose_raw_cache[pose_idx] = raw_pts
            
        if is_fresh:
            for i, lm in enumerate(pose_landmarks):
                raw_pts[i, 0] = lm.x
                raw_pts[i, 1] = lm.y
                raw_pts[i, 2] = lm.z
            
        if pose_idx not in state.pose_smooth_cache or state.pose_smooth_cache[pose_idx].x_prev.shape != raw_pts.shape:
            state.pose_smooth_cache[pose_idx] = OneEuroFilter(t, raw_pts, min_cutoff=0.004, beta=25.0)
        else:
            filtered_pts = state.pose_smooth_cache[pose_idx](t, raw_pts)
            for i, lm in enumerate(pose_landmarks):
                lm.x, lm.y, lm.z = filtered_pts[i, 0], filtered_pts[i, 1], filtered_pts[i, 2]

def get_browser_rect(target_titles=None):
    if target_titles is None:
        target_titles = ["opera", "chrome", "edge", "firefox"]
        
    user32 = ctypes.windll.user32
    EnumWindowsProc = ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)
    found_hwnd = None
    
    def foreach_window(hwnd, lParam):
        nonlocal found_hwnd
        if user32.IsWindowVisible(hwnd):
            length = user32.GetWindowTextLengthW(hwnd)
            if length > 0:
                buf = ctypes.create_unicode_buffer(length + 1)
                user32.GetWindowTextW(hwnd, buf, length + 1)
                title = buf.value.lower()
                if any(target in title for target in target_titles):
                    # Filter out tiny/hidden browser overlays by checking size
                    rect = wintypes.RECT()
                    user32.GetWindowRect(hwnd, ctypes.byref(rect))
                    if (rect.right - rect.left) > 400 and (rect.bottom - rect.top) > 400:
                        found_hwnd = hwnd
                        return False
        return True
    
    user32.EnumWindows(EnumWindowsProc(foreach_window), 0)
    
    if found_hwnd:
        rect = wintypes.RECT()
        user32.GetWindowRect(found_hwnd, ctypes.byref(rect))
        if rect.left >= -10000 and rect.top >= -10000:
            return {"top": rect.top, "left": rect.left, "width": rect.right - rect.left, "height": rect.bottom - rect.top}
    return None
