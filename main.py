import cv2
cv2.setUseOptimized(True)
cv2.ocl.setUseOpenCL(False)
print(f"OpenCL Enabled: {cv2.ocl.haveOpenCL()}")
import time
import os

# Crucial CPU optimizations: stop C++ backends from aggressively over-threading
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
cv2.setNumThreads(1)

import mss
import queue
import logging
from voice_assistant import VoiceAssistant
from renderer import update_file_network_physics
import threading
try:
    from websocket_server import HUDWebSocketServer
except ImportError:
    class HUDWebSocketServer:
        def start(self): pass
        def stop(self): pass

# Configure basic logging for error tracking
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PRIMARY_CAM = 2
FALLBACK_CAM = 0

class ThreadedCamera:
    def __init__(self, primary=PRIMARY_CAM, fallback=FALLBACK_CAM, target_width=1280, target_height=720):
        self.cap = cv2.VideoCapture(primary)
        self.cam_index = primary
        if not self.cap.isOpened():
            logger.warning(f"Primary camera {primary} not found. Falling back to {fallback}.")
            self.cap = cv2.VideoCapture(fallback)
            self.cam_index = fallback
            
        if self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, target_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, target_height)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            self.ret, self.frame = self.cap.read()
            self.running = True
            self.thread = threading.Thread(target=self.update, daemon=True)
            self.thread.start()
        else:
            self.ret, self.frame = False, None
            self.running = False

    def update(self):
        while self.running:
            if self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret:
                    self.ret = ret
                    self.frame = frame
            else:
                time.sleep(0.01)

    def read(self):
        return self.ret, (self.frame.copy() if self.frame is not None else None)

    def isOpened(self):
        return self.cap.isOpened() and self.running

    def release(self):
        self.running = False
        if hasattr(self, 'thread'):
            self.thread.join(timeout=1.0)
        self.source_mgr.release()

class ThreadedScreenCapture:
    def __init__(self, bbox=None):
        import mss
        self.sct = mss.mss()
        self.bbox = bbox if bbox else self.sct.monitors[0]
        self.running = True
        self.ret = False
        self.frame = None
        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()
        
    def update(self):
        import time
        import numpy as np
        while self.running:
            try:
                sct_img = self.sct.grab(self.bbox)
                self.frame = np.array(sct_img)[:, :, :3]
                self.ret = True
            except Exception:
                self.ret = False
            time.sleep(1/30.0)
            
    def read(self):
        return self.ret, (self.frame.copy() if self.frame is not None else None)
        
    def isOpened(self):
        return self.running
        
    def release(self):
        self.running = False
        if hasattr(self, 'thread'):
            self.thread.join(timeout=1.0)

class VideoSourceManager:
    def __init__(self, primary_cam_idx):
        self.source = ThreadedCamera(primary=primary_cam_idx)
        self.mode = "WEBCAM"
        self.window_list = []
        self.window_idx = 0
        
    def switch_to_desktop(self):
        self.source.release()
        self.source = ThreadedScreenCapture(bbox=None)
        self.mode = "DESKTOP"
        print("[SOURCE] Switched to Full Desktop")
        
    def switch_to_window(self):
        self.source.release()
        from utils import get_open_windows
        self.window_list = get_open_windows()
        if len(self.window_list) > 0:
            self.window_idx = (self.window_idx + 1) % len(self.window_list)
            win = self.window_list[self.window_idx]
            self.source = ThreadedScreenCapture(bbox=win['bbox'])
            self.mode = f"WINDOW: {win['title'][:15]}"
            print(f"[SOURCE] Switched to Window: {win['title']}")
        else:
            self.switch_to_desktop()
            
    def switch_to_webcam(self, idx=0):
        self.source.release()
        self.source = ThreadedCamera(primary=idx)
        self.mode = "WEBCAM"
        print("[SOURCE] Switched to Webcam")
        
    def read(self):
        return self.source.read()
        
    def isOpened(self):
        return self.source.isOpened()
        
    def release(self):
        self.source.release()

from state import HUDState
from tracker import start_inference_thread
from renderer import draw_full_hud
from utils import apply_tracking_smoothing, get_target_window_rect
import pyvirtualcam
import numpy as np

def main():
    print('=' * 50)
    print('  P.R.I.S.M TRACKING SYSTEM v4.0 (Modular)')
    print("==================================================")
    print("  [c] Cycle camera  |  [m] Change Face Mode  |  [h] Toggle HUD Data")
    print("  [w] Cycle Window  |  [s] Set Screen Crop   |  [f] Full Screen ")
    print("  [t] Toggle Target |  [q] Quit")
    print("==================================================")
    state = HUDState()
    frame_q = queue.Queue(maxsize=1)
    result_q = queue.Queue(maxsize=1)
    sct_instance = mss.mss()

    class DummyCam:
        def __init__(self, *args, **kwargs): pass
        def __enter__(self): 
            self.device = "DISABLED (OBS Virtual Camera not found)"
            return self
        def __exit__(self, *args): pass
        def send(self, frame): pass
        def sleep_until_next_frame(self): time.sleep(1/30.0)

    face_det, hand_det, pose_det, infer_thread = start_inference_thread(state, frame_q, result_q)

    source_mgr = VideoSourceManager(primary_cam_idx=2)
    if not source_mgr.isOpened():
        print("[CRITICAL ERROR] No webcam detected! Please plug in a camera.")
        return
    cam_index = 2

    while True:
        raw_width, raw_height = 1280, 720
        for _ in range(50):
            success, test_frame = source_mgr.read()
            if success:
                raw_height, raw_width = test_frame.shape[:2]
                break
            time.sleep(0.05)
            
        # VIRTUAL CAMERA INITIALIZATION:
        # Dynamic Resolution based on actual camera output
        canvas_w = raw_width
        canvas_h = raw_height

        # Initialize pyvirtualcam with RGB pixel format
        try:
            cam_context = pyvirtualcam.Camera(width=canvas_w, height=canvas_h, fps=30, fmt=pyvirtualcam.PixelFormat.RGB)
        except Exception as e:
            print(f"\n[WARNING] Could not start virtual camera: {e}")
            print("[WARNING] P.R.I.S.M will run locally, but will not stream to Discord/Zoom.\n")
            cam_context = DummyCam()

        with cam_context as cam:
            print(f'Virtual camera active: {cam.device}. Resolution: {canvas_w}x{canvas_h}')
            

            voice_agent = VoiceAssistant(state)
            ws_server = HUDWebSocketServer()
            ws_server.start()
            voice_agent.start()
            
            cycle_requested = False
            
            # Last known AI results — reuse if AI thread isn't done yet
            last_face   = []
            last_hands  = []
            last_handed = []
            last_pose   = []

            while source_mgr.isOpened():
                try:
                    success, image = source_mgr.read()
                except Exception as e:
                    logger.error(f"Failed to read from camera: {e}", exc_info=True)
                    success = False

                if not success:
                    time.sleep(0.005)
                    continue

                t = time.time()
                state.frame_count += 1
                state.scan_y = (state.scan_y + 4) % max(1, image.shape[0])

                # Downscale for async ML inference
                h, w = image.shape[:2]
                small_frame = cv2.resize(image, (int(w * state.ai_scale_factor), int(h * state.ai_scale_factor)), interpolation=cv2.INTER_LINEAR)
                # Convert BGR to RGB here to save tracker thread time
                small_rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

                # Push freshest downscaled frame to AI thread atomically
                try:
                    frame_q.put_nowait((small_rgb, t))
                except queue.Full:
                    try:
                        frame_q.get_nowait() # pop oldest
                        frame_q.put_nowait((small_rgb, t))
                    except queue.Empty:
                        pass
                    except queue.Full:
                        pass

                # Check if AI has a new result — non-blocking, always
                is_fresh = False
                try:
                    res = result_q.get_nowait()
                    _discard, ready_t, last_face, last_hands, last_handed, last_pose = res
                    is_fresh = True
                except queue.Empty:
                    ready_t = t
                    
                # Always render at full camera FPS — never wait for AI
                # Use current camera frame for display; AI landmarks are still valid even if 1-2 frames behind
                render_frame = image
                apply_tracking_smoothing(last_face, last_hands, last_handed, last_pose, state, is_fresh)

                render_start = time.perf_counter()
                if state.is_file_network_open:

                    update_file_network_physics(state, last_hands, ready_t, 1280, 720)
                hud_image = draw_full_hud(render_frame, last_face, last_hands, last_handed, last_pose, ready_t, state, sct_instance)
                state.render_latency_ms = (time.perf_counter() - render_start) * 1000.0

                if state.frame_count % 120 == 0:
                    state.last_profiler_str = f"Render:{state.render_latency_ms:.0f}ms AI:{state.inference_latency_ms:.0f}ms"

                # Flip for selfie-view
                flipped_image = cv2.flip(hud_image, 1)

                if cam is not None:
                    rgb_feed = cv2.cvtColor(flipped_image, cv2.COLOR_BGR2RGB)
                    cam_w, cam_h = cam.width, cam.height

                    # === GESTURE CALIBRATION MODE ===
                    if state.gesture_calibration_mode and len(last_hands) >= 2:
                        dx = abs(last_hands[0][8].x - last_hands[1][8].x)
                        dy = abs(last_hands[0][8].y - last_hands[1][8].y)
                        state.stretch_factor = max(0.2, min(5.0, dx * 3.0))
                        state.zoom_factor    = max(0.2, min(5.0, dy * 3.0))

                    # === DYNAMIC CALIBRATION CROP ===
                    result_feed = np.zeros((cam_h, cam_w, 3), dtype=np.uint8)
                    fh, fw = rgb_feed.shape[:2]
                    base_scale = cam_w / float(fw)
                    scale_x    = base_scale * state.zoom_factor
                    scale_y    = base_scale * state.zoom_factor * state.stretch_factor
                    new_w = max(1, int(fw * scale_x))
                    new_h = max(1, int(fh * scale_y))
                    scaled = cv2.resize(rgb_feed, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                    start_x  = (cam_w - new_w) // 2
                    start_y  = (cam_h - new_h) // 2
                    crop_x1  = max(0, -start_x);  crop_y1  = max(0, -start_y)
                    crop_x2  = min(new_w, crop_x1 + cam_w); crop_y2 = min(new_h, crop_y1 + cam_h)
                    paste_x1 = max(0, start_x);   paste_y1 = max(0, start_y)
                    paste_x2 = paste_x1 + (crop_x2 - crop_x1)
                    paste_y2 = paste_y1 + (crop_y2 - crop_y1)
                    result_feed[paste_y1:paste_y2, paste_x1:paste_x2] = scaled[crop_y1:crop_y2, crop_x1:crop_x2]

                    # Send to virtual cam without blocking display — skip if cam is behind
                    try:
                        cam.send(result_feed)
                        cam.sleep_until_next_frame()
                    except RuntimeError as e:
                        logger.error(f"Virtual camera pipe broken: {e}. Reinitializing...")
                        break
                    except Exception as e:
                        logger.error(f"Virtual camera send failed/dropped: {e}", exc_info=True)

                cv2.imshow('P.R.I.S.M Tracking', flipped_image)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    return # Exit the entire program
                elif key == ord('m'):
                    state.face_mesh_mode = (state.face_mesh_mode + 1) % 6

                elif key == ord('h'):
                    state.show_side_panels = not state.show_side_panels
                elif key == ord('v'):
                    state.is_file_network_open = not state.is_file_network_open
                    if state.is_file_network_open:
                        init_file_network(state)
                    state.show_side_panels = not state.show_side_panels
                elif key == ord('t'):
                    state.tracking_mode = (state.tracking_mode + 1) % 5

                elif key == ord('x'):
                    state.hologram_active = True
                    state.hologram_pos = np.array([canvas_w / 2.0, canvas_h / 2.0])
                    print("Hologram Spawned!")
                elif key == ord('n'):
                    if state.suit_up_complete:
                        state.is_retracting = True
                        state.suit_up_complete = True
                        state.deploy_y = state.last_h if hasattr(state, 'last_h') else 1080
                    else:
                        state.is_deploying = True
                        state.suit_up_complete = False
                        state.deploy_y = 0
                    state.deploy_start_time = time.time()
                elif key == ord('='):
                    state.zoom_factor += 0.05
                    print(f"[CALIBRATION] Zoom In: {state.zoom_factor:.2f}")
                elif key == ord('-'):
                    state.zoom_factor = max(0.1, state.zoom_factor - 0.05)
                    print(f"[CALIBRATION] Zoom Out: {state.zoom_factor:.2f}")
                elif key == ord(']'):
                    state.stretch_factor += 0.05
                    print(f"[CALIBRATION] Taller: {state.stretch_factor:.2f}")
                elif key == ord('['):
                    state.stretch_factor = max(0.1, state.stretch_factor - 0.05)
                    print(f"[CALIBRATION] Wider: {state.stretch_factor:.2f}")
                elif key == ord('b'):
                    state.gesture_calibration_mode = not state.gesture_calibration_mode
                    print(f"[CALIBRATION] Gesture Calibration Mode: {'ON' if state.gesture_calibration_mode else 'OFF'}")
                elif key == ord('i'):
                    state.bg_frame = image.copy()
                    state.invis_mode = not state.invis_mode
                    print(f"[INVISIBILITY CLOAK] {'ACTIVATED' if state.invis_mode else 'DEACTIVATED'}")

                # Process Voice Commands
                if hasattr(state, "voice_command_queue") and state.voice_command_queue:
                    while state.voice_command_queue:
                        cmd = state.voice_command_queue.pop(0)
                        if cmd.startswith("set_mode_"):
                            mode_idx = int(cmd.split("_")[-1])
                            state.face_mesh_mode = mode_idx
                        elif cmd.startswith("track_"):
                            track_idx = int(cmd.split("_")[-1])
                            state.tracking_mode = track_idx
                        elif cmd == "toggle_hud":
                            state.show_side_panels = not state.show_side_panels
                        elif cmd == "suit_up":
                            if state.suit_up_complete:
                                state.is_retracting = True
                                state.suit_up_complete = True
                                state.deploy_y = getattr(state, 'last_h', 1080)
                            else:
                                state.is_deploying = True
                                state.suit_up_complete = False
                                state.deploy_y = 0
                            state.deploy_start_time = time.time()
                elif key == ord('c'):
                    ws_server.stop()
                    source_mgr.release()
                    cam_index = (cam_index + 1) % 10
                    cap = cv2.VideoCapture(cam_index)
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 720)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1280)
                    cap.set(cv2.CAP_PROP_FPS, 60)
                    cycle_requested = True
                    break # Break inner loop to recreate virtual camera
                elif key == ord('f'):
                    state.crop_rect = None
                    print("[CALIBRATION] Restored to Full Window")
                elif key == ord('w'):
                    state.target_window_index += 1
                    bbox, title, count = get_target_window_rect(state.target_window_index)
                    if bbox is None or state.target_window_index >= count:
                        state.target_window_index = -1
                        print("\n[TARGET WINDOW] Switched to: Full Desktop Monitor")
                    else:
                        print(f"\n[TARGET WINDOW] Switched to ({state.target_window_index+1}/{count}): {title}")
                elif key == ord('s'):
                    bbox, title, count = get_target_window_rect(state.target_window_index)
                    
                    if bbox is None or state.target_window_index >= count:
                        state.target_window_index = -1
                        bbox = sct_instance.monitors[0]
                        print("\n[SCREEN CROP] Targeting: Full Desktop Monitor")
                    else:
                        print(f"\n[SCREEN CROP] Targeting Window ({state.target_window_index+1}/{count}): {title}")

                    sct_img = sct_instance.grab(bbox)
                    calib_img = cv2.cvtColor(np.array(sct_img), cv2.COLOR_BGRA2BGR)
                    r = cv2.selectROI('Select Video Feed (Press ENTER)', calib_img, False, False)
                    cv2.destroyWindow('Select Video Feed (Press ENTER)')
                    if r[2] > 0 and r[3] > 0:
                        state.crop_rect = r
                        
            if not cycle_requested:
                break # If the camera loop ended normally, exit out of outer loop

if __name__ == '__main__':
    main()
