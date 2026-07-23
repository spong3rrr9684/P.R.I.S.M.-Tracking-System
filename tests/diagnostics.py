import sys, os, platform
import time

def run_diagnostics():
    print("=== P.R.I.S.M. SYSTEM DIAGNOSTIC REPORT ===")
    print(f"Timestamp: {time.ctime()}")
    print(f"OS: {platform.system()} {platform.release()} ({platform.version()})")
    print(f"Python Version: {sys.version.split(' ')[0]}")
    
    print("\n--- 1. DEPENDENCY CHECK ---")
    deps = ["cv2", "numpy", "mediapipe", "mss", "psutil", "pyvirtualcam", "speech_recognition", "pyttsx3", "sounddevice"]
    for d in deps:
        try:
            __import__(d)
            print(f"[OK] {d} is installed.")
        except ImportError:
            print(f"[ERROR] {d} is MISSING!")
            
    print("\n--- 2. AI MODEL CHECK ---")
    models = ["face_landmarker.task", "hand_landmarker.task", "pose_landmarker_full.task", "selfie_segmenter.task"]
    for m in models:
        if os.path.exists(m):
            size = os.path.getsize(m) / (1024*1024)
            print(f"[OK] {m} found ({size:.1f} MB)")
        else:
            print(f"[ERROR] {m} is MISSING! (Did they run the auto-installer?)")
            
    print("\n--- 3. HARDWARE CAMERA CHECK ---")
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print(f"[OK] Default Webcam is working! Resolution: {frame.shape[1]}x{frame.shape[0]}")
            else:
                print("[WARNING] Webcam opened but failed to capture a frame (Might be used by another app like Zoom).")
            cap.release()
        else:
            print("[ERROR] No webcam detected on USB index 0!")
    except Exception as e:
        print(f"[ERROR] Webcam hardware check failed: {e}")

    print("\n--- 4. P.R.I.S.M RUNTIME CHECK ---")
    try:
        import psutil
        prism_running = False
        for p in psutil.process_iter(['name', 'cmdline']):
            try:
                if p.info['name'] in ['python.exe', 'pythonw.exe']:
                    if p.info['cmdline'] and 'main.py' in ' '.join(p.info['cmdline']):
                        print(f"[OK] P.R.I.S.M. is currently running in memory (PID: {p.pid})")
                        prism_running = True
            except: pass
        if not prism_running:
            print("[INFO] P.R.I.S.M. 'main.py' is not currently running.")
    except Exception as e:
        print(f"[ERROR] Could not check active processes: {e}")

    print("=== END OF REPORT ===")

if __name__ == "__main__":
    run_diagnostics()
