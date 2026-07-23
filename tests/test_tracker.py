import os, sys, time, traceback
import numpy as np

def run_tracker_test():
    print("=== P.R.I.S.M. AI TRACKING ENGINE DIAGNOSTIC ===")
    
    print("\n[1/3] Loading MediaPipe Framework...")
    try:
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
        from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode
        print("[OK] MediaPipe library loaded successfully.")
    except ImportError as e:
        print(f"[FAIL] Missing mediapipe! Run: pip install mediapipe")
        return

    print("\n[2/3] Initializing Neural Network Models...")
    script_dir = os.getcwd()
    models = {
        "Face": ('face_landmarker.task', vision.FaceLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=os.path.join(script_dir, 'face_landmarker.task')),
            num_faces=1, running_mode=VisionTaskRunningMode.IMAGE, min_face_detection_confidence=0.5)),
        
        "Hand": ('hand_landmarker.task', vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=os.path.join(script_dir, 'hand_landmarker.task')),
            num_hands=2, running_mode=VisionTaskRunningMode.IMAGE, min_hand_detection_confidence=0.5)),
            
        "Pose": ('pose_landmarker_full.task', vision.PoseLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=os.path.join(script_dir, 'pose_landmarker_full.task')),
            num_poses=1, running_mode=VisionTaskRunningMode.IMAGE, min_pose_detection_confidence=0.5))
    }

    detectors = {}
    for name, (filename, options) in models.items():
        if not os.path.exists(filename):
            print(f"[FAIL] {filename} is MISSING from your folder!")
            print("       -> FIX: Double check your downloaded files or run the RUN_PRISM.bat installer again.")
            return
            
        if os.path.getsize(filename) == 0:
            print(f"[FAIL] {filename} is 0 bytes (CORRUPTED DOWNLOAD)!")
            print("       -> FIX: Delete the file and download it again.")
            return

        try:
            if name == "Face":
                detectors[name] = vision.FaceLandmarker.create_from_options(options)
            elif name == "Hand":
                detectors[name] = vision.HandLandmarker.create_from_options(options)
            elif name == "Pose":
                detectors[name] = vision.PoseLandmarker.create_from_options(options)
            print(f"[OK] {name} Model Engine Started.")
        except Exception as e:
            print(f"[CRITICAL FAIL] Failed to initialize {name} Engine!")
            print(f"Error Details: {e}")
            if "Unsupported" in str(e) or "AVX" in str(e):
                print("       -> FIX: Your CPU might be too old to run Google's AI models natively (Missing AVX instructions).")
            return

    print("\n[3/3] Simulating CPU Inference Stress Test...")
    try:
        # Create a blank black image (RGB)
        dummy_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=dummy_frame)
        
        t1 = time.time()
        detectors["Face"].detect(mp_img)
        detectors["Hand"].detect(mp_img)
        detectors["Pose"].detect(mp_img)
        t2 = time.time()
        
        print(f"[OK] Inference test passed successfully in {((t2-t1)*1000):.1f}ms!")
    except Exception as e:
        print(f"[FAIL] The AI crashed during active inference!")
        print(f"Error Details: {e}")
        return

    print("\n=== SUCCESS! THE TRACKER IS 100% HEALTHY ===")
    print("If the script is still crashing for you, the problem is likely your webcam or OBS Virtual Camera.")

if __name__ == "__main__":
    run_tracker_test()
