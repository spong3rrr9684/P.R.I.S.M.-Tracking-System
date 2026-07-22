import cv2
import mediapipe as mp
import threading
import queue
import time
import os
import numpy as np
import traceback
import logging
import concurrent.futures

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode

logger = logging.getLogger(__name__)

def inference_loop(face_detector, hand_detector, pose_detector, state, frame_q, result_q):
    inference_ts_counter = 0
    frame_counter        = 0

    # Persist last results so skipped frames can reuse them
    last_face   = []
    last_hand   = []
    last_handed = []
    last_pose   = []
    
    missed_face = 0
    missed_hand = 0
    missed_pose = 0

    # Pre-allocate the thread pool to avoid per-frame overhead
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        while state.inference_running:
            try:
                frame, cam_ts = frame_q.get(timeout=0.5)
            except queue.Empty:
                continue
    
            try:
                inference_start  = time.perf_counter()
                frame_counter   += 1
    
                # frame is already downscaled and RGB from main.py
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(frame))
    
                # Use actual time elapsed so MediaPipe's internal motion tracking works correctly
                current_ts = int(cam_ts * 1000)
                if current_ts <= inference_ts_counter:
                    current_ts = inference_ts_counter + 1
                inference_ts_counter = current_ts
    
                # --- PARALLEL DISPATCH ---
                face_future = None
                hand_future = None
                pose_future = None
    
                # Face: every frame (cheap, most important)
                if state.tracking_mode in [0, 1, 2]:
                    face_future = executor.submit(face_detector.detect_for_video, mp_img, inference_ts_counter)
    
                # Hands: every frame for lowest latency
                if state.tracking_mode in [0, 1, 3]:
                    hand_future = executor.submit(hand_detector.detect_for_video, mp_img, inference_ts_counter)
    
                # Pose: every 2nd frame — body moves slowly, 15fps update is fine
                if state.tracking_mode in [0, 4]:
                    if frame_counter % 2 == 0:
                        pose_future = executor.submit(pose_detector.detect_for_video, mp_img, inference_ts_counter)
    
                # --- PARALLEL COLLECTION ---
                if face_future:
                    try:
                        r = face_future.result(timeout=1.0)
                        if r.face_landmarks:
                            last_face = r.face_landmarks
                            missed_face = 0
                        else:
                            missed_face += 1
                            if missed_face > 10:
                                last_face = []
                                
                    except concurrent.futures.TimeoutError:
                        logger.warning("Face detection timeout")
                    except concurrent.futures.CancelledError:
                        pass
                    except Exception as e:
                        logger.error(f"Face tracking crashed: {e}", exc_info=True)
                    finally:
                        del face_future

                if hand_future:
                    try:
                        r = hand_future.result(timeout=1.0)
                        if r.hand_landmarks:
                            last_hand   = r.hand_landmarks
                            last_handed = r.handedness
                            missed_hand = 0
                        else:
                            missed_hand += 1
                            if missed_hand > 10:
                                last_hand = []
                                last_handed = []
                    except concurrent.futures.TimeoutError:
                        logger.warning("Hand detection timeout")
                    except concurrent.futures.CancelledError:
                        pass
                    except Exception as e:
                        logger.error(f"Hand tracking crashed: {e}", exc_info=True)
                    finally:
                        del hand_future

                if pose_future:
                    try:
                        r = pose_future.result(timeout=1.0)
                        if r.pose_landmarks:
                            last_pose = r.pose_landmarks
                            missed_pose = 0
                        else:
                            missed_pose += 1
                            if missed_pose > 5:
                                last_pose = []
                                
                    except concurrent.futures.TimeoutError:
                        logger.warning("Pose detection timeout")
                    except concurrent.futures.CancelledError:
                        pass
                    except Exception as e:
                        logger.error(f"Pose tracking crashed: {e}", exc_info=True)
                    finally:
                        del pose_future
    
                del mp_img

                state.inference_latency_ms = (time.perf_counter() - inference_start) * 1000.0

                # Push result (drop stale if queue is full)
                if result_q.full():
                    try: result_q.get_nowait()
                    except queue.Empty: pass
                result_q.put((frame, cam_ts, last_face, last_hand, last_handed, last_pose))

            except Exception as e:
                logger.error(f"[AI THREAD ERROR] {e}")
                logger.error(traceback.format_exc())

def start_inference_thread(state, frame_q, result_q):
    script_dir = os.path.dirname(os.path.abspath(__file__))

    options = vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=os.path.join(script_dir, 'face_landmarker.task')),
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
        num_faces=1,
        running_mode=VisionTaskRunningMode.VIDEO,
        min_face_detection_confidence=0.6,
        min_face_presence_confidence=0.6,
        min_tracking_confidence=0.6)

    hand_options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=os.path.join(script_dir, 'hand_landmarker.task')),
        num_hands=2,
        running_mode=VisionTaskRunningMode.VIDEO,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.6)

    pose_options = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=os.path.join(script_dir, 'pose_landmarker_lite.task')),
        num_poses=1,
        running_mode=VisionTaskRunningMode.VIDEO,
        min_pose_detection_confidence=0.6,
        min_pose_presence_confidence=0.6,
        min_tracking_confidence=0.6)

    try:
        face_detector = vision.FaceLandmarker.create_from_options(options)
        hand_detector = vision.HandLandmarker.create_from_options(hand_options)
        pose_detector = vision.PoseLandmarker.create_from_options(pose_options)
    except Exception as e:
        logger.error(f"Failed to load MediaPipe models: {e}")
        raise e

    state.inference_running = True
    infer_thread = threading.Thread(
        target=inference_loop, 
        args=(face_detector, hand_detector, pose_detector, state, frame_q, result_q), 
        daemon=True)
    infer_thread.start()
    
    return face_detector, hand_detector, pose_detector, infer_thread
