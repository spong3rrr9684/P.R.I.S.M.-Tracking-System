import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Any, List, Dict, Optional, Tuple

@dataclass
class HUDState:
    # Tracking config
    tracking_mode: int = 0  # 0: All, 1: Face+Hands, 2: Face Only, 3: Hands Only, 4: Arms Only
    face_mesh_mode: int = 0  # 0: Thinned, 1: Full 3D, 2: Minimal, 3: Point Cloud, 4: Shield, 5: Tactical Red, 6: Faceless Void
    target_window_index: int = 0
    
    # Render config
    show_side_panels: bool = False
    crop_rect: Optional[Tuple[int, int, int, int]] = None
    
    # Interaction logic
    is_screen_open: bool = False
    invis_mode: bool = False
    bg_frame: Any = None
    is_file_network_open: bool = False
    fingers_touching: bool = False
    last_pinch_time: float = 0
    last_box_coords: Any = None
    was_pinching: bool = False
    pinch_count: int = 0
    
    # Custom calibration tweaks
    ai_scale_factor: float = 0.35
    stretch_factor: float = 1.0
    zoom_factor: float = 1.8
    gesture_calibration_mode: bool = False
    air_draw_points: List[Any] = field(default_factory=list)
    is_drawing: bool = False
    cameraman_active: bool = False
    cameraman_cx: float = 0.5
    cameraman_cy: float = 0.5
    cameraman_zoom: float = 1.0
    last_cameraman_toggle_time: float = 0
    
    # Temporal & AI variables
    last_face_landmarks: Any = None
    last_face_time: float = 0
    voice_command_queue: List[str] = field(default_factory=list)
    is_listening: bool = False
    last_voice_text: str = ""
    last_voice_time: float = 0
    
    # Nanotech Deployment state
    deploy_y: int = 0
    is_deploying: bool = False
    is_retracting: bool = False
    suit_up_complete: bool = False
    deploy_start_time: float = 0
    
    # Smoothing / Caches (Pre-allocated buffers & EMA states)
    smooth_cache: Dict[str, float] = field(default_factory=dict)
    face_smooth_cache: Dict[int, Any] = field(default_factory=dict)
    face_raw_cache: Dict[int, Any] = field(default_factory=dict)
    hand_smooth_caches: Dict[str, Any] = field(default_factory=dict)
    hand_raw_caches: Dict[str, Any] = field(default_factory=dict)
    pose_smooth_cache: Dict[int, Any] = field(default_factory=dict)
    pose_raw_cache: Dict[int, Any] = field(default_factory=dict)
    vignette_cache: Any = None
    scanline_mask: Any = None
    bloom_cache: Any = None
    small_cache: Any = None
    bloom_up_cache: Any = None
    face_raw_buf: Any = None
    face_pts_buf: Any = None
    hand_raw_bufs: Dict[str, Any] = field(default_factory=dict)
    hand_pts_bufs: Dict[str, Any] = field(default_factory=dict)
    virtual_cam_buf: Any = None
    
    sys_stats: Dict[str, Any] = field(default_factory=dict)
    sys_stats_time: float = 0
    cpu_history_buffer: List[float] = field(default_factory=list)
    real_process_list: List[str] = field(default_factory=list)
    process_update_time: float = 0

    def allocate_buffers(self, w: int, h: int, c: int = 3):
        quarter_w, quarter_h = w // 4, h // 4
            
        # Switched back to standard zero-allocation numpy arrays. 
        # Keeps memory local to your Ryzen CPU and avoids the RX 6400 PCIe bottleneck.
        if self.small_cache is None:
            self.small_cache = np.zeros((quarter_h, quarter_w, c), dtype=np.uint8)
        if self.bloom_up_cache is None:
            self.bloom_up_cache = np.zeros((h, w, c), dtype=np.uint8)
            
        if self.vignette_cache is None:
            vig = np.zeros((h, w), dtype=np.float32)
            cv2.ellipse(vig, (w // 2, h // 2), (int(w * 0.55), int(h * 0.6)), 0, 0, 360, 1.0, -1)
            cv2.GaussianBlur(vig, (101, 101), 0, dst=vig)
            vig_rgb = np.stack([vig, vig, vig], axis=2)
            self.vignette_cache = vig_rgb
            
        if self.face_raw_buf is None:
            self.face_raw_buf = np.zeros((478, 3), dtype=np.float32)
        if self.face_pts_buf is None:
            self.face_pts_buf = np.zeros((478, 2), dtype=np.int32)
            
        if self.bg_frame is None:
            import os
            if os.path.exists("empty_room_ref.jpg"):
                try:
                    loaded = cv2.imread("empty_room_ref.jpg")
                    if loaded is not None:
                        if loaded.shape[:2] != (h, w):
                            loaded = cv2.resize(loaded, (w, h))
                        self.bg_frame = loaded
                        print("[ROOM REFERENCE] Loaded saved empty room background from 'empty_room_ref.jpg'")
                except Exception as e:
                    pass
    
    # Threading controls
    inference_running: bool = False
    
    # Animation
    frame_count: int = 0
    scan_y: float = 0
    
    # Profiling
    render_latency_ms: float = 0.0
    inference_latency_ms: float = 0.0
