import numpy as np
import traceback
from hud_modes import MODES
from state import HUDState

class LM:
    def __init__(self):
        self.x = 0.5
        self.y = 0.5
        self.z = 0.5

lm_list = [LM() for _ in range(478)]
img = np.zeros((720, 1280, 3), dtype=np.uint8)
state = HUDState()
t = 0.1

pts = np.zeros((468, 2), dtype=np.int32)
nose = np.array([640, 360], dtype=np.int32)
lines = np.zeros((100, 2, 2), dtype=np.int32)
dist = 1.0

for i in range(6):
    print(f"Testing Mode {i}...")
    try:
        MODES[i].render(img, lm_list, 1280, 720, t, pts, nose, lines, dist, state)
        print(f"Mode {i} OK!")
    except Exception as e:
        print(f"Mode {i} FAILED:")
        traceback.print_exc()
