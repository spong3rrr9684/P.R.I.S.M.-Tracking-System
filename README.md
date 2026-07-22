# P.R.I.S.M. Tracking System
*(Pose Recognition & Intelligent Sensing Module)*

P.R.I.S.M. is a highly modular, real-time computer vision pipeline that overlays stunning sci-fi cinematic HUDs onto your live camera feed. It uses Google's MediaPipe for ultra-fast, multi-threaded face, hand, and body tracking, completely decoupled from the rendering engine to ensure buttery-smooth FPS even on entry-level CPU hardware.

## 🚀 Features

* **Zero-Lag Asynchronous Inference:** The AI inference runs on a totally isolated ThreadPool worker. It uses aggressive queue dropping to guarantee the UI is always locked to the absolute freshest frame.
* **Cinematic 1-Euro Smoothing:** Custom, dynamically tuned 1-Euro Filters adapt to your movement velocity. It heavily smooths micro-jitters when you are still, but instantly drops lag when you move fast.
* **Pure CPU Optimization:** Explicitly engineered for machines bottlenecked by PCIe bus transfers. OpenCL overhead is disabled, and all post-processing (Bloom, Vignette, Alpha Compositing) runs natively in zero-allocation NumPy arrays.
* **Voice-Activated Commands:** Complete hands-free control. Say *"Prism, suit up!"* to deploy the HUD, or *"Prism, quantum mode"* to switch aesthetics on the fly.
* **Virtual Camera Output:** Plugs directly into Discord, Zoom, or OBS via `pyvirtualcam` as a fully functional webcam feed.

## 🎨 HUD Modes
1. **Nano Armor** (Gold / Iron Man style)
2. **Quantum Blueprint** (Amber volumetric layers)
3. **Neon Cipher** (Green Matrix-style hacker UI)
4. **Surgical Edith** (White minimal, high-contrast)
5. **Infra-Thermal** (Predator heat-vision style)
6. **Ultron Void** (Purple glitch/malice aesthetics)

## 🛠️ Project Structure
* `main.py`: The entry point. Handles the camera loop, virtual camera streaming, and keystrokes.
* `renderer.py`: The core rendering engine for the HUDs, overlays, animations, and post-processing.
* `tracker.py`: Houses the MediaPipe Task APIs in a dedicated, blocking thread.
* `hud_modes.py`: Defines the visual aesthetic instructions (Strategy Pattern) for the 6 different HUD modes.
* `ui_components.py`: Contains the raw geometry drawing logic for side-panels, crosshairs, and data bars.
* `state.py`: Global dataclass defining the memory pools and tracking coordinates.
* `utils.py`: Contains the 1-Euro Filter logic, matrix scaling, and system telemetry gathering.
* `voice_assistant.py`: Threaded SpeechRecognition and pyttsx3 agent for voice commands.

## ⌨️ Controls
| Key | Action |
| :---: | :--- |
| `c` | Cycle physical webcams |
| `m` | Cycle HUD visual modes |
| `t` | Cycle tracking targets (Face/Hands/Pose) |
| `h` | Toggle side data panels |
| `s` | Dynamically crop a desktop window onto the screen |
| `n` | Manually trigger "Suit Up" deployment animation |
| `x` | Spawn holographic palm UI |
| `b` | Toggle Gesture Calibration Mode |
| `f` | Reset to Full Screen |
| `q` | Quit |

## ⚙️ Installation Guide

### Option A: The One-Click Auto Installer (Windows Only)
If you are on Windows, simply double-click the **`RUN_PRISM.bat`** file. 
It will automatically verify Python, install all pip dependencies, download the massive Google MediaPipe `.task` models into your folder, and launch the HUD. You don't have to touch a terminal!

---

### Option B: Manual Installation (Mac / Linux / Advanced Users)

**Step 1: Install Python Dependencies**
```bash
pip install opencv-python numpy mediapipe mss psutil pyvirtualcam SpeechRecognition pyttsx3 sounddevice
```

**Step 2: Download the AI Models**
P.R.I.S.M uses Google's MediaPipe machine learning models. You **MUST** download these 4 files and place them directly inside the `face_tracker` folder next to `main.py`.
* [face_landmarker.task](https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task)
* [hand_landmarker.task](https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task)
* [pose_landmarker_full.task](https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task)
* [selfie_segmenter.task](https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.task)

**Step 3: Setup Virtual Camera (Optional but Recommended)**
To stream the HUD directly into Discord, Zoom, or Twitch, the script uses a virtual webcam.
1. Download and install [OBS Studio](https://obsproject.com/).
2. Open OBS Studio and click **"Start Virtual Camera"** at least once to initialize the driver on your system.
*(If you do not want to use a virtual camera, you can safely ignore any `pyvirtualcam` warnings in the terminal; the local OpenCV preview window will still work perfectly).*

**Step 4: Run the System**
```bash
python main.py
```

---
*Built for extreme efficiency and cinematic aesthetics.*
