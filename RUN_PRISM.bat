@echo off
color 0B
title P.R.I.S.M. System Boot

echo =======================================================
echo          P.R.I.S.M. Tracking System - Auto Setup
echo =======================================================
echo.

:: Check for Python
python --version >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    color 0C
    echo [CRITICAL ERROR] Python is not installed or not added to PATH!
    echo Please install Python 3.10 or higher from python.org
    echo IMPORTANT: Make sure to check the box that says "Add Python to PATH" during installation.
    echo.
    pause
    exit /b
)

echo [1/3] Installing and Updating Python Dependencies...
pip install opencv-python numpy mediapipe mss psutil pyvirtualcam SpeechRecognition pyttsx3 sounddevice
echo.

echo [2/3] Checking for AI Neural Network Models...
IF NOT EXIST "face_landmarker.task" (
    echo Downloading Face Landmarker...
    curl -o face_landmarker.task "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)
IF NOT EXIST "hand_landmarker.task" (
    echo Downloading Hand Landmarker...
    curl -o hand_landmarker.task "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)
IF NOT EXIST "pose_landmarker_full.task" (
    echo Downloading Pose Landmarker...
    curl -o pose_landmarker_full.task "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task"
)
IF NOT EXIST "selfie_segmenter.task" (
    echo Downloading Selfie Segmenter...
    curl -o selfie_segmenter.task "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.task"
)
echo.

echo [3/3] Boot Sequence Initiated...
echo Launching P.R.I.S.M...
python main.py

pause
