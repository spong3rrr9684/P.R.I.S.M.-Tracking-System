@echo off
color 0B
title P.R.I.S.M. Diagnostic Suite

:MENU
cls
echo =======================================================
echo          P.R.I.S.M. Diagnostic Test Suite
echo =======================================================
echo.
echo Please select a diagnostic test to run:
echo.
echo [1] System Environment Diagnostic (Checks dependencies, webcam, missing files)
echo [2] AI Engine Stress Test (Tests the neural networks and CPU math limits)
echo [3] HUD Visual Modes Diagnostic (Tests the graphics engine rendering)
echo [4] Audio ^& Voice Recognition Diagnostic (Tests your mic and speakers)
echo [5] Exit
echo.
set /p choice="Enter your choice (1-5): "

if "%choice%"=="1" goto DIAG
if "%choice%"=="2" goto TRACKER
if "%choice%"=="3" goto MODES
if "%choice%"=="4" goto VOICE
if "%choice%"=="5" goto EOF
goto MENU

:DIAG
cls
color 0E
echo Running System Diagnostics... Please wait.
set PYTHONPATH=%cd%
python tests\diagnostics.py > temp_log.tmp
clip < temp_log.tmp
del temp_log.tmp
echo.
color 0A
echo ========================================================
echo   SUCCESS! The System Diagnostic Report has been 
echo   COPIED TO YOUR CLIPBOARD.
echo.
echo   Just press CTRL+V to paste it to the developer!
echo ========================================================
echo.
pause
goto MENU

:TRACKER
cls
color 0E
echo Running AI Engine Stress Test... Please wait.
set PYTHONPATH=%cd%
python tests\test_tracker.py > temp_log.tmp
type temp_log.tmp
clip < temp_log.tmp
del temp_log.tmp
echo.
color 0A
echo ========================================================
echo   SUCCESS! The AI Engine Report has been 
echo   COPIED TO YOUR CLIPBOARD. 
echo.
echo   Just press CTRL+V to paste it to the developer!
echo ========================================================
echo.
pause
goto MENU

:MODES
cls
color 0E
echo Running HUD Visual Modes Diagnostic... Please wait.
set PYTHONPATH=%cd%
python tests\test_modes.py > temp_log.tmp
type temp_log.tmp
clip < temp_log.tmp
del temp_log.tmp
echo.
color 0A
echo ========================================================
echo   SUCCESS! The HUD Modes Report has been 
echo   COPIED TO YOUR CLIPBOARD. 
echo.
echo   Just press CTRL+V to paste it to the developer!
echo ========================================================
echo.
pause
goto MENU

:VOICE
cls
color 0E
echo Running Audio ^& Voice Recognition Diagnostic... Please wait.
set PYTHONPATH=%cd%
python tests\test_voice.py > temp_log.tmp
type temp_log.tmp
clip < temp_log.tmp
del temp_log.tmp
echo.
color 0A
echo ========================================================
echo   SUCCESS! The Voice Diagnostic Report has been 
echo   COPIED TO YOUR CLIPBOARD. 
echo.
echo   Just press CTRL+V to paste it to the developer!
echo ========================================================
echo.
pause
goto MENU
