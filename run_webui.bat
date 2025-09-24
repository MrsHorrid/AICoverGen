@echo off
setlocal enableextensions enabledelayedexpansion

REM Change to repo root (folder of this script)
cd /d "%~dp0"

REM ---------- Config ----------
set "VENV_DIR=AICoverGen"
set "PYTHON_EXE=%VENV_DIR%\Scripts\python.exe"
set "PIP_EXE=%VENV_DIR%\Scripts\pip.exe"

REM Activate virtual environment
call "%VENV_DIR%\Scripts\activate.bat"

REM Force ONNX Runtime to use CPU provider (PyTorch will still use GPU)
set "ORT_DISABLE_CUDA=1"

REM Optional: choose GPU index for PyTorch (0 = first GPU)
set "CUDA_VISIBLE_DEVICES=0"

REM Ensure UTF-8 output
set "PYTHONUTF8=1"
REM ----------------------------

REM Check venv
if not exist "%PYTHON_EXE%" (
  echo [ERROR] Virtual environment not found at %VENV_DIR%.^>
  echo         Expected: %PYTHON_EXE%
  echo         Create one first, or ensure the repo was set up correctly.
  echo.
  echo Example to create venv:
  echo   python -m venv AICoverGen
  exit /b 1
)

REM Upgrade pip (silent-ish)
"%PYTHON_EXE%" -m pip install --upgrade pip --disable-pip-version-check 1>nul 2>nul

REM Install core requirements
"%PIP_EXE%" install -r requirements.txt --no-input
if errorlevel 1 (
  echo [ERROR] Failed installing requirements.
  exit /b 1
)

REM Ensure a compatible Gradio version (3.50.2 is known-good with this UI)
"%PIP_EXE%" install "gradio==3.50.2" --no-input
if errorlevel 1 (
  echo [ERROR] Failed installing Gradio.
  exit /b 1
)

REM Check ffmpeg availability (recommended)
where ffmpeg >nul 2>nul
if errorlevel 1 (
  echo [WARN] ffmpeg not found in PATH. Audio processing may fail.
  echo       Install ffmpeg and add it to PATH: https://ffmpeg.org/download.html
)

REM Optional: open the UI in browser after a short delay
start "" /b cmd /c "timeout /t 3 /nobreak >nul & start http://127.0.0.1:7860"

REM Run the WebUI. Add --listen to allow LAN access if desired.
"%PYTHON_EXE%" src\webui.py %*

REM Preserve exit code
set "EXIT_CODE=%ERRORLEVEL%"
echo.
echo Server exited with code %EXIT_CODE%.
exit /b %EXIT_CODE%
