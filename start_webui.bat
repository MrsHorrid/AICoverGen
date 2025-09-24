@echo off
cd /d "%~dp0"

REM Activate virtual environment
call AICoverGen\Scripts\activate.bat

REM Update yt-dlp to fix YouTube download issues
echo Updating yt-dlp...
pip install --upgrade yt-dlp --quiet

REM Run webui with CUDA enabled
python src\webui.py

pause
