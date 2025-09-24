# Change to script directory
Set-Location $PSScriptRoot

# Activate virtual environment and run webui with CUDA enabled
& "AICoverGen\Scripts\python.exe" "src\webui.py"
