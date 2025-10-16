@echo off
echo ========================================
echo Fingerprint Blood Group Predictor
echo ========================================
echo.

echo [1/4] Checking Python installation...
python --version
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH
    pause
    exit /b 1
)

echo [2/4] Installing dependencies...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo ERROR: Failed to install dependencies
    pause
    exit /b 1
)

echo [3/4] Checking required files...
if not exist "fingerprint_blood_group_resnet18.pth" (
    echo WARNING: Model file not found. Please ensure fingerprint_blood_group_resnet18.pth is in this directory.
    pause
)

if not exist "model.py" (
    echo ERROR: model.py not found
    pause
    exit /b 1
)

echo [4/4] Starting Streamlit application...
echo.
echo The application will open in your default browser.
echo Navigate to the Live Capture tab to use biometric device.
echo.

streamlit run app.py

pause