@echo off
setlocal enabledelayedexpansion

REM Usage: setup_env.bat [env_name]
REM If env_name is provided, use it. Otherwise create "pcprocessor".

set "ENV_NAME=%~1"
if "%ENV_NAME%"=="" set "ENV_NAME=pcprocessor"

REM Check if conda is available
where conda >nul 2>nul
if %errorlevel% neq 0 (
    echo ERROR: conda is not installed or not in PATH.
    echo Install Miniconda/Anaconda first: https://docs.conda.io/en/latest/miniconda.html
    exit /b 1
)

REM Check if environment already exists
conda env list | findstr /b "%ENV_NAME% " >nul 2>nul
if %errorlevel% equ 0 (
    echo Environment '%ENV_NAME%' already exists. Activating...
    call conda activate "%ENV_NAME%"
) else (
    echo Creating conda environment '%ENV_NAME%' with Python 3.11...
    conda create -n "%ENV_NAME%" python=3.11 -y
    call conda activate "%ENV_NAME%"
)

echo.
echo Installing dependencies...
pip install --upgrade pip
pip install numpy scipy
pip install PySide6
pip install vtk
pip install open3d
pip install PyOpenGL PyOpenGL_accelerate

echo.
echo ============================================
echo   Setup complete!
echo   Environment: %ENV_NAME%
echo.
echo   To run the application:
echo     conda activate %ENV_NAME%
echo     cd src && python main.py
echo.
echo   Or use:  run.bat %ENV_NAME%
echo ============================================