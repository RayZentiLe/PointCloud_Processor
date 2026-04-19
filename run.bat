@echo off
setlocal enabledelayedexpansion

REM Usage: run.bat [env_name]
REM If env_name is provided, use it. Otherwise use "pcprocessor".

set "ENV_NAME=%~1"
if "%ENV_NAME%"=="" set "ENV_NAME=pcprocessor"

REM Check if conda is available
where conda >nul 2>nul
if %errorlevel% neq 0 (
    echo ERROR: conda is not installed or not in PATH.
    echo Install Miniconda/Anaconda first: https://docs.conda.io/en/latest/miniconda.html
    exit /b 1
)

REM Activate conda
call conda activate "%ENV_NAME%"

REM Get script directory and cd to src
set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%src"

REM Run the app
python main.py