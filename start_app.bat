@echo off
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo Virtual environment not found. Create .venv and install the project dependencies first.
    pause
    exit /b 1
)

".venv\Scripts\python.exe" -m multimodalcv.cli.serve
if errorlevel 1 pause
