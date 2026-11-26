@echo off
setlocal

set "PY=python"
if exist "%~dp0..\.venv\Scripts\python.exe" set "PY=%~dp0..\.venv\Scripts\python.exe"

echo === Running Ruff (lint check only, no auto-fix) ===
"%PY%" -m ruff check .

if errorlevel 1 (
    echo.
    echo [FAIL] Lint errors found. Run .\scripts\format.bat to auto-fix.
    exit /b 1
)

echo.
echo Linting passed — no issues found.
