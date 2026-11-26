@echo off
setlocal

set "PY=python"
if exist "%~dp0..\.venv\Scripts\python.exe" set "PY=%~dp0..\.venv\Scripts\python.exe"

echo === Running Ruff --fix (auto-fix lint) ===
"%PY%" -m ruff check --fix .
if errorlevel 1 echo [WARN] Ruff reported remaining lint issues.

echo === Running isort (import sorting) ===
"%PY%" -m isort .
if errorlevel 1 echo [WARN] isort reported issues.

echo === Running Black (code formatting) ===
"%PY%" -m black .
if errorlevel 1 echo [WARN] Black reported issues.

echo.
echo Formatting complete.
