@echo off
REM ============================================================
REM  One-click launcher (Windows).
REM  Double-click this file, OR run it from a terminal.
REM  Add flags after it, e.g.:  run.bat --sensitivity max
REM  To run vision only:        run.bat --text     (etc.)
REM ============================================================

REM Move to this script's own folder, whatever the current directory is.
cd /d "%~dp0"

if not exist ".venv\Scripts\activate.bat" (
  echo.
  echo   Virtual environment not found in this folder.
  echo   Set it up first - see the README "Setup" section.
  echo.
  pause
  exit /b 1
)

call ".venv\Scripts\activate.bat"
python assistant.py %*

echo.
echo   (Assistant closed.)
pause
