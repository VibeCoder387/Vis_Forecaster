@echo off
cd /d "%~dp0"
if exist .venv\Scripts\python.exe (
  .venv\Scripts\python.exe optimize_pipeline.py
) else (
  python optimize_pipeline.py
)
pause
