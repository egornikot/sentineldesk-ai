@echo off
cd /d "%~dp0"
if not exist .env copy .env.example .env >nul
if exist .venv\Scripts\python.exe (
  .venv\Scripts\python.exe run.py %*
) else (
  py run.py %*
)
pause
