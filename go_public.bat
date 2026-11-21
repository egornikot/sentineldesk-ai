@echo off
rem Make https://egornikot.github.io/sentineldesk-ai/ triage any ticket live from this computer. Keep this window open.
cd /d "%~dp0"
set "PATH=C:\Program Files\Git\cmd;%PATH%"
if exist ".venv\Scripts\python.exe" (".venv\Scripts\python.exe" scripts\go_public.py) else (py scripts\go_public.py)
pause
