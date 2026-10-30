@echo off
rem Nightly job for Windows Task Scheduler (23:50): scrape FFWC, then forecast.
rem Register once:  schtasks /create /tn "FloodEngineNightly" /sc daily /st 23:50 /tr "\"%CD%\run_nightly.cmd\""
cd /d "%~dp0"
if not exist logs mkdir logs
"%~dp0..\.venv\Scripts\python.exe" engine.py all >> logs\nightly.log 2>&1
