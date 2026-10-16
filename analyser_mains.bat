@echo off
cd /d "%~dp0"
python analyse_mains.py %*
pause
