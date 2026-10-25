@echo off
cd /d "%~dp0"
where py >nul 2>nul
if %ERRORLEVEL%==0 (
    set PYCMD=py
) else (
    set PYCMD=python
)
%PYCMD% TEST_EARTHDATA_CREDENTIALS.py
pause
