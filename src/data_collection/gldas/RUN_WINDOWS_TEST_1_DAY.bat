@echo off
cd /d "%~dp0"
where py >nul 2>nul
if %ERRORLEVEL%==0 (
    set PYCMD=py
) else (
    set PYCMD=python
)
%PYCMD% -m pip install -r requirements.txt
if errorlevel 1 goto :fail
%PYCMD% run_gldas_to_daily.py --max-days 1
goto :end
:fail
echo.
echo Setup failed.
:end
pause
