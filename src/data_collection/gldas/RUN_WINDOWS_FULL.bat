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
%PYCMD% run_gldas_to_daily.py
if errorlevel 1 goto :fail
%PYCMD% merge_final.py
if errorlevel 1 goto :fail
echo.
echo COMPLETE. Check the output folder.
goto :end
:fail
echo.
echo Pipeline stopped because an error occurred. Read the message above.
:end
pause
