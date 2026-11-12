@echo off
rem Double-click to start the confocal GUI. Add --sim to use the simulator.
cd /d "%~dp0"
set "PY="
for %%P in ("%USERPROFILE%\anaconda3" "%USERPROFILE%\Anaconda3" "%USERPROFILE%\miniconda3" "%ProgramData%\anaconda3" "%ProgramData%\Anaconda3" "%ProgramData%\miniconda3" "%LOCALAPPDATA%\anaconda3" "%LOCALAPPDATA%\miniconda3" "C:\anaconda3" "C:\Anaconda3" "C:\miniconda3") do (
    if not defined PY if exist "%%~P\python.exe" set "PY=%%~P\python.exe"
)
if exist "python_path.txt" set /p PY=<python_path.txt
if not defined PY (
    for /f "delims=" %%P in ('where python 2^>nul') do (
        echo %%P | find /i "WindowsApps" >nul || if not defined PY set "PY=%%P"
    )
)
if not defined PY (
    echo No Python found. Put the full path of python.exe into python_path.txt next to this file.
    pause
    exit /b 1
)
"%PY%" "%~dp0start_confocal.py" %*