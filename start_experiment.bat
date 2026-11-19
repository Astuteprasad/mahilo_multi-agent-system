@echo off
setlocal
cd /d "%~dp0"
title Track Manager experiment

rem ---- First-time setup (when .venv is missing, broken or incomplete) ----
if exist ".venv\pyvenv.cfg" ".venv\Scripts\python.exe" -c "import fastapi, uvicorn, openai, litellm, mahilo" 2>nul && goto :venvok
echo First-time setup: this takes a few minutes and needs internet.
echo Do not close this window until the page opens.
if exist ".venv" rmdir /s /q .venv
py -3.12 -m venv .venv 2>nul
if not exist ".venv\Scripts\python.exe" python -m venv .venv
if not exist ".venv\Scripts\python.exe" goto :fail
".venv\Scripts\python.exe" -m pip install -r requirements.txt || goto :fail
".venv\Scripts\python.exe" -m pip install -e mahilo_src --no-deps || goto :fail
:venvok

rem ---- API key (asked once, then remembered for this Windows user) ----
if not "%OPENAI_API_KEY%"=="" goto :haskey
echo.
set /p OPENAI_API_KEY=Paste the OpenAI API key and press Enter: 
setx OPENAI_API_KEY "%OPENAI_API_KEY%" >nul
:haskey

rem ---- Start ----
set MAHILO_LLM_MODEL=openai/gpt-6-luna
echo.
echo Starting the experiment. The page opens in your browser.
echo Keep this window open during the session. Close it to stop.
start "" /min cmd /c "timeout /t 5 >nul & start http://127.0.0.1:8000"
".venv\Scripts\python.exe" human_app.py
echo.
echo The experiment server has stopped.
pause
goto :eof

:fail
echo.
echo Setup failed. Check that Python 3.12 is installed and that there is internet, then run this file again.
pause
