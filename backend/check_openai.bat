@echo off
setlocal

REM Use project venv’s python if present; otherwise fall back to system python
set "PYEXE=%~dp0.venv\Scripts\python.exe"
if not exist "%PYEXE%" set "PYEXE=python"

REM Require OPENAI_API_KEY in the environment
if "%OPENAI_API_KEY%"=="" (
  echo ERROR: OPENAI_API_KEY is empty in this shell.
  echo Set it first, e.g.:
  echo   set OPENAI_API_KEY=sk-...
  exit /b 1
)

"%PYEXE%" "%~dp0check_openai.py"
exit /b %errorlevel%
