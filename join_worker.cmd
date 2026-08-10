@echo off
setlocal
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0join_worker.ps1"
set EXITCODE=%ERRORLEVEL%
if not "%EXITCODE%"=="0" (
  echo.
  echo join_worker.cmd failed with exit code %EXITCODE%.
)
exit /b %EXITCODE%
