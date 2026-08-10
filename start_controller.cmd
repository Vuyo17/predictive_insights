@echo off
setlocal
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0start_controller.ps1"
set EXITCODE=%ERRORLEVEL%
if not "%EXITCODE%"=="0" (
  echo.
  echo start_controller.cmd failed with exit code %EXITCODE%.
)
exit /b %EXITCODE%
