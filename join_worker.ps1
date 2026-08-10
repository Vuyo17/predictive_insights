$ErrorActionPreference = 'Stop'

function Ensure-RepoShape {
  if (-not (Test-Path .\requirements.txt)) {
    throw 'requirements.txt not found. Run this script from the repository root.'
  }
  if (-not (Test-Path .\src\distributed_worker.py)) {
    throw 'src\distributed_worker.py not found. Run this script from the repository root.'
  }
}

function Ensure-Python {
  if (Get-Command py -ErrorAction SilentlyContinue) { return }
  if (Get-Command python -ErrorAction SilentlyContinue) { return }

  Write-Host 'Python not found. Attempting install via winget...'
  if (-not (Get-Command winget -ErrorAction SilentlyContinue)) {
    throw 'winget is not installed. Please install Python 3.12 manually, then rerun join_worker.ps1.'
  }

  winget install -e --id Python.Python.3.12 --accept-source-agreements --accept-package-agreements
}

try {
  Write-Host 'Starting worker setup...'
  Ensure-RepoShape
  Ensure-Python

  if (-not (Test-Path .\.venv\Scripts\python.exe)) {
    Write-Host 'Creating virtual environment (.venv)...'
    if (Get-Command py -ErrorAction SilentlyContinue) {
      py -3.12 -m venv .venv
    } else {
      python -m venv .venv
    }
  }

  Write-Host 'Installing dependencies...'
  .\.venv\Scripts\python.exe -m pip install --upgrade pip
  .\.venv\Scripts\python.exe -m pip install -r requirements.txt

  Write-Host 'Launching worker...'
  # Auto-discover controller on LAN and join.
  .\.venv\Scripts\python.exe .\src\distributed_worker.py
}
catch {
  Write-Host ''
  Write-Host 'join_worker.ps1 failed:' -ForegroundColor Red
  Write-Host $_.Exception.Message -ForegroundColor Red
  exit 1
}
