$ErrorActionPreference = 'Stop'

function Ensure-Python {
  if (Get-Command py -ErrorAction SilentlyContinue) { return }
  if (Get-Command python -ErrorAction SilentlyContinue) { return }

  Write-Host 'Python not found. Attempting install via winget...'
  if (-not (Get-Command winget -ErrorAction SilentlyContinue)) {
    throw 'winget is not installed. Please install Python 3.12 manually, then rerun join_worker.ps1.'
  }

  winget install -e --id Python.Python.3.12 --accept-source-agreements --accept-package-agreements
}

Ensure-Python

if (-not (Test-Path .\.venv\Scripts\python.exe)) {
  if (Get-Command py -ErrorAction SilentlyContinue) {
    py -3.12 -m venv .venv
  } else {
    python -m venv .venv
  }
}

.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

# Auto-discover controller on LAN and join.
.\.venv\Scripts\python.exe .\src\distributed_worker.py
