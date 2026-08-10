$ControllerUrl = ""
if ($args.Count -gt 0) {
  $ControllerUrl = [string]$args[0]
}

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
  $hasPy = [bool](Get-Command py -ErrorAction SilentlyContinue)
  $hasPython = [bool](Get-Command python -ErrorAction SilentlyContinue)

  $pyReady = $false
  $pythonReady = $false

  if ($hasPy) {
    try {
      & py -3.12 -c "import sys" *> $null
      $pyReady = ($LASTEXITCODE -eq 0)
    } catch {
      $pyReady = $false
    }
  }

  if ($hasPython) {
    try {
      & python -c "import sys" *> $null
      $pythonReady = ($LASTEXITCODE -eq 0)
    } catch {
      $pythonReady = $false
    }
  }

  if ($pyReady) { return 'py' }
  if ($pythonReady) { return 'python' }

  Write-Host 'Python 3.12+ runtime not found. Attempting install via winget...'
  if (-not (Get-Command winget -ErrorAction SilentlyContinue)) {
    throw 'winget is not installed. Install Python 3.12 manually, open a new terminal, then rerun join_worker.ps1.'
  }

  winget install -e --id Python.Python.3.12 --accept-source-agreements --accept-package-agreements

  # PATH updates may require a new shell.
  $postHasPy = [bool](Get-Command py -ErrorAction SilentlyContinue)
  $postHasPython = [bool](Get-Command python -ErrorAction SilentlyContinue)
  if ($postHasPy) {
    try {
      & py -3.12 -c "import sys" *> $null
      if ($LASTEXITCODE -eq 0) { return 'py' }
    } catch {}
  }
  if ($postHasPython) {
    try {
      & python -c "import sys" *> $null
      if ($LASTEXITCODE -eq 0) { return 'python' }
    } catch {}
  }

  throw 'Python was installed, but this shell cannot see it yet. Close this terminal, open a new one, and rerun join_worker.ps1.'
}

try {
  Write-Host 'Starting worker setup...'
  Ensure-RepoShape
  $bootstrap = Ensure-Python

  if (-not (Test-Path .\.venv\Scripts\python.exe)) {
    Write-Host 'Creating virtual environment (.venv)...'
    if ($bootstrap -eq 'py') {
      py -3.12 -m venv .venv
    } else {
      python -m venv .venv
    }
  }

  Write-Host 'Installing dependencies...'
  .\.venv\Scripts\python.exe -m pip install --upgrade pip
  .\.venv\Scripts\python.exe -m pip install -r requirements.txt

  Write-Host 'Launching worker...'
  if ([string]::IsNullOrWhiteSpace($ControllerUrl)) {
    # Auto-discover controller on LAN and join.
    .\.venv\Scripts\python.exe .\src\distributed_worker.py
  } else {
    # Direct connect fallback when LAN discovery is blocked.
    .\.venv\Scripts\python.exe .\src\distributed_worker.py --controller-url $ControllerUrl
  }
}
catch {
  Write-Host ''
  Write-Host 'join_worker.ps1 failed:' -ForegroundColor Red
  Write-Host $_.Exception.Message -ForegroundColor Red
  exit 1
}
