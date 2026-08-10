$ErrorActionPreference = 'Stop'

function Test-PythonRuntime {
  param([string]$Command)

  try {
    if ($Command -eq 'py') {
      & py -3.12 -c "import sys" *> $null
    } else {
      & python -c "import sys" *> $null
    }
    return ($LASTEXITCODE -eq 0)
  } catch {
    return $false
  }
}

function Ensure-Python {
  $hasPy = [bool](Get-Command py -ErrorAction SilentlyContinue)
  $hasPython = [bool](Get-Command python -ErrorAction SilentlyContinue)

  if ($hasPy -and (Test-PythonRuntime 'py')) { return 'py' }
  if ($hasPython -and (Test-PythonRuntime 'python')) { return 'python' }

  Write-Host 'Python 3.12+ runtime not found. Attempting install via winget...'
  if (-not (Get-Command winget -ErrorAction SilentlyContinue)) {
    throw 'winget is not installed. Install Python 3.12 manually, open a new terminal, then rerun start_controller.ps1.'
  }

  winget install -e --id Python.Python.3.12 --accept-source-agreements --accept-package-agreements

  # PATH updates may require a new shell. Try immediate detection first.
  if ([bool](Get-Command py -ErrorAction SilentlyContinue) -and (Test-PythonRuntime 'py')) { return 'py' }
  if ([bool](Get-Command python -ErrorAction SilentlyContinue) -and (Test-PythonRuntime 'python')) { return 'python' }

  throw 'Python was installed, but this shell cannot see it yet. Close this terminal, open a new one, and rerun start_controller.ps1.'
}

function Ensure-RepoShape {
  if (-not (Test-Path .\requirements.txt)) {
    throw 'requirements.txt not found. Run this script from the repository root.'
  }
  if (-not (Test-Path .\src\distributed_controller.py)) {
    throw 'src\distributed_controller.py not found. Run this script from the repository root.'
  }
}

try {
  Write-Host 'Starting distributed controller setup...'
  Ensure-RepoShape

  if (-not (Test-Path .\.venv\Scripts\python.exe)) {
    $bootstrap = Ensure-Python
    Write-Host 'Creating virtual environment (.venv)...'
    if ($bootstrap -eq 'py') {
      & py -3.12 -m venv .venv
    } else {
      & python -m venv .venv
    }
  } else {
    Ensure-Python | Out-Null
  }

  Write-Host 'Installing dependencies...'
  .\.venv\Scripts\python.exe -m pip install --upgrade pip
  .\.venv\Scripts\python.exe -m pip install -r requirements.txt

  Write-Host 'Launching controller...'
  .\.venv\Scripts\python.exe .\src\distributed_controller.py --host 0.0.0.0 --port 8765 --discovery-port 50555
}
catch {
  Write-Host ''
  Write-Host 'start_controller.ps1 failed:' -ForegroundColor Red
  Write-Host $_.Exception.Message -ForegroundColor Red
  exit 1
}
