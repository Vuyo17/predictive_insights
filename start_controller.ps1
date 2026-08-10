$ErrorActionPreference = 'Stop'

function Get-PythonBootstrapCommand {
  if (Get-Command py -ErrorAction SilentlyContinue) { return 'py -3.12' }
  if (Get-Command python -ErrorAction SilentlyContinue) { return 'python' }
  return $null
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
    $bootstrap = Get-PythonBootstrapCommand
    if (-not $bootstrap) {
      throw 'Python not found. Install Python 3.12+ and rerun start_controller.ps1.'
    }
    Write-Host 'Creating virtual environment (.venv)...'
    Invoke-Expression "$bootstrap -m venv .venv"
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
