$ErrorActionPreference = 'Stop'
$ScriptRoot = Split-Path -Parent $PSCommandPath
Set-Location $ScriptRoot

function Invoke-Checked {
  param(
    [string]$FilePath,
    [string[]]$Arguments,
    [string]$StepName
  )

  & $FilePath @Arguments
  if ($LASTEXITCODE -ne 0) {
    throw "$StepName failed with exit code $LASTEXITCODE."
  }
}

function Resolve-VenvPythonPath {
  $candidates = @(
    (Join-Path $ScriptRoot ".venv\Scripts\python.exe"),
    (Join-Path $ScriptRoot ".venv\python.exe"),
    (Join-Path $ScriptRoot ".venv\bin\python.exe"),
    (Join-Path $ScriptRoot ".venv\bin\python")
  )
  foreach ($path in $candidates) {
    if (Test-Path $path) {
      return $path
    }
  }
  return $null
}

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
  if ($LASTEXITCODE -ne 0) {
    throw 'Automatic Python install failed. Install Python 3.12 manually, open a new terminal, then rerun start_controller.ps1.'
  }

  # PATH updates may require a new shell. Try immediate detection first.
  if ([bool](Get-Command py -ErrorAction SilentlyContinue) -and (Test-PythonRuntime 'py')) { return 'py' }
  if ([bool](Get-Command python -ErrorAction SilentlyContinue) -and (Test-PythonRuntime 'python')) { return 'python' }

  throw 'Python was installed, but this shell cannot see it yet. Close this terminal, open a new one, and rerun start_controller.ps1.'
}

function Ensure-RepoShape {
  if (-not (Test-Path (Join-Path $ScriptRoot 'requirements.txt'))) {
    throw 'requirements.txt not found. Run this script from the repository root.'
  }
  if (-not (Test-Path (Join-Path $ScriptRoot 'src\distributed_controller.py'))) {
    throw 'src\distributed_controller.py not found. Run this script from the repository root.'
  }
}

try {
  Write-Host 'Starting distributed controller setup...'
  Ensure-RepoShape
  $venvPython = Resolve-VenvPythonPath

  if (-not $venvPython) {
    $bootstrap = Ensure-Python
    Write-Host 'Creating virtual environment (.venv)...'
    if ($bootstrap -eq 'py') {
      & py -3.12 -m venv (Join-Path $ScriptRoot '.venv')
      if ($LASTEXITCODE -ne 0) {
        throw 'Failed to create virtual environment with py -3.12.'
      }
    } else {
      & python -m venv (Join-Path $ScriptRoot '.venv')
      if ($LASTEXITCODE -ne 0) {
        throw 'Failed to create virtual environment with python -m venv.'
      }
    }
    $venvPython = Resolve-VenvPythonPath
    if (-not $venvPython) {
      throw 'Virtual environment was created but python executable was not found under .venv.'
    }
  } else {
    Ensure-Python | Out-Null
  }

  Write-Host 'Installing dependencies...'
  Invoke-Checked -FilePath $venvPython -Arguments @('-m', 'pip', 'install', '--upgrade', 'pip') -StepName 'pip self-upgrade'
  Invoke-Checked -FilePath $venvPython -Arguments @('-m', 'pip', 'install', '-r', (Join-Path $ScriptRoot 'requirements.txt')) -StepName 'requirements install'

  Write-Host 'Launching controller...'
  Invoke-Checked -FilePath $venvPython -Arguments @((Join-Path $ScriptRoot 'src\distributed_controller.py'), '--host', '0.0.0.0', '--port', '8765', '--discovery-port', '50555') -StepName 'controller startup'
}
catch {
  Write-Host ''
  Write-Host 'start_controller.ps1 failed:' -ForegroundColor Red
  Write-Host $_.Exception.Message -ForegroundColor Red
  exit 1
}
