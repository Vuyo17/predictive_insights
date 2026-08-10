$ControllerUrl = ""
if ($args.Count -gt 0) {
  $ControllerUrl = [string]$args[0]
}

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

function Ensure-RepoShape {
  if (-not (Test-Path (Join-Path $ScriptRoot 'requirements.txt'))) {
    throw 'requirements.txt not found. Run this script from the repository root.'
  }
  if (-not (Test-Path (Join-Path $ScriptRoot 'src\distributed_worker.py'))) {
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
      $ver = & python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
      if ($LASTEXITCODE -eq 0) {
        $parts = ($ver.Trim() -split '\\.')
        if ($parts.Length -ge 2) {
          $major = [int]$parts[0]
          $minor = [int]$parts[1]
          $pythonReady = ($major -gt 3) -or ($major -eq 3 -and $minor -ge 12)
        }
      }
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
  if ($LASTEXITCODE -ne 0) {
    throw 'Automatic Python install failed. Install Python 3.12 manually, open a new terminal, then rerun join_worker.ps1.'
  }

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
      $ver = & python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
      if ($LASTEXITCODE -eq 0) {
        $parts = ($ver.Trim() -split '\\.')
        if ($parts.Length -ge 2) {
          $major = [int]$parts[0]
          $minor = [int]$parts[1]
          if (($major -gt 3) -or ($major -eq 3 -and $minor -ge 12)) { return 'python' }
        }
      }
    } catch {}
  }

  throw 'Python was installed, but this shell cannot see it yet. Close this terminal, open a new one, and rerun join_worker.ps1.'
}

try {
  Write-Host 'Starting worker setup...'
  Ensure-RepoShape
  $bootstrap = Ensure-Python
  $venvPython = Resolve-VenvPythonPath

  if (-not $venvPython) {
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
  }

  Write-Host 'Installing dependencies...'
  Invoke-Checked -FilePath $venvPython -Arguments @('-m', 'pip', 'install', '--upgrade', 'pip') -StepName 'pip self-upgrade'
  Invoke-Checked -FilePath $venvPython -Arguments @('-m', 'pip', 'install', '-r', (Join-Path $ScriptRoot 'requirements.txt')) -StepName 'requirements install'

  Write-Host 'Launching worker...'
  if ([string]::IsNullOrWhiteSpace($ControllerUrl)) {
    # Auto-discover controller on LAN and join.
    Invoke-Checked -FilePath $venvPython -Arguments @((Join-Path $ScriptRoot 'src\distributed_worker.py')) -StepName 'worker startup'
  } else {
    # Direct connect fallback when LAN discovery is blocked.
    Invoke-Checked -FilePath $venvPython -Arguments @((Join-Path $ScriptRoot 'src\distributed_worker.py'), '--controller-url', $ControllerUrl) -StepName 'worker startup'
  }
}
catch {
  Write-Host ''
  Write-Host 'join_worker.ps1 failed:' -ForegroundColor Red
  Write-Host $_.Exception.Message -ForegroundColor Red
  exit 1
}
