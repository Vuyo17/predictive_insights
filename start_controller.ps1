$ErrorActionPreference = 'Stop'
$ScriptRoot = Split-Path -Parent $PSCommandPath
Set-Location $ScriptRoot
$LogDir = Join-Path $ScriptRoot 'logs'
New-Item -Path $LogDir -ItemType Directory -Force | Out-Null
$LogPath = Join-Path $LogDir ("start_controller_{0}.log" -f (Get-Date -Format 'yyyyMMdd_HHmmss'))
Start-Transcript -Path $LogPath -Append | Out-Null

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

function Get-RequirementsHash {
  return (Get-FileHash -Algorithm SHA256 -Path (Join-Path $ScriptRoot 'requirements.txt')).Hash
}

function Get-PythonVersionString {
  param([string]$PythonExe)
  return (& $PythonExe -c "import sys; print(sys.version.split()[0])").Trim()
}

function Should-InstallDependencies {
  param([string]$PythonExe)

  $statePath = Join-Path $ScriptRoot '.bootstrap_controller_state.json'
  $reqHash = Get-RequirementsHash
  $pyVersion = Get-PythonVersionString -PythonExe $PythonExe
  $fingerprint = "$reqHash|$pyVersion"

  if (-not (Test-Path $statePath)) {
    return @{ Install = $true; Fingerprint = $fingerprint }
  }

  try {
    $raw = Get-Content -Path $statePath -Raw
    $obj = $raw | ConvertFrom-Json
    if ($obj.fingerprint -eq $fingerprint) {
      return @{ Install = $false; Fingerprint = $fingerprint }
    }
  } catch {}

  return @{ Install = $true; Fingerprint = $fingerprint }
}

function Write-InstallState {
  param([string]$Fingerprint)
  $statePath = Join-Path $ScriptRoot '.bootstrap_controller_state.json'
  $payload = @{
    updated_at = (Get-Date).ToString('o')
    fingerprint = $Fingerprint
  } | ConvertTo-Json
  Set-Content -Path $statePath -Value $payload -Encoding UTF8
}

function Test-ControllerAlreadyRunning {
  $conn = Get-NetTCPConnection -LocalPort 8765 -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
  if ($null -eq $conn) {
    return $false
  }

  $proc = Get-CimInstance Win32_Process -Filter ("ProcessId = " + $conn.OwningProcess)
  if ($null -ne $proc -and $proc.CommandLine -like '*distributed_controller.py*') {
    Write-Host "Controller already running on port 8765 (PID $($conn.OwningProcess))."
    return $true
  }

  throw "Port 8765 is already in use by PID $($conn.OwningProcess)."
}

function Normalize-ServiceExitCode {
  param([int]$Code)
  if ($Code -eq 0) { return 0 }
  if ($Code -eq -1 -or $Code -eq -1073741510 -or $Code -eq 3221225786) {
    return 0
  }
  return $Code
}

try {
  Write-Host 'Starting distributed controller setup...'
  Ensure-RepoShape

  if (Test-ControllerAlreadyRunning) {
    Write-Host "Log file: $LogPath"
    exit 0
  }

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

  $installDecision = Should-InstallDependencies -PythonExe $venvPython
  if ($installDecision.Install) {
    Write-Host 'Installing dependencies (requirements changed or first run)...'
    Invoke-Checked -FilePath $venvPython -Arguments @('-m', 'pip', 'install', '-r', (Join-Path $ScriptRoot 'requirements.txt')) -StepName 'requirements install'
    Write-InstallState -Fingerprint $installDecision.Fingerprint
  } else {
    Write-Host 'Dependencies already up to date. Skipping installation.'
  }

  Write-Host 'Launching controller...'
  & $venvPython (Join-Path $ScriptRoot 'src\distributed_controller.py') --host 0.0.0.0 --port 8765 --discovery-port 50555
  $controllerExit = Normalize-ServiceExitCode -Code $LASTEXITCODE
  if ($controllerExit -eq 0) {
    Write-Host 'Controller stopped.'
    exit 0
  }
  throw "controller startup failed with exit code $controllerExit."
}
catch {
  Write-Host ''
  Write-Host 'start_controller.ps1 failed:' -ForegroundColor Red
  Write-Host $_.Exception.Message -ForegroundColor Red
  Write-Host "Log file: $LogPath" -ForegroundColor Yellow
  exit 1
}
finally {
  try { Stop-Transcript | Out-Null } catch {}
}
