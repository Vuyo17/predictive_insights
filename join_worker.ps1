$ControllerUrl = ""
if ($args.Count -gt 0) {
  $ControllerUrl = [string]$args[0]
}

$ErrorActionPreference = 'Stop'
$ScriptRoot = Split-Path -Parent $PSCommandPath
Set-Location $ScriptRoot
$LogDir = Join-Path $ScriptRoot 'logs'
New-Item -Path $LogDir -ItemType Directory -Force | Out-Null
$LogPath = Join-Path $LogDir ("join_worker_{0}.log" -f (Get-Date -Format 'yyyyMMdd_HHmmss'))
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
    (Join-Path $ScriptRoot '.venv\Scripts\python.exe'),
    (Join-Path $ScriptRoot '.venv\python.exe'),
    (Join-Path $ScriptRoot '.venv\bin\python.exe'),
    (Join-Path $ScriptRoot '.venv\bin\python')
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

  if ([bool](Get-Command py -ErrorAction SilentlyContinue)) {
    try {
      & py -3.12 -c "import sys" *> $null
      if ($LASTEXITCODE -eq 0) { return 'py' }
    } catch {}
  }
  if ([bool](Get-Command python -ErrorAction SilentlyContinue)) {
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

function Get-RequirementsHash {
  return (Get-FileHash -Algorithm SHA256 -Path (Join-Path $ScriptRoot 'requirements.txt')).Hash
}

function Get-PythonVersionString {
  param([string]$PythonExe)
  return (& $PythonExe -c "import sys; print(sys.version.split()[0])").Trim()
}

function Should-InstallDependencies {
  param([string]$PythonExe)

  $statePath = Join-Path $ScriptRoot '.bootstrap_worker_state.json'
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
  $statePath = Join-Path $ScriptRoot '.bootstrap_worker_state.json'
  $payload = @{
    updated_at = (Get-Date).ToString('o')
    fingerprint = $Fingerprint
  } | ConvertTo-Json
  Set-Content -Path $statePath -Value $payload -Encoding UTF8
}

function Discover-ControllerUrl {
  param([int]$DiscoveryPort, [int]$TimeoutMs)

  $client = New-Object System.Net.Sockets.UdpClient
  try {
    $client.EnableBroadcast = $true
    $client.Client.ReceiveTimeout = $TimeoutMs
    $bytes = [System.Text.Encoding]::UTF8.GetBytes('DISCOVER_IMPROVER')
    $endpoint = New-Object System.Net.IPEndPoint([System.Net.IPAddress]::Parse('255.255.255.255'), $DiscoveryPort)
    [void]$client.Send($bytes, $bytes.Length, $endpoint)
    $remote = New-Object System.Net.IPEndPoint([System.Net.IPAddress]::Any, 0)
    $resp = $client.Receive([ref]$remote)
    $text = [System.Text.Encoding]::UTF8.GetString($resp)
    $obj = $text | ConvertFrom-Json
    if ($obj.controller_url) {
      return [string]$obj.controller_url
    }
    return ''
  } catch {
    return ''
  } finally {
    $client.Close()
  }
}

function Resolve-ControllerUrl {
  param([string]$CliValue)

  if (-not [string]::IsNullOrWhiteSpace($CliValue)) {
    return $CliValue.Trim()
  }

  if (-not [string]::IsNullOrWhiteSpace($env:CONTROLLER_URL)) {
    return $env:CONTROLLER_URL.Trim()
  }

  $fromDiscovery = Discover-ControllerUrl -DiscoveryPort 50555 -TimeoutMs 2500
  if (-not [string]::IsNullOrWhiteSpace($fromDiscovery)) {
    Write-Host "Discovered controller: $fromDiscovery"
    return $fromDiscovery
  }

  Write-Host 'Controller auto-discovery failed.' -ForegroundColor Yellow
  $manual = Read-Host 'Enter controller URL (example: http://192.168.1.10:8765)'
  if ([string]::IsNullOrWhiteSpace($manual)) {
    throw 'No controller URL provided. Rerun join_worker.cmd and provide a URL when prompted.'
  }
  return $manual.Trim()
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

  $installDecision = Should-InstallDependencies -PythonExe $venvPython
  if ($installDecision.Install) {
    Write-Host 'Installing dependencies (requirements changed or first run)...'
    Invoke-Checked -FilePath $venvPython -Arguments @('-m', 'pip', 'install', '-r', (Join-Path $ScriptRoot 'requirements.txt')) -StepName 'requirements install'
    Write-InstallState -Fingerprint $installDecision.Fingerprint
  } else {
    Write-Host 'Dependencies already up to date. Skipping installation.'
  }

  $resolvedUrl = Resolve-ControllerUrl -CliValue $ControllerUrl
  if ($resolvedUrl -notmatch '^https?://') {
    throw "Invalid controller URL: $resolvedUrl"
  }

  Write-Host "Launching worker against $resolvedUrl ..."
  Invoke-Checked -FilePath $venvPython -Arguments @((Join-Path $ScriptRoot 'src\distributed_worker.py'), '--controller-url', $resolvedUrl) -StepName 'worker startup'
}
catch {
  Write-Host ''
  Write-Host 'join_worker.ps1 failed:' -ForegroundColor Red
  Write-Host $_.Exception.Message -ForegroundColor Red
  Write-Host "Log file: $LogPath" -ForegroundColor Yellow
  exit 1
}
finally {
  try { Stop-Transcript | Out-Null } catch {}
}
