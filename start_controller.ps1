$ErrorActionPreference = 'Stop'

if (-not (Test-Path .\.venv\Scripts\python.exe)) {
  py -3.12 -m venv .venv
}

.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

# Start controller + dashboard
.\.venv\Scripts\python.exe .\src\distributed_controller.py --host 0.0.0.0 --port 8765 --discovery-port 50555
