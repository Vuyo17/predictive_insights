# Local Submission Improver

This repository is intentionally minimal and local-only.

It does one thing: generate improved Kaggle submission files.

It also supports optional distributed compute so other PCs can join and accelerate candidate evaluation.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Data

Place competition files in `data/` and include `train` and `test` in filenames.

## Run (single command)

```powershell
py src/improve_submission.py `
  --train-file data/train.csv `
  --test-file data/test.csv `
  --n-splits 2 `
  --confirm-splits 3 `
  --batch-size 8 `
  --parallel-jobs -1 `
  --min-improvement 0.0005 `
  --target-improvements 3 `
  --output-dir outputs
```

## Naming and benchmark behavior

- New files are numbered: `submission1.csv`, `submission2.csv`, `submission3.csv`, ...
- Higher number means newer and better (based on internal CV benchmark).
- The script only writes a new numbered file when it beats the current best benchmark.
- Benchmark is persisted in `.improver_state.json`.

If numbered files already exist but `.improver_state.json` does not, initialize once:

```powershell
py src/improve_submission.py --baseline-override <best_cv_auc_so_far>
```

## Outputs

- `outputs/` contains only `submission*.csv` files.

## Distributed Mode (Controller + Joinable Workers)

Use this when you want other PCs to help compute.

### On your main PC (controller)

Run one command from repo root:

```powershell
.\start_controller.ps1
```

This does the following automatically:
- creates `.venv` if needed
- installs Python dependencies
- starts distributed controller API on port `8765`
- starts LAN discovery responder on UDP `50555`
- serves live dashboard at `http://<your-ip>:8765/dashboard`

### On each helper PC (worker)

Clone the same repo and run one command:

```powershell
.\join_worker.ps1
```

This script:
- installs Python via `winget` if missing
- creates `.venv` and installs dependencies
- auto-discovers your controller over LAN
- registers worker and starts contributing compute jobs

### Dashboard

Open:

`http://<controller-ip>:8765/dashboard`

The dashboard shows:
- number of joined PCs
- pending/inflight jobs
- best benchmark AUC
- new submission files created this session
- recent event logs

### Distributed scripts

- `src/distributed_controller.py`
- `src/distributed_worker.py`
- `src/distributed_smoke_test.py`
- `start_controller.ps1`
- `join_worker.ps1`

### Notes

- All machines should use the same `data/train.csv` and `data/test.csv` contents.
- Numbered file writing (`submission1.csv`, `submission2.csv`, ...) stays centralized on the controller.
- If LAN discovery fails, workers can connect directly with:

```powershell
.\.venv\Scripts\python.exe .\src\distributed_worker.py --controller-url http://<controller-ip>:8765
```
