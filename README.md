# Labour Market Status Hackathon Starter

This workspace is set up to help you generate a strong baseline submission for the Round 9 employment prediction challenge.

## 1) Project structure

- `data/` place competition files here (train and test).
- `src/train_and_submit.py` end-to-end training + inference pipeline.
- `outputs/` generated models, CV metrics, OOF predictions, and submission file.
- `notebooks/hackathon_starter.ipynb` quick experimentation notebook.

## 2) Environment setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 3) Add data

Put your official competition files in `data/` and ensure filenames contain `train` and `test`.

Examples:
- `data/labour_train.csv`
- `data/labour_test.csv`

If your files have different names or paths, pass them explicitly via command line arguments.

## 4) Train and generate submission

```powershell
python src/train_and_submit.py
```

For a very fast first submission during the hackathon:

```powershell
py src/quick_submit.py
```

This writes `outputs/submission_quick.csv` using a compact feature set and a lightweight logistic model.

Optional explicit arguments:

```powershell
python src/train_and_submit.py `
  --train-file data/my_train.csv `
  --test-file data/my_test.csv `
  --id-col anonymised_id `
  --target-col employed_status `
  --max-round 9 `
  --n-splits 5 `
  --output-dir outputs
```

## 5) Outputs

- `outputs/submission.csv` upload this to the competition.
- `outputs/cv_metrics.json` fold-wise AUC + summary.
- `outputs/oof_predictions.csv` out-of-fold probabilities for diagnostics.
- `outputs/model_ohe.joblib` and `outputs/model_ord.joblib` trained models.

## 6) Submission Metrics Frontend

A React dashboard is available in `frontend/` to track all submission CSV files and their metrics.

### Run locally

```powershell
cd frontend
npm.cmd install
npm.cmd run dev
```

### Build for production

```powershell
cd frontend
npm.cmd run build
```

The frontend automatically:
- scans `outputs/` for files matching `submission*.csv`
- computes per-file metrics and scores
- checks whether each new submission improves on the previous one
- copies submission files to `frontend/public/submissions/` so they can be downloaded from the UI

The frontend also includes a **Real Leaderboard Score** entry section:
- enter the true Kaggle score for each file
- keep proxy score visible next to real score
- save/clear local score overrides in-browser

## 7) Auto-Improve Pipeline

Use this pipeline when you want the code to keep searching candidates until it finds a better model score than your chosen baseline.

### Quick search (recommended during hackathon)

```powershell
py src/auto_improve_pipeline.py `
  --train-file data/train.csv `
  --test-file data/test.csv `
  --profile quick `
  --n-splits 2 `
  --baseline-override 0.62528 `
  --min-improvement 0.0005 `
  --output-dir outputs
```

### Full search (slower, deeper)

```powershell
py src/auto_improve_pipeline.py `
  --train-file data/train.csv `
  --test-file data/test.csv `
  --profile full `
  --n-splits 3 `
  --baseline-metrics-file outputs/cv_metrics.json `
  --min-improvement 0.002 `
  --output-dir outputs
```

The script writes:
- `outputs/submission_<winner>_<timestamp>.csv`
- `outputs/oof_<winner>_<timestamp>.csv`
- `outputs/auto_report_<winner>_<timestamp>.json`

If no model beats the target baseline, it writes a `outputs/auto_report_no_improvement_<timestamp>.json` report.

## 8) Continuous Improve Pipeline (Runs Until Stopped)

Use this for systematic continuous search. It keeps generating random candidate models and only writes a new submission when it finds improvement over the current best.

```powershell
py src/continuous_improve_pipeline.py `
  --train-file data/train.csv `
  --test-file data/test.csv `
  --n-splits 2 `
  --confirm-splits 3 `
  --batch-size 4 `
  --parallel-jobs -1 `
  --min-improvement 0.0005 `
  --leaderboard-scores-file frontend/src/data/leaderboard_scores.json `
  --output-dir outputs
```

Stop with `Ctrl+C`.

Useful options:
- `--worker-name server_a` to label outputs/state from a given machine
- `--parallel-jobs -1` to use all CPU cores
- `--batch-size 8` to screen more candidates per loop when CPU allows
- `--max-iterations 50` for a bounded run (default `0` means forever)
- `--state-file outputs/continuous_state.json` to track live progress

### Two-server setup

For non-overlapping divide-and-conquer, run both servers with the same `--seed` and `--server-count`, and different `--server-index` values.

If both servers can access the same filesystem path, also pass the same `--coordination-file` path for extra duplicate protection.

Server 1 (index 0):

```powershell
py src/continuous_improve_pipeline.py `
  --train-file data/train.csv `
  --test-file data/test.csv `
  --seed 42 `
  --server-count 2 `
  --server-index 0 `
  --n-splits 2 `
  --confirm-splits 3 `
  --batch-size 4 `
  --parallel-jobs -1 `
  --worker-name server_a `
  --coordination-file outputs/continuous_coordination.json `
  --leaderboard-scores-file frontend/src/data/leaderboard_scores.json `
  --output-dir outputs
```

Server 2 (index 1):

```powershell
py src/continuous_improve_pipeline.py `
  --train-file data/train.csv `
  --test-file data/test.csv `
  --seed 42 `
  --server-count 2 `
  --server-index 1 `
  --n-splits 2 `
  --confirm-splits 3 `
  --batch-size 4 `
  --parallel-jobs -1 `
  --worker-name server_b `
  --coordination-file outputs/continuous_coordination.json `
  --leaderboard-scores-file frontend/src/data/leaderboard_scores.json `
  --output-dir outputs
```

How overlap is prevented:
- Deterministic sharding: each server only samples candidate IDs assigned to its shard (`server-index` modulo `server-count`).
- Shared candidate claims: when `server-count > 1`, candidates are claimed in `continuous_coordination.json` before evaluation.
- Prediction hash dedupe: duplicate prediction vectors are still skipped before writing files.

Scale to more servers:
- Set `--server-count N` on every server.
- Assign each server a unique `--server-index` from `0` to `N-1`.
- Keep `--seed` and `--coordination-file` identical across all servers.

Example for 4 servers: use indices `0`, `1`, `2`, `3` with `--server-count 4`.

This path does not require a hardcoded benchmark override. It uses the real leaderboard scores file first, then local metrics only as fallback.

## 9) SSH Cluster Orchestrator (One Frontend+Backend, Many Backend Workers)

Use `ops/cluster_orchestrator.py` to control all servers through SSH with deterministic backend sharding (no overlap).

Install orchestrator dependency locally:

```powershell
pip install paramiko
```

Sync local project files to all servers (no git remote required):

```powershell
py ops/cluster_orchestrator.py sync --config ops/cluster_nodes.json --local-project-dir .
```

Create your node config:

```powershell
copy ops\cluster_nodes.example.json ops\cluster_nodes.json
```

Edit `ops/cluster_nodes.json`:
- exactly one node must have `"frontend": true`
- every backend worker must have `"backend": true`
- new servers are added by appending another node object with `"backend": true`

Start all services:

```powershell
py ops/cluster_orchestrator.py start --config ops/cluster_nodes.json
```

Collect runtime status + log tails from every server into frontend data:

```powershell
py ops/cluster_orchestrator.py sync-logs --config ops/cluster_nodes.json --server-logs-file frontend/src/data/server_logs.json --log-lines 60
```

This powers the **Server Runtime Logs** table in the frontend.

Optional first-time bootstrap (create venv, install Python deps, install frontend deps on frontend node):

```powershell
py ops/cluster_orchestrator.py bootstrap --config ops/cluster_nodes.json
```

If project code is not present on servers yet, bootstrap with git clone:

```powershell
py ops/cluster_orchestrator.py bootstrap --config ops/cluster_nodes.json --repo-url <YOUR_GIT_URL> --repo-branch main
```

Check status:

```powershell
py ops/cluster_orchestrator.py status --config ops/cluster_nodes.json
```

Stop all services:

```powershell
py ops/cluster_orchestrator.py stop --config ops/cluster_nodes.json
```

Security notes:
- do not hardcode passwords into config files
- the script prompts for each node password securely, unless env vars are set
- optional env var format per node name: `CLUSTER_SSH_PASSWORD_<NODE_NAME_UPPER_SNAKE>`
  - example for node name `worker_2`: `CLUSTER_SSH_PASSWORD_WORKER_2`

How no-overlap is guaranteed in this mode:
- backend shard count = number of nodes with `"backend": true`
- each backend node gets a unique shard index automatically
- all workers use the same seed and server-count with different server-index values
- therefore candidate generation is partitioned deterministically across workers

How to add another backend server:
- append a new node object in `ops/cluster_nodes.json` with `"backend": true` and unique `"name"`
- run `sync` once for that node set (or all nodes), then `bootstrap`
- run `start` again; shard count is recalculated automatically

## 10) Remote Frontend Hosting (Static Mode)

Use static mode when `npm` is not installed on the remote server.

1) Build frontend locally:

powershell: `cd frontend`
powershell: `npm run build`

2) Sync project to servers:

powershell: `cd ..`
powershell: `py ops/cluster_orchestrator.py sync --config ops/cluster_nodes.json --local-project-dir .`

3) Start backend workers:

powershell: `py ops/cluster_orchestrator.py start --config ops/cluster_nodes.json`

4) Start static frontend directly on primary server:

powershell: `py ops/start_primary_frontend_static.py`

5) Refresh server log snapshot for dashboard visibility:

powershell: `py ops/pull_server_runtime_logs.py --config ops/cluster_nodes.json --out frontend/src/data/server_logs.json --log-lines 80`

Expected frontend URL:
- `http://nightmare.cs.uct.ac.za:5173`

If direct browser access times out from another PC, use SSH tunnel:

windows/mac/linux terminal:
`ssh -L 5173:localhost:5173 mtmluv001@nightmare.cs.uct.ac.za`

Then open in browser on that PC:
- `http://localhost:5173`

## 11) Render Frontend Locally From Remote Servers

If remote browser access is blocked, use this mode to fetch remote outputs/logs to your local machine and render locally.

One-time dependency:

```powershell
pip install paramiko
```

Refresh local frontend data from remote servers:

```powershell
$env:CLUSTER_SSH_PASSWORD_PRIMARY='<primary_password>'
$env:CLUSTER_SSH_PASSWORD_WORKER_2='<worker2_password>'
py ops/render_frontend_locally_from_remote.py --config ops/cluster_nodes.json --project-root .
```

Then run local frontend and open it in browser:

```powershell
cd frontend
npm run dev
```

Open:
- `http://localhost:5173`

Single command to fetch and immediately serve local frontend:

```powershell
py ops/render_frontend_locally_from_remote.py --config ops/cluster_nodes.json --project-root . --serve --port 5173
```

When it finds improvement, it writes versioned files:
- `outputs/submission_cont_<...>.csv`
- `outputs/oof_cont_<...>.csv`
- `outputs/continuous_report_cont_<...>.json`

Submission format is:

```csv
anonymised_id,employed_status
ID_10,0.83
ID_100,0.12
ID_1000,0.47
```

## What the baseline does

- Automatically detects ID and target columns (with override options).
- Handles either:
  - wide panel style (columns like round-specific variables), or
  - long panel style (multiple rows per participant with a round column).
- Engineers time-aware features per participant:
  - first/last value,
  - change and trend for numeric variables,
  - mode and change flags for categorical variables,
  - panel history span features.
- Trains two complementary models and averages predictions:
  - Logistic regression over one-hot encoded features,
  - Histogram Gradient Boosted Trees over ordinal-encoded features.
- Evaluates using stratified CV AUC.

## Fast ways to improve score

1. Add lag-aware interaction features (e.g., education x province x last employment state).
2. Add external target encoding with fold-safe implementation.
3. Try CatBoost/LightGBM and blend with this baseline.
4. Tune folds so all records for each participant stay in one split if duplicates exist.
5. Calibrate and clip probabilities only after CV checks.

## Notes for final report (<=15 pages)

Include:
- Data quality and missingness analysis.
- Feature engineering strategy for longitudinal behavior.
- Model comparison table with CV AUC and variance.
- Error analysis by subgroup (gender, province, education, age band).
- Discussion on public/private leaderboard overfitting risks.
