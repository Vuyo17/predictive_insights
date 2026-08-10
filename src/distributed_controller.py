from __future__ import annotations

import argparse
import json
import queue
import socket
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from continuous_improve_pipeline import (
    auto_find_data_paths,
    build_panel_features,
    build_weighted_models,
    fit_predict_full,
    infer_id_col,
    infer_target_col,
    load_existing_prediction_hashes,
    load_table,
    maybe_to_wide_panel,
    next_submission_index,
    prediction_hash,
    sample_candidate,
)


@dataclass
class Job:
    job_id: str
    candidate_name: str
    params: Dict[str, float]
    cv_splits: int
    cv_seed: int
    created_at: float


class ControllerState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.workers: Dict[str, Dict[str, Any]] = {}
        self.jobs_pending: queue.Queue[Job] = queue.Queue()
        self.jobs_inflight: Dict[str, Job] = {}
        self.events: deque[Dict[str, Any]] = deque(maxlen=200)
        self.created_files: list[str] = []
        self.best_cv_auc: float = 0.0
        self.min_improvement: float = 0.0005
        self.next_index: int = 1
        self.seen_hashes: set[str] = set()
        self.seed: int = 42
        self.cv_splits: int = 3
        self.batch_size: int = 8
        self.running: bool = True

        self.X_train = None
        self.y = None
        self.X_test = None
        self.test_ids = None
        self.best_candidate_meta: Optional[Dict[str, object]] = None
        self.output_dir = Path("outputs")
        self.benchmark_state_file = Path(".improver_state.json")
        self.anchor_predictions: Optional[np.ndarray] = None
        self.anchor_name: str = ""
        self.anchor_max_mae: float = 0.08
        self.anchor_min_rank_corr: float = 0.985

    def log_event(self, kind: str, message: str, extra: Optional[Dict[str, Any]] = None) -> None:
        payload = {
            "ts": datetime.now().isoformat(),
            "kind": kind,
            "message": message,
            "extra": extra or {},
        }
        with self.lock:
            self.events.appendleft(payload)


STATE = ControllerState()


def _rank_corr(a: np.ndarray, b: np.ndarray) -> float:
    a_rank = pd.Series(a).rank(method="average").to_numpy(dtype=float)
    b_rank = pd.Series(b).rank(method="average").to_numpy(dtype=float)
    if a_rank.std() == 0.0 or b_rank.std() == 0.0:
        return 0.0
    return float(np.corrcoef(a_rank, b_rank)[0, 1])


def load_anchor_predictions(anchor_file: Path, test_ids: pd.Series) -> Optional[np.ndarray]:
    if not anchor_file.exists():
        return None

    df = pd.read_csv(anchor_file)
    if "anonymised_id" not in df.columns or "employed_status" not in df.columns:
        return None

    anchor = df[["anonymised_id", "employed_status"]].copy()
    merged = pd.DataFrame({"anonymised_id": test_ids}).merge(anchor, on="anonymised_id", how="left")
    if merged["employed_status"].isna().any():
        return None
    return merged["employed_status"].to_numpy(dtype=float)


def resolve_lan_ip() -> str:
    """Resolve a likely LAN IP for sharing the controller URL."""
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # No outbound packet is sent; this picks the local interface.
        probe.connect(("8.8.8.8", 80))
        return str(probe.getsockname()[0])
    except OSError:
        return "127.0.0.1"
    finally:
        probe.close()


def resolve_advertised_host(bind_host: str) -> str:
    host = (bind_host or "").strip()
    if host in {"", "0.0.0.0", "::"}:
        return resolve_lan_ip()
    return host


def load_benchmark_state(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            return payload
    except (json.JSONDecodeError, OSError):
        return {}
    return {}


def save_benchmark_state(path: Path, best_cv_auc: float, last_submission_index: int) -> None:
    payload = {
        "updated_at": datetime.now().isoformat(),
        "best_cv_auc": float(best_cv_auc),
        "last_submission_index": int(last_submission_index),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def prepare_data(data_dir: Path, train_file: Optional[Path], test_file: Optional[Path], max_round: int) -> None:
    if train_file is not None and test_file is not None:
        train_path = train_file
        test_path = test_file
    else:
        train_path, test_path = auto_find_data_paths(data_dir)

    train_df = load_table(train_path)
    test_df = load_table(test_path)

    id_col = infer_id_col(train_df)
    target_col = infer_target_col(train_df)

    train_wide = maybe_to_wide_panel(train_df, id_col=id_col, target_col=target_col)
    test_wide = maybe_to_wide_panel(test_df, id_col=id_col, target_col=None)

    X_train, y = build_panel_features(train_wide, id_col=id_col, target_col=target_col, max_round=max_round)
    X_test, _ = build_panel_features(test_wide, id_col=id_col, target_col=None, max_round=max_round)

    if y is None:
        raise ValueError("Target column missing after feature engineering")

    test_ids = X_test[id_col].copy()
    X_train = X_train.drop(columns=[id_col])
    X_test = X_test.drop(columns=[id_col])
    y = y.fillna(0).astype(int)

    STATE.X_train = X_train
    STATE.y = y
    STATE.X_test = X_test
    STATE.test_ids = test_ids


def job_producer_loop() -> None:
    while STATE.running:
        try:
            with STATE.lock:
                worker_count = max(1, len(STATE.workers))
                target_queue = worker_count * 3

            if STATE.jobs_pending.qsize() >= target_queue:
                time.sleep(0.2)
                continue

            for _ in range(STATE.batch_size):
                candidate_id = int(time.time() * 1000) % 10_000_000
                rng_seed = STATE.seed + candidate_id
                rng = np.random.default_rng(rng_seed)
                py_rng_seed = int(rng.integers(1, 10_000_000))
                import random

                local_rng = random.Random(py_rng_seed)
                candidate_name, params, _ = sample_candidate(local_rng, STATE.X_train, STATE.best_candidate_meta)
                job = Job(
                    job_id=str(uuid.uuid4()),
                    candidate_name=candidate_name,
                    params=params,
                    cv_splits=STATE.cv_splits,
                    cv_seed=rng_seed,
                    created_at=time.time(),
                )
                STATE.jobs_pending.put(job)

            time.sleep(0.1)
        except Exception as exc:  # pragma: no cover
            STATE.log_event("error", f"Job producer error: {exc}")
            time.sleep(1.0)


def udp_discovery_loop(host: str, port: int, controller_url: str) -> None:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((host, port))

    while STATE.running:
        try:
            data, addr = sock.recvfrom(2048)
            if data.decode("utf-8", errors="ignore").strip() != "DISCOVER_IMPROVER":
                continue
            payload = json.dumps({"controller_url": controller_url}).encode("utf-8")
            sock.sendto(payload, addr)
        except Exception:
            continue


class Handler(BaseHTTPRequestHandler):
    def _json(self, code: int, payload: Dict[str, Any]) -> None:
        raw = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _read_json(self) -> Dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length > 0 else b"{}"
        return json.loads(raw.decode("utf-8"))

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/api/status":
            with STATE.lock:
                workers = [
                    {
                        "worker_id": wid,
                        "name": meta.get("name", "worker"),
                        "last_seen": meta.get("last_seen"),
                        "jobs_done": meta.get("jobs_done", 0),
                    }
                    for wid, meta in STATE.workers.items()
                ]
                payload = {
                    "workers": workers,
                    "worker_count": len(workers),
                    "best_cv_auc": STATE.best_cv_auc,
                    "pending_jobs": STATE.jobs_pending.qsize(),
                    "inflight_jobs": len(STATE.jobs_inflight),
                    "created_files": STATE.created_files[-30:],
                    "events": list(STATE.events)[:30],
                }
            self._json(HTTPStatus.OK, payload)
            return

        if self.path == "/" or self.path == "/dashboard":
            html = """
<!doctype html>
<html>
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width,initial-scale=1\" />
  <title>Distributed Improver Dashboard</title>
  <style>
    body { font-family: Segoe UI, Arial, sans-serif; margin: 20px; background: #f7f9fc; color: #1f2937; }
    .card { background: #fff; border-radius: 12px; padding: 16px; margin-bottom: 14px; box-shadow: 0 2px 10px rgba(0,0,0,0.06); }
    h1 { margin-top: 0; }
    ul { margin: 0; padding-left: 20px; }
    .mono { font-family: Consolas, monospace; }
  </style>
</head>
<body>
  <h1>Distributed Improver</h1>
  <div class=\"card\"><strong>Workers:</strong> <span id=\"wc\">0</span> | <strong>Best CV AUC:</strong> <span id=\"best\">0</span> | <strong>Pending Jobs:</strong> <span id=\"pq\">0</span></div>
  <div class=\"card\"><h3>Connected PCs</h3><ul id=\"workers\"></ul></div>
  <div class=\"card\"><h3>New Submission Files</h3><ul id=\"files\"></ul></div>
  <div class=\"card\"><h3>Recent Events</h3><ul id=\"events\"></ul></div>
  <script>
    async function refresh() {
      const r = await fetch('/api/status');
      const s = await r.json();
      document.getElementById('wc').textContent = s.worker_count;
      document.getElementById('best').textContent = Number(s.best_cv_auc || 0).toFixed(6);
      document.getElementById('pq').textContent = s.pending_jobs;

      const workers = document.getElementById('workers');
      workers.innerHTML = '';
      s.workers.forEach(w => {
        const li = document.createElement('li');
        li.textContent = `${w.name} (${w.worker_id.slice(0,8)}) jobs=${w.jobs_done}`;
        workers.appendChild(li);
      });

      const files = document.getElementById('files');
      files.innerHTML = '';
      s.created_files.slice().reverse().forEach(f => {
        const li = document.createElement('li');
        li.textContent = f;
        li.className = 'mono';
        files.appendChild(li);
      });

      const events = document.getElementById('events');
      events.innerHTML = '';
      s.events.forEach(e => {
        const li = document.createElement('li');
        li.textContent = `${e.ts} | ${e.kind} | ${e.message}`;
        events.appendChild(li);
      });
    }
    setInterval(refresh, 2000);
    refresh();
  </script>
</body>
</html>
""".strip()
            raw = html.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
            return

        self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:  # noqa: N802
        try:
            if self.path == "/api/register":
                payload = self._read_json()
                worker_name = str(payload.get("worker_name", "worker"))
                worker_id = str(uuid.uuid4())
                with STATE.lock:
                    STATE.workers[worker_id] = {
                        "name": worker_name,
                        "last_seen": datetime.now().isoformat(),
                        "jobs_done": 0,
                    }
                STATE.log_event("worker_join", f"{worker_name} joined ({worker_id[:8]})")
                self._json(HTTPStatus.OK, {"worker_id": worker_id, "poll_seconds": 1.0})
                return

            if self.path == "/api/get_job":
                payload = self._read_json()
                worker_id = str(payload.get("worker_id", ""))
                if worker_id not in STATE.workers:
                    self._json(HTTPStatus.UNAUTHORIZED, {"error": "unknown worker"})
                    return

                with STATE.lock:
                    STATE.workers[worker_id]["last_seen"] = datetime.now().isoformat()

                try:
                    job = STATE.jobs_pending.get_nowait()
                except queue.Empty:
                    self._json(HTTPStatus.OK, {"job": None})
                    return

                with STATE.lock:
                    STATE.jobs_inflight[job.job_id] = job

                self._json(
                    HTTPStatus.OK,
                    {
                        "job": {
                            "job_id": job.job_id,
                            "candidate_name": job.candidate_name,
                            "params": job.params,
                            "cv_splits": job.cv_splits,
                            "cv_seed": job.cv_seed,
                            "current_best_cv_auc": STATE.best_cv_auc,
                            "min_improvement": STATE.min_improvement,
                        }
                    },
                )
                return

            if self.path == "/api/submit_result":
                payload = self._read_json()
                worker_id = str(payload.get("worker_id", ""))
                job_id = str(payload.get("job_id", ""))
                mean_auc = float(payload.get("mean_auc"))
                std_auc = float(payload.get("std_auc"))
                fold_scores = payload.get("fold_scores", [])

                if worker_id not in STATE.workers:
                    self._json(HTTPStatus.UNAUTHORIZED, {"error": "unknown worker"})
                    return

                with STATE.lock:
                    job = STATE.jobs_inflight.pop(job_id, None)
                    if job is None:
                        self._json(HTTPStatus.BAD_REQUEST, {"error": "unknown job"})
                        return
                    STATE.workers[worker_id]["jobs_done"] = int(STATE.workers[worker_id].get("jobs_done", 0)) + 1
                    STATE.workers[worker_id]["last_seen"] = datetime.now().isoformat()

                improved = mean_auc >= (STATE.best_cv_auc + STATE.min_improvement)
                if improved:
                    weighted_models = build_weighted_models(STATE.X_train, job.candidate_name, job.params)
                    test_pred = fit_predict_full(weighted_models, STATE.X_train, STATE.y, STATE.X_test)
                    pred_hash = prediction_hash(test_pred)
                    if pred_hash in STATE.seen_hashes:
                        STATE.log_event(
                            "duplicate",
                            f"Worker {worker_id[:8]} found duplicate prediction; skipped",
                            {"job_id": job_id},
                        )
                        self._json(HTTPStatus.OK, {"accepted": False, "reason": "duplicate_prediction"})
                        return

                    if STATE.anchor_predictions is not None:
                        anchor_mae = float(np.mean(np.abs(test_pred - STATE.anchor_predictions)))
                        anchor_rank_corr = _rank_corr(test_pred, STATE.anchor_predictions)
                        if anchor_mae > STATE.anchor_max_mae or anchor_rank_corr < STATE.anchor_min_rank_corr:
                            STATE.log_event(
                                "anchor_reject",
                                (
                                    f"Worker {worker_id[:8]} candidate rejected by anchor guard "
                                    f"(mae={anchor_mae:.6f}, rank_corr={anchor_rank_corr:.6f})"
                                ),
                                {
                                    "job_id": job_id,
                                    "anchor": STATE.anchor_name,
                                    "anchor_mae": anchor_mae,
                                    "anchor_rank_corr": anchor_rank_corr,
                                },
                            )
                            self._json(
                                HTTPStatus.OK,
                                {
                                    "accepted": False,
                                    "reason": "anchor_guard",
                                    "anchor_mae": anchor_mae,
                                    "anchor_rank_corr": anchor_rank_corr,
                                },
                            )
                            return

                    STATE.output_dir.mkdir(parents=True, exist_ok=True)
                    out_path = STATE.output_dir / f"submission{STATE.next_index}.csv"
                    pd.DataFrame(
                        {"anonymised_id": STATE.test_ids, "employed_status": test_pred}
                    ).to_csv(out_path, index=False)

                    STATE.seen_hashes.add(pred_hash)
                    STATE.best_cv_auc = mean_auc
                    STATE.best_candidate_meta = {
                        "candidate_name": job.candidate_name,
                        "params": job.params,
                        "mean_auc": mean_auc,
                        "std_auc": std_auc,
                    }
                    STATE.created_files.append(out_path.name)
                    save_benchmark_state(STATE.benchmark_state_file, STATE.best_cv_auc, STATE.next_index)
                    STATE.next_index += 1

                    STATE.log_event(
                        "improvement",
                        f"New best {mean_auc:.6f} from {job.candidate_name}; wrote {out_path.name}",
                        {"fold_scores": fold_scores},
                    )
                    self._json(HTTPStatus.OK, {"accepted": True, "new_best": mean_auc, "file": out_path.name})
                    return

                STATE.log_event(
                    "result",
                    f"Worker {worker_id[:8]} result {mean_auc:.6f} did not beat {STATE.best_cv_auc:.6f}",
                )
                self._json(HTTPStatus.OK, {"accepted": False, "new_best": STATE.best_cv_auc})
                return

            self.send_error(HTTPStatus.NOT_FOUND)
        except Exception as exc:
            self._json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)})



def run_controller(args: argparse.Namespace) -> None:
    STATE.output_dir = args.output_dir
    STATE.min_improvement = args.min_improvement
    STATE.seed = args.seed
    STATE.cv_splits = args.cv_splits
    STATE.batch_size = args.batch_size
    STATE.benchmark_state_file = args.benchmark_state_file

    prepare_data(args.data_dir, args.train_file, args.test_file, args.max_round)
    STATE.anchor_max_mae = float(args.anchor_max_mae)
    STATE.anchor_min_rank_corr = float(args.anchor_min_rank_corr)

    baseline = args.baseline_override if args.baseline_override is not None else args.fallback_baseline
    bench = load_benchmark_state(args.benchmark_state_file)
    persisted_best = float(bench.get("best_cv_auc", baseline))
    STATE.best_cv_auc = max(float(baseline), persisted_best)
    STATE.next_index = next_submission_index(args.output_dir)
    STATE.seen_hashes = load_existing_prediction_hashes(args.output_dir)

    if args.anchor_file is not None:
        anchor_pred = load_anchor_predictions(args.anchor_file, STATE.test_ids)
        if anchor_pred is None:
            STATE.log_event(
                "anchor",
                f"Anchor file not loaded: {args.anchor_file}. Continuing without anchor guard.",
            )
        else:
            STATE.anchor_predictions = anchor_pred
            STATE.anchor_name = args.anchor_file.name
            STATE.log_event(
                "anchor",
                (
                    f"Anchor loaded from {args.anchor_file.name} "
                    f"(max_mae={STATE.anchor_max_mae:.4f}, min_rank_corr={STATE.anchor_min_rank_corr:.4f})"
                ),
            )

    STATE.log_event("startup", f"Controller started with best={STATE.best_cv_auc:.6f} next=submission{STATE.next_index}.csv")

    producer = threading.Thread(target=job_producer_loop, daemon=True)
    producer.start()

    bind_url = f"http://{args.host}:{args.port}"
    share_host = resolve_advertised_host(args.host)
    controller_url = f"http://{share_host}:{args.port}"
    discover_thread = threading.Thread(
        target=udp_discovery_loop,
        args=(args.discovery_host, args.discovery_port, controller_url),
        daemon=True,
    )
    discover_thread.start()

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Controller bind: {bind_url}")
    print(f"Controller URL: {controller_url}")
    print(f"Dashboard (local): http://127.0.0.1:{args.port}/dashboard")
    if share_host != "127.0.0.1":
        print(f"Dashboard (LAN): {controller_url}/dashboard")
    print(f"UDP discovery on {args.discovery_host}:{args.discovery_port}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        pass
    finally:
        STATE.running = False
        server.shutdown()



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Distributed controller for submission improver")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--discovery-host", type=str, default="0.0.0.0")
    parser.add_argument("--discovery-port", type=int, default=50555)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--train-file", type=Path, default=None)
    parser.add_argument("--test-file", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--benchmark-state-file", type=Path, default=Path(".improver_state.json"))
    parser.add_argument("--fallback-baseline", type=float, default=0.0)
    parser.add_argument("--baseline-override", type=float, default=None)
    parser.add_argument("--min-improvement", type=float, default=0.0005)
    parser.add_argument("--cv-splits", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-round", type=int, default=9)
    parser.add_argument(
        "--anchor-file",
        type=Path,
        default=Path("outputs/submission2.csv"),
        help="Known good submission used as public-score anchor.",
    )
    parser.add_argument(
        "--anchor-max-mae",
        type=float,
        default=0.08,
        help="Reject improvements whose mean absolute deviation from anchor is above this threshold.",
    )
    parser.add_argument(
        "--anchor-min-rank-corr",
        type=float,
        default=0.985,
        help="Reject improvements whose rank-correlation to anchor falls below this threshold.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run_controller(parse_args())
