from __future__ import annotations

import argparse
import json
import socket
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from continuous_improve_pipeline import (
    auto_find_data_paths,
    build_panel_features,
    evaluate_candidate_config,
    infer_id_col,
    infer_target_col,
    load_table,
    maybe_to_wide_panel,
)


def discover_controller(discovery_port: int, timeout_seconds: float) -> Optional[str]:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    sock.settimeout(timeout_seconds)
    try:
        sock.sendto(b"DISCOVER_IMPROVER", ("255.255.255.255", discovery_port))
        data, _ = sock.recvfrom(2048)
        payload = json.loads(data.decode("utf-8"))
        return str(payload.get("controller_url", "")) or None
    except Exception:
        return None
    finally:
        sock.close()


def http_post_json(url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=body, method="POST")
    req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=30) as resp:
        raw = resp.read()
    return json.loads(raw.decode("utf-8"))


def prepare_data(data_dir: Path, train_file: Optional[Path], test_file: Optional[Path], max_round: int):
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

    X_train = X_train.drop(columns=[id_col])
    _ = X_test.drop(columns=[id_col])
    y = y.fillna(0).astype(int)
    return X_train, y


def run_worker(args: argparse.Namespace) -> None:
    controller_url = args.controller_url
    if not controller_url:
        controller_url = discover_controller(args.discovery_port, args.discovery_timeout)
        if not controller_url:
            raise RuntimeError("Could not discover controller on LAN. Provide --controller-url explicitly.")

    print(f"Controller: {controller_url}")
    X_train, y = prepare_data(args.data_dir, args.train_file, args.test_file, args.max_round)

    reg = http_post_json(f"{controller_url}/api/register", {"worker_name": args.worker_name})
    worker_id = str(reg["worker_id"])
    poll_seconds = float(reg.get("poll_seconds", 1.0))
    print(f"Registered worker_id={worker_id[:8]}")

    jobs_done = 0
    while True:
        try:
            job_res = http_post_json(f"{controller_url}/api/get_job", {"worker_id": worker_id})
            job = job_res.get("job")
            if not job:
                time.sleep(poll_seconds)
                continue

            result = evaluate_candidate_config(
                candidate_name=str(job["candidate_name"]),
                params=dict(job["params"]),
                X_train=X_train,
                y=y,
                n_splits=int(job["cv_splits"]),
                cv_seed=int(job["cv_seed"]),
                need_oof=False,
            )
            submit_payload = {
                "worker_id": worker_id,
                "job_id": str(job["job_id"]),
                "mean_auc": float(result["mean_auc"]),
                "std_auc": float(result["std_auc"]),
                "fold_scores": [float(x) for x in result["fold_scores"]],
            }
            submit_res = http_post_json(f"{controller_url}/api/submit_result", submit_payload)
            jobs_done += 1
            accepted = bool(submit_res.get("accepted", False))
            if accepted:
                print(f"[{jobs_done}] Accepted: new file {submit_res.get('file')} | best={submit_res.get('new_best'):.6f}")
            else:
                print(f"[{jobs_done}] Submitted: not accepted")

            if args.max_jobs > 0 and jobs_done >= args.max_jobs:
                print("Reached --max-jobs limit. Exiting.")
                return
        except urllib.error.URLError as exc:
            print(f"Connection issue: {exc}. Retrying...")
            time.sleep(2.0)
        except KeyboardInterrupt:
            print("Stopped by user.")
            return
        except Exception as exc:
            print(f"Worker error: {exc}")
            time.sleep(1.5)



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Distributed worker for submission improver")
    parser.add_argument("--controller-url", type=str, default="")
    parser.add_argument("--discovery-port", type=int, default=50555)
    parser.add_argument("--discovery-timeout", type=float, default=4.0)
    parser.add_argument("--worker-name", type=str, default=socket.gethostname())
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--train-file", type=Path, default=None)
    parser.add_argument("--test-file", type=Path, default=None)
    parser.add_argument("--max-round", type=int, default=9)
    parser.add_argument("--max-jobs", type=int, default=0, help="0 means run forever")
    return parser.parse_args()


if __name__ == "__main__":
    run_worker(parse_args())
