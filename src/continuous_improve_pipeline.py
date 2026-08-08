from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
from glob import glob
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

from train_and_submit import (
    auto_find_data_paths,
    build_panel_features,
    infer_id_col,
    infer_target_col,
    load_table,
    maybe_to_wide_panel,
)


def split_columns(X: pd.DataFrame) -> Tuple[List[str], List[str]]:
    numeric_features = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    categorical_features = [c for c in X.columns if c not in numeric_features]
    return numeric_features, categorical_features


def build_hgb_model(
    X: pd.DataFrame,
    learning_rate: float,
    max_depth: int,
    max_iter: int,
    min_samples_leaf: int,
    l2_regularization: float,
    random_state: int,
) -> Pipeline:
    numeric_features, categorical_features = split_columns(X)

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))]),
                numeric_features,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "encoder",
                            OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                        ),
                    ]
                ),
                categorical_features,
            ),
        ],
        remainder="drop",
    )

    model = HistGradientBoostingClassifier(
        learning_rate=learning_rate,
        max_depth=max_depth,
        max_iter=max_iter,
        min_samples_leaf=min_samples_leaf,
        l2_regularization=l2_regularization,
        random_state=random_state,
    )

    return Pipeline(steps=[("prep", preprocessor), ("model", model)])


def build_logreg_model(
    X: pd.DataFrame,
    c_value: float,
    max_iter: int,
    class_weight: Optional[str],
    random_state: int,
) -> Pipeline:
    numeric_features, categorical_features = split_columns(X)

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))]),
                numeric_features,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "encoder",
                            OneHotEncoder(handle_unknown="ignore", min_frequency=15),
                        ),
                    ]
                ),
                categorical_features,
            ),
        ],
        remainder="drop",
    )

    model = LogisticRegression(
        C=c_value,
        max_iter=max_iter,
        solver="saga",
        class_weight=class_weight,
        random_state=random_state,
    )

    return Pipeline(steps=[("prep", preprocessor), ("model", model)])


def bounded_int(value: int, low: int, high: int) -> int:
    return max(low, min(high, int(value)))


def bounded_float(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


def perturb_float(rng: random.Random, center: float, scale: float, low: float, high: float) -> float:
    return round(bounded_float(rng.gauss(center, scale), low, high), 6)


def perturb_int(rng: random.Random, center: int, scale: int, low: int, high: int) -> int:
    return bounded_int(round(rng.gauss(center, scale)), low, high)


def sample_candidate(
    rng: random.Random,
    X: pd.DataFrame,
    best_candidate: Optional[Dict[str, object]],
) -> Tuple[str, Dict[str, float], List[Tuple[float, Pipeline]]]:
    mix_choice = rng.random()
    use_refine = best_candidate is not None and rng.random() < 0.7

    best_params = best_candidate.get("params", {}) if best_candidate else {}

    if mix_choice < 0.6:
        if use_refine and best_candidate and str(best_candidate.get("candidate_name", "")).startswith("hgb"):
            params = {
                "learning_rate": perturb_float(
                    rng, float(best_params.get("learning_rate", 0.05)), 0.008, 0.025, 0.09
                ),
                "max_depth": perturb_int(rng, int(best_params.get("max_depth", 6)), 1, 4, 10),
                "max_iter": perturb_int(rng, int(best_params.get("max_iter", 420)), 55, 220, 820),
                "min_samples_leaf": perturb_int(
                    rng, int(best_params.get("min_samples_leaf", 22)), 3, 8, 40
                ),
                "l2_regularization": perturb_float(
                    rng, float(best_params.get("l2_regularization", 0.12)), 0.05, 0.0, 0.45
                ),
                "seed": int(rng.randint(1, 10_000_000)),
            }
            candidate_name = "hgb_refine"
        else:
            params = {
                "learning_rate": round(rng.uniform(0.035, 0.085), 6),
                "max_depth": int(rng.randint(5, 10)),
                "max_iter": int(rng.randint(260, 700)),
                "min_samples_leaf": int(rng.randint(12, 34)),
                "l2_regularization": round(rng.uniform(0.0, 0.4), 6),
                "seed": int(rng.randint(1, 10_000_000)),
            }
            candidate_name = "hgb_random"

        model = build_hgb_model(
            X,
            learning_rate=float(params["learning_rate"]),
            max_depth=int(params["max_depth"]),
            max_iter=int(params["max_iter"]),
            min_samples_leaf=int(params["min_samples_leaf"]),
            l2_regularization=float(params["l2_regularization"]),
            random_state=int(params["seed"]),
        )
        return candidate_name, params, [(1.0, model)]

    if use_refine and best_candidate and "hgb_learning_rate" in best_params:
        hgb_params = {
            "learning_rate": perturb_float(
                rng, float(best_params.get("hgb_learning_rate", 0.05)), 0.006, 0.03, 0.08
            ),
            "max_depth": perturb_int(rng, int(best_params.get("hgb_max_depth", 6)), 1, 4, 9),
            "max_iter": perturb_int(rng, int(best_params.get("hgb_max_iter", 380)), 45, 220, 700),
            "min_samples_leaf": perturb_int(
                rng, int(best_params.get("hgb_min_samples_leaf", 20)), 3, 10, 34
            ),
            "l2_regularization": perturb_float(
                rng, float(best_params.get("hgb_l2_regularization", 0.15)), 0.04, 0.0, 0.35
            ),
            "seed": int(rng.randint(1, 10_000_000)),
        }
        logreg_params = {
            "c": perturb_float(rng, float(best_params.get("logreg_c", 1.0)), 0.18, 0.35, 1.8),
            "max_iter": perturb_int(rng, int(best_params.get("logreg_max_iter", 3200)), 300, 2200, 6000),
            "seed": int(rng.randint(1, 10_000_000)),
        }
        hgb_weight = round(
            bounded_float(float(best_params.get("hgb_weight", 0.65)) + rng.uniform(-0.08, 0.08), 0.5, 0.85),
            4,
        )
        candidate_name = "blend_refine"
    else:
        hgb_params = {
            "learning_rate": round(rng.uniform(0.04, 0.07), 6),
            "max_depth": int(rng.randint(5, 8)),
            "max_iter": int(rng.randint(260, 500)),
            "min_samples_leaf": int(rng.randint(14, 30)),
            "l2_regularization": round(rng.uniform(0.05, 0.3), 6),
            "seed": int(rng.randint(1, 10_000_000)),
        }
        logreg_params = {
            "c": round(rng.uniform(0.5, 1.6), 6),
            "max_iter": int(rng.randint(2600, 5200)),
            "seed": int(rng.randint(1, 10_000_000)),
        }
        hgb_weight = round(rng.uniform(0.55, 0.8), 4)
        candidate_name = "blend_random"
    logreg_weight = round(1.0 - hgb_weight, 4)

    hgb_model = build_hgb_model(
        X,
        learning_rate=float(hgb_params["learning_rate"]),
        max_depth=int(hgb_params["max_depth"]),
        max_iter=int(hgb_params["max_iter"]),
        min_samples_leaf=int(hgb_params["min_samples_leaf"]),
        l2_regularization=float(hgb_params["l2_regularization"]),
        random_state=int(hgb_params["seed"]),
    )
    logreg_model = build_logreg_model(
        X,
        c_value=float(logreg_params["c"]),
        max_iter=int(logreg_params["max_iter"]),
        class_weight="balanced",
        random_state=int(logreg_params["seed"]),
    )

    params = {
        "hgb_weight": hgb_weight,
        "logreg_weight": logreg_weight,
        "hgb_learning_rate": hgb_params["learning_rate"],
        "hgb_max_depth": hgb_params["max_depth"],
        "hgb_max_iter": hgb_params["max_iter"],
        "hgb_min_samples_leaf": hgb_params["min_samples_leaf"],
        "hgb_l2_regularization": hgb_params["l2_regularization"],
        "hgb_seed": hgb_params["seed"],
        "logreg_c": logreg_params["c"],
        "logreg_max_iter": logreg_params["max_iter"],
        "logreg_seed": logreg_params["seed"],
    }

    return candidate_name, params, [(hgb_weight, hgb_model), (logreg_weight, logreg_model)]


def build_weighted_models(
    X: pd.DataFrame,
    candidate_name: str,
    params: Dict[str, float],
) -> List[Tuple[float, Pipeline]]:
    if candidate_name.startswith("hgb"):
        model = build_hgb_model(
            X,
            learning_rate=float(params["learning_rate"]),
            max_depth=int(params["max_depth"]),
            max_iter=int(params["max_iter"]),
            min_samples_leaf=int(params["min_samples_leaf"]),
            l2_regularization=float(params["l2_regularization"]),
            random_state=int(params["seed"]),
        )
        return [(1.0, model)]

    hgb_model = build_hgb_model(
        X,
        learning_rate=float(params["hgb_learning_rate"]),
        max_depth=int(params["hgb_max_depth"]),
        max_iter=int(params["hgb_max_iter"]),
        min_samples_leaf=int(params["hgb_min_samples_leaf"]),
        l2_regularization=float(params["hgb_l2_regularization"]),
        random_state=int(params.get("hgb_seed", 42)),
    )
    logreg_model = build_logreg_model(
        X,
        c_value=float(params["logreg_c"]),
        max_iter=int(params["logreg_max_iter"]),
        class_weight="balanced",
        random_state=int(params.get("logreg_seed", 42)),
    )
    return [
        (float(params["hgb_weight"]), hgb_model),
        (float(params["logreg_weight"]), logreg_model),
    ]


def evaluate_candidate_config(
    candidate_name: str,
    params: Dict[str, float],
    X_train: pd.DataFrame,
    y: pd.Series,
    n_splits: int,
    cv_seed: int,
    need_oof: bool,
) -> Dict[str, object]:
    weighted_models = build_weighted_models(X_train, candidate_name, params)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=cv_seed)
    oof = np.zeros(len(X_train), dtype=float) if need_oof else None
    fold_scores: List[float] = []

    for idx_tr, idx_va in cv.split(X_train, y):
        X_tr = X_train.iloc[idx_tr]
        X_va = X_train.iloc[idx_va]
        y_tr = y.iloc[idx_tr]
        y_va = y.iloc[idx_va]

        pred = np.zeros(len(X_va), dtype=float)

        for weight, model in weighted_models:
            model.fit(X_tr, y_tr)
            pred += weight * model.predict_proba(X_va)[:, 1]

        score = roc_auc_score(y_va, pred)
        fold_scores.append(float(score))
        if need_oof and oof is not None:
            oof[idx_va] = pred

    return {
        "candidate_name": candidate_name,
        "params": params,
        "fold_scores": fold_scores,
        "mean_auc": float(np.mean(fold_scores)),
        "std_auc": float(np.std(fold_scores)),
        "oof": oof,
    }


def evaluate_candidate(
    weighted_models: List[Tuple[float, Pipeline]],
    X_train: pd.DataFrame,
    y: pd.Series,
    cv: StratifiedKFold,
) -> Tuple[List[float], np.ndarray]:
    oof = np.zeros(len(X_train), dtype=float)
    fold_scores: List[float] = []

    for fold_idx, (idx_tr, idx_va) in enumerate(cv.split(X_train, y), start=1):
        X_tr = X_train.iloc[idx_tr]
        X_va = X_train.iloc[idx_va]
        y_tr = y.iloc[idx_tr]
        y_va = y.iloc[idx_va]

        pred = np.zeros(len(X_va), dtype=float)

        for weight, model in weighted_models:
            model.fit(X_tr, y_tr)
            pred += weight * model.predict_proba(X_va)[:, 1]

        score = roc_auc_score(y_va, pred)
        fold_scores.append(float(score))
        oof[idx_va] = pred
        print(f"    Fold {fold_idx}/{cv.n_splits}: AUC={score:.5f}")

    return fold_scores, oof


def resolve_parallel_jobs(parallel_jobs: int) -> int:
    if parallel_jobs == 0:
        return max(1, (os.cpu_count() or 1) - 1)
    if parallel_jobs < 0:
        return max(1, os.cpu_count() or 1)
    return max(1, parallel_jobs)


def fit_predict_full(
    weighted_models: List[Tuple[float, Pipeline]],
    X_train: pd.DataFrame,
    y: pd.Series,
    X_test: pd.DataFrame,
) -> np.ndarray:
    pred = np.zeros(len(X_test), dtype=float)

    for weight, model in weighted_models:
        model.fit(X_train, y)
        pred += weight * model.predict_proba(X_test)[:, 1]

    return np.clip(pred, 0.0, 1.0)


def save_improved_submission(
    output_dir: Path,
    iteration: int,
    candidate_name: str,
    worker_name: str,
    mean_auc: float,
    fold_scores: List[float],
    params: Dict[str, float],
    train_ids: pd.Series,
    y: pd.Series,
    oof: np.ndarray,
    test_ids: pd.Series,
    test_pred: np.ndarray,
    baseline_before: float,
    prediction_hash: str,
) -> Dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    worker_suffix = f"_{worker_name}" if worker_name else ""
    stem = f"cont_{candidate_name}{worker_suffix}_iter{iteration}_{ts}"

    submission_path = output_dir / f"submission_{stem}.csv"
    oof_path = output_dir / f"oof_{stem}.csv"
    report_path = output_dir / f"continuous_report_{stem}.json"

    pd.DataFrame(
        {
            "anonymised_id": test_ids,
            "employed_status": test_pred,
        }
    ).to_csv(submission_path, index=False)

    pd.DataFrame(
        {
            "anonymised_id": train_ids,
            "target": y.values,
            "oof_pred": oof,
        }
    ).to_csv(oof_path, index=False)

    report = {
        "status": "improvement_found",
        "iteration": iteration,
        "candidate_name": candidate_name,
        "params": params,
        "baseline_before": baseline_before,
        "cv_auc_mean": mean_auc,
        "cv_auc_gain": mean_auc - baseline_before,
        "prediction_hash": prediction_hash,
        "fold_scores": fold_scores,
        "timestamp": datetime.now().isoformat(),
        "artifacts": {
            "submission": str(submission_path),
            "oof": str(oof_path),
        },
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    return {
        "submission_path": str(submission_path),
        "oof_path": str(oof_path),
        "report_path": str(report_path),
    }


def read_best_manual_score(score_file: Path) -> Optional[float]:
    if not score_file.exists():
        return None

    try:
        payload = json.loads(score_file.read_text(encoding="utf-8"))
        scores = payload.get("scores", {})
        values = [float(v) for v in scores.values() if isinstance(v, (int, float))]
        return max(values) if values else None
    except (json.JSONDecodeError, OSError, ValueError):
        return None


def prediction_hash(values: np.ndarray) -> str:
    rounded = np.round(values.astype(float), 12)
    return hashlib.sha1(rounded.tobytes()).hexdigest()


def candidate_signature(candidate_name: str, params: Dict[str, float]) -> str:
    normalized: Dict[str, object] = {}
    for key, value in sorted(params.items()):
        if isinstance(value, float):
            normalized[key] = round(value, 12)
        elif isinstance(value, (int, str, bool)) or value is None:
            normalized[key] = value
        else:
            normalized[key] = str(value)

    payload = {
        "candidate_name": candidate_name,
        "params": normalized,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha1(encoded).hexdigest()


def load_json_or_default(path: Path, default: Dict[str, object]) -> Dict[str, object]:
    if not path.exists():
        return dict(default)

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            return payload
    except (json.JSONDecodeError, OSError):
        pass

    return dict(default)


def write_json_atomic(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(f"{path.suffix}.tmp")
    tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp_path.replace(path)


def acquire_lock_file(lock_path: Path, timeout_seconds: float = 10.0, stale_seconds: float = 180.0) -> bool:
    deadline = time.time() + timeout_seconds

    while time.time() < deadline:
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(datetime.now().isoformat())
            return True
        except FileExistsError:
            try:
                age_seconds = time.time() - lock_path.stat().st_mtime
                if age_seconds > stale_seconds:
                    lock_path.unlink(missing_ok=True)
                    continue
            except OSError:
                pass

            time.sleep(0.05)

    return False


def release_lock_file(lock_path: Path) -> None:
    try:
        lock_path.unlink(missing_ok=True)
    except OSError:
        pass


def claim_candidate_signature(
    coordination_file: Path,
    signature: str,
    worker_name: str,
    ttl_seconds: float,
) -> Tuple[bool, Optional[str]]:
    lock_path = coordination_file.with_suffix(".lock")
    lock_acquired = acquire_lock_file(lock_path)
    if not lock_acquired:
        return False, "lock-timeout"

    try:
        now = datetime.now()
        state = load_json_or_default(coordination_file, {"version": 1, "claims": {}})
        raw_claims = state.get("claims", {})
        claims = raw_claims if isinstance(raw_claims, dict) else {}

        existing = claims.get(signature)
        if isinstance(existing, dict):
            claimed_at = existing.get("claimed_at")
            if isinstance(claimed_at, str):
                try:
                    age = (now - datetime.fromisoformat(claimed_at)).total_seconds()
                    if age < ttl_seconds:
                        owner = existing.get("worker_name")
                        owner_name = str(owner) if owner is not None else "unknown"
                        return False, owner_name
                except ValueError:
                    pass

        claims[signature] = {
            "worker_name": worker_name,
            "claimed_at": now.isoformat(),
        }

        state["claims"] = claims
        state["updated_at"] = now.isoformat()
        write_json_atomic(coordination_file, state)
        return True, None
    finally:
        release_lock_file(lock_path)


def load_existing_prediction_hashes(output_dir: Path) -> set[str]:
    hashes: set[str] = set()
    for file_path in glob(str(output_dir / "submission*.csv")):
        try:
            df = pd.read_csv(file_path, usecols=["employed_status"])
            hashes.add(prediction_hash(df["employed_status"].to_numpy(dtype=float)))
        except Exception:
            continue
    return hashes


def get_initial_baseline(args: argparse.Namespace) -> float:
    if args.baseline_override is not None:
        return float(args.baseline_override)

    manual_best = read_best_manual_score(args.leaderboard_scores_file)
    if manual_best is not None:
        return manual_best

    if args.baseline_metrics_file.exists():
        try:
            payload = json.loads(args.baseline_metrics_file.read_text(encoding="utf-8"))
            value = payload.get("cv_auc_mean")
            if isinstance(value, (int, float)):
                return float(value)
        except (json.JSONDecodeError, OSError):
            pass

    return float(args.fallback_baseline)


def save_state(state_path: Path, payload: Dict[str, object]) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Continuous random-search pipeline: runs until stopped and emits a new submission whenever it finds improvement."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Directory containing train/test files")
    parser.add_argument("--train-file", type=Path, default=None, help="Optional explicit train file")
    parser.add_argument("--test-file", type=Path, default=None, help="Optional explicit test file")
    parser.add_argument("--id-col", type=str, default=None, help="Optional ID column")
    parser.add_argument("--target-col", type=str, default=None, help="Optional target column")
    parser.add_argument("--max-round", type=int, default=9, help="Largest valid round")
    parser.add_argument("--n-splits", type=int, default=2, help="CV folds")
    parser.add_argument("--confirm-splits", type=int, default=3, help="Confirmation CV folds")
    parser.add_argument("--min-improvement", type=float, default=0.0005, help="Required gain over current best")
    parser.add_argument(
        "--promote-window",
        type=float,
        default=0.003,
        help="How close a quick candidate must be to the best score before running confirmation CV",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--max-iterations", type=int, default=0, help="0 means run forever")
    parser.add_argument("--batch-size", type=int, default=4, help="Candidates to screen per loop")
    parser.add_argument(
        "--parallel-jobs",
        type=int,
        default=0,
        help="Parallel candidate evaluations per batch. 0 uses cpu_count-1, -1 uses all CPUs.",
    )
    parser.add_argument(
        "--worker-name",
        type=str,
        default="",
        help="Optional worker/server name added to output files and state for multi-server runs.",
    )
    parser.add_argument(
        "--server-count",
        type=int,
        default=1,
        help="Number of cooperating servers. Values >1 enable deterministic divide-and-conquer sharding.",
    )
    parser.add_argument(
        "--server-index",
        type=int,
        default=0,
        help="0-based shard index for this server. Must be in [0, server-count-1].",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"), help="Output directory")
    parser.add_argument(
        "--coordination-file",
        type=Path,
        default=None,
        help="Shared JSON file for candidate claims (default: <output-dir>/continuous_coordination.json).",
    )
    parser.add_argument(
        "--claim-ttl-seconds",
        type=float,
        default=12 * 3600,
        help="How long a claimed candidate stays reserved before another server may re-claim it.",
    )
    parser.add_argument(
        "--state-file",
        type=Path,
        default=None,
        help="Optional state file path (default: <output-dir>/continuous_state.json)",
    )
    parser.add_argument("--fallback-baseline", type=float, default=0.0, help="Fallback baseline if no reference score is found")
    parser.add_argument("--baseline-override", type=float, default=None, help="Explicit initial baseline to beat")
    parser.add_argument(
        "--baseline-metrics-file",
        type=Path,
        default=Path("outputs/cv_metrics.json"),
        help="Metrics file used as baseline fallback",
    )
    parser.add_argument(
        "--leaderboard-scores-file",
        type=Path,
        default=Path("frontend/src/data/leaderboard_scores.json"),
        help="Leaderboard score mapping file for real-score baseline",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.server_count < 1:
        raise ValueError("--server-count must be >= 1")
    if args.server_index < 0 or args.server_index >= args.server_count:
        raise ValueError("--server-index must be within [0, server-count-1]")

    if args.train_file is not None and args.test_file is not None:
        train_path = args.train_file
        test_path = args.test_file
    else:
        train_path, test_path = auto_find_data_paths(args.data_dir)

    train_df = load_table(train_path)
    test_df = load_table(test_path)

    id_col = args.id_col or infer_id_col(train_df)
    target_col = args.target_col or infer_target_col(train_df)

    if id_col not in test_df.columns:
        raise ValueError(f"ID column '{id_col}' not found in test data.")

    train_wide = maybe_to_wide_panel(train_df, id_col=id_col, target_col=target_col)
    test_wide = maybe_to_wide_panel(test_df, id_col=id_col, target_col=None)

    X_train, y = build_panel_features(train_wide, id_col=id_col, target_col=target_col, max_round=args.max_round)
    X_test, _ = build_panel_features(test_wide, id_col=id_col, target_col=None, max_round=args.max_round)

    if y is None:
        raise ValueError("Target column missing after feature engineering.")

    train_ids = X_train[id_col].copy()
    test_ids = X_test[id_col].copy()
    X_train = X_train.drop(columns=[id_col])
    X_test = X_test.drop(columns=[id_col])
    y = y.fillna(0).astype(int)

    baseline = get_initial_baseline(args)
    worker_slug = args.worker_name.strip().replace(" ", "_")
    default_state_name = f"continuous_state_{worker_slug}.json" if worker_slug else "continuous_state.json"
    state_path = args.state_file if args.state_file is not None else (args.output_dir / default_state_name)
    coordination_file = (
        args.coordination_file
        if args.coordination_file is not None
        else (args.output_dir / "continuous_coordination.json")
    )
    coordination_enabled = args.server_count > 1
    best_cv_auc = baseline
    best_candidate_meta: Optional[Dict[str, object]] = None
    seen_prediction_hashes = load_existing_prediction_hashes(args.output_dir)
    parallel_jobs = resolve_parallel_jobs(args.parallel_jobs)

    print(f"Continuous pipeline started at {datetime.now().isoformat()}")
    print(f"Initial baseline to beat: {best_cv_auc:.6f}")
    print(f"Parallel jobs: {parallel_jobs} | Batch size: {args.batch_size}")
    print(f"Server shard: index {args.server_index}/{args.server_count - 1}")
    if coordination_enabled:
        print(f"Coordination file: {coordination_file}")
    print("Press Ctrl+C to stop.")

    iteration = 0
    candidate_counter = 0
    shard_cursor = args.server_index
    history: List[Dict[str, object]] = []

    save_state(
        state_path,
        {
            "status": "running",
            "started_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "initial_baseline": baseline,
            "current_best_cv_auc": best_cv_auc,
            "iterations_completed": 0,
            "history_tail": [],
        },
    )

    try:
        while True:
            iteration += 1
            batch_specs: List[Tuple[int, int, str, Dict[str, float], str]] = []
            skipped_claims = 0
            max_attempts = max(args.batch_size * 10, 20)
            attempts = 0

            while len(batch_specs) < args.batch_size and attempts < max_attempts:
                attempts += 1
                global_candidate_id = shard_cursor
                shard_cursor += args.server_count

                candidate_rng = random.Random(args.seed + (global_candidate_id * 104_729))
                candidate_name, params, _ = sample_candidate(candidate_rng, X_train, best_candidate_meta)
                signature = candidate_signature(candidate_name, params)

                if coordination_enabled:
                    worker_label = worker_slug or f"server_{args.server_index}"
                    claimed, owner = claim_candidate_signature(
                        coordination_file=coordination_file,
                        signature=signature,
                        worker_name=worker_label,
                        ttl_seconds=args.claim_ttl_seconds,
                    )
                    if not claimed:
                        skipped_claims += 1
                        continue

                candidate_counter += 1
                batch_specs.append((candidate_counter, global_candidate_id, candidate_name, params, signature))

            if not batch_specs:
                print(
                    f"\nIteration {iteration}: no claimable candidates found after {attempts} attempts. Retrying next loop."
                )
                continue

            print(f"\nIteration {iteration}: screening batch of {len(batch_specs)} candidates")
            if skipped_claims > 0:
                print(f"  Skipped {skipped_claims} already-claimed candidate(s)")

            raw_screen_results = Parallel(n_jobs=parallel_jobs, prefer="processes")(
                delayed(evaluate_candidate_config)(
                    candidate_name,
                    params,
                    X_train,
                    y,
                    args.n_splits,
                    args.seed + global_candidate_id,
                    False,
                )
                for _, global_candidate_id, candidate_name, params, _ in batch_specs
            )

            screen_results: List[Dict[str, object]] = []
            for batch_spec, result in zip(batch_specs, raw_screen_results):
                _, global_candidate_id, _, _, signature = batch_spec
                enriched = dict(result)
                enriched["global_candidate_id"] = global_candidate_id
                enriched["candidate_signature"] = signature
                screen_results.append(enriched)

            best_screen = max(screen_results, key=lambda item: float(item["mean_auc"]))
            candidate_name = str(best_screen["candidate_name"])
            params = dict(best_screen["params"])
            global_candidate_id = int(best_screen["global_candidate_id"])
            signature = str(best_screen["candidate_signature"])
            quick_mean_auc = float(best_screen["mean_auc"])
            quick_std_auc = float(best_screen["std_auc"])

            should_confirm = quick_mean_auc >= (best_cv_auc - args.promote_window)
            print(
                f"  Best batch candidate={candidate_name} | Quick Mean AUC={quick_mean_auc:.6f} Std={quick_std_auc:.6f} | Best={best_cv_auc:.6f} | Confirm={should_confirm}"
            )

            if should_confirm:
                print("  Running confirmation CV on batch winner...")
                confirm_result = evaluate_candidate_config(
                    candidate_name,
                    params,
                    X_train,
                    y,
                    args.confirm_splits,
                    args.seed + 100000 + global_candidate_id,
                    True,
                )
                fold_scores = list(confirm_result["fold_scores"])
                mean_auc = float(confirm_result["mean_auc"])
                std_auc = float(confirm_result["std_auc"])
                oof = np.asarray(confirm_result["oof"], dtype=float)
            else:
                fold_scores = list(best_screen["fold_scores"])
                mean_auc = quick_mean_auc
                std_auc = quick_std_auc
                oof = np.zeros(len(X_train), dtype=float)

            improved = mean_auc >= (best_cv_auc + args.min_improvement)
            print(f"  Mean AUC={mean_auc:.6f} Std={std_auc:.6f} | Best={best_cv_auc:.6f} | Improved={improved}")

            run_info: Dict[str, object] = {
                "iteration": iteration,
                "batch_size": args.batch_size,
                "parallel_jobs": parallel_jobs,
                "candidate_name": candidate_name,
                "global_candidate_id": global_candidate_id,
                "candidate_signature": signature,
                "params": params,
                "quick_mean_auc": quick_mean_auc,
                "quick_std_auc": quick_std_auc,
                "used_confirmation": should_confirm,
                "mean_auc": mean_auc,
                "std_auc": std_auc,
                "fold_scores": fold_scores,
                "best_before": best_cv_auc,
                "improved": improved,
                "timestamp": datetime.now().isoformat(),
                "worker_name": worker_slug,
            }

            if improved:
                best_before = best_cv_auc
                weighted_models = build_weighted_models(X_train, candidate_name, params)
                test_pred = fit_predict_full(weighted_models, X_train, y, X_test)
                pred_hash = prediction_hash(test_pred)
                duplicate_prediction = pred_hash in seen_prediction_hashes
                run_info["prediction_hash"] = pred_hash
                run_info["duplicate_prediction"] = duplicate_prediction

                best_cv_auc = mean_auc
                best_candidate_meta = {
                    "candidate_name": candidate_name,
                    "params": params,
                    "mean_auc": mean_auc,
                    "std_auc": std_auc,
                }

                if duplicate_prediction:
                    print("  Improvement found, but predictions duplicate an existing submission. Skipping file write.")
                    run_info["best_after"] = best_cv_auc
                else:
                    artifacts = save_improved_submission(
                        output_dir=args.output_dir,
                        iteration=iteration,
                        candidate_name=candidate_name,
                        worker_name=worker_slug,
                        mean_auc=mean_auc,
                        fold_scores=fold_scores,
                        params=params,
                        train_ids=train_ids,
                        y=y,
                        oof=oof,
                        test_ids=test_ids,
                        test_pred=test_pred,
                        baseline_before=best_before,
                        prediction_hash=pred_hash,
                    )
                    seen_prediction_hashes.add(pred_hash)
                    run_info["artifacts"] = artifacts
                    run_info["best_after"] = best_cv_auc
                    print(f"  Improvement accepted. New best CV AUC: {best_cv_auc:.6f}")
                    print(f"  New submission: {artifacts['submission_path']}")

            history.append(run_info)
            state_payload = {
                "status": "running",
                "started_at": history[0]["timestamp"] if history else datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
                "initial_baseline": baseline,
                "current_best_cv_auc": best_cv_auc,
                "iterations_completed": iteration,
                "last_iteration": run_info,
                "history_tail": history[-30:],
            }
            save_state(state_path, state_payload)

            if args.max_iterations > 0 and iteration >= args.max_iterations:
                print("Max iterations reached. Stopping.")
                break

        save_state(
            state_path,
            {
                "status": "completed",
                "updated_at": datetime.now().isoformat(),
                "initial_baseline": baseline,
                "current_best_cv_auc": best_cv_auc,
                "iterations_completed": iteration,
                "history_tail": history[-30:],
            },
        )

    except KeyboardInterrupt:
        print("\nStopped by user.")
        state_payload = {
            "status": "stopped",
            "updated_at": datetime.now().isoformat(),
            "initial_baseline": baseline,
            "current_best_cv_auc": best_cv_auc,
            "iterations_completed": iteration,
            "history_tail": history[-30:],
        }
        save_state(state_path, state_payload)


if __name__ == "__main__":
    main()
