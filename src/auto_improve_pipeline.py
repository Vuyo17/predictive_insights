from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
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


@dataclass
class CandidateSpec:
    name: str
    builders: Sequence[Callable[[pd.DataFrame], Pipeline]]
    weights: Sequence[float]


def sanitize_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", name).strip("_").lower()


def split_columns(X: pd.DataFrame) -> Tuple[List[str], List[str]]:
    numeric_features = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    categorical_features = [c for c in X.columns if c not in numeric_features]
    return numeric_features, categorical_features


def build_ohe_logreg(
    X: pd.DataFrame,
    c_value: float,
    max_iter: int,
    class_weight: Optional[str],
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
        random_state=42,
    )

    return Pipeline(steps=[("prep", preprocessor), ("model", model)])


def build_ord_hgb(
    X: pd.DataFrame,
    learning_rate: float,
    max_depth: int,
    max_iter: int,
    min_samples_leaf: int,
    l2_regularization: float,
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
                            OrdinalEncoder(
                                handle_unknown="use_encoded_value",
                                unknown_value=-1,
                            ),
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
        random_state=42,
    )

    return Pipeline(steps=[("prep", preprocessor), ("model", model)])


def build_ord_extratrees(
    X: pd.DataFrame,
    n_estimators: int,
    max_depth: Optional[int],
    min_samples_leaf: int,
    max_features: str,
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
                            OrdinalEncoder(
                                handle_unknown="use_encoded_value",
                                unknown_value=-1,
                            ),
                        ),
                    ]
                ),
                categorical_features,
            ),
        ],
        remainder="drop",
    )

    model = ExtraTreesClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        random_state=42,
        n_jobs=-1,
    )

    return Pipeline(steps=[("prep", preprocessor), ("model", model)])


def candidate_space(profile: str) -> List[CandidateSpec]:
    quick = [
        CandidateSpec(
            name="hgb_quick_tuned",
            builders=[
                lambda X: build_ord_hgb(
                    X,
                    learning_rate=0.055,
                    max_depth=6,
                    max_iter=280,
                    min_samples_leaf=22,
                    l2_regularization=0.2,
                ),
            ],
            weights=[1.0],
        ),
        CandidateSpec(
            name="blend_quick_hgb_logreg",
            builders=[
                lambda X: build_ohe_logreg(X, c_value=0.9, max_iter=2800, class_weight="balanced"),
                lambda X: build_ord_hgb(
                    X,
                    learning_rate=0.05,
                    max_depth=7,
                    max_iter=320,
                    min_samples_leaf=20,
                    l2_regularization=0.15,
                ),
            ],
            weights=[0.35, 0.65],
        ),
        CandidateSpec(
            name="logreg_quick",
            builders=[
                lambda X: build_ohe_logreg(X, c_value=1.1, max_iter=3200, class_weight="balanced"),
            ],
            weights=[1.0],
        ),
    ]

    full = [
        CandidateSpec(
            name="blend_hgb_heavy_logreg_support",
            builders=[
                lambda X: build_ohe_logreg(X, c_value=0.9, max_iter=4000, class_weight="balanced"),
                lambda X: build_ord_hgb(
                    X,
                    learning_rate=0.04,
                    max_depth=8,
                    max_iter=550,
                    min_samples_leaf=18,
                    l2_regularization=0.1,
                ),
            ],
            weights=[0.3, 0.7],
        ),
        CandidateSpec(
            name="blend_hgb_extratrees",
            builders=[
                lambda X: build_ord_hgb(
                    X,
                    learning_rate=0.05,
                    max_depth=7,
                    max_iter=450,
                    min_samples_leaf=20,
                    l2_regularization=0.15,
                ),
                lambda X: build_ord_extratrees(
                    X,
                    n_estimators=500,
                    max_depth=None,
                    min_samples_leaf=2,
                    max_features="sqrt",
                ),
            ],
            weights=[0.65, 0.35],
        ),
        CandidateSpec(
            name="logreg_strong",
            builders=[
                lambda X: build_ohe_logreg(X, c_value=1.3, max_iter=5000, class_weight="balanced"),
            ],
            weights=[1.0],
        ),
        CandidateSpec(
            name="hgb_compact",
            builders=[
                lambda X: build_ord_hgb(
                    X,
                    learning_rate=0.06,
                    max_depth=6,
                    max_iter=350,
                    min_samples_leaf=22,
                    l2_regularization=0.25,
                ),
            ],
            weights=[1.0],
        ),
    ]

    return quick if profile == "quick" else full


def get_baseline_score(metrics_path: Path, fallback: float) -> float:
    if not metrics_path.exists():
        return fallback

    try:
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return fallback

    score = payload.get("cv_auc_mean")
    if isinstance(score, (int, float)):
        return float(score)
    return fallback


def evaluate_candidate(
    spec: CandidateSpec,
    X_train: pd.DataFrame,
    y: pd.Series,
    cv: StratifiedKFold,
) -> Dict[str, object]:
    oof = np.zeros(len(X_train), dtype=float)
    fold_scores: List[float] = []

    for fold_idx, (idx_tr, idx_va) in enumerate(cv.split(X_train, y), start=1):
        X_tr = X_train.iloc[idx_tr]
        X_va = X_train.iloc[idx_va]
        y_tr = y.iloc[idx_tr]
        y_va = y.iloc[idx_va]

        preds = np.zeros(len(X_va), dtype=float)

        for weight, builder in zip(spec.weights, spec.builders):
            model = builder(X_train)
            model.fit(X_tr, y_tr)
            preds += weight * model.predict_proba(X_va)[:, 1]

        score = roc_auc_score(y_va, preds)
        fold_scores.append(float(score))
        oof[idx_va] = preds
        print(f"  Fold {fold_idx}/{cv.n_splits}: AUC={score:.5f}")

    cv_mean = float(np.mean(fold_scores))
    cv_std = float(np.std(fold_scores))

    return {
        "name": spec.name,
        "fold_scores": fold_scores,
        "cv_auc_mean": cv_mean,
        "cv_auc_std": cv_std,
        "oof": oof,
    }


def train_full_and_predict(
    spec: CandidateSpec,
    X_train: pd.DataFrame,
    y: pd.Series,
    X_test: pd.DataFrame,
) -> np.ndarray:
    pred = np.zeros(len(X_test), dtype=float)

    for weight, builder in zip(spec.weights, spec.builders):
        model = builder(X_train)
        model.fit(X_train, y)
        pred += weight * model.predict_proba(X_test)[:, 1]

    return np.clip(pred, 0.0, 1.0)


def build_dataset(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    id_col: str,
    target_col: str,
    max_round: int,
) -> Tuple[pd.Series, pd.Series, pd.DataFrame, pd.DataFrame, pd.Series]:
    train_wide = maybe_to_wide_panel(train_df, id_col=id_col, target_col=target_col)
    test_wide = maybe_to_wide_panel(test_df, id_col=id_col, target_col=None)

    X_train, y = build_panel_features(train_wide, id_col=id_col, target_col=target_col, max_round=max_round)
    X_test, _ = build_panel_features(test_wide, id_col=id_col, target_col=None, max_round=max_round)

    if y is None:
        raise ValueError("Target column missing after feature engineering.")

    train_ids = X_train[id_col].copy()
    test_ids = X_test[id_col].copy()

    X_train = X_train.drop(columns=[id_col])
    X_test = X_test.drop(columns=[id_col])

    y = y.fillna(0).astype(int)

    return train_ids, test_ids, X_train, X_test, y


def save_outputs(
    output_dir: Path,
    candidate_name: str,
    test_ids: pd.Series,
    test_pred: np.ndarray,
    train_ids: pd.Series,
    y: pd.Series,
    oof: np.ndarray,
    report_payload: Dict[str, object],
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"{sanitize_name(candidate_name)}_{timestamp}"

    submission_path = output_dir / f"submission_{suffix}.csv"
    oof_path = output_dir / f"oof_{suffix}.csv"
    report_path = output_dir / f"auto_report_{suffix}.json"

    submission = pd.DataFrame(
        {
            "anonymised_id": test_ids,
            "employed_status": test_pred,
        }
    )
    submission.to_csv(submission_path, index=False)

    oof_df = pd.DataFrame(
        {
            "anonymised_id": train_ids,
            "target": y.values,
            "oof_pred": oof,
        }
    )
    oof_df.to_csv(oof_path, index=False)

    report_path.write_text(json.dumps(report_payload, indent=2), encoding="utf-8")

    return submission_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Iterative pipeline that evaluates model candidates until CV AUC improves over baseline."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Directory containing train/test files")
    parser.add_argument("--train-file", type=Path, default=None, help="Optional explicit train file")
    parser.add_argument("--test-file", type=Path, default=None, help="Optional explicit test file")
    parser.add_argument("--id-col", type=str, default=None, help="Optional ID column")
    parser.add_argument("--target-col", type=str, default=None, help="Optional target column")
    parser.add_argument("--max-round", type=int, default=9, help="Largest valid round")
    parser.add_argument("--n-splits", type=int, default=3, help="CV folds")
    parser.add_argument("--min-improvement", type=float, default=0.0020, help="Minimum AUC gain to accept")
    parser.add_argument(
        "--profile",
        type=str,
        default="quick",
        choices=["quick", "full"],
        help="Search depth profile",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=0,
        help="Optional cap on number of candidates to evaluate (0 = all)",
    )
    parser.add_argument(
        "--baseline-metrics-file",
        type=Path,
        default=Path("outputs/cv_metrics.json"),
        help="Baseline metrics JSON with cv_auc_mean",
    )
    parser.add_argument("--fallback-baseline", type=float, default=0.0, help="Fallback baseline AUC")
    parser.add_argument(
        "--baseline-override",
        type=float,
        default=None,
        help="Optional explicit baseline score to beat (overrides baseline metrics file)",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"), help="Output directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.train_file is not None and args.test_file is not None:
        train_path = args.train_file
        test_path = args.test_file
    else:
        train_path, test_path = auto_find_data_paths(args.data_dir)

    print(f"Loading train data: {train_path}")
    print(f"Loading test data:  {test_path}")

    train_df = load_table(train_path)
    test_df = load_table(test_path)

    id_col = args.id_col or infer_id_col(train_df)
    target_col = args.target_col or infer_target_col(train_df)

    if id_col not in test_df.columns:
        raise ValueError(f"ID column '{id_col}' not found in test data.")

    print(f"Using id_col={id_col} target_col={target_col}")

    train_ids, test_ids, X_train, X_test, y = build_dataset(
        train_df=train_df,
        test_df=test_df,
        id_col=id_col,
        target_col=target_col,
        max_round=args.max_round,
    )

    baseline_score = (
        float(args.baseline_override)
        if args.baseline_override is not None
        else get_baseline_score(args.baseline_metrics_file, args.fallback_baseline)
    )
    target_score = baseline_score + args.min_improvement

    print(f"Baseline CV AUC: {baseline_score:.6f}")
    print(f"Need at least:    {target_score:.6f}")

    cv = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=42)

    all_results: List[Dict[str, object]] = []
    winner_spec: Optional[CandidateSpec] = None
    winner_result: Optional[Dict[str, object]] = None

    specs = candidate_space(args.profile)
    if args.max_candidates > 0:
        specs = specs[: args.max_candidates]

    for spec in specs:
        print(f"\nEvaluating candidate: {spec.name}")
        result = evaluate_candidate(spec, X_train, y, cv)
        all_results.append({
            "name": result["name"],
            "cv_auc_mean": result["cv_auc_mean"],
            "cv_auc_std": result["cv_auc_std"],
            "fold_scores": result["fold_scores"],
        })

        mean_auc = float(result["cv_auc_mean"])
        print(f"Candidate mean AUC: {mean_auc:.6f}")

        if mean_auc >= target_score:
            winner_spec = spec
            winner_result = result
            print("Improvement found. Stopping search early.")
            break

    if winner_spec is None or winner_result is None:
        best = max(all_results, key=lambda r: float(r["cv_auc_mean"]))
        report_payload = {
            "status": "no_improvement_found",
            "baseline_cv_auc": baseline_score,
            "target_cv_auc": target_score,
            "best_candidate": best,
            "all_candidates": all_results,
            "timestamp": datetime.now().isoformat(),
        }

        args.output_dir.mkdir(parents=True, exist_ok=True)
        report_path = args.output_dir / f"auto_report_no_improvement_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        report_path.write_text(json.dumps(report_payload, indent=2), encoding="utf-8")
        print(f"No candidate exceeded target. Report saved to: {report_path}")
        return

    test_pred = train_full_and_predict(winner_spec, X_train, y, X_test)

    report_payload = {
        "status": "improvement_found",
        "baseline_cv_auc": baseline_score,
        "target_cv_auc": target_score,
        "winner": {
            "name": winner_result["name"],
            "cv_auc_mean": winner_result["cv_auc_mean"],
            "cv_auc_std": winner_result["cv_auc_std"],
            "fold_scores": winner_result["fold_scores"],
            "gain": float(winner_result["cv_auc_mean"]) - baseline_score,
        },
        "all_candidates": all_results,
        "timestamp": datetime.now().isoformat(),
    }

    submission_path = save_outputs(
        output_dir=args.output_dir,
        candidate_name=winner_spec.name,
        test_ids=test_ids,
        test_pred=test_pred,
        train_ids=train_ids,
        y=y,
        oof=np.asarray(winner_result["oof"], dtype=float),
        report_payload=report_payload,
    )

    print(f"Saved improved submission to: {submission_path}")


if __name__ == "__main__":
    main()
