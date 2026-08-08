from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder


ROUND_PATTERNS = [
    re.compile(r"(?i)(?:^|[_-])(r|round|wave)(\d{1,2})(?:$|[_-])"),
    re.compile(r"(?i)(\d{1,2})$"),
]


def load_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported file format: {path}")


def auto_find_data_paths(data_dir: Path) -> Tuple[Path, Path]:
    candidates = sorted(data_dir.glob("*"))
    train_candidates = [p for p in candidates if p.is_file() and "train" in p.stem.lower()]
    test_candidates = [p for p in candidates if p.is_file() and "test" in p.stem.lower()]

    if not train_candidates or not test_candidates:
        raise FileNotFoundError(
            "Could not auto-detect train/test files in data/. "
            "Name files with 'train' and 'test' in the filename (csv/parquet)."
        )

    return train_candidates[0], test_candidates[0]


def infer_id_col(df: pd.DataFrame) -> str:
    candidates = [
        "anonymised_id",
        "anonymous_id",
        "participant_id",
        "person_id",
        "id",
    ]
    lower_to_original = {c.lower(): c for c in df.columns}
    for candidate in candidates:
        if candidate in lower_to_original:
            return lower_to_original[candidate]

    string_cols = [c for c in df.columns if df[c].dtype == "object" or pd.api.types.is_string_dtype(df[c])]
    if string_cols:
        uniq_ratio = [(c, df[c].nunique(dropna=True) / max(len(df), 1)) for c in string_cols]
        uniq_ratio.sort(key=lambda x: x[1], reverse=True)
        return uniq_ratio[0][0]

    return df.columns[0]


def infer_target_col(df: pd.DataFrame) -> str:
    candidates = [
        "employed_status",
        "target",
        "label",
        "y",
    ]
    lower_to_original = {c.lower(): c for c in df.columns}
    for candidate in candidates:
        if candidate in lower_to_original:
            return lower_to_original[candidate]
    raise ValueError("Could not infer target column. Expected something like employed_status.")


def maybe_to_wide_panel(df: pd.DataFrame, id_col: str, target_col: Optional[str]) -> pd.DataFrame:
    lower_map = {c.lower(): c for c in df.columns}
    round_col = None
    for name in ["round", "survey_round", "wave"]:
        if name in lower_map:
            round_col = lower_map[name]
            break

    if round_col is None:
        return df

    if not df[id_col].duplicated().any():
        return df

    work = df.copy()
    work[round_col] = pd.to_numeric(work[round_col], errors="coerce")

    keep_cols = [c for c in work.columns if c not in {id_col, round_col, target_col}]

    wide_parts: List[pd.DataFrame] = []
    for col in keep_cols:
        pivot = work.pivot_table(index=id_col, columns=round_col, values=col, aggfunc="first")
        pivot.columns = [f"{col}_r{int(c)}" for c in pivot.columns]
        wide_parts.append(pivot)

    wide = pd.concat(wide_parts, axis=1).reset_index()

    if target_col is not None and target_col in work.columns:
        target = work.groupby(id_col, as_index=False)[target_col].max()
        wide = wide.merge(target, on=id_col, how="left")

    return wide


def detect_round_number(col_name: str, max_round: int) -> Optional[int]:
    for pattern in ROUND_PATTERNS:
        match = pattern.search(col_name)
        if match:
            value = int(match.group(2) if len(match.groups()) > 1 else match.group(1))
            if 1 <= value <= max_round:
                return value
    return None


def normalize_base_name(col_name: str) -> str:
    base = re.sub(r"(?i)(?:^|[_-])(r|round|wave)\d{1,2}(?:$|[_-])", "_", col_name)
    base = re.sub(r"\d{1,2}$", "", base)
    base = re.sub(r"__+", "_", base).strip("_-")
    return base if base else col_name


def row_slope(values: np.ndarray, rounds: np.ndarray) -> float:
    mask = ~np.isnan(values)
    if mask.sum() < 2:
        return np.nan
    x = rounds[mask]
    y = values[mask]
    return float(np.polyfit(x, y, 1)[0])


def build_panel_features(
    df: pd.DataFrame,
    id_col: str,
    target_col: Optional[str],
    max_round: int,
) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    work = df.copy()
    y: Optional[pd.Series] = None
    if target_col is not None and target_col in work.columns:
        y = pd.to_numeric(work[target_col], errors="coerce")
        work = work.drop(columns=[target_col])

    cols = [c for c in work.columns if c != id_col]

    round_groups: Dict[str, Dict[int, str]] = {}
    static_cols: List[str] = []

    for col in cols:
        round_num = detect_round_number(col, max_round)
        if round_num is None:
            static_cols.append(col)
            continue
        base = normalize_base_name(col)
        round_groups.setdefault(base, {})[round_num] = col

    features = pd.DataFrame(index=work.index)
    features[id_col] = work[id_col]

    if static_cols:
        static_block = work[static_cols].copy()
        if not static_block.empty:
            features = pd.concat([features, static_block], axis=1)

    all_round_columns: List[Tuple[int, str]] = []

    for base, mapping in round_groups.items():
        rounds_sorted = np.array(sorted(mapping.keys()), dtype=float)
        col_order = [mapping[r] for r in sorted(mapping.keys())]
        series_block = work[col_order]

        all_round_columns.extend((r, mapping[r]) for r in mapping)

        numeric_block = series_block.apply(pd.to_numeric, errors="coerce")
        numeric_share = float(np.isfinite(numeric_block.to_numpy(dtype=float)).mean())
        is_numeric = numeric_share >= 0.5

        if is_numeric:
            vals = numeric_block
            last_vals = vals.ffill(axis=1).iloc[:, -1]
            first_vals = vals.bfill(axis=1).iloc[:, 0]

            features[f"{base}__last"] = last_vals
            features[f"{base}__first"] = first_vals
            features[f"{base}__change"] = last_vals - first_vals
            features[f"{base}__mean"] = vals.mean(axis=1)
            features[f"{base}__std"] = vals.std(axis=1)
            features[f"{base}__min"] = vals.min(axis=1)
            features[f"{base}__max"] = vals.max(axis=1)
            features[f"{base}__obs_count"] = vals.notna().sum(axis=1)

            arr = vals.to_numpy(dtype=float)
            slopes = np.array([row_slope(row, rounds_sorted) for row in arr])
            features[f"{base}__trend"] = slopes
        else:
            cats = series_block.astype("string")
            last_vals = cats.ffill(axis=1).iloc[:, -1]
            first_vals = cats.bfill(axis=1).iloc[:, 0]

            features[f"{base}__last"] = last_vals
            features[f"{base}__first"] = first_vals
            features[f"{base}__mode"] = cats.mode(axis=1, dropna=True).iloc[:, 0]
            features[f"{base}__nunique"] = cats.nunique(axis=1, dropna=True)
            features[f"{base}__changed"] = (
                (last_vals.fillna("<NA>") != first_vals.fillna("<NA>")).astype(float)
            )

    if all_round_columns:
        round_frame = pd.DataFrame(
            {
                col_name: work[col_name].notna().astype(int) * round_num
                for round_num, col_name in all_round_columns
            }
        )
        features["panel__most_recent_round"] = round_frame.max(axis=1)
        min_round = round_frame.replace(0, np.nan).min(axis=1)
        features["panel__first_observed_round"] = min_round
        features["panel__history_span"] = (
            features["panel__most_recent_round"] - features["panel__first_observed_round"]
        )

    return features, y


def build_models(
    X: pd.DataFrame,
) -> Tuple[Pipeline, Pipeline]:
    feature_cols = [c for c in X.columns]
    numeric_features = [c for c in feature_cols if pd.api.types.is_numeric_dtype(X[c])]
    categorical_features = [c for c in feature_cols if c not in numeric_features]

    ohe_preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                    ]
                ),
                numeric_features,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "encoder",
                            OneHotEncoder(handle_unknown="ignore", min_frequency=20),
                        ),
                    ]
                ),
                categorical_features,
            ),
        ],
        remainder="drop",
    )

    ord_preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                    ]
                ),
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

    logreg = LogisticRegression(
        C=0.7,
        max_iter=3000,
        solver="saga",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    gbdt = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_depth=6,
        max_iter=350,
        min_samples_leaf=25,
        l2_regularization=0.2,
        random_state=42,
    )

    model_ohe = Pipeline(steps=[("prep", ohe_preprocessor), ("model", logreg)])
    model_ord = Pipeline(steps=[("prep", ord_preprocessor), ("model", gbdt)])

    return model_ohe, model_ord


def train_predict(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    id_col: str,
    target_col: str,
    max_round: int,
    n_splits: int,
    output_dir: Path,
) -> None:
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

    model_ohe, model_ord = build_models(X_train)

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    oof_pred = np.zeros(len(X_train), dtype=float)
    fold_scores: List[float] = []

    for fold, (idx_tr, idx_va) in enumerate(cv.split(X_train, y), start=1):
        X_tr, X_va = X_train.iloc[idx_tr], X_train.iloc[idx_va]
        y_tr, y_va = y.iloc[idx_tr], y.iloc[idx_va]

        model_ohe.fit(X_tr, y_tr)
        model_ord.fit(X_tr, y_tr)

        p1 = model_ohe.predict_proba(X_va)[:, 1]
        p2 = model_ord.predict_proba(X_va)[:, 1]
        p = 0.5 * p1 + 0.5 * p2

        oof_pred[idx_va] = p
        score = roc_auc_score(y_va, p)
        fold_scores.append(score)
        print(f"Fold {fold}/{n_splits} AUC: {score:.5f}")

    cv_mean = float(np.mean(fold_scores))
    cv_std = float(np.std(fold_scores))
    print(f"CV AUC mean={cv_mean:.5f} std={cv_std:.5f}")

    model_ohe.fit(X_train, y)
    model_ord.fit(X_train, y)

    test_pred = 0.5 * model_ohe.predict_proba(X_test)[:, 1] + 0.5 * model_ord.predict_proba(X_test)[:, 1]
    test_pred = np.clip(test_pred, 0.0, 1.0)

    output_dir.mkdir(parents=True, exist_ok=True)

    oof_df = pd.DataFrame(
        {
            id_col: train_ids,
            target_col: y.values,
            "oof_pred": oof_pred,
        }
    )
    oof_df.to_csv(output_dir / "oof_predictions.csv", index=False)

    submission = pd.DataFrame(
        {
            "anonymised_id": test_ids,
            "employed_status": test_pred,
        }
    )
    submission.to_csv(output_dir / "submission.csv", index=False)

    metrics = {
        "cv_auc_folds": fold_scores,
        "cv_auc_mean": cv_mean,
        "cv_auc_std": cv_std,
        "n_train_rows": int(len(X_train)),
        "n_test_rows": int(len(X_test)),
        "n_features": int(X_train.shape[1]),
    }
    (output_dir / "cv_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    joblib.dump(model_ohe, output_dir / "model_ohe.joblib")
    joblib.dump(model_ord, output_dir / "model_ord.joblib")

    print(f"Saved submission to: {output_dir / 'submission.csv'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train labour market prediction models and generate submission.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Directory containing train/test files")
    parser.add_argument("--train-file", type=Path, default=None, help="Optional explicit train file path")
    parser.add_argument("--test-file", type=Path, default=None, help="Optional explicit test file path")
    parser.add_argument("--id-col", type=str, default=None, help="Optional ID column name override")
    parser.add_argument("--target-col", type=str, default=None, help="Optional target column name override")
    parser.add_argument("--max-round", type=int, default=9, help="Largest valid survey round number")
    parser.add_argument("--n-splits", type=int, default=5, help="Number of CV folds")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"), help="Output directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.train_file is not None and args.test_file is not None:
        train_path = args.train_file
        test_path = args.test_file
    else:
        train_path, test_path = auto_find_data_paths(args.data_dir)

    print(f"Loading train data from: {train_path}")
    print(f"Loading test data from:  {test_path}")

    train_df = load_table(train_path)
    test_df = load_table(test_path)

    id_col = args.id_col or infer_id_col(train_df)
    target_col = args.target_col or infer_target_col(train_df)

    print(f"Using id_col={id_col} target_col={target_col}")

    if id_col not in test_df.columns:
        raise ValueError(f"ID column '{id_col}' not found in test data.")

    train_predict(
        train_df=train_df,
        test_df=test_df,
        id_col=id_col,
        target_col=target_col,
        max_round=args.max_round,
        n_splits=args.n_splits,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
