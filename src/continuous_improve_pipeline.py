from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
from glob import glob
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier
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
                            OneHotEncoder(handle_unknown="ignore", min_frequency=25),
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
        tol=3e-3,
        class_weight=class_weight,
        random_state=random_state,
    )

    return Pipeline(steps=[("prep", preprocessor), ("model", model)])


def build_extratrees_model(
    X: pd.DataFrame,
    n_estimators: int,
    max_depth: Optional[int],
    min_samples_leaf: int,
    max_features: str,
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
                            OneHotEncoder(handle_unknown="ignore", min_frequency=20),
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
        n_jobs=1,
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

    if mix_choice < 0.45:
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

    if mix_choice < 0.65:
        if use_refine and best_candidate and str(best_candidate.get("candidate_name", "")).startswith("et"):
            max_depth_raw = best_params.get("max_depth", 18)
            if max_depth_raw in (None, "none"):
                max_depth_center = 18
            else:
                max_depth_center = int(max_depth_raw)

            sampled_depth = perturb_int(rng, max_depth_center, 4, 8, 32)
            params = {
                "n_estimators": perturb_int(rng, int(best_params.get("n_estimators", 450)), 80, 200, 1000),
                "max_depth": int(sampled_depth),
                "min_samples_leaf": perturb_int(rng, int(best_params.get("min_samples_leaf", 3)), 1, 1, 10),
                "max_features": str(best_params.get("max_features", "sqrt")),
                "seed": int(rng.randint(1, 10_000_000)),
            }
            candidate_name = "et_refine"
        else:
            params = {
                "n_estimators": int(rng.randint(300, 900)),
                "max_depth": int(rng.randint(10, 30)),
                "min_samples_leaf": int(rng.randint(1, 8)),
                "max_features": "sqrt" if rng.random() < 0.75 else "log2",
                "seed": int(rng.randint(1, 10_000_000)),
            }
            candidate_name = "et_random"

        model = build_extratrees_model(
            X,
            n_estimators=int(params["n_estimators"]),
            max_depth=int(params["max_depth"]),
            min_samples_leaf=int(params["min_samples_leaf"]),
            max_features=str(params["max_features"]),
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
            "max_iter": perturb_int(rng, int(best_params.get("logreg_max_iter", 1500)), 180, 900, 2600),
            "seed": int(rng.randint(1, 10_000_000)),
        }
        hgb_weight = round(
            bounded_float(float(best_params.get("hgb_weight", 0.65)) + rng.uniform(-0.08, 0.08), 0.5, 0.85),
            4,
        )
        et_weight = round(bounded_float(float(best_params.get("et_weight", 0.0)) + rng.uniform(-0.06, 0.06), 0.0, 0.35), 4)
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
            "max_iter": int(rng.randint(900, 2200)),
            "seed": int(rng.randint(1, 10_000_000)),
        }
        et_params = {
            "n_estimators": int(rng.randint(320, 820)),
            "max_depth": int(rng.randint(10, 28)),
            "min_samples_leaf": int(rng.randint(1, 6)),
            "max_features": "sqrt" if rng.random() < 0.75 else "log2",
            "seed": int(rng.randint(1, 10_000_000)),
        }

        hgb_weight = round(rng.uniform(0.45, 0.75), 4)
        et_weight = round(rng.uniform(0.0, 0.3), 4)
        candidate_name = "blend_random"
    if "et_params" not in locals():
        et_params = {
            "n_estimators": int(best_params.get("et_n_estimators", 520)),
            "max_depth": int(best_params.get("et_max_depth", 18)),
            "min_samples_leaf": int(best_params.get("et_min_samples_leaf", 3)),
            "max_features": str(best_params.get("et_max_features", "sqrt")),
            "seed": int(rng.randint(1, 10_000_000)),
        }

    total = hgb_weight + et_weight
    if total >= 0.95:
        scale = 0.95 / total
        hgb_weight = round(hgb_weight * scale, 4)
        et_weight = round(et_weight * scale, 4)
    logreg_weight = round(1.0 - hgb_weight - et_weight, 4)
    if logreg_weight < 0.05:
        deficit = 0.05 - logreg_weight
        hgb_weight = round(max(0.35, hgb_weight - (deficit * 0.7)), 4)
        et_weight = round(max(0.0, et_weight - (deficit * 0.3)), 4)
        logreg_weight = round(1.0 - hgb_weight - et_weight, 4)

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
    et_model = build_extratrees_model(
        X,
        n_estimators=int(et_params["n_estimators"]),
        max_depth=int(et_params["max_depth"]),
        min_samples_leaf=int(et_params["min_samples_leaf"]),
        max_features=str(et_params["max_features"]),
        random_state=int(et_params["seed"]),
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
        "et_weight": et_weight,
        "et_n_estimators": et_params["n_estimators"],
        "et_max_depth": et_params["max_depth"],
        "et_min_samples_leaf": et_params["min_samples_leaf"],
        "et_max_features": et_params["max_features"],
        "et_seed": et_params["seed"],
        "logreg_c": logreg_params["c"],
        "logreg_max_iter": logreg_params["max_iter"],
        "logreg_seed": logreg_params["seed"],
    }

    weighted = [(hgb_weight, hgb_model), (logreg_weight, logreg_model)]
    if et_weight > 0.0:
        weighted.append((et_weight, et_model))
    return candidate_name, params, weighted


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

    if candidate_name.startswith("et"):
        model = build_extratrees_model(
            X,
            n_estimators=int(params["n_estimators"]),
            max_depth=int(params["max_depth"]),
            min_samples_leaf=int(params["min_samples_leaf"]),
            max_features=str(params["max_features"]),
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
    weighted: List[Tuple[float, Pipeline]] = [
        (float(params["hgb_weight"]), hgb_model),
        (float(params["logreg_weight"]), logreg_model),
    ]

    et_weight = float(params.get("et_weight", 0.0))
    if et_weight > 0.0:
        et_model = build_extratrees_model(
            X,
            n_estimators=int(params.get("et_n_estimators", 520)),
            max_depth=int(params.get("et_max_depth", 18)),
            min_samples_leaf=int(params.get("et_min_samples_leaf", 3)),
            max_features=str(params.get("et_max_features", "sqrt")),
            random_state=int(params.get("et_seed", 42)),
        )
        weighted.append((et_weight, et_model))

    return weighted


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


def next_submission_index(output_dir: Path) -> int:
    highest = 0
    for file_path in output_dir.glob("submission*.csv"):
        match = re.fullmatch(r"submission(\d+)\.csv", file_path.name)
        if match:
            highest = max(highest, int(match.group(1)))
    return highest + 1


def load_benchmark_state(path: Path) -> Dict[str, float | int]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            result: Dict[str, float | int] = {}
            best = payload.get("best_cv_auc")
            idx = payload.get("last_submission_index")
            if isinstance(best, (int, float)):
                result["best_cv_auc"] = float(best)
            if isinstance(idx, int):
                result["last_submission_index"] = idx
            return result
    except (json.JSONDecodeError, OSError):
        pass
    return {}


def save_benchmark_state(path: Path, best_cv_auc: float, last_submission_index: int) -> None:
    payload = {
        "updated_at": datetime.now().isoformat(),
        "best_cv_auc": float(best_cv_auc),
        "last_submission_index": int(last_submission_index),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def save_improved_submission(
    output_dir: Path,
    submission_index: int,
    test_ids: pd.Series,
    test_pred: np.ndarray,
) -> Dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)

    submission_path = output_dir / f"submission{submission_index}.csv"

    pd.DataFrame(
        {
            "anonymised_id": test_ids,
            "employed_status": test_pred,
        }
    ).to_csv(submission_path, index=False)

    return {
        "submission_path": str(submission_path),
    }


def prediction_hash(values: np.ndarray) -> str:
    rounded = np.round(values.astype(float), 12)
    return hashlib.sha1(rounded.tobytes()).hexdigest()


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

    if args.baseline_metrics_file.exists():
        try:
            payload = json.loads(args.baseline_metrics_file.read_text(encoding="utf-8"))
            value = payload.get("cv_auc_mean")
            if isinstance(value, (int, float)):
                return float(value)
        except (json.JSONDecodeError, OSError):
            pass

    return float(args.fallback_baseline)


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
    parser.add_argument(
        "--target-improvements",
        type=int,
        default=0,
        help="Stop after writing this many improved submission files (0 means unlimited).",
    )
    parser.add_argument("--batch-size", type=int, default=4, help="Candidates to screen per loop")
    parser.add_argument(
        "--parallel-jobs",
        type=int,
        default=0,
        help="Parallel candidate evaluations per batch. 0 uses cpu_count-1, -1 uses all CPUs.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"), help="Output directory")
    parser.add_argument("--fallback-baseline", type=float, default=0.0, help="Fallback baseline if no reference score is found")
    parser.add_argument("--baseline-override", type=float, default=None, help="Explicit initial baseline to beat")
    parser.add_argument(
        "--baseline-metrics-file",
        type=Path,
        default=Path("outputs/cv_metrics.json"),
        help="Metrics file used as baseline fallback",
    )
    parser.add_argument(
        "--benchmark-state-file",
        type=Path,
        default=Path(".improver_state.json"),
        help="Persistent local benchmark state file used across runs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

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
    benchmark_state = load_benchmark_state(args.benchmark_state_file)
    persisted_best = float(benchmark_state.get("best_cv_auc", baseline))
    best_cv_auc = max(baseline, persisted_best)
    best_candidate_meta: Optional[Dict[str, object]] = None
    seen_prediction_hashes = load_existing_prediction_hashes(args.output_dir)
    parallel_jobs = resolve_parallel_jobs(args.parallel_jobs)
    next_index = next_submission_index(args.output_dir)
    highest_existing_index = next_index - 1

    if (
        highest_existing_index > 0
        and "best_cv_auc" not in benchmark_state
        and args.baseline_override is None
    ):
        raise ValueError(
            "Found existing numbered submissions but no benchmark state file. "
            "Run once with --baseline-override <best_cv_auc_so_far> to initialize local benchmark tracking."
        )

    print(f"Continuous pipeline started at {datetime.now().isoformat()}")
    print(f"Initial baseline to beat: {best_cv_auc:.6f}")
    print(f"Parallel jobs: {parallel_jobs} | Batch size: {args.batch_size}")
    print(f"Next submission filename index: {next_index}")
    if args.target_improvements > 0:
        print(f"Target improved submissions to write: {args.target_improvements}")
    print("Press Ctrl+C to stop.")

    iteration = 0
    improvements_written = 0
    candidate_counter = 0
    history: List[Dict[str, object]] = []

    try:
        while True:
            iteration += 1
            batch_specs: List[Tuple[int, int, str, Dict[str, float]]] = []
            max_attempts = max(args.batch_size * 10, 20)
            attempts = 0

            while len(batch_specs) < args.batch_size and attempts < max_attempts:
                attempts += 1
                global_candidate_id = candidate_counter + 1

                candidate_rng = random.Random(args.seed + (global_candidate_id * 104_729))
                candidate_name, params, _ = sample_candidate(candidate_rng, X_train, best_candidate_meta)

                candidate_counter += 1
                batch_specs.append((candidate_counter, global_candidate_id, candidate_name, params))

            if not batch_specs:
                print(
                    f"\nIteration {iteration}: no candidates found after {attempts} attempts. Retrying next loop."
                )
                continue

            print(f"\nIteration {iteration}: screening batch of {len(batch_specs)} candidates")

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
                for _, global_candidate_id, candidate_name, params in batch_specs
            )

            screen_results: List[Dict[str, object]] = []
            for batch_spec, result in zip(batch_specs, raw_screen_results):
                _, global_candidate_id, _, _ = batch_spec
                enriched = dict(result)
                enriched["global_candidate_id"] = global_candidate_id
                screen_results.append(enriched)

            best_screen = max(screen_results, key=lambda item: float(item["mean_auc"]))
            candidate_name = str(best_screen["candidate_name"])
            params = dict(best_screen["params"])
            global_candidate_id = int(best_screen["global_candidate_id"])
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
            else:
                fold_scores = list(best_screen["fold_scores"])
                mean_auc = quick_mean_auc
                std_auc = quick_std_auc

            improved = mean_auc >= (best_cv_auc + args.min_improvement)
            print(f"  Mean AUC={mean_auc:.6f} Std={std_auc:.6f} | Best={best_cv_auc:.6f} | Improved={improved}")

            run_info: Dict[str, object] = {
                "iteration": iteration,
                "batch_size": args.batch_size,
                "parallel_jobs": parallel_jobs,
                "candidate_name": candidate_name,
                "global_candidate_id": global_candidate_id,
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
                        submission_index=next_index,
                        test_ids=test_ids,
                        test_pred=test_pred,
                    )
                    seen_prediction_hashes.add(pred_hash)
                    run_info["artifacts"] = artifacts
                    run_info["best_after"] = best_cv_auc
                    improvements_written += 1
                    run_info["improvements_written"] = improvements_written
                    run_info["submission_index"] = next_index
                    save_benchmark_state(
                        path=args.benchmark_state_file,
                        best_cv_auc=best_cv_auc,
                        last_submission_index=next_index,
                    )
                    next_index += 1
                    print(f"  Improvement accepted. New best CV AUC: {best_cv_auc:.6f}")
                    print(f"  New submission: {artifacts['submission_path']}")

                    if args.target_improvements > 0 and improvements_written >= args.target_improvements:
                        print(
                            f"Reached target improvements ({improvements_written}/{args.target_improvements}). Stopping."
                        )
                        return

            history.append(run_info)

            if args.max_iterations > 0 and iteration >= args.max_iterations:
                print("Max iterations reached. Stopping.")
                break

    except KeyboardInterrupt:
        print("\nStopped by user.")


if __name__ == "__main__":
    main()
