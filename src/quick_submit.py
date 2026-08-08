from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def main() -> None:
    train_path = Path("data/train.csv")
    test_path = Path("data/test.csv")
    out_path = Path("outputs/submission_quick.csv")

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    id_col = "anonymised_id"
    target_col = "employed_status"

    # Compact, high-signal feature set for a quick first submission.
    cat_cols = [
        "gender",
        "province",
        "education_level",
        "status_broad_lag",
        "sample",
        "sample_first",
    ]
    num_cols = [
        "age",
        "work_readiness_score",
        "days_since_last_obs",
        "total_historical_rounds",
        "lag_round",
        "current_round",
        "tenure_lag",
        "employed_lag",
    ]

    use_cat = [c for c in cat_cols if c in train.columns and c in test.columns]
    use_num = [c for c in num_cols if c in train.columns and c in test.columns]
    use_cols = use_cat + use_num

    tr = train[train[target_col].notna()].copy()
    y = tr[target_col].astype(int)

    X_train = tr[use_cols].copy()
    X_test = test[use_cols].copy()

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore", min_frequency=20)),
                    ]
                ),
                use_cat,
            ),
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                    ]
                ),
                use_num,
            ),
        ],
        remainder="drop",
    )

    model = Pipeline(
        steps=[
            ("prep", preprocessor),
            (
                "clf",
                LogisticRegression(
                    max_iter=1500,
                    solver="saga",
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )

    model.fit(X_train, y)
    pred = model.predict_proba(X_test)[:, 1]

    submission = pd.DataFrame(
        {
            "anonymised_id": test[id_col],
            "employed_status": pred,
        }
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(out_path, index=False)

    print(f"Saved quick submission to: {out_path}")
    print(f"Rows: {len(submission)}")
    print(f"Duplicate IDs: {submission['anonymised_id'].duplicated().sum()}")
    print(f"Missing predictions: {submission['employed_status'].isna().sum()}")


if __name__ == "__main__":
    main()
