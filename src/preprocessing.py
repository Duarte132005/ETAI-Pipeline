"""
Preprocessing -- deliberately minimal for week 2.

This is intentionally the weakest part of the pipeline:
    - missing values are simply dropped (no imputation strategy)
    - categorical columns are one-hot encoded with no thought given to unseen categories or cardinality
    - a single train/test split is used (no cross-validation)

You will replace this with something better in the coming weeks.

One thing that is NOT naive, on purpose: `sensitive_attr` (race) is kept out of the model's input features entirely. It's split alongside the data so it's still available afterwards -- not to train on, but to check whether the model treats different groups differently. See src/evaluate.py:fairness_report.
"""
import pandas as pd
from sklearn.model_selection import train_test_split


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data_diagnosis import flag_invalid_values

def _canonicalize_categories(df: pd.DataFrame, columns_and_maps: dict, placeholder_tokens: set) -> pd.DataFrame:
    out = df.copy()
    for col, mapping in columns_and_maps.items():
        if col not in out.columns:
            continue
        cleaned = out[col].astype(str).str.strip()
        lowered = cleaned.str.lower()
        out[col] = lowered.map(mapping).fillna(cleaned)
        out.loc[out[col].astype(str).str.strip().isin(placeholder_tokens), col] = np.nan
    return out


def clean_dataset(df: pd.DataFrame, diagnostics_config: dict) -> pd.DataFrame:
    """
    Applies this week's diagnosis: category cleanup, domain-rule/placeholder -> NaN
    conversion, de-duplication, and redundant-column removal. Target-agnostic -- safe
    to call on label-free inference data, since none of this depends on a target column.
    """
    out = df.copy()
    placeholder_tokens = set(diagnostics_config.get("placeholder_tokens", []))

    # numeric columns that load as text purely because of a placeholder token
    for col in diagnostics_config.get("numeric_text_columns", []):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col].replace(list(placeholder_tokens), np.nan), errors="coerce")

    flag_invalid_values(out, diagnostics_config.get("validity_rules", {}))

    out = _canonicalize_categories(out, diagnostics_config.get("canonical_categories", {}), placeholder_tokens)

    out = out.drop_duplicates()
    id_column = diagnostics_config.get("id_column")
    if id_column and id_column in out.columns:
        out = out.drop_duplicates(subset=id_column, keep="first")

    columns_to_drop = [c for c in diagnostics_config.get("redundant_columns", []) if c in out.columns]
    out = out.drop(columns=columns_to_drop)

    return out


def split_train_test(X, y, extras, test_size: float, random_state: int):
    """
    Stratified split of X, y, and the extras frame (race/score_text, kept aside for the
    fairness report) together, so all three stay row-aligned. This is the leak-safe
    boundary line -- everything downstream (imputation, encoding, scaling, inside
    build_preprocessor's ColumnTransformer) may only ever be fit on X_train, never on
    X_test or the full dataset.
    """
    X_train, X_test, y_train, y_test, extras_train, extras_test = train_test_split(
        X, y, extras, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test, extras_train, extras_test
