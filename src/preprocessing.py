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


import numpy as np
import pandas as pd

from src.data_diagnosis import flag_invalid_values,apply_consistency_rules

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
    """Clean data using configuration-driven rules."""
    out = df.copy()
    placeholder_tokens = set(
        diagnostics_config.get("placeholder_tokens", [])
    )

    # Convert configured text columns to numeric
    for col in diagnostics_config.get("numeric_text_columns", []):
        if col in out.columns:
            out[col] = pd.to_numeric(
                out[col].replace(list(placeholder_tokens), np.nan),
                errors="coerce",
            )

    # Convert values outside configured bounds to NaN
    flag_invalid_values(
        out,
        diagnostics_config.get("validity_rules", {}),
    )

    # Standardize categories and preserve missing values
    out = _canonicalize_categories(
        out,
        diagnostics_config.get("canonical_categories", {}),
        placeholder_tokens,
    )

    # NEW: fix relationships between columns using configured rules
    apply_consistency_rules(
        out,
        diagnostics_config.get("consistency_rules", []),
    )

    out = out.drop_duplicates()

    id_column = diagnostics_config.get("id_column")
    if id_column and id_column in out.columns:
        out = out.drop_duplicates(subset=id_column, keep="first")

    columns_to_drop = [
        c for c in diagnostics_config.get("redundant_columns", [])
        if c in out.columns
    ]
    out = out.drop(columns=columns_to_drop)

    return out



