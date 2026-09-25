import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency


def _cramers_v(confusion_matrix: pd.DataFrame) -> float:
    """Bias-corrected Cramer's V effect size for a chi-square test of association."""
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2_corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    r_corr = r - ((r - 1) ** 2) / (n - 1)
    k_corr = k - ((k - 1) ** 2) / (n - 1)
    return float(np.sqrt(phi2_corr / min(k_corr - 1, r_corr - 1)))


def test_missingness_mechanism(df: pd.DataFrame, target_col: str, candidate_predictors: list) -> pd.DataFrame:
    """
    For `target_col`'s missing-value indicator, test association against each column in
    `candidate_predictors` via chi-square + Cramer's V. Returns one row per predictor,
    sorted by association strength (strongest first).

    Reading the verdict: a max Cramer's V well under 0.1 across every predictor reads as
    MCAR (missing completely at random -- safe to impute simply). Anything well above
    that (roughly >= 0.2) reads as MAR/MNAR, and the top predictor usually points at why.
    """
    indicator = df[target_col].isna()
    rows = []
    for predictor in candidate_predictors:
        if predictor == target_col or predictor not in df.columns:
            continue
        sub = pd.DataFrame({"missing": indicator, "predictor": df[predictor]}).dropna(subset=["predictor"])
        if sub["predictor"].nunique() < 2 or sub["missing"].nunique() < 2:
            continue
        table = pd.crosstab(sub["missing"], sub["predictor"])
        chi2, p, _, _ = chi2_contingency(table)
        v = _cramers_v(table)
        rows.append({"predictor": predictor, "cramers_v": round(v, 3), "p_value": p, "n": len(sub)})
    return pd.DataFrame(rows).sort_values("cramers_v", ascending=False).reset_index(drop=True)


def flag_invalid_values(df: pd.DataFrame, rules: dict) -> pd.DataFrame:
    """
    Applies a dict of {column: {"min": ..., "max": ...}} domain rules (either bound is
    optional) and converts violations to NaN **in place** on `df`. An "impossible but
    not missing" value (an age of -3, a COMPAS decile score of 15) counts as missing
    once this runs -- `.isna()` alone would never have caught it.

    Returns a small report: how many violations were found per column.
    """
    report_rows = []
    for column, bounds in rules.items():
        if column not in df.columns:
            continue
        numeric = pd.to_numeric(df[column], errors="coerce")
        lower_ok = numeric >= bounds["min"] if "min" in bounds else pd.Series(True, index=numeric.index)
        upper_ok = numeric <= bounds["max"] if "max" in bounds else pd.Series(True, index=numeric.index)
        violations = numeric.notna() & ~(lower_ok & upper_ok)
        report_rows.append({"column": column, "rule": bounds, "violations": int(violations.sum())})
        df.loc[violations, column] = np.nan
    return pd.DataFrame(report_rows)


def find_duplicates(df: pd.DataFrame, id_column: str = None) -> dict:
    """
    Checks duplicates two ways: exact row duplication, and (if `id_column` is given)
    repeated identifiers. The two checks can disagree in general -- a duplicate case
    re-entered with even one typo'd field would only be caught by the id check -- so
    both are reported rather than trusting either one alone.
    """
    result = {"exact_row_duplicates": int(df.duplicated().sum())}
    if id_column and id_column in df.columns:
        result["repeated_ids"] = int(df[id_column].duplicated().sum())
    return result


def apply_consistency_rules(df: pd.DataFrame,rules: list,) -> pd.DataFrame:
    """Apply configured numeric-to-category rules in place."""
    report = []

    for rule in rules:
        source = rule["source"]
        destination = rule["destination"]
        action = rule.get("action", "replace")

        if action not in {"replace", "set_missing"}:
            raise ValueError(f"Unsupported consistency action: {action}")

        numeric = pd.to_numeric(df[source], errors="coerce")

        expected = pd.cut(
            numeric,
            bins=rule["bins"],
            labels=rule["labels"],
            right=rule.get("right", False),
            include_lowest=True,
        ).astype("string")

        actual = df[destination].astype("string").str.strip()

        # Only compare rows for which a category can be calculated.
        mismatch = (
            expected.notna()
            & actual.notna()
            & actual.ne(expected)
        ).fillna(False)

        missing_category = expected.notna() & actual.isna()

        if action == "replace":
            # Trust the source: correct mismatches and fill missing categories.
            change = mismatch | missing_category
            df.loc[change, destination] = expected.loc[change]
        else:
            # Mark inconsistent categories as missing for later imputation.
            change = mismatch
            df.loc[change, destination] = np.nan

        report.append({
            "source": source,
            "destination": destination,
            "mismatches": int(mismatch.sum()),
            "values_changed": int(change.sum()),
            "action": action,
        })

    return pd.DataFrame(report)