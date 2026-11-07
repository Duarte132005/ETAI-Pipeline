"""Run with ``python main.py``.

Load and clean data -> deduplicate -> lock the test set -> development-set CV
-> optional nested CV and final tuning -> save evaluation outputs -> reports
-> refit on all development rows -> full report.
"""

import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import yaml
from sklearn.model_selection import StratifiedKFold

from src.data import load_data
from src.evaluate import (
    build_scorers,
    cross_validate_pipeline,
    cv_report,
    fairness_report,
    oof_classification_report,
)
from src.model import build_pipeline
from src.preprocessing import (
    clean_dataset,
    drop_duplicate_rows,
    split_dev_test,
    split_features_target,
)
from src.results import save_run, save_scores
from src.tuning import nested_cross_validate, tune_pipeline, tuning_report


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("config.yaml must contain a configuration mapping.")
    return config


def prepare_oof_predictions(y_true, oof_proba, threshold: float = 0.5):
    """Return positive-class probabilities and 0/1 predictions for a binary target."""
    y_true = np.asarray(y_true)
    probabilities = np.asarray(oof_proba, dtype=float)
    if probabilities.ndim == 2 and probabilities.shape[1] == 2:
        probabilities = probabilities[:, 1]
    if probabilities.ndim != 1 or len(probabilities) != len(y_true):
        raise ValueError("OOF probabilities must contain one value per development row.")
    if y_true.ndim != 1 or not np.isin(y_true, [0, 1]).all():
        raise ValueError("This reporting code requires target labels encoded as 0 and 1.")
    if not np.isfinite(probabilities).all():
        raise ValueError("OOF probabilities contain missing or infinite values.")
    if ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("OOF probabilities must be between 0 and 1.")
    if not 0 < threshold < 1:
        raise ValueError("The probability threshold must be between 0 and 1.")
    return probabilities, (probabilities >= threshold).astype(int)


def save_evaluation_outputs(directory, config, fold_scores, final_scores,
                            y_dev, oof_proba, oof_pred, extras_dev, best_params):
    """Save computed results before report generation or final refitting can fail."""
    directory.mkdir(parents=True, exist_ok=True)
    fold_scores.to_csv(directory / "baseline_folds.csv", index=False)
    final_scores.to_csv(directory / "final_folds.csv", index=False)
    predictions = pd.DataFrame({
        "y_true": np.asarray(y_dev),
        "oof_proba": oof_proba,
        "oof_pred": oof_pred,
    })
    sensitive_attr = config["data"]["sensitive_attr"]
    if sensitive_attr in extras_dev:
        # Assign by position: development rows may retain their original index.
        predictions[sensitive_attr] = np.asarray(extras_dev[sensitive_attr])
    predictions.to_csv(directory / "oof_predictions.csv", index=False)
    with open(directory / "config.yaml", "w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    with open(directory / "best_params.yaml", "w", encoding="utf-8") as handle:
        yaml.safe_dump(best_params, handle, sort_keys=False)


def main():
    config = load_config()
    run = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    model_type = config["model"]["type"]
    cv_config = config["cv"]
    metrics = cv_config["metrics"]
    selection_metric = cv_config["selection_metric"]
    if selection_metric not in metrics:
        raise ValueError("cv.selection_metric must be included in cv.metrics.")
    scorers = build_scorers(metrics)
    n_jobs = cv_config.get("n_jobs", 1)

    selection_config = config.get("feature_selection") or {}
    selection_enabled = selection_config.get("enabled", False)
    tuning_config = config.get("tuning") or {}
    tuning_enabled = tuning_config.get("enabled", False)
    search_space = {}
    if tuning_enabled:
        # Validate before spending time on the initial cross-validation.
        search_spaces = tuning_config.get("search_spaces") or {}
        if model_type not in search_spaces:
            raise ValueError(
                f"Tuning is enabled but tuning.search_spaces has no entry for '{model_type}'."
            )
        search_space = dict(search_spaces[model_type])
        if selection_enabled:
            search_space.update(selection_config.get("search_space") or {})

    output_config = config.get("output") or {}
    results_dir = Path(output_config.get("results_dir", "results"))
    evaluation_dir = results_dir / "evaluations" / run
    scoreboard_path = Path(output_config.get("scoreboard", "results/runs.csv"))
    scoreboard_path.parent.mkdir(parents=True, exist_ok=True)
    threshold = 0.5

    df_raw = load_data(config["data"]["path"])
    df_clean = clean_dataset(df_raw, config["diagnostics"])
    df_clean = drop_duplicate_rows(df_clean, config["diagnostics"].get("id_column"))
    mnar_sources = config["preprocessing"].get("mnar_indicator_sources", [])
    X, y, extras = split_features_target(df_clean, config["data"], mnar_sources)
    X_dev, X_test, y_dev, y_test, extras_dev, extras_test = split_dev_test(
        X, y, extras,
        test_size=config["test_set"]["size"],
        random_state=config["test_set"]["random_state"],
    )
    # Only development rows are passed to evaluation, tuning and refitting.
    shuffle = cv_config.get("shuffle", True)
    cv = StratifiedKFold(
        n_splits=cv_config["n_splits"],
        shuffle=shuffle,
        random_state=cv_config.get("random_state", 42) if shuffle else None,
    )
    selection_label = (
        f"min_votes {selection_config.get('min_votes', 2)}" if selection_enabled else "off"
    )
    baseline_selection_label = selection_label
    print(
        f"Model: {model_type} | selection metric: {selection_metric} | "
        f"{cv_config['n_splits']}-fold CV | feature selection: {selection_label}"
    )
    print(f"Data: {len(X_dev)} development rows | {len(X_test)} test rows locked (not used)")

    # Cleanup also runs if fitting, reporting or saving raises an exception.
    with tempfile.TemporaryDirectory(prefix="pipeline_cache_") as cache_dir:
        pipeline = build_pipeline(
            config["preprocessing"], config["model"], selection_config, memory=cache_dir
        )
        floor_pipeline = build_pipeline(config["preprocessing"], {"type": "dummy"})
        floor_scores, _ = cross_validate_pipeline(
            floor_pipeline, X_dev, y_dev, cv,
            {selection_metric: scorers[selection_metric]}, selection_metric, n_jobs,
        )
        fold_scores, oof_proba = cross_validate_pipeline(
            pipeline, X_dev, y_dev, cv, scorers, selection_metric, n_jobs
        )
        floor = floor_scores["validation"].mean()
        untuned = fold_scores["validation"].mean()
        final_scores = fold_scores
        study = None
        best_params = {}
        print("\nDefault hyperparameters (config.yaml)")
        print(
            f"  {selection_metric}: {untuned:.3f} +/- "
            f"{fold_scores['validation'].std(ddof=1):.3f} | "
            f"train-validation gap: {fold_scores['gap'].mean():+.3f} | "
            f"no-model floor: {floor:.3f}"
        )

        if tuning_enabled:
            n_trials = tuning_config["n_trials"]
            tuning_seed = tuning_config["random_state"]
            optuna.logging.set_verbosity(optuna.logging.WARNING)
            inner_cv = StratifiedKFold(
                n_splits=tuning_config["n_splits"], shuffle=True, random_state=tuning_seed
            )
            print(f"\nTuning ({n_trials} trials per study, nested CV) ...", flush=True)
            final_scores, oof_proba = nested_cross_validate(
                pipeline, X_dev, y_dev, cv, inner_cv, scorers, selection_metric,
                search_space, n_trials, tuning_seed, n_jobs=n_jobs,
            )
            print(
                f"  Nested-CV {selection_metric}: {final_scores['validation'].mean():.3f} "
                f"+/- {final_scores['validation'].std(ddof=1):.3f}"
            )
            print("\nChoosing final hyperparameters on all development rows ...", flush=True)
            pipeline, study = tune_pipeline(
                pipeline, X_dev, y_dev, inner_cv, scorers[selection_metric],
                search_space, n_trials, tuning_seed, n_jobs=n_jobs,
            )
            best_params = dict(study.best_params)
            print("  Chosen hyperparameters:")
            for name, value in best_params.items():
                value_text = f"{value:.4g}" if isinstance(value, float) else str(value)
                print(f"    {name:<30s} {value_text}")

        label = "tuned procedure" if tuning_enabled else "default hyperparameters"
        print(f"\nAll metrics ({label}, mean +/- std)")
        for name in metrics:
            print(f"  {name:<18s} {final_scores[name].mean():.3f} "
                  f"+/- {final_scores[name].std(ddof=1):.3f}")

        # Classification/fairness reports receive labels; probability metrics keep probabilities.
        oof_proba, oof_pred = prepare_oof_predictions(y_dev, oof_proba, threshold)
        save_evaluation_outputs(
            evaluation_dir, config, fold_scores, final_scores,
            y_dev, oof_proba, oof_pred, extras_dev, best_params,
        )
        print(f"\nFold scores, OOF predictions and parameters saved: {evaluation_dir}", flush=True)

        selection_label = (
            f"min_votes {pipeline.named_steps['select'].min_votes}" if selection_enabled else "off"
        )
        row = {
            "run": run, "data": config["data"]["path"], "model": model_type,
            "tuned": tuning_enabled, "encoder": config["preprocessing"]["encoder"],
            "scaler": config["preprocessing"]["scaler"], "feature_selection": selection_label,
            "params": str(config["model"].get("params")),
            "tuned_params": str(best_params) if tuning_enabled else "",
            "metric": selection_metric, "mean": final_scores["validation"].mean(),
            "std": final_scores["validation"].std(ddof=1), "untuned": untuned, "floor": floor,
            **{f"fold_{i}": score for i, score in enumerate(final_scores["validation"], start=1)},
        }
        # Save evaluation scores before either report or the final refit can fail.
        runs = save_scores(str(scoreboard_path), row)
        same = runs[
            (runs["data"] == row["data"]) & (runs["metric"] == selection_metric)
            & (runs["encoder"] == row["encoder"]) & (runs["scaler"] == row["scaler"])
        ].copy()
        columns = ["run", "model", "tuned", "feature_selection", "mean", "std", "untuned", "floor"]
        board_rows = same.sort_values("mean", ascending=False).reindex(columns=columns)
        board_rows["feature_selection"] = board_rows["feature_selection"].fillna("not recorded")
        board = board_rows.rename(columns={"mean": selection_metric}).to_string(
            index=False, float_format=lambda value: f"{value:.3f}"
        )
        print(f"\nScoreboard (same data, metric, encoder and scaler; best first)\n{board}")

        # Formatting is deliberately deferred until after the raw results are saved.
        report = (
            f"Hyperparameters from config.yaml: {config['model'].get('params')}\n"
            f"Baseline feature selection: {baseline_selection_label}\n"
            f"Final-refit feature selection: {selection_label}\n"
            f"Classification/fairness threshold: {threshold}\n"
            f"Development rows: {len(X_dev)}; locked, unevaluated test rows: {len(X_test)}\n"
            f"Evaluation outputs: {evaluation_dir}\n\n"
            + cv_report(fold_scores, metrics, selection_metric, "Cross-validation")
            + f"\n\nNo-model floor (dummy, same folds): {selection_metric} = {floor:.3f}"
        )
        if tuning_enabled:
            report += "\n\n" + cv_report(
                final_scores, metrics, selection_metric, "Nested cross-validation"
            )
            report += "\n\n" + tuning_report(study, final_scores, selection_metric)
        report += "\n\n" + oof_classification_report(y_dev, oof_pred)
        report += "\n" + fairness_report(
            y_dev, oof_pred, extras_dev, sensitive_attr=config["data"]["sensitive_attr"]
        )

        pipeline.fit(X_dev, y_dev)
        refit = f"Final model: {model_type} refit on all {len(X_dev)} development rows."
        if selection_enabled:
            all_features = pipeline.named_steps["prep"].get_feature_names_out()
            kept = pipeline.named_steps["select"].get_feature_names_out(all_features)
            refit += f"\nFeatures kept: {len(kept)} of {len(all_features)}: {', '.join(map(str, kept))}"
        report += f"\n{refit}\n\nScoreboard:\n{board}\n"
        path = save_run(str(results_dir), run, config, report)
        print(f"\n{refit}\nFull report: {path}")


if __name__ == "__main__":
    main()

