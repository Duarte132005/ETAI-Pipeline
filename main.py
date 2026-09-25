"""Entry point for the baseline predictive pipeline."""

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

from src.data import load_data
from src.model import build_model
from src.evaluate import evaluate, fairness_report
from src.results import save_run
from src.preprocessing import clean_dataset


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    config = load_config()

    df_raw = load_data(config["data"]["path"])
    df_clean = clean_dataset(df_raw, config["diagnostics"]).dropna()

    target = config["data"]["target"]
    sensitive_attr = config["data"]["sensitive_attr"]

    y = df_clean[target]
    extras = df_clean[[sensitive_attr, "score_text"]]
    X = df_clean.drop(
        columns=[target, sensitive_attr, *config["data"]["drop_columns"]],
        errors="ignore",
    )

    X_train, X_test, y_train, y_test, extras_train, extras_test = train_test_split(
        X,
        y,
        extras,
        test_size=config["split"]["test_size"],
        random_state=config["split"]["random_state"],
        stratify=y,
    )

    # The model needs numbers, so encode text columns after splitting.
    X_train = pd.get_dummies(X_train)
    X_test = pd.get_dummies(X_test).reindex(
        columns=X_train.columns, fill_value=0
    )

    model = build_model(config["model"])
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    report = evaluate(y_train, y_train_pred, y_test, y_test_pred)
    report += "\n" + fairness_report(
        y_test, y_test_pred, extras_test, sensitive_attr=sensitive_attr
    )

    results_dir = config.get("output", {}).get("results_dir", "results")
    path = save_run(results_dir, config, report)
    print(f"Full results saved to {path}")


if __name__ == "__main__":
    main()