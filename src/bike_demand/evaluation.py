"""Validation-only selection, then a single held-out comparison of frozen families."""

from __future__ import annotations

import platform
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.metrics import make_scorer, mean_absolute_error, mean_squared_error, r2_score
from threadpoolctl import threadpool_limits

from bike_demand import __version__
from bike_demand.data import RAW_FEATURES, split_metadata, write_json
from bike_demand.models import SEED, candidates

SEASONS = {1: "Winter", 2: "Spring", 3: "Summer", 4: "Fall"}


def metrics(actual, predicted) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "r2": float(r2_score(actual, predicted)),
    }


def grouped_errors(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    rows = []
    for value, group in frame.groupby(column):
        rows.append(
            {
                column: value,
                "rows": len(group),
                **metrics(group.actual, group.predicted),
                "bias": float((group.predicted - group.actual).mean()),
                "mean_actual": float(group.actual.mean()),
            }
        )
    return pd.DataFrame(rows)


def run_benchmark(
    partitions: dict[str, pd.DataFrame],
    output: Path,
    provenance: dict,
    *,
    fast: bool = False,
) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    with threadpool_limits(limits=2):
        return _run(partitions, output, provenance, fast=fast)


def _run(partitions, output, provenance, *, fast):
    train, validation, test = (partitions[name] for name in ("train", "validation", "test"))
    configurations = candidates(fast=fast)
    validation_results, family_winners = [], {}
    for family, label, params, estimator in configurations:
        fitted = clone(estimator).fit(train[RAW_FEATURES], train.cnt)
        result = {
            "family": family,
            "label": label,
            "parameters": params,
            **metrics(validation.cnt, fitted.predict(validation[RAW_FEATURES])),
        }
        validation_results.append(result)
        previous = family_winners.get(family)
        if previous is None or result["mae"] < previous[0]["mae"]:
            family_winners[family] = (result, estimator)
        print(f"Validation: {label:32s} MAE {result['mae']:.3f}", flush=True)

    selected = min(family_winners, key=lambda name: family_winners[name][0]["mae"])
    train_validation = pd.concat([train, validation], ignore_index=True)
    test_results, final_models = [], {}
    predictions = test[["timestamp", "hr", "season", "workingday", "cnt"]].copy()
    predictions.rename(columns={"cnt": "actual"}, inplace=True)
    for family, (choice, estimator) in family_winners.items():
        fitted = clone(estimator).fit(train_validation[RAW_FEATURES], train_validation.cnt)
        predicted = fitted.predict(test[RAW_FEATURES])
        final_models[family] = fitted
        test_results.append(
            {
                "family": family,
                "label": choice["label"],
                "parameters": choice["parameters"],
                **metrics(test.cnt, predicted),
            }
        )
        predictions[f"prediction_{family}"] = predicted
    predictions["predicted"] = predictions[f"prediction_{selected}"]
    predictions.to_csv(output / "predictions.csv", index=False, float_format="%.8f")
    grouped_errors(predictions, "hr").to_csv(output / "error_by_hour.csv", index=False)
    seasons = grouped_errors(predictions, "season")
    seasons["label"] = seasons.season.map(SEASONS)
    seasons.to_csv(output / "error_by_season.csv", index=False)

    scorer = make_scorer(mean_absolute_error, greater_is_better=False)
    importance = permutation_importance(
        final_models[selected],
        test[RAW_FEATURES],
        test.cnt,
        scoring=scorer,
        n_repeats=5,
        random_state=SEED,
        n_jobs=1,
        max_samples=min(1000, len(test)),
    )
    pd.DataFrame(
        {
            "feature": RAW_FEATURES,
            "mae_increase": importance.importances_mean,
            "std": importance.importances_std,
        }
    ).sort_values("mae_increase", ascending=False).to_csv(
        output / "feature_importance.csv",
        index=False,
    )
    joblib.dump(final_models[selected], output / "model.joblib", compress=3)
    full_frame = pd.concat([train, validation, test])
    expected_hours = (
        int((full_frame.timestamp.max() - full_frame.timestamp.min()).total_seconds() / 3600) + 1
    )
    result = {
        "schema_version": 1,
        "project_version": __version__,
        "task": "Retrospective hourly demand prediction using observed weather",
        "source": provenance,
        "seed": SEED,
        "cpu_threads": 2,
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "data_quality": {
            "rows": len(full_frame),
            "missing_calendar_hours": expected_hours - len(full_frame),
            "missing_hours_policy": "Omitted hours stay omitted; no zero-count imputation.",
        },
        "features": RAW_FEATURES,
        "excluded_columns": ["cnt", "casual", "registered", "instant", "yr"],
        "splits": split_metadata(partitions),
        "protocol": {
            "selection": "Minimum validation MAE; fixed finite grid; no test-based tuning.",
            "refit": "Each family's validation winner refitted on train + validation only.",
            "test": "One final comparison of frozen family configurations on the same held-out dates.",
            "preprocessing": "Imputation, scaling and one-hot encoding fitted within each fit partition.",
            "predictions": "All regression predictions clipped at zero; counts remain continuous.",
            "importance": "Post-evaluation permutation diagnostic: five repeats, max 1000 test rows; "
            "not used for selection and not a causal effect.",
        },
        "selected_family": selected,
        "validation_results": validation_results,
        "test_results": test_results,
    }
    write_json(output / "benchmark.json", result)
    write_model_card(output / "MODEL_CARD.md", result)
    return result


def write_model_card(path: Path, result: dict) -> None:
    rows = ["| Model | Test MAE ↓ | Test RMSE ↓ | Test R² ↑ |", "|---|---:|---:|---:|"]
    for item in result["test_results"]:
        rows.append(
            f"| {item['label']} | {item['mae']:.3f} | {item['rmse']:.3f} | {item['r2']:.3f} |"
        )
    splits = result["splits"]
    selected = next(r for r in result["test_results"] if r["family"] == result["selected_family"])
    path.write_text(
        "# Bike Demand Lab · model card\n\n"
        f"Source: **{result['source']['kind']}**. Task: {result['task']}.\n\n"
        f"Selected by validation MAE: **{selected['label']}**. "
        "Scores below are computed after refitting each frozen family winner on train + validation.\n\n"
        + "\n".join(rows)
        + "\n\n"
        "## Evaluation design\n\n"
        + "\n".join(
            f"- {name}: {info['rows']:,} rows, {info['start']} to {info['end']}."
            for name, info in splits.items()
        )
        + "\n\nNo random split. All transformations fit within their training partition. "
        "The fixed grid uses validation MAE; test scores do not choose hyperparameters. "
        "The final model uses all train and validation rows, never test rows. "
        "This is a single historical holdout, not rolling-origin validation.\n\n"
        "## Inputs and target\n\n"
        "Target: total hourly rentals (`cnt`). Inputs: calendar fields, a deterministic date "
        "trend and recorded weather. `casual` and `registered` sum to the target and are "
        "excluded, along with the record ID, `instant`. `yr` is excluded because date already "
        "provides the time trend. The feature allowlist applies during training and inference. "
        "Numeric median imputation and scaling plus categorical most-frequent imputation and "
        "one-hot encoding are learned in the pipeline. No missing target hours are invented.\n\n"
        "## Intended use and limitations\n\n"
        "A reproducible educational benchmark of historical demand in Washington, D.C., "
        "2011–2012. Weather is observed, so these scores do not establish future forecast "
        "quality. Deployment would need forecast-available weather, rolling temporal evaluation, "
        "updated local data, and uncertainty estimates. Demand growth, missing hours and unusual "
        "events can shift errors. Aggregate rental counts do not describe station availability. "
        "No external, university or production validation is claimed.\n\n"
        "Permutation importance is the increase in held-out MAE after shuffling one raw input, "
        "averaged over five seeded repeats on at most 1,000 test rows. Correlated calendar and "
        "weather fields share information; shuffling can create unrealistic combinations. "
        "Standard deviations describe repeat variability, not confidence intervals. "
        "This diagnostic did not influence model choice.\n\n"
        "## Reproducibility and attribution\n\n"
        f"Seed: {result['seed']}; CPU thread limit: {result['cpu_threads']}. "
        f"Data SHA-256: `{result['source']['sha256']}`. "
        "See `benchmark.json` for versions, all validation candidates, exact boundaries and metrics.\n\n"
        "UCI Bike Sharing dataset, Hadi Fanaee-T (2013), "
        "[DOI 10.24432/C5W894](https://doi.org/10.24432/C5W894), CC BY 4.0. "
        "Fanaee-T and Gama, *Event labeling combining ensemble detectors and background knowledge*, "
        "DOI 10.1007/s13748-013-0040-3. See `data/PROVENANCE.md`. "
        "For synthetic demo runs, this citation identifies the schema inspiration only; "
        "their rows and scores are synthetic.\n"
    )
