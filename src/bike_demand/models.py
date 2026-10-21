"""Explicit feature allowlist and preprocessing fitted inside each pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from bike_demand.data import FORBIDDEN_FEATURES, RAW_FEATURES

SEED = 42
NUMERIC = ["temp", "atemp", "hum", "windspeed", "days_since_start"]
CATEGORICAL = ["hr", "mnth", "weekday", "season", "holiday", "workingday", "weathersit"]


class FeatureBuilder(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        self.transform(X)
        return self

    def transform(self, X):
        missing = set(RAW_FEATURES) - set(X.columns)
        if missing:
            raise ValueError(f"Missing inference features: {sorted(missing)}")
        result = X.loc[:, RAW_FEATURES].copy()
        result["days_since_start"] = (
            pd.to_datetime(result.pop("dteday"), errors="raise") - pd.Timestamp("2011-01-01")
        ).dt.total_seconds() / 86400
        for column in NUMERIC + CATEGORICAL:
            result[column] = pd.to_numeric(result[column], errors="raise")
        if FORBIDDEN_FEATURES & set(result.columns):
            raise ValueError("Target-related or record-identifier column entered the feature set.")
        return result


class NonnegativeRegressor(RegressorMixin, BaseEstimator):
    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y):
        self.estimator_ = clone(self.estimator).fit(X, y)
        return self

    def predict(self, X):
        return np.maximum(0, self.estimator_.predict(X))


class CalendarBaseline(RegressorMixin, BaseEstimator):
    """Training-only mean count for each (hour, workingday) pair."""

    def fit(self, X, y):
        data = X[["hr", "workingday"]].copy()
        data["count"] = np.asarray(y)
        self.means_ = data.groupby(["hr", "workingday"])["count"].mean().to_dict()
        self.fallback_ = float(np.mean(y))
        return self

    def predict(self, X):
        return np.array(
            [
                self.means_.get((hour, working), self.fallback_)
                for hour, working in X[["hr", "workingday"]].itertuples(index=False, name=None)
            ]
        )


def pipeline(estimator) -> Pipeline:
    numeric = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    preprocessing = ColumnTransformer(
        [
            ("numeric", numeric, NUMERIC),
            ("categorical", categorical, CATEGORICAL),
        ]
    )
    return Pipeline(
        [
            ("features", FeatureBuilder()),
            ("preprocess", preprocessing),
            ("regressor", NonnegativeRegressor(estimator)),
        ]
    )


def candidates(*, fast: bool = False) -> list[tuple[str, str, dict, BaseEstimator]]:
    result = [
        ("mean", "Training mean", {}, pipeline(DummyRegressor(strategy="mean"))),
        ("calendar", "Hour × workday mean", {}, CalendarBaseline()),
    ]
    for alpha in [10.0] if fast else [1.0, 10.0, 100.0]:
        result.append(
            ("ridge", f"Ridge α={alpha:g}", {"alpha": alpha}, pipeline(Ridge(alpha=alpha)))
        )
    for leaves in [15] if fast else [15, 31]:
        params = {
            "max_leaf_nodes": leaves,
            "max_iter": 80 if fast else 200,
            "learning_rate": 0.08,
            "min_samples_leaf": 30,
            "l2_regularization": 1.0,
            "random_state": SEED,
            "early_stopping": False,
        }
        result.append(
            (
                "hist_gradient_boosting",
                f"Gradient boosting · {leaves} leaves",
                params,
                pipeline(HistGradientBoostingRegressor(**params)),
            )
        )
    return result
