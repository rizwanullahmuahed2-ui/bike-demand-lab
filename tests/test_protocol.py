import numpy as np
import pytest
from sklearn.base import clone
from sklearn.linear_model import Ridge

from bike_demand.data import RAW_FEATURES, synthetic_frame, temporal_split, validate_frame
from bike_demand.evaluation import run_benchmark
from bike_demand.models import CalendarBaseline, FeatureBuilder, pipeline


def test_temporal_partitions_are_disjoint_and_cover_every_row():
    frame = synthetic_frame()
    parts = temporal_split(frame, "2011-03-02", "2011-03-22")
    assert sum(len(part) for part in parts.values()) == len(frame)
    assert parts["train"].timestamp.max() < parts["validation"].timestamp.min()
    assert parts["validation"].timestamp.max() < parts["test"].timestamp.min()
    with pytest.raises(ValueError, match="partition"):
        temporal_split(frame, "2020-01-01", "2021-01-01")


def test_target_columns_cannot_change_features_or_predictions():
    frame = synthetic_frame(days=4)
    estimator = pipeline(Ridge(alpha=10)).fit(frame, frame.cnt)
    poisoned = frame.copy()
    poisoned[["cnt", "casual", "registered", "instant"]] = 999999
    np.testing.assert_allclose(estimator.predict(frame), estimator.predict(poisoned))
    assert set(FeatureBuilder().transform(frame).columns).isdisjoint(
        {"cnt", "casual", "registered", "instant"}
    )


def test_preprocessing_is_fitted_on_training_only():
    frame = synthetic_frame(days=8)
    train, future = frame.iloc[:96].copy(), frame.iloc[96:].copy()
    future["temp"] = 900
    fitted = pipeline(Ridge()).fit(train[RAW_FEATURES], train.cnt)
    scaler = fitted["preprocess"].named_transformers_["numeric"]["scale"]
    assert scaler.mean_[0] == pytest.approx(train.temp.mean())
    assert np.isfinite(fitted.predict(future[RAW_FEATURES])).all()
    assert scaler.mean_[0] == pytest.approx(train.temp.mean())
    assert clone(fitted) is not fitted


@pytest.mark.parametrize("change", ["duplicate", "negative", "fractional_hour"])
def test_invalid_input_is_rejected(change):
    frame = synthetic_frame(days=2)
    if change == "duplicate":
        frame.loc[1, ["dteday", "hr"]] = frame.loc[0, ["dteday", "hr"]]
    elif change == "negative":
        frame.loc[0, "cnt"] = -1
    else:
        frame["hr"] = frame.hr.astype(float)
        frame.loc[0, "hr"] = 1.5
    with pytest.raises(ValueError):
        validate_frame(frame)


def test_calendar_baseline_has_training_only_fallback():
    frame = synthetic_frame(days=2)
    model = CalendarBaseline().fit(frame.iloc[:5], frame.cnt.iloc[:5])
    assert model.predict(frame.iloc[12:13])[0] == pytest.approx(frame.cnt.iloc[:5].mean())


def test_benchmark_artifacts_and_validation_selection(tmp_path):
    parts = temporal_split(synthetic_frame(), "2011-03-02", "2011-03-22")
    report = run_benchmark(parts, tmp_path, {"kind": "synthetic test", "sha256": "test"}, fast=True)
    winner = min(report["validation_results"], key=lambda row: row["mae"])
    assert report["selected_family"] == winner["family"]
    assert report["splits"]["test"]["rows"] == len(parts["test"])
    assert (tmp_path / "model.joblib").is_file()
    assert (tmp_path / "MODEL_CARD.md").is_file()
    assert all(np.isfinite(row["mae"]) for row in report["test_results"])
