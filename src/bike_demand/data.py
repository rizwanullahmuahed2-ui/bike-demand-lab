"""Verified UCI data acquisition, schema checks, and date-based splits."""

from __future__ import annotations

import hashlib
import io
import json
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ARCHIVE_URL = "https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip"
DATASET_URL = "https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset"
ARCHIVE_SHA256 = "b70182d0d0508e9abbb79306ce5c0cec34869000f8220175ac83d11dbe845401"
HOURLY_SHA256 = "e03de4ee4ef4dc376ac6e04bf829673c6269e8eba5c60fa121640fa2f829504f"
RAW_FEATURES = [
    "dteday",
    "hr",
    "mnth",
    "weekday",
    "season",
    "holiday",
    "workingday",
    "weathersit",
    "temp",
    "atemp",
    "hum",
    "windspeed",
]
FORBIDDEN_FEATURES = {"cnt", "casual", "registered", "instant"}
TRAIN_END = "2012-05-01"
VALIDATION_END = "2012-09-01"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def download(destination: Path) -> Path:
    """Fetch exact documented bytes, checking both archive and hourly file hashes."""
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / "hour.csv"
    if target.exists() and sha256(target) == HOURLY_SHA256:
        return target
    request = urllib.request.Request(ARCHIVE_URL, headers={"User-Agent": "BikeDemandLab/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = response.read(4_000_000)
    if hashlib.sha256(payload).hexdigest() != ARCHIVE_SHA256:
        raise ValueError(
            "UCI archive fingerprint changed; verify provenance before updating hashes."
        )
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        hourly = archive.read("hour.csv")
        readme = archive.read("Readme.txt")
    if hashlib.sha256(hourly).hexdigest() != HOURLY_SHA256:
        raise ValueError("Hourly dataset fingerprint does not match the documented UCI file.")
    target.write_bytes(hourly)
    (destination / "UCI-Readme.txt").write_bytes(readme)
    return target


def load_frame(path: Path, *, require_uci: bool = True) -> pd.DataFrame:
    if require_uci and sha256(path) != HOURLY_SHA256:
        raise ValueError("Expected the verified UCI hour.csv. Use demo for synthetic data.")
    return validate_frame(pd.read_csv(path))


def validate_frame(frame: pd.DataFrame) -> pd.DataFrame:
    missing = set(RAW_FEATURES + ["cnt"]) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    result = frame.copy()
    dates = pd.to_datetime(result["dteday"], errors="raise")
    hours = pd.to_numeric(result["hr"], errors="raise")
    if hours.isna().any() or not hours.between(0, 23).all() or (hours % 1 != 0).any():
        raise ValueError("hr must contain integer hours from 0 to 23.")
    result["timestamp"] = dates + pd.to_timedelta(hours, unit="h")
    if result["timestamp"].isna().any() or result["timestamp"].duplicated().any():
        raise ValueError("Hourly timestamps must be present and unique.")
    if (dates != dates.dt.normalize()).any():
        raise ValueError("dteday must contain dates without a time component.")
    counts = pd.to_numeric(result["cnt"], errors="raise")
    if not np.isfinite(counts).all() or (counts < 0).any() or (counts % 1 != 0).any():
        raise ValueError("Rental counts must be finite nonnegative integers.")
    result["cnt"] = counts
    return result.sort_values("timestamp").reset_index(drop=True)


def temporal_split(
    frame: pd.DataFrame,
    train_end: str = TRAIN_END,
    validation_end: str = VALIDATION_END,
) -> dict[str, pd.DataFrame]:
    left, right = pd.Timestamp(train_end), pd.Timestamp(validation_end)
    if left >= right:
        raise ValueError("Training cutoff must precede validation cutoff.")
    partitions = {
        "train": frame.loc[frame.timestamp < left].copy(),
        "validation": frame.loc[(frame.timestamp >= left) & (frame.timestamp < right)].copy(),
        "test": frame.loc[frame.timestamp >= right].copy(),
    }
    if any(part.empty for part in partitions.values()):
        raise ValueError("Each chronological partition needs at least one record.")
    if not (
        partitions["train"].timestamp.max() < partitions["validation"].timestamp.min()
        and partitions["validation"].timestamp.max() < partitions["test"].timestamp.min()
    ):
        raise ValueError("Temporal partitions overlap.")
    return partitions


def split_metadata(partitions: dict[str, pd.DataFrame]) -> dict:
    return {
        name: {
            "rows": len(part),
            "start": part.timestamp.min().isoformat(),
            "end": part.timestamp.max().isoformat(),
            "target_mean": float(part.cnt.mean()),
        }
        for name, part in partitions.items()
    }


def synthetic_frame(days: int = 100, seed: int = 42) -> pd.DataFrame:
    """Small deterministic fixture; never a substitute for UCI benchmark evidence."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2011-01-01", periods=24 * days, freq="h")
    hour = dates.hour.to_numpy()
    weekday = (dates.dayofweek.to_numpy() + 1) % 7
    working = (dates.dayofweek.to_numpy() < 5).astype(int)
    temp = np.clip(
        0.5 + 0.2 * np.sin(np.arange(len(dates)) / 400) + rng.normal(0, 0.04, len(dates)), 0, 1
    )
    humidity = rng.uniform(0.2, 0.9, len(dates))
    count = (
        np.maximum(
            0,
            35
            + 130 * np.exp(-(((hour - 8) / 2) ** 2)) * working
            + 190 * np.exp(-(((hour - 17) / 3) ** 2))
            + 80 * temp
            - 25 * humidity
            + rng.normal(0, 12, len(dates)),
        )
        .round()
        .astype(int)
    )
    return validate_frame(
        pd.DataFrame(
            {
                "dteday": dates.strftime("%Y-%m-%d"),
                "hr": hour,
                "mnth": dates.month,
                "weekday": weekday,
                "season": ((dates.month % 12) // 3 + 1),
                "holiday": 0,
                "workingday": working,
                "weathersit": rng.integers(1, 4, len(dates)),
                "temp": temp,
                "atemp": temp * 0.95,
                "hum": humidity,
                "windspeed": rng.uniform(0.05, 0.5, len(dates)),
                "cnt": count,
                "casual": count // 4,
                "registered": count - count // 4,
                "instant": np.arange(1, len(dates) + 1),
            }
        )
    )


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
