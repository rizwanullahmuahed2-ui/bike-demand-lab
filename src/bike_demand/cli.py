"""Command-line entry points; training never starts from the dashboard."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from bike_demand.data import (
    ARCHIVE_SHA256,
    ARCHIVE_URL,
    DATASET_URL,
    download,
    load_frame,
    sha256,
    synthetic_frame,
    temporal_split,
)
from bike_demand.evaluation import run_benchmark


def main() -> None:
    parser = argparse.ArgumentParser(description="Bike Demand Lab reproducible benchmark")
    commands = parser.add_subparsers(dest="command", required=True)
    fetch = commands.add_parser(
        "fetch", help="Download and fingerprint-check official UCI hourly data"
    )
    fetch.add_argument("--directory", type=Path, default=Path("data/raw"))
    train = commands.add_parser(
        "train", help="Run full UCI benchmark with fixed chronological splits"
    )
    train.add_argument("--data", type=Path, default=Path("data/raw/hour.csv"))
    train.add_argument("--output", type=Path, default=Path("artifacts"))
    demo = commands.add_parser("demo", help="Offline synthetic smoke run; not UCI performance")
    demo.add_argument("--output", type=Path, default=Path("artifacts/demo"))
    args = parser.parse_args()
    if args.command == "fetch":
        print(download(args.directory))
        return
    if args.command == "demo":
        frame = synthetic_frame()
        partitions = temporal_split(frame, "2011-03-02", "2011-03-22")
        provenance = {
            "kind": "SYNTHETIC DEMO — not UCI data",
            "seed": 42,
            "sha256": hashlib.sha256(frame.to_csv(index=False).encode()).hexdigest(),
        }
    else:
        if not args.data.exists():
            parser.error(
                "Data file missing. Run `bike-demand fetch` or use `bike-demand demo` offline."
            )
        frame = load_frame(args.data)
        partitions = temporal_split(frame)
        provenance = {
            "kind": "UCI Bike Sharing · official hour.csv",
            "sha256": sha256(args.data),
            "archive_sha256": ARCHIVE_SHA256,
            "download_url": ARCHIVE_URL,
            "dataset_url": DATASET_URL,
            "license": "CC BY 4.0",
            "doi": "10.24432/C5W894",
        }
    result = run_benchmark(partitions, args.output, provenance, fast=args.command == "demo")
    print(f"\nSelected on validation: {result['selected_family']}")
    for item in result["test_results"]:
        print(
            f"Test: {item['label']:32s} MAE={item['mae']:.3f} "
            f"RMSE={item['rmse']:.3f} R²={item['r2']:.3f}"
        )
    print(f"\nArtifacts: {args.output.resolve()}")


if __name__ == "__main__":
    main()
