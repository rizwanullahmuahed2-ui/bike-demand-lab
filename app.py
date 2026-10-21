"""Read saved benchmark evidence; no training or model loading on page visits."""

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).parent
st.set_page_config(page_title="Bike Demand Lab", page_icon="🚲", layout="wide")
st.markdown(
    """<style>
.block-container{max-width:1250px;padding-top:2.5rem}
h1{letter-spacing:-.045em} [data-testid="stMetric"]{border:1px solid #dce6df;
border-radius:12px;padding:18px;background:#f7faf8}
</style>""",
    unsafe_allow_html=True,
)
st.caption("BIKE DEMAND LAB / HISTORICAL MODEL EVALUATION")
st.title("When does a city need more bikes?")
st.write("An hourly demand benchmark using Washington, D.C. rental records from 2011–2012.")

with st.sidebar:
    st.header("Evidence browser")
    mode = st.radio("Run", ["UCI benchmark", "Synthetic demo"])
    st.caption(
        "The dashboard reads saved artifacts. Training runs separately from the command line."
    )
    st.markdown("[Source dataset ↗](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset)")
    st.info(
        "Observed weather is an input. These results measure retrospective prediction, not future forecast accuracy."
    )

directory = ROOT / "artifacts" / "demo" if mode == "Synthetic demo" else ROOT / "artifacts"
required = [
    "benchmark.json",
    "predictions.csv",
    "error_by_hour.csv",
    "error_by_season.csv",
    "feature_importance.csv",
]
missing = [name for name in required if not (directory / name).is_file()]
if missing:
    st.info("No complete run is available for this view.")
    st.code(
        "bike-demand demo" if mode == "Synthetic demo" else "bike-demand fetch\nbike-demand train"
    )
    st.stop()
try:
    report = json.loads((directory / "benchmark.json").read_text())
    predictions = pd.read_csv(directory / "predictions.csv", parse_dates=["timestamp"])
    hourly = pd.read_csv(directory / "error_by_hour.csv")
    seasonal = pd.read_csv(directory / "error_by_season.csv")
    importance = pd.read_csv(directory / "feature_importance.csv")
except (ValueError, OSError, KeyError) as exc:
    st.error(f"Saved artifacts could not be read: {exc}")
    st.stop()

if mode == "Synthetic demo":
    st.warning("Synthetic fixture: these scores are only a pipeline smoke check.")
selected = next(
    item for item in report["test_results"] if item["family"] == report["selected_family"]
)
st.caption(
    f"Selected by validation MAE: {selected['label']} · test window: "
    f"{report['splits']['test']['start'][:10]} → {report['splits']['test']['end'][:10]}"
)
for column, label, value in zip(
    st.columns(4),
    ["Test MAE · rentals", "Test RMSE · rentals", "Test R²", "Held-out hours"],
    [
        f"{selected['mae']:.2f}",
        f"{selected['rmse']:.2f}",
        f"{selected['r2']:.3f}",
        f"{len(predictions):,}",
    ],
):
    column.metric(label, value)

overview, diagnostics, protocol = st.tabs(
    ["Predictions", "Where errors grow", "Evaluation & provenance"]
)
with overview:
    st.subheader("Observed rentals and model estimates")
    dates = sorted(predictions.timestamp.dt.strftime("%Y-%m-%d").unique())
    start = st.select_slider("Start date", options=dates, value=dates[0])
    days = st.slider("Days to display", 1, 28, 7)
    left = pd.Timestamp(start)
    window = predictions.loc[
        (predictions.timestamp >= left) & (predictions.timestamp < left + pd.Timedelta(days=days))
    ]
    figure = px.line(
        window,
        x="timestamp",
        y=["actual", "predicted"],
        color_discrete_sequence=["#173e35", "#bc702a"],
        labels={
            "value": "Rentals per hour",
            "timestamp": "Recorded local hour",
            "variable": "Series",
        },
    )
    st.plotly_chart(figure, use_container_width=True)
    st.subheader("Frozen model comparison")
    st.dataframe(
        pd.DataFrame(report["test_results"])[["label", "mae", "rmse", "r2"]],
        hide_index=True,
        use_container_width=True,
    )
    st.caption(
        "Every family uses its validation winner. The selected family was chosen before looking at test scores."
    )
    st.download_button(
        "Download held-out predictions",
        (directory / "predictions.csv").read_bytes(),
        "predictions.csv",
        "text/csv",
    )
with diagnostics:
    st.subheader("Average absolute error by hour")
    st.plotly_chart(
        px.bar(
            hourly,
            x="hr",
            y="mae",
            color_discrete_sequence=["#397d6c"],
            labels={"hr": "Hour", "mae": "MAE · rentals"},
        ),
        use_container_width=True,
    )
    st.dataframe(seasonal, hide_index=True, use_container_width=True)
    st.subheader("Which recorded inputs does the model rely on?")
    st.plotly_chart(
        px.bar(
            importance.sort_values("mae_increase"),
            x="mae_increase",
            y="feature",
            orientation="h",
            error_x="std",
            color_discrete_sequence=["#bc702a"],
        ),
        use_container_width=True,
    )
    st.caption(
        "Permutation importance on held-out rows: higher values mean a larger error increase when shuffled. Correlated inputs share information; this is not causation or a confidence interval."
    )
with protocol:
    st.subheader("Time moves forward in every split")
    st.dataframe(pd.DataFrame(report["splits"]).T, use_container_width=True)
    st.write(
        "The fixed model grid is ranked on validation MAE. The chosen configurations are refitted on train + validation, then evaluated on the final dates. Preprocessing is fitted inside each pipeline."
    )
    st.warning(
        "`casual` + `registered` equals the target, `cnt`. Both columns and the record ID are excluded from features."
    )
    st.write(
        f"{report['data_quality']['missing_calendar_hours']} calendar hours are absent in the source. They remain absent; missing hours are not treated as zero demand."
    )
    st.code(report["source"]["sha256"], language=None)
    st.json(
        {"source": report["source"], "versions": report["versions"], "protocol": report["protocol"]}
    )
    st.download_button(
        "Download benchmark manifest",
        (directory / "benchmark.json").read_bytes(),
        "benchmark.json",
        "application/json",
    )
    st.markdown((directory / "MODEL_CARD.md").read_text())
