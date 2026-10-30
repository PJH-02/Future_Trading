"""R06 diagnostic online-state stream; no candidate is approved for use.

Joins R02 causal path candidates and R03 scores as-of each minute. Because no
Path/Shift model has passed R07 selection, the canonical runtime state deliberately
falls back to UNCERTAIN. Candidate outputs are explicitly marked diagnostic-only.
No future labels, 2026 rows, or existing/legacy selection artifacts are loaded.

Run after R02/R03:
  python3 regime_r02_path.py --market KOSPI
  python3 regime_r03_distribution.py --market KOSPI
  python3 regime_r06_detector.py --market KOSPI
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from regime_r00_validate import DEV_END, MINUTE, OUT, load_minute
from regime_r02_path import build_path_features, valid_dev_minutes

FUTURE_PREFIXES = ("future_", "continuation_", "reversion_", "target_")
FORBIDDEN_INPUT_COLUMNS = {
    "future_return", "future_rv", "future_er", "reversion_return",
    "target_start_ts", "target_end_ts",
}
DISTANCE_METRICS = ("w1_norm", "ad2_stat", "ks_stat", "js_divergence", "vol_ratio")
NULL_PERCENTILES = ("w1_norm_null_pct", "ad2_stat_null_pct", "ks_stat_null_pct", "js_divergence_null_pct")


def _load_r03_scores(market: str, result_dir: Path = OUT) -> pd.DataFrame:
    path = result_dir / f"r03_distribution_scores_dev_{market.lower()}.csv"
    if not path.exists():
        raise FileNotFoundError(f"R03 scores missing; generate them first: {path}")
    x = pd.read_csv(path, parse_dates=["timestamp"])
    for col in ("recent_start", "baseline_start", "baseline_end"):
        if col in x:
            x[col] = pd.to_datetime(x[col])
    x = x[x["timestamp"] <= DEV_END + pd.Timedelta(days=1)].copy()
    # The two recent-window variants share timestamps; preserve them in separate columns.
    value_cols = [*DISTANCE_METRICS, *NULL_PERCENTILES]
    present = [c for c in value_cols if c in x.columns]
    wide = x.pivot_table(index=["timestamp", "date"], columns="recent_n", values=present, aggfunc="first")
    wide.columns = [f"{name}_{int(window)}" for name, window in wide.columns]
    wide = wide.reset_index().sort_values(["timestamp", "date"]).reset_index(drop=True)
    # R03 emits scored rows only after its initial 60 calibration sessions;
    # do not discard another 60 scored days here.
    wide["shift_available_at"] = wide["timestamp"]
    return wide


def build_diagnostic_stream(path_features: pd.DataFrame, r03_scores: pd.DataFrame) -> pd.DataFrame:
    """As-of merge without future labels; canonical status remains conservative."""
    forbidden = [c for c in path_features.columns if c in FORBIDDEN_INPUT_COLUMNS or c.startswith(FUTURE_PREFIXES)]
    if forbidden:
        raise ValueError(f"future/evaluation columns forbidden in detector input: {forbidden[:8]}")
    x = path_features.copy()
    x["timestamp"] = pd.to_datetime(x["timestamp"])
    x["date"] = x["timestamp"].dt.strftime("%Y-%m-%d")
    x = x.sort_values(["timestamp", "date"]).reset_index(drop=True)
    d = r03_scores.copy()
    d["timestamp"] = pd.to_datetime(d["timestamp"])
    d["date"] = d["timestamp"].dt.strftime("%Y-%m-%d")
    d = d.sort_values(["timestamp", "date"]).reset_index(drop=True)
    if "shift_available_at" not in d:
        d["shift_available_at"] = d["timestamp"]
    x = pd.merge_asof(
        x,
        d,
        on="timestamp",
        by="date",
        direction="backward",
        tolerance=pd.Timedelta(minutes=5),
        suffixes=("", "_shift"),
    )
    x["available_at"] = x["timestamp"]  # source bar is confirmed at bar_end
    x["reference_age_min"] = (
        (x["timestamp"] - x["baseline_end_60"]).dt.total_seconds() / 60
        if "baseline_end_60" in x else np.nan
    )
    x["path_candidate_p1"] = x["state_p1"].fillna("UNCERTAIN")
    x["path_candidate_p2"] = x["state_p2"].fillna("UNCERTAIN")
    x["path_candidate_p5"] = x["state_p5"].fillna("UNCERTAIN")
    x["path_candidate_direction"] = x["trend_direction_p1"].fillna("NONE")
    x["shift_candidate_alarm"] = False
    for col in NULL_PERCENTILES:
        for recent in (30, 60):
            candidate = f"{col}_{recent}"
            if candidate in x:
                x["shift_candidate_alarm"] |= x[candidate].ge(.95).fillna(False)
    x["shift_candidate_alarm"] = x["shift_candidate_alarm"].astype(bool)
    # Preserve raw distance evidence, but do not call uncalibrated candidates a finalized shift state.
    pct_cols = [c for c in x.columns if c.startswith(NULL_PERCENTILES)]
    pct_cols += [f"{metric}_null_pct_{n}" for metric in DISTANCE_METRICS for n in (30, 60)
                 if f"{metric}_null_pct_{n}" in x.columns]
    x["shift_score"] = x[pct_cols].max(axis=1) if pct_cols else np.nan
    x["distribution_shift"] = "UNCERTAIN"
    x["change_alarm"] = "DISABLED"
    x["vol_state"] = "MID"  # placeholder label is not a forecast; overridden below until R05 model selected
    x["vol_state"] = "UNCERTAIN"
    x["path_state"] = "UNCERTAIN"
    x["trend_direction"] = "NONE"
    x["path_confidence"] = 0.0
    x["model_version"] = "diagnostic-unselected-r06"
    x["detector_status"] = "DIAGNOSTIC_ONLY_R07_SELECTION_REQUIRED"
    if "shift_available_at" not in x:
        x["shift_available_at"] = pd.NaT
    # Avoid implying shift scores are current when their timestamp is absent or stale.
    has_shift = x.get("w1_norm_30").notna() if "w1_norm_30" in x else pd.Series(False, index=x.index)
    x.loc[~has_shift, ["shift_score", "shift_candidate_alarm"]] = [np.nan, False]
    return x


def runtime_view(stream: pd.DataFrame) -> pd.DataFrame:
    """Return the runtime contract plus diagnostic candidate fields, no future targets."""
    cols = [
        "timestamp", "available_at", "shift_available_at", "date", "symbol", "path_state", "trend_direction",
        "path_confidence", "vol_state", "distribution_shift", "shift_score", "change_alarm",
        "reference_age_min", "model_version", "detector_status", "path_candidate_p1",
        "path_candidate_p2", "path_candidate_p5", "path_candidate_direction", "shift_candidate_alarm",
    ]
    cols += [c for c in stream.columns if c.startswith(tuple(f"{m}_" for m in DISTANCE_METRICS))]
    cols += [c for c in stream.columns if c.startswith(tuple(f"{m}_" for m in NULL_PERCENTILES))]
    cols = list(dict.fromkeys(c for c in cols if c in stream.columns))
    out = stream[cols].copy()
    forbidden = [c for c in out.columns if c in FORBIDDEN_INPUT_COLUMNS or c.startswith(FUTURE_PREFIXES)]
    if forbidden:
        raise AssertionError(f"runtime view leaked evaluation labels: {forbidden}")
    return out


def run(market: str = "KOSPI") -> tuple[pd.DataFrame, dict]:
    if market not in {"KOSPI", "KOSDAQ"}:
        raise ValueError(f"unsupported market: {market}")
    minute_path = MINUTE / f"{market}_TOSS_1m.csv"
    minutes = valid_dev_minutes(minute_path)
    path_features = build_path_features(minutes)
    r03 = _load_r03_scores(market)
    stream = build_diagnostic_stream(path_features, r03)
    result = runtime_view(stream)
    report = {
        "stage": "R06_ONLINE_DETECTOR_DIAGNOSTIC",
        "status": "DIAGNOSTIC_ONLY_NO_APPROVED_MODEL",
        "market": market,
        "data_period": {"start": str(minutes["date"].min()), "end": str(minutes["date"].max())},
        "rows": int(len(result)),
        "available_at_monotonic": bool(result["available_at"].is_monotonic_increasing),
        "path_candidate_coverage": {c: {str(k): int(v) for k, v in result[c].value_counts(dropna=False).items()}
                                     for c in ("path_candidate_p1", "path_candidate_p2", "path_candidate_p5")},
        "shift_candidate_alarm_rows": int(result["shift_candidate_alarm"].sum()),
        "canonical_path_state": "UNCERTAIN_PENDING_R07_SELECTION",
        "canonical_distribution_shift": "UNCERTAIN_PENDING_BLOCK_NULL_DAY_CALIBRATION",
        "future_label_columns_in_runtime": [],
        "true_oos_accessed": False,
        "caveats": [
            "P1/P2/P5 are preliminary explanatory rules; none is a selected or calibrated model.",
            "The p95 diagnostic alarm is a candidate flag from R03 block-null percentiles, not an approved FPR/day operating threshold.",
            "The score updates only at the R03 five-minute grid after the initial 60 DEV calibration sessions; the as-of join tolerates at most five minutes and never looks forward.",
            "No canonical TREND, MEAN_REVERTING, volatility state, or distribution-shift state is asserted until R07 selects and validates it.",
        ],
        "decision": "NO_MODEL_PROMOTED; R07_REQUIRED",
    }
    return result, report


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate R06 DEV diagnostic detector stream")
    parser.add_argument("--market", choices=("KOSPI", "KOSDAQ"), default="KOSPI")
    args = parser.parse_args()
    result, report = run(args.market)
    OUT.mkdir(parents=True, exist_ok=True)
    suffix = args.market.lower()
    result.to_csv(OUT / f"r06_detector_diagnostic_dev_{suffix}.csv", index=False)
    import json
    with open(OUT / f"r06_detector_report_dev_{suffix}.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps({"status": report["status"], "rows": report["rows"], "shift_candidate_alarm_rows": report["shift_candidate_alarm_rows"], "decision": report["decision"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
