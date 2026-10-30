"""R02 causal path features and descriptive tournament (spec v2, section 4).

Consumes only KOSPI DEV rows through 2025-12-31. No model is selected here:
future-anchor summaries are descriptive and R07 walk-forward decides adoption.
All features reset each session and never bridge missing minutes or overnight gaps.

Run: python3 regime_r02_path.py
Outputs live under jehyeong/dollarbar/results/regime_research/.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from regime_r00_validate import DEV_END, DEV_START, MINUTE, OUT, load_minute

HORIZONS = (5, 15, 30, 60)
ER_WINDOWS = (10, 30, 60)
VR_Q = (2, 5, 10)
VR_WINDOWS = (60, 120)


def valid_dev_minutes(path: Path = MINUTE / "KOSPI_TOSS_1m.csv") -> pd.DataFrame:
    """Load DEV only, discard post-session bars and days with >=30 missing minutes."""
    x = load_minute(path)
    x = x[(x["timestamp"] >= DEV_START) & (x["timestamp"] < DEV_END + pd.Timedelta(days=1))].copy()
    tm = x["timestamp"].dt.strftime("%H:%M")
    x = x[tm.between("09:01", "15:30")].copy()
    x["minute"] = x["timestamp"].dt.hour * 60 + x["timestamp"].dt.minute
    x = x.sort_values("timestamp").reset_index(drop=True)
    counts = x.groupby("date")["timestamp"].nunique()
    # Gaps are never filled; remove days whose broken session prevents reliable path inference.
    bad_dates = set(counts.index[(390 - counts) >= 30])
    x = x[~x["date"].isin(bad_dates)].copy().reset_index(drop=True)
    return x


def build_path_features(minutes: pd.DataFrame) -> pd.DataFrame:
    """Causal per-minute ER, MA, VR and lag-ACF features with session resets."""
    x = minutes.sort_values("timestamp").copy().reset_index(drop=True)
    if "date" not in x:
        x["date"] = x["timestamp"].dt.strftime("%Y-%m-%d")
    grp = x.groupby("date", sort=False, group_keys=False)
    # A return is valid only when observations are exactly one clock minute apart.
    prev_close = grp["close"].shift(1)
    prev_minute = grp["minute"].shift(1) if "minute" in x else grp["timestamp"].shift(1).dt.floor("min")
    if "minute" in x:
        continuous = x["minute"].sub(prev_minute).eq(1)
    else:
        continuous = x["timestamp"].sub(prev_minute).eq(pd.Timedelta(minutes=1))
    x["ret_1m"] = np.log(x["close"] / prev_close).where(continuous)
    # Start a new feature segment after every absent clock minute; no rolling window bridges it.
    x["segment_id"] = (x["date"].ne(x["date"].shift()) | ~continuous).cumsum()
    grp = x.groupby("segment_id", sort=False, group_keys=False)
    logc = np.log(x["close"])

    # Causal rolling windows; pandas rolling includes the current confirmed bar only.
    for k in ER_WINDOWS:
        net = (logc.groupby(x["segment_id"]).diff(k)).abs()
        path = x["ret_1m"].abs().groupby(x["segment_id"]).rolling(k, min_periods=k).sum().reset_index(level=0, drop=True)
        x[f"er_{k}"] = (net / path.replace(0, np.nan)).clip(0, 1)

    for span in (20, 60):
        x[f"ema{span}"] = grp["close"].transform(lambda s: s.ewm(span=span, adjust=False, min_periods=span).mean())
        x[f"ema{span}_log"] = np.log(x[f"ema{span}"])
        # Ten-observation slope is valid only when the intervening path is contiguous.
        x[f"ema{span}_slope10"] = x.groupby("segment_id")[f"ema{span}_log"].diff(10) / 10
    x["ma_spread"] = np.log(x["ema20"] / x["ema60"])
    x["ma_distance"] = np.log(x["close"] / x["ema20"])
    # Scale estimate uses prior observations only; current deviation cannot set its own denominator.
    x["ma_scale60"] = x.groupby("segment_id")["ma_distance"].transform(
        lambda s: s.abs().shift(1).rolling(60, min_periods=30).median()
    ) * 1.4826
    x["ma_z"] = x["ma_distance"] / x["ma_scale60"].replace(0, np.nan)

    # Rolling variance ratios. Both numerator and denominator use only within-session returns.
    for w in VR_WINDOWS:
        var1 = x.groupby("segment_id")["ret_1m"].transform(lambda s: s.rolling(w, min_periods=w).var(ddof=1))
        for q in VR_Q:
            qret = x.groupby("segment_id")["ret_1m"].transform(lambda s: s.rolling(q, min_periods=q).sum())
            varq = x.assign(_qret=qret).groupby("segment_id")["_qret"].transform(
                lambda s: s.rolling(w - q + 1, min_periods=w - q + 1).var(ddof=1)
            )
            x[f"vr_{w}_{q}"] = varq / (q * var1)

    # Rolling lag-1 return autocorrelation as a separate candidate, not a default state rule.
    for w in (60, 120):
        x[f"acf1_{w}"] = x.groupby("segment_id")["ret_1m"].transform(
            lambda s: s.rolling(w, min_periods=w).corr(s.shift(1))
        )

    # Deterministic candidate rules; low ER alone maps to UNCERTAIN, never MR.
    direction = np.sign(x["ema20_slope10"])
    ma_aligned = direction.eq(np.sign(x["ema60_slope10"])) & direction.ne(0)
    x["state_p0"] = "UNCERTAIN"
    # P1/P2 are explicitly diagnostic rules; P5 adds a deterministic future-free
    # three-way anchor for R05/R07 evaluation, but is not a fitted model.
    x["state_p1"] = np.where(ma_aligned, "TREND", "UNCERTAIN")
    x["trend_direction_p1"] = np.select(
        [x["state_p1"].eq("TREND") & direction.gt(0), x["state_p1"].eq("TREND") & direction.lt(0)],
        ["UP", "DOWN"], default="NONE"
    )
    er = x["er_30"]
    x["state_p2"] = np.where((er >= .55) & ma_aligned, "TREND", "UNCERTAIN")
    x["trend_direction_p2"] = np.where(
        x["state_p2"].eq("TREND"), np.where(direction.gt(0), "UP", "DOWN"), "NONE"
    )
    # P3/P4 are feature ablations only until R07 validates incremental OOS value.
    x["state_p3"] = x["state_p2"]
    x["state_p4"] = x["state_p2"]
    x["state_p5"] = np.where(
        x["state_p2"].eq("TREND"), "TREND",
        np.where((er <= .30) & x["ma_z"].abs().ge(1.5), "MEAN_REVERTING", "UNCERTAIN")
    )
    return x


def add_future_anchors(features: pd.DataFrame) -> pd.DataFrame:
    """Add evaluation-only future labels; never used by feature or state creation."""
    x = features.copy()
    for h in HORIZONS:
        future_close = x.groupby("segment_id")["close"].shift(-h)
        contiguous = x.groupby("segment_id")["minute"].shift(-h).sub(x["minute"]).eq(h)
        future_ret = np.log(future_close / x["close"]).where(contiguous)
        x[f"future_ret_{h}"] = future_ret
        # Use direct future close path for the signed continuation / reversion anchors.
        past_close = x.groupby("segment_id")["close"].shift(15)
        past_minute = x.groupby("segment_id")["minute"].shift(15)
        past_is_contiguous = x["minute"].sub(past_minute).eq(15)
        direction = np.sign(np.log(x["close"] / past_close)).where(past_is_contiguous)
        x[f"continuation_{h}"] = future_ret * direction
        x[f"reversion_{h}"] = -np.sign(x["ma_z"]) * future_ret
        # Sum squared future one-minute returns using a strict h-minute window.
        x[f"future_rv_{h}"] = x.groupby("segment_id")["ret_1m"].transform(
            lambda s: s.pow(2).shift(-1).iloc[::-1].rolling(h, min_periods=h).sum().iloc[::-1]
        ).where(contiguous)
        x[f"continuation_{h}"] = x[f"continuation_{h}"].where(contiguous)
        x[f"reversion_{h}"] = x[f"reversion_{h}"].where(contiguous)
    return x


def _anchor_summary(x: pd.DataFrame, state_col: str) -> dict:
    out = {"state_column": state_col, "states": {}}
    for state, g in x.groupby(state_col, dropna=False):
        s = {"n": int(len(g))}
        for h in HORIZONS:
            for key in (f"future_ret_{h}", f"future_rv_{h}", f"continuation_{h}", f"reversion_{h}"):
                vals = g[key].dropna()
                s[f"{key}_mean"] = float(vals.mean()) if len(vals) else None
                s[f"{key}_n"] = int(len(vals))
        out["states"][str(state)] = s
    return out


def run(market: str = "KOSPI", path: Path | None = None) -> tuple[pd.DataFrame, dict]:
    """Run the same schema-compatible R02 implementation for either index."""
    if market not in {"KOSPI", "KOSDAQ"}:
        raise ValueError(f"unsupported market: {market}")
    if path is None:
        path = MINUTE / f"{market}_TOSS_1m.csv"
    x = build_path_features(valid_dev_minutes(path))
    x = add_future_anchors(x)
    tournament = {
        "stage": "R02_DESCRIPTIVE_ONLY",
        "status": "FEATURES_AND_ANCHORS_COMPUTED; STATISTICAL_SELECTION_PENDING_R07",
        "market": market,
        "dev_start": str(DEV_START.date()),
        "dev_end": str(DEV_END.date()),
        "usable_days": int(x["date"].nunique()),
        "rows": int(len(x)),
        "candidates": {
            "P0_no_state": _anchor_summary(x, "state_p0"),
            "P1_MA_slope_alignment": _anchor_summary(x, "state_p1"),
            "P2_ER_MA": _anchor_summary(x, "state_p2"),
            "P3_ER_MA_VR": {"state_rule": "same as P2; VR features available, not yet selected", **_anchor_summary(x, "state_p3")},
            "P4_plus_lag_ACF": {"state_rule": "same as P2; lag-ACF features available, not yet selected", **_anchor_summary(x, "state_p4")},
            "P5_operational_anchor": {"state_rule": "P2 TREND; else ER30<=0.30 and |MA_z|>=1.5 is an evaluation candidate only", **_anchor_summary(x, "state_p5")},
        },
        "decision": "NO_SELECTION_R07_REQUIRED",
        "caveats": [
            "Future-anchor summaries are descriptive conditional outcomes, not independent OOS model scores.",
            "Low ER or VR below one never labels MEAN_REVERTING without validated price reversion.",
            "All windows reset each session and at every missing minute; no overnight returns or gap filling.",
            "Missing-minute sessions with at least 30 missing session minutes are excluded.",
            "No confidence intervals, purged walk-forward scores, or significance tests are included at this stage.",
            "Index volume bars remain a sampling candidate; this module uses time-minute OHLCV only."
        ],
    }
    return x, tournament


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Run causal R02 Path features for KOSPI/KOSDAQ.")
    parser.add_argument("--market", choices=("KOSPI", "KOSDAQ"), default="KOSPI")
    args = parser.parse_args()
    features, summary = run(args.market)
    OUT.mkdir(parents=True, exist_ok=True)
    # Persist features and evaluation-only anchors separately to make their roles explicit.
    feature_cols = [c for c in features.columns if not c.startswith(("future_", "continuation_", "reversion_"))]
    anchor_cols = ["timestamp", "date", "state_p0", "state_p1", "state_p2", "state_p3", "state_p4", "state_p5"] + [
        c for c in features.columns if c.startswith(("future_", "continuation_", "reversion_"))
    ]
    suffix = args.market.lower()
    features[feature_cols].to_csv(OUT / f"r02_path_features_dev_{suffix}.csv", index=False)
    features[anchor_cols].to_csv(OUT / f"r02_future_anchor_eval_dev_{suffix}.csv", index=False)
    with open(OUT / f"r02_path_tournament_{suffix}.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps({"usable_days": summary["usable_days"], "rows": summary["rows"], "decision": summary["decision"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
