"""R01 sampling comparison for KOSPI (spec v2, section 3).

Compares 30-minute time bars with causal index-volume bars. Index dollar bars are
not built because the index file has no KRW turnover field (spec 1.1).

Causality: the volume threshold for day d is median(daily volume of the previous
20 valid days) / N_target, where valid days come only from DEV data before d.
Bars reset every day; the last partial bar at session end is kept and flagged.
Only regular-session stamps 09:01..15:30 (bar_end) are used.

Run: python regime_r01_sampling.py  -> results/regime_research/r01_sampling.json
Reads input data from ../intraday/data/minute.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from regime_r00_validate import DEV_END, MINUTE, OUT, load_minute

N_TARGETS = (30, 50, 100)
LOOKBACK = 20
TIME_BAR_MIN = 30


def _regular(x: pd.DataFrame) -> pd.DataFrame:
    t = x["timestamp"].dt.strftime("%H:%M")
    x = x[t.between("09:01", "15:30")].copy()
    x["clock_minute"] = x["timestamp"].dt.hour * 60 + x["timestamp"].dt.minute
    x["t"] = x["timestamp"].dt.strftime("%H:%M")
    # Match R00: do not use days with a missing block >= 30 minutes.
    counts = x.groupby("date")["timestamp"].nunique()
    excluded = set(counts.index[(390 - counts) >= 30])
    x = x[~x["date"].isin(excluded)]
    return x.sort_values("timestamp").reset_index(drop=True)


def daily_volume_thresholds(x: pd.DataFrame, n_target: int) -> dict:
    """Causal per-day volume threshold; days without 20 prior valid days get None."""
    daily = x.groupby("date")["volume"].sum()
    dates = list(daily.index)
    thr = {}
    for i, d in enumerate(dates):
        prev = daily.iloc[max(0, i - LOOKBACK): i]
        thr[d] = float(prev.median() / n_target) if len(prev) == LOOKBACK else None
    return thr


def volume_bars(x: pd.DataFrame, n_target: int) -> pd.DataFrame:
    thr = daily_volume_thresholds(x, n_target)
    out = []
    for d, g in x.groupby("date", sort=True):
        T = thr[d]
        if T is None or T <= 0:
            continue
        cum, start = 0.0, 0
        ends = []
        for j, v in enumerate(g["volume"].to_numpy()):
            cum += v
            if cum >= T:
                ends.append((j, cum / T))
                cum, start = 0.0, j + 1
        if start < len(g):  # keep partial last bar, flagged
            ends.append((len(g) - 1, None))
        prev_idx = -1
        for j, ovr in ends:
            seg = g.iloc[prev_idx + 1: j + 1]
            out.append({"date": d, "bar_end": seg["timestamp"].iloc[-1], "close": seg["close"].iloc[-1],
                        "minutes": len(seg), "volume": seg["volume"].sum(),
                        "overshoot": (ovr - 1.0) if ovr is not None else None,
                        "partial": ovr is None})
            prev_idx = j
    return pd.DataFrame(out)


def time_bars(x: pd.DataFrame, minutes: int = TIME_BAR_MIN) -> pd.DataFrame:
    out = []
    for d, g in x.groupby("date", sort=True):
        # Fixed bar_end clock buckets: 09:01..09:30, 09:31..10:00, ...
        g = g.copy()
        g["bucket"] = (g["clock_minute"] - (9 * 60 + 1)) // minutes
        for _, seg in g.groupby("bucket", sort=True):
            expected = min(minutes, 390 - int(seg["bucket"].iloc[0]) * minutes)
            complete = len(seg) == expected
            out.append({"date": d, "bar_end": seg["timestamp"].iloc[-1], "close": seg["close"].iloc[-1],
                        "minutes": len(seg), "volume": seg["volume"].sum(),
                        "overshoot": None, "partial": not complete})
    return pd.DataFrame(out)


def add_returns(bars: pd.DataFrame) -> pd.DataFrame:
    """Within-day log returns between consecutive bars; no overnight gap."""
    b = bars.copy()
    b["ret"] = b.groupby("date")["close"].transform(lambda s: np.log(s).diff())
    return b


def diagnostics(bars: pd.DataFrame) -> dict:
    b = add_returns(bars).dropna(subset=["ret"])
    b = b[~b["partial"]]
    r = b["ret"].to_numpy()
    if len(r) < 10:
        return {"n_returns": int(len(r)), "status": "INSUFFICIENT"}
    r2 = r ** 2
    acf = lambda v: float(pd.Series(v).autocorr(1))
    per_day = bars.groupby("date").size()
    minutes = bars.loc[~bars["partial"], "minutes"]
    ov = bars["overshoot"].dropna()
    return {
        "n_bars": int(len(bars)),
        "bars_per_day_median": float(per_day.median()),
        "bars_per_day_p05": float(per_day.quantile(.05)),
        "bars_per_day_p95": float(per_day.quantile(.95)),
        "bar_minutes_median": float(minutes.median()),
        "bar_minutes_p95": float(minutes.quantile(.95)),
        "overshoot_p95": float(ov.quantile(.95)) if len(ov) else None,
        "acf1_ret": acf(r),
        "acf1_r2": acf(r2),
        "kurtosis_ret": float(pd.Series(r).kurt()),
        "n_returns": int(len(r)),
    }


def run(path: Path = MINUTE / "KOSPI_TOSS_1m.csv") -> dict:
    x = _regular(load_minute(path))
    x = x[(x["date"] >= "2023-09-21") & (x["date"] <= str(DEV_END.date()))]  # DEV only; 2026 never touched here
    result = {"sample_scope": "KOSPI DEV <= 2025-12-31", "time_bar_minutes": TIME_BAR_MIN, "variants": {}}
    tb = time_bars(x)
    base = diagnostics(tb)
    result["variants"]["time_30m"] = base
    for n in N_TARGETS:
        vb = volume_bars(x, n)
        d = diagnostics(vb)
        if "acf1_r2" in d and base.get("acf1_r2") is not None:
            d["acf1_r2_minus_time"] = d["acf1_r2"] - base["acf1_r2"]
        result["variants"][f"index_volume_N{n}"] = d
    return result


def main() -> None:
    res = run()
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "r01_sampling.json", "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    for k, v in res["variants"].items():
        print(k, {kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in v.items() if kk in
                  ("n_bars", "bars_per_day_median", "bar_minutes_median", "overshoot_p95", "acf1_ret", "acf1_r2")})


if __name__ == "__main__":
    main()
