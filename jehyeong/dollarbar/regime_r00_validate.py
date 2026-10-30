"""R00 data validation for the KRX regime research (spec v2, section 1).

Timestamp rule (confirmed on the data, TOSS): bar_end. A stamp t covers (t-1min, t].
Regular session stamps: 09:01 .. 15:30. Stamp 15:31 is reported separately as
post-session and is never used as a regular bar.

Run: python regime_r00_validate.py
Reads ../intraday/data/minute; writes results/regime_research/r00_data_validation.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parent
DATA_ROOT = PROJECT.parent / "intraday"
MINUTE = DATA_ROOT / "data" / "minute"
OUT = PROJECT / "results" / "regime_research"

DEV_START, DEV_END = pd.Timestamp("2023-09-21"), pd.Timestamp("2025-12-31")
OOS_START = pd.Timestamp("2026-01-01")
REG_FIRST, REG_LAST = "09:01", "15:30"
EXCLUDE_GAP_MINUTES = 30  # a day missing >= 30 session minutes is excluded from signal days
REQUIRED = ["timestamp", "date", "open", "high", "low", "close", "volume", "symbol", "source"]
SESSION_STAMPS = pd.date_range("2000-01-01 09:01", "2000-01-01 15:30", freq="1min").strftime("%H:%M").tolist()

# Market/instrument map. ETF multiple taken from the product name in the spec.
INSTRUMENTS = {
    "KOSPI": {"file": "KOSPI_TOSS_1m.csv", "kind": "index", "required": True},
    "KOSDAQ": {"file": "KOSDAQ_TOSS_1m.csv", "kind": "index", "required": True},
    "069500": {"file": "069500_TOSS_1m.csv", "kind": "etf", "market": "KOSPI", "leverage": 1},
    "122630": {"file": "122630_TOSS_1m.csv", "kind": "etf", "market": "KOSPI", "leverage": 2},
    "229200": {"file": "229200_TOSS_1m.csv", "kind": "etf", "market": "KOSDAQ", "leverage": 1},
    "251340": {"file": "251340_TOSS_1m.csv", "kind": "etf", "market": "KOSDAQ", "leverage": -1},
}


def load_minute(path: Path) -> pd.DataFrame:
    x = pd.read_csv(path, dtype={"date": str, "symbol": str}, encoding="utf-8-sig")
    x["timestamp"] = pd.to_datetime(x["timestamp"])
    return x


def validate_ohlcv(df: pd.DataFrame, market: str, instrument: str, timestamp_rule: str = "bar_end") -> dict:
    """Return a validation report with status DATA_PASS or DATA_BLOCKED and reasons.

    Hard failures block. Session gaps are counted and reported, never forward-filled.
    """
    reasons: list[str] = []
    missing_cols = [c for c in REQUIRED if c not in df.columns]
    if missing_cols:
        return {"market": market, "instrument": instrument, "status": "DATA_BLOCKED",
                "reasons": [f"missing columns {missing_cols}"]}

    if timestamp_rule != "bar_end":
        reasons.append(f"unsupported timestamp rule {timestamp_rule}")

    x = df.copy()
    x["t"] = x["timestamp"].dt.strftime("%H:%M")
    n = len(x)
    dup = int(x.duplicated("timestamp").sum())
    if dup:
        reasons.append(f"{dup} duplicate timestamps")
    if not x["timestamp"].is_monotonic_increasing:
        reasons.append("timestamps not sorted")

    ohlc = ["open", "high", "low", "close"]
    num = x[ohlc + ["volume"]].astype(float)
    nonfinite = int((~np.isfinite(num.to_numpy())).sum())
    if nonfinite:
        reasons.append(f"{nonfinite} nonfinite values")
    if (x[ohlc] <= 0).any().any():
        reasons.append("nonpositive prices")
    if (x["volume"] < 0).any():
        reasons.append("negative volume")
    bad_hi = x["high"] < x[["open", "close", "low"]].max(axis=1)
    bad_lo = x["low"] > x[["open", "close", "high"]].min(axis=1)
    ohlc_bad = int((bad_hi | bad_lo).sum())
    if ohlc_bad:
        reasons.append(f"{ohlc_bad} OHLC range violations")

    post = x[~x["t"].between(REG_FIRST, REG_LAST)]
    reg = x[x["t"].between(REG_FIRST, REG_LAST)].copy()
    reg_days = reg.groupby("date")
    days = sorted(reg["date"].unique())

    expected = {d: len(SESSION_STAMPS) for d in days}
    gap_minutes = 0
    missing_first_stamp = 0
    excluded_days = []  # days with a missing block >= EXCLUDE_GAP_MINUTES; never filled
    for d, g in reg_days:
        have = set(g["t"])
        miss = [s for s in SESSION_STAMPS if s not in have]
        gap_minutes += len(miss)
        if "09:01" in miss:
            missing_first_stamp += 1
        if len(miss) >= EXCLUDE_GAP_MINUTES:
            excluded_days.append({"date": d, "missing_minutes": len(miss), "first_missing": miss[0], "last_missing": miss[-1]})
    days_with_gap = int(sum(1 for d, g in reg_days if len(set(g["t"])) < expected[d]))

    dates = pd.to_datetime(pd.Series(days))
    dev_days = int((dates <= DEV_END).sum())
    oos_days = int((dates >= OOS_START).sum())
    last_bar = str(x["timestamp"].max())
    oos_last = str(x.loc[x["date"].str.startswith("2026"), "timestamp"].max()) if oos_days else None

    report = {
        "market": market,
        "instrument": instrument,
        "timestamp_rule": timestamp_rule,
        "rows": n,
        "days_regular": len(days),
        "first_date": days[0] if days else None,
        "last_date": days[-1] if days else None,
        "last_bar_stamp": last_bar,
        "dev_days": dev_days,
        "oos_days": oos_days,
        "oos_last_bar_stamp": oos_last,
        "duplicate_timestamps": dup,
        "ohlc_violations": ohlc_bad,
        "session_missing_minutes": int(gap_minutes),
        "days_with_session_gap": days_with_gap,
        "days_missing_0901_stamp": int(missing_first_stamp),
        "excluded_days": excluded_days,
        "usable_days": len(days) - len(excluded_days),
        "post_session_rows": int(len(post)),
        "post_session_stamps": post["t"].value_counts().sort_index().to_dict(),
    }
    report["status"] = "DATA_BLOCKED" if reasons else "DATA_PASS"
    report["reasons"] = reasons
    return report


def main() -> None:
    reports = []
    for key, spec in INSTRUMENTS.items():
        path = MINUTE / spec["file"]
        if not path.exists():
            reports.append({"market": spec.get("market", key), "instrument": key,
                            "status": "DATA_BLOCKED", "reasons": [f"missing file {spec['file']}"]})
            continue
        market = key if spec["kind"] == "index" else spec["market"]
        rep = validate_ohlcv(load_minute(path), market, key)
        if spec["kind"] == "etf":
            rep["leverage"] = spec["leverage"]
        reports.append(rep)
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "r00_data_validation.json", "w", encoding="utf-8") as f:
        json.dump(reports, f, ensure_ascii=False, indent=2)
    for r in reports:
        print(r["instrument"], r["status"], r.get("reasons", []))


if __name__ == "__main__":
    main()
