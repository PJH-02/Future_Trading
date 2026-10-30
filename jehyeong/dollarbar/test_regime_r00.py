"""pytest tests for regime_r00_validate (R00). Run: python -m pytest test_regime_r00.py -q"""
import pandas as pd
import pytest

from regime_r00_validate import validate_ohlcv


def _session(day, stamps, price=100.0):
    rows = []
    for s in stamps:
        ts = pd.Timestamp(f"{day} {s}")
        rows.append({"timestamp": ts, "date": day, "open": price, "high": price + 1,
                     "low": price - 1, "close": price, "volume": 10,
                     "symbol": "T", "source": "TOSS"})
    return pd.DataFrame(rows)


def _full_day(day):
    stamps = pd.date_range(f"2000-01-01 09:01", f"2000-01-01 15:30", freq="1min").strftime("%H:%M")
    return _session(day, list(stamps))


def test_complete_day_passes():
    rep = validate_ohlcv(_full_day("2024-03-04"), "KOSPI", "KOSPI")
    assert rep["status"] == "DATA_PASS"
    assert rep["session_missing_minutes"] == 0
    assert rep["dev_days"] == 1 and rep["oos_days"] == 0


def test_missing_0901_counted_not_filled():
    stamps = [s for s in pd.date_range("2000-01-01 09:01", "2000-01-01 15:30", freq="1min").strftime("%H:%M") if s != "09:01"]
    rep = validate_ohlcv(_session("2024-03-04", stamps), "KOSPI", "KOSPI")
    assert rep["status"] == "DATA_PASS"
    assert rep["days_missing_0901_stamp"] == 1
    assert rep["session_missing_minutes"] == 1


def test_ohlc_violation_blocks():
    df = _full_day("2024-03-04")
    df.loc[0, "high"] = 0.5  # high below close/open
    rep = validate_ohlcv(df, "KOSPI", "KOSPI")
    assert rep["status"] == "DATA_BLOCKED"
    assert rep["ohlc_violations"] >= 1


def test_duplicate_timestamp_blocks():
    df = _full_day("2024-03-04")
    rep = validate_ohlcv(pd.concat([df, df.iloc[[0]]]), "KOSPI", "KOSPI")
    assert rep["status"] == "DATA_BLOCKED"
    assert rep["duplicate_timestamps"] == 1


def test_negative_volume_blocks():
    df = _full_day("2024-03-04")
    df.loc[5, "volume"] = -1
    assert validate_ohlcv(df, "KOSPI", "KOSPI")["status"] == "DATA_BLOCKED"


def test_dev_oos_split_counts():
    df = pd.concat([_full_day("2025-12-31"), _full_day("2026-01-02")])
    rep = validate_ohlcv(df, "KOSPI", "KOSPI")
    assert rep["dev_days"] == 1
    assert rep["oos_days"] == 1
    assert rep["oos_last_bar_stamp"].startswith("2026-01-02 15:30")


def test_missing_columns_block():
    rep = validate_ohlcv(_full_day("2024-03-04").drop(columns=["volume"]), "KOSPI", "KOSPI")
    assert rep["status"] == "DATA_BLOCKED"


def test_unsupported_timestamp_rule_blocks():
    rep = validate_ohlcv(_full_day("2024-03-04"), "KOSPI", "KOSPI", timestamp_rule="bar_start")
    assert rep["status"] == "DATA_BLOCKED"
