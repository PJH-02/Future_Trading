import numpy as np
import pandas as pd

from g00_data_check import check_frame, GRID


def _day(date, labels, price=100.0):
    ts = pd.to_datetime([f"{date} {h}" for h in labels])
    n = len(ts)
    return pd.DataFrame({
        "timestamp": ts,
        "open": np.full(n, price), "high": np.full(n, price),
        "low": np.full(n, price), "close": np.full(n, price),
        "volume": np.full(n, 10.0),
    })


def _labels():
    return [h + ":00" for h in GRID]


def test_duplicate_and_ohlc_detected():
    x = pd.concat([_day("2024-01-02", _labels()), _day("2024-01-02", ["09:01:00"])], ignore_index=True)
    x.loc[0, "high"] = 0.5  # high below open/close
    rep = check_frame(x, "069500")
    assert rep["duplicate_keys"] == 1
    assert rep["ohlc_violations"] >= 1


def test_clean_day_has_no_gaps_and_exec_bar():
    rep = check_frame(_day("2024-01-02", _labels()), "069500")
    assert rep["intraday_gaps"] == 0
    assert rep["days_not_390_count"] == 0
    assert rep["exec_bar_label"] == "15:19"
    assert rep["exec_bar_days"] == 1


def test_bar_end_uses_end_label_for_execution():
    labels = [f"{(pd.Timestamp('2000-01-01 ' + h) + pd.Timedelta(minutes=1)).strftime('%H:%M')}:00" for h in _labels()]
    rep = check_frame(_day("2024-01-02", labels), "KOSPI")
    assert rep["exec_bar_label"] == "15:20"
    assert rep["exec_bar_days"] == 1
    assert rep["first_label_counts"] == {"09:02": 1}
    assert rep["last_label_counts"] == {"15:30": 1}


def test_short_day_is_listed_with_missing_labels():
    labels = [t for t in _labels() if t != "10:00:00"]
    rep = check_frame(_day("2024-01-02", labels), "069500")
    assert rep["days_not_390_count"] == 1
    assert rep["days_not_390"][0]["missing_labels"] == ["10:00"]
    assert rep["intraday_gaps"] >= 1


def test_jump_detected():
    x = _day("2024-01-02", _labels())
    x.loc[100, "close"] = 200.0  # +69% log jump
    rep = check_frame(x, "069500")
    assert rep["jump_minutes"] >= 1
    assert "2024-01-02" in rep["jump_dates"]
