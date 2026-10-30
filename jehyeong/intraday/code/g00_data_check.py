"""G00 data and time validation for the KOSPI event-bar research.

Timestamp conventions (verified empirically, see jehyeong/probability):
- KOSPI_TOSS_1m.csv: bar_end. Label T covers [T-1min, T).
- 069500 / 114800 _TOSS_1m.csv: bar_start. Label T covers [T, T+1min).
No shifting is performed. Raw files are never modified.
Missing session bars are forward-filled (previous close, volume 0, filled=1)
into separate *_filled_390.csv files.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data" / "minute"
OUT = HERE.parents[1] / "probability" / "results" / "g00"

CONVENTION = {"KOSPI": "bar_end", "069500": "bar_start", "114800": "bar_start"}
FILES = {"KOSPI": "KOSPI_TOSS_1m.csv", "069500": "069500_TOSS_1m.csv", "114800": "114800_TOSS_1m.csv"}
RANGE_START, RANGE_END = "2023-09-21", "2026-10-07"
SESSION_FIRST, SESSION_LAST = "09:01", "15:30"
EXPECTED_BARS = 390
JUMP_LOG_RET = 0.15
FIELDS = ["open", "high", "low", "close", "volume"]
GRID = [d.strftime("%H:%M") for d in pd.date_range("2000-01-01 09:01", "2000-01-01 15:30", freq="min")]


def load(path: Path) -> pd.DataFrame:
    x = pd.read_csv(path, encoding="utf-8-sig")
    x["timestamp"] = pd.to_datetime(x["timestamp"])
    x = x.sort_values("timestamp").reset_index(drop=True)
    sym = path.name.split("_")[0]
    if CONVENTION[sym] == "bar_end":
        # Normalize each bar to its actual interval start; 09:02 end-label
        # represents the same 09:01 interval as an ETF 09:01 start-label.
        x["timestamp"] -= pd.Timedelta(minutes=1)
    x["date"] = x.timestamp.dt.strftime("%Y-%m-%d")
    x["hm"] = x.timestamp.dt.strftime("%H:%M")
    return x[(x.date >= RANGE_START) & (x.date <= RANGE_END)].reset_index(drop=True)


def check_frame(x: pd.DataFrame, sym: str) -> dict:
    if "date" not in x:
        x = x.assign(date=x.timestamp.dt.strftime("%Y-%m-%d"), hm=x.timestamp.dt.strftime("%H:%M"))
    rep = {"convention": CONVENTION[sym], "reported_clock": "normalized_bar_start", "rows_in_range": int(len(x)), "days_in_range": int(x.date.nunique())}
    rep["duplicate_keys"] = int(x.timestamp.duplicated().sum())
    rep["nonfinite_or_nonpositive"] = int(
        (~np.isfinite(x[FIELDS].to_numpy(dtype=float))).sum()
        + (x[["open", "high", "low", "close"]] <= 0).sum().sum()
    )
    rep["negative_volume"] = int((x.volume < 0).sum())
    rep["ohlc_violations"] = int(
        ((x.high < x[["open", "low", "close"]].max(axis=1)) | (x.low > x[["open", "close"]].min(axis=1))).sum()
    )
    sess = x[(x.hm >= SESSION_FIRST) & (x.hm <= SESSION_LAST)]
    rep["rows_outside_session"] = int(len(x) - len(sess))
    rep["outside_session_labels"] = sorted(set(x.loc[~x.index.isin(sess.index), "hm"]))
    rep["session_days"] = int(sess.date.nunique())
    short, firsts, lasts = [], [], []
    for d, g in sess.groupby("date"):
        firsts.append(g.hm.iloc[0])
        lasts.append(g.hm.iloc[-1])
        if len(g) != EXPECTED_BARS:
            missing = [h for h in GRID if h not in set(g.hm)]
            short.append({"date": d, "bars": int(len(g)), "missing_labels": missing})
    rep["first_label_counts"] = {k: firsts.count(k) for k in sorted(set(firsts))}
    rep["last_label_counts"] = {k: lasts.count(k) for k in sorted(set(lasts))}
    rep["days_not_390"] = short
    rep["days_not_390_count"] = len(short)
    rep["intraday_gaps"] = int(sum(
        int((g.timestamp.diff().dropna() != pd.Timedelta(minutes=1)).sum()) for _, g in sess.groupby("date")
    ))
    exec_label = "15:19"
    source_exec_label = "15:20" if CONVENTION[sym] == "bar_end" else "15:19"
    exec_days = int(x.loc[x.hm == exec_label, "date"].nunique())
    rep["exec_bar_label"] = exec_label
    rep["source_exec_bar_label"] = source_exec_label
    rep["exec_bar_days"] = exec_days
    rep["exec_bar_missing_days"] = int(x.date.nunique() - exec_days)
    expected_first = "09:01"
    expected_last = "15:30" if CONVENTION[sym] == "bar_start" else "15:30"
    rep["first_bar_missing_days"] = int(sum(v != expected_first for v in firsts))
    rep["last_bar_missing_days"] = int(sum(v != expected_last for v in lasts))
    lr = np.log(x.close).groupby(x.date).diff().abs()
    jump = x.loc[lr > JUMP_LOG_RET]
    rep["jump_minutes"] = int(len(jump))
    rep["jump_dates"] = sorted(set(jump.date))
    return rep


def fill_session(x: pd.DataFrame) -> pd.DataFrame:
    """Build a 390-bar grid per day. Missing labels take the previous bar's close (volume 0, filled=1).
    The very first bar of the range has no predecessor, so if it is missing it is seeded from the first observed open."""
    sess = x[(x.hm >= SESSION_FIRST) & (x.hm <= SESSION_LAST)]
    prev_close = float(sess.open.iloc[0])
    rows = []
    for d, g in sess.groupby("date", sort=True):
        g = g.drop_duplicates("hm").set_index("hm").reindex(GRID)
        for h in GRID:
            if pd.isna(g.at[h, "open"]):
                g.loc[h, ["open", "high", "low", "close"]] = prev_close
                g.loc[h, "volume"] = 0.0
                g.loc[h, "filled"] = 1
            else:
                g.loc[h, "filled"] = 0
            prev_close = float(g.at[h, "close"])
        g = g.reset_index()
        g["date"] = d
        g["timestamp"] = pd.to_datetime(d + " " + g.hm)
        rows.append(g)
    res = pd.concat(rows, ignore_index=True)
    res["filled"] = res["filled"].astype(int)
    return res[["timestamp", "date", "hm", *FIELDS, "filled"]]


def run() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    raw = {sym: load(DATA / name) for sym, name in FILES.items()}
    # Align each file to its empirically verified real-bar date before selecting
    # the exact shared sessions; start-labeled ETFs sometimes carry a final
    # after-hours row on a date with no equivalent KOSPI session.
    common = sorted(set.intersection(*(set(x.date) for x in raw.values())))
    report = {"common_range": {"start": common[0], "end": common[-1], "days": len(common)}}
    for sym in FILES:
        excluded = sorted(set(raw[sym].date) - set(common))
        x = raw[sym][raw[sym].date.isin(common)].reset_index(drop=True)
        report[sym] = check_frame(x, sym)
        report[sym]["excluded_dates_not_in_all_files"] = excluded
        f = fill_session(x)
        f.to_csv(OUT / f"{sym}_filled_390.csv", index=False, encoding="utf-8")
        report[sym]["filled_bar_count"] = int(f.filled.sum())
        report[sym]["filled_days"] = int(f.loc[f.filled == 1, "date"].nunique())
        report[sym]["filled_day_list"] = sorted(f.loc[f.filled == 1, "date"].unique().tolist())
    (OUT / "g00_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))
