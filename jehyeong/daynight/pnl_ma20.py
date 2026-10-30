"""Realized P&L MA20 mode gate for the day/night shadow account."""

import numpy as np
import pandas as pd

COLUMNS = [
    "date",
    "reference_pnl",
    "ma20",
    "has_exit",
    "mode_before_exit",
    "mode_after_exit",
    "observations",
]


def _empty_result(calendar):
    return pd.DataFrame(
        {
            "date": calendar[:0],
            "reference_pnl": pd.Series(dtype=float),
            "ma20": pd.Series(dtype=float),
            "has_exit": pd.Series(dtype=bool),
            "mode_before_exit": pd.Series(dtype=int),
            "mode_after_exit": pd.Series(dtype=int),
            "observations": pd.Series(dtype=int),
        }
    )


def _session_exits(calendar, exits):
    if exits is None:
        exits = pd.DataFrame(columns=["exit_time", "reference_pnl"])
    if not isinstance(exits, pd.DataFrame):
        raise TypeError("exits must be a pandas DataFrame")

    if len(exits) == 0:
        return {}

    missing = {"exit_time", "reference_pnl"} - set(exits.columns)
    if missing:
        raise ValueError(f"exits missing columns: {sorted(missing)}")

    times = pd.DatetimeIndex(pd.to_datetime(exits["exit_time"]))
    pnls = pd.to_numeric(exits["reference_pnl"], errors="coerce").to_numpy(dtype=float)
    if not np.all(np.isfinite(pnls)):
        raise ValueError("reference_pnl must be finite")

    session_pos = {}
    for i, session in enumerate(calendar):
        session_pos[session.normalize()] = i

    result = {}
    for when, pnl in zip(times, pnls):
        pos = session_pos.get(when.normalize())
        if pos is None:
            raise ValueError(f"exit outside calendar: {when}")
        if pos in result:
            raise ValueError(f"duplicate exit for session: {when}")
        result[pos] = float(pnl)
    return result


def build_gate(calendar, exits, window=20):
    """Build the causal MA gate over ``calendar`` from realized exit P&L."""
    calendar = pd.DatetimeIndex(calendar)
    if calendar.hasnans:
        raise ValueError("calendar must not contain NaT")
    if not calendar.is_unique:
        raise ValueError("calendar must be unique")
    if not calendar.is_monotonic_increasing:
        raise ValueError("calendar must be ordered")
    if int(window) != window or window < 1:
        raise ValueError("window must be a positive integer")
    window = int(window)

    if len(calendar) == 0:
        return _empty_result(calendar)

    exit_by_pos = _session_exits(calendar, exits)

    history = []
    pnl = 0.0
    mode = 1
    ma20 = float("nan")

    rows = []
    for i, date in enumerate(calendar):
        mode_before = mode
        if i in exit_by_pos:
            pnl = exit_by_pos[i]
            prior = history[-(window - 1):] if window > 1 else []
            levels = prior + [pnl]
            if len(levels) >= window:
                ma20 = float(np.mean(levels))
                mode = -1 if pnl < ma20 else 1
            else:
                ma20 = float("nan")
                mode = 1
        observations = min(len(history) + 1, window)
        rows.append(
            {
                "date": date,
                "reference_pnl": pnl,
                "ma20": ma20,
                "has_exit": i in exit_by_pos,
                "mode_before_exit": mode_before,
                "mode_after_exit": mode,
                "observations": observations,
            }
        )
        history.append(pnl)

    return pd.DataFrame(rows, columns=COLUMNS)
