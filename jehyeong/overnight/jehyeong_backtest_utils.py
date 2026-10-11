"""Shared account returns, transaction costs and comparable daily metrics."""

import numpy as np
import pandas as pd

ANNUAL_SESSIONS = 252
COMMISSION_BP = 3.0
SLIPPAGE_BP = 2.0
ROUND_TRIP_COST = (COMMISSION_BP + SLIPPAGE_BP) / 10000.0


def account_returns(trades, calendar, start, end, eligible_entry_dates=None):
    """Each account starts in cash; book complete trades on their exit session.

    Exclude trades entered before the window or exited after it. Overnight
    returns therefore share the same realized calendar as intraday strategies.
    """
    dates = pd.DatetimeIndex(calendar).sort_values()
    dates = dates[(dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))]
    daily = pd.Series(0.0, index=dates, name="net_return")
    selected = trades.loc[
        (trades["entry_date"] >= pd.Timestamp(start))
        & (trades["exit_date"] <= pd.Timestamp(end))
    ].copy()
    if eligible_entry_dates is not None:
        selected = selected.loc[selected["entry_date"].isin(eligible_entry_dates)]
    if not selected.empty:
        if not selected["exit_date"].isin(dates).all():
            raise ValueError("Trade exit date is outside the evaluation calendar")
        realized = selected.groupby("exit_date")["net_return"].agg(lambda r: (1.0 + r).prod() - 1.0)
        daily.loc[realized.index] = realized
    return daily, selected


def performance_metrics(daily, trades=None):
    r = daily.astype(float)
    if r.empty or not np.isfinite(r).all():
        raise ValueError("Daily evaluation returns must be nonempty and finite")
    equity = (1.0 + r).cumprod()
    peaks = np.maximum.accumulate(np.r_[1.0, equity.to_numpy()])[1:]
    mdd = float((equity.to_numpy() / peaks - 1.0).min())
    gross = float(equity.iloc[-1])
    cagr = gross ** (ANNUAL_SESSIONS / len(r)) - 1.0 if gross > 0 else -1.0
    sd = float(r.std(ddof=1))
    sharpe = float(r.mean() / sd * np.sqrt(ANNUAL_SESSIONS)) if sd > 0 else np.nan
    calmar = cagr / abs(mdd) if mdd < 0 else (np.inf if cagr > 0 else np.nan)
    active = trades["net_return"] if trades is not None and len(trades) else pd.Series(dtype=float)
    return {
        "start": r.index.min().date().isoformat(), "end": r.index.max().date().isoformat(),
        "days": len(r), "trades": len(trades) if trades is not None else np.nan,
        "total_return": gross - 1.0, "CAGR": cagr, "MDD": mdd,
        "Sharpe": sharpe, "Calmar": calmar,
        "win_rate": float(active.gt(0).mean()) if len(active) else np.nan,
        "avg_trade_return": float(active.mean()) if len(active) else np.nan,
    }


def metrics_markdown(frame, label_col="strategy"):
    lines = [
        "| 전략 | 평가기간 | 거래 | 누적수익률 | CAGR | MDD | Sharpe | Calmar |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in frame.to_dict("records"):
        label = str(row[label_col]).replace("|", "\\|")
        lines.append(
            f"| {label} | {row['start']}~{row['end']} | {int(row['trades'])} "
            f"| {row['total_return']:.2%} | {row['CAGR']:.2%} | {row['MDD']:.2%} "
            f"| {row['Sharpe']:.3f} | {row['Calmar']:.3f} |"
        )
    return "\n".join(lines)
