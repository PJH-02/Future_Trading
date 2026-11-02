"""Compare fixed committed strategies on common KRX calendars, in separate accounts.

Run from any directory: python jehyeong/compare_committed_strategies.py
Peer entry/exit rules are loaded from Git, never reselected on the test windows.
"""

from __future__ import annotations

import ast
import contextlib
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import types

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
import jehyeong.overnight.f1_f2_f3_minute_eda_aligned_causal as own
from jehyeong.overnight.jehyeong_backtest_utils import (
    COMMISSION_BP, SLIPPAGE_BP, ROUND_TRIP_COST, account_returns,
    performance_metrics, metrics_markdown,
)

OUT = HERE / "results" / "team_strategy_comparison"
HOLDING_BARS = 30
LABELS = {
    "JEHYEONG_PROXY": "제형: 미국 종가 보정 proxy",
    "SUNNY_C": "sunny: Strategy C",
    "WOOHYUN_TOP40_30": "woohyun: VWAP+ORB+Top40, 30봉",
    "BEOMGU_C3_VT30": "beomgu: C3·VT30·ETF",
    "JEHYEONG_INTRADAY_KOSPI": "jehyeong: intraday final KOSPI200",
}
COLORS = {"JEHYEONG_PROXY": "tab:blue", "SUNNY_C": "tab:orange", "WOOHYUN_TOP40_30": "tab:green", "BEOMGU_C3_VT30": "tab:red", "JEHYEONG_INTRADAY_KOSPI": "tab:purple"}
PLOT_LABELS = {
    "JEHYEONG_PROXY": "JeHyeong overnight US-close proxy",
    "SUNNY_C": "Sunny Strategy C",
    "WOOHYUN_TOP40_30": "Woohyun VWAP + ORB + flow Top40",
    "BEOMGU_C3_VT30": "Beomgu C3 VT30 ETF",
    "JEHYEONG_INTRADAY_KOSPI": "JeHyeong intraday KOSPI200",
}
SOURCE_PATHS = {
    "SUNNY_C": "sunny/intraday_strategy_C/strategy3_smart_vs_dumb.py",
    "WOOHYUN_TOP40_30": "woohyun/Results/vwap+orb+flow40%.py",
    "BEOMGU_C3_VT30": "beomgu/overnight_lab/topics/r5_c3_levered.py",
}


def git(*args):
    result = subprocess.run(["git", "-C", str(ROOT), *args], check=True, capture_output=True)
    return result.stdout.decode("utf-8-sig").strip()


def committed_source(path):
    commit = git("log", "-1", "--format=%H", "--", path)
    source = git("show", f"{commit}:{path}")
    metadata = {
        "path": path, "commit": commit,
        "commit_date": git("show", "-s", "--format=%cI", commit),
        "commit_subject": git("show", "-s", "--format=%s", commit),
        "source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
        "working_copy_differs": bool(git("diff", "HEAD", "--", path)),
    }
    return source, metadata


def source_module(name, source, path):
    module = types.ModuleType(name)
    module.__file__ = str(ROOT / path)
    sys.modules[name] = module
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    return module


def same_day_ledger(name, panel, selected, position, gross, invested):
    dates = selected.index
    commission = invested.reindex(dates) * COMMISSION_BP / 10000.0
    slippage = invested.reindex(dates) * SLIPPAGE_BP / 10000.0
    return pd.DataFrame({
        "strategy": name, "entry_date": dates, "exit_date": dates,
        "entry_time": "09:00_OPEN", "exit_time": "15:30_CLOSE",
        "position": position.reindex(dates).values,
        "invested_weight": invested.reindex(dates).values,
        "gross_return": gross.reindex(dates).values,
        "commission": commission.values, "slippage": slippage.values,
        "net_return": (gross.reindex(dates) - commission - slippage).values,
    })


def prepare_sunny(source, metadata):
    m = source_module("team_sunny_committed", source, metadata["path"])
    panel = m.build_positions(m.build_signals(m.load_option_flow(), m.load_prices()))
    panel["available"] = panel[["for_put_z_lag", "ind_call_z_lag", "oc_ret"]].notna().all(axis=1)
    position = panel["pos_C"]
    gross = position * panel["oc_ret"]
    selected = panel.loc[panel["available"] & position.ne(0)]
    ledger = same_day_ledger("SUNNY_C", panel, selected, position, gross, position.abs())
    ledger["entry_price"] = selected["open"].values
    ledger["exit_price"] = selected["close"].values
    ledger["for_put_z_lag"] = selected["for_put_z_lag"].values
    ledger["ind_call_z_lag"] = selected["ind_call_z_lag"].values
    _, expected = m.backtest(panel, "pos_C", cost=ROUND_TRIP_COST)
    np.testing.assert_allclose(ledger["net_return"], expected.reindex(selected.index), atol=1e-14)
    return {
        "panel": panel, "ledger": ledger,
        "ready": panel.index[panel["available"]],
        "paths": [m.F_OPT, m.F_PRICE],
        "checks": {"source_net_return_equal": True},
    }


def woohyun_namespace(source, metadata):
    """Execute the committed data preparation and trade functions only.

    Stop before its performance/25-variant reporting code. Change only file
    locations to the existing repo inputs; retain the entire preparation AST.
    """
    paths = {
        "KODEX_FILE": ROOT / "woohyun/data/KODEX200_1min_2023_2026.csv",
        "BUY_FILE": ROOT / "woohyun/data/KRX foreign_buy.xlsx",
        "SELL_FILE": ROOT / "woohyun/data/KRX foreign_sell.xlsx",
    }
    nodes = []
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id in paths:
                node.value = ast.Constant(str(paths[target.id]))
        nodes.append(node)
        if isinstance(node, ast.FunctionDef) and node.name == "backtest":
            break
    else:
        raise ValueError("Committed Woohyun backtest function not found")
    namespace = {"__name__": "team_woohyun_committed", "__file__": str(ROOT / metadata["path"])}
    tree = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(tree, namespace["__file__"], "exec"), namespace)
    return namespace, list(paths.values())


def prepare_woohyun(source, metadata):
    ns, paths = woohyun_namespace(source, metadata)
    minute, flow = ns["kodex"], ns["flow"].copy()
    flow["date"] = pd.to_datetime(flow["date"])
    flow = flow.set_index("date")
    or_days = pd.DatetimeIndex(pd.to_datetime(minute.loc[
        minute["time"].between(ns["OR_START"], ns["OR_END"]), "date"
    ].unique()))
    price_days = pd.DatetimeIndex(pd.to_datetime(minute["date"].unique()))
    flow["has_opening_range"] = flow.index.isin(or_days)
    # Delayed market openings have valid prices/flow, but no 09:01~09:10 OR.
    # Preserve the committed no-trade rule; do not drop these calendar days.
    flow["available"] = flow[["previous_day_flow", "flow_top40_threshold"]].notna().all(axis=1) & flow.index.isin(price_days)
    trades = ns["backtest"](minute, HOLDING_BARS, "top40")
    dates = pd.to_datetime(trades["date"])
    ledger = pd.DataFrame({
        "strategy": "WOOHYUN_TOP40_30", "entry_date": dates, "exit_date": dates,
        "entry_time": trades["entry_time"], "exit_time": trades["exit_time"],
        "position": 1.0, "invested_weight": 1.0,
        "entry_price": trades["entry_price"], "exit_price": trades["exit_price"],
        "gross_return": trades["return"],
        "commission": COMMISSION_BP / 10000.0, "slippage": SLIPPAGE_BP / 10000.0,
        "net_return": trades["return"] - ROUND_TRIP_COST,
        "foreign_prev_flow": trades["foreign_flow"], "holding_bars": HOLDING_BARS,
    })
    if not ledger["entry_date"].isin(flow.index[flow["available"]]).all():
        raise ValueError("Woohyun trade occurs before its flow warmup or without OR data")
    entry_clock = pd.to_datetime(dates.dt.strftime("%Y-%m-%d") + " " + trades["entry_time"])
    exit_clock = pd.to_datetime(dates.dt.strftime("%Y-%m-%d") + " " + trades["exit_time"])
    ledger["entry_to_exit_label_minutes"] = (exit_clock - entry_clock).dt.total_seconds() / 60
    return {
        "panel": flow, "ledger": ledger, "ready": flow.index[flow["available"]],
        "paths": paths,
        "checks": {
            "committed_trade_functions_and_preparation": True,
            "full_ready_trades": len(ledger),
            "minimum_label_minutes": float(ledger["entry_to_exit_label_minutes"].min()),
            "maximum_label_minutes": float(ledger["entry_to_exit_label_minutes"].max()),
            "ready_days_without_opening_range": flow.index[flow["available"] & ~flow["has_opening_range"]].strftime("%Y-%m-%d").tolist(),
        },
    }


def etf_weights(w):
    """Cash allocations for the committed 1x/2x ETF vehicle (no borrowing)."""
    return pd.Series(np.where(w <= 1, w, 2 - w), index=w.index), pd.Series(np.where(w <= 1, 0.0, w - 1), index=w.index)


def prepare_beomgu(source, metadata):
    helper_paths = [
        "beomgu/overnight_lab/data.py", "beomgu/overnight_lab/engine.py",
        "beomgu/overnight_lab/common.py", "beomgu/overnight_lab/topics/r2_d2.py",
        "beomgu/overnight_lab/topics/r3_d2.py", "beomgu/overnight_lab/topics/bt_d2_2010_2026.py",
    ]
    for path in helper_paths:
        if git("diff", "HEAD", "--", path):
            raise ValueError(f"Committed peer helper has local changes: {path}")
    m = source_module("team_beomgu_committed", source, metadata["path"])
    # Force the repo's existing caches. Do not fetch or refresh historical data.
    m.data.RAW = ROOT / "beomgu/data"
    m.data.PX_DIR = m.data.RAW / "etf"
    paths = [m.data.PX_DIR / f"{code}.csv" for code in ("069500", "122630", "102110", "KPI200", "FUT")]
    paths += [m.data.RAW / "k200_index_daily.csv", m.data.RAW / "spx_daily.csv"]
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Offline committed-strategy cache missing: {path}")
    panel = m.load()
    variant = next(v for v in m.variants() if v["name"] == "C3|vt30|ETF")
    position = m.exposure(panel, variant)
    a, b = etf_weights(position)
    m.COST["K"] = (COMMISSION_BP + SLIPPAGE_BP, m.COST["K"][1])
    m.COST["L"] = (COMMISSION_BP + SLIPPAGE_BP, m.COST["L"][1])
    expected, gross = m.daily_net(panel, position, "ETF")
    panel["position"] = position
    panel["w_069500"], panel["w_122630"] = a, b
    panel["net_return"] = expected
    panel["available"] = panel[["C3", "sig_hat", "r_k", "r_l"]].notna().all(axis=1)
    selected = panel.loc[panel["available"] & position.gt(0)]
    ledger = same_day_ledger("BEOMGU_C3_VT30", panel, selected, position, gross, a + b)
    ledger["w_069500"], ledger["w_122630"] = a.reindex(selected.index).values, b.reindex(selected.index).values
    ledger["entry_price_069500"] = selected["k_o"].values
    ledger["exit_price_069500"] = selected["k_c"].values
    ledger["z_index"], ledger["mis"] = selected["z"].values, selected["mis"].values
    ledger["sig_hat"] = selected["sig_hat"].values
    np.testing.assert_allclose(ledger["net_return"], expected.reindex(selected.index), atol=1e-14)
    reference_path = ROOT / "beomgu/results/2026-09-29-overnight/round5_strategy/c3_levered/daily_returns.csv"
    reference = pd.read_csv(reference_path, parse_dates=["date"], index_col="date")
    shared = panel.index.intersection(reference.index)
    np.testing.assert_allclose(position.reindex(shared), reference.loc[shared, "w_chosen"], atol=1e-7)
    np.testing.assert_allclose(expected.reindex(shared), reference.loc[shared, "net5_chosen"], atol=1e-7)
    return {
        "panel": panel, "ledger": ledger, "ready": panel.index[panel["available"]],
        "paths": paths + [reference_path],
        "checks": {"source_net_return_equal": True, "committed_saved_weights_and_returns_equal": True},
        "helpers": helper_paths,
    }


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_benchmark(calendar):
    """KOSPI200 buy-and-hold proxy: unlevered KODEX200 close-to-close."""
    path = ROOT / "beomgu/data/etf/069500.csv"
    if not path.exists():
        raise FileNotFoundError(f"Benchmark cache missing: {path}")
    px = pd.read_csv(path, parse_dates=["date"]).set_index("date").sort_index()
    close = pd.to_numeric(px["close"], errors="coerce")
    return close.pct_change().reindex(pd.DatetimeIndex(calendar))


def make_plot(daily, benchmark, combined, path, title):
    plt.rcParams["font.family"] = "DejaVu Sans"
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    curves = {name: daily[name] for name in daily}
    curves["KOSPI200 Buy&Hold"] = benchmark
    curves["COMBINED"] = combined
    curve_colors = {**COLORS, "KOSPI200 Buy&Hold": "black", "COMBINED": "magenta"}
    for name, returns in curves.items():
        eq = (1.0 + returns).cumprod()
        axes[0].plot(eq.index, (eq - 1.0) * 100, label=name, color=curve_colors[name], linewidth=1.5 if name in {"KOSPI200 Buy&Hold", "COMBINED"} else 1.1)
        peaks = np.maximum.accumulate(np.r_[1.0, eq.to_numpy()])[1:]
        axes[1].plot(eq.index, (eq.to_numpy() / peaks - 1) * 100, label=name, color=curve_colors[name], linewidth=1.2 if name in {"KOSPI200 Buy&Hold", "COMBINED"} else 1.0)
    axes[0].set_ylabel("Cumulative PnL (%)")
    axes[0].set_title(title)
    axes[0].legend(fontsize=9)
    axes[1].set_ylabel("MDD / Drawdown (%)")
    axes[1].set_xlabel("KRX realization date")
    for ax in axes:
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def portfolio_metrics(daily, period, strategy_names, trade_count):
    """Equal-initial-capital composite: each account has the same starting cash."""
    r = daily[list(strategy_names)].mean(axis=1)
    eq = (1.0 + r).cumprod()
    peak = np.maximum.accumulate(np.r_[1.0, eq.to_numpy()])[1:]
    mdd = float((eq.to_numpy() / peak - 1.0).min())
    cagr = float(eq.iloc[-1] ** (252 / len(r)) - 1.0)
    sd = float(r.std(ddof=1))
    sharpe = float(r.mean() / sd * np.sqrt(252)) if sd > 0 else np.nan
    return {
        "period": period, "portfolio": f"동일자본 종합 ({len(strategy_names)}개)",
        "start": r.index.min().date().isoformat(), "end": r.index.max().date().isoformat(),
        "days": len(r), "trades": int(trade_count), "total_return": float(eq.iloc[-1] - 1.0),
        "CAGR": cagr, "MDD": mdd, "Sharpe": sharpe,
        "Calmar": cagr / abs(mdd) if mdd < 0 else np.nan,
    }


def plot_account_portfolio_pnl(equity, total_initial, path, title, extra_labels=None):
    """Plot all independently compounded PnL in KRW on one shared zero axis."""
    n_accounts = equity.shape[1]
    per_account = total_initial / n_accounts
    aggregate_equity = equity.mean(axis=1)
    aggregate_drawdown = aggregate_equity / np.maximum.accumulate(
        np.r_[1., aggregate_equity.to_numpy()]
    )[1:] - 1
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True,
                             gridspec_kw={"height_ratios": [1.6, 1]})
    palette = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown"]
    labels = extra_labels or {}
    for i, (name, series) in enumerate(equity.items()):
        axes[0].plot(equity.index, (series - 1) * per_account / 1_000_000,
                     color=palette[i], alpha=.78, linewidth=1.05,
                     label=f"{labels.get(name, PLOT_LABELS.get(name, name))} account PnL")
    combined_pnl = (aggregate_equity - 1) * total_initial / 1_000_000
    axes[0].plot(equity.index, combined_pnl, color="black", linewidth=2.2,
                 label=f"Combined PnL (KRW {total_initial/1_000_000:.0f}m initial)")
    axes[0].axhline(0, color="#555555", linewidth=.7)
    axes[0].set_ylabel("Cumulative net PnL (KRW million; same axis)")
    axes[0].legend(fontsize=8, ncol=2, loc="upper left")
    axes[1].fill_between(equity.index, aggregate_drawdown * 100, 0,
                         color="#333333", alpha=.25)
    axes[1].plot(equity.index, aggregate_drawdown * 100, color="#202020", linewidth=1.1,
                 label=f"Combined drawdown (MDD {aggregate_drawdown.min():.2%})")
    axes[1].set_ylabel("Combined drawdown from peak (%)")
    axes[1].set_xlabel("KRX PnL realization date")
    axes[1].legend(loc="lower left")
    for ax in axes:
        ax.grid(alpha=.24)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def build_five_strategy_portfolio(packs, calendar, benchmark_returns, provenance):
    """Combine five independently funded accounts on the requested full calendar."""
    start = pd.Timestamp("2022-12-13")
    end = min(packs[name]["coverage"][1] for name in
              ("JEHYEONG_PROXY", "SUNNY_C", "WOOHYUN_TOP40_30", "BEOMGU_C3_VT30"))
    intraday_dir = ROOT / "jehyeong/intraday/results/final"
    intraday_manifest_path = intraday_dir / "run_manifest.json"
    intraday_rules_path = intraday_dir / "selected_rules.json"
    intraday_summary_path = intraday_dir / "performance_summary.csv"
    intraday_daily_path = intraday_dir / "KOSPI200/daily_pnl.csv"
    intraday_trades_path = intraday_dir / "KOSPI200/trades.csv"
    for path in (intraday_manifest_path, intraday_rules_path, intraday_summary_path,
                 intraday_daily_path, intraday_trades_path):
        if not path.is_file():
            raise FileNotFoundError(f"Final intraday account result missing: {path}")
    run = json.loads(intraday_manifest_path.read_text(encoding="utf-8-sig"))
    rules = json.loads(intraday_rules_path.read_text(encoding="utf-8-sig"))
    if run.get("status") != "COMPLETE":
        raise ValueError(f"Final intraday run is not complete: {run.get('status')}")
    if set(rules.get("markets", {})) != {"KOSPI200", "KOSDAQ150"}:
        raise ValueError("Final intraday rules do not contain the selected KOSPI/KOSDAQ strategies")
    if rules["markets"]["KOSPI200"]["options"].get("direction_filter") != 1:
        raise ValueError("The intraday KOSPI account is not the final long-only selection")
    backtest_summary = pd.read_csv(intraday_summary_path)
    chosen = backtest_summary.loc[backtest_summary["market"].eq("KOSPI200")]
    if len(chosen) != 1 or not np.isclose(chosen.iloc[0]["commission_bps"], 1.5) or not np.isclose(chosen.iloc[0]["slippage_bps"], 1.0):
        raise ValueError("Intraday KOSPI results must use the frozen 1.5bp fee + 1bp slippage per side")
    intraday = pd.read_csv(intraday_daily_path, parse_dates=["date"]).set_index("date").sort_index()
    if intraday.index.has_duplicates or not np.isfinite(intraday["daily_return"]).all():
        raise ValueError("Intraday KOSPI daily PnL has duplicate dates or non-finite returns")
    daily_calendar = pd.DatetimeIndex(calendar).sort_values().unique()
    daily_calendar = daily_calendar[(daily_calendar >= start) & (daily_calendar <= end)]
    if not len(daily_calendar) or daily_calendar[0] != start:
        raise ValueError(f"Requested start {start.date()} is not available on the common KRX calendar")
    if daily_calendar[-1] != end:
        end = daily_calendar[-1]
        daily_calendar = daily_calendar[daily_calendar <= end]

    names = ["JEHYEONG_PROXY", "SUNNY_C", "WOOHYUN_TOP40_30",
             "BEOMGU_C3_VT30", "JEHYEONG_INTRADAY_KOSPI"]
    returns, selected_trades, rows = {}, {}, []
    for name in names[:-1]:
        daily, selected = account_returns(packs[name]["ledger"], daily_calendar, start, end)
        returns[name], selected_trades[name] = daily, selected
    # The intraday backtest intentionally omits dates failing its exact-bar quality
    # rules; these dates are kept as cash (zero return) in the independent account.
    returns[names[-1]] = intraday["daily_return"].reindex(daily_calendar).fillna(0.0)
    intraday_trades = pd.read_csv(intraday_trades_path, parse_dates=["entry_time", "exit_time"])
    intraday_trades = intraday_trades.loc[
        intraday_trades["entry_time"].dt.normalize().ge(start)
        & intraday_trades["exit_time"].dt.normalize().le(end)
    ].copy()
    selected_trades[names[-1]] = intraday_trades
    for name in names:
        selected = selected_trades[name]
        metric = performance_metrics(returns[name], selected if "net_return" in selected else None)
        metric["trades"] = len(selected)
        metric.update(strategy=name, label=LABELS[name])
        rows.append(metric)

    panel = pd.DataFrame(returns, index=daily_calendar)
    if panel.isna().any().any() or not np.isfinite(panel.to_numpy()).all():
        raise ValueError("Five-account daily return panel is incomplete")
    # Each account compounds its own capital without transfers or daily
    # rebalancing. Aggregate equity is therefore the mean of account equities,
    # not a daily arithmetic mean of returns.
    equity = (1 + panel).cumprod()
    combined_equity = equity.mean(axis=1)
    combined = combined_equity.pct_change()
    combined.iloc[0] = combined_equity.iloc[0] - 1
    trade_count = sum(len(v) for v in selected_trades.values())
    combined_metrics = performance_metrics(combined)
    stats = {"period": "FIVE_STRATEGIES_2022_12_13", "portfolio": "Five independent accounts",
             "start": combined.index.min().date().isoformat(), "end": combined.index.max().date().isoformat(),
             "days": len(combined), "trades": trade_count,
             "total_return": combined_metrics["total_return"], "CAGR": combined_metrics["CAGR"],
             "MDD": combined_metrics["MDD"], "Sharpe": combined_metrics["Sharpe"],
             "Calmar": combined_metrics["Calmar"]}
    total_initial = 5 * 100_000_000
    stats.update(initial_capital_per_account=100_000_000, initial_total_capital=total_initial,
                 final_total_equity=total_initial * (1 + stats["total_return"]))
    individual = pd.DataFrame(rows)
    out = ROOT / "jehyeong/overnight/results/five_strategy_portfolio_2022_12_13"
    out.mkdir(parents=True, exist_ok=True)
    panel.rename(columns=LABELS).to_csv(out / "daily_returns_by_strategy.csv", index_label="date", encoding="utf-8-sig")
    equity.rename(columns=LABELS).to_csv(out / "equity_by_strategy.csv", index_label="date", encoding="utf-8-sig")
    combined_stats = pd.DataFrame([stats])
    combined_stats.to_csv(out / "portfolio_summary.csv", index=False, encoding="utf-8-sig")
    individual.to_csv(out / "individual_strategy_summary.csv", index=False, encoding="utf-8-sig")
    benchmark = benchmark_returns.reindex(daily_calendar)
    if benchmark.isna().any():
        raise ValueError("KOSPI200 buy-and-hold benchmark has missing dates in the portfolio interval")
    dd = combined_equity / np.maximum.accumulate(np.r_[1., combined_equity.to_numpy()])[1:] - 1
    plot_account_portfolio_pnl(equity, total_initial, out / "five_strategy_pnl_mdd.png",
                               f"Five independent strategy accounts | {start.date()} to {end.date()} | KRW 100m each")

    portfolio_daily = pd.DataFrame({"date": daily_calendar, "portfolio_return": combined.to_numpy(),
                                    "portfolio_equity_multiple": combined_equity.to_numpy(),
                                    "portfolio_pnl_krw": (combined_equity.to_numpy() - 1) * total_initial,
                                    "portfolio_drawdown": dd})
    portfolio_daily.to_csv(out / "portfolio_daily_pnl.csv", index=False, encoding="utf-8-sig")
    all_trades = pd.concat([
        selected_trades[name].assign(strategy=name) for name in names
    ], ignore_index=True, sort=False)
    all_trades.to_csv(out / "trade_log.csv", index=False, encoding="utf-8-sig")
    intraday_run_hash = file_hash(intraday_manifest_path)
    provenance["five_strategy_portfolio"] = {
        "intraday_final_run_manifest_sha256": intraday_run_hash,
        "intraday_daily_pnl_sha256": file_hash(intraday_daily_path),
        "intraday_trades_sha256": file_hash(intraday_trades_path),
        "period_start_requested": start.date().isoformat(), "period_end_common": end.date().isoformat(),
        "initial_capital_each_account": 100_000_000, "accounts": names,
        "idle_cash_when_strategy_not_ready_or_day_not_eligible": True,
        "transaction_costs": "round-trip 5bp on traded capital/notional; intraday 3bp fee + 2bp slippage round trip from frozen side costs",
        "account_collision_assumption": "separate accounts, no shared buying power or order interaction",
        "output_directory": str(out.relative_to(ROOT)),
    }
    (out / "provenance.json").write_text(json.dumps(provenance["five_strategy_portfolio"],
                                                       indent=2, ensure_ascii=False), encoding="utf-8")
    report = [
        "# 5개 전략 별도 계좌 통합 백테스트", "",
        f"평가기간 **{start.date()}~{end.date()}**, KRX 거래일 {len(daily_calendar):,}일. "
        "종료일은 5개 전략 자료가 공통으로 존재하는 가장 최근 손익 실현일이다.", "",
        "각 계좌에 초기자본 1억 원을 각각 배정했다. 계좌는 각자 복리로 운용하고 자금을 이동하거나 매일 리밸런싱하지 않는다. "
        "매일의 통합 자산은 다섯 계좌 자산의 합이며 통합 초기자본은 5억 원이다. "
        "계좌 간 매수자금·포지션·체결은 독립적이고 현금일은 0% 수익이다. 수수료와 슬리피지를 포함한 순수익이다.", "",
        "## 통합 성과", "",
        "| 초기자본 | 최종자산 | 누적손익 | 누적수익률 | CAGR | MDD | Sharpe | 거래 수 |", "|---:|---:|---:|---:|---:|---:|---:|---:|",
        f"| {total_initial:,.0f}원 | {stats['final_total_equity']:,.0f}원 | {stats['final_total_equity']-total_initial:,.0f}원 | "
        f"{stats['total_return']:.2%} | {stats['CAGR']:.2%} | {stats['MDD']:.2%} | {stats['Sharpe']:.3f} | {trade_count:,} |", "",
        "![5개 전략 통합 PnL 및 MDD](five_strategy_pnl_mdd.png)", "",
        "## 전략별 성과", "",
        "| 전략 | 거래 수 | 누적수익률 | CAGR | MDD | Sharpe |", "|---|---:|---:|---:|---:|---:|",
    ]
    for row in individual.to_dict("records"):
        report.append(f"| {row['label']} | {int(row['trades'])} | {row['total_return']:.2%} | {row['CAGR']:.2%} | {row['MDD']:.2%} | {row['Sharpe']:.3f} |")
    report += ["", "## 해석 및 가정", "",
        "- woohyun 전략은 외국인 수급의 250일 준비 기간 때문에 2024-10-10부터 신호가 가능하다. 그 전까지 계좌는 현금으로 두어 수익률 0%를 반영했다.",
        "- jehyeong intraday는 최종 KOSPI200 롱온리·하루 1회 진입 전략의 저장된 일별 실현 손익을 사용했다. 유효 분봉 품질 검사를 통과하지 못한 날은 별도 계좌 현금일로 처리했다.",
        "- 익일 청산 전략의 순손익은 청산일에 기록한다. 모든 계좌를 같은 KRX 거래일 달력에 맞추고, 해당 날짜에 거래가 없으면 수익률 0이다.",
        "- Sharpe는 무위험수익률 0, 표본 일별 표준편차와 연 252거래일을 사용했다. CAGR도 252거래일 기준이다. MDD는 일별 손익 기준이므로 장중 미실현 낙폭은 포함하지 않는다.",
        "- beomgu 전략은 확정된 당일 시가를 본 뒤 같은 시가에 체결하는 원 전략 가정을 포함한다. 실행 가능성보다 원 전략의 과거 결과 재현에 초점을 둔다.",
        "- 전략별 손익의 상관관계와 동시 위험을 함께 반영하지만, 같은 시각 주문 충돌·시장충격·유동성 한도는 모델링하지 않는다. 별도 계좌라는 요청에 따라 자본 합산만 반영했다.",
        "- 그래프 위 패널은 개별 계좌 PnL과 통합 PnL을 같은 원화 축으로 표시하므로 모든 곡선이 0원에서 출발한다. 아래 패널은 통합 계좌의 일별 MDD다.", "",
        "`portfolio_summary.csv`, `individual_strategy_summary.csv`, `portfolio_daily_pnl.csv`, `daily_returns_by_strategy.csv`, `equity_by_strategy.csv`, `trade_log.csv`, `provenance.json`은 결과 재현과 점검을 위한 파일이다.", ""]
    (out / "report.md").write_text("\n".join(report), encoding="utf-8")
    return stats, out


def build_six_strategy_portfolio(five_out):
    """Add the final KOSDAQ150 account to the same five-account time series."""
    intraday_dir = ROOT / "jehyeong/intraday/results/final"
    daily_path = intraday_dir / "KOSDAQ150/daily_pnl.csv"
    trades_path = intraday_dir / "KOSDAQ150/trades.csv"
    rules_path = intraday_dir / "selected_rules.json"
    summary_path = intraday_dir / "performance_summary.csv"
    manifest_path = intraday_dir / "run_manifest.json"
    for path in (daily_path, trades_path, rules_path, summary_path, manifest_path,
                 five_out / "daily_returns_by_strategy.csv"):
        if not path.is_file():
            raise FileNotFoundError(f"Required saved strategy result missing: {path}")
    run = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    rules = json.loads(rules_path.read_text(encoding="utf-8-sig"))
    if run.get("status") != "COMPLETE":
        raise ValueError(f"Final intraday run is not complete: {run.get('status')}")
    options = rules["markets"]["KOSDAQ150"]["options"]
    if options.get("max_daily_entries") != 1 or options.get("minute_safety_stop_k") != .25:
        raise ValueError("KOSDAQ account is not the selected final one-entry safety-stop strategy")
    summary = pd.read_csv(summary_path)
    selected = summary.loc[summary["market"].eq("KOSDAQ150")]
    if len(selected) != 1 or not np.isclose(selected.iloc[0]["commission_bps"], 1.5) or not np.isclose(selected.iloc[0]["slippage_bps"], 1.):
        raise ValueError("KOSDAQ final results must use the frozen fee and slippage settings")

    five_returns = pd.read_csv(five_out / "daily_returns_by_strategy.csv", parse_dates=["date"]).set_index("date")
    five_equity = (1 + five_returns).cumprod()
    dates = five_returns.index
    raw_kosdaq = pd.read_csv(daily_path, parse_dates=["date"]).set_index("date").sort_index()
    if raw_kosdaq.index.has_duplicates or not np.isfinite(raw_kosdaq["daily_return"]).all():
        raise ValueError("KOSDAQ daily PnL has duplicate dates or non-finite returns")
    kosdaq_returns = raw_kosdaq["daily_return"].reindex(dates).fillna(0.)
    kosdaq_equity = (1 + kosdaq_returns).cumprod()
    label = "jehyeong: intraday final KOSDAQ150"
    equity = five_equity.copy()
    equity[label] = kosdaq_equity
    if equity.shape[1] != 6 or equity.isna().any().any():
        raise ValueError("Six independent account equity paths are not aligned")
    # No daily rebalance: total assets are the sum of six separately compounded accounts.
    aggregate = equity.mean(axis=1)
    aggregate_returns = aggregate.pct_change()
    aggregate_returns.iloc[0] = aggregate.iloc[0] - 1
    metrics = performance_metrics(aggregate_returns)
    trades_path_df = pd.read_csv(trades_path, parse_dates=["entry_time", "exit_time"])
    start, end = dates.min(), dates.max()
    kosdaq_trades = trades_path_df.loc[
        trades_path_df["entry_time"].dt.normalize().ge(start)
        & trades_path_df["exit_time"].dt.normalize().le(end)
    ]
    five_trades = pd.read_csv(five_out / "trade_log.csv")
    total_initial = 600_000_000
    final_equity = total_initial * aggregate.iloc[-1]
    stats = {"period": "SIX_STRATEGIES_2022_12_13", "portfolio": "Six independent accounts",
             **metrics, "start": start.date().isoformat(), "end": end.date().isoformat(),
             "days": len(dates), "trades": len(five_trades) + len(kosdaq_trades),
             "initial_capital_per_account": 100_000_000, "initial_total_capital": total_initial,
             "final_total_equity": final_equity, "total_net_pnl": final_equity - total_initial}

    out = ROOT / "jehyeong/overnight/results/six_strategy_portfolio_2022_12_13"
    out.mkdir(parents=True, exist_ok=True)
    equity.to_csv(out / "equity_by_strategy.csv", index_label="date", encoding="utf-8-sig")
    returns = equity.pct_change()
    returns.iloc[0] = equity.iloc[0] - 1
    returns.to_csv(out / "daily_returns_by_strategy.csv", index_label="date", encoding="utf-8-sig")
    pd.DataFrame([stats]).to_csv(out / "portfolio_summary.csv", index=False, encoding="utf-8-sig")
    aggregate_drawdown = aggregate / np.maximum.accumulate(np.r_[1., aggregate.to_numpy()])[1:] - 1
    portfolio_daily = pd.DataFrame({"date": dates, "portfolio_return": aggregate_returns.to_numpy(),
                                    "portfolio_equity_multiple": aggregate.to_numpy(),
                                    "portfolio_pnl_krw": (aggregate.to_numpy() - 1) * total_initial,
                                    "portfolio_drawdown": aggregate_drawdown.to_numpy()})
    portfolio_daily.to_csv(out / "portfolio_daily_pnl.csv", index=False, encoding="utf-8-sig")

    labels = {
        "제형: 미국 종가 보정 proxy": "JeHyeong overnight US-close proxy",
        "sunny: Strategy C": "Sunny Strategy C",
        "woohyun: VWAP+ORB+Top40, 30봉": "Woohyun VWAP + ORB + flow Top40",
        "beomgu: C3·VT30·ETF": "Beomgu C3 VT30 ETF",
        "jehyeong: intraday final KOSPI200": "JeHyeong intraday KOSPI200",
        label: "JeHyeong intraday KOSDAQ150",
    }
    plot_account_portfolio_pnl(equity, total_initial, out / "six_strategy_pnl_mdd.png",
                               f"Six independent strategy accounts | {start.date()} to {end.date()} | KRW 100m each",
                               extra_labels=labels)

    (out / "provenance.json").write_text(json.dumps({
        "period": [start.date().isoformat(), end.date().isoformat()], "trading_days": len(dates),
        "accounts": list(equity.columns), "initial_capital_each_account": 100_000_000,
        "independent_account_equity_sha256": file_hash(five_out / "equity_by_strategy.csv"),
        "kosdaq_intraday_manifest_sha256": file_hash(manifest_path),
        "kosdaq_intraday_daily_pnl_sha256": file_hash(daily_path),
        "kosdaq_intraday_trades_sha256": file_hash(trades_path),
        "commission_bps_per_side": 1.5, "slippage_bps_per_side": 1.0,
        "cost_basis": "saved final intraday per-account daily net returns; no costs deducted again",
        "aggregation": "mean of six individually compounded account equity paths; no transfers or rebalancing",
        "cash_on_missing_or_ineligible_days": True,
        "trade_count_six": stats["trades"],
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    report = [
        "# 6개 전략 독립 계좌 통합 성과", "",
        f"5개 전략 비교의 같은 기간인 **{start.date()}~{end.date()} ({len(dates):,} KRX 거래일)**에 "
        "jehyeong intraday final KOSDAQ150 전략을 초기자본 1억 원의 여섯 번째 독립 계좌로 추가했다.", "",
        "각 계좌 자산은 계좌별로 별도 복리 계산하고, 매일 여섯 자산을 합산했다. 계좌간 자금 이동과 매일 리밸런싱은 없다. "
        "비용이 반영된 기존 일별 손익을 사용해 중복 차감하지 않았다.", "",
        "| 기간 | 거래일 | 초기자본 | 최종자산 | 누적손익 | 누적수익률 | CAGR | MDD | Sharpe | 거래 수 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        f"| {stats['start']}~{stats['end']} | {stats['days']:,} | 600,000,000원 | {stats['final_total_equity']:,.0f}원 | "
        f"{stats['total_net_pnl']:,.0f}원 | {stats['total_return']:.2%} | {stats['CAGR']:.2%} | {stats['MDD']:.2%} | {stats['Sharpe']:.3f} | {stats['trades']:,} |", "",
        "![여섯 계좌 통합 PnL 및 MDD](six_strategy_pnl_mdd.png)", "",
        "상단의 개별 PnL과 통합 PnL은 모두 원화 기준 같은 세로축을 사용한다. 계좌별 손익과 합산 손익은 시작일 0원에서 출발한다.", "",
        "코스닥 계좌는 KODEX 코스닥150 신호와 기존 최종 코스닥 전략의 1분 보호청산을 그대로 사용했다. "
        "왕복 수수료 3bp와 슬리피지 2bp가 반영된 순손익이다. MDD는 일별 손익 기준이며 장중 미실현 낙폭은 포함하지 않는다.", "",
        "| 산출물 | 내용 |", "|---|---|",
        "| `portfolio_summary.csv` | 6계좌 통합 성과 지표 |",
        "| `portfolio_daily_pnl.csv` | 날짜별 통합 순수익, 자산, 손익, 낙폭 |",
        "| `equity_by_strategy.csv` | 6개 개별 계좌 자산 배수 |",
        "| `daily_returns_by_strategy.csv` | 6개 개별 계좌 일별 순수익 |",
        "| `provenance.json` | 사용 기간, 계좌 규칙, 데이터 SHA256 |", "",
    ]
    (out / "report.md").write_text("\n".join(report), encoding="utf-8")
    return stats, out


def analyze_six_strategy_pool(six_out):
    """Study realized net-return correlations and equal-initial-capital subsets."""
    import itertools
    raw = pd.read_csv(six_out / "daily_returns_by_strategy.csv", parse_dates=["date"]).set_index("date")
    codes = ["ON", "SC", "W40", "C3", "KP", "KQ"]
    if raw.shape[1] != 6 or raw.isna().any().any():
        raise ValueError("Correlation analysis requires the complete six-account return panel")
    expected = [LABELS[key] for key in ["JEHYEONG_PROXY", "SUNNY_C", "WOOHYUN_TOP40_30",
                                        "BEOMGU_C3_VT30", "JEHYEONG_INTRADAY_KOSPI"]]
    expected.append("jehyeong: intraday final KOSDAQ150")
    if set(raw.columns) != set(expected):
        raise ValueError("Correlation input columns do not identify the expected six strategies")
    raw = raw[expected]
    labels = dict(zip(codes, raw.columns))
    x = raw.copy()
    x.columns = codes
    out = six_out / "correlation_analysis"
    out.mkdir(parents=True, exist_ok=True)
    common = x.loc["2024-10-10":]
    weekly = (1 + x).resample("W-FRI").prod() - 1
    weekly = weekly.loc[x.resample("W-FRI").size().gt(0)]
    full_corr, common_corr, weekly_corr = x.corr(), common.corr(), weekly.corr()
    full_corr.to_csv(out / "daily_pearson_full.csv", encoding="utf-8-sig")
    common_corr.to_csv(out / "daily_pearson_common_ready.csv", encoding="utf-8-sig")
    weekly_corr.to_csv(out / "weekly_pearson.csv", encoding="utf-8-sig")
    x.rank().corr().to_csv(out / "daily_spearman_full.csv", encoding="utf-8-sig")
    active = x.abs().gt(1e-12)
    pair_rows = []
    for a, b in itertools.combinations(codes, 2):
        both = active[a] & active[b]
        union = active[a] | active[b]
        n = int(both.sum())
        losses_a, losses_b = x[a].lt(0), x[b].lt(0)
        pair_rows.append({"a": a, "b": b, "daily_corr": full_corr.loc[a, b],
                          "common_ready_corr": common_corr.loc[a, b], "weekly_corr": weekly_corr.loc[a, b],
                          "coactive_days": n, "coactive_corr": x.loc[both, [a, b]].corr().iloc[0, 1] if n >= 3 else np.nan,
                          "union_active_days": int(union.sum()),
                          "union_active_corr": x.loc[union, [a, b]].corr().iloc[0, 1],
                          "coactive_days_low_sample": n < 30,
                          "joint_loss_days": int((losses_a & losses_b).sum()),
                          "loss_overlap_given_a_loss": float((losses_a & losses_b).sum() / losses_a.sum()) if losses_a.any() else np.nan})
    pairs = pd.DataFrame(pair_rows)
    # Contiguous blocks preserve some short-range serial dependence. These are
    # descriptive intervals on the observed regimes, not future guarantees.
    rng = np.random.default_rng(20261008)
    sample = x.to_numpy()
    n = len(sample)
    correlations = []
    for _ in range(1000):
        starts = rng.integers(0, n - 20 + 1, size=int(np.ceil(n / 20)))
        indices = np.concatenate([np.arange(i, i + 20) for i in starts])[:n]
        correlations.append(np.corrcoef(sample[indices].T))
    correlations = np.asarray(correlations)
    for i, row in pairs.iterrows():
        a, b = codes.index(row.a), codes.index(row.b)
        lo, hi = np.nanquantile(correlations[:, a, b], [.025, .975])
        pairs.loc[i, "daily_corr_bootstrap_p025"] = lo
        pairs.loc[i, "daily_corr_bootstrap_p975"] = hi
    pairs.to_csv(out / "pair_diagnostics.csv", index=False, encoding="utf-8-sig")
    regime_corr = []
    for period, panel in [("2022_2024", x.loc[:"2024-12-31"]), ("2025", x.loc["2025"]), ("2026_YTD", x.loc["2026"])]:
        for a, b in itertools.combinations(codes, 2):
            regime_corr.append({"period": period, "days": len(panel), "a": a, "b": b, "corr": panel[a].corr(panel[b])})
    pd.DataFrame(regime_corr).to_csv(out / "period_correlations.csv", index=False, encoding="utf-8-sig")

    def pool_stats(panel):
        equity = (1 + panel).cumprod().mean(axis=1)
        returns = equity.pct_change()
        returns.iloc[0] = equity.iloc[0] - 1
        metrics = performance_metrics(returns)
        return {key: metrics[key] for key in ["total_return", "CAGR", "MDD", "Sharpe", "Calmar"]}

    single = []
    for c in codes:
        others = [s for s in codes if s != c]
        single.append({"strategy": c, "label": labels[c], **pool_stats(x[[c]]),
                       "active_days": int(active[c].sum()), "active_fraction": float(active[c].mean()),
                       "average_abs_daily_corr": full_corr.loc[c, others].abs().mean(),
                       "average_abs_weekly_corr": weekly_corr.loc[c, others].abs().mean(),
                       **{"common_" + k: v for k, v in pool_stats(common[[c]]).items()}})
    single = pd.DataFrame(single)
    single.to_csv(out / "individual_metrics.csv", index=False, encoding="utf-8-sig")
    baseline = pool_stats(x)
    leave_out = []
    for c in codes:
        stats = pool_stats(x.drop(columns=c))
        leave_out.append({"removed": c, **stats,
                          "CAGR_change": stats["CAGR"] - baseline["CAGR"],
                          "MDD_change": stats["MDD"] - baseline["MDD"],
                          "Sharpe_change": stats["Sharpe"] - baseline["Sharpe"]})
    pd.DataFrame(leave_out).to_csv(out / "leave_one_out.csv", index=False, encoding="utf-8-sig")
    subsets = []
    for size in range(1, 7):
        for subset in itertools.combinations(codes, size):
            mask = np.triu(np.ones((size, size), dtype=bool), k=1)
            corr = full_corr.loc[list(subset), list(subset)].to_numpy()
            stats = {"strategies": "+".join(subset), "number_of_accounts": size, **pool_stats(x[list(subset)]),
                     "mean_abs_daily_corr": float(np.abs(corr[mask]).mean()) if size > 1 else np.nan,
                     "max_abs_daily_corr": float(np.abs(corr[mask]).max()) if size > 1 else np.nan,
                     **{"common_" + k: v for k, v in pool_stats(common[list(subset)]).items()}}
            subsets.append(stats)
    subsets = pd.DataFrame(subsets)
    subsets.to_csv(out / "all_63_equal_capital_subsets.csv", index=False, encoding="utf-8-sig")
    candidates = ["ON+SC+W40+C3+KP+KQ", "ON+SC+C3+KP+KQ", "ON+C3+KP+KQ", "ON+C3+KQ", "C3+KP+KQ", "ON+C3"]
    subset_metrics = subsets.set_index("strategies")
    selected = subset_metrics.loc[candidates].reset_index()
    selected.to_csv(out / "candidate_portfolios.csv", index=False, encoding="utf-8-sig")
    period_stats = []
    for composition in candidates:
        cols = composition.split("+")
        for period, panel in [("2022_2024", x.loc[:"2024-12-31"]), ("2025", x.loc["2025"]), ("2026_YTD", x.loc["2026"])]:
            period_stats.append({"strategies": composition, "period": period, **pool_stats(panel[cols])})
    pd.DataFrame(period_stats).to_csv(out / "candidate_period_metrics.csv", index=False, encoding="utf-8-sig")

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.7), layout="constrained")
    for ax, (title, matrix) in zip(axes, [(f"Daily / full period ({len(x)} sessions)", full_corr),
                                         (f"Daily / all ready ({len(common)} sessions)", common_corr),
                                         (f"Weekly ({len(weekly)} weeks)", weekly_corr)]):
        graphic = ax.imshow(matrix, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(6), codes)
        ax.set_yticks(range(6), codes)
        ax.set_title(title, fontsize=10)
        for i in range(6):
            for j in range(6):
                ax.text(j, i, f"{matrix.iloc[i, j]:.3f}", ha="center", va="center", fontsize=8,
                        color="white" if abs(matrix.iloc[i, j]) > .6 else "black")
    fig.colorbar(graphic, ax=axes, label="Pearson correlation", shrink=.7)
    fig.suptitle("Six strategy net-return correlations | 2022-12-13 to 2026-09-17", fontsize=13)
    fig.savefig(out / "return_correlation_heatmap.png", dpi=180, facecolor="white")
    plt.close(fig)

    lines = ["# 6개 전략 알파 풀 상관관계와 조합 분석", "",
             f"{x.index.min().date()}~{x.index.max().date()}, {len(x)} KRX 거래일의 **거래비용 차감 후 일별 계좌 순수익률**로 계산했다. "
             "미거래일은 0이며 누적 PnL 곡선의 상관을 계산한 것이 아니다.", "",
             "| 코드 | 전략 |", "|---|---|"]
    lines += [f"| {c} | {labels[c]} |" for c in codes]
    lines += ["", "## 일별 Pearson 상관", "",
              "| 전략 | " + " | ".join(codes) + " |", "|---|" + "---:|" * 6]
    lines += ["| " + c + " | " + " | ".join(f"{v:.3f}" for v in full_corr.loc[c]) + " |" for c in codes]
    lines += ["", "![일별 및 주별 상관](return_correlation_heatmap.png)", "",
              "## 상관계수 해석", "",
              "- C3와 KP는 전체 일별 상관 0.230, 공통 준비기간 0.224다. 둘 다 손익이 발생한 57일의 조건부 상관은 0.727로 높다. 동일 코스피 장중 롱 성격이므로 서로 완전히 다른 알파로 보기는 어렵다.",
              "- ON과 C3, ON과 KQ, C3와 KQ의 전체 일별 상관은 각각 0.024, 0.017, 0.010이다. 관측 손익 기준으로 수익성 있는 전략 중 서로 덜 겹치는 조합이다.",
              "- ON과 W40은 일별 0.032이지만 주별 0.342다. SC와 W40도 주별 0.250이다. 모든 거래일의 일별 0 근처 값만으로 직교성을 판단하면 일부 같은 시기 수익/손실을 놓친다.",
              "- SC와 KP의 동시 손익 발생일 상관은 -0.884지만 표본은 12일뿐이다. SC와 C3, SC와 W40은 각각 5일이다. 이러한 작은 조건부 표본을 강한 헤지 근거로 사용하지 않는다.",
              "- W40의 2024-10-10 전 현금 구간을 제외한 474일 분석도 따로 계산했다. KP와 KQ 상관은 전체 0.067, 이 구간 -0.054이므로 상관은 기간에 따라 변한다.", "",
              "## 전략별 수익성과 활동일", "",
              "| 전략 | CAGR | MDD | Sharpe | 손익 발생일 | 평균 절대 일별 상관 |", "|---|---:|---:|---:|---:|---:|"]
    for row in single.to_dict("records"):
        lines.append(f"| {row['strategy']} | {row['CAGR']:.2%} | {row['MDD']:.2%} | {row['Sharpe']:.3f} | {row['active_days']} | {row['average_abs_daily_corr']:.3f} |")
    lines += ["", "## 동등 초기자본 조합 비교", "",
              "각 전략 계좌는 각자 복리 운용하고 계좌간 자금 이동을 하지 않는다. 모든 조합은 전체 초기자본으로 정규화하여 비교했다. "
              "전략을 제외한 계좌의 초기자본을 남은 전략에 동일하게 배분하는 비교이므로 같은 자본이 투자되는 경우의 효과다.", "",
              "| 구성 | CAGR | MDD | Sharpe | Calmar | 평균 절대 일별 상관 |", "|---|---:|---:|---:|---:|---:|"]
    for row in selected.to_dict("records"):
        lines.append(f"| {row['strategies']} | {row['CAGR']:.2%} | {row['MDD']:.2%} | {row['Sharpe']:.3f} | {row['Calmar']:.3f} | {row['mean_abs_daily_corr']:.3f} |")
    lines += ["", "## 운용 후보 해석", "",
              "직교성과 수익성을 우선하면 **ON+C3+KQ**를 핵심 검토 후보로 삼을 수 있다. 세 전략의 일별 상관이 모두 낮고 개별 CAGR가 높다. "
              "다만 세 계좌 조합의 MDD는 6계좌보다 크므로 같은 위험에서 더 높은 수익을 입증한 것은 아니다.", "",
              "더 간단한 **ON+C3**는 CAGR 12.87%, MDD 3.74%, Sharpe 1.807이다. KQ를 추가하면 Sharpe는 1.908로 높아지지만 "
              "MDD는 4.85%로 커지고 CAGR는 12.77%가 된다. KQ가 추가될 때 모든 위험 지표가 좋아지는 것은 아니다.", "",
              "낙폭과 Sharpe를 더 중시하면 **ON+C3+KP+KQ**를 비교한다. KP는 C3와 동시 거래 손익이 많이 겹치지만 "
              "이 표본에서는 전체 조합의 낙폭을 낮췄다. 신호 발생시점·동시 코스피 노출 제한을 다음 검토 대상으로 삼을 수 있다.", "",
              "**SC와 W40은 저상관이어도 낮은 수익성이 약점**이다. W40을 제외한 5전략은 CAGR와 Calmar가 높아지며 "
              "Sharpe는 6전략과 거의 같다. SC 역시 낮은 수익과 비용 부담을 점검할 후보다. "
              "다만 6전략은 전체 63조합 중 과거 전체기간 Sharpe가 가장 높아, 두 전략을 무조건 무가치하다고 판단해서는 안 된다.", "",
              "Calmar만으로 사후 고르면 C3+KP+KQ가 높지만 C3/KP의 조건부 상관이 크다. "
              "63조합의 최고값은 같은 자료에서 골랐으므로 독립 미관찰 기간 성과가 아니다. 가중치 최적화는 수행하지 않았다.", "",
              "## 계산의 범위", "",
              "- ON 손익은 익일 청산일에 기록하고 장중 전략은 당일 기록한다. 자본 합산의 일별 위험을 보는 데 적합하지만 실제 같은 시간의 시장 노출 독립성을 보장하지 않는다. 주별 상관은 기록 시점의 영향을 일부 완화한다.",
              "- 활동일은 절대 일별 순수익률이 1e-12보다 큰 날이다. 동시 손익 발생일 상관은 해당 부분표본에 조건을 건 값이므로 전체 위험 상관과 서로 다른 통계다.",
              "- pair_diagnostics.csv의 95% 구간은 20거래일 연속 블록을 1,000회 재추출한 관측자료 상관 구간이다. 미래 국면에서 상관이 유지된다는 보장은 아니다.",
              "- 구간별 성과는 기존 계좌 수익률을 구간 시작에서 재정규화한 분석이며 구간 전 진입의 청산 손익이 포함될 수 있다. 사전 규칙을 동결한 독립 검증으로 취급하지 않는다.",
              "- 원 전략의 시가 신호/동일 시가 체결 가정(C3), 선물 proxy(SC), 분봉 제외일의 현금 처리 등 기존 백테스트 제약을 그대로 물려받는다.", "",
              "전체 상관 CSV, 주별/공통기간/Spearman 행렬, 동시 손익일·손실 겹침·bootstrap 구간, 개별 성과, 전략 하나씩 제외한 결과, "
              "63개 동등 초기자본 조합과 후보의 연도별 지표를 함께 저장했다.", ""]
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    save = {"start": str(x.index.min().date()), "end": str(x.index.max().date()), "days": len(x),
            "common_ready_days": len(common), "weekly_observations": len(weekly), "labels": labels,
            "input_sha256": file_hash(six_out / "daily_returns_by_strategy.csv"),
            "analysis_code_sha256": file_hash(Path(__file__)),
            "annual_sessions": 252, "bootstrap_block_sessions": 20, "bootstrap_samples": 1000,
            "bootstrap_seed": 20261008, "subset_count": len(subsets), "selection_is_exploratory": True}
    (out / "analysis_manifest.json").write_text(json.dumps(save, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def build_report(summary, portfolio, coverage, provenance, audit, yearly):
    short = summary.loc[summary["period"] == "PERIOD_1_FOUR"]
    long = summary.loc[summary["period"] == "PERIOD_2_THREE"]
    lines = [
        "# 커밋된 전략의 공통 기간·별도 계좌 비교", "",
        "## 결과", "",
        "각 계좌의 초기자산을 1로 놓고 독립적으로 복리 운용했다. 아래 수치는 왕복 수수료 3bp와 슬리피지 2bp를 차감한 결과다. "
        "전략을 합친 포트폴리오 성과는 계산하지 않았다.", "",
        f"### 1기간: 4개 전략 ({short.iloc[0]['start']}~{short.iloc[0]['end']}, {short.iloc[0]['days']} KRX 거래일)", "",
        metrics_markdown(short, "label"), "",
        f"### 2기간: 3개 전략 ({long.iloc[0]['start']}~{long.iloc[0]['end']}, {long.iloc[0]['days']} KRX 거래일)", "",
        metrics_markdown(long, "label"), "",
        "## 동일자본으로 4/3개 전략을 종합한 성과", "",
        "각 전략 계좌에 동일한 초기자본을 배정하고, 매 평가일의 전략별 순수익률을 산술평균해 종합 계좌의 수익률을 계산했다. "
        "따라서 각 전략의 수익률이 같은 비중으로 반영되며, 개별 계좌의 거래금액을 합친 뒤 다시 초기 총자본으로 나누는 방식과 같다.", "",
        metrics_markdown(portfolio.rename(columns={"portfolio": "label"}), "label"), "",
        "### 성과 해석", "",
        "1기간에서는 제형 proxy가 CAGR·Calmar가 높고, beomgu는 Sharpe가 더 높으며 MDD가 작다. "
        "2기간에서는 beomgu가 CAGR·Sharpe·Calmar 모두 높다. 제형 proxy는 장기 구간에 비해 최근 구간 성과가 강해 기간 선택에 민감하다. "
        "장기 연도별 표에서 2022년을 포함한 수익 분포를 확인할 수 있다.",
        "제형 proxy는 2022~2024년 각각 −1.22%, −2.25%, −6.54%였고, 2025년 +1.82%, "
        "2026년은 공통 종료일까지 +71.32%다. 장기 수익의 상당 부분이 마지막 연도에 집중되어 있다.",
        "woohyun의 기존 README 평균 총수익은 거래당 약 6.93bp이다. 요청한 5bp 비용을 차감하고 미거래일을 포함하여 "
        "공통 기간에서 다시 계산하므로 비용 없는 기존 Sharpe와 차이가 크다. "
        "전략의 매매 대상·방향·비중·신호 시각은 서로 다르며, beomgu에는 아래의 시가 신호·시가 체결 가정이 포함된다. "
        "이 표는 고정된 각 전략의 계좌 성과 비교이며 같은 위험이나 실행 가능성을 보장하는 비교는 아니다.", "",
        "그래프의 검은색 선은 비용을 차감하지 않은 KOSPI200 buy-and-hold(069500 close-to-close), 자홍색 선은 동일자본 전략 결합 계좌다. "
        "위 패널은 누적 PnL, 아래 패널은 각 곡선의 고점 대비 drawdown이다.", "",
        "## 왜 이 기간인가", "",
        "- 우현 수급 파일은 2023-09-25부터 있으나, 전일 수급의 상위 40% 기준에 과거 250일이 필요하여 첫 유효일은 2024-10-10이다. "
        "번인 기간을 현금 운용 기간으로 늘려 성과를 계산하지 않았다.",
        "- 장기 구간은 proxy_panel.csv가 시작하는 2021-12-20부터다. 2022년 전체를 포함하면서 최대 겹침을 사용했다.",
        "- sunny의 커밋된 가격 파일이 2026-09-17까지여서 두 기간의 공통 종료일은 2026-09-17이다. "
        "그 이후의 가격으로 자료를 확장하지 않았다. 해당 지수 파일의 마지막 행은 원 저자 코드에 장중 스냅샷 가능성이 언급되어 있다.", "",
        "| 전략 | 계산 가능한 첫 신호일 | 마지막 신호일 | 비교 가능한 마지막 손익일 |",
        "|---|---|---|---|",
    ]
    for row in coverage.to_dict("records"):
        lines.append(f"| {LABELS[row['strategy']]} | {row['first_ready']} | {row['last_ready']} | {row['last_realization']} |")
    lines += ["", "## 전략별 규칙", "",
        "### 제형: 미국 종가 기준 보정 proxy", "",
        "QQQ proxy 304940·SPY proxy 219480의 당일 15:20까지 실제 가격으로 계산한다. "
        "E0는 직전 한국장 종가, E1은 오늘 15:20 가격, A는 직전 한국장 종가 당시 알려진 미국 종가, B는 오늘 15:20 당시 알려진 미국 종가다. "
        "`log(E1/E0) − log(B/A)`가 두 ETF 모두 > 0이고, KRX_foreign.xlsx의 직전 실제 한국 거래일 외국인 순매수 > 0이면 "
        "KODEX200 당일 종가 매수·다음 한국 거래일 시가 매도한다. 자본 100% 롱이다.",
        "미국 현지 세션 날짜와 뉴욕 16:00→한국 시간 변환, 서머타임·한미 휴일을 반영한다. "
        "보정 지표는 기존 관측 PREUS의 근사이며 ETF 추적 차이·직전 한국 종가 시점의 괴리 잔차가 남는다.", "",
        "### sunny: 최신 Strategy C의 case C", "",
        "`sunny/intraday_strategy_C/strategy3_smart_vs_dumb.py`의 핵심 case C를 사용한다. "
        "옵션 순매수 금액의 60일 rolling z-score를 하루 lag한다. 전일 외국인 풋 z > 1이고 개인 콜 z > 0이면 "
        "KOSPI200 당일 시가에 자본 대비 1배 숏, 당일 종가 청산한다. 다른 case A/B/D는 포함하지 않았다.",
        "원 전략의 KOSPI200 지수 open-to-close를 선물 수익률 proxy로 사용한다. 실제 선물 가격·증거금·베이시스·롤 비용은 별도 모델링하지 않았다.", "",
        "### woohyun: VWAP + ORB + Flow Top40, 30 설정", "",
        "전일 외국인 선물 `(매수−매도)/(매수+매도)`가 그 이전까지의 250일 rolling 60% 분위 이상인 날에만 거래한다. "
        "09:01~09:10의 opening range 고가보다 높고 누적 VWAP보다 높은 첫 종가를 09:11 이후 찾는다. "
        "신호 다음 유효 봉의 시가로 KODEX200 자본 100% 롱, 진입 인덱스 +30 봉 종가로 청산한다. 하루 최대 한 거래다.",
        "README에서 강조한 30 설정을 고정했으며 비교 성과로 보유시간을 고르지 않았다. "
        "원 코드가 거래량 0 봉을 제거하므로 **30 설정은 30개의 유효 봉 인덱스 차이이며 정확히 30분 뒤 주문은 아니다**. "
        "진입 봉부터 청산 봉까지 양쪽을 포함하면 31개 봉이며, 누락 봉이 있으면 시간은 더 길어진다. 이 동작을 그대로 유지했다.", "",
        "수능일·연초처럼 장이 늦게 열려 09:01~09:10 자료가 없는 날에는 원 코드대로 거래하지 않는다. "
        "자료가 존재하는 평가일이므로 그날을 삭제하지 않고 현금 수익률 0으로 포함한다.", "",
        "### beomgu: r5_c3_levered의 C3|vt30|ETF", "",
        "미국 S&P500 수익으로 설명한 KOSPI200 시가 갭의 회귀 잔차 z ≥ 1이며, "
        "`log(KODEX 시가/지수 시가)`가 직전 5일 종가 비율 로그의 중앙값 이하이면 C3 롱 신호다. "
        "회귀는 당일을 뺀 과거 250행(최소 200 유효행)으로 계산한다. "
        "전일까지만의 20일 지수 종가 수익률 변동성 σ에 대해 `w=C3×clip(0.30/σ,0,2)`를 적용한다.",
        "w≤1이면 KODEX200 비중 w와 현금 1−w, 1<w≤2이면 KODEX200 비중 2−w 및 실제 KODEX 레버리지 122630 비중 w−1이다. "
        "당일 시가→종가 수익률을 각 비중으로 합친다. 전체 구간에 실제 레버리지 ETF 자료가 있어 2배 합성 수익률을 쓰지 않는다.",
        "**당일 09:00 시가를 보고 신호를 계산하면서 같은 09:00 시가 단일가에 체결하는 가정이다. "
        "체결 이후 알게 되는 확정 시가를 같은 시가 체결에 사용하므로 실제 실행보다 유리할 수 있다.** "
        "이는 커밋된 주 전략의 가정을 유지한 비교이며, 실제 주문 가능한 성과 검증과 구분해야 한다. "
        "최종 선택 C3|vt30|ETF를 고정하고 비교 기간에서 42개 후보를 다시 선택하지 않았다.", "",
        "## 비용·계좌·성과 계산 규약", "",
        "- 제형·sunny·woohyun은 한 번 거래할 때 계좌 대비 1배 거래금액에 왕복 5bp를 한 번 차감한다. "
        "기존 순수익에 비용을 중복 차감하지 않고 각 원자료 총수익에서 다시 계산한다.",
        "- beomgu는 실제 매매하는 두 ETF의 **자본 비중 합**에 각각 수수료 3bp·슬리피지 2bp를 적용한다. "
        "레버리지 ETF에 전액 투자하여 지수 노출이 2배여도 현금 거래금액은 자본의 1배라 5bp다. w<1일 때 비용도 w에 비례한다.",
        "- 각 전략은 별도 계좌·자본 1로 시작한다. 계좌 간 신호, 포지션, 거래 순서, 주문을 합치지 않는다. "
        "장중 주문 혼잡이나 전략 간 체결 경쟁은 없다고 가정한다.",
        "- 모든 KRX 평가 거래일을 유지하고 미거래일은 수익률 0이다. 장중·오버나이트를 같은 달력으로 비교하도록 손익·왕복 비용을 **청산일**에 기록한다. "
        "제형 전략의 진입일과 청산일은 거래장부에서 별도로 확인할 수 있다.",
        "- 시작일 전에 들어간 포지션 및 종료일 후에 청산하는 거래를 제외한다. 종료일의 신규 오버나이트 진입은 비교 성과에 포함하지 않는다.",
        "- 모든 신호·rolling window는 전체 과거 자료로 계산한 뒤 평가 구간을 자른다. 구간 시작에서 번인을 다시 하지 않는다. "
        "자료가 없는 날에 미래·전일 가격을 채우지 않고 신규 진입하지 않는다.",
        "- CAGR = 최종자산^(252/N)−1. N은 공통 KRX 거래일 수다. Sharpe = 일별 순수익률 평균 / 표본 표준편차 ×√252, 무위험수익률 0. "
        "MDD = 초기자산 1을 포함한 고점 대비 최대 하락률. Calmar = CAGR/|MDD|. 오버나이트의 장중 미실현 낙폭은 일별 청산 손익 MDD에 들어가지 않는다.",
        "- sunny 원 보고서의 CAGR/연변동성 방식 및 woohyun의 거래일만으로 산출한 Sharpe는 재사용하지 않았다. "
        "모두 위의 동일한 일별 계좌 정의로 다시 계산했으므로 기존 폴더의 요약 숫자와 다를 수 있다.", "",
        "## 결측·원본 대조", "",
        "| 기간 | 전략 | 신호 자료 없는 평가일 수 | 해당 날짜 |", "|---|---|---:|---|",
    ]
    for row in audit:
        lines.append(f"| {row['period']} | {LABELS[row['strategy']]} | {row['missing_feature_days']} | {', '.join(row['missing_dates']) or '없음'} |")
    lines += ["", "제형의 자료 유무는 진입일 기준이다. 구간 끝 신규 진입 불가와는 별개다. "
              "peer source 함수의 순수익과 새 거래장부를 대조했고, beomgu의 저장된 w_chosen·net5_chosen도 수치 오차 1e−7 이내로 일치했다. "
              "우현은 커밋된 자료 준비 및 진입·청산 함수를 직접 실행했다.", "",
              "## 연도별 공통 계좌 순수익", "",
              "| 기간 | 연도 | 전략 | 순수익률 | 거래 |", "|---|---:|---|---:|---:|"]
    for row in yearly.to_dict("records"):
        lines.append(f"| {row['period']} | {row['year']} | {LABELS[row['strategy']]} | {row['return']:.2%} | {row['trades']} |")
    lines += ["", "연도별 수치는 기간 내 일부 연도도 포함하며 연환산하지 않았다.", "",
              "## 커밋·재현", "", "| 전략 | 커밋 | 커밋 시각 |", "|---|---|---|"]
    for name in SOURCE_PATHS:
        meta = provenance["sources"][name]
        lines.append(f"| {LABELS[name]} | `{meta['commit']}` | {meta['commit_date']} |")
    lines += ["", "peer 전략은 해당 파일의 최신 커밋 내용을 Git에서 읽었다. 로컬 우현 파일의 경로 수정은 신호 규칙으로 사용하지 않았다. "
        "우현의 커밋된 코드에 적힌 입력 파일명만 현재 저장소의 실제 파일 경로로 연결했다. peer 코드와 원자료를 수정하지 않는다.",
        "제형의 보정 proxy 구현 및 사용 데이터·helper 코드의 SHA256, 라이브러리 버전·체크 결과는 provenance.json에 저장했다.", "",
        "```powershell", "python jehyeong/f1_f2_f3_minute_eda_aligned_causal.py",
        "python jehyeong/compare_committed_strategies.py",
        "python -m unittest discover -s jehyeong/tests -v", "```", "",
        "![4전략 공통 기간](period_1_four_equity_drawdown.png)", "",
        "![3전략 장기 공통 기간](period_2_three_equity_drawdown.png)", "",
        "summary.csv: 개별 전략 지표 / portfolio_summary.csv: 동일자본 종합 지표 / 각 period_*_daily_returns.csv: 공통 달력 손익 / period_trade_log.csv: 실제 평가에 포함한 거래 / "
        "all_strategy_trade_log.csv: 전체 자료 거래 / 각 *_signal_panel.csv: 신호·비중 감사 / coverage.csv·provenance.json: 재현 정보.", "",
    ]
    return "\n".join(lines)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    prepared = own.prepare_strategies()
    calendar = prepared["research"].index
    proxy_model = prepared["models"]["US_CLOSE_PROXY"]
    packs = {"JEHYEONG_PROXY": {
        "panel": proxy_model, "ready": proxy_model.index[proxy_model["available"]],
        "ledger": prepared["ledgers"]["US_CLOSE_PROXY"].assign(strategy="JEHYEONG_PROXY"),
        "paths": list(prepared["paths"].values()), "checks": {"overnight_actual_next_session": True},
    }}
    provenance = {
        "git_head": git("rev-parse", "HEAD"), "sources": {}, "data_sha256": {}, "code_sha256": {},
        "settings": {"commission_round_trip_bp": COMMISSION_BP, "slippage_round_trip_bp": SLIPPAGE_BP,
                     "woohyun_holding_bars": HOLDING_BARS, "annual_sessions": 252, "risk_free_return": 0,
                     "separate_accounts": True, "pnl_date": "exit_date"},
        "versions": {"python": sys.version, "pandas": pd.__version__, "numpy": np.__version__, "matplotlib": matplotlib.__version__},
        "checks": {},
    }
    benchmark_path = ROOT / "beomgu/data/etf/069500.csv"
    provenance["data_sha256"][benchmark_path.relative_to(ROOT).as_posix()] = file_hash(benchmark_path)
    benchmark_returns = load_benchmark(calendar)
    for name, factory in (("SUNNY_C", prepare_sunny), ("WOOHYUN_TOP40_30", prepare_woohyun), ("BEOMGU_C3_VT30", prepare_beomgu)):
        print(f"Preparing {name} from committed source...", flush=True)
        source, meta = committed_source(SOURCE_PATHS[name])
        provenance["sources"][name] = meta
        packs[name] = factory(source, meta)
    next_session = calendar.to_series().shift(-1)
    coverage_rows = []
    for name, pack in packs.items():
        ready = pd.DatetimeIndex(pack["ready"])
        last_realization = next_session.loc[ready.max()] if name == "JEHYEONG_PROXY" else ready.max()
        pack["coverage"] = ready.min(), last_realization
        coverage_rows.append({
            "strategy": name, "first_ready": ready.min().date().isoformat(),
            "last_ready": ready.max().date().isoformat(), "last_realization": last_realization.date().isoformat(),
            "full_ready_days": len(ready),
        })
        pack["panel"].to_csv(OUT / f"{name.lower()}_signal_panel.csv", index_label="date", encoding="utf-8-sig")
        provenance["checks"][name] = pack["checks"]
        for path in pack["paths"]:
            path = Path(path).resolve()
            provenance["data_sha256"][path.relative_to(ROOT).as_posix()] = file_hash(path)
        for relative in pack.get("helpers", []):
            provenance["code_sha256"][relative] = file_hash(ROOT / relative)
    for path in [Path(__file__), Path(own.__file__), HERE / "jehyeong_backtest_utils.py"]:
        provenance["code_sha256"][path.relative_to(ROOT).as_posix()] = file_hash(path)
    coverage = pd.DataFrame(coverage_rows)
    coverage.to_csv(OUT / "coverage.csv", index=False, encoding="utf-8-sig")
    all_trades = pd.concat([p["ledger"] for p in packs.values()], ignore_index=True)
    all_trades.to_csv(OUT / "all_strategy_trade_log.csv", index=False, encoding="utf-8-sig")

    groups = {"PERIOD_1_FOUR": ["JEHYEONG_PROXY", "SUNNY_C", "WOOHYUN_TOP40_30", "BEOMGU_C3_VT30"],
              "PERIOD_2_THREE": ["JEHYEONG_PROXY", "SUNNY_C", "BEOMGU_C3_VT30"]}
    rows, portfolio_rows, logs, audit, annual = [], [], [], [], []
    for period, names in groups.items():
        start = max(packs[name]["coverage"][0] for name in names)
        end = min(packs[name]["coverage"][1] for name in names)
        series = {}
        for name in names:
            pack = packs[name]
            daily, selected = account_returns(pack["ledger"], calendar, start, end)
            series[name] = daily
            row = performance_metrics(daily, selected)
            row.update(period=period, strategy=name, label=LABELS[name])
            rows.append(row)
            logs.append(selected.assign(period=period))
            missing = daily.index.difference(pack["ready"])
            audit.append({"period": period, "strategy": name, "missing_feature_days": len(missing),
                          "missing_dates": missing.strftime("%Y-%m-%d").tolist()})
            if name != "JEHYEONG_PROXY" and len(missing):
                raise ValueError(f"Peer has missing feature sessions in common period: {name}: {missing}")
            for year in daily.index.year.unique():
                yr = daily[daily.index.year == year]
                annual.append({"period": period, "year": int(year), "strategy": name,
                               "return": float((1.0 + yr).prod() - 1),
                               "trades": int(selected["exit_date"].dt.year.eq(year).sum())})
        daily = pd.DataFrame(series)
        benchmark = benchmark_returns.reindex(daily.index)
        if benchmark.isna().any():
            raise ValueError(f"KOSPI200 benchmark has missing close-to-close returns: {benchmark[benchmark.isna()].index.tolist()}")
        combined = daily.mean(axis=1)
        period_trade_count = sum(row["trades"] for row in rows if row["period"] == period)
        portfolio_rows.append(portfolio_metrics(daily, period, names, period_trade_count))
        stem = period.lower()
        daily.to_csv(OUT / f"{stem}_daily_returns.csv", index_label="date", encoding="utf-8-sig")
        (1.0 + daily).cumprod().to_csv(OUT / f"{stem}_equity.csv", index_label="date", encoding="utf-8-sig")
        make_plot(daily, benchmark, combined, OUT / f"{stem}_equity_drawdown.png", f"{period}: {start.date()} to {end.date()} | PnL / MDD | round-trip cost 5bp")
    five_stats, five_out = build_five_strategy_portfolio(packs, calendar, benchmark_returns, provenance)
    six_stats, six_out = build_six_strategy_portfolio(five_out)
    correlation_out = analyze_six_strategy_pool(six_out)
    summary = pd.DataFrame(rows)
    portfolio = pd.DataFrame(portfolio_rows)
    summary.to_csv(OUT / "summary.csv", index=False, encoding="utf-8-sig")
    portfolio.to_csv(OUT / "portfolio_summary.csv", index=False, encoding="utf-8-sig")
    pd.concat(logs, ignore_index=True).to_csv(OUT / "period_trade_log.csv", index=False, encoding="utf-8-sig")
    yearly = pd.DataFrame(annual)
    yearly.to_csv(OUT / "yearly_returns.csv", index=False, encoding="utf-8-sig")
    provenance["availability_audit"] = audit
    (OUT / "provenance.json").write_text(json.dumps(provenance, indent=2, ensure_ascii=False), encoding="utf-8")
    (OUT / "comparison_report.md").write_text(build_report(summary, portfolio, coverage, provenance, audit, yearly), encoding="utf-8")
    print(summary[["period", "strategy", "start", "end", "days", "trades", "CAGR", "MDD", "Sharpe", "Calmar"]].to_string(index=False))
    print("\nFive independent accounts:")
    print(pd.DataFrame([five_stats])[["start", "end", "days", "trades", "total_return", "CAGR", "MDD", "Sharpe"]].to_string(index=False))
    print(f"Five-account report: {(five_out / 'report.md').resolve()}")
    print("\nSix independent accounts:")
    print(pd.DataFrame([six_stats])[["start", "end", "days", "trades", "total_return", "CAGR", "MDD", "Sharpe"]].to_string(index=False))
    print(f"Six-account report: {(six_out / 'report.md').resolve()}")
    print(f"Six-strategy correlation analysis: {(correlation_out / 'report.md').resolve()}")
    print(f"Report: {(OUT / 'comparison_report.md').resolve()}")


def run_transfer_leverage_experiments():
    """Offline, fixed-rule vehicle experiments; never refresh or collect inputs.

    Keep the original portfolio outputs and frozen intraday selections intact.
    Missing market-native signals / execution vehicles are reported, not imputed.
    """
    out = HERE / "results" / "transfer_leverage_experiments"
    out.mkdir(parents=True, exist_ok=True)
    prior = HERE / "results" / "six_strategy_portfolio_2022_12_13"
    old_daily = pd.read_csv(prior / "daily_returns_by_strategy.csv", index_col=0, parse_dates=True)
    old_daily = old_daily.rename(columns={label: code for code, label in LABELS.items()})
    calendar = old_daily.index
    start, end = calendar.min(), calendar.max()
    inputs = [prior / "daily_returns_by_strategy.csv"]

    def read_daily(code):
        choices = [ROOT / f"jehyeong/intraday/data/{code}_daily.csv", ROOT / f"beomgu/data/etf/{code}.csv"]
        path = next((p for p in choices if p.exists()), None)
        if path is None:
            return None
        d = pd.read_csv(path)
        d["date"] = pd.to_datetime(d["date"])
        d = d.set_index("date").sort_index()
        if not d.index.is_unique or not d[["open", "close"]].gt(0).all().all():
            raise ValueError(f"Invalid ETF daily input: {path}")
        inputs.append(path)
        return d

    prices = {code: read_daily(code) for code in ("069500", "122630", "229200", "233740", "114800", "252670", "251340")}
    panels = {}
    for code, filename in (("ON", "jehyeong_proxy"), ("C3", "beomgu_c3_vt30"), ("SC", "sunny_c"), ("W40", "woohyun_top40_30")):
        path = OUT / f"{filename}_signal_panel.csv"
        panels[code] = pd.read_csv(path, index_col="date", parse_dates=True)
        inputs.append(path)
    log_path = OUT / "all_strategy_trade_log.csv"
    legacy_trades = pd.read_csv(log_path, parse_dates=["entry_date", "exit_date"])
    inputs.append(log_path)
    minute = {}
    for code in ("069500", "122630", "229200", "233740"):
        path = ROOT / f"jehyeong/intraday/data/minute/{code}_TOSS_1m.csv"
        if not path.exists():
            minute[code] = None
            continue
        d = pd.read_csv(path, usecols=["timestamp", "open", "high", "low", "close", "volume", "timestamp_semantics"])
        if not d["timestamp_semantics"].eq("bar_end").all():
            raise ValueError(f"Unverified minute timestamp convention: {path}")
        d["timestamp"] = pd.to_datetime(d["timestamp"])
        d = d.set_index("timestamp").sort_index()
        if not d.index.is_unique:
            raise ValueError(f"Duplicate minute timestamps: {path}")
        minute[code] = d
        inputs.append(path)

    def volatility(code):
        # At most T-1 information, even for overnight entries at T close.
        d = prices[code]
        return d.close.pct_change(fill_method=None).rolling(20, min_periods=20).std().shift(1) * np.sqrt(252)

    def mix(exposure):
        if not np.isfinite(exposure) or not 0 <= exposure <= 2:
            raise ValueError(f"Invalid target exposure: {exposure}")
        return (exposure, 0.0) if exposure <= 1 else (2 - exposure, exposure - 1)

    def target(size, date, asset, sigma_override=None):
        if size == "1X":
            return 1.0
        if size == "2X":
            return 2.0
        sigma = sigma_override.loc[date] if sigma_override is not None else volatility_cache[asset].loc[date]
        if not np.isfinite(sigma) or sigma <= 0:
            raise ValueError(f"VT30 lacks causal warmup: {asset} {date}")
        return float(np.clip(.30 / sigma, 0, 2))

    volatility_cache = {code: volatility(code) for code in ("069500", "229200")}
    daily_series, rows, ledgers, unavailable, execution_audit = {}, [], [], [], []
    native_panels = {}
    foreign_path = ROOT / "jehyeong/intraday/data/KOSDAQ_150_fut_foreign.csv"
    index_path = ROOT / "jehyeong/intraday/data/KOSDAQ_150_ohlcv.csv"
    native_foreign = native_index = None
    data_audit = {}
    if foreign_path.exists():
        f = pd.read_csv(foreign_path)
        fdate = pd.to_datetime(f.iloc[:, 0])
        # Human-specified fifth column, not an inferred investor column.
        fvalue = pd.to_numeric(f.iloc[:, 4].astype(str).str.replace(",", "", regex=False), errors="coerce")
        native_foreign = pd.Series(fvalue.to_numpy(), index=fdate, name="foreign_net").sort_index()
        if not native_foreign.index.is_unique:
            raise ValueError("Duplicate KOSDAQ futures foreign-flow dates")
        data_audit["foreign_fifth_column"] = str(f.columns[4])
        data_audit["foreign_missing_evaluation_days"] = int(native_foreign.reindex(calendar).isna().sum())
        if data_audit["foreign_missing_evaluation_days"]:
            raise ValueError("KOSDAQ foreign input missing on evaluation sessions")
        inputs.append(foreign_path)
    if index_path.exists():
        raw_index = pd.read_csv(index_path)
        ix = raw_index.drop_duplicates().copy()
        data_audit["identical_index_rows_dropped"] = len(raw_index) - len(ix)
        ix.columns = ix.columns.str.lower()
        ix["date"] = pd.to_datetime(ix["date"])
        native_index = ix.set_index("date").sort_index()
        if not native_index.index.is_unique:
            raise ValueError("Conflicting duplicate KOSDAQ index dates")
        check = native_index.reindex(calendar)[["open", "close"]]
        if not check.notna().all().all() or not check.gt(0).all().all():
            raise ValueError("KOSDAQ index open/close missing on evaluation sessions")
        inputs.append(index_path)

    def evaluate(name, strategy, market, signal_mode, size, orders, vehicle, note=""):
        orders = pd.DataFrame(orders)
        if orders.empty:
            orders = pd.DataFrame(columns=["entry_date", "exit_date", "net_return"])
        else:
            orders["entry_date"] = pd.to_datetime(orders["entry_date"])
            orders["exit_date"] = pd.to_datetime(orders["exit_date"])
        orders["variant"] = name
        daily, selected = account_returns(orders, calendar, start, end)
        daily_series[name] = daily
        rows.append({"variant": name, "family": strategy, "market": market, "signal_mode": signal_mode,
                     "size": size, "vehicle": vehicle, "note": note, **performance_metrics(daily, selected)})
        ledgers.append(selected)

    def missing(name, reason):
        unavailable.append({"variant": name, "status": "NOT_RUN", "reason": reason})

    def daily_order(date, exit_date, asset, levered, exposure, overnight=False):
        a, b = mix(exposure)
        ep = prices[asset].loc[date, "close" if overnight else "open"]
        xp = prices[asset].loc[exit_date, "open" if overnight else "close"]
        rk = xp / ep - 1
        rl = np.nan
        le, lx = np.nan, np.nan
        if b:
            le = prices[levered].loc[date, "close" if overnight else "open"]
            lx = prices[levered].loc[exit_date, "open" if overnight else "close"]
            rl = lx / le - 1
        gross = a * rk + (b * rl if b else 0)
        invested = a + b
        return {"entry_date": date, "exit_date": exit_date, "entry_time": "15:30_CLOSE" if overnight else "09:00_OPEN",
                "exit_time": "NEXT_09:00_OPEN" if overnight else "15:30_CLOSE", "w_1x": a, "w_2x": b,
                "target_exposure": exposure, "asset_1x": asset, "asset_2x": levered,
                "entry_price_1x": ep, "exit_price_1x": xp, "entry_price_2x": le, "exit_price_2x": lx,
                "gross_return": gross, "commission": invested * COMMISSION_BP / 10000,
                "slippage": invested * SLIPPAGE_BP / 10000, "net_return": gross - invested * ROUND_TRIP_COST}

    # ON and C3 preserve all original signal dates; cross-market C3 is only a
    # signal-transfer diagnostic, not a newly measured KOSDAQ discount signal.
    for family, condition in (("ON", panels["ON"].signal.astype(str).str.lower().eq("true")), ("C3", panels["C3"].C3.eq(1))):
        selected = panels[family].index[condition]
        for market, asset, lev in (("KOSPI", "069500", "122630"), ("KOSDAQ", "229200", "233740")):
            for size in ("1X", "2X", "VT30"):
                name = f"{family}_{market}_KOSPI_SIGNAL_{size}"
                if size != "1X" and prices[lev] is None:
                    missing(name, f"Missing daily OHLC: {lev}")
                    continue
                orders = []
                for date in selected[(selected >= start) & (selected <= end)]:
                    i = calendar.get_loc(date)
                    exit_date = calendar[i + 1] if family == "ON" and i + 1 < len(calendar) else date
                    if family == "ON" and (i + 1 == len(calendar)):
                        continue
                    # Retain C3's original index volatility on KOSPI. On KOSDAQ,
                    # ETF volatility is explicitly an execution sizing proxy.
                    sigma = panels["C3"].sig_hat if family == "C3" and market == "KOSPI" else None
                    w = target(size, date, asset, sigma)
                    orders.append(daily_order(date, exit_date, asset, lev, w, family == "ON"))
                note = "Same confirmed-open signal/fill assumption as original C3" if family == "C3" else "Actual ETF close-to-next-open returns"
                if family == "C3" and market == "KOSDAQ":
                    note += "; retained KOSPI discount signal; KOSDAQ ETF volatility for VT30; not native C3"
                evaluate(name, family, market, "KOSPI_SIGNAL", size, orders, "ACTUAL_ETF", note)

    # Native overnight signal: same US proxy filter, previous KOSDAQ futures
    # foreign net buying, trade KOSDAQ ETFs close -> actual next session open.
    if native_foreign is not None:
        p = panels["ON"].copy()
        full_q_calendar = prices["229200"].index
        prev_net = native_foreign.reindex(full_q_calendar).shift(1)
        p["kosdaq_foreign_previous"] = prev_net.reindex(p.index)
        uscols = ["QQQ_proxy_preus_log_1520", "SPY_proxy_preus_log_1520"]
        p["native_available"] = p[[*uscols, "kosdaq_foreign_previous"]].notna().all(axis=1)
        p["native_signal"] = p.native_available & p[uscols].gt(0).all(axis=1) & p.kosdaq_foreign_previous.gt(0)
        native_panels["ON"] = p[[*uscols, "kosdaq_foreign_previous", "native_available", "native_signal"]]
        for size in ("1X", "2X", "VT30"):
            orders = []
            for date in p.index[p.native_signal & p.index.to_series().between(start, end)]:
                i = calendar.get_loc(date)
                if i + 1 == len(calendar):
                    continue
                w = target(size, date, "229200")
                orders.append(daily_order(date, calendar[i + 1], "229200", "233740", w, True))
            evaluate(f"ON_KOSDAQ_NATIVE_{size}", "ON", "KOSDAQ", "NATIVE", size, orders, "ACTUAL_ETF",
                     "US proxy unchanged; previous KOSDAQ150 futures foreign net-buy signal")

    # Re-estimate the original C3 OLS on KOSDAQ150 index gaps; retain SPX as
    # explanatory variable and all original thresholds/windows.
    if native_index is not None:
        q = prices["229200"]
        ix = native_index.reindex(q.index)
        spx_path = ROOT / "beomgu/data/spx_daily.csv"
        spx = pd.read_csv(spx_path, index_col="date", parse_dates=True).sort_index().close.dropna()
        pos = np.searchsorted(spx.index.values, q.index.values, side="left") - 1
        if (pos < 0).any():
            raise ValueError("SPX history insufficient for native C3")
        us = pd.Series(np.log(spx.to_numpy()[pos]), index=q.index).diff()
        # Extract the existing rolling OLS only; no peer data collection runs.
        r3_path = ROOT / "beomgu/overnight_lab/topics/r3_d2.py"
        node = next(n for n in ast.parse(r3_path.read_text(encoding="utf-8")).body
                    if isinstance(n, ast.FunctionDef) and n.name == "rolling_z1")
        ns = {"pd": pd, "np": np}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), str(r3_path), "exec"), ns)
        gi = np.log(ix.open / ix.close.shift(1))
        z = ns["rolling_z1"](gi, us, 250, 200)
        mis = np.log(q.open / ix.open) - np.log(q.close / ix.close).shift(1).rolling(5).median()
        sig_hat = ix.close.pct_change(fill_method=None).rolling(20).std().shift(1) * np.sqrt(252)
        p = pd.DataFrame({"us": us, "index_gap": gi, "z": z, "mis": mis, "sig_hat": sig_hat})
        p["native_available"] = p[["z", "mis", "sig_hat"]].notna().all(axis=1)
        p["native_signal"] = p.native_available & p.z.ge(1.) & p.mis.le(0.)
        native_panels["C3"] = p
        if not p.native_available.reindex(calendar).fillna(False).all():
            raise ValueError("Native C3 features missing on evaluation sessions")
        inputs.extend([spx_path, r3_path])
        for size in ("1X", "2X", "VT30"):
            orders = [daily_order(date, date, "229200", "233740", target(size, date, "229200", sig_hat))
                      for date in p.index[p.native_signal & p.index.to_series().between(start, end)]]
            evaluate(f"C3_KOSDAQ_NATIVE_{size}", "C3", "KOSDAQ", "NATIVE", size, orders, "ACTUAL_ETF",
                     "KOSDAQ index gap residual z>=1 and KOSDAQ ETF relative discount; same-open-fill assumption retained")

    # W40 KOSPI retains the original price/flow signal and 30 positive-bar exit.
    # KOSDAQ recalculates ORB/VWAP on 229200, retaining only KOSPI futures flow.
    w40_orders = {}
    old_w40 = legacy_trades.loc[legacy_trades.strategy.eq("WOOHYUN_TOP40_30")]
    w40_orders["KOSPI"] = [{"date": r.entry_date, "entry": pd.Timestamp(str(r.entry_date.date()) + " " + r.entry_time),
                            "exit": pd.Timestamp(str(r.exit_date.date()) + " " + r.exit_time)} for r in old_w40.itertuples()
                           if start <= r.entry_date <= end]
    kosdaq_orders = []
    flow = panels["W40"]
    for date, day in minute["229200"].groupby(minute["229200"].index.normalize()):
        if date < start or date > end or date not in flow.index:
            continue
        if not bool(flow.loc[date, "available"]) or not bool(flow.loc[date, "flow_top40"]):
            continue
        day = day.loc[day.volume.gt(0)].copy()
        hm = day.index.strftime("%H:%M")
        opening = day.loc[(hm >= "09:01") & (hm <= "09:10")]
        if opening.empty:
            continue
        vwap = (((day.high + day.low + day.close) / 3) * day.volume).cumsum() / day.volume.cumsum()
        signals = np.flatnonzero((hm >= "09:11") & day.close.gt(opening.high.max()) & day.close.gt(vwap))
        if not len(signals):
            continue
        entry_i, exit_i = signals[0] + 1, signals[0] + 1 + HOLDING_BARS
        if exit_i >= len(day):
            continue
        kosdaq_orders.append({"date": date, "entry": day.index[entry_i], "exit": day.index[exit_i]})
    w40_orders["KOSDAQ"] = kosdaq_orders
    if native_foreign is not None:
        qcal = prices["229200"].index
        net = native_foreign.reindex(qcal)
        threshold = net.rolling(250, min_periods=250).quantile(.6).shift(1)
        p = pd.DataFrame({"previous_net": net.shift(1), "top40_threshold": threshold})
        p["native_available"] = p.notna().all(axis=1)
        p["native_signal"] = p.native_available & p.previous_net.ge(p.top40_threshold)
        native_panels["W40_NET_PROXY"] = p
        if not p.native_available.reindex(calendar).fillna(False).all():
            raise ValueError("Native W40 net-buy proxy lacks 250-session warmup")
        proxy_orders = []
        for date, day in minute["229200"].groupby(minute["229200"].index.normalize()):
            if date < start or date > end or not bool(p.loc[date, "native_signal"]):
                continue
            day = day.loc[day.volume.gt(0)]
            hm = day.index.strftime("%H:%M")
            opening = day.loc[(hm >= "09:01") & (hm <= "09:10")]
            if opening.empty:
                continue
            vwap = (((day.high + day.low + day.close) / 3) * day.volume).cumsum() / day.volume.cumsum()
            signals = np.flatnonzero((hm >= "09:11") & day.close.gt(opening.high.max()) & day.close.gt(vwap))
            if not len(signals):
                continue
            ei, xi = signals[0] + 1, signals[0] + 1 + HOLDING_BARS
            if xi < len(day):
                proxy_orders.append({"date": date, "entry": day.index[ei], "exit": day.index[xi]})
        w40_orders["KOSDAQ_NET_PROXY"] = proxy_orders
    jobs = [("KOSPI", "069500", "122630"), ("KOSDAQ", "229200", "233740")]
    if "KOSDAQ_NET_PROXY" in w40_orders:
        jobs.append(("KOSDAQ_NET_PROXY", "229200", "233740"))
    for job, asset, lev in jobs:
        market = "KOSDAQ" if job == "KOSDAQ_NET_PROXY" else job
        mode = "NET_PROXY" if job == "KOSDAQ_NET_PROXY" else "KOSPI_FLOW"
        # All sizes use the same eligible base orders, so vehicle missing bars
        # do not manufacture an advantage for one position-size variant.
        supported = ["1X", "2X", "VT30"] if minute[lev] is not None else ["1X"]
        valid = []
        for order in w40_orders[job]:
            codes = [asset] + ([lev] if len(supported) > 1 else [])
            okay = all(order["entry"] in minute[c].index and order["exit"] in minute[c].index
                       and minute[c].loc[order["entry"], "volume"] > 0
                       and minute[c].loc[order["exit"], "volume"] > 0 for c in codes)
            if okay:
                valid.append(order)
            else:
                execution_audit.append({"family": "W40", "market": market, "date": order["date"],
                                        "reason": "Exact positive-volume execution bar missing in an ETF leg"})
        for size in ("1X", "2X", "VT30"):
            name = f"W40_{market}_{mode}_{size}"
            if size not in supported:
                missing(name, f"Missing verified bar-end minute OHLCV: {lev}; daily prices cannot replace intraday fills")
                continue
            orders = []
            for order in valid:
                date, ep_t, xp_t = order["date"], order["entry"], order["exit"]
                a, b = mix(target(size, date, asset))
                ep, xp = minute[asset].loc[ep_t, "open"], minute[asset].loc[xp_t, "close"]
                le, lx = (minute[lev].loc[ep_t, "open"], minute[lev].loc[xp_t, "close"]) if b else (np.nan, np.nan)
                gross = a * (xp / ep - 1) + (b * (lx / le - 1) if b else 0)
                orders.append({"entry_date": date, "exit_date": date, "entry_bar_end": ep_t, "exit_bar_end": xp_t,
                               "entry_time_actual": ep_t - pd.Timedelta(minutes=1), "exit_time_actual": xp_t,
                               "w_1x": a, "w_2x": b, "asset_1x": asset, "asset_2x": lev,
                               "entry_price_1x": ep, "exit_price_1x": xp, "entry_price_2x": le, "exit_price_2x": lx,
                               "gross_return": gross, "commission": (a + b) * COMMISSION_BP / 10000,
                               "slippage": (a + b) * SLIPPAGE_BP / 10000, "net_return": gross - (a + b) * ROUND_TRIP_COST})
            evaluate(name, "W40", market, mode, size, orders, "ACTUAL_ETF_MINUTE",
                     "1x ETF ORB/VWAP; next positive-bar open, 30-positive-bar exit; " +
                     ("KOSDAQ futures net-buy quantile proxy, NOT normalized buy/sell flow" if mode == "NET_PROXY" else "original KOSPI normalized flow"))

    # SC is SHORT only: buy inverse ETFs, never buy the positive 2x ETF.
    sc_dates = panels["SC"].index[panels["SC"].pos_C.eq(-1)]
    for market, asset in (("KOSPI", "114800"), ("KOSDAQ", "251340")):
        orders = [daily_order(d, d, asset, None, 1) for d in sc_dates if start <= d <= end]
        evaluate(f"SC_{market}_KOSPI_OPTIONS_1X", "SC", market, "KOSPI_OPTIONS", "1X", orders, "ACTUAL_INVERSE_ETF",
                 "Original short signal; inverse ETF bought at open and sold at close; differs from index short")
        for size in ("2X", "VT30"):
            if market == "KOSPI" and prices["252670"] is not None:
                orders = [daily_order(d, d, asset, "252670", target(size, d, "069500"))
                          for d in sc_dates if start <= d <= end]
                evaluate(f"SC_{market}_KOSPI_OPTIONS_{size}", "SC", market, "KOSPI_OPTIONS", size, orders,
                         "ACTUAL_INVERSE_ETF", "Inverse 1x/2x mix; volatility measured on underlying 069500")
                continue
            reason = "Missing 252670 daily OHLC (short-direction 2x ETF)" if market == "KOSPI" else "No KOSDAQ -2x ETF in the permitted vehicle universe; margin futures/ETN require a separate model"
            missing(f"SC_{market}_KOSPI_OPTIONS_{size}", reason)
    for family in ("ON", "C3", "W40", "SC"):
        if (family == "ON" and native_foreign is not None) or (family == "C3" and native_index is not None):
            continue
        reason = {"ON": "KOSDAQ150 futures foreign daily net flow missing; existing KOSDAQ cash-market flow is a different signal",
                  "C3": "KOSDAQ150 index daily open/close with at least 250-session warmup missing; cannot reconstruct residual-gap/discount signal",
                  "W40": "Only net-buy available; normalized net/(buy+sell) requires daily buy/sell amounts; separate NET_PROXY tested",
                  "SC": "KOSDAQ150 options foreign put / individual call daily buy/sell amounts and 60-session warmup missing"}[family]
        for size in ("1X", "2X", "VT30"):
            missing(f"{family}_KOSDAQ_NATIVE_{size}", reason)

    summary = pd.DataFrame(rows)
    trades = pd.concat(ledgers, ignore_index=True)
    returns = pd.DataFrame(daily_series)
    years = []
    for name in returns:
        for year, ret in returns[name].groupby(returns.index.year):
            years.append({"variant": name, "year": year, **performance_metrics(ret)})
    summary.to_csv(out / "summary.csv", index=False, encoding="utf-8-sig")
    trades.to_csv(out / "trades.csv", index=False, encoding="utf-8-sig")
    returns.to_csv(out / "daily_returns.csv", index_label="date", encoding="utf-8-sig")
    returns.corr().to_csv(out / "correlations.csv", index_label="variant", encoding="utf-8-sig")
    pd.DataFrame(years).to_csv(out / "yearly.csv", index=False, encoding="utf-8-sig")
    for name, panel in native_panels.items():
        panel.to_csv(out / f"{name.lower()}_native_signal_panel.csv", index_label="date", encoding="utf-8-sig")
    pd.DataFrame(unavailable).to_csv(out / "not_run.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(execution_audit, columns=["family", "market", "date", "reason"]).to_csv(out / "execution_exclusions.csv", index=False, encoding="utf-8-sig")
    # Independent leg arithmetic and ledger-to-calendar reconciliation.
    for r in trades.itertuples():
        a, b = r.w_1x, r.w_2x
        expected = a * (r.exit_price_1x / r.entry_price_1x - 1)
        if b:
            expected += b * (r.exit_price_2x / r.entry_price_2x - 1)
        np.testing.assert_allclose(r.gross_return, expected, atol=1e-14)
        np.testing.assert_allclose(r.net_return, expected - (a + b) * .0005, atol=1e-14)
        assert 0 <= a <= 1 and 0 <= b <= 1 and a + b <= 1 + 1e-12
    for name in returns:
        s = trades.loc[trades.variant.eq(name)]
        expected = pd.Series(0., index=calendar)
        realized = s.groupby("exit_date").net_return.agg(lambda v: (1 + v).prod() - 1)
        expected.loc[realized.index] = realized
        np.testing.assert_allclose(returns[name], expected, atol=1e-14)
    ref = old_daily[["JEHYEONG_PROXY", "BEOMGU_C3_VT30", "WOOHYUN_TOP40_30", "SUNNY_C"]]
    ref_rows = []
    for n in ref:
        selected = legacy_trades.loc[legacy_trades.strategy.eq(n)
                                    & legacy_trades.entry_date.ge(start)
                                    & legacy_trades.exit_date.le(end)]
        ref_rows.append({"strategy": n, **performance_metrics(ref[n], selected)})
    pd.DataFrame(ref_rows).to_csv(out / "previous_reference_metrics.csv", index=False, encoding="utf-8-sig")
    expanded = ref.join(returns)
    expanded.corr().to_csv(out / "correlations_with_original.csv", index_label="variant", encoding="utf-8-sig")
    base_pairs = {"ON": ("ON_KOSPI_KOSPI_SIGNAL_1X", "ON_KOSDAQ_KOSPI_SIGNAL_1X"),
                  "C3": ("C3_KOSPI_KOSPI_SIGNAL_VT30", "C3_KOSDAQ_KOSPI_SIGNAL_VT30"),
                  "W40": ("W40_KOSPI_KOSPI_FLOW_1X", "W40_KOSDAQ_KOSPI_FLOW_1X"),
                  "SC": ("SC_KOSPI_KOSPI_OPTIONS_1X", "SC_KOSDAQ_KOSPI_OPTIONS_1X")}
    market_pairs = []
    for family, (left, right) in base_pairs.items():
        active = returns[left].ne(0) & returns[right].ne(0)
        market_pairs.append({"family": family, "kospi": left, "kosdaq": right,
                             "daily_correlation": returns[left].corr(returns[right]), "both_active_days": int(active.sum()),
                             "both_active_correlation": returns.loc[active, left].corr(returns.loc[active, right])})
    pd.DataFrame(market_pairs).to_csv(out / "market_transfer_correlations.csv", index=False, encoding="utf-8-sig")
    manifest = {"start": str(start.date()), "end": str(end.date()), "days": len(calendar),
                "commission_round_trip_bp": 3, "slippage_round_trip_bp": 2, "missing_data_never_zero_imputed": True,
                "vt30": "clip(0.30 / past-20-session close-return annualized SD known by T-1, 0, 2)",
                "c3_kospi_vt30": "original KOSPI200 index SD retained", "data_collection": False,
                "source": "previous fixed signal panels / ledger; W40 KOSDAQ price signal recalculated", 
                "inputs": {p.relative_to(ROOT).as_posix(): file_hash(p) for p in dict.fromkeys(inputs)},
                "code_sha256": file_hash(Path(__file__)), "completed_variants": len(summary), "unrun_variants": len(unavailable),
                "new_data_audit": data_audit,
                "checks": {"actual_etf_leg_arithmetic": "PASS", "exit_date_ledger_reconciliation": "PASS", "unborrowed_cash_weights": "PASS"}}
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    plot_transfer_variants(out)
    np.testing.assert_allclose(returns["ON_KOSPI_KOSPI_SIGNAL_1X"], ref["JEHYEONG_PROXY"], atol=1e-12)
    write_transfer_report(out)
    print(summary[["variant", "trades", "CAGR", "MDD", "Sharpe", "Calmar"]].to_string(index=False))
    print(f"Completed {len(summary)} variants; NOT_RUN {len(unavailable)}; report: {out / 'report.md'}")
    return out


def plot_transfer_variants(out):
    rows = pd.read_csv(out / "summary.csv").to_dict("records")
    returns = pd.read_csv(out / "daily_returns.csv", index_col="date", parse_dates=True)
    fig, axes = plt.subplots(4, 3, figsize=(18, 15), sharex=True)
    plot_start = returns.index.min() - pd.Timedelta(days=1)
    for i, family in enumerate(("ON", "C3", "W40", "SC")):
        for j, market in enumerate(("KOSPI", "KOSDAQ", "KOSDAQ")):
            ax = axes[i, j]
            plotted = False
            for row in rows:
                native = row["signal_mode"] in ("NATIVE", "NET_PROXY")
                if row["family"] == family and row["market"] == market and (j == 2) == native:
                    eq = (1 + returns[row["variant"]]).cumprod()
                    ax.plot([plot_start, *eq.index], [0., *((eq - 1) * 100)], label=row["size"])
                    plotted = True
            signal_caption = ("foreign net-buy proxy" if family == "W40" else "native signal") if j == 2 else "KOSPI signal / flow"
            ax.set_title(f"{family} / {market} / " + signal_caption)
            ax.set_ylabel("Cumulative net P&L (%)")
            ax.axhline(0, color="grey", lw=.5)
            ax.grid(alpha=.2)
            if plotted:
                ax.legend()
            else:
                ax.text(.5, .5, "NOT RUN: native options flow missing", ha="center", transform=ax.transAxes)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out / "pnl_comparison.png", dpi=150)
    plt.close(fig)


POOL_SEARCH_CS = r'''
using System;
using System.IO;
using System.Collections.Generic;
using System.Globalization;
public class PoolSearch {
 class Row {
  public ulong Mask; public int K; public double CAGR, DD, Calmar, Abs, Signed, Weekly;
 }
 static int N,T,G; static int[][] Groups; static double[][] E,Work;
 static double[,] Corr,Weekly; static bool[] Positive; static int[] Picked;
 static List<Row>[] Top,Both; static Row[] Low,LowPositive,LowAllPositive;
 static long[] Counts; static long Leaves; static ulong Mandatory;
 static string Progress;
 static Row Make(ulong mask,int k,double ca,double dd,double ab,double si,double we) {
  return new Row {Mask=mask,K=k,CAGR=ca,DD=-dd,Calmar=dd>0?ca/dd:double.PositiveInfinity,
   Abs=k>1?ab/(k*(k-1)/2.0):double.NaN,Signed=k>1?si/(k*(k-1)/2.0):double.NaN,
   Weekly=k>1?we/(k*(k-1)/2.0):double.NaN};
 }
 static void Push(List<Row> list,Row row,int limit) {
  if(list.Count==limit && row.Calmar<=list[list.Count-1].Calmar) return;
  int at=0;while(at<list.Count && list[at].Calmar>=row.Calmar) at++;
  list.Insert(at,row); if(list.Count>limit)list.RemoveAt(limit);
 }
 static void Visit(int depth,int k,ulong mask,double[] sum,double ab,double si,double we,bool allPositive) {
  if(depth==G) {
   if(k==0)return;
   Leaves++;Counts[k]++;
   double ca=Math.Pow(sum[T-1]/k,252.0/T)-1.0;
   double peak=k,dd=0;
   for(int t=0;t<T;t++) {double v=sum[t];if(v>peak)peak=v;double d=1.0-v/peak;if(d>dd)dd=d;}
   Row row=Make(mask,k,ca,dd,ab,si,we);
   Push(Top[k],row,20);
   if((mask & Mandatory)==Mandatory)Push(Both[k],row,10);
   if(k>=2) {
    if(Low[k]==null || row.Abs<Low[k].Abs)Low[k]=row;
    if(ca>0 && (LowPositive[k]==null || row.Abs<LowPositive[k].Abs))LowPositive[k]=row;
    if(allPositive && (LowAllPositive[k]==null || row.Abs<LowAllPositive[k].Abs))LowAllPositive[k]=row;
   }
   if(Leaves%500000==0) {
    string msg="evaluated="+Leaves.ToString(CultureInfo.InvariantCulture);
    File.WriteAllText(Progress,msg);Console.WriteLine(msg);
   }
   return;
  }
  Visit(depth+1,k,mask,sum,ab,si,we,allPositive);
  foreach(int id in Groups[depth]) {
   double a=ab,s=si,w=we;
   for(int q=0;q<k;q++) {double c=Corr[id,Picked[q]];a+=Math.Abs(c);s+=c;w+=Math.Abs(Weekly[id,Picked[q]]);}
   double[] next=Work[depth]; double[] add=E[id];
   for(int t=0;t<T;t++)next[t]=sum[t]+add[t];
   Picked[k]=id;
   Visit(depth+1,k+1,mask|(1UL<<id),next,a,s,w,allPositive && Positive[id]);
  }
 }
 static string F(double v) {return v.ToString("R",CultureInfo.InvariantCulture);}
 static void Write(StreamWriter writer,string kind,Row r) {
  if(r==null)return;
  writer.WriteLine(kind+","+r.Mask+","+r.K+","+F(r.CAGR)+","+F(r.DD)+","+F(r.Calmar)+","+F(r.Abs)+","+F(r.Signed)+","+F(r.Weekly));
 }
 public static void Main(string[] args) {
  Progress=args[3];
  using(BinaryReader b=new BinaryReader(File.OpenRead(args[0]))) {
   N=b.ReadInt32();T=b.ReadInt32();G=b.ReadInt32();Mandatory=b.ReadUInt64();
   Groups=new int[G][];
   for(int g=0;g<G;g++){Groups[g]=new int[b.ReadInt32()];for(int j=0;j<Groups[g].Length;j++)Groups[g][j]=b.ReadInt32();}
   E=new double[N][];for(int i=0;i<N;i++){E[i]=new double[T];for(int t=0;t<T;t++)E[i][t]=b.ReadDouble();}
   Corr=new double[N,N];Weekly=new double[N,N];
   for(int i=0;i<N;i++)for(int j=0;j<N;j++)Corr[i,j]=b.ReadDouble();
   for(int i=0;i<N;i++)for(int j=0;j<N;j++)Weekly[i,j]=b.ReadDouble();
   Positive=new bool[N];for(int i=0;i<N;i++)Positive[i]=b.ReadInt32()!=0;
  }
  Work=new double[G][];for(int i=0;i<G;i++)Work[i]=new double[T];
  Picked=new int[G];Counts=new long[G+1];Top=new List<Row>[G+1];Both=new List<Row>[G+1];
  Low=new Row[G+1];LowPositive=new Row[G+1];LowAllPositive=new Row[G+1];
  for(int i=0;i<=G;i++){Top[i]=new List<Row>();Both[i]=new List<Row>();}
  Visit(0,0,0UL,new double[T],0,0,0,true);
  using(StreamWriter w=new StreamWriter(args[1])) {
   w.WriteLine("kind,mask,accounts,CAGR,MDD,Calmar,mean_abs_corr,mean_signed_corr,mean_abs_weekly_corr");
   for(int k=1;k<=G;k++) {
    foreach(Row r in Top[k])Write(w,"TOP_CALMAR",r);
    foreach(Row r in Both[k])Write(w,"TOP_CALMAR_WITH_KP_KD",r);
    Write(w,"MIN_ABS_CORR",Low[k]);Write(w,"MIN_ABS_CORR_POSITIVE_POOL",LowPositive[k]);
    Write(w,"MIN_ABS_CORR_ALL_POSITIVE",LowAllPositive[k]);
   }
  }
  using(StreamWriter w=new StreamWriter(args[2])) {
   w.WriteLine("accounts,combinations");for(int k=1;k<=G;k++)w.WriteLine(k+","+Counts[k]);
  }
  File.WriteAllText(Progress,"complete="+Leaves);Console.WriteLine("complete="+Leaves);
 }
}
'''


def research_table(frame):
    def cell(v):
        return str(v).replace("|", "\\|").replace("\n", " ")
    return "\n".join(["| " + " | ".join(map(cell, frame.columns)) + " |",
                     "| " + " | ".join("---" for _ in frame.columns) + " |",
                     *["| " + " | ".join(map(cell, row)) + " |" for row in frame.itertuples(index=False, name=None)]])


def write_transfer_report(out):
    summary = pd.read_csv(out / "summary.csv")
    unavailable = pd.read_csv(out / "not_run.csv")
    reference = pd.read_csv(out / "previous_reference_metrics.csv")
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    new = summary.loc[summary.signal_mode.isin(["NATIVE", "NET_PROXY"]) |
                      (summary.family.eq("SC") & summary.market.eq("KOSPI") & summary["size"].ne("1X")) |
                      (summary.family.eq("W40") & summary.market.eq("KOSDAQ") & summary["size"].ne("1X"))]
    lines = ["# 코스닥 이식·레버리지 변형 검증",
             f"평가 {manifest['start']}~{manifest['end']}, {manifest['days']}거래일. 완료 {len(summary)}개, 미실행 {len(unavailable)}개. 추가 수집은 실행하지 않았다.",
             "## 새 입력으로 완료한 변형", metrics_markdown(new, "variant"),
             "ON NATIVE는 미국 proxy 조건을 유지하고 전일 외국인 순매수를 코스닥150 선물로 교체했다. C3 NATIVE는 코스닥150 지수 갭의 직전 250일 OLS 잔차 z≥1와 229200 시가 할인≤0을 함께 확인한다. SPX 설명변수와 임계값은 원형을 유지했다.",
             "W40 NET_PROXY는 외국인 순매수값의 직전 250일 상위 40%다. 기존 순매수/(매수+매도) 정규화 수급과 다르며, 추후 매수·매도 금액을 확보하면 원형 이식을 별도로 검증한다. 가격 신호는 코스닥 1배 ETF의 ORB/VWAP, 다음 거래량>0 분봉 open 진입, 30개 유효 봉 뒤 close 청산이다.",
             "233740 분봉으로 W40 코스닥 레버리지 체결을, 252670 일봉으로 SC 코스피 인버스2X와 혼합을 실행했다. 실제 해당 ETF 가격을 사용하며 1배 손익을 단순 두 배로 만들지 않는다.",
             "## 전체 성과", metrics_markdown(summary, "variant"), "![누적 순손익](pnl_comparison.png)",
             "## 계산 규칙",
             "각 후보는 독립 계좌이며 미거래일은 0이다. 진입 ETF 비중×수수료 왕복 3bp+슬리피지 왕복 2bp를 차감하는 기존 공통 비교 방식을 유지한다. 매도 시 바뀐 평가금액 기준으로 비용을 다시 계산하는 방식과는 소폭 다르다. 분배금 현금 수령·재투자와 정수 주식 수 제약은 별도 반영하지 않았다.",
             "1X는 1배 ETF 전액, 2X는 방향이 같은 2배 ETF 전액. VT30은 전일까지의 20일 종가수익률 SD×√252로 σ를 계산하고 w=clip(0.30/σ,0,2). w≤1은 1배 ETF w+현금 1−w, w>1은 1배 ETF 2−w+2배 ETF w−1이다. C3 코스피와 코스닥 NATIVE는 각각 지수 변동성, 나머지는 거래 대상 1배 ETF 변동성이다. 비중은 진입 시 고정하고 청산까지 유지한다.",
             "분봉은 모두 끝시각 라벨이다. 다음 봉 open의 실제 시작 시각은 해당 라벨보다 1분 앞이다. 원형 W40의 30유효봉은 30시계분과 다를 수 있다. 여러 ETF는 같은 분봉의 open/close를 사용하며 체결량>0을 확인한다.",
             "## 입력 점검", research_table(pd.DataFrame([manifest['new_data_audit']])),
             "외국인 순매수는 사람이 지정한 파일의 5번째 foreign 열을 사용했다. 모든 입력 날짜를 오름차순으로 정렬했다. 코스닥 지수의 2024-10-08 완전 동일 중복행 1개는 계산에서만 제거하고 원본은 보존했다.",
             "## 미실행", research_table(unavailable),
             "필요한 추가 자료는 W40 원형용 코스닥150 선물 외국인 매수·매도 금액, SC 고유 신호용 코스닥150 옵션 외국인 풋·개인 콜 매수·매도 금액이다. 코스닥 2배 숏은 이번 허용 ETF 목록으로 구현할 차량이 없다.",
             "## 기존 전략 참조", metrics_markdown(reference),
             "SC는 과거 지수 숏 프록시와 달리 실제 인버스 ETF를 거래해 성과가 달라진다. C3 등 다른 통제군은 같은 신호를 유지하되 가능한 경우 TOSS 가격으로 재평가했다.",
             "C3는 확정된 시가 신호로 같은 시가에 체결하는 원형 가정을 그대로 갖고 있다. 이 결과는 다음 분봉 진입이나 장전 예상체결가의 실행 검증을 마친 성과가 아니다. 전체 표본에서 조합을 고르는 행위는 표본 외 검증이 아니며, 일별 MDD는 장중 최악의 낙폭까지 포함하지 않는다.",
             "## 조합 선정", "[전체 후보의 조합 선정 보고서](pool_selection/report.md)",
             "## 재실행", "`python jehyeong/overnight/compare_committed_strategies.py --transfer-leverage`", 
             "`python jehyeong/overnight/compare_committed_strategies.py --select-alpha-pool`"]
    (out / "report.md").write_text("\n\n".join(lines), encoding="utf-8")


def plot_selected_pool(out):
    """Plot each distinct portfolio once, including the initial cash baseline."""
    curves = pd.read_csv(out / "selected_equity.csv", index_col="date", parse_dates=True)
    winners = pd.read_csv(out / "selected_portfolios.csv")
    captions = {"MAX_CALMAR": "Highest Calmar portfolio",
                "MIN_ABS_CORR": "Lowest daily correlation portfolio",
                "MIN_ABS_CORR_6_POSITIVE": "Lowest correlation with 6 positive-CAGR accounts"}
    fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True)
    anchor = curves.index.min() - pd.Timedelta(days=1)
    seen = set()
    for row in winners.itertuples():
        if row.mask in seen:
            continue
        seen.add(row.mask)
        total = curves[row.selection]
        caption = captions.get(row.selection, row.selection)
        axes[0].plot([anchor, *total.index], [0., *((total - 1) * 100)], label=caption)
        peaks = np.maximum.accumulate(np.r_[1., total.to_numpy()])[1:]
        axes[1].plot([anchor, *total.index], [0., *((total.to_numpy() / peaks - 1) * 100)], label=caption)
    axes[0].set_ylabel("Cumulative net P&L (%)")
    axes[1].set_ylabel("Drawdown (%)")
    for ax in axes:
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "selected_pnl_mdd.png", dpi=150)
    plt.close(fig)


def write_alpha_pool_report(out, summary, ranked, winners, counts, experiment_out):
    candidate_returns = pd.read_csv(out / "candidate_daily_returns.csv", index_col="date", parse_dates=True)
    weekly = (1 + candidate_returns).resample("W-FRI").prod() - 1
    diagnostics = []
    for row in winners.itertuples():
        selected = row.members.split(";")
        for i, left in enumerate(selected):
            for right in selected[:i]:
                active = candidate_returns[left].ne(0) & candidate_returns[right].ne(0)
                diagnostics.append({"selection": row.selection, "left": left, "right": right,
                                    "daily_corr": candidate_returns[left].corr(candidate_returns[right]),
                                    "weekly_corr": weekly[left].corr(weekly[right]),
                                    "both_active_days": int(active.sum()),
                                    "both_active_corr": candidate_returns.loc[active, left].corr(candidate_returns.loc[active, right])})
    diagnostics = pd.DataFrame(diagnostics)
    diagnostics.to_csv(out / "selected_pair_diagnostics.csv", index=False, encoding="utf-8-sig")
    def display(frame):
        cols = ["selection", "accounts", "CAGR", "MDD", "Sharpe", "Calmar", "mean_abs_corr", "mean_abs_weekly_corr"]
        d = frame[cols].copy()
        for col in ("CAGR", "MDD"):
            d[col] = d[col].map(lambda v: f"{v:.2%}")
        for col in ("Sharpe", "Calmar", "mean_abs_corr", "mean_abs_weekly_corr"):
            d[col] = d[col].map(lambda v: f"{v:.4f}")
        return research_table(d)
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    lines = ["# 레버리지·코스닥 이식 전략의 조합 선정",
             f"평가 {manifest['start']}~{manifest['end']}, {manifest['days']}거래일. 신규 데이터가 더 최근까지 있어도 기존 6계좌 비교와 같은 공통 기간을 유지했다.",
             f"후보 {len(summary)}개, 신호 그룹 {manifest['group_count']}개. 비어 있지 않은 조합 {manifest['exhaustive_nonempty_portfolios']:,}개를 전수 탐색했고, 최종 조합 선정은 2계좌 이상 {manifest['exhaustive_portfolios_at_least_two']:,}개 중에서 했다. 같은 family/market/signal_mode의 1X·2X·VT30 중 하나만 선택한다. 서로 다른 시장 또는 서로 다른 수급/신호 정의는 별도 그룹이다. KP·KD는 선택 가능한 후보이며 필수는 아니다.",
             "각 선택 계좌에 동일한 초기자본을 배정하고 계좌별 복리 자산을 합산한다. 계좌 사이 리밸런싱·자금 이동은 없다. 통합 일별 수익은 합산 자산의 전일 대비 변화다. 평균 일별 수익률을 복리로 계산하지 않는다. 최초 자산을 고점에 포함해 MDD를 계산하고 CAGR=(최종자산/초기자산)^(252/919)−1, Calmar=CAGR/|MDD|, Sharpe=일별 평균/표준편차×√252를 사용한다.",
             "비용은 모든 후보에서 수수료 왕복 3bp+슬리피지 왕복 2bp다. ETF 변형은 기존 비교의 진입비중×왕복 5bp 차감, KP·KD는 기존 주문별 매수/매도 비용 모델이다. 거래하지 않는 날은 계좌 수익 0이다. W40 기존 수급 버전은 자료 준비 전 현금 기간이 있으며 새 코스닥 순매수 버전은 2015년 수급으로 번인한다.",
             "상관 최소 목적은 전략 쌍의 일별 순수익률 Pearson 상관 절대값의 평균이다. 음의 상관은 헤지에 유용할 수 있지만 절대값이 크면 독립적이지 않으므로 평균 부호 상관도 함께 저장했다. 주별 상관과 모든 개별 전략 CAGR가 양수인 경우도 별도 보고한다. 평균 절대상관의 무제약 최솟값은 두 전략 쌍에서 얻어지므로, 6계좌 및 계좌 수별 결과도 함께 제시한다.",
             "## 선정 결과", display(winners)]
    for row in winners.itertuples():
        lines += [f"### {row.selection}", "\n".join(f"- {name}" for name in row.members.split(";")),
                  f"각 계좌 최초 비중 {1/row.accounts:.2%}. 운용 중 비중은 각 계좌의 손익에 따라 달라진다."]
    lines += ["![선정 조합 손익과 낙폭](selected_pnl_mdd.png)",
              "## 선정 조합의 거래일·주별 상관", research_table(diagnostics.loc[diagnostics.selection.isin(["MAX_CALMAR", "MIN_ABS_CORR"])]),
              "최저 일별 상관 조합도 두 전략 모두 손익이 발생한 날은 21일이며 그 조건부 상관은 약 0.197, 주별 상관은 약 0.159다. 전체 일별 상관이 거의 0이라는 이유만으로 독립 알파가 입증된 것은 아니다. 해당 조합의 2026년 평가기간 수익이 약 79.30%로 전체 성과에 크게 기여했다.",
              "## 계좌 수별 최상위 조합"]
    byk = ranked.loc[ranked.kind.eq("TOP_CALMAR") & ranked.accounts.ge(2)].sort_values("Calmar", ascending=False).groupby("accounts", sort=True).head(1).sort_values("accounts").copy()
    byk["selection"] = byk.accounts.map(lambda k: f"{k}계좌 최대 Calmar")
    lines.append(display(byk))
    lowk = ranked.loc[ranked.kind.eq("MIN_ABS_CORR_ALL_POSITIVE")].sort_values("accounts").copy()
    lowk["selection"] = lowk.accounts.map(lambda k: f"{k}계좌 양수 전략 상관 최소")
    lines += [display(lowk), "각 조합의 전체 구성은 ranked_candidates.csv에 있다.",
              "## 개별 후보 성과와 정의"]
    d = summary[["short_code", "variant", "group", "CAGR", "MDD", "Sharpe", "Calmar"]].copy()
    for c in ("CAGR", "MDD"):
        d[c] = d[c].map(lambda v: f"{v:.2%}")
    lines += [research_table(d), "![전체 후보 상관](all_strategy_correlations.png)",
              "KP는 기존 코스피 intraday final(롱 전용, 최대 2배), KD는 기존 코스닥 intraday final(양방향, 최대 1배)이다. 원래 frozen 선택과 일별 수익률을 그대로 사용했다. 새 KD 레버리지 변형은 이번 후보에 추가로 만들지 않았다.",
              "## 새 데이터 처리와 미완료 버전",
              "코스닥 선물 파일의 5번째 foreign 열을 그대로 외국인 순매수로 사용했다. 원본은 내림차순이므로 날짜를 오름차순으로 정렬하고 실제 직전 KRX 거래일로 lag했다. 코스닥150 지수의 2024-10-08 중복 두 행은 모든 값이 같아 계산에서 하나만 사용했다. 원본 CSV는 변경하지 않았다.",
              "ON NATIVE는 미국 proxy 조건을 유지하고 수급을 전일 코스닥150 선물 순매수>0으로 교체했다. C3 NATIVE는 코스닥150 지수 시가 갭을 SPX 미국 종가 누적수익으로 설명하는 직전 250일 OLS(최소 200일), 잔차 z≥1 및 229200 상대 시가 할인≤0 규칙을 사용한다. VT30은 native 지수 변동성을 사용한다.",
              "W40 NET_PROXY는 외국인 순매수값의 직전 250일 60% 분위수 이상 조건이다. 원형의 순매수/(매수+매도) 비율과 다르며 원형 이식으로 부르지 않는다. W40 가격 신호는 229200 ORB/VWAP, 다음 거래량>0 분봉 open 진입, 원형처럼 30개의 유효 봉 뒤 close 청산이다. 레버리지는 가격 신호를 바꾸지 않고 실제 233740 체결가로 적용한다.",
              research_table(pd.read_csv(experiment_out / "not_run.csv")),
              "아직 필요한 것은 W40 원형용 외국인 매수·매도 금액, SC 코스닥 고유 신호용 옵션 콜·풋 투자자별 매수·매도 금액이다. SC 코스닥 2배 숏은 허용 ETF 목록에서 구현할 차량이 없어 제외했다.",
              "## 해석의 한계",
              "선정은 전체 과거 표본을 관측한 뒤 800만 개 이상 조합을 비교한 사후 결과다. 새로운 표본 외 성과가 아니다. selected_yearly.csv와 주별 상관으로 기간 의존성을 확인해야 한다. 낮은 전체 일별 상관은 현금일로 희석될 수 있다. 개별 전략의 양의 CAGR 역시 미래 기대수익을 보장하지 않는다.",
              "C3는 확정된 당일 지수/ETF 시가로 신호를 계산하고 같은 시가에 체결하는 원형의 시간 가정을 보존했다. C3가 포함된 최고 Calmar 조합은 실거래 가능한 진입 지연 검증을 마친 결과가 아니다. 다음 분봉 진입 등으로 재평가하면 조합 순위가 달라질 수 있다. 일별 MDD는 장중 및 야간 미실현 최악의 낙폭까지 측정한 값이 아니다.",
              "## 검증",
              "계좌 수별 탐색 개수는 생성함수 ∏(1+그룹 내 후보 수×x)의 계수와 정확히 일치했다. 저장된 모든 상위/최저상관 조합의 CAGR·MDD·Calmar·상관을 Python에서 계좌별 자산 합산으로 독립 재계산했다. 최소상관 결과는 허용되는 모든 두 전략 쌍을 별도로 계산해 확인했다.",
              "## 실행", "`python jehyeong/overnight/compare_committed_strategies.py --transfer-leverage` 후 `python jehyeong/overnight/compare_committed_strategies.py --select-alpha-pool`."]
    (out / "report.md").write_text("\n\n".join(lines), encoding="utf-8")


def select_alpha_pool(experiment_out, *, write_report=True):
    """Exhaustive grouped account subsets; never compound averaged returns."""
    import struct
    out = Path(experiment_out) / "pool_selection"
    out.mkdir(parents=True, exist_ok=True)
    returns = pd.read_csv(Path(experiment_out) / "daily_returns.csv", index_col="date", parse_dates=True)
    summary = pd.read_csv(Path(experiment_out) / "summary.csv")
    old_path = HERE / "results/six_strategy_portfolio_2022_12_13/daily_returns_by_strategy.csv"
    old = pd.read_csv(old_path, index_col="date", parse_dates=True)
    override_path = Path(experiment_out) / "intraday_daily_returns.csv"
    override = pd.read_csv(override_path, index_col="date", parse_dates=True) if override_path.exists() else None
    for name, label in (("KP", "jehyeong: intraday final KOSPI200"), ("KD", "jehyeong: intraday final KOSDAQ150")):
        returns[name] = (override[name] if override is not None else old[label]).reindex(returns.index)
        summary = pd.concat([summary, pd.DataFrame([{"variant": name, "family": name, "market": name,
                          "signal_mode": "FROZEN_FINAL", "size": "FINAL", **performance_metrics(returns[name])}])], ignore_index=True)
    if not np.isfinite(returns.to_numpy()).all():
        raise ValueError("Candidate returns missing on common calendar")
    names = returns.columns.tolist()
    summary = summary.set_index("variant").loc[names].reset_index()
    summary["group"] = summary[["family", "market", "signal_mode"]].agg("/".join, axis=1)
    summary["candidate_id"] = np.arange(len(names))
    summary["short_code"] = [f"S{i+1:02}" for i in range(len(names)-2)] + ["KP", "KD"]
    groups = [frame.candidate_id.tolist() for _, frame in summary.groupby("group", sort=False)]
    equity = (1 + returns).cumprod()
    daily_corr = returns.corr()
    weekly_returns = (1 + returns).resample("W-FRI").prod() - 1
    weekly_corr = weekly_returns.corr()
    if not np.isfinite(daily_corr.to_numpy()).all():
        raise ValueError("Zero-variance candidate makes correlations undefined")
    returns.to_csv(out / "candidate_daily_returns.csv", index_label="date", encoding="utf-8-sig")
    equity.to_csv(out / "candidate_equity.csv", index_label="date", encoding="utf-8-sig")
    summary.to_csv(out / "candidate_summary.csv", index=False, encoding="utf-8-sig")
    daily_corr.to_csv(out / "daily_correlations.csv", index_label="variant", encoding="utf-8-sig")
    weekly_corr.to_csv(out / "weekly_correlations.csv", index_label="variant", encoding="utf-8-sig")
    support = out / "support"
    support.mkdir(exist_ok=True)
    binary = support / "search_input.bin"
    with binary.open("wb") as f:
        f.write(struct.pack("<iiiQ", len(names), len(returns), len(groups), (1 << names.index("KP")) | (1 << names.index("KD"))))
        for group in groups:
            f.write(struct.pack("<i", len(group)))
            f.write(struct.pack("<" + "i" * len(group), *group))
        f.write(equity.to_numpy(dtype="<f8").T.copy(order="C").tobytes())
        f.write(daily_corr.to_numpy(dtype="<f8").tobytes())
        f.write(weekly_corr.to_numpy(dtype="<f8").tobytes())
        f.write(np.asarray(summary.CAGR.gt(0), dtype="<i4").tobytes())
    source, exe = support / "pool_search.cs", support / "pool_search.exe"
    source.write_text(POOL_SEARCH_CS, encoding="utf-8")
    compiler = Path("C:/Windows/Microsoft.NET/Framework64/v4.0.30319/csc.exe")
    subprocess.run([str(compiler), "/nologo", "/optimize+", "/platform:x64", f"/out:{exe}", str(source)], check=True)
    raw_path, counts_path = out / "search_candidates.csv", out / "search_counts.csv"
    print(f"Exhaustive search: {len(names)} candidates, {len(groups)} groups, "
          f"{int(np.prod([len(g)+1 for g in groups]))-1:,} nonempty portfolios", flush=True)
    subprocess.run([str(exe), str(binary), str(raw_path), str(counts_path), str(support / "progress.txt")], check=True)
    raw = pd.read_csv(raw_path)

    def members(mask):
        return [n for i, n in enumerate(names) if int(mask) & (1 << i)]

    def metrics_for(mask):
        selected = members(mask)
        total = equity[selected].mean(axis=1)
        daily = total.pct_change(fill_method=None)
        daily.iloc[0] = total.iloc[0] - 1.
        pairs = np.triu_indices(len(selected), 1)
        cm = daily_corr.loc[selected, selected].to_numpy()
        wm = weekly_corr.loc[selected, selected].to_numpy()
        return {**performance_metrics(daily), "accounts": len(selected), "members": ";".join(selected),
                "mean_abs_corr": float(np.abs(cm[pairs]).mean()) if len(selected)>1 else np.nan,
                "mean_signed_corr": float(cm[pairs].mean()) if len(selected)>1 else np.nan,
                "mean_abs_weekly_corr": float(np.abs(wm[pairs]).mean()) if len(selected)>1 else np.nan}, daily, total

    checked = []
    for row in raw.itertuples():
        metrics, _, _ = metrics_for(row.mask)
        for field in ("CAGR", "MDD", "Calmar", "mean_abs_corr", "mean_signed_corr", "mean_abs_weekly_corr"):
            if not np.isfinite(metrics[field]) and not np.isfinite(getattr(row, field)):
                continue
            np.testing.assert_allclose(metrics[field], getattr(row, field), rtol=1e-9, atol=1e-10)
        checked.append({"kind": row.kind, "mask": int(row.mask), **metrics})
    ranked = pd.DataFrame(checked)
    ranked.to_csv(out / "ranked_candidates.csv", index=False, encoding="utf-8-sig")
    best = ranked.loc[ranked.kind.eq("TOP_CALMAR") & ranked.accounts.ge(2)].sort_values("Calmar", ascending=False).iloc[0]
    low = ranked.loc[ranked.kind.eq("MIN_ABS_CORR")].sort_values("mean_abs_corr").iloc[0]
    low_profitable = ranked.loc[ranked.kind.eq("MIN_ABS_CORR_ALL_POSITIVE")].sort_values("mean_abs_corr").iloc[0]
    both = ranked.loc[ranked.kind.eq("TOP_CALMAR_WITH_KP_KD")].sort_values("Calmar", ascending=False).iloc[0]
    winners = [("MAX_CALMAR", best), ("MIN_ABS_CORR", low), ("MIN_ABS_CORR_ALL_POSITIVE", low_profitable),
               ("MAX_CALMAR_WITH_KP_KD", both)]
    six = ranked.loc[ranked.kind.eq("MIN_ABS_CORR_ALL_POSITIVE") & ranked.accounts.eq(6)]
    if len(six):
        winners.append(("MIN_ABS_CORR_6_POSITIVE", six.iloc[0]))
    winner_rows, year_rows, curves = [], [], {}
    for label, row in winners:
        metrics, daily, total = metrics_for(row["mask"])
        winner_rows.append({"selection": label, "mask": int(row["mask"]), **metrics})
        curves[label] = total
        selected = members(row["mask"])
        daily_corr.loc[selected, selected].to_csv(out / f"{label.lower()}_correlations.csv", index_label="variant", encoding="utf-8-sig")
        for year, ret in daily.groupby(daily.index.year):
            year_rows.append({"selection": label, "year": year, **performance_metrics(ret)})
    winner_frame = pd.DataFrame(winner_rows)
    winner_frame.to_csv(out / "selected_portfolios.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(year_rows).to_csv(out / "selected_yearly.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(curves).to_csv(out / "selected_equity.csv", index_label="date", encoding="utf-8-sig")
    counts = pd.read_csv(counts_path)
    expected_poly = np.array([1], dtype=np.int64)
    for group in groups:
        expected_poly = np.convolve(expected_poly, [1, len(group)])
    np.testing.assert_array_equal(counts.combinations, expected_poly[1:])
    # The global minimum mean absolute correlation is necessarily an eligible
    # two-strategy pair; independently enumerate all such pairs as a certificate.
    pair_min = min(abs(daily_corr.iloc[i,j]) for i in range(len(names)) for j in range(i)
                   if summary.loc[i,"group"] != summary.loc[j,"group"])
    np.testing.assert_allclose(low.mean_abs_corr, pair_min, atol=1e-14)
    plot_selected_pool(out)
    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(daily_corr, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(len(names)), summary.short_code, rotation=90)
    ax.set_yticks(range(len(names)), summary.short_code)
    ax.set_title("Daily net account-return Pearson correlations; cash days included")
    fig.colorbar(im, ax=ax, label="Pearson correlation")
    fig.tight_layout();fig.savefig(out / "all_strategy_correlations.png", dpi=150);plt.close(fig)
    manifest = {"candidate_count": len(names), "group_count": len(groups), "exhaustive_nonempty_portfolios": int(counts.combinations.sum()),
                "exhaustive_portfolios_at_least_two": int(counts.loc[counts.accounts.ge(2),"combinations"].sum()),
                "start": str(returns.index.min().date()), "end": str(returns.index.max().date()), "days": len(returns),
                "same_signal_one_size_only": True, "kp_kd_optional": True,
                "correlation_objective": "mean(abs(off-diagonal daily Pearson account-return correlations)), at least two accounts",
                "capital": "equal initial capital; independently compounded accounts summed; no interaccount rebalancing",
                "reference_file_sha256": file_hash(old_path), "candidate_returns_sha256": file_hash(Path(experiment_out)/"daily_returns.csv"),
                "code_sha256": file_hash(Path(__file__)), "input_manifest": str(Path(experiment_out)/"manifest.json"),
                "checks": {"all_saved_candidates_independently_recomputed": "PASS", "counts_per_cardinality": "PASS",
                           "minimum_correlation_independent_pair_certificate": "PASS", "initial_equity_in_peak": "PASS"}}
    if override is not None:
        manifest["intraday_return_override_sha256"] = file_hash(override_path)
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    if write_report:
        write_alpha_pool_report(out, summary, ranked, winner_frame, counts, Path(experiment_out))
    print(winner_frame[["selection", "accounts", "CAGR", "MDD", "Sharpe", "Calmar", "mean_abs_corr", "members"]].to_string(index=False), flush=True)
    return out


def write_c3_delay_diagnostics(out, base):
    baseline = pd.read_csv(base / "trades.csv", parse_dates=["entry_date"])
    baseline = baseline[baseline.variant.str.startswith("C3_")]
    base_stats = pd.read_csv(base / "summary.csv").set_index("variant")
    results = []
    for scenario in ("c3_0901_close", "c3_0901_close_zero_volume_price"):
        delayed = pd.read_csv(out / scenario / "trade_intents.csv", parse_dates=["entry_date"])
        delayed = delayed[delayed.variant.str.startswith("C3_")]
        merged = baseline.merge(delayed, on=["variant", "entry_date"], suffixes=("_base", "_new"), validate="one_to_one")
        drift = pd.Series(0., index=merged.index)
        for leg in ("1x", "2x"):
            used = merged[f"w_{leg}_base"].gt(0)
            drift.loc[used] += merged.loc[used, f"w_{leg}_base"] * (merged.loc[used, f"entry_price_{leg}_new"] / merged.loc[used, f"entry_price_{leg}_base"] - 1)
        merged["entry_drift_bp"] = drift / (merged.w_1x_base + merged.w_2x_base) * 10000
        stats = pd.read_csv(out / scenario / "summary.csv").set_index("variant")
        for name, group in merged.groupby("variant", sort=False):
            filled = group[group.execution_status.eq("FILLED")]
            results.append({"scenario": scenario, "variant": name, "original_trades": len(group), "filled_trades": len(filled),
                            "unfilled_trades": len(group) - len(filled), "baseline_CAGR": base_stats.loc[name, "CAGR"],
                            "delayed_CAGR": stats.loc[name, "CAGR"], "baseline_Calmar": base_stats.loc[name, "Calmar"],
                            "delayed_Calmar": stats.loc[name, "Calmar"], "mean_entry_drift_bp": filled.entry_drift_bp.mean(),
                            "median_entry_drift_bp": filled.entry_drift_bp.median(),
                            "fraction_gross_return_worse": filled.gross_return_new.lt(filled.gross_return_base).mean(),
                            "mean_gross_return_lost_bp": ((filled.gross_return_base - filled.gross_return_new) * 10000).mean()})
    diagnostics = pd.DataFrame(results)
    diagnostics.to_csv(out / "c3_delay_diagnostics.csv", index=False, encoding="utf-8-sig")
    audit_rows = []
    for asset in ("069500", "122630", "229200", "233740"):
        minute = pd.read_csv(ROOT / f"jehyeong/intraday/data/minute/{asset}_TOSS_1m.csv", usecols=["timestamp", "close", "volume"])
        minute["timestamp"] = pd.to_datetime(minute.timestamp)
        minute = minute[minute.timestamp.dt.strftime("%H:%M").eq("09:01")].copy()
        minute["date"] = minute.timestamp.dt.normalize()
        minute = minute.set_index("date")
        daily = pd.read_csv(ROOT / f"jehyeong/intraday/data/{asset}_daily.csv", parse_dates=["date"]).set_index("date")
        needed = baseline.loc[(baseline.asset_1x.eq(int(asset)) & baseline.w_1x.gt(0)) |
                              (baseline.asset_2x.eq(int(asset)) & baseline.w_2x.gt(0)), "entry_date"].unique()
        for date in needed:
            if date not in minute.index:
                audit_rows.append({"asset": asset, "date": date, "status": "MISSING_0901"})
                continue
            bar, day = minute.loc[date], daily.loc[date]
            inside = day["low"] <= bar.close <= day["high"]
            audit_rows.append({"asset": asset, "date": date, "close_0901": bar.close, "volume_0901": bar.volume,
                               "daily_low": day["low"], "daily_high": day["high"], "inside_daily_range": inside,
                               "status": "OK" if inside and bar.volume > 0 else "ZERO_VOLUME" if bar.volume == 0 else "PRICE_RANGE_CONFLICT"})
    audit = pd.DataFrame(audit_rows)
    # This is an after-the-fact consistency audit, not a signal or fill filter.
    if audit.status.eq("PRICE_RANGE_CONFLICT").any():
        raise ValueError("Positive-volume C3 09:01 price conflicts with daily range")
    audit.to_csv(out / "first_minute_price_audit.csv", index=False, encoding="utf-8-sig")
    report_path = out / "report.md"
    text = report_path.read_text(encoding="utf-8").split("\n\n## C3 진입 지연 진단")[0]
    chosen = diagnostics[diagnostics.variant.eq("C3_KOSPI_KOSPI_SIGNAL_VT30")]
    text += "\n\n## C3 진입 지연 진단\n\n" + research_table(chosen)
    text += "\n\n진입가격 변화는 기존 비중으로 가중한 투자 ETF 가격 변화이며, 수수료·슬리피지와 별개다. gross_return_lost는 공통 체결 거래의 기존 시가 진입 수익에서 지연 진입 수익을 뺀 값이다. 미체결 거래는 가격 변화 평균에서 제외했다. 양수면 첫 1분 상승분을 놓쳐 진입 이후 수익이 줄었다. 개별 CAGR는 미체결도 포함한 전체 평가기간 기준이다."
    text += "\n\n## 09:01 표시 가격 검증\n\n실제 C3 진입에 필요한 ETF·거래일의 09:01 가격을 일봉 고가·저가와 대조했다. 양의 거래량 봉은 모두 일봉 범위 안이었다. 거래량 0인 표시 봉 중에는 범위 밖의 값도 있다. 예를 들어 2026-06-08의 069500 09:01 종가는 131,030원이지만 일봉 고가는 125,665원이고, 첫 거래량>0 봉은 09:03이다. 따라서 ZERO_VOLUME_PRICE는 표시값에 기계적으로 체결시킨 보조 가정이며 실제 체결 성과로 해석할 수 없다. 주 결과는 양의 거래량 09:01 봉에서만 체결한 경우다. 상세 대조는 first_minute_price_audit.csv에 저장했다."
    report_path.write_text(text, encoding="utf-8")
    return diagnostics


def run_cost_execution_sensitivity():
    """Offline factorial fee/fill comparison on the same frozen signal universe."""
    import importlib.util
    import shutil
    base = HERE / "results/transfer_leverage_experiments"
    out = base / "execution_cost_sensitivity"
    out.mkdir(parents=True, exist_ok=True)
    baseline_returns = pd.read_csv(base / "pool_selection/candidate_daily_returns.csv", index_col="date", parse_dates=True)
    calendar = baseline_returns.index
    summary = pd.read_csv(base / "summary.csv")
    ledger = pd.read_csv(base / "trades.csv", dtype={"asset_1x": str, "asset_2x": str}, parse_dates=["entry_date", "exit_date"])
    baseline_members = pd.read_csv(base / "pool_selection/selected_portfolios.csv").set_index("selection").loc["MAX_CALMAR", "members"].split(";")
    inputs = [base / "trades.csv", base / "summary.csv", base / "pool_selection/candidate_daily_returns.csv"]
    # 09:01 bar_end close is the end of 09:00--09:01, never the next bar.
    first_bars = {}
    for asset in ("069500", "122630", "229200", "233740"):
        path = ROOT / f"jehyeong/intraday/data/minute/{asset}_TOSS_1m.csv"
        d = pd.read_csv(path, usecols=["timestamp", "close", "volume", "timestamp_semantics"])
        if not d.timestamp_semantics.eq("bar_end").all():
            raise ValueError(f"Unverified C3 minute time: {path}")
        d["timestamp"] = pd.to_datetime(d.timestamp)
        if not d.timestamp.is_unique:
            raise ValueError(f"Duplicate C3 minute timestamps: {asset}")
        d = d[d.timestamp.dt.strftime("%H:%M").eq("09:01")].copy()
        d["date"] = d.timestamp.dt.normalize()
        first_bars[asset] = d.set_index("date")
        inputs.append(path)

    def trade_daily(t):
        daily = pd.Series(0., index=calendar)
        if len(t):
            realized = t.groupby("exit_date").net_return.agg(lambda x: np.prod(1 + x) - 1)
            if not realized.index.isin(calendar).all():
                raise ValueError("Trade exit outside evaluation calendar")
            daily.loc[realized.index] = realized
        return daily

    # Original ledger reconciliation before modifying any cost or fill.
    for name in summary.variant:
        np.testing.assert_allclose(trade_daily(ledger[ledger.variant.eq(name)]), baseline_returns[name], atol=1e-12)

    # Reexecute frozen intraday rules with fractional quantities and exact side costs.
    engine_path = ROOT / "jehyeong/intraday/code/backtest_intraday_momentum.py"
    selection_path = ROOT / "jehyeong/intraday/results/final/selected_rules.json"
    sys.path.insert(0, str(engine_path.parent))
    spec = importlib.util.spec_from_file_location("sensitivity_intraday", engine_path)
    engine = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = engine
    spec.loader.exec_module(engine)
    selected = json.loads(selection_path.read_text(encoding="utf-8"))
    inputs.extend([engine_path, selection_path])
    data = engine.load_data(engine.DATA)
    inputs.extend(p for p in engine.DATA.glob("*_daily.csv"))
    inputs.extend(engine.DATA / "minute" / f"{asset}_TOSS_1m.csv" for asset in selected["required_symbols"])
    nofee_intraday = pd.DataFrame(index=calendar)
    replay_checks = []
    for name, market in (("KP", "KOSPI200"), ("KD", "KOSDAQ150")):
        rule = selected["markets"][market]
        signal = engine.MARKETS[market]["signal"]
        features = engine.make_features(data[signal]["1m"], data[signal]["daily"])
        valid, _ = engine.eligible_days(market, data, features, str(calendar.min().date()), str(calendar.max().date()))
        days = [day for day in selected["sample_dates"][market] if pd.Timestamp(day) in calendar]
        if not set(days).issubset(valid):
            raise ValueError(f"Missing frozen intraday dates: {market}")
        features, _ = engine.add_vol_buckets(features, days)
        for fee in (1.5, 0.):
            config = engine.Config(initial_capital=100_000_000., commission_bps=fee, whole_shares=False, slippage_bps=1.)
            d, t, orders, marks = engine.simulate(market, rule["model"], features, data, days, engine.VERSIONS[0], config, **rule["options"])
            daily = d.assign(date=pd.to_datetime(d.date)).set_index("date").daily_return.reindex(calendar, fill_value=0.)
            np.testing.assert_allclose(orders.commission_cost, orders.filled_notional * fee / 10000, atol=1e-7)
            np.testing.assert_allclose(orders.slippage_cost, orders.reference_notional * .0001, atol=1e-7)
            np.testing.assert_allclose(t.net_pnl.sum(), d.equity.iloc[-1] - config.initial_capital, atol=1e-4)
            if fee:
                np.testing.assert_allclose(daily, baseline_returns[name], atol=1e-10)
                baseline_trade_times = t[["entry_time", "exit_time", "direction"]].reset_index(drop=True)
                replay_checks.append(f"{name}: baseline daily returns reproduced")
            else:
                pd.testing.assert_frame_equal(t[["entry_time", "exit_time", "direction"]].reset_index(drop=True), baseline_trade_times)
                nofee_intraday[name] = daily
                dest = out / "intraday_zero_fee" / name
                dest.mkdir(parents=True, exist_ok=True)
                for filename, frame in (("daily_pnl.csv", d), ("trades.csv", t), ("orders.csv", orders)):
                    frame.to_csv(dest / filename, index=False, encoding="utf-8-sig")
                replay_checks.append(f"{name}: zero fee side costs and unchanged trade timing verified")
            del marks
            print(f"[{name}] replay fee={fee}bp/side complete", flush=True)
    nofee_intraday.to_csv(out / "intraday_zero_fee/daily_returns.csv", index_label="date", encoding="utf-8-sig")
    del data

    scenarios = [("ZERO_FEE", 0., False, True),
                 ("C3_0901_CLOSE", 3., True, True),
                 ("ZERO_FEE_C3_0901_CLOSE", 0., True, True),
                 ("C3_0901_CLOSE_ZERO_VOLUME_PRICE", 3., True, False),
                 ("ZERO_FEE_C3_0901_CLOSE_ZERO_VOLUME_PRICE", 0., True, False)]
    comparisons, constituent_rows, curve_map = [], [], {}
    def portfolio_stats(returns, members):
        eq = (1 + returns[members]).cumprod().mean(axis=1)
        daily = eq.pct_change(fill_method=None)
        daily.iloc[0] = eq.iloc[0] - 1
        return performance_metrics(daily), eq
    initial_metrics, initial_eq = portfolio_stats(baseline_returns, baseline_members)
    comparisons.append({"scenario": "BASELINE", "selection": "OPTIMAL", "members": ";".join(baseline_members), "accounts": len(baseline_members), "same_as_baseline": True, **initial_metrics})
    curve_map["BASELINE optimal"] = initial_eq
    for scenario, fee_bp, delayed, require_volume in scenarios:
        print(f"Preparing {scenario}", flush=True)
        dest = out / scenario.lower()
        dest.mkdir(exist_ok=True)
        trades = ledger.copy()
        trades["commission"] = (trades.w_1x + trades.w_2x) * fee_bp / 10000
        trades["net_return"] = trades.gross_return - trades.commission - trades.slippage
        trades["execution_status"] = "FILLED"
        audit = []
        if delayed:
            for idx, row in trades.loc[trades.variant.str.startswith("C3_")].iterrows():
                gross, problems = 0., []
                for leg in ("1x", "2x"):
                    weight = row[f"w_{leg}"]
                    if not weight:
                        continue
                    asset = f"{int(float(row[f'asset_{leg}'])):06d}"
                    bars = first_bars[asset]
                    reason = ""
                    if row.entry_date not in bars.index:
                        reason = "MISSING_0901_BAR"
                    else:
                        bar = bars.loc[row.entry_date]
                        if not np.isfinite(bar.close) or bar.close <= 0:
                            reason = "INVALID_0901_CLOSE"
                        elif require_volume and bar.volume <= 0:
                            reason = "ZERO_VOLUME_0901_BAR"
                        else:
                            trades.loc[idx, f"entry_price_{leg}"] = bar.close
                            gross += weight * (row[f"exit_price_{leg}"] / bar.close - 1)
                    if reason:
                        problems.append(reason)
                        audit.append({"variant": row.variant, "date": row.entry_date, "asset": asset, "reason": reason})
                trades.loc[idx, "entry_time"] = "09:01_CLOSE"
                trades.loc[idx, "entry_time_actual"] = row.entry_date + pd.Timedelta(hours=9, minutes=1)
                if problems:
                    trades.loc[idx, "execution_status"] = "UNFILLED"
                    trades.loc[idx, ["gross_return", "commission", "slippage", "net_return"]] = 0.
                else:
                    trades.loc[idx, "gross_return"] = gross
                    trades.loc[idx, "net_return"] = gross - trades.loc[idx, "commission"] - trades.loc[idx, "slippage"]
        trades.to_csv(dest / "trade_intents.csv", index=False, encoding="utf-8-sig")
        filled = trades[trades.execution_status.eq("FILLED")].copy()
        filled.to_csv(dest / "trades.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(audit, columns=["variant", "date", "asset", "reason"]).to_csv(dest / "execution_exclusions.csv", index=False, encoding="utf-8-sig")
        returns = pd.DataFrame({name: trade_daily(filled[filled.variant.eq(name)]) for name in summary.variant})
        info = summary[["variant", "family", "market", "signal_mode", "size", "vehicle"]].copy()
        for i, row in info.iterrows():
            stats = performance_metrics(returns[row.variant], filled[filled.variant.eq(row.variant)])
            for field, value in stats.items():
                info.loc[i, field] = value
        info.to_csv(dest / "summary.csv", index=False, encoding="utf-8-sig")
        returns.to_csv(dest / "daily_returns.csv", index_label="date", encoding="utf-8-sig")
        intraday = nofee_intraday if fee_bp == 0 else baseline_returns[["KP", "KD"]]
        intraday.to_csv(dest / "intraday_daily_returns.csv", index_label="date", encoding="utf-8-sig")
        shutil.copyfile(base / "not_run.csv", dest / "not_run.csv")
        scenario_manifest = {"scenario": scenario, "start": str(calendar.min().date()), "end": str(calendar.max().date()),
                             "days": len(calendar), "commission_round_trip_bp": fee_bp, "slippage_round_trip_bp": 2.,
                             "c3_entry": "09:01 bar_end close" if delayed else "daily open", "c3_signal": "unchanged confirmed daily open features",
                             "c3_exit": "unchanged daily close", "require_positive_0901_volume": require_volume,
                             "missing_0901_policy": "unfilled entire allocation; cash; no fallback", "unfilled_intents": int(trades.execution_status.eq("UNFILLED").sum()),
                             "baseline_manifest_sha256": file_hash(base / "manifest.json"), "data_collection": False}
        (dest / "manifest.json").write_text(json.dumps(scenario_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        for row in filled.itertuples():
            gross = sum(getattr(row, f"w_{leg}") * (getattr(row, f"exit_price_{leg}") / getattr(row, f"entry_price_{leg}") - 1)
                        for leg in ("1x", "2x") if getattr(row, f"w_{leg}"))
            np.testing.assert_allclose(row.net_return, gross - (row.w_1x + row.w_2x) * (fee_bp + 2) / 10000, atol=1e-12)
        pool = select_alpha_pool(dest, write_report=False)
        chosen = pd.read_csv(pool / "selected_portfolios.csv").set_index("selection").loc["MAX_CALMAR"]
        all_returns = pd.read_csv(pool / "candidate_daily_returns.csv", index_col="date", parse_dates=True)
        for label, members in (("OPTIMAL", chosen.members.split(";")), ("BASELINE_MEMBERS", baseline_members)):
            metrics, eq = portfolio_stats(all_returns, members)
            comparisons.append({"scenario": scenario, "selection": label, "members": ";".join(members), "accounts": len(members),
                                "same_as_baseline": set(members) == set(baseline_members), **metrics})
            if label == "OPTIMAL":
                np.testing.assert_allclose(metrics["Calmar"], chosen.Calmar, atol=1e-10)
            if require_volume:
                curve_map[f"{scenario} {label}"] = eq
        chosen_info = pd.read_csv(pool / "candidate_summary.csv")
        for name in baseline_members + ["C3_KOSPI_KOSPI_SIGNAL_1X", "C3_KOSPI_KOSPI_SIGNAL_2X"]:
            constituent_rows.append({"scenario": scenario, **chosen_info.set_index("variant").loc[name].to_dict(), "variant": name})
    comparison = pd.DataFrame(comparisons)
    comparison.to_csv(out / "comparison.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(constituent_rows).to_csv(out / "constituent_comparison.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(curve_map).to_csv(out / "comparison_equity.csv", index_label="date", encoding="utf-8-sig")
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
    anchor = calendar.min() - pd.Timedelta(days=1)
    for caption, eq in curve_map.items():
        if caption.endswith("BASELINE_MEMBERS"):
            continue
        axes[0].plot([anchor, *calendar], [0., *((eq - 1) * 100)], label=caption)
        peaks = np.maximum.accumulate(np.r_[1., eq])[1:]
        axes[1].plot([anchor, *calendar], [0., *((eq.to_numpy() / peaks - 1) * 100)], label=caption)
    axes[0].set_ylabel("Cumulative net P&L (%)")
    axes[1].set_ylabel("Drawdown (%)")
    for ax in axes:
        ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out / "comparison_pnl_mdd.png", dpi=150); plt.close(fig)
    verify = {"inputs": {str(p.relative_to(ROOT)): file_hash(p) for p in dict.fromkeys(inputs)},
              "code_sha256": file_hash(Path(__file__)), "checks": replay_checks + ["original variant ledger reconciled", "all scenario ETF leg arithmetic verified", "all searches independently recomputed"],
              "data_collection": False}
    (out / "verification.json").write_text(json.dumps(verify, ensure_ascii=False, indent=2), encoding="utf-8")
    display = comparison[["scenario", "selection", "members", "accounts", "CAGR", "MDD", "Sharpe", "Calmar", "same_as_baseline"]].copy()
    for column in ("CAGR", "MDD"):
        display[column] = display[column].map(lambda v: f"{v:.2%}")
    report = ["# 수수료 및 C3 09:01 종가 체결 민감도",
              f"평가 {calendar.min().date()}~{calendar.max().date()}, {len(calendar)}거래일. 기존 33개 후보·13개 그룹, 같은 신호 비중 한 버전, KP·KD 선택 가능, 2계좌 이상 전체 조합을 각 시나리오에서 다시 검색했다. 초기자본 동일, 계좌별 복리 자산 합산, 계좌 간 리밸런싱 없음.",
              "수수료만 왕복 3bp에서 0bp로 변경한다. 슬리피지는 모든 경우 왕복 2bp 유지. KP·KD는 frozen 거래일·규칙으로 주문 수량과 매수·매도 수수료를 다시 계산했고, 원래 비용의 일별 수익이 기존 결과와 일치하는지 확인했다. 원본 데이터와 기존 결과는 보존했다.",
              "C3는 원래 시가 신호 및 진입비중을 그대로 고정하고, ETF 진입가격만 09:01 bar_end close(09:00~09:01 종가)로 변경했다. 청산은 기존 일봉 close. 지연 시점에 신호를 재판단하지 않는다. 정확한 09:01 가격이 없으면 미체결로 현금 보유한다. 기본 시나리오는 양의 거래량을 요구하며, ZERO_VOLUME_PRICE 보조 시나리오는 거래량 0이라도 CSV 표시가격 체결을 허용한다. 실제 주문은 체결 보장이 없으며 지수 시가의 실제 공표 지연까지 검증한 결과는 아니다.",
              research_table(display), "OPTIMAL은 해당 조건에서 새로 선정한 최고 Calmar 조합, BASELINE_MEMBERS는 기존 C3 코스피 VT30+KP+KD 조합을 그대로 유지한 결과다. 동일한 구성이라도 계좌의 최초 비중은 계좌 수의 역수이고 운용 중에는 손익에 따라 달라진다.",
              "![최적 조합 손익·낙폭](comparison_pnl_mdd.png)",
              "각 시나리오 pool_selection/ranked_candidates.csv에 상위 조합, execution_exclusions.csv에 미체결 사유, trade_intents.csv에 가격·비중·비용이 있다. 전체 표본에서 선택한 사후 최적 결과이며 일별 MDD를 사용한다."]
    (out / "report.md").write_text("\n\n".join(report), encoding="utf-8")
    write_c3_delay_diagnostics(out, base)
    print(comparison[["scenario", "selection", "CAGR", "MDD", "Sharpe", "Calmar", "members"]].to_string(index=False), flush=True)
    return out


if __name__ == "__main__":
    if "--cost-execution-sensitivity" in sys.argv:
        run_cost_execution_sensitivity()
    elif "--select-alpha-pool" in sys.argv:
        select_alpha_pool(HERE / "results/transfer_leverage_experiments")
    elif "--transfer-leverage" in sys.argv:
        run_transfer_leverage_experiments()
    else:
        main()
