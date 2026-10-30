from __future__ import annotations

"""Combine the six requested KRX strategy accounts across their actual data windows.

Run from any directory with: python3 jehyeong/overnight/combine_requested_six_strategies.py
The script is offline and reads existing, audited strategy result panels only.
"""

import base64
import hashlib
import html
import itertools
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "jehyeong/overnight/results/requested_six_strategy_combinations"
CAPITAL = 100_000_000.0
ANN = 252

STRATEGIES = [
    "BEOMGU_C3_VT30",
    "SUNNY_C",
    "WOOHYUN_TOP40_30",
    "DAYNIGHT_V1_COMBINED",
    "INTRADAY_KOSPI200",
    "INTRADAY_KOSDAQ150",
]
LABELS = {
    "BEOMGU_C3_VT30": "Beomgu r5 C3|vt30|ETF",
    "SUNNY_C": "Sunny Strategy C (pos_C)",
    "WOOHYUN_TOP40_30": "Woohyun Flow Top40, 30 positive bars",
    "DAYNIGHT_V1_COMBINED": "JeHyeong daynight V1 combined",
    "INTRADAY_KOSPI200": "JeHyeong intraday KOSPI200",
    "INTRADAY_KOSDAQ150": "JeHyeong intraday KOSDAQ150",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def check_series(name: str, frame: pd.DataFrame, ready: pd.Series,
                 returns: pd.Series, turnover: pd.Series,
                 commission: pd.Series, slippage: pd.Series,
                 paths: list[Path], assumptions: str) -> dict:
    idx = pd.DatetimeIndex(frame.index).normalize()
    if idx.has_duplicates or not idx.is_monotonic_increasing:
        raise ValueError(f"{name}: duplicate or unsorted dates")
    ready = pd.Series(ready.to_numpy(dtype=bool), index=idx)
    ret = pd.Series(pd.to_numeric(returns, errors="coerce").to_numpy(), index=idx)
    turn = pd.Series(pd.to_numeric(turnover, errors="coerce").to_numpy(), index=idx)
    fee = pd.Series(pd.to_numeric(commission, errors="coerce").to_numpy(), index=idx)
    slip = pd.Series(pd.to_numeric(slippage, errors="coerce").to_numpy(), index=idx)
    if not ready.any():
        raise ValueError(f"{name}: no ready observations")
    start, end = idx[ready.to_numpy()].min(), idx[ready.to_numpy()].max()
    covered = (idx >= start) & (idx <= end)
    if not np.isfinite(ret.to_numpy()[covered]).all():
        raise ValueError(f"{name}: non-finite returns inside coverage")
    if not np.isfinite(turn.to_numpy()[covered]).all() or (turn.to_numpy()[covered] < -1e-10).any():
        raise ValueError(f"{name}: invalid turnover inside coverage")
    if not np.isfinite(fee.to_numpy()[covered]).all() or not np.isfinite(slip.to_numpy()[covered]).all():
        raise ValueError(f"{name}: invalid costs inside coverage")
    if (ret.to_numpy()[covered] <= -1).any():
        raise ValueError(f"{name}: daily return at or below -100%")
    return {
        "returns": ret.loc[covered], "turnover": turn.loc[covered],
        "commission_rate": fee.loc[covered], "slippage_rate": slip.loc[covered],
        "start": start, "end": end, "source_paths": paths,
        "assumptions": assumptions,
    }


def load_strategies() -> tuple[dict, dict]:
    packs, hashes = {}, {}
    base = ROOT / "jehyeong/overnight/results/team_strategy_comparison"
    c3_path = base / "beomgu_c3_vt30_signal_panel.csv"
    c3 = pd.read_csv(c3_path, parse_dates=["date"]).set_index("date").sort_index()
    c3_ready = c3["available"].astype(bool)
    # Original r5 C3|vt30|ETF has 5bp round-trip costs applied to invested cash;
    # the leverage ETF's exposure is not extra cash turnover.
    c3_weight = c3["w_069500"].fillna(0) + c3["w_122630"].fillna(0)
    c3_fee = c3_weight * 0.0003
    c3_slip = c3_weight * 0.0002
    packs["BEOMGU_C3_VT30"] = check_series(
        "BEOMGU_C3_VT30", c3, c3_ready, c3["net_return"], 2 * c3_weight,
        c3_fee, c3_slip, [c3_path],
        "C3|vt30|ETF source net_return; original same-open signal/fill assumption; 3bp commission + 2bp slippage on invested ETF cash; turnover is both sides.",
    )

    sunny_path = base / "sunny_c_signal_panel.csv"
    sunny = pd.read_csv(sunny_path, parse_dates=["date"]).set_index("date").sort_index()
    sunny_ready = sunny["available"].astype(bool)
    sunny_pos = sunny["pos_C"].fillna(0)
    sunny_trade = sunny_pos.ne(0).astype(float)
    sunny_fee = sunny_trade * 0.0003
    packs["SUNNY_C"] = check_series(
        "SUNNY_C", sunny, sunny_ready,
        sunny_pos * sunny["oc_ret"] - sunny_fee,
        2 * sunny_trade, sunny_fee, sunny_trade * 0,
        [sunny_path, ROOT / "sunny/intraday_strategy_C/strategy3_smart_vs_dumb.py"],
        "Original Strategy C pos_C, open-to-close index proxy; original stated 3bp round-trip cost (commission); turnover is both sides.",
    )

    flow_path = base / "woohyun_top40_30_signal_panel.csv"
    trade_path = base / "all_strategy_trade_log.csv"
    flow = pd.read_csv(flow_path, parse_dates=["date"]).set_index("date").sort_index()
    flow_ready = flow["available"].astype(bool)
    # The original file's Flow Top40 / 30-bar selection is the fixed variant.
    trades = pd.read_csv(trade_path, parse_dates=["entry_date", "exit_date"])
    trades = trades.loc[trades["strategy"].eq("WOOHYUN_TOP40_30")].copy()
    if trades["entry_date"].duplicated().any() or not trades["entry_date"].eq(trades["exit_date"]).all():
        raise ValueError("Woohyun selected trades must be one same-day trade per session")
    w_daily = pd.Series(0.0, index=flow.index)
    w_comm = pd.Series(0.0, index=flow.index)
    w_slip = pd.Series(0.0, index=flow.index)
    w_turn = pd.Series(0.0, index=flow.index)
    for r in trades.itertuples():
        d = pd.Timestamp(r.exit_date).normalize()
        if d not in flow.index or not bool(flow.loc[d, "available"]):
            raise ValueError(f"Woohyun trade date is outside the ready flow panel: {d.date()}")
        # all_strategy_trade_log stores gross_return and its audited 3+2bp costs.
        w_daily.loc[d] += float(r.net_return)
        w_comm.loc[d] += float(r.commission)
        w_slip.loc[d] += float(r.slippage)
        w_turn.loc[d] += 2.0 * float(r.invested_weight)
    packs["WOOHYUN_TOP40_30"] = check_series(
        "WOOHYUN_TOP40_30", flow, flow_ready, w_daily, w_turn, w_comm, w_slip,
        [flow_path, trade_path, ROOT / "woohyun/Results/vwap+orb+flow40%.py"],
        "Original Flow Top40 and 30-positive-bar trade set; 3bp commission + 2bp slippage round trip per invested capital; absent OR/trades are flat; turnover is both sides.",
    )

    day_path = ROOT / "jehyeong/daynight/results/daily_nav.csv"
    day_orders_path = ROOT / "jehyeong/daynight/results/orders.csv"
    day = pd.read_csv(day_path, parse_dates=["date"])
    day = day.loc[day["version"].eq("V1") & day["account"].eq("combined")].set_index("date").sort_index()
    if day.empty:
        raise ValueError("Stored daynight V1 combined account is missing")
    orders = pd.read_csv(day_orders_path, parse_dates=["timestamp"])
    orders = orders.loc[orders["version"].eq("V1") & orders["account"].eq("combined")].copy()
    order_cost = (orders.assign(date=orders["timestamp"].dt.normalize())
                  .groupby("date")[["commission", "slippage_cost"]].sum())
    order_cost = order_cost.reindex(day.index).fillna(0)
    packs["DAYNIGHT_V1_COMBINED"] = check_series(
        "DAYNIGHT_V1_COMBINED", day, pd.Series(True, index=day.index), day["daily_return"],
        day["turnover"], order_cost["commission"] / day["previous_nav"],
        order_cost["slippage_cost"] / day["previous_nav"], [day_path, day_orders_path,
        ROOT / "jehyeong/daynight/backtest_daynight.py"],
        "Stored V1 combined independent account, including sequential intraday+overnight legs; source costs and filled-notional turnover; no double deduction.",
    )

    for code, market in (("INTRADAY_KOSPI200", "KOSPI200"), ("INTRADAY_KOSDAQ150", "KOSDAQ150")):
        pnl_path = ROOT / f"jehyeong/intraday/results/final/{market}/daily_pnl.csv"
        daily = pd.read_csv(pnl_path, parse_dates=["date"]).set_index("date").sort_index()
        packs[code] = check_series(
            code, daily, pd.Series(True, index=daily.index), daily["daily_return"],
            daily["turnover"], daily["commission_cost"] / daily["start_equity"],
            daily["slippage_cost"] / daily["start_equity"], [pnl_path,
            ROOT / "jehyeong/intraday/results/final/run_manifest.json",
            ROOT / "jehyeong/intraday/code/backtest_intraday_momentum.py"],
            f"Stored final {market} selected strategy daily net PnL; 1.5bp commission + 1bp slippage per side already included; ineligible/missing evaluation sessions are flat; source turnover.",
        )
    return packs, hashes


def metrics(returns: np.ndarray) -> dict:
    if len(returns) == 0 or not np.isfinite(returns).all():
        raise ValueError("Portfolio return vector is empty or non-finite")
    equity = np.r_[1.0, np.cumprod(1.0 + returns)]
    if equity[-1] <= 0:
        cagr = np.nan
    else:
        cagr = float(equity[-1] ** (ANN / len(returns)) - 1.0)
    sd = float(np.std(returns, ddof=1)) if len(returns) > 1 else np.nan
    sharpe = float(np.mean(returns) / sd * np.sqrt(ANN)) if sd > 0 else np.nan
    mdd = float(np.min(equity / np.maximum.accumulate(equity) - 1.0))
    return {
        "total_return": float(equity[-1] - 1.0), "cagr": cagr,
        "mdd": mdd, "sharpe": sharpe,
        "calmar": cagr / abs(mdd) if mdd < 0 else np.nan,
        "final_time_weighted_equity": float(equity[-1]),
    }


def evaluate_subset(codes: tuple[str, ...], packs: dict) -> tuple[dict, pd.DataFrame]:
    start = min(packs[c]["start"] for c in codes)
    end = min(packs[c]["end"] for c in codes)
    dates = pd.DatetimeIndex(sorted(set().union(*[
        set(packs[c]["returns"].loc[start:end].index) for c in codes
    ])))
    if not len(dates):
        raise ValueError(f"Empty combination period: {codes}")

    prior_nav = {c: 1.0 for c in codes}
    started: set[str] = set()
    rows = []
    linked_equity = 1.0
    cumulative_pnl = 0.0
    cumulative_fees = 0.0
    for date in dates:
        new = [c for c in codes if c not in started and packs[c]["start"] <= date]
        prior_total = sum(prior_nav[c] * CAPITAL for c in started)
        contribution = len(new) * CAPITAL
        opening_assets = prior_total + contribution
        daily_pnl = 0.0
        daily_fees = 0.0
        daily_slippage = 0.0
        turnover_notional = 0.0
        for c in codes:
            p = packs[c]
            if not (p["start"] <= date <= p["end"]):
                continue
            started.add(c)
            account_return = float(p["returns"].get(date, 0.0))
            account_start_nav = 1.0 if c in new else prior_nav[c]
            account_capital = CAPITAL * account_start_nav
            daily_pnl += account_capital * account_return
            daily_fees += account_capital * float(p["commission_rate"].get(date, 0.0))
            daily_slippage += account_capital * float(p["slippage_rate"].get(date, 0.0))
            turnover_notional += account_capital * float(p["turnover"].get(date, 0.0))
            prior_nav[c] = account_start_nav * (1.0 + account_return)
        port_return = daily_pnl / opening_assets
        if not np.isfinite(port_return) or port_return <= -1:
            raise ValueError(f"Invalid portfolio return {port_return} on {date}")
        linked_equity *= 1.0 + port_return
        cumulative_pnl += daily_pnl
        fees = daily_fees + daily_slippage
        cumulative_fees += fees
        rows.append({
            "date": date, "active_accounts": len(started),
            "new_accounts": "+".join(new), "capital_contribution": contribution,
            "opening_invested_capital": opening_assets, "portfolio_return": port_return,
            "time_weighted_equity": linked_equity,
            "aggregate_net_pnl_krw": cumulative_pnl,
            "daily_net_pnl_krw": daily_pnl,
            "daily_turnover_notional_krw": turnover_notional,
            "daily_turnover": turnover_notional / opening_assets,
            "daily_turnover_fee_krw": daily_fees,
            "daily_slippage_krw": daily_slippage,
            "daily_total_cost_krw": fees,
            "cumulative_total_cost_krw": cumulative_fees,
        })
    daily = pd.DataFrame(rows).set_index("date")
    daily["drawdown"] = daily["time_weighted_equity"] / np.maximum.accumulate(
        np.r_[1.0, daily["time_weighted_equity"].to_numpy()]
    )[1:] - 1.0
    m = metrics(daily["portfolio_return"].to_numpy())
    n = len(daily)
    # Total turnover and fees are expressed against each day's currently funded NAV.
    annual_turnover = float(daily["daily_turnover"].sum() * ANN / n)
    avg_turnover = float(daily["daily_turnover"].mean())
    total_notional = float(daily["daily_turnover_notional_krw"].sum())
    total_fee = float(daily["daily_total_cost_krw"].sum())
    fee_total = float(daily["daily_turnover_fee_krw"].sum())
    slippage_total = float(daily["daily_slippage_krw"].sum())
    if not np.isclose(fee_total + slippage_total, total_fee, rtol=1e-10, atol=1e-3):
        raise AssertionError(f"{codes}: cost reconciliation failed {fee_total + slippage_total} != {total_fee}")
    start_capital = sum(CAPITAL for c in codes if packs[c]["start"] == start)
    total_contributions = CAPITAL * len(codes)
    end_capital = sum(CAPITAL * prior_nav[c] for c in codes)
    if not np.isclose(end_capital, total_contributions + cumulative_pnl, rtol=1e-10, atol=1.0):
        raise AssertionError(f"{codes}: account NAV/PnL reconciliation failed")
    row = {
        "combination": "+".join(codes), "number_of_accounts": len(codes),
        "start": dates[0].date().isoformat(), "end": dates[-1].date().isoformat(),
        "sessions": n, "initial_capital_per_account": CAPITAL,
        "capital_contributed_first_day_krw": start_capital,
        "total_capital_contributions_krw": total_contributions,
        "final_total_account_equity_krw": end_capital,
        "net_pnl_krw": cumulative_pnl,
        "turnover_fee_krw": fee_total, "slippage_krw": slippage_total,
        "cumulative_fees_krw": total_fee,
        "turnover_notional_krw": total_notional, "turnover_x_total": float(daily["daily_turnover"].sum()),
        "average_daily_turnover_x": avg_turnover,
        "annualized_turnover_x_per_year": annual_turnover,
        "fee_bps_per_turnover_notional": total_fee / total_notional * 10000 if total_notional else np.nan,
        **m,
    }
    return row, daily


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    packs, _ = load_strategies()
    coverage = []
    for code in STRATEGIES:
        p = packs[code]
        coverage.append({
            "strategy": code, "label": LABELS[code],
            "first_ready": p["start"].date().isoformat(),
            "last_available": p["end"].date().isoformat(),
            "observations": len(p["returns"]),
            "source_assumption": p["assumptions"],
        })
    pd.DataFrame(coverage).to_csv(OUT / "strategy_coverage.csv", index=False, encoding="utf-8-sig")

    rows, daily_paths = [], {}
    for size in range(1, len(STRATEGIES) + 1):
        for codes in itertools.combinations(STRATEGIES, size):
            row, daily = evaluate_subset(codes, packs)
            rows.append(row)
            daily_paths[row["combination"]] = daily
    summary = pd.DataFrame(rows)
    # Rank independently by each requested objective; lower MDD means closer to zero.
    winners = {
        "highest_sharpe": summary.loc[summary["sharpe"].idxmax(), "combination"],
        "highest_cagr": summary.loc[summary["cagr"].idxmax(), "combination"],
        "lowest_mdd": summary.loc[summary["mdd"].idxmax(), "combination"],
        "highest_calmar": summary.loc[summary["calmar"].idxmax(), "combination"],
    }
    rank_map = {}
    for key, code in winners.items():
        rank_map.setdefault(code, []).append(key)
    summary["best_for"] = summary["combination"].map(
        lambda code: ";".join(rank_map.get(code, []))
    )
    summary.to_csv(OUT / "all_combinations_performance.csv", index=False,
                   encoding="utf-8-sig", float_format="%.10g")
    daynight_combinations = summary.loc[
        summary["combination"].str.contains("DAYNIGHT_V1_COMBINED", regex=False)
        & summary["number_of_accounts"].gt(1)
    ].copy()
    daynight_combinations = daynight_combinations.sort_values(
        ["number_of_accounts", "sharpe"], ascending=[True, False]
    )
    daynight_combinations.to_csv(
        OUT / "daynight_v1_combinations.csv", index=False, encoding="utf-8-sig",
        float_format="%.10g",
    )

    winners_daily = []
    for key, code in winners.items():
        daily = daily_paths[code].copy()
        daily["objective"] = key
        daily["combination"] = code
        winners_daily.append(daily.reset_index())
    pd.concat(winners_daily, ignore_index=True).to_csv(
        OUT / "best_combinations_daily_pnl.csv", index=False,
        encoding="utf-8-sig", float_format="%.10g")

    # Save daily return/equity and actual contributed-capital P&L for every subset.
    all_daily = []
    for code, daily in daily_paths.items():
        x = daily.reset_index().copy()
        x["combination"] = code
        all_daily.append(x)
    pd.concat(all_daily, ignore_index=True).to_csv(
        OUT / "all_combinations_daily_pnl.csv", index=False,
        encoding="utf-8-sig", float_format="%.10g")

    best_rows = summary.set_index("combination").loc[list(dict.fromkeys(winners.values()))].reset_index()
    best_rows.to_csv(OUT / "best_combinations.csv", index=False, encoding="utf-8-sig")
    chart_path = OUT / "best_combinations_pnl.png"
    write_plot(winners, daily_paths, chart_path)
    write_report(summary, best_rows, winners, coverage, OUT)
    write_html_report(summary, winners, coverage, chart_path, OUT / "report.html")

    source_files = sorted({path for p in packs.values() for path in p["source_paths"]})
    manifest = {
        "status": "COMPLETE", "strategies": STRATEGIES,
        "strategy_labels": LABELS, "combinations_evaluated": len(summary),
        "initial_capital_per_strategy_account_krw": CAPITAL,
        "annualization_sessions": ANN, "risk_free_rate": 0,
        "coverage_policy": "Each selected account starts at its first ready observation and ends at the earliest last-available date among selected strategies; newly ready accounts receive a fresh equal-size capital contribution; earlier dates use only accounts already ready.",
        "portfolio_aggregation": "Flow-adjusted time-weighted returns: daily portfolio PnL divided by prior account equity plus same-day capital contributions. Each active account compounds independently; no daily rebalance or cross-account position/netting.",
        "metrics": "CAGR = linked_equity^(252/sessions)-1; Sharpe = mean daily TWR return / sample std * sqrt(252); MDD from linked equity including initial unit NAV; Calmar = CAGR/abs(MDD).",
        "transaction_costs": "Net daily returns and costs as specified per original strategy/artifact; no second deduction. Turnover sums both buy and sell notional where available. For source trade-only panels, notional is reconstructed from invested weight.",
        "source_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in source_files if p.is_file()},
        "outputs": ["all_combinations_performance.csv", "all_combinations_daily_pnl.csv", "best_combinations.csv", "best_combinations_daily_pnl.csv", "best_combinations_pnl.png", "strategy_coverage.csv", "report.md", "report.html", "daynight_v1_combinations.csv"],
        "winners": winners,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(summary.sort_values("sharpe", ascending=False)[
        ["combination", "start", "end", "cagr", "mdd", "sharpe", "calmar", "average_daily_turnover_x", "cumulative_fees_krw"]
    ].head(10).to_string(index=False))
    print("\nWinners:", json.dumps(winners, ensure_ascii=False))
    print("Output:", OUT)


def write_plot(winners: dict, daily_paths: dict, path: Path) -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False,
                         "axes.spines.right": False})
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=False,
                             gridspec_kw={"height_ratios": [2, 1]})
    colors = {"highest_sharpe": "#1769aa", "highest_cagr": "#d1495b",
              "lowest_mdd": "#2a9d8f", "highest_calmar": "#8e5bb7"}
    seen = set()
    for objective, code in winners.items():
        if code in seen:
            continue
        seen.add(code)
        d = daily_paths[code]
        stats = d["time_weighted_equity"]
        label = f"{objective}: {code}"
        axes[0].plot(d.index, (stats - 1) * 100, label=label,
                     color=colors[objective], linewidth=1.6)
        axes[1].plot(d.index, d["drawdown"] * 100, label=label,
                     color=colors[objective], linewidth=1.2)
    axes[0].set_ylabel("Cumulative time-weighted P&L (%)")
    axes[0].set_title("Best requested strategy combinations | net of source transaction costs")
    axes[1].set_ylabel("Drawdown from linked equity (%)")
    axes[1].set_xlabel("KRX trading date")
    for ax in axes:
        ax.grid(alpha=.25)
        ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_html_report(summary: pd.DataFrame, winners: dict, coverage: list[dict],
                      chart_path: Path, path: Path) -> None:
    """Render an offline HTML report with all combinations and V1 composites."""
    dnv1 = summary.loc[
        summary["combination"].str.contains("DAYNIGHT_V1_COMBINED", regex=False)
        & summary["number_of_accounts"].gt(1)
    ].copy()
    dnv1 = dnv1.sort_values(["number_of_accounts", "sharpe"], ascending=[True, False])
    table_columns = [
        ("combination", "조합"), ("start", "시작일"), ("end", "종료일"),
        ("number_of_accounts", "계좌 수"), ("cagr", "CAGR"), ("mdd", "MDD"),
        ("sharpe", "Sharpe"), ("calmar", "Calmar"),
        ("annualized_turnover_x_per_year", "연환산 회전율"),
        ("cumulative_fees_krw", "수수료+슬리피지(원)"),
    ]

    def performance_table(frame: pd.DataFrame, table_id: str) -> str:
        heads = "".join(f"<th>{html.escape(label)}</th>" for _, label in table_columns)
        rows = []
        for _, r in frame.iterrows():
            cells = []
            for column, label in table_columns:
                value = r[column]
                if column == "combination":
                    display = html.escape(" + ".join(LABELS.get(code, code) for code in str(value).split("+")))
                elif column in ("start", "end"):
                    display = html.escape(str(value))
                elif column in ("cagr", "mdd"):
                    display = f"{value:.2%}"
                elif column == "sharpe":
                    display = f"{value:.3f}"
                elif column == "calmar":
                    display = f"{value:.3f}"
                elif column == "cumulative_fees_krw":
                    display = f"{value:,.0f}"
                elif column == "annualized_turnover_x_per_year":
                    display = f"{value:.2f}x"
                else:
                    display = str(int(value))
                cells.append(f"<td>{display}</td>")
            rows.append("<tr>" + "".join(cells) + "</tr>")
        return (
            f'<div class="table-wrap"><table id="{table_id}"><thead><tr>{heads}</tr></thead>'
            f"<tbody>{''.join(rows)}</tbody></table></div>"
        )

    winner_rows = []
    winner_labels = {
        "highest_sharpe": "최고 Sharpe", "highest_cagr": "최고 CAGR",
        "lowest_mdd": "최저 MDD", "highest_calmar": "최고 Calmar",
    }
    indexed = summary.set_index("combination")
    for objective, code in winners.items():
        r = indexed.loc[code]
        winner_rows.append(
            "<tr>" + f"<td>{winner_labels[objective]}</td>"
            + f"<td>{html.escape(' + '.join(LABELS.get(c, c) for c in code.split('+')))}</td>"
            + f"<td>{r['start']} ~ {r['end']}</td>"
            + f"<td>{r['cagr']:.2%}</td><td>{r['mdd']:.2%}</td>"
            + f"<td>{r['sharpe']:.3f}</td><td>{r['calmar']:.3f}</td>"
            + f"<td>{r['cumulative_fees_krw']:,.0f}</td></tr>"
        )
    coverage_rows = "".join(
        f"<tr><td>{html.escape(r['label'])}</td><td>{r['first_ready']}</td>"
        f"<td>{r['last_available']}</td><td>{r['observations']:,}</td></tr>"
        for r in coverage
    )
    chart_data = base64.b64encode(chart_path.read_bytes()).decode("ascii")
    all_table = performance_table(summary, "all-combinations")
    dnv1_table = performance_table(dnv1, "daynight-combinations")
    document = f'''<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>6개 전략 조합 성과 보고서</title>
<style>
:root {{ color-scheme: light; --ink:#182230; --muted:#536273; --line:#dbe2ea; --accent:#145ea8; --paper:#fff; --wash:#f2f5f9; }}
* {{ box-sizing:border-box }} body {{ margin:0; background:var(--wash); color:var(--ink); font:15px/1.6 system-ui,-apple-system,"Segoe UI",sans-serif }}
main {{ max-width:1500px; margin:0 auto; padding:32px 24px 64px }} h1 {{ margin:0 0 8px; font-size:30px }} h2 {{ margin:36px 0 12px; font-size:22px }} p {{ color:var(--muted) }} .card {{ background:var(--paper); border:1px solid var(--line); border-radius:12px; padding:20px; margin:16px 0 }}
.table-wrap {{ overflow:auto; max-height:650px; border:1px solid var(--line); border-radius:8px; background:#fff }} table {{ border-collapse:collapse; width:100%; white-space:nowrap; font-variant-numeric:tabular-nums }} th,td {{ padding:9px 11px; border-bottom:1px solid var(--line); text-align:right }} th {{ position:sticky; top:0; background:#eaf0f7; z-index:1; color:#27394c }} td:first-child,th:first-child {{ text-align:left }} tbody tr:hover {{ background:#f5f9ff }} .callout {{ padding:14px 18px; background:#e9f3ff; border-left:4px solid var(--accent); border-radius:6px }} img {{ width:100%; height:auto; border-radius:8px }} .controls {{ display:flex; gap:12px; align-items:center; margin:12px 0 }} input {{ min-width:260px; max-width:100%; padding:9px 12px; border:1px solid var(--line); border-radius:7px; font:inherit }} small {{ color:var(--muted) }}
</style></head><body><main>
<h1>요청한 6개 전략 조합 성과</h1>
<p>63개 비어 있지 않은 조합을 각 전략의 실제 준비 구간에 따라 평가했습니다. 동일 규모의 별도 계좌로 복리 운용하고 거래비용을 포함했습니다.</p>
<section class="card"><h2 style="margin-top:0">목표별 최고 조합</h2>
<div class="table-wrap"><table><thead><tr><th>목표</th><th>조합</th><th>구간</th><th>CAGR</th><th>MDD</th><th>Sharpe</th><th>Calmar</th><th>비용(원)</th></tr></thead><tbody>{''.join(winner_rows)}</tbody></table></div>
<img src="data:image/png;base64,{chart_data}" alt="목표별 최고 조합 P&amp;L 및 낙폭"></section>
<section class="card"><h2 style="margin-top:0">daynight V1 + 다른 전략 조합</h2>
<p>V1 결합 조합은 31개입니다: V1에 1~5개 다른 전략을 더한 모든 경우입니다(V1 단독은 제외). 아래 기본 정렬은 전략 수, 같은 전략 수 안에서는 Sharpe 내림차순입니다.</p>
<div class="callout">최고 CAGR은 daynight V1 단독, 최고 MDD는 Woohyun + KOSPI200 intraday 조합입니다. daynight V1은 단독 성과보다 조합에서 기간이 짧은 다른 전략들과 겹치는 구간이 크게 달라, 각 행의 실제 시작/종료일을 함께 확인하세요.</div>
<div class="controls"><input id="daynight-filter" type="search" placeholder="전략 이름으로 필터"><small>{len(dnv1)}개 조합 · 표 머리글 클릭으로 정렬</small></div>
{dnv1_table}</section>
<section class="card"><h2 style="margin-top:0">6개 전략의 데이터 가용 구간</h2>
<div class="table-wrap"><table><thead><tr><th>전략</th><th>첫 준비일</th><th>마지막 데이터일</th><th>관측일</th></tr></thead><tbody>{coverage_rows}</tbody></table></div></section>
<section class="card"><h2 style="margin-top:0">전체 63개 조합 성과</h2>
<div class="controls"><input id="all-filter" type="search" placeholder="전략 이름으로 필터"><small>모든 행의 CAGR, MDD, Sharpe, Calmar, 회전율, 총 거래비용 포함</small></div>
{all_table}</section>
<section class="card"><h2 style="margin-top:0">평가 방법 및 가정</h2><ul>
<li>각 전략은 1억원의 초기자본을 가진 독립 계좌입니다. 준비되기 전 계좌는 조합에서 제외하고, 준비일에 동일 금액의 새 계좌가 추가됩니다. 각 전략의 일별 손익은 실제 준비 기간 안에서만 반영합니다.</li>
<li>계좌별 복리와 중간 자금 유입을 반영한 연결(time-weighted) 수익률로 CAGR, Sharpe, MDD, Calmar를 계산했습니다. Calmar=CAGR/|MDD|, 무위험수익률 0, 연 252거래일입니다.</li>
<li>각 조합은 포함된 전략들의 마지막 가용일 중 가장 이른 날에 끝납니다. 초기 투자액과 최종 자산·PnL은 총 계좌 단위이며, CAGR·위험 지표는 연결 수익률 단위입니다.</li>
<li>수수료와 슬리피지는 원 전략 결과에 기록된 조건을 사용하고 재차 차감하지 않았습니다. 회전율은 양방향 거래대금 기준이며 일별 원화 turnover를 합산했습니다.</li>
<li>조합 목표별 최고 선택은 같은 전체 표본에서 수행되어 표본내 선택 편향이 있습니다. Beomgu는 확정 시가 기반 원 전략 체결 가정을 유지하며, 실거래 성과를 보장하지 않습니다.</li>
</ul><p><small>원자료: all_combinations_performance.csv, all_combinations_daily_pnl.csv, daynight_v1_combinations.csv, strategy_coverage.csv</small></p></section>
</main><script>
function setupTable(tableId,inputId) {{ const table=document.getElementById(tableId), input=document.getElementById(inputId); let asc=true; input.addEventListener('input',()=>{{const q=input.value.toLowerCase(); for(const row of table.tBodies[0].rows) row.hidden=!row.textContent.toLowerCase().includes(q);}}); table.querySelectorAll('th').forEach((th,col)=>th.addEventListener('click',()=>{{const body=table.tBodies[0],rows=[...body.rows]; rows.sort((a,b)=>{{const x=a.cells[col].textContent.trim(),y=b.cells[col].textContent.trim(),nx=parseFloat(x.replace(/[,%x\\s]/g,'')),ny=parseFloat(y.replace(/[,%x\\s]/g,'')); return (Number.isFinite(nx)&&Number.isFinite(ny)?nx-ny:x.localeCompare(y,'ko'))*(asc?1:-1);}}); asc=!asc; rows.forEach(r=>body.appendChild(r));}})); }}
setupTable('daynight-combinations','daynight-filter'); setupTable('all-combinations','all-filter');
</script></body></html>'''
    path.write_text(document, encoding="utf-8")


def write_report(summary: pd.DataFrame, best: pd.DataFrame, winners: dict,
                 coverage: list[dict], out: Path) -> None:
    by_code = summary.set_index("combination")
    lines = [
        "# 6개 요청 전략의 63개 조합 성과", "",
        "서로 다른 데이터 준비 기간을 보존해 6개 전략의 모든 비어 있지 않은 조합(63개)을 계산했다. "
        "각 전략은 별도의 동일 초기자본 계좌로 복리 운용하며, 전략 준비 전에는 조합에서 제외하고 준비일에 같은 규모의 신규 계좌를 추가한다. "
        "조합 종료일은 포함된 전략 중 가장 이른 마지막 평가일이다.", "",
        "성과는 계좌 간 현금 유입을 제거한 연결(time-weighted) 일별 수익률 기준이다. 실제 합산 P&L과 원화 비용·회전율도 CSV에 포함했다. "
        "기존 결과에 비용이 반영되어 있는 경우 재차 차감하지 않았다.", "",
        "## 목표별 최고 조합", "",
        "| 목표 | 조합 | 구간 | CAGR | MDD | Sharpe | Calmar | 평균 일 회전율 | 총 수수료+슬리피지 |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    objective_labels = {"highest_sharpe": "최고 Sharpe", "highest_cagr": "최고 CAGR",
                        "lowest_mdd": "최저 MDD", "highest_calmar": "최고 Calmar"}
    for objective, code in winners.items():
        r = by_code.loc[code]
        lines.append(
            f"| {objective_labels[objective]} | {code} | {r['start']}~{r['end']} | "
            f"{r['cagr']:.2%} | {r['mdd']:.2%} | {r['sharpe']:.3f} | {r['calmar']:.3f} | "
            f"{r['average_daily_turnover_x']:.3f}x | {r['cumulative_fees_krw']:,.0f}원 |"
        )
    lines += ["", "![목표별 최고 조합 P&L 및 낙폭](best_combinations_pnl.png)", "",
              "## 전략별 데이터 준비 구간", "",
              "| 전략 | 첫 준비일 | 마지막 평가일 | 관측일 |", "|---|---|---|---:|"]
    for r in coverage:
        lines.append(f"| {r['label']} | {r['first_ready']} | {r['last_available']} | {r['observations']:,} |")
    lines += ["", "## 전체 조합 성과표", "",
              "`all_combinations_performance.csv`에 63개 전체 조합의 CAGR, MDD, Sharpe, Calmar, 거래회전율, 거래비용이 수록되어 있다. "
              "조합 간 기간이 다르므로 순위는 각 조합이 실제로 관측 가능한 전체 기간에서 계산했다.", "",
              "## 자본·산식·비용 가정", "",
              f"- 각 전략 계좌 초기자본: {CAPITAL:,.0f}원. 기존 계좌는 독립 복리 운용하며 계좌 사이에 주문·자금 이동·상계가 없다.",
              "- 계좌는 첫 유효 준비일에 시작한다. 준비일에 추가되는 계좌 자본은 현금유입으로 처리하고, CAGR/Sharpe/MDD는 연결 수익률로 계산해 유입금이 수익이나 고점으로 오인되지 않게 했다.",
              "- 조합 마지막 날은 해당 조합 전략의 마지막 가용일 중 가장 빠른 날짜다. 이 날짜 이후 전략의 손익을 외삽하지 않는다.",
              "- Sharpe는 무위험수익률 0, 일별 표본 표준편차, 연 252 거래일을 쓴다. CAGR은 연결 자산의 252/관측 세션 수 연율화, Calmar=CAGR/|MDD|다.",
              "- turnover는 매수+매도 거래대금/해당일 운용 NAV다. 거래 로그만 보유한 당일 전략은 거래금액 비중의 양방향 합계로 재구성했다.",
              "- 비용은 원 전략 결과의 순수익과 기록된 주문 비용을 사용한다. Sunny는 원 코드의 왕복 3bp, Woohyun/Beomgu는 기존 감사 결과의 3bp 수수료+2bp 슬리피지, daynight/intraday는 저장 결과의 편도 1.5bp+1bp를 따른다.",
              "- 모든 순수익은 원자료의 선택된 고정전략 결과다. 조합 성과를 보고 개별 전략의 임계값이나 설정을 다시 최적화하지 않았다. 다만 63개 중 목표별 최고를 고르는 과정 자체는 동일 표본 내 선택이므로 표본외 성과 보장은 아니다.",
              "- Beomgu C3는 확정 시가를 보고 같은 시가 체결하는 원 전략 가정을 유지한다. Woohyun은 첫 ORB 신호 뒤 30개의 양의 거래량 봉을 기준으로 한 기존 선정 거래를 사용한다. intraday는 저장된 final KOSPI/KOSDAQ 규칙 결과다.",
              "", "## 재현", "",
              "```bash", "python3 jehyeong/overnight/combine_requested_six_strategies.py", "```", "",
              "산출물: `report.html` (전체 HTML 보고서 및 daynight V1 조합 표), `daynight_v1_combinations.csv`, `all_combinations_performance.csv`, `all_combinations_daily_pnl.csv`, `best_combinations.csv`, `best_combinations_daily_pnl.csv`, `best_combinations_pnl.png`, `strategy_coverage.csv`, `manifest.json`.", ""]
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
