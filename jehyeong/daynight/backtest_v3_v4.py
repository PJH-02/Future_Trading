"""V3: 09:05 minute close entry. V4: independent realized-P&L MA20 gates."""
from __future__ import annotations

import argparse
import base64
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

import backtest_daynight as base
from pnl_ma20 import build_gate

HERE = Path(__file__).resolve().parent
VERSIONS = ("V3", "V4")


def execution_plan(market, dates, day, strategy, direction, multiplier=1):
    """Resolve actual ETF prices AFTER selecting a direction; no return negation."""
    if not direction:
        return None, ""
    actual_direction = int(direction * multiplier)
    symbol = "233740" if actual_direction == 1 else "251340"
    common = dict(strategy=strategy, symbol=symbol, direction=actual_direction,
                  base_direction=int(direction), entry_mode=int(multiplier), entry_reference_field="close")
    if strategy == "intraday":
        entry = market.bar(symbol, base.stamp(day, "09:06"))
        exit_ = market.bar(symbol, base.stamp(day, "15:01"))
        if not base.executable(entry) or not base.executable(exit_):
            return None, "missing_or_zero_volume_intraday_close_execution"
        return dict(common, entry_time=base.stamp(day, "09:06"), exit_time=base.stamp(day, "15:00"),
                    entry_reference_bar_end=entry.timestamp, exit_reference_bar_end=exit_.timestamp,
                    entry_reference=float(entry.close), exit_reference=float(exit_.open),
                    signal_available_time=base.stamp(day, "09:05")), ""
    nxt = market.next.get(day)
    if nxt is None or nxt > dates[-1]:
        return None, "next_session_outside_sample"
    exit_ = market.bar(symbol, base.stamp(nxt, "09:01"))
    mark = market.bar(symbol, base.stamp(day, "15:30"))
    g = market.groups[symbol].get(day)
    candidates = None if g is None else g[g.timestamp.between(base.stamp(day, "09:01"), base.stamp(day, "15:20")) & g.volume.gt(0)]
    if not base.executable(exit_):
        return None, "missing_or_zero_volume_next_session_0900_exit"
    if mark is None:
        return None, "missing_entry_day_1530_mark"
    if candidates is None or candidates.empty:
        return None, "missing_last_traded_price_before_1520"
    entry = candidates.iloc[-1]
    return dict(common, entry_time=base.stamp(day, "15:20"), exit_time=base.stamp(nxt, "09:00"),
                entry_reference_bar_end=entry.timestamp, exit_reference_bar_end=exit_.timestamp,
                entry_reference=float(entry.close), exit_reference=float(exit_.open), mark_close=float(mark.close),
                signal_available_time=base.stamp(day, "15:20")), ""


def corrected_simulate(version, account, dates, plans, config):
    daily, orders, trades = base.simulate(version, account, dates, plans, config)
    if orders.empty:
        orders = pd.DataFrame({name: pd.Series(dtype="object") for name in
                               ("version", "account", "strategy", "symbol", "side", "reference_field", "reference_kind")})
        for name in ("timestamp", "reference_bar_end"):
            orders[name] = pd.Series(dtype="datetime64[ns]")
        for name in ("quantity", "reference_price", "fill_price", "filled_notional", "commission"):
            orders[name] = pd.Series(dtype=float)
    if trades.empty:
        trades = pd.DataFrame({name: pd.Series(dtype="object") for name in
                               ("version", "account", "strategy", "symbol")})
        for name in ("entry_time", "exit_time"):
            trades[name] = pd.Series(dtype="datetime64[ns]")
        for name in ("exit_equity", "entry_mode"):
            trades[name] = pd.Series(dtype=float)
    if not orders.empty:
        buys = orders.side.eq("BUY")
        orders.loc[buys, "reference_field"] = "close"
        orders.loc[buys & orders.strategy.eq("intraday"), "reference_kind"] = "minute_close"
    return daily, orders, trades


def build_v3_plans(market, dates):
    _, original_signals, _ = base.make_plans(market, dates)
    signals = original_signals[original_signals.version.eq("V2")].copy()
    signals["version"] = "V3"
    plans, skips = {}, []
    for row in signals.itertuples():
        day_plan, dr = execution_plan(market, dates, row.date, "intraday", row.intraday_direction)
        night_plan, nr = execution_plan(market, dates, row.date, "overnight", row.overnight_direction)
        plans["V3", row.date] = day_plan, night_plan
        for strategy, reason in [("intraday", dr or row.intraday_skip_reason if not row.intraday_direction else dr),
                                 ("overnight", nr or row.overnight_skip_reason if not row.overnight_direction else nr)]:
            if reason:
                skips.append(dict(version="V3", strategy=strategy, date=row.date, reason=reason))
    return plans, signals, skips


def shadow_gates(dates, normal_trades, initial_capital):
    """Each normal standalone account drives its own gate; combined P&L is excluded."""
    output = []
    for strategy in ("intraday", "overnight"):
        t = normal_trades[normal_trades.account.eq(strategy)].copy()
        exits = t[["exit_time"]].copy()
        exits["reference_pnl"] = t.exit_equity - initial_capital
        gate = build_gate(dates, exits, window=20)
        gate = gate.rename(columns={"ma20": "decision_ma20"})
        gate["daily_ma20"] = gate.reference_pnl.rolling(20, min_periods=20).mean()
        gate["strategy"] = strategy
        gate["entry_mode"] = gate.mode_before_exit if strategy == "intraday" else gate.mode_after_exit
        gate["exit_time"] = gate.date + pd.Timedelta(hours=15 if strategy == "intraday" else 9)
        gate["last_update_time"] = gate.exit_time.where(gate.has_exit).ffill()
        if strategy == "intraday":
            gate["reference_pnl_at_entry"] = gate.reference_pnl.shift(1).fillna(0.)
            gate["decision_ma20_at_entry"] = gate.decision_ma20.shift(1)
            gate["mode_source_time_at_entry"] = gate.last_update_time.shift(1)
        else:
            gate["reference_pnl_at_entry"] = gate.reference_pnl
            gate["decision_ma20_at_entry"] = gate.decision_ma20
            gate["mode_source_time_at_entry"] = gate.last_update_time
        output.append(gate)
    return pd.concat(output, ignore_index=True)


def build_v4_plans(market, dates, signals, gates):
    lookup = gates.set_index(["date", "strategy"])
    plans, skips, rows = {}, [], []
    for row in signals.itertuples():
        day = row.date
        pair = []
        for strategy, direction in [("intraday", row.intraday_direction), ("overnight", row.overnight_direction)]:
            gate = lookup.loc[(day, strategy)]
            mode = int(gate.entry_mode)
            plan, reason = execution_plan(market, dates, day, strategy, direction, mode)
            pair.append(plan)
            rows.append(dict(date=day, strategy=strategy, raw_direction=int(direction), entry_mode=mode,
                             actual_direction=int(direction * mode), executed=plan is not None,
                             symbol="" if plan is None else plan["symbol"], skip_reason=reason,
                             reference_pnl_at_entry=gate.reference_pnl_at_entry, decision_ma20_at_entry=gate.decision_ma20_at_entry,
                             mode_source_time_at_entry=gate.mode_source_time_at_entry))
            if reason:
                skips.append(dict(version="V4", strategy=strategy, date=day, reason=reason))
        plans["V4", day] = tuple(pair)
    return plans, skips, pd.DataFrame(rows)


def charts(daily, gates, comparison_daily, output):
    colors = {"intraday": "#2769b0", "overnight": "#c17820", "combined": "#15846a"}
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False})
    files = []
    def save(fig, name):
        fig.tight_layout()
        for extension in ("png", "svg"):
            fig.savefig(output / f"{name}.{extension}", dpi=150, bbox_inches="tight")
        plt.close(fig)
        files.append(name + ".png")
    for version in VERSIONS:
        fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
        for strategy in base.SLEEVES:
            g = daily[daily.version.eq(version) & daily.account.eq(strategy)]
            axes[0].plot(g.date, g.cumulative_pnl / 1e6, color=colors[strategy], label=strategy.title())
            axes[1].plot(g.date, g.drawdown, color=colors[strategy], lw=1)
        axes[0].set_title(version + " | 09:05 minute CLOSE entry | Net of costs")
        axes[0].set_ylabel("Cumulative P&L (KRW million)")
        axes[0].legend(frameon=False)
        axes[1].set_ylabel("Daily MTM drawdown")
        axes[1].yaxis.set_major_formatter(PercentFormatter(1))
        for ax in axes:
            ax.grid(alpha=.18)
        save(fig, "pnl_" + version)
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    for version, g in comparison_daily[comparison_daily.account.eq("combined")].groupby("version"):
        axes[0].plot(g.date, g.cumulative_pnl / 1e6, label=version)
        axes[1].plot(g.date, g.drawdown, label=version)
    axes[0].set_title("Combined account | V1 / V2 / V3 / V4")
    axes[0].set_ylabel("Cumulative P&L (KRW million)")
    axes[0].legend(frameon=False)
    axes[1].set_ylabel("Daily MTM drawdown")
    axes[1].yaxis.set_major_formatter(PercentFormatter(1))
    for ax in axes:
        ax.grid(alpha=.18)
    save(fig, "pnl_comparison")
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    for ax, strategy in zip(axes, ("intraday", "overnight")):
        g = gates[gates.strategy.eq(strategy)]
        ax.plot(g.date, g.reference_pnl / 1e6, label="Normal shadow realized P&L", color=colors[strategy])
        ax.plot(g.date, g.daily_ma20 / 1e6, label="20-session MA", color="#333333", lw=1)
        ax.fill_between(g.date, 0, 1, where=g.entry_mode.eq(-1), transform=ax.get_xaxis_transform(), color="#cc4040", alpha=.12, label="Entry uses inversion")
        ax.set_title(strategy.title() + " | Normal shadow P&L and MA20")
        ax.set_ylabel("KRW million")
        ax.legend(frameon=False)
        ax.grid(alpha=.18)
    save(fig, "shadow_pnl_ma20")
    years = sorted(daily.date.dt.year.unique())
    fig, axes = plt.subplots(len(years), 2, figsize=(14, len(years) * 3.1), squeeze=False)
    for i, year in enumerate(years):
        for j, version in enumerate(VERSIONS):
            ax = axes[i, j]
            for strategy in base.SLEEVES:
                g = daily[daily.version.eq(version) & daily.account.eq(strategy) & daily.date.dt.year.eq(year)]
                ax.plot(g.date, (g.nav - g.previous_nav.iloc[0]) / 1e6, label=strategy.title(), color=colors[strategy])
            ax.set_title(f"{year} | {version} | {g.date.iloc[0]:%m-%d} to {g.date.iloc[-1]:%m-%d}")
            ax.set_ylabel("Year P&L (KRW million)")
            ax.grid(alpha=.18)
            if not i:
                ax.legend(frameon=False)
    save(fig, "pnl_yearly")
    return files


RULES = """V3는 V2를 기준으로 장중 진입 가격만 09:05 시작 분봉 종가(원본 09:06 종료 봉 종가)로 변경합니다. 신호 판단은 기존 09:05 시가를 그대로 사용하고 장중 청산은 15:00 시가입니다.
오버나이트는 V2와 같이 15:19 분봉 종가로 IBS를 계산하고 15:20 직전 마지막 실제 체결가를 참고가격으로 진입하며, 다음 거래일 09:00 시가에 청산합니다.
V4는 V3에 두 개의 독립적인 정상 전략 모의 계좌를 추가합니다. 각 모의 계좌는 최초 1억원에서 시작해 비용을 반영하고 정상 방향으로 계속 운용됩니다. 이 계좌들은 매매 방향 판단용이며 결합 계좌의 자금을 차감하지 않습니다.
각 전략의 포지션이 청산되면 모의 계좌의 누적 확정 P&L을 현재 관측값으로 삼고, 직전 19거래일 말의 확정 P&L과 함께 MA20을 계산합니다. 거래가 없는 날은 직전 확정 P&L을 유지해 평균에 포함하며, 20개 관측치가 쌓이기 전에는 정상 방향입니다.
청산 직후 정상 모의 P&L < MA20이면 해당 전략의 다음 시그널 방향을 반전하고, P&L >= MA20이면 정상으로 복귀합니다. 포지션 청산이 없는 날에는 반전 상태를 변경하지 않습니다.
장중 진입은 당일 15:00 청산 전 상태를 사용하고, 오버나이트 진입은 당일 09:00 청산 후 상태를 사용합니다. 당일 뒤에 발생할 청산손익을 미리 사용하지 않습니다.
롱·숏 반전은 레버리지 ETF(233740) 매수와 인버스 ETF(251340) 매수를 교환합니다. 실제 반전 ETF의 가격·수량·비용으로 계산하므로 수익률의 단순 부호 반전과 다릅니다. 매번 현금 전액을 투자하며 소수점 수량을 허용합니다.
정상 전략 모의 P&L은 청산된 거래의 손익만 반영합니다. 표시 성과의 NAV·MDD·Sharpe는 종전과 같은 매 거래일 15:30 종가 평가 기준이며, 두 기준을 구분합니다.
매수·매도 편도 수수료 1.5bp와 불리한 슬리피지 1bp를 적용합니다. CAGR은 관측 달력 기간, Sharpe는 무위험수익률 0·표본표준편차·sqrt(252), 회전율은 양방향 거래대금/전일 NAV의 합입니다.
15:20 참고가격 체결과 분봉 종가 체결은 이상적 체결가격 가정입니다. 필수 체결봉·다음 거래일 시가의 거래량이 없으면 해당 ETF 거래를 제외하고 사유를 남깁니다. 반전 ETF의 거래 가능 여부는 정상 ETF와 별도로 확인합니다.
초기 관측치 이전의 모의 계좌 이력은 사용하지 않습니다. 2022·2026년은 부분연도이며 모든 성과는 비용 차감 후입니다. V1·V2 원본 파일과 결과는 보존합니다.
"""


def report(summary, yearly, comparison, figures, output):
    full, annual, compare = (base.formatted_table(x) for x in (summary, yearly, comparison))
    full["반전거래수"] = summary.inverted_trades.to_numpy()
    annual["반전거래수"] = yearly.inverted_trades.to_numpy()
    def image_html(name):
        return f'<img alt="{name}" src="data:image/png;base64,{base64.b64encode((output/name).read_bytes()).decode()}">'
    rules_html = "".join("<li>" + line.replace("<", "&lt;").replace(">", "&gt;") + "</li>" for line in RULES.strip().splitlines())
    intro = f"평가 구간: {summary.start.iloc[0]}~{summary.end.iloc[0]} · {int(summary.sessions.iloc[0])}거래일 · 각 계좌 초기 {summary.starting_nav.iloc[0]:,.0f}원"
    html = f'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Kosdaq V3 & V4</title>
<style>body{{font:15px/1.65 system-ui,sans-serif;margin:28px;color:#17352d;background:#f5f7f5}}main{{max-width:1480px;margin:auto}}.scroll{{overflow-x:auto;background:white}}table{{border-collapse:collapse;white-space:nowrap;font-size:13px;width:100%}}th,td{{padding:9px;border-bottom:1px solid #ddd;text-align:right}}th{{background:#e7eeea}}img{{width:100%;height:auto;margin:18px 0}}li{{margin:8px 0}}.note{{padding:14px;background:#fff3cd}}</style>
<main><h1>Kosdaq day & night · V3 / V4</h1><p>{intro}</p><p class="note">V3: 09:05 분봉 종가 진입 · V4: V3 + 정상 모의 누적 확정 P&L의 MA20으로 방향 반전. 15:20 직전 체결가 참고가격 가정을 유지합니다.</p>
<h2>V3·V4 전체 성과</h2><div class="scroll">{full.to_html(index=False,border=0)}</div><h2>V1~V4 비교</h2><div class="scroll">{compare.to_html(index=False,border=0)}</div>
<h2>V3·V4 연도별 개별·결합 성과</h2><div class="scroll">{annual.to_html(index=False,border=0)}</div><h2>P&L·낙폭·정상 모의 P&L과 MA20</h2>{''.join(image_html(f) for f in figures)}
<h2>전략·시간·산식 정의</h2><ul>{rules_html}</ul><p>상세 추적: shadow_gate.csv, switch_events.csv, execution_signals.csv, orders.csv, trades.csv, skips.csv, independent_check.json, acceptance.json.</p></main></html>'''
    (output / "report.html").write_text(html, encoding="utf-8")
    md = ["# Kosdaq V3 / V4", "", intro, "", "## 전체 성과", "", base.markdown_table(full), "", "## V1~V4 비교", "", base.markdown_table(compare), "", "## 연도별 성과", "", base.markdown_table(annual), "", RULES]
    md += [f"![{f}]({f})\n" for f in figures]
    (output / "report.md").write_text("\n".join(md) + "\n", encoding="utf-8")


def run(data_dir, output, config):
    output.mkdir(parents=True, exist_ok=True)
    (output / "acceptance.json").write_text('{"status":"PENDING"}\n')
    market = base.Market(data_dir)
    dates = market.calendar(config)
    plans, signals, skips = build_v3_plans(market, dates)
    daily_parts, order_parts, trade_parts = [], [], []
    for account in base.SLEEVES:
        d, o, t = corrected_simulate("V3", account, dates, plans, config)
        daily_parts.append(d); order_parts.append(o); trade_parts.append(t)
    normal_trades = pd.concat(trade_parts, ignore_index=True)
    gates = shadow_gates(dates, normal_trades, config.initial_capital)
    flipped, flip_skips, execution_signals = build_v4_plans(market, dates, signals, gates)
    plans.update(flipped)
    skips.extend(flip_skips)
    for account in base.SLEEVES:
        d, o, t = corrected_simulate("V4", account, dates, plans, config)
        daily_parts.append(d); order_parts.append(o); trade_parts.append(t)
    daily, orders, trades = (pd.concat(x, ignore_index=True) for x in (daily_parts, order_parts, trade_parts))
    summary, yearly = base.performance_tables(daily, trades)
    for frame in (summary, yearly):
        frame["inverted_trades"] = [int(((trades.version == row.version) & (trades.account == row.account) & trades.entry_mode.eq(-1) & (trades.exit_time.dt.year.eq(row.year) if hasattr(row, "year") else True)).sum()) for row in frame.itertuples()]
    comparison, comparison_yearly, comparison_daily = summary.copy(), yearly.copy(), daily.copy()
    old = HERE / "results"
    old_manifest = json.loads((old / "manifest.json").read_text())
    comparable = old_manifest["config"] == asdict(config) and old_manifest["common_start"] == str(dates[0].date()) and old_manifest["common_end"] == str(dates[-1].date())
    comparable = comparable and {s["path"]:s["sha256"] for s in old_manifest["source_files"]} == {s["path"]:s["sha256"] for s in market.sources}
    if comparable:
        comparison = pd.concat([pd.read_csv(old / "summary.csv"), summary], ignore_index=True)
        comparison_yearly = pd.concat([pd.read_csv(old / "yearly.csv"), yearly], ignore_index=True)
        comparison_daily = pd.concat([pd.read_csv(old / "daily_nav.csv", parse_dates=["date"]), daily], ignore_index=True)
    switch_events = gates[gates.has_exit & gates.mode_before_exit.ne(gates.mode_after_exit)]
    tables = dict(summary=summary, yearly=yearly, comparison_summary=comparison, comparison_yearly=comparison_yearly,
                  daily_nav=daily, orders=orders, trades=trades, normal_signals=signals, execution_signals=execution_signals,
                  shadow_gate=gates, switch_events=switch_events,
                  skips=pd.DataFrame(skips, columns=["version", "strategy", "date", "reason"]), data_quality=market.quality(dates))
    for name, frame in tables.items():
        frame.to_csv(output / f"{name}.csv", index=False, encoding="utf-8-sig", lineterminator="\n")
    figures = charts(daily, gates, comparison_daily, output)
    report(summary, yearly, comparison, figures, output)
    manifest = dict(config=asdict(config), common_start=str(dates[0].date()), common_end=str(dates[-1].date()), sessions=len(dates),
                    generated_at_utc=datetime.now(timezone.utc).isoformat(), source_files=market.sources,
                    code_sha256={p.name:base.sha256(p) for p in (Path(__file__), HERE / "pnl_ma20.py", HERE / "backtest_daynight.py")},
                    versions={"V3":"V2 with 09:05 minute close entry", "V4":"V3 with separate normal-shadow realized-P&L MA20 gates"},
                    normal_reference="Standalone V3 intraday and overnight accounts; net cumulative realized P&L, 20 exchange sessions including cash days",
                    gate_clock="Only normal shadow exits update mode. Intraday uses mode before today's15:00 exit; overnight uses mode after today's09:00 exit.",
                    reversal="Swap 233740 and 251340 buys; use actual ETF prices, never negate baseline returns", assumptions=RULES.strip().splitlines(),
                    historical_results_preserved=str(old.resolve()), prior_versions_comparable=comparable,
                    owner_acceptance="PENDING independent verification; see acceptance.json")
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(summary[["version", "account", "cagr", "mdd", "sharpe", "annualized_turnover", "total_return", "closed_trades", "inverted_trades"]].to_string(index=False))
    print("Saved:", output.resolve())
    return tables


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=HERE.parent / "intraday/data")
    parser.add_argument("--output-dir", type=Path, default=HERE / "results_v3_v4")
    parser.add_argument("--start", default="2022-01-01")
    parser.add_argument("--end", default="2026-12-31")
    parser.add_argument("--initial-capital", type=float, default=100_000_000.)
    args = parser.parse_args()
    run(args.data_dir.resolve(), args.output_dir.resolve(), base.Config(initial_capital=args.initial_capital, start=args.start, end=args.end))
