"""Causal, single-account Kosdaq day & night backtest; local CSVs only."""
from __future__ import annotations

import argparse
import base64
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

HERE = Path(__file__).resolve().parent
SYMBOLS = ("229200", "233740", "251340")
SLEEVES = ("intraday", "overnight", "combined")
VERSIONS = {"V1": "15:15", "V2": "15:19"}
LABELS = {"intraday": "장중", "overnight": "오버나이트", "combined": "결합"}


@dataclass(frozen=True)
class Config:
    initial_capital: float = 100_000_000.0
    commission_bps: float = 1.5
    slippage_bps: float = 1.0
    start: str = "2022-01-01"
    end: str = "2026-12-31"

    def __post_init__(self):
        if not np.isfinite(self.initial_capital) or self.initial_capital <= 0:
            raise ValueError("initial capital must be positive")
        if pd.Timestamp(self.start) > pd.Timestamp(self.end):
            raise ValueError("start must precede end")
        if not all(np.isfinite(v) and 0 <= v < 10000 for v in (self.commission_bps, self.slippage_bps)):
            raise ValueError("costs must be finite, nonnegative and below 10000bps")


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def minute_start(end):
    return pd.Timestamp(end) - pd.Timedelta(minutes=1)


def stamp(day, time):
    return pd.Timestamp(f"{pd.Timestamp(day).date()} {time}")


def prior_features(daily):
    """Yesterday belongs to the last ten COMPLETED sessions; today never does."""
    daily = daily.sort_index()
    return pd.DataFrame({"previous_close": daily.close.shift(1),
                         "ma10": daily.close.rolling(10, min_periods=10).mean().shift(1)})


def day_direction(previous_close, ma10, opening, current):
    if not np.isfinite([previous_close, ma10, opening, current]).all() or opening <= 0:
        return 0
    move = current / opening - 1
    # A 0.3% boundary must survive floating-point division.
    if previous_close > ma10 and move >= .003 - 1e-12:
        return 1
    if previous_close < ma10 and move <= -.003 + 1e-12:
        return -1
    return 0


def ibs_direction(close, high, low):
    if not np.isfinite([close, high, low]).all() or high <= low:
        return np.nan, 0
    ibs = (close - low) / (high - low)
    return ibs, 1 if ibs < .3 else -1 if ibs > .7 else 0


def next_sessions(calendar):
    calendar = pd.DatetimeIndex(calendar).sort_values()
    if calendar.has_duplicates:
        raise ValueError("duplicate daily calendar")
    return dict(zip(calendar[:-1], calendar[1:]))


def buy(cash, reference, config):
    fill = reference * (1 + config.slippage_bps / 10000)
    quantity = cash / (fill * (1 + config.commission_bps / 10000))
    return quantity, fill, quantity * fill * config.commission_bps / 10000


def sell(quantity, reference, config):
    fill = reference * (1 - config.slippage_bps / 10000)
    commission = quantity * fill * config.commission_bps / 10000
    return quantity * fill - commission, fill, commission


def validate_frame(x, symbol, minute):
    required = {"timestamp", "date", "symbol", "source", "open", "high", "low", "close", "volume"}
    if minute:
        required.add("timestamp_semantics")
    if not required <= set(x.columns):
        raise ValueError(f"{symbol}: missing columns {required - set(x.columns)}")
    if not x.symbol.astype(str).str.zfill(6).eq(symbol).all() or not x.source.eq("TOSS").all():
        raise ValueError(f"{symbol}: unexpected symbol/source")
    x = x.copy()
    x["timestamp"] = pd.to_datetime(x.timestamp, errors="raise")
    x["date"] = pd.to_datetime(x.date, errors="raise").dt.normalize()
    if x.timestamp.dt.tz is not None:
        x["timestamp"] = x.timestamp.dt.tz_convert("Asia/Seoul").dt.tz_localize(None)
    if not x.timestamp.dt.normalize().eq(x.date).all():
        raise ValueError(f"{symbol}: timestamp/date mismatch")
    if x.duplicated("timestamp" if minute else "date").any():
        raise ValueError(f"{symbol}: duplicate timestamp/date")
    cols = ["open", "high", "low", "close", "volume"]
    x[cols] = x[cols].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(x[cols].to_numpy()).all() or (x[cols[:4]] <= 0).any().any() or (x.volume < 0).any():
        raise ValueError(f"{symbol}: invalid OHLCV")
    if ((x.high < x[["open", "close", "low"]].max(axis=1)) |
            (x.low > x[["open", "close", "high"]].min(axis=1))).any():
        raise ValueError(f"{symbol}: incoherent OHLC")
    if minute:
        if not x.timestamp_semantics.eq("bar_end").all() or not x.timestamp.eq(x.timestamp.dt.floor("min")).all():
            raise ValueError(f"{symbol}: expected exact minute-end timestamps")
        x["bar_start"] = x.timestamp - pd.Timedelta(minutes=1)
    return x.sort_values("timestamp").reset_index(drop=True)


class Market:
    def __init__(self, directory):
        self.minutes, self.daily, self.groups, self.lookup, self.sources = {}, {}, {}, {}, []
        for symbol in SYMBOLS:
            for kind in ("daily", "minute"):
                path = directory / (f"{symbol}_daily.csv" if kind == "daily" else f"minute/{symbol}_TOSS_1m.csv")
                x = validate_frame(pd.read_csv(path, dtype={"symbol": str}), symbol, kind == "minute")
                self.sources.append({"path": str(path.resolve()), "symbol": symbol, "kind": kind,
                                     "sha256": sha256(path), "rows": len(x),
                                     "first_date": str(x.date.min().date()), "last_date": str(x.date.max().date())})
                if kind == "daily":
                    self.daily[symbol] = x.set_index("date")
                else:
                    self.minutes[symbol] = x
                    self.lookup[symbol] = x.set_index("timestamp", drop=False)
                    self.groups[symbol] = {d: g for d, g in x.groupby("date", sort=False)}
        self.features = prior_features(self.daily["229200"])
        self.next = next_sessions(self.daily["229200"].index)

    def bar(self, symbol, end):
        try:
            return self.lookup[symbol].loc[end]
        except KeyError:
            return None

    def calendar(self, config):
        lower = max(pd.Timestamp(config.start), *(x.date.min() for x in self.minutes.values()))
        upper = min(pd.Timestamp(config.end), *(x.date.max() for x in self.minutes.values()))
        dates = self.daily["229200"].index
        dates = dates[(dates >= lower) & (dates <= upper)]
        if dates.empty:
            raise ValueError("no overlapping daily/minute data in requested date range")
        return dates

    def quality(self, dates):
        rows = []
        for symbol in SYMBOLS:
            for day in dates:
                g = self.groups[symbol].get(day)
                expected = pd.date_range(stamp(day, "09:01"), stamp(day, "15:30"), freq="min")
                present = pd.DatetimeIndex([] if g is None else g.timestamp)
                opening = self.bar(symbol, stamp(day, "09:01"))
                daily = self.daily[symbol]
                rows.append({"symbol": symbol, "date": day, "rows": 0 if g is None else len(g),
                             "missing_standard_minutes": len(expected.difference(present)),
                             "nonstandard_minutes": len(present.difference(expected)),
                             "zero_volume_minutes": 0 if g is None else int(g.volume.eq(0).sum()),
                             "has_0900_open": opening is not None,
                             "daily_minute_open_match": bool(opening is not None and day in daily.index and
                                                            np.isclose(opening.open, daily.loc[day, "open"]))})
        return pd.DataFrame(rows)


def ibs_window(group, day, label):
    """label is a minute START; its CLOSE is available at label+1 minute."""
    end = stamp(day, label) + pd.Timedelta(minutes=1)
    expected = pd.date_range(stamp(day, "09:01"), end, freq="min")
    if group is None:
        return np.nan, 0, "missing_signal_day"
    window = group[group.timestamp.between(expected[0], end)]
    if len(window) != len(expected) or not pd.DatetimeIndex(window.timestamp).equals(expected):
        return np.nan, 0, "incomplete_ibs_window_or_delayed_open"
    ibs, direction = ibs_direction(window.close.iloc[-1], window.high.max(), window.low.min())
    return ibs, direction, "zero_ibs_range" if not np.isfinite(ibs) else ""


def executable(bar):
    return bar is not None and bar.open > 0 and bar.volume > 0


def make_plans(market, dates):
    plans, signals, skips = {}, [], []
    for version, label in VERSIONS.items():
        for day in dates:
            ds = 0
            b0, b5 = market.bar("229200", stamp(day, "09:01")), market.bar("229200", stamp(day, "09:06"))
            feat = market.features.loc[day]
            reason = ""
            if b0 is None or b5 is None:
                reason = "missing_0900_or_0905_signal_open"
            elif not np.isfinite(feat.to_numpy()).all():
                reason = "insufficient_completed_ma10_history"
            else:
                ds = day_direction(feat.previous_close, feat.ma10, b0.open, b5.open)
            day_plan = None
            if ds:
                symbol = "233740" if ds == 1 else "251340"
                entry, exit_ = market.bar(symbol, stamp(day, "09:07")), market.bar(symbol, stamp(day, "15:01"))
                if not executable(entry) or not executable(exit_):
                    reason = "missing_or_zero_volume_intraday_execution"
                else:
                    day_plan = dict(strategy="intraday", symbol=symbol, direction=ds,
                                    entry_time=stamp(day, "09:06"), exit_time=stamp(day, "15:00"),
                                    entry_reference_bar_end=entry.timestamp, exit_reference_bar_end=exit_.timestamp,
                                    entry_reference=float(entry.open), exit_reference=float(exit_.open),
                                    signal_available_time=stamp(day, "09:05"))
            if reason:
                skips.append(dict(version=version, strategy="intraday", date=day, reason=reason))
            ibs, ns, nr = ibs_window(market.groups["229200"].get(day), day, label)
            night_plan = None
            if ns:
                symbol = "233740" if ns == 1 else "251340"
                nxt = market.next.get(day)
                if nxt is None or nxt > dates[-1]:
                    nr = "next_session_outside_sample"
                else:
                    exit_ = market.bar(symbol, stamp(nxt, "09:01"))
                    mark = market.bar(symbol, stamp(day, "15:30"))
                    g = market.groups[symbol].get(day)
                    candidates = None if g is None else g[g.timestamp.between(stamp(day, "09:01"), stamp(day, "15:20")) & g.volume.gt(0)]
                    if not executable(exit_):
                        nr = "missing_or_zero_volume_next_session_0900_exit"
                    elif mark is None:
                        nr = "missing_entry_day_1530_mark"
                    elif candidates is None or candidates.empty:
                        nr = "missing_last_traded_price_before_1520"
                    else:
                        entry = candidates.iloc[-1]
                        night_plan = dict(strategy="overnight", symbol=symbol, direction=ns,
                                          entry_time=stamp(day, "15:20"), exit_time=stamp(nxt, "09:00"),
                                          entry_reference_bar_end=entry.timestamp, exit_reference_bar_end=exit_.timestamp,
                                          entry_reference=float(entry.close), exit_reference=float(exit_.open),
                                          mark_close=float(mark.close), signal_available_time=stamp(day, label) + pd.Timedelta(minutes=1))
            if nr:
                skips.append(dict(version=version, strategy="overnight", date=day, reason=nr))
            signals.append(dict(version=version, date=day, previous_close=feat.previous_close, ma10=feat.ma10,
                                open_0900=np.nan if b0 is None else b0.open,
                                open_0905=np.nan if b5 is None else b5.open,
                                intraday_direction=ds, intraday_accepted=day_plan is not None,
                                intraday_skip_reason=reason, ibs_label=label, ibs=ibs,
                                ibs_available_time=stamp(day, label) + pd.Timedelta(minutes=1),
                                overnight_direction=ns, overnight_accepted=night_plan is not None, overnight_skip_reason=nr))
            plans[version, day] = (day_plan, night_plan)
    return plans, pd.DataFrame(signals), pd.DataFrame(skips, columns=["version", "strategy", "date", "reason"])


def simulate(version, sleeve, dates, plans, config):
    cash, position, previous_nav, peak = config.initial_capital, None, config.initial_capital, config.initial_capital
    daily, orders, trades = [], [], []

    def enter(plan):
        nonlocal cash, position
        if position is not None:
            raise AssertionError("overlapping positions")
        quantity, fill, commission = buy(cash, plan["entry_reference"], config)
        trade_id = f"{version}_{sleeve}_{len(trades) + 1:05d}"
        position = dict(plan, quantity=quantity, entry_fill=fill, entry_commission=commission,
                        entry_equity=cash, trade_id=trade_id)
        orders.append(dict(version=version, account=sleeve, strategy=plan["strategy"], trade_id=trade_id,
                           timestamp=plan["entry_time"], reference_bar_end=plan["entry_reference_bar_end"],
                           reference_field="close" if plan["strategy"] == "overnight" else "open",
                           reference_kind="last_traded_price_proxy" if plan["strategy"] == "overnight" else "minute_open",
                           symbol=plan["symbol"], side="BUY", quantity=quantity, reference_price=plan["entry_reference"],
                           fill_price=fill, filled_notional=quantity * fill, commission=commission,
                           slippage_cost=quantity * (fill - plan["entry_reference"])))
        cash = 0.0

    def exit_position():
        nonlocal cash, position
        p = position
        cash, fill, commission = sell(p["quantity"], p["exit_reference"], config)
        orders.append(dict(version=version, account=sleeve, strategy=p["strategy"], trade_id=p["trade_id"],
                           timestamp=p["exit_time"], reference_bar_end=p["exit_reference_bar_end"],
                           reference_field="open", reference_kind="minute_open", symbol=p["symbol"], side="SELL",
                           quantity=p["quantity"], reference_price=p["exit_reference"], fill_price=fill,
                           filled_notional=p["quantity"] * fill, commission=commission,
                           slippage_cost=p["quantity"] * (p["exit_reference"] - fill)))
        trade = {k: v for k, v in p.items() if k != "mark_close"}
        trade.update(version=version, account=sleeve, exit_fill=fill, exit_commission=commission,
                     exit_equity=cash, net_pnl=cash - p["entry_equity"], net_return=cash / p["entry_equity"] - 1,
                     gross_pnl=p["quantity"] * (p["exit_reference"] - p["entry_reference"]))
        trade["commission_cost"] = commission + p["entry_commission"]
        trade["slippage_cost"] = p["quantity"] * ((p["entry_fill"] - p["entry_reference"]) + (p["exit_reference"] - fill))
        if not np.isclose(trade["net_pnl"], trade["gross_pnl"] - trade["commission_cost"] - trade["slippage_cost"], atol=1e-6):
            raise AssertionError("trade P&L reconciliation failure")
        trades.append(trade)
        position = None

    for day in dates:
        first_order = len(orders)
        if position is not None:
            if position["exit_time"].normalize() != day:
                raise AssertionError("overnight holding extended past next session")
            exit_position()
        day_plan, night_plan = plans[version, day]
        if day_plan is not None and sleeve in ("intraday", "combined"):
            enter(day_plan)
            exit_position()
        if night_plan is not None and sleeve in ("overnight", "combined"):
            enter(night_plan)
        nav = cash if position is None else position["quantity"] * position["mark_close"]
        if not np.isfinite(nav) or nav <= 0:
            raise AssertionError("nonpositive NAV")
        notional = sum(o["filled_notional"] for o in orders[first_order:])
        peak = max(peak, nav)
        daily.append(dict(version=version, account=sleeve, date=day, previous_nav=previous_nav, nav=nav,
                          daily_return=nav / previous_nav - 1, cumulative_pnl=nav - config.initial_capital,
                          drawdown=nav / peak - 1, gross_traded_notional=notional, turnover=notional / previous_nav,
                          orders=len(orders) - first_order, holding_symbol="" if position is None else position["symbol"]))
        previous_nav = nav
    if position is not None:
        raise AssertionError("unclosed final position")
    return pd.DataFrame(daily), pd.DataFrame(orders), pd.DataFrame(trades)


def metrics(frame, trades):
    initial = float(frame.previous_nav.iloc[0])
    nav = frame.nav.to_numpy()
    returns = frame.daily_return
    std = returns.std(ddof=1)
    elapsed_days = (frame.date.iloc[-1] - frame.date.iloc[0]).days + 1
    total_return = nav[-1] / initial - 1
    drawdown = nav / np.maximum.accumulate(np.r_[initial, nav])[1:] - 1
    start, end = frame.date.iloc[0], frame.date.iloc[-1]
    closed = 0 if trades.empty else int(trades.exit_time.dt.normalize().between(start, end).sum())
    return dict(start=str(start.date()), end=str(end.date()), sessions=len(frame),
                starting_nav=initial, ending_nav=nav[-1], pnl=nav[-1] - initial,
                total_return=total_return, cagr=(1 + total_return) ** (365.25 / elapsed_days) - 1,
                mdd=drawdown.min(), sharpe=returns.mean() / std * np.sqrt(252) if std > 0 else np.nan,
                turnover=frame.turnover.sum(), annualized_turnover=frame.turnover.sum() * 252 / len(frame),
                order_count=int(frame.orders.sum()), closed_trades=closed, elapsed_calendar_days=elapsed_days)


def performance_tables(daily, trades):
    summary, yearly = [], []
    for (version, account), frame in daily.groupby(["version", "account"], sort=False):
        selected = trades[(trades.version == version) & (trades.account == account)] if not trades.empty else trades
        keys = dict(version=version, account=account)
        summary.append(dict(keys, **metrics(frame, selected)))
        for year, group in frame.groupby(frame.date.dt.year):
            partial = (year == frame.date.iloc[0].year and frame.date.iloc[0] > pd.Timestamp(year, 1, 7)) or (year == frame.date.iloc[-1].year and frame.date.iloc[-1] < pd.Timestamp(year, 12, 24))
            yearly.append(dict(keys, year=year, partial_year=bool(partial), **metrics(group, selected)))
    return pd.DataFrame(summary), pd.DataFrame(yearly)


def save_charts(daily, directory):
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False})
    colors = {"intraday": "#2769b0", "overnight": "#c17820", "combined": "#15846a"}
    files = []

    def save(fig, name):
        fig.tight_layout()
        for ext in ("png", "svg"):
            fig.savefig(directory / f"{name}.{ext}", dpi=150, bbox_inches="tight")
        files.append(f"{name}.png")
        plt.close(fig)

    for version, cutoff in VERSIONS.items():
        fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
        for sleeve in SLEEVES:
            g = daily[(daily.version == version) & (daily.account == sleeve)]
            axes[0].plot(g.date, g.cumulative_pnl / 1e6, label=sleeve.title(), color=colors[sleeve])
            axes[1].plot(g.date, g.drawdown, color=colors[sleeve], lw=1)
        axes[0].set_title(f"Kosdaq day & night | {version} | IBS minute {cutoff} | Net of costs")
        axes[0].set_ylabel("Cumulative P&L (KRW million)")
        axes[0].legend(frameon=False)
        axes[1].set_ylabel("Daily MTM drawdown")
        axes[1].yaxis.set_major_formatter(PercentFormatter(1))
        for ax in axes:
            ax.grid(alpha=.18)
        save(fig, f"pnl_{version}")
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    for version, color in [("V1", "#15846a"), ("V2", "#8255aa")]:
        g = daily[(daily.version == version) & (daily.account == "combined")]
        axes[0].plot(g.date, g.cumulative_pnl / 1e6, label=version, color=color)
        axes[1].plot(g.date, g.drawdown, color=color, label=version)
    axes[0].set_title("Combined account | V1 vs V2 | Last-traded-price entry proxy")
    axes[0].set_ylabel("Cumulative P&L (KRW million)")
    axes[0].legend(frameon=False)
    axes[1].set_ylabel("Daily MTM drawdown")
    axes[1].yaxis.set_major_formatter(PercentFormatter(1))
    for ax in axes:
        ax.grid(alpha=.18)
    save(fig, "pnl_comparison")
    years = sorted(daily.date.dt.year.unique())
    fig, axes = plt.subplots(len(years), 2, figsize=(14, 3.2 * len(years)), squeeze=False)
    for row, year in enumerate(years):
        for col, version in enumerate(VERSIONS):
            ax = axes[row, col]
            for sleeve in SLEEVES:
                g = daily[(daily.version == version) & (daily.account == sleeve) & daily.date.dt.year.eq(year)]
                ax.plot(g.date, (g.nav - g.previous_nav.iloc[0]) / 1e6, label=sleeve.title(), color=colors[sleeve])
            ax.set_title(f"{year} | {version} | {g.date.iloc[0]:%m-%d} to {g.date.iloc[-1]:%m-%d}")
            ax.set_ylabel("Year P&L (KRW million)")
            ax.grid(alpha=.18)
            if row == 0:
                ax.legend(frameon=False)
    save(fig, "pnl_yearly")
    return files


ASSUMPTIONS = """시그널: KODEX 코스닥150(229200), 매수: 레버리지(233740) 또는 인버스(251340).
V1은 15:15 시작 분봉의 종가(15:16 확정), V2는 15:19 시작 분봉의 종가(15:20 확정)를 사용합니다.
저장된 원본 분봉은 종료 시각이므로 09:06 시가 체결은 원본 09:07 봉 시가입니다.
두 버전의 오버나이트 진입은 사용자가 승인한 15:20 직전 마지막 양의 거래량 분봉 종가를 참고가격으로 사용합니다.
이는 실제 15:20 시장가 체결을 보장하지 않는 연구용 가격 가정이며 V1의 원안 대비 승인된 체결가격 변경입니다.
V2는 15:20 신호 확정 직후 지연 없이 해당 참고가격으로 거래한다고 가정합니다.
각 거래에 계좌 현금 전액을 재투자하며 소수점 수량을 허용합니다. 추가 차입은 없고 현금 이자는 0입니다.
매수·매도 각각 수수료 1.5bp와 불리한 슬리피지 1bp를 적용합니다. 가격이 같으면 왕복 비용은 약 5bp입니다.
수수료를 포함하여 매수 수량을 계산하며 ETF 실제 가격을 사용합니다. 별도 배당·세금은 더하지 않습니다.
결합 계좌는 09:00 오버나이트 청산 → 09:06 장중 매수 → 15:00 장중 청산 → 15:20 오버나이트 매수 순으로 운용합니다.
일별 NAV는 15:30 종가 평가액입니다. 오버나이트 진입 당일의 비용·평가손익과 다음 거래일 청산손익을 각각 귀속합니다.
매매하지 않는 거래일도 일별 수익률에 포함합니다. 연도 사이 계좌 자산은 이어지며 연도별 수익률 기준은 전년도 마지막 NAV입니다.
CAGR=(기말NAV/기초NAV)^(365.25/관측기간 달력일수)-1; 달력일수는 첫 NAV 날짜의 전날부터 마지막 NAV 날짜까지입니다.
연도별 total_return은 실제 해당 연도 수익률이고 CAGR은 같은 공식을 적용한 연환산 값입니다. 부분연도의 CAGR은 실제 수익률과 다릅니다.
MDD는 기초자산을 고점에 포함한 일별 평가 NAV 낙폭이며 장중 최대 손실은 아닙니다.
Sharpe=일별 수익률 평균/표본표준편차(ddof=1)×sqrt(252), 무위험수익률 0입니다. 수익률 분산이 0이면 정의되지 않아 NaN으로 표시합니다.
Turnover(x)=Σ(당일 매수·매도 체결대금 합계/전일NAV); annualized_turnover=turnover×252/거래일수입니다.
매수·매도 양방향 합산 회전율이므로 한 번 전액 매수·청산하면 약 2x입니다. 거래횟수와는 다른 지표입니다.
필수 체결봉이 없거나 거래량이 0이면 거래를 제외합니다. IBS는 09:00부터 해당 신호 분봉까지 모두 존재해야 계산합니다.
오버나이트의 다음 거래일은 필터링 전 229200 일봉 달력으로 정하며, 그날 09:00 체결봉이 없으면 신규 진입을 제외합니다.
이 사전 데이터 완전성 검사는 체결 가능성을 보장하지 않으며, 거래량 제한·시장 충격·호가잔량은 별도 모형화하지 않습니다.
마지막 표본일에는 표본 밖 청산이 필요한 오버나이트 신규 진입을 제외합니다. 제외 내역은 skips.csv에 기록합니다.
2022년 1월부터의 분봉은 보유하지 않아 2022년은 11월부터, 2026년은 데이터 종료일까지의 부분연도입니다.
"""


def formatted_table(frame):
    columns = [c for c in ["version", "account", "year", "partial_year", "start", "end", "sessions", "total_return", "cagr", "mdd", "sharpe", "turnover", "annualized_turnover", "closed_trades", "pnl"] if c in frame]
    x = frame[columns].copy()
    for c in ("total_return", "cagr", "mdd"):
        x[c] = x[c].map(lambda v: f"{v:.2%}")
    for c in ("sharpe", "turnover", "annualized_turnover"):
        x[c] = x[c].map(lambda v: "N/A" if not np.isfinite(v) else f"{v:.2f}")
    x["pnl"] = x.pnl.map(lambda v: f"{v:,.0f}")
    x["account"] = x.account.map(LABELS)
    return x.rename(columns={"version": "버전", "account": "전략", "year": "연도", "partial_year": "부분연도", "start": "시작일", "end": "종료일", "sessions": "거래일", "total_return": "수익률", "cagr": "CAGR", "mdd": "MDD", "sharpe": "Sharpe", "turnover": "회전율(x)", "annualized_turnover": "연환산 회전율(x/년)", "closed_trades": "청산거래수", "pnl": "P&L(원)"})


def markdown_table(x):
    header = "| " + " | ".join(x.columns) + " |"
    return "\n".join([header, "| " + " | ".join(["---"] * len(x.columns)) + " |"] + ["| " + " | ".join(map(str, row)) + " |" for row in x.itertuples(index=False, name=None)])


def write_reports(summary, yearly, charts, directory, skips, config):
    full, annual = formatted_table(summary), formatted_table(yearly)
    period = f"{summary.start.iloc[0]} ~ {summary.end.iloc[0]} ({int(summary.sessions.iloc[0])}거래일)"
    warning = "15:20 진입은 승인된 직전 체결가 참고가격 가정입니다. 2022·2026년은 부분연도이며 모든 성과는 비용 차감 후입니다."
    intro = f"공통 평가 구간: {period}. 각 계좌 초기 자산: {config.initial_capital:,.0f}원."
    comparison = summary[summary.account.eq("combined")].set_index("version")
    result = f"결합 계좌 CAGR은 V1 {comparison.loc['V1', 'cagr']:.2%}, V2 {comparison.loc['V2', 'cagr']:.2%}입니다."
    md = ["# Kosdaq day & night", "", intro, "", warning, "", result, "", "## 전체 성과", "", markdown_table(full), "", "## 연도별 개별·결합 성과", "", markdown_table(annual), "", "## P&L 및 낙폭", ""]
    for chart in charts:
        md += [f"![{chart}]({chart})", ""]
    md += ["## 산식·체결·데이터 가정", "", ASSUMPTIONS, "", f"제외 내역 {len(skips):,}행(버전별 신호/데이터 제외 사유).", "", "KRX는 15:20~15:30을 종가 동시호가 구간으로 구분합니다: [KRX 거래 안내](https://global.krx.co.kr/contents/GLB/01/0109/0109000000/guide_to_trading_in_the_korean_stock_market.pdf).", "", "기계 판독 원자료: summary.csv, yearly.csv, daily_nav.csv, orders.csv, trades.csv, signals.csv, skips.csv, data_quality.csv, manifest.json."]
    (directory / "report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    pictures = "".join(f'<img alt="{f}" src="data:image/png;base64,{base64.b64encode((directory/f).read_bytes()).decode()}">' for f in charts)
    assumptions = "".join(f"<li>{line}</li>" for line in ASSUMPTIONS.strip().splitlines())
    html = f'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Kosdaq day & night</title>
<style>body{{font:15px/1.65 system-ui,sans-serif;color:#17352d;background:#f5f7f5;margin:0;padding:28px}}main{{max-width:1480px;margin:auto}}h1{{font-size:32px}}h2{{margin-top:40px}}.note{{background:#fff3cd;padding:16px;border-left:4px solid #b17b13}}.scroll{{overflow-x:auto;background:white}}table{{border-collapse:collapse;white-space:nowrap;font-size:13px;width:100%}}th,td{{padding:9px 12px;text-align:right;border-bottom:1px solid #ddd}}th{{background:#e7eeea}}img{{width:100%;height:auto;background:white;margin:16px 0}}li{{margin:8px 0}}</style>
<main><h1>Kosdaq day & night</h1><p>{intro}</p><p class="note">{warning}</p><p>{result}</p><p>V1: IBS 15:15 분봉 종가 · V2: IBS 15:19 분봉 종가. 표의 회전율은 매수·매도 양방향 합계입니다.</p>
<h2>전체 개별·결합 성과</h2><div class="scroll">{full.to_html(index=False, border=0)}</div>
<h2>연도별 개별·결합 성과</h2><div class="scroll">{annual.to_html(index=False, border=0)}</div>
<h2>P&L 및 일별 평가 낙폭</h2>{pictures}<h2>산식·체결·데이터 가정</h2><ul>{assumptions}</ul>
<p>제외 내역 {len(skips):,}행은 skips.csv에 기록했습니다.</p><p>거래시간 참고: <a href="https://global.krx.co.kr/contents/GLB/01/0109/0109000000/guide_to_trading_in_the_korean_stock_market.pdf">KRX 거래 안내</a>의 15:20~15:30 종가 동시호가 구간.</p>
<p>모든 그래프는 이 HTML에 포함되어 있어 인터넷 연결 없이 열 수 있습니다. 수치 원본은 같은 폴더의 CSV 및 manifest.json에 저장됩니다.</p></main></html>'''
    (directory / "report.html").write_text(html, encoding="utf-8")


def run(data_dir, output_dir, config):
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "acceptance.json").write_text(json.dumps({
        "status": "PENDING", "reason": "Backtest regenerated; independent checks and owner review must match current output hashes."
    }, indent=2) + "\n", encoding="utf-8")
    market = Market(data_dir)
    dates = market.calendar(config)
    plans, signals, skips = make_plans(market, dates)
    daily_parts, order_parts, trade_parts = [], [], []
    for version in VERSIONS:
        for sleeve in SLEEVES:
            daily, orders, trades = simulate(version, sleeve, dates, plans, config)
            daily_parts.append(daily)
            if not orders.empty:
                order_parts.append(orders)
            if not trades.empty:
                trade_parts.append(trades)
    daily = pd.concat(daily_parts, ignore_index=True)
    orders = pd.concat(order_parts, ignore_index=True) if order_parts else pd.DataFrame(columns=["version", "account", "timestamp"])
    trades = pd.concat(trade_parts, ignore_index=True) if trade_parts else pd.DataFrame(columns=["version", "account", "exit_time"])
    summary, yearly = performance_tables(daily, trades)
    tables = dict(summary=summary, yearly=yearly, daily_nav=daily, orders=orders, trades=trades,
                  signals=signals, skips=skips, data_quality=market.quality(dates))
    for name, table in tables.items():
        table.to_csv(output_dir / f"{name}.csv", index=False, encoding="utf-8-sig", lineterminator="\n")
    charts = save_charts(daily, output_dir)
    write_reports(summary, yearly, charts, output_dir, skips, config)
    manifest = dict(generated_at_utc=datetime.now(timezone.utc).isoformat(), config=asdict(config),
                    common_start=str(dates[0].date()), common_end=str(dates[-1].date()), sessions=len(dates),
                    source_files=market.sources, code_sha256=sha256(Path(__file__)), versions=VERSIONS,
                    python_libraries={"pandas": pd.__version__, "numpy": np.__version__, "matplotlib": matplotlib.__version__},
                    assumptions=ASSUMPTIONS.strip().splitlines(), signal_instrument="229200",
                    commission_bps_per_side=config.commission_bps, slippage_bps_per_side=config.slippage_bps,
                    timing_semantics="minute-start economics; source minute-end = minute-start + 1 minute; 15:30 close is actual EOD auction mark",
                    execution_approximation="USER APPROVED: overnight entry uses last positive-volume close with bar_end <=15:20; idealized reference-price fill",
                    partial_years=sorted(yearly.loc[yearly.partial_year, "year"].unique().tolist()),
                    selected_worker_model="opencode/deepseek-v4.1-flash-free", worker_acceptance="FAIL: reasoning output limit, no code",
                    implementation_owner="Codex direct completion, transparently announced after two exhausted repair delegations",
                    owner_acceptance="PENDING independent verification; see acceptance.json")
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(summary[["version", "account", "total_return", "cagr", "mdd", "sharpe", "annualized_turnover", "closed_trades"]].to_string(index=False))
    print(f"Saved: {output_dir.resolve()}")
    return tables


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=HERE.parent / "intraday/data")
    parser.add_argument("--output-dir", type=Path, default=HERE / "results")
    parser.add_argument("--initial-capital", type=float, default=100_000_000)
    parser.add_argument("--start", default="2022-01-01")
    parser.add_argument("--end", default="2026-12-31")
    args = parser.parse_args()
    run(args.data_dir.resolve(), args.output_dir.resolve(), Config(initial_capital=args.initial_capital, start=args.start, end=args.end))


if __name__ == "__main__":
    main()
