#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""FOREIGN_POS_BOTH_POS: original QQQ/SPY and US-close-anchored ETF proxy only.
Round trip: 3bp commission + 2bp slippage. Separate accounts, exit-date P&L.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# Config
# ============================================================

DATA_DIR = Path(__file__).resolve().parent
import sys
sys.path.insert(0, str(DATA_DIR))
from jehyeong.overnight.jehyeong_backtest_utils import (
    COMMISSION_BP, SLIPPAGE_BP, ROUND_TRIP_COST, account_returns,
    performance_metrics, metrics_markdown,
)
DATA = DATA_DIR / "data"
RESULTS = DATA_DIR / "results"
OUT_DIR = RESULTS / "analysis_output_foreign_preus_pnl"


RESEARCH_CANDIDATES = [
    DATA / "research_dataset.csv"
]

FOREIGN_CANDIDATES = [
    DATA / "KRX_foreign.xlsx"
]

PROXY_CANDIDATES = [
    DATA / "proxy_panel.csv"
]

FOREIGN_DATE_COL = "일자"
FOREIGN_VALUE_COL = "외국인 합계"

BACKTEST_START = pd.Timestamp("2021-01-01")
PROXY_CODES = ("304940", "219480")  # QQQ proxy, SPY proxy
PROXY_RETURN_COLS = [
    f"{code}_asof_return_from_previous_krx_close_1520"
    for code in PROXY_CODES
]
ORIGINAL_RETURN_COLS = ["QQQ_preus_ret_1520", "SPY_preus_ret_1520"]
US_SYMBOLS = ("QQQ", "SPY")
ANCHORED_RETURN_COLS = [f"{symbol}_proxy_preus_log_1520" for symbol in US_SYMBOLS]
US_DAILY_CANDIDATES = {symbol: [DATA / f"{symbol}_daily.csv"] for symbol in US_SYMBOLS}

# Recalculate costs from gross kr_gap; never subtract fees twice.


# ============================================================
# Load
# ============================================================

def first_existing(candidates: list[Path]) -> Path:
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        "file not found:\n"
        + "\n".join(str(p) for p in candidates)
    )


def load_research(include_original: bool = False) -> tuple[pd.DataFrame, Path]:
    path = first_existing(RESEARCH_CANDIDATES)
    df = pd.read_csv(path)

    if "date" not in df.columns:
        raise ValueError(
            f"{path}: date column 없음. columns={list(df.columns)}"
        )

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce",
    ).dt.normalize()

    df = (
        df.dropna(subset=["date"])
        .drop_duplicates("date", keep="last")
        .set_index("date")
        .sort_index()
    )

    required = {
        "KODEX_close",
        "kr_gap",
        "net_gap",
    }
    if include_original:
        required.update(ORIGINAL_RETURN_COLS)

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"{path}: missing={sorted(missing)}"
        )

    # QQQ/SPY columns must not limit the proxy backtest's observation period.
    selected = set(required)
    if include_original:
        selected.update(
            col for symbol in US_SYMBOLS
            for col in (f"{symbol}_prev_close", f"{symbol}_preus_last_price")
            if col in df.columns
        )
    return df[sorted(selected)].apply(pd.to_numeric, errors="coerce"), path


def load_proxy() -> tuple[pd.DataFrame, Path]:
    path = first_existing(PROXY_CANDIDATES)
    df = pd.read_csv(path)
    required = {"date", *PROXY_RETURN_COLS}
    for code in PROXY_CODES:
        required.update({
            f"{code}_asof_usable_1520",
            f"{code}_asof_last_trade_available_at_1520",
        })

    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing={sorted(missing)}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df = (
        df.dropna(subset=["date"])
        .drop_duplicates("date", keep="last")
        .set_index("date")
        .sort_index()
    )
    out = pd.DataFrame(index=df.index)
    session_date = df.index.tz_localize("Asia/Seoul")
    cutoff = session_date + pd.Timedelta(hours=15, minutes=20)

    for code, return_col in zip(PROXY_CODES, PROXY_RETURN_COLS):
        usable_col = f"{code}_asof_usable_1520"
        time_col = f"{code}_asof_last_trade_available_at_1520"
        usable = (
            df[usable_col].astype("string").str.strip().str.lower()
            .isin(["true", "1"])
        )
        last_trade = pd.to_datetime(
            df[time_col], errors="coerce", utc=True,
        ).dt.tz_convert("Asia/Seoul")
        proxy_return = pd.to_numeric(df[return_col], errors="coerce")

        # Accept an earlier same-day trade, but never a future/post-close price.
        out[usable_col] = (
            usable
            & (last_trade.dt.normalize() == session_date)
            & (last_trade <= cutoff)
            & np.isfinite(proxy_return)
        )
        out[return_col] = proxy_return.where(out[usable_col])
        out[time_col] = last_trade
        for price_col in (f"{code}_previous_krx_close", f"{code}_asof_price_1520"):
            if price_col in df:
                out[price_col] = pd.to_numeric(df[price_col], errors="coerce")

    return out, path


def load_us_daily(symbol: str) -> tuple[pd.DataFrame, Path]:
    path = first_existing(US_DAILY_CANDIDATES[symbol])
    df = pd.read_csv(path)
    missing = {"date", "close"} - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing={sorted(missing)}")
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["date"]).drop_duplicates("date", keep="last").sort_values("date")
    df["close"] = df["close"].where(np.isfinite(df["close"]) & df["close"].gt(0))
    # date labels US sessions. Convert the NY close to its actual KST availability;
    # DST is handled by the timezone, not a fixed calendar-day or hour shift.
    # Using 16:00 on early-close days is conservative and still before the KRX cutoff.
    df["available_at"] = (
        (df["date"] + pd.Timedelta(hours=16)).dt.tz_localize("America/New_York")
        .dt.tz_convert("Asia/Seoul")
    )
    return df[["date", "close", "available_at"]], path


def add_us_close_anchored_returns(
    proxy: pd.DataFrame,
    krx_calendar: pd.DatetimeIndex,
    us_daily: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Scale US closes into proxy units using the previous actual KRX close.

    log(P_proxy_T_1520 / P_proxy_prev_KRX_close)
      - log(C_US_latest_at_T_1520 / C_US_latest_at_prev_KRX_close).

    This is an estimated signal, not an observed QQQ/SPY PREUS return. Its base
    assumes the previous KRX proxy close maps to the US close known at that time.
    """
    calendar = pd.DatetimeIndex(krx_calendar).sort_values()
    previous_date = calendar.to_series().shift(1)
    queries = pd.DataFrame({
        "date": calendar,
        "signal_at": calendar.tz_localize("Asia/Seoul") + pd.Timedelta(hours=15, minutes=20),
        "previous_krx_close_at": pd.DatetimeIndex(previous_date).tz_localize("Asia/Seoul")
        + pd.Timedelta(hours=15, minutes=30),
    })
    out = proxy.copy()
    for symbol, code, raw_col, anchored_col in zip(US_SYMBOLS, PROXY_CODES, PROXY_RETURN_COLS, ANCHORED_RETURN_COLS):
        daily = us_daily[symbol].rename(columns={"date": "us_session_date"}).sort_values("available_at")
        if daily.empty:
            raise ValueError(f"{symbol}: US daily data empty")
        for query_col, suffix in (("signal_at", "at_signal"), ("previous_krx_close_at", "at_previous_krx_close")):
            aligned = pd.merge_asof(
                queries[["date", query_col]].dropna(subset=[query_col]).sort_values(query_col),
                daily, left_on=query_col, right_on="available_at", direction="backward",
            ).set_index("date")
            for source_col, label in (("close", "us_close"), ("us_session_date", "us_session_date"), ("available_at", "us_close_available_at")):
                out[f"{symbol}_{label}_{suffix}"] = aligned[source_col].reindex(out.index)

        latest = out[f"{symbol}_us_close_at_signal"]
        base = out[f"{symbol}_us_close_at_previous_krx_close"]
        # Do not extrapolate the last input close into future missing US sessions.
        # A next-weekday boundary permits the normal Friday -> KRX Monday case.
        coverage_end = daily["us_session_date"].max() + pd.offsets.BDay(1)
        valid = (out.index <= coverage_end) & latest.gt(0) & base.gt(0) & out[raw_col].gt(-1)
        us_gross = (latest / base).where(valid)
        proxy_gross = (1.0 + out[raw_col]).where(valid)
        out[f"{symbol}_us_close_change_log"] = np.log(us_gross)
        out[f"{symbol}_estimated_price_1520_us_units"] = base * proxy_gross
        out[anchored_col] = np.log(proxy_gross) - np.log(us_gross)
        price_col = f"{code}_previous_krx_close"
        if price_col in out:
            out[f"{code}_us_close_reference_in_proxy_units"] = out[price_col] * us_gross
    return out


def load_foreign() -> tuple[pd.DataFrame, Path]:
    path = first_existing(FOREIGN_CANDIDATES)
    df = pd.read_excel(path)

    if not {
        FOREIGN_DATE_COL,
        FOREIGN_VALUE_COL,
    }.issubset(df.columns):
        raise ValueError(
            f"{path}: columns={list(df.columns)}"
        )

    date = pd.to_datetime(
        df[FOREIGN_DATE_COL],
        errors="coerce",
    ).dt.normalize()

    value = pd.to_numeric(
        df[FOREIGN_VALUE_COL]
        .astype(str)
        .str.replace(",", "", regex=False),
        errors="coerce",
    )

    out = pd.DataFrame({
        "date": date,
        "foreign_netbuy": value,
    })

    out = (
        out.dropna()
        .drop_duplicates("date", keep="last")
        .set_index("date")
        .sort_index()
    )

    return out, path


def build_backtest_dataset(research, proxy, foreign, proxy_return_cols=None):
    return_cols = ANCHORED_RETURN_COLS if proxy_return_cols is None else proxy_return_cols
    x = research.sort_index().join(proxy, how="left").join(
        foreign.rename(columns={"foreign_netbuy": "foreign_current"}), how="left",
    )
    x["foreign_prev"] = x["foreign_current"].shift(1)
    x["foreign_prev_date"] = x.index.to_series().shift(1)
    x["net_gap"] = x["kr_gap"] - ROUND_TRIP_COST
    features = ["foreign_prev", *return_cols]
    valid = (x.index >= BACKTEST_START) & x[[*features, "net_gap", "KODEX_close"]].notna().all(axis=1)
    if not valid.any():
        raise RuntimeError("2021년 이후 전략의 공통 데이터 없음")
    d = x.loc[x.index[valid].min():x.index[valid].max()].dropna(subset=["net_gap", "KODEX_close"]).copy()
    d["available"] = d[features].notna().all(axis=1)
    d["signal"] = d["available"] & d["foreign_prev"].gt(0) & d[return_cols].gt(0).all(axis=1)
    return d


def build_comparison_dataset(research, proxy, foreign):
    original = build_backtest_dataset(research, proxy, foreign, ORIGINAL_RETURN_COLS)
    anchored = build_backtest_dataset(research, proxy, foreign, ANCHORED_RETURN_COLS)
    start, end = max(original.index.min(), anchored.index.min()), min(original.index.max(), anchored.index.max())
    d = original.loc[start:end].drop(columns=["available", "signal"]).copy()
    for prefix, model in (("original", original), ("anchored", anchored)):
        d[f"{prefix}_available"] = model["available"].reindex(d.index).fillna(False)
        d[f"{prefix}_signal"] = model["signal"].reindex(d.index).fillna(False)
    d["common_available"] = d["original_available"] & d["anchored_available"]
    for prefix in ("original", "anchored"):
        d[f"{prefix}_common_signal"] = d[f"{prefix}_signal"] & d["common_available"]
    return d


def overnight_trades(model, calendar, name):
    next_session = pd.DatetimeIndex(calendar).to_series().shift(-1)
    selected = model.loc[model["signal"]].copy()
    trades = pd.DataFrame({
        "strategy": name, "entry_date": selected.index,
        "exit_date": next_session.reindex(selected.index).values,
        "entry_time": "KRX_CLOSE", "exit_time": "NEXT_KRX_OPEN", "position": 1.0,
        "entry_price": selected["KODEX_close"].values,
        "exit_price": (selected["KODEX_close"] * (1.0 + selected["kr_gap"])).values,
        "gross_return": selected["kr_gap"].values,
        "commission": COMMISSION_BP / 10000.0,
        "slippage": SLIPPAGE_BP / 10000.0,
        "net_return": selected["net_gap"].values,
        "foreign_prev_date": selected["foreign_prev_date"].values,
        "foreign_prev": selected["foreign_prev"].values,
    })
    if trades["exit_date"].isna().any():
        raise ValueError("Overnight trade lacks an actual next KRX session")
    return trades


def prepare_strategies():
    research, research_path = load_research(include_original=True)
    raw_proxy, proxy_path = load_proxy()
    foreign, foreign_path = load_foreign()
    us_daily, paths = {}, {}
    for symbol in US_SYMBOLS:
        us_daily[symbol], paths[symbol] = load_us_daily(symbol)
    proxy = add_us_close_anchored_returns(raw_proxy, research.index, us_daily)
    models = {
        "US_CLOSE_PROXY": build_backtest_dataset(research, proxy, foreign, ANCHORED_RETURN_COLS),
        "QQQ_SPY_ORIGINAL": build_backtest_dataset(research, proxy, foreign, ORIGINAL_RETURN_COLS),
    }
    ledgers = {name: overnight_trades(model, research.index, name) for name, model in models.items()}
    next_session = research.index.to_series().shift(-1)
    coverage = {
        name: (model.index.min(), next_session.loc[model.index.max()])
        for name, model in models.items()
    }
    return {
        "research": research, "proxy": proxy, "foreign": foreign, "models": models,
        "ledgers": ledgers, "coverage": coverage,
        "paths": {"research": research_path, "proxy": proxy_path, "foreign": foreign_path, **paths},
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    prepared = prepare_strategies()
    research, models, ledgers, coverage = (prepared[key] for key in ("research", "models", "ledgers", "coverage"))
    labels = {"US_CLOSE_PROXY": "미국 종가 기준 보정 proxy", "QQQ_SPY_ORIGINAL": "기존 QQQ·SPY PREUS"}
    full_rows, full_daily = [], {}
    for name in models:
        start, end = coverage[name]
        daily, trades = account_returns(ledgers[name], research.index, start, end)
        row = performance_metrics(daily, trades)
        row.update(strategy=name, label=labels[name], scope="FULL_AVAILABLE")
        full_rows.append(row)
        full_daily[name] = daily
    summary = pd.DataFrame(full_rows)
    summary.to_csv(OUT_DIR / "performance_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(full_daily).to_csv(OUT_DIR / "daily_pnl.csv", index_label="date", encoding="utf-8-sig")
    pd.concat(ledgers.values(), ignore_index=True).to_csv(OUT_DIR / "strategy_trade_log.csv", index=False, encoding="utf-8-sig")

    comparison = build_comparison_dataset(research, prepared["proxy"], prepared["foreign"])
    comparison.to_csv(OUT_DIR / "strategy_signal_panel.csv", index_label="date", encoding="utf-8-sig")
    start = max(value[0] for value in coverage.values())
    end = min(value[1] for value in coverage.values())
    shared_entries = comparison.index[comparison["common_available"]]
    matched_rows, series = [], {}
    for scope, eligible in (("MATCHED_PERIOD", None), ("COMMON_FEATURE_DAYS", shared_entries)):
        for name in models:
            daily, trades = account_returns(ledgers[name], research.index, start, end, eligible)
            row = performance_metrics(daily, trades)
            row.update(strategy=name, label=labels[name], scope=scope)
            matched_rows.append(row)
            series[(scope, name)] = daily
    matched = pd.DataFrame(matched_rows)
    matched.to_csv(OUT_DIR / "strategy_comparison_summary.csv", index=False, encoding="utf-8-sig")
    wide = pd.DataFrame({f"{scope}__{name}": value for (scope, name), value in series.items()})
    wide.to_csv(OUT_DIR / "strategy_comparison_daily.csv", index_label="date", encoding="utf-8-sig")

    common = comparison.loc[comparison["common_available"]]
    disagree = common["original_signal"].ne(common["anchored_signal"])
    common.loc[disagree].to_csv(OUT_DIR / "entry_disagreements.csv", index_label="date", encoding="utf-8-sig")
    foreign_positive = common.loc[common["foreign_prev"].gt(0)]
    conditional_agreement = foreign_positive["original_signal"].eq(foreign_positive["anchored_signal"]).mean()
    period_metrics = matched.loc[matched["scope"] == "MATCHED_PERIOD"].set_index("strategy")
    original_metrics, proxy_metrics = period_metrics.loc["QQQ_SPY_ORIGINAL"], period_metrics.loc["US_CLOSE_PROXY"]
    reference_rows = []
    for symbol in US_SYMBOLS:
        old_col, new_col = f"{symbol}_prev_close", f"{symbol}_us_close_at_signal"
        if old_col in research:
            reference = research[[old_col]].join(prepared["proxy"][[new_col]], how="inner").dropna()
            same = np.isclose(reference[old_col], reference[new_col], rtol=1e-9, atol=1e-9)
            reference_rows.append({"symbol": symbol, "checked_days": len(reference), "same_reference_days": int(same.sum())})
    pd.DataFrame(reference_rows).to_csv(OUT_DIR / "us_close_reference_audit.csv", index=False, encoding="utf-8-sig")
    fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True)
    for ax, scope, title in zip(axes, ("MATCHED_PERIOD", "COMMON_FEATURE_DAYS"), ("Same account evaluation period", "Shared feature entry days only")):
        for name, color in (("QQQ_SPY_ORIGINAL", "tab:blue"), ("US_CLOSE_PROXY", "tab:green")):
            daily = series[(scope, name)]
            ax.plot(daily.index, (1.0 + daily).cumprod(), label=name, color=color)
        ax.set_title(title)
        ax.set_ylabel("Growth of 1.0")
        ax.grid(alpha=0.25)
        ax.legend()
    axes[1].set_xlabel("KRX realization date")
    fig.suptitle("FOREIGN_POS_BOTH_POS: 3bp commission + 2bp slippage")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "foreign_pos_both_pos_pnl.png", dpi=170)
    plt.close(fig)

    lines = [
        "# FOREIGN_POS_BOTH_POS: 두 전략의 설명과 성과", "",
        "## 비교 결과의 해석", "",
        f"같은 기간에서 기존 전략 CAGR {original_metrics['CAGR']:.2%}, 보정 proxy {proxy_metrics['CAGR']:.2%}로 "
        f"proxy가 {(original_metrics['CAGR'] - proxy_metrics['CAGR']) * 100:.2f}%p 낮다. "
        f"MDD도 기존 {original_metrics['MDD']:.2%} 대비 proxy {proxy_metrics['MDD']:.2%}로 깊다. "
        "보정으로 비교할 수 있는 신호를 만들었지만, 기존 전략과 같은 성과·진입일을 복원했다고 볼 수 없다.",
        f"공통 feature 날짜 중 외국인 순매수가 양수인 {len(foreign_positive)}일에서 진입 결정 일치율은 {conditional_agreement:.2%}다. "
        "양쪽 모두 쉬는 외국인 비양수 날짜를 제외해 비교했다. 불일치 일자는 entry_disagreements.csv에서 확인할 수 있다.",
        "직전 미국 종가 대조: " + "; ".join(
            f"{row['symbol']} {row['same_reference_days']}/{row['checked_days']}일 일치" for row in reference_rows
        ) + ". 같은 한국 의사결정일에 사용한 미국 기준 종가를 대조한 결과다. "
        "외국인 수급은 두 전략 모두 원래의 전체 KRX 달력에서 1일 lag하므로 자료를 자른 후의 하루 이동은 없다. "
        "남은 차이는 ETF의 미국 선물·추적·헤지 특성과 직전 한국 종가의 괴리 잔차가 직접 QQQ/SPY PREUS와 다르게 움직이는 데서 올 수 있다. "
        "이 비교만으로 각 원인의 기여도를 분리할 수는 없다.", "",
        "## 공통 거래 규칙", "",
        "- 외국인 조건: KRX_foreign.xlsx의 직전 실제 KRX 거래일 외국인 순매수 > 0.",
        "- 미국 방향 조건: 15:20까지 관측한 두 지표가 모두 양수이면 당일 KODEX200 종가 매수, 다음 실제 KRX 거래일 시가 매도.",
        "- 롱 전용, 거래당 계좌 자본 100%, 미거래일은 현금 수익률 0. 전략마다 별도 계좌를 두며 계좌 간 주문·포지션을 합치지 않는다.",
        "- 비용: 왕복 수수료 3bp + 왕복 슬리피지 2bp = 5bp. 기존 net_gap을 재차 차감하지 않고 kr_gap에서 정확히 0.0005를 한 번 차감한다.",
        "", "## 1. 기존 QQQ·SPY 전략", "",
        "QQQ와 SPY 각각 직전 미국 정규장 종가 대비 한국 당일 15:20까지 마지막 가격의 로그수익률을 사용한다. "
        "research_dataset.csv의 QQQ_preus_ret_1520·SPY_preus_ret_1520 두 값이 모두 > 0이어야 한다. "
        "현재 직접 관측한 PREUS는 2023년부터 이용 가능하다.",
        "", "## 2. 미국 종가 기준 보정 proxy 전략", "",
        "QQQ proxy 304940와 SPY proxy 219480의 15:20까지 마지막 실제 거래 가격을 사용한다. "
        "직전 한국 종가 이후 미국 종가까지의 움직임을 미국 일봉으로 제거한다.", "",
        "```text", "A = 직전 KRX 종가 당시 이미 알려진 QQQ/SPY 종가",
        "B = 오늘 15:20 당시 이미 알려진 QQQ/SPY 종가",
        "E0 = 직전 실제 KRX 거래일 proxy ETF 종가",
        "E1 = 오늘 15:20까지 proxy ETF 마지막 거래 가격",
        "보정 지표 = log(E1 / E0) - log(B / A)",
        "          = log(E1 / (E0 * B / A))", "```", "",
        "미국 일봉 날짜는 미국 세션 날짜이며, 뉴욕 16:00 마감을 Asia/Seoul로 변환하여 의사결정 시각 이전 종가만 조회한다. "
        "서머타임·한미 휴일을 반영한다. 거래량이 없는 15:20 봉은 이전의 당일 실제 거래를 사용하며, 미래·전일 체결 가격은 제외한다.", "",
        "이 값은 관측된 QQQ/SPY PREUS를 그대로 복원한 수익률은 아니다. E0를 A에 대응시킨 가정과 ETF 추적 차이의 잔차가 남으므로 "
        "두 전략의 진입일이 완전히 같을 수는 없다. 미래 가격이나 성과에 맞춘 임계값은 사용하지 않는다.",
        "", "## 평가 규약", "",
        "모든 거래는 진입일·청산일을 별도로 기록한다. 손익과 왕복 비용은 청산일에 반영한다. "
        "평가 시작일 이전에 진입한 거래와 평가 종료일 이후에 청산하는 거래를 제외하여 각 계좌를 같은 시점에 현금 1로 시작한다.",
        "CAGR = 최종자산^(252 / 전체 KRX 평가일 수) − 1. Sharpe = 일별 순수익률 평균 / 표본 SD × √252, 무위험수익률 0. "
        "MDD에는 최초 자산 1을 포함하고 Calmar = CAGR / |MDD|이다. 미거래일도 모든 지표에 포함한다.",
        "이번 성과는 이전 대화의 왕복 3bp·진입일 손익 표기와 비용 및 일자 정렬 규약이 달라 직접 같은 숫자로 비교하면 안 된다.",
        "", "## 각 전략의 최대 이용 가능 기간", "",
        metrics_markdown(summary, "label"), "", "## 동일 기간 비교", "",
        metrics_markdown(matched.loc[matched["scope"] == "MATCHED_PERIOD"], "label"),
        "", "## 공통 feature 날짜에만 진입", "",
        f"공통 feature {len(common)}일. 결측일에는 양쪽 모두 신규 진입하지 않으며, 전체 KRX 평가 달력은 유지한다.", "",
        metrics_markdown(matched.loc[matched["scope"] == "COMMON_FEATURE_DAYS"], "label"), "",
        f"공통 자료 기준 동시 진입 {int((common['original_signal'] & common['anchored_signal']).sum())}회, "
        f"기존만 진입 {int((common['original_signal'] & ~common['anchored_signal']).sum())}회, "
        f"보정 proxy만 진입 {int((~common['original_signal'] & common['anchored_signal']).sum())}회. "
        "날짜를 하루 밀어서 생긴 차이가 아니라 신호 부호 차이이다.", "",
        "![두 전략 비교](foreign_pos_both_pos_pnl.png)", "",
        "## 결과 파일", "",
        "performance_summary.csv / strategy_comparison_summary.csv / daily_pnl.csv / strategy_trade_log.csv / "
        "strategy_signal_panel.csv / entry_disagreements.csv. 보정 전 proxy와 단순 보유·매일 매매 비교는 현재 실행 코드에서 제거했다.", "",
    ]
    (OUT_DIR / "strategy_explanation_and_performance.md").write_text("\n".join(lines), encoding="utf-8")
    print(summary[["strategy", "start", "end", "trades", "CAGR", "MDD", "Sharpe", "Calmar"]].to_string(index=False))
    print(matched[["scope", "strategy", "CAGR", "MDD", "Sharpe", "Calmar"]].to_string(index=False))
    print(f"Report: {(OUT_DIR / 'strategy_explanation_and_performance.md').resolve()}")


if __name__ == "__main__":
    main()
