import pandas as pd
import numpy as np


# =========================================================
# 0. 파일 설정
# =========================================================

KODEX_FILE = "KODEX200_1min_2023_2026.csv"
BUY_FILE = "KRX 외국인 매수.xlsx"
SELL_FILE = "KRX 외국인 매도.xlsx"

# Flow 상위 구간
# Top 20% = 80% quantile
# Top 30% = 70% quantile
# Top 40% = 60% quantile
FLOW_QUANTILES = {
    "Top20": 0.80,
    "Top30": 0.70,
    "Top40": 0.60,
}
ROLLING_DAYS = 250

# Opening Range
OR_START = "09:01"
OR_END = "09:10"

# OR 돌파 탐색 시작
ENTRY_START = "09:11"

MARKET_CLOSE = "15:30"


# =========================================================
# 1. KODEX 데이터 읽기
# =========================================================

print("=" * 70)
print("1. KODEX 200 1분봉 데이터 읽는 중...")
print("=" * 70)

kodex = pd.read_csv(
    KODEX_FILE,
    parse_dates=["datetime"]
)

kodex = (
    kodex
    .sort_values("datetime")
    .reset_index(drop=True)
)

print(f"KODEX 데이터 개수: {len(kodex):,}")


# =========================================================
# 2. KODEX 데이터 정리
# =========================================================

required_columns = [
    "datetime",
    "open",
    "high",
    "low",
    "close",
    "volume"
]

missing = [
    c for c in required_columns
    if c not in kodex.columns
]

if missing:
    raise ValueError(
        f"KODEX 파일에 다음 컬럼이 없습니다: {missing}"
    )

for col in [
    "open",
    "high",
    "low",
    "close",
    "volume"
]:
    kodex[col] = pd.to_numeric(
        kodex[col],
        errors="coerce"
    )

kodex["date"] = kodex["datetime"].dt.date
kodex["time"] = kodex["datetime"].dt.strftime("%H:%M")

kodex = kodex[
    kodex["volume"] > 0
].copy()

kodex = (
    kodex
    .dropna(
        subset=[
            "datetime",
            "open",
            "high",
            "low",
            "close",
            "volume"
        ]
    )
    .reset_index(drop=True)
)

print(
    f"거래량 0 데이터 제거 후: {len(kodex):,}"
)


# =========================================================
# 3. 외국인 매수/매도 데이터
# =========================================================

print()
print("=" * 70)
print("2. 외국인 선물 수급 데이터 읽는 중...")
print("=" * 70)

buy = pd.read_excel(BUY_FILE)
sell = pd.read_excel(SELL_FILE)

print("매수 데이터 컬럼:", buy.columns.tolist())
print("매도 데이터 컬럼:", sell.columns.tolist())


# 날짜 변환

buy["date"] = pd.to_datetime(
    buy["일자"]
).dt.date

sell["date"] = pd.to_datetime(
    sell["일자"]
).dt.date


# 실제 컬럼명: 외국인 합계

buy = buy[
    ["date", "외국인 합계"]
].rename(
    columns={
        "외국인 합계": "foreign_buy"
    }
)

sell = sell[
    ["date", "외국인 합계"]
].rename(
    columns={
        "외국인 합계": "foreign_sell"
    }
)


# =========================================================
# 4. 외국인 Flow 계산
# =========================================================

flow = (
    buy
    .merge(
        sell,
        on="date",
        how="inner"
    )
    .sort_values("date")
    .reset_index(drop=True)
)

flow["foreign_net"] = (
    flow["foreign_buy"]
    - flow["foreign_sell"]
)

denom = (
    flow["foreign_buy"]
    + flow["foreign_sell"]
)

flow["foreign_flow"] = np.where(
    denom != 0,
    flow["foreign_net"] / denom,
    np.nan
)

print()
print(
    f"Foreign Flow 데이터: {len(flow):,}일"
)


# =========================================================
# 5. 전일 Flow를 다음 거래일에 연결
# =========================================================

flow["previous_day_flow"] = (
    flow["foreign_flow"]
    .shift(1)
)


# =========================================================
# 6. Flow 상위 20% / 30% / 40% 계산
# =========================================================

for label, quantile in FLOW_QUANTILES.items():

    threshold_col = f"flow_{label.lower()}_threshold"
    signal_col = f"flow_{label.lower()}"

    flow[threshold_col] = (
        flow["foreign_flow"]
        .rolling(
            ROLLING_DAYS,
            min_periods=ROLLING_DAYS
        )
        .quantile(quantile)
        .shift(1)
    )

    flow[signal_col] = (
        flow["previous_day_flow"]
        >=
        flow[threshold_col]
    )


# =========================================================
# 7. KODEX + Flow 결합
# =========================================================

kodex = kodex.merge(
    flow[
        [
            "date",
            "previous_day_flow",
            "flow_top20",
            "flow_top30",
            "flow_top40"
        ]
    ],
    on="date",
    how="left"
)

print()
print("데이터 결합 완료")


# =========================================================
# 8. VWAP 계산
# =========================================================

kodex["typical_price"] = (
    kodex["high"]
    + kodex["low"]
    + kodex["close"]
) / 3

kodex["pv"] = (
    kodex["typical_price"]
    * kodex["volume"]
)

kodex["cum_pv"] = (
    kodex
    .groupby("date")["pv"]
    .cumsum()
)

kodex["cum_volume"] = (
    kodex
    .groupby("date")["volume"]
    .cumsum()
)

kodex["vwap"] = (
    kodex["cum_pv"]
    /
    kodex["cum_volume"]
)


# =========================================================
# 9. Opening Range 계산
# =========================================================

def prepare_day(day):

    day = day.copy()

    or_data = day[
        (day["time"] >= OR_START)
        &
        (day["time"] <= OR_END)
    ]

    if len(or_data) == 0:
        return None

    or_high = or_data["high"].max()
    or_low = or_data["low"].min()

    day["or_high"] = or_high
    day["or_low"] = or_low

    return day


# =========================================================
# 10. ORB + VWAP 신호
# =========================================================

def make_signal(day):

    day = prepare_day(day)

    if day is None:
        return None

    day = day.copy()

    day["long_signal"] = False

    after_or = (
        day["time"] >= ENTRY_START
    )

    breakout = (
        day["close"]
        > day["or_high"]
    )

    above_vwap = (
        day["close"]
        > day["vwap"]
    )

    day.loc[
        after_or
        & breakout
        & above_vwap,
        "long_signal"
    ] = True

    return day


# =========================================================
# 11. 백테스트
# =========================================================

def backtest(
    data,
    holding_minutes,
    flow_filter=None
):

    trades = []

    for date, day in data.groupby("date"):

        day = make_signal(day)

        if day is None:
            continue

        day = day.reset_index(drop=True)

        # -------------------------------------------------
        # Flow filter
        # -------------------------------------------------

        if flow_filter == "positive":

            previous_flow = (
                day["previous_day_flow"].iloc[0]
            )

            if (
                pd.isna(previous_flow)
                or previous_flow <= 0
            ):
                continue

        elif flow_filter in ("top20", "top30", "top40"):

            flow_signal = day[f"flow_{flow_filter}"].iloc[0]

            if pd.isna(flow_signal) or not bool(flow_signal):
                continue

        # -------------------------------------------------
        # 장중 첫 번째 ORB + VWAP 신호
        # -------------------------------------------------

        signal_idx = day.index[
            day["long_signal"]
        ]

        if len(signal_idx) == 0:
            continue

        signal_idx = signal_idx[0]

        # -------------------------------------------------
        # ★ 원본 로직: 신호가 발생한 다음 1분봉 시가에 진입
        # -------------------------------------------------

        entry_idx = signal_idx + 1

        if entry_idx >= len(day):
            continue

        entry_price = day.loc[
            entry_idx,
            "open"
        ]

        # -------------------------------------------------
        # 보유기간 후 청산
        # -------------------------------------------------

        exit_idx = entry_idx + holding_minutes

        if exit_idx >= len(day):
            continue

        exit_price = day.loc[
            exit_idx,
            "close"
        ]

        ret = (
            exit_price
            /
            entry_price
            - 1
        )

        trades.append({

            "date": date,

            "entry_time":
                day.loc[
                    entry_idx,
                    "time"
                ],

            "exit_time":
                day.loc[
                    exit_idx,
                    "time"
                ],

            "entry_price":
                entry_price,

            "exit_price":
                exit_price,

            "return":
                ret,

            "return_bp":
                ret * 10000,

            "foreign_flow":
                day[
                    "previous_day_flow"
                ].iloc[0]

        })

    return pd.DataFrame(trades)


# =========================================================
# 12. 성과 계산
# =========================================================

def summarize(trades, name):

    if len(trades) == 0:

        return {
            "strategy": name,
            "trades": 0,
            "win_rate": np.nan,
            "avg_return": np.nan,
            "median_return": np.nan,
            "profit_factor": np.nan,
            "sharpe": np.nan,
            "mdd": np.nan,
            "total_return": np.nan
        }

    r = trades["return"]

    win_rate = (
        (r > 0).mean()
        * 100
    )

    avg_return = r.mean()

    median_return = r.median()

    gross_profit = r[r > 0].sum()

    gross_loss = (
        abs(r[r < 0].sum())
    )

    if gross_loss != 0:
        profit_factor = (
            gross_profit
            /
            gross_loss
        )
    else:
        profit_factor = np.nan

    if r.std() != 0:

        sharpe = (
            r.mean()
            /
            r.std()
            *
            np.sqrt(252)
        )

    else:
        sharpe = np.nan

    equity = (
        1 + r
    ).cumprod()

    running_max = equity.cummax()

    drawdown = (
        equity
        /
        running_max
        - 1
    )

    mdd = drawdown.min()

    total_return = (
        equity.iloc[-1]
        - 1
    )

    return {

        "strategy": name,

        "trades":
            len(trades),

        "win_rate":
            win_rate,

        "avg_return":
            avg_return * 100,

        "median_return":
            median_return * 100,

        "profit_factor":
            profit_factor,

        "sharpe":
            sharpe,

        "mdd":
            mdd * 100,

        "total_return":
            total_return * 100
    }


# =========================================================
# 13. 전략 비교
# =========================================================

strategies = {

    # A: Flow 필터 없음
    "A_ORB_VWAP":
        None,

    # B: Flow > 0
    "B_FlowPositive_ORB_VWAP":
        "positive",

    # C: 전일 Flow 상위 20%
    "C_FlowTop20_ORB_VWAP":
        "top20",

    # D: 전일 Flow 상위 30%
    "D_FlowTop30_ORB_VWAP":
        "top30",

    # E: 전일 Flow 상위 40%
    "E_FlowTop40_ORB_VWAP":
        "top40"
}


holding_periods = [
    5,
    10,
    20,
    30,
    60
]


all_results = []

all_trades = []


# =========================================================
# 14. 실행
# =========================================================

for strategy_name, flow_filter in strategies.items():

    for holding in holding_periods:

        print()
        print(
            f"{strategy_name} / "
            f"{holding}분 백테스트..."
        )

        trades = backtest(
            kodex,
            holding,
            flow_filter
        )

        summary = summarize(
            trades,
            f"{strategy_name}_{holding}min"
        )

        all_results.append(
            summary
        )

        if len(trades) > 0:

            trades["strategy"] = (
                f"{strategy_name}_{holding}min"
            )

            all_trades.append(
                trades
            )


# =========================================================
# 15. 결과 출력
# =========================================================

results = pd.DataFrame(
    all_results
)

print()
print("=" * 100)
print("A / B / C / D / E 전략 비교 결과")
print("=" * 100)

print(
    results.to_string(
        index=False
    )
)


# =========================================================
# 16. 결과 저장
# =========================================================

results.to_csv(
    "flow_top20_30_40_strategy_summary.csv",
    index=False,
    encoding="utf-8-sig"
)

if len(all_trades) > 0:

    trades_all = pd.concat(
        all_trades,
        ignore_index=True
    )

    trades_all.to_csv(
        "flow_top20_30_40_strategy_trades.csv",
        index=False,
        encoding="utf-8-sig"
    )

print()
print("=" * 70)
print("백테스트 완료!")
print("=" * 70)

print()
print(
    "결과 파일:"
)

print(
    "ABC_strategy_summary.csv"
)

print(
    "ABC_strategy_trades.csv"
)