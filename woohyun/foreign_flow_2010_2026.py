from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm


# ============================================================
# KOSPI200 선물 외국인 수급 -> 다음날 KOSPI200 방향성 분석
# 기간: 2010~2026
# ============================================================

DATA_DIR = Path(__file__).resolve().parent

PRICE_FILE = DATA_DIR / "KRX_KOSPI200_주가_2010_2026.csv"
SELL_FILE = DATA_DIR / "KRX 코스피200 선물 매도_2010_2026.csv"
BUY_FILE = DATA_DIR / "KRX 코스피200 선물 매수_2010_2026.csv"


# ============================================================
# 1. 데이터 불러오기
# ============================================================

price = pd.read_csv(PRICE_FILE, encoding="utf-8-sig")
sell = pd.read_csv(SELL_FILE, encoding="utf-8-sig")
buy = pd.read_csv(BUY_FILE, encoding="utf-8-sig")

print("가격 컬럼:")
print(price.columns.tolist())

print("\n매도 컬럼:")
print(sell.columns.tolist())

print("\n매수 컬럼:")
print(buy.columns.tolist())


# ============================================================
# 2. 날짜 및 숫자형 변환
# ============================================================

for df in [price, sell, buy]:

    df["date"] = pd.to_datetime(df["일자"], errors="coerce")

    df.sort_values("date", inplace=True)

# 숫자형 변환
price_cols = ["시가", "고가", "저가", "종가"]

for col in price_cols:
    price[col] = pd.to_numeric(
        price[col],
        errors="coerce"
    )

for df in [sell, buy]:

    df["외국인 합계"] = pd.to_numeric(
        df["외국인 합계"],
        errors="coerce"
    )


# ============================================================
# 3. 필요한 열만 사용
# ============================================================

price = price[
    ["date", "시가", "고가", "저가", "종가"]
].copy()

sell = sell[
    ["date", "외국인 합계"]
].rename(
    columns={
        "외국인 합계": "foreign_sell"
    }
)

buy = buy[
    ["date", "외국인 합계"]
].rename(
    columns={
        "외국인 합계": "foreign_buy"
    }
)


# ============================================================
# 4. 날짜 기준 병합
# ============================================================

df = (
    price
    .merge(
        sell,
        on="date",
        how="inner"
    )
    .merge(
        buy,
        on="date",
        how="inner"
    )
    .sort_values("date")
    .reset_index(drop=True)
)


# 중복 날짜 확인
if df["date"].duplicated().any():

    print("\n[주의] 중복 날짜 발견")

    duplicated_dates = df[
        df["date"].duplicated(keep=False)
    ]["date"]

    print(
        duplicated_dates
        .drop_duplicates()
        .tolist()
    )


# ============================================================
# 5. Foreign Flow 계산
# ============================================================

# 순매수
df["foreign_net"] = (
    df["foreign_buy"]
    - df["foreign_sell"]
)


# 1-day Foreign Flow
denom_1d = (
    df["foreign_buy"]
    + df["foreign_sell"]
)

df["foreign_flow_1d"] = np.where(

    denom_1d != 0,

    df["foreign_net"]
    / denom_1d,

    np.nan
)


# 3-day cumulative Foreign Flow

buy_3 = (
    df["foreign_buy"]
    .rolling(3)
    .sum()
)

sell_3 = (
    df["foreign_sell"]
    .rolling(3)
    .sum()
)

denom_3d = buy_3 + sell_3

df["foreign_flow_3d"] = np.where(

    denom_3d != 0,

    (buy_3 - sell_3)
    / denom_3d,

    np.nan
)


# ============================================================
# 6. 다음 거래일 KOSPI200 수익률
# ============================================================

df["t1_open"] = df["시가"].shift(-1)
df["t1_high"] = df["고가"].shift(-1)
df["t1_low"] = df["저가"].shift(-1)
df["t1_close"] = df["종가"].shift(-1)


# 전일 종가 -> 다음날 시가
df["ret_overnight"] = (
    df["t1_open"]
    / df["종가"]
    - 1
)


# 다음날 시가 -> 다음날 종가
df["ret_next_intraday"] = (
    df["t1_close"]
    / df["t1_open"]
    - 1
)


# 전일 종가 -> 다음날 종가
df["ret_close_to_close"] = (
    df["t1_close"]
    / df["종가"]
    - 1
)


# 이전 종가 -> 현재 종가
df["prev_close_to_close"] = (
    df["종가"]
    / df["종가"].shift(1)
    - 1
)


# ============================================================
# 7. 분석용 데이터
# ============================================================

analysis = df.dropna(
    subset=[
        "foreign_flow_1d",
        "foreign_flow_3d",
        "ret_next_intraday",
        "ret_overnight",
        "ret_close_to_close"
    ]
).copy()


print("\n========================================")
print("전체 분석기간")
print("========================================")

print(
    "시작:",
    analysis["date"].min().date()
)

print(
    "종료:",
    analysis["date"].max().date()
)

print(
    "관측치:",
    len(analysis)
)


# ============================================================
# 8. 분위수 분석
# ============================================================

def quintile_table(
    data,
    signal_col,
    return_col
):

    temp = data[
        [signal_col, return_col]
    ].dropna().copy()

    temp["quintile"] = (
        pd.qcut(
            temp[signal_col],
            5,
            labels=False,
            duplicates="drop"
        )
        + 1
    )

    result = (
        temp
        .groupby("quintile")[return_col]
        .agg(
            mean_return="mean",
            hit_rate=lambda x: (x > 0).mean(),
            n="size"
        )
    )

    result["mean_return_pct"] = (
        result["mean_return"] * 100
    )

    result["hit_rate_pct"] = (
        result["hit_rate"] * 100
    )

    return result


q_1d = quintile_table(
    analysis,
    "foreign_flow_1d",
    "ret_next_intraday"
)


q_3d = quintile_table(
    analysis,
    "foreign_flow_3d",
    "ret_next_intraday"
)


print("\n========================================")
print("1-Day Foreign Flow 분위수")
print("========================================")

print(
    q_1d.to_string()
)


print("\n========================================")
print("3-Day Foreign Flow 분위수")
print("========================================")

print(
    q_3d.to_string()
)


# ============================================================
# 9. HAC 회귀분석
# ============================================================

def run_hac(
    data,
    signal_col,
    return_col,
    label
):

    temp = data[
        [signal_col, return_col]
    ].dropna()

    X = sm.add_constant(
        temp[signal_col]
    )

    y = temp[return_col]

    model = sm.OLS(
        y,
        X
    ).fit(
        cov_type="HAC",
        cov_kwds={
            "maxlags": 5
        }
    )

    print("\n========================================")
    print(label)
    print("========================================")

    print(
        "Beta:",
        round(
            model.params[signal_col],
            6
        )
    )

    print(
        "t-stat:",
        round(
            model.tvalues[signal_col],
            3
        )
    )

    print(
        "p-value:",
        round(
            model.pvalues[signal_col],
            4
        )
    )

    print(
        "R-squared:",
        round(
            model.rsquared,
            4
        )
    )

    return {
        "signal": signal_col,
        "return": return_col,
        "beta": model.params[signal_col],
        "t_stat": model.tvalues[signal_col],
        "p_value": model.pvalues[signal_col],
        "r_squared": model.rsquared,
        "n": len(temp)
    }


regression_results = []


regression_results.append(
    run_hac(
        analysis,
        "foreign_flow_1d",
        "ret_next_intraday",
        "1-Day Foreign Flow -> Next-day Intraday Return"
    )
)


regression_results.append(
    run_hac(
        analysis,
        "foreign_flow_3d",
        "ret_next_intraday",
        "3-Day Foreign Flow -> Next-day Intraday Return"
    )
)


# ============================================================
# 10. Previous Return 통제
# ============================================================

control_data = analysis.dropna(
    subset=[
        "foreign_flow_3d",
        "prev_close_to_close",
        "ret_next_intraday"
    ]
).copy()


X = sm.add_constant(
    control_data[
        [
            "foreign_flow_3d",
            "prev_close_to_close"
        ]
    ]
)

y = control_data[
    "ret_next_intraday"
]


control_model = sm.OLS(
    y,
    X
).fit(
    cov_type="HAC",
    cov_kwds={
        "maxlags": 5
    }
)


print("\n========================================")
print("3-Day Foreign Flow + Previous Return 통제")
print("========================================")

print(
    control_model.summary()
)


# ============================================================
# 11. 연도별 안정성
# ============================================================

def annual_stability(
    data,
    signal_col,
    return_col
):

    rows = []

    for year, group in data.groupby(
        data["date"].dt.year
    ):

        group = group[
            [
                signal_col,
                return_col
            ]
        ].dropna().copy()

        if len(group) < 30:
            continue

        X = sm.add_constant(
            group[signal_col]
        )

        y = group[return_col]

        model = sm.OLS(
            y,
            X
        ).fit(
            cov_type="HAC",
            cov_kwds={
                "maxlags": 5
            }
        )

        group["q"] = (
            pd.qcut(
                group[signal_col],
                5,
                labels=False,
                duplicates="drop"
            )
            + 1
        )

        qmean = (
            group
            .groupby("q")[return_col]
            .mean()
        )

        q1 = qmean.get(
            1,
            np.nan
        )

        q5 = qmean.get(
            5,
            np.nan
        )

        rows.append({

            "year": int(year),

            "n": len(group),

            "beta":
                model.params[signal_col],

            "t_stat":
                model.tvalues[signal_col],

            "p_value":
                model.pvalues[signal_col],

            "r_squared":
                model.rsquared,

            "Q1_return_pct":
                q1 * 100,

            "Q5_return_pct":
                q5 * 100,

            "Q5_minus_Q1_pct":
                (q5 - q1) * 100
        })

    return pd.DataFrame(rows)


annual_1d = annual_stability(
    analysis,
    "foreign_flow_1d",
    "ret_next_intraday"
)


annual_3d = annual_stability(
    analysis,
    "foreign_flow_3d",
    "ret_next_intraday"
)


print("\n========================================")
print("연도별 안정성 - 1-Day")
print("========================================")

print(
    annual_1d.to_string(
        index=False
    )
)


print("\n========================================")
print("연도별 안정성 - 3-Day")
print("========================================")

print(
    annual_3d.to_string(
        index=False
    )
)


# ============================================================
# 12. 상위 20% Long-only 탐색 백테스트
# ============================================================

strategy_rows = []


for signal_col in [
    "foreign_flow_1d",
    "foreign_flow_3d"
]:

    temp = analysis.copy()

    threshold = (
        temp[signal_col]
        .quantile(0.80)
    )

    trade = (
        temp[
            temp[signal_col] >= threshold
        ]["ret_next_intraday"]
    )

    if (
        len(trade) > 0
        and trade.std() > 0
    ):

        avg_ret = trade.mean()

        hit = (
            trade > 0
        ).mean()

        sharpe = (
            np.sqrt(252)
            * avg_ret
            / trade.std()
        )

    else:

        avg_ret = np.nan
        hit = np.nan
        sharpe = np.nan


    strategy_rows.append({

        "signal":
            signal_col,

        "threshold":
            threshold,

        "trades":
            len(trade),

        "avg_trade_return_pct":
            avg_ret * 100,

        "hit_rate_pct":
            hit * 100,

        "annualized_sharpe":
            sharpe
    })


strategy_results = pd.DataFrame(
    strategy_rows
)


print("\n========================================")
print("상위 20% Long-only 탐색전략")
print("========================================")

print(
    strategy_results.to_string(
        index=False
    )
)


# ============================================================
# 13. 그래프 저장
# ============================================================

def save_bar_chart(
    result,
    title,
    filename
):

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.bar(
        result.index.astype(str),
        result["mean_return_pct"]
    )

    ax.axhline(
        0,
        linewidth=1
    )

    ax.set_xlabel(
        "Quintile"
    )

    ax.set_ylabel(
        "Next-day Intraday Return (%)"
    )

    ax.set_title(
        title
    )

    fig.tight_layout()

    fig.savefig(
        DATA_DIR / filename,
        dpi=180,
        bbox_inches="tight"
    )

    plt.close(fig)


save_bar_chart(
    q_1d,
    "1-Day Foreign Flow vs Next-Day KOSPI200 Return",
    "foreign_flow_2010_2026_1d.png"
)


save_bar_chart(
    q_3d,
    "3-Day Foreign Flow vs Next-Day KOSPI200 Return",
    "foreign_flow_2010_2026_3d.png"
)


# ============================================================
# 14. 결과 Excel 저장
# ============================================================

output_file = (
    DATA_DIR
    / "foreign_flow_2010_2026_result.xlsx"
)


with pd.ExcelWriter(
    output_file,
    engine="openpyxl"
) as writer:

    analysis.to_excel(
        writer,
        sheet_name="merged_analysis",
        index=False
    )

    q_1d.to_excel(
        writer,
        sheet_name="1day_quintile"
    )

    q_3d.to_excel(
        writer,
        sheet_name="3day_quintile"
    )

    pd.DataFrame(
        regression_results
    ).to_excel(
        writer,
        sheet_name="regression",
        index=False
    )

    annual_1d.to_excel(
        writer,
        sheet_name="annual_1day",
        index=False
    )

    annual_3d.to_excel(
        writer,
        sheet_name="annual_3day",
        index=False
    )

    strategy_results.to_excel(
        writer,
        sheet_name="strategy",
        index=False
    )


print("\n========================================")
print("전체 분석 완료")
print("========================================")

print(
    "결과 파일:",
    output_file
)

print(
    "그래프:",
    "foreign_flow_2010_2026_1d.png"
)

print(
    "그래프:",
    "foreign_flow_2010_2026_3d.png"
)