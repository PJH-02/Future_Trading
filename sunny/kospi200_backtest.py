"""
KOSPI200 방향 예측 백테스트 - B안 (완성)
════════════════════════════════════════
1차 신호 : 외인 선물 순매수 방향
2차 필터 : 외인 풋 Z-score (미스프라이싱 프록시)

데이터
  · 가격       : FinanceDataReader 자동 (전체 기간)
  · 선물 순매수 : 매수_전체.csv + 매도_전체.csv (2010~2023)
                + 외국인 매수/매도.xlsx (2023~2026)
  · 옵션 순매수 : 옵션 콜:풋 순매수_전체.csv (2010~2026)

진입/청산 : 전일 수급 → 다음날 시가 진입 → 종가 청산
비용      : 왕복 0.03% + 틱 0.01%
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import FinanceDataReader as fdr
except ImportError:
    raise ImportError("pip install finance-datareader 실행 후 재시작하세요.")

warnings.filterwarnings("ignore")
plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

# ═══════════════════════════════════════════════════════════
# 0. 경로 & 설정
# ═══════════════════════════════════════════════════════════
BASE_DIR = Path("/Users/sunkyuhan/Desktop/HYFE/코스피200_옵션전략")
DATA_DIR = BASE_DIR / "data"
OUT_DIR  = BASE_DIR / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

F_FUT_BUY  = DATA_DIR / "KRX 코스피200 선물 매수_전체.csv"
F_FUT_SELL = DATA_DIR / "KRX 코스피200 선물 매도_전체.csv"
F_FUT_BUY_XLS  = DATA_DIR / "KRX 외국인 매수.xlsx"
F_FUT_SELL_XLS = DATA_DIR / "KRX 외국인 매도.xlsx"
F_OPT      = DATA_DIR / "KRX 코스피200 옵션 콜:풋 순매수_전체.csv"

START_DATE    = "20100101"
END_DATE      = "20260928"
TRANS_COST    = 0.0003   # 왕복 3bp (1.5bp × 2, 키움 기준)
TICK_COST     = 0.0002   # 슬리피지 2bp
ZSCORE_WINDOW = 60
ZSCORE_THRESH = 1.0

# ═══════════════════════════════════════════════════════════
# 1. 가격 (FinanceDataReader)
# ═══════════════════════════════════════════════════════════
def load_prices() -> pd.DataFrame:
    print("[1/3] KOSPI200 가격 수집...")
    df = fdr.DataReader("KS200",
                        f"{START_DATE[:4]}-{START_DATE[4:6]}-{START_DATE[6:]}",
                        f"{END_DATE[:4]}-{END_DATE[4:6]}-{END_DATE[6:]}")
    df = (df.reset_index()
            .rename(columns={"Date": "date", "Open": "open", "Close": "close"})
            [["date", "open", "close"]]
            .dropna()
            .sort_values("date")
            .reset_index(drop=True))
    df["date"] = pd.to_datetime(df["date"])
    print(f"      → {len(df)}일  {df['date'].min().date()} ~ {df['date'].max().date()}")
    return df


# ═══════════════════════════════════════════════════════════
# 2. 선물 순매수 (CSV 2010~2023 + xlsx 2023~2026 병합)
# ═══════════════════════════════════════════════════════════
def load_futures_net() -> pd.DataFrame:
    print("[2/3] 선물 순매수 로드...")

    buy_csv  = pd.read_csv(str(F_FUT_BUY),  encoding="utf-8-sig", thousands=",")
    sell_csv = pd.read_csv(str(F_FUT_SELL), encoding="utf-8-sig", thousands=",")
    buy_xls  = pd.read_excel(str(F_FUT_BUY_XLS))
    sell_xls = pd.read_excel(str(F_FUT_SELL_XLS))

    def _prep(df, col_name):
        df.columns = df.columns.str.strip()
        df = df.rename(columns={"일자": "date", "외국인 합계": col_name})
        df["date"]   = pd.to_datetime(df["date"])
        df[col_name] = pd.to_numeric(df[col_name], errors="coerce")
        return df[["date", col_name]].dropna()

    buy_csv  = _prep(buy_csv,  "buy")
    sell_csv = _prep(sell_csv, "sell")
    buy_xls  = _prep(buy_xls,  "buy")
    sell_xls = _prep(sell_xls, "sell")

    # CSV 이후 구간만 xlsx에서 추가
    cutoff = buy_csv["date"].max()
    buy  = pd.concat([buy_csv,  buy_xls[buy_xls["date"]   > cutoff]], ignore_index=True)
    sell = pd.concat([sell_csv, sell_xls[sell_xls["date"] > cutoff]], ignore_index=True)

    df = buy.merge(sell, on="date")
    df["fut_net"] = df["buy"] - df["sell"]
    df = df[["date", "fut_net"]].drop_duplicates("date").sort_values("date").reset_index(drop=True)
    print(f"      → {len(df)}일  {df['date'].min().date()} ~ {df['date'].max().date()}")
    return df


# ═══════════════════════════════════════════════════════════
# 3. 옵션 콜/풋 순매수
# ═══════════════════════════════════════════════════════════
def load_options_flow() -> pd.DataFrame:
    print("[3/3] 옵션 콜/풋 순매수 로드...")
    df = pd.read_csv(str(F_OPT), encoding="utf-8-sig", thousands=",")
    df.columns = df.columns.str.strip()
    df = df.rename(columns={
        "일자":         "date",
        "콜_순매수대금_원": "call_net",
        "풋_순매수대금_원": "put_net",
    })
    df["date"]     = pd.to_datetime(df["date"])
    df["call_net"] = pd.to_numeric(df["call_net"], errors="coerce")
    df["put_net"]  = pd.to_numeric(df["put_net"],  errors="coerce")
    df = df[["date", "call_net", "put_net"]].dropna().sort_values("date").reset_index(drop=True)
    print(f"      → {len(df)}일  {df['date'].min().date()} ~ {df['date'].max().date()}")
    return df


# ═══════════════════════════════════════════════════════════
# 4. 신호 생성
# ═══════════════════════════════════════════════════════════
def build_signals(prices, fut, opt, z_window=ZSCORE_WINDOW) -> pd.DataFrame:
    df = (prices
          .merge(fut, on="date", how="inner")
          .merge(opt, on="date", how="inner")
          .sort_values("date")
          .reset_index(drop=True))

    # ── 1차 신호: 외인 선물 순매수 방향 ──
    df["futures_signal"] = np.sign(df["fut_net"])

    # ── 옵션 플로우 정규화 ──
    total = df["call_net"].abs() + df["put_net"].abs()
    df["call_flow"] = df["call_net"] / total.replace(0, np.nan)
    df["put_flow"]  = df["put_net"]  / total.replace(0, np.nan)
    df["opt_flow"]  = df["call_flow"] - df["put_flow"]   # + = 콜 우세 (강세)

    # ── 옵션 방향 신호 ──
    df["options_dir"] = np.sign(df["opt_flow"])           # +1 = 강세, -1 = 약세

    # ── 풋 Z-score: 미스프라이싱 강도 ──
    mu    = df["put_flow"].rolling(z_window, min_periods=20).mean()
    sigma = df["put_flow"].rolling(z_window, min_periods=20).std()
    df["put_zscore"] = (df["put_flow"] - mu) / sigma.replace(0, np.nan)

    # ── 다음날 시가→종가 수익률 ──
    df["next_ret"] = df["close"].shift(-1) / df["open"].shift(-1) - 1

    print(f"\n신호 생성 완료: 유효 {df['next_ret'].notna().sum()}일")
    return df


# ═══════════════════════════════════════════════════════════
# 5. 포지션 정의
# ═══════════════════════════════════════════════════════════
def build_positions(df: pd.DataFrame, thresh=ZSCORE_THRESH) -> pd.DataFrame:

    # Base: 외인 선물 방향만
    df["pos_base"] = df["futures_signal"]

    # Case A: 선물 + 옵션 방향 일치 시만 진입
    df["pos_A"] = df["futures_signal"].where(
        df["futures_signal"] == df["options_dir"], 0)

    # Case B: 선물 + 풋 Z-score 방향 일치 AND |Z| >= 임계값 (미스프라이싱 필터)
    put_dir = -np.sign(df["put_zscore"])   # 풋 급증 = 하락 내재
    strong  = df["put_zscore"].abs() >= thresh
    df["pos_B"] = df["futures_signal"].where(
        (df["futures_signal"] == put_dir) & strong, 0)

    # Case C: A + B 동시 충족 (가장 엄격)
    df["pos_C"] = df["futures_signal"].where(
        (df["futures_signal"] == df["options_dir"]) &
        (df["futures_signal"] == put_dir) & strong, 0)

    return df


# ═══════════════════════════════════════════════════════════
# 6. 백테스트 & 성과
# ═══════════════════════════════════════════════════════════
ROUND_TRIP = TRANS_COST * 2 + TICK_COST

def backtest(df, pos_col):
    valid = df.dropna(subset=[pos_col, "next_ret"])
    pos   = valid[pos_col]
    net   = valid["next_ret"] * pos - ROUND_TRIP * pos.abs()
    cum   = (1 + net).cumprod()
    cum.index = net.index = valid["date"]
    return cum, net

def metrics(cum, net):
    n_years  = (cum.index[-1] - cum.index[0]).days / 365.25
    ann_ret  = cum.iloc[-1] ** (1 / n_years) - 1
    ann_vol  = net.std() * 252 ** 0.5
    sharpe   = ann_ret / ann_vol if ann_vol > 0 else np.nan
    max_dd   = (cum / cum.cummax() - 1).min()
    win_rate = (net[net != 0] > 0).mean()
    return {
        "연환산수익률": f"{ann_ret:+.2%}",
        "연변동성":     f"{ann_vol:.2%}",
        "샤프지수":     f"{sharpe:.3f}",
        "최대낙폭":     f"{max_dd:.2%}",
        "승률":         f"{win_rate:.2%}",
        "거래횟수":     int((net != 0).sum()),
    }


# ═══════════════════════════════════════════════════════════
# 7. 시각화
# ═══════════════════════════════════════════════════════════
CASES = {
    "Base (선물만)":          ("pos_base", "tab:blue"),
    "A  (선물+옵션방향일치)":  ("pos_A",    "tab:orange"),
    "B  (선물+풋Z필터)":       ("pos_B",    "tab:green"),
    "C  (A+B 동시)":           ("pos_C",    "tab:red"),
}

def plot_results(df, results):
    fig, axes = plt.subplots(4, 1, figsize=(14, 16))

    # (1) 누적 수익률
    ax = axes[0]
    for label, (col, color) in CASES.items():
        results[label][0].plot(ax=ax, label=label, color=color, lw=1.5)
    ax.set_title("케이스별 누적 수익률 (시가 진입 → 종가 청산)")
    ax.set_ylabel("누적 수익률"); ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # (2) 외인 선물 순매수 vs KOSPI200
    ax = axes[1]; ax2 = ax.twinx()
    ax.bar(df["date"], df["fut_net"] / 1e6, alpha=0.4, color="steelblue", label="선물 순매수")
    ax2.plot(df["date"], df["close"], color="black", lw=0.8, label="KOSPI200")
    ax.set_title("외인 선물 순매수 vs KOSPI200")
    ax.set_ylabel("순매수"); ax2.set_ylabel("KOSPI200")
    ax.legend(loc="upper left", fontsize=9); ax2.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3)

    # (3) 옵션 플로우 (콜-풋)
    ax = axes[2]
    ax.fill_between(df["date"], df["opt_flow"], 0,
                    where=(df["opt_flow"] > 0), color="blue",  alpha=0.4, label="콜 우세")
    ax.fill_between(df["date"], df["opt_flow"], 0,
                    where=(df["opt_flow"] < 0), color="red",   alpha=0.4, label="풋 우세")
    ax.set_title("옵션 플로우 (CallFlow - PutFlow)")
    ax.set_ylabel("OptFlow"); ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # (4) 풋 Z-score
    ax = axes[3]
    ax.fill_between(df["date"], df["put_zscore"], 0,
                    where=(df["put_zscore"] > 0), color="red",   alpha=0.4, label="풋 급증")
    ax.fill_between(df["date"], df["put_zscore"], 0,
                    where=(df["put_zscore"] < 0), color="green", alpha=0.4, label="풋 감소")
    ax.axhline( ZSCORE_THRESH, color="red",   ls="--", lw=0.8, alpha=0.7)
    ax.axhline(-ZSCORE_THRESH, color="green", ls="--", lw=0.8, alpha=0.7)
    ax.set_title(f"외인 풋 Z-score (윈도우 {ZSCORE_WINDOW}일, 임계값 ±{ZSCORE_THRESH})")
    ax.set_ylabel("Z-score"); ax.legend(fontsize=9); ax.grid(alpha=0.3)

    plt.tight_layout()
    out = OUT_DIR / "backtest_result.png"
    plt.savefig(out, dpi=150); plt.show()
    print(f"차트 저장 → {out}")


# ═══════════════════════════════════════════════════════════
# 8. 메인
# ═══════════════════════════════════════════════════════════
def main():
    prices = load_prices()
    fut    = load_futures_net()
    opt    = load_options_flow()

    df = build_signals(prices, fut, opt)
    df = build_positions(df)

    results = {lb: backtest(df, col) for lb, (col, _) in CASES.items()}

    rows = {lb: metrics(cum, net) for lb, (cum, net) in results.items()}
    perf = pd.DataFrame(rows).T

    from tabulate import tabulate
    print("\n" + "=" * 60)
    print(tabulate(perf, headers="keys", tablefmt="simple", stralign="right", numalign="right"))
    print("=" * 60)

    perf.to_csv(OUT_DIR / "performance_summary.csv", encoding="utf-8-sig")
    df.to_csv(OUT_DIR / "signals.csv", index=False, encoding="utf-8-sig")

    plot_results(df, results)
    return df, results


if __name__ == "__main__":
    df, results = main()
