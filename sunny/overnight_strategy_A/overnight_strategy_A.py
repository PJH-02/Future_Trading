"""
오버나잇 전략 A: 당일 종가 진입 → 익일 시가 청산
════════════════════════════════════════════════════════════
기존 인트라데이 연구 결과를 오버나잇(종가→익일시가) 구조에 적용

──── 근거 (이전 전략 연구) ──────────────────────────────────
전략 1B  외인 풋z > 1.0 → 숏 (인트라데이 Sharpe 0.365)
전략 3C  외인 풋z > 1.0 & 개인 콜z > 0 → 숏 (Sharpe 0.380, MDD -9.3%)
전략 2   ES 역추세 + 외인선물 결합 (Sharpe 0.50)
T05      외인 풋 z 를 오버나잇 롱 거부권으로 활용

──── 오버나잇 신호 설계 ──────────────────────────────────────
포지션: T 종가 결정 → T+1 시가 청산
신호 기준일: T (T 장 마감 후 확정 데이터, 룩어헤드 없음)

  ① SHORT 신호: 외인 풋z[T] > 1.0 (헤지 급증 → 갭다운 내재)
  ② SHORT 강화: ① + 개인 콜z[T] > 0 (개인 낙관 = 역지표)
  ③ LONG 신호:  외인선물z[T] > 0.5 AND 외인 풋z[T] < 1.5 (거부권 포함)
  ④ 롱 거부:    외인 풋z[T] > 1.5 일 때 롱 불가

케이스
  Base  항상 롱 (기준선, 오버나잇 드리프트 확인용)
  A     외인 풋z > 1.0 → 숏 (미포지 기타)
  B     외인 풋z > 1.0 & 개인 콜z > 0 → 숏 (Strategy 3C 오버나잇, 개인 데이터 필요)
  C     외인선물z > 0.5 → 롱 (핵심 케이스, 거부권 포함)
  D     숏 우선 결합: B 조건 → 숏 / C 조건 → 롱 / 기타 미포지

진입·청산: T 종가 단일가 → T+1 시가 단일가
비용: 왕복 5bp (KODEX200 기준), 스트레스 7bp

──── 백테스트 결과 (2010~2026) ──────────────────────────────
  Base  +8.1% / Sharpe 0.57 / MDD -34.3%  (오버나잇 드리프트 자체가 강함)
  A     -2.1% / Sharpe -0.42              (풋 숏이 오버나잇에서는 역방향)
  B     (개인 데이터 필요)
  C     +5.4% / Sharpe 0.77 / MDD -15.8%  ← 핵심: 외인 선물 롱 필터
  D     +4.1% / Sharpe 0.49

──── 데이터 ──────────────────────────────────────────────────
  beomgu/data/krx_opt_investor_daily.csv    (외인+개인 옵션 투자자별)
  beomgu/data/naver_fut_investor_daily.csv  (외인 선물 순매수)
  beomgu/data/k200_index_daily.csv          (K200 지수 일봉)
  beomgu/data/spx_daily.csv                 (S&P500 일봉)

실행: cd Future_Trading && python sunny/overnight_strategy_A/overnight_strategy_A.py
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

# ════════════════════════════════════════════════════════
# 0. 경로 & 설정
# ════════════════════════════════════════════════════════
REPO_DIR = Path(__file__).resolve().parents[2]   # Future_Trading/
DATA_DIR = REPO_DIR / "beomgu" / "data"
OUT_DIR  = Path(__file__).parent / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

START_DATE    = "2010-01-01"
END_DATE      = "2026-11-01"
ROUND_TRIP    = 0.0005   # 왕복 5bp
STRESS_COST   = 0.0007   # 스트레스 7bp
ZSCORE_WINDOW = 60
PUT_SHORT_THR = 1.0      # 외인 풋z SHORT 임계
IND_CALL_THR  = 0.0      # 개인 콜z 확인 임계
FUT_LONG_THR  = 0.5      # 외인선물z LONG 임계
PUT_VETO_THR  = 1.5      # 롱 거부권 풋z 임계


# ════════════════════════════════════════════════════════
# 1. 가격 데이터
# ════════════════════════════════════════════════════════
def load_prices() -> pd.DataFrame:
    f = DATA_DIR / "k200_index_daily.csv"
    if f.exists():
        df = pd.read_csv(f, encoding="utf-8-sig")
        df.columns = df.columns.str.strip()
        df["date"] = pd.to_datetime(df["date"])
        df = df[["date", "open", "close"]].dropna()
    else:
        print("  k200_index_daily.csv 없음 → FinanceDataReader 사용")
        raw = fdr.DataReader("KS200", START_DATE, END_DATE)
        df = (raw.reset_index()
                 .rename(columns={"Date": "date", "Open": "open", "Close": "close"})
                 [["date", "open", "close"]]
                 .dropna())
        df["date"] = pd.to_datetime(df["date"])

    df = (df.sort_values("date")
            .drop_duplicates("date")
            .reset_index(drop=True))
    df = df[(df["date"] >= START_DATE) & (df["date"] <= END_DATE)]
    print(f"  K200 가격: {len(df)}일  {df['date'].min().date()} ~ {df['date'].max().date()}")
    return df


# ════════════════════════════════════════════════════════
# 2. 옵션 투자자별 데이터
# ════════════════════════════════════════════════════════
def load_option_flows() -> pd.DataFrame:
    """반환: date, for_put_net, ind_call_net (금액 기준 순매수)"""
    f = DATA_DIR / "krx_opt_investor_daily.csv"
    if not f.exists():
        raise FileNotFoundError(f"파일 없음: {f}\nbeomgu/data/ 폴더에 krx_opt_investor_daily.csv 필요")

    df = pd.read_csv(f, encoding="utf-8-sig")
    df["date"] = pd.to_datetime(df["date"])
    df["net_amt"] = df["buy_amt"] - df["sell_amt"]

    for_put = (df[(df["investor"] == "foreign") & (df["cp"] == "P")]
               .groupby("date")["net_amt"].sum()
               .rename("for_put_net"))
    ind_call = (df[(df["investor"] == "individual") & (df["cp"] == "C")]
                .groupby("date")["net_amt"].sum()
                .rename("ind_call_net"))

    result = pd.concat([for_put, ind_call], axis=1).reset_index()
    result = (result.dropna(subset=["for_put_net"])
                    .sort_values("date")
                    .reset_index(drop=True))
    result = result[(result["date"] >= START_DATE) & (result["date"] <= END_DATE)]
    print(f"  옵션: {len(result)}일  {result['date'].min().date()} ~ {result['date'].max().date()}")
    return result


# ════════════════════════════════════════════════════════
# 3. 외인 선물 순매수
# ════════════════════════════════════════════════════════
def load_futures_net() -> pd.Series:
    f = DATA_DIR / "naver_fut_investor_daily.csv"
    if not f.exists():
        raise FileNotFoundError(f"파일 없음: {f}")

    df = pd.read_csv(f, encoding="utf-8-sig")
    df["date"] = pd.to_datetime(df["date"])
    s = df.set_index("date")["net_foreign"].dropna().sort_index()
    s = s[(s.index >= START_DATE) & (s.index <= END_DATE)]
    print(f"  선물: {len(s)}일  {s.index.min().date()} ~ {s.index.max().date()}")
    return s


# ════════════════════════════════════════════════════════
# 4. SPX 일봉
# ════════════════════════════════════════════════════════
def load_spx() -> pd.Series:
    f = DATA_DIR / "spx_daily.csv"
    if f.exists():
        df = pd.read_csv(f, encoding="utf-8-sig")
        df["date"] = pd.to_datetime(df["date"])
        s = df.set_index("date")["close"].dropna().sort_index()
    else:
        print("  spx_daily.csv 없음 → FinanceDataReader 사용")
        raw = fdr.DataReader("^GSPC", "2009-01-01", END_DATE)
        s = raw["Close"]
        s.index = pd.to_datetime(s.index)

    spx_ret = s.pct_change().rename("spx_ret")
    print(f"  SPX: {len(spx_ret.dropna())}일  {s.index.min().date()} ~ {s.index.max().date()}")
    return spx_ret


# ════════════════════════════════════════════════════════
# 5. Z-score
# ════════════════════════════════════════════════════════
def rolling_z(s: pd.Series, window: int = ZSCORE_WINDOW) -> pd.Series:
    mu    = s.shift(1).rolling(window, min_periods=max(20, window // 2)).mean()
    sigma = s.shift(1).rolling(window, min_periods=max(20, window // 2)).std()
    return ((s - mu) / sigma.replace(0, np.nan)).clip(-5, 5)


# ════════════════════════════════════════════════════════
# 6. 데이터 결합 및 신호 생성
# ════════════════════════════════════════════════════════
def build_dataset() -> pd.DataFrame:
    print("[데이터 로드]")
    prices = load_prices()
    opt    = load_option_flows()
    fut    = load_futures_net()
    spx    = load_spx()

    df = prices.merge(opt, on="date", how="inner")

    fut_df = fut.reset_index().rename(columns={"date": "date", "net_foreign": "fut_net"})
    df = df.merge(fut_df, on="date", how="left")

    spx_df = spx.dropna().reset_index().rename(columns={"date": "us_date"})
    df = pd.merge_asof(df.sort_values("date"), spx_df.sort_values("us_date"),
                       left_on="date", right_on="us_date", direction="backward")
    df = df.drop(columns=["us_date"])
    df = df.sort_values("date").reset_index(drop=True)

    # 오버나잇 수익률: T 종가 진입 → T+1 시가 청산
    # on_ret[T] = open[T] / close[T-1] - 1
    # 신호는 T-1 데이터 → shift(1) 로 T 수익률을 예측
    df["on_ret"] = df["open"] / df["close"].shift(1) - 1

    # Z-score 계산
    df["for_put_z"]  = rolling_z(df["for_put_net"])
    df["ind_call_z"] = rolling_z(df["ind_call_net"])
    df["fut_z"]      = rolling_z(df["fut_net"])

    # shift(1): 전일 신호 → 당일 오버나잇 수익
    df["sig_for_put_z"]  = df["for_put_z"].shift(1)
    df["sig_ind_call_z"] = df["ind_call_z"].shift(1)
    df["sig_fut_z"]      = df["fut_z"].shift(1)
    df["sig_spx_ret"]    = df["spx_ret"].shift(1)

    print(f"\n[신호 생성] 유효 오버나잇 수익률: {df['on_ret'].notna().sum()}일")
    return df


# ════════════════════════════════════════════════════════
# 7. 포지션 정의
# ════════════════════════════════════════════════════════
def build_positions(df: pd.DataFrame) -> pd.DataFrame:
    s_pz  = df["sig_for_put_z"]
    s_icz = df["sig_ind_call_z"]
    s_fz  = df["sig_fut_z"]

    # Base: 항상 롱 (오버나잇 드리프트 기준선)
    df["pos_base"] = 1.0

    # A: 외인 풋z > 임계 → 숏만 (기타 미포지)
    df["pos_A"] = np.where(s_pz > PUT_SHORT_THR, -1.0, 0.0)

    # B: 외인 풋z > 임계 AND 개인 콜z > 임계 → 숏 (Strategy 3C 오버나잇)
    df["pos_B"] = np.where(
        (s_pz > PUT_SHORT_THR) & (s_icz > IND_CALL_THR), -1.0, 0.0)

    # C: 외인선물z > 롱임계 → 롱 (풋 거부권 포함)
    long_cond = (s_fz > FUT_LONG_THR) & (s_pz < PUT_VETO_THR)
    df["pos_C"] = np.where(long_cond, 1.0, 0.0)

    # D: 숏 우선 결합 (B 숏 OR C 롱)
    short_cond = (s_pz > PUT_SHORT_THR) & (s_icz > IND_CALL_THR)
    df["pos_D"] = np.where(short_cond, -1.0,
                  np.where(long_cond,   1.0, 0.0))

    return df


# ════════════════════════════════════════════════════════
# 8. 백테스트 & 성과
# ════════════════════════════════════════════════════════
def backtest(df: pd.DataFrame, pos_col: str, cost: float = ROUND_TRIP):
    valid = df.dropna(subset=[pos_col, "on_ret"])
    pos   = valid[pos_col]
    net   = valid["on_ret"] * pos - cost * pos.abs()
    cum   = (1 + net).cumprod()
    cum.index = net.index = valid["date"]
    return cum, net


def metrics(cum: pd.Series, net: pd.Series) -> dict:
    n_years  = (cum.index[-1] - cum.index[0]).days / 365.25
    ann_ret  = cum.iloc[-1] ** (1 / n_years) - 1
    ann_vol  = net.std() * 252 ** 0.5
    sharpe   = ann_ret / ann_vol if ann_vol > 0 else np.nan
    max_dd   = (cum / cum.cummax() - 1).min()
    win_rate = (net[net != 0] > 0).mean()
    n_trades = int((net != 0).sum())
    return {
        "CAGR":     f"{ann_ret:+.2%}",
        "연변동성":  f"{ann_vol:.2%}",
        "샤프지수":  f"{sharpe:+.3f}",
        "최대낙폭":  f"{max_dd:.2%}",
        "승률":      f"{win_rate:.2%}",
        "거래횟수":  n_trades,
    }


# ════════════════════════════════════════════════════════
# 9. 시각화
# ════════════════════════════════════════════════════════
CASES = {
    "Base (항상 롱)":         ("pos_base", "tab:gray"),
    "A  (외인 풋z → 숏)":     ("pos_A",    "tab:red"),
    "B  (풋z+개인콜z → 숏)":  ("pos_B",    "tab:orange"),
    "C  (외인선물z → 롱)":    ("pos_C",    "tab:blue"),
    "D  (롱숏 결합)":          ("pos_D",    "tab:green"),
}


def plot_results(df: pd.DataFrame, results: dict):
    fig, axes = plt.subplots(4, 1, figsize=(14, 18))

    ax = axes[0]
    for label, (col, color) in CASES.items():
        results[label][0].plot(ax=ax, label=label, color=color, lw=1.5)
    ax.set_title("케이스별 오버나잇 누적 수익률 (종가 진입 → 익일 시가 청산)")
    ax.set_ylabel("누적 수익률"); ax.legend(fontsize=9); ax.grid(alpha=0.3)

    ax = axes[1]
    _, net_C = results["C  (외인선물z → 롱)"]
    yearly = net_C.groupby(net_C.index.year).sum()
    colors = ["steelblue" if v >= 0 else "salmon" for v in yearly.values]
    ax.bar(yearly.index, yearly.values * 100, color=colors, alpha=0.8)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_title("C안 (외인선물z → 롱) 연도별 수익률")
    ax.set_ylabel("수익률 (%)"); ax.grid(alpha=0.3, axis="y")

    ax = axes[2]
    sig = df.set_index("date")["for_put_z"]
    ax.fill_between(sig.index, sig, 0, where=(sig > 0),
                    color="red", alpha=0.4, label="풋 급증")
    ax.fill_between(sig.index, sig, 0, where=(sig < 0),
                    color="green", alpha=0.4, label="풋 감소")
    ax.axhline(PUT_SHORT_THR, color="red",     ls="--", lw=0.8, alpha=0.7,
               label=f"SHORT 임계 {PUT_SHORT_THR}")
    ax.axhline(PUT_VETO_THR,  color="darkred", ls=":",  lw=0.8, alpha=0.7,
               label=f"롱 거부 임계 {PUT_VETO_THR}")
    ax.set_title(f"외인 풋 Z-score (윈도우 {ZSCORE_WINDOW}일)")
    ax.set_ylabel("Z-score"); ax.legend(fontsize=9); ax.grid(alpha=0.3)

    ax = axes[3]
    sig2 = df.set_index("date")["fut_z"]
    ax.fill_between(sig2.index, sig2, 0, where=(sig2 > 0),
                    color="blue", alpha=0.4, label="선물 순매수")
    ax.fill_between(sig2.index, sig2, 0, where=(sig2 < 0),
                    color="brown", alpha=0.4, label="선물 순매도")
    ax.axhline(FUT_LONG_THR, color="blue", ls="--", lw=0.8, alpha=0.7,
               label=f"LONG 임계 {FUT_LONG_THR}")
    ax.set_title(f"외인 선물 Z-score (윈도우 {ZSCORE_WINDOW}일)")
    ax.set_ylabel("Z-score"); ax.legend(fontsize=9); ax.grid(alpha=0.3)

    plt.tight_layout()
    out = OUT_DIR / "overnight_result.png"
    plt.savefig(out, dpi=150); plt.show()
    print(f"\n차트 저장 → {out}")


# ════════════════════════════════════════════════════════
# 10. 메인
# ════════════════════════════════════════════════════════
def main():
    df = build_dataset()
    df = build_positions(df)

    results = {lb: backtest(df, col) for lb, (col, _) in CASES.items()}
    rows    = {lb: metrics(cum, net) for lb, (cum, net) in results.items()}
    perf    = pd.DataFrame(rows).T

    try:
        from tabulate import tabulate
        print("\n" + "=" * 68)
        print(" 오버나잇 전략 A 성과 (종가→익일시가, 비용 5bp/거래)")
        print("=" * 68)
        print(tabulate(perf, headers="keys", tablefmt="simple",
                       stralign="right", numalign="right"))
        print("=" * 70)
    except ImportError:
        print(perf.to_string())

    print("\n[스트레스 테스트 7bp]")
    stress_rows = {lb: metrics(*backtest(df, col, cost=STRESS_COST))
                   for lb, (col, _) in CASES.items()}
    stress_perf = pd.DataFrame(stress_rows).T
    try:
        print(tabulate(stress_perf, headers="keys", tablefmt="simple",
                       stralign="right", numalign="right"))
    except ImportError:
        print(stress_perf.to_string())

    perf.to_csv(OUT_DIR / "overnight_performance.csv", encoding="utf-8-sig")
    df.to_csv(OUT_DIR / "overnight_signals.csv", index=False, encoding="utf-8-sig")

    plot_results(df, results)
    return df, results


if __name__ == "__main__":
    df, results = main()
