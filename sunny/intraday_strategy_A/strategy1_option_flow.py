"""
전략 1 · 외국인 옵션 플로우 방향 신호
════════════════════════════════════════
개요
  외국인 투자자의 콜/풋 Z-score를 이용한 시장 방향 예측.

핵심 발견 (2010~2026, 인트라데이 open→close 기준)
  ┌────────────────────────────────────────────────────────┐
  │ call_z > 1 → 롱  :  Sharpe -0.26  (역방향 — 쓰지 말것) │
  │ put_z  > 1 → 숏  :  Sharpe +0.37  (유효)               │
  │ put_z  > 1 → 숏  :  Sharpe +0.37  (단독 유효)          │
  │ put_z  > 1 and call_z < 0 → 숏 : Sharpe +0.23 (정밀)  │
  └────────────────────────────────────────────────────────┘
  → 외국인 풋 급매수는 하락 예측 신호 (헤지 or 방향성)
    외국인 콜 급매수는 역방향 신호 (마켓메이커 헤지 추정)

신호 계산
  call_net(d)  = 외인 콜 매수금액 − 매도금액
  put_net(d)   = 외인 풋 매수금액 − 매도금액
  call_z(d)    = call_net 의 과거 60일 z-score
  put_z(d)     = put_net  의 과거 60일 z-score
  [모두 T-1 기준 → 당일 진입 신호]

케이스 정의
  A. 풋z>0.5 → 숏  (약한 필터)
  B. 풋z>1.0 → 숏  (강한 필터, 핵심)
  C. 풋z>1.0 AND 콜z<0 → 숏  (풋↑ + 콜↓ 동시, 정밀)
  D. 풋z>1.0 → 숏 / 콜z>1.0 AND 풋z<0 → 롱  (롱숏 혼합)

진입/청산
  전일 신호 → 당일 시가 진입 → 당일 종가 청산 (open-to-close)

데이터
  · 옵션 수급 : beomgu/data/krx_opt_investor_daily.csv
  · 가격      : beomgu/data/k200_index_daily.csv  (KOSPI200 지수)

비용
  왕복 2bp (미니선물 기준)

판정 구간 : 2010-01-04 ~ 2026-09-30
"""

import unicodedata
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

warnings.filterwarnings("ignore")
plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

# ═══════════════════════════════════════════════════════════
# 0. 경로 & 하이퍼파라미터
# ═══════════════════════════════════════════════════════════
REPO_DIR  = Path(__file__).resolve().parents[2]
DATA_DIR  = REPO_DIR / "beomgu" / "data"
OUT_DIR   = Path(__file__).parent / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

F_OPT   = DATA_DIR / "krx_opt_investor_daily.csv"
F_PRICE = DATA_DIR / "k200_index_daily.csv"

START_DATE  = "2010-01-01"
END_DATE    = "2026-09-30"
Z_WINDOW    = 60
BURN_IN     = 60
THR_WEAK    = 0.5
THR_STRONG  = 1.0

# 거래비용 (왕복)
# K200 미니선물: 수수료 ~0.5bp × 2 + 슬리피지 1틱(0.25pt/300pt ≈ 8bp) ≈ 10bp
# 바이낸스 무기한선물: taker 0.05% × 2 = 10bp, maker 0.02% × 2 = 4bp → 평균 6bp
COST_KR      = 0.0002   # 왕복  2bp (K200 미니선물: 수수료 ~1bp + 슬리피지 1틱 ~0.7bp)
COST_BINANCE = 0.0008   # 왕복  8bp (바이낸스 USDT 무기한: taker 10bp/maker 4bp 혼합 ~7bp + 슬리피지 0.5bp)
ROUND_TRIP   = COST_KR  # 기본값


# ═══════════════════════════════════════════════════════════
# 1. 데이터 로드
# ═══════════════════════════════════════════════════════════
def load_option_flow() -> pd.DataFrame:
    """외국인 콜/풋 순매수 금액. 반환: date | call_net | put_net"""
    raw = pd.read_csv(F_OPT)
    raw["date"] = pd.to_datetime(raw["date"])
    fo = raw[raw["investor"] == "foreign"].copy()

    call = fo[fo["cp"] == "C"].groupby("date")[["buy_amt", "sell_amt"]].sum()
    put  = fo[fo["cp"] == "P"].groupby("date")[["buy_amt", "sell_amt"]].sum()

    df = pd.DataFrame({
        "call_net": call["buy_amt"] - call["sell_amt"],
        "put_net":  put["buy_amt"]  - put["sell_amt"],
    })
    df.index = pd.to_datetime(df.index)
    return df.loc[(df.index >= START_DATE) & (df.index <= END_DATE)].sort_index()


def load_prices() -> pd.DataFrame:
    """KOSPI200 일봉. 반환: date | open | close | oc_ret"""
    df = pd.read_csv(F_PRICE, encoding="utf-8-sig")
    df.columns = df.columns.str.strip()
    df["date"] = pd.to_datetime(df["date"])
    df = (df[["date", "open", "close"]].dropna()
          .set_index("date")
          .sort_index())
    df = df.loc[(df.index >= START_DATE) & (df.index <= END_DATE)]
    df["oc_ret"] = df["close"] / df["open"] - 1
    return df


# ═══════════════════════════════════════════════════════════
# 2. 신호 생성
# ═══════════════════════════════════════════════════════════
def zscore_rolling(s: pd.Series, window: int = Z_WINDOW) -> pd.Series:
    mu  = s.rolling(window, min_periods=BURN_IN).mean()
    sig = s.rolling(window, min_periods=BURN_IN).std()
    return (s - mu) / sig.replace(0, np.nan)


def build_signals(flow: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    df = prices.join(flow, how="inner")

    df["call_z"] = zscore_rolling(df["call_net"])
    df["put_z"]  = zscore_rolling(df["put_net"])

    # T-1 신호 (당일 시가 진입용)
    df["call_z_lag"] = df["call_z"].shift(1)
    df["put_z_lag"]  = df["put_z"].shift(1)

    # 참고용
    df["call_net_b"] = df["call_net"] / 1e9
    df["put_net_b"]  = df["put_net"]  / 1e9

    return df.dropna(subset=["oc_ret"])


# ═══════════════════════════════════════════════════════════
# 3. 포지션 생성
# ═══════════════════════════════════════════════════════════
def build_positions(df: pd.DataFrame) -> pd.DataFrame:
    cz = df["call_z_lag"]
    pz = df["put_z_lag"]

    # A. 풋 z > 0.5 → 숏
    df["pos_A"] = np.where(pz > THR_WEAK, -1.0, 0.0)

    # B. 풋 z > 1.0 → 숏 (핵심 전략)
    df["pos_B"] = np.where(pz > THR_STRONG, -1.0, 0.0)

    # C. 풋z>1.0 AND 콜z<0 → 숏 (풋↑ + 콜 정상/감소, 정밀 필터)
    df["pos_C"] = np.where((pz > THR_STRONG) & (cz < 0), -1.0, 0.0)

    # D. 롱숏 혼합:
    #    풋z>1.0 → 숏
    #    콜z>1.0 AND 풋z<0 → 롱 (콜 급매수 + 풋 감소 = 순수 강세 시그널)
    df["pos_D"] = np.where(pz > THR_STRONG, -1.0,
                  np.where((cz > THR_STRONG) & (pz < 0), 1.0, 0.0))

    return df


# ═══════════════════════════════════════════════════════════
# 4. 백테스트 & 지표
# ═══════════════════════════════════════════════════════════
def backtest(df: pd.DataFrame, pos_col: str,
             cost: float = ROUND_TRIP) -> tuple[pd.Series, pd.Series]:
    v   = df.dropna(subset=[pos_col, "oc_ret"])
    pos = v[pos_col]
    net = v["oc_ret"] * pos - cost * pos.abs()
    cum = (1 + net).cumprod()
    return cum, net


def metrics(cum: pd.Series, net: pd.Series) -> dict:
    n_years = max((cum.index[-1] - cum.index[0]).days / 365.25, 1e-9)
    ann_ret = cum.iloc[-1] ** (1 / n_years) - 1
    ann_vol = net.std() * np.sqrt(252)
    sharpe  = ann_ret / ann_vol if ann_vol > 1e-9 else np.nan
    max_dd  = (cum / cum.cummax() - 1).min()
    active  = net[net != 0]
    win     = (active > 0).mean() if len(active) else np.nan
    return {
        "연환산수익률": f"{ann_ret:+.2%}",
        "연변동성":     f"{ann_vol:.2%}",
        "샤프지수":     f"{sharpe:.3f}",
        "최대낙폭":     f"{max_dd:.2%}",
        "승률":         f"{win:.2%}",
        "유효거래일":   int((net != 0).sum()),
        "전체거래일":   len(net),
    }


# ── 한글 포함 테이블 출력 헬퍼 ────────────────────────────
def _dw(s: str) -> int:
    """터미널 출력 폭 (한글·CJK = 2칸, 나머지 = 1칸)."""
    return sum(2 if unicodedata.east_asian_width(c) in ("W", "F") else 1
               for c in str(s))


def _rjust(s: str, width: int) -> str:
    return " " * max(width - _dw(str(s)), 0) + str(s)


def _ljust(s: str, width: int) -> str:
    return str(s) + " " * max(width - _dw(str(s)), 0)


def print_perf_table(perf: pd.DataFrame) -> None:
    cols      = list(perf.columns)
    rows      = list(perf.index)
    row_w     = max(_dw(r) for r in rows) + 2
    col_ws    = [max(_dw(c), max(_dw(str(perf.loc[r, c])) for r in rows)) + 2
                 for c in cols]
    sep_len   = row_w + sum(col_ws) + len(col_ws) * 2
    sep       = "─" * sep_len

    header = _ljust("", row_w) + "  " + "  ".join(
        _rjust(c, w) for c, w in zip(cols, col_ws))
    print(sep)
    print(header)
    print(sep)
    for r in rows:
        line = _ljust(r, row_w) + "  " + "  ".join(
            _rjust(str(perf.loc[r, c]), w) for c, w in zip(cols, col_ws))
        print(line)
    print(sep)


# ═══════════════════════════════════════════════════════════
# 5. 시각화
# ═══════════════════════════════════════════════════════════
CASES = {
    "A. 풋z > 0.5 → 숏":            "pos_A",
    "B. 풋z > 1.0 → 숏 (핵심)":     "pos_B",
    "C. 풋z > 1.0 & 콜z < 0 → 숏":  "pos_C",
    "D. 풋z↑ 숏 / 콜z↑ 롱 혼합":   "pos_D",
}
COLORS = ["tab:orange", "tab:red", "tab:purple", "tab:blue"]


def plot_results(df: pd.DataFrame, results: dict):
    fig = plt.figure(figsize=(16, 22))
    gs  = fig.add_gridspec(5, 1, hspace=0.45)

    # ── (1) 누적 수익률 ───────────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    for (label, _), color in zip(CASES.items(), COLORS):
        cum, _ = results[label]
        ax1.plot(cum.index, cum.values, label=label, color=color, lw=1.5)
    ax1.axhline(1, color="black", lw=0.7, ls="--")
    ax1.set_title("케이스별 누적 수익률  (시가 진입 → 종가 청산, 왕복 2bp)", fontsize=11)
    ax1.set_ylabel("누적 배수")
    ax1.legend(fontsize=8.5)
    ax1.grid(alpha=0.25)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # ── (2) 연도별 수익률 (B안) ───────────────────────────
    ax2 = fig.add_subplot(gs[1])
    _, net_b = results["B. 풋z > 1.0 → 숏 (핵심)"]
    yearly = net_b.resample("YE").apply(lambda x: (1 + x).prod() - 1)
    bar_colors = ["steelblue" if v >= 0 else "tomato" for v in yearly]
    ax2.bar(yearly.index, yearly.values * 100, color=bar_colors, width=300, alpha=0.85)
    for x, v in zip(yearly.index, yearly.values):
        ax2.text(x, v * 100 + np.sign(v) * 0.4,
                 f"{v*100:.1f}%", ha="center",
                 va="bottom" if v >= 0 else "top", fontsize=7.5)
    ax2.axhline(0, color="black", lw=0.6)
    ax2.set_title("연도별 수익률 — B안 (풋z > 1.0 → 숏)", fontsize=11)
    ax2.set_ylabel("수익률 (%)")
    ax2.grid(axis="y", alpha=0.25)

    # ── (3) 외인 콜/풋 순매수 (십억 원) ─────────────────
    ax3 = fig.add_subplot(gs[2])
    ax3.bar(df.index, df["call_net_b"], alpha=0.5, color="blue",
            width=1, label="콜 순매수")
    ax3.bar(df.index, -df["put_net_b"], alpha=0.5, color="red",
            width=1, label="풋 순매도(부호 반전)")
    ax3.axhline(0, color="black", lw=0.5)
    ax3.set_title("외국인 콜/풋 순매수 (십억 원, 풋은 부호 반전)", fontsize=11)
    ax3.set_ylabel("십억 원")
    ax3.legend(fontsize=8)
    ax3.grid(alpha=0.2)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # ── (4) 콜 z vs 풋 z ─────────────────────────────────
    ax4 = fig.add_subplot(gs[3])
    ax4.plot(df.index, df["call_z"], color="blue",   lw=0.6, alpha=0.7, label="콜 z-score")
    ax4.plot(df.index, df["put_z"],  color="red",    lw=0.6, alpha=0.7, label="풋 z-score")
    ax4.axhline( THR_STRONG, color="gray", ls="--", lw=0.8)
    ax4.axhline(-THR_STRONG, color="gray", ls="--", lw=0.8)
    ax4.axhline( THR_WEAK,   color="gray", ls=":",  lw=0.6)
    ax4.axhline(-THR_WEAK,   color="gray", ls=":",  lw=0.6)
    ax4.set_title("콜 / 풋 z-score (점선: ±1.0, 점점선: ±0.5)", fontsize=11)
    ax4.set_ylabel("Z-Score")
    ax4.legend(fontsize=8)
    ax4.grid(alpha=0.25)
    ax4.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # ── (5) KOSPI200 + B안 진입 구간 ──────────────────────
    ax5 = fig.add_subplot(gs[4])
    ax5.plot(df.index, df["close"], color="black", lw=0.9, label="KOSPI200 종가")
    short_days = df.index[df["pos_B"] != 0]
    for d in short_days:
        ax5.axvspan(d, d + pd.Timedelta(days=1), color="red", alpha=0.12)
    ax5.set_title("KOSPI200 + B안 숏 진입일 (빨간 구간: 풋z > 1.0)", fontsize=11)
    ax5.set_ylabel("KOSPI200")
    ax5.legend(fontsize=8)
    ax5.grid(alpha=0.25)
    ax5.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plt.suptitle("전략 1 · 외국인 옵션 플로우 방향 신호\n[open→close 인트라데이, 왕복 2bp 포함]",
                 fontsize=13, fontweight="bold", y=1.005)

    out = OUT_DIR / "strategy1_option_flow.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"\n차트 저장 → {out}")


# ═══════════════════════════════════════════════════════════
# 6. 메인
# ═══════════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print("전략 1 · 외국인 옵션 플로우 방향 신호")
    print("=" * 60)

    print("\n[1/3] 데이터 로드...")
    flow   = load_option_flow()
    prices = load_prices()
    print(f"      옵션 플로우: {len(flow)}일  "
          f"{flow.index.min().date()} ~ {flow.index.max().date()}")
    print(f"      가격 데이터: {len(prices)}일  "
          f"{prices.index.min().date()} ~ {prices.index.max().date()}")

    print("\n[2/3] 신호 생성 및 포지션 구성...")
    df = build_signals(flow, prices)
    df = build_positions(df)
    valid = df.dropna(subset=["put_z_lag"])
    print(f"      유효 구간: {len(valid)}일  "
          f"{valid.index.min().date()} ~ {valid.index.max().date()}")
    print(f"      콜z 범위: {valid['call_z_lag'].min():.2f} ~ {valid['call_z_lag'].max():.2f}")
    print(f"      풋z 범위: {valid['put_z_lag'].min():.2f} ~ {valid['put_z_lag'].max():.2f}")

    print("\n[3/3] 백테스트...")
    COST_SCENARIOS = {
        "K200 미니선물 (2bp)":         COST_KR,
        "바이낸스 무기한선물 (8bp)":   COST_BINANCE,
    }

    all_perf = {}
    all_results = {}
    for cost_label, cost in COST_SCENARIOS.items():
        results = {label: backtest(df, col, cost=cost) for label, col in CASES.items()}
        all_results[cost_label] = results
        rows = {label: metrics(cum, net) for label, (cum, net) in results.items()}
        all_perf[cost_label] = pd.DataFrame(rows).T

    for cost_label, perf in all_perf.items():
        print(f"\n[ {cost_label} ]")
        print_perf_table(perf)

    # 연도별 B안 비교
    print("\n[B안] 연도별 수익률 비교 (풋z > 1.0 → 숏)")
    print(f"  {'연도':>4}  {'K200 미니(2bp)':>14}  {'바이낸스(8bp)':>13}")
    print("  " + "─" * 38)
    b_kr  = all_results["K200 미니선물 (2bp)"]["B. 풋z > 1.0 → 숏 (핵심)"][1]
    b_bnc = all_results["바이낸스 무기한선물 (8bp)"]["B. 풋z > 1.0 → 숏 (핵심)"][1]
    for yr in b_kr.resample("YE").apply(lambda x: (1+x).prod()-1).items():
        year, ret_kr  = yr
        ret_bnc = b_bnc.resample("YE").apply(lambda x: (1+x).prod()-1).get(year, float("nan"))
        bar = "█" * int(abs(ret_kr) * 300)
        s_kr  = f"{'+' if ret_kr >=0 else '-'}{abs(ret_kr)*100:4.1f}%"
        s_bnc = f"{'+' if ret_bnc>=0 else '-'}{abs(ret_bnc)*100:4.1f}%"
        print(f"  {year.year}  {s_kr:>15}  {s_bnc:>13}  {bar}")

    # 저장 (기본: K200 미니선물 비용 기준)
    perf = all_perf["K200 미니선물 (10bp)"]
    perf.to_csv(OUT_DIR / "strategy1_performance.csv", encoding="utf-8-sig")
    sig_cols = ["open", "close", "oc_ret",
                "call_net_b", "put_net_b",
                "call_z", "put_z", "call_z_lag", "put_z_lag",
                "pos_A", "pos_B", "pos_C", "pos_D"]
    df[sig_cols].to_csv(OUT_DIR / "strategy1_signals.csv", encoding="utf-8-sig")
    print(f"\n결과 저장 → {OUT_DIR}")

    plot_results(df, all_results["K200 미니선물 (2bp)"])
    return df, all_results


if __name__ == "__main__":
    df, results = main()
