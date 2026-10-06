"""
전략 3 · 외국인 vs 개인 옵션 방향 갈림 (스마트머니 vs 역지표)
════════════════════════════════════════════════════════════
개요
  외국인(스마트머니)과 개인(역지표)의 옵션 방향이 갈릴 때 외국인을 따르는 전략.

핵심 아이디어 (전략 1·2와의 차별점)
  ┌──────────────────────────────────────────────────────────────────┐
  │ 전략 1 : 외국인 풋z > 1 → 숏 (외국인 단독)                       │
  │ 전략 3 : (외국인 풋z > 1) AND (개인 콜z > 0) → 숏               │
  │          외국인이 풋 급매수(하락 베팅) + 개인이 콜 순매수(낙관)   │
  │          → 두 지표 동시 확인 = 더 신뢰도 높은 숏 신호             │
  └──────────────────────────────────────────────────────────────────┘

개인 역지표 근거
  · 개인 콜 급매수 = FOMO / 추격 매수 (시장 고점 근처) → 역추세 숏
  · 개인 풋 급매수 = 패닉 / 공포 매도 (시장 저점 근처) → 역추세 롱

케이스 정의
  A. 개인 콜z > 1.0 → 숏           (개인 역추세 단독 숏)
  B. 개인 풋z > 1.0 → 롱           (개인 패닉 역추세 단독 롱)
  C. 외인 풋z > 1.0 AND 개인 콜z > 0 → 숏   [핵심: 이중 확인]
  D. C 숏 OR (외인 콜z < -1 AND 개인 풋z > 1 → 롱)  [롱숏 혼합]

진입/청산
  전일 신호 → 당일 시가 진입 → 당일 종가 청산 (open-to-close)

데이터
  · 옵션 수급 : beomgu/data/krx_opt_investor_daily.csv
  · 가격      : beomgu/data/k200_index_daily.csv

비용
  왕복 3bp (키움증권 미니선물 기준: 수수료 ~1bp + 슬리피지 ~2bp)

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

START_DATE = "2010-01-01"
END_DATE   = "2026-09-30"
Z_WINDOW   = 60
BURN_IN    = 60
THR_STRONG = 1.0
THR_WEAK   = 0.0

# 왕복 3bp (키움증권 미니선물: 수수료 ~1bp + 슬리피지 ~2bp)
ROUND_TRIP = 0.0003


# ═══════════════════════════════════════════════════════════
# 1. 데이터 로드
# ═══════════════════════════════════════════════════════════
def load_option_flow() -> pd.DataFrame:
    """
    외국인 / 개인 콜·풋 순매수 금액 추출.
    반환: date | for_call_net | for_put_net | ind_call_net | ind_put_net
    """
    raw = pd.read_csv(F_OPT)
    raw["date"] = pd.to_datetime(raw["date"])
    raw = raw[(raw["date"] >= START_DATE) & (raw["date"] <= END_DATE)]

    def net(investor: str, cp: str) -> pd.Series:
        sub = raw[(raw["investor"] == investor) & (raw["cp"] == cp)]
        g = sub.groupby("date")[["buy_amt", "sell_amt"]].sum()
        return (g["buy_amt"] - g["sell_amt"]).rename(f"{investor}_{cp}_net")

    for_call = net("foreign",    "C")
    for_put  = net("foreign",    "P")
    ind_call = net("individual", "C")
    ind_put  = net("individual", "P")

    df = pd.concat([for_call, for_put, ind_call, ind_put], axis=1).dropna()
    df.columns = ["for_call_net", "for_put_net", "ind_call_net", "ind_put_net"]
    df.index = pd.to_datetime(df.index)
    return df.sort_index()


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

    # z-score 계산 (당일 기준)
    df["for_call_z"] = zscore_rolling(df["for_call_net"])
    df["for_put_z"]  = zscore_rolling(df["for_put_net"])
    df["ind_call_z"] = zscore_rolling(df["ind_call_net"])
    df["ind_put_z"]  = zscore_rolling(df["ind_put_net"])

    # T-1 lag (당일 시가 진입 신호, 룩어헤드 없음)
    df["for_call_z_lag"] = df["for_call_z"].shift(1)
    df["for_put_z_lag"]  = df["for_put_z"].shift(1)
    df["ind_call_z_lag"] = df["ind_call_z"].shift(1)
    df["ind_put_z_lag"]  = df["ind_put_z"].shift(1)

    # 참고용: 십억 원 단위
    for col in ["for_call_net", "for_put_net", "ind_call_net", "ind_put_net"]:
        df[f"{col}_b"] = df[col] / 1e9

    return df.dropna(subset=["oc_ret"])


# ═══════════════════════════════════════════════════════════
# 3. 포지션 생성
# ═══════════════════════════════════════════════════════════
def build_positions(df: pd.DataFrame) -> pd.DataFrame:
    fc  = df["for_call_z_lag"]
    fp  = df["for_put_z_lag"]
    ic  = df["ind_call_z_lag"]
    ip  = df["ind_put_z_lag"]

    # A. 개인 콜z > 1 → 숏 (FOMO 역추세)
    df["pos_A"] = np.where(ic > THR_STRONG, -1.0, 0.0)

    # B. 개인 풋z > 1 → 롱 (패닉 역추세)
    df["pos_B"] = np.where(ip > THR_STRONG, 1.0, 0.0)

    # C. 외인 풋z > 1 AND 개인 콜z > 0 → 숏 (이중 확인, 핵심)
    df["pos_C"] = np.where(
        (fp > THR_STRONG) & (ic > THR_WEAK), -1.0, 0.0
    )

    # D. C 숏 + (외인 콜z < -1 AND 개인 풋z > 1 → 롱) 롱숏 혼합
    short_cond = (fp > THR_STRONG) & (ic > THR_WEAK)
    long_cond  = (fc < -THR_STRONG) & (ip > THR_STRONG)
    df["pos_D"] = np.where(short_cond, -1.0,
                  np.where(long_cond,   1.0, 0.0))

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
    return sum(2 if unicodedata.east_asian_width(c) in ("W", "F") else 1
               for c in str(s))


def _rjust(s: str, width: int) -> str:
    return " " * max(width - _dw(str(s)), 0) + str(s)


def _ljust(s: str, width: int) -> str:
    return str(s) + " " * max(width - _dw(str(s)), 0)


def print_perf_table(perf: pd.DataFrame) -> None:
    cols   = list(perf.columns)
    rows   = list(perf.index)
    row_w  = max(_dw(r) for r in rows) + 2
    col_ws = [max(_dw(c), max(_dw(str(perf.loc[r, c])) for r in rows)) + 2
              for c in cols]
    sep    = "─" * (row_w + sum(col_ws) + len(col_ws) * 2)
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
    "A. 개인 콜z > 1 → 숏 (역추세)":           "pos_A",
    "B. 개인 풋z > 1 → 롱 (패닉 역추세)":       "pos_B",
    "C. 외인 풋z > 1 & 개인 콜z > 0 → 숏 (핵심)": "pos_C",
    "D. C + 외인 콜z<-1 & 개인 풋z>1 → 롱 혼합": "pos_D",
}
COLORS = ["tab:orange", "tab:green", "tab:red", "tab:purple"]


def plot_results(df: pd.DataFrame, results: dict):
    fig = plt.figure(figsize=(16, 24))
    gs  = fig.add_gridspec(6, 1, hspace=0.45)

    # ── (1) 누적 수익률 ───────────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    for (label, _), color in zip(CASES.items(), COLORS):
        cum, _ = results[label]
        ax1.plot(cum.index, cum.values, label=label, color=color, lw=1.5)
    ax1.axhline(1, color="black", lw=0.7, ls="--")
    ax1.set_title("케이스별 누적 수익률  (시가 진입 → 종가 청산, 왕복 3bp)", fontsize=11)
    ax1.set_ylabel("누적 배수")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.25)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # ── (2) 연도별 수익률 (C안) ───────────────────────────
    ax2 = fig.add_subplot(gs[1])
    core_label = "C. 외인 풋z > 1 & 개인 콜z > 0 → 숏 (핵심)"
    _, net_c = results[core_label]
    yearly = net_c.resample("YE").apply(lambda x: (1 + x).prod() - 1)
    bar_colors = ["steelblue" if v >= 0 else "tomato" for v in yearly]
    ax2.bar(yearly.index, yearly.values * 100, color=bar_colors, width=300, alpha=0.85)
    for x, v in zip(yearly.index, yearly.values):
        ax2.text(x, v * 100 + np.sign(v) * 0.4,
                 f"{v*100:.1f}%", ha="center",
                 va="bottom" if v >= 0 else "top", fontsize=7.5)
    ax2.axhline(0, color="black", lw=0.6)
    ax2.set_title("연도별 수익률 — C안 (외인 풋z > 1 & 개인 콜z > 0 → 숏)", fontsize=11)
    ax2.set_ylabel("수익률 (%)")
    ax2.grid(axis="y", alpha=0.25)

    # ── (3) 외인 풋 z-score vs 개인 콜 z-score ──────────
    ax3 = fig.add_subplot(gs[2])
    ax3.plot(df.index, df["for_put_z"],  color="red",   lw=0.8, alpha=0.8, label="외인 풋 z-score")
    ax3.plot(df.index, df["ind_call_z"], color="blue",  lw=0.8, alpha=0.8, label="개인 콜 z-score")
    ax3.axhline( THR_STRONG, color="gray", ls="--", lw=0.8)
    ax3.axhline(-THR_STRONG, color="gray", ls="--", lw=0.8)
    ax3.axhline(0, color="black", lw=0.4, alpha=0.5)
    ax3.fill_between(df.index, 0, df["for_put_z"],
                     where=(df["for_put_z"] > THR_STRONG) & (df["ind_call_z"] > 0),
                     color="red", alpha=0.2, label="C안 진입 구간")
    ax3.set_title("외인 풋 z-score vs 개인 콜 z-score  (빨간 영역: C안 진입 조건)", fontsize=11)
    ax3.set_ylabel("Z-Score")
    ax3.legend(fontsize=8)
    ax3.grid(alpha=0.25)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # ── (4) 외인 콜 z vs 개인 풋 z (D안 롱 조건) ─────────
    ax4 = fig.add_subplot(gs[3])
    ax4.plot(df.index, df["for_call_z"], color="blue", lw=0.8, alpha=0.8, label="외인 콜 z-score")
    ax4.plot(df.index, df["ind_put_z"],  color="orange", lw=0.8, alpha=0.8, label="개인 풋 z-score")
    ax4.axhline( THR_STRONG, color="gray", ls="--", lw=0.8)
    ax4.axhline(-THR_STRONG, color="gray", ls="--", lw=0.8)
    ax4.axhline(0, color="black", lw=0.4, alpha=0.5)
    ax4.fill_between(df.index, 0, df["ind_put_z"],
                     where=(df["for_call_z"] < -THR_STRONG) & (df["ind_put_z"] > THR_STRONG),
                     color="green", alpha=0.25, label="D안 롱 진입 구간")
    ax4.set_title("외인 콜 z-score vs 개인 풋 z-score  (녹색 영역: D안 롱 진입 조건)", fontsize=11)
    ax4.set_ylabel("Z-Score")
    ax4.legend(fontsize=8)
    ax4.grid(alpha=0.25)
    ax4.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # ── (5) 외인·개인 순매수 금액 (십억 원) ──────────────
    ax5 = fig.add_subplot(gs[4])
    ax5.bar(df.index, df["for_put_net_b"],  alpha=0.5, color="red",   width=1, label="외인 풋 순매수")
    ax5.bar(df.index, -df["ind_call_net_b"], alpha=0.5, color="blue", width=1, label="개인 콜 순매수(부호반전)")
    ax5.axhline(0, color="black", lw=0.5)
    ax5.set_title("외인 풋 순매수 vs 개인 콜 순매수 (십억 원, 개인 부호 반전)", fontsize=11)
    ax5.set_ylabel("십억 원")
    ax5.legend(fontsize=8)
    ax5.grid(alpha=0.2)
    ax5.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # ── (6) KOSPI200 + C안 진입 구간 ─────────────────────
    ax6 = fig.add_subplot(gs[5])
    ax6.plot(df.index, df["close"], color="black", lw=0.9, label="KOSPI200 종가")
    short_days = df.index[df["pos_C"] == -1]
    for d in short_days:
        ax6.axvspan(d, d + pd.Timedelta(days=1), color="red", alpha=0.12)
    ax6.set_title("KOSPI200 + C안 숏 진입일 (빨간 구간)", fontsize=11)
    ax6.set_ylabel("KOSPI200")
    ax6.legend(fontsize=8)
    ax6.grid(alpha=0.25)
    ax6.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plt.suptitle(
        "전략 3 · 외국인 vs 개인 옵션 방향 갈림 (스마트머니 vs 역지표)\n"
        "[open→close 인트라데이, 왕복 3bp 포함]",
        fontsize=13, fontweight="bold", y=1.005)

    out = OUT_DIR / "strategy3_smart_vs_dumb.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"\n차트 저장 → {out}")


# ═══════════════════════════════════════════════════════════
# 6. 메인
# ═══════════════════════════════════════════════════════════
def main():
    print("=" * 65)
    print("전략 3 · 외국인 vs 개인 옵션 방향 갈림 (스마트머니 vs 역지표)")
    print("=" * 65)

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
    valid = df.dropna(subset=["for_put_z_lag", "ind_call_z_lag"])
    print(f"      유효 구간: {len(valid)}일  "
          f"{valid.index.min().date()} ~ {valid.index.max().date()}")
    print(f"      외인 풋z 범위: {valid['for_put_z_lag'].min():.2f} ~ "
          f"{valid['for_put_z_lag'].max():.2f}")
    print(f"      개인 콜z 범위: {valid['ind_call_z_lag'].min():.2f} ~ "
          f"{valid['ind_call_z_lag'].max():.2f}")
    print(f"\n      진입 빈도:")
    for label, col in CASES.items():
        n = int((df[col] != 0).sum())
        print(f"        {label}: {n}일")

    print(f"\n[3/3] 백테스트...")
    print(f"      거래비용: 왕복 {ROUND_TRIP*10000:.1f}bp "
          f"(키움 수수료 ~1bp + 슬리피지 ~2bp)")
    results = {label: backtest(df, col) for label, col in CASES.items()}

    rows = {label: metrics(cum, net) for label, (cum, net) in results.items()}
    perf = pd.DataFrame(rows).T
    print()
    print_perf_table(perf)

    # 연도별 C안
    core_label = "C. 외인 풋z > 1 & 개인 콜z > 0 → 숏 (핵심)"
    _, net_c = results[core_label]
    yearly = net_c.resample("YE").apply(lambda x: (1 + x).prod() - 1)
    print(f"\n[C안] 연도별 수익률 (외인 풋z > 1 & 개인 콜z > 0 → 숏):")
    for year, ret in yearly.items():
        bar  = "█" * int(abs(ret) * 300)
        sign = "+" if ret >= 0 else "-"
        print(f"  {year.year}  {sign}{abs(ret)*100:4.1f}%  {bar}")

    # 저장
    perf.to_csv(OUT_DIR / "strategy3_performance.csv", encoding="utf-8-sig")
    sig_cols = ["open", "close", "oc_ret",
                "for_call_net_b", "for_put_net_b",
                "ind_call_net_b", "ind_put_net_b",
                "for_call_z", "for_put_z", "ind_call_z", "ind_put_z",
                "for_call_z_lag", "for_put_z_lag",
                "ind_call_z_lag", "ind_put_z_lag",
                "pos_A", "pos_B", "pos_C", "pos_D"]
    df[sig_cols].to_csv(OUT_DIR / "strategy3_signals.csv", encoding="utf-8-sig")
    print(f"\n결과 저장 → {OUT_DIR}")

    plot_results(df, results)
    return df, results


if __name__ == "__main__":
    df, results = main()
