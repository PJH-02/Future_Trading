"""
전략 2 · 미국 야간 역추세 + 외국인 선물 순방향 결합
════════════════════════════════════════════════════
핵심 발견
  ┌─────────────────────────────────────────────────────────┐
  │ ES 순방향 (따라가기) : Sharpe -0.63  → 시가에 이미 반영  │
  │ ES 역추세 (역방향)   : Sharpe +0.37  → 갭 되돌림 패턴   │
  │ 외인선물 순방향      : Sharpe +0.52  → 실제 매수 압력    │
  │ ES역추세 + 외인순방향: Sharpe +0.48  (결합, 핵심)        │
  │ 위 + 외인z > ±0.5   : Sharpe +0.50  (정밀 필터)         │
  └─────────────────────────────────────────────────────────┘

신호 해석
  S1 (ES 역추세): 미국 상승 → 한국 시가 갭업 → 장 중 되돌림 → 숏
                  미국 하락 → 한국 시가 갭다운 → 장 중 회복 → 롱
  S2 (외인 선물): 외국인 순매수 → 실제 매수 압력 → 롱
                  외국인 순매도 → 실제 매도 압력 → 숏

케이스 정의
  A. ES 역추세만                          (항상 포지션)
  B. 외인선물 순방향만                     (항상 포지션)
  C. A 방향 = B 방향 일치 시 진입         (핵심, AND 결합)
  D. C 조건 + 외인 z-score > ±0.5        (신호 강도 필터)

진입/청산
  전일 신호 → 당일 시가 진입 → 당일 종가 청산 (open-to-close)

비용
  왕복 4.5bp (수수료 3bp + 슬리피지 1.5bp, K200 미니선물 기준)

데이터
  · 미국 선물 : beomgu/data/es_daily.csv  (S&P500 E-mini, 2000~)
  · 외인 선물 : beomgu/data/naver_fut_investor_daily.csv  (2010~)
  · 가격      : beomgu/data/k200_index_daily.csv

판정 구간 : 2010-09-20 ~ 2026-09-17
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
# 0. 경로 & 파라미터
# ═══════════════════════════════════════════════════════════
REPO_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_DIR / "beomgu" / "data"
OUT_DIR  = Path(__file__).parent / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

F_ES    = DATA_DIR / "es_daily.csv"
F_FUT   = DATA_DIR / "naver_fut_investor_daily.csv"
F_PRICE = DATA_DIR / "k200_index_daily.csv"

START_DATE = "2010-09-20"
END_DATE   = "2026-09-30"
Z_WINDOW   = 60
BURN_IN    = 60
THR_Z      = 0.5

# 키움증권 미니선물 수수료 ~1bp + 슬리피지 ~2bp = 왕복 3bp
ROUND_TRIP = 0.0003


# ═══════════════════════════════════════════════════════════
# 1. 데이터 로드
# ═══════════════════════════════════════════════════════════
def load_es() -> pd.Series:
    """ES 종가 일별 수익률."""
    df = pd.read_csv(F_ES, encoding="utf-8-sig")
    df["date"] = pd.to_datetime(df["date"])
    df = df[["date", "close"]].dropna().set_index("date").sort_index()
    ret = df["close"].pct_change()
    return ret.loc[(ret.index >= START_DATE) & (ret.index <= END_DATE)]


def load_foreign_futures() -> pd.Series:
    """외국인 코스피200 선물 일별 순매수 계약수."""
    df = pd.read_csv(F_FUT, encoding="utf-8-sig")
    df["date"] = pd.to_datetime(df["date"])
    s = (df[["date", "net_foreign"]].dropna()
           .set_index("date")["net_foreign"].sort_index())
    return s.loc[(s.index >= START_DATE) & (s.index <= END_DATE)]


def load_prices() -> pd.DataFrame:
    """KOSPI200 일봉."""
    df = pd.read_csv(F_PRICE, encoding="utf-8-sig")
    df.columns = df.columns.str.strip()
    df["date"] = pd.to_datetime(df["date"])
    df = (df[["date", "open", "close"]].dropna()
            .set_index("date").sort_index())
    df = df.loc[(df.index >= START_DATE) & (df.index <= END_DATE)]
    df["oc_ret"] = df["close"] / df["open"] - 1
    return df


# ═══════════════════════════════════════════════════════════
# 2. 신호 생성
# ═══════════════════════════════════════════════════════════
def build_signals(es_ret: pd.Series,
                  fut_net: pd.Series,
                  prices: pd.DataFrame) -> pd.DataFrame:
    df = prices.copy()
    df = df.join(es_ret.rename("es_ret"),   how="left")
    df = df.join(fut_net.rename("fut_net"), how="left")

    # S1: ES 역추세 방향 (T-1) — 미국 상승 → 숏(-1), 미국 하락 → 롱(+1)
    df["es_rev_lag"] = -np.sign(df["es_ret"].shift(1))

    # S2: 외인 선물 순방향 (T-1) — 순매수 → 롱(+1), 순매도 → 숏(-1)
    df["fut_dir_lag"] = np.sign(df["fut_net"].shift(1))

    # 외인 선물 z-score (T-1)
    mu  = df["fut_net"].rolling(Z_WINDOW, min_periods=BURN_IN).mean()
    sig = df["fut_net"].rolling(Z_WINDOW, min_periods=BURN_IN).std()
    df["fut_z"]     = (df["fut_net"] - mu) / sig.replace(0, np.nan)
    df["fut_z_lag"] = df["fut_z"].shift(1)

    return df.dropna(subset=["oc_ret"])


# ═══════════════════════════════════════════════════════════
# 3. 포지션 생성
# ═══════════════════════════════════════════════════════════
def build_positions(df: pd.DataFrame) -> pd.DataFrame:
    es  = df["es_rev_lag"]    # ES 역추세 방향
    fd  = df["fut_dir_lag"]   # 외인 선물 순방향
    fz  = df["fut_z_lag"]

    # A. ES 역추세만
    df["pos_A"] = es.where(es != 0, 0)

    # B. 외인 선물 순방향만
    df["pos_B"] = fd.where(fd != 0, 0)

    # C. 두 신호 방향 일치 시 진입 (핵심)
    agree = (es == fd) & (es != 0)
    df["pos_C"] = es.where(agree, 0)

    # D. C 조건 + 외인 z-score > ±0.5
    strong = fz.abs() >= THR_Z
    df["pos_D"] = es.where(agree & strong, 0)

    return df


# ═══════════════════════════════════════════════════════════
# 4. 백테스트 & 지표
# ═══════════════════════════════════════════════════════════
def backtest(df: pd.DataFrame, pos_col: str) -> tuple[pd.Series, pd.Series]:
    v   = df.dropna(subset=[pos_col, "oc_ret"])
    pos = v[pos_col]
    net = v["oc_ret"] * pos - ROUND_TRIP * pos.abs()
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
    "A. ES 역추세만":                "pos_A",
    "B. 외인선물 순방향만":           "pos_B",
    "C. ES역추세 + 외인순방향 (핵심)": "pos_C",
    "D. C + 외인z > ±0.5":           "pos_D",
}
COLORS = ["tab:blue", "tab:orange", "tab:red", "tab:purple"]


def plot_results(df: pd.DataFrame, results: dict):
    fig = plt.figure(figsize=(16, 20))
    gs  = fig.add_gridspec(5, 1, hspace=0.45)

    # (1) 누적 수익률
    ax1 = fig.add_subplot(gs[0])
    for (label, _), color in zip(CASES.items(), COLORS):
        cum, _ = results[label]
        ax1.plot(cum.index, cum.values, label=label, color=color, lw=1.5)
    ax1.axhline(1, color="black", lw=0.7, ls="--")
    ax1.set_title("케이스별 누적 수익률  (시가→종가, 왕복 4.5bp)", fontsize=11)
    ax1.set_ylabel("누적 배수")
    ax1.legend(fontsize=8.5)
    ax1.grid(alpha=0.25)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # (2) 연도별 수익률 (C안)
    ax2 = fig.add_subplot(gs[1])
    _, net_c = results["C. ES역추세 + 외인순방향 (핵심)"]
    yearly = net_c.resample("YE").apply(lambda x: (1 + x).prod() - 1)
    bc = ["steelblue" if v >= 0 else "tomato" for v in yearly]
    ax2.bar(yearly.index, yearly * 100, color=bc, width=300, alpha=0.85)
    for x, v in zip(yearly.index, yearly):
        ax2.text(x, v * 100 + np.sign(v) * 0.4, f"{v*100:.1f}%",
                 ha="center", va="bottom" if v >= 0 else "top", fontsize=7.5)
    ax2.axhline(0, color="black", lw=0.6)
    ax2.set_title("연도별 수익률 — C안 (ES역추세 + 외인선물 순방향 일치)", fontsize=11)
    ax2.set_ylabel("수익률 (%)")
    ax2.grid(axis="y", alpha=0.25)

    # (3) ES 전일 수익률
    ax3 = fig.add_subplot(gs[2])
    ax3.fill_between(df.index, df["es_ret"] * 100, 0,
                     where=(df["es_ret"] >= 0), color="blue", alpha=0.4, label="ES 상승 → 다음날 숏 신호")
    ax3.fill_between(df.index, df["es_ret"] * 100, 0,
                     where=(df["es_ret"] < 0),  color="red",  alpha=0.4, label="ES 하락 → 다음날 롱 신호")
    ax3.set_title("S&P500 E-mini (ES) 전일 수익률 — 역추세 신호원", fontsize=11)
    ax3.set_ylabel("%")
    ax3.legend(fontsize=8)
    ax3.grid(alpha=0.25)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # (4) 외인 선물 z-score
    ax4 = fig.add_subplot(gs[3])
    ax4.fill_between(df.index, df["fut_z"], 0,
                     where=(df["fut_z"] >= 0), color="blue", alpha=0.35, label="순매수")
    ax4.fill_between(df.index, df["fut_z"], 0,
                     where=(df["fut_z"] < 0),  color="red",  alpha=0.35, label="순매도")
    ax4.axhline( THR_Z, color="gray", ls="--", lw=0.8)
    ax4.axhline(-THR_Z, color="gray", ls="--", lw=0.8)
    ax4.set_title(f"외국인 선물 순매수 z-score (점선: ±{THR_Z})", fontsize=11)
    ax4.set_ylabel("Z-Score")
    ax4.legend(fontsize=8)
    ax4.grid(alpha=0.25)
    ax4.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # (5) KOSPI200 + C안 진입
    ax5 = fig.add_subplot(gs[4])
    ax5.plot(df.index, df["close"], color="black", lw=0.9, label="KOSPI200")
    pos_c = df["pos_C"].fillna(0)
    for d in df.index[pos_c > 0]:
        ax5.axvspan(d, d + pd.Timedelta(days=1), color="blue", alpha=0.12)
    for d in df.index[pos_c < 0]:
        ax5.axvspan(d, d + pd.Timedelta(days=1), color="red",  alpha=0.12)
    ax5.set_title("KOSPI200 + C안 진입일 (파랑=롱, 빨강=숏)", fontsize=11)
    ax5.set_ylabel("KOSPI200")
    ax5.legend(fontsize=8)
    ax5.grid(alpha=0.25)
    ax5.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plt.suptitle("전략 2 · 미국 야간 역추세 + 외국인 선물 순방향 결합  [open→close, 왕복 4.5bp]",
                 fontsize=13, fontweight="bold", y=1.005)

    out = OUT_DIR / "strategy2_us_foreign_flow.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"\n차트 저장 → {out}")


# ═══════════════════════════════════════════════════════════
# 6. 메인
# ═══════════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print("전략 2 · 미국 야간 역추세 + 외국인 선물 순방향 결합")
    print("=" * 60)

    print("\n[1/3] 데이터 로드...")
    es_ret  = load_es()
    fut_net = load_foreign_futures()
    prices  = load_prices()
    print(f"      ES 수익률   : {len(es_ret)}일  "
          f"{es_ret.index.min().date()} ~ {es_ret.index.max().date()}")
    print(f"      외인 선물   : {len(fut_net)}일  "
          f"{fut_net.index.min().date()} ~ {fut_net.index.max().date()}")
    print(f"      가격 데이터 : {len(prices)}일  "
          f"{prices.index.min().date()} ~ {prices.index.max().date()}")

    print("\n[2/3] 신호 생성 및 포지션 구성...")
    df = build_signals(es_ret, fut_net, prices)
    df = build_positions(df)
    valid = df.dropna(subset=["es_rev_lag", "fut_dir_lag"])
    print(f"      유효 구간   : {len(valid)}일  "
          f"{valid.index.min().date()} ~ {valid.index.max().date()}")
    agree = (valid["es_rev_lag"] == valid["fut_dir_lag"]) & (valid["es_rev_lag"] != 0)
    print(f"      신호 일치율 : {agree.mean():.1%}  ({agree.sum()}일 / {len(valid)}일)")

    print(f"\n[3/3] 백테스트... (왕복 {ROUND_TRIP*10000:.1f}bp)")
    results = {label: backtest(df, col) for label, col in CASES.items()}

    rows = {label: metrics(cum, net) for label, (cum, net) in results.items()}
    perf = pd.DataFrame(rows).T
    print()
    print_perf_table(perf)

    # 연도별 C안
    _, net_c = results["C. ES역추세 + 외인순방향 (핵심)"]
    yearly = net_c.resample("YE").apply(lambda x: (1 + x).prod() - 1)
    print("\n[C안] 연도별 수익률:")
    for year, ret in yearly.items():
        bar  = "█" * int(abs(ret) * 200)
        sign = "+" if ret >= 0 else "-"
        print(f"  {year.year}  {sign}{abs(ret)*100:4.1f}%  {bar}")

    # 저장
    perf.to_csv(OUT_DIR / "strategy2_performance.csv", encoding="utf-8-sig")
    sig_cols = ["open", "close", "oc_ret",
                "es_ret", "fut_net", "fut_z",
                "es_rev_lag", "fut_dir_lag", "fut_z_lag",
                "pos_A", "pos_B", "pos_C", "pos_D"]
    df[sig_cols].to_csv(OUT_DIR / "strategy2_signals.csv", encoding="utf-8-sig")
    print(f"\n결과 저장 → {OUT_DIR}")

    plot_results(df, results)
    return df, results


if __name__ == "__main__":
    df, results = main()
