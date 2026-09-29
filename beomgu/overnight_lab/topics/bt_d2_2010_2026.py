"""D2 3차 셀 전략 백테스트 2010-01 ~ 2026-09 (성과 보고용, 판정 아님)

매매 규칙(3차 고정 사양 round3_D2/spec.json 그대로, 파라미터 변경 없음)
- 전날 밤: 직전 250거래일(T 제외, 유효 200행 이상)로 회귀 g = a + b·us 를 추정한다.
    g  = 코스피200 시가 갭 ln(시가(T) / 종가(T−1))
    us = 한국 T 이전에 끝난 미국 S&P500 세션들의 종가 누적 로그수익(미국 종가는 06:00 KST 무렵 확정)
    s  = 회귀 잔차 표준편차
- 아침: 갭 잔차 z = (g − a − b·us) / s 를 계산한다(= 미국으로 설명되지 않은 갭이 평소의 몇 배인지).
- R3-C1: KODEX 200 자체 시가 갭으로 만든 z ≥ 1   → 09:00 시가 매수, 15:30 종가 매도
- R3-C2: 코스피200 지수 시가 갭으로 만든 z ≥ 1  → 같은 방식(2차 D2-C1 과 같은 규칙)
- R3-C3: R3-C2 조건 + KODEX 시가가 지수 대비 싸게 열림
         (ln(ETF 시가/지수 시가) ≤ 직전 5일 ln(ETF 종가/지수 종가) 중앙값) → 같은 방식
- 롱 전용, 하루 1회, 비용 왕복 5bp(팀 기준 수수료 3bp + 슬리피지 2bp), 스트레스 7bp
- 가정: 백테스트는 09:00 시가를 신호로 쓰고 같은 시가 단일가에 체결된다고 둔다. 실거래는 08:59 예상지수·예상체결가로
  신호를 내야 하며, 그 근사 오차는 과거 자료가 없어 반영하지 못했다.
기준선: 매일 시가 매수·종가 매도(같은 비용), KODEX 200 매수 보유(비용 없음)
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import data  # noqa: E402
import engine as E  # noqa: E402
import r2_d2 as R2  # noqa: E402
import r3_d2 as R3  # noqa: E402

TOPIC = "BT_D2_2010_2026"
OUT = data.OUT / "strategy_d2_2010_2026"
START = "2010-01-01"
COSTS = (5.0, 7.0)
ANN = 252
CELLS = {"R3-C3": "지수 갭 z≥1 + ETF 싸게 열림", "R3-C2": "지수 갭 z≥1", "R3-C1": "ETF 자체 갭 z≥1(실행형)"}
COLORS = {"R3-C3": "#2a78d6", "R3-C2": "#eb6834", "R3-C1": "#1baf7a"}     # 검증된 범주 순서 1~3


def signals():
    px, ix, tg, cal, listed = R3.load()
    ge = np.log(px["open"] / px["close"].shift(1))
    gi = np.log(ix["open"] / ix["close"].shift(1))
    us = R2.us_cum(cal, "spx")
    ze, zi = R3.rolling_z1(ge, us, 250, 200), R3.rolling_z1(gi, us, 250, 200)
    mis = np.log(px["open"] / ix["open"]) - np.log(px["close"] / ix["close"]).shift(1).rolling(5).median()
    sig = {"R3-C1": R3.B_(ze, 1.0), "R3-C2": R3.B_(zi, 1.0), "R3-C3": R3.both(R3.B_(zi, 1.0), R3.le0(mis))}
    return px, sig, pd.DataFrame({"g_index": gi, "g_etf": ge, "us": us, "z_index": zi, "z_etf": ze, "mis": mis})


def metrics(daily, pos):
    """daily: 일별 순수익(무포지션 0 포함), pos: 1/0 포지션."""
    d = daily.dropna()
    p = pos.reindex(d.index).fillna(0)
    tr = d[p != 0]
    eq = (1 + d).cumprod()
    yrs = len(d) / ANN
    cagr = eq.iloc[-1] ** (1 / yrs) - 1 if len(d) else np.nan
    mdd = (eq / eq.cummax() - 1).min() if len(d) else np.nan
    vol = d.std() * np.sqrt(ANN)
    return {"기간": f"{d.index.min().date()}~{d.index.max().date()}" if len(d) else "-", "거래일": len(d), "거래": int((p != 0).sum()),
            "노출": (p != 0).mean(), "승률": (tr > 0).mean() if len(tr) else np.nan, "거래당순bp": tr.mean() * 1e4 if len(tr) else np.nan,
            "누적수익": eq.iloc[-1] - 1 if len(d) else np.nan, "연환산": cagr, "연변동성": vol,
            "샤프": d.mean() / d.std() * np.sqrt(ANN) if d.std() > 0 else np.nan, "MDD": mdd, "칼마": cagr / -mdd if mdd < 0 else np.nan}


def fmt(df):
    out = df.copy()
    for c in ("노출", "승률", "누적수익", "연환산", "연변동성", "MDD"):
        if c in out:
            out[c] = out[c].map(lambda v: "-" if pd.isna(v) else f"{v * 100:.1f}%")
    for c in ("거래당순bp", "샤프", "칼마"):
        if c in out:
            out[c] = out[c].map(lambda v: "-" if pd.isna(v) else f"{v:+.2f}")
    return out.to_markdown(index=False)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    px, sig, feats = signals()
    r_oc = (px["close"] / px["open"] - 1)
    r_cc = px["close"].pct_change()
    valid = r_oc.notna() & sig["R3-C2"].notna() & sig["R3-C1"].notna() & sig["R3-C3"].notna()
    idx = r_oc.index[valid & (r_oc.index >= START)]
    end = idx.max()

    series, rows, yearly = {}, [], []
    for cost in COSTS:
        strat = {}
        for c in CELLS:
            pos = sig[c].reindex(idx)
            strat[c] = (pos * r_oc.reindex(idx) - pos * cost / 1e4, pos)
        strat["매일 시가→종가"] = (r_oc.reindex(idx) - cost / 1e4, pd.Series(1.0, index=idx))
        strat["KODEX 200 보유"] = (r_cc.reindex(idx), pd.Series(1.0, index=idx))
        if cost == COSTS[0]:
            series = strat
        windows = {"전체": (START, end), "2010~2019": (START, "2019-12-31"), "2020~2025-06-05": ("2020-01-01", "2025-06-05"),
                   "2025-06-09~": ("2025-06-09", end), "3년": (end - pd.DateOffset(years=3), end),
                   "1년": (end - pd.DateOffset(years=1), end), "3개월": (end - pd.DateOffset(months=3), end)}
        for w, (a, b) in windows.items():
            for name, (dr, pos) in strat.items():
                m = metrics(dr.loc[a:b], pos.loc[a:b])
                rows.append({"비용bp": cost, "구간": w, "전략": name, **m})
        if cost == COSTS[0]:
            for y in sorted(set(idx.year)):
                for name, (dr, pos) in strat.items():
                    m = metrics(dr[dr.index.year == y], pos[pos.index.year == y])
                    yearly.append({"연도": y, "전략": name, "거래": m["거래"], "연수익": m["누적수익"], "샤프": m["샤프"], "MDD": m["MDD"]})
    summ = pd.DataFrame(rows)
    yr = pd.DataFrame(yearly)

    # 일별 기록(팀 재현·대조용)
    daily = pd.DataFrame({"kodex_open": px["open"], "kodex_close": px["close"], "r_oc": r_oc}).reindex(idx)
    daily = daily.join(feats.reindex(idx))
    for c in CELLS:
        daily[f"sig_{c}"] = sig[c].reindex(idx).astype(int)
        daily[f"net5_{c}"] = series[c][0]
    daily.to_csv(OUT / "daily_returns.csv", encoding="utf-8-sig", float_format="%.8g")
    summ.to_csv(OUT / "summary.csv", index=False, encoding="utf-8-sig")
    yr.to_csv(OUT / "yearly.csv", index=False, encoding="utf-8-sig")

    chart(series, end)

    md = ["# D2 3차 전략 백테스트 2010-01 ~ " + str(end.date()) + "\n",
          "```\n" + __doc__.strip() + "\n```\n",
          f"표본: 신호 계산 가능 거래일 {len(idx)}일({idx.min().date()}~{end.date()}). 미국 S&P500 종가가 2026-09-21 세션까지(09-22 행은 종가 결측)라 마지막 신호일은 {end.date()}.\n",
          "## 누적 수익(비용 5bp, 로그 축. 매일 시가→종가 기준선은 ×0.05 라 표에만 표시)\n", "![equity](equity.png)\n",
          "## 구간별 성과 (비용 5bp)\n", fmt(summ[summ["비용bp"] == 5].drop(columns=["비용bp"])),
          "\n## 구간별 성과 (비용 7bp, 스트레스)\n", fmt(summ[(summ["비용bp"] == 7) & summ["구간"].isin(["전체", "2010~2019", "2020~2025-06-05", "2025-06-09~"])].drop(columns=["비용bp"])),
          "\n## 연도별 (비용 5bp)\n"]
    piv = yr.pivot(index="연도", columns="전략", values="연수익")[[*CELLS, "매일 시가→종가", "KODEX 200 보유"]]
    ntr = yr.pivot(index="연도", columns="전략", values="거래")[list(CELLS)]
    tab = piv.map(lambda v: f"{v * 100:+.1f}%")
    for c in CELLS:
        tab[c] = tab[c] + ntr[c].map(lambda n: f" ({n})")
    md += ["연수익(괄호 = 거래 수)\n", tab.to_markdown(), "\n",
           "## 읽는 법\n",
           "- 성과 보고용 백테스트다. 셀 판정은 3차 보고서(2003~2009 H0)와 2차 보고서(2020~2025-06 B)를 따른다. 규칙과 파라미터는 3차 사양 그대로다.",
           "- R3-C2·C3 의 수익은 '09:00 시가를 보고 같은 시가 단일가에 체결' 가정 위에 있다. 실거래는 08:59 예상지수(C2)와 ETF 예상체결가(C3)가 필요하다.",
           "- R3-C1 은 ETF 예상체결가만 있으면 되는 실행형이지만 2003~2009 판정에서 기각됐다.",
           "- 2023-07-31(선물 08:45 개장) 이후 ETF 시가 과소 반영 몫이 C2 에서 사라졌다. 최근 구간 성과를 해석할 때 참고한다.",
           "- 일별 신호·수익은 `daily_returns.csv`, 표는 `summary.csv`·`yearly.csv` 에 있다."]
    (OUT / "report.md").write_text("\n".join(md), encoding="utf-8")
    spec = {"rules": "round3_D2 spec R3-C1/C2/C3, r_oc, long-only", "start": START, "end": str(end.date()), "costs_bp": list(COSTS)}
    led = summ.rename(columns={"전략": "rule", "구간": "window", "비용bp": "cost_bp", "거래": "trades", "거래일": "days", "노출": "exposure",
                               "승률": "win", "거래당순bp": "net_bp", "연환산": "cagr", "샤프": "sharpe", "MDD": "mdd", "칼마": "calmar"})
    led["asset"] = "069500"
    if "--no-ledger" not in sys.argv:                       # 그림·문구만 다시 만들 때는 장부를 쓰지 않음
        E.ledger_append(data.OUT, TOPIC, spec, led, note="전략 백테스트 2010~2026(판정 아님)")
    print("\n".join(md[4:8]))


def chart(series, end):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False})
    fig, ax = plt.subplots(figsize=(10, 5.2), dpi=130)
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    order = [*CELLS, "KODEX 200 보유"]                       # 매일 시가→종가(×0.05)는 축을 왜곡해 표에만 둔다
    style = {"매일 시가→종가": dict(color="#8a8984", lw=1.5, ls="--"), "KODEX 200 보유": dict(color="#52514e", lw=1.5, ls=":")}
    for name in order:
        dr = series[name][0].dropna()
        eq = (1 + dr).cumprod()
        kw = style.get(name, dict(color=COLORS.get(name), lw=2))
        ax.plot(eq.index, eq.values, label=f"{name} {CELLS.get(name, '')}".strip(), **kw)
        ax.annotate(f"{name} ×{eq.iloc[-1]:.2f}", xy=(eq.index[-1], eq.iloc[-1]), xytext=(4, 0), textcoords="offset points",
                    fontsize=8, color="#0b0b0b", va="center")
    ax.set_yscale("log")
    ticks = [0.8, 1, 1.5, 2, 3, 5, 8]
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{t:g}" for t in ticks])
    ax.minorticks_off()
    ax.set_ylabel("누적 가치(시작 = 1, 로그 축)", color="#52514e")
    ax.set_title(f"D2 3차 셀 누적 수익, 비용 왕복 5bp (2010-01 ~ {end.date()})", color="#0b0b0b", fontsize=11, loc="left")
    ax.grid(True, color="#e6e5e1", lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c3c2b7")
    ax.tick_params(colors="#52514e", labelsize=8)
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    ax.set_xlim(right=ax.get_xlim()[1] + 420)
    fig.tight_layout()
    fig.savefig(OUT / "equity.png", facecolor=fig.get_facecolor())
    plt.close(fig)


if __name__ == "__main__":
    main()
