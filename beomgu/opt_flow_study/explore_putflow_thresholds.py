"""탐색: 외인 풋 순매도 '강도'별 매매 결과 (판정 아님, 사후 탐색).

PutFlow = (외인 풋 매수대금 − 매도대금) / (매수 + 매도), T−1 값. 음수 = 외인 풋 순매도.
조건: 절대 임계(<0, <−0.01, <−0.02, <−0.03, <−0.05)와 과거 20일 z 임계(z ≤ −0.84, ≤ −1.28, 룩어헤드 없음).
매매: KODEX 200 T일 시가 매수 → 종가, 비용 왕복 0.03% + 1틱. 기간: 3개월·1년·3년·최대(종료 2026-09-23).
출력: 분포 백분위, 조건별 매매일·비중·승률·평균 순수익·t값(매매당 평균)·누적·샤프·샤프 표준오차.
"""
import argparse
import datetime as dt
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from backtest_periods import COMM_ONE_WAY, TICK_KRW, WINDOWS, zscore_lagged
from optflow.config import PREREG, resolve_out_dir, resolve_raw_dir
from optflow.indicators import build_indicators, lag_to_target
from optflow.io import build_panel


def stats(g, c, s):
    d = pd.DataFrame({"g": g, "c": c, "s": s}).dropna(subset=["g", "c"])
    d["s"] = d["s"].fillna(False).astype(bool)
    net = pd.Series(np.where(d["s"], d["g"] - d["c"], 0.0), index=d.index)
    tr = net[d["s"]]
    n, k = len(net), len(tr)
    sd_d = net.std()
    sr_d = net.mean() / sd_d if sd_d > 0 else np.nan
    return {"days": n, "trades": k, "share_pct": 100 * k / n if n else np.nan,
            "win_pct": 100 * (tr > 0).mean() if k else np.nan,
            "avg_net_bp": 1e4 * tr.mean() if k else np.nan,
            "t_trade": tr.mean() / (tr.std() / np.sqrt(k)) if k > 2 and tr.std() > 0 else np.nan,
            "cum_net_pct": 100 * ((1 + net).prod() - 1),
            "sharpe": sr_d * np.sqrt(252) if sr_d == sr_d else np.nan,
            "sharpe_se": np.sqrt((1 + 0.5 * sr_d ** 2) / n) * np.sqrt(252) if n and sr_d == sr_d else np.nan}


def main():
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--end", default=PREREG["window_end"])
    a = ap.parse_args()
    panel, _ = build_panel(resolve_raw_dir(a.raw_dir), PREREG["window_start"])
    ind = build_indicators(panel)
    X = lag_to_target(ind, panel)
    pf = X["putflow_1d"]
    zp = zscore_lagged(ind["putflow_1d"])
    g = panel["kdx_close"] / panel["kdx_open"] - 1
    c = 2 * COMM_ONE_WAY + TICK_KRW / panel["kdx_open"]
    conds = {"PutFlow<0 (표의 R3)": pf < 0, "PutFlow<−0.01": pf < -0.01, "PutFlow<−0.02": pf < -0.02,
             "PutFlow<−0.03": pf < -0.03, "PutFlow<−0.05": pf < -0.05,
             "z≤−0.84 (과거20일 하위≈20%)": zp <= -0.8416, "z≤−1.28 (하위≈10%)": zp <= -1.2816,
             "매일 매수 (기준선)": pd.Series(True, index=panel.index)}
    end = pd.Timestamp(a.end)
    base = pd.Timestamp(PREREG["window_start"])
    valid = pf.notna() & zp.notna()

    L = [f"# 탐색: 외인 풋 순매도 강도별 결과 (종료 {a.end}, 사후 탐색·판정 아님)", ""]
    q = [1, 5, 10, 20, 25, 50, 75, 80, 90, 95, 99]
    full = pf[(pf.index >= base) & (pf.index <= end)].dropna()
    y1 = pf[(pf.index > end - pd.DateOffset(years=1)) & (pf.index <= end)].dropna()
    L += ["## 외인 PutFlow(T−1) 분포", "| 백분위 | " + " | ".join(f"{p}%" for p in q) + " |", "|---|" + "---|" * len(q),
          "| 최대 기간 | " + " | ".join(f"{full.quantile(p / 100):+.3f}" for p in q) + " |",
          "| 최근 1년 | " + " | ".join(f"{y1.quantile(p / 100):+.3f}" for p in q) + " |", ""]
    rows = []
    for wname, off in WINDOWS:
        m = ((panel.index > end - off) if off is not None else (panel.index >= base)) & (panel.index <= end) & valid
        idx = panel.index[m]
        for cn, s in conds.items():
            rows.append({"기간": wname, "조건": cn, **stats(g[idx], c[idx], s[idx])})
    R = pd.DataFrame(rows)
    for wname, _ in WINDOWS:
        L += [f"## {wname}", "| 조건 | 매매일 (비중) | 승률 | 평균 순수익/회 | t(매매당) | 누적 순수익 | 샤프 ± 표준오차 |", "|---|---|---|---|---|---|---|"]
        for _, r in R[R["기간"] == wname].iterrows():
            L.append(f"| {r['조건']} | {r['trades']}/{r['days']} ({r['share_pct']:.0f}%) | {r['win_pct']:.1f}% | {r['avg_net_bp']:+.1f}bp | "
                     f"{r['t_trade']:+.2f} | {r['cum_net_pct']:+.1f}% | {r['sharpe']:+.2f} ± {r['sharpe_se']:.2f} |")
        L.append("")
    L += ["- 임계값(−0.01~−0.05)은 분포를 보고 고른 사후 조건이다. 여러 조건을 봤으므로 가장 좋은 칸은 우연으로 부풀려져 있다",
          "- 샤프 표준오차는 √((1+SR²/2)/n)×√252 근사. 1년(약 243일) 샤프의 표준오차는 약 1이다",
          "- 누적 순수익은 해당 기간 한 번의 결과이며 연환산이 아니다(1년 구간은 사실상 1년 수익)"]
    out = resolve_out_dir(a.out_dir) / (dt.datetime.now().strftime("%Y%m%d-%H%M%S") + "-putflow-explore")
    out.mkdir(parents=True, exist_ok=False)
    R.to_csv(out / "putflow_thresholds.csv", index=False, encoding="utf-8-sig")
    (out / "putflow_thresholds.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))
    print(f"\n→ {out}")


if __name__ == "__main__":
    main()
