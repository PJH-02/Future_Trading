"""기간별(3개월·1년·3년·최대) 결과표 — 보고용. 판정 아님(사전등록 1단계는 미통과).

두 가지를 기간마다 낸다.
  A. 방향성: 목표 K200 지수 시가→종가, 지표 외인 OptFlow(T−1) — 5분위 Q5−Q1, β, HAC(lag 5) p, R²
  B. 매매 규칙 백테스트: KODEX 200 T일 시가 매수 → 종가 청산, 신호 없으면 현금(수익 0)
     R0 매일 매수(기준선) / R1 OptFlow(T−1) > 0 / R2 z20(OptFlow) ≥ 0.8416(설계 문서 2단계 규칙)
     R3 외인 PutFlow(T−1) < 0 (탐색에서 사후 발견 — 참고만)
     비용: 왕복 2×0.015%(가정) + 1틱(5원)/시가, 날짜별 계산
     지표: 매매일수, 승률(비용 후), 평균 순수익/매매(bp), 누적 순수익(복리 %), 연환산 샤프(현금일 0 포함, √252), MDD

사용: python backtest_periods.py [--raw-dir PATH] [--out-dir PATH] [--end 2026-09-23]
"""
import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from optflow import stats as st
from optflow.config import PREREG, resolve_out_dir, resolve_raw_dir
from optflow.indicators import build_indicators, lag_to_target
from optflow.io import build_panel

COMM_ONE_WAY = 0.00015
TICK_KRW = 5.0
Z_THR = 0.8416
WINDOWS = [("3개월", pd.DateOffset(months=3)), ("1년", pd.DateOffset(years=1)), ("3년", pd.DateOffset(years=3)), ("최대", None)]


def zscore_lagged(ind_col, win=20, burn=60):
    """신호일 s 의 z = (x_s − 평균(x_{s−win..s−1})) / 표준편차(같은 창). 그 뒤 T일 행으로 한 칸 민다."""
    mu = ind_col.rolling(win, min_periods=win).mean().shift(1)
    sd = ind_col.rolling(win, min_periods=win).std().shift(1)
    z = (ind_col - mu) / sd
    valid = z.dropna().index
    if len(valid) > burn:
        z.loc[: valid[burn - 1]] = np.nan
    return z.shift(1)


def trade_stats(gross, cost, signal):
    """gross, cost: 소수 수익률. signal: bool(T일 매수 여부). 현금일 수익 0."""
    d = pd.DataFrame({"g": gross, "c": cost, "s": signal}).dropna(subset=["g", "c"])
    d["s"] = d["s"].fillna(False).astype(bool)
    net = np.where(d["s"], d["g"] - d["c"], 0.0)
    net = pd.Series(net, index=d.index)
    tr = net[d["s"]]
    eq = (1 + net).cumprod()
    mdd = float((eq / eq.cummax() - 1).min()) if len(eq) else np.nan
    sd = net.std()
    return {"days": int(len(d)), "trades": int(d["s"].sum()),
            "win_pct": float(100 * (tr > 0).mean()) if len(tr) else np.nan,
            "avg_net_bp": float(1e4 * tr.mean()) if len(tr) else np.nan,
            "avg_gross_bp": float(1e4 * d.loc[d["s"], "g"].mean()) if len(tr) else np.nan,
            "cum_net_pct": float(100 * (eq.iloc[-1] - 1)) if len(eq) else np.nan,
            "sharpe": float(net.mean() / sd * np.sqrt(252)) if sd and sd > 0 else np.nan,
            "mdd_pct": 100 * mdd}


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
    raw = resolve_raw_dir(a.raw_dir)
    panel, _ = build_panel(raw, PREREG["window_start"])
    ind = build_indicators(panel)
    X = lag_to_target(ind, panel)
    z = zscore_lagged(ind["optflow_1d"])

    gross = panel["kdx_close"] / panel["kdx_open"] - 1
    cost = 2 * COMM_ONE_WAY + TICK_KRW / panel["kdx_open"]
    rules = {
        "R0 매일 매수": pd.Series(True, index=panel.index),
        "R1 OptFlow>0": X["optflow_1d"] > 0,
        "R2 z≥0.84": z >= Z_THR,
        "R3 외인풋순매도(탐색)": X["putflow_1d"] < 0,
    }
    has_sig = X["optflow_1d"].notna() & z.notna()      # 모든 규칙을 같은 날짜 집합에서 비교
    end = pd.Timestamp(a.end)
    base_start = pd.Timestamp(PREREG["window_start"])

    rows_dir, rows_bt = [], []
    for name, off in WINDOWS:
        start = base_start if off is None else end - off
        m = (panel.index > start if off is not None else panel.index >= start) & (panel.index <= end) & has_sig
        idx = panel.index[m]
        y, x = panel.loc[idx, "y_idx"], X.loc[idx, "optflow_1d"]
        r = st.hac_ols(y, x.to_frame("x"), lags=PREREG["hac_lags"], min_n=20)
        q = st.quintile_table(y, x, min_per_bin=5)
        rows_dir.append({"기간": name, "시작": str(idx.min().date()), "끝": str(idx.max().date()), "거래일": len(idx),
                         "Q5−Q1(%p)": q["q5_q1"] if q else np.nan, "β": r.get("coef", {}).get("x", np.nan),
                         "HAC p": r.get("p", {}).get("x", np.nan), "R²": r.get("r2", np.nan)})
        for rn, sig in rules.items():
            s = trade_stats(gross[idx], cost[idx], sig[idx])
            rows_bt.append({"기간": name, "규칙": rn, **s})

    D, T = pd.DataFrame(rows_dir), pd.DataFrame(rows_bt)
    run_id = dt.datetime.now().strftime("%Y%m%d-%H%M%S") + "-periods"
    out = resolve_out_dir(a.out_dir) / run_id
    out.mkdir(parents=True, exist_ok=False)
    D.to_csv(out / "periods_direction.csv", index=False, encoding="utf-8-sig")
    T.to_csv(out / "periods_backtest.csv", index=False, encoding="utf-8-sig")

    def fmt(v, d=2, sign=True):
        return "—" if v is None or (isinstance(v, float) and np.isnan(v)) else (f"{v:+.{d}f}" if sign else f"{v:.{d}f}")

    L = [f"# 기간별 결과 — 외인 옵션 매수세(OptFlow) (종료일 {a.end}, 보고용·판정 아님)", "",
         "## A. 방향성 (K200 지수 시가→종가, 비용 전)", "| 기간 | 표본 | 거래일 | Q5−Q1 | β | HAC p | R² |", "|---|---|---|---|---|---|---|"]
    for _, r in D.iterrows():
        L.append(f"| {r['기간']} | {r['시작']} ~ {r['끝']} | {r['거래일']} | {fmt(r['Q5−Q1(%p)'], 3)}%p | {fmt(r['β'])} | {fmt(r['HAC p'], 3, False)} | {fmt(r['R²'], 4, False)} |")
    L += ["", "## B. 매매 규칙 백테스트 (KODEX 200 시가 매수→종가 청산, 비용 반영: 왕복 0.03% + 1틱)",
          "| 기간 | 규칙 | 매매일 | 승률 | 평균 순수익/회 | 누적 순수익 | 샤프(연) | MDD |", "|---|---|---|---|---|---|---|---|"]
    for _, r in T.iterrows():
        L.append(f"| {r['기간']} | {r['규칙']} | {r['trades']}/{r['days']} | {fmt(r['win_pct'], 1, False)}% | {fmt(r['avg_net_bp'], 1)}bp | "
                 f"{fmt(r['cum_net_pct'], 1)}% | {fmt(r['sharpe'])} | {fmt(r['mdd_pct'], 1)}% |")
    L += ["", "- 샤프는 현금으로 쉰 날 수익 0을 포함한 일별 순수익으로 계산(√252 연환산). 무위험 수익률은 빼지 않음",
          "- R3 은 전체 기간 탐색에서 찾은 뒤 붙인 규칙이라 표본 내 성과가 부풀려져 있다(참고만)",
          "- 3개월(약 60거래일)은 표본이 작아 방향성·샤프 모두 우연의 폭이 크다"]
    (out / "periods_report.md").write_text("\n".join(L), encoding="utf-8")
    (out / "meta.json").write_text(json.dumps({"end": a.end, "comm_one_way": COMM_ONE_WAY, "tick_krw": TICK_KRW, "z_thr": Z_THR,
                                               "raw_dir": str(raw)}, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n".join(L))
    print(f"\n→ {out}")


if __name__ == "__main__":
    main()
