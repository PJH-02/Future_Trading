"""주제 공통 판정 블록: 신호 1개(롱 전용, 1박)를 공통 규약으로 판정하고 마크다운을 만든다."""
from math import erf, sqrt

import numpy as np
import pandas as pd

import data
import engine as E


def zscore_past(x, win):
    """T 제외 과거 win 일 평균·SD 로 표준화."""
    m = x.shift(1).rolling(win, min_periods=win).mean()
    s = x.shift(1).rolling(win, min_periods=win).std()
    return (x - m) / s


def vol_adj(ret, win=60):
    sd = ret.shift(1).rolling(win, min_periods=win).std()
    return ret / sd


def hac_reg(y, X, lags=5):
    """OLS + Newey-West. X: DataFrame(상수 자동 추가). 계수·t 반환."""
    d = pd.concat([y.rename("y"), X], axis=1).dropna()
    Xm = np.column_stack([np.ones(len(d)), d[X.columns].to_numpy()])
    yv = d["y"].to_numpy()
    b, *_ = np.linalg.lstsq(Xm, yv, rcond=None)
    u = yv - Xm @ b
    n, k = Xm.shape
    inv = np.linalg.inv(Xm.T @ Xm)
    S = (Xm * u[:, None]).T @ (Xm * u[:, None])
    for L in range(1, lags + 1):
        G = (Xm[L:] * u[L:, None]).T @ (Xm[:-L] * u[:-L, None])
        S += (1 - L / (lags + 1)) * (G + G.T)
    V = inv @ S @ inv * n / (n - k)
    se = np.sqrt(np.diag(V))
    return pd.DataFrame({"coef": b, "t": b / se}, index=["const", *X.columns]), n


def _f(x, fmt="{:+.2f}"):
    return "-" if x is None or (isinstance(x, float) and np.isnan(x)) else fmt.format(x)


def save_verdict(path, obj):
    """verdict 를 JSON 으로 저장(numpy 수치는 float, NaN 은 null)."""
    import json

    def clean(o):
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        if isinstance(o, (np.floating, float)):
            return None if np.isnan(o) else float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.bool_,)):
            return bool(o)
        return o
    path.write_text(json.dumps(clean(obj), ensure_ascii=False, indent=1), encoding="utf-8")


def judge(name, sig, ret, neighbors=None, transfer=None, controls=None, veto=False, judge_start=None):
    """sig: 0/1(NaN = 신호 계산 불가) 결정일 T 인덱스, ret: r_on.
    veto=True 면 '신호 밤 = 쉬는 밤'이고 Δveto = 평균(비신호) − 평균(신호) 로 판정한다.
    반환: (verdict dict, 마크다운 줄 목록)"""
    J = ret.loc[judge_start:E.JUDGE_END].dropna()
    s = sig.reindex(J.index)
    ok_idx = s.notna()
    J, s = J[ok_idx], s[ok_idx]
    sign = -1 if veto else 1
    d, t, n = E.delta_test(s, J)
    d, t = sign * d, sign * t
    p1 = 0.5 * (1 - erf(t / sqrt(2)))
    held = J[s == 0] if veto else J[s == 1]
    net7 = (held - 7e-4).mean() * 1e4
    sig_mean = J[s == 1].mean() * 1e4
    dA = sign * E.delta_test(s.loc[:E.A_END], J.loc[:E.A_END])[0]
    dB = sign * E.delta_test(s.loc["2020-01-01":], J.loc["2020-01-01":])[0]
    dv = sign * E.delta_test(s, vol_adj(ret).reindex(J.index))[0]
    plc = E.placebo_year_strat(s, J) if not veto else E.placebo_year_strat(1 - s, J)
    neigh = {k: float(sign * E.delta_test(v.reindex(J.index), J)[0] * 1e4) for k, v in (neighbors or {}).items()}
    tr = None
    if transfer is not None:
        st, rt = transfer
        Jt = rt.loc[:E.JUDGE_END].dropna()
        tr = sign * E.delta_test(st.reindex(Jt.index), Jt)[0] * 1e4
    ctrl = None
    if controls is not None:
        X = pd.concat([s.rename("signal"), controls.reindex(J.index)], axis=1)
        ctrl, _ = hac_reg(J, X)
    v = dict(n_sig=n, n_days=len(J), delta_bp=d * 1e4, t=t, p_one=p1, p_bonf5=min(1.0, p1 * 5), net7_bp=net7, sig_mean_bp=sig_mean,
             dA_bp=dA * 1e4, dB_bp=dB * 1e4, dvol=dv, placebo_p=plc, neighbors=neigh, transfer_delta_bp=tr)
    crit = [v["delta_bp"] > 0 and v["p_bonf5"] < 0.05,
            (sig_mean < 5) if veto else (net7 > 0),
            v["dA_bp"] > 0 and v["dB_bp"] > 0 and v["dB_bp"] >= 0.5 * v["dA_bp"],
            (plc is not None and not np.isnan(plc) and plc <= 0.05),
            dv > 0,
            all(np.sign(x) == np.sign(v["delta_bp"]) for x in neigh.values()) if neigh else True,
            (tr is None) or tr > 0]
    if ctrl is not None:
        crit.append(ctrl.loc["signal", "coef"] * sign > 0 and abs(ctrl.loc["signal", "t"]) > 1.96)
    grade = "통과" if all(crit) else ("보류" if crit[0] and crit[1] and crit[3] else "기각")
    v.update(criteria="".join("O" if c else "X" for c in crit), grade=grade)

    lab = "Δveto(비신호−신호)" if veto else "Δ(신호−비신호)"
    md = [f"### {name}: **{grade}** (기준 충족 {v['criteria']})\n",
          f"| 항목 | 값 | 기준 |\n|---|---|---|",
          f"| 판정 구간 밤 / 신호 밤 | {len(J)} / {n} | |",
          f"| {lab} | {v['delta_bp']:+.2f}bp (HAC t {t:+.2f}, 단측 p {p1:.4f}, ×5 {v['p_bonf5']:.4f}) | > 0, ×5 p < 0.05 |",
          f"| 신호 밤 평균(비용 전) | {sig_mean:+.2f}bp | {'< 5bp' if veto else ''} |",
          f"| 보유 밤 순평균(7bp) | {net7:+.2f}bp | {'' if veto else '> 0'} |",
          f"| {lab} A(~2019) / B(2020~2025-06) | {_f(v['dA_bp'])} / {_f(v['dB_bp'])} | 둘 다 > 0, B ≥ 0.5·A |",
          f"| 변동성 조정 {lab} | {dv:+.4f} SD | > 0 |",
          f"| 연도 층화 플라시보 p | {plc:.4f} | ≤ 0.05 |",
          f"| 이웃 {lab} | {', '.join(f'{a} {b:+.1f}' for a, b in neigh.items()) or '-'} | 부호 동일 |",
          f"| 이전 자산 {lab} | {'-' if tr is None else f'{tr:+.2f}bp'} | > 0 |"]
    if ctrl is not None:
        md.append(f"| 통제 회귀 신호 계수 | {ctrl.loc['signal', 'coef'] * 1e4:+.2f}bp (t {ctrl.loc['signal', 't']:+.2f}) | 부호 유지, |t| > 1.96 |")
        md.append("\n통제 회귀 전체(계수 × 1e4 = 설명변수 1단위당 bp): " + ", ".join(f"{i} {r.coef * 1e4:+.2f}(t {r.t:+.2f})" for i, r in ctrl.iterrows()))
    return v, md


def period_block(rules, ret, cost=5.0):
    t = E.evaluate(rules, ret, cost_bp=cost)
    return t, E.fmt_table(t)
