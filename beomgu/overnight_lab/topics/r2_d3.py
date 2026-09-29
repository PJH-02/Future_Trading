"""R2 · D3 국면 조건 확인 — 변동성 국면 게이트(KODEX 200 오버나잇 보유 여부), B 구간 확인 전용

사전 고정(실행 전 기록, 2026-09-29). 사양 원문: <OUT>/round2/D3/spec.json (탐색 담당 고정, 변경 금지)
- 자산·목표: 069500 KODEX 200 (data.price('069500'), 네이버 분배 수정주가, 2002-10~ 전체 이력)
  r_on(T) = O(T+1)/C(T) - 1. 진입 T 종가 단일가(체결 C(T)), 청산 T+1 시가 단일가(체결 O(T+1)). 0/1 롱, 보유 밤마다 왕복 1회.
- D3-USRV: S&P500 종가(spx_daily 'close', 결측 행 제거) 세션 수익률의 20세션 표본 SD(ddof=1) → 과거 250세션(D 제외) 백분위
  (동률 0.5, 과거 창에 결측 있으면 NaN) → 한국 T 에 D < T 인 마지막 미국 세션 값(T - D > 7일이면 NaN) → gate = pct >= 0.5
- D3-RVON: ron_hist(T) = O(T)/C(T-1) - 1 (= r_on(T-1), T 시가 확정) 의 20밤 표본 SD → 과거 250거래일(T 제외) 백분위 → gate = pct >= 0.5
- 게이트 NaN 밤은 무포지션이며 표본에서 뺀다(매일 보유 기준선도 같은 밤 집합).
- 이웃: W=120, W=500, 임계 0.4, 임계 0.6 (n=20) + 추가 n=10, n=40
- 구현은 spec.json 의 reference_impl_python 을 그대로 옮겼고, 실행 때 사양 코드를 exec 해 게이트가 완전히 같은지 단언한다.
  A 구간 통계(사양 A_stats)도 재현 점검한다.

확인 판정(셀별, B = 결정일 T 2020-01-01 ~ 2025-06-05):
  ① Δ = 평균 r_on(gate=1) - 평균 r_on(gate=0) > 0. NW(lag5) HAC t 의 단측 p(정규 근사)는 원 p 로 보고하고,
     2차 전체 셀(최대 8) Holm 보정은 종합에서 한다(원 p >= 0.05 면 어떤 순위에서도 Holm 통과 불가).
  ② 7bp 차감 후 보유 밤 순평균 > 0
  ③ 연도 층화 플라시보 p <= 0.05 (engine.placebo_year_strat, 2000회, seed 20260929)
  ④ 변동성 조정 Δ > 0 (common.vol_adj: r_on / 과거 60밤 r_on SD, T 제외)
  ⑤ 이웃 Δ 부호가 본 셀 Δ 와 모두 같음
  등급(Holm 전 잠정): ①~⑤ 모두 → '통과 후보(Holm 대기)', ①②③ 만 → '보류', 그 밖 → '기각'.
  가설(변동성 국면이 밤 프리미엄을 가른다) 확인은 두 셀 모두 통과일 때만으로 본다(사양 권고, 두 셀 독립 아님).

모호한 곳의 보수적 해석(기록):
  - ⑤ 이웃: 사양이 n=10·n=40 을 '추가 보고용'으로 두었지만 A 근거에서는 6개 모두를 이웃으로 불렀다.
    보수적으로 6개 전부(W120·W500·임계.4·임계.6·n10·n40) 부호 동일을 ⑤ 로 쓰고, 지정 4개만의 결과도 병기한다.
  - ② '신호 거래 순평균' = B 보유 밤 r_on 평균 - 7bp (비용 전 평균에서 비용 차감, 무보유 밤 제외).
  - B 마지막 밤 T = 2025-06-05 의 청산 시가는 2025-06-09 이다(06-06 휴장). 기존 규약(JUDGE_END)대로 B 에 넣는다.
  - 가격 캐시는 갱신하지 않는다(수정주가 재기록 방지, 탐색과 같은 캐시).
보고만: A(2010-01-01~2019-12-31, 표본 내) · E4(2025-06-09~), 기준선 대비 샤프·MDD·CAGR·거래당 순평균(5·7bp), 연도별,
  사양 권고 보조 점검(순샤프·MDD·5bp 순평균), 지수·타 자산 측정(캐시 있는 것만).
"""
import json
import sys
from math import erf, sqrt
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common as C  # noqa: E402
import data  # noqa: E402
import engine as E  # noqa: E402

TOPIC = "R2_D3"
OUT = data.OUT / "round2" / "D3"
SPEC_FILE = data.spec_file("D3")   # 출력 폴더에 없으면 저장소 결과 폴더의 사양을 읽음
WIN = {"A": ("2010-01-01", E.A_END), "B": ("2020-01-01", E.JUDGE_END), "E4": (E.E4_START, None)}
N_FAMILY_MAX = 8
CELLS = ("D3-USRV", "D3-RVON")
NEIGH = {"W120": dict(W=120), "W500": dict(W=500), "임계0.4": dict(thr=0.4), "임계0.6": dict(thr=0.6),
         "n10": dict(n=10), "n40": dict(n=40)}
NEIGH_DESIGNATED = ("W120", "W500", "임계0.4", "임계0.6")


# ---- 사양 참조 구현(spec.json reference_impl_python 그대로) ----
def pct_rank_past(x, W=250):
    """pct(t) = [#{x(t-k) < x(t)} + 0.5*#{x(t-k) == x(t)}] / W, k = 1..W (t 제외 과거 W 개가 모두 있어야 함, 아니면 NaN)."""
    v = x.to_numpy(float)
    out = np.full(len(v), np.nan)
    for i in range(W, len(v)):
        w = v[i - W:i]
        if np.isnan(v[i]) or np.isnan(w).any():
            continue
        out[i] = ((w < v[i]).sum() + 0.5 * (w == v[i]).sum()) / W
    return pd.Series(out, index=x.index)


def map_us_to_kr(s_us, kr_index, max_gap_days=7):
    """한국 거래일 T 에 미국 날짜 D < T 인 마지막 세션 값을 붙인다(T - D > 7일이면 NaN)."""
    s = s_us.dropna()
    pos = np.searchsorted(s.index.values, kr_index.values, side="left") - 1
    ok = pos >= 0
    gap = (kr_index.values - s.index.values[np.clip(pos, 0, None)]).astype("timedelta64[D]").astype(int)
    ok &= gap <= max_gap_days
    return pd.Series(np.where(ok, s.to_numpy()[np.clip(pos, 0, None)], np.nan), index=kr_index)


def gate_usrv(kodex, spx_close, n=20, W=250, thr=0.5):
    """D3-USRV: S&P500 20세션 실현변동성의 과거 250세션 백분위 >= 0.5 인 밤만 보유."""
    r = spx_close.dropna().pct_change()
    rv = r.rolling(n, min_periods=n).std()          # 표본 SD(ddof=1), 세션 D 포함 직전 n 개
    p = map_us_to_kr(pct_rank_past(rv, W), kodex.index)
    return (p >= thr).astype(float).where(p.notna())


def gate_rvon(kodex, n=20, W=250, thr=0.5):
    """D3-RVON: KODEX 200 오버나잇 수익률 20일 실현변동성의 과거 250거래일 백분위 >= 0.5 인 밤만 보유."""
    ron_hist = kodex["open"] / kodex["close"].shift(1) - 1   # T 행 = r_on(T-1), T 시가에 확정
    rv = ron_hist.rolling(n, min_periods=n).std()           # r_on(T-20..T-1)
    p = pct_rank_past(rv, W)
    return (p >= thr).astype(float).where(p.notna())


def r_on(kodex):
    return kodex["open"].shift(-1) / kodex["close"] - 1      # T 종가 매수 -> T+1 시가 매도


# ---- 측정 ----
def p_one(t):
    return np.nan if t is None or np.isnan(t) else 0.5 * (1 - erf(t / sqrt(2)))


def sample(g, ret, a, b):
    r = ret.loc[a:b]
    s = g.reindex(r.index)
    ok = r.notna() & s.notna()
    return s[ok], r[ok]


def block(g, ret, a, b, rv, placebo=True):
    """한 셀·한 구간의 모든 수치. 기준선 = 같은 밤 집합의 매일 보유."""
    s, r = sample(g, ret, a, b)
    d, t, n = E.delta_test(s, r)
    held, off = r[s == 1], r[s == 0]
    lo, hi = r.quantile([0.01, 0.99])
    dw, tw, _ = E.delta_test(s, r.clip(lo, hi))
    years = len(r) / E.ANN
    o = dict(start=str(r.index.min().date()), end=str(r.index.max().date()), nights=len(r), held=n, exposure=n / len(r),
             held_bp=held.mean() * 1e4, off_bp=off.mean() * 1e4, delta_bp=d * 1e4, t=t, p_one=p_one(t),
             net5_bp=(held.mean() - 5e-4) * 1e4, net7_bp=(held.mean() - 7e-4) * 1e4,
             dvol=E.delta_test(s, rv.reindex(r.index))[0], delta_wins_bp=dw * 1e4, t_wins=tw,
             median_diff_bp=(held.median() - off.median()) * 1e4, switches_per_year=float((s.diff().abs() > 0).sum() / years),
             base_mean_bp=r.mean() * 1e4, base_t=E.nw_t(r))
    for cost in E.COSTS_BP:
        c = int(cost)
        st, bs = E.stats(s, r, cost), E.stats(pd.Series(1.0, index=r.index), r, cost)
        o.update({f"sharpe_{c}": st["sharpe"], f"mdd_{c}": st["mdd"], f"cagr_{c}": st["cagr"], f"trade_net_{c}_bp": st["net_bp"],
                  f"base_sharpe_{c}": bs["sharpe"], f"base_mdd_{c}": bs["mdd"], f"base_cagr_{c}": bs["cagr"],
                  f"base_trade_net_{c}_bp": bs["net_bp"]})
    o["placebo_p"] = E.placebo_year_strat(s, r) if placebo else np.nan
    return o


def neighbor_deltas(neigh, ret, a, b):
    out = {}
    for k, g in neigh.items():
        s, r = sample(g, ret, a, b)
        out[k] = E.delta_test(s, r)[0] * 1e4
    return out


def yearly_rows(gates, ret):
    periods = [(str(y), f"{y}-01-01", f"{y}-12-31") for y in range(2020, 2025)]
    periods += [("2025(~06-05)", "2025-01-01", E.JUDGE_END), ("E4 2025-06-09~", E.E4_START, None)]
    rows = []
    for lab, a, b in periods:
        row = {"구간": lab}
        for cid, g in gates.items():
            s, r = sample(g, ret, a, b)
            row[f"{cid} 보유"] = int((s == 1).sum())
            row[f"{cid} Δ"] = E.delta_test(s, r)[0] * 1e4 if (s == 1).sum() >= 5 and (s == 0).sum() >= 5 else np.nan
            row[f"{cid} 순합5"] = float((s * r - s * 5e-4).sum() * 100)
            row["밤"] = len(r)
            row["매일 순합5"] = float((r - 5e-4).sum() * 100)
        rows.append(row)
    return rows


def year_decomp(g, ret, a, b):
    """사후 진단(보고만, 결과 확인 뒤 추가): Δ = 연도 간 구성(연도 평균 r_on 의 Δ) + 연도 안(연도 평균을 뺀 r_on 의 Δ), 정확히 가법."""
    s, r = sample(g, ret, a, b)
    ym = r.groupby(r.index.year).transform("mean")
    return (ym[s == 1].mean() - ym[s == 0].mean()) * 1e4, ((r - ym)[s == 1].mean() - (r - ym)[s == 0].mean()) * 1e4


def f(x, fmt="{:+.2f}"):
    return "-" if x is None or (isinstance(x, (float, np.floating)) and np.isnan(x)) else fmt.format(x)


def pc(x):
    return f(x * 100 if x is not None and not (isinstance(x, float) and np.isnan(x)) else np.nan, "{:+.1f}%")


def main(dry=False):
    OUT.mkdir(parents=True, exist_ok=True)
    spec = json.loads(SPEC_FILE.read_text(encoding="utf-8"))
    kodex = data.price("069500")
    spx = data.us_daily("spx")["close"]
    assert kodex.index.is_unique and spx.index.is_unique
    ret = r_on(kodex)
    assert np.allclose(ret.dropna(), E.legs(kodex)["on"].dropna())
    rv = C.vol_adj(ret)

    makers = {"D3-USRV": lambda **kw: gate_usrv(kodex, spx, **kw), "D3-RVON": lambda **kw: gate_rvon(kodex, **kw)}
    gates = {cid: makers[cid]() for cid in CELLS}
    neigh = {cid: {k: makers[cid](**kw) for k, kw in NEIGH.items()} for cid in CELLS}

    # 사양 참조 구현과 게이트 일치 단언
    ns = {}
    exec(spec["reference_impl_python"], ns)
    ref = {"D3-USRV": ns["gate_usrv"](kodex, spx), "D3-RVON": ns["gate_rvon"](kodex)}
    for cid in CELLS:
        assert ref[cid].equals(gates[cid]), f"{cid}: 사양 참조 구현과 게이트 불일치"

    res = {cid: {w: block(gates[cid], ret, a, b, rv) for w, (a, b) in WIN.items()} for cid in CELLS}
    nb = {cid: {w: neighbor_deltas(neigh[cid], ret, a, b) for w, (a, b) in WIN.items()} for cid in CELLS}

    # 판정(B)
    verdict = {"family": spec["family"], "judge_window_B": [WIN["B"][0], WIN["B"][1]], "n_family_max_for_holm": N_FAMILY_MAX,
               "spec_file": str(SPEC_FILE), "cells": {}}
    for cid in CELLS:
        b = res[cid]["B"]
        sgn = np.sign(b["delta_bp"])
        same6 = all(np.sign(v) == sgn for v in nb[cid]["B"].values())
        same4 = all(np.sign(nb[cid]["B"][k]) == sgn for k in NEIGH_DESIGNATED)
        crit = [b["delta_bp"] > 0, b["net7_bp"] > 0, (not np.isnan(b["placebo_p"])) and b["placebo_p"] <= 0.05, b["dvol"] > 0, same6]
        grade = "통과 후보(Holm 대기)" if all(crit) else ("보류" if crit[0] and crit[1] and crit[2] else "기각")
        aux = {"sharpe5_gt_base": b["sharpe_5"] > b["base_sharpe_5"], "mdd5_shallower": b["mdd_5"] > b["base_mdd_5"],
               "net5_gt_0": b["net5_bp"] > 0}
        verdict["cells"][cid] = dict(criteria="".join("O" if c else "X" for c in crit), grade_pre_holm=grade,
                                     holm_impossible=bool(b["p_one"] >= 0.05), neighbors_same_sign_6=same6,
                                     neighbors_same_sign_designated4=same4, spec_aux_checks=aux,
                                     B=b, A=res[cid]["A"], E4=res[cid]["E4"], neighbors=nb[cid])
    both = all(verdict["cells"][c]["grade_pre_holm"].startswith("통과") for c in CELLS)
    verdict["hypothesis"] = "두 셀 모두 통과 후보(Holm 대기)" if both else "가설 미확인(두 셀 모두 통과가 아님)"
    sB = {cid: sample(gates[cid], ret, *WIN["B"])[0] for cid in CELLS}
    agree = (sB["D3-USRV"] == sB["D3-RVON"].reindex(sB["D3-USRV"].index)).mean()
    verdict["gate_agreement_B"] = float(agree)

    # 보고용 측정 점검(캐시가 있는 것만, 판정 무관)
    meas = {}
    for sym in ("KPI200", "102110", "122630", "229200"):
        if (data.PX_DIR / f"{sym}.csv").exists():
            rt = E.legs(data.price(sym))["on"]
            meas[sym] = {cid: {w: E.delta_test(*sample(gates[cid], rt, a, b))[0:2] for w, (a, b) in WIN.items() if w != "A" or sym != "KPI200"}
                         for cid in CELLS}
    verdict["measurement_checks"] = {s: {c: {w: {"delta_bp": v[0] * 1e4, "t": v[1]} for w, v in d2.items()} for c, d2 in d1.items()}
                                     for s, d1 in meas.items()}

    # ---- 보고서 ----
    L = ["# R2 · D3 국면 조건 확인 — 변동성 국면 게이트 (KODEX 200 오버나잇)\n",
         f"실행 {pd.Timestamp.now(tz='Asia/Seoul'):%Y-%m-%d %H:%M} KST · 스크립트 `code/overnight_lab/topics/r2_d3.py` · 사양 `spec.json`(같은 폴더, 변경 없음)\n",
         "```\n" + __doc__.strip() + "\n```\n",
         "## 0. 구현 점검\n",
         "- 게이트 두 개가 spec.json 참조 구현(exec)과 전 기간에서 완전히 같다(단언 통과).",
         f"- 가격 캐시 069500: {kodex.index.min().date()} ~ {kodex.index.max().date()}, S&P500 종가(결측 제거): {spx.dropna().index.min().date()} ~ {spx.dropna().index.max().date()}.",
         f"- B 표본: 결정일 {res['D3-USRV']['B']['start']} ~ {res['D3-USRV']['B']['end']}, 게이트 NaN 밤 제외 뒤 USRV {res['D3-USRV']['B']['nights']}밤 · RVON {res['D3-RVON']['B']['nights']}밤. B 게이트 일치율 {agree:.1%}.",
         "\nA 재현(사양 A_stats 대비, 5bp):\n",
         "| 셀 | 항목 | 사양 | 재계산 |", "|---|---|---|---|"]
    for i, cid in enumerate(CELLS):
        As, a = spec["frozen_specs"][i]["A_stats"], res[cid]["A"]
        for lab, sv, rv_ in (("밤/보유", f"{As['nights']}/{As['held']}", f"{a['nights']}/{a['held']}"),
                             ("보유/비보유 bp", f"{As['held_bp']:+.2f}/{As['off_bp']:+.2f}", f"{a['held_bp']:+.2f}/{a['off_bp']:+.2f}"),
                             ("Δ bp (t)", f"{As['delta_bp']:+.2f} ({As['t_nw']:.2f})", f"{a['delta_bp']:+.2f} ({a['t']:.2f})"),
                             ("순샤프 5bp / 기준선", f"{As['sharpe_5']:+.3f} / {As['sharpe_base_5']:+.3f}", f"{a['sharpe_5']:+.3f} / {a['base_sharpe_5']:+.3f}"),
                             ("MDD 5bp / 기준선", f"{As['mdd_5']:.3f} / {As['mdd_base_5']:.3f}", f"{a['mdd_5']:.3f} / {a['base_mdd_5']:.3f}"),
                             ("순샤프 7bp / 기준선", f"{As['sharpe_7']:+.3f} / {As['sharpe_base_7']:+.3f}", f"{a['sharpe_7']:+.3f} / {a['base_sharpe_7']:+.3f}"),
                             ("게이트 전환/년", f"{As['gate_switches_per_year']:.1f}", f"{a['switches_per_year']:.1f}")):
            L.append(f"| {cid} | {lab} | {sv} | {rv_} |")

    L += ["\n## 1. 확인 판정 요약 (B 2020-01-01 ~ 2025-06-05)\n",
          "| 셀 | 밤 / 보유 | ① Δ bp (HAC t, 단측 원 p) | ② 7bp 순평균 | ③ 플라시보 p | ④ 변동성 조정 Δ | ⑤ 이웃 6개 부호 | 기준 | 잠정 등급 |",
          "|---|---|---|---|---|---|---|---|---|"]
    for cid in CELLS:
        v, b = verdict["cells"][cid], res[cid]["B"]
        L.append(f"| {cid} | {b['nights']} / {b['held']} | {b['delta_bp']:+.2f} ({b['t']:+.2f}, p {b['p_one']:.4f}) | {b['net7_bp']:+.2f}bp | "
                 f"{f(b['placebo_p'], '{:.4f}')} | {b['dvol']:+.4f} SD | {'동일' if v['neighbors_same_sign_6'] else '불일치'} "
                 f"(지정 4개 {'동일' if v['neighbors_same_sign_designated4'] else '불일치'}) | {v['criteria']} | **{v['grade_pre_holm']}** |")
    L += ["",
          f"- Holm: 원 p 는 2차 전체 셀(최대 {N_FAMILY_MAX}) 종합에서 보정한다. 가장 엄한 단계 문턱은 0.05/{N_FAMILY_MAX} = {0.05 / N_FAMILY_MAX:.5f}, "
          "가장 느슨한 마지막 단계도 0.05 다. " + "; ".join(
              f"{cid} 원 p {res[cid]['B']['p_one']:.4f} → {'어떤 순위에서도 Holm 통과 불가' if verdict['cells'][cid]['holm_impossible'] else '순위에 따라 가능'}"
              for cid in CELLS) + ".",
          f"- 가설 수준(사양 권고: 두 셀 모두 통과해야 확인): **{verdict['hypothesis']}**.",
          "\n사양 권고 보조 점검(B, 5bp; 판정 기준 아님):\n",
          "| 셀 | 순샤프 > 매일 보유 | MDD 더 얕음 | 보유 밤 순평균(5bp) > 0 |", "|---|---|---|---|"]
    for cid in CELLS:
        b, aux = res[cid]["B"], verdict["cells"][cid]["spec_aux_checks"]
        L.append(f"| {cid} | {'O' if aux['sharpe5_gt_base'] else 'X'} ({b['sharpe_5']:+.2f} vs {b['base_sharpe_5']:+.2f}) | "
                 f"{'O' if aux['mdd5_shallower'] else 'X'} ({pc(b['mdd_5'])} vs {pc(b['base_mdd_5'])}) | "
                 f"{'O' if aux['net5_gt_0'] else 'X'} ({b['net5_bp']:+.2f}bp) |")

    L += ["\n## 2. 구간별 셀 수치 (A 표본 내 · B 판정 · E4 보고)\n",
          "| 셀 | 구간 | 밤 | 보유(노출) | 보유 / 비보유 bp | Δ bp (t) | 단측 p | 윈저화 Δ (t) | 중앙값 차 | 변동성 조정 Δ | 5bp 순평균 | 7bp 순평균 | 플라시보 p | 전환/년 |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for cid in CELLS:
        for w in WIN:
            o = res[cid][w]
            L.append(f"| {cid} | {w} {o['start']}~{o['end']} | {o['nights']} | {o['held']} ({o['exposure']:.0%}) | {o['held_bp']:+.2f} / {o['off_bp']:+.2f} | "
                     f"{o['delta_bp']:+.2f} ({o['t']:+.2f}) | {o['p_one']:.4f} | {o['delta_wins_bp']:+.2f} ({o['t_wins']:+.2f}) | {o['median_diff_bp']:+.2f} | "
                     f"{o['dvol']:+.4f} | {o['net5_bp']:+.2f} | {o['net7_bp']:+.2f} | {f(o['placebo_p'], '{:.4f}')} | {o['switches_per_year']:.1f} |")

    L += ["\n## 3. 매일 보유 기준선 대비 (같은 밤 집합, 5bp · 7bp)\n",
          "순샤프·MDD·CAGR 은 engine.stats(표본 전체 밤, 무보유 밤 0 포함, √252). 거래당 순평균 = 보유 밤 평균 − 비용.\n",
          "| 구간 | 비용 | 규칙 | 거래 | 거래당 순평균 bp | 순샤프 | MDD | CAGR |", "|---|---|---|---|---|---|---|---|"]
    for w in WIN:
        for cost in (5, 7):
            b0 = res[CELLS[0]][w]
            L.append(f"| {w} | {cost}bp | 매일 보유 | {b0['nights']} | {b0[f'base_trade_net_{cost}_bp']:+.2f} | {b0[f'base_sharpe_{cost}']:+.2f} | "
                     f"{pc(b0[f'base_mdd_{cost}'])} | {pc(b0[f'base_cagr_{cost}'])} |")
            for cid in CELLS:
                o = res[cid][w]
                note = "" if o["nights"] == b0["nights"] else f" (밤 {o['nights']})"
                L.append(f"| {w} | {cost}bp | {cid}{note} | {o['held']} | {o[f'trade_net_{cost}_bp']:+.2f} | {o[f'sharpe_{cost}']:+.2f} | "
                         f"{pc(o[f'mdd_{cost}'])} | {pc(o[f'cagr_{cost}'])} |")

    L += ["\n## 4. 이웃 Δ bp (⑤ 는 B, 6개 전부 부호 동일을 요구 — 보수적 해석)\n",
          "| 셀 | 구간 | 본 셀 | " + " | ".join(NEIGH) + " |", "|---|---|---|" + "---|" * len(NEIGH)]
    for cid in CELLS:
        for w in WIN:
            L.append(f"| {cid} | {w} | {res[cid][w]['delta_bp']:+.2f} | " + " | ".join(f"{nb[cid][w][k]:+.2f}" for k in NEIGH) + " |")

    yr = yearly_rows(gates, ret)
    L += ["\n## 5. 연도별 (Δ bp, 순수익 합 % = Σ(포지션·r − 포지션·5bp))\n",
          "| 구간 | 밤 | 매일 보유 순합 | USRV 보유 | USRV Δ | USRV 순합 | RVON 보유 | RVON Δ | RVON 순합 |", "|---|---|---|---|---|---|---|---|---|"]
    for r_ in yr:
        L.append(f"| {r_['구간']} | {r_['밤']} | {r_['매일 순합5']:+.1f} | {r_['D3-USRV 보유']} | {f(r_['D3-USRV Δ'])} | {r_['D3-USRV 순합5']:+.1f} | "
                 f"{r_['D3-RVON 보유']} | {f(r_['D3-RVON Δ'])} | {r_['D3-RVON 순합5']:+.1f} |")
    dec = {cid: {w: year_decomp(gates[cid], ret, a, b) for w, (a, b) in WIN.items()} for cid in CELLS}
    verdict["year_decomposition_bp"] = {c: {w: {"between_year": v[0], "within_year": v[1]} for w, v in d1.items()} for c, d1 in dec.items()}
    L += ["\n사후 진단(보고만, 결과 확인 뒤 추가한 분해): Δ = 연도 간 구성 + 연도 안 선별 (정확히 가법, bp)\n",
          "| 셀 | A 연도 간 / 연도 안 | B 연도 간 / 연도 안 | E4 연도 간 / 연도 안 |", "|---|---|---|---|"]
    for cid in CELLS:
        L.append(f"| {cid} | " + " | ".join(f"{dec[cid][w][0]:+.2f} / {dec[cid][w][1]:+.2f}" for w in WIN) + " |")
    L.append("\n연도 층화 플라시보의 몬테카를로 표준오차(2000회): " + ", ".join(
        f"{cid} B p {res[cid]['B']['placebo_p']:.4f} ± {np.sqrt(res[cid]['B']['placebo_p'] * (1 - res[cid]['B']['placebo_p']) / 2000):.4f}"
        for cid in CELLS) + " (사전 고정 draws·seed 그대로, 재추첨하지 않음).")

    if meas:
        L += ["\n## 6. 보고용 측정 점검 (같은 게이트를 다른 r_on 에 적용, 판정 무관)\n",
              "KPI200 = 코스피200 지수(fchart 2014-07~, 거래 불가, ETF 시가 잡음 점검용). A 열은 KPI200 이 2014-07 부터라 생략.\n",
              "| 자산 | 셀 | A Δ (t) | B Δ (t) | E4 Δ (t) |", "|---|---|---|---|---|"]
        for sym, d1 in meas.items():
            for cid in CELLS:
                cells = [f"{d1[cid][w][0] * 1e4:+.2f} ({d1[cid][w][1]:+.2f})" if w in d1[cid] else "-" for w in WIN]
                L.append(f"| {sym} {data.NAMES.get(sym, '')} | {cid} | " + " | ".join(cells) + " |")

    # 기간별 표·장부 (본 셀 + 이웃 전부 기록)
    rules = {cid: gates[cid] for cid in CELLS}
    for cid in CELLS:
        rules.update({f"{cid} {k}": g for k, g in neigh[cid].items()})
    rj = ret.loc[WIN["A"][0]:]
    t5 = E.evaluate(rules, rj, 5.0)
    t7 = E.evaluate(rules, rj, 7.0)
    main_rows = t5[t5["rule"].isin(["매일 보유", *CELLS])]
    L += ["\n## 7. 기간별 (engine.evaluate, 5bp, 본 셀만; 이웃 포함 전체는 시행 장부)\n", E.fmt_table(main_rows)]
    L += ["\n## 8. 해석\n", INTERP]
    if dry:                                           # 점검 실행: 파일·장부 미기록
        print("\n".join(L[3:]))
        return
    (OUT / "report.md").write_text("\n".join(L), encoding="utf-8")

    t5["asset"], t7["asset"] = "069500", "069500"
    spec_compact = {"family": spec["family"], "spec_file": str(SPEC_FILE), "judge_B": list(WIN["B"]), "cost_bp": [5, 7],
                    "cells": {s_["cell_id"]: {"rule": s_["rule"], "params": s_["params"]} for s_ in spec["frozen_specs"]},
                    "neighbors": {k: v for k, v in NEIGH.items()}}
    E.ledger_append(data.OUT, TOPIC, spec_compact, pd.concat([t5, t7], ignore_index=True), note="2차 확인")
    C.save_verdict(OUT / "verdict.json", verdict)
    print("\n".join(L[3:]))


INTERP = """(점검 실행 `--dry` 1회로 파일을 쓰지 않고 결과를 본 뒤 이 절만 적었다. 규칙·판정 기준·seed 는 바꾸지 않았다. 수치는 결정적이다.)

**결론: D3 가설(변동성 국면 게이트)은 B 에서 확인되지 않았다.**
- 두 셀 모두 잠정 등급이 '기각'이다(USRV OOXXX, RVON OOXOX).
- 원 단측 p 가 0.28·0.23 이라, 종합의 Holm 보정에서는 어떤 순위로도 통과할 수 없다.
- 보수적 해석(⑤ 이웃 6개)을 지정 이웃 4개로 바꿔도 결론은 같다. RVON 은 ③ 플라시보에서 이미 떨어진다.

1. **방향은 A 와 같지만 크기가 약간 작고 잡음은 훨씬 크다.**
   - B Δ 는 USRV +2.6bp, RVON +3.2bp 다. A(+3.6·+3.7)의 72~87% 수준이다.
   - B 의 오버나잇 SD 는 86bp 로 A(58bp)보다 1.5배 크다. 그래서 HAC t 는 0.58·0.75 에 그친다.
   - 같은 수의 밤을 무작위로 고른 규칙과 비교한 백분위(rand_pct)는 71%·77%다. 선별 효과는 우연 범위 안이다.
2. **'비용 절감' 경로는 남았지만 작다.**
   - 게이트는 프리미엄이 비용보다 작은 밤을 건너뛴다.
   - 5bp 기준 순샤프는 0.23·0.26 으로 매일 보유 0.15 보다 높고, MDD 는 −22~24% 로 매일 보유 −27.5% 보다 얕다(사양 권고 보조 점검 3개 모두 O).
   - 다만 A 에서처럼 MDD 가 절반으로 줄지는 않았다.
   - 7bp 에서는 보유 밤 순평균이 +0.11·+0.16bp, 순샤프가 0.01·0.02 다. 비용이 조금만 커져도 절대 수익이 없다.
3. **연도 효과가 Δ 를 희석했고, 위기 급락 국면의 약점이 재현됐다.**
   - 게이트가 켜진 밤이 2020·2022 에 몰렸다(USRV 170·223밤). 2022 는 평균 r_on 이 −2.7bp 인 해였다.
   - 반대로 평균이 높았던 2021(+11.0)·2023(+7.5)에는 게이트가 대부분 꺼져 있었다.
   - 그래서 Δ 를 나누면 연도 간 구성이 −3.7·−3.2bp, 연도 안 선별이 +6.3·+6.4bp 다(사후 진단).
   - 연도 안 비교인 연도 층화 플라시보 p 는 0.051·0.054 로 경계선이다. 몬테카를로 오차(±0.005) 안이지만, 사전 고정 문턱을 넘었으므로 실패로 기록한다.
   - 연도별로 보면 2022~2025 는 Δ 가 모두 양수였고, 2020(코로나 급락·반등)은 음수였다(USRV −8.3, RVON −16.3).
   - 탐색 메모가 경고한 '위기 국면 방어 없음'(2008~09 Δ −13~−16bp)이 B 에서 다시 나타났다.
4. **강건성이 약해졌다.**
   - USRV 의 변동성 조정 Δ 는 −0.007 SD 로 사실상 0 이다. 보유 밤 SD 104bp 대 비보유 밤 64bp 다.
   - 이는 "프리미엄이 위험에 비례한다"는 가설 메커니즘과 모순되지는 않는다. 그러나 위험 단위로 더 나은 밤을 고른다는 근거는 없다는 뜻이고, ④ 는 사전 고정 기준이므로 실패다.
   - 이웃에서는 USRV W500(−0.55)·n40(−1.39), RVON n40(−1.10)이 부호가 뒤집혔다.
5. **E4(보고만): 게이트보다 무조건 보유가 나았다.**
   - 오버나잇 평균이 +32bp 로 국면이 바뀌었다. 게이트 Δ 도 +26·+16bp(t 1.2·1.1)로 컸다.
   - 그러나 게이트는 E4 밤의 42%·74% 만 보유했다. 그 결과 순샤프(1.60·1.62)와 CAGR 모두 매일 보유(1.87)에 못 미쳤다.
6. **측정 점검(보고만).**
   - 코스피200 지수·TIGER 200·레버리지·코스닥150 모두 B Δ 가 양수였다(+3.3~+9.0bp).
   - t 는 코스닥150 만 2 근처(1.9·2.0)다. 코스닥150 은 D3 사양 대상이 아니고 확인 구간을 본 뒤의 관찰이므로, 쓰려면 새 가설로 A 에서 설계부터 해야 한다.

한 줄 요약:
- 변동성 국면 게이트는 B 에서 부호는 유지했지만 통계적으로 확인되지 않았다(t 0.6~0.75, 플라시보 경계선 밖, 이웃 부호 흔들림).
- 2020 급락 국면에서 역전되는 약점이 재현됐다.
- 비용 절감에 따른 순샤프·MDD 개선은 있으나, 7bp 에서는 절대 수익이 0 에 가깝다. 운용 규칙으로 채택할 근거는 부족하다."""

if __name__ == "__main__":
    main(dry="--dry" in sys.argv)
