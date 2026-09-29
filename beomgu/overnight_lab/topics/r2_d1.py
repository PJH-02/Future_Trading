"""R2 · D1 유동성 압력 반전 확인(2차): 종가 위치 매도압력(CLV) × 시장 거래대금 급증 → 그날 밤 반등

고정 사양: <OUT>/round2/D1/spec.json (탐색 담당 고정, 이 스크립트는 읽기만 한다. 규칙·임계·창 변경 없음)
셀(같은 정의, 입력 시장만 다름):
- D1-K200-CLVTV : 069500 KODEX 200,     TV = 코스피 투자자별 12분류 매수금액 합(naver_kospi_investor_daily)
- D1-KQ150-CLVTV: 229200 KODEX 코스닥150, TV = 코스닥 투자자별 12분류 매수금액 합(naver_kosdaq_investor_daily)
신호(T) = 1 ⇔ CLV(T) ≤ 0.20 AND dTV20(T) > 0, 둘 중 하나라도 NaN 이면 NaN(표본 제외)
  CLV = (C−L)/(H−L) (해당 ETF 일봉, H=L 이면 NaN), dTV20 = ln TV(T) − 직전 20거래일(ETF 달력, T 제외) ln TV 평균(20개 필요)
목표: r_on(T) = O(T+1)/C(T) − 1. T 시간외 종가 매수(체결가 = C(T)) → T+1 시가 매도. 롱 전용, 신호 밤마다 왕복 비용 5bp(스트레스 7bp)
구간: A = 결정일 2010-01-01~2019-12-31(표본 내, 사양 재현 확인), B = 2020-01-01~2025-06-05(확인, 1회), E4 = 2025-06-09~(보고만)
확인 판정(셀별, B):
 ① Δ(신호−비신호) > 0, Newey-West(lag 5) t 의 단측 p(정규 근사) 원값 보고 — Holm(m ≤ 8) 보정은 종합에서
 ② 7bp 차감 후 신호 밤 순평균 > 0
 ③ 연도 층화 플라시보 p ≤ 0.05 (engine.placebo_year_strat, 2000회, seed 20260929)
 ④ 변동성 조정 Δ > 0 (r_on / 직전 60밤 r_on SD, T 제외 = common.vol_adj)
 ⑤ 이웃 N1(CLV≤0.15)·N2(CLV≤0.25)·N3(창 60)·N4(창 10)의 B Δ 가 모두 > 0 (본 셀과 같은 양의 부호)
보고용(판정 불사용): 상호 이전(두 셀), 레버리지 배율(122630·233740), 코스피200 지수 오버나잇(069500 신호), 통제 회귀(r_cc·r_id·dTV20),
  상위 5밤 제외·중앙값·1% 윈저화·옵션 만기일 제외 Δ, 연도별 Δ, 기준선(매일 보유) 대비 샤프·MDD·거래당 순평균(5·7bp)
실행: python topics/r2_d1.py            (산출: <OUT>/round2/D1/report.md·verdict.json, 시행 장부 <OUT>/ledger.csv 'R2_D1')
      python topics/r2_d1.py --a-only   (A 구간 사양 재현만 출력, 파일·장부 쓰지 않음)
"""
import argparse
import datetime as dt
import json
import sys
from math import erf, sqrt
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common as C  # noqa: E402
import data  # noqa: E402
import engine as E  # noqa: E402

TOPIC = "R2_D1"
OUT = data.OUT / "round2" / "D1"
SPEC_FILE = data.spec_file("D1")   # 출력 폴더에 없으면 저장소 결과 폴더의 사양을 읽음
PERIODS = {"A": ("2010-01-01", E.A_END), "B": ("2020-01-01", E.JUDGE_END), "E4": (E.E4_START, None)}
CATS = ["other_foreign", "foreign", "individual", "other_corp", "gov", "pension", "other_fin", "bank", "private_fund", "trust",
        "insurance", "fin_inv"]
MAIN = dict(clv_max=0.20, tv_win=20, tv_thr=0.0)
NEIGH = {"N1": dict(clv_max=0.15, tv_win=20, tv_thr=0.0), "N2": dict(clv_max=0.25, tv_win=20, tv_thr=0.0),
         "N3": dict(clv_max=0.20, tv_win=60, tv_thr=0.0), "N4": dict(clv_max=0.20, tv_win=10, tv_thr=0.0)}
NEIGH_LABEL = {"N1": "N1 CLV≤0.15", "N2": "N2 CLV≤0.25", "N3": "N3 창60", "N4": "N4 창10"}
CELLS = [
    dict(cell_id="D1-K200-CLVTV", asset="069500", market="KOSPI", lev="122630", index="KPI200",
         A_ref=dict(n_sig=248, delta_bp=8.09, t=1.65, placebo_p=0.053)),
    dict(cell_id="D1-KQ150-CLVTV", asset="229200", market="KOSDAQ", lev="233740", index=None,
         A_ref=dict(n_sig=126, delta_bp=9.18, t=1.35, placebo_p=0.091)),
]
A_CTRL = {"D1-K200-CLVTV": "+13.85bp, t +2.73", "D1-KQ150-CLVTV": "+14.97bp, t +1.55"}   # spec.json A_stats(보고 비교용)
A_LEV = {"D1-K200-CLVTV": "1.68배", "D1-KQ150-CLVTV": "2.04배"}
HOLM_M = 8
Z_HOLM_MIN = NormalDist().inv_cdf(1 - 0.05 / HOLM_M)     # 가장 엄격한 Holm 순위(0.05/8) 단측 z
Z_05 = NormalDist().inv_cdf(0.95)


# ---------- 사양 대조(읽기 전용) ----------
def check_spec():
    sp = json.loads(SPEC_FILE.read_text(encoding="utf-8"))
    for cell in CELLS:
        sc = next(c for c in sp["cells"] if c["cell_id"] == cell["cell_id"])
        assert sc["asset"] == cell["asset"], sc["asset"]
        p = sc["params"]
        assert (p["clv_max"], p["tv_window"], p["tv_threshold"]) == (MAIN["clv_max"], MAIN["tv_win"], MAIN["tv_thr"]), p
        for n in sc["neighbors_list"]:
            q = NEIGH[n["id"]]
            assert (n["clv_max"], n["tv_window"], n["tv_threshold"]) == (q["clv_max"], q["tv_win"], q["tv_thr"]), n
    return sp


# ---------- 신호(사양 reference_impl_python 그대로) ----------
def market_tv(market):
    inv = data.naver_investor("kospi") if market == "KOSPI" else data.naver_kosdaq_investor()
    return inv[[f"buyv_{c}" for c in CATS]].sum(axis=1, min_count=len(CATS)), inv


def d1_parts(px, tv, tv_win=20):
    rng = px["high"] - px["low"]
    clv = ((px["close"] - px["low"]) / rng).where(rng > 0)
    ltv = np.log(tv.reindex(px.index))
    dtv = ltv - ltv.shift(1).rolling(tv_win, min_periods=tv_win).mean()
    return clv, dtv


def d1_signal(px, tv, clv_max=0.20, tv_win=20, tv_thr=0.0):
    clv, dtv = d1_parts(px, tv, tv_win)
    ok = clv.notna() & dtv.notna()
    return ((clv <= clv_max) & (dtv > tv_thr)).astype(float).where(ok)


# ---------- 통계 블록 ----------
def p_one(t):
    return np.nan if t is None or np.isnan(t) else 0.5 * (1 - erf(t / sqrt(2)))


def sample(sig, ret, period):
    a, b = PERIODS[period]
    return pd.DataFrame({"s": sig, "r": ret}).loc[a:b].dropna()


def block(sig, ret, period, vol=None, placebo=False):
    d = sample(sig, ret, period)
    s, r = d["s"], d["r"]
    dl, t, n = E.delta_test(s, r)
    held, non = r[s == 1], r[s == 0]
    out = dict(start=str(d.index.min().date()), end=str(d.index.max().date()), n_days=len(d), n_sig=int(n),
               rate=n / len(d) if len(d) else np.nan,
               sig_mean_bp=held.mean() * 1e4, non_mean_bp=non.mean() * 1e4, all_mean_bp=r.mean() * 1e4,
               delta_bp=dl * 1e4, t=t, p_one=p_one(t),
               net5_bp=(held.mean() - 5e-4) * 1e4, net7_bp=(held.mean() - 7e-4) * 1e4, t_net7=E.nw_t(held - 7e-4),
               sig_median_bp=held.median() * 1e4, non_median_bp=non.median() * 1e4,
               win_sig=(held > 0).mean(), win_non=(non > 0).mean())
    out["median_diff_bp"] = out["sig_median_bp"] - out["non_median_bp"]
    if vol is not None:
        out["dvol"] = E.delta_test(s, vol.reindex(d.index))[0]
    if placebo:
        out["placebo_p"] = E.placebo_year_strat(s, r)
    return out, d


def yearly_delta(d):
    rows = []
    for y, g in d.groupby(d.index.year):
        hs, hn = g["r"][g["s"] == 1], g["r"][g["s"] == 0]
        rows.append(dict(year=int(y), n=len(g), n_sig=len(hs), sig_mean_bp=hs.mean() * 1e4, non_mean_bp=hn.mean() * 1e4,
                         delta_bp=(hs.mean() - hn.mean()) * 1e4 if len(hs) and len(hn) else np.nan))
    return rows


def expiry_days(idx):
    """월물 옵션 만기일 = 매월 둘째 목요일, 휴장이면 그 전 거래일(ETF 거래일 달력 기준)."""
    out = set()
    for (y, m) in sorted({(d.year, d.month) for d in idx}):
        first = pd.Timestamp(y, m, 1)
        thu = first + pd.Timedelta(days=(3 - first.weekday()) % 7 + 7)
        cand = idx[(idx <= thu) & (idx >= first)]
        if len(cand):
            out.add(cand.max())
    return pd.DatetimeIndex(sorted(out))


def supplementary(cell, sig, ret, px, dtv, other_ret=None):
    """B 구간 보고용 항목(판정 불사용)."""
    d = sample(sig, ret, "B")
    s, r = d["s"], d["r"]
    rep = {}
    # 상위 5밤 제외
    top = r[s == 1].nlargest(5).index
    rep["delta_ex_top5_bp"] = E.delta_test(s.drop(top), r.drop(top))[0] * 1e4
    rep["top5_nights"] = [f"{x.date()} {r[x] * 1e4:+.0f}bp" for x in top]
    # 1% 윈저화
    lo, hi = r.quantile(0.01), r.quantile(0.99)
    rep["delta_winsor1_bp"] = E.delta_test(s, r.clip(lo, hi))[0] * 1e4
    # 옵션 만기일 제외
    ex = expiry_days(px.index)
    keep = ~d.index.isin(ex)
    rep["n_sig_expiry"] = int((s[~keep] == 1).sum())
    dl, t, _ = E.delta_test(s[keep], r[keep])
    rep["delta_ex_expiry_bp"], rep["t_ex_expiry"] = dl * 1e4, t
    # 통제 회귀 r_on ~ 신호 + r_cc(T) + r_id(T) + dTV20(T)
    X = pd.DataFrame({"signal": s, "r_cc": (px["close"] / px["close"].shift(1) - 1).reindex(d.index),
                      "r_id": (px["close"] / px["open"] - 1).reindex(d.index), "dTV20": dtv.reindex(d.index)})
    ctrl, n = C.hac_reg(r, X)
    rep["ctrl"] = {k: dict(coef_bp=float(v.coef * 1e4), t=float(v.t)) for k, v in ctrl.iterrows()}
    rep["ctrl_n"] = n
    # 레버리지 배율(같은 1배 ETF 신호)
    rl = E.legs(data.price(cell["lev"]))["on"]
    lv = {}
    for per in ("A", "B", "E4"):
        a, b = PERIODS[per]
        dd = pd.DataFrame({"s": sig, "r1": ret, "rl": rl}).loc[a:b].dropna()
        d1, t1, n1 = E.delta_test(dd["s"], dd["r1"])
        dL, tL, _ = E.delta_test(dd["s"], dd["rl"])
        lv[per] = dict(n=len(dd), n_sig=n1, delta_1x_bp=d1 * 1e4, delta_lev_bp=dL * 1e4, t_lev=tL,
                       ratio=(dL / d1) if (d1 > 0 and dL > 0) else np.nan)
    rep["leverage"] = lv
    # 코스피200 지수 오버나잇(069500 신호)
    if cell["index"]:
        ri = E.legs(data.price(cell["index"]))["on"]
        ix = {}
        for per in ("B", "E4"):
            dd = sample(sig, ri, per)
            dl, t, n = E.delta_test(dd["s"], dd["r"])
            ix[per] = dict(n=len(dd), n_sig=n, delta_bp=dl * 1e4, t=t)
        rep["index"] = ix
    return rep


def baseline_rows(sig, ret, label):
    """기준선(매일 보유 오버나잇, 매일 왕복 비용) 대비. 표본 = 신호 계산 가능 밤."""
    rows = []
    for per in ("A", "B", "E4"):
        d = sample(sig, ret, per)
        for cost in E.COSTS_BP:
            for name, pos in (("매일 보유", pd.Series(1.0, index=d.index)), (label, d["s"])):
                st = E.stats(pos, d["r"], cost)
                rows.append(dict(period=per, cost_bp=cost, rule=name, **st))
    return rows


# ---------- 표시 ----------
def f(x, fmt="{:+.2f}"):
    return C._f(x, fmt)


def pct(x):
    return "-" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x * 100:.1f}%"


def holm_status(p):
    if np.isnan(p):
        return "p 계산 불가"
    if p >= 0.05:
        return f"원 p {p:.4f} ≥ 0.05 → Holm(m={HOLM_M}) 어느 순위에서도 통과 불가"
    if p <= 0.05 / HOLM_M:
        return (f"원 p {p:.4f} ≤ 0.05/{HOLM_M} = {0.05 / HOLM_M:.5f} → 다른 셀 p 와 무관하게 Holm(m={HOLM_M}) 기각 문턱 충족"
                f"(더 작은 p 도 모두 0.05/{HOLM_M} 이하라 단계가 끊기지 않음)")
    return f"원 p {p:.4f}: 0.05/{HOLM_M}={0.05 / HOLM_M:.5f} 초과·0.05 미만 → 종합에서 Holm 순위에 따라 결정"


def run_cell(cell, a_only=False):
    px = data.price(cell["asset"])
    tv, inv = market_tv(cell["market"])
    ret = E.legs(px)["on"]
    sig = d1_signal(px, tv, **MAIN)
    _, dtv = d1_parts(px, tv, MAIN["tv_win"])
    vol = C.vol_adj(ret)
    res = {"cell_id": cell["cell_id"], "asset": cell["asset"], "asset_name": data.NAMES.get(cell["asset"], ""),
           "market_tv": cell["market"]}
    A, dA = block(sig, ret, "A", vol=vol, placebo=True)
    res["A"] = A
    res["A_reproduced"] = bool(A["n_sig"] == cell["A_ref"]["n_sig"] and abs(A["delta_bp"] - cell["A_ref"]["delta_bp"]) < 0.01)
    res["A_neighbors"] = {k: block(d1_signal(px, tv, **q), ret, "A")[0]["delta_bp"] for k, q in NEIGH.items()}
    if a_only:
        return res, None
    # 자료 점검
    tvr = tv.reindex(px.index)
    sv = inv[[f"sellv_{c}" for c in CATS]].sum(axis=1, min_count=len(CATS))
    mism = ((tv - sv).abs() / tv > 1e-6).reindex(px.index, fill_value=False).astype(bool)
    chk = {}
    for per, (a, b) in PERIODS.items():
        seg = px.loc[a:b]
        chk[per] = dict(etf_days=len(seg), tv_nan=int(tvr.loc[a:b].isna().sum()), h_eq_l=int((seg["high"] == seg["low"]).sum()),
                        buy_sell_mismatch_days=int(mism.loc[a:b].sum()))
    chk["investor_last_date"] = str(inv.index.max().date())
    chk["etf_last_date"] = str(px.index.max().date())
    res["data_check"] = chk
    # B 판정
    B, dB = block(sig, ret, "B", vol=vol, placebo=True)
    res["B"] = B
    nb = {}
    for k, q in NEIGH.items():
        o, _ = block(d1_signal(px, tv, **q), ret, "B")
        nb[k] = dict(delta_bp=o["delta_bp"], t=o["t"], n_sig=o["n_sig"], net7_bp=o["net7_bp"])
    res["B_neighbors"] = nb
    res["B_yearly"] = yearly_delta(dB)
    E4, dE = block(sig, ret, "E4", vol=vol, placebo=False)
    res["E4"] = E4
    res["E4_yearly"] = yearly_delta(dE)
    res["E4_neighbors"] = {k: block(d1_signal(px, tv, **q), ret, "E4")[0]["delta_bp"] for k, q in NEIGH.items()}
    res["supp_B"] = supplementary(cell, sig, ret, px, dtv)
    # 판정
    neigh_vals = [v["delta_bp"] for v in nb.values()]
    same_sign = bool(all(np.sign(x) == np.sign(B["delta_bp"]) for x in neigh_vals))
    crit = {"①": bool(B["delta_bp"] > 0), "②": bool(B["net7_bp"] > 0),
            "③": bool(not np.isnan(B["placebo_p"]) and B["placebo_p"] <= 0.05), "④": bool(B["dvol"] > 0),
            "⑤": bool(all(x > 0 for x in neigh_vals))}
    res["neighbors_same_sign_B"] = same_sign
    res["criteria"] = crit
    res["criteria_str"] = "".join("O" if v else "X" for v in crit.values())
    res["grade"] = "통과 후보(종합 Holm 대기)" if all(crit.values()) else "미통과"
    res["holm_raw_p_one"] = B["p_one"]
    res["holm_status"] = holm_status(B["p_one"])
    se = abs(B["delta_bp"] / B["t"]) if B["t"] and not np.isnan(B["t"]) and B["t"] != 0 else np.nan
    res["power"] = dict(se_bp=se, delta_needed_p05_bp=Z_05 * se, delta_needed_holm_min_bp=Z_HOLM_MIN * se,
                        ratio_B_to_A=B["delta_bp"] / A["delta_bp"] if A["delta_bp"] else np.nan)
    # 기준선 표·장부용 기간 표
    res["baseline"] = baseline_rows(sig, ret, cell["cell_id"])
    rv = ret.where(sig.notna()).loc[PERIODS["A"][0]:]
    rules = {cell["cell_id"]: sig, **{f"{cell['cell_id']} {NEIGH_LABEL[k]}": d1_signal(px, tv, **q) for k, q in NEIGH.items()}}
    tabs = []
    for cost in E.COSTS_BP:
        t = E.evaluate(rules, rv, cost_bp=cost)
        t["asset"] = cell["asset"]
        tabs.append(t)
    return res, pd.concat(tabs, ignore_index=True)


def cell_md(r, other):
    A, B, E4 = r["A"], r["B"], r["E4"]
    s = r["supp_B"]
    md = [f"## {r['cell_id']} — {r['asset']} {r['asset_name']} (TV = {r['market_tv']} 시장 거래대금)\n",
          f"**판정: {r['grade']}** (기준 ①②③④⑤ = {r['criteria_str']}). {r['holm_status']}.\n",
          "| 항목 | A 2010~2019 (표본 내) | **B 2020-01~2025-06-05 (확인)** | E4 2025-06-09~ (보고) | B 기준 |",
          "|---|---|---|---|---|",
          f"| 결정일 범위 | {A['start']}~{A['end']} | {B['start']}~{B['end']} | {E4['start']}~{E4['end']} | |",
          f"| 밤 / 신호 밤(비율) | {A['n_days']} / {A['n_sig']} ({pct(A['rate'])}) | {B['n_days']} / {B['n_sig']} ({pct(B['rate'])}) | "
          f"{E4['n_days']} / {E4['n_sig']} ({pct(E4['rate'])}) | |",
          f"| 신호 밤 평균 / 비신호 평균(bp, 비용 전) | {f(A['sig_mean_bp'])} / {f(A['non_mean_bp'])} | {f(B['sig_mean_bp'])} / {f(B['non_mean_bp'])} | "
          f"{f(E4['sig_mean_bp'])} / {f(E4['non_mean_bp'])} | |",
          f"| ① Δ(신호−비신호) bp | {f(A['delta_bp'])} | **{f(B['delta_bp'])}** | {f(E4['delta_bp'])} | > 0 |",
          f"| NW(lag5) t / 단측 p(원값) | {f(A['t'])} / {f(A['p_one'], '{:.4f}')} | **{f(B['t'])} / {f(B['p_one'], '{:.4f}')}** | "
          f"{f(E4['t'])} / {f(E4['p_one'], '{:.4f}')} | Holm 은 종합 |",
          f"| 신호 밤 순평균 5bp / ② 7bp | {f(A['net5_bp'])} / {f(A['net7_bp'])} | {f(B['net5_bp'])} / **{f(B['net7_bp'])}** | "
          f"{f(E4['net5_bp'])} / {f(E4['net7_bp'])} | 7bp > 0 |",
          f"| ③ 연도 층화 플라시보 p | {f(A['placebo_p'], '{:.4f}')} | **{f(B['placebo_p'], '{:.4f}')}** | - | ≤ 0.05 |",
          f"| ④ 변동성 조정 Δ(SD) | {f(A['dvol'], '{:+.4f}')} | **{f(B['dvol'], '{:+.4f}')}** | {f(E4['dvol'], '{:+.4f}')} | > 0 |",
          f"| ⑤ 이웃 Δ bp (N1/N2/N3/N4) | {' / '.join(f(v, '{:+.1f}') for v in r['A_neighbors'].values())} | "
          f"**{' / '.join(f(v['delta_bp'], '{:+.1f}') for v in r['B_neighbors'].values())}** | "
          f"{' / '.join(f(v, '{:+.1f}') for v in r['E4_neighbors'].values())} | 모두 > 0 |",
          f"| 중앙값 신호 / 비신호(차) bp | {f(A['sig_median_bp'])} / {f(A['non_median_bp'])} ({f(A['median_diff_bp'])}) | "
          f"{f(B['sig_median_bp'])} / {f(B['non_median_bp'])} ({f(B['median_diff_bp'])}) | "
          f"{f(E4['sig_median_bp'])} / {f(E4['non_median_bp'])} ({f(E4['median_diff_bp'])}) | |",
          f"| 승률 신호 / 비신호 | {pct(A['win_sig'])} / {pct(A['win_non'])} | {pct(B['win_sig'])} / {pct(B['win_non'])} | "
          f"{pct(E4['win_sig'])} / {pct(E4['win_non'])} | |",
          f"| 무조건 오버나잇 평균(같은 표본) bp | {f(A['all_mean_bp'])} | {f(B['all_mean_bp'])} | {f(E4['all_mean_bp'])} | |",
          "",
          f"A 사양 재현: 신호 {A['n_sig']}밤, Δ {A['delta_bp']:+.2f}bp → {'사양 수치와 일치' if r['A_reproduced'] else '사양 수치와 불일치(아래 해석 참조)'}.\n",
          "이웃 B 상세: " + ", ".join(f"{NEIGH_LABEL[k]} Δ {f(v['delta_bp'])} (t {f(v['t'])}, 신호 {v['n_sig']}밤, 7bp 순 {f(v['net7_bp'])})"
                                     for k, v in r["B_neighbors"].items()) + "\n",
          "연도별 Δ(bp, 신호 밤 수) — B: " + ", ".join(f"{y['year']} {f(y['delta_bp'], '{:+.1f}')}({y['n_sig']})" for y in r["B_yearly"])
          + " / E4: " + ", ".join(f"{y['year']} {f(y['delta_bp'], '{:+.1f}')}({y['n_sig']})" for y in r["E4_yearly"]) + "\n",
          "### 보고용 부가 확인 (B, 판정 불사용)\n",
          "| 항목 | 값 |", "|---|---|",
          f"| 상호 이전(같은 정의의 상대 셀 {other['cell_id']} B Δ) | {f(other['B']['delta_bp'])}bp (t {f(other['B']['t'])}) |",
          f"| 상위 5개 신호 밤 제외 Δ | {f(s['delta_ex_top5_bp'])}bp (제외: {', '.join(s['top5_nights'])}) |",
          f"| 1% 윈저화 Δ | {f(s['delta_winsor1_bp'])}bp |",
          f"| 옵션 만기일 제외 Δ | {f(s['delta_ex_expiry_bp'])}bp (t {f(s['t_ex_expiry'])}, 만기일 신호 {s['n_sig_expiry']}밤) |",
          f"| 통제 회귀 신호 계수 (r_cc·r_id·dTV20 동시, n={s['ctrl_n']}) | {f(s['ctrl']['signal']['coef_bp'])}bp (t {f(s['ctrl']['signal']['t'])}); "
          + ", ".join(f"{k} {f(v['coef_bp'])}(t {f(v['t'])})" for k, v in s["ctrl"].items() if k != "signal") + " |"]
    for per, lv in s["leverage"].items():
        md.append(f"| 레버리지 {r['asset']}→{CELLS_BY_ID[r['cell_id']]['lev']} {per} (공통 {lv['n']}밤, 신호 {lv['n_sig']}) | "
                  f"1배 Δ {f(lv['delta_1x_bp'])} / 레버리지 Δ {f(lv['delta_lev_bp'])} (t {f(lv['t_lev'])}) → 배율 {f(lv['ratio'], '{:.2f}')} |")
    if "index" in s:
        for per, ix in s["index"].items():
            md.append(f"| 코스피200 지수 오버나잇 {per} (신호 {ix['n_sig']}/{ix['n']}밤) | Δ {f(ix['delta_bp'])}bp (t {f(ix['t'])}) |")
    md.append("")
    return md


def baseline_md(r):
    md = [f"### {r['cell_id']} 기준선(매일 보유 오버나잇) 대비 — 표본 = 신호 계산 가능 밤\n",
          "| 구간 | 비용 | 규칙 | 밤 | 거래 | 노출 | 거래당 순평균 bp | 순평균 t | 승률 | 샤프 | CAGR | MDD |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for x in r["baseline"]:
        md.append(f"| {x['period']} | {x['cost_bp']:.0f}bp | {x['rule']} | {x['days']} | {x['trades']} | {pct(x['exposure'])} | {f(x['net_bp'])} | "
                  f"{f(x['t_net'])} | {pct(x['win'])} | {f(x['sharpe'])} | {pct(x['cagr'])} | {pct(x['mdd'])} |")
    md.append("")
    return md


def narrative(r):
    B, A, E4, pw = r["B"], r["A"], r["E4"], r["power"]
    fails = [k for k, v in r["criteria"].items() if not v]
    L = [f"- **{r['cell_id']}**: B Δ {B['delta_bp']:+.2f}bp (t {B['t']:+.2f}, 단측 p {B['p_one']:.4f}), 신호 {B['n_sig']}밤. "
         f"A Δ {A['delta_bp']:+.2f}bp 대비 {pw['ratio_B_to_A'] * 100:.0f}%. "
         + ("판정 기준 ①~⑤ 모두 충족 — 다만 셀 단위 통과는 종합 Holm 보정 결과에 달려 있다." if not fails
            else f"미충족 기준: {', '.join(fails)}.")]
    L.append(f"  - 검정력: B 의 Δ 표준오차 ≈ {pw['se_bp']:.1f}bp. 단측 5%에 필요한 Δ ≈ {pw['delta_needed_p05_bp']:.1f}bp, "
             f"Holm 최엄격 순위(0.05/{HOLM_M})에 필요한 Δ ≈ {pw['delta_needed_holm_min_bp']:.1f}bp. {r['holm_status']}.")
    if B["delta_bp"] > 0 and B["net7_bp"] <= 0:
        L.append(f"  - Δ 는 양이지만 신호 밤 평균 {B['sig_mean_bp']:+.2f}bp 가 7bp 비용을 넘지 못한다(순 {B['net7_bp']:+.2f}bp).")
    if B["delta_bp"] <= 0:
        L.append(f"  - B 에서 부호가 A 와 반대다(신호 밤 {B['sig_mean_bp']:+.2f} vs 비신호 {B['non_mean_bp']:+.2f}bp). 가설 방향의 확인 실패.")
    s = r["supp_B"]
    cs, cc = s["ctrl"]["signal"], s["ctrl"]["r_cc"]
    L.append(f"  - 보고용 견고성(B): 상위 5밤 제외 Δ {s['delta_ex_top5_bp']:+.2f}, 1% 윈저화 {s['delta_winsor1_bp']:+.2f}, "
             f"만기일 제외 {s['delta_ex_expiry_bp']:+.2f}bp, 중앙값 차 {B['median_diff_bp']:+.2f}bp; 통제 회귀 신호 계수 {cs['coef_bp']:+.2f}bp "
             f"(t {cs['t']:+.2f}), r_cc 계수 t {cc['t']:+.2f}.")
    if B["delta_bp"] > 0 and s["delta_ex_top5_bp"] <= 0.5 * B["delta_bp"]:
        L.append(f"  - 상위 5개 신호 밤({', '.join(x.split()[0] for x in s['top5_nights'])})을 빼면 Δ 가 절반 이하로 줄어 소수 극단 밤 의존이 크다.")
    if B["delta_bp"] > 0 and abs(cs["t"]) < 1.96:
        L.append("  - 당일 종가→종가 수익률(r_cc)·장중 수익률(r_id)·dTV20 을 함께 넣으면 신호 계수가 유의하지 않다. "
                 f"B 에서는 신호 효과의 상당 부분이 당일 하락 뒤 반등(r_cc 음의 계수)과 겹친다(A 사양의 통제 계수는 {A_CTRL[r['cell_id']]}).")
    lv = s["leverage"]["B"]
    if not np.isnan(lv["ratio"]):
        L.append(f"  - 레버리지 ETF 로 같은 신호를 옮기면 B Δ {lv['delta_lev_bp']:+.2f}bp, 배율 {lv['ratio']:.2f} (A 사양 {A_LEV[r['cell_id']]}).")
    L.append(f"  - E4(보고만): Δ {E4['delta_bp']:+.2f}bp (t {f(E4['t'])}, 신호 {E4['n_sig']}밤, 7bp 순 {E4['net7_bp']:+.2f}bp). 판정에는 쓰지 않았다.")
    return L


CELLS_BY_ID = {c["cell_id"]: c for c in CELLS}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a-only", action="store_true", help="A 구간 사양 재현만 출력(파일·장부 쓰지 않음)")
    ap.add_argument("--no-ledger", action="store_true", help="장부 기록 생략(재출력용)")
    ap.add_argument("--note", default="", help="보고서 머리에 덧붙일 실행 메모")
    args = ap.parse_args()
    spec = check_spec()
    if args.a_only:
        for cell in CELLS:
            r, _ = run_cell(cell, a_only=True)
            A = r["A"]
            print(cell["cell_id"], "A", A["start"], A["end"], "n", A["n_days"], "sig", A["n_sig"], f"delta {A['delta_bp']:+.2f} t {A['t']:+.2f}",
                  f"placebo {A['placebo_p']:.3f} net5 {A['net5_bp']:+.2f} net7 {A['net7_bp']:+.2f} dvol {A['dvol']:+.4f}",
                  "neigh", {k: round(v, 2) for k, v in r["A_neighbors"].items()}, "reproduced", r["A_reproduced"])
        return
    res, tabs = {}, []
    for cell in CELLS:
        r, t = run_cell(cell)
        res[cell["cell_id"]] = r
        tabs.append(t)
    ids = [c["cell_id"] for c in CELLS]
    run_at = dt.datetime.now(E.KST).strftime("%Y-%m-%d %H:%M")
    md = ["# R2 · D1 유동성 압력 반전 — 확인(B) 결과\n",
          f"실행 {run_at} KST. 스크립트 `code/overnight_lab/topics/r2_d1.py`, 고정 사양 `round2/D1/spec.json`(변경 없음, 탐색 변형 {spec['explored_variants']}개)."
          + (f" {args.note}" if args.note else "") + "\n",
          "```\n" + __doc__.strip() + "\n```\n",
          "## 요약\n",
          "| 셀 | 자산 | B 신호 밤 | B Δ bp | NW t | 단측 p(원) | 7bp 순 | 플라시보 p | 변동성 조정 Δ | 이웃 부호 | A Δ | E4 Δ | 기준 ①~⑤ | 판정 |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for i in ids:
        r = res[i]
        B = r["B"]
        md.append(f"| {i} | {r['asset']} | {B['n_sig']} | {f(B['delta_bp'])} | {f(B['t'])} | {f(B['p_one'], '{:.4f}')} | {f(B['net7_bp'])} | "
                  f"{f(B['placebo_p'], '{:.4f}')} | {f(B['dvol'], '{:+.4f}')} | {'O' if r['criteria']['⑤'] else 'X'} | {f(r['A']['delta_bp'])} | "
                  f"{f(r['E4']['delta_bp'])} | {r['criteria_str']} | {r['grade']} |")
    md += ["", "해석:\n"]
    for i in ids:
        md += narrative(res[i])
    both_pos = all(res[i]["B"]["delta_bp"] > 0 for i in ids)
    md += [f"- 상호 이전(코스피↔코스닥, 같은 정의): 두 셀 B Δ 가 {'모두 양' if both_pos else '같은 양의 부호가 아님'} "
           f"({', '.join(f'{i} {res[i]['B']['delta_bp']:+.2f}' for i in ids)}bp).",
           f"- Holm 보정: 이번 2차 전체 셀 수 최대 {HOLM_M}로 종합에서 한다. 여기서는 원 단측 p 만 낸다 "
           f"({', '.join(f'{i} {res[i]['B']['p_one']:.4f}' for i in ids)}).",
           "- 실행 제약(백테스트 미반영): 투자자별 거래대금 일별 확정치는 장 마감 후 공표라 실거래는 15:40 전후 잠정치로 판단해야 하고(임계 0 근처에서만 영향), "
           "시간외 종가 호가 잔량 부족에 따른 체결 실패 위험이 있다.", ""]
    md += ["## 판정 기준(셀별, B)\n",
           "① Δ(신호−비신호) > 0, NW(lag5) 단측 p 원값 보고(Holm 은 종합) · ② 7bp 차감 후 신호 밤 순평균 > 0 · ③ 연도 층화 플라시보 p ≤ 0.05 · "
           "④ 변동성 조정 Δ > 0 · ⑤ 이웃 N1~N4 Δ 모두 > 0. 다섯 개 모두 충족이면 '통과 후보(종합 Holm 대기)', 하나라도 미충족이면 '미통과'.\n",
           "해석 선택(모호한 곳은 보수적으로): 단측 p 는 NW t 의 정규 근사(common.judge 와 같음). ⑤ 는 '본 셀과 같은 부호'를 '모두 양(가설 방향)'으로 읽었다. "
           "변동성 조정은 r_on 을 직전 60밤 r_on SD(T 제외)로 나눈 값(common.vol_adj)의 Δ. 표본은 결정일이 구간 안이고 r_on·신호가 모두 있는 밤. "
           "신호의 20일 창은 구간 경계를 넘어 과거 자료를 쓴다(진입 시점에 알 수 있는 정보). 기준선은 같은 표본의 매일 오버나잇 보유(매일 왕복 비용).\n"]
    for k, i in enumerate(ids):
        md += cell_md(res[i], res[ids[1 - k]])
    md += ["## 기준선 대비 표 (5·7bp)\n",
           "기준선 = 매일 종가(시간외) 매수·익일 시가 매도, 매일 왕복 비용(engine 규약). 샤프·CAGR·MDD 는 현금 밤(수익 0) 포함 일별 계열 기준.\n"]
    for i in ids:
        md += baseline_md(res[i])
    md += ["## 자료 점검\n", "| 셀 | 구간 | ETF 거래일 | TV 결측 | H=L | 매수≠매도(>1e-6) 일수 |", "|---|---|---|---|---|---|"]
    for i in ids:
        ch = res[i]["data_check"]
        for per in PERIODS:
            c = ch[per]
            md.append(f"| {i} | {per} | {c['etf_days']} | {c['tv_nan']} | {c['h_eq_l']} | {c['buy_sell_mismatch_days']} |")
    md += ["", "투자자별 자료 마지막 날짜: " + ", ".join(f"{i} {res[i]['data_check']['investor_last_date']}(ETF {res[i]['data_check']['etf_last_date']})" for i in ids)
           + ". 투자자 자료가 없는 날은 신호 NaN 으로 표본에서 빠진다. 자료는 캐시 그대로 썼다(재수집 없음).\n"]
    md += ["## 부록: 기간별 표 (engine.evaluate, 5bp, 본 셀·이웃·기준선)\n"]
    t5 = pd.concat(tabs, ignore_index=True)
    t5 = t5[t5["cost_bp"] == 5.0]
    for i in ids:
        a = CELLS_BY_ID[i]["asset"]
        md += [f"### {i} ({a})\n", E.fmt_table(t5[t5["asset"] == a]), ""]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.md").write_text("\n".join(md), encoding="utf-8")
    verdict = {"family": spec["family"], "topic": TOPIC, "run_at": run_at, "spec_file": str(SPEC_FILE),
               "script": str(Path(__file__).resolve()), "periods": PERIODS, "holm_m": HOLM_M,
               "criteria_def": {"①": "B Δ>0 (원 단측 NW p 보고, Holm 은 종합)", "②": "B 7bp 순평균>0", "③": "B 연도 층화 플라시보 p≤0.05",
                                "④": "B 변동성 조정 Δ>0", "⑤": "B 이웃 N1~N4 Δ 모두>0"},
               "cells": res}
    C.save_verdict(OUT / "verdict.json", verdict)
    if not args.no_ledger:
        lspec = {"family": "D1 CLV×dTV20", "spec_file": "round2/D1/spec.json", "rule": "CLV<=0.20 & dTV20>0 (TV=investor buyv 12cats sum)",
                 "cells": {c["cell_id"]: c["asset"] for c in CELLS}, "neighbors": NEIGH, "entry": "T after-hours close", "exit": "T+1 open",
                 "periods": PERIODS, "cost_bp": list(E.COSTS_BP)}
        E.ledger_append(data.OUT, TOPIC, lspec, pd.concat(tabs, ignore_index=True), note="2차 확인")
    print("\n".join(md[3:md.index("## 판정 기준(셀별, B)\n")]))


if __name__ == "__main__":
    main()
