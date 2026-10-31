# -*- coding: utf-8 -*-
"""16차 봉인 구간 1회 평가. 사전 등록 alpha-search-prereg-2026-10-31.md 의 다리·장부·k·판정·선택 규칙을 그대로 적용한다.

입력: r16_eval_legs.py 캐시(없거나 더 오래되면 재생성). 2024-12-30 까지 다리 전 필드는 그쪽에서 스캔과 대조했다.
장부 함수 legs_of·parts·sub·notional·run 은 r16s_portfolio.py 에서 옮겼다. 쓰지 않는 배분·위험 규칙 분기만 뺐고 고정 k·50:50·bookcap 경로는 같은 식이다.
창(사전 등록): D 2022-12-13~2024-12-30, S 2025-01-02~평가 끝, W D+S, X 2022-12-13~2025-12-30, L 2016-10-04~평가 끝. 장부 평가 끝 = 다리 자료가 모두 있는 마지막 날.
비용: 기본 net, 스트레스 stress, 하한 = 기본 + 단일가 ETF 1틱 + 선물 3bp. 스윙 두 다리의 하한만 회전 기준이다(아래 F["low"] 주석).
통계: 활성일(스윙은 거래) 순수익 bp 상수 회귀 HAC(5) → NW t. CAGR = Π(1+r)^(252/일수)−1(비활성일 0), MDD = min(누적/누적최고−1).
산출: round16_alpha_search/eval/ leg_results.csv, book_results.csv, book_yearly.csv, contributions.csv, end_dates.csv, futures_check.csv, judgement.json, notes.md, fig_books.png
실행 순서: r16_eval_legs.py → r16_eval_tail.py(달력 뒤 민감도, tail_sensitivity.csv) → r16_eval.py. 앞 둘은 PYTHONUTF8=1 로 각각 실행.
"""
import json
import pickle
import re
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import statsmodels.api as sm

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
pd.set_option("display.width", 250, "display.max_columns", 50, "display.max_rows", 400, "display.max_colwidth", 60)
matplotlib.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = True

DESIGN_END = pd.Timestamp("2024-12-30")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
LAB = Path("<OUT>")
SCAN = LAB / "round16_alpha_search/scan"
OUT = LAB / "round16_alpha_search/eval"
OUT.mkdir(parents=True, exist_ok=True)
RAW = Path("<raw>")
GIT = Path("<repo>")
SCRATCH = Path("<local>")
CACHE = SCRATCH / "r16_eval_legs.pkl"
if not CACHE.exists() or CACHE.stat().st_mtime < (HERE / "r16_eval_legs.py").stat().st_mtime:
    import r16_eval_legs  # noqa: F401  캐시 재생성
d0 = pickle.load(open(CACHE, "rb"))
L, CAL, TR, ENDS, LEG_END, INTEG = d0["L"], d0["CAL"], d0["TR"], d0["ENDS"], d0["LEG_END"], d0["INTEG"]
FUT, FUT_S = 1.8e-4, 3e-4
BANS = (("2020-03-16", "2021-05-02"), ("2023-11-06", "2025-03-30"))  # 공매도 전면 금지. 2023 기간 끝은 설계 표(2024-12-30 절단)가 아니라 실제 재개 전날
SWING = ("X_MRq", "X_FLzVT")
LEGS = tuple(LEG_END)
F = {f: pd.DataFrame({k: getattr(v, f) for k, v in L.items()}) for f in ("net", "stress", "gross", "pos", "fut", "auc")}
assert F["net"].index.equals(CAL)
F["low_scan"] = F["net"] - F["auc"].fillna(0) - (FUT_S - FUT) * F["pos"].abs().fillna(0) * F["fut"].fillna(0)  # 15차 하한 묶음과 같은 정의(하루 다리에서 왕복 3bp)
F["low"] = F["low_scan"].copy()
for _k in SWING:  # 스윙에 위 식을 쓰면 보유일마다 1.2bp 가 빠져 사전 등록 '선물 3bp'(왕복)와 달라진다. 순수 선물 다리는 회전 기준 3bp = 스트레스 흐름
    F["low"][_k] = F["stress"][_k]
for _k in ("KA_STf", "X_K2", "X_D1"):
    assert (F["low"][_k] - F["stress"][_k]).abs().max() < 1e-12  # 하루 선물 다리는 두 정의가 같다
ZERO = pd.Series(0.0, index=CAL)
FALSE = pd.Series(False, index=CAL)
_i = CAL.searchsorted(TR["X_FLzVT"].index) + 1  # FLzVT 는 T+1 시가 체결이라 거래 창 배정은 체결일. MRq 는 T 15:45 체결이라 결정일 = 체결일
TR2 = {**TR, "X_FLzVT": TR["X_FLzVT"][_i < len(CAL)].set_axis(CAL[_i[_i < len(CAL)]])}


def hac_t(x):
    x = np.asarray(x, float)
    if len(x) < 5 or x.std() == 0:
        return np.nan
    return float(sm.OLS(x, np.ones(len(x))).fit(cov_type="HAC", cov_kwds={"maxlags": 5}).tvalues[0])


def eqm(r):
    e = (1 + r).cumprod()
    return float(e.iloc[-1] ** (252 / len(r)) - 1), float((e / e.cummax() - 1).min())


def sharpe(r):
    return float(r.mean() * 252 / (r.std() * 252 ** .5)) if r.std() > 0 else np.nan


def dd_span(r):
    e = (1 + r).cumprod()
    t = (e / e.cummax() - 1).idxmin()
    return e.loc[:t].idxmax().date(), t.date()


def ban_mask(idx):
    m = np.zeros(len(idx), bool)
    for a, b in BANS:
        m |= (idx >= a) & (idx <= b)
    return m


def wins(end):
    return {"D": ("2022-12-13", "2024-12-30"), "S": ("2025-01-02", end), "W": ("2022-12-13", end), "X": ("2022-12-13", "2025-12-30"), "L": ("2016-10-04", end)}


def dot(d):
    return pd.Timestamp(d).strftime("%Y.%m.%d")


# ---------------------------------------------------------------- 장부 (r16s_portfolio.py 에서 옮김)
def sp(name, K=None, Q=None):
    return dict(name=name, K=K or {}, Q=Q or {})


_KX3 = dict(A=["KA_STf"], B=["KB_E1s3"], C=["KC_KP"], X=["X_K2"])
BOOKS = {"P1": (sp("E3s", dict(B=["KB_E1s3"], C=["KC_KP"]), dict(A=["QA_F"], C=["QC_KQ"])), 1.25, None),
         "P2": (sp("X2s", dict(A=["KA_STf"], B=["KB_E1s3"], C=["KC_KP"]), dict(A=["QA_F"], C=["QC_KQ"])), 1.00, None),
         "P3": (sp("X10s", _KX3, dict(A=["QA_F"], C=["QC_KQ"], X=["X_D1"])), 1.25, 3.0),
         "P4": (sp("P3 + 스윙", _KX3, dict(A=["QA_F"], C=["QC_KQ"], X=["X_D1", ("X_MRq", .5), ("X_FLzVT", .5)])), 1.00, 3.0)}


def legs_of(s):
    out = []
    for m in "KQ":
        for slot, items in s[m].items():
            for it in items:
                k, wt = (it, 1.0) if isinstance(it, str) else it
                out.append((m, slot, k, wt))
    return out


def parts(s, field):
    P = {}
    for m in "KQ":
        P[m] = {x: ZERO.copy() for x in "ABCX"}
        act = FALSE.copy()
        for slot, items in s[m].items():
            for it in items:
                k, wt = (it, 1.0) if isinstance(it, str) else it
                P[m][slot] = P[m][slot] + F[field][k].fillna(0.0) * wt
                if slot == "B":
                    act = act | (F["pos"][k].fillna(0.0) != 0)
        P[m]["C"] = P[m]["C"].where(~act, 0.0)
    return P


def notional(s, sc):
    """총 명목(장부 자본 대비). 밤·낮은 시간이 겹치지 않아 큰 쪽이 그 날 최대."""
    sA, sD, wk, wq = sc
    nn, dd = ZERO.copy(), ZERO.copy()
    for m, wm in (("K", wk), ("Q", wq)):
        act = FALSE.copy()
        for slot, items in s[m].items():
            if slot == "B":
                for it in items:
                    act = act | (F["pos"][it if isinstance(it, str) else it[0]].fillna(0) != 0)
        for ms, slot, k, wt in legs_of(s):
            if ms != m:
                continue
            p = F["pos"][k].fillna(0).abs() * wt
            if slot == "A":
                nn = nn + wm * p
            elif slot == "C":
                dd = dd + wm * p.where(~act, 0.0)
            elif slot == "B":
                dd = dd + wm * p
            else:
                nn, dd = nn + wm * p, dd + wm * p
    return sA * nn, sD * dd


def run(s, field="net", k=1.0, bookcap=None):
    """장부 일수익. 50:50, 고정 k. bookcap = 장부 총 명목 상한(그 시간대 다리를 비례 축소). X 슬롯은 스캔 그대로 밤 배수 sA 를 탄다."""
    wk, wq = .5, .5
    start = min(L[x].start for _, _, x, _ in legs_of(s))
    sA, sD = pd.Series(k, index=CAL), pd.Series(k, index=CAL)
    if bookcap:
        nn, dd = notional(s, (sA, sD, wk, wq))
        sA, sD = sA * (bookcap / nn.where(nn > 0, bookcap)).clip(upper=1.0), sD * (bookcap / dd.where(dd > 0, bookcap)).clip(upper=1.0)
    P = parts(s, field)
    night = wk * (P["K"]["A"] + P["K"]["X"]) + wq * (P["Q"]["A"] + P["Q"]["X"])
    day = wk * (P["K"]["C"] + P["K"]["B"]) + wq * (P["Q"]["C"] + P["Q"]["B"])
    return (sA * night + sD * day).where(CAL >= start, 0.0), (sA, sD, wk, wq)


def contrib(s, sc, field):
    """다리별 장부 기여 일수익. 합 = run() 일수익(아래에서 단언)."""
    sA, sD, wk, wq = sc
    out = {}
    for m, wm in (("K", wk), ("Q", wq)):
        act = FALSE.copy()
        for ms, sl, lk, _ in legs_of(s):
            if ms == m and sl == "B":
                act = act | (F["pos"][lk].fillna(0) != 0)
        for ms, sl, lk, wt in legs_of(s):
            if ms != m:
                continue
            x = F[field][lk].fillna(0) * wt * wm * (sD if sl in "BC" else sA)
            out[(m, sl, lk, wt)] = x.where(~act, 0.0) if sl == "C" else x
    return out


# ---------------------------------------------------------------- 스캔 흐름 대조(장부)
sc_ = pd.read_csv(SCAN / "portfolio/series_candidates.csv", encoding="utf-8-sig", parse_dates=["date"]).set_index("date")
BOOK_DIFF = {}
for tag in ("P1", "P2", "P3"):
    s, k0, _ = BOOKS[tag]
    BOOK_DIFF[tag] = float((run(s, "net", k0)[0].reindex(sc_.index) - sc_[tag]).abs().max())  # 저장 흐름은 상한 없음
    assert BOOK_DIFF[tag] < 6e-9, (tag, BOOK_DIFF[tag])  # 저장 소수 8자리 반올림 안
print("series_candidates 장부 일수익 최대 차:", BOOK_DIFF)
_r3 = run(BOOKS["P3"][0], "net", 1.25, 3.0)[0]
_c1, _c2 = eqm(_r3.loc["2022-12-13":DESIGN_END]), eqm(_r3.loc["2016-10-04":DESIGN_END])
print(f"스캔 상한 3 재현: 창 D {_c1[0] * 100:.2f}/{_c1[1] * 100:.2f}(기준 64.12/−8.23), 창 L {_c2[0] * 100:.2f}/{_c2[1] * 100:.2f}(기준 34.38/−8.23)")
assert abs(_c1[0] * 100 - 64.12) < .01 and abs(_c2[0] * 100 - 34.38) < .01 and abs(_c1[1] * 100 + 8.23) < .01

# ---------------------------------------------------------------- 단독 다리
YEARS = range(2016, 2027)


def leg_row(k, w, cost, a=None, b=None, mask=None):
    v = L[k]
    day = (v.net if cost == "기본" else v.stress).dropna()
    day = day.loc[a:b] if mask is None else day[mask[day.index]]
    if not len(day):
        return None
    if k in SWING:
        t = TR2[k]
        t = t.loc[a:b] if mask is None else t[mask.reindex(t.index, fill_value=False)]
        x, g, unit = t["net" if cost == "기본" else "stress"], t.gross, "거래"
    else:
        act = v.net.reindex(day.index) != 0
        x, g, unit = day[act], v.gross.reindex(day.index)[act], "활성일"
    c, m = eqm(day)
    return dict(leg=k, market=v.mkt, slot=v.slot, window=w, cost=cost, start=day.index[0].date(), end=day.index[-1].date(), days=len(day), unit=unit, n=len(x), gross_bp=g.mean() * 1e4 if len(x) else np.nan,
                net_bp=x.mean() * 1e4 if len(x) else np.nan, t_nw=hac_t(x.values * 1e4), cagr_pct=c * 100, mdd_pct=m * 100, top5_share_pct=x.nlargest(5).sum() / x.sum() * 100 if len(x) and x.sum() > 0 else np.nan)


rows = []
for k in LEGS:
    end = LEG_END[k]
    w_ = wins(end)
    w_start = pd.Timestamp(w_["W"][0])
    in_ban = pd.Series(ban_mask(CAL), index=CAL)
    in_w = pd.Series((CAL >= w_start) & (CAL <= end), index=CAL)
    for cost in ("기본", "스트레스"):
        for w, (a, b) in w_.items():
            if w == "L" and L[k].start > pd.Timestamp(a):
                continue
            rows.append(leg_row(k, w, cost, a, b))
        for y in YEARS:
            rows.append(leg_row(k, f"Y{y}", cost, f"{y}-01-01", min(pd.Timestamp(f"{y}-12-31"), end)))
        rows.append(leg_row(k, "BAN_IN", cost, mask=in_w & in_ban))
        rows.append(leg_row(k, "BAN_OUT", cost, mask=in_w & ~in_ban))
LR = pd.DataFrame([r for r in rows if r])

# ---------------------------------------------------------------- 설계 선별(사전 등록 다리 판정의 첫 조건)
DW = {"KB_E1s3": ("2016-10-04", "L"), "KC_KP": ("2022-12-13", "D"), "QA_F": ("2016-10-04", "L"), "QC_KQ": ("2022-12-13", "D"), "KA_STf": ("2010-11-01", "2010.11.01~"), "X_K2": ("2011-01-01", "2011~"), "X_D1": ("2016-01-01", "2016~"),
      "X_N1": ("2015-01-01", "2015.11~"), "X_MRq": ("2016-01-04", "2016~"), "X_FLzVT": ("2016-01-04", "2016~"), "QC_Df": ("2022-12-13", "D"), "M_zfQ": ("2023-07-31", "2023.07.31~")}


def design(k):
    a, tag = DW[k]
    if k in TR:
        t = TR[k].loc[a:DESIGN_END]
        x, st = t.net, t.stress
    else:
        r = L[k].net.loc[a:DESIGN_END].dropna()
        x = r[r != 0]
        st = L[k].stress.reindex(x.index)
    y = x.groupby(x.index.year)
    ys = y.sum()[y.size() >= 5]
    h = len(x) // 2
    c = dict(순수익=x.mean() > 0, t2=hac_t(x.values * 1e4) >= 2, 연도70=(ys > 0).mean() >= .7, 상위5_50=x.nlargest(5).sum() / x.sum() < .5, 전반=x.iloc[:h].mean() > 0, 후반=x.iloc[h:].mean() > 0, 스트레스=st.mean() > 0)
    return dict(창=tag, n=len(x), 순bp=round(float(x.mean() * 1e4), 2), t=round(hac_t(x.values * 1e4), 2), 연도양수=f"{int((ys > 0).sum())}/{len(ys)}", 상위5pct=round(float(x.nlargest(5).sum() / x.sum() * 100), 1),
                스트레스bp=round(float(st.mean() * 1e4), 2), 미달=[a_ for a_, b_ in c.items() if not b_], 통과=bool(all(c.values())))


DES = {k: design(k) for k in DW}
_rv = pd.read_csv(SCAN / "rv_arb/cells.csv", encoding="utf-8-sig").set_index("id").loc["F4g_0787"]  # C1 은 선물 1분봉이 설계 구간에 없어 NAV 대리 스캔 셀로 판정
_c = dict(순수익=_rv.net_bp > 0, t2=_rv.t_nw >= 2, 연도70=_rv.yearly_pos >= .7, 상위5_50=_rv.top5_share_pct < 50, 전반=_rv.half1_net_bp > 0, 후반=_rv.half2_net_bp > 0, 스트레스=_rv.stress_net_bp > 0)
DES["C1_RB"] = dict(창="2022.11.23~ NAV 대리", n=int(_rv.n), 순bp=float(_rv.net_bp), t=float(_rv.t_nw), 연도양수="2/2", 상위5pct=float(_rv.top5_share_pct), 스트레스bp=float(_rv.stress_net_bp), 미달=[a_ for a_, b_ in _c.items() if not b_], 통과=bool(all(_c.values())) and "기준·인접 통과" in _rv.notes)

_s = LR[(LR.window == "S") & (LR.cost == "기본")].set_index("leg")
VERD = {}
for k in LEGS:
    ok = bool(DES[k]["통과"] and _s.net_bp[k] > 0 and _s.t_nw[k] >= 1)
    VERD[k] = dict(설계선별=DES[k], S순bp=round(float(_s.net_bp[k]), 2), S_t=round(float(_s.t_nw[k]), 2), S_n=int(_s.n[k]), 판정="채택 후보" if ok else "미채택")
LR["판정"] = LR.leg.map({k: v["판정"] for k, v in VERD.items()})
LR.to_csv(OUT / "leg_results.csv", index=False, encoding="utf-8-sig", float_format="%.4f")
print(pd.DataFrame({k: dict(설계창=v["설계선별"]["창"], 설계n=v["설계선별"]["n"], 설계순=v["설계선별"]["순bp"], 설계t=v["설계선별"]["t"], 설계통과=v["설계선별"]["통과"], 미달=",".join(v["설계선별"]["미달"]), S_n=v["S_n"], S순=v["S순bp"], S_t=v["S_t"], 판정=v["판정"]) for k, v in VERD.items()}).T.to_string())

# ---------------------------------------------------------------- 장부
COST = {"기본": "net", "하한": "low", "스트레스": "stress"}
rows, yrows, crows = [], [], []
BOOK_END = {b: min(LEG_END[x] for _, _, x, _ in legs_of(v[0])) for b, v in BOOKS.items()}
SER = {}
for b, (s, k0, cap) in BOOKS.items():
    end = BOOK_END[b]
    w_ = wins(end)
    for k in sorted({1.0, 1.25, 1.5, k0}):
        for cost, fld in COST.items():
            r_, sc = run(s, fld, k, cap)
            r_ = r_.loc[:end]
            nn, dd = notional(s, sc)
            top = pd.concat([nn, dd], axis=1).max(axis=1).loc[:end]
            capped = ((sc[0] < k - 1e-12) | (sc[1] < k - 1e-12)).loc[:end]
            if k == k0:
                SER[(b, cost)] = r_
                ct = contrib(s, sc, fld)
                assert (sum(ct.values()).loc[r_.index[r_.index >= L[legs_of(s)[0][2]].start]] - r_.loc[L[legs_of(s)[0][2]].start:]).abs().max() < 1e-12
                for y in (2023, 2024, 2025, 2026):
                    x = r_.loc[str(y)]
                    c, m = eqm(x)
                    yrows.append(dict(book=b, k=k, cost=cost, year=y, days=len(x), start=x.index[0].date(), end=x.index[-1].date(), ret_pct=((1 + x).prod() - 1) * 100, cagr_pct=c * 100, mdd_pct=m * 100, sharpe=sharpe(x), worst_day_pct=x.min() * 100))
            for w, (a, e_) in w_.items():
                x = r_.loc[a:e_]
                c, m = eqm(x)
                act = x[x != 0]
                ban = pd.Series(ban_mask(x.index), index=x.index)
                t5 = act.nlargest(5).index
                tw = top.loc[a:e_]
                pk, tr_ = dd_span(x)
                rows.append(dict(book=b, spec=s["name"], k=k, 사전등록k=k == k0, 상한=cap, cost=cost, window=w, start=x.index[0].date(), end=x.index[-1].date(), days=len(x), active=len(act), cagr_pct=c * 100, mdd_pct=m * 100, mdd_peak=pk, mdd_trough=tr_, sharpe=sharpe(x),
                                 worst_day_pct=x.min() * 100, worst_day=x.idxmin().date(), ret_pct=((1 + x).prod() - 1) * 100, 활성일순bp=act.mean() * 1e4, t_nw=hac_t(act.values * 1e4), cagr_ex_top1=eqm(x.drop(act.nlargest(1).index))[0] * 100, mdd_ex_top1=eqm(x.drop(act.nlargest(1).index))[1] * 100,
                                 cagr_ex_top5=eqm(x.drop(t5))[0] * 100, mdd_ex_top5=eqm(x.drop(t5))[1] * 100, top1_day=act.nlargest(1).index[0].date(), top1_pct=act.max() * 100, cagr_ex_ban=eqm(x[~ban.values])[0] * 100, mdd_ex_ban=eqm(x[~ban.values])[1] * 100,
                                 ban_days=int(ban.sum()), notional_mean=float(tw[tw > 0].mean()), notional_max=float(tw.max()), cap_days=int(capped.loc[a:e_].sum()), note="2022.12 전 분봉 다리 없는 부분 장부" if w == "L" else ""))
                if k == k0 and cost in ("기본", "하한"):
                    tot = x.sum()
                    for (m_, sl, lk, wt), y_ in ct.items():
                        z = y_.loc[a:e_]
                        z_a = z[z != 0]
                        crows.append(dict(book=b, k=k, cost=cost, window=w, 시장=m_, 슬롯=sl, 다리=lk, 다리비중=wt, 활성일=len(z_a), 평균bp=z_a.mean() * 1e4 if len(z_a) else np.nan, 합계pct=z.sum() * 100, 기여비중pct=z.sum() / tot * 100 if tot else np.nan, 유효=L[lk].start <= pd.Timestamp(a)))
BR = pd.DataFrame(rows)
BY = pd.DataFrame(yrows)
CT = pd.DataFrame(crows)
BR.to_csv(OUT / "book_results.csv", index=False, encoding="utf-8-sig", float_format="%.4f")
BY.to_csv(OUT / "book_yearly.csv", index=False, encoding="utf-8-sig", float_format="%.4f")
CT.to_csv(OUT / "contributions.csv", index=False, encoding="utf-8-sig", float_format="%.4f")
ENDS.to_csv(OUT / "end_dates.csv", index=False, encoding="utf-8-sig")
pd.DataFrame([dict(leg="장부 " + b, 자료="다리 평가 끝 최솟값", 자료끝=BOOK_END[b].date(), 시차=0, 손익끝=BOOK_END[b].date(), 달력끝=CAL[-1].date(), 다리평가끝=BOOK_END[b].date(), 묶는자료=True) for b in BOOKS]).to_csv(OUT / "end_dates.csv", mode="a", header=False, index=False, encoding="utf-8")

_v = BR[BR.사전등록k].pivot_table(index=["book", "cost"], columns="window", values=["cagr_pct", "mdd_pct"], sort=False)[[("cagr_pct", w) for w in "DSWXL"] + [("mdd_pct", w) for w in "DSWXL"]].round(2)
print(_v.to_string())
print(BY[BY.cost == "하한"].pivot_table(index="book", columns="year", values="ret_pct").round(2).to_string())

# ---------------------------------------------------------------- 하한 스캔 식 민감도(P4 만 영향)
_alt = {}
_r4 = run(BOOKS["P4"][0], "low_scan", BOOKS["P4"][1], BOOKS["P4"][2])[0].loc[:BOOK_END["P4"]]
for w, (a, e_) in wins(BOOK_END["P4"]).items():
    c, m = eqm(_r4.loc[a:e_])
    _alt[w] = (round(c * 100, 2), round(m * 100, 2))
print("P4 하한 스캔 식(보유일마다 1.2bp):", _alt)

# ---------------------------------------------------------------- 최종 선택 규칙
LOW = BR[BR.사전등록k & (BR.cost == "하한")].set_index(["book", "window"])


def test1(b):
    d = {w: (float(LOW.cagr_pct[(b, w)]), float(LOW.mdd_pct[(b, w)])) for w in "WXSL"}
    each = {w: bool(d[w][0] >= 30 and d[w][1] > -15) for w in "WXS"}
    lm = bool(d["L"][1] > -15)
    return dict(하한=d, 창별통과=each, L_MDD통과=lm, 통과=bool(all(each.values()) and lm))


def test2(b):
    return bool(LOW.cagr_pct[(b, "W")] >= 30 and LOW.mdd_pct[(b, "W")] > -15 and LOW.ret_pct[(b, "S")] > 0)


T1 = {b: test1(b) for b in BOOKS}
T2 = {b: test2(b) for b in BOOKS}
p1 = [b for b in BOOKS if T1[b]["통과"]]
p2 = [b for b in BOOKS if T2[b]]
xc = lambda b: float(LOW.cagr_pct[(b, "X")])
if p1:
    SEL = dict(등급="1순위", 장부=max(p1, key=xc))
elif p2:
    SEL = dict(등급="조건부", 장부=max(p2, key=xc))
else:
    SEL = dict(등급="목표 미달", 장부=None)
print("1순위 조건:", {b: T1[b]["통과"] for b in BOOKS}, "2순위 조건:", T2, "선택:", SEL)
_lk = BR[(BR.cost == "하한") & (BR.window == "W")]
KS = dict(셀수=len(_lk), W_CAGR30_MDD15_통과=int(((_lk.cagr_pct >= 30) & (_lk.mdd_pct > -15)).sum()))  # 보고만, 선택 금지
print("k 민감도 하한 W:", KS)

# ---------------------------------------------------------------- 실측 대조


def read_1min(*paths):
    f = pd.concat([pd.read_csv(p, usecols=["date", "time", "open", "high", "low", "close", "contract"]) for p in paths], ignore_index=True)
    f["ts"] = pd.to_datetime(f.date + " " + f.time)
    f["date"] = pd.to_datetime(f.date)
    return f.set_index("ts").sort_index()


def price_at(f1, ts, kind):
    """r15_02 price_at 그대로: 봉 time == ts 값, 봉이 없으면 같은 날 직전 봉 종가."""
    if ts in f1.index:
        return float(f1.loc[ts, kind]), False
    day = f1.loc[ts.normalize():ts]
    assert len(day), ts
    return float(day.close.iloc[-1]), True


fq = read_1min(RAW / "intraday/kis_minute/kq150_full_day/futures_1min.csv")
fk = read_1min(RAW / "intraday/kis_minute/full_day/futures_1min.csv", RAW / "intraday/kis_minute/full_day_ext/futures_1min.csv")
FC = []


def fc_row(check, mkt, a, b, real, proxy, note):
    d = (real - proxy) * 1e4
    FC.append(dict(check=check, market=mkt, start=a.date(), end=b.date(), n=len(d), real_bp=real.mean() * 1e4, proxy_bp=proxy.mean() * 1e4, diff_bp=d.mean(), diff_t=hac_t(d.values), diff_sum_pct=d.sum() / 100, abs_max_bp=d.abs().max(), note=note))


# QC_KQ 숏: r15_02 와 같은 거래별 시각. 선물 봉 time + 1분 = ETF 라벨
tr = pd.read_csv(GIT / "jehyeong/intraday/results/final/KOSDAQ150/trades.csv", encoding="utf-8-sig", parse_dates=["date", "entry_time", "exit_bar_end"])
tr = tr[(tr.direction < 0) & (tr.date >= fq.date.min()) & (tr.date <= LEG_END["QC_KQ"])].set_index("date")
assert tr.index.is_unique and np.allclose(L["QC_KQ"].pos.reindex(tr.index).abs(), tr.leverage) and (L["QC_KQ"].pos.reindex(tr.index) < 0).all()
_q = [(price_at(fq, r.entry_time - pd.Timedelta(minutes=1), "close"), price_at(fq, r.exit_bar_end - pd.Timedelta(minutes=1), "close" if r.exit_reason == "forced_1520" else "open")) for r in tr.itertuples()]
_fin, _fout = pd.Series([a[0] for a, _ in _q], index=tr.index), pd.Series([b[0] for _, b in _q], index=tr.index)
_fb = sum(a[1] + b[1] for a, b in _q)
_g = -tr.leverage * (_fout / _fin - 1)
fc_row("QC_KQ 숏 총수익: 실측 − 대리(series gross)", "KOSDAQ", tr.index[0], tr.index[-1], _g, L["QC_KQ"].gross.reindex(tr.index), f"거래별 선물 1분봉, 직전 봉 대체 {_fb}건")
fc_row("QC_KQ 숏 순수익: 실측 1.8bp − 다리 net(1.8 + 1.06bp)", "KOSDAQ", tr.index[0], tr.index[-1], _g - FUT * tr.leverage, L["QC_KQ"].net.reindex(tr.index), "양수면 다리 대리가 보수적")
# QC_Df 숏: 09:07 라벨 시가 ↔ 선물 09:06 봉 시가, 15:01 라벨 시가 ↔ 선물 15:00 봉 시가
_p = L["QC_Df"].pos.loc[fq.date.min():LEG_END["QC_Df"]]
_d = _p[_p < 0].index
_q = [(price_at(fq, d + pd.Timedelta(hours=9, minutes=6), "open"), price_at(fq, d + pd.Timedelta(hours=15), "open")) for d in _d]
_g = -(pd.Series([b[0] for _, b in _q], index=_d) / pd.Series([a[0] for a, _ in _q], index=_d) - 1)
_fb = sum(a[1] + b[1] for a, b in _q)
fc_row("QC_Df 숏 총수익: 실측 − 대리(229200 1분봉)", "KOSDAQ", _d[0], _d[-1], _g, L["QC_Df"].gross.reindex(_d), f"직전 봉 대체 {_fb}건")
fc_row("QC_Df 숏 순수익: 실측 1.8bp − 다리 net(1.8 + 1.06bp)", "KOSDAQ", _d[0], _d[-1], _g - FUT, L["QC_Df"].net.reindex(_d), "양수면 다리 대리가 보수적")


# X_K2·X_D1: 일봉 시가·종가가 1분봉 첫 시가·끝 종가와 같은지, 다리 일수익이 1분봉 재계산과 같은지
def day_check(key, f1, name, side):
    d_ = pd.read_csv(RAW / f"r15_external/kis_fut_{name}_daily.csv", dtype={"code": str}, parse_dates=["date"])
    fr = d_[d_.front == 1].set_index("date").sort_index()
    p = L[key].pos.dropna()
    p = p[p != 0].loc[f1.date.min():LEG_END[key]]
    g = f1.groupby("date")
    o, c, ct = g.open.first().reindex(p.index), g.close.last().reindex(p.index), g.contract.first().reindex(p.index)
    same = ct == fr.code.reindex(p.index)  # full_day 는 만기일에 만기물을 담아 일봉 front 와 다르다
    assert o[same].notna().all()
    mis = int(((o - fr.open.reindex(p.index)).abs()[same] > 1e-6).sum() + ((c - fr.close.reindex(p.index)).abs()[same] > 1e-6).sum())
    r_ = side * (c / o - 1)[same] - FUT
    fc_row(f"{key} 일수익: 1분봉 재계산 − 다리 net", L[key].mkt, p.index[0], p.index[-1], r_, L[key].net.reindex(r_.index), f"일봉 시가·종가와 1분봉 불일치 {mis}값, 만기일 제외 {int((~same).sum())}일")


day_check("X_K2", fk, "k200", -1)
day_check("X_D1", fq, "kq150", 1)
_c1 = LR[(LR.leg == "C1_RB") & (LR.cost == "기본") & (LR.window == "S")].iloc[0]
_c1s = LR[(LR.leg == "C1_RB") & (LR.cost == "스트레스") & (LR.window == "S")].iloc[0]
FC.append(dict(check="C1_RB 실측 순수익(1.8bp), 등급 estimate", market="KOSDAQ", start=_c1.start, end=_c1.end, n=int(_c1.n), real_bp=_c1.net_bp, proxy_bp=float(_rv.net_bp), diff_bp=_c1.net_bp - float(_rv.net_bp), diff_t=_c1.t_nw, diff_sum_pct=np.nan, abs_max_bp=np.nan,
               note=f"proxy_bp = 설계 NAV 대리 n {int(_rv.n)}, diff_t 칸은 실측 NW t, 스트레스 {_c1s.net_bp:.2f}bp, CAGR {_c1.cagr_pct:.2f}%, MDD {_c1.mdd_pct:.2f}%, {_c1.days}일"))
FCD = pd.DataFrame(FC)
FCD.to_csv(OUT / "futures_check.csv", index=False, encoding="utf-8-sig", float_format="%.4f")
print(FCD.drop(columns=["note"]).round(3).to_string())
print(FCD.note.to_string())

# ---------------------------------------------------------------- 그림
PAL = dict(zip(BOOKS, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]))
INK, MUTED, GRIDC, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
fig, axs = plt.subplots(2, 1, figsize=(10, 6.8), height_ratios=[2.4, 1], sharex=True, facecolor=SURF)
_ends = {}
for b in BOOKS:
    x = SER[(b, "하한")].loc["2022-12-13":]
    e = (1 + x).cumprod()
    axs[0].plot(e.index, e.values, lw=2, color=PAL[b], label=f"{b} k {BOOKS[b][1]:g}")
    axs[1].plot(e.index, (e / e.cummax() - 1).values * 100, lw=1.5, color=PAL[b])
    _ends[b] = e
_y = sorted(_ends, key=lambda b: _ends[b].iloc[-1])
_gap = max(_ends[_y[-1]].iloc[-1] * .05, .12)
_pos = {}
for i, b in enumerate(_y):
    _pos[b] = max(_ends[b].iloc[-1], _pos[_y[i - 1]] + _gap) if i else _ends[b].iloc[-1]
for b in BOOKS:
    x = _ends[b]
    axs[0].annotate(f"{b} {eqm(SER[(b, '하한')].loc['2022-12-13':])[0] * 100:.1f}%", (x.index[-1], _pos[b]), xytext=(6, 0), textcoords="offset points", fontsize=9, color=INK, va="center")
axs[0].axvline(pd.Timestamp("2025-01-02"), color=MUTED, lw=1, ls=":")
axs[0].annotate("봉인 구간 시작", (pd.Timestamp("2025-01-02"), axs[0].get_ylim()[1]), xytext=(4, -12), textcoords="offset points", fontsize=9, color=MUTED)
axs[1].axvline(pd.Timestamp("2025-01-02"), color=MUTED, lw=1, ls=":")
axs[0].set_ylabel("누적(시작 1배)", color=MUTED)
axs[1].set_ylabel("낙폭(%)", color=MUTED)
axs[0].legend(frameon=False, fontsize=9, loc="upper left")
_be = min(BOOK_END.values())
axs[0].set_title(f"하한 비용 장부 누적, 창 W {dot('2022-12-13')}~{dot(_be)}, 끝 라벨 = 창 W CAGR", loc="left", fontsize=11, color=INK)
for ax in axs:
    ax.set_facecolor(SURF)
    ax.grid(axis="y", color=GRIDC, lw=.8)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(colors=MUTED, length=0)
axs[0].margins(x=.09)
fig.tight_layout()
fig.savefig(OUT / "fig_books.png", dpi=140, facecolor=SURF)
plt.close(fig)

# ---------------------------------------------------------------- 판정 기록
_dm = float(INTEG.max_abs_diff.max())
DM = f"2024.12.30 까지 스캔 다리 {INTEG.leg.nunique()}개 × 6필드 최대 차 {_dm:.1e}, series_candidates P1·P2·P3 일수익 최대 차 {max(BOOK_DIFF.values()):.2e}(저장 소수 8자리), 상한 3 재현 일치"
_tail = pd.read_csv(OUT / "tail_sensitivity.csv", encoding="utf-8-sig")  # r16_eval_tail.py 산출(보고만)
J = dict(평가끝={b: dot(e) for b, e in BOOK_END.items()}, 다리평가끝={k: dot(e) for k, e in LEG_END.items()}, 무결성=DM, 다리=VERD, 장부_1순위조건=T1, 장부_2순위조건=T2, 선택=SEL, k민감도_하한W=KS,
         P4_하한_스캔식=_alt, 비용정의="기본·하한·스트레스 = 사전 등록", 공매도금지=[f"{dot(a)}~{dot(b)}" for a, b in BANS], 달력뒤민감도=_tail.to_dict("records"))
(OUT / "judgement.json").write_text(json.dumps(J, ensure_ascii=False, indent=1, default=float), encoding="utf-8")


def fmt(x):
    return f"{x:.2f}".replace("-", "−")


_be = dot(BOOK_END["P1"])
N = ["# 16차 봉인 평가 실행 기록", "",
     "• 코드: r16_eval_legs.py(다리, 스캔 파일 복사 뒤 봉인선만 이동), r16_eval_tail.py(달력 뒤 민감도), r16_eval.py(평가). 이 순서로 PYTHONUTF8=1 python 실행. 캐시는 스크래치 r16_eval_legs.pkl.",
     f"• 평가 끝: 장부 P1·P2·P3·P4 {_be}(beomgu_C3VT30 series 끝이 묶음). 달력 끝 {dot(CAL[-1])}(069500 수정가 일봉). 다리별 표 end_dates.csv.",
     "• X_D1·X_MRq·QC_Df·C1_RB 는 입력이 달력 뒤까지 있어도 달력 끝에서 멈춘다. M_zfQ 는 SPX 끝 때문에 2026.09.23 까지. KB_E1s3 는 series 끝 2026.09.22.",
     f"• 무결성: {DM}.", "",
     "## 봉인선 이동 외 코드 차이(규칙 변경 없음)", "",
     "• 첫 1분 평균 차감값을 설계 구간(≤ 2024.12.30) C3 날로 고정(KB_E1 069500 +0.12185%, 122630 +0.17706%, n 48). 스캔 값과 같음을 단언.",
     "• 낮 숏 보정 1.06bp 고정. z·회귀·국면은 인과 롤링 그대로. 틱은 그해 평균 원가격, 2025·2026 은 그해 값(2026 은 2026.09.29 까지 평균).",
     "• 분봉 첫·끝 라벨 단언: 2025.01.02·2025.11.13·2026.01.02 는 파일에 09:01~10:00 봉이 없고(수능일은 16:30 까지) 스캔 구간 4일과 형식이 달라 이 3일만 허용. 세 날 모두 SPECIAL 로 분봉 다리에서 제외됨을 단언.",
     "• M_zfQ 의 post 창은 me.minute 이 위 단언에 걸려 Vo 09:01 > 0 으로 같은 날을 고른다. 설계 n 38, +73.66bp, t 2.58 재현 단언 통과.",
     "• 스윙 두 다리의 하한: 스캔 식은 보유일마다 1.2bp 를 빼서 사전 등록 왕복 3bp 와 다르다. P4 만 영향. 회전 기준(= 스트레스 흐름)을 쓰고 스캔 식 값은 judgement.json P4_하한_스캔식 에 남겼다.",
     "• 스윙 거래 집계: X_MRq 는 진입일, X_FLzVT 는 T+1 체결일 배정. 봉인선을 넘은 FLzVT 거래(결정 2024.12.30)는 설계 재현에서 끝까지 간 값이라 순 33.51 대 스캔 33.21, 일수익 차는 0.",
     "• 공매도 금지 구간: 2020.03.16~2021.05.02, 2023.11.06~2025.03.30(스캔 표의 2024.12.30 절단은 봉인 때문이므로 쓰지 않음).", "",
     "## 설계 선별 기준", "",
     "• 순수익 > 0, NW t ≥ 2, 연도 부호 ≥ 70%(거래 5건 이상 해), 상위 5건 < 50%, 전·후반 양수, 스트레스 뒤 양수. 인접 문턱은 스캔 판정 승계. 창: 기존 다리 중 장기 있는 KB_E1s3·QA_F 는 L(≤ 2024.12.30), KC_KP·QC_KQ 는 D, 새 다리는 사전 등록 표기 창. C1_RB 는 NAV 대리 스캔 셀 F4g_0787.", ""]
N += ["## 판정 요약", ""]
for k, v in VERD.items():
    N.append(f"• {k}: 설계 {'통과' if v['설계선별']['통과'] else '미달(' + ','.join(v['설계선별']['미달']) + ')'}, S n {v['S_n']}, 순 {fmt(v['S순bp'])}bp, NW t {fmt(v['S_t'])} → {v['판정']}")
N.append("")
for b in BOOKS:
    t = T1[b]
    N.append(f"• {b} k {BOOKS[b][1]:g}: 하한 " + ", ".join(f"{w} {fmt(t['하한'][w][0])}%/{fmt(t['하한'][w][1])}%" for w in "WXSL") + f" → 1순위 {'통과' if t['통과'] else '미달'}, 2순위 {'통과' if T2[b] else '미달'}")
_pk = BR[BR.사전등록k & (BR.cost == "하한") & (BR.window == "W")].set_index("book")
_ct = CT[(CT.cost == "기본") & (CT.window == "S")].set_index(["book", "다리"]).합계pct
_fc = FCD.set_index("check")
N += [f"• 선택: {SEL['등급']} {SEL['장부'] or ''}",
      f"• k 민감도(보고만): 하한 창 W {KS['셀수']}셀 중 CAGR ≥ 30 ∧ MDD < 15 통과 {KS['W_CAGR30_MDD15_통과']}셀.",
      "• 낙폭 구간(하한, 창 W): " + ", ".join(f"{b} {fmt(_pk.mdd_pct[b])}%({dot(_pk.mdd_peak[b])}→{dot(_pk.mdd_trough[b])})" for b in BOOKS) + ".",
      f"• P3 창 S 다리 합계(기본, 장부 비중 반영 %): " + ", ".join(f"{k} {fmt(v)}" for k, v in _ct["P3"].items()) + ". KA_STf 는 2026.08.19·08.25·08.31 밤 갭 손실이 크다.",
      "", "## 실측 대조", ""]
N += [f"• {r.check}: {dot(r.start)}~{dot(r.end)}, n {r.n}, 실측 {fmt(r.real_bp)}bp, 대리 {fmt(r.proxy_bp)}bp, 차 {fmt(r.diff_bp)}bp" + ("" if pd.isna(r.diff_t) else f", t {fmt(r.diff_t)}") + f". {r.note}" for r in FCD.itertuples()]
N += ["", "## 달력 뒤 민감도(보고만, 판정 불변)", ""]
N += [f"• {r.leg}: 마지막 거래·자료 {dot(r.끝)}, n {r.n}, 순 {fmt(r.net_bp)}bp, NW t {fmt(r.t_nw)}" for r in _tail.itertuples()]
N += ["", "## 한계", "",
      "• 기존 다리·스윙 일부는 S 가 깨끗한 표본 밖이 아니다(사전 등록 봉인 노출 고지).",
      "• 2026 표본에 일중 −14%·+19% 수준의 코스피200 선물 일봉, 서킷 추정 공백(2026.03.04·06.08·07.28·07.29)이 있다. 아침 손절은 일봉 저가 기준이라 그날 체결 지연을 반영하지 않는다.",
      "• 장부 상한 3 에서 X 슬롯(낮 다리)은 스캔 코드대로 밤 배수를 탄다. X 명목은 밤·낮 합산에 모두 잡혀 상한이 낮 다리 기준보다 일찍 걸릴 수 있다.",
      "• X_FLzVT 설계 미달은 거래 단위 상위 5건 54.3%(스캔 표기 기준 초과). 일 단위로는 18.7% 지만 S t 0.40 이라 판정은 같다.",
      "• C1_RB 설계 판정은 선물 실측이 아닌 NAV 대리 셀이다. 실측은 230일, estimate 등급."]
(OUT / "notes.md").write_text(re.sub(r"(?<![\w.~])-(?=\d)", "−", "\n".join(N)) + "\n", encoding="utf-8")

# StructuredOutput 재료(스크래치)
_S = dict(end_date=_be, design_match=DM,
          legs=[dict(key=k, window=f"{w} {dot(r.start)}~{dot(r.end)}", n=int(r.n), net_bp=round(float(r.net_bp), 2), t_nw=round(float(r.t_nw), 2), cagr_pct=round(float(r.cagr_pct), 2), mdd_pct=round(float(r.mdd_pct), 2),
                    stress_net_bp=round(float(LR[(LR.leg == k) & (LR.window == w) & (LR.cost == "스트레스")].net_bp.iloc[0]), 2), verdict=VERD[k]["판정"]) for k in LEGS for w in "DS" for r in LR[(LR.leg == k) & (LR.window == w) & (LR.cost == "기본")].itertuples()],
          books=[dict(book=r.book, k=r.k, window=f"{r.window} {dot(r.start)}~{dot(r.end)}", cost=r.cost, cagr_pct=round(float(r.cagr_pct), 2), mdd_pct=round(float(r.mdd_pct), 2), sharpe=round(float(r.sharpe), 2), worst_day_pct=round(float(r.worst_day_pct), 2)) for r in BR[BR.사전등록k].itertuples()],
          selection=SEL, ks=KS, alt=_alt)
(SCRATCH / "r16_eval_structured.json").write_text(json.dumps(_S, ensure_ascii=False, indent=0, default=float), encoding="utf-8")
print("완료")
