"""verify_r15_v2_review.py: 15차 보정판(r15_final_book_v2.py) 반박 검토용 독립 재계산.

v2 코드를 import 하지 않고 13차 series·원자료에서 여섯 다리를 다시 짜 공유 수치를 재현한 뒤
(1) 코스닥 롱 밤 선물 대 233740 차이 분해·신호 시각 (2) 아침 혼합 (3) 단일가 다리 1틱·09:01 실행형
(4) 낮 다리 틱 (5) k 경계·선물 3bp (6) 코드 점검을 잰다. 읽기만 함. 산출은 final_v2/verify/ 의 새 CSV 만.
실행: PYTHONUTF8=1 python "<code>/topics/verify_r15_v2_review.py"
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)

LAB = Path("<OUT>")
R13, R15 = LAB / "round13_slot_book", LAB / "round15_alpha_share"
RAW = Path("<raw>")
MIN = Path("<repo>/jehyeong/intraday/data/minute")
KQMIN = RAW / "intraday/kis_minute/kq150_full_day/futures_1min.csv"
K2MIN = RAW / "intraday/kis_minute/full_day/futures_1min.csv"
OUT = R15 / "final_v2" / "verify"
OUT.mkdir(parents=True, exist_ok=True)

FUT, KQ_ADJ = 1.8e-4, 1.06e-4
WIN = {"W_all": ("2022-12-13", "2026-09-17"), "W_ex2026": ("2022-12-13", "2025-12-30"),
       "Y2024": ("2024-01-02", "2024-12-30"), "Y2025": ("2025-01-02", "2025-12-30"), "Y2026": ("2026-01-02", "2026-09-17")}
W_LONG = ("2016-10-04", "2026-09-17")
FUT0845 = "2023-07-31"                     # 이날부터 코스닥150·코스피200 선물 08:45 개장(5차 보고서), 그 전은 09:00
CAL = pd.DatetimeIndex(pd.read_csv(RAW / "etf/069500.csv", parse_dates=["date"])["date"])
assert CAL.is_monotonic_increasing and CAL.is_unique
PREV = pd.Series(CAL[:-1], index=CAL[1:])
ROWS = []


def rec(sec, item, value, note=""):
    ROWS.append(dict(section=sec, item=item, value=value, note=note))
    print(f"[{sec}] {item}: {value}" + (f"  ({note})" if note else ""))


def ser(h):
    s = pd.read_csv(R13 / "series" / f"{h}.csv", parse_dates=["date"]).set_index("date")
    m = json.loads((R13 / "series" / f"{h}.meta.json").read_text(encoding="utf-8"))
    return s.loc[m["first_valid"]:m["last_valid"]].fillna(0.0)


def etf(code):
    p = RAW / "etf" / f"{code}.csv"
    if not p.exists():
        p = RAW / "etf_extra" / f"{code}.csv"
    return pd.read_csv(p, parse_dates=["date"]).set_index("date").sort_index()


def metrics(r):
    nan = int(r.isna().sum())
    r = r.dropna()
    eq = (1 + r).cumprod()
    dd = eq / eq.cummax() - 1
    dd0 = eq / np.maximum(eq.cummax(), 1.0) - 1                      # 시작 자본 1 을 고점에 넣은 MDD
    cum = float(eq.iloc[-1])
    t = dd.idxmin()
    return dict(n=len(r), nan=nan, cagr=cum ** (252 / len(r)) - 1, mdd=float(dd.min()), mdd0=float(dd0.min()), cum=cum - 1,
                worst=float(r.min()), worst_date=str(r.idxmin().date()), trough=str(t.date()), peak=str(eq.loc[:t].idxmax().date()))


def bp(x):
    return f"{x * 1e4:+.1f}"


# ======================================================================== 0. 자료
S = {h: ser(h) for h in ("jehyeong_ON", "beomgu_C3VT30", "jehyeong_KP", "blog_ON_Q", "beomgu_C3Q", "jehyeong_KQ")}
book = pd.read_csv(R13 / "book/book_daily.csv", parse_dates=["date"]).set_index("date")
A_W = book.index
rec("0_달력", "book_daily 달력 = 069500 달력(W_all 구간)", bool(A_W.equals(CAL[(CAL >= WIN["W_all"][0]) & (CAL <= WIN["W_all"][1])])), f"{len(A_W)}일")
PX = {c: etf(c) for c in ("069500", "122630", "229200", "233740")}
OC = {c: PX[c].close / PX[c].open - 1 for c in PX}
ON = {c: PX[c].open / PX[c].close.shift(1) - 1 for c in PX}          # 날짜 = 청산일(T 종가 → d 시가)
TKY = {c: 5 / PX[c].close.groupby(PX[c].index.year).mean() for c in PX}


def tk(c, idx):
    idx = pd.DatetimeIndex(idx)
    return pd.Series(TKY[c].reindex(idx.year).values, index=idx)


IX = {mk: etf(f) for mk, f in (("K", "K200_index_kis"), ("Q", "KQ150_index_kis"))}
IOC = {mk: (IX[mk].close / IX[mk].open - 1).where(IX[mk].open != IX[mk].close) for mk in IX}
for c, e in PX.items():
    assert np.isfinite(OC[c]).all(), c

# 코스닥150 선물(일봉)
fut = pd.read_csv(RAW / "r15_external/kis_fut_kq150_daily.csv", parse_dates=["date"])
fut = fut[fut.date < "2026-10-30"]
FPX = fut.set_index(["date", "code"]).sort_index()
assert FPX.index.is_unique
FRONT = fut[fut.front == 1].set_index("date").code
assert FRONT.index.is_unique
q = S["blog_ON_Q"]
qa_on = q.index[q.pos != 0]
NT = pd.DataFrame({"pos": q.pos[qa_on].values, "T": PREV.reindex(qa_on).values, "code": FRONT.reindex(qa_on).values}, index=qa_on)
NT["cT"] = FPX.close.reindex(pd.MultiIndex.from_arrays([NT["T"], NT.code])).values
NT["oD"] = FPX.open.reindex(pd.MultiIndex.from_arrays([NT.index, NT.code])).values
NT["vT"] = FPX.volume.reindex(pd.MultiIndex.from_arrays([NT["T"], NT.code])).values
NT["fr"] = NT.oD / NT.cT - 1
NT["front_T"] = FRONT.reindex(NT["T"]).values
_rf = NT.code != NT.front_T                                          # 롤 밤(d = 만기일): T 의 front(만기물)로 잡은 대안 수익
NT["fr_frontT"] = NT.fr.where(~_rf, FPX.open.reindex(pd.MultiIndex.from_arrays([NT.index, NT.front_T])).values /
                              FPX.close.reindex(pd.MultiIndex.from_arrays([NT["T"], NT.front_T])).values - 1)

# ======================================================================== 1. 독립 재조립(v2 사양)
C3 = {"K_B": ("beomgu_C3VT30", "069500", "122630", "K"), "Q_B": ("beomgu_C3Q", "229200", "233740", "Q")}


def mix_ret(x, o1, o2, oi):
    x = np.asarray(x, float)
    return np.where(x <= 1, x * o1, np.where(x <= 2, (x - 1) * o2 + (2 - x) * o1, o2 + (x - 2) * oi))


def mix_tick(x, t1, t2):
    xe = np.minimum(np.asarray(x, float), 2.0)                       # ETF 몫(명목 ≤ 2)만, 초과 몫은 선물
    return np.where(xe <= 1, xe * t1, (xe - 1) * t2 + (2 - xe) * t1)


BASE = dict(fut=FUT, adj=KQ_ADJ, day_tick=True, auc_tick=False, c3="cap", c3oc=None, c3proxy=None, qa_long="fut", qa_etf_dates=None, qa_roll_frontT=False, ka="scaled")
OC_K2F = None                                                        # 코스피 밤 'mix' 모드용 K200 선물 밤 수익(아래에서 채움)


def c3_leg(name, k, p):
    h, c1, c2, mk = C3[name]
    s = S[h]
    idx = s.index
    on = (s.pos != 0).values
    if p["c3"] == "scaled":
        return k * s.net
    o1, o2, oi = OC[c1].reindex(idx), OC[c2].reindex(idx), IOC[mk].reindex(idx).fillna(OC[c1].reindex(idx))
    if p["c3oc"] is not None:                                        # 09:01 실행형: 분봉 있는 날만 덮어씀
        a1, a2 = p["c3oc"][name]
        a1, a2 = a1.reindex(idx), a2.reindex(idx)
        ok = a1.notna() & a2.notna()
        o1, o2, oi = o1.where(~ok, a1), o2.where(~ok, a2), oi.where(~ok, a1)
        if p["c3proxy"] is not None:                                 # 분봉 없는 날(2022-11 전): W_all 실측 첫 1분 평균 변화만큼 시가를 올려 근사(지수 몫은 그대로)
            m1, m2 = p["c3proxy"][name]
            o1 = o1.where(ok, (1 + o1) / (1 + m1) - 1)
            o2 = o2.where(ok, (1 + o2) / (1 + m2) - 1)
    x = k * s.pos.abs().values
    g = mix_ret(x, o1.values, o2.values, oi.values)
    cost = 5e-4 * np.minimum(x, 1) + p["fut"] * np.maximum(x - 2, 0)
    if p["auc_tick"]:
        cost = cost + mix_tick(x, tk(c1, idx).values, tk(c2, idx).values)
    return pd.Series(np.where(on, g - cost, 0.0), index=idx)


def ka_leg(k, p):
    s = S["jehyeong_ON"]
    on = (s.pos != 0).values
    if p["ka"] == "scaled":
        r = k * s.net
    else:                                                            # 실행 일치형: 명목 k 를 069500/122630(≤2)+K200 선물(초과)로
        x = k * s.pos.abs().values
        o1, o2 = ON["069500"].reindex(s.index).values, ON["122630"].reindex(s.index).values
        of = OC_K2F.reindex(s.index).fillna(ON["069500"].reindex(s.index)).values
        g = mix_ret(x, o1, o2, of)
        r = pd.Series(np.where(on, g - 5e-4 * k * s.pos.abs().values, 0.0), index=s.index)   # 비용은 사양(k×5bp) 그대로
    if p["auc_tick"]:
        r = r - np.where(on, mix_tick(k * s.pos.abs().values, tk("069500", s.index).values, tk("122630", s.index).values), 0.0)
    return r


KP = S["jehyeong_KP"]
KP_TICK = pd.Series(np.where(KP.pos != 0, mix_tick(KP.pos.abs().values, tk("069500", KP.index).values, tk("122630", KP.index).values), 0.0), index=KP.index)
KQ = S["jehyeong_KQ"]
SH = KQ.pos < 0
KQ_TICK = pd.Series(np.where(KQ.pos > 0, KQ.pos.values * tk("229200", KQ.index).values, 0.0), index=KQ.index)


def kc_leg(k, p):
    return k * (KP.net - (KP_TICK if p["day_tick"] else 0.0))


def qc_leg(k, p):
    net = KQ.net.copy()
    net[SH] = KQ.gross[SH] - (p["fut"] + p["adj"]) * KQ.pos[SH].abs()
    return k * (net - (KQ_TICK if p["day_tick"] else 0.0))


def qa_leg(k, p):
    net = q.net.copy()
    rf = p["qa_roll_frontT"]
    if rf is True:
        frr = NT.fr_frontT
    elif rf:                                                         # 날짜 목록: 그 롤 밤만 만기물
        frr = NT.fr.where(~NT.index.isin(pd.DatetimeIndex(rf)), NT.fr_frontT)
    else:
        frr = NT.fr
    f = NT.pos * frr - p["fut"] * NT.pos.abs()
    use = NT.fr.notna() & ((NT.pos < 0) | (p["qa_long"] == "fut"))
    if p["qa_etf_dates"] is not None:                                # 지정한 롱 밤만 233740(13차) 경로로 되돌림
        use = use & ~(NT.index.isin(p["qa_etf_dates"]) & (NT.pos > 0))
    net.loc[NT.index[use]] = f[use].values
    return k * net


def legs(k, **kw):
    p = {**BASE, **kw}
    return {"K_A": ka_leg(k, p), "K_B": c3_leg("K_B", k, p), "K_C": kc_leg(k, p),
            "Q_A": qa_leg(k, p), "Q_B": c3_leg("Q_B", k, p), "Q_C": qc_leg(k, p)}


def assemble(L):
    g = {n: L[n].reindex(A_W) for n in L}
    nanc = {n: int(v.isna().sum()) for n, v in g.items()}
    g = {n: v.fillna(0.0) for n, v in g.items()}
    kb = S["beomgu_C3VT30"].pos.reindex(A_W).fillna(0) != 0
    qb = S["beomgu_C3Q"].pos.reindex(A_W).fillna(0) != 0
    g["K_C"] = g["K_C"].where(~kb, 0.0)
    g["Q_C"] = g["Q_C"].where(~qb, 0.0)
    wall = 0.5 * sum(g.values())
    parts = [L[n].loc[W_LONG[0]:W_LONG[1]] for n in ("K_B", "Q_A", "Q_B")]
    li = parts[0].index.intersection(parts[1].index).intersection(parts[2].index)
    wl = 0.5 * sum(pp.reindex(li) for pp in parts)
    return wall, wl, nanc, g


def summ(wall, wl):
    out = {}
    for w, (a, z) in WIN.items():
        m = metrics(wall.loc[a:z])
        out[f"{w}_cagr"], out[f"{w}_mdd"], out[f"{w}_ret"] = m["cagr"], m["mdd"], m["cum"]
    m = metrics(wall)
    out.update(W_all_mdd0=m["mdd0"], W_all_trough=m["trough"], W_all_worst=m["worst"], W_all_worst_date=m["worst_date"])
    ml = metrics(wl)
    out.update(W_long_cagr=ml["cagr"], W_long_mdd=ml["mdd"], W_long_mdd0=ml["mdd0"], W_long_n=ml["n"], W_long_nan=ml["nan"],
               W_long_peak=ml["peak"], W_long_trough=ml["trough"], W_long_worst=ml["worst"], W_long_worst_date=ml["worst_date"])
    return out


def run(k, **kw):
    wall, wl, _, _ = assemble(legs(k, **kw))
    d = summ(wall, wl)
    d["k"] = k
    return d


KS = np.round(np.arange(1.0, 3.0001, 0.05), 2)


def grid(**kw):
    G = pd.DataFrame([run(float(k), **kw) for k in KS])
    kr = G[(G.W_all_mdd >= -0.15) & (G.W_long_mdd >= -0.15)].k.max()
    km = G[(G.W_all_mdd >= -0.20) & (G.W_long_mdd >= -0.20)].k.max()
    return G, float(kr), float(km)


CW = ("W_all", "W_ex2026", "Y2024", "Y2025", "Y2026", "W_long")


def judge(d):
    """창별 기준(CAGR ≥ 30% ∧ MDD ≥ −20%). 연도 창은 그 해 누적 수익을 쓴다."""
    out = {}
    for w in CW:
        c = d[f"{w}_ret"] if w.startswith("Y") else d[f"{w}_cagr"]
        out[w] = bool(c >= 0.30 and d[f"{w}_mdd"] >= -0.20)
    return out


def say(d):
    return (f"W_all {d['W_all_cagr']:.1%}/{d['W_all_mdd']:.1%}, ex2026 {d['W_ex2026_cagr']:.1%}/{d['W_ex2026_mdd']:.1%}, "
            f"2024 {d['Y2024_ret']:+.1%} 2025 {d['Y2025_ret']:+.1%} 2026 {d['Y2026_ret']:+.1%}, "
            f"W_long {d['W_long_cagr']:.1%}/{d['W_long_mdd']:.2%}")


def jstr(j):
    return " ".join(f"{w}:{'O' if v else 'X'}" for w, v in j.items())


# --- v2 산출과 대조
V2 = pd.read_csv(R15 / "final_v2/k_grid_v2.csv")
G0, KR0, KM0 = grid()
cmpc = ["W_all_cagr", "W_all_mdd", "W_ex2026_cagr", "W_ex2026_mdd", "Y2024_ret", "Y2025_ret", "Y2026_ret", "W_long_cagr", "W_long_mdd"]
for cfg, kw in (("v2", {}), ("v2_notick", dict(day_tick=False)), ("v2_c3scaled", dict(c3="scaled"))):
    Gm = G0 if cfg == "v2" else pd.DataFrame([run(float(k), **kw) for k in KS])
    Vv = V2[V2.config == cfg].sort_values("k").reset_index(drop=True)
    dmax = max(float((Gm[c].values - Vv[c].values).__abs__().max()) for c in cmpc)
    rec("1_재현", f"{cfg} 41개 k 전 지표 최대 차(내 재조립 − k_grid_v2)", f"{dmax:.2e}")
rec("1_재현", "v2 k_rec / k_max(내 재조립)", f"{KR0} / {KM0}")
BASEROW = {}
for kk in (1.0, KR0, KM0):
    d = G0[np.isclose(G0.k, kk)].iloc[0].to_dict()
    BASEROW[kk] = d
    rec("1_재현", f"v2 k {kk}", say(d), f"판정 {jstr(judge(d))}; W_long 최악일 {d['W_long_worst']:.1%}({d['W_long_worst_date']}), 고점 {d['W_long_peak']} → 저점 {d['W_long_trough']}, W_all 저점 {d['W_all_trough']}")
mono = bool((G0.W_all_mdd.diff().dropna() <= 1e-12).all() and (G0.W_long_mdd.diff().dropna() <= 1e-12).all())
rec("1_재현", "k 에 대해 MDD 단조 감소(W_all·W_long)", mono)
for kk, lim in ((KR0, -0.15), (KM0, -0.20)):
    a = G0[np.isclose(G0.k, kk)].iloc[0]
    b = G0[np.isclose(G0.k, round(kk + 0.05, 2))].iloc[0]
    rec("5_경계", f"k {kk} 한도 {lim:.0%}: W_long MDD 여유 / 다음 격자 k {round(kk + 0.05, 2)} W_long MDD",
        f"{(a.W_long_mdd - lim) * 100:+.2f}%p / {b.W_long_mdd:.2%}", f"W_all MDD {a.W_all_mdd:.2%}(여유 {(a.W_all_mdd - lim) * 100:+.2f}%p)")

# ======================================================================== 2. 코스닥 롱 밤: 선물 대 233740 분해
lg = NT[NT.pos > 0].copy()
e233 = PX["233740"]
lg["e"] = e233.open.reindex(lg.index).values / e233.close.reindex(lg["T"]).values - 1
rec("2_롱밤", "13차 롱 밤 gross − 233740(T 종가→d 시가) 최대 차(bp)", f"{(q.gross.reindex(lg.index) - lg.e).abs().max() * 1e4:.3f}", f"n {len(lg)}, 선물 가격 결측 {int(lg.fr.isna().sum())}")
lg["f"] = 2 * lg.fr
ixq = IX["Q"]
lg["iT"] = ixq.close.reindex(lg["T"]).values
lg["iO"] = ixq.open.reindex(lg.index).values
lg["iflat"] = (ixq.open == ixq.close).reindex(lg.index).fillna(True).values
lg["i2"] = (2 * (lg.iO / lg.iT - 1)).where(~lg.iflat)
nav = pd.read_csv(RAW / "r15_external/kis_etf_nav_daily.csv", parse_dates=["date"], dtype={"code": str})
nav["code"] = nav.code.str.zfill(6)
nav233 = nav[(nav.code == "233740") & (nav.nav > 0)].set_index("date").nav
lg["premT"] = e233.close.reindex(lg["T"]).values / nav233.reindex(lg["T"]).values - 1
lg["b_close"] = lg.cT / lg.iT - 1                                    # 선물 15:45 종가 / 지수 15:30 종가 − 1
lg["b_open"] = lg.oD / lg.iO - 1                                     # 선물 시가(08:45, 2023-07-31 전 09:00) / 지수 09:00 시가 − 1
lg["net_f"] = lg.f - 2 * FUT
lg["net_e"] = lg.e - 5e-4
lg["adv"] = lg.net_f - lg.net_e
lg["etf_loss"] = lg.i2 - lg.e                                        # 지수 대비 ETF 밤 손실(종가 괴리 되돌림 + 시가 괴리)
lg["fut_gain"] = lg.f - lg.i2                                        # 지수 대비 선물 밤 이득(베이시스 변화·창 차이)
lg["prem_open"] = (1 + lg.e) * (1 + lg.premT) / (1 + lg.i2) - 1      # d 09:00 시가의 공정가 대비 괴리(NAV_T×(1+2×지수 밤) 기준)
rec("2_롱밤", "롱 밤 평균 net: 선물 / 233740 / 차(bp)", f"{bp(lg.net_f.mean())} / {bp(lg.net_e.mean())} / {bp(lg.adv.mean())}", f"n {len(lg)} (v2 출력 +29.4 / +16.3)")
ok = lg.i2.notna()
rec("2_롱밤", "분해(지수 시가 유효 밤): 차 = 비용 차 + ETF 손실(지수 − ETF) + 선물 이득(선물 − 지수), bp",
    f"{bp(lg.adv[ok].mean())} = +1.4 + {bp(lg.etf_loss[ok].mean())} + {bp(lg.fut_gain[ok].mean())}", f"n {int(ok.sum())}, 지수 시가 평탄·결측 제외 {int((~ok).sum())}")
okp = ok & lg.premT.notna()
rec("2_롱밤", "ETF 손실 분해: T 종가 괴리(종가/NAV − 1) / d 시가 괴리(공정가 대비) 평균 bp", f"{bp(lg.premT[okp].mean())} / {bp(lg.prem_open[okp].mean())}",
    f"ETF 손실 ≈ 종가 괴리 − 시가 괴리 = {bp((lg.premT - lg.prem_open)[okp].mean())}; n {int(okp.sum())}")
rec("2_롱밤", "선물 이득 분해: d 시가 베이시스(선물 시가/지수 시가 − 1) / T 종가 베이시스(선물 15:45/지수 15:30 − 1) 평균 bp",
    f"{bp(lg.b_open[ok].mean())} / {bp(lg.b_close[ok].mean())}", f"선물 이득 ≈ 2 × (시가 − 종가 베이시스) = {bp((2 * (lg.b_open - lg.b_close))[ok].mean())}")
# 조건 없는 밤(모든 거래일 밤)과 인버스 밤: 롱 밤에만 있는 현상인지
allN = pd.DataFrame(index=CAL[(CAL >= "2016-08-10") & (CAL <= "2026-09-29")])
allN["T"] = PREV.reindex(allN.index).values
allN["code"] = FRONT.reindex(allN.index).values
allN["cT"] = FPX.close.reindex(pd.MultiIndex.from_arrays([allN["T"], allN.code])).values
allN["oD"] = FPX.open.reindex(pd.MultiIndex.from_arrays([allN.index, allN.code])).values
allN["f"] = 2 * (allN.oD / allN.cT - 1)
allN["e"] = e233.open.reindex(allN.index).values / e233.close.reindex(allN["T"]).values - 1
allN["iT"] = ixq.close.reindex(allN["T"]).values
allN["iO"] = ixq.open.reindex(allN.index).values
allN["i2"] = (2 * (allN.iO / allN.iT - 1)).where((ixq.open != ixq.close).reindex(allN.index).fillna(False).values)
allN["premT"] = e233.close.reindex(allN["T"]).values / nav233.reindex(allN["T"]).values - 1
allN["b_close"] = allN.cT / allN.iT - 1
allN["b_open"] = allN.oD / allN.iO - 1
allN["kind"] = "무신호"
allN.loc[allN.index.isin(lg.index), "kind"] = "롱"
allN.loc[allN.index.isin(NT.index[NT.pos < 0]), "kind"] = "인버스"
allN = allN.dropna(subset=["f", "e", "i2"])
for kind, g in allN.groupby("kind"):
    rec("2_롱밤", f"[{kind} 밤] gross 차(선물 − 233740) / ETF 손실 / 선물 이득 / T 종가 괴리 / T 종가 베이시스 / d 시가 베이시스 평균 bp",
        f"{bp((g.f - g.e).mean())} / {bp((g.i2 - g.e).mean())} / {bp((g.f - g.i2).mean())} / {bp(g.premT.mean())} / {bp(g.b_close.mean())} / {bp(g.b_open.mean())}", f"n {len(g)}")
for per, g in (("선물 09:00 개장기(~2023-07-28)", lg[lg.index < FUT0845]), ("선물 08:45 개장기(2023-07-31~)", lg[lg.index >= FUT0845])):
    gg = g[g.i2.notna()]
    rec("2_롱밤", f"{per} 롱 밤 차 / ETF 손실 / 선물 이득(bp)", f"{bp(g.adv.mean())} / {bp(gg.etf_loss.mean())} / {bp(gg.fut_gain.mean())}", f"n {len(g)}")
yr = lg.groupby(lg.index.year).agg(n=("adv", "size"), adv=("adv", "mean"), etf_loss=("etf_loss", "mean"), fut_gain=("fut_gain", "mean"))
rec("2_롱밤", "연도별 롱 밤 n / 차 / ETF 손실 / 선물 이득(bp)", "; ".join(f"{y}:{int(r.n)}/{r.adv * 1e4:+.0f}/{r.etf_loss * 1e4:+.0f}/{r.fut_gain * 1e4:+.0f}" for y, r in yr.iterrows()))
for wn, (a, z) in (("W_all", WIN["W_all"]), ("W_long", W_LONG)):
    g = lg.loc[a:z]
    rec("2_롱밤", f"{wn} 롱 밤 n / 선물 net / 233740 net / 차(bp)", f"{len(g)} / {bp(g.net_f.mean())} / {bp(g.net_e.mean())} / {bp(g.adv.mean())}",
        f"차의 t {g.adv.mean() / (g.adv.std(ddof=1) / np.sqrt(len(g))):.2f}, 차 중앙값 {bp(g.adv.median())}")
# 자료 이상
d_fi = (lg.f - lg.i2).abs()
rec("2_롱밤", "롱 밤 |선물 − 지수 2배| > 2%p 밤 / 선물 시가 = T 종가 밤 / T 거래량 < 1000 밤", f"{int((d_fi > 0.02).sum())} / {int((lg.oD == lg.cT).sum())} / {int((lg.vT < 1000).sum())}",
    f"상위 |차| 5밤: {[(str(i.date()), round(r.adv * 1e4)) for i, r in lg.reindex(lg.adv.abs().sort_values(ascending=False).index[:5]).iterrows()]}")
tr = lg.adv.copy()
lo, hi = tr.quantile(0.01), tr.quantile(0.99)
rec("2_롱밤", "롱 밤 차 1%·99% 절사 평균 / 상하위 1% 제외 평균(bp)", f"{bp(tr.clip(lo, hi).mean())} / {bp(tr[(tr > lo) & (tr < hi)].mean())}")
roll = lg[lg.code != lg.front_T].copy()
roll["cT_front"] = FPX.close.reindex(pd.MultiIndex.from_arrays([roll["T"], roll.front_T])).values
roll["oD_front"] = FPX.open.reindex(pd.MultiIndex.from_arrays([roll.index, roll.front_T])).values
roll["f_front"] = 2 * (roll.oD_front / roll.cT_front - 1)
rec("2_롱밤", "롱 밤 중 롤(d = 만기일, 계약 = 차근월) 밤 / 그 밤 차 평균(bp) / 같은 밤 만기물로 잡았을 때 선물 gross − 차근월 gross 평균(bp)",
    f"{len(roll)} / {bp(roll.adv.mean())} / {bp((roll.f_front - roll.f).mean())}",
    f"롤 밤 T 거래량 최소 {int(roll.vT.min())}; 롤 밤이 전체 차에 보탠 몫 {roll.adv.sum() / len(lg) * 1e4:+.1f}bp/밤")
# 모든 만기 전날 밤(신호 무관): 차근월물 밤 수익이 지수·만기물과 어긋나는지 → T 차근월 종가 품질
fr_s = FRONT.sort_index()
swd = fr_s.index[1:][fr_s.values[1:] != fr_s.values[:-1]]
EX = pd.DataFrame({"T": PREV.reindex(swd).values, "nxt": fr_s.reindex(swd).values}, index=swd)
EX["old"] = fr_s.reindex(EX["T"]).values
for c in ("nxt", "old"):
    EX["r_" + c] = 2 * (FPX.open.reindex(pd.MultiIndex.from_arrays([EX.index, EX[c]])).values /
                        FPX.close.reindex(pd.MultiIndex.from_arrays([EX["T"], EX[c]])).values - 1)
EX["vT_nxt"] = FPX.volume.reindex(pd.MultiIndex.from_arrays([EX["T"], EX.nxt])).values
EX["i2"] = 2 * (ixq.open.reindex(EX.index).values / ixq.close.reindex(EX["T"]).values - 1)
EX = EX.dropna(subset=["r_nxt", "r_old", "i2"])
EX = EX[EX.index >= "2016-08-10"]
rec("2_롱밤", "모든 만기 전날 밤: 차근월 − 만기물 / 차근월 − 지수 2배 / 만기물 − 지수 2배 평균(bp), 차근월 > 만기물 비율",
    f"{bp((EX.r_nxt - EX.r_old).mean())} / {bp((EX.r_nxt - EX.i2).mean())} / {bp((EX.r_old - EX.i2).mean())}, {(EX.r_nxt > EX.r_old).mean():.0%}",
    f"n {len(EX)}; 비교: 모든 밤 선물 − 지수 2배 {bp((allN.f - allN.i2).mean())}; 2021-09-09 롤 밤(W_long 낙폭 구간 안) 차근월 {bp(EX.r_nxt.get(pd.Timestamp('2021-09-09'), np.nan))}, 만기물 {bp(EX.r_old.get(pd.Timestamp('2021-09-09'), np.nan))}, 지수 2배 {bp(EX.i2.get(pd.Timestamp('2021-09-09'), np.nan))}")
flat = lg[lg.oD == lg.cT]
rec("2_롱밤", "롱 밤 중 선물 시가 = T 종가(평탄) 밤 / 그 밤 차 평균(bp)", f"{len(flat)} / {bp(flat.adv.mean())}", "평탄 시가는 선물 경로를 0 으로 눌러 차를 줄이는 쪽(분봉 2026-07-03 은 08:45 단일가 실체결 1557.9 확인)")
o17 = lg.loc["2017-10-16"]
rec("2_롱밤", "2017-10-16 롱 밤(얇은 시장 시가 = 고가)", f"선물 {o17.cT:.1f}→{o17.oD:.1f}(gross {bp(o17.f)}), 지수 2배 {bp(o17.i2)}, 233740 {bp(o17.e)}",
    f"T 종가 베이시스 {bp(o17.b_close)}, d 시가 베이시스 {bp(o17.b_open)}; 이 한 밤이 W_long 롱 밤 평균 차에 보탠 몫 {o17.adv / len(lg.loc[W_LONG[0]:W_LONG[1]]) * 1e4:+.1f}bp/밤")
# 분봉(최근 1년): 선물 창 차이(15:30→15:45, 08:45→09:00)와 같은 시각 베이시스 변화
fm = pd.read_csv(KQMIN, parse_dates=["date"])


def at(df, hm, field, fallback_before=True):
    x = df[df.time == hm].set_index("date")[field]
    if fallback_before:
        prior = df[df.time < hm].groupby("date").close.last()
        x = x.reindex(prior.index.union(x.index)).fillna(prior)
    return x


F1530 = at(fm, "15:30", "open")
F0900 = at(fm, "09:00", "open")
FO =fm.groupby("date").open.first()
FC = fm.groupby("date").close.last()
FCON = fm.groupby("date").contract.first()
mm = lg[(lg.index >= fm.date.min()) & (lg["T"] >= fm.date.min())].copy()
mm = mm[(FCON.reindex(mm["T"]).values == mm.code.values) & (FCON.reindex(mm.index).values == mm.code.values)]
mm["chk_c"] = FC.reindex(mm["T"]).values - mm.cT.values
mm["chk_o"] = FO.reindex(mm.index).values - mm.oD.values
mm["win_close"] = 2 * (mm.cT / F1530.reindex(mm["T"]).values - 1)   # 15:30→15:45 선물 움직임(+면 선물 진입이 비쌈)
mm["win_open"] = 2 * (mm.oD / F0900.reindex(mm.index).values - 1)   # 08:45 시가 − 09:00 가격(+면 08:45 청산이 유리)
mm["f_match"] = 2 * (F0900.reindex(mm.index).values / F1530.reindex(mm["T"]).values - 1)
mm["basis_chg"] = mm.f_match - mm.i2
rec("2_롱밤", "분봉 1년 롱 밤 n / 일봉 대조 최대 차(종가, 시가)", f"{len(mm)} / {np.nanmax(np.abs(mm.chk_c)):.2f}, {np.nanmax(np.abs(mm.chk_o)):.2f}", f"{mm.index.min().date() if len(mm) else None}~{mm.index.max().date() if len(mm) else None}")
rec("2_롱밤", "분봉 1년 롱 밤: 선물 이득 = 같은 시각 베이시스 변화 + 08:45 창 − 15:45 창(bp)",
    f"{bp(mm.fut_gain.mean())} ≈ {bp(mm.basis_chg.mean())} + {bp(mm.win_open.mean())} − {bp(mm.win_close.mean())}", f"ETF 손실 {bp(mm.etf_loss.mean())}, 차 {bp(mm.adv.mean())}")
# 같은 해 모든 밤(조건 없음)
al = allN[(allN.index >= fm.date.min()) & (allN["T"] >= fm.date.min())].copy()
al = al[(FCON.reindex(al["T"]).values == al.code.values) & (FCON.reindex(al.index).values == al.code.values)]
al["win_close"] = 2 * (al.cT / F1530.reindex(al["T"]).values - 1)
al["win_open"] = 2 * (al.oD / F0900.reindex(al.index).values - 1)
al["basis_chg"] = 2 * (F0900.reindex(al.index).values / F1530.reindex(al["T"]).values - 1) - al.i2
for kind, g in al.groupby("kind"):
    rec("2_롱밤", f"분봉 1년 [{kind} 밤] 15:45 창 / 08:45 창 / 같은 시각 베이시스 변화 / ETF 손실(bp)",
        f"{bp(g.win_close.mean())} / {bp(g.win_open.mean())} / {bp(g.basis_chg.mean())} / {bp((g.i2 - g.e).mean())}", f"n {len(g)}")
# 233740 분봉: 09:00 단일가 매도 대신 09:01 첫 체결 매도(롱 밤 W_all)


def tossm(code):
    m = pd.read_csv(MIN / f"{code}_TOSS_1m.csv", encoding="utf-8-sig", usecols=["timestamp", "date", "open", "close", "volume"],
                    parse_dates=["timestamp", "date"]).sort_values("timestamp")
    m = m[m.volume > 0]
    first = m.groupby("date").head(1).set_index("date")                 # 시가 단일가 봉(끝 라벨 09:01 = 09:00:00~59)
    second = m.groupby("date").nth(1).set_index("date")                 # 다음 체결 봉(09:01 첫 체결)
    last = m.groupby("date").tail(1).set_index("date")
    return pd.DataFrame({"o_auc": first.open, "lab_auc": first.timestamp.dt.strftime("%H:%M"), "p0901": second.open,
                         "lab_0901": second.timestamp.dt.strftime("%H:%M"), "c_last": last.close, "lab_last": last.timestamp.dt.strftime("%H:%M")})


TM = {c: tossm(c) for c in ("069500", "122630", "229200", "233740")}
for c, t in TM.items():
    tt = t.join(PX[c][["open", "close"]], how="inner")
    rec("3_분봉", f"{c} TOSS 분봉 일수 / 첫 봉 시가 = 일봉 시가 / 마지막 봉 종가 = 일봉 종가 / 첫 봉 라벨 09:01 비율",
        f"{len(tt)} / {(tt.o_auc == tt.open).mean():.3f} / {(tt.c_last == tt.close).mean():.3f} / {(tt.lab_auc == '09:01').mean():.3f}", f"{tt.index.min().date()}~{tt.index.max().date()}")
lgw = lg.loc[WIN["W_all"][0]:WIN["W_all"][1]].copy()
lgw["e0901"] = TM["233740"].p0901.reindex(lgw.index).values / e233.close.reindex(lgw["T"]).values - 1
rec("2_롱밤", "W_all 롱 밤: 233740 09:01 첫 체결 매도 − 09:00 단일가 매도(bp) / 분봉 있는 밤", f"{bp((lgw.e0901 - lgw.e).mean())} / {int(lgw.e0901.notna().sum())}",
    f"선물 net − 233740(09:01 매도) net = {bp((lgw.net_f - (lgw.e0901 - 5e-4)).mean())}")

# ======================================================================== 3. 신호 시각(미래 정보)
e2 = etf("229200")
ma10 = e2.close.rolling(10).mean()
reg_up = (e2.close > ma10).shift(1)
ibs = (e2.close - e2.low) / (e2.high - e2.low)
sig = pd.Series(0.0, index=e2.index)
sig[(ibs < 0.30) & (reg_up == True)] = 2.0
sig[(ibs > 0.70) & (reg_up == False)] = -1.0
rep = pd.Series(sig.reindex(PREV.reindex(q.index).values).values, index=q.index).fillna(0.0)
rec("3_신호", "blog_ON_Q pos = T 의 229200 일봉(15:30 확정) 신호 재현", f"{int((rep == q.pos).sum())}/{len(q)}", "신호 입력은 T 고가·저가·종가와 T−1 까지 MA10 → 15:30 에 확정, 선물 진입은 15:35~15:45 종가 단일가")
rec("3_신호", "선물 T 종가 = 15:45 종가 단일가(분봉 1년 대조)", f"{int((np.abs(mm.chk_c) < 1e-9).sum())}/{len(mm)}", "분봉 마지막 봉 = 15:45 단일가 봉")
# 15:30→15:45 사이 정보로 신호가 바뀔 수 없는지: 신호는 229200 의 15:30 이전 가격만 쓴다(코드상 일봉 OHLC). 선물 종가 단일가 이후 정보 사용 없음.
rec("3_신호", "롱 밤 T 와 d 사이 거래일 빠짐(달력 간격 > 1 거래일)", int((pd.Series(CAL.get_indexer(lg.index) - CAL.get_indexer(pd.DatetimeIndex(lg["T"])), index=lg.index) != 1).sum()))

# ======================================================================== 4. 아침 혼합
for name in ("K_B", "Q_B"):
    h, c1, c2, mk = C3[name]
    s = S[h]
    on = s.pos != 0
    g1 = pd.Series(mix_ret(s.pos.abs().values, OC[c1].reindex(s.index).values, OC[c2].reindex(s.index).values,
                           IOC[mk].reindex(s.index).fillna(OC[c1].reindex(s.index)).values), index=s.index)
    dg = (g1 - s.gross)[on]
    top = dg.abs().sort_values(ascending=False).index[:3]
    rec("4_아침", f"{h} k1 혼합 gross − 13차 gross 평균 / |차| 최대(bp)", f"{bp(dg.mean())} / {dg.abs().max() * 1e4:.2f}",
        f"활성 {int(on.sum())}일, 명목 > 1 {int((s.pos[on] > 1).sum())}일; 큰 차 3일 {[(str(d.date()), round(s.pos[d], 2), round(OC[c1].get(d, np.nan) * 1e4), round(OC[c2].get(d, np.nan) * 1e4), round(dg[d] * 1e4)) for d in top]}(날짜, pos, oc1 bp, oc2 bp, 차 bp)")
    for wn, (a, z) in (("W_all", WIN["W_all"]), ("W_long", W_LONG)):
        ss = s.loc[a:z]
        oo = ss.pos != 0
        miss = IOC[mk].reindex(ss.index)[oo].isna()
        rec("4_아침", f"{h} {wn} 활성일 / 지수 시가 결측·평탄(1배 ETF 대체) / 그중 pos > 1(초과 몫 생길 수 있는 날)",
            f"{int(oo.sum())} / {int(miss.sum())} / {int((miss & (ss.pos[oo] > 1)).sum())}")
        for kk in (KR0, KM0):
            x = kk * ss.pos[oo].abs()
            o1 = OC[c1].reindex(x.index)
            oi = IOC[mk].reindex(x.index).fillna(o1)
            exc = ((x - 2).clip(lower=0) * oi)
            etfp = pd.Series(mix_ret(x.clip(upper=2).values, o1.values, OC[c2].reindex(x.index).values, oi.values), index=x.index)
            rec("4_아침", f"{h} {wn} k {kk}: 명목 > 2 날 / 초과 명목 평균 / 활성일 평균 bp(ETF 몫, 초과 지수 몫)",
                f"{int((x > 2).sum())} / {(x - 2)[x > 2].mean():.2f} / {bp(etfp.mean())}, {bp(exc.mean())}")
# 코스닥 ETF 시가 단일가 잡음(얇은 초기 연도): 233740 시가→종가 − 2×229200 시가→종가 |차| > 50bp 날 비율, C3Q 연도별 ETF 몫
dd = (OC["233740"] - 2 * OC["229200"]).dropna()
c3q = S["beomgu_C3Q"]
aq = c3q[c3q.pos != 0]
etfq = (aq.gross - aq.pos * IOC["Q"].reindex(aq.index)).dropna()
rec("4_아침", "연도별 |233740 oc − 2×229200 oc| > 50bp 날 비율 / C3Q 활성일 수·ETF 몫 평균 bp",
    "; ".join(f"{y}:{(dd[dd.index.year == y].abs() > 0.005).mean():.1%}/{int((etfq.index.year == y).sum())}·{etfq[etfq.index.year == y].mean() * 1e4:+.0f}" for y in range(2016, 2027)),
    "ETF 몫 = gross − pos×KQ150 지수 시가→종가")
# 지수 시가→종가 대 선물 09:00→15:30(분봉): 초과 몫을 지수로 둔 잡음
for mk, path, h in (("Q", KQMIN, "beomgu_C3Q"), ("K", K2MIN, "beomgu_C3VT30")):
    f1 = pd.read_csv(path, parse_dates=["date"])
    p09, p1530 = at(f1, "09:00", "open"), at(f1, "15:30", "open")
    fo = (p1530 / p09 - 1)
    fc = f1.groupby("date").close.last() / p09 - 1                     # 09:00 → 15:45 종가
    d = pd.DataFrame({"idx": IOC[mk].reindex(fo.index), "f1530": fo, "f1545": fc}).dropna()
    on = S[h].pos.reindex(d.index).fillna(0) != 0
    rec("4_아침", f"{mk} 지수 시가→종가 − 선물 09:00→15:30 평균 / 표준편차(bp), 전체일 / C3 활성일",
        f"{bp((d.idx - d.f1530).mean())} / {(d.idx - d.f1530).std() * 1e4:.1f}, {bp((d.idx - d.f1530)[on].mean())} / {(d.idx - d.f1530)[on].std() * 1e4:.1f}",
        f"n {len(d)} / {int(on.sum())}; 선물 09:00→15:45 기준 C3 활성일 {bp((d.idx - d.f1545)[on].mean())}")

# ======================================================================== 5. 단일가 다리: 1틱, 09:01 실행형
SC = {}                                                              # 시나리오 → (G, kr, km)


def scenario(tag, sec, **kw):
    G, kr, km = grid(**kw)
    SC[tag] = (G, kr, km)
    rec(sec, f"[{tag}] k_rec / k_max", f"{kr} / {km}", f"v2 {KR0} / {KM0}")
    for kk in sorted(x for x in {1.0, KR0, KM0, kr, km} if np.isfinite(x)):
        d = G[np.isclose(G.k, kk)].iloc[0].to_dict()
        b = BASEROW.get(kk)
        diff = ""
        if b is not None:
            diff = (f"v2 대비 W_all {(d['W_all_cagr'] - b['W_all_cagr']) * 100:+.1f}%p, ex2026 {(d['W_ex2026_cagr'] - b['W_ex2026_cagr']) * 100:+.1f}%p, "
                    f"2024 {(d['Y2024_ret'] - b['Y2024_ret']) * 100:+.1f}%p, 2025 {(d['Y2025_ret'] - b['Y2025_ret']) * 100:+.1f}%p, W_long {(d['W_long_cagr'] - b['W_long_cagr']) * 100:+.1f}%p/MDD {(d['W_long_mdd'] - b['W_long_mdd']) * 100:+.2f}%p; "
                    f"판정 바뀜 {[w for w in CW if judge(d)[w] != judge(b)[w]]}")
        rec(sec, f"[{tag}] k {kk}", say(d), f"판정 {jstr(judge(d))}; {diff}")
    return G, kr, km


scenario("단일가 ETF 다리 1틱(K_A·K_B·Q_B)", "5_단일가", auc_tick=True)
# 아침 C3 09:01 실행형(공통 규약 4: 09:00 봉 마감 판단 → 09:01 첫 체결): 분봉 있는 날만, 초과 지수 몫은 1배 ETF 09:01→종가로 근사
C3OC = {}
for name in ("K_B", "Q_B"):
    h, c1, c2, mk = C3[name]
    # 069500·229200 일봉은 수정주가(TOSS 원가격과 배율이 해마다 다름) → 수익은 TOSS 안에서만 잰다(15:30 봉 종가 / 09:01 첫 체결)
    a1 = (TM[c1].c_last / TM[c1].p0901 - 1).where(TM[c1].lab_last.isin(["15:30", "16:30"]))
    a2 = (TM[c2].c_last / TM[c2].p0901 - 1).where(TM[c2].lab_last.isin(["15:30", "16:30"]))
    a1_900 = TM[c1].c_last / TM[c1].o_auc - 1
    rec("5_단일가", f"{c1} TOSS 안 09:00 단일가→종가 − 일봉 시가→종가 최대 차(bp, W_all 활성일)", f"{((a1_900 - OC[c1]).reindex(S[h].index[S[h].pos != 0]).loc[WIN['W_all'][0]:].abs().max() * 1e4):.2f}")
    C3OC[name] = (a1, a2)
    s = S[h]
    oo = s.index[s.pos != 0]
    ow = oo[(oo >= WIN["W_all"][0]) & (oo <= WIN["W_all"][1])]
    have = a1.reindex(ow).notna() & a2.reindex(ow).notna()
    x = s.pos.abs().reindex(ow)
    g900 = pd.Series(mix_ret(x.values, OC[c1].reindex(ow).values, OC[c2].reindex(ow).values, IOC[mk].reindex(ow).fillna(OC[c1].reindex(ow)).values), index=ow)
    g901 = pd.Series(mix_ret(x.values, a1.reindex(ow).values, a2.reindex(ow).values, a1.reindex(ow).values), index=ow)
    rec("5_단일가", f"{h} W_all 활성 {len(ow)}일 중 분봉 있는 날 / k1 gross 평균 09:00 단일가 → 09:01 첫 체결(bp)", f"{int(have.sum())} / {bp(g900[have].mean())} → {bp(g901[have].mean())}",
        f"남는 몫 {g901[have].mean() / g900[have].mean():.0%}; {c1} 첫 1분 변화 {bp((TM[c1].p0901 / TM[c1].o_auc - 1).reindex(ow).mean())}, {c2} {bp((TM[c2].p0901 / TM[c2].o_auc - 1).reindex(ow).mean())}")
scenario("아침 C3 09:01 실행형(W_all 구간만, W_long 의 2022-11 전은 단일가)", "5_단일가", c3oc=C3OC)
scenario("아침 09:01 + 단일가 ETF 1틱", "5_단일가", c3oc=C3OC, auc_tick=True)
# 2022-11 전 날의 근사: W_all C3 날 실측 첫 1분 평균 변화(ETF 별)만큼 시가를 올림(지수 몫은 그대로)
PROXY = {}
for name in ("K_B", "Q_B"):
    h, c1, c2, mk = C3[name]
    ow = S[h].index[(S[h].pos != 0) & (S[h].index >= WIN["W_all"][0]) & (S[h].index <= WIN["W_all"][1])]
    PROXY[name] = tuple(float((TM[c].p0901 / TM[c].o_auc - 1).reindex(ow).mean()) for c in (c1, c2))
rec("5_단일가", "근사용 첫 1분 평균 변화(bp): K_B 069500·122630 / Q_B 229200·233740", f"{PROXY['K_B'][0] * 1e4:+.1f}·{PROXY['K_B'][1] * 1e4:+.1f} / {PROXY['Q_B'][0] * 1e4:+.1f}·{PROXY['Q_B'][1] * 1e4:+.1f}",
    "W_long 의 2016-10~2022-11 C3 날 ETF 몫이 W_all 보다 컸으므로(v1 검토 C3Q +38.7 vs +24.5bp) 이 근사는 덜 가혹한 쪽")
scenario("아침 09:01 실행형 + 2022-11 전 근사(W_long 포함)", "5_단일가", c3oc=C3OC, c3proxy=PROXY)

# ======================================================================== 6. 낮 다리 틱
KB_W = S["beomgu_C3VT30"].pos.reindex(A_W).fillna(0) != 0
QB_W = S["beomgu_C3Q"].pos.reindex(A_W).fillna(0) != 0
kpW = KP_TICK.reindex(A_W).where(~KB_W, 0.0)
kqW = KQ_TICK.reindex(A_W).where(~QB_W, 0.0)
rec("6_낮", "W_all 틱 합(자본 1, %) / 활성일 평균 bp: K_C / Q_C", f"{kpW.sum() * 100:.2f} / {kpW[kpW > 0].mean() * 1e4:.1f}, {kqW.sum() * 100:.2f} / {kqW[kqW > 0].mean() * 1e4:.1f}")
alt_kp = pd.Series(np.where(KP.pos != 0, KP.pos.abs().values / 2 * tk("122630", KP.index).values, 0.0), index=KP.index)
alt_kq = pd.Series(np.where(KQ.pos > 0, KQ.pos.values / 2 * tk("233740", KQ.index).values, 0.0), index=KQ.index)
rec("6_낮", "대안 틱(차량 = 2배 ETF 만, 자본 pos/2): K_C 122630 / Q_C 233740 W_all 합(%)",
    f"{alt_kp.reindex(A_W).where(~KB_W, 0.0).sum() * 100:.2f} / {alt_kq.reindex(A_W).where(~QB_W, 0.0).sum() * 100:.2f}", "v2 혼합식 합과 비교")
d_sh = (qc_leg(1.0, BASE) / 1.0 - (KQ.net - KQ_TICK)).reindex(A_W)[SH.reindex(A_W).fillna(False)]
rec("6_낮", "Q_C 숏일 v2 net − 13차 net 평균(bp/숏일)", bp(d_sh.mean()), f"숏일 {int(SH.reindex(A_W).fillna(False).sum())}")

# ======================================================================== 7. 선물 3bp, 묶음
scenario("선물 비용 3bp(밤 롱·인버스, 낮 숏, 아침 초과 몫)", "7_선물3bp", fut=3e-4)
scenario("선물 3bp + 단일가 ETF 1틱", "7_선물3bp", fut=3e-4, auc_tick=True)
scenario("선물 3bp + 단일가 1틱 + 아침 09:01", "7_선물3bp", fut=3e-4, auc_tick=True, c3oc=C3OC)
scenario("롱 밤 233740 복귀(참고: 선물 경로 효과 크기)", "7_선물3bp", qa_long="etf")
scenario("2016~2017 얇은 선물 시장 롱 밤만 233740(참고)", "7_선물3bp", qa_etf_dates=lg.index[lg.index < "2018-01-01"])
scenario("롤 밤(d = 만기일)을 T 의 front(만기물)로", "7_선물3bp", qa_roll_frontT=True)
scenario("2021-09-09 롤 밤 하나만 만기물로", "7_선물3bp", qa_roll_frontT=["2021-09-09"])
scenario("2016~2017 롱 밤 233740 + 롤 밤 만기물(자료 처리 선택 묶음)", "7_선물3bp", qa_etf_dates=lg.index[lg.index < "2018-01-01"], qa_roll_frontT=True)
# 선물 비용 손익분기: k_max 의 W_long CAGR ≥ 30% 판정이 뒤집히는 선물 비용
for fc in (2.0e-4, 2.2e-4, 2.4e-4, 2.6e-4, 2.8e-4):
    G, kr, km = grid(fut=fc)
    d = G[np.isclose(G.k, km)].iloc[0]
    rec("7_선물3bp", f"선물 {fc * 1e4:.1f}bp: k_rec / k_max / k_max 의 W_long CAGR·MDD / k_max 의 2024", f"{kr} / {km} / {d.W_long_cagr:.2%}·{d.W_long_mdd:.2%} / {d.Y2024_ret:+.1%}",
        f"판정 {jstr(judge(d.to_dict()))}")

# 코스피 밤 실행 일치형(명목 k 를 069500/122630 + K200 선물로)
f2 = pd.read_csv(RAW / "r15_external/kis_fut_k200_daily.csv", parse_dates=["date"])
f2 = f2[f2.date < "2026-10-30"]
F2PX = f2.set_index(["date", "code"]).sort_index()
F2FR = f2[f2.front == 1].drop_duplicates("date").set_index("date").code
kaD = S["jehyeong_ON"].index
c2d = F2FR.reindex(kaD)
OC_K2F = pd.Series(F2PX.open.reindex(pd.MultiIndex.from_arrays([kaD, c2d.values])).values /
                   F2PX.close.reindex(pd.MultiIndex.from_arrays([PREV.reindex(kaD).values, c2d.values])).values - 1, index=kaD)
kon = S["jehyeong_ON"].index[S["jehyeong_ON"].pos != 0]
rec("7_선물3bp", "K_A 활성 밤: 122630 밤 − 2×069500 밤 / K200 선물 밤 − 069500 밤 평균(bp)",
    f"{bp((ON['122630'] - 2 * ON['069500']).reindex(kon).mean())} / {bp((OC_K2F - ON['069500']).reindex(kon).mean())}", f"n {len(kon)}, 13차 gross − 069500 밤 최대 차 {(S['jehyeong_ON'].gross.reindex(kon) - ON['069500'].reindex(kon)).abs().max():.1e}")
scenario("코스피 밤 실행 일치형(069500/122630 + K200 선물)", "7_선물3bp", ka="mix")

# ======================================================================== 8. 코드 점검
L1 = legs(1.0)
wall1, wl1, nanc, g1 = assemble(L1)
rec("8_코드", "W_all 다리별 결측(재색인 후 0 처리 전)", nanc)
for name in ("K_B", "Q_B"):
    h = C3[name][0]
    on = S[h].pos != 0
    rec("8_코드", f"{name} 활성일 NaN(전 구간 / W_long)", f"{int(L1[name][on].isna().sum())} / {int(L1[name][on].loc[W_LONG[0]:W_LONG[1]].isna().sum())}")
rec("8_코드", "W_long 일수 / 069500 달력 같은 구간 일수 / NaN", f"{len(wl1)} / {int(((CAL >= W_LONG[0]) & (CAL <= W_LONG[1])).sum())} / {int(wl1.isna().sum())}", f"첫날 {wl1.index[0].date()} 끝날 {wl1.index[-1].date()}")
kb_on = S["beomgu_C3VT30"].pos.reindex(A_W).fillna(0) != 0
qb_on = S["beomgu_C3Q"].pos.reindex(A_W).fillna(0) != 0
rec("8_코드", "B 활성 판정 vs book_daily B 열 ≠ 0 불일치 일수(코스피 / 코스닥)", f"{int((kb_on != (book.KOSPI_B_beomgu_C3VT30 != 0)).sum())} / {int((qb_on != (book.KOSDAQ_B_beomgu_C3Q != 0)).sum())}")
rec("8_코드", "B 활성일에 C 활성(겹침, C = 0 처리) 일수(코스피 / 코스닥)", f"{int((kb_on & (KP.pos.reindex(A_W).fillna(0) != 0)).sum())} / {int((qb_on & (KQ.pos.reindex(A_W).fillna(0) != 0)).sum())}",
    f"book_daily clash 열 합 {int(book.KOSPI_clash.sum())} / {int(book.KOSDAQ_clash.sum())}")
rec("8_코드", "k1 W_all 일수익 − 0.5×Σ다리 최대 차(검산)", f"{(wall1 - 0.5 * sum(g1.values())).abs().max():.1e}")
d1 = BASEROW[1.0]
rec("8_코드", "MDD 시작 자본 포함 vs 미포함(k1 W_all, W_long)", f"{d1['W_all_mdd0']:.4%} vs {d1['W_all_mdd']:.4%}, {d1['W_long_mdd0']:.4%} vs {d1['W_long_mdd']:.4%}")
for w in ("Y2024", "Y2025", "Y2026"):
    a, z = WIN[w]
    yy = A_W[A_W.year == int(w[1:])]
    rec("8_코드", f"{w} 창 = 그 해 달력 첫날~끝날", bool(len(yy) and str(yy[0].date()) == a and (str(yy[-1].date()) == z or w == "Y2026")), f"달력 {yy[0].date()}~{yy[-1].date()}")
contrib = {n: float(v.sum()) for n, v in g1.items()}
rec("8_코드", "k1 W_all 다리별 일수익 합(×0.5 전, %)", {n: round(v * 100, 2) for n, v in contrib.items()})
gl = {n: L1[n].reindex(wl1.index) for n in ("K_B", "Q_A", "Q_B")}
pk, tr_ = pd.Timestamp(BASEROW[KR0]["W_long_peak"]), pd.Timestamp(BASEROW[KR0]["W_long_trough"])
rec("8_코드", f"k {KR0} W_long 낙폭 구간 {pk.date()}~{tr_.date()} 다리별 합(×0.5 전, k1 기준 %)", {n: round(float(v.loc[pk:tr_].sum()) * 100, 2) for n, v in gl.items()})

pd.DataFrame(ROWS).to_csv(OUT / "review_v2_checks.csv", index=False, encoding="utf-8-sig")
SG = []
for tag, (G, kr, km) in SC.items():
    G = G.copy()
    G.insert(0, "scenario", tag)
    G["k_rec"], G["k_max"] = kr, km
    SG.append(G)
pd.concat(SG).to_csv(OUT / "review_v2_scenarios.csv", index=False, encoding="utf-8-sig")
print("저장:", OUT)
