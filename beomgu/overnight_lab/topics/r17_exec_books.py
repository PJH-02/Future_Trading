"""r17_exec_books.py: 13차 장부·15차 v3 장부에서 '신호 가격 = 체결 가격' 다리를 실행 가능한 체결로 바꿔 다시 백테스트.

규칙·신호·비중·배수는 13차 series·15차 v3 그대로. 바꾼 것은 체결 시각과 그에 따른 비용만. 결과를 보고 고치지 않는다.
체결 규약
  • 아침 C3(069500+122630)·C3Q(229200+233740): 09:00 단일가(09:01 봉 시가) 신호 확정 → 09:01:00 첫 연속 체결(09:02 봉 시가, 무거래면 다음 거래 봉) 진입
    → 15:19:00(15:20 봉 시가, 무거래면 15:30 마감 단일가) 청산. 09:01 봉 무거래(늦은 개장)면 그날 쉼. x = |pos|(VT30, ≤ 2), 수익 = mix_ret(x, a1, a2, a1), TOSS 원가격 안에서만.
  • 코스닥 밤 blog_ON_Q 실행판: 229200 09:01~15:20 봉 고저·15:20 종가로 IBS20 → 15:30 단일가 체결(IBS<0.3 ∧ 전일 종가>MA10 → 233740 현금 1, IBS>0.7 ∧ 전일 종가<MA10 → 251340 현금 1)
    → 다음날 09:00 단일가(09:01 봉 무거래면 쉼). 특수일(r16 SPECIAL) 제외. 손익 날짜 = 청산일. 16차 캐시 L["QA_bm"].net 과 일치 단언.
  • 그대로: 코스피 밤 jehyeong_ON(15:20 신호 → 15:30 → 09:00), 팀 장중 jehyeong_KP·KQ(자체 1분봉 체결), 15차 코스닥 밤 선물·낮 숏 선물.
비용
  • 13차 비용: beomgu·blog 다리 ETF 왕복 5bp(현금 min(x,1)), 팀 다리 자체 비용(틱 없음). 폭포의 '체결만 바꾼 단계'는 이 비용.
  • 기본: 13차 비용 + 연속 체결이 끼는 왕복마다 1틱 왕복(mix_tick, 틀의 tk = 5원/그해 평균가) → 아침 실행판, 팀 낮 다리(KP mix_tick, KQ 롱 229200·숏 251340). 밤 ETF 단일가 다리 틱 없음. 선물 1.8bp×명목, 낮 숏 대리 +1.06bp.
  • 하한: 기본 + 단일가 ETF 다리(코스피 밤 069500, 코스닥 밤 ETF)에도 1틱 + 선물 3bp.
장부
  • A 13차 실행판: k 1, ETF 만. 코스피 = jehyeong_ON + C3 실행판 + KP(C3 활성일 0), 코스닥 = blog_ON_Q 실행판 + C3Q 실행판 + KQ(C3Q 활성일 0). 합산 = 50%·50% 매일 재조정.
  • B 15차 v3 실행판: k 1.40, 틀 legs(qa_etf_dates=thin, qa_roll_frontT=True) + c3_leg 덮어쓰기. 2022-11-23 전(W_long)은 근사 (1+oc)/(1+m)/(1+e)−1, m = 틀 PROXY(첫 1분), e = W_all 활성일 평균(15:30 종가/15:19:00 시가 − 1).
창: W_all 2022-12-13~2026-09-17, W_des ~2024-12-30, W_val 2025-01-02~, W_ex2026 ~2025-12-30, 연도 2023~2026, W_long 2016-10-04~(15차만).
지표: 누적 Π(1+r), CAGR = 누적^(252/일수)−1, MDD, 샤프 = mean·252/(std·√252), 최악일. 다리: 활성일 순 bp, NW t(HAC 5).
틀 verify_r15_v2_review.py 는 경로 치환 뒤 exec(시나리오 격자만 생략, 함수·자료는 그대로). 기존 파일 수정 없음. 산출 round17_exec_books/.
실행: PYTHONUTF8=1 python "<code>/topics/r17_exec_books.py"
"""
import contextlib
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)
HERE = Path(__file__).resolve().parent
LAB = Path("<OUT>")
R13, R15, R16 = LAB / "round13_slot_book", LAB / "round15_alpha_share", LAB / "round16_alpha_search"
OUT = LAB / "round17_exec_books"
(OUT / "from_review").mkdir(parents=True, exist_ok=True)
JH = Path("<repo>/jehyeong/intraday/data")
CACHE16 = Path("<local>")
ETF, FUT, FUT_S = 5e-4, 1.8e-4, 3e-4
WIN = {"W_all": ("2022-12-13", "2026-09-17"), "W_des": ("2022-12-13", "2024-12-30"), "W_val": ("2025-01-02", "2026-09-17"), "W_ex2026": ("2022-12-13", "2025-12-30"),
       "Y2023": ("2023-01-02", "2023-12-28"), "Y2024": ("2024-01-02", "2024-12-30"), "Y2025": ("2025-01-02", "2025-12-30"), "Y2026": ("2026-01-02", "2026-09-17")}
NOTES = []


def note(s=""):
    NOTES.append(s)
    print(s)


# ---------------------------------------------------------------- 0. 틀 exec(경로 치환, 격자 생략)
src = (HERE / "verify_r15_v2_review.py").read_text(encoding="utf-8")
for a, b in (('LAB = Path("<OUT>")', f'LAB = Path("{LAB.as_posix()}")'),
             ('OUT = R15 / "final_v2" / "verify"', f'OUT = Path("{(OUT / "from_review").as_posix()}")'),
             ('for cfg, kw in (("v2", {}), ("v2_notick", dict(day_tick=False)), ("v2_c3scaled", dict(c3="scaled"))):', 'for cfg, kw in (("v2", {}),):'),
             ("    G, kr, km = grid(**kw)\n    SC[tag] = (G, kr, km)", "    return None, None, None  # r17: 격자 생략\n    G, kr, km = grid(**kw)\n    SC[tag] = (G, kr, km)"),
             ("for fc in (2.0e-4, 2.2e-4, 2.4e-4, 2.6e-4, 2.8e-4):", "for fc in ():"),
             ('pd.concat(SG).to_csv(OUT / "review_v2_scenarios.csv"', 'SG and pd.concat(SG).to_csv(OUT / "review_v2_scenarios.csv"')):
    assert src.count(a) == 1, a
    src = src.replace(a, b)
ns = {"__name__": "r17_template", "__file__": str(HERE / "verify_r15_v2_review.py")}
with open(OUT / "from_review" / "stdout.txt", "w", encoding="utf-8") as fh, contextlib.redirect_stdout(fh):
    exec(compile(src, "verify_r15_v2_review.py", "exec"), ns)
S, PX, OC, IOC, TKY, tk, mix_ret, mix_tick, C3 = (ns[n] for n in ("S", "PX", "OC", "IOC", "TKY", "tk", "mix_ret", "mix_tick", "C3"))
legs, assemble, run, A_W, W_LONG, CAL, PREV, lg, C3OC, PROXY, book13, q, etf, c3_leg0 = (
    ns[n] for n in ("legs", "assemble", "run", "A_W", "W_LONG", "CAL", "PREV", "lg", "C3OC", "PROXY", "book", "q", "etf", "c3_leg"))
TKY["251340"] = 5 / etf("251340").close.groupby(etf("251340").index.year).mean()          # 틀 tk 에 251340 추가(틀과 같은 수정가 연평균 규약)
NXT = pd.Series(CAL[1:], index=CAL[:-1])
ON_, C3K_, KP_, ONQ_, C3Q_, KQ_ = (S[h] for h in ("jehyeong_ON", "beomgu_C3VT30", "jehyeong_KP", "blog_ON_Q", "beomgu_C3Q", "jehyeong_KQ"))
KB_ON, QB_ON = (s.pos.reindex(A_W).fillna(0) != 0 for s in (C3K_, C3Q_))


def met(r):
    r = r.dropna()
    eq = (1 + r).cumprod()
    n, cum, sd = len(r), float(eq.iloc[-1]), float(r.std(ddof=1))
    return dict(cagr_pct=(cum ** (252 / n) - 1) * 100, mdd_pct=float((eq / eq.cummax() - 1).min()) * 100, ret_pct=(cum - 1) * 100,
                sharpe=float(r.mean()) * 252 / (sd * np.sqrt(252)) if sd > 0 else np.nan, worst_day=str(r.idxmin().date()), worst_pct=float(r.min()) * 100, n_days=n)


def nw_t(x, lag=5):
    x = np.asarray(x, float)
    n = len(x)
    if n < 3:
        return np.nan
    e = x - x.mean()
    s = e @ e / n
    for j in range(1, min(lag, n - 1) + 1):
        s += 2 * (1 - j / (lag + 1)) * (e[j:] @ e[:-j]) / n
    return float(x.mean() / np.sqrt(s / n)) if s > 0 else np.nan


# ---------------------------------------------------------------- 1. 분봉(봉 끝 라벨), 특수일, 체결가 표
def read_min(c):
    m = pd.read_csv(JH / f"minute/{c}_TOSS_1m.csv", encoding="utf-8-sig", usecols=["timestamp", "date", "open", "high", "low", "close", "volume"], parse_dates=["date"])
    m = m[m.date <= CAL[-1]]
    m["hm"] = m.timestamp.str[11:16]
    return {k: m.pivot(index="date", columns="hm", values=k).sort_index() for k in ("open", "high", "low", "close", "volume")}


M = {c: read_min(c) for c in ("069500", "122630", "229200", "233740", "251340")}
Op, Cl, Vo = ({c: M[c][k] for c in M} for k in ("open", "close", "volume"))
DAYS = Op["229200"].index
assert CAL[(CAL >= DAYS[0]) & (CAL <= DAYS[-1])].equals(DAYS) and all(M[c]["open"].index.equals(DAYS) for c in M), "분봉 달력 ≠ 거래일 달력"
_v = Vo["229200"]
SPECIAL = DAYS[(Cl["229200"].notna().sum(axis=1) != 390) | (_v.loc[:, "09:01":"10:00"].sum(axis=1) == 0) |
               ((Op["229200"].loc[:, "15:21":"15:29"] != Cl["229200"].loc[:, "15:21":"15:29"]).sum(axis=1) >= 5)]


def exec_table(c):
    """09:01 봉(단일가) 거래량·시가, 09:02 봉 시가(무거래면 다음 거래 봉), 15:30 종가, 15:20 봉 시가(무거래면 15:30 마감 단일가)."""
    o, v, cl = Op[c], Vo[c], Cl[c]
    keep = [h for h in o.columns if not "15:21" <= h <= "15:29"]                     # 예상가 봉 제외
    po = o[keep].where(v[keep] > 0).bfill(axis=1)
    return pd.DataFrame({"v0901": v["09:01"].fillna(0.0), "p0901": o["09:01"], "e1": po["09:02"], "e1_push": v["09:02"].fillna(0.0) <= 0,
                         "x1530": cl["15:30"], "x1519": o["15:20"].where(v["15:20"] > 0, cl["15:30"]), "x1519_push": v["15:20"].fillna(0.0) <= 0})


TX = {c: exec_table(c) for c in M}
EXIT = {"1530": "x1530", "1519": "x1519"}


def c3_parts(name, k, ex, rest="0901"):
    """C3 다리 날짜별 체결 표: x, ETF 별 체결 수익 a1·a2, 쉼·밀기 표시. 분봉 없는 날은 have=False.
    rest='0901': 쓰는 ETF 의 09:01 봉 무거래면 쉼(사양). 'special': 16차 격자 재현용(특수일만 쉼, 09:01 무거래는 다음 봉으로 밀기)."""
    h, c1, c2, _ = C3[name]
    s = S[h]
    idx = s.index
    T1, T2 = TX[c1].reindex(idx), TX[c2].reindex(idx)
    x = k * s.pos.abs()
    have = pd.Series(idx.isin(DAYS), index=idx)
    rest = have & (((T1.v0901 <= 0) | ((T2.v0901 <= 0) & (x > 1))) if rest == "0901" else idx.isin(SPECIAL))
    return pd.DataFrame({"x": x, "on": s.pos != 0, "have": have, "rest": rest, "a1": T1[EXIT[ex]] / T1.e1 - 1, "a2": T2[EXIT[ex]] / T2.e1 - 1,
                         "e1": T1.e1, "e2": T2.e1, "x1": T1[EXIT[ex]], "x2": T2[EXIT[ex]],
                         "push_in": (T1.e1_push | (T2.e1_push & (x > 1))).fillna(False).astype(bool),
                         "push_out": ((T1.x1519_push | (T2.x1519_push & (x > 1))) & (ex == "1519")).fillna(False).astype(bool)})


def eprox(name):
    h, c1, c2, _ = C3[name]
    t = c3_parts(name, 1.0, "1519").loc[WIN["W_all"][0]:WIN["W_all"][1]]
    w = t.index[t.on & t.have & ~t.rest]
    return tuple(float((TX[c].x1530 / TX[c].x1519 - 1).reindex(w).mean()) for c in (c1, c2))


EPX = {"1519": {n: eprox(n) for n in C3}, "1530": {n: (0.0, 0.0) for n in C3}}       # 마지막 구간 근사 e(2022-11 전, W_long 만 해당)


def c3_leg17(name, k, p):
    """틀 c3_leg 덮어쓰기: p['c3exit'] 가 있으면 실행판(09:02 봉 시가 → 15:20 봉 시가/15:30 종가, 09:01 무거래 쉼, p['c3tick'] 이면 1틱 왕복), 없으면 틀 그대로."""
    if "c3exit" not in p:
        return c3_leg0(name, k, p)
    h, c1, c2, mk = C3[name]
    s = S[h]
    idx = s.index
    on = (s.pos != 0).values
    o1, o2, oi = OC[c1].reindex(idx), OC[c2].reindex(idx), IOC[mk].reindex(idx).fillna(OC[c1].reindex(idx))
    t = c3_parts(name, k, p["c3exit"], p.get("c3rest", "0901"))
    ok = (t.have & ~t.rest).values
    assert t.a1[ok & on].notna().all() and t.a2[ok & on].notna().all(), name
    o1, o2, oi = o1.where(~ok, t.a1), o2.where(~ok, t.a2), oi.where(~ok, t.a1)     # 초과(선물) 몫은 1배 ETF 체결 수익으로 근사(틀과 같음)
    m1, m2 = PROXY[name]
    e1, e2 = EPX[p["c3exit"]][name]
    nomin = ~t.have.values
    o1 = o1.where(~nomin, (1 + o1) / (1 + m1) / (1 + e1) - 1)                        # 분봉 없는 날(2022-11-23 전) 근사, 지수 몫은 그대로
    o2 = o2.where(~nomin, (1 + o2) / (1 + m2) / (1 + e2) - 1)
    x = t.x.values
    g = mix_ret(x, o1.values, o2.values, oi.values)
    cost = ETF * np.minimum(x, 1) + p["fut"] * np.maximum(x - 2, 0)
    if p["c3tick"]:
        cost = cost + mix_tick(x, tk(c1, idx).values, tk(c2, idx).values)
    act = on & ~t.rest.values
    return pd.Series(np.where(act, g - cost, 0.0), index=idx)


ns["c3_leg"] = c3_leg17                                                              # 틀 legs() 가 이 함수를 쓴다

# ---------------------------------------------------------------- 2. 코스닥 밤 실행판(r16_eval_legs night_auc(NSF) 재현)
SIG, LNG, SHT = "229200", "233740", "251340"
d229 = pd.read_csv(JH / f"{SIG}_daily.csv", encoding="utf-8-sig", usecols=["date", "close"]).drop_duplicates("date")
d229["date"] = pd.to_datetime(d229.date)
d229 = d229[d229.date <= CAL[-1]].sort_values("date").set_index("date").close
prev, ma10 = d229.shift(1), d229.rolling(10, min_periods=10).mean().shift(1)
UP, DN = (prev > ma10).reindex(DAYS).fillna(False), (prev < ma10).reindex(DAYS).fillna(False)
h20, l20 = M[SIG]["high"].loc[:, "09:01":"15:20"].max(axis=1), M[SIG]["low"].loc[:, "09:01":"15:20"].min(axis=1)
IBS20 = ((Cl[SIG]["15:20"] - l20) / (h20 - l20)).where(h20 > l20)
NS0 = pd.Series(np.where(IBS20 < .3, 1, np.where(IBS20 > .7, -1, 0)), index=DAYS).where(~DAYS.isin(SPECIAL), 0)
NSF = NS0.where(((NS0 == 1) & UP) | ((NS0 == -1) & DN), 0)
X09 = {c: Op[c]["09:01"].where(Vo[c]["09:01"] > 0).shift(-1) for c in (LNG, SHT)}
ONQ = pd.DataFrame(0.0, index=CAL, columns=["g", "p", "tick"])
ONQ_DROP = []
for sd, c, x in ((1, LNG, 2.0), (-1, SHT, -1.0)):
    ix = NSF.index[NSF == sd]
    r_all = (X09[c] / Cl[c]["15:30"] - 1).reindex(ix)
    r = r_all.dropna()
    d = NXT.reindex(r.index).dropna()
    r = r.reindex(d.index)
    ONQ_DROP += [str(t.date()) for t in r_all.index[r_all.isna()] if t < CAL[-1]]
    ONQ.loc[d.values, "g"] += r.values
    ONQ.loc[d.values, "p"] += x
    ONQ.loc[d.values, "tick"] += tk(c, d.index).values
ONQ["n"] = ONQ.g - ETF * (ONQ.p != 0)
ONQ.loc[CAL < DAYS[0]] = np.nan                                                      # 분봉 전은 유효 전(NaN)
assert ONQ.n.reindex(A_W).notna().all()

# ---------------------------------------------------------------- 3. 틱(기본·하한), 장부 A 조립
ON_TICK = pd.Series(np.where(ON_.pos != 0, mix_tick(ON_.pos.abs().values, tk("069500", ON_.index).values, tk("122630", ON_.index).values), 0.0), index=ON_.index)
KP_TICK = pd.Series(np.where(KP_.pos != 0, mix_tick(KP_.pos.abs().values, tk("069500", KP_.index).values, tk("122630", KP_.index).values), 0.0), index=KP_.index)
KQ_TICK = pd.Series(np.where(KQ_.pos > 0, KQ_.pos.values * tk("229200", KQ_.index).values, KQ_.pos.abs().values * tk("251340", KQ_.index).values), index=KQ_.index)
PA = dict(fut=FUT)


def legsA(c3k="orig", c3q="orig", onq="orig", c3tick=False, team_tick=False, night_tick=False):
    """13차 장부 다리. c3k·c3q ∈ {orig, 1530, 1519}(실행판 청산), onq ∈ {orig, exec}. 틱 플래그는 기본·하한 비용."""
    return {"K_A": ON_.net - (ON_TICK if night_tick else 0.0),
            "K_B": C3K_.net if c3k == "orig" else c3_leg17("K_B", 1.0, {**PA, "c3exit": c3k, "c3tick": c3tick}),
            "K_C": KP_.net - (KP_TICK if team_tick else 0.0),
            "Q_A": ONQ_.net if onq == "orig" else ONQ.n - (ONQ.tick if night_tick else 0.0),
            "Q_B": C3Q_.net if c3q == "orig" else c3_leg17("Q_B", 1.0, {**PA, "c3exit": c3q, "c3tick": c3tick}),
            "Q_C": KQ_.net - (KQ_TICK if team_tick else 0.0)}


def bookA(L, qb_zero=True):
    """합산(a) = 0.5·코스피 + 0.5·코스닥, B 활성일(원 series pos ≠ 0, 실행판이 쉰 날도 포함 = 보수) C 다리 0."""
    g = {n: v.reindex(A_W).fillna(0.0) for n, v in L.items()}
    g["K_C"] = g["K_C"].where(~KB_ON, 0.0)
    if qb_zero:
        g["Q_C"] = g["Q_C"].where(~QB_ON, 0.0)
    K, Q = g["K_A"] + g["K_B"] + g["K_C"], g["Q_A"] + g["Q_B"] + g["Q_C"]
    return 0.5 * K + 0.5 * Q, K, Q, g


A_EXEC = dict(c3k="1519", c3q="1519", onq="exec")
A_BASE = dict(A_EXEC, c3tick=True, team_tick=True)
A_LOW = dict(A_BASE, night_tick=True)

# ---------------------------------------------------------------- 4. 장부 B(15차 v3) 다리
thin = lg.index[lg.index < "2018-01-01"]
V3 = dict(qa_etf_dates=thin, qa_roll_frontT=True)
THIN_TICK = pd.Series(0.0, index=q.index)
THIN_TICK[thin] = tk("233740", thin).values


def legsB(k, ex="1519", tick=True, low=False):
    kw = dict(V3, c3exit=ex, c3tick=tick)
    if low:
        kw.update(fut=FUT_S, auc_tick=True)
    L = legs(k, **kw)
    if low:
        L["Q_A"] = L["Q_A"] - k * THIN_TICK                                           # 2016~2017 233740 롱 밤(단일가 ETF)에도 1틱(W_long 만 해당)
    return L


# ---------------------------------------------------------------- 5. 출발점 재현 R1~R4
note("# 17차 실행판 장부 notes")
note()
note("## R1~R4 출발점 재현")
w13, K13, Q13, _ = bookA(legsA())
d1 = max((w13 - book13.comb_a).abs().max(), (K13 - book13.KOSPI_book).abs().max(), (Q13 - book13.KOSDAQ_book).abs().max())
m13 = met(book13.comb_a)
assert d1 <= 1e-8 and abs(m13["cagr_pct"] - 31.42) < .005 and abs(m13["mdd_pct"] + 6.10) < .005, (d1, m13)
note(f"• R1 13차: book_daily comb_a W_all CAGR {m13['cagr_pct']:.2f}% MDD {m13['mdd_pct']:.2f}% (기준 31.42 / −6.10). series 재조립 − comb_a 일수익 최대 차 {d1:.1e}(코스피·코스닥 장부 포함).")
picks = pd.read_csv(R15 / "final_v3/k_picks_v3.csv", encoding="utf-8-sig")
rep, mid = run(1.40, **V3), run(1.40, **V3, c3oc=C3OC, c3proxy=PROXY)
for tag, d, pfx in (("대표", rep, "대표"), ("중간", mid, "중간")):
    pr = picks[(picks.pick == "k1.40") & picks.bundle.str.startswith(pfx)].iloc[0]
    dm = max(abs(d[c] - pr[c]) for c in ("W_all_cagr", "W_all_mdd", "W_ex2026_cagr", "Y2024_ret", "Y2025_ret", "Y2026_ret", "W_long_cagr", "W_long_mdd"))
    assert dm < 1e-9, (tag, dm)
    note(f"• R2 15차 v3 {tag} k 1.40: W_all CAGR {d['W_all_cagr'] * 100:.2f}% MDD {d['W_all_mdd'] * 100:.2f}%, W_long {d['W_long_cagr'] * 100:.2f}% / {d['W_long_mdd'] * 100:.2f}%; k_picks_v3 와 최대 차 {dm:.1e}.")
assert round(rep["W_all_cagr"] * 100, 2) == 49.41 and round(rep["W_all_mdd"] * 100, 2) == -7.91 and round(mid["W_all_cagr"] * 100, 2) == 43.33 and round(mid["W_all_mdd"] * 100, 2) == -8.68
# R3: 16차 격자 C3K·C3Q E1·15:19 W(2022-11-23~2026-09-22)
grid = pd.read_csv(R16 / "open_close_grid/grid_cells.csv", encoding="utf-8-sig")
RAWC = {c: pd.read_csv(JH / f"{c}_daily.csv", encoding="utf-8-sig", usecols=["date", "close"]).drop_duplicates("date").assign(date=lambda d: pd.to_datetime(d.date)).set_index("date").close for c in ("069500", "122630", "229200", "233740")}
TK_RAW = {c: 5 / v.groupby(v.index.year).mean() for c, v in RAWC.items()}
for name, gl in (("K_B", "C3K"), ("Q_B", "C3Q")):
    h, c1, c2, _ = C3[name]
    cell = grid[(grid.leg == gl) & (grid.entry == "E1") & (grid.exit == "15:19") & (grid.window == "W")].iloc[0]
    a, z = pd.Timestamp(cell.start), pd.Timestamp(cell.end)
    cal_w = DAYS[(DAYS >= a) & (DAYS <= z)]
    out = {}
    for mode in ("0901", "special"):
        r = c3_leg17(name, 1.0, {**PA, "c3exit": "1519", "c3tick": True, "c3rest": mode}).loc[a:z]
        t = c3_parts(name, 1.0, "1519", mode).loc[a:z]
        act = r.index[(t.on & ~t.rest).values]
        x = t.x.reindex(act).values
        dtick = mix_tick(x, tk(c1, act).values, tk(c2, act).values) - mix_tick(x, TK_RAW[c1].reindex(act.year).values, TK_RAW[c2].reindex(act.year).values)
        mm = met(r.reindex(cal_w).fillna(0.0))
        out[mode] = dict(n=len(act), net=r.reindex(act).mean() * 1e4, net_raw=(r.reindex(act) + dtick).mean() * 1e4, cagr=mm["cagr_pct"], mdd=mm["mdd_pct"], rest=t.index[(t.on & t.rest).values])
    s_, g_ = out["0901"], out["special"]
    extra = s_["rest"].difference(g_["rest"])
    note(f"• R3 {gl} E1·15:19 {a.date()}~{z.date()}: 사양판(09:01 무거래 쉼) n {s_['n']}, 순 {s_['net']:+.2f}bp, CAGR {s_['cagr']:.2f}%, MDD {s_['mdd']:.2f}% / 격자 n {int(cell.n)}, {cell.net_bp:+.2f}bp, {cell.cagr_pct:.2f}%, {cell.mdd_pct:.2f}%. "
         f"격자 재현판(특수일만 쉼, 09:01 무거래는 다음 봉으로 밀기) n {g_['n']}, 순 {g_['net']:+.2f}bp(차 {g_['net'] - cell.net_bp:+.2f}bp), 원가격 틱이면 {g_['net_raw']:+.2f}bp(차 {g_['net_raw'] - cell.net_bp:+.2f}bp), CAGR {g_['cagr']:.2f}%, MDD {g_['mdd']:.2f}%. "
         f"사양판이 더 쉰 날: {', '.join(str(d.date()) for d in extra) or '없음'}.")
    assert g_["n"] == int(cell.n) and abs(g_["net_raw"] - cell.net_bp) < .05 and abs(g_["net"] - cell.net_bp) < .5, gl
note("  차이 원인: (1) 격자는 09:01 봉 무거래를 다음 봉으로 밀고 특수일만 뺐으나 사양은 09:01 무거래면 쉼 → 활성일 수가 다를 수 있다. (2) 격자 비용 = ETF 5bp×min(x,1) + 연속 체결 한쪽당 0.5틱(원가격 연평균)으로 기본 비용과 틱 수는 같고 틱 분모만 다르다"
     "(틀 tk 는 수정가 연평균 → 틱이 0.05~0.13bp 큼, 보수). 같은 활성일·원가격 틱이면 차 0.05bp 이내.")
# R4: 코스닥 밤 실행판 = 16차 캐시 QA_bm
C16 = pickle.load(open(CACHE16, "rb"))
assert list(SPECIAL) == list(C16["SPECIAL"]), "특수일 불일치"
qa16 = C16["L"]["QA_bm"].net
both = qa16.index.intersection(ONQ.index)
d4 = float((qa16.reindex(both).fillna(0.0) - ONQ.n.reindex(both).fillna(0.0)).abs().max())
assert d4 < 1e-12 and qa16.reindex(both).isna().equals(ONQ.n.reindex(both).isna()), d4
note(f"• R4 코스닥 밤 실행판: 활성 {int((ONQ.p.fillna(0) != 0).sum())}밤, 16차 캐시 L['QA_bm'].net 과 최대 차 {d4:.1e}(유효 구간 일치). 특수일 {len(SPECIAL)}일 일치.")

# ---------------------------------------------------------------- 6. 장부·폭포·연도·다리
ROWS, LEGS, WF, YR = [], [], [], []


def add_book(book, version, cost, subs, wl=None):
    for sub, r in subs.items():
        for w, (a, z) in WIN.items():
            ROWS.append(dict(book=book, version=version, cost=cost, sub=sub, window=w, start=a, end=z, **met(r.loc[a:z])))
        if sub == "합산":
            for y in (2023, 2024, 2025, 2026):
                YR.append(dict(book=book, version=version, cost=cost, year=y, **met(r.loc[WIN[f"Y{y}"][0]:WIN[f"Y{y}"][1]])))
    if wl is not None:
        ROWS.append(dict(book=book, version=version, cost=cost, sub="합산", window="W_long", start=str(wl.index[0].date()), end=str(wl.index[-1].date()), **met(wl)))


def add_legs(book, version, L):
    for n, r in L.items():
        for w in ("W_all", "W_val"):
            a, z = WIN[w]
            x = r.reindex(A_W).fillna(0.0).loc[a:z]
            act = x[x != 0]
            m = met(x)
            LEGS.append(dict(book=book, version=version, leg=n, window=w, n=len(act), net_bp=act.mean() * 1e4, t_nw=nw_t(act.values * 1e4), cagr_pct=m["cagr_pct"], mdd_pct=m["mdd_pct"]))


def wf_rows(book, steps):
    prev_, first = None, None
    for i, (label, wall) in enumerate(steps):
        row = dict(book=book, step=i, label=label)
        for w in ("W_all", "W_val"):
            m = met(wall.loc[WIN[w][0]:WIN[w][1]])
            row[f"{w}_cagr"], row[f"{w}_mdd"] = m["cagr_pct"], m["mdd_pct"]
        if prev_ is not None:
            for w in ("W_all", "W_val"):
                row[f"{w}_d_cagr"], row[f"{w}_d_mdd"] = row[f"{w}_cagr"] - prev_[f"{w}_cagr"], row[f"{w}_mdd"] - prev_[f"{w}_mdd"]
                row[f"{w}_d_cagr_from0"], row[f"{w}_d_mdd_from0"] = row[f"{w}_cagr"] - first[f"{w}_cagr"], row[f"{w}_mdd"] - first[f"{w}_mdd"]
        WF.append(row)
        prev_, first = row, first or row


# A 13차
A_VER = [("원형", "13차 비용", {}), ("실행판", "13차 비용", A_EXEC), ("실행판", "기본", A_BASE), ("실행판", "하한", A_LOW)]
A_OUT = {}
for ver, cost, kw in A_VER:
    L = legsA(**kw)
    w_, K_, Q_, g_ = bookA(L)
    A_OUT[(ver, cost)] = (w_, K_, Q_, g_)
    add_book("13차", ver, cost, {"합산": w_, "코스피": K_, "코스닥": Q_})
    add_legs("13차", f"{ver}·{cost}", L)
wf_rows("13차", [("원형(13차)", w13),
                 ("코스피 C3 09:01 진입(15:30 청산 유지)", bookA(legsA(c3k="1530"))[0]),
                 ("코스피 C3 15:19 청산", bookA(legsA(c3k="1519"))[0]),
                 ("코스닥 C3Q 09:01 진입(15:30 청산 유지)", bookA(legsA(c3k="1519", c3q="1530"))[0]),
                 ("코스닥 C3Q 15:19 청산", bookA(legsA(c3k="1519", c3q="1519"))[0]),
                 ("코스닥 밤 15:20 신호·15:30 단일가", A_OUT[("실행판", "13차 비용")][0]),
                 ("아침 C3·C3Q 연속 체결 1틱", bookA(legsA(**A_EXEC, c3tick=True))[0]),
                 ("팀 낮 다리 1틱(기본 비용 규약 완성)", A_OUT[("실행판", "기본")][0]),
                 ("하한(밤 ETF 단일가 다리 1틱)", A_OUT[("실행판", "하한")][0])])
# B 15차 v3 k 1.40
B_VER = [("대표", "기본(15차)", legs(1.40, **V3)), ("중간(09:01 진입, 15:30 청산)", "기본(15차)", legs(1.40, **V3, c3oc=C3OC, c3proxy=PROXY)),
         ("실행판", "기본", legsB(1.40)), ("실행판", "하한", legsB(1.40, low=True))]
B_OUT = {}
for ver, cost, L in B_VER:
    wall, wl, nanc, g_ = assemble(L)
    assert not any(nanc.values()), nanc
    B_OUT[(ver, cost)] = (wall, wl, g_)
    add_book("15차 v3", ver, cost, {"합산": wall}, wl)
    add_legs("15차 v3", f"{ver}·{cost}", L)
wf_rows("15차 v3", [("대표(k 1.40)", B_OUT[("대표", "기본(15차)")][0]),
                    ("중간(09:01 진입, 15:30 청산, 틀 근사)", B_OUT[("중간(09:01 진입, 15:30 청산)", "기본(15차)")][0]),
                    ("15:19 청산 + 09:01 무거래 쉼(틱 없음)", assemble(legsB(1.40, tick=False))[0]),
                    ("기본(아침 1틱)", B_OUT[("실행판", "기본")][0]),
                    ("하한(선물 3bp + 단일가 ETF 1틱)", B_OUT[("실행판", "하한")][0])])

# ---------------------------------------------------------------- 7. 민감도(기준 값 함께)
SENS = []


def sens(book, item, variant, wall, base):
    for w in ("W_all", "W_val"):
        a, z = WIN[w]
        m, b = met(wall.loc[a:z]), met(base.loc[a:z])
        SENS.append(dict(book=book, item=item, variant=variant, window=w, cagr_pct=m["cagr_pct"], mdd_pct=m["mdd_pct"], base_cagr_pct=b["cagr_pct"], base_mdd_pct=b["mdd_pct"],
                         d_cagr=m["cagr_pct"] - b["cagr_pct"], d_mdd=m["mdd_pct"] - b["mdd_pct"]))


baseA, baseB = A_OUT[("실행판", "기본")][0], B_OUT[("실행판", "기본")][0]
sens("13차", "s1", "아침 청산 15:30 단일가(09:02 진입, 1틱)", bookA(legsA(**dict(A_BASE, c3k="1530", c3q="1530")))[0], baseA)
sens("15차 v3", "s1", "아침 청산 15:30 단일가(09:02 진입, 1틱)", assemble(legsB(1.40, ex="1530"))[0], baseB)
sens("15차 v3", "s1", "참고: 15차 중간 묶음(틀 근사, 틱 없음)", B_OUT[("중간(09:01 진입, 15:30 청산)", "기본(15차)")][0], baseB)


def r0901(c):
    """청산일 09:01:00 첫 연속 체결가 / 09:00 단일가(09:01 봉 무거래면 1 = 단일가 그대로)."""
    t = TX[c]
    return (t.e1 / t.p0901).where(t.v0901 > 0, 1.0).reindex(CAL).fillna(1.0)


KA_0901 = pd.Series(np.where(ON_.pos != 0, (1 + ON_.gross) * r0901("069500").reindex(ON_.index) - 1 - ETF - ON_TICK, 0.0), index=ON_.index)
rq = pd.Series(np.where(ONQ.p > 0, r0901(LNG), np.where(ONQ.p < 0, r0901(SHT), 1.0)), index=CAL)
QA_0901 = ((1 + ONQ.g) * rq - 1 - ETF - ONQ.tick).where(ONQ.p != 0, 0.0)
L2 = legsA(**A_BASE)
L2["K_A"], L2["Q_A"] = KA_0901, QA_0901
sens("13차", "s2", "밤 ETF 다리 청산 09:01:00 첫 연속 체결 + 1틱(jehyeong_ON, blog_ON_Q 실행판)", bookA(L2)[0], baseA)
L2b = legsB(1.40)
L2b["K_A"] = 1.40 * KA_0901
sens("15차 v3", "s2", "코스피 밤 069500 청산 09:01:00 첫 연속 체결 + 1틱", assemble(L2b)[0], baseB)
for kk in (1.0, 1.25, 1.40):
    for lab, low in (("기본", False), ("하한", True)):
        wall, wl, _, _ = assemble(legsB(kk, low=low))
        sens("15차 v3", "s3", f"k {kk:.2f} {lab}", wall, baseB)
        ml = met(wl)
        SENS[-1].update(W_long_cagr_pct=ml["cagr_pct"], W_long_mdd_pct=ml["mdd_pct"])
        SENS[-2].update(W_long_cagr_pct=ml["cagr_pct"], W_long_mdd_pct=ml["mdd_pct"])
L4 = legsA(**A_BASE)
L4["Q_B"] = pd.Series(0.0, index=C3Q_.index)
sens("13차", "s4", "아침 C3Q 제외(KQ 는 전 활성일 사용, 사후 변형·보고만)", bookA(L4, qb_zero=False)[0], baseA)

# ---------------------------------------------------------------- 8. 저장
def fmt(df):
    return df.round(6)


fmt(pd.DataFrame(ROWS)).to_csv(OUT / "books.csv", index=False, encoding="utf-8-sig")
fmt(pd.DataFrame(LEGS)).to_csv(OUT / "legs.csv", index=False, encoding="utf-8-sig")
fmt(pd.DataFrame(WF)).to_csv(OUT / "waterfall.csv", index=False, encoding="utf-8-sig")
fmt(pd.DataFrame(YR)).to_csv(OUT / "yearly.csv", index=False, encoding="utf-8-sig")
fmt(pd.DataFrame(SENS)).to_csv(OUT / "sensitivity.csv", index=False, encoding="utf-8-sig")
wA, KA, QA, gA = A_OUT[("실행판", "기본")]
D13 = pd.DataFrame({**{n: gA[n] for n in ("K_A", "K_B", "K_C")}, "KOSPI": KA, **{n: gA[n] for n in ("Q_A", "Q_B", "Q_C")}, "KOSDAQ": QA, "comb": wA,
                    "comb_13cost": A_OUT[("실행판", "13차 비용")][0], "comb_low": A_OUT[("실행판", "하한")][0], "comb_orig13": w13})
D13.to_csv(OUT / "daily_13_exec.csv", float_format="%.8f", encoding="utf-8-sig")
wB, wlB, gB = B_OUT[("실행판", "기본")]
D15 = pd.DataFrame({**gB, "comb": wB, "comb_low": B_OUT[("실행판", "하한")][0], "comb_rep": B_OUT[("대표", "기본(15차)")][0]})
D15.to_csv(OUT / "daily_15_exec.csv", float_format="%.8f", encoding="utf-8-sig")
pd.DataFrame({"comb_long_base": wlB, "comb_long_low": B_OUT[("실행판", "하한")][1], "comb_long_rep": B_OUT[("대표", "기본(15차)")][1]}).to_csv(OUT / "daily_15_exec_long.csv", float_format="%.8f", encoding="utf-8-sig")
TRD = []
for name in C3:
    h, c1, c2, _ = C3[name]
    t = c3_parts(name, 1.0, "1519")
    for d, r in t[t.on & t.have].iterrows():
        for code, cash, e, xx, a in ((c1, r.x if r.x <= 1 else 2 - r.x, r.e1, r.x1, r.a1), (c2, max(r.x - 1, 0.0), r.e2, r.x2, r.a2)):
            if cash > 0:
                tick = float(tk(code, pd.DatetimeIndex([d])).iloc[0])
                gross = 0.0 if r.rest else cash * a
                cost = 0.0 if r.rest else cash * (ETF + tick)
                TRD.append(dict(date=d.date(), leg=h, code=code, x=r.x, cash=cash, entry_px=e, exit_px=xx, rest=bool(r.rest), push_in=bool(r.push_in), push_out=bool(r.push_out),
                                gross=gross, cost_etf5bp=0.0 if r.rest else cash * ETF, cost_tick=0.0 if r.rest else cash * tick, cost=cost, net=gross - cost))
TRD = pd.DataFrame(TRD)
TRD.round(8).to_csv(OUT / "c3_exec_trades.csv", index=False, encoding="utf-8-sig")

# ---------------------------------------------------------------- 9. notes
B = pd.DataFrame(ROWS)
W = pd.DataFrame(WF)
note()
note("## 자료")
note(f"• 13차 series(R13/series, first~last_valid, 비활성 0), book_daily comb_a 달력 {A_W[0].date()}~{A_W[-1].date()} {len(A_W)}일. 15차 틀 verify_r15_v2_review.py 를 LAB·OUT 경로 치환 뒤 exec(시나리오 격자 생략, from_review/ 에 틀 출력).")
note(f"• 분봉 TOSS 1분(봉 끝 라벨) 069500·122630·229200·233740·251340, {DAYS[0].date()}~{DAYS[-1].date()} {len(DAYS)}일(달력 끝 {CAL[-1].date()} 까지 자름, 그 뒤 분봉은 쓰지 않음). 수익은 분봉 원가격 안에서만.")
note(f"• 229200 원가격 일봉(JH) MA10·전일 종가, 16차 캐시 r16_eval_legs.pkl(QA_bm 대조·SPECIAL 대조), 16차 격자 grid_cells.csv(R3), k_picks_v3.csv(R2). 틱 = 틀 tk(5원 / 그해 평균 수정가, 251340 은 etf_extra 수정가로 같은 규약).")
note()
note("## 체결 규약·보수적 판단(사양에 없는 것)")
note("• C 다리(KP·KQ) 0 처리일 = 원 series 의 B 활성일. 실행판 아침 다리가 쉰 날(09:01 무거래)에도 C 다리를 되살리지 않는다(보수).")
note("• 15:20 봉 무거래면 15:30 마감 단일가로 청산(15:21~15:29 는 예상가 봉). 09:02 봉 무거래면 다음 거래 봉 시가. 쉰 날은 수익·비용 모두 0.")
note("• 틱 분모는 틀 tk(수정가 연평균)로 통일. 원가격 연평균보다 작아 틱이 0.05~0.13bp 크다(보수). 기본의 1틱은 연속 체결이 끼면 양쪽 합쳐 1틱 왕복(16차 사전 등록 규약, 격자의 한쪽당 0.5틱과 같은 양).")
note("• s1(09:02 진입·15:30 단일가 청산)도 1틱 왕복을 그대로 둔다(연속 체결은 한쪽만이나 사전 등록 규약을 유지, 보수).")
note("• 코스닥 밤 실행판은 16차 QA_bm 규약 그대로: 청산일 09:01 봉 무거래(늦은 개장일)면 그 밤 거래를 뺀다(실제로는 10:00 단일가 청산이 가능하나 캐시와 같게 둠).")
note(f"• 15차 2022-11-23 전 근사 e(15:30 종가/15:19:00 시가 − 1, W_all 활성·비쉼일 평균, bp): K_B 069500 {EPX['1519']['K_B'][0] * 1e4:+.2f}·122630 {EPX['1519']['K_B'][1] * 1e4:+.2f}, Q_B 229200 {EPX['1519']['Q_B'][0] * 1e4:+.2f}·233740 {EPX['1519']['Q_B'][1] * 1e4:+.2f}. "
     f"m(첫 1분, 틀 PROXY, bp): K_B {PROXY['K_B'][0] * 1e4:+.1f}·{PROXY['K_B'][1] * 1e4:+.1f}, Q_B {PROXY['Q_B'][0] * 1e4:+.1f}·{PROXY['Q_B'][1] * 1e4:+.1f}. 근사는 W_long 에만 영향.")
note("• 15차 하한에 2016~2017 233740 롱 밤(단일가 ETF 경로) 1틱을 더했다(v3 하한에는 없던 항목, W_long 만 영향). 15차 아침 초과(선물) 몫은 틱 없음.")
note()
note("## 제외일·쉼·밀림")
note(f"• 특수일(SPECIAL, 코스닥 밤 신호 제외) {len(SPECIAL)}일: {', '.join(str(d.date()) for d in SPECIAL)}.")
for name in C3:
    h, c1, c2, _ = C3[name]
    t = c3_parts(name, 1.0, "1519")
    tw = t.loc[WIN["W_all"][0]:WIN["W_all"][1]]
    on_w = tw[tw.on]
    note(f"• {h} W_all 활성 {len(on_w)}일: 쉼(09:01 무거래) {int(on_w.rest.sum())}일 [{', '.join(str(d.date()) for d in on_w.index[on_w.rest])}], "
         f"진입 밀림(09:02 무거래) {int((on_w.push_in & ~on_w.rest).sum())}건, 청산 밀림(15:20 무거래 → 15:30 단일가) {int((on_w.push_out & ~on_w.rest).sum())}건. "
         f"분봉 전 구간 활성(근사 적용, 15차 W_long 만) {int((t.on & ~t.have).sum())}일.")
note(f"• 코스닥 밤 실행판: 신호 뒤 청산일 09:01 무거래로 뺀 밤 {len(ONQ_DROP)}건 [{', '.join(ONQ_DROP)}]. 활성 {int((ONQ.p.fillna(0) != 0).sum())}밤(롱 {int((ONQ.p > 0).sum())}, 숏 {int((ONQ.p < 0).sum())}), 13차 원형 blog_ON_Q W_all 활성 {int((ONQ_.pos.reindex(A_W).fillna(0) != 0).sum())}밤.")
note("• 2026-06-08(코스피 C3 쉼): 069500·122630 의 09:01·09:02 봉이 전일 종가 수준 평탄·무거래이고 첫 체결이 09:03 봉(일봉 시가 = 09:03 봉 시가, −9% 갭). 개장 지연 또는 자료 결손으로 보고 사양대로 쉼. 16차 격자는 이 날을 다음 봉으로 밀어 넣었다(R3 의 n 99 대 100).")
note()
note("## 비용 몫(13차 실행판, W_all, 각 장부 자본 대비 누적 % / 활성일 평균 bp × 일수)")
g13, gBs, gLw = A_OUT[("실행판", "13차 비용")][3], A_OUT[("실행판", "기본")][3], A_OUT[("실행판", "하한")][3]
kq_pos = KQ_.pos.reindex(A_W).fillna(0.0)
for n_, lab, g0_, g1_ in (("K_B", "코스피 아침 C3 1틱(기본)", g13, gBs), ("K_C", "코스피 낮 KP 1틱(기본)", g13, gBs), ("Q_B", "코스닥 아침 C3Q 1틱(기본)", g13, gBs), ("Q_C", "코스닥 낮 KQ 1틱(기본)", g13, gBs),
                         ("K_A", "코스피 밤 069500 1틱(하한)", gBs, gLw), ("Q_A", "코스닥 밤 233740·251340 1틱(하한)", gBs, gLw)):
    d_ = g0_[n_] - g1_[n_]
    a_ = d_[d_ > 0]
    extra = ""
    if n_ == "Q_C":
        lo, sh = d_[kq_pos > 0], d_[kq_pos < 0]
        extra = f", 롱 229200 {lo.sum() * 100:.2f}% / {lo[lo > 0].mean() * 1e4:.1f}bp × {int((lo > 0).sum())}일, 숏 251340 {sh.sum() * 100:.2f}% / {sh[sh > 0].mean() * 1e4:.1f}bp × {int((sh > 0).sum())}일"
    note(f"• {lab}: {d_.sum() * 100:.2f}% / {a_.mean() * 1e4:.1f}bp × {len(a_)}일{extra}")
note("• 틱(5원 / 그해 평균가, bp): " + "; ".join(f"{c} " + ", ".join(f"{y} {v * 1e4:.1f}" for y, v in TKY[c].loc[2023:2026].items()) for c in ("069500", "122630", "229200", "233740", "251340")) + ".")
note("• 코스닥 낮 숏을 251340(2,300~4,900원) 1틱으로 치면 왕복 12~21bp 라 KQ 다리 순수익이 거의 사라진다. 15차 v3 는 같은 숏을 선물 대리(1.8bp + 1.06bp)로 두어 이 몫이 없다. 사양대로 두고 결과에 맞춰 바꾸지 않았다.")
note()
note("## 폭포(합산 장부, CAGR % / MDD %)")
for bk, g_ in W.groupby("book", sort=False):
    note(f"### {bk}")
    note("| 단계 | W_all CAGR | W_all MDD | ΔCAGR | ΔMDD | W_val CAGR | W_val MDD | ΔCAGR | ΔMDD |")
    note("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for _, r in g_.iterrows():
        dc = lambda k: "" if pd.isna(r.get(k, np.nan)) else f"{r[k]:+.2f}"
        note(f"| {r.label} | {r.W_all_cagr:.2f} | {r.W_all_mdd:.2f} | {dc('W_all_d_cagr')} | {dc('W_all_d_mdd')} | {r.W_val_cagr:.2f} | {r.W_val_mdd:.2f} | {dc('W_val_d_cagr')} | {dc('W_val_d_mdd')} |")
note()
note("## 실행판 장부(합산)")
note("| 장부 | 판 | 비용 | W_all | W_des | W_val | W_ex2026 | 2023 | 2024 | 2025 | 2026 | W_long |")
note("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
for (bk, ver, cost), g_ in B[B["sub"] == "합산"].groupby(["book", "version", "cost"], sort=False):
    g_ = g_.set_index("window")
    cell = lambda w, k="cagr_pct": f"{g_.loc[w, k]:.2f} / {g_.loc[w, 'mdd_pct']:.2f}" if w in g_.index else ""
    note(f"| {bk} | {ver} | {cost} | {cell('W_all')} | {cell('W_des')} | {cell('W_val')} | {cell('W_ex2026')} | {cell('Y2023', 'ret_pct')} | {cell('Y2024', 'ret_pct')} | {cell('Y2025', 'ret_pct')} | {cell('Y2026', 'ret_pct')} | {cell('W_long')} |")
note("연도 열은 그 해 누적 수익 / MDD. 코스피·코스닥 장부, 샤프, 최악일은 books.csv.")
note()
note("## 해석")
a0, a1 = W[(W.book == "13차") & (W.step == 0)].iloc[0], W[(W.book == "13차") & (W.label.str.startswith("팀 낮"))].iloc[0]
b0, b1 = W[(W.book == "15차 v3") & (W.step == 0)].iloc[0], W[(W.book == "15차 v3") & (W.label.str.startswith("기본"))].iloc[0]
note(f"• 13차 장부: 원형 {a0.W_all_cagr:.2f}% / {a0.W_all_mdd:.2f}% → 실행판 기본 {a1.W_all_cagr:.2f}% / {a1.W_all_mdd:.2f}% (W_all), 봉인 W_val {a0.W_val_cagr:.2f}% → {a1.W_val_cagr:.2f}%. 하한 {A_OUT[('실행판', '하한')][0].pipe(lambda r: met(r.loc[WIN['W_all'][0]:WIN['W_all'][1]]))['cagr_pct']:.2f}%.")
note(f"• 15차 v3 k 1.40: 대표 {b0.W_all_cagr:.2f}% / {b0.W_all_mdd:.2f}% → 실행판 기본 {b1.W_all_cagr:.2f}% / {b1.W_all_mdd:.2f}%, 봉인 W_val {b0.W_val_cagr:.2f}% → {b1.W_val_cagr:.2f}%. 하한 {met(B_OUT[('실행판', '하한')][0].loc[WIN['W_all'][0]:WIN['W_all'][1]])['cagr_pct']:.2f}%, W_long 기본 {met(wlB)['cagr_pct']:.2f}% / {met(wlB)['mdd_pct']:.2f}%.")
Lg = pd.DataFrame(LEGS)
for bk, ver0, ver1 in (("13차", "원형·13차 비용", "실행판·기본"), ("15차 v3", "대표·기본(15차)", "실행판·기본")):
    x0, x1 = Lg[(Lg.book == bk) & (Lg.version == ver0) & (Lg.window == "W_all")].set_index("leg"), Lg[(Lg.book == bk) & (Lg.version == ver1) & (Lg.window == "W_all")].set_index("leg")
    note(f"• {bk} 다리 W_all 활성일 순 bp(원형 → 실행판 기본): " + ", ".join(f"{n} {x0.loc[n, 'net_bp']:+.1f}→{x1.loc[n, 'net_bp']:+.1f}(n {x0.loc[n, 'n']}→{x1.loc[n, 'n']}, t {x1.loc[n, 't_nw']:.2f})" for n in ("K_A", "K_B", "K_C", "Q_A", "Q_B", "Q_C")) + ".")
for bk, g_ in W.groupby("book", sort=False):
    g_ = g_[g_.step > 0].copy()
    g_["abs_d"] = g_.W_all_d_cagr.abs()
    top = g_.sort_values("abs_d", ascending=False).head(3)
    note(f"• {bk} 폭포에서 W_all CAGR 변화가 큰 단계: " + "; ".join(f"{r.label} {r.W_all_d_cagr:+.2f}%p(W_val {r.W_val_d_cagr:+.2f}%p)" for _, r in top.iterrows()) + ".")
note(f"• 아침 다리의 09:01 진입은 첫 1분 변화(C3 활성일 평균 069500 {PROXY['K_B'][0] * 1e4:+.0f}bp, 229200 {PROXY['Q_B'][0] * 1e4:+.0f}bp)를 놓치는 몫이고, 15:19 청산은 마지막 11분 평균 변화(069500 {EPX['1519']['K_B'][0] * 1e4:+.0f}bp, 229200 {EPX['1519']['Q_B'][0] * 1e4:+.0f}bp)의 반대 부호만큼 움직인다. "
     "코스닥 밤 실행판은 신호 시각을 15:20 으로 당겨 16차 QA_bm 과 같은 수치다.")
note(f"• 15차 2022-11 전 근사의 e 부호가 W_long 에서 15:19 청산 효과의 방향을 정한다(실행판 기본 W_long {met(wlB)['cagr_pct']:.2f}% 대 중간 {met(B_OUT[('중간(09:01 진입, 15:30 청산)', '기본(15차)')][1])['cagr_pct']:.2f}%, 차에는 틱도 섞임). 측정값을 그대로 썼고 방향을 고르지 않았다.")
note("• 민감도(sensitivity.csv)는 기준 값과 함께 읽고 헤드라인으로 쓰지 않는다. s4 는 사후 변형이다.")
note()
note("## 한계")
note("• shortcut: 아침 실행판 초과(선물) 몫·15차 2022-11 전은 근사(1배 ETF 체결 수익, (1+oc)/(1+m)/(1+e)−1). 선물 1분봉이 전 구간 갖춰지면 실측으로 바꿀 것.")
note("• shortcut: 체결가는 봉 시가 1건 전량 체결 가정(호가 깊이·부분 체결 미반영). 호가 자료가 생기면 수량 대비 충격을 더할 것.")
note("• shortcut: 팀 장중 다리 KP·KQ 는 저장 일수익 + 1틱만 반영(자체 체결 규약은 팀 코드). 팀 코드가 바뀌면 series 재생성 뒤 다시 돌릴 것.")
note("• 모든 수치는 단일 백테스트 추정치. 설계 구간에서 고른 규칙이며 봉인 결과는 소수 극단일 의존(13차·15차 보고와 같은 단서).")
(OUT / "notes.md").write_text("\n".join(NOTES).replace("-0.00", "0.00") + "\n", encoding="utf-8")
print("저장:", OUT)
