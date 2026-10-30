"""15차 공유 구성 보정판(v2): 반박 검토(wf_16313768-4e3) major 3건을 집행 방식으로 고친 계산.

v1(r15_final_book.py) 대비 바뀐 점
  1. 코스닥 밤(blog_ON_Q) 롱 밤도 코스닥150 선물로 집행. 신호(IBS·MA10)는 15:30 종가로 확정된 뒤 15:45 선물 종가에 진입 → 종가 신호·종가 체결의 미래 정보 제거.
     롱 밤 pos +2 = 선물 명목 2, 인버스 밤 pos −1 = 명목 1. 수익 = pos × (시가_d / 종가_T − 1) − 1.8bp × |pos|, 계약 = d 의 front.
  2. 아침 C3 두 다리의 배수는 ETF 한도 안(2배 ETF 혼합, 명목 ≤ 2)만 ETF 수익으로, 그 위 초과 명목은 지수 시가→종가 수익 − 선물 1.8bp 로 계산
     (ETF 시가 괴리 몫은 선물로 늘지 않음). 혼합 = 명목 x ≤ 1: x·oc_1배, 1 < x ≤ 2: (x − 1)·oc_2배 + (2 − x)·oc_1배.
     코스피 069500/122630 + K200 지수, 코스닥 229200/233740 + KQ150 지수. 비용 = 5bp × min(x, 1) + 1.8bp × max(x − 2, 0).
     (13차 C3Q 는 명목 > 1 날을 229200 × pos 로 둠 → 여기서는 233740 혼합으로 바꿈)
  3. 호가 1틱 비용(5원 / 연평균 종가, 14차 규약)을 연속 매매 다리(낮 KP 069500/122630 혼합, 낮 KQ 롱 229200)에 전액 추가.
     단일가 경매(09:00 시가·15:30 종가)로 체결하는 밤·아침 다리에는 호가 차이 비용을 두지 않는다.
  4. 낮 숏(KQ) 선물 경로는 v1 보수판에서 실측 보정 −1.06bp × |pos| 를 더 뺀다.
나머지(B 활성일 C = 0, 합산 = 코스피 50% + 코스닥 50%, 코스피 밤·낮은 명목 k 배 scaled)는 v1 과 같다.
k_rec = W_all·W_long 둘 다 MDD ≤ 15% 최대 k, k_max = ≤ 20%(격자 0.05). 실행: PYTHONUTF8=1 python r15_final_book_v2.py
"""
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
LAB = Path("<OUT>")
R13 = LAB / "round13_slot_book"
RAW = Path("<raw>")
OUT = LAB / "round15_alpha_share" / "final_v2"
OUT.mkdir(parents=True, exist_ok=True)
FUT = 0.00018
KQ_ADJ = 0.000106
WIN = {"W_all": ("2022-12-13", "2026-09-17"), "W_ex2026": ("2022-12-13", "2025-12-30"), "Y2023": ("2023-01-02", "2023-12-28"),
       "Y2024": ("2024-01-02", "2024-12-30"), "Y2025": ("2025-01-02", "2025-12-30"), "Y2026": ("2026-01-02", "2026-09-17")}
W_LONG = ("2016-10-04", "2026-09-17")


def s13(h):
    s = pd.read_csv(R13 / "series" / f"{h}.csv", parse_dates=["date"]).set_index("date")
    m = json.loads((R13 / "series" / f"{h}.meta.json").read_text(encoding="utf-8"))
    return s.loc[m["first_valid"]:m["last_valid"]].fillna(0.0)


def px(path):
    e = pd.read_csv(path, parse_dates=["date"]).set_index("date").sort_index()
    return e


def metrics(r):
    r = r.dropna()
    eq = (1 + r).cumprod()
    n = len(r)
    cum = float(eq.iloc[-1])
    return dict(cagr=cum ** (252 / n) - 1, mdd=float((eq / eq.cummax() - 1).min()), cum=cum - 1, worst=float(r.min()),
                worst_date=str(r.idxmin().date()), sharpe=float(r.mean()) * 252 / (float(r.std(ddof=1)) * np.sqrt(252)))


cal = pd.read_csv(RAW / "etf/069500.csv", parse_dates=["date"])["date"]
prev = pd.Series(cal.shift(1).values, index=cal.values)
OC = {c: (lambda e: e["close"] / e["open"] - 1)(px(RAW / "etf" / f"{c}.csv")) for c in ("069500", "122630", "229200", "233740")}
TICK = {c: 5 / px(RAW / "etf" / f"{c}.csv")["close"].groupby(lambda d: d.year).mean() for c in ("069500", "122630", "229200", "233740")}
IDXOC = {}
for mk, f in (("K", "K200_index_kis"), ("Q", "KQ150_index_kis")):
    ix = px(RAW / "etf_extra" / f"{f}.csv")
    IDXOC[mk] = (ix["close"] / ix["open"] - 1).where(ix["open"] != ix["close"])


def tk(c, idx):
    return pd.Series(TICK[c].reindex(pd.Index(idx).year).values, index=idx)


# ------------------------------------------------------------------ 다리 자료
S = {h: s13(h) for h in ("jehyeong_ON", "beomgu_C3VT30", "jehyeong_KP", "blog_ON_Q", "beomgu_C3Q", "jehyeong_KQ")}
book = pd.read_csv(R13 / "book/book_daily.csv", parse_dates=["date"]).set_index("date")

# 1. 코스닥 밤 전부 선물
fut = pd.read_csv(RAW / "r15_external/kis_fut_kq150_daily.csv", parse_dates=["date"])
fut = fut[fut["date"] < "2026-10-30"]
fpx = fut.set_index(["date", "code"])
front = fut[fut["front"] == 1].set_index("date")["code"]
q = S["blog_ON_Q"]
qF = q.copy()
miss = 0
for d in q.index[q["pos"] != 0]:
    T, code, p = prev.get(d), front.get(d), q.at[d, "pos"]
    try:
        r = float(fpx.loc[(d, code), "open"]) / float(fpx.loc[(T, code), "close"]) - 1
    except KeyError:
        miss += 1
        continue
    qF.at[d, "gross"] = p * r
    qF.at[d, "net"] = p * r - FUT * abs(p)
lg = q.index[q["pos"] > 0]
print(f"코스닥 밤 선물 경로: 롱 밤 {len(lg)}·인버스 밤 {int((q['pos'] < 0).sum())}, 결측 {miss}; 롱 밤 평균 net 선물 {qF.loc[lg, 'net'].mean() * 1e4:+.1f}bp vs 233740 {q.loc[lg, 'net'].mean() * 1e4:+.1f}bp")

# 2. 아침 C3 혼합 함수
def mix(x, oc1, oc2, oci):
    x = np.asarray(x, dtype=float)
    r = np.where(x <= 1, x * oc1, np.where(x <= 2, (x - 1) * oc2 + (2 - x) * oc1, oc2 + (x - 2) * oci))
    return r


def c3_leg(h, c1, c2, mk, k):
    s = S[h]
    act = s["pos"] != 0
    idx = s.index
    o1, o2, oi = OC[c1].reindex(idx), OC[c2].reindex(idx), IDXOC[mk].reindex(idx)
    oi = oi.fillna(o1)                                   # 지수 결측일은 1배 ETF 로 대체(개수 보고)
    x = k * s["pos"].abs()
    g = pd.Series(mix(x, o1.values, o2.values, oi.values), index=idx).where(act, 0.0)
    cost = (0.0005 * np.minimum(x, 1) + FUT * np.maximum(x - 2, 0)).where(act, 0.0)
    return g - cost, g, cost


for h, c1, c2, mk in (("beomgu_C3VT30", "069500", "122630", "K"), ("beomgu_C3Q", "229200", "233740", "Q")):
    n1, g1, _ = c3_leg(h, c1, c2, mk, 1.0)
    s = S[h]
    act = s["pos"] != 0
    dg = ((g1 - s["gross"])[act]) * 1e4
    print(f"{h}: k1 혼합 gross − 13차 gross 평균 {dg.mean():+.2f}bp, |차| 최대 {dg.abs().max():.2f}bp, 명목>1 날 {(s['pos'][act] > 1).sum()}일, 지수 결측(활성일) {int(IDXOC[mk].reindex(s.index)[act].isna().sum())}일")

# 3·4. 낮 다리
kp = S["jehyeong_KP"].copy()
kp_tick = pd.Series(np.where(kp["pos"].abs() <= 1, kp["pos"].abs() * tk("069500", kp.index),
                             (kp["pos"].abs() - 1) * tk("122630", kp.index) + (2 - kp["pos"].abs()) * tk("069500", kp.index)), index=kp.index).where(kp["pos"] != 0, 0.0)
kq = S["jehyeong_KQ"].copy()
sh = kq["pos"] < 0
kq.loc[sh, "net"] = kq.loc[sh, "gross"] - (FUT + KQ_ADJ) * kq.loc[sh, "pos"].abs()
kq_tick = (kq["pos"].clip(lower=0) * tk("229200", kq.index)).where(kq["pos"] > 0, 0.0)


def leg_series(k, tick=True, c3mode="cap"):
    L = {}
    L["K_A"] = k * S["jehyeong_ON"]["net"]
    L["K_C"] = k * (kp["net"] - (kp_tick if tick else 0.0))
    L["Q_A"] = k * qF["net"]
    L["Q_C"] = k * (kq["net"] - (kq_tick if tick else 0.0))
    for name, h, c1, c2, mk in (("K_B", "beomgu_C3VT30", "069500", "122630", "K"), ("Q_B", "beomgu_C3Q", "229200", "233740", "Q")):
        if c3mode == "cap":
            L[name] = c3_leg(h, c1, c2, mk, k)[0]
        else:                                             # scaled(v1 방식, 비교용)
            L[name] = k * S[h]["net"]
    return L


def assemble(L):
    idx = book.index
    g = {n: s.reindex(idx).fillna(0.0) for n, s in L.items()}
    kb_on = S["beomgu_C3VT30"]["pos"].reindex(idx).fillna(0) != 0
    qb_on = S["beomgu_C3Q"]["pos"].reindex(idx).fillna(0) != 0
    assert (kb_on == (book["KOSPI_B_beomgu_C3VT30"] != 0)).mean() > 0.99
    g["K_C"] = g["K_C"].where(~kb_on, 0.0)
    g["Q_C"] = g["Q_C"].where(~qb_on, 0.0)
    wall = 0.5 * (g["K_A"] + g["K_B"] + g["K_C"] + g["Q_A"] + g["Q_B"] + g["Q_C"])
    parts = [L[n].loc[W_LONG[0]:W_LONG[1]] for n in ("K_B", "Q_A", "Q_B")]
    li = parts[0].index.intersection(parts[1].index).intersection(parts[2].index)
    wl = 0.5 * sum(p.reindex(li) for p in parts)
    return wall, wl


def row(tag, k, **kw):
    wall, wl = assemble(leg_series(k, **kw))
    out = dict(config=tag, k=k)
    for w, (a, z) in WIN.items():
        m = metrics(wall.loc[a:z])
        out[f"{w}_cagr"], out[f"{w}_mdd"], out[f"{w}_ret"] = m["cagr"], m["mdd"], m["cum"]
        if w == "W_all":
            out["W_all_sharpe"], out["W_all_worst"], out["W_all_worst_date"] = m["sharpe"], m["worst"], m["worst_date"]
    ml = metrics(wl)
    out.update(W_long_cagr=ml["cagr"], W_long_mdd=ml["mdd"], W_long_worst=ml["worst"], W_long_worst_date=ml["worst_date"], W_long_sharpe=ml["sharpe"])
    return out


CONFIGS = {"v2": dict(tick=True, c3mode="cap"), "v2_notick": dict(tick=False, c3mode="cap"), "v2_c3scaled": dict(tick=True, c3mode="scaled")}
grid = [row(t, float(k), **kw) for t, kw in CONFIGS.items() for k in np.round(np.arange(1.0, 3.0001, 0.05), 2)]
G = pd.DataFrame(grid)
G.to_csv(OUT / "k_grid_v2.csv", index=False, encoding="utf-8-sig")
picks = []
for t, g in G.groupby("config"):
    kr = g[(g.W_all_mdd >= -0.15) & (g.W_long_mdd >= -0.15)]["k"].max()
    km = g[(g.W_all_mdd >= -0.20) & (g.W_long_mdd >= -0.20)]["k"].max()
    for tag, kk in (("k1", 1.0), ("k_rec", kr), ("k_max", km), ("k1.30", 1.30), ("k1.45", 1.45)):
        r = g[np.isclose(g["k"], kk)].iloc[0].to_dict()
        r["pick"] = tag
        picks.append(r)
P = pd.DataFrame(picks)
cols = ["config", "pick", "k", "W_all_cagr", "W_all_mdd", "W_ex2026_cagr", "W_ex2026_mdd", "Y2023_ret", "Y2024_ret", "Y2025_ret", "Y2026_ret",
        "W_long_cagr", "W_long_mdd", "W_all_worst", "W_long_worst", "W_long_worst_date", "W_all_sharpe"]
P = P[cols]
P.to_csv(OUT / "k_picks_v2.csv", index=False, encoding="utf-8-sig")
pd.set_option("display.width", 260)
print(P.round(4).to_string(index=False))
