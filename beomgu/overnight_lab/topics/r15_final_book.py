"""15차 공유 구성: 13차 시간대별 장부 + 코스닥 숏 다리 코스닥150 선물 집행(보수판) + 명목 k배.

구성
  코스피 = jehyeong_ON(밤) + beomgu_C3VT30(아침) + jehyeong_KP(낮), 코스닥 = blog_ON_Q(밤) + beomgu_C3Q(아침) + jehyeong_KQ(낮).
  같은 시장에서 아침(B) 활성일엔 낮(C) 0, 합산 = 코스피 50% + 코스닥 50% 매일 재조정(13차 규칙 그대로).
  교체(코스닥 숏 다리만):
    A 밤 인버스(pos −1): 251340 대신 코스닥150 선물 front, T 15:45 종가 매도 → T+1 08:45 시가 환매, 단순수익, 왕복 1.8bp.
       계약 = T+1 의 front(자료의 front 는 만기 당일부터 차근월). KIS 선물 일봉에서 독립 재계산해 02 교체본과 대조.
    C 낮 숏(pos < 0): 원 ETF 체결 모형(다음 봉 시가) 총수익을 그대로 두고 비용만 선물 1.8bp×|pos| 로 바꾼다.
       02 교체본(신호 봉 가격 체결)은 체결 지연 몫을 빼서 낙관이므로 보수판을 대표로 쓴다.
  현실 기준선(ETF 그대로): 251340 거래마다 연도별 틱 비용(2024 13.6, 2025 14.2, 2026 21.5, 그 외 10bp) 추가 차감(02 tick 시리즈).
레버리지
  scaled: k×net(비용도 k배, 대표·보수). fixed: k·gross − cost(참고, 낙관).
  noC3: 아침 C3 두 다리는 k=1 유지(ETF 시가 괴리 몫은 선물로 늘릴 수 없다는 보수 가정), 나머지 다리만 k배.
  k_rec = W_all·W_long 둘 다 MDD ≤ 15% 인 최대 k, k_max = 둘 다 MDD ≤ 20% 인 최대 k(격자 0.05).
단일가 가정(ETF 09:00·15:30, 선물 08:45·15:45 체결)은 13차와 같다. 원본 수정 없음.
실행: PYTHONUTF8=1 python r15_final_book.py
"""
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.stdout.reconfigure(encoding="utf-8")
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

LAB = Path("<OUT>")
R13 = LAB / "round13_slot_book"
R15 = LAB / "round15_alpha_share"
RAW = Path("<raw>")
OUT = R15 / "final"
OUT.mkdir(parents=True, exist_ok=True)

FUT_COST = 0.00018
WIN = {"W_all": ("2022-12-13", "2026-09-17"), "W_ex2026": ("2022-12-13", "2025-12-30"),
       "Y2023": ("2023-01-02", "2023-12-28"), "Y2024": ("2024-01-02", "2024-12-30"),
       "Y2025": ("2025-01-02", "2025-12-30"), "Y2026": ("2026-01-02", "2026-09-17")}
W_LONG = ("2016-10-04", "2026-09-17")
LEGS = {"KOSPI_A_jehyeong_ON": "jehyeong_ON", "KOSPI_B_beomgu_C3VT30": "beomgu_C3VT30", "KOSPI_C_jehyeong_KP": "jehyeong_KP",
        "KOSDAQ_A_blog_ON_Q": "blog_ON_Q", "KOSDAQ_B_beomgu_C3Q": "beomgu_C3Q", "KOSDAQ_C_jehyeong_KQ": "jehyeong_KQ"}
C3 = ("KOSPI_B_beomgu_C3VT30", "KOSDAQ_B_beomgu_C3Q")


def load(path, meta_path=None):
    s = pd.read_csv(path, parse_dates=["date"]).set_index("date")
    if meta_path is not None and Path(meta_path).exists():
        m = json.loads(Path(meta_path).read_text(encoding="utf-8"))
        s = s.loc[m["first_valid"]:m["last_valid"]]
    return s.fillna(0.0)


def s13(h):
    return load(R13 / "series" / f"{h}.csv", R13 / "series" / f"{h}.meta.json")


def s15(h):
    return load(R15 / "series" / f"{h}.csv", R15 / "series" / f"{h}.meta.json")


def metrics(r):
    r = r.dropna()
    eq = (1 + r).cumprod()
    n = len(r)
    cum = float(eq.iloc[-1])
    vol = float(r.std(ddof=1)) * np.sqrt(252)
    return dict(n_days=n, cagr=cum ** (252 / n) - 1 if cum > 0 else -1.0, mdd=float((eq / eq.cummax() - 1).min()),
                sharpe=float(r.mean()) * 252 / vol if vol else np.nan, worst_day=float(r.min()),
                worst_date=str(r.idxmin().date()), cum=cum - 1)


# ------------------------------------------------------------------ 1. 13차 장부 재현
book = pd.read_csv(R13 / "book/book_daily.csv", parse_dates=["date"]).set_index("date")
legs_net = {c: book[c].copy() for c in LEGS}
legs_cost = {}
for c, h in LEGS.items():
    s = s13(h).reindex(book.index).fillna(0.0)
    act = book[c] != 0
    assert np.allclose(book[c][act], s["net"][act]), c
    legs_cost[c] = (s["gross"] - s["net"]).where(act, 0.0)
clashQ = book["KOSDAQ_clash"] != 0


def assemble(net, cost=None, k=1.0, mode="scaled", lev_c3=True):
    out = {}
    for mkt in ("KOSPI", "KOSDAQ"):
        tot = 0.0
        for c in LEGS:
            if not c.startswith(mkt):
                continue
            kk = k if (lev_c3 or c not in C3) else 1.0
            tot = tot + (kk * net[c] if mode == "scaled" else kk * net[c] + (kk - 1) * cost[c])
        out[mkt] = tot
    out["comb"] = 0.5 * (out["KOSPI"] + out["KOSDAQ"])
    return out


b0 = assemble(legs_net, legs_cost)
m0 = metrics(b0["comb"].loc[WIN["W_all"][0]:WIN["W_all"][1]])
assert abs(m0["cagr"] - 0.3142) < 5e-4 and abs(m0["mdd"] + 0.0610) < 5e-4, m0
print(f"13차 재현 W_all {m0['cagr']:.4f} / {m0['mdd']:.4f}")

# ------------------------------------------------------------------ 2. 밤 인버스 선물 경로 독립 재계산
cal = pd.read_csv(RAW / "etf/069500.csv", parse_dates=["date"])["date"]
prev = pd.Series(cal.shift(1).values, index=cal.values)
fut = pd.read_csv(RAW / "r15_external/kis_fut_kq150_daily.csv", parse_dates=["date"])
fut = fut[fut["date"] < "2026-10-30"]
px = fut.set_index(["date", "code"])
front = fut[fut["front"] == 1].set_index("date")["code"]
q = s13("blog_ON_Q")
qf = s15("blog_ON_Q_fut")
inv = q.index[q["pos"] < 0]
rows = []
for d in inv:
    T = prev.get(d)
    code = front.get(d)
    try:
        c_T = float(px.loc[(T, code), "close"])
        o_d = float(px.loc[(d, code), "open"])
    except KeyError:
        rows.append((d, T, code, np.nan, np.nan))
        continue
    g = 1 - o_d / c_T                      # 숏 단순수익
    rows.append((d, T, code, g, g - FUT_COST))
chk = pd.DataFrame(rows, columns=["date", "T", "code", "gross_me", "net_me"]).set_index("date")
chk["net_02"] = qf["net"].reindex(chk.index)
chk["diff_bp"] = (chk["net_me"] - chk["net_02"]) * 1e4
print(f"인버스 밤 {len(chk)}건, 재계산 결측 {chk['net_me'].isna().sum()}, |차| 평균 {chk['diff_bp'].abs().mean():.2f}bp 최대 {chk['diff_bp'].abs().max():.2f}bp (02 는 로그수익, 여기는 단순수익)")
chk.to_csv(OUT / "check_inverse_nights_fut.csv", encoding="utf-8-sig")
qA = q.copy()
qA.loc[chk.index, "gross"] = chk["gross_me"]
qA.loc[chk.index, "net"] = chk["net_me"]

# ------------------------------------------------------------------ 3. 낮 숏 보수판
kq = s13("jehyeong_KQ")
kqC = kq.copy()
sh = kq.index[kq["pos"] < 0]
kqC.loc[sh, "net"] = kq.loc[sh, "gross"] - FUT_COST * kq.loc[sh, "pos"].abs()
kqF = s15("jehyeong_KQ_fut")
d_opt = ((kqF.loc[sh, "net"] - kqC.loc[sh, "net"]) * 1e4)
print(f"낮 숏 {len(sh)}일: 02 교체본 − 보수판 평균 {d_opt.mean():+.2f}bp/일 (체결 지연 몫), 보수판 − 원 평균 {((kqC.loc[sh,'net']-kq.loc[sh,'net'])*1e4).mean():+.2f}bp/일")


def put(net, cost, col, s):
    s = s.reindex(book.index).fillna(0.0)
    keep = ~clashQ if col == "KOSDAQ_C_jehyeong_KQ" else pd.Series(True, index=book.index)
    net[col] = s["net"].where(keep, 0.0)
    cost[col] = (s["gross"] - s["net"]).where(keep & (s["pos"] != 0), 0.0)


variants = {}
for name, A, Cs in (("orig", None, None), ("tick", s15("blog_ON_Q_tick"), s15("jehyeong_KQ_tick")),
                    ("fut02", qf, kqF), ("fut_cons", qA, kqC)):
    n, c = dict(legs_net), dict(legs_cost)
    if A is not None:
        put(n, c, "KOSDAQ_A_blog_ON_Q", A)
        put(n, c, "KOSDAQ_C_jehyeong_KQ", Cs)
    variants[name] = (n, c, A if A is not None else q)
# 교체 안 한 행은 원본과 같아야 한다
nC, _, _ = variants["fut_cons"]
same = book["KOSDAQ_C_jehyeong_KQ"].index.difference(sh)
assert np.allclose(nC["KOSDAQ_C_jehyeong_KQ"].reindex(same), book["KOSDAQ_C_jehyeong_KQ"].reindex(same))


def long_book(A, k=1.0, mode="scaled", lev_c3=True):
    parts = {}
    for col, s in (("KOSPI_B_beomgu_C3VT30", s13("beomgu_C3VT30")), ("KOSDAQ_A_blog_ON_Q", A), ("KOSDAQ_B_beomgu_C3Q", s13("beomgu_C3Q"))):
        s = s.loc[W_LONG[0]:W_LONG[1]]
        kk = k if (lev_c3 or col not in C3) else 1.0
        parts[col] = kk * s["net"] if mode == "scaled" else kk * s["net"] + (kk - 1) * (s["gross"] - s["net"])
    idx = parts["KOSPI_B_beomgu_C3VT30"].index.intersection(parts["KOSDAQ_A_blog_ON_Q"].index).intersection(parts["KOSDAQ_B_beomgu_C3Q"].index)
    return 0.5 * (parts["KOSPI_B_beomgu_C3VT30"].reindex(idx) + parts["KOSDAQ_A_blog_ON_Q"].reindex(idx) + parts["KOSDAQ_B_beomgu_C3Q"].reindex(idx))


def row(name, k, mode, lev_c3):
    n, c, A = variants[name]
    b = assemble(n, c, k, mode, lev_c3)
    out = dict(config=name, k=k, mode=mode, lev_c3=lev_c3)
    for w, (a, z) in WIN.items():
        m = metrics(b["comb"].loc[a:z])
        out[f"{w}_cagr"], out[f"{w}_mdd"], out[f"{w}_ret"] = m["cagr"], m["mdd"], m["cum"]
        if w == "W_all":
            out["W_all_sharpe"], out["W_all_worst"], out["W_all_worst_date"] = m["sharpe"], m["worst_day"], m["worst_date"]
    ml = metrics(long_book(A, k, mode, lev_c3))
    out.update(W_long_cagr=ml["cagr"], W_long_mdd=ml["mdd"], W_long_sharpe=ml["sharpe"], W_long_worst=ml["worst_day"], W_long_worst_date=ml["worst_date"])
    return out


base = pd.DataFrame([row(v, 1.0, "scaled", True) for v in variants])
base.to_csv(OUT / "base_k1.csv", index=False, encoding="utf-8-sig")
pd.set_option("display.width", 250)
show = ["config", "W_all_cagr", "W_all_mdd", "W_ex2026_cagr", "W_ex2026_mdd", "Y2023_ret", "Y2024_ret", "Y2025_ret", "Y2026_ret", "W_long_cagr", "W_long_mdd", "W_all_sharpe"]
print(base[show].round(4).to_string(index=False))

# ------------------------------------------------------------------ 4. 레버리지 격자
grid = []
for k in np.round(np.arange(1.0, 3.0001, 0.05), 2):
    for mode in ("scaled", "fixed"):
        for lev_c3 in (True, False):
            grid.append(row("fut_cons", float(k), mode, lev_c3))
    grid.append(row("tick", float(k), "scaled", True))
G = pd.DataFrame(grid)
G.to_csv(OUT / "k_grid.csv", index=False, encoding="utf-8-sig")


def kpick(g, lim):
    ok = g[(g["W_all_mdd"] >= -lim) & (g["W_long_mdd"] >= -lim)]
    return float(ok["k"].max()) if len(ok) else np.nan


summ = []
for (cfg, mode, lc), g in G.groupby(["config", "mode", "lev_c3"]):
    kr, km = kpick(g, 0.15), kpick(g, 0.20)
    for tag, kk in (("k1", 1.0), ("k_rec", kr), ("k_max", km)):
        if np.isnan(kk):
            continue
        r = g[np.isclose(g["k"], kk)].iloc[0].to_dict()
        r["pick"] = tag
        summ.append(r)
S = pd.DataFrame(summ)
cols = ["config", "mode", "lev_c3", "pick", "k", "W_all_cagr", "W_all_mdd", "W_ex2026_cagr", "W_ex2026_mdd", "Y2023_ret", "Y2024_ret", "Y2025_ret", "Y2026_ret", "Y2024_mdd", "Y2025_mdd",
        "W_long_cagr", "W_long_mdd", "W_all_sharpe", "W_all_worst", "W_long_worst", "W_long_worst_date"]
S = S[cols]
S.to_csv(OUT / "k_picks.csv", index=False, encoding="utf-8-sig")
print(S.round(4).to_string(index=False))

# 기준 충족(CAGR ≥ 30 ∧ MDD ≤ 20) 창별
crit = []
for _, r in S.iterrows():
    for w in ("W_all", "W_ex2026", "W_long"):
        crit.append(dict(config=r["config"], mode=r["mode"], lev_c3=r["lev_c3"], pick=r["pick"], k=r["k"], window=w,
                         cagr=r[f"{w}_cagr"], mdd=r[f"{w}_mdd"], meets=bool(r[f"{w}_cagr"] >= 0.30 and r[f"{w}_mdd"] >= -0.20)))
pd.DataFrame(crit).to_csv(OUT / "criteria.csv", index=False, encoding="utf-8-sig")

# ------------------------------------------------------------------ 5. 그림
fig, ax = plt.subplots(1, 2, figsize=(14, 5))
for name, lab in (("orig", "13차 원수치(251340 5bp 가정)"), ("tick", "ETF 현실(251340 틱 비용)"), ("fut_cons", "코스닥 숏 선물 집행(보수)")):
    n, c, A = variants[name]
    b = assemble(n, c)["comb"].loc[WIN["W_all"][0]:WIN["W_all"][1]]
    ax[0].plot((1 + b).cumprod(), label=lab)
kr = S[(S.config == "fut_cons") & (S["mode"] == "scaled") & (S.lev_c3) & (S.pick == "k_rec")]["k"]
if len(kr):
    n, c, A = variants["fut_cons"]
    b = assemble(n, c, float(kr.iloc[0]))["comb"].loc[WIN["W_all"][0]:WIN["W_all"][1]]
    ax[0].plot((1 + b).cumprod(), label=f"선물 집행 × k {float(kr.iloc[0]):.2f}", ls="--")
ax[0].set_title("합산 장부 W_all 2022-12-13~2026-09-17 누적")
ax[0].legend()
g = G[(G.config == "fut_cons") & (G["mode"] == "scaled") & (G.lev_c3)]
ax[1].plot(-g["W_all_mdd"] * 100, g["W_all_cagr"] * 100, marker="o", ms=3, label="W_all")
ax[1].plot(-g["W_long_mdd"] * 100, g["W_long_cagr"] * 100, marker="o", ms=3, label="W_long 2016-10~(두 장부)")
ax[1].axhline(30, color="gray", lw=0.8)
ax[1].axvline(20, color="gray", lw=0.8)
ax[1].set_xlabel("MDD %")
ax[1].set_ylabel("CAGR %")
ax[1].set_title("선물 집행 장부 × k (scaled, 격자 0.05)")
ax[1].legend()
fig.tight_layout()
fig.savefig(OUT / "fig_final.png", dpi=110)
print("저장:", OUT)
