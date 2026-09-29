"""R2 · D4 확인: 코스피200 선물로 오버나잇·종가 베팅 집행 (총수익·손익분기 비용)

사전 고정 사양(2026-09-29 동결, 탐색 담당 작성): <OUT>/round2/D4/spec.json. 이 스크립트는 사양을 바꾸지 않고 구현만 한다.
구간: A = 2014-07-08 ~ 2019-12-31 (표본 내 병기. 오버나잇은 청산일 <= 2019-12-31 인 밤만)
      B = 결정일 T in [2020-01-01, 2025-06-05] (확인)
      B 하위: 2020-01-01 ~ 2023-07-28 (선물 시가 09:00) / 2023-07-31 ~ 2025-06-05 (08:45)
      E4 = 2025-06-09 ~ (보고만)
공통: FUT = 네이버 fchart 'FUT' 연결 일봉(정규장만, 근월물을 만기일까지, 백조정 없음).
      분기 만기일(3·6·9·12월 둘째 목요일, 휴장이면 직전 거래일)과 만기 다음 거래일은 거래·표본 제외(t07_basis.expiries).
      선물 왕복 비용(bp) = 수수료 + k × 0.05 / C_F(T) × 1e4. 기본 0.3bp+1틱, 낙관 0.3bp+0.5틱, 스트레스 0.6bp+2틱.
      KODEX 200(069500) 왕복 5bp(7bp).

셀
- D4-K6F (알파): 신호 = t06.us_signal(FUT 날짜, 60, 0.7, 0.3)['H-rev']
  (T 직전 미국 세션 S&P500 종가→종가 < 그 세션 직전 60세션 30분위). 목표 r_oc_F(T) = C_F(T)/O_F(T) − 1
  (T 시가 단일가 매수 → 같은 날 종가 단일가 매도). 이웃: 창40·창120·q20(창60). 참고: KODEX r_oc 같은 규칙.
- D4-UF (실행 기준선, 알파 아님): 모든 거래일 r_on_F(T) = O_F(T+1)/C_F(T) − 1. 비교 KODEX 같은 밤.
  쌍차 d = r_on_F − r_on_E, 순수익 차 = (r_on_F − 0.3bp − 1틱) − (r_on_E − 5bp).
  이웃: 평일 밤 / 주말·연휴 밤 / 비용 0.5틱·2틱 / KODEX 같은 밤 5bp.

확인 판정(셀별, B): ① Δ > 0 (NW lag5 단측 원 p 보고, Holm 보정은 2차 종합에서) ② 7bp 차감 순평균 > 0
③ 연도 층화 플라시보 p <= 0.05 ④ 변동성 조정 Δ > 0 ⑤ 이웃 부호 동일. 사양 자체의 B 기준도 따로 판정한다.
모호한 점은 가장 보수적인 쪽으로 해석하고 report.md '해석 선택' 절에 적는다.

실행: cwd = overnight_lab, python topics/r2_d4.py
산출: <OUT>/round2/D4/report.md, verdict.json, 시행 장부 <OUT>/ledger.csv (topic R2_D4)
"""
import datetime as dt
import hashlib
import sys
from math import erf, sqrt
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import common as C  # noqa: E402
import data  # noqa: E402
import engine as E  # noqa: E402
from t06_us_close_to_open_entry import us_signal  # noqa: E402
from t07_basis import expiries  # noqa: E402

TOPIC = "R2_D4"
OUT = data.OUT / "round2" / "D4"
SPEC_FILE = data.spec_file("D4")   # 출력 폴더에 없으면 저장소 결과 폴더의 사양을 읽음
A0, A1 = "2014-07-08", "2019-12-31"
B0, B1 = "2020-01-01", E.JUDGE_END
B1_END, B2_START = "2023-07-28", "2023-07-31"
E4_0 = E.E4_START
FEE, FEE_STRESS, TICK = 0.3, 0.6, 0.05
N_CELLS_R2 = 8                                   # 2차 전체 셀 수 상한(Holm 은 종합에서)
WLAB = {"A": "A(2014-07-08~2019)", "B": "B(2020~2025-06-05)", "B1": "B1(2020~2023-07-28, 시가 09:00)",
        "B2": "B2(2023-07-31~2025-06-05, 시가 08:45)", "E4": "E4(2025-06-09~, 보고)"}


def p_one(t):
    return np.nan if t is None or np.isnan(t) else 0.5 * (1 - erf(t / sqrt(2)))


def f2(x, fmt="{:+.2f}"):
    return "-" if x is None or (isinstance(x, (float, np.floating)) and np.isnan(x)) else fmt.format(x)


def ox(b):
    return "O" if b else "X"


# ---------------------------------------------------------------- 자료
def load():
    f = data.price("FUT")
    e_all = data.price("069500")
    k_all = data.price("KPI200")
    idx = f.index
    rng = (e_all.index >= idx.min()) & (e_all.index <= idx.max())
    chk = {"fut_rows": len(idx), "fut_first": str(idx.min().date()), "fut_last": str(idx.max().date()),
           "kodex_only_dates": [str(d.date()) for d in e_all.index[rng].difference(idx)],
           "fut_only_vs_kodex": [str(d.date()) for d in idx.difference(e_all.index)],
           "fut_only_vs_k200": [str(d.date()) for d in idx.difference(k_all.index)]}
    e, k = e_all.reindex(idx), k_all.reindex(idx)
    ex = expiries(idx)
    pos = pd.Series(np.arange(len(idx)), index=idx)
    bad = pd.Series(False, index=idx)
    for d in ex:
        i = pos[d]
        bad.iloc[i:i + 2] = True                  # 만기일 + 만기 다음 거래일
    return f, e, k, bad, ex, chk


def windows(idx, leg):
    """결정일 T 기준 창. 오버나잇(leg='on')의 A 는 청산일(T+1) <= 2019-12-31 인 밤만."""
    nxt = pd.Series(list(idx[1:]) + [pd.NaT], index=idx)
    a_end = nxt[nxt <= pd.Timestamp(A1)].index.max() if leg == "on" else pd.Timestamp(A1)
    return {"A": (pd.Timestamp(A0), a_end), "B": (pd.Timestamp(B0), pd.Timestamp(B1)),
            "B1": (pd.Timestamp(B0), pd.Timestamp(B1_END)), "B2": (pd.Timestamp(B2_START), pd.Timestamp(B1)),
            "E4": (pd.Timestamp(E4_0), idx.max())}


def vol_adj_valid(ret, win=60):
    """변동성 조정: T 를 뺀 과거 win 개 유효일(만기 제외일을 뺀 표본)의 SD 로 나눈다."""
    v = ret.dropna()
    sd = v.shift(1).rolling(win, min_periods=win).std()
    return (v / sd).reindex(ret.index)


def perf(pos, ret, cost):
    """engine.stats 와 같은 정의에 날짜별 비용(bp, 스칼라 또는 Series)을 허용한다. 롱 전용 0/1."""
    d = pd.DataFrame({"p": pos, "r": ret}).dropna()
    c = pd.Series(float(cost), index=d.index) if np.isscalar(cost) else cost.reindex(d.index)
    tr = d["p"] != 0
    daily = d["p"] * d["r"] - tr * c / 1e4
    trd = daily[tr]
    if len(d) == 0:
        return {"days": 0, "trades": 0}
    eq = (1 + daily).cumprod()
    years = len(d) / E.ANN
    cagr = eq.iloc[-1] ** (1 / years) - 1 if eq.iloc[-1] > 0 else np.nan
    mdd = (eq / eq.cummax() - 1).min()
    sd = daily.std()
    return {"days": len(d), "trades": int(tr.sum()), "exposure": tr.mean(), "gross_bp": (d["p"] * d["r"])[tr].mean() * 1e4,
            "net_bp": trd.mean() * 1e4, "t_net": E.nw_t(trd), "win": (trd > 0).mean(),
            "sharpe": daily.mean() / sd * np.sqrt(E.ANN) if sd > 0 else np.nan, "cagr": cagr, "mdd": mdd,
            "calmar": cagr / -mdd if mdd < 0 else np.nan, "cost_bp": c[tr].mean()}


# ---------------------------------------------------------------- D4-K6F
def k6f_block(sig, r, rv, tick, a, b, neigh=None, rE=None, gap=None, placebo=True):
    d = pd.DataFrame({"s": sig.loc[a:b], "r": r.loc[a:b]}).dropna()
    D, t, n = E.delta_test(d["s"], d["r"])
    h = d["r"][d["s"] == 1] * 1e4
    tk = tick.reindex(h.index)
    o = {"days": len(d), "n_sig": n, "sig_bp": h.mean(), "non_bp": d["r"][d["s"] == 0].mean() * 1e4, "all_bp": d["r"].mean() * 1e4,
         "delta_bp": D * 1e4, "t": t, "p_one": p_one(t), "sig_median_bp": h.median(),
         "net5_bp": (h - 5).mean(), "net7_bp": (h - 7).mean(), "t_net7": E.nw_t(h - 7),
         "net_f05_bp": (h - FEE - 0.5 * tk).mean(), "net_f1_bp": (h - FEE - tk).mean(), "t_net_f1": E.nw_t(h - FEE - tk),
         "net_f2s_bp": (h - FEE_STRESS - 2 * tk).mean(), "tick_bp": tk.mean(), "be_ticks": (h.mean() - FEE) / tk.mean(),
         "dvol": E.delta_test(d["s"], rv.reindex(d.index))[0]}
    o["placebo_p"] = E.placebo_year_strat(d["s"], d["r"]) if placebo else np.nan
    if neigh:
        o["neighbors"] = {}
        for k_, v in neigh.items():
            dn, tn, nn = E.delta_test(v.loc[a:b], r.loc[a:b])
            o["neighbors"][k_] = {"delta_bp": dn * 1e4, "t": tn, "n_sig": nn}
    if rE is not None:
        dE = pd.DataFrame({"s": sig.loc[a:b], "r": rE.loc[a:b]}).dropna()
        DE, tE, nE = E.delta_test(dE["s"], dE["r"])
        hE = dE["r"][dE["s"] == 1] * 1e4
        o["kodex"] = {"delta_bp": DE * 1e4, "t": tE, "n_sig": nE, "sig_bp": hE.mean(), "all_bp": dE["r"].mean() * 1e4,
                      "net5_bp": (hE - 5).mean(), "net7_bp": (hE - 7).mean()}
    if gap is not None:
        X = pd.DataFrame({"signal": d["s"], "gap": gap.reindex(d.index)})
        reg, nreg = C.hac_reg(d["r"], X)
        o["gap_reg"] = {"signal_bp": reg.loc["signal", "coef"] * 1e4, "t_signal": reg.loc["signal", "t"],
                        "gap_coef": reg.loc["gap", "coef"], "t_gap": reg.loc["gap", "t"], "n": nreg,
                        "gapdown_share_sig": (gap.reindex(h.index) < 0).mean(),
                        "gapdown_share_non": (gap.reindex(d.index[d["s"] == 0]) < 0).mean()}
    return o


def yearly_delta(sig, r, a, b):
    d = pd.DataFrame({"s": sig.loc[a:b], "r": r.loc[a:b]}).dropna()
    yrs = sorted(set(d.index.year))
    by = {int(y): E.delta_test(d["s"][d.index.year == y], d["r"][d.index.year == y])[0] * 1e4 for y in yrs}
    loo = {}
    for y in yrs:
        m = d.index.year != y
        D, t, _ = E.delta_test(d["s"][m], d["r"][m])
        loo[int(y)] = {"delta_bp": D * 1e4, "t": t}
    return by, loo


# ---------------------------------------------------------------- D4-UF
def uf_block(rF, rE, rv, tick, wk, logdec, a, b):
    d = (pd.DataFrame({"F": rF.loc[a:b], "E": rE.loc[a:b]}).dropna() * 1e4)
    F, Eo = d["F"], d["E"]
    tk = tick.reindex(d.index)
    w = wk.reindex(d.index).astype(bool)
    nf1 = F - FEE - tk
    ndiff = nf1 - (Eo - 5)
    ndiff_s = (F - FEE_STRESS - 2 * tk) - (Eo - 7)
    tF = E.nw_t(F)
    ld = logdec.reindex(d.index).dropna()
    o = {"nights": len(d), "F_bp": F.mean(), "F_sd": F.std(), "t_F": tF, "p_one": p_one(tF), "F_median": F.median(), "F_up": (F > 0).mean(),
         "E_bp": Eo.mean(), "E_sd": Eo.std(), "t_E": E.nw_t(Eo), "d_bp": (F - Eo).mean(), "t_d": E.nw_t(F - Eo), "corr": F.corr(Eo),
         "tick_bp": tk.mean(), "cost_f1_bp": (FEE + tk).mean(),
         "netF05_bp": (F - FEE - 0.5 * tk).mean(), "netF1_bp": nf1.mean(), "t_netF1": E.nw_t(nf1), "netF2s_bp": (F - FEE_STRESS - 2 * tk).mean(),
         "netF5_bp": (F - 5).mean(), "netF7_bp": (F - 7).mean(), "t_netF7": E.nw_t(F - 7),
         "netE5_bp": (Eo - 5).mean(), "t_netE5": E.nw_t(Eo - 5), "netE7_bp": (Eo - 7).mean(),
         "netdiff_bp": ndiff.mean(), "t_netdiff": E.nw_t(ndiff), "netdiff_stress_bp": ndiff_s.mean(), "t_netdiff_stress": E.nw_t(ndiff_s),
         "be_bp": F.mean(), "be_ticks": (F.mean() - FEE) / tk.mean(), "be_E_bp": Eo.mean(),
         "dvol": rv.reindex(d.index).mean(),
         "wk_n": int(w.sum()), "wk_F_bp": F[w].mean(), "wk_t": E.nw_t(F[w]), "wk_E_bp": Eo[w].mean(),
         "we_n": int((~w).sum()), "we_F_bp": F[~w].mean(), "we_t": E.nw_t(F[~w]), "we_E_bp": Eo[~w].mean(),
         "basis_chg_bp": ld["basis_chg"].mean() * 1e4, "t_basis_chg": E.nw_t(ld["basis_chg"]),
         "k200_on_log_bp": ld["k_on"].mean() * 1e4, "f_on_log_bp": ld["f_on"].mean() * 1e4, "n_logdec": len(ld)}
    return o


# ---------------------------------------------------------------- 성과표·장부
def perf_rows(cell, asset, rule, pos, ret, costs, wins, with_delta=True):
    rows = []
    for wk_, (a, b) in wins.items():
        p, r = pos.loc[a:b], ret.loc[a:b]
        extra = {}
        if with_delta:
            dd, tt, _ = E.delta_test((p != 0).astype(float).where(p.notna()), r)
            extra = {"delta_bp": dd * 1e4, "t_delta": tt, "rand_pct": E.random_pct(p, r, 0.0)}
        for clab, c in costs.items():
            s = perf(p, r, c if np.isscalar(c) else c.loc[a:b])
            s.update(rule=f"{cell} {rule} [{clab}]", window=WLAB[wk_], asset=asset, **extra)
            rows.append(s)
    return rows


def md_table(df, cols, heads):
    out = ["| " + " | ".join(heads) + " |", "|" + "---|" * len(heads)]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r.get(c, np.nan)
            if isinstance(v, str):
                cells.append(v)
            elif c in ("win", "cagr", "mdd", "exposure"):
                cells.append(f2(v * 100 if v == v else v, "{:.1f}%"))
            elif c in ("trades", "days"):
                cells.append(f"{int(v)}")
            elif c == "sharpe":
                cells.append(f2(v, "{:+.2f}"))
            else:
                cells.append(f2(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


# ---------------------------------------------------------------- 실행
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    spec_bytes = SPEC_FILE.read_bytes()
    spec_sha = hashlib.sha256(spec_bytes.replace(bytes([13, 10]), bytes([10]))).hexdigest()   # 줄바꿈(CRLF→LF) 무관
    f, e, k, bad, ex, chk = load()
    idx = f.index
    tick = TICK / f["close"] * 1e4                                    # 1틱 bp, C_F(T) 기준(두 셀 공통)
    cost_f = {"0.3bp+0.5틱": FEE + 0.5 * tick, "0.3bp+1틱": FEE + tick, "0.6bp+2틱": FEE_STRESS + 2 * tick}

    # ---- D4-K6F
    r_oc = (f["close"] / f["open"] - 1).where(~bad)
    r_ocE = (e["close"] / e["open"] - 1).where(~bad)
    sig = us_signal(idx, 60, 0.7, 0.3)["H-rev"].where(~bad)
    neigh = {"창40": us_signal(idx, 40, 0.7, 0.3)["H-rev"].where(~bad),
             "창120": us_signal(idx, 120, 0.7, 0.3)["H-rev"].where(~bad),
             "q20(창60)": us_signal(idx, 60, 0.8, 0.2)["H-rev"].where(~bad)}
    rv_oc = vol_adj_valid(r_oc)
    gap = f["open"] / f["close"].shift(1) - 1
    Woc = windows(idx, "oc")
    K = {w: k6f_block(sig, r_oc, rv_oc, tick, a, b, neigh=neigh, rE=r_ocE, gap=gap) for w, (a, b) in Woc.items()}
    by_B, loo_B = yearly_delta(sig, r_oc, *Woc["B"])
    by_A, _ = yearly_delta(sig, r_oc, *Woc["A"])
    by_E4, _ = yearly_delta(sig, r_oc, *Woc["E4"])

    kb = K["B"]
    nb_same = all(np.sign(v["delta_bp"]) == np.sign(kb["delta_bp"]) for v in kb["neighbors"].values())
    crit_k = {"①Δ>0": kb["delta_bp"] > 0, "②7bp순>0": kb["net7_bp"] > 0, "③플라시보≤0.05": kb["placebo_p"] <= 0.05,
              "④변동성조정Δ>0": kb["dvol"] > 0, "⑤이웃부호동일": nb_same}
    spec_k = {"(1) Δ>0, 단측 p×2<0.05": kb["delta_bp"] > 0 and kb["p_one"] * 2 < 0.05,
              "(2) 순평균(0.3bp+1틱)>0": kb["net_f1_bp"] > 0,
              "(3) 플라시보 p≤0.05": kb["placebo_p"] <= 0.05,
              "(4) 이웃 3개 Δ 부호 동일": nb_same,
              "(5) 하위 B1·B2 Δ>0(보고)": K["B1"]["delta_bp"] > 0 and K["B2"]["delta_bp"] > 0,
              "(6) 참고 KODEX r_oc 5bp 순>0(보고)": kb["kodex"]["net5_bp"] > 0}

    # ---- D4-UF
    nxt_open = f["open"].shift(-1)
    r_on = (nxt_open / f["close"] - 1).where(~bad)
    r_onE = (e["open"].shift(-1) / e["close"] - 1).where(~bad)
    gapdays = pd.Series((idx[1:] - idx[:-1]).days, index=idx[:-1]).reindex(idx)
    wk = gapdays == 1
    logdec = pd.DataFrame({"basis_chg": np.log(f["open"].shift(-1) / k["open"].shift(-1)) - np.log(f["close"] / k["close"]),
                           "k_on": np.log(k["open"].shift(-1) / k["close"]), "f_on": np.log(f["open"].shift(-1) / f["close"])}).where(~bad, np.nan)
    rv_on = vol_adj_valid(r_on)
    Won = windows(idx, "on")
    U = {w: uf_block(r_on, r_onE, rv_on, tick, wk, logdec, a, b) for w, (a, b) in Won.items()}
    ub = U["B"]
    # ⑤ 보수 해석: 표본 이웃(평일·주말)은 총평균 부호가 전체와 같고, 비용 이웃(0.5틱·2틱)은 순평균 부호가 기본(1틱)과 같고,
    #    KODEX 5bp 순평균 부호가 FUT 기본 순평균 부호와 같아야 한다.
    uf_nb = {"평일 총평균": np.sign(ub["wk_F_bp"]) == np.sign(ub["F_bp"]), "주말·연휴 총평균": np.sign(ub["we_F_bp"]) == np.sign(ub["F_bp"]),
             "순 0.5틱 vs 1틱": np.sign(ub["netF05_bp"]) == np.sign(ub["netF1_bp"]), "순 2틱(0.6bp) vs 1틱": np.sign(ub["netF2s_bp"]) == np.sign(ub["netF1_bp"]),
             "KODEX 5bp 순 vs FUT 1틱 순": np.sign(ub["netE5_bp"]) == np.sign(ub["netF1_bp"])}
    uf_nb_same = all(uf_nb.values())
    crit_u = {"①Δ>0": ub["F_bp"] > 0, "②7bp순>0": ub["netF7_bp"] > 0, "③플라시보≤0.05": False,
              "④변동성조정Δ>0": ub["dvol"] > 0, "⑤이웃부호동일": uf_nb_same}
    spec_u = {"(1) FUT 순평균(0.3bp+1틱)>0 → 무조건 오버나잇 흑자": ub["netF1_bp"] > 0,
              "(2) 쌍차 d<0 유지(A −2.10)": ub["d_bp"] < 0,
              "(3) 순수익 차 FUT(1틱)−KODEX(5bp)>0, t>1.96 → 선물 집행 우위": ub["netdiff_bp"] > 0 and ub["t_netdiff"] > 1.96}

    # ---- 성과표(기준선 대비) + 장부
    base_oc = pd.Series(1.0, index=idx).where(sig.notna() & r_oc.notna())
    base_ocE = pd.Series(1.0, index=idx).where(sig.notna() & r_ocE.notna())
    rows_k = []
    rows_k += perf_rows("D4-K6F", "FUT", "FUT r_oc 신호일", sig, r_oc, {"5bp": 5.0, "7bp": 7.0, **{k_: cost_f[k_] for k_ in ("0.3bp+1틱", "0.6bp+2틱")}}, Woc)
    rows_k += perf_rows("D4-K6F", "FUT", "FUT r_oc 매일(기준선)", base_oc, r_oc, {"5bp": 5.0, "7bp": 7.0, **{k_: cost_f[k_] for k_ in ("0.3bp+1틱", "0.6bp+2틱")}}, Woc, with_delta=False)
    for nk, nv in neigh.items():
        rows_k += perf_rows("D4-K6F", "FUT", f"이웃 {nk} FUT r_oc", nv, r_oc, {"7bp": 7.0, "0.3bp+1틱": cost_f["0.3bp+1틱"]}, Woc)
    rows_k += perf_rows("D4-K6F", "069500", "참고 KODEX r_oc 같은 규칙", sig, r_ocE, {"5bp": 5.0, "7bp": 7.0}, Woc)
    rows_k += perf_rows("D4-K6F", "069500", "참고 KODEX r_oc 매일(기준선)", base_ocE, r_ocE, {"5bp": 5.0, "7bp": 7.0}, Woc, with_delta=False)
    base_on = pd.Series(1.0, index=idx).where(r_on.notna() & r_onE.notna())
    rows_u = []
    rows_u += perf_rows("D4-UF", "FUT", "FUT r_on 매일", base_on, r_on, {**cost_f, "5bp": 5.0, "7bp": 7.0}, Won, with_delta=False)
    rows_u += perf_rows("D4-UF", "069500", "KODEX r_on 매일(같은 밤, 기준선)", base_on, r_onE, {"5bp": 5.0, "7bp": 7.0}, Won, with_delta=False)
    rows_u += perf_rows("D4-UF", "FUT", "이웃 FUT 평일 밤", base_on.where(wk, 0.0).where(base_on.notna()), r_on, {"0.3bp+1틱": cost_f["0.3bp+1틱"], "7bp": 7.0}, Won)
    rows_u += perf_rows("D4-UF", "FUT", "이웃 FUT 주말·연휴 밤", base_on.where(~wk, 0.0).where(base_on.notna()), r_on, {"0.3bp+1틱": cost_f["0.3bp+1틱"], "7bp": 7.0}, Won)
    tk_ = pd.DataFrame(rows_k)
    tu_ = pd.DataFrame(rows_u)
    SPEC = {"family": "D4 실행 상품(코스피200 선물)", "spec_file": str(SPEC_FILE), "spec_sha256_16": spec_sha[:16], "frozen_on": "2026-09-29",
            "cells": {"D4-K6F": "us_signal(FUT,60,0.7,0.3)[H-rev] -> FUT r_oc(T), excl expiry & next day",
                      "D4-UF": "every day FUT r_on(T) vs KODEX r_on same night, excl expiry & next day"},
            "cost_fut": "fee + k*0.05/C_F(T): 0.3+1tick base, 0.3+0.5tick, 0.6+2tick; flat 5/7bp", "confirm": [B0, B1]}
    E.ledger_append(data.OUT, TOPIC, SPEC, pd.concat([tk_, tu_], ignore_index=True), note="2차 확인")

    # ---- 판정 요약
    def grade(crit):
        failed = [c for c, v in crit.items() if not v]
        if not failed:
            return "통과 후보(①~⑤ 충족, 원 p 는 종합 Holm 대기)"
        return "불통과(미충족: " + ", ".join(failed) + ")"

    g_k, g_u = grade(crit_k), grade(crit_u)
    run_at = dt.datetime.now(E.KST).strftime("%Y-%m-%d %H:%M")

    verdict = {
        "family": "D4 실행 상품: 코스피200 선물로 오버나잇·종가 베팅 집행(총수익·손익분기 비용)", "run_at": run_at,
        "script": str(Path(__file__).resolve()), "spec_file": str(SPEC_FILE), "spec_sha256": spec_sha,
        "confirm_window": [B0, B1], "data_check": chk, "expiries_in_B": [str(d.date()) for d in ex if pd.Timestamp(B0) <= d <= pd.Timestamp(B1)],
        "cells": {
            "D4-K6F": {"role": "알파(1차 K6 H-rev 를 선물 r_oc 로 이전)", "windows": {WLAB[w]: v for w, v in K.items()},
                       "yearly_delta_bp": {"A": by_A, "B": by_B, "E4": by_E4}, "leave_one_year_out_B": loo_B,
                       "criteria": {k_: bool(v) for k_, v in crit_k.items()}, "criteria_str": "".join(ox(v) for v in crit_k.values()),
                       "spec_B_criteria": {k_: bool(v) for k_, v in spec_k.items()},
                       "p_one_B_raw": kb["p_one"], "p_one_B_x2": min(1.0, kb["p_one"] * 2), f"p_one_B_x{N_CELLS_R2}": min(1.0, kb["p_one"] * N_CELLS_R2),
                       "grade": g_k},
            "D4-UF": {"role": "실행 기준선(알파 아님). 판정표의 Δ = FUT 매일 오버나잇 총평균(보유 − 무포지션)",
                      "windows": {WLAB[w]: v for w, v in U.items()}, "neighbors_check_B": {k_: bool(v) for k_, v in uf_nb.items()},
                      "criteria": {k_: bool(v) for k_, v in crit_u.items()}, "criteria_str": "".join(ox(v) for v in crit_u.values()),
                      "placebo_note": "무조건 셀이라 연도 층화 플라시보가 정의되지 않음(모든 밤이 선택됨) → ③ 미충족으로 처리",
                      "spec_B_criteria": {k_: bool(v) for k_, v in spec_u.items()},
                      "p_one_B_raw": ub["p_one"], "grade": g_u},
        },
    }
    C.save_verdict(OUT / "verdict.json", verdict)

    # ---------------------------------------------------------------- report.md
    md = []
    A_ = md.append
    A_("# R2 · D4 확인: 코스피200 선물로 오버나잇·종가 베팅 집행\n")
    A_(f"실행 {run_at} KST. 사양 `spec.json`(동결 2026-09-29, sha256 `{spec_sha[:16]}…`)을 바꾸지 않고 구현했다. "
       f"코드 `overnight_lab/topics/r2_d4.py`. 확인 구간 B = 결정일 T ∈ [{B0}, {B1}]. "
       "A 는 표본 내 값(탐색 구간)이고 E4 는 보고만 한다.\n")
    A_("## 0. 판정 요약 (B 구간)\n")
    A_("| 셀 | B 일수 / 거래일 | Δ bp (NW t, 단측 원 p) | ② 7bp 순평균 | ③ 플라시보 p | ④ 변동성 조정 Δ | ⑤ 이웃 부호 | 기준 ①~⑤ | 예비 판정 |")
    A_("|---|---|---|---|---|---|---|---|---|")
    A_(f"| D4-K6F (알파) | {kb['days']} / {kb['n_sig']} | {kb['delta_bp']:+.2f} ({kb['t']:+.2f}, {kb['p_one']:.4f}) | {kb['net7_bp']:+.2f} | "
       f"{kb['placebo_p']:.4f} | {kb['dvol']:+.4f} SD | {ox(nb_same)} | {verdict['cells']['D4-K6F']['criteria_str']} | {g_k} |")
    A_(f"| D4-UF (실행 기준선) | {ub['nights']} / {ub['nights']} | {ub['F_bp']:+.2f} ({ub['t_F']:+.2f}, {ub['p_one']:.4f}) | {ub['netF7_bp']:+.2f} | "
       f"해당 없음(미충족 처리) | {ub['dvol']:+.4f} SD | {ox(uf_nb_same)} | {verdict['cells']['D4-UF']['criteria_str']} | {g_u} |")
    A_("")
    A_("- 원 p 는 NW(lag 5) t 의 정규 근사 단측 p 다. Holm 보정은 2차 종합(최대 8셀)에서 한다. "
       f"참고로 D4-K6F 의 p×2(사양 권고 본페로니) = {min(1, kb['p_one'] * 2):.4f}, p×8 = {min(1, kb['p_one'] * 8):.4f}.")
    A_(f"- A(표본 내) Δ: D4-K6F {K['A']['delta_bp']:+.2f}bp (t {K['A']['t']:+.2f}), D4-UF 총평균 {U['A']['F_bp']:+.2f}bp (t {U['A']['t_F']:+.2f}). "
       f"E4(보고) Δ: D4-K6F {K['E4']['delta_bp']:+.2f}bp (t {f2(K['E4']['t'])}, 신호 {K['E4']['n_sig']}일), D4-UF 총평균 {U['E4']['F_bp']:+.2f}bp (t {U['E4']['t_F']:+.2f}, {U['E4']['nights']}밤).")
    A_("- D4-UF 는 조건 없는 셀이다. 판정표의 Δ 는 '매일 선물 보유 − 무포지션' 즉 총평균이다. 알파가 아니라 집행 기준선이라 사양의 기준(아래)이 본 판정이다.\n")
    A_("### 사양 자체의 B 기준\n")
    A_("| 셀 | 기준 | 값 | 충족 |\n|---|---|---|---|")
    vals_k = [f"Δ {kb['delta_bp']:+.2f}, p×2 {min(1, kb['p_one'] * 2):.4f}", f"{kb['net_f1_bp']:+.2f}bp (t {kb['t_net_f1']:+.2f})",
              f"{kb['placebo_p']:.4f}", ", ".join(f"{a} {v['delta_bp']:+.2f}" for a, v in kb["neighbors"].items()),
              f"B1 {K['B1']['delta_bp']:+.2f} / B2 {K['B2']['delta_bp']:+.2f}", f"KODEX 5bp 순 {kb['kodex']['net5_bp']:+.2f}bp"]
    for (c, v), s in zip(spec_k.items(), vals_k):
        A_(f"| D4-K6F | {c} | {s} | {ox(v)} |")
    vals_u = [f"{ub['netF1_bp']:+.2f}bp (t {ub['t_netF1']:+.2f}); 총평균 {ub['F_bp']:+.2f} vs 비용 {ub['cost_f1_bp']:.2f}",
              f"{ub['d_bp']:+.2f}bp (t {ub['t_d']:+.2f})", f"{ub['netdiff_bp']:+.2f}bp (t {ub['t_netdiff']:+.2f})"]
    for (c, v), s in zip(spec_u.items(), vals_u):
        A_(f"| D4-UF | {c} | {s} | {ox(v)} |")
    A_("")
    A_("### 결론\n")
    A_(f"- **D4-K6F**: A 의 Δ {K['A']['delta_bp']:+.2f}bp(t {K['A']['t']:+.2f})가 B 에서 {kb['delta_bp']:+.2f}bp(t {kb['t']:+.2f}, 플라시보 {kb['placebo_p']:.3f})로 "
       f"{'유지됐다' if all(crit_k.values()) else '확인되지 않았다'}. B1(시가 09:00) {K['B1']['delta_bp']:+.2f} → B2(시가 08:45) {K['B2']['delta_bp']:+.2f}. "
       f"순평균은 선물 기본 비용(0.3bp+1틱) {kb['net_f1_bp']:+.2f}bp, 7bp {kb['net7_bp']:+.2f}bp 다. 예비 판정: {g_k}.")
    A_(f"- **D4-UF**: 알파 기준(①~⑤)으로는 {g_u}. 실행 기준선으로서 사양 기준은 "
       f"(1) FUT 순평균(0.3bp+1틱) {ub['netF1_bp']:+.2f}bp(t {ub['t_netF1']:+.2f}) {ox(spec_u[list(spec_u)[0]])}, "
       f"(2) 쌍차 {ub['d_bp']:+.2f}bp(t {ub['t_d']:+.2f}) {ox(spec_u[list(spec_u)[1]])}, "
       f"(3) 순수익 차 {ub['netdiff_bp']:+.2f}bp(t {ub['t_netdiff']:+.2f}) {ox(spec_u[list(spec_u)[2]])} 이다. "
       + ("선물은 같은 밤 총수익이 KODEX 보다 낮지만(쌍차 음), 비용 차가 그보다 커서 순수익은 선물이 높다. "
          if ub["d_bp"] < 0 and ub["netdiff_bp"] > 0 else "쌍차·순수익 차의 부호는 위 표와 같다. ")
       + f"B 에서 FUT 총평균 {ub['F_bp']:+.2f}bp 와 7bp 비용의 차(7bp 순 {ub['netF7_bp']:+.2f})로 보면, 무조건 오버나잇의 흑자 여부는 선물 비용 가정(B 1틱 ≈ {ub['tick_bp']:.2f}bp)에 달려 있다.")
    A_(f"- 주말·연휴 밤(B {ub['we_n']}밤): FUT {ub['we_F_bp']:+.2f}bp / KODEX {ub['we_E_bp']:+.2f}bp (쌍차 {ub['we_F_bp'] - ub['we_E_bp']:+.2f}). "
       f"평일 밤({ub['wk_n']}밤): FUT {ub['wk_F_bp']:+.2f}bp / KODEX {ub['wk_E_bp']:+.2f}bp (쌍차 {ub['wk_F_bp'] - ub['wk_E_bp']:+.2f}). "
       "다른 D 셀의 오버나잇을 선물로 집행할 때 요일 구성에 따라 선물 열위 크기가 달라진다는 점을 참고할 것.\n")

    A_("## 1. 자료 점검\n")
    nB = (idx >= pd.Timestamp(B0)) & (idx <= pd.Timestamp(B1))
    A_(f"- FUT {chk['fut_rows']}행 ({chk['fut_first']} ~ {chk['fut_last']}). B 구간 FUT 거래일 {int(nB.sum())}일, 그중 만기일·다음날 제외 {int(bad[nB].sum())}일"
       f"(B 만기일 {len(verdict['expiries_in_B'])}개).")
    A_(f"- 날짜 정렬: KODEX 에만 있는 날(FUT 기간 안) {len(chk['kodex_only_dates'])}일, FUT 에만 있는 날(KODEX 대비) {len(chk['fut_only_vs_kodex'])}일, "
       f"K200 대비 {len(chk['fut_only_vs_k200'])}일. 모두 0 이면 같은 밤·같은 날 비교가 성립한다.")
    A_(f"- 1틱 bp(0.05/C_F(T)): A 평균 {K['A']['tick_bp']:.2f}bp(K6F 신호일), B {ub['tick_bp']:.2f}bp, E4 {U['E4']['tick_bp']:.2f}bp. "
       "지수 수준이 오르면 틱 비용이 bp 로 줄어든다.\n")

    A_("## 2. D4-K6F: 전일 미국 약세 → 선물 시가 매수·종가 매도\n")
    A_("### 2.1 구간별\n")
    A_("| 구간 | 일수 / 신호 | 신호일 평균 | 비신호 평균 | Δ bp (NW t, 단측 p) | 플라시보 p | 변동성 조정 Δ | 순 0.3bp+0.5틱 | 순 0.3bp+1틱 (t) | 순 0.6bp+2틱 | 순 5bp | 순 7bp | 손익분기 틱 |")
    A_("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for w, v in K.items():
        A_(f"| {WLAB[w]} | {v['days']} / {v['n_sig']} | {v['sig_bp']:+.2f} | {v['non_bp']:+.2f} | {v['delta_bp']:+.2f} ({f2(v['t'])}, {f2(v['p_one'], '{:.4f}')}) | "
           f"{f2(v['placebo_p'], '{:.4f}')} | {f2(v['dvol'], '{:+.4f}')} | {v['net_f05_bp']:+.2f} | {v['net_f1_bp']:+.2f} ({f2(v['t_net_f1'])}) | "
           f"{v['net_f2s_bp']:+.2f} | {v['net5_bp']:+.2f} | {v['net7_bp']:+.2f} | {f2(v['be_ticks'])} |")
    A_("\n손익분기 틱 = (신호일 총평균 − 0.3bp) / 신호일 평균 1틱 bp.\n")
    A_("### 2.2 이웃·참고(KODEX 같은 규칙)\n")
    A_("| 구간 | 창40 Δ (t) | 창120 Δ (t) | q20 Δ (t) | KODEX r_oc Δ (t) | KODEX 신호일 순 5bp / 7bp |")
    A_("|---|---|---|---|---|---|")
    for w, v in K.items():
        nb = v["neighbors"]
        A_(f"| {WLAB[w]} | " + " | ".join(f"{nb[x]['delta_bp']:+.2f} ({f2(nb[x]['t'])})" for x in ("창40", "창120", "q20(창60)")) +
           f" | {v['kodex']['delta_bp']:+.2f} ({f2(v['kodex']['t'])}) | {v['kodex']['net5_bp']:+.2f} / {v['kodex']['net7_bp']:+.2f} |")
    A_("\n### 2.3 진단 (판정 밖)\n")
    A_("| 구간 | 갭 통제 신호 계수 bp (t) | 갭 계수 (t) | 신호일 갭하락 비율 / 비신호 | 신호일 중앙값 |")
    A_("|---|---|---|---|---|")
    for w, v in K.items():
        g = v["gap_reg"]
        A_(f"| {WLAB[w]} | {g['signal_bp']:+.2f} ({g['t_signal']:+.2f}) | {g['gap_coef']:+.3f} ({g['t_gap']:+.2f}) | "
           f"{g['gapdown_share_sig'] * 100:.1f}% / {g['gapdown_share_non'] * 100:.1f}% | {v['sig_median_bp']:+.2f} |")
    A_("\n연도별 Δ bp: " + "; ".join(f"{y} {d_:+.1f}" for y, d_ in {**by_A, **by_B, **by_E4}.items()) + " (2014~2019 = A, 2020~2025-06-05 = B, 2025-06-09~ = E4)")
    A_("\nB 한 해 빼기 Δ bp (t): " + "; ".join(f"−{y} {v['delta_bp']:+.2f} ({f2(v['t'])})" for y, v in loo_B.items()) + "\n")
    A_("### 2.4 성과표: 신호일 vs 매일 시가→종가 기준선 (비노출일 수익 0)\n")
    cols = ["rule", "trades", "gross_bp", "cost_bp", "net_bp", "t_net", "win", "sharpe", "cagr", "mdd"]
    heads = ["규칙 [비용]", "거래", "총평균", "평균 비용", "거래당 순평균", "t", "승률", "샤프", "CAGR", "MDD"]
    for w in ("B", "B1", "B2", "A", "E4"):
        sub = tk_[(tk_["window"] == WLAB[w]) & ~tk_["rule"].str.contains("이웃")].copy()
        sub["rule"] = sub["rule"].str.replace("D4-K6F ", "", regex=False)
        A_(f"**{WLAB[w]}**\n")
        A_(md_table(sub, cols, heads) + "\n")

    A_("## 3. D4-UF: 매일 선물 오버나잇 (실행 기준선)\n")
    A_("### 3.1 구간별 총수익·쌍차·손익분기\n")
    A_("| 구간 | 밤 | FUT 총평균 (SD, NW t) | KODEX 총평균 (t) | 쌍차 FUT−KODEX (t) | 상관 | 베이시스 변화 (t) | 손익분기 FUT bp (틱) | 손익분기 KODEX bp | 1틱 bp |")
    A_("|---|---|---|---|---|---|---|---|---|---|")
    for w, v in U.items():
        A_(f"| {WLAB[w]} | {v['nights']} | {v['F_bp']:+.2f} ({v['F_sd']:.1f}, {v['t_F']:+.2f}) | {v['E_bp']:+.2f} ({v['t_E']:+.2f}) | "
           f"{v['d_bp']:+.2f} ({v['t_d']:+.2f}) | {v['corr']:.3f} | {v['basis_chg_bp']:+.2f} ({v['t_basis_chg']:+.2f}) | "
           f"{v['be_bp']:+.2f} (0.3bp + {v['be_ticks']:+.2f}틱) | {v['be_E_bp']:+.2f} | {v['tick_bp']:.2f} |")
    A_("\n베이시스 변화 = log(F_open(T+1)/K200_open(T+1)) − log(F_close(T)/K200_close(T)) (로그 분해: log F_on = log K200_on + 베이시스 변화). "
       "2023-07-31 이후 선물 시가(08:45)와 지수 시가(09:00) 시각이 달라 분해는 참고만.\n")
    A_("### 3.2 비용별 순평균 (bp/밤)\n")
    A_("| 구간 | FUT 0.3bp+0.5틱 | FUT 0.3bp+1틱 (t) | FUT 0.6bp+2틱 | FUT 5bp | FUT 7bp (t) | KODEX 5bp (t) | KODEX 7bp | 순수익 차 FUT 1틱 − KODEX 5bp (t) | 스트레스 차 FUT 0.6bp+2틱 − KODEX 7bp (t) |")
    A_("|---|---|---|---|---|---|---|---|---|---|")
    for w, v in U.items():
        A_(f"| {WLAB[w]} | {v['netF05_bp']:+.2f} | {v['netF1_bp']:+.2f} ({v['t_netF1']:+.2f}) | {v['netF2s_bp']:+.2f} | {v['netF5_bp']:+.2f} | "
           f"{v['netF7_bp']:+.2f} ({v['t_netF7']:+.2f}) | {v['netE5_bp']:+.2f} ({v['t_netE5']:+.2f}) | {v['netE7_bp']:+.2f} | "
           f"{v['netdiff_bp']:+.2f} ({v['t_netdiff']:+.2f}) | {v['netdiff_stress_bp']:+.2f} ({v['t_netdiff_stress']:+.2f}) |")
    A_("\n### 3.3 이웃: 평일 밤 / 주말·연휴 밤\n")
    A_("| 구간 | 평일 밤 수 | 평일 FUT (t) / KODEX | 주말·연휴 밤 수 | 주말 FUT (t) / KODEX | 변동성 조정 평균 |")
    A_("|---|---|---|---|---|---|")
    for w, v in U.items():
        A_(f"| {WLAB[w]} | {v['wk_n']} | {v['wk_F_bp']:+.2f} ({f2(v['wk_t'])}) / {v['wk_E_bp']:+.2f} | {v['we_n']} | "
           f"{v['we_F_bp']:+.2f} ({f2(v['we_t'])}) / {v['we_E_bp']:+.2f} | {v['dvol']:+.4f} SD |")
    A_("\nB 이웃 부호 점검(보수 해석): " + ", ".join(f"{a} {ox(b)}" for a, b in uf_nb.items()) + "\n")
    A_("### 3.4 성과표: FUT 매일 오버나잇 vs KODEX 매일 보유(같은 밤) 기준선\n")
    for w in ("B", "B1", "B2", "A", "E4"):
        sub = tu_[tu_["window"] == WLAB[w]].copy()
        sub["rule"] = sub["rule"].str.replace("D4-UF ", "", regex=False)
        A_(f"**{WLAB[w]}**\n")
        A_(md_table(sub, cols, heads) + "\n")

    A_("## 4. 해석 선택 (사양이 모호한 곳은 가장 보수적인 쪽)\n")
    notes = [
        "② 의 비용: 2차 공통 판정 규약은 '7bp 차감'이고 사양의 선물 비용은 0.3bp+1틱(스트레스 0.6bp+2틱)이다. 판정표 ② 는 보수적으로 **정액 7bp** 로 판정했다. "
        "B 에서 선물 스트레스 비용(0.6bp+2틱)은 평균 약 " + f"{0.6 + 2 * ub['tick_bp']:.2f}bp 로 7bp 보다 작다. 사양 기준 (2)(0.3bp+1틱)는 사양 B 기준 표에 따로 판정했다.",
        "D4-UF 는 조건이 없어 신호−비신호 Δ 가 없다. 판정표의 Δ 는 '매일 보유 − 무포지션' = FUT 총평균으로 두고, NW t 로 단측 p 를 냈다. "
        "연도 층화 플라시보는 모든 밤이 선택돼 정의되지 않으므로 ③ 은 미충족으로 처리했다(플라시보 p 는 '해당 없음').",
        "D4-UF 의 ⑤ 이웃: 사양에 적힌 이웃 네 종류를 모두 요구했다. 평일·주말은 총평균 부호가 전체와 같아야 하고, 비용 0.5틱·2틱은 순평균 부호가 기본(1틱)과 같아야 하며, "
        "KODEX 5bp 순평균 부호가 FUT 기본 순평균 부호와 같아야 한다.",
        "D4-K6F 의 ⑤ 이웃: 창40·창120·q20 의 B Δ 부호가 본 셀 B Δ 부호와 같아야 한다(common.judge 규약).",
        "변동성 조정: 만기일·다음날을 뺀 유효일만으로 T 를 뺀 과거 60개 유효일 SD 를 구해 나눴다(만기 롤 점프가 SD 를 부풀리지 않도록). "
        "K6F 는 Δ(신호−비신호)를, UF 는 평균을 SD 단위로 보고한다. A 의 K6F 변동성 조정 Δ 가 탐색 메모(+0.123)와 조금 다른 것은 이 SD 표본 정의 차이 때문이다.",
        "만기 제외: 이웃·KODEX 참고·기준선에도 같은 제외일을 적용해 같은 날 비교가 되게 했다. 시가 갭 통제 회귀의 갭은 O_F(T)/C_F(T−1) − 1 이다.",
        "구간: 사양대로 T 기준. B 하위 구간 경계(2023-07-28 / 2023-07-31)도 T 기준이라 UF 의 T = 2023-07-28 밤(청산 07-31 08:45)은 B1 에 들어간다. "
        "UF 의 A 는 청산일 ≤ 2019-12-31(T ≤ 2019-12-27)이고, T = 2019-12-30 밤은 A·B 어디에도 넣지 않았다(사양).",
        "비용 C_F(T) 는 두 셀 모두 T 의 FUT 종가(사양의 단순화). 플라시보는 engine 기본(2,000회, 시드 20260929)을 썼다(탐색은 1,000회).",
        "성과표의 샤프·CAGR·MDD 는 engine.stats 와 같은 정의(비노출일 수익 0, 연 252일)다. 선물 수익률은 명목 대비이며 증거금 대비 레버리지 수익이 아니다.",
    ]
    for n_ in notes:
        A_(f"- {n_}")
    A_("\n## 5. 한계\n")
    A_("- FUT 는 연결 일봉이라 롤·만기 체결을 검증할 수 없어 분기마다 2일을 뺐다. 집행 대안인 미니선물(매월 만기, 틱 0.02pt)은 캐시에 일봉이 없어 재현하지 못했다.")
    A_("- 단일가 체결을 가정한다. 시가·종가 단일가의 실제 체결 가능 수량과 호가 스프레드는 일봉으로 확인할 수 없다.")
    A_("- FUT − KODEX 오버나잇 열위의 원인(15:30~15:45 선물 움직임 vs 개장 베이시스)은 일봉으로 가를 수 없다. 1분봉이 필요하다.")
    A_("- 자금 비용(선물은 증거금 외 현금에 이자) 이점은 순수익에 넣지 않았다(보수적).")
    A_("- 2025-06-09 이후(E4)는 KRX 야간장이 있지만 오버나잇 셀은 야간장을 거래하지 않고 보유만 한다. E4 는 판정에 쓰지 않는다.\n")
    A_("## 6. 재현\n")
    A_("```\ncd overnight_lab\npython topics/r2_d4.py\n```\n")
    A_("시행 장부: `<OUT>/ledger.csv` 에 topic `R2_D4`, run_note `2차 확인` 으로 이 실행의 모든 규칙·비용·구간 행을 남겼다.")
    (OUT / "report.md").write_text("\n".join(md), encoding="utf-8")

    print("K6F B:", {k_: kb[k_] for k_ in ("days", "n_sig", "delta_bp", "t", "p_one", "net7_bp", "net_f1_bp", "placebo_p", "dvol")})
    print("K6F neighbors B:", kb["neighbors"])
    print("K6F A:", {k_: K["A"][k_] for k_ in ("days", "n_sig", "delta_bp", "t", "placebo_p", "dvol", "net_f1_bp")})
    print("K6F E4:", {k_: K["E4"][k_] for k_ in ("days", "n_sig", "delta_bp", "t")})
    print("K6F B1/B2:", K["B1"]["delta_bp"], K["B2"]["delta_bp"], "crit", verdict["cells"]["D4-K6F"]["criteria_str"], g_k)
    print("UF B:", {k_: ub[k_] for k_ in ("nights", "F_bp", "t_F", "p_one", "netF1_bp", "netF7_bp", "d_bp", "t_d", "netdiff_bp", "t_netdiff", "dvol")})
    print("UF A:", {k_: U["A"][k_] for k_ in ("nights", "F_bp", "t_F", "E_bp", "d_bp", "t_d", "netdiff_bp", "t_netdiff", "basis_chg_bp")})
    print("UF E4:", {k_: U["E4"][k_] for k_ in ("nights", "F_bp", "t_F", "d_bp", "netdiff_bp")})
    print("UF crit", verdict["cells"]["D4-UF"]["criteria_str"], g_u, uf_nb)
    print("chk", {k_: (len(v) if isinstance(v, list) else v) for k_, v in chk.items()})


if __name__ == "__main__":
    main()
