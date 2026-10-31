"""16차 스캔 · 스윙 2단계: 후보 검증. 봉인 2024-12-30 유지(로더에서 절단).

후보(1단계 격자에서 고정, 이 파일에서 바꾸지 않는다)
  MRq   코스닥150 지수 IBS < 0.1 → T 15:45 선물 진입, IBS > 0.7 또는 10거래일 → 그날 15:45 청산
  MRqG  MRq + T−1 까지의 수급 앙상블 z̄ > 0 일 때만 진입
  MRqD  코스닥150 20일 고점 대비 낙폭 ≤ −1.5σ√20 → 청산 동일
  MRp   코스피200 IBS < 0.1 → 청산 동일(코스피200 선물)
  FLq   코스닥 프로그램 비차익 10일 누적 z > 0.5 롱, z < −0.5 숏, z 가 0 을 건너면 청산, T+1 시가 선물 체결
  FLqVT FLq × min(2, 15%/20일 실현 변동성)
  FLz   수급 앙상블 z̄ 선형 크기 clip(z̄,−1,1)
  FLzVT FLz × min(2, 15%/20일 실현 변동성)

검증: 차량·창·문턱 인접, HAC 알파, 신호 이동 플라시보(가족 단위), 디플레이티드 샤프, 선택 축소(이전 3년 최고 셀을 다음 해에 씀),
      사전 구간(2006-01~2015-09 코스닥 종합지수, 규칙 고정, 거래 불가 지수 경로), 슬리브 결합, MDD 15% 배수, 기준 장부 상관.
산출: scan/swing/{s2_*.csv, s2_results.json, fig*.png}, cells.csv 에 S2_ 행 추가
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import r16s_swing as S  # noqa: E402
import r16s_swing_data as D  # noqa: E402

plt.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = True
OUT = S.OUT
c = S.Ctx()
Sf = S.flow_series()
RES = {}
SEED = 20261030
Q_S, P_S = "2016-01-04", "2010-01-04"
END = "2024-12-30"
CAND = {"MRq": ("KQ", "c"), "MRqG": ("KQ", "c"), "MRqD": ("KQ", "c"), "MRp": ("KP", "c"), "FLq": ("KQ", "o"), "FLqVT": ("KQ", "o"), "FLz": ("KQ", "o"), "FLzVT": ("KQ", "o")}


def zprog(idx, n=10, zw=250):
    s = Sf["KQ"]["prog"].astype(float).rolling(n).sum()
    return ((s - s.rolling(zw, min_periods=int(zw * 0.8)).mean()) / s.rolling(zw, min_periods=int(zw * 0.8)).std()).reindex(idx)


def zb(idx):
    return pd.concat([S.zscore(Sf["KQ"][k].astype(float), n).reindex(idx) for k in ("prog", "spot") for n in (5, 10)], axis=1).mean(axis=1, skipna=False)


def rule(nm, f, idx, th=0.5, n=10, zw=250, ibs=0.1, cap=10):
    """f: 특성 표, idx: 달력. 거래 차량과 무관한 목표 노출 w."""
    x70 = S.exit_cond(f, "ibs70")
    if nm in ("MRq", "MRp"):
        return S.hold(S.trig(f, "ibs", ibs).fillna(False).to_numpy(), x70, cap)
    if nm == "MRqG":
        return S.hold(S.trig(f, "ibs", ibs).fillna(False).to_numpy() & (zb(idx).shift(1).to_numpy() > 0), x70, cap)
    if nm == "MRqD":
        return S.hold(S.trig(f, "dd20", 1.5).fillna(False).to_numpy(), x70, cap)
    if nm == "FLq":
        return S.flow_pos(zprog(idx, n, zw).to_numpy(), th, True)
    if nm == "FLqVT":
        return S.vt(S.flow_pos(zprog(idx, n, zw).to_numpy(), th, True), f["vol"].to_numpy(), 0.15, 2.0)
    if nm == "FLz":
        return S.band(np.clip(np.nan_to_num(zb(idx).to_numpy()), -1, 1))
    if nm == "FLzVT":
        return S.vt(np.clip(np.nan_to_num(zb(idx).to_numpy()), -1, 1), f["vol"].to_numpy(), 0.15, 2.0)
    raise ValueError(nm)


def sim_rule(nm, vk="F", fill=None, **kw):
    m, f0 = CAND[nm]
    w = c.mask(m, vk, rule(nm, c.Fv[m][vk], c.V[m][vk].idx, **kw))
    return w, c.run(m, w, vk, fill or f0), m


def qm(res, V, a, b):
    idx = V.idx
    ma = (idx >= pd.Timestamp(a)) & (idx <= pd.Timestamp(b))
    r = res["r"][ma]
    tid = idx[res["ti"]]
    k = (tid >= pd.Timestamp(a)) & (tid <= pd.Timestamp(b))
    nt, st = res["nt"][k], res["st"][k]
    eq = np.cumprod(1 + r)
    return dict(n=int(k.sum()), net_bp=float(nt.mean() * 1e4) if k.any() else np.nan, t_nw=float(S.E.nw_t(nt * 1e4)) if k.sum() >= 10 else np.nan, stress_bp=float(st.mean() * 1e4) if k.any() else np.nan,
                cagr=float((eq[-1] ** (252 / len(r)) - 1) * 100), mdd=float((eq / np.maximum.accumulate(eq) - 1).min() * 100), sharpe=float(r.mean() / r.std() * np.sqrt(252)) if r.std() > 0 else np.nan)


def ser(res, V):
    return pd.Series(res["r"], index=V.idx)


def st_d(r):
    eq = (1 + r).cumprod()
    return dict(cagr=round(float((eq.iloc[-1] ** (252 / len(r)) - 1) * 100), 2), mdd=round(float((eq / eq.cummax() - 1).min() * 100), 2), sharpe=round(float(r.mean() / r.std() * np.sqrt(252)), 2))


def fmt(d):
    return {k: (round(v, 2) if isinstance(v, float) else v) for k, v in d.items()}


def hac_alpha(r, mk):
    d = pd.DataFrame({"r": r, "m": mk, "m1": mk.shift(1)}).dropna()
    f = sm.OLS(d["r"], sm.add_constant(d[["m", "m1"]])).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
    return dict(alpha_ann=round(f.params["const"] * 252 * 100, 2), t_alpha=round(f.tvalues["const"], 2), beta0=round(f.params["m"], 3), beta1=round(f.params["m1"], 3), r2=round(f.rsquared, 3))


def scale_to(r, tgt=-0.15):
    lo, hi = 0.02, 10.0
    for _ in range(50):
        k = (lo + hi) / 2
        eq = (1 + k * r).cumprod()
        if (eq / eq.cummax() - 1).min() < tgt:
            hi = k
        else:
            lo = k
    return lo


# ---------------------------------------------------------------- 1 후보 기본·차량·창
def sect_cands():
    rows, yearly, daily = [], {}, {}
    for nm, (m, f0) in CAND.items():
        w, res, _ = sim_rule(nm)
        V = c.V[m]["F"]
        a = Q_S if m == "KQ" else P_S
        r = ser(res, V).loc[a:]
        daily[nm] = r
        yearly[nm] = ((1 + r).groupby(r.index.year).prod() - 1).mul(100).round(1).to_dict()
        s = S.summary(res, V, a, c.base, c.mret[m]["F"])
        mk = pd.Series(c.mret[m]["F"], index=V.idx).loc[a:]
        h = dict(hac_alpha(r, mk))
        o = dict(name=nm, market=m, window=s["window"], n=s["n"], gross_bp=s["gross_bp"], net_bp=s["net_bp"], t_nw=s["t_nw"], t_daily=s["t_daily"], cagr=s["cagr_pct"], mdd=s["mdd_pct"], sharpe=s["sharpe"], yearly=s["yearly_pos"],
                 top5=s["top5_share_pct"], h1=s["half1_net_bp"], h2=s["half2_net_bp"], stress=s["stress_net_bp"], hold=s["avg_hold"], expo=s["expo"], maxexpo=s["max_expo"], win=s["win_pct"], **h,
                 mkt_cagr=st_d(mk)["cagr"], mkt_mdd=st_d(mk)["mdd"], ban_share=float(r.loc["2023-11-06":].sum() / r.sum() * 100), top5d=float(r.nlargest(5).sum() / r.sum() * 100),
                 sh_ex2020=float(r[r.index.year != 2020].mean() / r[r.index.year != 2020].std() * np.sqrt(252)), worst_day=float(r.min() * 100),
                 mon_pos=float(((1 + r).groupby([r.index.year, r.index.month]).prod() > 1).mean() * 100), **{k: s[k] for k in s if k.startswith("rho_")})
        rows.append(o)
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "s2_candidates.csv", index=False, encoding="utf-8-sig")
    print(t.round(2).T.to_string())
    y = pd.DataFrame(yearly)
    y.to_csv(OUT / "s2_yearly.csv", encoding="utf-8-sig")
    print(y)
    return daily


def sect_vehicle():
    rows = []
    def add(tag, nm, vk, fill, a, b=END, note=""):
        m, f0 = CAND[nm]
        vv = {"E2": "E"}.get(vk, vk)
        w = rule(nm, c.Fv[m][vv], c.V[m][vv].idx)
        w = c.mask(m, vv, w)
        w = np.where(c.V[m][vv].idx < pd.Timestamp(a), 0.0, w)
        V = c.V[m][vk]
        if vk == "E2":
            res = S.sim(np.clip(w, 0, None) * V.valid, V, fill)
        elif vk == "E":
            res = c.run(m, w, "E", fill)
        else:
            res = S.sim(w, V, fill)
        rows.append(dict(tag=tag, rule=nm, vehicle=vk, fill=fill, a=a, **fmt(qm(res, V, a, b)), note=note))
    for nm in ("MRq", "MRqG", "MRqD"):
        add("기본", nm, "F", "c", Q_S)
        add("2018~", nm, "F", "c", "2018-01-02", note="선물 유동성 확보 뒤")
        add("첫 밤 제외 T+1 시가 진입", nm, "F", "o", Q_S)
        add("청산 T+1 시가", nm, "F", "co", Q_S)
        add("ETF 1배 T+1 시가", nm, "E", "o", "2015-10-01")
        add("ETF 1배 하루 늦은 종가", nm, "E", "l", "2015-10-01")
        add("ETF 2배 T+1 시가", nm, "E2", "o", "2015-12-17", note="233740 실제 가격, 시장 노출 2배")
    add("기본", "MRp", "F", "c", P_S)
    add("2018~", "MRp", "F", "c", "2018-01-02")
    add("첫 밤 제외 T+1 시가 진입", "MRp", "F", "o", P_S)
    add("ETF 1배 T+1 시가", "MRp", "E", "o", P_S)
    add("ETF 2배 T+1 시가", "MRp", "E2", "o", "2010-02-22", note="122630 실제 가격, 시장 노출 2배")
    for nm in ("FLq", "FLqVT"):
        add("기본", nm, "F", "o", Q_S)
        add("2018~", nm, "F", "o", "2018-01-02", note="선물 유동성 확보 뒤")
        add("2020~", nm, "F", "o", "2020-01-02")
        add("하루 늦은 T+1 종가", nm, "F", "l", Q_S)
        add("ETF 1배·인버스 T+1 시가", nm, "E", "o", "2016-08-10", note="251340 실제 가격")
    add("ETF 2배 T+1 시가(롱만)", "FLq", "E2", "o", "2015-12-17", note="233740 실제 가격")
    add("기본", "FLz", "F", "o", Q_S)
    add("기본", "FLzVT", "F", "o", Q_S)
    add("2018~", "FLzVT", "F", "o", "2018-01-02")
    add("하루 늦은 T+1 종가", "FLzVT", "F", "l", Q_S)
    add("ETF 1배·인버스 T+1 시가", "FLzVT", "E", "o", "2016-08-10", note="251340 실제 가격")
    # 인접 문턱
    for kw, tag in ((dict(n=7), "N7"), (dict(n=15), "N15"), (dict(th=0.25), "θ0.25"), (dict(th=0.75), "θ0.75"), (dict(zw=120), "z창120"), (dict(zw=500), "z창500")):
        w, res, m = sim_rule("FLq", **kw)
        rows.append(dict(tag="인접 " + tag, rule="FLq", vehicle="F", fill="o", a=Q_S, **fmt(qm(res, c.V[m]["F"], Q_S, END)), note=""))
    for kw, tag in ((dict(ibs=0.05), "IBS0.05"), (dict(ibs=0.15), "IBS0.15"), (dict(cap=5), "상한5일"), (dict(cap=20), "상한20일")):
        w, res, m = sim_rule("MRq", **kw)
        rows.append(dict(tag="인접 " + tag, rule="MRq", vehicle="F", fill="c", a=Q_S, **fmt(qm(res, c.V[m]["F"], Q_S, END)), note=""))
    idx = c.V["KQ"]["F"].idx
    zz = zprog(idx).to_numpy()
    cr = pd.Series(zz > 0.5, index=idx)
    e_l = (cr & ~cr.shift(1, fill_value=False)).to_numpy()
    crs = pd.Series(zz < -0.5, index=idx)
    e_s = (crs & ~crs.shift(1, fill_value=False)).to_numpy()
    w = S.hold(e_l, np.zeros(len(idx), bool), 10) + S.hold(e_s, np.zeros(len(idx), bool), 10, -1.0)
    res = S.sim(c.mask("KQ", "F", np.clip(w, -1, 1)), c.V["KQ"]["F"], "o")
    rows.append(dict(tag="z 돌파 뒤 10일 고정 보유", rule="FLq", vehicle="F", fill="o", a=Q_S, **fmt(qm(res, c.V["KQ"]["F"], Q_S, END)), note=""))
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "s2_vehicle.csv", index=False, encoding="utf-8-sig")
    print(t.drop(columns=["a"]).to_string())
    return t


# ---------------------------------------------------------------- 2 플라시보·디플레이티드·선택 축소
def fam_cells(kind):
    out = []
    if kind == "MRq":
        f = c.Fv["KQ"]["F"]
        for t, k in S.TRG:
            e0 = S.trig(f, t, k).fillna(False).to_numpy()
            for fl in S.FILT:
                e = e0 & S.filt(f, fl)
                for x, h in S.EXITS:
                    out.append((f"MR_KQ_{t}{k}_{fl}_{x}{h}_Fc", S.hold(e, S.exit_cond(f, x), h)))
    else:
        idx = c.V["KQ"]["F"].idx
        for n in (5, 10, 20, 60):
            zs = {k: S.zscore(v.astype(float), n).reindex(idx) for k, v in Sf["KQ"].items()}
            zs["combo"] = pd.concat(zs.values(), axis=1).mean(axis=1, skipna=False)
            for k, z in zs.items():
                for th in (0.0, 0.5, 1.0):
                    for short in (False, True):
                        out.append((f"FL_KQ_{k}_N{n}_th{th}_{'LS' if short else 'LF'}_fol", S.flow_pos(z.to_numpy(), th, short)))
    return out


def sh_mat(W, V, fill):
    on, oc = np.nan_to_num(V.on), np.nan_to_num(V.oc)
    n = W.shape[0]
    w1 = np.c_[np.zeros((n, 1)), W[:, :-1]]
    if fill == "c":
        r = w1 * (on + oc) - np.abs(W - w1) * V.c1
    else:
        w2 = np.c_[np.zeros((n, 2)), W[:, :-2]]
        r = w2 * on + w1 * oc - np.abs(w1 - w2) * V.c1
    return r.mean(1) / r.std(1) * np.sqrt(252)


def sect_placebo(draws=2000):
    V = c.V["KQ"]["F"]
    m0 = V.idx.searchsorted(pd.Timestamp(Q_S))
    Vw = S.Veh(V.idx[m0:], V.on[m0:], V.oc[m0:], V.roll[m0:], V.c1[m0:], V.cs[m0:], "w")
    rng = np.random.default_rng(SEED)
    out, hist = {}, {}
    for kind, fill in (("MRq", "c"), ("FLq", "o")):
        cells = fam_cells(kind)
        ids = [a for a, _ in cells]
        W = np.array([c.mask("KQ", "F", b)[m0:] for _, b in cells])
        act = sh_mat(W, Vw, fill)
        T = W.shape[1]
        sel = {"MRq": "MR_KQ_ibs0.1_none_ibs7010_Fc", "FLq": "FL_KQ_prog_N10_th0.5_LS_fol"}[kind]
        j = ids.index(sel)
        mx, md, one = [], [], []
        for _ in range(draws):
            s = rng.integers(100, T - 100)
            x = sh_mat(np.roll(W, s, axis=1), Vw, fill)
            mx.append(np.nanmax(x))
            md.append(np.nanmedian(x))
            one.append(x[j])
        mx, md, one = np.array(mx), np.array(md), np.array(one)
        out[kind] = dict(cells=len(ids), act_max=round(float(np.nanmax(act)), 3), act_med=round(float(np.nanmedian(act)), 3), pos_share=round(float((act > 0).mean()), 3), null_max95=round(float(np.quantile(mx, 0.95)), 3),
                         null_med95=round(float(np.quantile(md, 0.95)), 3), p_max=round(float((mx >= np.nanmax(act)).mean()), 4), p_med=round(float((md >= np.nanmedian(act)).mean()), 4),
                         sel=sel, sel_sh=round(float(act[j]), 3), p_sel=round(float((one >= act[j]).mean()), 4), null_sel95=round(float(np.quantile(one, 0.95)), 3))
        hist[kind] = dict(med=md, act=float(np.nanmedian(act)), mx=mx, amax=float(np.nanmax(act)))
    # 결합: 두 슬리브를 같은 크기로 이동(슬리브 사이 상관 구조는 유지, 시장과의 시점만 무효화)
    def rp(a_, b_):
        on, oc = np.nan_to_num(Vw.on), np.nan_to_num(Vw.oc)
        a1 = np.r_[0.0, a_[:-1]]
        b1, b2 = np.r_[0.0, b_[:-1]], np.r_[0.0, 0.0, b_[:-2]]
        return a1 * (on + oc) - np.abs(a_ - a1) * Vw.c1 + b2 * on + b1 * oc - np.abs(b1 - b2) * Vw.c1
    for tag, nm in (("P1", "FLqVT"), ("P1z", "FLzVT")):
        w1 = c.mask("KQ", "F", rule("MRq", c.Fv["KQ"]["F"], V.idx))[m0:]
        w2 = c.mask("KQ", "F", rule(nm, c.Fv["KQ"]["F"], V.idx))[m0:]
        r0 = rp(w1, w2)
        a0 = r0.mean() / r0.std() * np.sqrt(252)
        null = []
        for _ in range(draws):
            k = rng.integers(100, len(w1) - 100)
            r = rp(np.roll(w1, k), np.roll(w2, k))
            null.append(r.mean() / r.std() * np.sqrt(252))
        null = np.array(null)
        out[tag] = dict(act=round(float(a0), 3), null95=round(float(np.quantile(null, 0.95)), 3), null99=round(float(np.quantile(null, 0.99)), 3), p=round(float((null >= a0).mean()), 4))
    print("플라시보", json.dumps(out, ensure_ascii=False))
    RES["placebo"] = out
    return hist


def dsr(r, n_trials, sr_std_ann):
    x = r.to_numpy()
    T = len(x)
    sr = x.mean() / x.std()
    g3 = ((x - x.mean()) ** 3).mean() / x.std() ** 3
    g4 = ((x - x.mean()) ** 4).mean() / x.std() ** 4
    sd = sr_std_ann / np.sqrt(252)
    em = 0.5772156649
    sr0 = sd * ((1 - em) * norm.ppf(1 - 1 / n_trials) + em * norm.ppf(1 - 1 / (n_trials * np.e)))
    z = (sr - sr0) * np.sqrt(T - 1) / np.sqrt(1 - g3 * sr + (g4 - 1) / 4 * sr ** 2)
    return float(norm.cdf(z)), float(sr0 * np.sqrt(252))


def sect_dsr(daily):
    d = pd.read_csv(OUT / "cells_ext.csv", encoding="utf-8-sig")
    sd = float(d["sharpe"].std())
    o = dict(sr_std=round(sd, 3), n_cells=len(d))
    for nm in ("MRq", "MRqG", "MRp", "FLq", "FLqVT", "FLzVT"):
        for n in (2319, 200):
            p, s0 = dsr(daily[nm], n, sd)
            o[f"{nm}_N{n}"] = dict(dsr=round(p, 3), sr0=round(s0, 2))
    print("DSR", json.dumps(o, ensure_ascii=False))
    RES["dsr"] = o


def sect_wf():
    """선택 축소: (1) 전·후반 교차, (2) 매년 직전 3년 최고 셀을 다음 해에 쓴다."""
    V = c.V["KQ"]["F"]
    out = {}
    for kind, fill in (("MRq", "c"), ("FLq", "o")):
        cells = fam_cells(kind)
        R = pd.DataFrame({a: S.sim(c.mask("KQ", "F", b), V, fill)["r"] for a, b in cells}, index=V.idx).loc[Q_S:]
        sr = lambda x: x.mean() / x.std() * np.sqrt(252)
        A, B = R.loc[:"2020-06-30"], R.loc["2020-07-01":]
        sa, sb = sr(A), sr(B)
        out[kind] = dict(rank_corr=round(float(sa.rank().corr(sb.rank())), 3), A_best=sa.idxmax(), A_best_sh=round(float(sa.max()), 2), A_best_inB=round(float(sb[sa.idxmax()]), 2), B_best=sb.idxmax(), B_best_sh=round(float(sb.max()), 2), B_best_inA=round(float(sa[sb.idxmax()]), 2),
                         A_top10_inB=round(float(sb[sa.nlargest(10).index].mean()), 2), B_top10_inA=round(float(sa[sb.nlargest(10).index].mean()), 2), med_A=round(float(sa.median()), 2), med_B=round(float(sb.median()), 2))
        pick, parts = {}, []
        for y in range(2019, 2025):
            tr = R.loc[f"{y - 3}-01-01":f"{y - 1}-12-31"]
            te = R.loc[f"{y}-01-01":f"{y}-12-31"]
            s = sr(tr)
            top = s.nlargest(5).index
            pick[y] = top[0]
            parts.append(te[top].mean(axis=1))
        wf = pd.concat(parts)
        out[kind].update(wf=st_d(wf), wf_yearly=((1 + wf).groupby(wf.index.year).prod() - 1).mul(100).round(1).to_dict(), picks=pick)
    print("선택 축소", json.dumps(out, ensure_ascii=False))
    RES["wf"] = out


# ---------------------------------------------------------------- 2b 선택 없는 가족 평균·분해·기제
def sect_family():
    """셀을 고르지 않은 가족 동일가중. 선택 편향 없는 효과 크기 추정."""
    V = c.V["KQ"]["F"]
    fam = {"MR 225셀": (fam_cells("MRq"), "c", lambda a: True), "MR 청산 ibs70 45셀": (fam_cells("MRq"), "c", lambda a: "ibs7010" in a), "FL 96셀": (fam_cells("FLq"), "o", lambda a: True),
           "FL N5·10 48셀": (fam_cells("FLq"), "o", lambda a: "_N5_" in a or "_N10_" in a)}
    rows, keep = [], {}
    for k, (cells, fill, fn) in fam.items():
        R = pd.DataFrame({a: S.sim(c.mask("KQ", "F", b), V, fill)["r"] for a, b in cells if fn(a)}, index=V.idx).loc[Q_S:]
        r = R.mean(axis=1)
        keep[k] = r
        y = ((1 + r).groupby(r.index.year).prod() - 1).mul(100).round(1).to_dict()
        rows.append(dict(fam=k, cells=R.shape[1], **st_d(r), h1=st_d(r.loc[:"2020-06-30"])["sharpe"], h2=st_d(r.loc["2020-07-01":])["sharpe"], t_daily_nw=round(float(S.E.nw_t(r.to_numpy() * 1e4)), 2), k15=round(scale_to(r), 2), cagr_k15=st_d(r * scale_to(r))["cagr"], **{str(a): b for a, b in y.items()}))
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "s2_family.csv", index=False, encoding="utf-8-sig")
    print(t.T.to_string())
    a, b = keep["MR 225셀"], keep["FL N5·10 48셀"]
    j = (a / a.std() + b / b.std()) / 2 * 0.01
    RES["family_combo"] = dict(rho=round(float(a.corr(b)), 3), **st_d(j), h1=st_d(j.loc[:"2020-06-30"])["sharpe"], h2=st_d(j.loc["2020-07-01":])["sharpe"], k15=round(scale_to(j), 3), cagr_k15=st_d(j * scale_to(j))["cagr"])
    print("가족 평균 결합(위험 동일)", RES["family_combo"])


def sect_split():
    """평균회귀 첫 밤·나머지, 경로별 거래 수익, 기저 기제, 수급 분해."""
    rows = []
    V = c.V["KQ"]["F"]
    idx = V.idx
    ix = D.kq150_index().reindex(idx)
    e = D.etf("229200").reindex(idx)
    on, oc = np.nan_to_num(V.on), np.nan_to_num(V.oc)
    lf = pd.Series(np.log1p(on) + np.log1p(oc), index=idx).cumsum()
    for nm in ("MRq", "MRqG", "MRqD"):
        w, res, _ = sim_rule(nm)
        st = (w != 0) & (np.r_[0.0, w[:-1]] == 0)
        ex = (w == 0) & (np.r_[0.0, w[:-1]] != 0)
        i, j = np.flatnonzero(st), np.flatnonzero(ex)
        i = i[:len(j)]
        n1 = on[i + 1]
        tot = res["gt"][:len(i)]
        rest = (1 + tot) / (1 + n1) - 1
        pf = (lf.iloc[j].values - lf.iloc[i].values) * 1e4
        pi = (np.log(ix["close"]).iloc[j].values - np.log(ix["close"]).iloc[i].values) * 1e4
        pe = (np.log(e["close"]).iloc[j].values - np.log(e["close"]).iloc[i].values) * 1e4
        d = pf - pi
        if nm == "MRq":
            RES["mrq_basis_year"] = pd.Series(d, index=idx[i]).groupby(idx[i].year).mean().round(1).to_dict()
        rows.append(dict(rule=nm, n=len(i), total=round(tot.mean() * 1e4, 1), first_night=round(n1.mean() * 1e4, 1), rest=round(rest.mean() * 1e4, 1), t_rest=round(S.E.nw_t(rest * 1e4), 2), fut=round(pf.mean(), 1), idx=round(pi.mean(), 1), t_idx=round(S.E.nw_t(pi), 2), etf=round(pe.mean(), 1), t_etf=round(S.E.nw_t(pe[~np.isnan(pe)]), 2), basis=round(d.mean(), 1), t_basis=round(S.E.nw_t(d), 2), basis_share=round(d.mean() / pf.mean() * 100, 1)))
    # 코스피
    Vp = c.V["KP"]["F"]
    w, res, _ = sim_rule("MRp")
    ixp = D.k200_index().reindex(Vp.idx)
    onp, ocp = np.nan_to_num(Vp.on), np.nan_to_num(Vp.oc)
    lfp = pd.Series(np.log1p(onp) + np.log1p(ocp), index=Vp.idx).cumsum()
    st = (w != 0) & (np.r_[0.0, w[:-1]] == 0)
    ex = (w == 0) & (np.r_[0.0, w[:-1]] != 0)
    i, j = np.flatnonzero(st), np.flatnonzero(ex)
    i = i[:len(j)]
    n1 = onp[i + 1]
    tot = res["gt"][:len(i)]
    rest = (1 + tot) / (1 + n1) - 1
    pf = (lfp.iloc[j].values - lfp.iloc[i].values) * 1e4
    pi = (np.log(ixp["close"]).iloc[j].values - np.log(ixp["close"]).iloc[i].values) * 1e4
    d = pf - pi
    rows.append(dict(rule="MRp", n=len(i), total=round(tot.mean() * 1e4, 1), first_night=round(n1.mean() * 1e4, 1), rest=round(rest.mean() * 1e4, 1), t_rest=round(S.E.nw_t(rest * 1e4), 2), fut=round(pf.mean(), 1), idx=round(pi.mean(), 1), t_idx=round(S.E.nw_t(pi), 2), etf=np.nan, t_etf=np.nan, basis=round(d.mean(), 1), t_basis=round(S.E.nw_t(d), 2), basis_share=round(d.mean() / pf.mean() * 100, 1)))
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "s2_mr_split.csv", index=False, encoding="utf-8-sig")
    print(t.to_string())
    # 기저 기제: 일별 기저 변화 ~ 당일 지수 수익, 다음 날 되돌림
    fut = {"KQ": D.fut_ext("kq150"), "KP": D.fut_ext("k200")}
    ixs = {"KQ": D.kq150_index(), "KP": D.k200_index()}
    o = {}
    for m in ("KQ", "KP"):
        fr = fut[m][fut[m]["front"] == 1].drop_duplicates("date").set_index("date")["close"].astype(float)
        roll = fut[m][fut[m]["front"] == 1].drop_duplicates("date").set_index("date")["code"]
        b = (fr / ixs[m]["close"].reindex(fr.index) - 1) * 1e4
        db = b.diff().where(roll == roll.shift(1))
        ret = ixs[m]["close"].pct_change().reindex(fr.index) * 1e4
        d = pd.DataFrame({"db": db, "r": ret, "db1": db.shift(-1)}).loc[Q_S:].dropna()
        f1 = sm.OLS(d["db"], sm.add_constant(d["r"])).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
        f2 = sm.OLS(d["db1"], sm.add_constant(d["db"])).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
        o[m] = dict(mean_basis=round(float(b.loc[Q_S:].mean()), 1), slope=round(float(f1.params["r"]), 4), t=round(float(f1.tvalues["r"]), 2), rev=round(float(f2.params["db"]), 3), t_rev=round(float(f2.tvalues["db"]), 2),
                    by_year={int(y): round(float(v), 1) for y, v in b.loc[Q_S:].groupby(b.loc[Q_S:].index.year).mean().items()})
    print("기저 기제", json.dumps(o, ensure_ascii=False))
    RES["basis"] = o
    # 수급: 날짜 정렬, 롱·숏, 밤·낮, 가격 추세 포섭, 전후 구간
    lead = {}
    for m in ("KQ", "KP"):
        r = c.F[m]["C"].pct_change()
        lead[m] = {}
        for k, v in Sf[m].items():
            x = v.astype(float)
            x = (x - x.rolling(250, min_periods=200).mean()) / x.rolling(250, min_periods=200).std()
            lead[m][k] = {str(j): round(float(x.corr(r.shift(-j))), 3) for j in (-1, 0, 1, 2)}
    print("수급 날짜 정렬(T 수급 대 T+k 수익 상관)", json.dumps(lead, ensure_ascii=False))
    RES["lead"] = lead
    w, res, _ = sim_rule("FLq")
    w1, w2 = np.r_[0.0, w[:-1]], np.r_[0.0, 0.0, w[:-2]]
    o = dict(long_night=round(float(((w2 > 0) * on).sum() * 100), 1), long_day=round(float(((w1 > 0) * oc).sum() * 100), 1), short_night=round(float(-((w2 < 0) * on).sum() * 100), 1), short_day=round(float(-((w1 < 0) * oc).sum() * 100), 1))
    y = ser(res, V).loc[Q_S:]
    o["sharpe_by"] = {f"{a}~{b}": st_d(y.loc[f"{a}-01-01":f"{b}-12-31"])["sharpe"] for a, b in ((2016, 2017), (2018, 2020), (2021, 2022), (2023, 2024))}
    fq = c.Fv["KQ"]["F"]
    X = {}
    for sig in ("ma10", "mom5", "mom10", "mom20"):
        X[sig] = ser(S.sim(c.mask("KQ", "F", S.trend_w(fq, sig, True)), V, "c"), V).loc[Q_S:]
    X = pd.DataFrame(X)
    X["mkt"] = pd.Series(c.mret["KQ"]["F"], index=idx).loc[Q_S:]
    f = sm.OLS(y, sm.add_constant(X)).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
    o["subsume"] = dict(alpha_ann=round(float(f.params["const"] * 252 * 100), 2), t=round(float(f.tvalues["const"]), 2), r2=round(float(f.rsquared), 3), betas={k: round(float(v), 3) for k, v in f.params.items() if k != "const"})
    print("FLq 분해", json.dumps(o, ensure_ascii=False))
    RES["fl_split"] = o


def sect_vt():
    """MDD 15% 안에 드는 변동성 목표: (1) 상수 배수 (2) 사전 변동성 스케일(상한 3배). 목표치는 사후 탐색."""
    rows = []
    for nm, (m, f0) in CAND.items():
        if nm in ("FLqVT", "FLzVT"):
            continue
        V = c.V[m]["F"]
        a = Q_S if m == "KQ" else P_S
        base = rule(nm, c.Fv[m]["F"], V.idx)
        base = c.mask(m, "F", base)
        vol = c.Fv[m]["F"]["vol"].to_numpy()
        r0 = ser(c.run(m, base, "F", f0), V).loc[a:]
        k = scale_to(r0)
        best = None
        for tgt in np.arange(0.04, 0.81, 0.01):
            wv = c.mask(m, "F", S.vt(base, vol, tgt, 3.0))
            q = st_d(ser(c.run(m, wv, "F", f0), V).loc[a:])
            if q["mdd"] >= -15.0 and (best is None or q["cagr"] > best[1]["cagr"]):
                best = (round(float(tgt), 2), q)
        rows.append(dict(rule=nm, **{f"raw_{x}": y for x, y in st_d(r0).items()}, k15=round(k, 2), cagr_k15=st_d(r0 * k)["cagr"], vt_tgt=best[0] if best else np.nan, vt_cagr=best[1]["cagr"] if best else np.nan, vt_mdd=best[1]["mdd"] if best else np.nan, vt_sharpe=best[1]["sharpe"] if best else np.nan))
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "s2_vt.csv", index=False, encoding="utf-8-sig")
    print(t.to_string())


def sect_lean():
    """규칙별 사전 구간 코스피 전 구간(2006~2024) 이어 붙인 수치."""
    Vp = c.V["KP"]["F"]
    fp = c.Fv["KP"]["F"]
    w = np.where(Vp.idx < pd.Timestamp("2006-01-02"), 0.0, rule("MRp", fp, Vp.idx))
    res = S.sim(w, Vp, "c")
    RES["kp_full"] = fmt(qm(res, Vp, "2006-01-02", END))
    r = ser(res, Vp).loc["2006-01-02":]
    RES["kp_full_year"] = ((1 + r).groupby(r.index.year).prod() - 1).mul(100).round(1).to_dict()
    print("코스피 MRp 2006~2024", RES["kp_full"], RES["kp_full_year"])


# ---------------------------------------------------------------- 3 사전 구간
def sect_pre():
    comp = D.kosdaq_comp_ext()
    comp = comp[(comp["open"] > 0) & (comp["high"] > 0) & (comp["low"] > 0)]
    f = S.features(comp)
    idx = f.index
    n = len(idx)
    V = S.Veh(idx, (comp["open"] / comp["close"].shift() - 1).to_numpy(), (comp["close"] / comp["open"] - 1).to_numpy(), np.zeros(n, bool), np.full(n, S.C_F), np.full(n, S.CS_F), "KQcomp")
    PRE, POST = ("2006-10-02", "2015-09-30"), ("2015-10-01", END)
    rows = []
    for nm, fill in (("MRq", "c"), ("MRq", "o"), ("MRqD", "c"), ("MRqD", "o"), ("MRqG", "c"), ("FLq", "o"), ("FLq", "l"), ("FLqVT", "o"), ("FLz", "o"), ("FLzVT", "o"), ("FLzVT", "l")):
        w = rule(nm, f, idx)
        w = np.where(idx < pd.Timestamp(PRE[0]), 0.0, np.nan_to_num(w))
        res = S.sim(w, V, fill)
        for tag, (a, b) in (("사전 2006~2015-09", PRE), ("사후 2015-10~2024", POST)):
            q = qm(res, V, a, b)
            mk = pd.Series(V.on, index=idx).fillna(0).add(1).mul(pd.Series(V.oc, index=idx).fillna(0).add(1)).sub(1).loc[a:b]
            rows.append(dict(rule=nm, fill=fill, tag=tag, **fmt(q), yr_pos=((1 + ser(res, V).loc[a:b]).groupby(ser(res, V).loc[a:b].index.year).prod() > 1).mean().round(2), **hac_alpha(ser(res, V).loc[a:b], mk)))
        if nm in ("MRq", "FLq", "FLzVT") and fill in ("c", "o"):
            r = ser(res, V).loc[PRE[0]:PRE[1]]
            print(nm, fill, "사전 구간 연도별", ((1 + r).groupby(r.index.year).prod() - 1).mul(100).round(1).to_dict())
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "s2_presample.csv", index=False, encoding="utf-8-sig")
    print(t.to_string())
    # 코스피 사전 구간: 2006~2009 코스피200 선물
    Vp = c.V["KP"]["F"]
    fp = c.Fv["KP"]["F"]
    w = rule("MRp", fp, Vp.idx)
    w = np.where(Vp.idx < pd.Timestamp("2006-01-02"), 0.0, w)
    res = S.sim(w, Vp, "c")
    RES["pre_kp"] = {"2006~2009": fmt(qm(res, Vp, "2006-01-02", "2009-12-30"))}
    print("코스피 MRp 사전", RES["pre_kp"])
    return t


# ---------------------------------------------------------------- 4 결합·기준 장부
PF = {"P1": ["MRq", "FLqVT"], "P1z": ["MRq", "FLzVT"], "P1G": ["MRqG", "FLqVT"], "P1D": ["MRqD", "FLqVT"], "P2": ["MRq", "FLqVT", "MRp"], "P3": ["MRqD", "MRqG", "FLqVT"]}


def sect_port(daily):
    X = pd.DataFrame({k: v for k, v in daily.items()}).loc[Q_S:].fillna(0)
    print(X.corr().round(2).to_string())
    X.corr().round(3).to_csv(OUT / "s2_sleeve_corr.csv", encoding="utf-8-sig")
    W = {nm: pd.Series(np.abs(sim_rule(nm)[0]), index=c.V[CAND[nm][0]]["F"].idx).reindex(X.index).fillna(0) for nm in CAND}
    rows, yr = [], {}
    for nm, cols in PF.items():
        r = X[cols].sum(axis=1)
        k = scale_to(r)
        gross = sum(W[x] for x in cols) * k
        rk = r * k
        half = ("2016-01-04", "2020-06-30"), ("2020-07-01", END)
        vt_ = {}
        for tgt in (0.15, 0.20):
            sg = r.rolling(60).std().shift(1) * np.sqrt(252)
            wv = np.minimum(3.0, tgt / sg).fillna(0)
            vt_[tgt] = st_d((r * wv).loc["2016-06-01":])
        rows.append(dict(name=nm, sleeves="+".join(cols), **{f"sum_{a}": b for a, b in st_d(r).items()}, k15=round(k, 3), **{f"k_{a}": b for a, b in st_d(rk).items()}, gross_mean=round(float(gross.mean()), 2), gross_max=round(float(gross.max()), 2),
                         margin_max_pct=round(float(gross.max() * 21), 1), worst_day=round(float(rk.min() * 100), 2), mon_pos=round(float(((1 + rk).groupby([rk.index.year, rk.index.month]).prod() > 1).mean() * 100), 1),
                         t_daily_nw=round(float(S.E.nw_t(r.to_numpy() * 1e4)), 2), sh_h1=st_d(r.loc[half[0][0]:half[0][1]])["sharpe"], sh_h2=st_d(r.loc[half[1][0]:half[1][1]])["sharpe"], sh_ex2020=st_d(r[r.index.year != 2020])["sharpe"],
                         ban_share=round(float(r.loc["2023-11-06":].sum() / r.sum() * 100), 1), top5d=round(float(r.nlargest(5).sum() / r.sum() * 100), 1), vt15=vt_[0.15], vt20=vt_[0.20]))
        yr[nm] = ((1 + rk).groupby(rk.index.year).prod() - 1).mul(100).round(1).to_dict()
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "s2_portfolio.csv", index=False, encoding="utf-8-sig")
    print(t.T.to_string())
    print(pd.DataFrame(yr).to_string())
    # 기준 장부
    b = c.base
    comb = 0.5 * b["KP_book"] + 0.5 * b["KQ_book"]
    comb = comb.dropna()
    k1 = scale_to(X[PF["P1"]].sum(axis=1))
    RES["book_add"], RES["book_rho"], RES["leg_rho"] = {}, {}, {}
    for nm in ("P1", "P1z"):
        r1 = X[PF[nm]].sum(axis=1)
        RES["book_add"][nm] = [dict(lam=lam, **st_d(comb + lam * r1.reindex(comb.index).fillna(0))) for lam in (0, 0.25, 0.5, 1.0)]
        RES["book_rho"][nm] = {k: round(float(r1.reindex(comb.index).corr(v.reindex(comb.index))), 3) for k, v in {**{x: b[x] for x in b.columns}, "comb": comb}.items()}
        lg = {}
        for h in ("blog_ON_Q", "blog_ON_K", "beomgu_C3Q", "beomgu_C3VT30", "jehyeong_KQ", "jehyeong_KP", "jehyeong_ON"):
            q = D.rd(D.LAB / f"round13_slot_book/series/{h}.csv")["net"]
            lg[h] = dict(rho=round(float(r1.reindex(q.index).corr(q)), 3), n=int(r1.reindex(q.index).notna().sum()))
        RES["leg_rho"][nm] = lg
    print("13차 합산 + λ·결합(배수 1, 2022-12~2024-12)", RES["book_add"])
    print("결합 상관", RES["book_rho"])
    print("13차 다리 상관", RES["leg_rho"])
    return X, t, k1


# ---------------------------------------------------------------- 5 그림·셀
def figs(X, hist, k1):
    C1, C2, C3, G = "#2a78d6", "#eb6834", "#1baf7a", "#8a8987"
    fig, ax = plt.subplots(figsize=(10, 4.6))
    r1 = X[PF["P1"]].sum(axis=1) * k1
    mk = pd.Series(c.mret["KQ"]["F"], index=c.V["KQ"]["F"].idx).reindex(X.index).fillna(0)
    for s, lab, col, lw, dy in ((X["MRq"], "평균회귀 MRq", C1, 1.5, 7), (X["FLqVT"], "수급 추세 FLqVT", C2, 1.5, -7), (r1, f"결합 P1 × {k1:.2f}", C3, 2.0, 0), (mk, "코스닥150 지수 경로", G, 1.2, 0)):
        e = (1 + s).cumprod()
        ax.plot(e.index, e.values, color=col, lw=lw, label=lab)
        ax.annotate(lab, (e.index[-1], e.values[-1]), xytext=(6, dy), textcoords="offset points", fontsize=9, color="#3a3a38", va="center", annotation_clip=False)
    ax.set_yscale("log")
    ax.set_title("스윙 슬리브 누적(순수익, 2016.01~2024.12, 선물 체결, 사후 선택 표본)", fontsize=11, loc="left")
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    ax.set_xlim(X.index[0], X.index[-1])
    fig.tight_layout(rect=(0, 0, 0.86, 1))
    fig.savefig(OUT / "fig1_equity.png", dpi=130)
    plt.close(fig)
    d = pd.read_csv(OUT / "cells_ext.csv", encoding="utf-8-sig")
    import ast
    m = d[(d.fam == "MR") & (d.market == "KOSDAQ") & (d.vk == "F") & (d.fill == "c")].copy()
    p = m["params"].map(ast.literal_eval)
    tn = {"ibs": "IBS<{}", "rsi": "RSI2<{}", "dn": "연속 하락 {}일", "d5": "5일 낙폭 {}σ", "d10": "10일 낙폭 {}σ", "dd20": "20일 고점 낙폭 {}σ"}
    m["trg"] = p.map(lambda q: tn[q["t"]].format(int(q["k"]) if q["t"] in ("rsi", "dn") else q["k"]))
    m["ex"] = p.map(lambda q: f"{q['x']}{q['h']}")
    m["flt"] = p.map(lambda q: q["fl"])
    a_ = m[m.flt == "none"].pivot_table(index="trg", columns="ex", values="sharpe")
    order = [tn[t].format(int(k) if t in ("rsi", "dn") else k) for t, k in S.TRG]
    a_ = a_.loc[order, ["ibs7010", "rsi7010", "ma510", "t5", "t2"]]
    a_.columns = ["IBS>0.7", "RSI2>70", "종가>MA5", "5일 보유", "2일 보유"]
    f_ = d[(d.fam == "FL") & (d.market == "KOSDAQ")].copy()
    p = f_["params"].map(ast.literal_eval)
    kn = {"prog": "프로그램", "spot": "외인 현물", "fut": "코스닥150 선물 외인", "combo": "셋 평균"}
    f_["k"], f_["N"], f_["th"], f_["short"], f_["dr"] = p.map(lambda q: kn[q["k"]]), p.map(lambda q: q["N"]), p.map(lambda q: q["th"]), p.map(lambda q: q["short"]), p.map(lambda q: q["dr"])
    f_["lab"] = f_["k"] + " " + f_["N"].astype(str) + "일"
    b_ = f_[(f_.dr == 1) & (f_.short == True)].pivot_table(index="lab", columns="th", values="sharpe")  # noqa: E712
    b_ = b_.loc[[f"{kn[k]} {n}일" for k in ("prog", "spot", "combo", "fut") for n in (5, 10, 20, 60)]]
    b_.columns = [f"문턱 {x}" for x in b_.columns]
    fig, axs = plt.subplots(1, 2, figsize=(11.5, 6.0))
    for ax, tb, ttl in ((axs[0], a_, "코스닥 평균회귀 샤프(필터 없음, 선물 종가 체결), 청산별"), (axs[1], b_, "코스닥 수급 추세 샤프(추종·롱숏, 선물 시가 체결)")):
        ax.imshow(tb.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(range(tb.shape[1]), [str(x) for x in tb.columns], fontsize=8.5)
        ax.set_yticks(range(tb.shape[0]), list(tb.index), fontsize=8.5)
        ax.tick_params(length=0)
        for i in range(tb.shape[0]):
            for j in range(tb.shape[1]):
                ax.text(j, i, f"{tb.values[i, j]:.2f}".replace("-", "−"), ha="center", va="center", fontsize=8, color="#0b0b0b" if abs(tb.values[i, j]) < 0.6 else "#ffffff")
        ax.set_title(ttl, fontsize=10, loc="left")
        for sp in ax.spines.values():
            sp.set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "fig2_grid.png", dpi=130)
    plt.close(fig)
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, k, ttl in ((axs[0], "MRq", "평균회귀 225셀 샤프 중앙값"), (axs[1], "FLq", "수급 추세 96셀 샤프 중앙값")):
        ax.hist(hist[k]["med"], bins=45, color=G)
        ax.axvline(hist[k]["act"], color=C2, lw=2)
        ax.annotate(f"실제 {hist[k]['act']:.2f}".replace("-", "−"), (hist[k]["act"], ax.get_ylim()[1] * 0.9), xytext=(-6, 0), textcoords="offset points", ha="right", fontsize=9, color="#3a3a38")
        ax.set_title(ttl + "(신호 이동 2,000회)", fontsize=10, loc="left")
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "fig3_placebo.png", dpi=130)
    plt.close(fig)


def add_cells(t_veh, t_pre, X, tport):
    """cells.csv 에 2단계 셀을 덧붙인다(같은 ID 는 교체)."""
    p = OUT / "cells.csv"
    d = pd.read_csv(p, encoding="utf-8-sig")
    d = d[~d["id"].astype(str).str.startswith("S2_")]
    rows = []
    for i, r in t_veh.iterrows():
        rows.append(dict(id=f"S2_V{i:03d}", name=f"{r['rule']} {r['tag']}", market="KOSPI" if r["rule"] == "MRp" else "KOSDAQ", horizon="swing", window=f"{r['a']}~{END}", n=r["n"], net_bp=r["net_bp"], t_nw=r["t_nw"], cagr_pct=r["cagr"], mdd_pct=r["mdd"], stress_net_bp=r["stress_bp"], exec_ok=1,
                         notes=f"후보 점검 {r['vehicle']}/{r['fill']} {r['note']}"))
    for i, r in t_pre.iterrows():
        rows.append(dict(id=f"S2_P{i:03d}", name=f"{r['rule']} {r['tag']} 체결 {r['fill']}", market="KOSDAQ종합", horizon="swing", window=r["tag"], n=r["n"], net_bp=r["net_bp"], t_nw=r["t_nw"], cagr_pct=r["cagr"], mdd_pct=r["mdd"], stress_net_bp=r["stress_bp"], exec_ok=0,
                         notes="사전 구간 검증, 지수 경로, 거래 불가, 깃 밖 자료"))
    for i, r in tport.iterrows():
        rows.append(dict(id=f"S2_B{i:03d}", name=f"결합 {r['name']} {r['sleeves']}", market="KOSPI+KOSDAQ" if "MRp" in r["sleeves"] else "KOSDAQ", horizon="swing", window=f"{Q_S}~{END}", cagr_pct=r["sum_cagr"], mdd_pct=r["sum_mdd"], exec_ok=1, notes=f"슬리브 합, 자본 배수 1, 샤프 {r['sum_sharpe']}"))
    pd.concat([d, pd.DataFrame(rows)], ignore_index=True).to_csv(p, index=False, encoding="utf-8-sig")
    print("cells.csv 행 수", len(d) + len(rows), "2단계", len(rows))


def main():
    daily = sect_cands()
    tv = sect_vehicle()
    hist = sect_placebo()
    sect_dsr(daily)
    sect_wf()
    sect_family()
    sect_split()
    sect_vt()
    sect_lean()
    tp = sect_pre()
    X, tport, k1 = sect_port(daily)
    figs(X, hist, k1)
    add_cells(tv, tp, X, tport)
    (OUT / "s2_results.json").write_text(json.dumps(RES, ensure_ascii=False, indent=1, default=float), encoding="utf-8")


if __name__ == "__main__":
    main()
