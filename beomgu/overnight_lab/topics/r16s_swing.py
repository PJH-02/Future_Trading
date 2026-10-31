"""16차 스캔 · 스윙 관점(2~20일 보유). 1단계: 사전 정의 격자 전수 실행.

[봉인] 모든 자료는 r16s_swing_data 로더에서 2024-12-30 으로 잘린다. 격자는 결과를 보기 전 이 파일에 고정했다.

[체결 규약]
  Fc  선물 종가: T 15:30 지수 종가로 신호 → T 15:45 선물 종가 체결(2016-08 전은 15:00 → 15:15). 진입·청산 모두.
  Fco 선물 종가 진입 + T 신호 청산은 T+1 시가(밤 다리를 한 번 더 받는 변형).
  Fo  선물 시가: T 장 마감 뒤 확정 자료 → T+1 시가(2023-07-31 전 09:00, 뒤 08:45).
  Eo  ETF 시가: T 신호 → T+1 09:00 시가 단일가.
  같은 시가·종가로 계산하고 같은 가격에 체결하는 셀은 없다(체크리스트 1).

[비용] ETF 왕복 5bp, 스트레스 7bp + 1틱(5원/그해 평균 원가격, 단일가라 기본에는 없음). 선물 왕복 1.8bp, 스트레스 3bp, 롤 왕복 1회 추가.
[통계] 거래 = 같은 부호 노출의 연속 구간. 거래별 순수익에 engine.nw_t(HAC 5). CAGR·MDD 는 달력 전체(비활성일 0).
       상위 5건 = 상위 5건 합/총합. 전·후반 = 건수 반분 평균. 연도 부호 = 활동 있는 해의 복리 순수익 부호.
산출: round16_alpha_search/scan/swing/{cells.csv, cells_ext.csv}
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import engine as E  # noqa: E402
import r16s_swing_data as D  # noqa: E402

OUT = D.LAB / "round16_alpha_search/scan/swing"
OUT.mkdir(parents=True, exist_ok=True)
C_E, C_F, CS_E, CS_F = 2.5e-4, 0.9e-4, 3.5e-4, 1.5e-4     # 편도
STARTS = {("KP", "E"): "2010-01-04", ("KP", "F"): "2010-01-04", ("KQ", "E"): "2015-10-01", ("KQ", "F"): "2016-01-04"}
CODES = {"KP": ("069500", "114800", "122630", "k200"), "KQ": ("229200", "251340", "233740", "kq150")}


# ---------------------------------------------------------------- 차량
class Veh:
    def __init__(s, idx, on, oc, roll, c1, cs, name):
        s.idx, s.on, s.oc, s.roll, s.c1, s.cs, s.name = idx, on, oc, roll, c1, cs, name
        s.valid = ~np.isnan(on)


def veh_etf(code, cal=None):
    e = D.etf(code)
    raw = D.etf_raw(code)["close"]
    tick = (5.0 / raw.groupby(raw.index.year).transform("mean")).reindex(e.index).ffill().bfill()
    df = pd.DataFrame({"on": e["open"] / e["close"].shift() - 1, "oc": e["close"] / e["open"] - 1, "t": tick})
    if cal is not None:
        df = df.reindex(cal)
    t = df["t"].ffill().bfill().to_numpy()
    return Veh(df.index, df["on"].to_numpy(), df["oc"].to_numpy(), np.zeros(len(df), bool), np.full(len(df), C_E), CS_E + t / 2, code)


def veh_fut(name):
    f = D.fut_chain(name)
    return Veh(f.index, f["on"].to_numpy(), f["oc"].to_numpy(), f["roll"].to_numpy(bool), np.full(len(f), C_F), np.full(len(f), CS_F), "F" + name)


def sim(w, V, fill):
    """w[i]: i 일 종가에 결정한 목표 노출. 반환: 일별 순수익(기본·스트레스), 거래별 배열."""
    w = np.nan_to_num(np.asarray(w, float))
    w = np.where(V.valid, w, 0.0) if fill != "c" else w
    n = len(w)
    w1, w2 = np.r_[0.0, w[:-1]], np.r_[0.0, 0.0, w[:-2]]
    s = np.sign(w)
    st = (s != 0) & (s != np.r_[0.0, s[:-1]])
    tid = np.cumsum(st) * (s != 0)
    t1, t2 = np.r_[0, tid[:-1]], np.r_[0, 0, tid[:-2]]
    zero = np.zeros(n)
    if fill == "c":        # 당일 종가 체결
        eon, eoc, tn, toc = w1, w1, t1, t1
        ev = [(w1, w, t1, tid)]
    elif fill == "o":      # 다음 날 시가 체결
        eon, eoc, tn, toc = w2, w1, t2, t1
        ev = [(w2, w1, t2, t1)]
    elif fill == "l":      # 다음 날 종가 체결(하루 늦은 신호)
        eon, eoc, tn, toc = w2, w2, t2, t2
        ev = [(w2, w1, t2, t1)]
    elif fill == "co":     # 진입 종가, 청산 다음 날 시가(고정 크기 전용)
        eoc, toc = w1, t1
        eon, tn = np.where(w1 != 0, w1, w2), np.where(w1 != 0, t1, t2)
        ev = [(w1, np.where(w == 0, w1, w), t1, tid), (w2, np.where(w1 == 0, 0.0, w2), t2, t2)]
    else:
        raise ValueError(fill)
    on, oc = np.nan_to_num(V.on), np.nan_to_num(V.oc)
    a, b = eon * on, eoc * oc
    g = (1 + a) * (1 + b) - 1
    K = int(tid.max())
    lg = np.bincount(tn, np.log1p(a), K + 1) + np.bincount(toc, np.log1p(b), K + 1)
    cu, csu, cd, csd = zero.copy(), zero.copy(), np.zeros(K + 1), np.zeros(K + 1)
    for x, y, tx, ty in ev:
        dx = np.abs(y - x)
        flip = (np.sign(x) * np.sign(y)) < 0
        out = np.where(flip, np.abs(x), np.maximum(np.abs(x) - np.abs(y), 0.0))
        inn = dx - out
        cu += dx * V.c1
        csu += dx * V.cs
        cd += np.bincount(tx, out * V.c1, K + 1) + np.bincount(ty, inn * V.c1, K + 1)
        csd += np.bincount(tx, out * V.cs, K + 1) + np.bincount(ty, inn * V.cs, K + 1)
    if V.roll.any():
        hold = np.minimum(np.abs(eon), np.abs(eoc)) * (V.roll & (eon != 0) & (eoc != 0)) * (np.sign(eon) == np.sign(eoc))
        cu += hold * 2 * V.c1
        csu += hold * 2 * V.cs
        cd += np.bincount(toc, hold * 2 * V.c1, K + 1)
        csd += np.bincount(toc, hold * 2 * V.cs, K + 1)
    r = (1 + g) * (1 - cu) - 1
    rs = (1 + g) * (1 - csu) - 1
    gt = np.expm1(lg[1:])
    return dict(r=r, rs=rs, g=g, gt=gt, nt=np.expm1(lg[1:] + np.log1p(-cd[1:])), st=np.expm1(lg[1:] + np.log1p(-csd[1:])),
                ti=np.flatnonzero(st), expo=np.abs(eoc), hold=np.bincount(toc, minlength=K + 1)[1:])


def merge(*R):
    o = dict(r=sum(x["r"] for x in R), rs=sum(x["rs"] for x in R), g=sum(x["g"] for x in R), expo=sum(x["expo"] for x in R))
    for k in ("gt", "nt", "st", "ti", "hold"):
        o[k] = np.concatenate([x[k] for x in R])
    i = np.argsort(o["ti"], kind="stable")
    for k in ("gt", "nt", "st", "ti", "hold"):
        o[k] = o[k][i]
    return o


# ---------------------------------------------------------------- 특성
def features(ix):
    C, H, L = ix["close"], ix["high"], ix["low"]
    d = C.diff()
    up = d.clip(lower=0).ewm(alpha=0.5, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=0.5, adjust=False).mean()
    f = pd.DataFrame({"C": C, "rsi2": (100 - 100 / (1 + up / dn)).where(dn > 0, 100.0), "ibs": ((C - L) / (H - L)).where(H > L)})
    r = C.pct_change()
    sig = r.rolling(20).std()
    f["sig"], f["vol"] = sig, sig * np.sqrt(252)
    for n in (5, 10, 20, 50, 60, 100, 120, 200):
        f[f"ma{n}"] = C.rolling(n).mean()
    for n in (5, 10, 20, 60, 120):
        f[f"mom{n}"] = C / C.shift(n) - 1
    for n, m in ((20, 10), (55, 20), (120, 60)):
        f[f"hi{n}"], f[f"lo{n}"] = C.shift(1).rolling(n).max(), C.shift(1).rolling(n).min()
        f[f"hi{m}"], f[f"lo{m}"] = C.shift(1).rolling(m).max(), C.shift(1).rolling(m).min()
    f["dnrun"] = (d < 0).groupby((d >= 0).cumsum()).cumsum()
    f["uprun"] = (d > 0).groupby((d <= 0).cumsum()).cumsum()
    f["d5"] = (C / C.shift(5) - 1) / (sig * np.sqrt(5))
    f["d10"] = (C / C.shift(10) - 1) / (sig * np.sqrt(10))
    f["dd20"] = (C / C.rolling(20).max() - 1) / (sig * np.sqrt(20))
    return f


def hold(ent, ex, maxh, side=1.0):
    """진입 신호일 종가에 w=side, ex 또는 maxh 일 경과로 0. 청산일 진입 신호가 있으면 유지."""
    n = len(ent)
    w = np.zeros(n)
    h = 0
    for i in range(n):
        if h > 0:
            h += 1
            if ex[i] or h > maxh:
                h = 0
            else:
                w[i] = side
        if h == 0 and ent[i]:
            w[i], h = side, 1
    return w


def donchian(C, hi, lo, xhi, xlo, short):
    n = len(C)
    w = np.zeros(n)
    p = 0.0
    for i in range(n):
        if p > 0 and C[i] < xlo[i]:
            p = 0.0
        elif p < 0 and C[i] > xhi[i]:
            p = 0.0
        if p == 0:
            if C[i] > hi[i]:
                p = 1.0
            elif short and C[i] < lo[i]:
                p = -1.0
        w[i] = p
    return w


def band(w, b=0.1):
    """크기 조절 잡음 억제: |Δ| < b 면 이전 크기 유지(부호가 바뀌거나 0 이 되면 즉시 반영)."""
    o = w.copy()
    for i in range(1, len(o)):
        if o[i] != 0 and np.sign(o[i]) == np.sign(o[i - 1]) and abs(o[i] - o[i - 1]) < b:
            o[i] = o[i - 1]
    return o


def vt(w, vol, tgt, cap):
    return band(w * np.minimum(cap, tgt / np.where(vol > 0, vol, np.nan)))


# ---------------------------------------------------------------- 지표
def summary(res, V, start, base, mret):
    idx = V.idx
    m0 = idx.searchsorted(pd.Timestamp(start))
    r, rs = res["r"][m0:], res["rs"][m0:]
    d = idx[m0:]
    keep = res["ti"] >= 0
    nt, gt, st = res["nt"][keep], res["gt"][keep], res["st"][keep]
    n = len(nt)
    o = dict(n=n, window=f"{d[0].date()}~{d[-1].date()}")
    if n == 0:
        return o
    eq = np.cumprod(1 + r)
    tot = nt.sum()
    h = n // 2
    y = pd.Series(r, index=d)
    act = pd.Series(res["expo"][m0:], index=d)
    yr = (1 + y).groupby(d.year).prod() - 1
    ya = act.groupby(d.year).sum()
    yr = yr[ya > 0]
    ypos = int((yr > 0).sum())
    sd = r.std()
    mm = mret[m0:]
    b = np.cov(r, mm)[0, 1] / mm.var() if mm.var() > 0 else np.nan
    corr = {}
    if base is not None:
        bb = base.reindex(d)
        for c in bb.columns:
            ok = bb[c].notna().to_numpy()
            corr[c] = np.corrcoef(r[ok], bb[c].to_numpy()[ok])[0, 1] if ok.sum() > 50 and r[ok].std() > 0 else np.nan
    o.update(gross_bp=gt.mean() * 1e4, net_bp=nt.mean() * 1e4, t_nw=E.nw_t(nt * 1e4), cagr_pct=(eq[-1] ** (252 / len(r)) - 1) * 100 if eq[-1] > 0 else -100.0,
             mdd_pct=(eq / np.maximum.accumulate(eq) - 1).min() * 100, yearly_pos=f"{ypos}/{len(yr)}", ypos_share=ypos / max(len(yr), 1),
             top5_share_pct=np.sort(nt)[-5:].sum() / tot * 100 if tot > 0 and n >= 5 else np.nan,
             half1_net_bp=nt[:h].mean() * 1e4 if h else np.nan, half2_net_bp=nt[h:].mean() * 1e4 if h else np.nan, stress_net_bp=st.mean() * 1e4,
             win_pct=(nt > 0).mean() * 100, avg_hold=res["hold"].mean(), expo=res["expo"][m0:].mean(), max_expo=res["expo"][m0:].max(),
             sharpe=r.mean() / sd * np.sqrt(252) if sd > 0 else np.nan, t_daily=r.mean() / sd * np.sqrt(len(r)) if sd > 0 else np.nan,
             beta=b, alpha_ann_pct=(r.mean() - b * mm.mean()) * 252 * 100, tpy=n / (len(r) / 252), **{f"rho_{k}": v for k, v in corr.items()})
    return o


COLS = ["id", "name", "market", "horizon", "window", "n", "gross_bp", "net_bp", "t_nw", "cagr_pct", "mdd_pct", "yearly_pos", "top5_share_pct",
        "half1_net_bp", "half2_net_bp", "stress_net_bp", "corr_base", "exec_ok", "notes"]
FAM_NOTE = {"MR": "지수 신호 평균회귀", "MRS": "지수 신호 과매수 숏", "TR": "지수 추세", "XS": "코스피·코스닥 교차", "FL": "수급 N일 누적 추세", "US": "미국 추세 필터",
            "USMR": "미국 필터 평균회귀", "VK": "VKOSPI 공포 뒤 매수(깃 밖 자료)", "X": "사후 추가"}


def us_known(s, idx):
    """KR 날짜 T 09:00 전에 알 수 있는 값: 미국 날짜 d < T 의 마지막 값."""
    a = s.copy()
    a.index = a.index + pd.Timedelta(days=1)
    return a.reindex(a.index.union(idx)).ffill().reindex(idx)


class Ctx:
    def __init__(s):
        s.base = D.base_streams()
        ix = {"KP": D.k200_index(), "KQ": D.kq150_index()}
        s.F = {m: features(ix[m]) for m in ix}
        s.V = {}
        for m, (e1, ei, e2, fu) in CODES.items():
            E1 = veh_etf(e1)
            s.V[m] = {"E": E1, "Ei": veh_etf(ei, E1.idx), "E2": veh_etf(e2, E1.idx), "F": veh_fut(fu)}
        s.Fv = {m: {k: s.F[m].reindex(s.V[m][k].idx) for k in ("E", "F")} for m in s.V}
        s.mret = {}
        for m in s.V:
            cc = (1 + np.nan_to_num(s.V[m]["E"].on)) * (1 + np.nan_to_num(s.V[m]["E"].oc)) - 1
            cs = pd.Series(cc, index=s.V[m]["E"].idx)
            s.mret[m] = {"E": cc, "F": cs.reindex(s.V[m]["F"].idx).fillna(0).to_numpy()}
        s.rows = []
        s.keep = {}

    def run(s, m, w, vk, fill):
        V = s.V[m][vk]
        if vk == "E" and (np.nan_to_num(w) < 0).any():
            Vi = s.V[m]["Ei"]
            ws = np.clip(-np.nan_to_num(w), 0, None) * Vi.valid
            return merge(sim(np.clip(np.nan_to_num(w), 0, None), V, fill), sim(ws, Vi, fill))
        return sim(w, V, fill)

    def mask(s, m, vk, w):
        w = np.nan_to_num(np.asarray(w, float)).copy()
        w[s.V[m][vk].idx < pd.Timestamp(STARTS[(m, vk)])] = 0.0
        return w

    def add(s, fam, cid, name, m, w, vk, fill, params, note=""):
        w = s.mask(m, vk, w)
        res = s.run(m, w, vk, fill)
        o = summary(res, s.V[m][vk], STARTS[(m, vk)], s.base, s.mret[m][vk])
        o.update(id=cid, name=name, market={"KP": "KOSPI", "KQ": "KOSDAQ", "X": "KOSPI+KOSDAQ"}[m], horizon="swing", fam=fam, vk=vk, fill=fill, exec_ok=1,
                 notes=f"{FAM_NOTE[fam]}; 체결 {fill}/{vk}{'; ' + note if note else ''}", params=params)
        s.rows.append(o)


# ---------------------------------------------------------------- 격자
TRG = [("ibs", 0.1), ("ibs", 0.2), ("ibs", 0.3), ("rsi", 5), ("rsi", 10), ("rsi", 20), ("dn", 2), ("dn", 3), ("dn", 4),
       ("d5", 1.5), ("d5", 2.0), ("d10", 1.5), ("d10", 2.0), ("dd20", 1.5), ("dd20", 2.0)]
EXITS = [("t", 2), ("t", 5), ("ibs70", 10), ("ma5", 10), ("rsi70", 10)]
FILT = ("none", "up", "dn")


def trig(f, t, k):
    if t == "ibs":
        return f["ibs"] < k
    if t == "rsi":
        return f["rsi2"] < k
    if t == "dn":
        return f["dnrun"] >= k
    return f[t] <= -k


def exit_cond(f, x):
    if x == "t":
        return np.zeros(len(f), bool)
    if x.startswith("ibs"):
        return (f["ibs"] > int(x[3:]) / 100).to_numpy()
    if x == "ma5":
        return (f["C"] > f["ma5"]).to_numpy()
    if x == "rsi70":
        return (f["rsi2"] > 70).to_numpy()
    raise ValueError(x)


def filt(f, k):
    if k == "none":
        return np.ones(len(f), bool)
    up = (f["C"] > f["ma200"]).to_numpy()
    return up if k == "up" else (f["C"] < f["ma200"]).fillna(False).to_numpy() & f["ma200"].notna().to_numpy()


def fam_mr(c):
    for m in ("KP", "KQ"):
        for vk, fill in (("F", "c"), ("F", "co"), ("E", "o")):
            f = c.Fv[m][vk]
            for (t, k) in TRG:
                e0 = trig(f, t, k).fillna(False).to_numpy()
                for fl in FILT:
                    e = e0 & filt(f, fl)
                    for (x, h) in EXITS:
                        w = hold(e, exit_cond(f, x), h)
                        c.add("MR", f"MR_{m}_{t}{k}_{fl}_{x}{h}_{vk}{fill}", f"{m} {t}{k} 필터 {fl} 청산 {x}{h} {vk}{fill}", m, w, vk, fill,
                              dict(m=m, t=t, k=k, fl=fl, x=x, h=h, vk=vk, fill=fill))


def fam_mrs(c):
    trg = [("ibs", 0.8), ("ibs", 0.9), ("rsi", 90), ("rsi", 95), ("up", 3), ("up", 4)]
    ex = [("t", 2), ("t", 5), ("ibs30", 10), ("ma5", 10)]
    for m in ("KP", "KQ"):
        f = c.Fv[m]["F"]
        for t, k in trg:
            e0 = (f["ibs"] > k if t == "ibs" else f["rsi2"] > k if t == "rsi" else f["uprun"] >= k).fillna(False).to_numpy()
            for fl in ("none", "dn"):
                e = e0 & filt(f, fl)
                for x, h in ex:
                    xc = np.zeros(len(f), bool) if x == "t" else (f["ibs"] < 0.3).to_numpy() if x == "ibs30" else (f["C"] < f["ma5"]).to_numpy()
                    c.add("MRS", f"MRS_{m}_{t}{k}_{fl}_{x}{h}", f"{m} 과매수 숏 {t}{k} 필터 {fl} 청산 {x}{h}", m, hold(e, xc, h, -1.0), "F", "c",
                          dict(m=m, t=t, k=k, fl=fl, x=x, h=h))


TRSIG = ["ma20", "ma50", "ma100", "ma200", "x5_20", "x20_60", "x50_200", "mom20", "mom60", "mom120", "vote", "don20", "don55", "don120"]


def trend_w(f, sig, short):
    C = f["C"].to_numpy()
    if sig.startswith("don"):
        n = int(sig[3:])
        x = {20: 10, 55: 20, 120: 60}[n]
        return donchian(C, f[f"hi{n}"].to_numpy(), f[f"lo{n}"].to_numpy(), f[f"hi{x}"].to_numpy(), f[f"lo{x}"].to_numpy(), short)
    if sig == "vote":
        v = sum(np.sign(f[f"mom{n}"].to_numpy()) for n in (20, 60, 120)) / 3
        return np.nan_to_num(v) if short else np.nan_to_num(np.clip(v, 0, None))
    if sig.startswith("ma"):
        s = (C > f[sig].to_numpy()) * 2.0 - 1
        ok = f[sig].notna().to_numpy()
    elif sig.startswith("mom"):
        s = np.sign(f[sig].to_numpy())
        ok = f[sig].notna().to_numpy()
    else:
        a, b = sig[1:].split("_")
        s = (f[f"ma{a}"].to_numpy() > f[f"ma{b}"].to_numpy()) * 2.0 - 1
        ok = f[f"ma{b}"].notna().to_numpy()
    s = np.where(ok, s, 0.0)
    return s if short else np.clip(s, 0, None)


def fam_trend(c):
    for m in ("KP", "KQ"):
        for vk, fill in (("F", "c"), ("E", "o")):
            f = c.Fv[m][vk]
            vol = f["vol"].to_numpy()
            for sig in TRSIG:
                for short in (False, True):
                    w0 = trend_w(f, sig, short)
                    for sz in ("1x", "vt15"):
                        w = w0 if sz == "1x" else vt(w0, vol, 0.15, 2.0 if vk == "F" else 1.0)
                        c.add("TR", f"TR_{m}_{sig}_{'LS' if short else 'LF'}_{sz}_{vk}{fill}", f"{m} 추세 {sig} {'롱숏' if short else '롱현금'} {sz} {vk}{fill}", m, w, vk, fill,
                              dict(m=m, sig=sig, short=short, sz=sz, vk=vk, fill=fill))


def add_pair(c, fam, cid, name, wp, wq, vk, fill, params, note=""):
    """두 시장 동시. 달력은 코스피 차량 기준으로 일수익을 합친다(코스닥 차량 시작 뒤 구간)."""
    wp, wq = c.mask("KP", vk, wp), c.mask("KQ", vk, wq)
    start = max(STARTS[("KP", vk)], STARTS[("KQ", vk)])
    wp = np.where(c.V["KP"][vk].idx < pd.Timestamp(start), 0.0, wp)
    a, b = c.run("KP", wp, vk, fill), c.run("KQ", wq, vk, fill)
    Vp, Vq = c.V["KP"][vk], c.V["KQ"][vk]
    pos = Vp.idx.get_indexer(Vq.idx)
    assert (pos >= 0).all()
    rb, rsb = np.zeros(len(Vp.idx)), np.zeros(len(Vp.idx))
    ex = np.zeros(len(Vp.idx))
    rb[pos], rsb[pos], ex[pos] = b["r"], b["rs"], b["expo"]
    bb = dict(b, r=rb, rs=rsb, g=rb, expo=ex, ti=pos[b["ti"]])
    res = merge(a, bb)
    o = summary(res, Vp, start, c.base, c.mret["KP"][vk] * 0.5 + pd.Series(c.mret["KQ"][vk], index=Vq.idx).reindex(Vp.idx).fillna(0).to_numpy() * 0.5)
    o.update(id=cid, name=name, market="KOSPI+KOSDAQ", horizon="swing", fam=fam, vk=vk, fill=fill, exec_ok=1, notes=f"{FAM_NOTE[fam]}; 체결 {fill}/{vk}{'; ' + note if note else ''}", params=params)
    c.rows.append(o)


def fam_cross(c):
    for vk, fill in (("F", "c"), ("E", "o")):
        fp, fq = c.Fv["KP"][vk], c.Fv["KQ"][vk].reindex(c.V["KP"][vk].idx)
        for n in (20, 60, 120):
            a, b = fp[f"mom{n}"].to_numpy(), fq[f"mom{n}"].to_numpy()
            pick_q = (b > a).astype(float)
            pick_p = (a >= b).astype(float)
            ok = ~np.isnan(a) & ~np.isnan(b)
            for ab in (False, True):
                wp0 = pick_p * ok * ((a > 0) if ab else 1.0)
                wq0 = pick_q * ok * ((b > 0) if ab else 1.0)
                for sz in ("1x", "vt15"):
                    cap = 2.0 if vk == "F" else 1.0
                    wp = wp0 if sz == "1x" else vt(wp0, fp["vol"].to_numpy(), 0.15, cap)
                    wq = wq0 if sz == "1x" else vt(wq0, fq["vol"].to_numpy(), 0.15, cap)
                    wq_v = pd.Series(wq, index=c.V["KP"][vk].idx).reindex(c.V["KQ"][vk].idx).fillna(0).to_numpy()
                    add_pair(c, "XS", f"XS_RS{n}_{'abs' if ab else 'all'}_{sz}_{vk}{fill}", f"상대강도 {n}일 {'절대필터' if ab else '항상보유'} {sz} {vk}{fill}", wp, wq_v, vk, fill,
                             dict(n=n, ab=ab, sz=sz, vk=vk, fill=fill))
    vk, fill = "F", "c"
    fp, fq = c.Fv["KP"][vk], c.Fv["KQ"][vk].reindex(c.V["KP"][vk].idx)
    for n in (20, 60, 120):
        s = np.sign(fq[f"mom{n}"].to_numpy() - fp[f"mom{n}"].to_numpy())
        s = np.nan_to_num(s)
        for sz in ("0.5x", "1x"):
            k = 0.5 if sz == "0.5x" else 1.0
            wq_v = pd.Series(s * k, index=c.V["KP"][vk].idx).reindex(c.V["KQ"][vk].idx).fillna(0).to_numpy()
            add_pair(c, "XS", f"XS_LS{n}_{sz}", f"코스닥 대 코스피 상대 추세 롱숏 {n}일 다리당 {sz}", -s * k, wq_v, vk, fill, dict(n=n, sz=sz), note="기각된 스프레드 z20 평균회귀와 달리 추세 추종, 선물 양다리")


def zscore(x, n):
    s = x.rolling(n).sum()
    return (s - s.rolling(250, min_periods=200).mean()) / s.rolling(250, min_periods=200).std()


def flow_series():
    kp, kq, fu = D.investor("kospi"), D.investor("kosdaq"), D.investor("fut")
    pk, pq = D.program("kospi"), D.program("kosdaq")
    kf = D.kq_fut_foreign()
    return {"KP": {"spot": kp["net_foreign"], "fut": fu["net_foreign"], "prog": pk["nonarb_net"]},
            "KQ": {"spot": kq["net_foreign"], "fut": kf["foreign"].astype(float), "prog": pq["nonarb_net"]}}


def flow_pos(z, th, short):
    """z > th 진입 → z < 0 청산. short 면 대칭."""
    w = np.zeros(len(z))
    p = 0.0
    for i, v in enumerate(z):
        if np.isnan(v):
            w[i] = p
            continue
        if p > 0 and v < 0:
            p = 0.0
        elif p < 0 and v > 0:
            p = 0.0
        if p == 0:
            if v > th:
                p = 1.0
            elif short and v < -th:
                p = -1.0
        w[i] = p
    return w


def fam_flow(c):
    S = flow_series()
    for m in ("KP", "KQ"):
        idx = c.V[m]["F"].idx
        for n in (5, 10, 20, 60):
            zs = {k: zscore(v.astype(float), n).reindex(idx) for k, v in S[m].items()}
            zs["combo"] = pd.concat(zs.values(), axis=1).mean(axis=1, skipna=False)
            for k, z in zs.items():
                zz = z.to_numpy()
                for th in (0.0, 0.5, 1.0):
                    for short in (False, True):
                        for dr in (1, -1):
                            w = flow_pos(zz * dr, th, short)
                            tag = f"FL_{m}_{k}_N{n}_th{th}_{'LS' if short else 'LF'}_{'fol' if dr > 0 else 'con'}"
                            c.add("FL", tag, f"{m} {k} {n}일 누적 z {th} {'롱숏' if short else '롱현금'} {'추종' if dr > 0 else '역행'} Fo", m, w, "F", "o",
                                  dict(m=m, k=k, N=n, th=th, short=short, dr=dr), note="기각된 1일 z·밤 보유와 달리 N일 누적·청산 신호까지 보유")


USF = ["spx_ma200", "spx_ma100", "spx_ma50", "ndx_ma200", "ndx_ma100", "ndx_ma50", "spx_mom60", "ndx_mom60", "ndx_mom20"]


def us_flags():
    o = {}
    for nm, key in (("spx", "spx"), ("ndx", "ndx_comp")):
        c = D.us(key)["close"]
        for n in (50, 100, 200):
            o[f"{nm}_ma{n}"] = (c > c.rolling(n).mean()).where(c.rolling(n).count() >= n).astype(float)
        for n in (20, 60):
            o[f"{nm}_mom{n}"] = (c > c.shift(n)).astype(float)
    return o


def fam_us(c):
    U = us_flags()
    for m in ("KP", "KQ"):
        for vk, fill in (("F", "c"), ("E", "o")):
            f = c.Fv[m][vk]
            idx = c.V[m][vk].idx
            for k in USF:
                u = us_known(U[k], idx)
                if fill == "o":                 # T+1 시가 체결이면 T+1 아침까지 나온 미국 종가를 쓸 수 있다
                    u = u.shift(-1)
                u = u.fillna(0).to_numpy()
                for base in ("all", "ma50", "ma200"):
                    kr = np.ones(len(f)) if base == "all" else (f["C"] > f[base]).to_numpy().astype(float)
                    c.add("US", f"US_{m}_{k}_{base}_{vk}{fill}", f"{m} 미국 {k} × 한국 {base} {vk}{fill}", m, u * kr, vk, fill, dict(m=m, k=k, base=base, vk=vk, fill=fill))
        f = c.Fv[m]["F"]
        idx = c.V[m]["F"].idx
        for k in ("spx_ma200", "ndx_ma50", "ndx_mom20"):
            u = us_known(U[k], idx).fillna(0).to_numpy() > 0
            for (t, th) in (("ibs", 0.3), ("rsi", 10)):
                for x, h in (("ibs70", 10), ("ma5", 10)):
                    e = trig(f, t, th).fillna(False).to_numpy() & u
                    c.add("USMR", f"USMR_{m}_{k}_{t}{th}_{x}", f"{m} 미국 {k} 게이트 {t}{th} 청산 {x}", m, hold(e, exit_cond(f, x), h), "F", "c", dict(m=m, k=k, t=t, th=th, x=x))


def fam_vk(c):
    v = D.vkospi_ext()["close"]
    z = (v - v.rolling(250, min_periods=200).mean()) / v.rolling(250, min_periods=200).std()
    for m in ("KP", "KQ"):
        idx = c.V[m]["F"].idx
        vv, zz = v.reindex(idx), z.reindex(idx)
        fall = (vv < vv.shift(1)).to_numpy()
        for tag, e0 in (("lv25", (vv >= 25).to_numpy()), ("lv30", (vv >= 30).to_numpy()), ("z15", (zz >= 1.5).to_numpy()), ("z20", (zz >= 2.0).to_numpy())):
            for h in (5, 10, 20):
                c.add("VK", f"VK_{m}_{tag}_t{h}", f"{m} VKOSPI {tag} 하락 확인 뒤 {h}일 보유 Fo", m, hold(e0 & fall, np.zeros(len(idx), bool), h), "F", "o", dict(m=m, tag=tag, h=h), note="깃 밖 자료")


def zbar(c, m, S=None):
    """수급 앙상블 z: 프로그램·외인 현물 × N 5·10 평균(같은 비중)."""
    S = S or flow_series()
    idx = c.V[m]["F"].idx
    return pd.concat([zscore(S[m][k].astype(float), n).reindex(idx) for k in ("prog", "spot") for n in (5, 10)], axis=1).mean(axis=1, skipna=False)


def composite(f):
    return (trig(f, "ibs", 0.1) | trig(f, "rsi", 5) | trig(f, "d5", 1.5) | trig(f, "dd20", 1.5)).fillna(False).to_numpy()


def fam_x(c):
    """1단계 결과를 본 뒤 추가한 격자(사후 추가 표시). 인접 문턱 확인, 수급 기제 분해, 포섭 점검 목적."""
    N2 = "2단계 추가(1단계 결과를 본 뒤)"
    S = flow_series()
    kq = D.investor("kosdaq")
    inst = kq[["net_trust", "net_pension", "net_fin_inv", "net_insurance", "net_bank", "net_other_fin", "net_private_fund"]].fillna(0).sum(axis=1)
    for m in ("KQ", "KP"):                                     # X1 복합 트리거, X2 청산 문턱 인접
        f = c.Fv[m]["F"]
        for x in ("ibs60", "ibs70", "ibs80"):
            c.add("X", f"X1_{m}_composite_{x}", f"{m} 복합 과매도(ibs<0.1 또는 rsi<5 또는 d5 또는 dd20) 청산 {x} Fc", m, hold(composite(f), exit_cond(f, x), 10), "F", "c", dict(g="X1", m=m, x=x), note=N2)
        tr = [("ibs", 0.1), ("dd20", 1.5), ("d5", 2.0), ("rsi", 5)] if m == "KQ" else [("ibs", 0.1), ("ibs", 0.2)]
        for t, k in tr:
            for x in ("ibs60", "ibs80"):
                c.add("X", f"X2_{m}_{t}{k}_{x}", f"{m} {t}{k} 청산 {x} Fc", m, hold(trig(f, t, k).fillna(False).to_numpy(), exit_cond(f, x), 10), "F", "c", dict(g="X2", m=m, t=t, k=k, x=x), note=N2)
    idx = c.V["KQ"]["F"].idx
    for k, ser, dr in (("inst", inst, 1), ("indiv", kq["net_individual"].astype(float), -1)):   # X3 기관·개인 역행
        for n in (5, 10):
            z = zscore(ser, n).reindex(idx).to_numpy()
            for th in (0.0, 0.5, 1.0):
                for short in (False, True):
                    c.add("X", f"X3_KQ_{k}_N{n}_th{th}_{'LS' if short else 'LF'}", f"KQ {k} {n}일 누적 z {th} {'롱숏' if short else '롱현금'} {'추종' if dr > 0 else '역행'} Fo", "KQ",
                          flow_pos(z * dr, th, short), "F", "o", dict(g="X3", k=k, N=n, th=th, short=short), note=N2)
    zb = zbar(c, "KQ", S)
    zz = zb.to_numpy()
    for th in (0.0, 0.5, 1.0):                                  # X4 수급 앙상블
        for short in (False, True):
            c.add("X", f"X4_KQ_zbar_th{th}_{'LS' if short else 'LF'}", f"KQ 수급 앙상블 z̄ {th} {'롱숏' if short else '롱현금'} Fo", "KQ", flow_pos(zz, th, short), "F", "o", dict(g="X4", th=th, short=short), note=N2)
    c.add("X", "X4_KQ_zbar_linear", "KQ 수급 앙상블 선형 크기 clip(z̄,−1,1) Fo", "KQ", band(np.clip(np.nan_to_num(zz), -1, 1)), "F", "o", dict(g="X4", th=-1, short=True), note=N2)
    f = c.Fv["KQ"]["F"]
    zlag = zb.shift(1).to_numpy()                               # X5 평균회귀 수급 게이트: T 수급은 15:30 에 미확정이라 T−1 까지
    for tag, e0 in (("ibs0.1", trig(f, "ibs", 0.1).fillna(False).to_numpy()), ("comp", composite(f))):
        for th in (-0.5, 0.0):
            c.add("X", f"X5_KQ_{tag}_gate{th}", f"KQ {tag} 청산 ibs70 수급 게이트 z̄(T−1) > {th} Fc", "KQ", hold(e0 & (zlag > th), exit_cond(f, "ibs70"), 10), "F", "c", dict(g="X5", tag=tag, th=th), note=N2)
    vol = f["vol"].to_numpy()                                   # X6 수급 변동성 목표
    wp = flow_pos(zscore(S["KQ"]["prog"].astype(float), 10).reindex(idx).to_numpy(), 0.5, True)
    wz = flow_pos(zz, 0.5, True)
    for tag, w0 in (("prog10", wp), ("zbar", wz)):
        for tgt in (0.10, 0.15, 0.20):
            c.add("X", f"X6_KQ_{tag}_vt{int(tgt * 100)}", f"KQ 수급 {tag} θ0.5 롱숏 변동성 목표 {int(tgt * 100)}% 상한 2배 Fo", "KQ", vt(w0, vol, tgt, 2.0), "F", "o", dict(g="X6", tag=tag, tgt=tgt), note=N2)
    for m in ("KP", "KQ"):                                      # X7 단기 가격 모멘텀(수급 포섭 대조군)
        f = c.Fv[m]["F"]
        for sig in ("ma5", "ma10", "mom5", "mom10"):
            for short in (False, True):
                c.add("X", f"X7_{m}_{sig}_{'LS' if short else 'LF'}", f"{m} 단기 추세 {sig} {'롱숏' if short else '롱현금'} Fc", m, trend_w(f, sig, short), "F", "c", dict(g="X7", m=m, sig=sig, short=short), note=N2)
    for src, dst in (("KQ", "KP"), ("KP", "KQ")):               # X8 수급 교차 시장
        z = zscore(S[src]["prog"].astype(float), 10).reindex(c.V[dst]["F"].idx).to_numpy()
        for short in (False, True):
            c.add("X", f"X8_{src}flow_{dst}_{'LS' if short else 'LF'}", f"{src} 프로그램 10일 z 0.5 → {dst} 선물 {'롱숏' if short else '롱현금'} Fo", dst, flow_pos(z, 0.5, short), "F", "o", dict(g="X8", src=src, short=short), note=N2)


def selftest(c):
    V = c.V["KP"]["F"]
    w = np.ones(len(V.idx))
    r = sim(w, V, "c")
    cc = (1 + np.nan_to_num(V.on)) * (1 + np.nan_to_num(V.oc))
    assert abs(r["g"][1:] + 1 - cc[1:]).max() < 1e-12
    assert abs(np.log1p(r["nt"]).sum() - np.log1p(r["r"]).sum()) < 1e-3    # 비용 합 vs 곱 2차항 차이만 허용
    ww = c.Fv["KP"]["E"]["C"].notna().to_numpy().astype(float)
    Ve = c.V["KP"]["E"]
    a = sim(ww, Ve, "o")
    assert abs(np.log1p(a["nt"]).sum() - np.log1p(a["r"]).sum()) < 1e-3
    wt = np.zeros(20)
    wt[5:8] = 1
    T = Veh(pd.date_range("2020-01-01", periods=20), np.full(20, 0.01), np.full(20, 0.02), np.zeros(20, bool), np.zeros(20), np.zeros(20), "t")
    assert abs(sim(wt, T, "c")["nt"][0] - (1.01 * 1.02) ** 3 + 1) < 1e-12              # 종가 진입: 3일 전체
    assert abs(sim(wt, T, "o")["nt"][0] - 1.02 ** 3 * 1.01 ** 3 + 1) < 1e-12            # 시가 체결: 진입일 낮 ~ 청산일 밤
    assert abs(sim(wt, T, "co")["nt"][0] - (1.01 * 1.02) ** 3 * 1.01 + 1) < 1e-12       # 종가 진입 + 다음 날 시가 청산: 밤 1회 추가
    print("자체 점검 통과")


def neighbor_ok(df):
    """인접 문턱·창(같은 나머지 사양)의 순수익 부호가 모두 양수인지. 인접 셀이 없으면 NaN."""
    ord_ = {"k": {"ibs": [0.1, 0.2, 0.3], "rsi": [5, 10, 20], "dn": [2, 3, 4], "d5": [1.5, 2.0], "d10": [1.5, 2.0], "dd20": [1.5, 2.0]},
            "N": [5, 10, 20, 60], "th": [0.0, 0.5, 1.0]}
    trs = {"ma": ["ma20", "ma50", "ma100", "ma200"], "x": ["x5_20", "x20_60", "x50_200"], "mom": ["mom20", "mom60", "mom120"], "don": ["don20", "don55", "don120"]}
    key = {}
    for i, p in zip(df.index, df["params"]):
        key[(df.at[i, "fam"], tuple(sorted(p.items())))] = i
    out = {}
    for i, p in zip(df.index, df["params"]):
        fam = df.at[i, "fam"]
        alts = []
        if fam == "MR":
            a = ord_["k"][p["t"]]
            j = a.index(p["k"])
            alts = [dict(p, k=a[x]) for x in (j - 1, j + 1) if 0 <= x < len(a)]
        elif fam == "TR":
            for lst in trs.values():
                if p["sig"] in lst:
                    j = lst.index(p["sig"])
                    alts = [dict(p, sig=lst[x]) for x in (j - 1, j + 1) if 0 <= x < len(lst)]
        elif fam == "FL":
            a, j = ord_["N"], ord_["N"].index(p["N"])
            alts = [dict(p, N=a[x]) for x in (j - 1, j + 1) if 0 <= x < len(a)]
            a, j = ord_["th"], ord_["th"].index(p["th"])
            alts += [dict(p, th=a[x]) for x in (j - 1, j + 1) if 0 <= x < len(a)]
        vals = [df.at[key[(fam, tuple(sorted(q.items())))], "net_bp"] for q in alts if (fam, tuple(sorted(q.items()))) in key]
        out[i] = (min(vals) > 0) if vals else np.nan
    return pd.Series(out)


def bh(p, q=0.05):
    p = np.asarray(p, float)
    o = np.argsort(p)
    m = len(p)
    th = (np.arange(1, m + 1) / m) * q
    ok = p[o] <= th
    k = np.flatnonzero(ok).max() + 1 if ok.any() else 0
    r = np.zeros(m, bool)
    r[o[:k]] = True
    return r


def main():
    c = Ctx()
    selftest(c)
    for fn in (fam_mr, fam_mrs, fam_trend, fam_cross, fam_flow, fam_us, fam_vk, fam_x):
        k0 = len(c.rows)
        fn(c)
        print(fn.__name__, "셀", len(c.rows) - k0)
    df = pd.DataFrame(c.rows)
    df["neighbor_ok"] = neighbor_ok(df)
    from scipy.stats import norm
    t = df["t_nw"].fillna(0).to_numpy()
    p = 1 - norm.cdf(t)
    df["bh5"] = bh(p, 0.05)
    df["pass_core"] = (df["net_bp"] > 0) & (df["t_nw"] >= 2) & (df["ypos_share"] >= 0.7) & (df["top5_share_pct"] < 50) & (df["half1_net_bp"] > 0) & (df["half2_net_bp"] > 0) & (df["stress_net_bp"] > 0)
    df["pass_all"] = df["pass_core"] & (df["neighbor_ok"].astype(object).where(df["neighbor_ok"].notna(), True).astype(bool))
    rho = [x for x in df.columns if x.startswith("rho_")]
    df["corr_base"] = df[rho].apply(lambda r: " ".join(f"{k[4:]} {v:+.2f}".replace("-", "−") for k, v in r.items() if pd.notna(v)), axis=1)
    df[COLS].to_csv(OUT / "cells.csv", index=False, encoding="utf-8-sig")
    df.drop(columns=["params"]).assign(params=df["params"].astype(str)).to_csv(OUT / "cells_ext.csv", index=False, encoding="utf-8-sig")
    print("전체 셀", len(df), "pass_core", int(df["pass_core"].sum()), "pass_all", int(df["pass_all"].sum()), "BH5", int(df["bh5"].sum()))
    print(df.groupby(["fam"]).agg(n=("id", "size"), pos=("net_bp", lambda x: (x > 0).sum()), t2=("t_nw", lambda x: (x >= 2).sum()), core=("pass_core", "sum"), all=("pass_all", "sum"), bh=("bh5", "sum")))
    return df


if __name__ == "__main__":
    main()
