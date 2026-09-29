"""R3 · D2 디벨롭 탐색 (A 구간 2010-01-01~2019-12-31 전용)

목적: 2차 D2(코스피200 지수 시가 갭을 S&P500 누적 수익률로 설명하고 남은 잔차 z ≥ 1 → KODEX 200 시가 매수·종가 매도)를
  (a) 실행 가능한 신호(ETF 자체 시가 갭, 신호 잡음), (b) 메커니즘 분해(지수 몫 vs ETF 단일가 몫), (c) 설명 변수(ES 선물·나스닥·전일 장중),
  (d) 청산(당일 종가 vs 익일 시가), (e) 크기 조절(z 비례 vs 이진) 축으로 넓혀 보고, 확인 구간 H0(2003~2009)에 걸 셀을 고른다.

A 전용 장치:
- 가격·지수·미국 자료는 읽자마자 2019-12-31 에서 자른다(B·E4 가격이 메모리에 올라오지 않는다). 자른 뒤 최대 날짜를 assert 한다.
- 신호 입력(갭·미국 수익률)은 번인을 위해 2007-06 부터 쓰지만, 수익률 계열은 결정일 T 가 A 안인 행만 남긴다(mask).
  익일 시가 청산(r_on, r_oo)은 T+1 시가가 A 안에 있어야 하므로 A 마지막 날(2019-12-30)은 NaN 이다.
- 원자료 캐시는 읽기만 한다(data.price 를 부르지 않아 새로 받거나 덮어쓰지 않는다). 기존 시행 장부(ledger.csv)에는 쓰지 않는다.

산출(<OUT>/round3_D2/):
- explore_variants.csv : A 에서 신호→수익률을 계산한 모든 규칙(중복 키는 1회만 셈)과 연속형 분석·잡음 설정 목록
- explore_results.json : 탐색 메모(explore.md)·사양(spec.json)에 옮긴 수치

실행: python tools/preopen_d2_explore.py            (전체, 잡음 모의 포함 수 분)
      python tools/preopen_d2_explore.py --no-noise (잡음 모의 생략)
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common as C  # noqa: E402
import data  # noqa: E402
import engine as E  # noqa: E402

A0, A1 = "2010-01-01", E.A_END          # A = 결정일 2010-01-01~2019-12-31
SIG0 = "2007-06-01"                       # 신호 입력 시작(번인 전용, 수익률 계산 안 함)
OUTD = data.OUT / "round3_D2"
SEED = 20260929


# ------------------------------------------------------------------ 자료(A 까지만)
def rd(p):
    return pd.read_csv(p, parse_dates=["date"], index_col="date", encoding="utf-8-sig").sort_index()


def cut(df, start=None):
    df = df.loc[start:A1] if start else df.loc[:A1]
    assert df.index.max() <= pd.Timestamp(A1), "A 밖 자료가 남았다"
    return df


def load():
    px = cut(rd(data.PX_DIR / "069500.csv"), SIG0)
    cal = px.index
    fdr = cut(rd(data.RAW / "k200_index_daily.csv")[["open", "close"]])
    fch = cut(rd(data.PX_DIR / "KPI200.csv")[["open", "close"]])
    ix = pd.concat([fdr.loc[:"2014-07-07"], fch.loc["2014-07-08":].combine_first(fdr.loc["2014-07-08":])]).sort_index()
    ix = ix[(ix["open"] > 0) & (ix["close"] > 0)].reindex(cal)
    other = {k: cut(rd(data.PX_DIR / f"{k}.csv")).reindex(cal) for k in ("114800", "102110", "FUT")}
    us = {k: cut(data.us_daily(k)) for k in ("spx", "ndx_comp", "es", "nq")}
    return px, ix, other, us


# ------------------------------------------------------------------ 미국 변수
def us_cum(close, cal):
    """D(T) = 미국 날짜 < T 인 마지막 세션, us(T) = ln 종가(D(T)) − ln 종가(D(prev T)) (같은 세션이면 0)."""
    s = close.dropna()
    pos = np.searchsorted(s.index.values, cal.values, side="left") - 1
    v = np.log(s.to_numpy(float))
    last = pd.Series(np.where(pos >= 0, v[np.clip(pos, 0, None)], np.nan), index=cal)
    return last - last.shift(1)


def expiry_sessions(dates):
    """분기 만기 세션 F(3·6·9·12월 셋째 금요일 또는 그 직전 세션)와 다음 세션 S."""
    dates = pd.DatetimeIndex(dates)
    F, S = [], []
    for y in range(dates.min().year, dates.max().year + 1):
        for m in (3, 6, 9, 12):
            d = pd.Timestamp(y, m, 1)
            fri3 = d + pd.Timedelta(days=(4 - d.dayofweek) % 7 + 14)
            k = dates.searchsorted(fri3, side="right") - 1
            if 0 <= k < len(dates) - 1 and (fri3 - dates[k]).days <= 3:
                F.append(dates[k])
                S.append(dates[k + 1])
    return pd.DatetimeIndex(F), pd.DatetimeIndex(S)


def fut_clean(fd, spot):
    """yfinance 연결 선물(ES=F·NQ=F): 만기 금요일 종가가 SOQ 정산가이고 다음 세션에 월물이 바뀌어 기저가 뛴다.
    F·S 두 세션의 선물 로그수익을 현물 로그수익으로 바꿔 이은 수준 계열을 돌려준다(그 두 세션의 장후 정보는 버림)."""
    c = fd["close"].dropna()
    lr = np.log(c).diff()
    F, S = expiry_sessions(c.index)
    sr = np.log(spot.dropna()).diff().reindex(c.index)
    bad = c.index.isin(F) | c.index.isin(S)
    lr[bad] = sr[bad].fillna(0.0)
    return np.exp(lr.fillna(0.0).cumsum())


# ------------------------------------------------------------------ 롤링 잔차 z
def rolling_z1(g, x, win=250, minr=None):
    """단일 회귀변수 닫힌 해: T 제외 직전 win 행 중 g·x 유효 행 n ≥ minr 이면 OLS, s = sqrt(SSR/(n−2)), z = 잔차(T)/s."""
    minr = int(round(win * 0.8)) if minr is None else minr
    G, X = g.to_numpy(float), x.to_numpy(float)
    ok = ~np.isnan(G) & ~np.isnan(X)
    g0, x0, o = np.where(ok, G, 0.0), np.where(ok, X, 0.0), ok.astype(float)
    cs = [np.concatenate([[0.0], np.cumsum(v)]) for v in (o, x0, g0, x0 * x0, x0 * g0, g0 * g0)]
    i = np.arange(len(G))
    lo = np.maximum(0, i - win)
    n, sx, sy, sxx, sxy, syy = [c[i] - c[lo] for c in cs]
    with np.errstate(invalid="ignore", divide="ignore"):
        mx, my = sx / n, sy / n
        vxx, vxy, vyy = sxx - n * mx * mx, sxy - n * mx * my, syy - n * my * my
        b = vxy / vxx
        a = my - b * mx
        s = np.sqrt((vyy - b * vxy) / (n - 2))
        z = (G - a - b * X) / s
    z[~ok | (n < minr)] = np.nan
    return pd.Series(z, index=g.index)


def rolling_zk(g, X, win=250, minr=None):
    """다변수(r2_d2.rolling_z 와 같은 규약): s = sqrt(SSR/(n−p))."""
    minr = int(round(win * 0.8)) if minr is None else minr
    G = g.to_numpy(float)
    Xv = np.asarray(X, float).reshape(len(G), -1)
    ok = ~np.isnan(G) & ~np.isnan(Xv).any(axis=1)
    z = np.full(len(G), np.nan)
    for i in range(len(G)):
        if not ok[i]:
            continue
        lo = max(0, i - win)
        m = ok[lo:i]
        n = int(m.sum())
        if n < minr:
            continue
        A = np.column_stack([np.ones(n), Xv[lo:i][m]])
        y = G[lo:i][m]
        b, *_ = np.linalg.lstsq(A, y, rcond=None)
        e = y - A @ b
        z[i] = (G[i] - np.r_[1.0, Xv[i]] @ b) / np.sqrt(e @ e / (n - A.shape[1]))
    return pd.Series(z, index=g.index)


def zpast(x, win=250, minr=200):
    """T 제외 과거 win 행 평균·SD 로 표준화."""
    return (x - x.shift(1).rolling(win, min_periods=minr).mean()) / x.shift(1).rolling(win, min_periods=minr).std()


# ------------------------------------------------------------------ 평가·기록
LOG = {}


def _log(key, family, row):
    row = dict(key=key, family=family, **row)
    LOG.setdefault(key, row)          # 같은 키(같은 규칙·목표)는 한 번만 센다
    return row


def ev(key, family, sig, ret, cost=5.0):
    """이진 신호(1/0, NaN 제외) → 신호일 평균·Δ(신호−비신호, NW lag5 HAC t)·하위 구간·꼬리 민감도."""
    d = pd.DataFrame({"s": sig, "r": ret}).dropna()
    s, r = d["s"].astype(float), d["r"]
    h = r[s == 1]
    out = dict(days=len(d), n=int(s.sum()))
    if len(h) >= 5 and (s == 0).sum() >= 5:
        dl, t, _ = E.delta_test(s, r)
        hs = h.sort_values()
        yrs = h.groupby(h.index.year).mean()
        out.update(gross=h.mean() * 1e4, t_gross=E.nw_t(h), delta=dl * 1e4, t_delta=t, med=h.median() * 1e4, hit=(h > 0).mean(),
                   net5=(h.mean() - cost / 1e4) * 1e4, net7=(h.mean() - 7e-4) * 1e4,
                   A1=h.loc[:"2014"].mean() * 1e4, A2=h.loc["2015":].mean() * 1e4, n_A1=int((h.index.year <= 2014).sum()),
                   yrs_pos5=int(((yrs - cost / 1e4) > 0).sum()), n_yrs=len(yrs),
                   drop5=hs.iloc[:-5].mean() * 1e4, drop10=hs.iloc[:-10].mean() * 1e4)
    return _log(key, family, out)


def sized(key, family, w, ret, cost=5.0):
    """w ∈ [0,1] 포지션, 일 손익 = w·r − w·비용. 전체 유효일 기준 샤프·NW t."""
    d = pd.DataFrame({"w": w, "r": ret}).dropna()
    p = d["w"] * d["r"] - d["w"] * cost / 1e4
    out = dict(days=len(d), n=int((d["w"] > 0).sum()), expo=d["w"].mean(), pnl_day=p.mean() * 1e4, t_pnl=E.nw_t(p),
               sharpe=p.mean() / p.std() * np.sqrt(252), gross_unit=(d["w"] * d["r"]).sum() / d["w"].sum() * 1e4,
               net_unit=p.sum() / d["w"].sum() * 1e4, A1_pnl=p.loc[:"2014"].mean() * 1e4, A2_pnl=p.loc["2015":].mean() * 1e4)
    return _log(key, family, out)


def cont(key, family, y, X):
    res, n = C.hac_reg(y, X)
    out = dict(days=n, **{f"b_{i}": r.coef for i, r in res.iterrows()}, **{f"t_{i}": r.t for i, r in res.iterrows()})
    return _log(key, family, out)


def B(z, k, up=True):
    s = (z >= k) if up else (z <= -k)
    return s.astype(float).where(z.notna())


def both(a, b):
    return (a * b).where(a.notna() & b.notna())


def lin(z, z0, z1):
    return ((z - z0) / (z1 - z0)).clip(0, 1).where(z.notna())


def rnd(o, nd=2):
    return {k: (round(float(v), nd) if isinstance(v, (float, np.floating)) else v) for k, v in o.items()}


# ------------------------------------------------------------------ 본체
def main(noise=True, draws=100):
    px, ix, other, usd = load()
    cal = px.index
    inA = (cal >= pd.Timestamp(A0)) & (cal <= pd.Timestamp(A1))
    inv, tg, fut = other["114800"], other["102110"], other["FUT"]

    # 갭·괴리(신호 입력)
    ge = np.log(px["open"] / px["close"].shift(1))                # KODEX 200 자체 시가 갭
    gi = np.log(ix["open"] / ix["close"].shift(1))                # 코스피200 지수 시가 갭
    dev = ge - gi                                                 # ETF 갭 − 지수 갭
    prem_c = np.log(px["close"] / ix["close"])                    # 종가 괴리 수준(수정주가라 12월 배당락 때마다 오른다)
    prem_o = np.log(px["open"] / ix["open"])
    mis = prem_o - prem_c.shift(1).rolling(5).median()            # 시가 괴리 − 직전 5개 종가 괴리 중앙값(개장 전에 알 수 있음)
    mis10 = prem_o - prem_c.shift(1).rolling(10).median()
    gt = np.log(tg["open"] / tg["close"].shift(1))                # TIGER 200 갭(교차 ETF 점검)
    basis_jump = np.log(fut["close"] / ix["close"]).diff().abs() > 0.004     # FUT 연결 월물 교체일
    gf = np.log(fut["open"] / fut["close"].shift(1)).where(~basis_jump)

    # 수익률(결정일 T 가 A 안인 행만)
    def mask(s):
        return s.where(inA)

    r_oc = mask(px["close"] / px["open"] - 1)
    r_on = mask(px["open"].shift(-1) / px["close"] - 1)
    r_oo = mask(px["open"].shift(-1) / px["open"] - 1)
    ri_oc = mask(ix["close"] / ix["open"] - 1)
    rinv_oc = mask(inv["close"] / inv["open"] - 1)
    rt_oc = mask(tg["close"] / tg["open"] - 1)
    rf_oc = mask(fut["close"] / fut["open"] - 1)
    assert r_oc.dropna().index.max() <= pd.Timestamp(A1) and r_oc.dropna().index.min() >= pd.Timestamp(A0)

    # 미국·설명 변수
    es_c = fut_clean(usd["es"], usd["spx"]["close"])
    nq_c = fut_clean(usd["nq"], usd["ndx_comp"]["close"])
    U = {"spx": us_cum(usd["spx"]["close"], cal), "ixic": us_cum(usd["ndx_comp"]["close"], cal),
         "es": us_cum(es_c, cal), "nq": us_cum(nq_c, cal), "es_raw": us_cum(usd["es"]["close"], cal),
         "krprev_i": np.log(ix["close"] / ix["open"]).shift(1), "krprev_e": np.log(px["close"] / px["open"]).shift(1)}
    U["es_post"] = U["es"] - U["spx"]
    XS = {"spx": ["spx"], "es": ["es"], "spx+ixic": ["spx", "ixic"], "es+nq": ["es", "nq"], "spx+krprev": ["spx", "krprev"]}

    def Xmat(names, gname):
        cols = [U[("krprev_e" if gname == "etf" else "krprev_i") if n == "krprev" else n] for n in names]
        return pd.concat(cols, axis=1).to_numpy()

    def zmake(g, gname, xname, win):
        names = XS[xname]
        return rolling_z1(g, U[names[0]], win) if len(names) == 1 else rolling_zk(g, Xmat(names, gname), win)

    R = {}
    # 검산: 닫힌 해 = lstsq
    zi = rolling_z1(gi, U["spx"], 250)
    chk = (zi - rolling_zk(gi, U["spx"].to_numpy(), 250)).abs().max()
    assert chk < 1e-8, chk
    ze = rolling_z1(ge, U["spx"], 250)

    # S0 2차 재현(A)
    R["S0_repro"] = {"C1": rnd(ev("idx|spx|W250|k1.00|>=|069500|oc", "S0_repro", B(zi, 1), r_oc)),
                     "C2": rnd(ev("idx|spx|W250|k1.00|<=|114800|oc", "S0_repro", B(zi, 1, False), rinv_oc))}

    # S1 격자: 갭 원천 × 설명 변수 × 창 × 임계 → 069500 시가→종가
    Z = {}
    grid = []
    for gname, g in (("etf", ge), ("idx", gi)):
        for xname in XS:
            for win in (120, 250, 500):
                z = zmake(g, gname, xname, win)
                Z[(gname, xname, win)] = z
                for k in (0.75, 1.0, 1.25, 1.5):
                    o = ev(f"{gname}|{xname}|W{win}|k{k:.2f}|>=|069500|oc", "S1_grid", B(z, k), r_oc)
                    grid.append(dict(g=gname, x=xname, win=win, k=k, **rnd(o)))
    G = pd.DataFrame(grid)
    R["S1_grid"] = grid
    R["S1_family_median"] = G.groupby(["g", "x"])[["gross", "t_delta", "drop10", "A2"]].median().round(2).reset_index().to_dict("records")
    R["S1_window_median"] = G.groupby(["g", "win"])[["gross", "t_delta", "drop10"]].median().round(2).reset_index().to_dict("records")
    # 설계 점검(수익률 미사용): 갭 회귀 설명력
    r2 = []
    for gname, g in (("idx", gi), ("etf", ge)):
        for names in (["spx"], ["es_raw"], ["es"], ["spx", "ixic"], ["es", "nq"], ["spx", "es_post"], ["spx", "krprev"]):
            cols = [U[("krprev_e" if gname == "etf" else "krprev_i") if n == "krprev" else n].rename(n) for n in names]
            d = pd.concat([g.rename("g"), *cols], axis=1).loc[A0:A1].dropna()
            res, n = C.hac_reg(d["g"], d[names])
            Xm = np.column_stack([np.ones(len(d)), d[names].to_numpy()])
            b, *_ = np.linalg.lstsq(Xm, d["g"].to_numpy(), rcond=None)
            rr = 1 - ((d["g"].to_numpy() - Xm @ b) ** 2).sum() / ((d["g"] - d["g"].mean()) ** 2).sum()
            r2.append(dict(g=gname, x="+".join(names), R2=round(rr, 3), **{f"b_{c}": round(res.loc[c, "coef"], 3) for c in names},
                           **{f"t_{c}": round(res.loc[c, "t"], 1) for c in names}))
    R["S1_gap_R2"] = r2

    # S2 신호 잡음(ETF 예상체결가·예상지수 오차 모의): 갭에 오차를 더해 z 를 다시 만들고 체결은 실제 시가
    if noise:
        rng = np.random.default_rng(SEED)
        rows = []
        for gname, g in (("etf", ge), ("idx", gi)):
            for kind, sc in (("N", 5), ("N", 10), ("U", 5), ("U", 10), ("N", 20)):
                a = []
                for _ in range(draws):
                    eps = (rng.normal(0, sc, len(g)) if kind == "N" else rng.uniform(-sc, sc, len(g))) / 1e4
                    z = rolling_z1(g + eps, U["spx"], 250)
                    d = pd.DataFrame({"s": B(z, 1), "r": r_oc}).dropna()
                    h = d["r"][d["s"] == 1]
                    dl, t, _ = E.delta_test(d["s"], d["r"])
                    a.append((len(h), h.mean() * 1e4, dl * 1e4, t, h.sort_values().iloc[:-10].mean() * 1e4))
                a = np.array(a)
                row = dict(g=gname, noise=f"{kind}{sc}bp", draws=draws, n=a[:, 0].mean(), gross=a[:, 1].mean(),
                           gross_p5=np.percentile(a[:, 1], 5), gross_p95=np.percentile(a[:, 1], 95), delta=a[:, 2].mean(),
                           t=a[:, 3].mean(), t_p5=np.percentile(a[:, 3], 5), drop10=a[:, 4].mean())
                rows.append(rnd(row))
                _log(f"noise|{gname}|{kind}{sc}|W250|k1.00|069500|oc", "S2_noise", rnd(row))
        R["S2_noise"] = rows

    # S3 메커니즘(연속형, HAC)
    zd = zpast(dev)
    zm = zpast(mis)
    D = pd.DataFrame({"zi": zi, "ze": ze, "zd": zd, "zm": zm, "r_oc": r_oc * 1e4, "ri_oc": ri_oc * 1e4, "spread": (r_oc - ri_oc) * 1e4,
                      "rf_oc": rf_oc * 1e4, "r_on": r_on * 1e4, "r_oo": r_oo * 1e4})
    DA = D.loc[A0:A1]
    R["S3_corr"] = DA[["zi", "ze", "zd", "zm"]].corr().round(3).to_dict()
    mech = []
    for y in ("r_oc", "ri_oc", "spread"):
        for xs in (["zi"], ["ze"], ["zd"], ["zi", "zd"], ["ze", "zd"], ["zi", "ze"]):
            mech.append(dict(y=y, x="+".join(xs), **rnd(cont(f"reg|{y}~{'+'.join(xs)}|A", "S3_mech", DA[y], DA[xs]))))
    FS = D.loc["2014-07-09":A1]
    for y in ("rf_oc", "ri_oc", "r_oc"):
        for xs in (["zi"], ["ze"], ["zi", "zd"]):
            mech.append(dict(y=y, x="+".join(xs), span="2014-07-09~", **rnd(cont(f"reg|{y}~{'+'.join(xs)}|2014-07~", "S3_mech", FS[y], FS[xs]))))
    for y in ("r_on", "r_oo"):
        for xs in (["zi"], ["ze"], ["zi", "zd"]):
            mech.append(dict(y=y, x="+".join(xs), **rnd(cont(f"reg|{y}~{'+'.join(xs)}|A", "S3_mech", DA[y], DA[xs]))))
    R["S3_mech"] = mech
    dec = {}
    for zc in ("zi", "ze"):
        q = DA.dropna(subset=[zc]).copy()
        q["dec"] = pd.qcut(q[zc], 10, labels=False)
        q["dev_bp"] = (dev * 1e4).reindex(q.index)
        dec[zc] = q.groupby("dec")[["zi", "ze", "dev_bp", "r_oc", "ri_oc", "spread"]].mean().round(2).reset_index().to_dict("records")
        _log(f"decile|{zc}|A", "S3_mech", dict(days=len(q)))
    R["S3_deciles"] = dec
    R["S3_bin"] = {"zi->index_oc": rnd(ev("idx|spx|W250|k1.00|>=|KPI200|oc", "S3_mech", B(zi, 1), ri_oc)),
                   "ze->index_oc": rnd(ev("etf|spx|W250|k1.00|>=|KPI200|oc", "S3_mech", B(ze, 1), ri_oc))}
    # 설계 점검(수익률 미사용): 갭 기울기(지수 갭 vs 선물·ETF 갭, 2014-07~, 월물 교체일 제외)
    gg = pd.DataFrame({"ge": ge, "gi": gi, "gf": gf}).loc["2014-07-09":A1].dropna()
    R["S3_gap_slopes"] = {lab: dict(n=len(x), gi_on_gf=round(float(np.polyfit(x["gf"], x["gi"], 1)[0]), 3),
                                    ge_on_gf=round(float(np.polyfit(x["gf"], x["ge"], 1)[0]), 3))
                          for lab, x in (("2014-09~2016-07", gg.loc["2014-09":"2016-07"]), ("2016-08~2019-12", gg.loc["2016-08":]))}
    gy = pd.DataFrame({"ge": ge, "gi": gi}).dropna()
    R["S3_gap_year"] = [dict(year=y, sd_ge=round(q["ge"].std() * 1e4, 1), sd_gi=round(q["gi"].std() * 1e4, 1),
                             sd_dev=round((q["ge"] - q["gi"]).std() * 1e4, 1),
                             beta_gi_on_ge=round(float(np.polyfit(q["ge"], q["gi"], 1)[0]), 3))
                        for y, q in ((y, gy.loc[str(y)]) for y in range(2010, 2020))]

    # S4 청산: 당일 종가(oc, 격자에 있음) vs 익일 시가(oo = oc + on), on 단독
    cands = {"C2형 idx|spx|W250": zi, "C1형 etf|spx|W250": ze, "etf|es+nq|W250": Z[("etf", "es+nq", 250)],
             "etf|spx+ixic|W250": Z[("etf", "spx+ixic", 250)]}
    ckey = {"C2형 idx|spx|W250": "idx|spx|W250", "C1형 etf|spx|W250": "etf|spx|W250", "etf|es+nq|W250": "etf|es+nq|W250",
            "etf|spx+ixic|W250": "etf|spx+ixic|W250"}
    ex = []
    for nm, z in cands.items():
        for tgt, ret in (("oc", r_oc), ("oo", r_oo), ("on", r_on)):
            ex.append(dict(cand=nm, target=tgt, **rnd(ev(f"{ckey[nm]}|k1.00|>=|069500|{tgt}", "S4_exit", B(z, 1), ret))))
    R["S4_exit"] = ex
    R["S4_uncond_on_bp"] = round(float(r_on.loc[A0:A1].mean() * 1e4), 2)

    # S5 크기 조절: 이진 vs z 비례(상한 1)
    sz = []
    for nm, z in cands.items():
        for sname, w in (("이진 k1", B(z, 1)), ("이진 k0.75", B(z, 0.75)), ("선형 0→2", lin(z, 0, 2)), ("선형 0.5→2", lin(z, 0.5, 2)),
                         ("선형 0.5→1.5", lin(z, 0.5, 1.5)), ("선형 1→2", lin(z, 1, 2))):
            fam = "S5_size" if sname.startswith("선형") else "S5_size_bin"
            sz.append(dict(cand=nm, size=sname, **rnd(sized(f"{ckey[nm]}|{sname}|069500|oc|sized", fam, w, r_oc), 3)))
    R["S5_size"] = sz

    # S6 교차 ETF·선물 갭·괴리 조합
    zt = rolling_z1(gt, U["spx"], 250)
    za = rolling_z1((ge + gt) / 2, U["spx"], 250)
    zf = rolling_z1(gf, U["spx"], 250)
    s6 = {}
    for key, sig, ret in (("etf|spx|W250|k1.00|>=|102110|oc", B(ze, 1), rt_oc), ("tiger|spx|W250|k1.00|>=|069500|oc", B(zt, 1), r_oc),
                          ("tiger|spx|W250|k1.00|>=|102110|oc", B(zt, 1), rt_oc), ("etfavg|spx|W250|k1.00|>=|069500|oc", B(za, 1), r_oc),
                          ("etfavg|spx|W250|k1.00|>=|102110|oc", B(za, 1), rt_oc), ("idx|spx|W250|k1.00|>=|102110|oc", B(zi, 1), rt_oc)):
        s6[key] = rnd(ev(key, "S6_cross_etf", sig, ret))
    zf0 = zf.first_valid_index()
    for key, sig, ret in (("fut|spx|W250|k1.00|>=|069500|oc", B(zf, 1), r_oc), ("fut|spx|W250|k1.00|>=|FUT|oc", B(zf, 1), rf_oc),
                          ("idx|spx|W250|k1.00|>=|FUT|oc|2015-06~", B(zi, 1).loc[zf0:], rf_oc),
                          ("idx|spx|W250|k1.00|>=|069500|oc|2015-06~", B(zi, 1).loc[zf0:], r_oc),
                          ("etf|spx|W250|k1.00|>=|069500|oc|2015-06~", B(ze, 1).loc[zf0:], r_oc)):
        s6[key] = rnd(ev(key, "S6_futures", sig, ret))
    s6["_zf_first_valid"] = str(zf0.date())
    for k in (0.75, 1.0, 1.5):
        key = f"dev_z|W250|k{k:.2f}|<=|069500|oc"
        s6[key] = rnd(ev(key, "S6_dev", B(zd, k, False), r_oc))
    combos = {"mis_z|W250|k1.00|<=|069500|oc": B(zm, 1, False),
              "dev_z<=-1 & zi<1|069500|oc": both(B(zd, 1, False), (zi < 1).astype(float).where(zi.notna())),
              "zi>=1 & dev_z<=0|069500|oc": both(B(zi, 1), (zd <= 0).astype(float).where(zd.notna())),
              "zi>=1 & dev_z>0|069500|oc": both(B(zi, 1), (zd > 0).astype(float).where(zd.notna())),
              "zi>=1 & |dev_z|<=1|069500|oc": both(B(zi, 1), (zd.abs() <= 1).astype(float).where(zd.notna())),
              "(zi-dev_z)/sqrt2>=1.00|069500|oc": B((zi - zd) / np.sqrt(2), 1.0),
              "(zi-dev_z)/sqrt2>=1.25|069500|oc": B((zi - zd) / np.sqrt(2), 1.25),
              "ze>=1 & dev_z<=0|069500|oc": both(B(ze, 1), (zd <= 0).astype(float).where(zd.notna())),
              "ze>=1 & dev_z>0|069500|oc": both(B(ze, 1), (zd > 0).astype(float).where(zd.notna())),
              "zi>=1 & dev<=0|069500|oc": both(B(zi, 1), (dev <= 0).astype(float).where(dev.notna())),
              "zi>=1 & mis5<=0|069500|oc": both(B(zi, 1), (mis <= 0).astype(float).where(mis.notna()))}
    for key, sig in combos.items():
        s6[key] = rnd(ev(key, "S6_combo", sig, r_oc))
    s6["_overlap"] = {"zd<=-1 & zi>=1": int(((zd <= -1) & (zi >= 1)).loc[A0:A1].sum()), "zi>=1": int((zi >= 1).loc[A0:A1].sum()),
                      "zd<=-1": int((zd <= -1).loc[A0:A1].sum())}
    R["S6"] = s6
    R["S6_mis_sd_bp"] = round(float(mis.loc[A0:A1].std() * 1e4), 1)

    # S7 최종 셀과 이웃
    c3 = combos["zi>=1 & mis5<=0|069500|oc"]
    finals = {"R3-C1": ("etf|spx|W250|k1.00|>=|069500|oc", B(ze, 1)),
              "R3-C2": ("idx|spx|W250|k1.00|>=|069500|oc", B(zi, 1)),
              "R3-C3": ("zi>=1 & mis5<=0|069500|oc", c3)}
    zi120, zi500 = Z[("idx", "spx", 120)], Z[("idx", "spx", 500)]
    ze120, ze500 = Z[("etf", "spx", 120)], Z[("etf", "spx", 500)]
    neigh = {"R3-C1": {"창120": ("etf|spx|W120|k1.00|>=|069500|oc", B(ze120, 1)), "창500": ("etf|spx|W500|k1.00|>=|069500|oc", B(ze500, 1)),
                       "임계0.75": ("etf|spx|W250|k0.75|>=|069500|oc", B(ze, 0.75)), "임계1.25": ("etf|spx|W250|k1.25|>=|069500|oc", B(ze, 1.25))},
             "R3-C2": {"창120": ("idx|spx|W120|k1.00|>=|069500|oc", B(zi120, 1)), "창500": ("idx|spx|W500|k1.00|>=|069500|oc", B(zi500, 1)),
                       "임계0.75": ("idx|spx|W250|k0.75|>=|069500|oc", B(zi, 0.75)), "임계1.25": ("idx|spx|W250|k1.25|>=|069500|oc", B(zi, 1.25))},
             "R3-C3": {"임계0.75": ("zi>=0.75 & mis5<=0|069500|oc", both(B(zi, 0.75), (mis <= 0).astype(float).where(mis.notna()))),
                       "임계1.25": ("zi>=1.25 & mis5<=0|069500|oc", both(B(zi, 1.25), (mis <= 0).astype(float).where(mis.notna()))),
                       "괴리 기준 10일": ("zi>=1 & mis10<=0|069500|oc", both(B(zi, 1), (mis10 <= 0).astype(float).where(mis10.notna()))),
                       "갭 차 dev≤0": ("zi>=1 & dev<=0|069500|oc", both(B(zi, 1), (dev <= 0).astype(float).where(dev.notna())))}}
    fin = {}
    for cid, (key, sig) in finals.items():
        o = ev(key, "S1_grid" if cid != "R3-C3" else "S6_combo", sig, r_oc)
        d = pd.DataFrame({"s": sig, "r": r_oc}).dropna()
        h = d["r"][d["s"] == 1]
        vr = C.vol_adj(r_oc.dropna(), 60).reindex(d.index)
        dv = E.delta_test(d["s"][vr.notna()], vr.dropna())[0]
        yrs = {int(y): dict(n=int(len(v)), gross=round(v.mean() * 1e4, 1)) for y, v in h.groupby(h.index.year)}
        f = dict(**rnd(o), vol_adj_delta=round(float(dv), 4), placebo_p=round(float(E.placebo_year_strat(d["s"], d["r"])), 4),
                 yearly=yrs, neighbors={}, extra={})
        for nn, (nk, ns) in neigh[cid].items():
            f["neighbors"][nn] = rnd(ev(nk, "S7_neighbor", ns, r_oc))
        f["extra"]["익일 시가 청산(oo)"] = rnd(ev(key.replace("|oc", "|oo"), "S7_extra", sig, r_oo))
        f["extra"]["102110 이전"] = rnd(ev(key.replace("069500", "102110"), "S7_extra", sig, rt_oc))
        f["extra"]["지수 시가→종가(거래 불가)"] = rnd(ev(key.replace("069500", "KPI200"), "S7_extra", sig, ri_oc))
        fin[cid] = f
    over = pd.DataFrame({k: v[1] for k, v in finals.items()}).loc[A0:A1]
    R["S7_overlap"] = {"C1&C2": int(((over["R3-C1"] == 1) & (over["R3-C2"] == 1)).sum()),
                       "C1&C3": int(((over["R3-C1"] == 1) & (over["R3-C3"] == 1)).sum()),
                       "C2&C3": int(((over["R3-C2"] == 1) & (over["R3-C3"] == 1)).sum()),
                       "C3 NaN days": int(over["R3-C3"].isna().sum())}
    # 최종 셀 C3 신호 잡음: 예상지수·ETF 예상체결가에 각각 독립 오차
    if noise:
        rng = np.random.default_rng(SEED + 3)
        rows = []
        for sc in (5, 10):
            a = []
            for _ in range(draws):
                ei, ee = rng.normal(0, sc, len(gi)) / 1e4, rng.normal(0, sc, len(gi)) / 1e4
                zin = rolling_z1(gi + ei, U["spx"], 250)
                misn = mis + ee - ei
                sig = both(B(zin, 1), (misn <= 0).astype(float).where(misn.notna()))
                d = pd.DataFrame({"s": sig, "r": r_oc}).dropna()
                h = d["r"][d["s"] == 1]
                dl, t, _ = E.delta_test(d["s"], d["r"])
                a.append((len(h), h.mean() * 1e4, dl * 1e4, t, h.sort_values().iloc[:-10].mean() * 1e4))
            a = np.array(a)
            row = rnd(dict(noise=f"N{sc}bp(지수·ETF 각각)", draws=draws, n=a[:, 0].mean(), gross=a[:, 1].mean(),
                           gross_p5=np.percentile(a[:, 1], 5), delta=a[:, 2].mean(), t=a[:, 3].mean(), drop10=a[:, 4].mean()))
            rows.append(row)
            _log(f"noise|C3|N{sc}", "S7_noise", row)
        fin["R3-C3"]["noise"] = rows
    R["S7_finals"] = fin

    # 기록
    OUTD.mkdir(parents=True, exist_ok=True)
    V = pd.DataFrame(list(LOG.values()))
    V.to_csv(OUTD / "explore_variants.csv", index=False, encoding="utf-8-sig")
    fam = V["family"].value_counts().to_dict()
    is_cont = V["key"].str.startswith(("reg|", "decile|"))
    is_noise = V["family"].isin(["S2_noise", "S7_noise"])
    ret_rules = int((~is_cont & ~is_noise & (V["family"] != "S5_size_bin")).sum())   # 2차 셀 재현 2개 포함
    R["count"] = dict(by_family=fam, total_rows=len(V), repro=int(fam.get("S0_repro", 0)),
                      continuous=int(V["key"].str.startswith(("reg|", "decile|")).sum()),
                      noise_configs=int(V["family"].isin(["S2_noise", "S7_noise"]).sum()),
                      rules_with_return=ret_rules, size_bin_dup=int(fam.get("S5_size_bin", 0)))
    R["meta"] = dict(A=[A0, A1], signal_input_start=SIG0, z_valid_A=int(zi.loc[A0:A1].notna().sum()), days_A=int(inA.sum()),
                     ze_valid_A=int(ze.loc[A0:A1].notna().sum()), mis_valid_A=int(mis.loc[A0:A1].notna().sum()), zcheck=float(chk))
    C.save_verdict(OUTD / "explore_results.json", R)
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(json.dumps(R["count"], ensure_ascii=False))
    for cid, f in fin.items():
        print(cid, {k: f[k] for k in ("n", "gross", "t_gross", "delta", "t_delta", "net5", "net7", "A1", "A2", "yrs_pos5", "drop10",
                                      "placebo_p", "vol_adj_delta") if k in f})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-noise", action="store_true")
    ap.add_argument("--draws", type=int, default=100)
    a = ap.parse_args()
    main(noise=not a.no_noise, draws=a.draws)
