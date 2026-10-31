# -*- coding: utf-8 -*-
"""R16 스캔 관점 morning_exec: 13차 아침 C3 다리의 09:00 동일가 문제를 없앤 실행 가능판 찾기 (봉인 2024-12-30)

원형 C3(바꾸지 않음, 비교 기준, exec_ok=False)
  g(T) = ln(지수 시가(T)/지수 종가(T−1)), us(T) = T 이전 미국 세션 SPX 누적 로그수익
  z(T) = 직전 250행(T 제외, 유효 200행 이상) OLS g~us 잔차/s
  mis(T) = ln(ETF 시가/지수 시가) − 직전 5일 ln(ETF 종가/지수 종가) 중앙값
  C3 = z ≥ 1 ∧ mis ≤ 0 → 09:00 단일가 매수, 15:30 종가 매도
  w = C3·clip(0.30/σ20, 0, cap), cap 기본 2, w ≤ 1 → 1배 ETF w, 1 < w ≤ 2 → 1배 (2−w) + 2배 (w−1)
  코스피 = 069500·122630(코스피200), 코스닥 = 229200·233740(코스닥150)

실행 경로(모두 신호 확정 시각 < 체결 시각)
  E1 사후 진입: 09:00 단일가로 확정한 신호 → 09:02 봉 시가(= 09:01:00 첫 체결) 이후 진입. 분봉 창만.
  E2 선매수 확인: 09:00 전 필터 F 로 단일가 매수 → 09:00 뒤 C3 확인 → 통과면 15:30 까지, 실패면 09:02 봉 시가 청산.
  E4 지정가 단일가: 09:00 전에 지정가 L 제출(08:45 선물 시가로 계산) → 단일가 < L 이면 체결. 시가를 보고 판단하지 않고 체결 조건만 시가에 걸린다.
  E5 사전 신호: 미국 마감·전일 한국·전일 수급으로 09:00 단일가 시장가 매수. 일봉 장기창.

비용: ETF 왕복 5bp, 연속 체결이 끼면 1틱(5원/그해 평균 원가격) 더. 스트레스 = 7bp + 1틱(단일가만) / 7bp + 2틱(연속 체결 포함).
  선물 왕복 1.8bp, 스트레스 3bp. 금액은 자본 대비: 비용 = (a + b)·5bp, a·1배 수익 + b·2배 수익.
분봉: 봉 끝 라벨, 09:01 봉 시가 = 09:00 단일가, 15:30 봉 종가 = 마감 단일가. 09:01 봉 거래량 0 인 4일(늦은 개장)은 제외.
일봉은 수정가라 수익만, 분봉은 원가격이라 분봉 수익은 분봉 안에서만 잰다(두 원천 혼합 금지).
FUT 시가는 2023-07-31 이후만 08:45(사전 정보). 그 전 FUT 시가를 쓴 셀은 exec_ok=False 진단.
깃 밖 자료: yf_ewy_daily, fdr_usdkrw_daily, kis_fut_k200_daily, kis_fut_kq150_daily (셀 notes 에 표기).
실행: PYTHONUTF8=1 python r16s_morning_exec.py
"""
import sys
from pathlib import Path

import matplotlib
import matplotlib.dates
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False})
sys.stdout.reconfigure(encoding="utf-8")
pd.set_option("display.width", 250, "display.max_columns", 50, "display.max_rows", 400, "display.max_colwidth", 60)

SEAL = pd.Timestamp("2024-12-30")
GIT = "<repo>"
EXT = "<raw>"
BT = "<OUT>"
ROOT = Path(BT) / "round16_alpha_search/scan"
OUT = ROOT / "morning_exec"
ANN, VT = 252, 0.30
FUT_0845 = pd.Timestamp("2023-07-31")
START = {"KOSPI": "2010-02-22", "KOSDAQ": "2016-05-09"}
MKT = {"KOSPI": ("069500", "122630"), "KOSDAQ": ("229200", "233740")}
INV = {"KOSPI": ("114800", "252670"), "KOSDAQ": ("251340",)}
LEGS = ("069500", "122630", "229200", "233740", "114800", "252670", "251340")
AUC = "09:01"
FST = "09:02"                                                  # 09:00 뒤 확정 신호의 가장 빠른 체결(라벨)
C1, C2, MUTED, GRID, SURF, INK = "#2a78d6", "#eb6834", "#898781", "#e1e0d9", "#fcfcfb", "#0b0b0b"


# ---------------------------------------------------------------- 봉인 읽기
def seal(df, col="date"):
    """봉인: 읽은 직후 호출. 2025-01-02 이후 행은 수익 계산 전에 버린다."""
    out = df[pd.to_datetime(df[col]) <= SEAL]
    assert pd.to_datetime(out[col]).max() <= SEAL
    return out


def daily(path, **kw):
    d = pd.read_csv(path, encoding="utf-8-sig", **kw)
    d["date"] = pd.to_datetime(d["date"])
    return seal(d).drop_duplicates("date").set_index("date").sort_index()


# ---------------------------------------------------------------- 통계·신호 보조
def nw_t(x, lags=5):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 10:
        return np.nan
    e = x - x.mean()
    s = e @ e / n
    for k in range(1, min(lags, n - 1) + 1):
        s += 2 * (1 - k / (lags + 1)) * (e[k:] @ e[:-k]) / n
    return x.mean() / np.sqrt(s / n) if s > 0 else np.nan


def rolling_fit(g, x, win=250, minr=200):
    """3차 D2 그대로: T 제외 직전 win 행 중 유효 n ≥ minr 이면 OLS g = a + b x, s = sqrt(SSR/(n−2))."""
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
    bad = n < minr
    a[bad], b[bad], s[bad] = np.nan, np.nan, np.nan
    return pd.DataFrame({"a": a, "b": b, "s": s}, index=g.index)


def rolling_z1(g, x, win=250, minr=200):
    f = rolling_fit(g, x, win, minr)
    return (g - f["a"] - f["b"] * x) / f["s"]


def last_before(cal, s):
    """각 한국일 T 에 대해 T 이전 마지막 미국 날짜의 ln 값."""
    s = s.dropna().sort_index()
    pos = np.searchsorted(s.index.values, cal.values, side="left") - 1
    v = np.log(s.to_numpy(float))
    return pd.Series(np.where(pos >= 0, v[np.clip(pos, 0, None)], np.nan), index=cal)


def us_cum(cal, s):
    last = last_before(cal, s)
    return last - last.shift(1)


def zroll(x, w=60):
    return (x - x.rolling(w, min_periods=40).mean().shift(1)) / x.rolling(w, min_periods=40).std().shift(1)


def mix(w):
    a = pd.Series(np.where(w <= 1, w, 2 - w), index=w.index)
    b = pd.Series(np.where(w <= 1, 0.0, w - 1), index=w.index)
    return a, b


def wt(sig, s, cap=2.0):
    return (sig.fillna(0) * (VT / s).clip(0, cap).fillna(1.0)).fillna(0.0)


def tick_of(code, cal):
    """1틱(수익 단위) = 호가/그해 평균 원가격. 호가는 2,000원 이상 5원, 미만 1원."""
    r = daily(f"{GIT}/jehyeong/intraday/data/{code}_daily.csv", usecols=["date", "close"])
    m = r["close"].groupby(r.index.year).mean()
    t = pd.Series(np.where(m >= 2000, 5.0, 1.0) / m.to_numpy(), index=m.index)
    return pd.Series(cal.year, index=cal).map(t)


# ---------------------------------------------------------------- 시장 표
def market(mkt):
    c1, c2 = MKT[mkt]
    e = f"{GIT}/beomgu/data/etf"
    p1, p2 = daily(f"{e}/{c1}.csv"), daily(f"{e}/{c2}.csv")
    if mkt == "KOSPI":
        fdr = daily(f"{GIT}/beomgu/data/k200_index_daily.csv")[["open", "close"]]
        fdr = fdr[(fdr["open"] > 0) & (fdr["close"] > 0)]
        kpi = daily(f"{e}/KPI200.csv")
        ix = pd.concat([fdr.loc[:"2014-07-07"], kpi[["open", "close"]].loc["2014-07-08":].combine_first(fdr.loc["2014-07-08":])]).sort_index()
        cal = p1.index
    else:
        q = pd.read_csv(f"{GIT}/jehyeong/intraday/data/KOSDAQ_150_ohlcv.csv", encoding="utf-8-sig").rename(columns=str.lower)
        q["date"] = pd.to_datetime(q["date"])
        q = seal(q).drop_duplicates("date").set_index("date").sort_index()      # 깃 파일 2024-10-08 행 중복
        ix = q[(q["open"] > 0) & (q["close"] > 0) & (q["volume"] > 0) & (q.index >= "2015-07-13")][["open", "close"]]
        cal = ix.index.union(p1.index)
        cal = cal[cal >= "2015-07-13"]
    ix, p1, p2 = ix.reindex(cal), p1.reindex(cal), p2.reindex(cal)
    d = pd.DataFrame(index=cal)
    d["o1"], d["h1"], d["l1"], d["c1"] = p1["open"], p1["high"], p1["low"], p1["close"]
    d["o2"], d["c2"], d["io"], d["ic"] = p2["open"], p2["close"], ix["open"], ix["close"]
    d["r1"], d["r2"], d["ri"] = p1["close"] / p1["open"] - 1, p2["close"] / p2["open"] - 1, ix["close"] / ix["open"] - 1
    d["g"] = np.log(ix["open"] / ix["close"].shift(1))
    d["ge"] = np.log(p1["open"] / p1["close"].shift(1))
    d["us"] = us_cum(cal, daily(f"{GIT}/beomgu/data/spx_daily.csv")["close"])
    f = rolling_fit(d["g"], d["us"])
    d["fa"], d["fb"], d["fs"] = f["a"], f["b"], f["s"]
    d["z"] = (d["g"] - f["a"] - f["b"] * d["us"]) / f["s"]
    d["mis"] = np.log(p1["open"] / ix["open"]) - np.log(p1["close"] / ix["close"]).shift(1).rolling(5).median()
    d["sig"] = (ix["close"] / ix["close"].shift(1) - 1).rolling(20).std().shift(1) * np.sqrt(ANN)
    d["C2"] = (d["z"] >= 1).astype(float).where(d["z"].notna())
    d["C3"] = ((d["z"] >= 1) & (d["mis"] <= 0)).astype(float).where(d["z"].notna() & d["mis"].notna())
    d["t1"], d["t2"] = tick_of(c1, cal), tick_of(c2, cal)
    return d


def minute(code):
    t = pd.read_csv(f"{GIT}/jehyeong/intraday/data/minute/{code}_TOSS_1m.csv", encoding="utf-8-sig", usecols=["timestamp", "date", "open", "high", "low", "close", "volume"])
    t["date"] = pd.to_datetime(t["date"])
    t = seal(t)
    t["hm"] = t["timestamp"].str[11:16]
    assert t.groupby("date")["hm"].first().eq(AUC).all() and t.groupby("date")["hm"].last().eq("15:30").all()   # 봉 끝 라벨 확인
    po, pc = t.pivot(index="date", columns="hm", values="open"), t.pivot(index="date", columns="hm", values="close")
    pv = t.pivot(index="date", columns="hm", values="volume")
    bad = pv[AUC] <= 0                                        # 늦은 개장·시가 미형성: 09:01 봉 시가가 자리표시
    return {"o": po.where(pv > 0).bfill(axis=1).where(~bad), "c": pc.where(~bad), "h": t.pivot(index="date", columns="hm", values="high").where(~bad),
            "l": t.pivot(index="date", columns="hm", values="low").where(~bad), "bad": list(bad[bad].index)}


def fut_tab(path, e_c):
    """근월물(front=1, 만기 당일은 차근월) 시가·전일 종가(같은 월물)·직전 5일 ln(ETF 종가/선물 종가) 중앙값(같은 월물)."""
    f = pd.read_csv(path, dtype={"code": str})
    f["date"] = pd.to_datetime(f["date"])
    f = seal(f)
    f = f[(f["open"] > 0) & (f["close"] > 0)]
    fr = f[f["front"] == 1].drop_duplicates("date").set_index("date").sort_index()
    cl = f.pivot_table(index="date", columns="code", values="close", aggfunc="last").sort_index()
    lr = pd.DataFrame({c: np.log(e_c.reindex(cl.index)) - np.log(cl[c]) for c in cl.columns})
    m5 = lr.shift(1).rolling(5, min_periods=3).median()
    prev = cl.shift(1)
    out = pd.DataFrame({"fo": fr["open"], "fc": fr["close"]})
    out["fp"] = [prev.at[d, c] for d, c in zip(fr.index, fr["code"])]
    out["m5"] = [m5.at[d, c] for d, c in zip(fr.index, fr["code"])]
    return out


def add_futures(D, path):
    F = fut_tab(path, D["c1"]).reindex(D.index)
    D["fo"], D["rf"] = F["fo"], F["fc"] / F["fo"] - 1
    D["gf"] = np.log(F["fo"] / F["fp"])
    D["zf"] = rolling_z1(D["gf"], D["us"])
    D["misf"] = np.log(D["o1"] / F["fo"]) - F["m5"]


def add_ewy(K):
    y = pd.read_csv(f"{EXT}/r15_external/yf_ewy_daily.csv", encoding="utf-8-sig", skiprows=[1, 2]).rename(columns={"Price": "date", "Adj Close": "adj"})
    y["date"] = pd.to_datetime(y["date"])
    ewy = seal(y).set_index("date").sort_index()["adj"].astype(float)
    x = pd.read_csv(f"{EXT}/r15_external/fdr_usdkrw_daily.csv", encoding="utf-8-sig").rename(columns={"Unnamed: 0": "date"})
    x["date"] = pd.to_datetime(x["date"])
    fx = seal(x).set_index("date").sort_index()["Close"].astype(float)
    L = last_before(K.index, ewy) - np.log(K["ic"]).shift(1)
    Lx = L + last_before(K.index, fx)
    K["ewy"] = L - L.shift(1).rolling(20, min_periods=15).median()             # 수준 잔차: 한국 종가 뒤 EWY 가 더 움직인 몫
    K["ewyx"] = Lx - Lx.shift(1).rolling(20, min_periods=15).median()
    K["zewy"], K["zewyx"] = rolling_z1(K["ewy"], K["us"]), rolling_z1(K["ewyx"], K["us"])


def add_pre(D, mk):
    """09:00 전 확정 변수. 전일 수급·전일 시세는 shift(1), 미국은 T 이전 세션."""
    b = f"{GIT}/beomgu/data"
    cal = D.index
    nq, es = us_cum(cal, daily(f"{b}/nq_daily.csv")["close"]), us_cum(cal, daily(f"{b}/es_daily.csv")["close"])
    inv = daily(f"{b}/naver_{'kospi' if mk == 'KOSPI' else 'kosdaq'}_investor_daily.csv").reindex(cal)
    prg = daily(f"{b}/naver_{'program' if mk == 'KOSPI' else 'kosdaq_program'}_daily.csv").reindex(cal)
    if mk == "KOSPI":
        fu = daily(f"{b}/naver_fut_investor_daily.csv").reindex(cal)
        ffrn, find = fu["net_foreign"], fu["net_individual"]
    else:
        kf = pd.read_csv(f"{GIT}/jehyeong/intraday/data/KOSDAQ_150_fut_foreign.csv", encoding="utf-8-sig").rename(columns={"Date": "date"})
        kf["date"] = pd.to_datetime(kf["date"])
        kf = seal(kf).drop_duplicates("date").set_index("date").sort_index().reindex(cal)
        ffrn, find = kf["foreign"], kf["individual"]
    ic = np.log(D["ic"])
    X = {"spx": D["us"], "nq_spx": nq - D["us"], "es_after": es - D["us"],
         "prev_cc": (ic - ic.shift(1)).shift(1), "prev_oc": np.log(D["c1"] / D["o1"]).shift(1), "prev_ibs": ((D["c1"] - D["l1"]) / (D["h1"] - D["l1"])).shift(1),
         "prev_gap": D["g"].shift(1), "prev_z": D["z"].shift(1), "prev_mis": D["mis"].shift(1),
         "frn": inv["net_foreign"].shift(1), "ind": inv["net_individual"].shift(1), "pen": inv["net_pension"].shift(1), "trust": inv["net_trust"].shift(1),
         "nonarb": prg["nonarb_net"].shift(1), "arb": prg["arb_net"].shift(1), "fut_frn": ffrn.shift(1), "fut_ind": find.shift(1)}
    return {k: zroll(v) for k, v in X.items()}


# ---------------------------------------------------------------- 셀 기록
CELLS, FAM, BASE, SER = [], {}, None, {}


def cell(cid, name, mk, cal, pos, gross, net, stress, ok, notes="", fam=None, hz="day"):
    """pos·gross·net·stress: 일별 자본 대비(비활성 0). 거래 = pos ≠ 0 인 날."""
    p = pos.reindex(cal).fillna(0.0)
    act = p != 0
    d, g, s = [x.reindex(cal).fillna(0.0).where(act, 0.0) for x in (net, gross, stress)]
    ev, n = d[act], int(act.sum())
    eq = (1 + d).cumprod()
    row = dict(id=cid, name=name, market=mk, horizon=hz, window=f"{cal[0].date()}~{cal[-1].date()}", n=n, gross_bp=np.nan, net_bp=np.nan, t_nw=np.nan,
               cagr_pct=round((eq.iloc[-1] ** (ANN / len(cal)) - 1) * 100, 2), mdd_pct=round((eq / eq.cummax() - 1).min() * 100, 2), yearly_pos="0/0", top5_share_pct=np.nan,
               half1_net_bp=np.nan, half2_net_bp=np.nan, stress_net_bp=np.nan, corr_base="", exec_ok=ok, notes=notes)
    if n:
        yr = ev.groupby(ev.index.year).agg(["sum", "count"])
        yr = yr[yr["count"] >= 5]
        tot = ev.sum()
        mid = cal[len(cal) // 2]
        h1, h2 = ev[ev.index < mid], ev[ev.index >= mid]
        row.update(gross_bp=round(g[act].mean() * 1e4, 2), net_bp=round(ev.mean() * 1e4, 2), t_nw=round(nw_t(ev), 2), yearly_pos=f"{int((yr['sum'] > 0).sum())}/{len(yr)}",
                   top5_share_pct=round(ev.nlargest(5).sum() / tot * 100, 1) if tot > 0 else np.nan,
                   half1_net_bp=round(h1.mean() * 1e4, 2) if len(h1) else np.nan, half2_net_bp=round(h2.mean() * 1e4, 2) if len(h2) else np.nan,
                   stress_net_bp=round(s[act].mean() * 1e4, 2))
        c = pd.concat([d.rename("c"), BASE], axis=1, join="inner")
        if len(c) > 30 and c["c"].std() > 0:
            row["corr_base"] = f"ov {c['c'].corr(c['ov']):+.2f}/id {c['c'].corr(c['id']):+.2f}/kp {c['c'].corr(c['kp']):+.2f}"
    CELLS.append(row)
    SER[cid] = (d, act, p)
    if fam:
        FAM.setdefault(fam, []).append(cid)
    return row


def crit(r):
    y = r["yearly_pos"].split("/")
    ys = int(y[0]) / int(y[1]) if int(y[1]) else 0
    c = [r["net_bp"] > 0, r["t_nw"] >= 2, ys >= 0.7, (r["top5_share_pct"] < 50) if pd.notna(r["top5_share_pct"]) else False, r["half1_net_bp"] > 0, r["half2_net_bp"] > 0, r["stress_net_bp"] > 0]
    return "".join("O" if bool(v) else "X" for v in c)


# ---------------------------------------------------------------- 수익 경로
def run_auc(D, sig, cap=2.0, w=None):
    """09:00 단일가 → 15:30 단일가(일봉 수정가 수익). 비용 5bp, 스트레스 7bp + 1틱."""
    w = wt(sig, D["sig"], cap) if w is None else w
    w = w.where(D["r1"].notna() & D["r2"].notna(), 0.0)
    a, b = mix(w)
    g = a * D["r1"].fillna(0) + b * D["r2"].fillna(0)
    return w, g, g - (a + b) * 5e-4, g - (a + b) * 7e-4 - (a * D["t1"] + b * D["t2"])


def px(M, lab):
    return M["c"][lab[:-1]] if lab.endswith("c") else M["o"][lab]


def run_min(D, M1, M2, sig, ent=FST, ex="15:30c", cap=2.0):
    """분봉 원가격 수익. ent = 진입 라벨 시가(AUC = 09:00 단일가), ex = 청산 라벨(끝 c 는 그 봉 종가)."""
    days = M1["o"].index[M1["o"][AUC].notna()]
    r1, r2 = px(M1, ex).reindex(days) / M1["o"][ent].reindex(days) - 1, px(M2, ex).reindex(days) / M2["o"][ent].reindex(days) - 1
    w = wt(sig.reindex(days), D["sig"].reindex(days), cap)
    w = w.where(r1.notna() & r2.notna(), 0.0)
    a, b = mix(w)
    t = a * D["t1"].reindex(days) + b * D["t2"].reindex(days)
    g = a * r1.fillna(0) + b * r2.fillna(0)
    auction_only = ent == AUC and ex == "15:30c"
    return w, g, g - (a + b) * 5e-4 - (0 if auction_only else t), g - (a + b) * 7e-4 - (1 if auction_only else 2) * t


def run_verify(D, M1, M2, F, hold, back=None, cap=2.0):
    """E2: F 날 단일가 매수, hold 면 15:30 까지, 아니면 09:02 봉 시가 청산. back: F 밖이면서 back 인 날은 E1 진입으로 메움."""
    days = M1["o"].index[M1["o"][AUC].notna()]
    A1, A2 = M1["o"][AUC].reindex(days), M2["o"][AUC].reindex(days)
    H1, H2 = M1["c"]["15:30"].reindex(days), M2["c"]["15:30"].reindex(days)
    P1, P2 = M1["o"][FST].reindex(days), M2["o"][FST].reindex(days)
    w = wt(F.reindex(days), D["sig"].reindex(days), cap)
    a, b = mix(w)
    t = a * D["t1"].reindex(days) + b * D["t2"].reindex(days)
    h = hold.reindex(days).fillna(0) > 0
    g = a * ((H1 / A1 - 1).where(h, P1 / A1 - 1)).fillna(0) + b * ((H2 / A2 - 1).where(h, P2 / A2 - 1)).fillna(0)
    c0 = (a + b) * 5e-4 + t.where(~h, 0.0)
    c1 = (a + b) * 7e-4 + t.where(h, 2 * t)
    if back is None:
        return w, g, g - c0, g - c1, h
    bk = (back.reindex(days).fillna(0) > 0) & (w == 0)
    wb, gb, nb, sb = run_min(D, M1, M2, bk.astype(float))
    return w + wb, g + gb, g - c0 + nb, g - c1 + sb, h


def run_short(D, M1, sig, veh, Mv=None, code=None, ent=FST, ex="15:30c"):
    """명목 1 숏. veh='fut': 1배 ETF 경로의 부호 반전 + 선물 비용(선물 분봉 미보유, 대용). veh='inv': 인버스 ETF 실경로."""
    days = M1["o"].index[M1["o"][AUC].notna()]
    s = sig.reindex(days).fillna(0).astype(float)
    if veh == "fut":
        g = -s * (px(M1, ex).reindex(days) / M1["o"][ent].reindex(days) - 1).fillna(0)
        return s, g, g - s * 1.8e-4, g - s * 3e-4
    t = tick_of(code, days)
    g = s * (px(Mv, ex).reindex(days) / Mv["o"][ent].reindex(days) - 1).fillna(0)
    return s, g, g - s * (5e-4 + t), g - s * (7e-4 + 2 * t)


# ---------------------------------------------------------------- 본체
def repro(K, Q):
    r5 = daily(f"{BT}/round5_strategy/c3_levered/daily_returns.csv")
    r9 = daily(f"{BT}/round9_c3_etfs/kq150_etfs/daily_returns.csv")
    x = K.join(r5[["C3", "z_index", "w_chosen", "net5_chosen"]], rsuffix="_r5").loc[START["KOSPI"]:]
    w, g, n, s = run_auc(K, K["C3"])
    assert (x["C3"] != x["C3_r5"]).sum() == 0 and (n.reindex(x.index) - x["net5_chosen"]).abs().max() < 1e-8, "코스피 C3 재현 실패"
    y = Q.join(r9[["z_spx", "w_KODEX_vt30", "net5_KODEX_vt30"]])
    y = y[y["w_KODEX_vt30"].notna()]
    w2, g2, n2, s2 = run_auc(Q, Q["C3"])
    assert (n2.reindex(y.index) - y["net5_KODEX_vt30"]).abs().max() < 1e-8 and (y["z"] - y["z_spx"]).abs().max() < 1e-5, "코스닥 C3 재현 실패"
    return dict(k_days=len(x), k_on=int((x["C3"] == 1).sum()), q_days=len(y), q_on=int((w2.reindex(y.index) > 0).sum()))


def path_tables(D, M, mk):
    """그룹별 ETF 자체 가격 경로(단일가 기준 bp)와 구간 증분(C3 − 전체 평균)."""
    labs = ["09:02", "09:03", "09:04", "09:06", "09:08", "09:11", "09:16", "09:21", "09:31", "10:01", "11:01", "13:01"]
    rows, inc = [], []
    for c in MKT[mk]:
        o, cl = M[c]["o"], M[c]["c"]
        days = o.index[o[AUC].notna()]
        x = D.reindex(days)
        grp = {"C3": x["C3"] == 1, "z≥1∧mis>0": (x["z"] >= 1) & (x["mis"] > 0), "z<1∧mis≤0": (x["z"] < 1) & (x["mis"] <= 0), "z<1∧mis>0": (x["z"] < 1) & (x["mis"] > 0), "전체": x["z"].notna()}
        U = pd.DataFrame({L: np.log(o.loc[days, L] / o.loc[days, AUC]) for L in labs})
        U["15:20c"], U["15:30c"] = np.log(cl.loc[days, "15:20"] / o.loc[days, AUC]), np.log(cl.loc[days, "15:30"] / o.loc[days, AUC])
        for g, m in grp.items():
            r = {"market": mk, "code": c, "group": g, "n": int(m.sum())}
            r.update({L: round(U.loc[m, L].mean() * 1e4, 1) for L in U.columns})
            r["t_09:02"] = round(nw_t(U.loc[m, "09:02"]), 2)
            r["t_후반(09:02→15:30)"] = round(nw_t(U.loc[m, "15:30c"] - U.loc[m, "09:02"]), 2)
            rows.append(r)
        cols = ["09:02", "09:03", "09:06", "09:11", "09:16", "09:31", "10:01", "13:01", "15:30c"]
        allm = U[grp["전체"]].mean()
        prev = None
        for L in cols:
            step = U[L] - (U[prev] if prev else 0.0)
            s3 = (step - (allm[L] - (allm[prev] if prev else 0.0)))[grp["C3"]]
            inc.append({"market": mk, "code": c, "구간": f"{prev or '단일가'}→{L}", "C3 평균 bp": round(step[grp['C3']].mean() * 1e4, 1), "전체 평균 bp": round(step[grp['전체']].mean() * 1e4, 1),
                        "초과 bp": round(s3.mean() * 1e4, 1), "초과 t": round(nw_t(s3), 2)})
            prev = L
    return pd.DataFrame(rows), pd.DataFrame(inc)


def decomp(D, M, mk):
    """C3 날 분해: 첫 1분 초과(ETF 시가 저평가 몫) vs 09:02 시가 뒤(지수 갭 잔차 방향 몫). 분봉 ETF 자체 가격만."""
    c1, c2 = MKT[mk]
    days = M[c1]["o"].index[M[c1]["o"][AUC].notna()]
    x = D.reindex(days)
    w = wt(x["C3"], x["sig"])
    a, b = mix(w)
    on = w > 0
    u1 = np.log(M[c1]["o"].loc[days, FST] / M[c1]["o"].loc[days, AUC])
    u2 = np.log(M[c2]["o"].loc[days, FST] / M[c2]["o"].loc[days, AUC])
    t1 = np.log(M[c1]["c"].loc[days, "15:30"] / M[c1]["o"].loc[days, AUC])
    t2 = np.log(M[c2]["c"].loc[days, "15:30"] / M[c2]["o"].loc[days, AUC])
    n1 = x["z"].notna()
    base1, base2 = u1[n1].mean(), u2[n1].mean()                  # 신호 없는 평소 첫 1분(모든 날 평균)
    tot = (a * t1 + b * t2)[on]
    und = (a * (u1 - base1) + b * (u2 - base2))[on]
    day1 = (a * base1 + b * base2)[on]
    dir_ = (a * (t1 - u1) + b * (t2 - u2))[on]
    return {"market": mk, "n": int(on.sum()), "평균 w": round(w[on].mean(), 3), "합계 bp": round(tot.mean() * 1e4, 2), "시가 저평가 몫 bp(첫 1분 − 평소)": round(und.mean() * 1e4, 2),
            "저평가 t": round(nw_t(und), 2), "평소 첫 1분 bp": round(day1.mean() * 1e4, 2), "방향 몫 bp(09:02 시가→15:30)": round(dir_.mean() * 1e4, 2), "방향 t": round(nw_t(dir_), 2),
            "저평가 비중 %": round(und.mean() / tot.mean() * 100, 1), "방향 비중 %": round(dir_.mean() / tot.mean() * 100, 1)}


def noise_sim(D, mk, reps=300):
    """08:59 예상가 오차 민감도: 지수·ETF 관측 오차 N(0, σ) 를 넣고 신호만 바꾼다(체결은 실제 단일가). 재현율·수익 잔존."""
    x = D.loc[START[mk]:].copy()
    w0, g0, n0, _ = run_auc(x, x["C3"])
    base = n0.sum()
    rng = np.random.default_rng(20261030)
    out = []
    for sd in (5, 10, 20, 40):
        s = sd / 1e4
        keep, ret, cnt = [], [], []
        for _ in range(reps):
            e1, e2 = rng.normal(0, s, len(x)), rng.normal(0, s, len(x))
            z = rolling_z1(x["g"] + e2, x["us"])
            sig = ((z >= 1) & (x["mis"] + e1 - e2 <= 0)).astype(float).where(x["C3"].notna())
            _, _, n, _ = run_auc(x, sig)
            keep.append(n.sum() / base)
            ret.append(((sig == 1) & (x["C3"] == 1)).sum() / (x["C3"] == 1).sum())
            cnt.append((sig == 1).sum())
        out.append({"market": mk, "오차 SD bp": sd, "순수익 합 잔존 %": round(np.mean(keep) * 100, 1), "C3 재현율 %": round(np.mean(ret) * 100, 1), "평균 거래 수": round(np.mean(cnt), 1)})
    return out


CAND = ["F|KOSPI|verb|1.0", "F|KOSDAQ|moo|1.0", "F|KOSDAQ|e1all|1.0", "E1|KOSPI|C3", "E1|KOSPI|ex10:01", "S|KOSDAQ|fut|0", "S|KOSPI|fut|0", "E5|KOSPI|prev_gap|상위|1.25"]


def style(ax):
    ax.set_facecolor(SURF)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.tick_params(colors=MUTED, labelsize=9)


def figs(DS, M, WIN):
    """그림 1: C3 날 ETF 자체 가격 첫 60분. 그림 2: post 창 실행판 누적."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), facecolor=SURF, sharey=True)
    for ax, (mk, c) in zip(axes, (("KOSPI", "069500"), ("KOSDAQ", "229200"))):
        D, o = DS[mk], M[c]["o"]
        days = WIN[mk]["min"]
        labs = [f"09:{i:02d}" for i in range(1, 60)] + ["10:00", "10:01"]
        U = pd.DataFrame({L: np.log(o.loc[days, L] / o.loc[days, AUC]) * 1e4 for L in labs})
        c3 = (D["C3"] == 1).reindex(days).fillna(False)
        x = np.arange(len(labs))
        y1, y0 = U[c3].mean(), U.mean()
        ax.plot(x, y1.values, color=C1, lw=2, label=f"C3 날 (n {int(c3.sum())})")
        ax.plot(x, y0.values, color=C2, lw=2, label=f"모든 날 (n {len(U)})")
        ax.axvline(1, color=MUTED, lw=1, ls="--")
        ax.text(1.6, ax.get_ylim()[1] * 0.97, "09:02 시가 = 사후 첫 체결", color=MUTED, fontsize=8, va="top")
        ax.set_xticks([0, 15, 30, 45, 60])
        ax.set_xticklabels(["09:00", "09:15", "09:30", "09:45", "10:00"])
        ax.set_title(f"{mk} {c} 단일가 대비 평균 가격 경로", fontsize=10, color=INK, loc="left")
        ax.legend(frameon=False, fontsize=9, loc="center right")
        style(ax)
    axes[0].set_ylabel("단일가 대비 평균 로그수익(bp)", color=MUTED, fontsize=9)
    fig.suptitle("C3 날 첫 60분 (TOSS 1분봉 2022-11-23~2024-12-30, 분봉 시가 기준, 비용 전)", fontsize=10, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(OUT / "first_60min_path.png", dpi=150, facecolor=SURF)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), facecolor=SURF)
    sets = (("KOSPI", (("R|KOSPI|post", "원형(동일가, 실행 불가)", C2, "-"), ("F|KOSPI|verb|1.0", "선매수 확인 E2b", C1, "-"), ("E1p|KOSPI|C3", "사후 진입 E1", MUTED, "--"))),
            ("KOSDAQ", (("R|KOSDAQ|post", "C3Q 원형(동일가, 실행 불가)", C2, "-"), ("F|KOSDAQ|moo|1.0", "z_f≥1 단일가 시장가", C1, "-"))))
    for ax, (mk, ss) in zip(axes, sets):
        for cid, lab, col, ls in ss:
            eq = (1 + SER[cid][0]).cumprod()
            ax.plot(eq.index, eq.values, color=col, lw=2, ls=ls, label=f"{lab} ×{eq.iloc[-1]:.3f}")
        ax.set_title(f"{mk} 아침 다리 누적(자본 1, 2023-07-31~2024-12-30)", fontsize=10, color=INK, loc="left")
        ax.legend(frameon=False, fontsize=9, loc="upper left")
        ax.xaxis.set_major_locator(matplotlib.dates.MonthLocator(bymonth=(1, 4, 7, 10)))
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%Y-%m"))
        style(ax)
    axes[0].set_ylabel("누적 배수(비용 차감)", color=MUTED, fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "equity_post.png", dpi=150, facecolor=SURF)
    plt.close(fig)


def hac(y, X, lags=5):
    """y ~ 상수 + X, Newey-West t(lags). 반환: {이름: (계수, t)}."""
    d = pd.concat([y.rename("y")] + [x.rename(n) for n, x in X.items()], axis=1).dropna()
    A = np.column_stack([np.ones(len(d))] + [d[n].to_numpy() for n in X])
    yv = d["y"].to_numpy()
    b = np.linalg.lstsq(A, yv, rcond=None)[0]
    u = yv - A @ b
    inv = np.linalg.inv(A.T @ A)
    S_ = (A * u[:, None]).T @ (A * u[:, None])
    for L in range(1, lags + 1):
        G = (A[L:] * u[L:, None]).T @ (A[:-L] * u[:-L, None])
        S_ += (1 - L / (lags + 1)) * (G + G.T)
    V = inv @ S_ @ inv * len(d) / (len(d) - A.shape[1])
    return {n: (round(float(b[i + 1]), 3), round(float(b[i + 1] / np.sqrt(V[i + 1, i + 1])), 2)) for i, n in enumerate(X)}


def lag_check(D, M1, mk):
    """지수 시가가 ETF 단일가보다 늦게 찍혀 첫 순간 하락이 mis 에 섞였는지: 첫 1분을 통제해도 mis 가 남는가."""
    o, c = M1["o"], M1["c"]
    days = o.index[o[AUC].notna()]
    x = D.reindex(days)
    y = np.log(c.loc[days, "15:30"] / o.loc[days, FST]) * 1e4
    y6 = np.log(c.loc[days, "15:30"] / o.loc[days, "09:06"]) * 1e4
    u1, u5 = np.log(o.loc[days, FST] / o.loc[days, AUC]) * 1e4, np.log(o.loc[days, "09:06"] / o.loc[days, FST]) * 1e4
    mm = x["mis"] * 1e4
    sig = (x["z"] < 1) & (x["mis"] > 0)
    ev = -y[sig] - 1.8
    ex = ev.drop(pd.Timestamp("2024-08-05"), errors="ignore")
    a, b, e = hac(y, {"mis": mm}), hac(y, {"mis": mm, "u1": u1}), hac(y6, {"mis": mm, "u1": u1, "u5": u5})
    return {"market": mk, "mis 단독 계수": a["mis"][0], "t": a["mis"][1], "첫 1분 통제 계수": b["mis"][0], "통제 t": b["mis"][1], "09:06 진입 계수": e["mis"][0], "09:06 t": e["mis"][1],
            "숏 순 bp": round(ev.mean(), 2), "2024-08-05 제외 bp": round(ex.mean(), 2), "제외 t": round(nw_t(ex), 2), "09:06 진입 숏 총 bp": round(-y6[sig].mean(), 2), "09:06 t2": round(nw_t(-y6[sig]), 2)}


def report(C, R):
    """notes.md 와 scan_morning_exec.md. 숫자는 표에서 그대로 가져온다."""
    g = C.set_index("id")
    m = lambda s: str(s).replace("-", "−")                     # noqa: E731  음수 기호
    v = lambda x, f="{:+.2f}": "결측" if pd.isna(x) else m(f.format(x))     # noqa: E731
    rt = R["RT"].set_index("market")
    dc = R["DC"].set_index("market")
    dt = R["DT"].set_index("id")
    cap = R["CAP"]
    bk = R["BK"]

    def row(cid, label):
        r = g.loc[cid]
        return f"| {label} | {r['window']} | {r['n']} | {v(r['gross_bp'])} | {v(r['net_bp'])} | {v(r['t_nw'])} | {v(r['cagr_pct'])} | {v(r['mdd_pct'])} | {r['yearly_pos']} | {v(r['top5_share_pct'], '{:.1f}')} | {v(r['half1_net_bp'])} / {v(r['half2_net_bp'])} | {v(r['stress_net_bp'])} |"

    HDR = "| 구분 | 창 | n | 총 bp | 순 bp | NW t | CAGR % | MDD % | 연도 + | 상위 5건 % | 전반 / 후반 bp | 스트레스 bp |\n|---|---|---:|---:|---:|---:|---:|---:|---|---:|---|---:|"

    def keep(a, b):
        ra, rb = g.loc[a], g.loc[b]
        return dict(g=rb["n"] * rb["gross_bp"] / (ra["n"] * ra["gross_bp"]) * 100, n=rb["n"] * rb["net_bp"] / (ra["n"] * ra["net_bp"]) * 100, c=rb["cagr_pct"] / ra["cagr_pct"] * 100)

    k_e1m, k_e1p, k_e2 = keep("R|KOSPI|min", "E1|KOSPI|C3"), keep("R|KOSPI|post", "E1p|KOSPI|C3"), keep("R|KOSPI|post", "F|KOSPI|verb|1.0")
    k_e1x = keep("R|KOSPI|min", "E1|KOSPI|ex10:01")
    e5 = C[C["id"].str.startswith("E5|")]
    pos2 = int((e5["t_nw"] >= 2).sum())
    q, qq = g.loc["F|KOSDAQ|moo|1.0"], g.loc["R|KOSDAQ|post"]
    kv, kr = g.loc["F|KOSPI|verb|1.0"], g.loc["R|KOSPI|post"]
    e1 = g.loc["E1|KOSPI|C3"]
    bwin = bk[bk["창"] == "W_des"]
    bval = lambda a, b="합산 50:50": bwin[(bwin["market"] == b) & (bwin["B 다리"] == a)].iloc[0]   # noqa: E731
    hyb = "K E2b(2023-07-31~) + 그 전 E1 / Q z_f 단일가(2023-07-31~) + 그 전 z_f 09:02"
    bo, be, bh, bx, bn = bval("K 원형 / Q 원형"), bval("K E1 15:30 / Q E1 15:30"), bval(hyb), bval("K E1 10:01 / Q E1 10:01"), bval("K B 없음 / Q B 없음")
    capd = cap[(cap["규칙"] == "orig")]
    lc = R["LC"].set_index("market")
    PTx = R["PT"].set_index(["code", "group"])

    L = ["# 16차 스캔 morning_exec: 아침 C3 실행 가능판", "",
         f"• 범위: 설계 구간만(수익 자료 ~2024-12-30). 2025-01-02 이후 수익 미열람. 돌린 셀 {len(C)}개(실행 가능 표시 {int(C['exec_ok'].astype(bool).sum())}개). 코드 r16s_morning_exec.py, 결과 morning_exec/.",
         "• 비용: ETF 왕복 5bp, 연속 체결이 끼면 1틱 추가, 스트레스 7bp + 1틱(단일가)·2틱(연속). 선물 왕복 1.8bp, 스트레스 3bp. 수익은 자본 대비.",
         "• 깃 밖 자료: kis_fut_k200_daily, kis_fut_kq150_daily(z_f), yf_ewy_daily, fdr_usdkrw_daily. 나머지는 깃 자료.",
         "• 용어: E1 = 09:00 뒤 09:02 봉 시가 진입. E2 = 09:00 전 필터로 단일가 선매수 뒤 C3 확인. E2b = E2 + 필터 밖 C3 는 E1. E4 = 지정가 단일가. E5 = 사전 신호 시장가. post = 2023-07-31~2024-12-30.",
         "• 봉인 고지: 지시된 ADR·조사 문서를 읽는 과정에서 13차 합산 장부의 2025·2026 수치가 화면에 나왔다. 이 스캔의 신호·문턱·창 선택에는 쓰지 않았다. 스크립트의 모든 읽기는 2024-12-30 으로 자른 뒤 수익을 계산한다.", "",
         "## 결론", "",
         f"• 코스피 실행 가능판 E2b: 2023-07-31~2024-12-30, n {kv['n']}, 순 {v(kv['net_bp'])}bp, NW t {v(kv['t_nw'])}, CAGR {v(kv['cagr_pct'])}%, MDD {v(kv['mdd_pct'])}%. 원형 C3 같은 창(n {kr['n']}, 순 {v(kr['net_bp'])}bp, CAGR {v(kr['cagr_pct'])}%, MDD {v(kr['mdd_pct'])}%) 대비 잔존: 총수익 합 {k_e2['g']:.1f}%, 순수익 합 {k_e2['n']:.1f}%, CAGR {k_e2['c']:.1f}%.",
         f"• 코스피 대체판 E1(선물 자료 불필요): 2022-11-23~2024-12-30, n {e1['n']}, 순 {v(e1['net_bp'])}bp, NW t {v(e1['t_nw'])}, CAGR {v(e1['cagr_pct'])}%. 원형 대비 총수익 합 {k_e1m['g']:.1f}%, 순수익 합 {k_e1m['n']:.1f}%, CAGR {k_e1m['c']:.1f}%. 10:01 청산이면 순 {v(g.loc['E1|KOSPI|ex10:01', 'net_bp'])}bp, t {v(g.loc['E1|KOSPI|ex10:01', 't_nw'])}, MDD {v(g.loc['E1|KOSPI|ex10:01', 'mdd_pct'])}%.",
         f"• 코스닥 C3Q 는 실행 문제 이전에 설계 분봉 창에서 수익이 없다: 2022-11-23~2024-12-30 n {g.loc['R|KOSDAQ|min', 'n']}, 동일가 원형 순 {v(g.loc['R|KOSDAQ|min', 'net_bp'])}bp(t {v(g.loc['R|KOSDAQ|min', 't_nw'])}), 09:02 진입 순 {v(g.loc['E1|KOSDAQ|C3', 'net_bp'])}bp. 잔존율은 분모가 0 근처라 계산하지 않는다. 장기창 2016-05-09~2024-12-30 은 n {g.loc['R|KOSDAQ|long', 'n']}, 순 {v(g.loc['R|KOSDAQ|long', 'net_bp'])}bp(전반 {v(g.loc['R|KOSDAQ|long', 'half1_net_bp'])}, 후반 {v(g.loc['R|KOSDAQ|long', 'half2_net_bp'])}).",
         f"• 코스닥 대체 후보: 08:45 코스닥150 선물 갭의 미국 잔차 z_f ≥ 1 → 09:00 단일가 시장가 → 15:30. 2023-07-31~2024-12-30, n {q['n']}, 순 {v(q['net_bp'])}bp, NW t {v(q['t_nw'])}, CAGR {v(q['cagr_pct'])}%, MDD {v(q['mdd_pct'])}%, 스트레스 {v(q['stress_net_bp'])}bp. 2023-11-06 제외 순 {v(dt.loc['F|KOSDAQ|moo|1.0', '2023-11-06 제외 순 bp'])}bp(t {v(dt.loc['F|KOSDAQ|moo|1.0', '제외 t'])}). 상위 5건 {v(q['top5_share_pct'], '{:.1f}')}% 로 선별 기준 미달이고 공매도 금지 구간 거래 비중 {v(dt.loc['F|KOSDAQ|moo|1.0', '금지 구간 거래 비중 %'], '{:.1f}')}%.",
         f"• ETF 시가 저평가 몫은 첫 1분 안에 끝난다(봉 해상도 안). 코스피 C3 날 분해(분봉 ETF 가격, n {dc.loc['KOSPI', 'n']}): 합계 {v(dc.loc['KOSPI', '합계 bp'])}bp 중 저평가 {v(dc.loc['KOSPI', '시가 저평가 몫 bp(첫 1분 − 평소)'])}bp({dc.loc['KOSPI', '저평가 비중 %']:.1f}%, t {v(dc.loc['KOSPI', '저평가 t'])}), 방향 {v(dc.loc['KOSPI', '방향 몫 bp(09:02 시가→15:30)'])}bp({dc.loc['KOSPI', '방향 비중 %']:.1f}%, t {v(dc.loc['KOSPI', '방향 t'])}).",
         f"• 13차 장부 합산 50:50(W_des 2022-12-13~2024-12-30, A·C 다리 그대로): 원형 {v(bo['CAGR %'])}%/{v(bo['MDD %'])}%, E1 두 시장 {v(be['CAGR %'])}%/{v(be['MDD %'])}%, 실행판 혼합(코스피 E2b·코스닥 z_f, 2023-07-31 전은 코스피 E1·코스닥 z_f 09:02) {v(bh['CAGR %'])}%/{v(bh['MDD %'])}%, 두 시장 E1 10:01 {v(bx['CAGR %'])}%/{v(bx['MDD %'])}%, B 없음 {v(bn['CAGR %'])}%/{v(bn['MDD %'])}%.",
         f"• 명목 상한 1.0·1.5 는 2020-03-19 손실을 줄이지 못한다. 그날 w 가 코스피 0.89, 코스닥 0.589 로 상한 아래였다.",
         f"• 부산물: 코스피 z<1 ∧ mis>0 날 09:02 시가 낮 숏(선물 비용, ETF 경로 대용): 2022-11-23~2024-12-30, n {g.loc['S|KOSPI|fut|0', 'n']}, 순 {v(g.loc['S|KOSPI|fut|0', 'net_bp'])}bp, t {v(g.loc['S|KOSPI|fut|0', 't_nw'])}, 선물 왕복 1.8bp. 코스닥판은 연속 관계가 없어 기각. 9절.", "",
         "## 1. 실행 가능판 규칙", "",
         "• 공통: z_f(T) = (ln(근월 선물 시가(T) / 같은 월물 종가(T−1)) − a − b·us(T)) / s. (a, b, s) 는 T 제외 직전 250행 OLS(유효 200행 이상). us(T) = T 이전 미국 세션 SPX 누적 로그수익. 선물 시가 08:45 뒤 확정.",
         "• 코스피 E2b",
         "  · 08:45~09:00: z_f ≥ 1 이면 09:00 단일가 시장가로 069500·122630 매수. w = clip(0.30/σ20, 0, 2), w ≤ 1 → 069500 w, 1 < w ≤ 2 → 069500 (2−w) + 122630 (w−1).",
         "  · 09:00 단일가 뒤: 원형 C3(z ≥ 1 ∧ mis ≤ 0) 확인. 통과면 15:30 단일가 청산, 실패면 09:02 봉 시가(09:01:00 첫 체결) 청산.",
         "  · z_f < 1 인데 C3 통과: 09:02 봉 시가 진입, 15:30 단일가 청산.",
         "• 코스닥 z_f 시장가: z_f ≥ 1 이면 09:00 단일가 시장가로 229200·233740 매수(w 는 코스닥150 σ20 기준 같은 식), 15:30 단일가 청산. mis 확인 없음.",
         "• 코스피 E1: C3 → 09:02 봉 시가 진입 → 15:30(또는 10:01 봉 시가) 청산. 깃 자료만으로 계산.",
         "• 2023-07-31 전에는 선물 시가가 09:00 이라 사전 신호가 없다. 그 구간 실행판은 E1 뿐이고, 코스닥은 z_f 를 09:02 진입으로만 쓸 수 있다.", "",
         "## 2. 원형 대비 잔존", "",
         "창·비용은 표 안 창 열과 머리말 기준. 2023-07-31~ 창 이름 post, 분봉 창 2022-11-23~.", "", HDR,
         row("R|KOSPI|min", "코스피 원형 VT30(실행 불가)"), row("E1|KOSPI|C3", "코스피 E1 09:02→15:30"), row("E1|KOSPI|ex10:01", "코스피 E1 09:02→10:01"), row("E1f|KOSPI|C3", "코스피 E1 선물 비용 대용"),
         row("R|KOSPI|post", "코스피 원형 post"), row("E1p|KOSPI|C3", "코스피 E1 post"), row("E1p|KOSPI|ex10:01", "코스피 E1 10:01 post"), row("F|KOSPI|verb|1.0", "코스피 E2b z_f≥1"),
         row("R|KOSDAQ|min", "코스닥 C3Q 원형(실행 불가)"), row("E1|KOSDAQ|C3", "코스닥 E1 09:02→15:30"), row("E1|KOSDAQ|ex10:01", "코스닥 E1 09:02→10:01"), row("R|KOSDAQ|post", "코스닥 C3Q 원형 post"),
         row("F|KOSDAQ|moo|1.0", "코스닥 z_f≥1 단일가"), row("F|KOSDAQ|e1|1.0", "코스닥 z_f≥1 09:02 진입 post"), row("F|KOSDAQ|e1all|1.0", "코스닥 z_f≥1 09:02 진입 전체"), "",
         f"• 코스피 잔존(원형 대비, 같은 창): E1 분봉 창 총 {k_e1m['g']:.1f}%·순 {k_e1m['n']:.1f}%·CAGR {k_e1m['c']:.1f}%, E1 10:01 총 {k_e1x['g']:.1f}%·순 {k_e1x['n']:.1f}%·CAGR {k_e1x['c']:.1f}%, E1 post 총 {k_e1p['g']:.1f}%·순 {k_e1p['n']:.1f}%·CAGR {k_e1p['c']:.1f}%, E2b post 총 {k_e2['g']:.1f}%·순 {k_e2['n']:.1f}%·CAGR {k_e2['c']:.1f}%.",
         f"• 후보 표본 품질은 7절. E2b 평균 명목 {dt.loc['F|KOSPI|verb|1.0', '평균 명목']:.3f}배, 코스닥 z_f 평균 명목 {dt.loc['F|KOSDAQ|moo|1.0', '평균 명목']:.3f}배(모두 ETF 현금, 증거금 없음).", "",
         "## 3. 저평가 회복 종료 시각(ETF 자체 분봉, 2022-11-23~2024-12-30, 비용 전)", "",
         "• 구간별 C3 날 평균과 모든 날 평균의 차. 단위 bp. 초과 t 는 C3 날 값에서 모든 날 평균을 뺀 NW t.", "",
         "| 종목 | 구간 | C3 평균 | 모든 날 평균 | 초과 | 초과 t |", "|---|---|---:|---:|---:|---:|"]
    ic = R["IC"]
    for _, r in ic[ic["code"].isin(["069500", "122630", "229200", "233740"])].iterrows():
        L.append(f"| {r['code']} | {r['구간']} | {v(r['C3 평균 bp'], '{:+.1f}')} | {v(r['전체 평균 bp'], '{:+.1f}')} | {v(r['초과 bp'], '{:+.1f}')} | {v(r['초과 t'])} |")
    i0 = ic[(ic["code"] == "069500") & (ic["구간"] == "단일가→09:02")].iloc[0]
    i1 = ic[(ic["code"] == "069500") & (ic["구간"] == "09:02→09:03")].iloc[0]
    q0 = ic[(ic["code"] == "229200") & (ic["구간"] == "단일가→09:02")].iloc[0]
    q1 = ic[(ic["code"] == "229200") & (ic["구간"] == "09:02→09:03")].iloc[0]
    L += ["", f"• 회복 종료: 09:02 봉 시가(09:01:00) 이전. 069500 초과 {v(i0['초과 bp'], '{:+.1f}')}bp(t {v(i0['초과 t'])}) 뒤 다음 1분 {v(i1['초과 bp'], '{:+.1f}')}bp(t {v(i1['초과 t'])}), 229200 초과 {v(q0['초과 bp'], '{:+.1f}')}bp(t {v(q0['초과 t'])}) 뒤 {v(q1['초과 bp'], '{:+.1f}')}bp(t {v(q1['초과 t'])}). 초 단위 경로는 자료 없음(미확인).",
          f"• 첫 봉 안 체결 진단(09:01 봉 (시가+종가)/2, 규약 위반이라 실행 불가): 코스피 총 {v(g.loc['E1|KOSPI|mid', 'gross_bp'])}bp, 원형 {v(g.loc['R|KOSPI|min', 'gross_bp'])}bp, E1 {v(g.loc['E1|KOSPI|C3', 'gross_bp'])}bp.",
          "• 09:06 뒤 C3 날 초과는 방향 몫으로 분류. 069500 은 09:11~10:01 에 초과 t 2 이상 구간이 이어지고 그 뒤 초과는 없다. 229200 은 09:06→09:11 초과 뒤 13:01→15:30 에 되돌린다.",
          "• 그림 first_60min_path.png.", "",
          "## 4. C3 수익 분해(분봉 ETF 자체 가격만, VT30 혼합)", "",
          "• 저평가 몫 = 단일가→09:02 시가 수익 − 같은 창 모든 날 평균. 방향 몫 = 09:02 시가→15:30. 합계에는 평소 첫 1분이 더해진다.", "",
          "| 시장 | n | 평균 w | 합계 bp | 저평가 bp | 저평가 t | 평소 첫 1분 bp | 방향 bp | 방향 t |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for mk, r in dc.iterrows():
        L.append(f"| {mk} | {int(r['n'])} | {r['평균 w']:.3f} | {v(r['합계 bp'])} | {v(r['시가 저평가 몫 bp(첫 1분 − 평소)'])} | {v(r['저평가 t'])} | {v(r['평소 첫 1분 bp'])} | {v(r['방향 몫 bp(09:02 시가→15:30)'])} | {v(r['방향 t'])} |")
    L += ["", "• post 창 경로별 잔존(C3 날 합계 bp, 2023-07-31~2024-12-30, 자본 대비, 총수익은 비용 전).", "",
          "| 시장 | C3 일 | 필터 안 C3 | 필터 실패일 | 합계 | 저평가 | 방향 | E2b 저평가 잔존 | 잔존 % | 실패일 총손익 | E2b 총 | E2b 순 |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for mk, r in rt.iterrows():
        L.append(f"| {mk} | {int(r['C3 일'])} | {int(r['필터 안 C3'])} | {int(r['필터 실패일'])} | {m(int(r['합계 bp']))} | {m(int(r['저평가 몫 bp']))} | {m(int(r['방향 몫(평소 첫 1분 포함) bp']))} | {m(int(r['E2b 저평가 잔존 bp']))} | {r['E2b 저평가 잔존 %']:.1f} | {m(int(r['실패일 총손익 bp']))} | {m(int(r['E2b 총 bp']))} | {m(int(r['E2b 순 bp']))} |")
    L += ["", f"• 잔존: E1 은 저평가 몫 0%, 방향 몫 100%(진입 1틱 별도). 코스피 E2b 는 저평가 몫 {rt.loc['KOSPI', 'E2b 저평가 잔존 %']:.1f}%(필터 안 C3 {int(rt.loc['KOSPI', '필터 안 C3'])}일), 방향 몫 100%, 필터 실패 {int(rt.loc['KOSPI', '필터 실패일'])}일 총손익 {m(int(rt.loc['KOSPI', '실패일 총손익 bp']))}bp.",
          "• 선물 대조(일봉 시가→종가, 2023-07-28 까지 선물 시가는 09:00): z ≥ 1 ∧ mis ≤ 0 날 선물 수익이 ETF 와 비슷해서 ETF 고유 저평가보다 지수 시가가 ETF·선물보다 높게 찍힌 날의 방향 성격이 크다.", "",
          "| 시장 | 창 | 그룹 | n | ETF bp | ETF t | 지수 bp | 선물 bp | 선물 t |", "|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for _, r in R["T22"].iterrows():
        L.append(f"| {r['market']} | {r['창']} | {r['그룹']} | {r['n']} | {v(r['ETF bp'])} | {v(r['ETF t'])} | {v(r['지수 bp'])} | {v(r['선물 bp'])} | {v(r['선물 t'])} |")
    L += ["", "• 코스닥은 z ≥ 1 날 mis 부호가 수익을 가르지 않는다. 그래서 코스닥 실행판에서 mis 확인을 뺐다(설계 구간 판단).", "",
          "## 5. 09:00 전 신호 → 09:00 단일가", "",
          f"• 사전 신호 E5(일봉 장기창, 코스피 2010-02-22~, 코스닥 2016-05-09~, ETF 5bp, 스트레스 7bp + 1틱): {len(e5)}셀 중 NW t ≥ 2 는 {pos2}셀. t ≥ 2 단측 기준 우연 기대치는 약 {len(e5) * 0.023:.0f}셀이라 이 묶음 전체가 다중 비교 수준이다.", "", HDR,
          row("E5|KOSPI|prev_gap|상위|1.25", "전일 지수 갭 z60 ≥ 1.25"), row("E5x|KOSPI|prev_gap|C3 제외", "같은 신호, C3 날 제외"), row("E5|KOSPI|zewyx|0.75", "EWY·환율 잔차 ≥ 0.75"), row("E5|KOSPI|zewyx|1.0", "EWY·환율 잔차 ≥ 1"), row("E5|KOSPI|zewy|1.0", "EWY 잔차(환율 없음) ≥ 1"),
          "", "• 선물 08:45 경로(2023-07-31~2024-12-30, 깃 밖 자료).", "", HDR,
          row("F|KOSPI|moo|1.0", "코스피 z_f≥1 단일가 시장가"), row("F|KOSPI|lim|1.0|0", "코스피 z_f≥1 지정가"), row("F|KOSPI|lim|0.5|0", "코스피 z_f≥0.5 지정가"),
          row("F|KOSPI|verb|0.75", "코스피 E2b k 0.75"), row("F|KOSPI|verb|1.25", "코스피 E2b k 1.25"), row("F|KOSPI|vermb|1.0", "코스피 E2b 확인을 mis≤0 로"),
          row("F|KOSDAQ|moo|0.75", "코스닥 z_f≥0.75"), row("F|KOSDAQ|moo|1.25", "코스닥 z_f≥1.25"), row("F|KOSDAQ|moo1|win120", "코스닥 추정창 120행"), row("F|KOSDAQ|moo1|win500", "코스닥 추정창 500행"),
          row("F|KOSDAQ|raw|100", "코스닥 원시 선물 갭 ≥ 100bp"), row("F|KOSDAQ|moo1|ex10:01", "코스닥 z_f≥1 10:01 청산"), row("F|KOSDAQ|moo1|ex15:20c", "코스닥 z_f≥1 15:20 청산"), row("F|KOSDAQ|lim|1.0|0", "코스닥 z_f≥1 지정가"),
          row("Fd|KOSDAQ|moo|1.0", "진단 코스닥 09:00 선물 시가 z_f≥1(실행 불가)"), row("Fd|KOSPI|moo|1.0", "진단 코스피 09:00 선물 시가 z_f≥1(실행 불가)"), "",
          "• 기각 축과의 차이: 14차 D2 갭 지속은 ETF 원시 갭 ±1%·09:05 진입. 이번 셀은 선물 갭의 미국 잔차·09:00 전 확정·단일가 진입. 표에서 원시 갭 ≥ 100bp 는 약하고 잔차만 남는다.",
          "• 지정가 단일가(E4): 체결 조건이 08:45→09:00 하락 쏠림을 고른다. 코스피 전 셀 음수 또는 표본 부족. 사전 등록 대상에서 제외.",
          "• 사전 필터 재현율은 prefilter_recall.csv. 코스피 z_f≥1 재현율 "
          f"{R['TM'].loc[('KOSPI', 'z_f≥1.0'), '재현율']}%·정밀도 {R['TM'].loc[('KOSPI', 'z_f≥1.0'), '정밀도']}%, zewyx≥0.5 재현율 {R['TM'].loc[('KOSPI', 'zewyx≥0.5'), '재현율']}%·정밀도 {R['TM'].loc[('KOSPI', 'zewyx≥0.5'), '정밀도']}%, zewy≥0.5 재현율 {R['TM'].loc[('KOSPI', 'zewy≥0.5'), '재현율']}%.", "",
          "• 08:59 예상가 오차 민감도(2010~ 코스피, 2016~ 코스닥, 신호만 흔들고 체결은 실제 단일가, 300회 평균).", "",
          "| 시장 | 오차 SD bp | 순수익 합 잔존 % | C3 재현율 % | 평균 거래 수 |", "|---|---:|---:|---:|---:|"]
    for _, r in R["NS"].iterrows():
        L.append(f"| {r['market']} | {r['오차 SD bp']} | {r['순수익 합 잔존 %']} | {r['C3 재현율 %']} | {r['평균 거래 수']} |")
    L += ["", "## 6. 장기창과 VT30 명목 상한", "",
          "• 일봉 장기창(코스피 2010-02-22~2024-12-30, 코스닥 2016-05-09~2024-12-30, 비용 5bp). 원형은 동일가 위반, 근사는 분봉 창 C3 날 평균 첫 1분을 전 구간에 차감하고 1틱을 더한 값(2022-11 전 분봉 없음).", "",
          "| 시장 | 규칙 | 상한 | 평균 w | CAGR % | MDD % | 최악일 % | 최악일 |", "|---|---|---:|---:|---:|---:|---:|---|"]
    for _, r in cap.iterrows():
        L.append(f"| {r['market']} | {'원형' if r['규칙'] == 'orig' else '근사'} | {r['상한']} | {r['평균 w']} | {v(r['CAGR %'])} | {v(r['MDD %'])} | {v(r['최악일 %'])} | {r['최악일']} |")
    L += ["", f"• 2020-03-19: 코스피 w 0.89(σ20 0.337, z 7.32, ETF 시가→종가 −9.37%), 코스닥 w 0.589(σ20 0.509, z 9.22, −11.86%). VT30 의 σ20 이 급락 속도를 따라가지 못했고 상한은 닿지 않았다.",
          f"• 코스피 상한은 MDD 를 줄이지 못하고 CAGR 만 낮춘다. 코스닥은 상한 1.0 이 MDD 를 {v(capd[(capd['market'] == 'KOSDAQ') & (capd['상한'] == 2.0)].iloc[0]['MDD %'])}% → {v(capd[(capd['market'] == 'KOSDAQ') & (capd['상한'] == 1.0)].iloc[0]['MDD %'])}% 로 줄이고 CAGR 은 {v(capd[(capd['market'] == 'KOSDAQ') & (capd['상한'] == 2.0)].iloc[0]['CAGR %'])}% → {v(capd[(capd['market'] == 'KOSDAQ') & (capd['상한'] == 1.0)].iloc[0]['CAGR %'])}%.",
          f"• post 실행판 상한(코스피 E2b 상한 1.0/1.5/2.0): CAGR {v(g.loc['Cx|KOSPI|verb|1.0', 'cagr_pct'])}/{v(g.loc['Cx|KOSPI|verb|1.5', 'cagr_pct'])}/{v(g.loc['Cx|KOSPI|verb|2.0', 'cagr_pct'])}%, MDD {v(g.loc['Cx|KOSPI|verb|1.0', 'mdd_pct'])}/{v(g.loc['Cx|KOSPI|verb|1.5', 'mdd_pct'])}/{v(g.loc['Cx|KOSPI|verb|2.0', 'mdd_pct'])}%. 코스닥 z_f: CAGR {v(g.loc['Cx|KOSDAQ|moo|1.0', 'cagr_pct'])}/{v(g.loc['Cx|KOSDAQ|moo|1.5', 'cagr_pct'])}/{v(g.loc['Cx|KOSDAQ|moo|2.0', 'cagr_pct'])}%, MDD {v(g.loc['Cx|KOSDAQ|moo|1.0', 'mdd_pct'])}/{v(g.loc['Cx|KOSDAQ|moo|1.5', 'mdd_pct'])}/{v(g.loc['Cx|KOSDAQ|moo|2.0', 'mdd_pct'])}%. 상한 판단 근거로는 표본이 짧다.",
          f"• z 상한 진단: C3 중 z ≥ 4 는 코스피 7일(평균 순 −27.1bp), 코스닥 8일(+125.1bp)로 부호가 갈려 쓰지 않는다.",
          f"• EWY·환율 필터 장기 근사(zewyx≥1, 실패일 분봉 평균·필터 밖 E1 근사): 코스피 CAGR {v(g.loc['Vl|KOSPI|zewyx|1.0', 'cagr_pct'])}%·MDD {v(g.loc['Vl|KOSPI|zewyx|1.0', 'mdd_pct'])}%, E1 근사 CAGR {v(g.loc['C|KOSPI|approx|2.0', 'cagr_pct'])}%·MDD {v(g.loc['C|KOSPI|approx|2.0', 'mdd_pct'])}%. 개선 없음.", "",
          "## 7. 후보 표본 품질", "",
          "| 셀 | n | 평균 명목 | 연도 순합 % | 상위 5일 bp | 2023-11-06 제외 bp (t) | 상위 3건 제외 bp (t) | 금지 구간 비중 % | 금지 구간 / 금지 전 bp |", "|---|---:|---:|---|---|---|---|---:|---|"]
    for cid, r in dt.iterrows():
        L.append(f"| {cid.replace('|', ' ')} | {r['n']} | {r['평균 명목']:.3f} | {m(r['연도 순합 %'])} | {r['상위 5일(bp)']} | {v(r['2023-11-06 제외 순 bp'])} ({v(r['제외 t'])}) | {v(r['상위 3건 제외 순 bp'])} ({v(r['상위 3건 제외 t'])}) | {r['금지 구간 거래 비중 %']:.1f} | {v(r['금지 구간 순 bp'])} / {v(r['금지 전 순 bp'])} |")
    L += ["", "• 분기별 거래 수/평균 순 bp 는 cand_quarter.csv.", "",
          "## 8. 장부 효과(A·C 다리 13차 그대로, B 활성일 C=0, 봉인 안)", "",
          "| 장부 | 창 | B 다리 | 일수 | CAGR % | MDD % |", "|---|---|---|---:|---:|---:|"]
    for _, r in bk.iterrows():
        L.append(f"| {r['market']} | {r['창']} | {r['B 다리']} | {r['일수']} | {v(r['CAGR %'])} | {v(r['MDD %'])} |")
    L += ["", "• 코스피 E2b 실패일은 09:02 청산 뒤 C 다리 허용. 13차 C 다리 수치는 251340 틱 비용 미반영이라 수준은 낙관적이고 B 다리 비교용으로만 쓴다.",
          "• 기준 흐름 상관(일수익, 봉인 안, ov = V2 overnight, id = V2 intraday, kp = 13차 KOSPI_book): "
          f"E2b {m(g.loc['F|KOSPI|verb|1.0', 'corr_base'])}, 코스닥 z_f {m(g.loc['F|KOSDAQ|moo|1.0', 'corr_base'])}, E1 10:01 {m(g.loc['E1|KOSPI|ex10:01', 'corr_base'])}.", "",
          "## 9. 부산물: mis > 0 낮 숏", "", HDR,
          row("S|KOSPI|fut|all", "코스피 매일 숏 기준선"), row("S|KOSPI|fut|0", "코스피 z<1∧mis>0 09:02 숏"), row("Sd|KOSPI|fut|all", "진단 코스피 매일 선물 숏"), row("Sd|KOSPI|fut|0", "진단 코스피 선물 시가 숏"),
          row("S|KOSDAQ|fut|all", "코스닥 매일 숏 기준선"), row("S|KOSDAQ|fut|0", "코스닥 z<1∧mis>0 09:02 숏"), row("Sd|KOSDAQ|fut|all", "진단 코스닥 매일 선물 숏"), row("Sd|KOSDAQ|fut|0", "진단 코스닥 선물 시가 숏"),
          row("S|KOSPI|114800|0", "코스피 114800 실경로"), row("S|KOSDAQ|251340|0", "코스닥 251340 실경로"), "",
          "• 분봉 셀 경로는 1배 ETF 부호 반전, 비용만 선물(선물 분봉 2025 이후뿐, 봉인). 15:45 아닌 15:30 청산 경로.",
          "• 진단 셀은 첫 1분 하락이 포함돼 있고 체결은 신호와 동시라 실행 불가. 2024-08-05 하루 비중이 크다(7절).",
          "• 인버스 ETF 실경로는 1틱 비용에 사라진다. 선물 집행 전제.",
          f"• 지수 시가 시차 점검(분봉 창 512일, y = 09:02 시가→15:30, 계수는 mis 1bp 당 bp): 코스피 mis 단독 {v(lc.loc['KOSPI', 'mis 단독 계수'], '{:+.3f}')}(t {v(lc.loc['KOSPI', 't'])}), 첫 1분 통제 뒤 {v(lc.loc['KOSPI', '첫 1분 통제 계수'], '{:+.3f}')}(t {v(lc.loc['KOSPI', '통제 t'])}), 09:06 진입 {v(lc.loc['KOSPI', '09:06 진입 계수'], '{:+.3f}')}(t {v(lc.loc['KOSPI', '09:06 t'])}). 코스닥 단독 {v(lc.loc['KOSDAQ', 'mis 단독 계수'], '{:+.3f}')}(t {v(lc.loc['KOSDAQ', 't'])}), 통제 뒤 {v(lc.loc['KOSDAQ', '첫 1분 통제 계수'], '{:+.3f}')}(t {v(lc.loc['KOSDAQ', '통제 t'])}).",
          f"• 2024-08-05 제외(선물 비용 1.8bp): 코스피 순 {v(lc.loc['KOSPI', '2024-08-05 제외 bp'])}bp(t {v(lc.loc['KOSPI', '제외 t'])}), 코스닥 순 {v(lc.loc['KOSDAQ', '2024-08-05 제외 bp'])}bp(t {v(lc.loc['KOSDAQ', '제외 t'])}). 코스닥은 z<1 ∧ mis>0 묶음 평균만 음수이고 연속 관계가 없어 후보에서 뺐다.",
          f"• 긴 창 크기 추정(진단 셀 순수익 − 분봉 창 첫 1분 몫): 코스피 {v(g.loc['Sd|KOSPI|fut|0', 'net_bp'] + float(PTx.loc[('069500', 'z<1∧mis>0'), '09:02']))}bp, 코스닥 {v(g.loc['Sd|KOSDAQ|fut|0', 'net_bp'] + float(PTx.loc[('229200', 'z<1∧mis>0'), '09:02']))}bp. 설계 분봉 창 값보다 작게 볼 근거.",
          f"• 장부에 명목 1 로 얹으면(W_des, 실행판 혼합 기준, 증거금 약 21% 별도, 분봉 창 경로라 긴 창 크기보다 큼): 코스피 숏만 {v(bval('실행판 혼합 + 코스피 낮 숏만')['CAGR %'])}%/{v(bval('실행판 혼합 + 코스피 낮 숏만')['MDD %'])}%, 두 시장 {v(bval('실행판 혼합 + 낮 숏(mis>0) 명목 1')['CAGR %'])}%/{v(bval('실행판 혼합 + 낮 숏(mis>0) 명목 1')['MDD %'])}%, 얹기 전 {v(bh['CAGR %'])}%/{v(bh['MDD %'])}%.",
          "• 함정 기록: C 다리 상태로 숏을 거르면 순 +84bp(코스닥 C 숏, 09:32 진입) 같은 값이 나오지만 C 다리는 09:30 뒤 어느 시각에든 진입하므로(entry_cutoff 없음) 미래 정보다. Sx 셀로 격리했다.", "",
          "## 10. 실행 가능성 점검표", "",
          "1. 신호 확정 < 체결: E1·E2·z_f 전부 충족. 지정가 셀은 지정가를 09:00 전에 확정하고 체결만 단일가에 걸린다. 첫 봉 안 체결·C 상태 조건 셀은 exec_ok=False.",
          "2. 분봉 라벨: 7종 전부 첫 라벨 09:01·끝 라벨 15:30·하루 390봉(봉 끝 라벨 assert). 09:01 봉 거래량 0 인 4일(2023-01-02, 2023-11-16, 2024-01-02, 2024-11-14) 제외. KOSPI 지수 분봉은 첫 라벨이 09:02 라 체결 판단에 쓰지 않았다.",
          f"3. 호가: 연속 체결 1틱 반영. 1틱 bp(그해 평균 원가격): " + "; ".join(f"{c} {R['ticks'].loc[c, 2023]}/{R['ticks'].loc[c, 2024]}" for c in LEGS) + " (2023/2024).",
          "4. 비용: 머리말 기준. E2b 실패일은 왕복 5bp + 1틱, 통과일은 5bp.",
          f"5. 선물: z_f 는 근월물 같은 월물 갭, 2023-07-31 전 셀은 exec_ok=False 또는 09:02 진입. post 창 {int(R['fq'].loc['post 일수', 'KOSPI'])}일 z_f 결측 0, 코스피 KIS 근월 시가와 깃 FUT.csv 시가 {R['n_fut'][0]}일 중 {R['n_fut'][1]}일 일치. 코스닥150 선물 초기(2016~2017)는 진단 셀에만 포함.",
          "6. 일봉 수정가·분봉 원가격: 수익 계산에서 섞지 않음. 예외 하나, E1 z1 신호는 수정가 갭 + 분봉 1분 수익을 신호에만 씀.",
          "7. 자본 회전: 아침 다리는 ETF 현금 100%, A 다리 09:00 매도 대금 재사용(13차 규약). E2b 실패일은 09:02 뒤 현금. 선물 증거금 없음. 코스피 평균 명목은 7절.",
          "8. 레버리지 ETF: 낮 보유만 사용(122630·233740 은 w > 1 몫). 밤 구조 손실 해당 없음. 2배 ETF 단일가 호가 품질은 미확인.",
          f"9. 표본: 7절. post 창은 공매도 금지 구간(2023-11-06~)과 대부분 겹쳐 금지 해제 뒤 성과는 미확인.", "",
          "## 11. 셀 수와 기각", "",
          f"• 돌린 셀 {len(C)}개: " + ", ".join(f"{k} {v_}" for k, v_ in C["id"].str.split("|").str[0].value_counts().items()) + ". 전부 cells.csv.",
          f"• 선별 7기준 전부 통과(실행 가능 셀): {', '.join(C[C['pass']]['id'])}. 실행판 후보는 상위 5건 기준 미달이고 장부 역할 때문에 올렸다.",
          f"• 기각: 지정가 단일가(F lim, L), EWY 필터 E2, 첫 1분 회복 필터(코스피 {v(g.loc['E1|KOSPI|rec미회복', 'net_bp'])}bp, 코스닥 {v(g.loc['E1|KOSDAQ|rec미회복', 'net_bp'])}bp 로 부호 반대), z1 ETF 자체 잔차, 코스피 z_f 단독, z_f 하락 쪽 인버스, 낮 다리 mis 오버레이, 밤 다리 승계(겹침 코스피 {int(R['INH'].iloc[0]['C3 ∧ 밤 롱'])}일, 코스닥 롱 {int(R['INH'].iloc[1]['C3 ∧ 밤 롱'])}일), z 상한, 코스닥 mis 낮 숏.", "",
          "## 12. 미확인과 다음", "",
          "• 08:59 예상체결가·예상지수: 과거 자료 없음. 예상가 경로 수치는 5절 민감도뿐.",
          "• 코스닥 z_f 를 선물로 08:46 에 집행하는 경로: 선물 분봉이 2025 이후뿐이라 미검정.",
          "• 단일가 시장가 체결 수량·충격, 지수 시가 공표 시각(09:01:00 전 확정 가정), 09:00~09:05 LP 공백 때 실제 호가 폭: 미확인.",
          "• z_f 사전 신호의 실행 표본은 2023-07-31~ 뿐이라 긴 창 확인이 불가능. 봉인 평가 때 코스닥 z_f 와 E2b 를 함께 등록할지 판단 필요.",
          "• 그림: first_60min_path.png, equity_post.png."]
    (ROOT / "scan_morning_exec.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    N = ["# morning_exec 자료 메모", "",
         f"• C3 재현: 코스피 {R['repro']['k_days']}일 중 불일치 0, 활성 {R['repro']['k_on']}일(net5_chosen 최대 차 < 1e−8). 코스닥 {R['repro']['q_days']}일, 활성 {R['repro']['q_on']}일(net5_KODEX_vt30 최대 차 < 1e−8). 비교 파일도 봉인 뒤 읽음.",
         "• 코스닥150 지수: 깃 KOSDAQ_150_ohlcv.csv 는 2024-10-08 행이 중복. 중복 제거 뒤 KIS 지수 파일과 2,328일 시가·종가 일치.",
         "• TOSS 1분봉: 봉 끝 라벨, 7종 516일. 09:01 봉 거래량 0 인 4일 제외 512일. 일봉 시가→종가 대조 차이 최대 2.9bp(수정가 반올림).",
         "• 라벨 L 진입가 = L 이후 첫 체결 봉 시가. 15:21~15:29 봉은 쓰지 않음.",
         "• 선물: KIS 일봉 front=1 근월(만기 당일 차근월). 갭·중앙값은 같은 월물 안에서 계산.",
         "• EWY: yfinance Adj Close, 미국 날짜 < T 인 마지막 세션. L(T) = ln EWY − ln 지수 종가(T−1), 잔차는 직전 20일 중앙값 차감. 환율판은 ln USDKRW 를 더함.",
         "• 전일 수급·프로그램: T−1 값을 z60(T−1 까지 평균·표준편차)으로 표준화.",
         "• C 다리 시리즈는 2022-12-13~. 진입 시각이 09:30 뒤 가변이라 C 상태 조건 셀은 미래 정보.",
         "• 자체 점검 assert: 혼합 비중, 전부 통과일 때 E2 = 단일가 경로, 전부 실패일 때 E2 = 첫 1분 − 비용, 분봉·일봉 경로 차 < 3bp.",
         "• 독립 재계산(스크립트 함수 미사용, 반복문 OLS, 원자료 직접, 임시 코드라 저장 안 함): 코스피 E2b n 71, 순 21.07bp, CAGR 11.35%(일치). 코스닥 z_f n 38, 순 73.31bp(원가격 일봉 사용, 수정가판 73.66bp).", ""]
    (OUT / "notes.md").write_text("\n".join(N), encoding="utf-8")


def selfcheck(K, A, B):
    """경로 함수 자체 점검: 혼합 비중, 전부 통과일 때 E2 = 단일가 경로, 전부 실패일 때 E2 = 첫 1분 + 비용."""
    w = pd.Series([0.0, 0.5, 1.0, 1.5, 2.0])
    a, b = mix(w)
    assert ((a + 2 * b) - w).abs().max() < 1e-12
    days = A["o"].index[A["o"][AUC].notna()]
    sig = (K["C3"] == 1).astype(float)
    w1, g1, n1, s1 = run_min(K, A, B, sig, ent=AUC)
    w2, g2, n2, s2, h = run_verify(K, A, B, sig, sig)
    assert (n1 - n2).abs().max() < 1e-12 and (s1 - s2).abs().max() < 1e-12 and (w1 - w2).abs().max() < 1e-12
    w3, g3, n3, s3, h3 = run_verify(K, A, B, sig, sig * 0)
    aa, bb = mix(w3)
    ch = aa * (A["o"][FST] / A["o"][AUC] - 1).reindex(days).fillna(0) + bb * (B["o"][FST] / B["o"][AUC] - 1).reindex(days).fillna(0)
    assert (ch - g3).abs().max() < 1e-12 and (g3 - n3 - (aa + bb) * 5e-4 - (aa * K["t1"].reindex(days) + bb * K["t2"].reindex(days))).abs().max() < 1e-12
    g_d = run_auc(K, sig)[1].reindex(days)                       # 분봉·일봉 경로 대조(분배 수정 차이만 허용)
    assert (g_d - g1).abs().max() < 3e-4


def main():
    global BASE
    OUT.mkdir(parents=True, exist_ok=True)
    K, Q = market("KOSPI"), market("KOSDAQ")
    assert K.index.max() <= SEAL and Q.index.max() <= SEAL
    R = {"repro": repro(K, Q)}
    print("재현:", R["repro"])
    add_ewy(K)
    Q["zewy"], Q["zewyx"], Q["ewy"] = K["zewy"].reindex(Q.index), K["zewyx"].reindex(Q.index), K["ewy"].reindex(Q.index)
    add_futures(K, f"{EXT}/r15_external/kis_fut_k200_daily.csv")
    add_futures(Q, f"{EXT}/r15_external/kis_fut_kq150_daily.csv")
    M = {c: minute(c) for c in LEGS}
    DS = {"KOSPI": K, "KOSDAQ": Q}
    selfcheck(K, M["069500"], M["122630"])

    # 기준 흐름(봉인 안)
    nav = pd.read_csv(f"{GIT}/jehyeong/daynight/results/daily_nav.csv", encoding="utf-8-sig")
    nav = seal(nav)
    nav = nav[nav["version"] == "V2"].assign(date=lambda x: pd.to_datetime(x["date"])).pivot(index="date", columns="account", values="daily_return")
    bk = daily(f"{BT}/round13_slot_book/book/book_daily.csv")
    BASE = pd.concat([nav["overnight"].rename("ov"), nav["intraday"].rename("id"), bk["KOSPI_book"].rename("kp")], axis=1).dropna()

    # ---- 1. 경로·회복 시각·분해
    pt, ic, dc = [], [], []
    for mk, D in DS.items():
        p, i = path_tables(D, M, mk)
        pt.append(p)
        ic.append(i)
        dc.append(decomp(D, M, mk))
    PT, IC, DC = pd.concat(pt), pd.concat(ic), pd.DataFrame(dc)
    PT.to_csv(OUT / "path_by_group.csv", index=False, encoding="utf-8-sig")
    IC.to_csv(OUT / "recovery_increment.csv", index=False, encoding="utf-8-sig")
    DC.to_csv(OUT / "decomposition.csv", index=False, encoding="utf-8-sig")
    print(PT.to_string(index=False))
    print(IC.to_string(index=False))
    print(DC.T.to_string())
    # 일봉 2x2: mis·z 조합별 ETF·지수·선물 시가→종가(선물은 2023-07-28 까지 09:00 시가, 이후 08:45 시가)
    wins = {"KOSPI": [("2010-02-22", "2014-07-07"), ("2014-07-08", "2019-12-30"), ("2020-01-02", "2023-07-28"), ("2023-07-31", "2024-12-30")],
            "KOSDAQ": [("2016-05-09", "2019-12-30"), ("2020-01-02", "2023-07-28"), ("2023-07-31", "2024-12-30")]}
    t22 = []
    for mk, D in DS.items():
        for a0, b0 in wins[mk]:
            x = D.loc[a0:b0].dropna(subset=["z", "mis", "r1"])
            for g, m in (("z≥1∧mis≤0", (x["z"] >= 1) & (x["mis"] <= 0)), ("z≥1∧mis>0", (x["z"] >= 1) & (x["mis"] > 0)), ("z<1∧mis≤0", (x["z"] < 1) & (x["mis"] <= 0)), ("z<1∧mis>0", (x["z"] < 1) & (x["mis"] > 0))):
                y = x[m]
                t22.append({"market": mk, "창": f"{a0}~{b0}", "그룹": g, "n": len(y), "ETF bp": round(y["r1"].mean() * 1e4, 2), "ETF t": round(nw_t(y["r1"]), 2), "지수 bp": round(y["ri"].mean() * 1e4, 2),
                            "선물 bp": round(y["rf"].mean() * 1e4, 2), "선물 t": round(nw_t(y["rf"]), 2)})
    T22 = pd.DataFrame(t22)
    T22.to_csv(OUT / "mis_z_2x2.csv", index=False, encoding="utf-8-sig")
    print(T22.to_string(index=False))
    fq = pd.DataFrame({mk: {"post 일수": len(D.loc[FUT_0845:]), "z_f 결측": int(D.loc[FUT_0845:, "zf"].isna().sum()), "z_f·z 상관(post)": round(D.loc[FUT_0845:, "zf"].corr(D.loc[FUT_0845:, "z"]), 3),
                            "z_f·z 상관(그 전)": round(D.loc[:"2023-07-28", "zf"].corr(D.loc[:"2023-07-28", "z"]), 3)} for mk, D in DS.items()})
    ff = daily(f"{GIT}/beomgu/data/etf/FUT.csv")
    jn = pd.DataFrame({"kis": K["fo"], "git": ff["open"].reindex(K.index)}).loc[FUT_0845:].dropna()
    print(fq.to_string())
    print("KIS 근월 시가 vs 깃 FUT.csv 시가(post):", len(jn), "일 중 일치", int(((jn["kis"] - jn["git"]).abs() < 1e-6).sum()))

    # ---- 2. 기준 셀(원형, 실행 불가)
    WIN = {}
    for mk, D in DS.items():
        c1, c2 = MKT[mk]
        L = D.loc[START[mk]:].index
        mdays = M[c1]["o"].index[M[c1]["o"][AUC].notna()]
        post = mdays[mdays >= FUT_0845]
        WIN[mk] = {"long": L, "min": mdays, "post": post}
        w, g, n, s = run_auc(D, D["C3"])
        cell(f"R|{mk}|long", f"C3 원형 VT30 09:00 동일가 {mk}", mk, L, w, g, n, s, False, "기준, 동일가 위반")
        w1 = D["C3"].fillna(0)
        w1g, g1, n1, s1 = run_auc(D, D["C3"], w=w1)
        cell(f"R|{mk}|long|1x", f"C3 원형 1x {mk}", mk, L, w1g, g1, n1, s1, False, "기준, 동일가 위반")
        for wn in ("min", "post"):
            cal = WIN[mk][wn]
            w, g, n, s = run_min(D, M[c1], M[c2], D["C3"], ent=AUC)
            cell(f"R|{mk}|{wn}", f"C3 원형 VT30 분봉 {wn} {mk}", mk, cal, w, g, n, s, False, "기준, 동일가 위반. 분봉 원가격")

    # ---- 3. E1 사후 진입
    for mk, D in DS.items():
        c1, c2 = MKT[mk]
        cal = WIN[mk]["min"]
        A, B = M[c1], M[c2]
        C3 = D["C3"]
        for k, sig in (("C3", C3), ("C2", D["C2"])):
            w, g, n, s = run_min(D, A, B, sig)
            cell(f"E1|{mk}|{k}", f"{k} → 09:02 봉 시가 → 15:30", mk, cal, w, g, n, s, True, "09:00 단일가 확정 신호 → 다음 분봉 시가", fam=f"E1|{mk}|adj")
        for zk in (0.75, 1.25):
            sig = ((D["z"] >= zk) & (D["mis"] <= 0)).astype(float)
            w, g, n, s = run_min(D, A, B, sig)
            cell(f"E1|{mk}|z{zk}", f"z≥{zk}∧mis≤0 → 09:02 시가", mk, cal, w, g, n, s, True, "인접 z 문턱", fam=f"E1|{mk}|adj")
        for mk_bp in (-10, 10):
            sig = ((D["z"] >= 1) & (D["mis"] <= mk_bp / 1e4)).astype(float)
            w, g, n, s = run_min(D, A, B, sig)
            cell(f"E1|{mk}|mis{mk_bp}", f"z≥1∧mis≤{mk_bp}bp → 09:02 시가", mk, cal, w, g, n, s, True, "인접 mis 문턱", fam=f"E1|{mk}|adj")
        for ent in ("09:03", "09:04", "09:06", "09:11", "09:16", "09:31"):
            w, g, n, s = run_min(D, A, B, C3, ent=ent)
            cell(f"E1|{mk}|ent{ent}", f"C3 → {ent} 봉 시가 → 15:30", mk, cal, w, g, n, s, True, "진입 지연", fam=f"E1|{mk}|ent")
        for ex in ("09:31", "10:01", "10:31", "11:01", "13:01", "15:20c"):
            w, g, n, s = run_min(D, A, B, C3, ex=ex)
            cell(f"E1|{mk}|ex{ex}", f"C3 → 09:02 시가 → {ex} 청산", mk, cal, w, g, n, s, True, "청산 변경", fam=f"E1|{mk}|ex")
            if ex == "10:01":
                cell(f"E1p|{mk}|ex{ex}", f"C3 → 09:02 시가 → {ex} 청산(2023-07-31~)", mk, WIN[mk]["post"], w, g, n, s, True, "청산 변경, post 창")
        w, g, n, s = run_min(D, A, B, C3)
        cell(f"E1p|{mk}|C3", "C3 → 09:02 봉 시가 → 15:30 (2023-07-31~)", mk, WIN[mk]["post"], w, g, n, s, True, "E1 을 post 창에서")
        fg = w * (A["c"]["15:30"] / A["o"][FST] - 1).reindex(cal).fillna(0)
        cell(f"E1f|{mk}|C3", "C3 → 09:02 선물 진입 대용(1배 ETF 경로 × w, 선물 비용)", mk, cal, w, fg, fg - w * 1.8e-4, fg - w * 3e-4, True, "선물 분봉 없음: 경로는 1배 ETF, 비용만 선물. 15:45 아닌 15:30 청산 경로")
        rec = np.log(A["c"]["09:01"] / A["o"][AUC]).reindex(D.index)
        for nm, m in (("미회복", rec <= 10e-4), ("회복", rec > 10e-4)):
            w, g, n, s = run_min(D, A, B, C3.where(m, 0.0))
            cell(f"E1|{mk}|rec{nm}", f"C3 ∧ 첫 1분 상승 {'≤' if nm == '미회복' else '>'} 10bp → 09:02 시가", mk, cal, w, g, n, s, True, "09:01 봉 종가 확정", fam=f"E1|{mk}|rec")
        # z1: ETF 자체 갭 + 첫 1분(지수 시가 불필요), 회귀 계수는 지수 갭 z 의 것을 그대로 씀
        z1 = (D["ge"] + rec - D["fa"] - D["fb"] * D["us"]) / D["fs"]
        for zk in (0.75, 1.0, 1.25):
            w, g, n, s = run_min(D, A, B, (z1 >= zk).astype(float))
            cell(f"E1|{mk}|z1_{zk}", f"ETF 09:01 봉 종가 갭 잔차 z1≥{zk} → 09:02 시가", mk, cal, w, g, n, s, True, "지수 시가 없이 ETF 자체 가격만, 일봉 수정가 갭 + 분봉 1분 수익(신호에만 혼합)", fam=f"E1|{mk}|z1")
        # 진단: 첫 봉 안 체결(규약 위반, 회복 속도 가늠)
        days = cal
        mid1, mid2 = (A["o"][AUC] + A["c"]["09:01"]) / 2, (B["o"][AUC] + B["c"]["09:01"]) / 2
        w = wt(C3.reindex(days), D["sig"].reindex(days))
        a, b = mix(w)
        g = a * (A["c"]["15:30"] / mid1 - 1).reindex(days).fillna(0) + b * (B["c"]["15:30"] / mid2 - 1).reindex(days).fillna(0)
        t = a * D["t1"].reindex(days) + b * D["t2"].reindex(days)
        cell(f"E1|{mk}|mid", "C3 → 09:01 봉 (시가+종가)/2 → 15:30", mk, cal, w, g, g - (a + b) * 5e-4 - t, g - (a + b) * 7e-4 - 2 * t, False, "진단: 첫 봉 안 체결 가정, 규약 2 위반")

    # ---- 4. E5 사전 신호 → 09:00 단일가 시장가(일봉 장기창, 깃 자료)
    PRE = {mk: add_pre(D, mk) for mk, D in DS.items()}
    for mk, D in DS.items():
        L = WIN[mk]["long"]
        for nm, x in PRE[mk].items():
            for side, tag in ((1, "상위"), (-1, "하위")):
                for k in (0.75, 1.0, 1.25):
                    sig = (x * side >= k).astype(float).where(x.notna())
                    w, g, n, s = run_auc(D, sig)
                    cell(f"E5|{mk}|{nm}|{tag}|{k}", f"{nm} z60 {tag} {k} → 단일가 매수", mk, L, w, g, n, s, True, "전일까지 확정 또는 미국 마감. 깃 자료", fam=f"E5|{mk}|{nm}|{tag}")
        for nm in ("zewy", "zewyx"):
            for k in (0.75, 1.0, 1.25):
                sig = (D[nm] >= k).astype(float).where(D[nm].notna())
                w, g, n, s = run_auc(D, sig)
                cell(f"E5|{mk}|{nm}|{k}", f"EWY 잔차 {nm}≥{k} → 단일가 매수", mk, L, w, g, n, s, True, "깃 밖 자료 yf_ewy" + ("·fdr_usdkrw" if nm == "zewyx" else ""), fam=f"E5|{mk}|{nm}")

    # ---- 5. 선물 08:45 경로(2023-07-31~) + 장기 진단
    tmis = {}
    for mk, D in DS.items():
        c1, c2 = MKT[mk]
        A, B = M[c1], M[c2]
        post, mdays = WIN[mk]["post"], WIN[mk]["min"]
        pre = D.loc["2016-06-01" if mk == "KOSDAQ" else "2014-07-08":"2023-07-28"].index
        tb = D["t1"]
        for k in (0.5, 1.0, 1.5):
            F = (D["zf"] >= k).astype(float).where(D["zf"].notna())
            w, g, n, s = run_auc(D, F)
            cell(f"F|{mk}|moo|{k}", f"z_f≥{k}(08:45 선물 갭) → 단일가 시장가", mk, post, w, g, n, s, True, "깃 밖 자료 kis_fut", fam=f"F|{mk}|moo")
            cell(f"Fd|{mk}|moo|{k}", f"진단 z_f≥{k}(09:00 선물 시가) → 단일가", mk, pre, w, g, n, s, False, "진단: 선물 시가가 ETF 시가와 동시, 사전 신호 아님")
            for dl in (0, 5, 10):
                lim = (F == 1) & (D["misf"] <= -(dl / 1e4) - tb)
                w, g, n, s = run_auc(D, lim.astype(float))
                cell(f"F|{mk}|lim|{k}|{dl}", f"z_f≥{k} ∧ 지정가(선물 환산 −{dl}bp−1틱) 단일가", mk, post, w, g, n, s, True, "E4. 지정가는 09:00 전 확정, 단일가 < L 체결. 깃 밖 자료", fam=f"F|{mk}|lim")
                cell(f"Fd|{mk}|lim|{k}|{dl}", f"진단 z_f≥{k} ∧ 지정가 −{dl}bp", mk, pre, w, g, n, s, False, "진단: 선물 시가 동시")
                w, g, n, s, _ = run_verify(D, A, B, lim.astype(float), D["C3"])
                cell(f"F|{mk}|limv|{k}|{dl}", f"z_f≥{k} ∧ 지정가 −{dl}bp + 09:00 뒤 C3 확인", mk, post, w, g, n, s, True, "E4+E2", fam=f"F|{mk}|limv")
            w, g, n, s, h = run_verify(D, A, B, F, D["C3"])
            cell(f"F|{mk}|ver|{k}", f"z_f≥{k} 단일가 매수 + C3 확인, 실패 09:02 청산", mk, post, w, g, n, s, True, "E2. 깃 밖 자료", fam=f"F|{mk}|ver")
            w, g, n, s, h = run_verify(D, A, B, F, D["C3"], back=D["C3"])
            cell(f"F|{mk}|verb|{k}", f"z_f≥{k} 단일가 매수 + C3 확인, F 밖 C3 는 09:02 진입", mk, post, w, g, n, s, True, "E2+E1", fam=f"F|{mk}|verb")
            v = (F == 1).reindex(post).fillna(False)
            c3 = (D["C3"] == 1).reindex(post)
            tmis[(mk, f"z_f≥{k}")] = dict(창="post", 필터일=int(v.sum()), C3일=int(c3.sum()), 겹침=int((v & c3).sum()), 재현율=round((v & c3).sum() / max(c3.sum(), 1) * 100, 1), 정밀도=round((v & c3).sum() / max(v.sum(), 1) * 100, 1))
        # E2 EWY 필터(분봉 창 전체)
        for nm, F0 in (("zewy", D["zewy"]), ("zewyx", D["zewyx"])):
            for k in (0.0, 0.5, 1.0):
                F = (F0 >= k).astype(float).where(F0.notna())
                w, g, n, s, h = run_verify(D, A, B, F, D["C3"])
                cell(f"V|{mk}|{nm}|{k}", f"{nm}≥{k} 단일가 매수 + C3 확인", mk, mdays, w, g, n, s, True, "E2. 깃 밖 자료 yf_ewy", fam=f"V|{mk}|{nm}")
                w, g, n, s, h = run_verify(D, A, B, F, D["C3"], back=D["C3"])
                cell(f"V|{mk}|{nm}|{k}|b", f"{nm}≥{k} 단일가 매수 + C3 확인, F 밖 C3 는 09:02 진입", mk, mdays, w, g, n, s, True, "E2+E1", fam=f"V|{mk}|{nm}|b")
                v = (F == 1).reindex(mdays).fillna(False)
                c3 = (D["C3"] == 1).reindex(mdays)
                tmis[(mk, f"{nm}≥{k}")] = dict(창="min", 필터일=int(v.sum()), C3일=int(c3.sum()), 겹침=int((v & c3).sum()), 재현율=round((v & c3).sum() / max(c3.sum(), 1) * 100, 1), 정밀도=round((v & c3).sum() / max(v.sum(), 1) * 100, 1))
        F = (D["us"] > 0).astype(float).where(D["us"].notna())
        w, g, n, s, h = run_verify(D, A, B, F, D["C3"])
        cell(f"V|{mk}|spx0", "SPX 상승 다음날 단일가 매수 + C3 확인", mk, mdays, w, g, n, s, True, "E2. 깃 자료")
        # z_f 인접 문턱·진입·청산·추정창·원시 갭
        for k in (0.75, 1.25):
            F = (D["zf"] >= k).astype(float).where(D["zf"].notna())
            w, g, n, s = run_auc(D, F)
            cell(f"F|{mk}|moo|{k}", f"z_f≥{k}(08:45 선물 갭) → 단일가 시장가", mk, post, w, g, n, s, True, "깃 밖 자료 kis_fut", fam=f"F|{mk}|moo")
            w, g, n, s, h = run_verify(D, A, B, F, D["C3"], back=D["C3"])
            cell(f"F|{mk}|verb|{k}", f"z_f≥{k} 단일가 매수 + C3 확인, F 밖 C3 는 09:02 진입", mk, post, w, g, n, s, True, "E2+E1", fam=f"F|{mk}|verb")
        for k in (0.5, 0.75, 1.0, 1.25, 1.5):
            F = (D["zf"] >= k).astype(float).where(D["zf"].notna())
            w, g, n, s = run_min(D, A, B, F)
            cell(f"F|{mk}|e1|{k}", f"z_f≥{k} → 09:02 봉 시가 → 15:30", mk, post, w, g, n, s, True, "단일가 대신 09:02 진입", fam=f"F|{mk}|e1")
            cell(f"F|{mk}|e1all|{k}", f"z_f≥{k} → 09:02 봉 시가 → 15:30 (분봉 창 전체)", mk, mdays, w, g, n, s, True, "2023-07-28 까지는 09:00 선물 시가 신호, 그 뒤 08:45. 어느 쪽이든 09:02 진입 전에 확정", fam=f"F|{mk}|e1all")
            hm = (D["mis"] <= 0).astype(float).where(D["mis"].notna())
            w, g, n, s, h = run_verify(D, A, B, F, hm, back=D["C3"])
            cell(f"F|{mk}|vermb|{k}", f"z_f≥{k} 단일가 매수 + mis≤0 확인, F 밖 C3 는 09:02 진입", mk, post, w, g, n, s, True, "확인을 z 없이 mis 만", fam=f"F|{mk}|vermb")
        F1 = (D["zf"] >= 1).astype(float).where(D["zf"].notna())
        for ex in ("10:01", "11:01", "13:01", "15:20c"):
            w, g, n, s = run_min(D, A, B, F1, ent=AUC, ex=ex)
            cell(f"F|{mk}|moo1|ex{ex}", f"z_f≥1 단일가 매수 → {ex} 청산", mk, post, w, g, n, s, True, "단일가 진입, 장중 청산", fam=f"F|{mk}|moo1ex")
        w1 = F1.fillna(0.0)
        cell(f"F|{mk}|moo1|1x", "z_f≥1 단일가 매수 1배만(w=1, 레버리지 없음)", mk, post, w1, w1 * D["r1"].fillna(0), w1 * D["r1"].fillna(0) - w1 * 5e-4, w1 * D["r1"].fillna(0) - w1 * (7e-4 + D["t1"]), True, "차량 점검")
        for wn, mr in ((120, 100), (500, 400)):
            zv = rolling_z1(D["gf"], D["us"], wn, mr)
            w, g, n, s = run_auc(D, (zv >= 1).astype(float).where(zv.notna()))
            cell(f"F|{mk}|moo1|win{wn}", f"z_f≥1 추정창 {wn}행 → 단일가 시장가", mk, post, w, g, n, s, True, "추정창 인접", fam=f"F|{mk}|moo1win")
        for x in (50, 100, 150):
            w, g, n, s = run_auc(D, (D["gf"] >= x / 1e4).astype(float).where(D["gf"].notna()))
            cell(f"F|{mk}|raw|{x}", f"선물 원시 갭 ≥ {x}bp → 단일가 시장가", mk, post, w, g, n, s, True, "미국 잔차 없이 원시 갭(기각 축 갭 지속 대조)", fam=f"F|{mk}|raw")
        for code in INV[mk][:1]:
            for k in (1.0, 1.5):
                Fi = (D["zf"] <= -k).astype(float).reindex(post).fillna(0.0)
                Mv = M[code]
                ri = (Mv["c"]["15:30"] / Mv["o"][AUC] - 1).reindex(post).fillna(0)
                ti = tick_of(code, post)
                cell(f"F|{mk}|inv|{k}", f"z_f≤−{k} → {code} 단일가 매수", mk, post, Fi, Fi * ri, Fi * (ri - 5e-4), Fi * (ri - 7e-4 - ti), True, "하락 쪽 대칭, 1배 인버스", fam=f"F|{mk}|inv")
        # EWY 지정가(장기): ETF 갭이 EWY 내재 갭보다 낮을 때만 체결
        ff = rolling_fit(D["g"], D["ewy"])
        ghat = ff["a"] + ff["b"] * D["ewy"]
        for dl in (0, 30, 60):
            lim = (D["ge"] <= ghat - dl / 1e4 - tb).astype(float).where(ghat.notna())
            w, g, n, s = run_auc(D, lim)
            cell(f"L|{mk}|ewy|{dl}", f"지정가 = EWY 내재 갭 −{dl}bp−1틱 단일가", mk, WIN[mk]["long"], w, g, n, s, True, "E4 장기. 깃 밖 자료 yf_ewy", fam=f"L|{mk}|ewy")
            for k in (0.5, 1.0):
                lim2 = ((D["zewy"] >= k) & (D["ge"] <= ghat - dl / 1e4 - tb)).astype(float).where(ghat.notna())
                w, g, n, s = run_auc(D, lim2)
                cell(f"L|{mk}|zewy{k}|{dl}", f"zewy≥{k} ∧ 지정가 EWY 내재 −{dl}bp", mk, WIN[mk]["long"], w, g, n, s, True, "E4 장기. 깃 밖 자료", fam=f"L|{mk}|zewy{k}")
    TM = pd.DataFrame(tmis).T
    TM.index.names = ["market", "filter"]
    TM.to_csv(OUT / "prefilter_recall.csv", encoding="utf-8-sig")
    print(TM.to_string())

    # ---- 6. mis 양수 낮 숏(부산물) + 선물 시가 진단
    for mk, D in DS.items():
        c1, c2 = MKT[mk]
        A = M[c1]
        cal = WIN[mk]["min"]
        for dl in (0, 10, 20):
            sig = ((D["mis"] > dl / 1e4) & (D["z"] < 1)).astype(float)
            s_, g, n, s = run_short(D, A, sig, "fut")
            cell(f"S|{mk}|fut|{dl}", f"z<1∧mis>{dl}bp → 09:02 숏(선물 비용, ETF 경로 대용)", mk, cal, s_, g, n, s, True, "선물 분봉 없음. 경로는 1배 ETF 부호 반전", fam=f"S|{mk}|fut")
            for code in INV[mk]:
                s_, g, n, s = run_short(D, A, sig, "inv", Mv=M[code], code=code)
                cell(f"S|{mk}|{code}|{dl}", f"z<1∧mis>{dl}bp → {code} 09:02 시가 매수", mk, cal, s_, g, n, s, True, "인버스 ETF 실경로, 1틱 반영", fam=f"S|{mk}|{code}")
            d = D.loc["2016-06-01" if mk == "KOSDAQ" else "2014-07-08":"2023-07-28"]
            act = ((d["mis"] > dl / 1e4) & (d["z"] < 1)).astype(float).where(d["rf"].notna(), 0.0)
            g = -act * d["rf"].fillna(0)
            cell(f"Sd|{mk}|fut|{dl}", f"진단 z<1∧mis>{dl}bp → 선물 시가 숏 → 종가", mk, d.index, act, g, g - act * 1.8e-4, g - act * 3e-4, False, "진단: 신호와 선물 시가 동시, 2023-07-28 까지")

        leg = daily(f"{BT}/round13_slot_book/series/{'jehyeong_KP' if mk == 'KOSPI' else 'jehyeong_KQ'}.csv")["pos"].reindex(cal).fillna(0)
        sig0 = ((D["mis"] > 0) & (D["z"] < 1)).astype(float)
        for tag, m in (("C없음", leg == 0), ("C롱", leg > 0), ("C숏", leg < 0)):
            s_, g, n, s = run_short(D, A, sig0.reindex(cal).where(m, 0.0), "fut")
            cell(f"Sx|{mk}|fut|0|{tag}", f"z<1∧mis>0 09:02 숏, C 다리 {tag} 날", mk, cal, s_, g, n, s, False, "미래 정보: C 상태는 09:31 확정인데 09:02 진입에 조건을 걸었다. 겹침 크기 진단 전용")
            s_, g, n, s = run_short(D, A, sig0.reindex(cal).where(m, 0.0), "fut", ent="09:32")
            cell(f"Sx|{mk}|fut|0|{tag}|0932", f"z<1∧mis>0, C 다리 {tag} 날 09:32 숏", mk, cal, s_, g, n, s, False, "미래 정보: C 다리는 entry_cutoff 없이 09:30 뒤 어느 시각에든 진입하므로 09:32 에도 C 상태 미확정. 진단 전용")
        al = pd.Series(1.0, index=cal)
        s_, g, n, s = run_short(D, A, al, "fut")
        cell(f"S|{mk}|fut|all", "매일 09:02 숏(선물 비용, 기준선)", mk, cal, s_, g, n, s, True, "기준선: 조건 없는 낮 숏")
        d = D.loc["2016-06-01" if mk == "KOSDAQ" else "2014-07-08":"2023-07-28"]
        act = d["rf"].notna().astype(float)
        g = -act * d["rf"].fillna(0)
        cell(f"Sd|{mk}|fut|all", "진단 매일 선물 시가 숏 → 종가(기준선)", mk, d.index, act, g, g - act * 1.8e-4, g - act * 3e-4, False, "기준선: 2023-07-28 까지")

    LC = pd.DataFrame([lag_check(D, M[MKT[mk][0]], mk) for mk, D in DS.items()])
    LC.to_csv(OUT / "mis_lag_check.csv", index=False, encoding="utf-8-sig")
    print(LC.T.to_string())

    # ---- 6b. 낮 다리 오버레이: 09:00 에 확정된 mis 부호로 09:31 C 다리를 거른다(체결 시각 > 신호 시각)
    ovl = []
    for h, mk in (("jehyeong_KP", "KOSPI"), ("jehyeong_KQ", "KOSDAQ")):
        D = DS[mk]
        ser = daily(f"{BT}/round13_slot_book/series/{h}.csv")
        cal = ser.index.intersection(WIN[mk]["min"])
        for side, sm in (("롱", ser["pos"] > 0), ("숏", ser["pos"] < 0)):
            for tag, m in (("전체", pd.Series(True, index=ser.index)), ("mis≤0", D["mis"].reindex(ser.index) <= 0), ("mis>0", D["mis"].reindex(ser.index) > 0)):
                if not sm.any():
                    continue
                p = ser["pos"].where(sm & m, 0.0)
                cell(f"O|{h}|{side}|{tag}", f"{h} {side} 다리 {tag} 날만", mk, cal, p, ser["gross"], ser["net"], ser["net"] - 2e-4 * p.abs(), True, "13차 시리즈 그대로, 09:31 진입 전 mis 확정")
                x = ser["net"][sm & m].reindex(cal).dropna()
                ovl.append({"다리": h, "방향": side, "조건": tag, "n": len(x), "평균 bp": round(x.mean() * 1e4, 2), "t": round(nw_t(x), 2)})
    OV = pd.DataFrame(ovl)
    OV.to_csv(OUT / "overlay_day_leg.csv", index=False, encoding="utf-8-sig")
    print(OV.to_string(index=False))

    # 밤 다리 승계 경로 점검: 밤 롱을 09:00 에 팔지 않고 09:02 시가까지 미룰 때의 변화
    inh = []
    for h, mk in (("jehyeong_ON", "KOSPI"), ("blog_ON_Q", "KOSDAQ")):
        D = DS[mk]
        c = "069500" if mk == "KOSPI" else "233740"
        days = WIN[mk]["min"]
        ser = daily(f"{BT}/round13_slot_book/series/{h}.csv")["pos"].reindex(days).fillna(0)
        lg = (ser > 0).astype(float)
        d1 = (M[c]["o"][FST] / M[c]["o"][AUC] - 1).reindex(days).fillna(0) * lg
        tk = tick_of(c, days) * lg
        cell(f"A|{mk}|delay", f"{h} 밤 롱 청산을 09:00 단일가에서 09:02 시가로 미룸", mk, days, lg, d1, d1 - tk, d1 - 2 * tk, True, "밤 다리 수익 변화분만. 1틱 반영")
        c3 = (D["C3"] == 1).reindex(days).fillna(False)
        inh.append({"market": mk, "밤 다리": h, "차량": c, "밤 롱 일": int((ser > 0).sum()), "C3 일": int(c3.sum()), "C3 ∧ 밤 롱": int(((ser > 0) & c3).sum()), "C3 ∧ 밤 숏": int(((ser < 0) & c3).sum())})
    INH = pd.DataFrame(inh)
    INH.to_csv(OUT / "night_inherit.csv", index=False, encoding="utf-8-sig")
    print(INH.to_string(index=False))

    # ---- 6c. 경로별 잔존 분해(2023-07-31~, C3 날 기준 합계 bp)
    routes = []
    for mk, D in DS.items():
        c1, c2 = MKT[mk]
        A, B = M[c1], M[c2]
        post = WIN[mk]["post"]
        x = D.reindex(post)
        w = wt(x["C3"], x["sig"])
        a, b = mix(w)
        on = w > 0
        F = (x["zf"] >= 1)
        r_u1, r_u2 = (A["o"][FST] / A["o"][AUC] - 1).reindex(post), (B["o"][FST] / B["o"][AUC] - 1).reindex(post)
        r_t1, r_t2 = (A["c"]["15:30"] / A["o"][AUC] - 1).reindex(post), (B["c"]["15:30"] / B["o"][AUC] - 1).reindex(post)
        b1, b2 = r_u1.mean(), r_u2.mean()
        tot = (a * r_t1 + b * r_t2)[on]
        und = (a * (r_u1 - b1) + b * (r_u2 - b2))[on]
        dirn = tot - und
        wf, gf_, nf, sf, hf = run_verify(D, A, B, F.astype(float).where(F.index >= FUT_0845, 0.0), D["C3"], back=D["C3"])
        wf, gf_, nf = wf.reindex(post), gf_.reindex(post), nf.reindex(post)      # 보충 진입이 post 창 밖으로 새지 않게
        fail = (wf > 0) & (x["C3"] != 1)
        routes.append({"market": mk, "C3 일": int(on.sum()), "필터 안 C3": int((on & F).sum()), "필터 실패일": int(fail.sum()), "합계 bp": round(tot.sum() * 1e4), "저평가 몫 bp": round(und.sum() * 1e4), "방향 몫(평소 첫 1분 포함) bp": round(dirn.sum() * 1e4),
                       "E2b 저평가 잔존 bp": round(und[on & F].sum() * 1e4), "E2b 저평가 잔존 %": round(und[on & F].sum() / und.sum() * 100, 1) if und.sum() else np.nan, "실패일 총손익 bp": round(gf_[fail].sum() * 1e4),
                       "E2b 총 bp": round(gf_.sum() * 1e4), "E2b 순 bp": round(nf.sum() * 1e4)})
    RT = pd.DataFrame(routes)
    RT.to_csv(OUT / "route_retention.csv", index=False, encoding="utf-8-sig")
    print(RT.T.to_string())

    # ---- 7. 장기창 근사(첫 1분 차감) + VT 상한
    cap_rows = []
    u_hat = {}
    for mk, D in DS.items():
        c1, c2 = MKT[mk]
        days = WIN[mk]["min"]
        on = (D["C3"] == 1).reindex(days).fillna(False)
        u_hat[mk] = tuple(float((M[c]["o"].loc[days, FST] / M[c]["o"].loc[days, AUC] - 1)[on].mean()) for c in (c1, c2))
        L = WIN[mk]["long"]
        for cap in (1.0, 1.5, 2.0):
            w, g, n, s = run_auc(D, D["C3"], cap=cap)
            a, b = mix(w)
            ga = g - a * u_hat[mk][0] - b * u_hat[mk][1]
            t = a * D["t1"] + b * D["t2"]
            na, sa = ga - (a + b) * 5e-4 - t, ga - (a + b) * 7e-4 - 2 * t
            cell(f"C|{mk}|orig|{cap}", f"C3 원형 VT30 상한 {cap}", mk, L, w, g, n, s, False, "상한 비교. 동일가 위반", fam=f"C|{mk}|orig")
            cell(f"C|{mk}|approx|{cap}", f"E1 근사(첫 1분 평균 차감) 상한 {cap}", mk, L, w, ga, na, sa, False, "근사: 2022-11 전은 분봉 없음. 분봉 창 평균 회복을 전 구간에 차감", fam=f"C|{mk}|approx")
            for nm, dn in (("orig", n), ("approx", na)):
                x = dn.loc[L[0]:]
                eq = (1 + x).cumprod()
                w20 = dn.loc["2020-02-20":"2020-04-10"]
                cap_rows.append({"market": mk, "규칙": nm, "상한": cap, "평균 w": round(w[w > 0].loc[L[0]:].mean(), 3), "최대 w": round(w.loc[L[0]:].max(), 3), "CAGR %": round((eq.iloc[-1] ** (ANN / len(x)) - 1) * 100, 2),
                                 "MDD %": round((eq / eq.cummax() - 1).min() * 100, 2), "최악일 %": round(x.min() * 100, 2), "최악일": str(x.idxmin().date()),
                                 "2020-02-20~04-10 합 %": round(w20.sum() * 100, 2), "그 구간 최악일 %": round(w20.min() * 100, 2), "그날": str(w20.idxmin().date())})
    CAP = pd.DataFrame(cap_rows)
    CAP.to_csv(OUT / "vt_cap.csv", index=False, encoding="utf-8-sig")
    print(CAP.to_string(index=False))
    print("첫 1분 평균(수익, C3 날):", {k: tuple(round(x * 1e4, 2) for x in v) for k, v in u_hat.items()})
    for mk, D in DS.items():
        wd = pd.Timestamp("2020-03-19")
        print(mk, "2020-03-19 w", round(float(wt(D["C3"], D["sig"]).get(wd, np.nan)), 3), "σ20", round(float(D["sig"].get(wd, np.nan)), 3), "r1 %", round(float(D["r1"].get(wd, np.nan)) * 100, 2), "z", round(float(D["z"].get(wd, np.nan)), 2), "mis bp", round(float(D["mis"].get(wd, np.nan)) * 1e4, 1))
        sg = ((D["C3"] == 1) & (D["z"] < 4)).astype(float).where(D["C3"].notna())
        w, g, n, s = run_auc(D, sg)
        print(mk, "C3 중 z≥4 일수", int(((D["C3"] == 1) & (D["z"] >= 4)).loc[START[mk]:].sum()), "그날 평균 순 bp", round(run_auc(D, D["C3"])[2][(D["C3"] == 1) & (D["z"] >= 4)].loc[START[mk]:].mean() * 1e4, 1))
        cell(f"Cz|{mk}|z4", "C3 ∧ z<4 (극단 잔차 제외)", mk, WIN[mk]["long"], w, g, n, s, False, "진단: 최악일 z 가 극단이라 확인. 동일가 위반", fam=None)
    pg = PRE["KOSPI"]["prev_gap"]
    for tag, m in (("C3 제외", K["C3"] != 1), ("C3 날", K["C3"] == 1)):
        w, g, n, s = run_auc(K, ((pg >= 1.25) & m).astype(float).where(pg.notna()))
        cell(f"E5x|KOSPI|prev_gap|{tag}", f"prev_gap z60≥1.25, {tag}", "KOSPI", WIN["KOSPI"]["long"], w, g, n, s, True if tag == "C3 제외" else False, "C3 와 겹침 분리")
    # 사전 필터 + 확인의 장기 근사(zewyx): 실패일 손익은 분봉 창 평균, 필터 밖 C3 는 E1 근사
    for mk, D in DS.items():
        c1, c2 = MKT[mk]
        days = WIN[mk]["min"]
        L = WIN[mk]["long"]
        for k in (0.5, 1.0):
            F = (D["zewyx"] >= k) & D["zewyx"].notna()
            c3 = D["C3"] == 1
            mf = (F & ~c3).reindex(days).fillna(False)
            u1f, u2f = [float((M[c]["o"].loc[days, FST] / M[c]["o"].loc[days, AUC] - 1)[mf].mean()) for c in (c1, c2)]
            w = wt(F.astype(float), D["sig"]).where(D["r1"].notna() & D["r2"].notna(), 0.0)
            wb = wt((c3 & ~F).astype(float), D["sig"]).where(D["r1"].notna() & D["r2"].notna(), 0.0)
            a, b = mix(w)
            ab, bb = mix(wb)
            t, tb_ = a * D["t1"] + b * D["t2"], ab * D["t1"] + bb * D["t2"]
            ghold = (a * D["r1"].fillna(0) + b * D["r2"].fillna(0)).where(c3, a * u1f + b * u2f)
            gback = ab * D["r1"].fillna(0) + bb * D["r2"].fillna(0) - ab * u_hat[mk][0] - bb * u_hat[mk][1]
            g = ghold + gback
            n = g - (a + b + ab + bb) * 5e-4 - t.where(~c3, 0.0) - tb_
            s = g - (a + b + ab + bb) * 7e-4 - t.where(c3, 2 * t) - 2 * tb_
            cell(f"Vl|{mk}|zewyx|{k}", f"zewyx≥{k} 선매수 + C3 확인 장기 근사", mk, L, w + wb, g, n, s, False, "근사: 실패일 첫 1분은 분봉 창 평균, F 밖 C3 는 첫 1분 차감. 깃 밖 자료", fam=f"Vl|{mk}")
    for mk, D in DS.items():
        c1, c2 = MKT[mk]
        A, B = M[c1], M[c2]
        post = WIN[mk]["post"]
        F = (D["zf"] >= 1).astype(float).where(D["zf"].notna())
        for cap in (1.0, 1.5, 2.0):
            w, g, n, s, h = run_verify(D, A, B, F, D["C3"], back=D["C3"], cap=cap)
            cell(f"Cx|{mk}|verb|{cap}", f"E2b z_f≥1 상한 {cap}", mk, post, w, g, n, s, True, "상한 비교")
            w, g, n, s = run_auc(D, F, cap=cap)
            cell(f"Cx|{mk}|moo|{cap}", f"z_f≥1 시장가 상한 {cap}", mk, post, w, g, n, s, True, "상한 비교")

    # ---- 7b. 13차 장부에 넣었을 때(봉인 안, A 다리·C 규칙 그대로, B 활성일 C=0)
    srK, srQ = daily(f"{BT}/round13_slot_book/series/jehyeong_KP.csv")["net"], daily(f"{BT}/round13_slot_book/series/jehyeong_KQ.csv")["net"]
    rows = []

    def mets(r):
        r = r.dropna()
        eq = (1 + r).cumprod()
        return {"일수": len(r), "CAGR %": round((eq.iloc[-1] ** (ANN / len(r)) - 1) * 100, 2), "MDD %": round((eq / eq.cummax() - 1).min() * 100, 2)}

    def var(mk, nm, rb, act):
        col_a, col_c, raw = ("KOSPI_A_jehyeong_ON", "KOSPI_C_jehyeong_KP", srK) if mk == "KOSPI" else ("KOSDAQ_A_blog_ON_Q", "KOSDAQ_C_jehyeong_KQ", srQ)
        idx = bk.index
        rc = raw.reindex(idx).fillna(0.0).where(~act.reindex(idx).fillna(False), 0.0)
        return bk[col_a] + rb.reindex(idx).fillna(0.0) + rc

    BK = {}
    for mk, D in DS.items():
        c1, c2 = MKT[mk]
        A, B = M[c1], M[c2]
        C3, ncol = D["C3"], f"{mk}_B_beomgu_{'C3VT30' if mk == 'KOSPI' else 'C3Q'}"
        F = (D["zf"] >= 1).astype(float).where(D["zf"].notna())
        Fo = F.where(F.index >= FUT_0845, 0.0)
        w0 = wt(C3, D["sig"]) > 0
        v = {"원형": (bk[ncol], w0)}
        w, g, n, s = run_min(D, A, B, C3)
        v["E1 15:30"] = (n, w > 0)
        w, g, n, s = run_min(D, A, B, C3, ex="10:01")
        v["E1 10:01"] = (n, w > 0)
        v["B 없음"] = (pd.Series(0.0, index=bk.index), pd.Series(False, index=bk.index))
        w, g, n, s, h = run_verify(D, A, B, Fo, C3, back=C3)
        v["E2b(2023-07-31~) + 그 전 E1"] = (n.where(n.index >= FUT_0845, run_min(D, A, B, C3)[2]), ((w > 0) & (h | (C3.reindex(w.index) == 1))).where(w.index >= FUT_0845, wt(C3, D["sig"]).reindex(w.index) > 0))
        w, g, n, s = run_auc(D, Fo)
        ee = run_min(D, A, B, F)                                 # 2023-07-28 까지: 09:00 선물 시가 z_f → 09:02 진입
        v["z_f 단일가(2023-07-31~) + 그 전 z_f 09:02"] = (n.where(n.index >= FUT_0845, ee[2].reindex(n.index)), (w > 0).where(w.index >= FUT_0845, ee[0].reindex(w.index).fillna(0) > 0))
        BK[mk] = {nm: var(mk, nm, rb, ac) for nm, (rb, ac) in v.items()}
        for wn, (a0, b0) in {"W_des": ("2022-12-13", "2024-12-30"), "post": ("2023-07-31", "2024-12-30")}.items():
            for nm, r in BK[mk].items():
                rows.append({"market": mk, "창": wn, "B 다리": nm, **mets(r.loc[a0:b0])})
    assert (BK["KOSPI"]["원형"] - bk["KOSPI_book"]).abs().max() < 2e-6 and (BK["KOSDAQ"]["원형"] - bk["KOSDAQ_book"]).abs().max() < 2e-6, "장부 재현 실패"
    for wn, (a0, b0) in {"W_des": ("2022-12-13", "2024-12-30"), "post": ("2023-07-31", "2024-12-30")}.items():
        for nk, nq in (("원형", "원형"), ("E1 15:30", "E1 15:30"), ("E1 10:01", "E1 10:01"), ("E1 15:30", "B 없음"), ("E1 10:01", "B 없음"), ("B 없음", "B 없음"),
                       ("E2b(2023-07-31~) + 그 전 E1", "z_f 단일가(2023-07-31~) + 그 전 z_f 09:02"), ("E2b(2023-07-31~) + 그 전 E1", "B 없음"), ("E1 10:01", "z_f 단일가(2023-07-31~) + 그 전 z_f 09:02")):
            rows.append({"market": "합산 50:50", "창": wn, "B 다리": f"K {nk} / Q {nq}", **mets((0.5 * BK["KOSPI"][nk] + 0.5 * BK["KOSDAQ"][nq]).loc[a0:b0])})
    sK, sQ = [SER[f"S|{mk}|fut|0"][0].reindex(bk.index).fillna(0.0) for mk in ("KOSPI", "KOSDAQ")]
    hk, hq = "E2b(2023-07-31~) + 그 전 E1", "z_f 단일가(2023-07-31~) + 그 전 z_f 09:02"
    for wn, (a0, b0) in {"W_des": ("2022-12-13", "2024-12-30"), "post": ("2023-07-31", "2024-12-30")}.items():
        rows.append({"market": "합산 50:50", "창": wn, "B 다리": "실행판 혼합 + 낮 숏(mis>0) 명목 1", **mets((0.5 * (BK["KOSPI"][hk] + sK) + 0.5 * (BK["KOSDAQ"][hq] + sQ)).loc[a0:b0])})
        rows.append({"market": "합산 50:50", "창": wn, "B 다리": "실행판 혼합 + 코스피 낮 숏만", **mets((0.5 * (BK["KOSPI"][hk] + sK) + 0.5 * BK["KOSDAQ"][hq]).loc[a0:b0])})
    BT_ = pd.DataFrame(rows)
    BT_.to_csv(OUT / "book_effect.csv", index=False, encoding="utf-8-sig")
    print(BT_.to_string(index=False))

    # ---- 8. 예상가 오차 민감도
    NS = pd.DataFrame(noise_sim(K, "KOSPI") + noise_sim(Q, "KOSDAQ"))
    NS.to_csv(OUT / "preopen_noise.csv", index=False, encoding="utf-8-sig")
    print(NS.to_string(index=False))

    # ---- 저장
    C = pd.DataFrame(CELLS)
    C["crit"] = C.apply(crit, axis=1)
    C["pass"] = C["crit"].eq("OOOOOOO") & C["exec_ok"].astype(bool)
    bymap = C.set_index("id")["net_bp"]
    adj = {}
    for fam, ids in FAM.items():
        ok = bool((bymap.reindex(ids) > 0).all())
        for i in ids:
            adj[i] = ok
    C["adj_ok"] = C["id"].map(adj)
    C["notes"] = C["notes"] + " | 기준" + C["crit"] + C["adj_ok"].map(lambda v: "" if pd.isna(v) else (" 인접O" if v else " 인접X"))
    C[["id", "name", "market", "horizon", "window", "n", "gross_bp", "net_bp", "t_nw", "cagr_pct", "mdd_pct", "yearly_pos", "top5_share_pct", "half1_net_bp", "half2_net_bp", "stress_net_bp", "corr_base", "exec_ok", "notes"]].to_csv(OUT / "cells.csv", index=False, encoding="utf-8-sig")
    assert (pd.to_datetime(C["window"].str[-10:]) <= SEAL).all()
    # 후보 상세: 연도·분기, 상위 거래, 하루 제외, 금지 구간 비중
    dt, qr = [], []
    for cid in CAND:
        d, act, p_ = SER[cid]
        ev = d[act]
        top = ev.nlargest(5)
        exd = ev.drop(pd.Timestamp("2023-11-06"), errors="ignore")
        ex3 = ev.drop(top.index[:3])
        ban = ev.loc["2023-11-06":]
        dt.append({"id": cid, "n": len(ev), "평균 명목": round(p_[act].abs().mean(), 3), "최대 명목": round(p_[act].abs().max(), 3), "순 bp": round(ev.mean() * 1e4, 2), "t": round(nw_t(ev), 2), "연도 순합 %": "; ".join(f"{y} {v * 100:+.1f}" for y, v in ev.groupby(ev.index.year).sum().items()),
                   "상위 5일(bp)": "; ".join(f"{i.date()} {v * 1e4:+.0f}" for i, v in top.items()), "2023-11-06 제외 순 bp": round(exd.mean() * 1e4, 2), "제외 t": round(nw_t(exd), 2),
                   "상위 3건 제외 순 bp": round(ex3.mean() * 1e4, 2), "상위 3건 제외 t": round(nw_t(ex3), 2), "금지 구간 거래 비중 %": round(len(ban) / len(ev) * 100, 1), "금지 구간 순 bp": round(ban.mean() * 1e4, 2) if len(ban) else np.nan,
                   "금지 전 순 bp": round(ev.loc[:"2023-11-03"].mean() * 1e4, 2) if len(ev.loc[:"2023-11-03"]) else np.nan})
        q = ev.groupby(ev.index.to_period("Q")).agg(["count", "mean"])
        qr.append({"id": cid, **{str(p): f"{int(r['count'])}/{r['mean'] * 1e4:+.0f}" for p, r in q.iterrows()}})
    DT, QR = pd.DataFrame(dt), pd.DataFrame(qr)
    DT.to_csv(OUT / "cand_detail.csv", index=False, encoding="utf-8-sig")
    QR.to_csv(OUT / "cand_quarter.csv", index=False, encoding="utf-8-sig")
    print(DT.T.to_string())
    print(QR.T.to_string())
    figs(DS, M, WIN)
    ticks = pd.DataFrame({c: {y: round(tick_of(c, pd.DatetimeIndex([f"{y}-06-30"])).iloc[0] * 1e4, 2) for y in (2022, 2023, 2024)} for c in LEGS}).T
    print(ticks.to_string())
    report(C, dict(PT=PT, IC=IC, DC=DC, T22=T22, TM=TM, RT=RT, CAP=CAP, NS=NS, BK=BT_, INH=INH, DT=DT, QR=QR, OV=OV, LC=LC, ticks=ticks, repro=R["repro"], fq=fq, n_fut=(len(jn), int(((jn["kis"] - jn["git"]).abs() < 1e-6).sum()))))
    print("셀 수", len(C), "실행 가능", int(C["exec_ok"].astype(bool).sum()), "통과", int(C["pass"].sum()))
    show = ["id", "n", "gross_bp", "net_bp", "t_nw", "cagr_pct", "mdd_pct", "yearly_pos", "top5_share_pct", "half1_net_bp", "half2_net_bp", "stress_net_bp", "crit", "adj_ok", "corr_base"]
    print(C[C["id"].str.match(r"^(R|E1|C|F|V|L|S|Sd)\|")][show].to_string(index=False))
    print(C[C["pass"]][show].to_string(index=False))
    return C


if __name__ == "__main__":
    main()
