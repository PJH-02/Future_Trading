"""R5 · daily_swing — 일봉 스윙(눌림목·돌파·외국인 선물 수급 필터) 5차 전략 탐색

사전 고정(실행 전 기록, 결과를 본 뒤 바꾸지 않는다)
- 신호 계열: 코스피200 지수 일봉 OHLC (2014-07-07 이전 FDR k200_index_daily, 2014-07-08 이후 fchart KPI200, 빈 행만 FDR).
  069500(KODEX 200)·122630(KODEX 레버리지) 은 같은 지수 신호로 매매한다. 229200(KODEX 코스닥150) 이전 검증은 229200 자체 OHLC 로
  같은 규칙을 계산한다(재조정 없음). 외국인 선물 수급 필터는 세 자산 모두 코스피200 선물 외국인 수급을 쓴다.
- 체결: 결정일 s 종가(15:30) 정보로 결정 → s+1 09:00 시가 단일가 체결(진입·청산 모두). 일별 시가평가:
  진입일 = 종가/시가, 보유일 = 종가/전일 종가, 청산일 = 시가/전일 종가. 무포지션일 0. 비용은 왕복(진입·청산에 절반씩 곱셈 차감).
  비용: 069500 5bp(스트레스 7bp), 122630 5bp(스트레스 8bp), 229200 5bp(7bp), 미니선물 대용 = 지수 경로 2bp(3bp; 베이시스·롤 미반영).
- 지표(신호 계열): RSI(2) Wilder, IBS = (C−L)/(H−L), MA(n) 단순이동평균 종가, ATR(14) Wilder, 채널 = 직전 N일(당일 제외) 종가 최고/최저.
- 외국인 선물 수급(naver_investor('fut')): ratio(s) = (buyq_foreign − sellq_foreign)/(buyq_foreign + sellq_foreign),
  pct(s) = s 제외 직전 250행 중 ratio < ratio(s) 비율(유효 200행 이상). 수급은 s 장 마감 후 확정 → s+1 시가 체결이라 사용 가능.
  veto20: pct(s) < 0.20 이면 진입 금지(pct NaN 이면 허용). conf60: pct(s) ≥ 0.60 일 때만 진입(NaN 이면 금지).

격자(56개, 모두 grid.csv 에 기록)
 PB(눌림목, 18): 추세 C > MA{100,200} × 트리거 {RSI2<10, IBS<0.2, C < 직전 3일 최저 종가} × 청산 {C > MA5(최대 10일),
     RSI2 > 70(최대 10일), 5일 보유}
 BO(돌파, 12): 진입 C > 직전 N{20,55}일 최고 종가 × 청산 {C < 직전 10일 최저 종가, C < 직전 20일 최저 종가,
     C < 진입 후 최고 종가 − 3×ATR14} × MA200 필터 {있음, 없음}
 FL(수급 필터, 26): PB(MA200 고정) 9개 × {veto20, conf60} = 18, BO(MA200 있음) N{20,55} × 청산 {20일 저가, ATR3} × {veto20, conf60} = 8
 포지션 1개(롱 전용, 피라미딩 없음). 청산 체결일 종가에 다시 신호를 보고 재진입 가능.

기간(실행 전 고정): DESIGN = 2010-01-01~2019-12-31, VALIDATION = 2020-01-01~2026-09-29,
 VALIDATION(최근 제외) = 2020-01-01~2025-06-08, 최근 = 2025-06-09~, 추가 표본 외 = 2003-01-01~2009-12-31(지수 신호·069500 경로).
선택 규칙(DESIGN 지표만 사용): 069500·5bp DESIGN 연간 거래 수 ≥ 8 인 변형 중 DESIGN Sharpe 최대(동률이면 DESIGN CAGR).
 2·3위도 같은 순위로 정해 VALIDATION 을 함께 보고한다(검증 성과로 고르지 않는다). 레버리지(122630)는 같은 신호의 크기 변형이며 선택에 쓰지 않는다.
보고만(선택에 쓰지 않음): 체결 민감도 — 결정일 15:30 종가 체결(종가로 계산한 신호를 15:19 신호의 대용으로 둔 낙관적 근사)과,
 2023~2026 KODEX 1분봉으로 15:19 가격 기준 신호와 종가 기준 신호의 일치율.

실행: python topics/r5_daily_swing.py [--dry: 시행 장부 기록 생략]
산출: <OUT>/round5_strategy/daily_swing/ (report.md, daily_returns.csv, trades.csv, grid.csv, equity.png), 시행 장부 주제 R5_DAILY_SWING
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import data  # noqa: E402
import engine as E  # noqa: E402

TOPIC = "R5_DAILY_SWING"
OUTD = data.OUT / "round5_strategy" / "daily_swing"
ANN = 252
DESIGN = ("2010-01-01", "2019-12-31")
VALID = ("2020-01-01", "2026-09-29")
VALID_EX = ("2020-01-01", "2025-06-08")
RECENT = ("2025-06-09", "2026-09-29")
H0 = ("2003-01-01", "2009-12-31")
PERIODS = {"추가OOS(2003~2009)": H0, "DESIGN(2010~2019)": DESIGN, "VALIDATION(2020~2026-09)": VALID,
           "VALID 최근제외(~2025-06-08)": VALID_EX, "최근(2025-06-09~)": RECENT}
MIN_TPY = 8.0
COST = {"069500": (5.0, 7.0), "122630": (5.0, 8.0), "229200": (5.0, 7.0), "IDX": (2.0, 3.0)}
NAME = {"069500": "KODEX 200", "122630": "KODEX 레버리지", "229200": "KODEX 코스닥150", "IDX": "지수 경로(미니선물 대용)"}
MIN1 = data._dir("HYFE_TEAM_REPO", "team_repo_dir", "../..") / "woohyun" / "data" / "KODEX200_1min_2023_2026.csv"   # 팀 저장소 루트: config team_repo_dir, 저장소 안에서는 ../..


# ---------------------------------------------------------------- 자료
def index_ohlc():
    fdr = pd.read_csv(data.RAW / "k200_index_daily.csv", parse_dates=["date"], index_col="date").sort_index()[["open", "high", "low", "close"]]
    fch = data.price("KPI200")[["open", "high", "low", "close"]]
    late = fch.loc["2014-07-08":].combine_first(fdr.loc["2014-07-08":])
    ix = pd.concat([fdr.loc["2000-01-01":"2014-07-07"], late]).sort_index()
    return ix[(ix["open"] > 0) & (ix["close"] > 0)]


def flow_pct():
    f = data.naver_investor("fut")
    b, s = f["buyq_foreign"].astype(float), f["sellq_foreign"].astype(float)
    r = ((b - s) / (b + s)).where((b + s) > 0)
    v = r.to_numpy()
    out = np.full(len(v), np.nan)
    for i in range(len(v)):
        past = v[max(0, i - 250):i]
        past = past[~np.isnan(past)]
        if len(past) >= 200 and not np.isnan(v[i]):
            out[i] = (past < v[i]).mean()
    return pd.DataFrame({"ratio": r, "pct": out}, index=f.index)


def features(sig):
    """sig: open/high/low/close. 결정일 s 종가까지의 정보만 쓴다."""
    C, H, L = sig["close"], sig["high"], sig["low"]
    d = C.diff()
    up = d.clip(lower=0).ewm(alpha=0.5, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=0.5, adjust=False).mean()
    rsi2 = (100 - 100 / (1 + up / dn)).where(dn > 0, 100.0)
    rsi2[d.isna()] = np.nan
    rng = H - L
    tr = pd.concat([H - L, (H - C.shift()).abs(), (L - C.shift()).abs()], axis=1).max(axis=1)
    f = pd.DataFrame({"C": C, "rsi2": rsi2, "ibs": ((C - L) / rng).where(rng > 0),
                      "ma5": C.rolling(5).mean(), "ma100": C.rolling(100).mean(), "ma200": C.rolling(200).mean(),
                      "min3": C.shift(1).rolling(3).min(), "atr": tr.ewm(alpha=1 / 14, adjust=False).mean()})
    for n in (20, 55):
        f[f"hi{n}"] = C.shift(1).rolling(n).max()
    for n in (10, 20):
        f[f"lo{n}"] = C.shift(1).rolling(n).min()
    return f


# ---------------------------------------------------------------- 격자
def grid():
    G = []
    for ma in (100, 200):
        for trg in ("rsi2<10", "ibs<0.2", "c<min3"):
            for ex in ("ma5", "rsi70", "n5"):
                G.append(dict(fam="PB", ma=ma, trg=trg, ex=ex, flow="none"))
    for n in (20, 55):
        for ex in ("lo10", "lo20", "atr3"):
            for ma in (200, 0):
                G.append(dict(fam="BO", ma=ma, trg=f"hi{n}", ex=ex, flow="none"))
    for flow in ("veto20", "conf60"):
        for trg in ("rsi2<10", "ibs<0.2", "c<min3"):
            for ex in ("ma5", "rsi70", "n5"):
                G.append(dict(fam="FL", ma=200, trg=trg, ex=ex, flow=flow))
        for n in (20, 55):
            for ex in ("lo20", "atr3"):
                G.append(dict(fam="FL", ma=200, trg=f"hi{n}", ex=ex, flow=flow))
    for i, g in enumerate(G, 1):
        g["id"] = f"V{i:02d}"
        g["label"] = f"{g['fam']}/{'C>MA' + str(g['ma']) if g['ma'] else 'MA없음'}/{g['trg']}/{g['ex']}/{g['flow']}"
    return G


def entry_sig(f, flow, spec):
    C = f["C"]
    t = spec["trg"]
    if t == "rsi2<10":
        e = f["rsi2"] < 10
    elif t == "ibs<0.2":
        e = f["ibs"] < 0.2
    elif t == "c<min3":
        e = C < f["min3"]
    else:
        e = C > f[t]
    if spec["ma"]:
        e = e & (C > f[f"ma{spec['ma']}"])
    if spec["flow"] == "veto20":
        e = e & ~(flow["pct"] < 0.20)
    elif spec["flow"] == "conf60":
        e = e & (flow["pct"] >= 0.60)
    return e.fillna(False).to_numpy(bool)


def simulate(f, flow, px, spec, cost_bp, fill="open"):
    """f·flow: px.index 에 맞춘 신호 계열. fill='open' 이면 s+1 시가 체결, 'close' 면 결정일 s 종가 체결(보고용 낙관 근사)."""
    O, Cp = px["open"].to_numpy(float), px["close"].to_numpy(float)
    n = len(px)
    ent = entry_sig(f, flow, spec)
    Cs = f["C"].to_numpy(float)
    ex, ok = spec["ex"], ~np.isnan(Cs)
    if ex == "ma5":
        xc = Cs > f["ma5"].to_numpy(float)
    elif ex == "rsi70":
        xc = f["rsi2"].to_numpy(float) > 70
    elif ex in ("lo10", "lo20"):
        xc = Cs < f[ex].to_numpy(float)
    else:
        xc = np.zeros(n, bool)
    atr = f["atr"].to_numpy(float)
    cap = {"ma5": 10, "rsi70": 10, "n5": 5}.get(ex)
    h_on, h_oc, fills = np.zeros(n), np.zeros(n), np.zeros(n)
    trades = []
    pos, pend_in, pend_out, e, peak = False, False, False, -1, np.nan
    for i in range(n):
        if fill == "open":
            if pend_out:
                h_on[i], fills[i], pos, pend_out = 1, fills[i] + 1, False, False
                trades.append((e, i))
            elif pos:
                h_on[i] = 1
            if pend_in:
                pos, e, pend_in, peak = True, i, False, Cs[i]
                fills[i] += 1
            if pos:
                h_oc[i] = 1
                if ok[i]:
                    peak = np.nanmax([peak, Cs[i]])
                held = i - e + 1
                out = bool(xc[i]) or (cap is not None and held >= cap) or (ex == "atr3" and ok[i] and Cs[i] < peak - 3 * atr[i])
                if out and i + 1 < n:
                    pend_out = True
            elif ent[i] and i + 1 < n:
                pend_in = True
        else:                                   # 종가 체결: 결정일 종가에 진입·청산. 보유 구간 = 종가→종가
            if pos:
                h_on[i] = h_oc[i] = 1
                if ok[i]:
                    peak = np.nanmax([peak, Cs[i]])
                held = i - e
                out = bool(xc[i]) or (cap is not None and held >= cap) or (ex == "atr3" and ok[i] and Cs[i] < peak - 3 * atr[i])
                if out:
                    pos, fills[i] = False, fills[i] + 1
                    trades.append((e, i))
                    continue
            elif ent[i]:
                pos, e, peak, fills[i] = True, i, Cs[i], fills[i] + 1
    r_on = np.r_[np.nan, O[1:] / Cp[:-1] - 1]
    r_oc = Cp / O - 1
    if fill == "close":
        r_on, r_oc = np.r_[np.nan, Cp[1:] / Cp[:-1] - 1], np.zeros(n)
    daily = (1 + h_on * np.nan_to_num(r_on)) * (1 + h_oc * np.nan_to_num(r_oc)) * (1 - cost_bp / 2e4) ** fills - 1
    idx = px.index
    P = O if fill == "open" else Cp
    rows = []
    for a, b in trades:
        g = P[b] / P[a] - 1
        rows.append(dict(entry=idx[a], exit=idx[b], days=b - a, entry_px=P[a], exit_px=P[b], gross_bp=g * 1e4,
                         net_bp=((1 + g) * (1 - cost_bp / 2e4) ** 2 - 1) * 1e4))
    if pos:
        rows.append(dict(entry=idx[e], exit=pd.NaT, days=n - 1 - e, entry_px=P[e], exit_px=np.nan, gross_bp=np.nan, net_bp=np.nan))
    tr = pd.DataFrame(rows, columns=["entry", "exit", "days", "entry_px", "exit_px", "gross_bp", "net_bp"])
    return pd.Series(daily, index=idx), pd.Series(h_oc, index=idx), tr


# ---------------------------------------------------------------- 지표
def metrics(daily, held, tr, a, b, bench=None):
    d = daily.loc[a:b].dropna()
    if len(d) < 20:
        return None
    t = tr[(tr["entry"] >= pd.Timestamp(a)) & (tr["entry"] <= pd.Timestamp(b)) & tr["exit"].notna()]
    eq = (1 + d).cumprod()
    yrs = len(d) / ANN
    cagr = eq.iloc[-1] ** (1 / yrs) - 1
    mdd = (eq / eq.cummax() - 1).min()
    sd = d.std()
    net = t["net_bp"].to_numpy() / 1e4
    out = dict(start=str(d.index[0].date()), end=str(d.index[-1].date()), days=len(d), trades=len(t), trades_yr=len(t) / yrs,
               exposure=held.loc[a:b].mean(), win=(net > 0).mean() if len(t) else np.nan,
               gross_bp=t["gross_bp"].mean() if len(t) else np.nan, net_bp=t["net_bp"].mean() if len(t) else np.nan,
               t_net=E.nw_t(net) if len(t) >= 10 else np.nan, avg_days=t["days"].mean() if len(t) else np.nan,
               avg_win_bp=net[net > 0].mean() * 1e4 if (net > 0).any() else np.nan,
               avg_loss_bp=net[net <= 0].mean() * 1e4 if (net <= 0).any() else np.nan,
               cagr=cagr, sharpe=d.mean() / sd * np.sqrt(ANN) if sd > 0 else np.nan, mdd=mdd,
               calmar=cagr / -mdd if mdd < 0 else np.nan)
    if bench is not None:
        bb = bench.loc[a:b].dropna()
        be = (1 + bb).cumprod()
        out["bh_cagr"] = be.iloc[-1] ** (ANN / len(bb)) - 1
        out["bh_mdd"] = (be / be.cummax() - 1).min()
    return out


def pct(x, k=1):
    return "-" if x is None or pd.isna(x) else f"{x * 100:.{k}f}%"


def num(x, f="{:+.2f}"):
    return "-" if x is None or pd.isna(x) else f.format(x)


def md(rows, cols):
    """rows: list of dict. cols: [(키, 제목, 형식)]"""
    h = "| " + " | ".join(c[1] for c in cols) + " |\n|" + "---|" * len(cols) + "\n"
    for r in rows:
        cells = []
        for k, _, fm in cols:
            v = r.get(k)
            if fm == "s":
                cells.append("-" if v is None else str(v))
            elif fm == "p":
                cells.append(pct(v))
            elif fm == "i":
                cells.append("-" if v is None or pd.isna(v) else f"{int(v)}")
            else:
                cells.append(num(v, fm))
        h += "| " + " | ".join(cells) + " |\n"
    return h


PCOLS = [("period", "구간", "s"), ("trades", "거래", "i"), ("trades_yr", "연 거래", "{:.1f}"), ("exposure", "보유비중", "p"),
         ("win", "승률", "p"), ("net_bp", "거래당 순bp", "{:+.1f}"), ("avg_win_bp", "평균 이익bp", "{:+.0f}"), ("avg_loss_bp", "평균 손실bp", "{:+.0f}"),
         ("t_net", "NW t", "{:+.2f}"), ("avg_days", "평균 보유일", "{:.1f}"),
         ("cagr", "CAGR", "p"), ("sharpe", "Sharpe", "{:.2f}"), ("mdd", "MDD", "p"), ("calmar", "Calmar", "{:.2f}"),
         ("bh_cagr", "B&H CAGR", "p"), ("bh_mdd", "B&H MDD", "p")]


def placebo(px, tr, a, b, draws=2000, seed=20261005):
    """같은 구간·같은 거래 수·같은 보유일 수로 진입일만 무작위(시가→시가)로 둔 거래당 평균 총수익 분포 대비 백분위."""
    O = px["open"].to_numpy(float)
    idx = px.index
    t = tr[(tr["entry"] >= pd.Timestamp(a)) & (tr["entry"] <= pd.Timestamp(b)) & tr["exit"].notna()]
    if len(t) < 10:
        return np.nan, np.nan
    lo, hi = idx.searchsorted(pd.Timestamp(a)), idx.searchsorted(pd.Timestamp(b), side="right") - 1
    L = t["days"].to_numpy(int)
    rng = np.random.default_rng(seed)
    e = lo + (rng.random((draws, len(L))) * np.maximum(1, hi - L - lo + 1)).astype(int)
    x = np.minimum(e + L, len(O) - 1)
    sims = (O[x] / O[e] - 1).mean(axis=1)
    act = t["gross_bp"].mean() / 1e4
    return float((sims < act).mean()), float(np.median(sims) * 1e4)


# ---------------------------------------------------------------- 1분봉 신호 일치율(보고용)
def min1_agreement(spec, ix_feat_full, sig_ix):
    """2023~2026 KODEX 1분봉 15:19 봉 종가로 계산한 진입 신호 vs 15:30 종가 기준 신호(KODEX 자체 일봉 계열에서 계산) 일치율."""
    if not MIN1.exists():
        return None
    m = pd.read_csv(MIN1)
    tcol = next(c for c in m.columns if "date" in c.lower() or "time" in c.lower())
    m[tcol] = pd.to_datetime(m[tcol], utc=True).dt.tz_convert("Asia/Seoul")
    m = m.set_index(tcol).sort_index()
    ccol = next(c for c in m.columns if c.lower() == "close")
    hm = m.index.strftime("%H:%M")
    m = m[(hm >= "09:01") & (hm <= "15:30")]
    day = m.index.normalize().tz_localize(None)
    hm = m.index.strftime("%H:%M")
    hcol = next(c for c in m.columns if c.lower() == "high")
    lcol = next(c for c in m.columns if c.lower() == "low")
    upto = m[hm <= "15:19"]
    dd = upto.index.normalize().tz_localize(None)
    c1519 = upto[ccol].groupby(dd).last()
    h1519 = upto[hcol].groupby(dd).max()
    l1519 = upto[lcol].groupby(dd).min()
    c1530 = m[ccol].groupby(day).last()                     # 원가격 종가(15:30 단일가)
    px = data.price("069500")
    rows = []
    for dte in c1519.index:
        if dte not in px.index or dte not in c1530.index or not c1530[dte] > 0:
            continue
        adj = px.at[dte, "close"] / c1530[dte]              # 1분봉은 원가격 → 수정주가 배율로 환산
        hist = px.loc[:dte].copy()
        hist.iloc[-1, hist.columns.get_loc("close")] = c1519[dte] * adj
        hist.iloc[-1, hist.columns.get_loc("high")] = h1519[dte] * adj
        hist.iloc[-1, hist.columns.get_loc("low")] = l1519[dte] * adj
        rows.append((dte, hist.iloc[-260:]))
    if not rows:
        return None
    a = b = both_ = 0
    full = features(px)
    flow = pd.DataFrame({"pct": np.nan}, index=px.index)
    e_close = pd.Series(entry_sig(full, flow, {**spec, "flow": "none"}), index=px.index)
    agree = 0
    for dte, h in rows:
        fe = features(h)
        e_1519 = bool(entry_sig(fe, flow.reindex(fe.index), {**spec, "flow": "none"})[-1])
        e_c = bool(e_close.get(dte, False))
        a += e_1519
        b += e_c
        both_ += e_1519 and e_c
        agree += e_1519 == e_c
    return dict(days=len(rows), sig_1519=a, sig_close=b, both=both_, agree=agree / len(rows))


# ---------------------------------------------------------------- 실행
def main(dry=False):
    OUTD.mkdir(parents=True, exist_ok=True)
    ix = index_ohlc()
    fx = features(ix)
    fl = flow_pct()
    px = {k: data.price(k) for k in ("069500", "122630", "229200")}
    px["IDX"] = ix
    bench = px["069500"]["close"].pct_change()
    feat = {k: fx.reindex(px[k].index) for k in ("069500", "122630", "IDX")}
    feat["229200"] = features(px["229200"])
    flows = {k: fl.reindex(px[k].index) for k in px}
    bh = {k: px[k]["close"].pct_change() for k in px}

    G = grid()
    grows = []
    for g in G:
        dly, held, tr = simulate(feat["069500"], flows["069500"], px["069500"], g, 5.0)
        row = dict(id=g["id"], label=g["label"], fam=g["fam"], ma=g["ma"], trg=g["trg"], ex=g["ex"], flow=g["flow"])
        for tag, (a, b) in (("d", DESIGN), ("v", VALID), ("x", VALID_EX), ("h", H0)):
            m = metrics(dly, held, tr, a, b)
            for k in ("trades", "trades_yr", "exposure", "win", "net_bp", "t_net", "cagr", "sharpe", "mdd", "calmar"):
                row[f"{tag}_{k}"] = m[k] if m else np.nan
        dl, hl, tl = simulate(feat["122630"], flows["122630"], px["122630"], g, 5.0)
        m = metrics(dl, hl, tl, *DESIGN)
        row["d_lev_cagr"], row["d_lev_sharpe"], row["d_lev_mdd"] = m["cagr"], m["sharpe"], m["mdd"]
        grows.append(row)
    gd = pd.DataFrame(grows)
    elig = gd[gd["d_trades_yr"] >= MIN_TPY].sort_values(["d_sharpe", "d_cagr"], ascending=False)
    gd["design_rank"] = gd["id"].map({v: i + 1 for i, v in enumerate(elig["id"])})
    gd.to_csv(OUTD / "grid.csv", index=False, encoding="utf-8-sig")
    top = [next(g for g in G if g["id"] == v) for v in elig["id"].iloc[:3]]
    chosen = top[0]

    # ---- 선택 사양 기간별
    res, yearly, all_daily, all_trades = {}, {}, {}, []
    for k in ("069500", "122630", "IDX", "229200"):
        for c in COST[k]:
            dly, held, tr = simulate(feat[k], flows[k], px[k], chosen, c)
            res[(k, c)] = {p: metrics(dly, held, tr, a, b, bench=bh[k]) for p, (a, b) in PERIODS.items()}
            if c == COST[k][0]:
                all_daily[f"{k}_net{int(c)}bp"] = dly
                all_daily[f"{k}_held"] = held
                t2 = tr.copy()
                t2.insert(0, "asset", k)
                all_trades.append(t2)
                yrs = sorted(y for y in set(px[k].index.year) if y >= 2003)
                yearly[k] = [{**(metrics(dly, held, tr, f"{y}-01-01", f"{y}-12-31", bench=bh[k]) or {}), "period": str(y)} for y in yrs
                             if metrics(dly, held, tr, f"{y}-01-01", f"{y}-12-31") is not None]
    dfd = pd.DataFrame(all_daily)
    dfd["bh_069500"] = bench
    dfd.loc["2003-01-01":].to_csv(OUTD / "daily_returns.csv", encoding="utf-8-sig")
    pd.concat(all_trades).to_csv(OUTD / "trades.csv", index=False, encoding="utf-8-sig")

    # ---- 2·3위 견고성
    robust = []
    for rk, g in enumerate(top, 1):
        dly, held, tr = simulate(feat["069500"], flows["069500"], px["069500"], g, 5.0)
        for p in ("DESIGN(2010~2019)", "VALIDATION(2020~2026-09)", "VALID 최근제외(~2025-06-08)", "추가OOS(2003~2009)"):
            m = metrics(dly, held, tr, *PERIODS[p], bench=bench)
            robust.append({**m, "period": p, "rank": rk, "label": g["label"]})

    # ---- 무작위 진입 플라시보(보고용, 069500·선택 사양)
    dly, held, tr = simulate(feat["069500"], flows["069500"], px["069500"], chosen, 5.0)
    plc = {p: placebo(px["069500"], tr, a, b) for p, (a, b) in PERIODS.items()}

    # ---- 수급 veto 짝 비교(보고용): 같은 PB(MA200) 규칙의 필터 없음 vs veto20 vs conf60
    pairs = []
    for trg in ("rsi2<10", "ibs<0.2", "c<min3"):
        for ex_ in ("ma5", "rsi70", "n5"):
            sel = gd[(gd["ma"] == 200) & (gd["trg"] == trg) & (gd["ex"] == ex_)]
            r0, rv, rc = (sel[sel["flow"] == f].iloc[0] for f in ("none", "veto20", "conf60"))
            pairs.append(dict(rule=f"{trg}/{ex_}", **{f"{tg}_{k}_{nm}": r[f"{tg}_{k}"] for nm, r in (("0", r0), ("v", rv), ("c", rc))
                                                      for tg in ("d", "x") for k in ("sharpe", "net_bp")}))

    # ---- 체결 민감도(보고용)
    sens = {}
    for k in ("069500", "122630"):
        dly, held, tr = simulate(feat[k], flows[k], px[k], chosen, COST[k][0], fill="close")
        sens[k] = {p: metrics(dly, held, tr, a, b, bench=bh[k]) for p, (a, b) in PERIODS.items()}
    try:
        agree = min1_agreement(chosen, fx, ix)
    except Exception as ex:          # 1분봉 형식 문제는 보고만 생략
        agree = {"error": str(ex)}

    # ---- 그림
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams["font.family"] = "Malgun Gothic"
        plt.rcParams["axes.unicode_minus"] = False
        fig, ax = plt.subplots(figsize=(11, 5.5))
        s0 = "2003-01-01"
        for col, lab, colr in (("069500_net5bp", "전략 · KODEX 200 (5bp)", "#2a78d6"), ("122630_net5bp", "전략 · KODEX 레버리지 (5bp)", "#eb6834"),
                               ("bh_069500", "KODEX 200 매수보유", "#8a8a8a")):
            s = dfd[col].loc[s0:].fillna(0)
            ax.plot((1 + s).cumprod(), label=lab, color=colr, lw=1.3)
        ax.set_yscale("log")
        ax.axvspan(pd.Timestamp(DESIGN[0]), pd.Timestamp(DESIGN[1]), color="#2a78d6", alpha=0.06, label="DESIGN")
        ax.axvspan(pd.Timestamp(VALID[0]), pd.Timestamp(VALID[1]), color="#1baf7a", alpha=0.06, label="VALIDATION")
        ax.set_title(f"일봉 스윙 선택 사양: {chosen['label']}")
        ax.legend(loc="upper left", fontsize=9)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(OUTD / "equity.png", dpi=120)
        plt.close(fig)
    except Exception as ex:
        print("그림 생략:", ex)

    # ---- 장부
    spec = dict(family="daily_swing", chosen=chosen, top3=[g["id"] for g in top], selection=f"069500 5bp DESIGN 연간 거래≥{MIN_TPY} 중 DESIGN Sharpe 최대",
                n_variants=len(G), design=DESIGN, valid=VALID)
    if not dry:
        led = []
        for _, r in gd.iterrows():
            led.append(dict(asset="069500", days=np.nan, trades=r["d_trades"], exposure=r["d_exposure"], net_bp=r["d_net_bp"], t_net=r["d_t_net"],
                            win=r["d_win"], sharpe=r["d_sharpe"], cagr=r["d_cagr"], mdd=r["d_mdd"], calmar=r["d_calmar"],
                            rule=f"{r['id']} {r['label']}", window="DESIGN(2010~2019)", cost_bp=5.0))
        for (k, c), per in res.items():
            for p, m in per.items():
                if m:
                    led.append(dict(asset=k, days=m["days"], trades=m["trades"], exposure=m["exposure"], gross_bp=m["gross_bp"], net_bp=m["net_bp"],
                                    t_net=m["t_net"], win=m["win"], sharpe=m["sharpe"], cagr=m["cagr"], mdd=m["mdd"], calmar=m["calmar"],
                                    rule=f"선택 {chosen['id']} {chosen['label']}", window=p, cost_bp=c))
        E.ledger_append(data.OUT, TOPIC, spec, pd.DataFrame(led), note="5차 전략 탐색")

    out = dict(chosen=chosen, top=top, grid=gd, elig=elig, res=res, yearly=yearly, robust=robust, sens=sens, agree=agree, n=len(G),
               plc=plc, pairs=pd.DataFrame(pairs))
    write_report(out)
    with open(OUTD / "summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary_json(out), fh, ensure_ascii=False, indent=1, default=lambda o: None if pd.isna(o) else float(o))
    return out


def summary_json(o):
    r = o["res"]
    def g(k, c, p):
        m = r[(k, c)][p]
        return {kk: m[kk] for kk in ("trades", "trades_yr", "exposure", "win", "net_bp", "t_net", "cagr", "sharpe", "mdd", "calmar", "bh_cagr", "bh_mdd")} if m else None
    return {"chosen": o["chosen"], "top": [t["id"] for t in o["top"]],
            "res": {f"{k}@{int(c)}bp": {p: g(k, c, p) for p in PERIODS} for (k, c) in r}, "agree": o["agree"]}


def write_report(o):
    ch, gd, res = o["chosen"], o["grid"], o["res"]
    L = []
    L.append("# 5차 전략 탐색 · 일봉 스윙(daily_swing)\n")
    L.append("코스피200 지수 일봉으로 신호를 만들고 KODEX 200(069500)·KODEX 레버리지(122630)를 다음 날 09:00 시가 단일가에 매매하는 "
             "눌림목·돌파 스윙 규칙을 56개 격자로 탐색했다. 선택은 DESIGN(2010~2019) 지표만으로 하고, VALIDATION(2020-01~2026-09)은 선택 뒤에 한 번 읽었다.\n")
    v0, vx0, d0, h0 = (res[("069500", 5.0)][p] for p in ("VALIDATION(2020~2026-09)", "VALID 최근제외(~2025-06-08)", "DESIGN(2010~2019)", "추가OOS(2003~2009)"))
    vl0, vlx0 = res[("122630", 5.0)]["VALIDATION(2020~2026-09)"], res[("122630", 5.0)]["VALID 최근제외(~2025-06-08)"]
    L.append("\n## 요약\n\n")
    if ch["id"] == "V34":
        L.append("- 선택 사양 V34: 코스피200 종가 > MA200 이고 IBS < 0.2(당일 저가권 마감)인 날, 외국인 선물 순매수 비율이 직전 250일 하위 20%가 아니면 "
                 "다음 날 09:00 시가에 KODEX 200 매수, 코스피200 종가 > MA5 가 되면(최대 10거래일) 다음 날 시가 매도.\n")
    else:
        L.append(f"- 선택 사양 {ch['id']} {ch['label']}\n")
    L.append(f"- VALIDATION(2020-01~2026-09, 5bp): 거래 {v0['trades']}회(연 {v0['trades_yr']:.1f}), 승률 {pct(v0['win'])}, 거래당 순 {v0['net_bp']:+.1f}bp, "
             f"CAGR {pct(v0['cagr'])}, Sharpe {v0['sharpe']:.2f}, MDD {pct(v0['mdd'])} (같은 기간 매수보유 CAGR {pct(v0['bh_cagr'])}, MDD {pct(v0['bh_mdd'])}).\n")
    L.append(f"- 2025-06-09 이후 급등·고변동 구간을 빼면(2020-01~2025-06-08): 승률 {pct(vx0['win'])}, 거래당 순 {vx0['net_bp']:+.1f}bp, CAGR {pct(vx0['cagr'])}, "
             f"Sharpe {vx0['sharpe']:.2f}. DESIGN(2010~2019)도 CAGR {pct(d0['cagr'])}, 거래당 {d0['net_bp']:+.1f}bp 로 사실상 0 이다.\n")
    L.append(f"- 승률 51% 이상은 모든 구간에서 충족(DESIGN {pct(d0['win'])}, VALIDATION {pct(v0['win'])}, 최근 제외 {pct(vx0['win'])}, 2003~2009 {pct(h0['win'])}). "
             "다만 눌림목 규칙의 구조적 특성(작은 이익 다수·큰 손실 소수)이라 승률만으로 수익성을 뜻하지 않는다.\n")
    L.append(f"- CAGR ≈10% 는 VALIDATION 전체로는 {pct(v0['cagr'])} 로 닿지만 2025-06 이후 상승장에 전적으로 기댄다. 레버리지(122630)로 같은 신호를 매매하면 "
             f"VALIDATION CAGR {pct(vl0['cagr'])}(최근 제외 {pct(vlx0['cagr'])}), MDD {pct(vl0['mdd'])}.\n")
    L.append(f"- 2003~2009 추가 표본 외: CAGR {pct(h0['cagr'])}, 승률 {pct(h0['win'])}, 거래당 {h0['net_bp']:+.1f}bp (NW t {h0['t_net']:+.2f}). "
             "상승 추세·고변동 국면(2003~2009, 2025-06~)에서만 돈을 벌고 2010~2025-06 박스권에서는 본전인 국면 의존 전략이다.\n")
    pl = o["plc"]
    L.append(f"- 무작위 진입(같은 거래 수·보유일) 대비 거래당 총수익 백분위: DESIGN {pct(pl['DESIGN(2010~2019)'][0], 0)}, "
             f"VALIDATION {pct(pl['VALIDATION(2020~2026-09)'][0], 0)}, 최근 제외 {pct(pl['VALID 최근제외(~2025-06-08)'][0], 0)}, "
             f"2003~2009 {pct(pl['추가OOS(2003~2009)'][0], 0)}. 95% 를 넘는 구간이 없어 '진입 타이밍 우위'는 통계적으로 확인되지 않는다"
             "(수익의 대부분은 상승장에 롱으로 있었던 효과).\n")
    L.append("- 코스닥150(229200) 이전(재조정 없음)은 VALIDATION 에서 손실이다. 발표에서는 '승률 목표 충족, 수익률은 국면 의존'으로 제시하는 것이 정직하다.\n\n")
    L.append("## 1. 사전 고정 사항\n")
    L.append("- 신호 계열: 코스피200 지수 일봉 OHLC(2014-07-07 이전 FDR, 이후 네이버 fchart). 069500·122630 은 같은 지수 신호를 쓴다. "
             "229200(KODEX 코스닥150) 이전 검증은 229200 자체 OHLC 로 같은 규칙을 계산(재조정 없음).\n"
             "- 체결: 결정일 종가(15:30) 정보 → 다음 날 09:00 시가 단일가 진입·청산. 일별 시가평가, 무포지션일 0. 비용 왕복 차감(진입·청산 절반씩).\n"
             "- 비용: 069500 5bp(스트레스 7bp), 122630 5bp(8bp), 229200 5bp(7bp), 지수 경로 2bp(3bp, 미니선물 대용 — 베이시스·롤·증거금 미반영).\n"
             "- 지표: RSI(2) Wilder, IBS=(종가−저가)/(고가−저가), MA 단순이평, ATR(14) Wilder, 채널 = 당일 제외 직전 N일 종가 최고/최저.\n"
             "- 외국인 선물 수급: ratio = (외국인 매수계약−매도계약)/(매수+매도), 결정일 s 값을 s 제외 직전 250일 분포의 백분위로 환산. "
             "veto20 = 하위 20% 면 진입 금지, conf60 = 상위 40%(≥60 백분위)일 때만 진입. 수급은 장 마감 후 확정되므로 다음 날 시가 체결에 쓸 수 있다.\n"
             f"- 격자 {o['n']}개: PB 눌림목 18 + BO 돌파 12 + FL 수급 필터 26(PB 18, BO 8). 정의는 스크립트 docstring.\n"
             f"- 선택 규칙(실행 전 고정): 069500·5bp DESIGN 연간 거래 ≥ {MIN_TPY:.0f}회인 변형 중 DESIGN Sharpe 최대(동률이면 CAGR). "
             "2·3위는 같은 순위로 정하고 VALIDATION 을 함께 보인다. 레버리지는 같은 신호의 크기 변형이며 선택에 쓰지 않았다.\n")
    L.append("## 2. 선택된 사양\n")
    L.append(f"**{ch['id']} · {ch['label']}**\n")
    L.append(rule_text(ch) + "\n")
    L.append(f"DESIGN 적격 변형 {len(o['elig'])}개 / 전체 {o['n']}개. 상위 5개(DESIGN 지표, 069500·5bp):\n")
    cols = [("id", "ID", "s"), ("label", "규칙", "s"), ("d_trades_yr", "연 거래", "{:.1f}"), ("d_win", "승률", "p"), ("d_net_bp", "거래당 순bp", "{:+.1f}"),
            ("d_cagr", "CAGR", "p"), ("d_sharpe", "Sharpe", "{:.2f}"), ("d_mdd", "MDD", "p"), ("d_lev_cagr", "레버리지 CAGR", "p")]
    L.append(md(o["elig"].head(5).to_dict("records"), cols))
    L.append("\n## 3. 선택 사양 성과(구간별)\n")
    for k in ("069500", "122630", "IDX", "229200"):
        for c in COST[k]:
            rows = [{**m, "period": p} for p, m in res[(k, c)].items() if m]
            note = "" if k != "229200" else " — 신호도 229200 자체 가격(재조정 없음)"
            L.append(f"\n**{NAME[k]} ({k}) · 비용 {c:g}bp{note}** (B&H = 같은 자산 종가 매수보유, 비용 없음)\n\n")
            L.append(md(rows, PCOLS))
    L.append("\n## 4. 목표 판정(VALIDATION, 5bp)\n")
    v = res[("069500", 5.0)]["VALIDATION(2020~2026-09)"]
    vl = res[("122630", 5.0)]["VALIDATION(2020~2026-09)"]
    vi = res[("IDX", 2.0)]["VALIDATION(2020~2026-09)"]
    L.append(f"- KODEX 200: CAGR {pct(v['cagr'])}, 거래 승률 {pct(v['win'])} → 목표(CAGR≈10% 또는 승률≥51%) "
             f"{'충족' if (v['cagr'] >= 0.095 or v['win'] >= 0.51) else '미충족'}"
             f" (CAGR 기준 {'충족' if v['cagr'] >= 0.095 else '미충족'}, 승률 기준 {'충족' if v['win'] >= 0.51 else '미충족'})\n")
    L.append(f"- 같은 기준을 2025-06-08 까지로 자르면 KODEX 200 CAGR {pct(vx0['cagr'])}, 승률 {pct(vx0['win'])} → CAGR 미충족, 승률 충족\n")
    L.append(f"- KODEX 레버리지: CAGR {pct(vl['cagr'])}, 승률 {pct(vl['win'])}, MDD {pct(vl['mdd'])}\n")
    L.append(f"- 지수 경로 2bp(미니선물 대용): CAGR {pct(vi['cagr'])}, 승률 {pct(vi['win'])}\n")
    L.append("\n## 5. 견고성: DESIGN 2·3위의 성과(069500·5bp)\n\n")
    L.append(md(o["robust"], [("rank", "DESIGN 순위", "i"), ("label", "규칙", "s"), ("period", "구간", "s"), ("trades", "거래", "i"), ("win", "승률", "p"),
                              ("net_bp", "거래당 순bp", "{:+.1f}"), ("cagr", "CAGR", "p"), ("sharpe", "Sharpe", "{:.2f}"), ("mdd", "MDD", "p")]))
    un = gd.sort_values(["d_sharpe", "d_cagr"], ascending=False).iloc[0]
    L.append(f"\n참고: 연 거래 ≥ {MIN_TPY:.0f} 제약이 없었다면 DESIGN Sharpe 1위는 {un['id']} {un['label']}(연 {un['d_trades_yr']:.1f}회, "
             f"DESIGN Sharpe {un['d_sharpe']:.2f})였다. 그 VALIDATION: 승률 {pct(un['v_win'])}, CAGR {pct(un['v_cagr'])}, Sharpe {un['v_sharpe']:.2f}; "
             f"최근 제외 CAGR {pct(un['x_cagr'])}, Sharpe {un['x_sharpe']:.2f}.\n")
    L.append("\n무작위 진입 플라시보(069500, 같은 구간·같은 거래 수·같은 보유일 수, 진입일만 무작위, 시가→시가 총수익, 2000회): "
             "선택 사양의 거래당 평균 총수익이 무작위보다 큰 비율(백분위)과 무작위 중앙값.\n\n")
    prow = []
    for p, v in o["plc"].items():
        m = res[("069500", 5.0)][p]
        prow.append({"period": p, "pctile": v[0], "med": v[1], "act": m["gross_bp"] if m else np.nan})
    L.append(md(prow, [("period", "구간", "s"), ("act", "선택 사양 거래당 총bp", "{:+.1f}"), ("med", "무작위 중앙값 bp", "{:+.1f}"), ("pctile", "백분위", "p")]))
    pr = o["pairs"]
    L.append("\n수급 필터 짝 비교(PB·MA200 같은 규칙 9쌍, 069500·5bp; d = DESIGN, x = VALIDATION 최근 제외):\n\n")
    L.append(md(pr.to_dict("records"), [("rule", "규칙", "s"), ("d_sharpe_0", "d Sharpe 없음", "{:.2f}"), ("d_sharpe_v", "d veto20", "{:.2f}"),
                                       ("d_sharpe_c", "d conf60", "{:.2f}"), ("x_sharpe_0", "x Sharpe 없음", "{:.2f}"), ("x_sharpe_v", "x veto20", "{:.2f}"),
                                       ("x_sharpe_c", "x conf60", "{:.2f}"), ("d_net_bp_0", "d 순bp 없음", "{:+.1f}"), ("d_net_bp_v", "d veto20", "{:+.1f}"),
                                       ("x_net_bp_0", "x 순bp 없음", "{:+.1f}"), ("x_net_bp_v", "x veto20", "{:+.1f}")]))
    L.append(f"\nveto20 이 Sharpe 를 높인 쌍: DESIGN {int((pr['d_sharpe_v'] > pr['d_sharpe_0']).sum())}/9, "
             f"VALIDATION 최근 제외 {int((pr['x_sharpe_v'] > pr['x_sharpe_0']).sum())}/9. "
             f"conf60: DESIGN {int((pr['d_sharpe_c'] > pr['d_sharpe_0']).sum())}/9, 최근 제외 {int((pr['x_sharpe_c'] > pr['x_sharpe_0']).sum())}/9.\n")
    gv = gd.dropna(subset=["v_sharpe"])
    L.append(f"\n전체 {len(gd)}개 변형의 VALIDATION 분포(선택에 쓰지 않음, 참고): Sharpe 중앙값 {gv['v_sharpe'].median():.2f}, "
             f"승률 중앙값 {pct(gv['v_win'].median())}, 승률 ≥51% 변형 {int((gv['v_win'] >= 0.51).sum())}개, "
             f"CAGR ≥9.5% 변형 {int((gv['v_cagr'] >= 0.095).sum())}개. 선택 사양의 VALIDATION Sharpe 순위 "
             f"{int((gv['v_sharpe'] > gd.loc[gd['id'] == ch['id'], 'v_sharpe'].iloc[0]).sum()) + 1}/{len(gv)}.\n")
    fam = gd.groupby("fam")[["d_sharpe", "v_sharpe", "d_win", "v_win", "d_cagr", "v_cagr"]].median()
    L.append("\n규칙군별 중앙값(069500·5bp):\n\n")
    L.append(md([{**r, "fam": i} for i, r in fam.iterrows()], [("fam", "규칙군", "s"), ("d_sharpe", "DESIGN Sharpe", "{:.2f}"), ("v_sharpe", "VALID Sharpe", "{:.2f}"),
                                                            ("d_win", "DESIGN 승률", "p"), ("v_win", "VALID 승률", "p"), ("d_cagr", "DESIGN CAGR", "p"),
                                                            ("v_cagr", "VALID CAGR", "p")]))
    L.append("\n## 6. 연도별(선택 사양, 5bp)\n")
    for k in ("069500", "122630"):
        L.append(f"\n**{NAME[k]}**\n\n")
        L.append(md(o["yearly"][k], [("period", "연도", "s"), ("trades", "거래", "i"), ("win", "승률", "p"), ("net_bp", "거래당 순bp", "{:+.1f}"),
                                     ("exposure", "보유비중", "p"), ("cagr", "수익률", "p"), ("mdd", "MDD", "p"), ("bh_cagr", "B&H", "p")]))
    L.append("\n## 7. 체결 민감도(보고용, 선택·판정에 쓰지 않음)\n\n")
    L.append("결정일 15:30 종가에 바로 체결한다고 둔 낙관적 근사(신호는 종가로 계산 — 실거래는 15:19 가격으로 신호를 내고 15:20~15:30 "
             "종가 단일가에 주문해야 한다).\n\n")
    for k in ("069500", "122630"):
        rows = [{**m, "period": p} for p, m in o["sens"][k].items() if m]
        L.append(f"\n**{NAME[k]} · 종가 체결 · {COST[k][0]:g}bp**\n\n")
        L.append(md(rows, PCOLS))
    a = o["agree"]
    if a and "error" not in a:
        L.append(f"\n15:19 신호 일치율(2023~2026 KODEX 1분봉, KODEX 자체 계열로 계산, 수급 필터 제외): {a['days']}일 중 일치 {pct(a['agree'])}, "
                 f"15:19 기준 진입 신호 {a['sig_1519']}일 / 종가 기준 {a['sig_close']}일 / 둘 다 {a['both']}일.\n")
    elif a:
        L.append(f"\n15:19 신호 일치율 계산 생략: {a['error']}\n")
    L.append("\n## 8. 해석과 한계\n\n" + caveats_text(o) + "\n")
    L.append("\n## 9. 파일\n\n- grid.csv: 56개 변형의 DESIGN 지표(선택용)와 VALIDATION 지표(참고), design_rank\n"
             "- daily_returns.csv: 선택 사양 자산별 일별 순수익률(5bp 등 기본 비용)·보유 여부, KODEX 200 매수보유\n"
             "- trades.csv: 선택 사양 자산별 거래(진입·청산일, 체결가, 총·순 bp)\n- equity.png: 누적 곡선(로그)\n"
             "- 코드: overnight_lab/topics/r5_daily_swing.py\n")
    out_lines = []
    for ln in "".join(L).split("\n"):
        prev = out_lines[-1] if out_lines else ""
        is_tab, prev_tab = ln.startswith("|"), prev.startswith("|")
        if prev and ((ln.startswith("#") and not prev.startswith("#")) or (is_tab and not prev_tab) or (prev_tab and not is_tab and ln)):
            out_lines.append("")                      # 제목·표 앞뒤 빈 줄
        out_lines.append(ln)
    (OUTD / "report.md").write_text("\n".join(out_lines), encoding="utf-8")


def rule_text(g):
    trg = {"rsi2<10": "코스피200 RSI(2) < 10", "ibs<0.2": "코스피200 IBS < 0.2(종가가 당일 범위 하단 20%)",
           "c<min3": "코스피200 종가 < 직전 3일 최저 종가", "hi20": "코스피200 종가 > 직전 20일 최고 종가", "hi55": "코스피200 종가 > 직전 55일 최고 종가"}[g["trg"]]
    ex = {"ma5": "종가 > MA5 이면 다음 날 시가 청산(최대 10거래일)", "rsi70": "RSI(2) > 70 이면 다음 날 시가 청산(최대 10거래일)",
          "n5": "진입 후 5거래일째 종가에 결정 → 다음 날 시가 청산", "lo10": "종가 < 직전 10일 최저 종가면 다음 날 시가 청산",
          "lo20": "종가 < 직전 20일 최저 종가면 다음 날 시가 청산", "atr3": "종가 < 진입 후 최고 종가 − 3×ATR14 면 다음 날 시가 청산"}[g["ex"]]
    ma = f"코스피200 종가 > MA{g['ma']}" if g["ma"] else "추세 필터 없음"
    fl = {"none": "수급 필터 없음", "veto20": "외국인 선물 순매수 비율이 직전 250일 하위 20%가 아님(하위 20%면 진입 금지)",
          "conf60": "외국인 선물 순매수 비율이 직전 250일 60 백분위 이상일 때만 진입"}[g["flow"]]
    return f"- 진입: {ma} 그리고 {trg} 이고 {fl} → 다음 날 09:00 시가 매수\n- 청산: {ex}\n"


def caveats_text(o):
    return ("- 선택은 DESIGN 지표만으로 했다. 그래도 56개 중 1개를 고른 것이라 DESIGN 수치는 낙관 편향이 있다(VALIDATION 이 정직한 추정).\n"
            "- VALIDATION 2025-06-09 이후는 코스피200이 크게 오른 구간이다. 최근 제외 행을 함께 보라.\n"
            "- 보유비중이 낮은 전략은 CAGR 이 매수보유보다 작게 나오는 것이 정상이다. 낮은 MDD 와 Sharpe 가 장점이다.\n"
            "- 레버리지 ETF 는 일일 2배 재조정 상품이라 며칠 보유 시 경로 의존(변동성 손실)이 있다. 실제 가격으로 계산했으므로 반영돼 있다.\n"
            "- 지수 경로 2bp 는 미니선물의 근사다(베이시스 변화·만기 롤·증거금·계약 단위 미반영).\n"
            "- 수정주가(분배 반영)라 분배락 갭은 수익률에 섞이지 않는다. 2003~2009 은 069500 초기 유동성이 낮았다.\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    o = main(dry=a.dry)
    ch = o["chosen"]
    print("선택:", ch["id"], ch["label"])
    for (k, c), per in o["res"].items():
        for p, m in per.items():
            if m:
                print(f"{k}@{c:g} {p}: tr={m['trades']} tpy={m['trades_yr']:.1f} win={m['win']:.3f} net={m['net_bp']:+.1f} "
                      f"cagr={m['cagr']:.3%} sh={m['sharpe']:.2f} mdd={m['mdd']:.2%} bh={m['bh_cagr']:.2%}")
    print("agree:", o["agree"])
