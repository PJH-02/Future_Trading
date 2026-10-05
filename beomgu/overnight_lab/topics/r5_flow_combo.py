"""R5 · flow_combo — 팀 수급 근거 + 기술 필터 1개 결합 일봉 전략 (5차 전략 탐색)

사전 고정(그리드 실행 전 기록, 2026-10-05). 설계 구간(2010~2019) 진단표 1회(수급별 상·하위 20% 다음날 수익)를 본 뒤 고정했다.
검증 구간(2020-01~)은 사양 선택에 쓰지 않는다.

자료·신호(결정일 T, 모두 T 장 마감 후 확정되는 일별 값 → T+1 시가부터 체결)
- FF  외인 코스피200 선물 순매수 계약(네이버 선물 투자자별 net_foreign)
- FS  외인 코스피 현물 순매수 비율 = 순매수 / (매수 + 매도) 금액
- PS  연기금 코스피 현물 순매수 비율(같은 정의)
- PN  프로그램 비차익 순매수 비율(보고용 단독 변형만)
- OI  개인 옵션 역발상 = −[(개인 콜 순매수 − 개인 풋 순매수) / 개인 콜·풋 매수+매도] 계약 기준(보고용 단독 변형만)
- SC  결합 점수(1개) = z60(FF) + z60(FS) + z60(PS), 각 z 는 T 제외 직전 60일 평균·SD, ±3 절단
- 각 수급의 백분위 = T 값이 T 제외 직전 250개 관측 중 차지하는 순위(최소 200개). 상위 = ≥ 0.8, 하위 = ≤ 0.2
기술 필터(규칙마다 하나, KODEX 200 수정 종가 T 기준)
- none: 필터 없음 / MA60: 롱 종가 > 60일 이평, 숏 종가 < 60일 이평
- RSI2: 롱 RSI(2, Wilder) < 20(눌림), 숏 RSI2 > 80 / BO20: 롱 종가 = 20일 종가 최고(돌파), 숏 종가 = 20일 종가 최저
- 롱 신호 = 수급 상위 20% ∧ 기술 롱 조건, 숏 신호 = 수급 하위 20% ∧ 기술 숏 조건
보유·체결(실행 규약)
- 신호 T → T+1 09:00 시가 단일가 진입, 보유 h 거래일(h = 1 이면 T+1 시가→T+1 종가, h = 5 면 T+5 종가 청산)
- 겹치는 신호는 가장 최근 신호가 이긴다(롱·숏 동시면 최근 신호 방향). 종가에서 다음 날로 넘길지(오버나잇 보유)는
  그날 장중에 아는 정보(T−1 까지의 신호)로만 정한다. 넘기지 않았는데 그날 밤 새 신호가 나오면 다음 날 시가에 다시 산다(비용 재부과).
- 롱 = KODEX 200(069500) 왕복 5bp(스트레스 7bp). 숏 = 미니 코스피200 선물 왕복 2bp(스트레스 3bp), 경로는 KODEX 200 대리.
- 보고용 크기 변형: 롱 다리를 KODEX 레버리지(122630, 5bp·스트레스 8bp)로 바꾼 판, 양방향 모두 미니 선물 2bp 판.
그리드(57개, 전부 grid.csv 기록)
- 블록1(48) 흐름 {FF, PS, SC} × 필터 {none, MA60, RSI2, BO20} × 보유 {1, 5} × 방향 {롱, 롱숏}
- 블록2(3)  단독 수급 {FS, PN, OI} × 필터 none × 보유 1 × 롱숏
- 블록3(2)  FF 숏 전용(팀 근거: 매도 측이 더 강함) × 필터 {none, MA60} × 보유 1
- 블록4(4)  거부권: 평소 KODEX 200 상시 보유, 수급 {FF, SC} 하위 20% 다음 날은 {쉼, 미니 선물 숏}(시가 매도 → 종가 재매수)
선택 규칙(설계 구간 값만): 설계 연 거래 ≥ 20회 ∧ 스트레스 비용 거래당 순평균 > 0 인 변형 중 설계 샤프(기본 비용) 최대.
 동률이면 번호 작은 것. 2·3위 변형의 검증 성과도 함께 보고한다.
구간: 설계 = 공통 신호 시작일 ~ 2019-12-31(모든 수급 백분위가 유효한 첫날부터; 명목 2010-01-01~), 검증 = 2020-01-01 ~ 자료 끝,
 검증(최근 제외) = 2020-01-01 ~ 2025-06-05, 연도별. 선택 규칙이 현물 수급(FS·PS·PN)만 쓰면 2006~2009 도 추가 표본 외로 보고.
지표: 거래 = 진입~청산 한 덩어리, 승률 = 거래 순수익 > 0 비율, CAGR = 일별 전략 순수익(무포지션 0) 복리 연환산(252일),
 샤프 = 평균/SD·√252, MDD, 칼마, 거래 순수익 NW t(lag 5). 기준선 KODEX 200 매수 보유(종가→종가, 비용 없음).
실행: python topics/r5_flow_combo.py [--no-ledger]
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import data  # noqa: E402
import engine as E  # noqa: E402

TOPIC = "R5_FLOW_COMBO"
OUTD = data.OUT / "round5_strategy" / "flow_combo"
DESIGN_END = "2019-12-31"
VALID_START = "2020-01-01"
EXREC_END = "2025-06-05"
ANN = 252
TOP, BOT = 0.8, 0.2
MIN_TPY = 20
COST = {"etf": 5.0, "lev": 5.0, "fut": 2.0}
STRESS = {"etf": 7.0, "lev": 8.0, "fut": 3.0}


# ---------------------------------------------------------------- 신호
def pct_past(x, win=250, minp=200):
    x = x.dropna()
    a = x.to_numpy(float)
    out = np.full(len(a), np.nan)
    for i in range(minp, len(a)):
        w = a[max(0, i - win):i]
        out[i] = ((w < a[i]).sum() + 0.5 * (w == a[i]).sum()) / len(w)
    return pd.Series(out, index=x.index)


def z_past(x, win=60):
    x = x.dropna()
    m = x.shift(1).rolling(win, min_periods=win).mean()
    s = x.shift(1).rolling(win, min_periods=win).std()
    return ((x - m) / s).clip(-3, 3)


def rsi_wilder(c, n=2):
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn)


def flows():
    fut = data.naver_investor("fut")
    k = data.naver_investor("kospi")
    prog = data.naver_program()
    o = data.krx_options()
    g = o[o["investor"] == "individual"].pivot_table(index="date", columns="cp", values=["buy_qty", "sell_qty"], aggfunc="sum")
    ind_bull = ((g[("buy_qty", "C")] - g[("sell_qty", "C")]) - (g[("buy_qty", "P")] - g[("sell_qty", "P")])) / (
        g[("buy_qty", "C")] + g[("sell_qty", "C")] + g[("buy_qty", "P")] + g[("sell_qty", "P")])
    raw = {
        "FF": fut["net_foreign"],
        "FS": k["net_foreign"] / (k["buyv_foreign"] + k["sellv_foreign"]),
        "PS": (k["net_pension"] / (k["buyv_pension"] + k["sellv_pension"])).where(k["buyv_pension"] + k["sellv_pension"] > 0),
        "PN": prog["nonarb_net"] / (prog["nonarb_buy"] + prog["nonarb_sell"]),
        "OI": -ind_bull,
    }
    sc = pd.concat([z_past(raw[n]) for n in ("FF", "FS", "PS")], axis=1).dropna().sum(axis=1)
    raw["SC"] = sc
    pct = {n: pct_past(v) for n, v in raw.items()}
    return raw, pct


def technicals(c):
    ma60 = c.rolling(60, min_periods=60).mean()
    rsi2 = rsi_wilder(c, 2)
    hi20, lo20 = c.rolling(20, min_periods=20).max(), c.rolling(20, min_periods=20).min()
    ok = ma60.notna()
    T = {"none": (pd.Series(True, index=c.index), pd.Series(True, index=c.index)),
         "MA60": (c > ma60, c < ma60), "RSI2": (rsi2 < 20, rsi2 > 80), "BO20": (c >= hi20, c <= lo20)}
    return {k: (a & ok, b & ok) for k, (a, b) in T.items()}, rsi2, ma60


# ---------------------------------------------------------------- 그리드
def grid():
    V = []
    for fl in ("FF", "PS", "SC"):
        for te in ("none", "MA60", "RSI2", "BO20"):
            for h in (1, 5):
                for di in ("L", "LS"):
                    V.append(dict(block=1, flow=fl, tech=te, hold=h, dir=di, mode="hold"))
    for fl in ("FS", "PN", "OI"):
        V.append(dict(block=2, flow=fl, tech="none", hold=1, dir="LS", mode="hold"))
    for te in ("none", "MA60"):
        V.append(dict(block=3, flow="FF", tech=te, hold=1, dir="S", mode="hold"))
    for fl in ("FF", "SC"):
        for act in ("flat", "short"):
            V.append(dict(block=4, flow=fl, tech="none", hold=1, dir="base-L", mode=f"veto_{act}"))
    for i, v in enumerate(V, 1):
        v["id"] = f"V{i:02d}"
        v["name"] = (f"{v['flow']}+{v['tech']} h{v['hold']} {v['dir']}" if v["mode"] == "hold"
                     else f"상시롱 · {v['flow']} 하위20% 다음날 {'쉼' if v['mode'] == 'veto_flat' else '숏'}")
    return V


def signal_series(v, pct, tech, cal):
    p = pct[v["flow"]].reindex(cal)
    tl, ts = tech[v["tech"]]
    valid = p.notna() & tl.notna()
    if v["mode"] != "hold":
        return (p <= BOT).astype(float).where(p.notna()), valid
    long_ = (p >= TOP) & tl
    short = (p <= BOT) & ts
    s = pd.Series(0.0, index=cal)
    if v["dir"] in ("L", "LS"):
        s[long_] = 1.0
    if v["dir"] in ("S", "LS"):
        s[short] = -1.0
    return s.where(valid), valid


def positions(s, h, mode):
    """s: 결정일 T 신호(±1/0, NaN = 계산 불가 → 0). 반환 p_day[i](i 일 시가→종가), p_night[i](i 종가→i+1 시가)."""
    a = np.nan_to_num(s.to_numpy(float))
    n = len(a)
    pd_, pn_ = np.zeros(n), np.zeros(n)
    if mode == "hold":
        def latest(lo, hi):                       # a[lo..hi] 중 가장 최근 0 아닌 값
            for j in range(hi, lo - 1, -1):
                if j >= 0 and a[j] != 0:
                    return a[j]
            return 0.0
        for i in range(n):
            pd_[i] = latest(i - h, i - 1)
            q = latest(i + 1 - h, i - 1) if h > 1 else 0.0
            pn_[i] = pd_[i] if (pd_[i] != 0 and q == pd_[i]) else 0.0
    else:
        act = 0.0 if mode == "veto_flat" else -1.0
        veto = np.r_[0.0, a[:-1]]                 # T 하위 20% → T+1 장중
        pd_ = np.where(veto == 1, act, 1.0)
        pn_ = np.ones(n)
    return pd_, pn_


def simulate(p_day, p_night, R, cl, cs):
    """R: dict(l_oc, l_on, s_oc, s_on) numpy. cl/cs: 롱·숏 왕복 비용 bp. 일별 순수익과 거래 목록."""
    n = len(p_day)
    daily = np.zeros(n)
    trades = []
    cur = None                                     # [side, entry_i, growth, cost]

    def tcost(a, b):
        return (abs(max(a, 0) - max(b, 0)) * cl / 2 + abs(max(-a, 0) - max(-b, 0)) * cs / 2) / 1e4

    def seg(p, i, kind):
        if p == 0:
            return 0.0
        if p > 0:
            r = R["l_oc"][i] if kind == "d" else R["l_on"][i]
        else:
            r = R["s_oc"][i] if kind == "d" else R["s_on"][i]
        return p * r if np.isfinite(r) else 0.0

    def switch(old, new, i, exit_i):
        nonlocal cur
        if old == new:
            return
        c = tcost(old, new)
        if old != 0 and cur is not None:
            cur[3] += c if new == 0 else tcost(old, 0)
            trades.append(dict(side=int(np.sign(cur[0])), entry_i=cur[1], exit_i=exit_i, gross=cur[2] - 1, net=cur[2] - 1 - cur[3]))
            cur = None
        if new != 0:
            cur = [new, i, 1.0, c if old == 0 else tcost(0, new)]

    prev_night = 0.0
    for i in range(n):
        rn = seg(prev_night, i - 1, "n") if i > 0 else 0.0
        if cur is not None:
            cur[2] *= 1 + rn
        c_open = tcost(prev_night, p_day[i])
        switch(prev_night, p_day[i], i, i)
        rd = seg(p_day[i], i, "d")
        if cur is not None:
            cur[2] *= 1 + rd
        c_close = tcost(p_day[i], p_night[i])
        switch(p_day[i], p_night[i], i, i)
        daily[i] = (1 + rn) * (1 + rd) - 1 - c_open - c_close
        prev_night = p_night[i]
    if cur is not None:                            # 자료 끝 미청산: 마지막 종가 청산 + 비용
        cur[3] += tcost(cur[0], 0)
        trades.append(dict(side=int(np.sign(cur[0])), entry_i=cur[1], exit_i=n - 1, gross=cur[2] - 1, net=cur[2] - 1 - cur[3]))
    expo = (p_day != 0) | (np.r_[0.0, p_night[:-1]] != 0)
    return daily, pd.DataFrame(trades, columns=["side", "entry_i", "exit_i", "gross", "net"]), expo


def metrics(daily, trades, expo, cal, a, b, bench_cc):
    m = (cal >= pd.Timestamp(a)) & (cal <= pd.Timestamp(b))
    d = daily[m]
    nd = len(d)
    if nd == 0:
        return {}
    t = trades[(trades["entry_i"] >= np.argmax(m)) & (trades["entry_i"] <= np.where(m)[0][-1])] if len(trades) else trades
    eq = np.cumprod(1 + d)
    yrs = nd / ANN
    cagr = eq[-1] ** (1 / yrs) - 1
    mdd = (eq / np.maximum.accumulate(np.r_[1.0, eq])[1:] - 1).min()
    mdd = min(mdd, 0.0)
    sd = d.std(ddof=1)
    beq = np.cumprod(1 + np.nan_to_num(bench_cc[m]))
    bmdd = min((beq / np.maximum.accumulate(np.r_[1.0, beq])[1:] - 1).min(), 0.0)
    return {"start": str(cal[m][0].date()), "end": str(cal[m][-1].date()), "days": int(nd), "trades": int(len(t)),
            "tpy": len(t) / yrs, "exposure": float(expo[m].mean()), "win": float((t["net"] > 0).mean()) if len(t) else np.nan,
            "gross_bp": float(t["gross"].mean() * 1e4) if len(t) else np.nan, "net_bp": float(t["net"].mean() * 1e4) if len(t) else np.nan,
            "t_net": float(E.nw_t(t["net"].to_numpy())) if len(t) >= 10 else np.nan,
            "cagr": float(cagr), "sharpe": float(d.mean() / sd * np.sqrt(ANN)) if sd > 0 else np.nan, "mdd": float(mdd),
            "calmar": float(cagr / -mdd) if mdd < 0 else np.nan,
            "bh_cagr": float(beq[-1] ** (1 / yrs) - 1), "bh_mdd": float(bmdd)}


# ---------------------------------------------------------------- 실행
def legs(cal, long_sym="069500"):
    px = data.price("069500").reindex(cal)
    lp = data.price(long_sym).reindex(cal)
    return {"l_oc": (lp["close"] / lp["open"] - 1).to_numpy(), "l_on": (lp["open"].shift(-1) / lp["close"] - 1).to_numpy(),
            "s_oc": (px["close"] / px["open"] - 1).to_numpy(), "s_on": (px["open"].shift(-1) / px["close"] - 1).to_numpy()}


def run_variant(v, pct, tech, cal, R, cl, cs):
    s, _ = signal_series(v, pct, tech, cal)
    pdy, pnt = positions(s, v["hold"], v["mode"])
    daily, tr, expo = simulate(pdy, pnt, R, cl, cs)
    return daily, tr, expo, pdy, pnt, s


def design_diag(pct, cal, px, d_end=DESIGN_END):
    """그리드 고정 전 본 설계 구간(2010~2019) 진단: 수급 백분위 구간별 다음날 시가→종가, T+1 시가→T+5 종가(비용 전)."""
    o, c = px["open"].reindex(cal), px["close"].reindex(cal)
    f1 = c.shift(-1) / o.shift(-1) - 1
    f5 = c.shift(-5) / o.shift(-1) - 1
    rows = []
    for n in ("FF", "FS", "PS", "PN", "OI", "SC"):
        d = pd.DataFrame({"p": pct[n].reindex(cal), "f1": f1, "f5": f5}).loc[:d_end].dropna()
        d = d[d.index >= "2010-01-01"]
        r = {"수급": n}
        for lab, m in (("하위20%", d.p <= BOT), ("중간", (d.p > BOT) & (d.p < TOP)), ("상위20%", d.p >= TOP)):
            r[f"{lab} n"] = int(m.sum())
            r[f"{lab} 1일bp"] = d.f1[m].mean() * 1e4
            r[f"{lab} 5일bp"] = d.f5[m].mean() * 1e4
        rows.append(r)
    allr = pd.DataFrame({"f1": f1, "f5": f5}).loc["2010-01-01":d_end].dropna()
    return pd.DataFrame(rows), allr.f1.mean() * 1e4, allr.f5.mean() * 1e4


def fut_proxy_check(cal):
    """숏 다리 대리(KODEX 200 경로) 점검: fchart 코스피200 선물 연결 일봉(2014-07~)과 시가→종가·오버나잇 비교."""
    f, k = data.price("FUT"), data.price("069500")
    d = pd.DataFrame({"f_oc": f.close / f.open - 1, "k_oc": k.close / k.open - 1, "f_on": f.open.shift(-1) / f.close - 1,
                      "k_on": k.open.shift(-1) / k.close - 1}).dropna()
    out = []
    for lab, a, b in (("2014-07~2023-07-28(선물 09:00 개장)", "2014-07-08", "2023-07-28"), ("2023-07-31~2025-06-05(08:45 개장)", "2023-07-31", "2025-06-05"),
                      ("2025-06-09~(야간 시장)", "2025-06-09", None)):
        x = d.loc[a:b]
        out.append(f"| {lab} | {len(x)} | {x.f_oc.corr(x.k_oc):.3f} | {x.f_oc.mean() * 1e4:+.2f} | {x.k_oc.mean() * 1e4:+.2f} | "
                   f"{x.f_on.mean() * 1e4:+.2f} | {x.k_on.mean() * 1e4:+.2f} |")
    return ["| 구간 | 일수 | 시가→종가 상관 | 선물 시가→종가 평균bp | KODEX 시가→종가 평균bp | 선물 오버나잇bp | KODEX 오버나잇bp |",
            "|---|---|---|---|---|---|---|", *out]


def pf(x, k=1):
    return "-" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x * 100:.{k}f}%"


def nf(x, fmt="{:+.2f}"):
    return "-" if x is None or (isinstance(x, float) and np.isnan(x)) else fmt.format(x)


def row_md(lab, m):
    return (f"| {lab} | {m['start']}~{m['end']} | {m['trades']} | {m['tpy']:.1f} | {pf(m['exposure'])} | {pf(m['win'])} | "
            f"{nf(m['net_bp'])} | {nf(m['t_net'])} | {pf(m['cagr'])} | {nf(m['sharpe'])} | {pf(m['mdd'])} | {nf(m['calmar'])} | "
            f"{pf(m['bh_cagr'])} / {pf(m['bh_mdd'])} |")


HDR = ("| 구간 | 기간 | 거래 | 연거래 | 노출 | 승률 | 거래당 순bp | NW t | CAGR | 샤프 | MDD | 칼마 | KODEX200 보유 CAGR/MDD |\n"
       "|---|---|---|---|---|---|---|---|---|---|---|---|---|")


def main():
    OUTD.mkdir(parents=True, exist_ok=True)
    px = data.price("069500")
    cal = px.index[px.index >= "2005-01-01"]
    c = px["close"].reindex(cal)
    bench_cc = (c / c.shift(1) - 1).to_numpy()
    raw, pct = flows()
    tech, rsi2, ma60 = technicals(c)
    V = grid()

    # 공통 설계 시작일: 블록1~4 에 쓰는 모든 수급 백분위가 유효한 첫날
    starts = {n: pct[n].dropna().index.min() for n in pct}
    d_start = max(starts.values())
    d_start = cal[cal >= d_start][0]
    last = min(pct[n].dropna().index.max() for n in ("FF", "FS", "PS"))
    end = cal[-1]
    W = {"설계": (d_start, DESIGN_END), "검증": (VALID_START, end), "검증(2025-06-05까지)": (VALID_START, EXREC_END),
         "검증 2025-06-09~": ("2025-06-09", end)}

    R1 = legs(cal)
    Rlev = legs(cal, "122630")
    rows, runs = [], {}
    for v in V:
        out = {}
        for tag, cl, cs in (("base", COST["etf"], COST["fut"]), ("stress", STRESS["etf"], STRESS["fut"])):
            daily, tr, expo, pdy, pnt, s = run_variant(v, pct, tech, cal, R1, cl, cs)
            out[tag] = (daily, tr, expo, pdy, pnt, s)
        runs[v["id"]] = out
        md_ = metrics(*out["base"][:3], cal, *W["설계"], bench_cc)
        ms_ = metrics(*out["stress"][:3], cal, *W["설계"], bench_cc)
        mv_ = metrics(*out["base"][:3], cal, *W["검증"], bench_cc)
        r = {**{k: v[k] for k in ("id", "block", "name", "flow", "tech", "hold", "dir", "mode")},
             **{f"d_{k}": md_[k] for k in ("trades", "tpy", "exposure", "win", "gross_bp", "net_bp", "t_net", "cagr", "sharpe", "mdd", "calmar")},
             "d_net_bp_stress": ms_["net_bp"],
             **{f"v_{k}": mv_[k] for k in ("trades", "tpy", "win", "net_bp", "t_net", "cagr", "sharpe", "mdd")}}
        rows.append(r)
    G = pd.DataFrame(rows)
    G["eligible"] = (G["d_tpy"] >= MIN_TPY) & (G["d_net_bp_stress"] > 0)
    rank = G[G["eligible"]].sort_values(["d_sharpe", "id"], ascending=[False, True])
    G["design_rank"] = G["id"].map({k: i + 1 for i, k in enumerate(rank["id"])})
    G.to_csv(OUTD / "grid.csv", index=False, encoding="utf-8-sig", float_format="%.6g")
    if rank.empty:
        print("적격 변형 없음")
        print(G.sort_values("d_sharpe", ascending=False).head(15).to_string())
        return
    top = rank["id"].tolist()[:3]
    chosen = top[0]
    vmap = {v["id"]: v for v in V}
    cv = vmap[chosen]
    print(G.sort_values("d_sharpe", ascending=False)[["id", "name", "d_tpy", "d_win", "d_net_bp", "d_net_bp_stress", "d_cagr", "d_sharpe", "eligible"]].head(20).to_string())
    print("chosen", chosen, cv["name"], "top3", top)

    # ---------------- 선택 사양 상세
    daily, tr, expo, pdy, pnt, s = runs[chosen]["base"]
    dailyS, trS, expoS = runs[chosen]["stress"][:3]
    per = {w: metrics(daily, tr, expo, cal, a, b, bench_cc) for w, (a, b) in W.items()}
    perS = {w: metrics(dailyS, trS, expoS, cal, a, b, bench_cc) for w, (a, b) in W.items()}
    yrs = sorted(set(cal[cal >= d_start].year))
    peryear = {y: metrics(daily, tr, expo, cal, f"{max(y, d_start.year)}-01-01" if y > d_start.year else d_start, f"{y}-12-31", bench_cc) for y in yrs}

    # 크기·상품 변형(보고만)
    dL, tL, eL, *_ = run_variant(cv, pct, tech, cal, Rlev, COST["lev"], COST["fut"])
    dLs, tLs, eLs, *_ = run_variant(cv, pct, tech, cal, Rlev, STRESS["lev"], STRESS["fut"])
    dF, tF, eF, *_ = run_variant(cv, pct, tech, cal, R1, COST["fut"], COST["fut"])
    lev = {w: metrics(dL, tL, eL, cal, a, b, bench_cc) for w, (a, b) in W.items()}
    levS = {w: metrics(dLs, tLs, eLs, cal, a, b, bench_cc) for w, (a, b) in W.items()}
    futp = {w: metrics(dF, tF, eF, cal, a, b, bench_cc) for w, (a, b) in W.items()}
    # 방향 분해(롱숏이면)
    side = {}
    if cv["dir"] == "LS":
        for dname in ("L", "S"):
            v2 = {**cv, "dir": dname}
            dd, tt, ee, *_ = run_variant(v2, pct, tech, cal, R1, COST["etf"], COST["fut"])
            side[dname] = {w: metrics(dd, tt, ee, cal, a, b, bench_cc) for w, (a, b) in W.items()}
    # 2·3위 검증
    others = {}
    for oid in top[1:]:
        dd, tt, ee = runs[oid]["base"][:3]
        others[oid] = {w: metrics(dd, tt, ee, cal, a, b, bench_cc) for w, (a, b) in W.items()}
    # 2006~2009 추가 표본 외(현물 수급만 쓰는 사양일 때)
    extra = None
    if cv["flow"] in ("FS", "PS", "PN") and cv["mode"] == "hold":
        e0 = pct[cv["flow"]].dropna().index.min()
        extra = metrics(daily, tr, expo, cal, max(e0, pd.Timestamp("2006-01-01")), "2009-12-31", bench_cc)
    # 검증 전체 그리드 분포(선택에 쓰지 않음, 정직성 점검)
    el = G[G["eligible"]]
    dist = {"n_eligible": int(len(el)), "val_sharpe_median": float(el["v_sharpe"].median()), "val_pos_sharpe_share": float((el["v_sharpe"] > 0).mean()),
            "chosen_val_sharpe_rank": int((el["v_sharpe"] > G.set_index("id").loc[chosen, "v_sharpe"]).sum() + 1)}

    # ---------------- 파일
    trd = tr.copy()
    trd["entry_date"] = cal[trd["entry_i"].to_numpy()]
    trd["exit_date"] = cal[trd["exit_i"].to_numpy()]
    trd["side"] = trd["side"].map({1: "long KODEX200", -1: "short mini-fut(KODEX path)"})
    trd["period"] = np.where(trd["entry_date"] < pd.Timestamp(VALID_START), "design", "validation")
    trd["gross_bp"], trd["net_bp"] = trd["gross"] * 1e4, trd["net"] * 1e4
    trd = trd[trd["entry_date"] >= d_start]
    trd[["entry_date", "exit_date", "side", "period", "gross_bp", "net_bp"]].to_csv(OUTD / "trades.csv", index=False, encoding="utf-8-sig", float_format="%.4f")
    sig_raw = pct[cv["flow"]].reindex(cal)
    dr = pd.DataFrame({"kodex_open": px["open"].reindex(cal), "kodex_close": c, f"pct_{cv['flow']}": sig_raw, "rsi2": rsi2, "ma60": ma60,
                       "signal_T": s, "pos_intraday": pdy, "pos_overnight": pnt, "net_ret_5bp": daily, "net_ret_stress": dailyS,
                       "net_ret_lev122630": dL, "net_ret_fut2bp": dF}, index=cal)
    dr = dr[dr.index >= d_start]
    dr["period"] = np.where(dr.index < pd.Timestamp(VALID_START), "design", "validation")
    dr.to_csv(OUTD / "daily_returns.csv", encoding="utf-8-sig", float_format="%.8g")
    chart(dr, d_start, cv)

    # ---------------- 보고서
    gsorted = G.sort_values(["eligible", "d_sharpe"], ascending=[False, False])
    gtab = ["| 번호 | 변형 | 설계 연거래 | 설계 승률 | 설계 순bp | 스트레스 순bp | 설계 CAGR | 설계 샤프 | 설계 MDD | 적격 | 설계 순위 |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in gsorted.iterrows():
        gtab.append(f"| {r['id']} | {r['name']} | {r['d_tpy']:.1f} | {pf(r['d_win'])} | {nf(r['d_net_bp'])} | {nf(r['d_net_bp_stress'])} | "
                    f"{pf(r['d_cagr'])} | {nf(r['d_sharpe'])} | {pf(r['d_mdd'])} | {'O' if r['eligible'] else '-'} | "
                    f"{'' if pd.isna(r['design_rank']) else int(r['design_rank'])} |")
    md = [f"# R5 · flow_combo — 수급 + 기술 필터 1개 결합 일봉 전략\n",
          "```\n" + __doc__.strip() + "\n```\n",
          f"자료: KODEX 200 일봉(수정주가) {cal[0].date()}~{end.date()}, 수급 마지막 확정일 FF {pct['FF'].dropna().index.max().date()}·현물 {last.date()}. "
          f"공통 설계 시작일 {d_start.date()}(선물 투자자별 2010-09-20 시작 + 백분위 번인 200일).\n",
          f"## 1. 선택 사양: {chosen} · {cv['name']}\n",
          f"- 규칙: 결정일 T 장 마감 후 {cv['flow']} 수급 백분위(직전 250일 대비)와 기술 필터 {cv['tech']} 로 신호 → T+1 09:00 시가 진입, "
          f"{cv['hold']}거래일 보유 후 종가 청산. 방향 {cv['dir']}.",
          f"- 선택 근거: 설계 적격 {len(el)}개 중 설계 샤프 1위(2위 {top[1] if len(top) > 1 else '-'}, 3위 {top[2] if len(top) > 2 else '-'}). 검증 구간 값은 선택에 쓰지 않았다.\n",
          "### 1-1. 구간별 성과 (기본 비용: KODEX 5bp·미니 선물 2bp)\n", HDR, *[row_md(w, m) for w, m in per.items()], "",
          "### 1-2. 스트레스 비용 (KODEX 7bp·미니 선물 3bp)\n", HDR, *[row_md(w, m) for w, m in perS.items()], "",
          "### 1-3. 연도별 (기본 비용)\n", HDR.replace("| 구간 |", "| 연도 |"), *[row_md(str(y), m) for y, m in peryear.items() if m], ""]
    if extra:
        md += ["### 1-4. 추가 표본 외 2006~2009 (현물 수급만 쓰므로 계산 가능)\n", HDR, row_md("2006~2009", extra), ""]
    md += ["### 1-5. 크기·상품 변형 (같은 신호, 선택에 쓰지 않음)\n",
           "KODEX 레버리지(122630) 롱 다리, 5bp:\n", HDR, *[row_md(w, m) for w, m in lev.items()], "",
           "KODEX 레버리지 롱 다리, 스트레스 8bp:\n", HDR, *[row_md(w, m) for w, m in levS.items()], "",
           "양방향 미니 선물 2bp(KODEX 200 경로 대리):\n", HDR, *[row_md(w, m) for w, m in futp.items()], ""]
    if side:
        md += ["### 1-6. 방향 분해 (기본 비용)\n"]
        for dname, mm in side.items():
            md += [f"{'롱 다리만' if dname == 'L' else '숏 다리만'}:\n", HDR, *[row_md(w, m) for w, m in mm.items()], ""]
    md += ["## 2. 설계 2·3위 변형의 검증 (강건성, 선택 아님)\n"]
    for oid, mm in others.items():
        md += [f"{oid} · {vmap[oid]['name']}:\n", HDR, *[row_md(w, m) for w, m in mm.items()], ""]
    md += ["## 3. 그리드 전체 (설계 지표, 57개)\n", "\n".join(gtab), "",
           f"참고(선택에 쓰지 않음): 적격 {dist['n_eligible']}개의 검증 샤프 중앙값 {dist['val_sharpe_median']:+.2f}, 검증 샤프 > 0 비율 "
           f"{dist['val_pos_sharpe_share'] * 100:.0f}%, 선택 사양의 검증 샤프 순위 {dist['chosen_val_sharpe_rank']}/{dist['n_eligible']}. 모든 변형의 검증 값은 grid.csv 의 v_ 열.\n"]
    diag, u1, u5 = design_diag(pct, cal, px)
    dtab = ["| 수급 | 하위20% n | 하위 1일bp | 하위 5일bp | 중간 1일bp | 중간 5일bp | 상위20% n | 상위 1일bp | 상위 5일bp |", "|---|---|---|---|---|---|---|---|---|"]
    for _, r in diag.iterrows():
        dtab.append(f"| {r['수급']} | {r['하위20% n']} | {r['하위20% 1일bp']:+.2f} | {r['하위20% 5일bp']:+.2f} | {r['중간 1일bp']:+.2f} | {r['중간 5일bp']:+.2f} | "
                    f"{r['상위20% n']} | {r['상위20% 1일bp']:+.2f} | {r['상위20% 5일bp']:+.2f} |")
    md += ["### 3-1. 그리드 고정 전에 본 설계 구간 진단 (2010~2019, 비용 전, KODEX 200)\n",
           f"1일 = T+1 시가→T+1 종가, 5일 = T+1 시가→T+5 종가. 무조건 평균 1일 {u1:+.2f}bp, 5일 {u5:+.2f}bp.\n", "\n".join(dtab), "",
           "### 3-2. 숏 다리 대리 점검 (선물 연결 일봉 vs KODEX 200)\n", "\n".join(fut_proxy_check(cal)), "",
           "선물 연결은 근월물 기준이며 시가 시각이 2023-07-31 부터 08:45 로 바뀌었다. 장중 상관이 높아 대리는 방향·크기 모두 근사로 쓸 만하나, "
           "선물 시가→종가 평균이 KODEX 보다 약간 높아(베이시스 수렴 등) 숏 다리는 실제로 대리보다 조금 불리할 수 있다.\n"]
    vm = per["검증"]
    md += ["## 4. 목표 점검\n",
           f"- 검증(2020-01~{end.date()}) 기본 비용: CAGR {pf(vm['cagr'])}, 승률 {pf(vm['win'])}, 샤프 {nf(vm['sharpe'])}, MDD {pf(vm['mdd'])}, "
           f"연 {vm['tpy']:.1f}회. 목표(CAGR ≥ 약 10% 또는 승률 ≥ 51%) {'충족' if (vm['cagr'] >= 0.095 or vm['win'] >= 0.51) else '미충족'}.",
           f"- 최근 랠리 제외(2020-01~2025-06-05): CAGR {pf(per['검증(2025-06-05까지)']['cagr'])}, 승률 {pf(per['검증(2025-06-05까지)']['win'])}.",
           f"- 레버리지 롱 다리판 검증 CAGR {pf(lev['검증']['cagr'])}, MDD {pf(lev['검증']['mdd'])}."]
    if side:
        md.append(f"- 방향 분해: 롱 다리만 검증 CAGR {pf(side['L']['검증']['cagr'])}·승률 {pf(side['L']['검증']['win'])}, "
                  f"숏 다리만 검증 CAGR {pf(side['S']['검증']['cagr'])}·승률 {pf(side['S']['검증']['win'])} → 검증 부진의 출처를 보여 준다.")
    for oid, mm in others.items():
        ok = mm["검증"]["cagr"] >= 0.095 or mm["검증"]["win"] >= 0.51
        md.append(f"- 설계 {top.index(oid) + 1}위 {oid}({vmap[oid]['name']}): 검증 CAGR {pf(mm['검증']['cagr'])}, 승률 {pf(mm['검증']['win'])}, "
                  f"샤프 {nf(mm['검증']['sharpe'])}, 최근 제외 CAGR {pf(mm['검증(2025-06-05까지)']['cagr'])} → 목표 수준 {'도달' if ok else '미달'}. "
                  "사전 규칙상 선택 사양이 아니므로 검증을 보고 바꾸면 사후 선택이 된다. 쓰려면 포워드(페이퍼) 기록으로 따로 확인해야 한다.")
    md += ["",
           "## 5. 실행 메모\n",
           "- 신호는 T 장 마감 뒤 확정되는 일별 수급(네이버 투자자별·프로그램 일별, 선물 투자자별)만 쓴다 → T+1 09:00 시가 단일가(동시호가) 주문.",
           "- 청산은 보유 마지막 날 15:20~15:30 종가 단일가. 오버나잇 보유 여부는 전날까지 신호로 정하므로 장중에 확정 가능.",
           "- 숏 다리는 미니 코스피200 선물(파생 계좌 필요)이며 백테스트는 KODEX 200 시가→종가 경로로 대리했다(2023-07-31 이후 선물은 08:45 개장, 2025-06-09 이후 야간 시장).",
           "- 산출: grid.csv(57개 전부, 설계 d_·검증 v_), daily_returns.csv(선택 사양 일별), trades.csv(선택 사양 거래), equity.png."]
    (OUTD / "report.md").write_text("\n".join(md), encoding="utf-8")

    # ---------------- 장부
    led = []
    for _, r in G.iterrows():
        led.append({"rule": f"{r['id']} {r['name']}", "window": "설계", "cost_bp": "5/2", "asset": "069500(+mini fut proxy)", "days": np.nan,
                    "trades": r["d_trades"], "exposure": r["d_exposure"], "gross_bp": r["d_gross_bp"], "net_bp": r["d_net_bp"], "t_net": r["d_t_net"],
                    "win": r["d_win"], "sharpe": r["d_sharpe"], "cagr": r["d_cagr"], "mdd": r["d_mdd"], "calmar": r["d_calmar"]})
    for w, m in per.items():
        led.append({"rule": f"{chosen} {cv['name']} [선택]", "window": w, "cost_bp": "5/2", "asset": "069500(+mini fut proxy)", **{k: m[k] for k in (
            "days", "trades", "exposure", "gross_bp", "net_bp", "t_net", "win", "sharpe", "cagr", "mdd", "calmar")}})
    for w, m in perS.items():
        led.append({"rule": f"{chosen} {cv['name']} [선택]", "window": w, "cost_bp": "7/3", "asset": "069500(+mini fut proxy)", **{k: m[k] for k in (
            "days", "trades", "exposure", "gross_bp", "net_bp", "t_net", "win", "sharpe", "cagr", "mdd", "calmar")}})
    for oid, mm in others.items():
        for w, m in mm.items():
            led.append({"rule": f"{oid} {vmap[oid]['name']} [설계 차순위]", "window": w, "cost_bp": "5/2", "asset": "069500(+mini fut proxy)", **{k: m[k] for k in (
                "days", "trades", "exposure", "gross_bp", "net_bp", "t_net", "win", "sharpe", "cagr", "mdd", "calmar")}})
    spec = {"family": "flow_combo", "grid": len(V), "selection": "design tpy>=20 & stress net>0, max design Sharpe", "chosen": chosen,
            "chosen_name": cv["name"], "design": [str(d_start.date()), DESIGN_END], "validation": [VALID_START, str(end.date())]}
    if "--no-ledger" not in sys.argv:
        E.ledger_append(data.OUT, TOPIC, spec, pd.DataFrame(led), note="5차 전략 탐색")
    summary = {"chosen": chosen, "name": cv["name"], "top3": top, "per": per, "stress": perS, "lev": lev, "levS": levS, "fut2": futp,
               "side": side, "others": others, "extra": extra, "dist": dist, "years": {str(k): v for k, v in peryear.items()}}
    (OUTD / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print(json.dumps({"per": per, "lev": {k: {kk: lev[k][kk] for kk in ("cagr", "mdd", "win", "sharpe")} for k in lev}, "dist": dist}, ensure_ascii=False, indent=1, default=float))


def chart(dr, d_start, cv):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(10, 4.8))
    eq = (1 + dr["net_ret_5bp"]).cumprod()
    eqL = (1 + dr["net_ret_lev122630"].fillna(0)).cumprod()
    bh = dr["kodex_close"] / dr["kodex_close"].iloc[0]
    ax.plot(eq.index, eq, color="#2a78d6", lw=1.6, label=f"{cv['id']} 전략(KODEX 5bp·선물 2bp)")
    ax.plot(eqL.index, eqL, color="#eb6834", lw=1.1, label="같은 신호, 롱 다리 KODEX 레버리지")
    ax.plot(bh.index, bh, color="#888888", lw=1.0, label="KODEX 200 보유")
    ax.axvspan(pd.Timestamp(VALID_START), dr.index[-1], color="#f2f2f2", zorder=0)
    ax.text(pd.Timestamp(VALID_START), ax.get_ylim()[1], " 검증", va="top", fontsize=9, color="#555555")
    ax.set_yscale("log")
    from matplotlib.ticker import FuncFormatter, LogLocator
    ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 3.0, 5.0)))
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs=()))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.1f}x"))
    ax.set_title(f"{cv['id']} {cv['name']} — 누적(로그)", fontsize=11)
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTD / "equity.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
