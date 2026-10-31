# -*- coding: utf-8 -*-
"""16차 스캔 · 관점 flows: 지금까지 쓰지 않은 수급 자료로 새 신호 탐색 (봉인 2024-12-30, 선별 단계)

실행: PYTHONUTF8=1 python "<code>/topics/r16s_flows.py"

[봉인] 모든 가격·수급·기준 흐름 파일은 읽은 직후 sealed() 로 2024-12-30 이후 행을 버리고 assert 한다. 2025 이후 수익 계산·출력 없음.
[구간] 세그먼트 배열 길이 2n: seg[2t] = t−1 종가→t 시가(밤), seg[2t+1] = t 시가→t 종가(낮).
  신호 s[T] 는 T 장 마감 뒤 확정. 지평:
    D = T+1 시가→T+1 종가(낮) / N = T+1 종가→T+2 시가(밤) / F = T+1 시가→T+2 시가 / 3·5·10 = T+1 시가→T+k 종가(스윙, 겹침은 연장, 마지막 신호 우선)
    N0 = T 종가→T+1 시가: J·후보 밤 다리. IBS 는 T 15:30 ETF 종가 확정 → 15:45 선물 종가 체결, 수급 조건은 T−1 확정치만.
  군: A KQ선물 투자자별 / B 선물×현물 동시·불일치 / C 가격 효과 제거 잔차·괴리 / D K200선물 투자자별(계약·금액) / E 교차 시장 / F 동시 순매도 다른 형태 / G 옵션 투자자별·sunny 재현 /
    H woohyun Flow / I 프로그램 차익 / J IBS×수급 / K ETF 투자자별(깃 밖) / S 현물 기준선 / L 매개변수 지도 / M 지속성·가격 상태 / P 후보 규칙·장부 / R 후보 인접 문턱 / V 차량 교차.
  체결: 선물 시가·종가는 단일가(2023-07-31 전 09:00, 뒤 08:45, 마감 15:45), ETF 시가 09:00·종가 15:30 단일가. 연속 체결 없음 → 기본 비용에 호가 없음, 스트레스에 1틱.
[비용] 선물 왕복 1.8bp(스트레스 3bp), ETF 왕복 5bp(스트레스 7bp + 호가 5원/그해 평균 원가격). 선물 스윙이 만기를 넘으면 롤마다 왕복 1회 추가.
[통계] 거래별 순수익 bp 상수 회귀 HAC(maxlags 5, Bartlett, 소표본 보정 없음) → NW t. CAGR = Π(1+r)^(252/일수)−1(비활성일 0), MDD = min(누적/누적최고−1).
  연도 부호 = 거래 ≥ 5 인 해 중 순손익 합 양수 비율(해 ≥ 3 필요). 상위 5건 비중 = 상위 5건 순손익 합/총 순손익 합. 전·후반 = 거래 수 반분 평균.
  Δ = 신호일 평균 − 비교일 평균(롱 부호, 비용 전) 더미 회귀 HAC t. 스윙 Δ 는 겹침 근사(maxlags = max(5, k)).
[선별] 순수익 > 0 ∧ NW t ≥ 2 ∧ 연도 부호 ≥ 70% ∧ 상위 5건 < 50% ∧ 전·후반 양수 ∧ 스트레스 뒤 양수 ∧ 인접 문턱 부호 유지. 돌린 셀 전부 cells.csv 기록.
  다중 비교: 신호를 원형 이동한 위약으로 스캔 전체 최대 |t| 분포를 만들어 셀별 족 전체 p(fwer_p)를 notes 에 남긴다.
  재현: 달력 반분(전반 ~2020-12-30, 후반 2021-01-04~)의 셀별 순수익·t 를 notes 에 남기고, 전반 자동 선별 규칙의 후반 성과(nested)를 따로 잰다.
[자료] 깃: beomgu/data, jehyeong/intraday/data, jehyeong/daynight/results/daily_nav.csv, woohyun/data. 깃 밖: kis_fut_kq150_daily·kis_fut_k200_daily·naver_etf_investor_daily_*(notes 에 '깃 밖 자료').
"""
import json
import re
from types import SimpleNamespace
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

SEAL = pd.Timestamp("2024-12-30")
GIT = Path("<repo>")
RAW = Path("<raw>")
LAB = Path("<OUT>")
OUT = LAB / "round16_alpha_search/scan/flows"
SCRATCH = Path("<local>")  # 보고서 작성용 임시 표
EXPIRY = LAB / "round14_alpha_sprint/S1_expiry_program/expiry_calendar.csv"
BAN = ("2023-11-06", "2025-03-30")
LAGS = 5
HS = ("D", "N", "F", 3, 5, 10)
HNAME = {"D": "day", "N": "night", "N0": "night", "F": "mixed", 3: "swing3", 5: "swing5", 10: "swing10"}
CELL_COLS = ["id", "name", "market", "horizon", "window", "n", "gross_bp", "net_bp", "t_nw", "cagr_pct", "mdd_pct", "yearly_pos", "top5_share_pct",
             "half1_net_bp", "half2_net_bp", "stress_net_bp", "corr_base", "exec_ok", "notes"]


def sealed(df, col="date"):
    """봉인: 읽은 직후 수익 계산 전에 2024-12-30 이후 행 제거."""
    df = df.copy()
    df[col] = pd.to_datetime(df[col])
    df = df[df[col] <= SEAL].sort_values(col).reset_index(drop=True)
    assert df[col].max() <= SEAL, "봉인 위반"
    return df


def rd(p, **kw):
    return pd.read_csv(p, encoding="utf-8-sig", **kw)


# ---------------------------------------------------------------- 통계
def nw_t(x, lags=LAGS):
    x = np.asarray(x, float)
    n = len(x)
    if n < 3:
        return np.nan
    u = x - x.mean()
    s = float(u @ u)
    for L in range(1, min(lags, n - 1) + 1):
        s += 2 * (1 - L / (lags + 1)) * float(u[L:] @ u[:-L])
    return float(x.mean() / (np.sqrt(s) / n)) if s > 0 else np.nan


def delta_hac(y, d, lags=LAGS):
    """y ~ 1 + d 의 d 계수, HAC t. 수기 2×2 샌드위치."""
    n = len(y)
    n1 = d.sum()
    if n1 < 3 or n - n1 < 3:
        return np.nan, np.nan
    m1, m0 = y[d].mean(), y[~d].mean()
    u = y - np.where(d, m1, m0)
    X = np.column_stack([np.ones(n), d.astype(float)])
    Xu = X * u[:, None]
    S = Xu.T @ Xu
    for L in range(1, lags + 1):
        G = Xu[L:].T @ Xu[:-L]
        S += (1 - L / (lags + 1)) * (G + G.T)
    XXi = np.linalg.inv(X.T @ X)
    V = XXi @ S @ XXi
    return float(m1 - m0), float((m1 - m0) / np.sqrt(V[1, 1])) if V[1, 1] > 0 else np.nan


def z_past(x, w):
    """(x_t − 직전 w 평균)/직전 w 표준편차, t 제외. 결측일은 건너뛰고 계산."""
    s = x.dropna()
    p = s.shift(1)
    return (s - p.rolling(w).mean()) / p.rolling(w).std()


def z_k(x, w, k, days):
    s = x.dropna()
    if k > 1:
        s = s.rolling(k).sum().dropna()
    return z_past(s, w).reindex(days)


# ---------------------------------------------------------------- 차량
class Veh:
    def __init__(self, name, market, rl, rs, cl, cs, sl, ss, roll, tradable=True, local=False, longonly_note=""):
        self.name, self.market, self.rl, self.rs = name, market, rl, rs
        self.cl, self.cs, self.sl, self.ss = cl, cs, sl, ss
        self.tradable, self.local, self.note = tradable, local, longonly_note
        self.rc = None if roll is None else np.cumsum(roll)
        self.cum = np.r_[0.0, np.cumsum(np.nan_to_num(rl))]
        self.first = int(np.flatnonzero(~np.isnan(rl))[0]) // 2


def seg_px(x):
    s = np.full(2 * len(x), np.nan)
    s[0::2] = (x.open / x.close.shift(1) - 1).values
    s[1::2] = (x.close / x.open - 1).values
    return s


def fut_veh(path, days, name, market):
    raw = sealed(pd.read_csv(path))
    fr = raw[raw.front == 1].set_index("date").sort_index()
    assert fr.index.is_unique and (fr.open > 0).all() and (fr.close > 0).all()
    fr = fr.reindex(days)
    code = fr.code
    prev = code.shift(1)
    o = raw.set_index(["date", "code"]).open
    open_same = pd.Series(o.reindex(pd.MultiIndex.from_arrays([days, prev])).values, index=days)  # 전일 front 계약의 당일 시가
    n = len(days)
    seg = np.full(2 * n, np.nan)
    seg[0::2] = (open_same / fr.close.shift(1) - 1).values
    seg[1::2] = (fr.close / fr.open - 1).values
    roll = np.zeros(2 * n)
    roll[1::2] = ((code != prev) & code.notna() & prev.notna()).astype(float).values  # 최종거래일 시가에 차근월로 이동
    c = np.ones(n)
    return Veh(name, market, seg, -seg, c * 1.8e-4, c * 1.8e-4, c * 3e-4, c * 3e-4, roll, local=True), fr


def etf_px(code, days, adj=True):
    p = GIT / f"beomgu/data/etf/{code}.csv" if adj else GIT / f"jehyeong/intraday/data/{code}_daily.csv"
    d = sealed(pd.read_csv(p))
    d = d[~d.date.duplicated()].set_index("date")
    assert (d[["open", "close"]] > 0).all().all()
    return d.reindex(days)


def tick_frac(code, days):
    d = sealed(pd.read_csv(GIT / f"jehyeong/intraday/data/{code}_daily.csv"))
    avg = d.groupby(d.date.dt.year).close.mean()
    return (5.0 / pd.Series(days.year, index=days).map(avg)).fillna(0.0).values


def etf_veh(name, market, long_code, short_code, days, long_adj=True, short_adj=False):
    rl = seg_px(etf_px(long_code, days, long_adj))
    rs = seg_px(etf_px(short_code, days, short_adj))
    n = len(days)
    c = np.ones(n)
    return Veh(name, market, rl, rs, c * 5e-4, c * 5e-4, c * 7e-4 + tick_frac(long_code, days), c * 7e-4 + tick_frac(short_code, days), None)


# ---------------------------------------------------------------- 시뮬레이션
def build_seg(s, H, n):
    seg = np.zeros(2 * n)
    idx = np.flatnonzero(s)
    if H == "D":
        i = 2 * (idx + 1) + 1
        k = i < 2 * n
        seg[i[k]] = s[idx[k]]
    elif H == "N":
        i = 2 * (idx + 2)
        k = i < 2 * n
        seg[i[k]] = s[idx[k]]
    elif H == "N0":
        i = 2 * (idx + 1)
        k = i < 2 * n
        seg[i[k]] = s[idx[k]]
    elif H == "G0":  # 진단: T 종가→T+1 시가(실행 불가)
        i = 2 * (idx + 1)
        k = i < 2 * n
        seg[i[k]] = s[idx[k]]
    elif H == "F":
        for T in idx:
            seg[2 * (T + 1) + 1:min(2 * (T + 2) + 1, 2 * n)] = s[T]
    else:
        for T in idx:
            seg[2 * (T + 1) + 1:min(2 * (T + H) + 2, 2 * n)] = s[T]
    return seg


def fwd_long(V, H):
    """T 인덱스의 지평 롱 총수익(겹침 없음 가정, 합산). Δ 계산용."""
    n = len(V.rl) // 2
    a = {"D": lambda t: 2 * (t + 1) + 1, "N": lambda t: 2 * (t + 2), "N0": lambda t: 2 * (t + 1), "G0": lambda t: 2 * (t + 1), "F": lambda t: 2 * (t + 1) + 1}.get(H, lambda t: 2 * (t + 1) + 1)
    b = {"D": lambda t: 2 * (t + 1) + 2, "N": lambda t: 2 * (t + 2) + 1, "N0": lambda t: 2 * (t + 1) + 1, "G0": lambda t: 2 * (t + 1) + 1, "F": lambda t: 2 * (t + 2) + 1}.get(H, lambda t: 2 * (t + H) + 2)
    t = np.arange(n)
    ia, ib = a(t), b(t)
    ok = ib <= 2 * n
    y = np.full(n, np.nan)
    y[ok] = V.cum[ib[ok]] - V.cum[ia[ok]]
    nanm = np.isnan(V.rl)
    nc = np.r_[0, np.cumsum(nanm)]
    bad = np.zeros(n, bool)
    bad[ok] = (nc[ib[ok]] - nc[ia[ok]]) > 0
    y[bad] = np.nan
    return y


def simulate(s, H, V, n):
    seg = build_seg(s, H, n)
    rr = np.where(seg > 0, V.rl, np.where(seg < 0, V.rs, 0.0))
    bad = np.isnan(rr)
    if bad.any():
        seg = np.where(bad, 0.0, seg)
        rr = np.where(bad, 0.0, rr)
    r = np.abs(seg) * rr
    d = np.diff(np.r_[0.0, seg, 0.0])
    b = np.flatnonzero(d)
    if len(b) < 2:
        return None
    on = seg[b[:-1]] != 0
    st, en = b[:-1][on], b[1:][on] - 1
    cl = np.r_[0.0, np.cumsum(np.log1p(r))]
    gross = np.expm1(cl[en + 1] - cl[st])
    tx = en // 2
    w = np.abs(seg[st])
    lg = seg[st] > 0
    mult = 1.0 if V.rc is None else 1.0 + (V.rc[en] - V.rc[st])
    c = np.where(lg, V.cl[tx], V.cs[tx]) * w * mult
    c2 = np.where(lg, V.sl[tx], V.ss[tx]) * w * mult
    daily = r[0::2] + r[1::2]
    daily_s = daily.copy()
    np.subtract.at(daily, tx, c)
    np.subtract.at(daily_s, tx, c2)
    return dict(gross=gross, net=gross - c, stress=gross - c2, tx=tx, daily=daily, daily_s=daily_s)


class Ctx:
    pass


def cell(C, fam, sig, name, mask, side, H, V, th=np.nan, tail="", base=None, valid_from=None, weight=None, exec_ok=None, extra=""):
    """mask: T 인덱스 bool. side ±1. base: Δ 비교군 bool(없으면 신호 정의일 전체). weight: 명목 배수(T 인덱스)."""
    n = C.n
    s = mask.astype(float) * side
    if weight is not None:
        s = s * weight
    res = simulate(s, H, V, n)
    first = max(V.first, int(np.flatnonzero(mask)[0]) if mask.any() else 0) if valid_from is None else max(V.first, valid_from)
    first = min(first + 1, n - 1)
    rid = f"{fam}{len(C.rows):05d}"
    row = dict(id=rid, name=name, market=V.market, horizon=HNAME.get(H, str(H)), window=f"{C.days[first]:%Y-%m-%d}~{C.days[-1]:%Y-%m-%d}", n=0, gross_bp=np.nan, net_bp=np.nan,
               t_nw=np.nan, cagr_pct=np.nan, mdd_pct=np.nan, yearly_pos="0/0", top5_share_pct=np.nan, half1_net_bp=np.nan, half2_net_bp=np.nan, stress_net_bp=np.nan,
               corr_base="", exec_ok=bool(V.tradable if exec_ok is None else exec_ok), notes="",
               _fam=fam, _sig=sig, _th=th, _tail=tail, _side=side, _H=str(H), _veh=V.name)
    if res is None:
        C.rows.append(row)
        C.spec[rid] = (mask, side, H, V, weight)
        return row
    net, gross, tx = res["net"], res["gross"], res["tx"]
    k = len(net)
    yrs = C.year[tx]
    u, inv = np.unique(yrs, return_inverse=True)
    ysum, ycnt = np.bincount(inv, net), np.bincount(inv)
    yk = ycnt >= 5
    ypos, ytot = int((ysum[yk] > 0).sum()), int(yk.sum())
    tot = net.sum()
    top = np.sort(net)[::-1]
    top5 = top[:5].sum() / tot * 100 if tot > 0 else np.nan
    top1 = top[0] / tot * 100 if tot > 0 else np.nan
    h = k // 2
    dly = res["daily"][first:]
    eq = np.cumprod(1 + dly)
    cagr = (eq[-1] ** (252 / len(dly)) - 1) * 100
    mdd = (eq / np.maximum.accumulate(eq) - 1).min() * 100
    ban = net[(C.days[tx] >= BAN[0]) & (C.days[tx] <= BAN[1])]
    ban_sh = ban.sum() / tot * 100 if tot > 0 else np.nan
    # Δ: 롱 부호 지평 수익, 비교군 대비
    y = C.fwd[V.name, H]
    dom = ~np.isnan(y) & (C.defined if base is None else base)
    dom[:first] = False
    dm = dom & mask
    dt_, tt_ = delta_hac(y[dom] * 1e4, dm[dom], lags=max(LAGS, H if isinstance(H, int) else 1)) if dm.sum() >= 3 else (np.nan, np.nan)
    a1 = tx < C.cut
    x1, x2 = net[a1] * 1e4, net[~a1] * 1e4
    corr = {}
    for b_name, (bm, bx) in C.base.items():
        x = res["daily"][bm]
        corr[b_name] = float(np.corrcoef(x, bx)[0, 1]) if x.std() > 0 else np.nan
    row.update(n=k, gross_bp=gross.mean() * 1e4, net_bp=net.mean() * 1e4, t_nw=nw_t(net * 1e4), cagr_pct=cagr, mdd_pct=mdd, yearly_pos=f"{ypos}/{ytot}", top5_share_pct=top5,
               half1_net_bp=net[:h].mean() * 1e4 if h else np.nan, half2_net_bp=net[h:].mean() * 1e4, stress_net_bp=res["stress"].mean() * 1e4,
               corr_base=";".join(f"{a}={b:+.3f}" for a, b in corr.items()),
               _delta=dt_, _tdelta=tt_, _top1=top1, _ban=ban_sh, _ypct=ypos / ytot if ytot else np.nan, _ytot=ytot, _extra=extra,
               _n1=len(x1), _n2=len(x2), _net1=x1.mean() if len(x1) else np.nan, _net2=x2.mean() if len(x2) else np.nan, _t1=nw_t(x1), _t2=nw_t(x2),
               _top5_1=np.sort(x1)[::-1][:5].sum() / x1.sum() * 100 if len(x1) and x1.sum() > 0 else np.nan)
    C.rows.append(row)
    C.spec[rid] = (mask, side, H, V, weight)
    return row


def add_grid(C, fam, sig, z, vehs, label, hs=HS, ths=(0.5, 1.0, 1.5), extra=None, base=None, sides=(1, -1), tails=("hi", "lo"), valid_from=None, note=""):
    C.defined = z.notna().values
    for th in ths:
        for tail in tails:
            m = (z >= th).values if tail == "hi" else (z <= -th).values
            if extra is not None:
                m = m & extra
            if not m.any():
                continue
            op = "≥" if tail == "hi" else "≤−"
            for side in sides:
                for V in vehs:
                    for H in hs:
                        cell(C, fam, sig, f"{label} {op}{th:g} → {'롱' if side > 0 else '숏'} {V.name} {H}", m, side, H, V, th=th, tail=tail, base=base, valid_from=valid_from, extra=note)


# ---------------------------------------------------------------- 자료 적재
def load():
    C = Ctx()
    e069 = sealed(pd.read_csv(GIT / "beomgu/data/etf/069500.csv"))
    days = pd.DatetimeIndex(e069.date)
    days = days[days >= "2010-01-04"]
    assert days.is_unique and days.max() <= SEAL
    C.days, C.n, C.year = days, len(days), days.year.values
    C.cut = int(days.searchsorted(pd.Timestamp("2020-12-30"), side="right"))
    C.rows, C.spec = [], {}
    C.v = {}
    C.v["F_KQ"], C.fq = fut_veh(RAW / "r15_external/kis_fut_kq150_daily.csv", days, "F_KQ", "KOSDAQ")
    C.v["F_KP"], C.fk = fut_veh(RAW / "r15_external/kis_fut_k200_daily.csv", days, "F_KP", "KOSPI")
    C.v["E_KQ"] = etf_veh("E_KQ", "KOSDAQ", "229200", "251340", days)
    C.v["E_KP"] = etf_veh("E_KP", "KOSPI", "069500", "114800", days, short_adj=True)
    C.v["L_KQ"] = etf_veh("L_KQ", "KOSDAQ", "233740", "251340", days)
    # 지수(체결 불가, 재현 전용)
    k2 = sealed(rd(GIT / "beomgu/data/k200_index_daily.csv"))
    k2 = k2[~k2.date.duplicated()].set_index("date").reindex(days)
    seg = seg_px(k2)
    c = np.ones(C.n)
    C.v["I_KP"] = Veh("I_KP", "KOSPI", seg, -seg, c * 3e-4, c * 3e-4, c * 3e-4, c * 3e-4, None, tradable=False)
    # 깃 자료 단독 검산: FUT.csv 연속선물은 같은 날 시가→종가만 쓴다(밤 구간은 롤 틈이 있어 NaN)
    fg = sealed(pd.read_csv(GIT / "beomgu/data/etf/FUT.csv"))
    fg = fg[~fg.date.duplicated()].set_index("date").reindex(days)
    seg_g = seg_px(fg)
    seg_g[0::2] = np.nan
    C.v["G_KP"] = Veh("G_KP", "KOSPI", seg_g, -seg_g, c * 1.8e-4, c * 1.8e-4, c * 3e-4, c * 3e-4, None)
    C.k2 = k2
    C.fwd = {(v.name, H): fwd_long(v, H) for v in C.v.values() for H in list(HS) + ["N0", "G0"]}
    # 기준 흐름(봉인 전)
    nav = sealed(pd.read_csv(GIT / "jehyeong/daynight/results/daily_nav.csv"))
    nv = nav[nav.version == "V2"].pivot(index="date", columns="account", values="daily_return")
    book = sealed(pd.read_csv(LAB / "round13_slot_book/book/book_daily.csv")).set_index("date")
    C.base = {}
    C.base_raw = {"ON": nv.overnight, "ID": nv.intraday, "KP13": book.KOSPI_book, "KQ13": book.KOSDAQ_book}
    for nm, s in C.base_raw.items():
        m = days.isin(s.index)
        C.base[nm] = (m, s.reindex(days[m]).values)
    # 수급
    kq = sealed(rd(GIT / "jehyeong/intraday/data/KOSDAQ_150_fut_foreign.csv").rename(columns={"Date": "date"})).set_index("date")
    fu = sealed(rd(GIT / "beomgu/data/naver_fut_investor_daily.csv")).set_index("date")
    fu = fu[fu.index >= "2010-12-22"]  # 그 전 행은 buyq 가 순매수, 매수·매도 무효
    ks = sealed(rd(GIT / "beomgu/data/naver_kospi_investor_daily.csv")).set_index("date")
    kd = sealed(rd(GIT / "beomgu/data/naver_kosdaq_investor_daily.csv")).set_index("date")
    pg = sealed(rd(GIT / "beomgu/data/naver_program_daily.csv")).set_index("date")
    pq = sealed(rd(GIT / "beomgu/data/naver_kosdaq_program_daily.csv")).set_index("date")
    op = sealed(rd(GIT / "beomgu/data/krx_opt_investor_daily.csv"))
    INST = ["pension", "trust", "fin_inv", "insurance", "bank", "other_fin", "private_fund"]
    for d in (ks, kd):
        d["net_inst"] = d[[f"net_{g}" for g in INST]].sum(axis=1, min_count=1)
    fu["net_inst"] = fu[[f"net_{g}" for g in ("gov", "pension", "bank", "trust", "insurance", "other_fin", "fin_inv")]].sum(axis=1, min_count=1)
    for g in ("foreign", "individual"):
        fu[f"netv_{g}"] = fu[f"buyv_{g}"] - fu[f"sellv_{g}"]
    C.kq, C.fu, C.ks, C.kd, C.pg, C.pq = kq, fu, ks, kd, pg, pq
    for nm, df in (("kq", kq), ("fu", fu), ("ks", ks), ("kd", kd), ("pg", pg), ("pq", pq)):
        print(f"  {nm}: {df.index.min():%Y-%m-%d}~{df.index.max():%Y-%m-%d} {len(df)}행, 달력 밖 {len(df.index.difference(days))}행")
    p = op.pivot_table(index="date", columns=["investor", "cp"], values=["buy_amt", "sell_amt", "buy_qty", "sell_qty"], aggfunc="sum")
    C.opt = {(i, c_): (p["buy_amt"][i][c_] - p["sell_amt"][i][c_]) for i in ("foreign", "individual", "institution") for c_ in "CP"}
    C.optq = {(i, c_): (p["buy_qty"][i][c_] - p["sell_qty"][i][c_]) for i in ("foreign", "individual", "institution") for c_ in "CP"}
    # 만기
    e = pd.read_csv(EXPIRY, parse_dates=["expiry"])
    C.exp = e.expiry[e.expiry <= SEAL + pd.Timedelta(days=60)].sort_values()
    C.expq = e.expiry[e.quarterly & (e.expiry <= SEAL + pd.Timedelta(days=60))].sort_values()
    ix = np.searchsorted(C.exp.values, days.values, side="left")  # 다음(≥ t) 만기
    nxt = C.exp.values[np.minimum(ix, len(C.exp) - 1)]
    C.dte = np.searchsorted(days.values, nxt) - np.arange(C.n)  # 거래일 기준 잔존, 만기일 0
    # 지수 수익
    kq150 = sealed(rd(GIT / "jehyeong/intraday/data/KOSDAQ_150_ohlcv.csv").rename(columns={"Date": "date"})).set_index("date")
    C.kq150 = kq150[~kq150.index.duplicated()].reindex(days)
    C.r_kq = C.kq150.Close.pct_change()
    C.r_kp = k2.close.pct_change()
    return C


# ---------------------------------------------------------------- 신호군
def fam_A(C):
    """코스닥150 선물 투자자별 → 코스닥 차량."""
    V = [C.v["F_KQ"]]
    for g, lab in (("foreign", "KQ선물 외인"), ("institution", "KQ선물 기관"), ("individual", "KQ선물 개인")):
        for w in (60, 250):
            for k in (1, 5):
                z = z_k(C.kq[g].where(C.kq[g] != 0) if g == "foreign" else C.kq[g], w, k, C.days)
                add_grid(C, "A", f"{g}.z{w}.k{k}", z, V, f"{lab} z{w} k{k}")


def fam_B(C):
    """동시·불일치: KQ선물 × 코스닥 현물, 코스피200 선물 교차."""
    V = [C.v["F_KQ"]]
    for w in (60, 250):
        ff = z_k(C.kq.foreign.where(C.kq.foreign != 0), w, 1, C.days)
        sf = z_k(C.kd.net_foreign, w, 1, C.days)
        pf = z_k(C.kq.individual, w, 1, C.days)
        ps = z_k(C.kd.net_individual, w, 1, C.days)
        kf = z_k(C.fu.net_foreign, w, 1, C.days)
        for nm, a, b in (("외인선물×외인현물", ff, sf), ("개인선물×개인현물", pf, ps), ("KQ외인선물×K200외인선물", ff, kf)):
            C.defined = (a.notna() & b.notna()).values
            for th in (0.5, 1.0, 1.5):
                for qa, qb in ((1, 1), (-1, -1), (1, -1), (-1, 1)):
                    m = ((a * qa >= th) & (b * qb >= th)).values
                    if m.sum() < 10:
                        continue
                    for side in (1, -1):
                        for H in HS:
                            cell(C, "B", f"{nm}.z{w}.{qa}{qb}", f"{nm} z{w} {qa:+d}{qb:+d} ≥{th:g} → {'롱' if side > 0 else '숏'} F_KQ {H}", m, side, H, V[0], th=th, tail=f"{qa}{qb}")


def rolling_resid(x, r, w):
    """x_t − (α + β r_t), α·β·잔차 표준편차는 직전 w 일(t 제외). 반환: 잔차 z."""
    d = pd.DataFrame({"x": x, "r": r}).dropna()
    px, pr = d.x.shift(1), d.r.shift(1)
    mx, mr = px.rolling(w).mean(), pr.rolling(w).mean()
    cov = (px * pr).rolling(w).mean() - mx * mr
    var = (pr * pr).rolling(w).mean() - mr * mr
    b = cov / var
    a = mx - b * mr
    e = d.x - (a + b * d.r)
    pe = (px - (a + b * pr))
    return (e - pe.rolling(w).mean()) / pe.rolling(w).std()


def fam_C(C):
    """가격 효과 제거: 당일 수익 대비 수급 잔차·가격 괴리."""
    V = [C.v["F_KQ"]]
    for g, lab in (("foreign", "KQ외인"), ("institution", "KQ기관"), ("individual", "KQ개인")):
        x = C.kq[g].where(C.kq[g] != 0) if g == "foreign" else C.kq[g]
        for w in (60, 250):
            z = rolling_resid(x, C.r_kq, w).reindex(C.days)
            add_grid(C, "C", f"resid.{g}.z{w}", z, V, f"{lab} 잔차 z{w}", hs=("D", "N", "F", 5))
    V = [C.v["F_KP"]]
    for g, lab in (("net_foreign", "K200외인"), ("net_individual", "K200개인"), ("net_inst", "K200기관")):
        for w in (60, 250):
            z = rolling_resid(C.fu[g], C.r_kp, w).reindex(C.days)
            add_grid(C, "C", f"resid.{g}.z{w}", z, V, f"{lab} 잔차 z{w}", hs=("D", "N", "F", 5))
    # 가격·수급 괴리(당일 수익 부호와 수급 반대)
    for w in (60, 250):
        z = z_k(C.kq.foreign.where(C.kq.foreign != 0), w, 1, C.days)
        for lab, m0, side in (("KQ 하락일 외인선물 매수", (C.r_kq < 0).values, 1), ("KQ 상승일 외인선물 매도", (C.r_kq > 0).values, -1)):
            tl = "hi" if side > 0 else "lo"
            add_grid(C, "C", f"div.{lab}.z{w}", z, [C.v["F_KQ"]], f"{lab} z{w}", hs=("D", "N", "F", 5), extra=m0, tails=(tl,), sides=(1, -1))
        z = z_k(C.fu.net_foreign, w, 1, C.days)
        for lab, m0, side in (("K200 하락일 외인선물 매수", (C.r_kp < 0).values, 1), ("K200 상승일 외인선물 매도", (C.r_kp > 0).values, -1)):
            tl = "hi" if side > 0 else "lo"
            add_grid(C, "C", f"div.{lab}.z{w}", z, [C.v["F_KP"]], f"{lab} z{w}", hs=("D", "N", "F", 5), extra=m0, tails=(tl,), sides=(1, -1))


def fam_D(C):
    """코스피200 선물 투자자별(계약·금액) → 코스피 차량."""
    V = [C.v["F_KP"]]
    for g, lab in (("net_foreign", "K200외인 계약"), ("netv_foreign", "K200외인 금액"), ("net_individual", "K200개인 계약"), ("netv_individual", "K200개인 금액"),
                   ("net_inst", "K200기관합 계약"), ("net_fin_inv", "K200금투 계약"), ("net_trust", "K200투신 계약")):
        for w in (60, 250):
            for k in (1, 5):
                add_grid(C, "D", f"{g}.z{w}.k{k}", z_k(C.fu[g], w, k, C.days), V, f"{lab} z{w} k{k}")


def fam_E(C):
    """교차 시장: K200 선물 → 코스닥, KQ 선물 → 코스피."""
    for w in (60, 250):
        for k in (1, 5):
            add_grid(C, "E", f"kpfor.to.kq.z{w}.k{k}", z_k(C.fu.net_foreign, w, k, C.days), [C.v["F_KQ"]], f"K200외인 z{w} k{k} → 코스닥")
            add_grid(C, "E", f"kqfor.to.kp.z{w}.k{k}", z_k(C.kq.foreign.where(C.kq.foreign != 0), w, k, C.days), [C.v["F_KP"]], f"KQ외인 z{w} k{k} → 코스피")
            add_grid(C, "E", f"kqind.to.kp.z{w}.k{k}", z_k(C.kq.individual, w, k, C.days), [C.v["F_KP"]], f"KQ개인 z{w} k{k} → 코스피")


def fam_F(C):
    """15차 외인 동시 순매도 숏의 상위 5건 의존을 푸는 다른 형태."""
    V = C.v["F_KP"]
    days = C.days
    zs = z_k(C.ks.net_foreign, 250, 1, days)
    zf = z_k(C.fu.net_foreign, 250, 1, days)
    zv = z_k(C.fu.netv_foreign, 250, 1, days)
    zi = z_k(C.fu.net_individual, 250, 1, days)
    zt = z_k(C.fu.net_inst, 250, 1, days)
    C.defined = (zs.notna() & zf.notna()).values
    k2c = C.k2.close
    ma20 = k2c.rolling(20).mean()
    vol20 = C.k2.close.pct_change().rolling(20).std()
    conds = {}
    for th in (0.5, 0.84, 1.0, 1.25):
        dual = (zs <= -th) & (zf <= -th)
        conds[f"동시매도 z≤−{th:g}", th, "base"] = dual
        conds[f"동시매도(금액) z≤−{th:g}", th, "value"] = (zs <= -th) & (zv <= -th)
        conds[f"동시매도 + 개인선물 매수 z≥0.5, 문턱 {th:g}", th, "indbuy"] = dual & (zi >= 0.5)
        conds[f"동시매도 + 기관선물 매수 z≥0.5, 문턱 {th:g}", th, "instbuy"] = dual & (zt >= 0.5)
        conds[f"동시매도 + 종가<MA20, 문턱 {th:g}", th, "belowma"] = dual & (k2c < ma20)
        conds[f"동시매도 + 종가>MA20, 문턱 {th:g}", th, "abovema"] = dual & (k2c > ma20)
        conds[f"동시매도 + 당일 하락, 문턱 {th:g}", th, "down"] = dual & (C.r_kp < 0)
        conds[f"동시매도 + 당일 상승, 문턱 {th:g}", th, "up"] = dual & (C.r_kp > 0)
        c3 = dual.astype(int).rolling(3).sum()
        conds[f"최근 3일 중 동시매도 2일 이상, 문턱 {th:g}", th, "persist"] = c3 >= 2
        conds[f"이틀 연속 동시매도, 문턱 {th:g}", th, "consec"] = dual & dual.shift(1, fill_value=False)
        z3s, z3f = z_k(C.ks.net_foreign, 250, 3, days), z_k(C.fu.net_foreign, 250, 3, days)
        conds[f"3일 합 동시매도 z≤−{th:g}", th, "sum3"] = (z3s <= -th) & (z3f <= -th)
        z5s, z5f = z_k(C.ks.net_foreign, 250, 5, days), z_k(C.fu.net_foreign, 250, 5, days)
        conds[f"5일 합 동시매도 z≤−{th:g}", th, "sum5"] = (z5s <= -th) & (z5f <= -th)
        conds[f"동시매수 z≥{th:g}", th, "dualbuy"] = (zs >= th) & (zf >= th)
    for (nm, th, tag), m in conds.items():
        mv = m.fillna(False).values
        if mv.sum() < 8:
            continue
        for H in ("D", "F", 3, 5, 10):
            side = 1 if tag == "dualbuy" else -1
            cell(C, "F", tag, f"{nm} → {'롱' if side > 0 else '숏'} F_KP {H}", mv, side, H, V, th=th, tail=tag)
            if H == "D":
                # 변동성 목표 명목(직전 20일 일변동 1.2% 기준, 상한 1배): 상위 건 쏠림 완화용
                w = (0.012 / vol20).clip(upper=1.0).fillna(0).values
                cell(C, "F", tag + ".vt", f"{nm} → 숏 F_KP D, 변동성 목표", mv, side, H, V, th=th, tail=tag + ".vt", weight=w)
    # 연속 강도: 명목 = clip(((−z_s − z_f)/2 − 0.5), 0, 1), 양쪽 다 음수일 때만
    for lo in (0.25, 0.5, 0.75):
        score = (-(zs + zf) / 2 - lo).clip(lower=0, upper=1)
        mv = ((zs < 0) & (zf < 0) & (score > 0)).fillna(False).values
        for H in ("D", "F", 3):
            cell(C, "F", "cont", f"연속 강도 숏 하한 {lo:g} F_KP {H}", mv, -1, H, V, th=lo, tail="cont", weight=score.fillna(0).values)


def opt_z(s, w, days, incl=False):
    if incl:  # sunny 원 정의: t 포함 60일
        s2 = s.dropna()
        return ((s2 - s2.rolling(w, min_periods=w).mean()) / s2.rolling(w, min_periods=w).std().replace(0, np.nan)).reindex(days)
    return z_past(s.dropna(), w).reindex(days)


def fam_G(C):
    """옵션 투자자별 순매수(금액). sunny C 재현 포함."""
    V = C.v["F_KP"]
    for (i, c), s in C.opt.items():
        for w in (60, 250):
            add_grid(C, "G", f"opt.{i}.{c}.z{w}", opt_z(s, w, C.days), [V], f"옵션 {i} {c} z{w}", hs=("D", "N", "F", 5))
    for i in ("individual", "institution"):
        s = C.opt[i, "C"] - C.opt[i, "P"]
        for w in (60, 250):
            add_grid(C, "G", f"opt.{i}.CmP.z{w}", opt_z(s, w, C.days), [V], f"옵션 {i} 콜−풋 z{w}", hs=("D", "N", "F", 5))
    # 만기 주(잔존 ≤ 4 거래일) 조건
    wk = C.dte <= 4
    for (i, c), s in C.opt.items():
        add_grid(C, "G", f"optexp.{i}.{c}.z60", opt_z(s, 60, C.days), [V], f"만기주 옵션 {i} {c} z60", hs=("D", "N", "F"), ths=(0.5, 1.0), extra=wk)
    # 코스닥 교차: 옵션 외인 풋·개인 콜 → 코스닥
    for (i, c) in (("foreign", "P"), ("individual", "C"), ("individual", "P")):
        add_grid(C, "G", f"opt.{i}.{c}.z60.kq", opt_z(C.opt[i, c], 60, C.days), [C.v["F_KQ"]], f"옵션 {i} {c} z60 → 코스닥", hs=("D", "N", "F"), ths=(1.0,))
    # sunny 전략 3 케이스 A~D (t 포함 z60, 1일 지연) 재현 + 실행 차량
    fp, ic = opt_z(C.opt["foreign", "P"], 60, C.days, True), opt_z(C.opt["individual", "C"], 60, C.days, True)
    fc, ip = opt_z(C.opt["foreign", "C"], 60, C.days, True), opt_z(C.opt["individual", "P"], 60, C.days, True)
    C.defined = (fp.notna() & ic.notna()).values
    cases = {"A 개인콜z>1 숏": ((ic > 1).values, -1), "B 개인풋z>1 롱": ((ip > 1).values, 1), "C 외인풋z>1∧개인콜z>0 숏": (((fp > 1) & (ic > 0)).values, -1),
             "Cb 외인콜z<−1∧개인풋z>1 롱": (((fc < -1) & (ip > 1)).values, 1)}
    for nm, (m, side) in cases.items():
        for v in ("I_KP", "F_KP", "E_KP"):
            for H in ("D", "N", "F", 3):
                if v == "I_KP" and H != "D":
                    continue
                cell(C, "G", f"sunny.{nm}", f"sunny {nm} {v} {H}", m, side, H, C.v[v], th=1.0, tail="sunny", exec_ok=(v != "I_KP"), extra="sunny 재현" if v == "I_KP" else "sunny 실행 차량")


def rolling_top(x, q, w=250):
    """(woohyun 규칙) 직전 w 일 분위 문턱(t 제외)."""
    return x.rolling(w, min_periods=w).quantile(q).shift(1)


def fam_H(C):
    """woohyun Flow(금액 (매수−매도)/(매수+매도))를 밤·스윙에."""
    buy = sealed(rd(GIT / "woohyun/data/KRX 코스피200 선물 매수_2010_2026.csv").rename(columns={"일자": "date"})).set_index("date")
    sell = sealed(rd(GIT / "woohyun/data/KRX 코스피200 선물 매도_2010_2026.csv").rename(columns={"일자": "date"})).set_index("date")
    b = pd.to_numeric(buy["외국인 합계"], errors="coerce")
    s = pd.to_numeric(sell["외국인 합계"], errors="coerce")
    d = pd.DataFrame({"b": b, "s": s}).dropna()
    d = d[~d.index.duplicated()]
    f1 = (d.b - d.s) / (d.b + d.s)
    f3 = (d.b.rolling(3).sum() - d.s.rolling(3).sum()) / (d.b.rolling(3).sum() + d.s.rolling(3).sum())
    V = C.v["F_KP"]
    for nm, f in (("flow1d", f1), ("flow3d", f3)):
        C.defined = f.reindex(C.days).notna().values & rolling_top(f, 0.8).reindex(C.days).notna().values
        for q in (0.8, 0.7, 0.6):
            hi = (f >= rolling_top(f, q)).reindex(C.days, fill_value=False).values
            lo = (f <= rolling_top(f, 1 - q)).reindex(C.days, fill_value=False).values
            for tail, m in (("top", hi), ("bot", lo)):
                for side in (1, -1):
                    for H in HS:
                        cell(C, "H", f"woohyun.{nm}.{tail}", f"woohyun {nm} {tail}{round((1 - q) * 100)}% → {'롱' if side > 0 else '숏'} F_KP {H}", m, side, H, V, th=1 - q, tail=tail)


def fam_I(C):
    """프로그램 차익 잔고 대용."""
    pg = C.pg
    idx = pg.index
    days = C.days
    V = [C.v["F_KP"]]
    for lab, exp in (("월물", C.exp), ("분기", C.expq)):
        gid = np.searchsorted(exp.values, idx.values, side="right") - 1
        eff = pg.arb_net.where(~idx.isin(exp), 0.0)
        cum = eff.groupby(gid).cumsum()
        for w in (60, 250):
            add_grid(C, "I", f"arbcum.{lab}.z{w}", z_past(cum.dropna(), w).reindex(days), V, f"차익 누적({lab} 리셋) z{w}")
            add_grid(C, "I", f"arbcum.{lab}.z{w}.expwk", z_past(cum.dropna(), w).reindex(days), V, f"만기주 차익 누적({lab} 리셋) z{w}", hs=("D", "N", "F"), extra=C.dte <= 4)
    for w in (60, 250):
        for k in (1, 5):
            add_grid(C, "I", f"arb.z{w}.k{k}", z_k(pg.arb_net, w, k, days), V, f"차익 순매수 z{w} k{k}")
            add_grid(C, "I", f"kqarb.z{w}.k{k}", z_k(C.pq.arb_net, w, k, days), [C.v["F_KQ"]], f"코스닥 차익 순매수 z{w} k{k}")
            add_grid(C, "I", f"kqnon.z{w}.k{k}", z_k(C.pq.nonarb_net, w, k, days), [C.v["F_KQ"]], f"코스닥 비차익 순매수 z{w} k{k}")


def ibs_of(p):
    return (p.close - p.low) / (p.high - p.low).replace(0, np.nan)


def fam_J(C):
    """IBS 밤 신호 × 수급: T 15:30 ETF 종가 IBS → 15:45 선물 종가 진입 → T+1 시가 청산, 수급은 T−1 확정치."""
    days = C.days
    p = etf_px("229200", days)
    ibs = ibs_of(p)
    pc = p.close.shift(1)
    ma10 = p.close.rolling(10, min_periods=10).mean().shift(1)
    up, dn = (pc > ma10), (pc < ma10)
    lo_, hi_ = (ibs < 0.3).values, (ibs > 0.7).values
    V = C.v["F_KQ"]
    C.defined = ibs.notna().values
    bl = cell(C, "J", "ibs.base.long", "IBS<0.3 롱 F_KQ N0", lo_, 1, "N0", V, th=0.3, tail="base")
    bs = cell(C, "J", "ibs.base.short", "IBS>0.7 숏 F_KQ N0", hi_, -1, "N0", V, th=0.7, tail="base")
    cell(C, "J", "ibs.reg.long", "IBS<0.3∧전일종가>MA10 롱 F_KQ N0", lo_ & up.values, 1, "N0", V, th=0.3, tail="reg")
    cell(C, "J", "ibs.reg.short", "IBS>0.7∧전일종가<MA10 숏 F_KQ N0", hi_ & dn.values, -1, "N0", V, th=0.7, tail="reg")
    # 문턱 인접(0.2/0.25/0.35/0.4)
    for a in (0.2, 0.25, 0.35, 0.4):
        cell(C, "J", "ibs.adj.long", f"IBS<{a:g} 롱 F_KQ N0", (ibs < a).values, 1, "N0", V, th=a, tail="adj")
        cell(C, "J", "ibs.adj.short", f"IBS>{1 - a:g} 숏 F_KQ N0", (ibs > 1 - a).values, -1, "N0", V, th=1 - a, tail="adj")
    filt = {"KQ외인선물z60": C.kq.foreign.where(C.kq.foreign != 0), "KQ기관선물z60": C.kq.institution, "KQ개인선물z60": C.kq.individual,
            "코스닥외인현물z60": C.kd.net_foreign, "코스닥개인현물z60": C.kd.net_individual, "코스닥기관현물z60": C.kd.net_inst, "K200외인선물z60": C.fu.net_foreign, "코스닥비차익z60": C.pq.nonarb_net}
    for nm, x in filt.items():
        z1 = z_k(x, 60, 1, days).shift(1)  # T 밤에 쓰는 값 = T−1 확정치
        for th in (0.0, 0.5, 1.0):
            for tail, m in (("hi", (z1 >= th).values), ("lo", (z1 <= -th).values)):
                op = "≥" if tail == "hi" else "≤−"
                cell(C, "J", f"ibs.long.{nm}", f"IBS<0.3 ∧ {nm}(T−1){op}{th:g} 롱 F_KQ N0", lo_ & m, 1, "N0", V, th=th, tail=tail, base=lo_ & z1.notna().values)
                cell(C, "J", f"ibs.short.{nm}", f"IBS>0.7 ∧ {nm}(T−1){op}{th:g} 숏 F_KQ N0", hi_ & m, -1, "N0", V, th=th, tail=tail, base=hi_ & z1.notna().values)
    # 코스피 IBS(069500) → K200 선물 동일 구조
    q = etf_px("069500", days)
    ik = ibs_of(q)
    Vk = C.v["F_KP"]
    C.defined = ik.notna().values
    l2, h2 = (ik < 0.3).values, (ik > 0.7).values
    cell(C, "J", "kibs.base.long", "코스피 IBS<0.3 롱 F_KP N0", l2, 1, "N0", Vk, th=0.3, tail="base")
    cell(C, "J", "kibs.base.short", "코스피 IBS>0.7 숏 F_KP N0", h2, -1, "N0", Vk, th=0.7, tail="base")
    for nm, x in (("K200외인선물z60", C.fu.net_foreign), ("K200개인선물z60", C.fu.net_individual), ("K200기관선물z60", C.fu.net_inst), ("코스피외인현물z60", C.ks.net_foreign), ("코스피개인현물z60", C.ks.net_individual)):
        z1 = z_k(x, 60, 1, days).shift(1)
        for th in (0.0, 0.5, 1.0):
            for tail, m in (("hi", (z1 >= th).values), ("lo", (z1 <= -th).values)):
                op = "≥" if tail == "hi" else "≤−"
                cell(C, "J", f"kibs.long.{nm}", f"코스피 IBS<0.3 ∧ {nm}(T−1){op}{th:g} 롱 F_KP N0", l2 & m, 1, "N0", Vk, th=th, tail=tail, base=l2 & z1.notna().values)
                cell(C, "J", f"kibs.short.{nm}", f"코스피 IBS>0.7 ∧ {nm}(T−1){op}{th:g} 숏 F_KP N0", h2 & m, -1, "N0", Vk, th=th, tail=tail, base=h2 & z1.notna().values)
    C.ibs_q, C.ibs_k = ibs, ik


def fam_K(C):
    """ETF 투자자별(깃 밖): 개인 레버리지·인버스 쏠림."""
    def etf_flow(code):
        p = pd.read_csv(RAW / f"r15_external/naver_etf_investor_daily_{code}.csv")
        p["date"] = pd.to_datetime(p.bizdate.astype(str))
        p = sealed(p).drop_duplicates("date").set_index("date")
        return p
    days = C.days
    note = "깃 밖 자료"
    for market, lev, inv, V, lab in (("KQ", "233740", "251340", C.v["F_KQ"], "코스닥"), ("KP", "122630", "252670", C.v["F_KP"], "코스피")):
        a, b = etf_flow(lev), etf_flow(inv)
        for who, col in (("개인", "individualPureBuyQuant"), ("기관", "organPureBuyQuant"), ("외인", "foreignerPureBuyQuant")):
            sent = (a[col] * a.closePrice).sub(b[col] * b.closePrice, fill_value=np.nan)
            for w in (60, 250):
                for k in (1, 5):
                    add_grid(C, "K", f"{market}.{who}.levinv.z{w}.k{k}", z_k(sent, w, k, days), [V], f"{lab} {who} 레버리지−인버스 순매수 z{w} k{k}", hs=("D", "N", "F", 5), note=note)


def fam_S(C):
    """현물 기준선(기각 축 참조): 같은 형태에서 선물이 현물에 더하는 것 대조."""
    for mk, d, V, lab in (("KQ", C.kd, C.v["F_KQ"], "코스닥"), ("KP", C.ks, C.v["F_KP"], "코스피")):
        for g, gl in (("net_foreign", "외인"), ("net_individual", "개인"), ("net_inst", "기관")):
            for w in (60, 250):
                for k in (1, 5):
                    add_grid(C, "S", f"spot.{mk}.{g}.z{w}.k{k}", z_k(d[g], w, k, C.days), [V], f"{lab} 현물 {gl} z{w} k{k}", hs=("D", "N", "F", 5), note="기각 축 참조")


def fam_L(C):
    """매개변수 지도: KQ선물 외인·기관 누적의 낮 효과가 고원인지."""
    V = C.v["F_KQ"]
    days = C.days
    for g, x, side, lab in (("foreign", C.kq.foreign.where(C.kq.foreign != 0), 1, "외인"), ("institution", C.kq.institution, -1, "기관")):
        for w in (40, 60, 90, 120, 180, 250):
            for k in (1, 2, 3, 4, 5, 6, 7, 8, 10, 15, 20):
                z = z_k(x, w, k, days)
                C.defined = z.notna().values
                for th in (1.0, 1.25, 1.5, 1.75, 2.0):
                    cell(C, "L", f"map.{g}.w{w}.k{k}", f"지도 KQ{lab} z{w} k{k} ≥{th:g} → {'롱' if side > 0 else '숏'} F_KQ D", (z >= th).values, side, "D", V, th=th, tail="hi")
        for hl in (1, 2, 3, 5, 8):
            z = z_past(x.dropna().ewm(halflife=hl, adjust=False).mean(), 60).reindex(days)
            C.defined = z.notna().values
            for th in (1.0, 1.5, 2.0):
                cell(C, "L", f"ewm.{g}.hl{hl}", f"지수가중 KQ{lab} 반감 {hl}일 z60 ≥{th:g} → {'롱' if side > 0 else '숏'} F_KQ D", (z >= th).values, side, "D", V, th=th, tail="hi")


def fam_M(C):
    """외인 누적 꼬리의 성격 규명: 지속성·가격 상태·현물 교차."""
    V = C.v["F_KQ"]
    days = C.days
    ff = C.kq.foreign.where(C.kq.foreign != 0)
    C.defined = ff.reindex(days).notna().values
    pos = (ff > 0).astype(float).where(ff.notna()).dropna()
    for k, c in ((3, 3), (5, 4), (5, 5), (10, 8)):
        m = (pos.rolling(k).sum().reindex(days) >= c).values
        for H in ("D", "N"):
            cell(C, "M", f"count.{k}.{c}", f"최근 {k}일 중 외인 순매수 {c}일 이상 → 롱 F_KQ {H}", m, 1, H, V, th=c, tail="count")
    z1 = z_k(ff, 60, 1, days)
    for th in (0.5, 1.0):
        m2 = ((z1 >= th) & (z1.shift(1) >= th)).values
        m3 = ((z1 >= th) & (z1.shift(1) >= th) & (z1.shift(2) >= th)).values
        cell(C, "M", "consec2", f"이틀 연속 외인 z60≥{th:g} → 롱 F_KQ D", m2, 1, "D", V, th=th, tail="consec2")
        cell(C, "M", "consec3", f"사흘 연속 외인 z60≥{th:g} → 롱 F_KQ D", m3, 1, "D", V, th=th, tail="consec3")
    z5 = z_k(ff, 60, 5, days)
    hi = z5 >= 1.5
    r5 = C.kq150.Close.pct_change(5)
    ma20 = C.kq150.Close.rolling(20).mean()
    for nm, m in (("5일 수익>0", r5 > 0), ("5일 수익≤0", r5 <= 0), ("당일 상승", C.r_kq > 0), ("당일 하락", C.r_kq <= 0), ("종가>MA20", C.kq150.Close > ma20), ("종가≤MA20", C.kq150.Close <= ma20)):
        cell(C, "M", "z5.state", f"KQ외5≥1.5 ∧ {nm} → 롱 F_KQ D", (hi & m).values, 1, "D", V, th=1.5, tail=nm)
    sz = z_k(C.kd.net_foreign, 60, 5, days)
    for nm, m in (("선물≥1.5∧현물≥1", hi & (sz >= 1)), ("선물≥1.5∧현물<1", hi & (sz < 1)), ("선물≥1.5∧현물<0", hi & (sz < 0)), ("선물<1∧현물≥1.5", (z5 < 1) & (sz >= 1.5))):
        for H in ("D", "N"):
            cell(C, "M", "z5.spot", f"{nm}(5일 합 z60) → 롱 F_KQ {H}", m.values, 1, H, V, th=1.5, tail=nm)


def flow3d(C):
    buy = sealed(rd(GIT / "woohyun/data/KRX 코스피200 선물 매수_2010_2026.csv").rename(columns={"일자": "date"})).set_index("date")
    sell = sealed(rd(GIT / "woohyun/data/KRX 코스피200 선물 매도_2010_2026.csv").rename(columns={"일자": "date"})).set_index("date")
    d = pd.DataFrame({"b": pd.to_numeric(buy["외국인 합계"], errors="coerce"), "s": pd.to_numeric(sell["외국인 합계"], errors="coerce")}).dropna()
    d = d[~d.index.duplicated()]
    return (d.b.rolling(3).sum() - d.s.rolling(3).sum()) / (d.b.rolling(3).sum() + d.s.rolling(3).sum())


def rules(C):
    """후보 규칙 고정안. 밤 다리는 N0 축으로 통일(수급은 T−1 확정치, IBS 는 T 15:30 ETF 종가, 체결은 T 15:45 선물 종가)."""
    days = C.days
    ff = C.kq.foreign.where(C.kq.foreign != 0)
    z5f, z5i = z_k(ff, 60, 5, days), z_k(C.kq.institution, 60, 5, days)
    zi, zf = z_k(C.kq.institution, 250, 1, days), z_k(ff, 250, 1, days)
    zk5 = z_k(C.fu.net_foreign, 250, 5, days)
    zp = z_k(C.kq.individual, 60, 1, days)
    ibs, ik = ibs_of(etf_px("229200", days)), ibs_of(etf_px("069500", days))
    zs, zf2, zi2 = z_k(C.ks.net_foreign, 250, 1, days), z_k(C.fu.net_foreign, 250, 1, days), z_k(C.fu.net_individual, 250, 1, days)
    zi60, zfin = z_k(C.fu.net_individual, 60, 1, days), z_k(C.fu.net_fin_inv, 60, 1, days)
    f3 = flow3d(C)
    top40 = (f3 >= rolling_top(f3, 0.6)).reindex(days, fill_value=False)
    d1, d2 = z5f >= 1.5, z5i >= 1.5
    n1 = ((zi <= -1) | (zf >= 1) | (zk5 >= 1.5)).shift(1, fill_value=False)
    n2 = (ibs < 0.3) & (zp.shift(1) >= 0)
    k1 = (zs <= -0.5) & (zf2 <= -0.5) & (zi2 >= 0.5)
    k2 = zi60 >= 0.5
    k3 = (zfin <= -1).shift(1, fill_value=False)
    k4 = top40.shift(1, fill_value=False)
    k6 = (ik < 0.3) & (z_k(C.fu.net_individual, 60, 1, days).shift(1) >= 0)
    FQ, FP = C.v["F_KQ"], C.v["F_KP"]
    w12 = (d1.astype(float) - d2.astype(float)).values
    return {"D1": dict(m=d1.values, side=1, H="D", V=FQ, w=None, desc="KQ선물 외인 5일 합 z60 ≥ 1.5 → T+1 시가 F_KQ 매수 → T+1 15:45"),
            "D2": dict(m=d2.values, side=-1, H="D", V=FQ, w=None, desc="KQ선물 기관 5일 합 z60 ≥ 1.5 → T+1 시가 F_KQ 매도 → T+1 15:45"),
            "D12": dict(m=(d1 ^ d2).values, side=1, H="D", V=FQ, w=w12, desc="D1 + D2 양방향(둘 다면 쉼)"),
            "N1": dict(m=n1.values, side=1, H="N0", V=FQ, w=None, desc="T−1 기준 KQ기관 z250 ≤ −1 ∪ KQ외인 z250 ≥ 1 ∪ K200외인 5일 합 z250 ≥ 1.5 → T 15:45 F_KQ 매수 → T+1 시가"),
            "N2": dict(m=n2.values, side=1, H="N0", V=FQ, w=None, desc="T IBS(229200) < 0.3 ∧ T−1 KQ개인 z60 ≥ 0 → T 15:45 F_KQ 매수 → T+1 시가"),
            "N12": dict(m=(n1 | n2).values, side=1, H="N0", V=FQ, w=None, desc="N1 ∪ N2"),
            "K1": dict(m=k1.values, side=-1, H="D", V=FP, w=None, desc="코스피 외인 현물 z250 ≤ −0.5 ∧ K200 외인 z250 ≤ −0.5 ∧ K200 개인 z250 ≥ 0.5 → T+1 시가 F_KP 매도 → 15:45"),
            "K2": dict(m=k2.values, side=-1, H="D", V=FP, w=None, desc="K200 개인 z60 ≥ 0.5 → T+1 시가 F_KP 매도 → 15:45"),
            "K3": dict(m=k3.values, side=1, H="N0", V=FP, w=None, desc="T−1 K200 금융투자 z60 ≤ −1 → T 15:45 F_KP 매수 → T+1 시가"),
            "K4": dict(m=k4.values, side=1, H="N0", V=FP, w=None, desc="T−1 woohyun flow3d 상위 40%(250일 문턱) → T 15:45 F_KP 매수 → T+1 시가"),
            "K5": dict(m=(k3 | k4).values, side=1, H="N0", V=FP, w=None, desc="K3 ∪ K4"),
            "K6": dict(m=k6.values, side=1, H="N0", V=FP, w=None, desc="T IBS(069500) < 0.3 ∧ T−1 K200 개인 z60 ≥ 0 → T 15:45 F_KP 매수 → T+1 시가")}


def run_rules(C):
    R_ = rules(C)
    out = {}
    C.defined = np.ones(C.n, bool)
    for k, r in R_.items():
        row = cell(C, "P", f"rule.{k}", f"[{k}] {r['desc']}", r["m"], r["side"], r["H"], r["V"], th=1.0, tail=k, weight=r["w"], extra="후보 규칙")
        out[k] = dict(r, row=row)
    # N1 여집합: 조건 밖 밤에 프리미엄이 남는지(정의 가능한 날만)
    days = C.days
    zv = (z_k(C.kq.institution, 250, 1, days).notna() & z_k(C.fu.net_foreign, 250, 5, days).notna()).shift(1, fill_value=False).values
    row = cell(C, "P", "rule.N1c", "[N1c] N1 여집합 밤 → T 15:45 F_KQ 매수 → T+1 시가", zv & ~R_["N1"]["m"], 1, "N0", C.v["F_KQ"], th=1.0, tail="N1c", extra="대조")
    out["N1c"] = dict(R_["N1"], m=zv & ~R_["N1"]["m"], row=row)
    # 차량 교차
    cross = {"D1": [("E_KQ", "D"), ("L_KQ", "D")], "D2": [("E_KQ", "D")], "N1": [("E_KQ", "N0"), ("L_KQ", "N0")], "K1": [("E_KP", "D"), ("G_KP", "D")], "K2": [("E_KP", "D"), ("G_KP", "D")], "K3": [("E_KP", "N0")], "K4": [("E_KP", "N0")]}
    for k, lst in cross.items():
        r = R_[k]
        for vn, H in lst:
            row = cell(C, "V", f"cross.{k}.{vn}", f"[{k}] {vn} 체결 {H}", r["m"], r["side"], H, C.v[vn], th=1.0, tail=f"{k}.{vn}", extra="차량 교차")
            out[f"{k}|{vn}"] = dict(r, row=row)
    return out


def adj_rules(C):
    """후보 규칙의 인접 문턱. 표는 규칙별 (문턱, 순 bp, t, n)."""
    days = C.days
    FQ, FP = C.v["F_KQ"], C.v["F_KP"]
    ff = C.kq.foreign.where(C.kq.foreign != 0)
    zi, zf, zk5 = z_k(C.kq.institution, 250, 1, days), z_k(ff, 250, 1, days), z_k(C.fu.net_foreign, 250, 5, days)
    zp = z_k(C.kq.individual, 60, 1, days).shift(1)
    ibs = ibs_of(etf_px("229200", days))
    zs, zf2, zi2 = z_k(C.ks.net_foreign, 250, 1, days), z_k(C.fu.net_foreign, 250, 1, days), z_k(C.fu.net_individual, 250, 1, days)
    zi60, zfin = z_k(C.fu.net_individual, 60, 1, days), z_k(C.fu.net_fin_inv, 60, 1, days)
    f3 = flow3d(C)
    C.defined = np.ones(C.n, bool)
    A = {k: [] for k in ("N1", "N2", "K1", "K2", "K3", "K4")}

    def add(key, lab, m, side, H, V):
        r = cell(C, "R", f"adj.{key}", f"[{key} 인접] {lab}", m, side, H, V, th=len(A[key]), tail=key, extra="후보 인접")
        A[key].append((lab, round(float(r["net_bp"]), 2), round(float(r["t_nw"]), 2), int(r["n"])))
    for a in (0.75, 1.0, 1.25):
        for b in (1.25, 1.5, 1.75):
            add("N1", f"KQ {a:g} / K200 {b:g}", ((zi <= -a) | (zf >= a) | (zk5 >= b)).shift(1, fill_value=False).values, 1, "N0", FQ)
    for ib in (0.25, 0.3, 0.35):
        for t in (-0.25, 0.0, 0.25):
            add("N2", f"IBS<{ib:g} / 개인≥{t:g}", ((ibs < ib) & (zp >= t)).values, 1, "N0", FQ)
    for a in (0.4, 0.5, 0.6):
        for b in (0.4, 0.5, 0.6):
            add("K1", f"동시 {a:g} / 개인 {b:g}", ((zs <= -a) & (zf2 <= -a) & (zi2 >= b)).values, -1, "D", FP)
    for t in (0.25, 0.4, 0.5, 0.6, 0.75, 1.0):
        add("K2", f"z≥{t:g}", (zi60 >= t).values, -1, "D", FP)
    for t in (0.75, 1.0, 1.25):
        add("K3", f"z≤−{t:g}", (zfin <= -t).shift(1, fill_value=False).values, 1, "N0", FP)
    for q in (0.5, 0.6, 0.7):
        add("K4", f"상위 {round((1 - q) * 100)}%", (f3 >= rolling_top(f3, q)).reindex(days, fill_value=False).shift(1, fill_value=False).values, 1, "N0", FP)
    return A


def win_stat(x):
    e = (1 + x).cumprod()
    return dict(cagr=float((e.iloc[-1] ** (252 / len(x)) - 1) * 100), mdd=float((e / e.cummax() - 1).min() * 100), n=len(x))


def book(C, name, legs, lo, hi, market, hz, scale=1.0, note=""):
    d = np.zeros(C.n)
    ds = np.zeros(C.n)
    for s, H, V, w in legs:
        r = simulate(np.nan_to_num(s), H, V, C.n)
        d += w * r["daily"]
        ds += w * r["daily_s"]
    x = pd.Series(d, index=C.days).loc[lo:hi] * scale
    xs = pd.Series(ds, index=C.days).loc[lo:hi] * scale
    eq = (1 + x).cumprod()
    a = x[x != 0]
    h = len(a) // 2
    yr = x.groupby(x.index.year).apply(lambda g: (1 + g).prod() - 1) * 100
    corr = {}
    for nm, (bm, bx) in C.base.items():
        u = pd.Series(d, index=C.days)[C.days[bm]]
        corr[nm] = float(np.corrcoef(u.values, bx)[0, 1]) if u.std() > 0 else np.nan
    row = dict(id=f"P{len(C.rows):05d}", name=name, market=market, horizon=hz, window=f"{x.index[0]:%Y-%m-%d}~{x.index[-1]:%Y-%m-%d}", n=len(a), gross_bp=np.nan, net_bp=a.mean() * 1e4, t_nw=nw_t(a.values * 1e4),
               cagr_pct=(eq.iloc[-1] ** (252 / len(x)) - 1) * 100, mdd_pct=(eq / eq.cummax() - 1).min() * 100, yearly_pos=f"{int((yr > 0).sum())}/{len(yr)}",
               top5_share_pct=a.nlargest(5).sum() / a.sum() * 100 if a.sum() > 0 else np.nan, half1_net_bp=a.iloc[:h].mean() * 1e4, half2_net_bp=a.iloc[h:].mean() * 1e4, stress_net_bp=xs[xs != 0].mean() * 1e4,
               corr_base=";".join(f"{u}={v:+.3f}" for u, v in corr.items()), exec_ok=True, notes="", _fam="P", _sig=name, _th=np.nan, _tail="book", _side=0, _H="book", _veh="book", _extra=note,
               _delta=np.nan, _tdelta=np.nan, _top1=a.max() / a.sum() * 100 if a.sum() > 0 else np.nan, _ban=x.loc[BAN[0]:BAN[1]].sum() / x.sum() * 100 if x.sum() > 0 else np.nan, _ypct=float((yr > 0).mean()), _ytot=len(yr))
    C.rows.append(row)
    sh = x.mean() * 252 / (x.std() * np.sqrt(252)) if x.std() > 0 else np.nan
    return dict(row=row, daily=x, yearly={int(k): round(float(v), 1) for k, v in yr.items()}, sharpe=float(sh), active=float((x != 0).mean()), worst=float(x.min() * 100), worst_day=str(x.idxmin().date()))


def event_path(C, mask, V):
    idx = np.flatnonzero(mask)
    n = C.n
    out = {}
    for lab, off in (("D(T)", 1), ("N(T)", 2), ("D+1", 3), ("N+1", 4), ("D+2", 5), ("N+2", 6), ("D+3", 7), ("N+3", 8), ("D+4", 9)):
        i = 2 * idx + off
        i = i[i < 2 * n]
        x = V.rl[i]
        x = x[~np.isnan(x)] * 1e4
        out[lab] = (round(float(x.mean()), 1), round(float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))), 2), len(x))
    return out


def ibs_minute(C):
    """TOSS 229200 1분봉(봉 끝 라벨): 09:01~15:19 라벨 봉의 고저·15:19 종가로 IBS. 15:30 단일가 체결 전제."""
    m = pd.read_csv(GIT / "jehyeong/intraday/data/minute/229200_TOSS_1m.csv", usecols=["timestamp", "date", "high", "low", "close"])
    m = sealed(m).sort_values("timestamp")  # sealed 는 날짜만 정렬, 하루 안 순서는 시각으로 다시
    t = m.timestamp.str[11:16]
    assert t.min() == "09:01"
    w = m[t <= "15:19"]
    g = w.groupby("date")
    last = w.groupby("date").tail(1).set_index("date")
    tl = last.timestamp.str[11:16]
    print(f"  분봉 IBS: {len(last)}일, 마지막 봉 15:19 비율 {(tl == '15:19').mean():.3f}, 15:10 이후 {(tl >= '15:10').mean():.3f}")
    assert (tl >= "15:10").mean() > 0.97
    last = last[tl >= "15:10"]  # 봉이 일찍 끊긴 날은 IBS 미정의
    hi, lo = g.high.max(), g.low.min()
    return ((last.close - lo) / (hi - lo).replace(0, np.nan)).reindex(C.days)


def minute_profile(C, mask, side):
    """신호 다음날 229200 구간별 수익(2022-11-23~). 09:01 봉 시가 = 09:00 단일가, 15:30 봉 종가 = 마감 단일가."""
    m = pd.read_csv(GIT / "jehyeong/intraday/data/minute/229200_TOSS_1m.csv", usecols=["timestamp", "date", "open", "close"])
    m = sealed(m)
    m["hm"] = m.timestamp.str[11:16]
    o = m[m.hm == "09:01"].set_index("date").open
    p = {hm: m[m.hm == hm].set_index("date").close for hm in ("09:30", "12:00", "15:20", "15:30")}
    T = C.days[np.flatnonzero(mask)]
    pos = C.days.get_indexer(T) + 1
    D = C.days[pos[pos < C.n]]
    f = pd.DataFrame({"o": o, **p}).reindex(D).dropna()
    r = pd.DataFrame({"09:00→09:30": f["09:30"] / f.o - 1, "09:30→12:00": f["12:00"] / f["09:30"] - 1, "12:00→15:20": f["15:20"] / f["12:00"] - 1, "15:20→15:30": f["15:30"] / f["15:20"] - 1}) * 1e4 * side
    return {c: (round(float(r[c].mean()), 1), round(float(r[c].mean() / (r[c].std(ddof=1) / np.sqrt(len(r)))), 2)) for c in r} | {"n": len(r)}


def mech_check(C):
    """KQ선물 기관 수급이 레버리지·인버스 ETF 설정·환매 헤지를 반영하는지(깃 밖 자료)."""
    def ef(code):
        p = pd.read_csv(RAW / f"r15_external/naver_etf_investor_daily_{code}.csv")
        p["date"] = pd.to_datetime(p.bizdate.astype(str))
        return sealed(p).drop_duplicates("date").set_index("date")
    a, b = ef("233740"), ef("251340")
    s = (a.individualPureBuyQuant * a.closePrice).sub(b.individualPureBuyQuant * b.closePrice).dropna()
    d = pd.DataFrame({"개인레버−인버": s, "KQ기관": C.kq.institution, "KQ외인": C.kq.foreign, "KQ개인": C.kq.individual}).dropna()
    d5 = d.rolling(5).sum().dropna()
    return {"1일": d.corr().iloc[0].round(3).to_dict(), "5일 합": d5.corr().iloc[0].round(3).to_dict(), "n": len(d)}


# ---------------------------------------------------------------- 선별
def post(C):
    df = pd.DataFrame(C.rows)
    df["_adj"] = False
    key = ["_fam", "_sig", "_tail", "_side", "_H", "_veh"]
    for _, g in df.groupby(key, sort=False, dropna=False):
        if len(g) < 2:
            continue
        g = g.sort_values("_th")
        v = g.net_bp.values
        for i, ix in enumerate(g.index):
            nb = [v[j] for j in (i - 1, i + 1) if 0 <= j < len(v)]
            nb = [x for x in nb if not np.isnan(x)]
            df.loc[ix, "_adj"] = len(nb) > 0 and all(x > 0 for x in nb)
    df["_pass"] = (df.net_bp > 0) & (df.t_nw >= 2) & (df._ypct >= 0.7) & (df._ytot >= 3) & (df.top5_share_pct < 50) & (df.half1_net_bp > 0) & (df.half2_net_bp > 0) & (df.stress_net_bp > 0) & df._adj & df.exec_ok
    return df


def placebo(C, B=2000, seed=16):
    """신호를 원형 이동해 학생화 t 의 족 전체 분포를 만든다. 신호일 내부 표준편차를 써서 변동성 쏠림을 반영, 지평별로 위약 표준편차로 다시 표준화."""
    rng = np.random.default_rng(seed)
    groups, keymap = {}, {}
    for rid, (mask, side, H, V, w) in C.spec.items():
        if not V.tradable or mask.sum() < 30 or w is not None or H == "G0":
            continue
        k = (V.name, H, hash(mask.tobytes()))
        keymap[rid] = k
        groups.setdefault((V.name, H), {}).setdefault(k, mask)
    keys, real, sims, fam_of = [], [], [], []
    for (vn, H), d in groups.items():
        V = C.v[vn]
        y = C.fwd[vn, H].copy()
        ok = np.flatnonzero(~np.isnan(y))
        lo, hi = max(V.first + 1, ok[0]), ok[-1] + 1
        y0 = np.nan_to_num(y[lo:hi])
        y0 = y0 - y0.mean()
        ks = list(d)
        M = np.column_stack([d[k][lo:hi] for k in ks]).astype(np.float32)
        ns = M.sum(0)
        keep = ns >= 30
        M, ns = M[:, keep], ns[keep]
        ks = [k for k, f in zip(ks, keep) if f]

        def tstat(a):
            s1 = M.T @ a
            s2 = M.T @ (a * a)
            mean = s1 / ns
            var = np.maximum(s2 / ns - mean * mean, 1e-18)
            return mean / np.sqrt(var / ns)
        N = hi - lo
        T = np.empty((B, len(ks)), np.float32)
        for b in range(B):
            T[b] = tstat(np.roll(y0, rng.integers(60, N - 60)))
        keys += ks
        real.append(tstat(y0))
        sims.append(T)
    real = np.concatenate(real)
    sims = np.column_stack(sims)
    s = sims.std(0)
    tz, zs = real / s, sims / s
    maxz = np.abs(zs).max(1)
    res = {"tests": len(keys), "max_real": float(np.abs(tz).max()), "p95": float(np.quantile(maxz, .95)), "p99": float(np.quantile(maxz, .99)), "median": float(np.median(maxz))}
    cnt = {}
    for th in (2.5, 3.0, 3.5, 4.0):
        pl = (np.abs(zs) >= th).sum(1)
        cnt[th] = dict(real=int((np.abs(tz) >= th).sum()), mean=float(pl.mean()), p95=float(np.quantile(pl, .95)), p99=float(np.quantile(pl, .99)))
    res["counts"] = cnt
    k2t = {k: (float(t), float((maxz >= abs(t)).mean())) for k, t in zip(keys, tz)}
    C.pl = {rid: k2t[k] for rid, k in keymap.items() if k in k2t}
    return res


def nested(C, df):
    """전반 성적만으로 슬롯별 상위 3규칙(겹침 Jaccard < 0.5)을 자동 선택 → 후반 평가. 후속 가공군(L·M·P·V·S)은 제외."""
    sel = df[df.exec_ok & df._fam.isin(list("ABCDEFGHIJK")) & (df._n1 >= 30) & (df._t1 >= 2.5) & (df._top5_1 < 50)]
    slots = {"KQ낮롱": ("F_KQ", ("D",), 1), "KQ낮숏": ("F_KQ", ("D",), -1), "KQ밤롱": ("F_KQ", ("N", "N0"), 1), "KP낮롱": ("F_KP", ("D",), 1), "KP낮숏": ("F_KP", ("D",), -1), "KP밤롱": ("F_KP", ("N", "N0"), 1)}
    n, cut = C.n, C.cut
    out, total = {}, np.zeros(n)
    tot_h1 = np.zeros(n)
    for nm, (vn, hs, side) in slots.items():
        g = sel[(sel._veh == vn) & (sel._side == side) & sel._H.isin(hs)].sort_values("_t1", ascending=False)
        picks, masks = [], []
        for rid, hh in zip(g.id, g._H):
            m = C.spec[rid][0].copy()
            if hh == "N":  # N(T 수급) = N0(T+1 축)
                m = np.r_[False, m[:-1]]
            m1 = m[:cut]
            if all(((m1 & x[:cut]).sum() / max(1, (m1 | x[:cut]).sum())) < 0.5 for x in masks):
                picks.append(df.loc[df.id == rid, "name"].iloc[0])
                masks.append(m)
            if len(masks) == 3:
                break
        if not masks:
            out[nm] = dict(picks=[], n2=0)
            continue
        u = np.logical_or.reduce(masks)
        H = "D" if hs == ("D",) else "N0"
        V = C.v[vn]
        s1 = (u * side).astype(float)
        s2 = s1.copy()
        s1[cut - 1:] = 0  # 전반 신호만
        s2[:cut - 1] = 0  # 후반 진입만
        r1, r2 = simulate(s1, H, V, n), simulate(s2, H, V, n)
        total += r2["daily"]
        tot_h1 += r1["daily"]
        d2 = r2["daily"][cut:]
        eq = np.cumprod(1 + d2)
        out[nm] = dict(picks=picks, n1=len(r1["net"]), net1=float(r1["net"].mean() * 1e4), t1=nw_t(r1["net"] * 1e4), n2=len(r2["net"]), net2=float(r2["net"].mean() * 1e4), t2=nw_t(r2["net"] * 1e4),
                       cagr2=float((eq[-1] ** (252 / len(d2)) - 1) * 100), mdd2=float((eq / np.maximum.accumulate(eq) - 1).min() * 100))
    d2 = total[cut:]
    eq = np.cumprod(1 + d2)
    y = pd.Series(d2, index=C.days[cut:])
    d1 = pd.Series(tot_h1[:cut], index=C.days[:cut]).loc["2017-01-01":]
    e1 = (1 + d1).cumprod()
    out["합산"] = dict(cagr2=float((eq[-1] ** (252 / len(d2)) - 1) * 100), mdd2=float((eq / np.maximum.accumulate(eq) - 1).min() * 100), yearly2={int(k): round(float((1 + g).prod() - 1) * 100, 1) for k, g in y.groupby(y.index.year)},
                     cagr1=float((e1.iloc[-1] ** (252 / len(d1)) - 1) * 100), mdd1=float((e1 / e1.cummax() - 1).min() * 100))
    return out


def notes_str(C, r):
    pl = C.pl.get(r.id)
    bits = [f"fam={r._fam}", f"veh={r._veh}", f"sig={r._sig}", f"th={r._th:g}" if not np.isnan(r._th) else "", f"delta={r._delta:.1f}" if not pd.isna(r._delta) else "", f"tdelta={r._tdelta:.2f}" if not pd.isna(r._tdelta) else "",
            f"top1={r._top1:.1f}" if not pd.isna(r._top1) else "", f"ban={r._ban:.1f}" if not pd.isna(r._ban) else "", f"adj={int(r._adj)}", f"pass={int(r._pass)}",
            f"h1={r._net1:.1f}/t{r._t1:.2f}/n{int(r._n1)}" if not pd.isna(getattr(r, "_net1", np.nan)) else "", f"h2={r._net2:.1f}/t{r._t2:.2f}/n{int(r._n2)}" if not pd.isna(getattr(r, "_net2", np.nan)) else "",
            f"t_std={pl[0]:.2f};p_fwer={pl[1]:.3f}" if pl else "", "깃 밖 자료" if (r._veh in ("F_KQ", "F_KP") or r._fam == "K") else "", str(r._extra) if isinstance(r._extra, str) and r._extra else ""]
    return ";".join(b for b in bits if b)


# ---------------------------------------------------------------- 그림
PAL = {"blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a", "yellow": "#eda100", "ink": "#52514e", "muted": "#a8a79f", "surface": "#fcfcfb"}


def _ax(ax):
    ax.set_facecolor(PAL["surface"])
    ax.grid(alpha=0.2)
    ax.tick_params(colors=PAL["ink"], labelsize=9)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(PAL["muted"])


def figures(C, df, books, rr):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    # 그림 1: 반분 재현
    d = df[df.exec_ok & df._fam.isin(list("ABCDEFGHIJK")) & (df._n1 >= 30) & (df._n2 >= 30)]
    fig, ax = plt.subplots(figsize=(8.2, 6), facecolor=PAL["surface"])
    ax.scatter(d._t1, d._t2, s=8, color=PAL["blue"], alpha=0.25, linewidths=0, label=f"전체 셀 {len(d):,}개")
    hl = df[(df._sig == "foreign.z60.k5") & (df._th == 1.5) & (df._H == "D") & (df._side == 1) & (df._fam == "A") & (df._tail == "hi")]
    ax.scatter(hl._t1, hl._t2, s=70, color=PAL["orange"], edgecolors=PAL["surface"], linewidths=1.5, zorder=3, label="KQ외인 5일 합 → 낮 롱")
    for _, r in hl.iterrows():
        ax.annotate(f"전반 t {r._t1:.2f} / 후반 t {r._t2:.2f}", (r._t1, r._t2), xytext=(8, 8), textcoords="offset points", fontsize=9, color=PAL["ink"])
    ax.axhline(0, color=PAL["muted"], lw=1)
    ax.axvline(0, color=PAL["muted"], lw=1)
    ax.axvline(2, color=PAL["ink"], lw=1, ls=":")
    ax.set_xlabel("전반 NW t (~2020-12-30)", color=PAL["ink"])
    ax.set_ylabel("후반 NW t (2021-01-04~2024-12-30)", color=PAL["ink"])
    rho = d._t1.corr(d._t2, method="spearman")
    ax.set_title(f"전반 t 가 후반을 약하게만 예측한다 (순위상관 {rho:.2f}, 비용 반영)", loc="left", fontsize=11, color="#0b0b0b")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    _ax(ax)
    fig.tight_layout()
    fig.savefig(OUT / "fig1_half_replication.png", dpi=140, facecolor=PAL["surface"])
    plt.close(fig)
    # 그림 2: 코스닥 수급 장부 누적
    fig, axs = plt.subplots(2, 1, figsize=(9.5, 7), height_ratios=[3, 1.3], facecolor=PAL["surface"])
    ax = axs[0]
    for nm, col, lab in (("낮", PAL["blue"], "낮 D12"), ("밤", PAL["aqua"], "밤 N12"), ("합", PAL["orange"], "낮 + 밤")):
        x = books["KQ_" + nm]["daily"]
        y = x.cumsum() * 100
        ax.plot(y.index, y.values, lw=2, color=col, label=lab)
        ax.annotate(f"{lab} +{y.iloc[-1]:.0f}%", (y.index[-1], y.iloc[-1]), xytext=(6, 0), textcoords="offset points", fontsize=9, color=PAL["ink"], va="center")
    ax.set_ylabel("누적 순손익 합(%, 명목 1배, 복리 아님)", color=PAL["ink"])
    ax.set_title("코스닥 수급 낮·밤 후보 누적 (설계 구간 2017~2024, 선물 왕복 1.8bp, 같은 창에서 고른 규칙)", loc="left", fontsize=11, color="#0b0b0b")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.margins(x=0.08)
    _ax(ax)
    ax = axs[1]
    yr = books["KQ_합"]["yearly"]
    xs = list(yr)
    vals = [yr[k] for k in xs]
    ax.bar(xs, vals, color=PAL["orange"], width=0.55)
    for a, b in zip(xs, vals):
        ax.annotate(f"{b:+.1f}%".replace("-", "−"), (a, b), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9, color=PAL["ink"])
    ax.axhline(0, color=PAL["muted"], lw=1)
    ax.set_ylabel("연수익(%)", color=PAL["ink"])
    ax.set_xticks(xs)
    ax.margins(y=0.2)
    _ax(ax)
    fig.tight_layout()
    fig.savefig(OUT / "fig2_kq_book.png", dpi=140, facecolor=PAL["surface"])
    plt.close(fig)
    # 그림 3: IBS × 수급 구간
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 4.4), sharey=True, facecolor=PAL["surface"])
    for ax, key, title in ((axs[0], "ibsL", "IBS < 0.3 롱 밤: T−1 KQ개인 z60 구간별"), (axs[1], "ibsS", "IBS > 0.7 숏 밤: T−1 KQ외인 z60 구간별")):
        t = rr[key]
        ax.bar(range(len(t)), [u["net"] for u in t], color=PAL["blue"], width=0.55)
        for i, u in enumerate(t):
            ax.annotate(f"{u['net']:+.1f}bp\nn {u['n']}".replace("-", "−"), (i, u["net"]), xytext=(0, 4 if u["net"] >= 0 else -4), textcoords="offset points", ha="center", va="bottom" if u["net"] >= 0 else "top", fontsize=8.5, color=PAL["ink"])
        ax.set_xticks(range(len(t)), [u["lab"] for u in t], fontsize=8.5)
        ax.axhline(0, color=PAL["muted"], lw=1)
        ax.set_title(title, loc="left", fontsize=10.5, color="#0b0b0b")
        ax.margins(y=0.22)
        _ax(ax)
    axs[0].set_ylabel("밤당 순수익(bp, F_KQ 1.8bp)", color=PAL["ink"])
    fig.tight_layout()
    fig.savefig(OUT / "fig3_ibs_flow.png", dpi=140, facecolor=PAL["surface"])
    plt.close(fig)


def self_check(C, RR):
    """엔진과 무관한 독립 재계산으로 핵심 규칙 대조. 어긋나면 중단."""
    import statsmodels.api as sm
    x = np.random.default_rng(0).normal(0.3, 1, 400)
    assert abs(nw_t(x) - sm.OLS(x, np.ones(400)).fit(cov_type="HAC", cov_kwds={"maxlags": 5}).tvalues[0]) < 1e-9
    # z 는 T 까지 자른 자료로 다시 계산해도 같아야 한다(미래 미사용)
    T = pd.Timestamp("2020-06-30")
    ff = C.kq.foreign.where(C.kq.foreign != 0)
    assert abs(z_k(ff, 60, 5, C.days).loc[T] - z_k(ff.loc[:T], 60, 5, C.days[C.days <= T]).loc[T]) < 1e-12
    raw = {}
    for nm, f in (("KQ", "kis_fut_kq150_daily.csv"), ("KP", "kis_fut_k200_daily.csv")):
        d = sealed(pd.read_csv(RAW / "r15_external" / f))
        raw[nm] = d
    out = {}
    for key, nm in (("D1", "KQ"), ("K2", "KP")):
        d = raw[nm]
        fr = d[d.front == 1].set_index("date").sort_index()
        r = fr.close / fr.open - 1
        idx = np.flatnonzero(RR[key]["m"])
        idx = idx[idx + 1 < C.n]
        t1 = C.days[idx + 1]
        g = RR[key]["side"] * r.reindex(t1).values - 1.8e-4
        g = g[~np.isnan(g)]
        out[key] = (float(g.mean() * 1e4), float(RR[key]["row"]["net_bp"]))
        assert abs(g.mean() * 1e4 - RR[key]["row"]["net_bp"]) < 1e-6, (key, out[key])
    # N1: T front 종가 매수 → T+1 같은 계약 시가 매도
    d = raw["KQ"]
    fr = d[d.front == 1].set_index("date").sort_index()
    op = d.set_index(["date", "code"]).open
    idx = np.flatnonzero(RR["N1"]["m"])
    idx = idx[idx + 1 < C.n]
    T0, T1 = C.days[idx], C.days[idx + 1]
    code = fr.code.reindex(T0).values
    e, x_ = fr.close.reindex(T0).values, op.reindex(pd.MultiIndex.from_arrays([T1, code])).values
    g = x_ / e - 1 - 1.8e-4
    g = g[~np.isnan(g)]
    out["N1"] = (float(g.mean() * 1e4), float(RR["N1"]["row"]["net_bp"]))
    assert abs(g.mean() * 1e4 - RR["N1"]["row"]["net_bp"]) < 1e-6, out["N1"]
    print("  자체 점검 통과(독립 재계산 순 bp, 엔진 순 bp):", {k: (round(a, 4), round(b, 4)) for k, (a, b) in out.items()})
    return out


def price_control(C):
    """수급 더미가 가격 정보로 설명되는지: 같은 슬롯 수익을 수급 더미 + 가격 변수에 회귀(HAC 5). 셀이 아닌 진단."""
    import statsmodels.api as sm
    days = C.days
    ff = C.kq.foreign.where(C.kq.foreign != 0)
    z5, i5 = z_k(ff, 60, 5, days), z_k(C.kq.institution, 60, 5, days)
    zi, zf, zk = z_k(C.kq.institution, 250, 1, days), z_k(ff, 250, 1, days), z_k(C.fu.net_foreign, 250, 5, days)
    ki = z_k(C.fu.net_individual, 60, 1, days)
    gq, gk = pd.Series(C.fwd["F_KQ", "G0"], index=days) * 100, pd.Series(C.fwd["F_KP", "G0"], index=days) * 100

    def fit(y, X, lo):
        d = pd.concat([y.rename("y"), X], axis=1).dropna().loc[lo:]
        m = sm.OLS(d.y, sm.add_constant(d.drop(columns="y"))).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
        return dict(n=len(d), b={k: round(float(v), 2) for k, v in m.params.items()}, t={k: round(float(v), 2) for k, v in m.tvalues.items()})
    yd, yn = pd.Series(C.fwd["F_KQ", "D"], index=days) * 1e4, pd.Series(C.fwd["F_KQ", "N"], index=days) * 1e4
    yk = pd.Series(C.fwd["F_KP", "D"], index=days) * 1e4
    return {"KQ낮": fit(yd, pd.DataFrame({"외5≥1.5": (z5 >= 1.5).astype(float), "기5≥1.5": (i5 >= 1.5).astype(float), "rT%": C.r_kq * 100, "r5%": C.kq150.Close.pct_change(5) * 100, "gap%": gq}), "2016-06-01"),
            "KQ밤": fit(yn, pd.DataFrame({"기관≤−1": (zi <= -1).astype(float), "외인≥1": (zf >= 1).astype(float), "K외5≥1.5": (zk >= 1.5).astype(float), "rT%": C.r_kq * 100, "rT+1%": C.r_kq.shift(-1) * 100, "rK%": C.r_kp * 100, "rK+1%": C.r_kp.shift(-1) * 100}), "2016-06-01"),
            "KP낮": fit(yk, pd.DataFrame({"개인≥0.5": (ki >= 0.5).astype(float), "rT%": C.r_kp * 100, "r5%": C.k2.close.pct_change(5) * 100, "gap%": gk}), "2011-04-01")}


def bucket_paths(C):
    """z 구간별 이벤트 경로: 꼬리 효과의 모양과 현물 대조."""
    days = C.days
    FQ, FP = C.v["F_KQ"], C.v["F_KP"]
    z5 = z_k(C.kq.foreign.where(C.kq.foreign != 0), 60, 5, days)
    sz = z_k(C.kd.net_foreign, 60, 5, days)
    iz = z_k(C.fu.net_individual, 60, 1, days)
    out = {"KQ외5": {}, "코스닥현물외5": {}, "K200개인": {}}
    for lab, m in (("z≥1.5", z5 >= 1.5), ("1~1.5", (z5 >= 1) & (z5 < 1.5)), ("0.5~1", (z5 >= .5) & (z5 < 1)), ("−0.5~0.5", z5.abs() < .5), ("−1~−0.5", (z5 > -1) & (z5 <= -.5)), ("z≤−1", z5 <= -1)):
        out["KQ외5"][lab] = event_path(C, m.values, FQ)
    for lab, m in (("z≥1.5", sz >= 1.5), ("z≥1", sz >= 1), ("z≤−1.5", sz <= -1.5)):
        out["코스닥현물외5"][lab] = event_path(C, m.values, FQ)
    for lo, hi in ((-9, -1), (-1, -.5), (-.5, 0), (0, .5), (.5, 1), (1, 1.5), (1.5, 9)):
        out["K200개인"][f"{lo}~{hi}"] = event_path(C, ((iz >= lo) & (iz < hi)).values, FP)
    return out


def write_notes(df, res):
    L = ["# 16차 스캔 flows 실행 기록", "", "• 코드: r16s_flows.py, 봉인 2024-12-30, 모든 파일 읽은 직후 sealed() assert.",
         f"• 셀 {res['n_cells']}행(cells.csv 전부), 선별 통과 {res['pass_n']}행. 통과 판정은 notes 열 pass=1.",
         "• 지평 표기: D 낮, N 밤(T 수급 → T+1 밤), N0 밤(T−1 수급·T IBS → T 밤), F 낮+밤, 3·5·10 스윙. 차량: F_KQ·F_KP 선물 front(깃 밖), E_KQ 229200/251340, E_KP 069500/114800, L_KQ 233740, G_KP FUT.csv(깃, 낮만), I_KP 지수(체결 불가).",
         "• cells.csv notes 키: delta·tdelta(신호일−비교일 Δ, HAC t), top1(상위 1건 비중), ban(공매도 금지 구간 손익 비중), adj(인접 문턱 부호 유지), pass, h1·h2(달력 반분 순 bp/t/n), t_std·p_fwer(학생화 위약).", "", "## 군별", ""]
    for f, g in df.groupby("_fam"):
        L.append(f"• {f}: 셀 {len(g)}, 순수익>0 ∧ t≥2 {int(((g.t_nw >= 2) & (g.net_bp > 0)).sum())}, t≥3 {int(((g.t_nw >= 3) & (g.net_bp > 0)).sum())}, 통과 {int(g._pass.sum())}")
    p = res["placebo"]
    L += ["", "## 위약(신호 원형 이동 2000회, 학생화)", "", f"• 고유 검정 {p['tests']}, 실제 최대 |t_std| {p['max_real']:.2f}, 위약 최대값 중앙 {p['median']:.2f}·95% {p['p95']:.2f}·99% {p['p99']:.2f}"]
    for th, c in p["counts"].items():
        L.append(f"• |t_std| ≥ {th}: 실제 {c['real']}, 위약 평균 {c['mean']:.1f}, 95% {c['p95']:.0f}, 99% {c['p99']:.0f}")
    L += ["", "## 전반 자동 선별 → 후반", ""]
    for k, v in res["nested"].items():
        if k == "합산":
            L.append(f"• 합산: 전반 내부(2017~2020) CAGR {v['cagr1']:.1f}% MDD {v['mdd1']:.1f}%, 후반(2021~2024) CAGR {v['cagr2']:.1f}% MDD {v['mdd2']:.1f}%, 연도 {v['yearly2']}")
        elif v.get("n2"):
            L.append(f"• {k}: 전반 n {v['n1']} {v['net1']:+.1f}bp t {v['t1']:.2f} → 후반 n {v['n2']} {v['net2']:+.1f}bp t {v['t2']:.2f}, 후반 CAGR {v['cagr2']:.1f}% MDD {v['mdd2']:.1f}%, 규칙 {v['picks']}")
    L += ["", "## 후보 규칙", ""]
    for k, r in res["rules"].items():
        L.append(f"• {k}: {r['name']} | n {r['n']} 총 {r['gross_bp']:+.2f} 순 {r['net_bp']:+.2f} t {r['t_nw']:.2f} CAGR {r['cagr_pct']:.2f}% MDD {r['mdd_pct']:.2f}% 연도 {r['yearly_pos']} 상위5 {r['top5_share_pct']:.1f}% 전·후반 {r['half1_net_bp']:+.1f}/{r['half2_net_bp']:+.1f} 스트레스 {r['stress_net_bp']:+.1f} 달력 전반 {r['_net1']:+.1f}(t {r['_t1']:.2f}) 후반 {r['_net2']:+.1f}(t {r['_t2']:.2f}) 상관 {r['corr_base']}")
    L += ["", "## 인접 문턱(규칙별 문턱, 순 bp, t, n)", ""]
    for k, v in res["adj"].items():
        L.append(f"• {k}: " + "; ".join(f"{a} {b:+.1f}/{c:.2f}/{d}" for a, b, c, d in v))
    L += ["", "## 장부", ""]
    for k, b in res["books"].items():
        r = b["row"]
        L.append(f"• {k}: {r['window']} CAGR {r['cagr_pct']:.1f}% MDD {r['mdd_pct']:.1f}% 샤프 {b['sharpe']:.2f} 활성 {b['active'] * 100:.0f}% 연수익 {b['yearly']} 상관 {r['corr_base']}")
    L += [f"• 전·후반: {res['splits']}", "", "## 기제·보조", "", f"• 개인 레버리지−인버스 순매수(깃 밖)와 KQ선물 수급 상관: {res['mech']}", f"• D1·D2 다음날 229200 구간별(2022-11~): {res['minute_profile']}", f"• blog_ON_Q 오버레이 2017~2024: {res['overlay']}",
          f"• 가격 통제 회귀: {res['price_control']}", "", "## z 구간별 이벤트 경로(구간 평균 bp, 단순 t, n)", ""]
    for g, d in res["paths"].items():
        for lab, p in d.items():
            L.append(f"• {g} {lab}: " + " ".join(f"{a} {b[0]:+.1f}({b[1]:+.2f})" for a, b in p.items()) + f" n {p['D+1'][2]}")
    (OUT / "notes.md").write_text(re.sub(r"(?<![\w.])-(?=\d)", "−", "\n".join(L)) + "\n", encoding="utf-8")  # 음수 표기 −, 날짜 하이픈은 유지


# ---------------------------------------------------------------- 메인
def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    print("[자료]")
    C = load()
    assert C.days[-1] == SEAL
    print(f"  달력 {C.days[0]:%Y-%m-%d}~{C.days[-1]:%Y-%m-%d} {C.n}일")
    C.defined = np.ones(C.n, bool)
    for nm, fn in (("A", fam_A), ("B", fam_B), ("C", fam_C), ("D", fam_D), ("E", fam_E), ("F", fam_F), ("G", fam_G), ("H", fam_H), ("I", fam_I), ("J", fam_J), ("K", fam_K), ("S", fam_S), ("L", fam_L), ("M", fam_M)):
        k0 = len(C.rows)
        fn(C)
        print(f"  {nm}: {len(C.rows) - k0}셀, 누적 {len(C.rows)} ({time.time() - t0:.0f}s)")
    # 후보 규칙·차량 교차
    RR = run_rules(C)
    SC = self_check(C, RR)
    AD = adj_rules(C)
    # IBS × 수급 구간(그림 3)
    p = etf_px("229200", C.days)
    ibs = ibs_of(p)
    zp = z_k(C.kq.individual, 60, 1, C.days).shift(1)
    zf = z_k(C.kq.foreign.where(C.kq.foreign != 0), 60, 1, C.days).shift(1)
    C.defined = ibs.notna().values
    bk = [(-99, -1, "≤−1"), (-1, -0.5, "−1~−0.5"), (-0.5, 0, "−0.5~0"), (0, 0.5, "0~0.5"), (0.5, 1, "0.5~1"), (1, 99, "≥1")]
    rr = {"ibsL": [], "ibsS": []}
    for key, base_m, z, side, lab in (("ibsL", (ibs < 0.3), zp, 1, "IBS<0.3 ∧ KQ개인(T−1)"), ("ibsS", (ibs > 0.7), zf, -1, "IBS>0.7 ∧ KQ외인(T−1)")):
        for lo, hi, tag in bk:
            m = (base_m & (z >= lo) & (z < hi)).values
            row = cell(C, "J", f"bucket.{key}.{tag}", f"{lab} z60 {tag} {'롱' if side > 0 else '숏'} F_KQ N0", m, side, "N0", C.v["F_KQ"], th=lo, tail="bucket", base=(base_m & z.notna()).values)
            rr[key].append(dict(lab=tag, net=row["net_bp"], n=row["n"], t=row["t_nw"]))
    # 짧은 창 재현: 분봉 15:19 IBS
    im = ibs_minute(C)
    C.defined = im.notna().values
    zp2 = zp
    m_min = ((im < 0.3) & (zp2 >= 0)).values
    m_all = (im < 0.3).values
    for nm, m in (("분봉 IBS<0.3", m_all), ("분봉 IBS<0.3 ∧ KQ개인(T−1)≥0", m_min)):
        cell(C, "J", "minute." + nm, f"{nm} 15:45 F_KQ N0(15:19 확정)", m, 1, "N0", C.v["F_KQ"], th=0.3, tail="minute")
        cell(C, "J", "minute." + nm, f"{nm} 15:30 233740 N0(15:19 확정, 2배)", m, 1, "N0", C.v["L_KQ"], th=0.3, tail="minute")
    # 장부(같은 창에서 고른 규칙, 명목 1배)
    lo, hi = "2017-01-02", "2024-12-30"
    s_ = {k: np.nan_to_num(RR[k]["m"].astype(float) * RR[k]["side"] * (1.0 if RR[k]["w"] is None else RR[k]["w"])) for k in ("D1", "D2", "D12", "N1", "N2", "N12", "K1", "K2", "K3", "K4", "K5", "K6")}
    FQ, FP = C.v["F_KQ"], C.v["F_KP"]
    B = {}
    B["KQ_낮"] = book(C, "장부 KQ 낮 D12", [(s_["D12"], "D", FQ, 1)], lo, hi, "KOSDAQ", "day")
    B["KQ_밤"] = book(C, "장부 KQ 밤 N12", [(s_["N12"], "N0", FQ, 1)], lo, hi, "KOSDAQ", "night")
    B["KQ_합"] = book(C, "장부 KQ 낮 D12 + 밤 N12", [(s_["D12"], "D", FQ, 1), (s_["N12"], "N0", FQ, 1)], lo, hi, "KOSDAQ", "mixed")
    B["KQ_낮롱합"] = book(C, "장부 KQ 낮 D1 + 밤 N12", [(s_["D1"], "D", FQ, 1), (s_["N12"], "N0", FQ, 1)], lo, hi, "KOSDAQ", "mixed")
    B["KP_낮K1"] = book(C, "장부 KP 낮 K1", [(s_["K1"], "D", FP, 1)], lo, hi, "KOSPI", "day")
    B["KP_낮K2"] = book(C, "장부 KP 낮 K2", [(s_["K2"], "D", FP, 1)], lo, hi, "KOSPI", "day")
    B["KP_밤"] = book(C, "장부 KP 밤 K5", [(s_["K5"], "N0", FP, 1)], lo, hi, "KOSPI", "night")
    B["KP_합1"] = book(C, "장부 KP 낮 K1 + 밤 K5", [(s_["K1"], "D", FP, 1), (s_["K5"], "N0", FP, 1)], lo, hi, "KOSPI", "mixed")
    B["KP_합2"] = book(C, "장부 KP 낮 K2 + 밤 K5", [(s_["K2"], "D", FP, 1), (s_["K5"], "N0", FP, 1)], lo, hi, "KOSPI", "mixed")
    legs_all = [(s_["D12"], "D", FQ, 1), (s_["N12"], "N0", FQ, 1), (s_["K1"], "D", FP, 1), (s_["K5"], "N0", FP, 1)]
    B["전체_1대1"] = book(C, "장부 KQ·KP 각 명목 1배 합", legs_all, lo, hi, "KOSPI+KOSDAQ", "mixed")
    B["전체_반반"] = book(C, "장부 KQ·KP 자본 반반", [(a, b, c, 0.5) for a, b, c, _ in legs_all], lo, hi, "KOSPI+KOSDAQ", "mixed")
    B["전체_0.65"] = book(C, "장부 KQ·KP 각 명목 0.65배", [(a, b, c, 0.65) for a, b, c, _ in legs_all], lo, hi, "KOSPI+KOSDAQ", "mixed")
    # 핵심 3다리(D1·K2·N1)만, 각 명목 1배: 다중 비교 부담이 가장 작은 조합
    B["핵심3"] = book(C, "장부 핵심 3다리 D1 + K2 + N1", [(s_["D1"], "D", FQ, 1), (s_["K2"], "D", FP, 1), (s_["N1"], "N0", FQ, 1)], lo, hi, "KOSPI+KOSDAQ", "mixed")
    splits = {k: {"2017~2020": win_stat(B[k]["daily"].loc[:"2020-12-30"]), "2021~2024": win_stat(B[k]["daily"].loc["2021-01-04":])} for k in ("KQ_합", "KP_합1", "전체_반반", "전체_0.65", "핵심3")}
    # 후보 부가 정보
    extra = {}
    for k in ("D1", "D2", "N1", "N2", "K1", "K2", "K3", "K4", "K6"):
        r = RR[k]
        sim = simulate(np.nan_to_num(r["m"].astype(float) * r["side"]), r["H"], r["V"], C.n)
        tx = C.days[sim["tx"]]
        s = pd.Series(sim["net"] * 1e4, index=tx)
        y = s.groupby(tx.year).agg(["count", "sum", "mean"]).round(1)
        a18 = s.loc["2018-01-01":]
        xb = s[(s.index < BAN[0])]
        extra[k] = dict(year={int(i): (int(u["count"]), float(u["sum"]), float(u["mean"])) for i, u in y.iterrows()}, y2018=(len(a18), float(a18.mean()), nw_t(a18.values)), exban=(len(xb), float(xb.mean()), nw_t(xb.values)), top=[f"{i:%Y-%m-%d} {v:+.0f}bp" for i, v in s.nlargest(5).items()],
                        bottom=[f"{i:%Y-%m-%d} {v:+.0f}bp" for i, v in s.nsmallest(3).items()], path=event_path(C, r["m"], r["V"]) if r["H"] == "D" else None)
    prof = {k: minute_profile(C, RR[k]["m"], RR[k]["side"]) for k in ("D1", "D2")}
    mech = mech_check(C)
    # blog_ON_Q 오버레이(봉인 전)
    bq = sealed(pd.read_csv(LAB / "round13_slot_book/series/blog_ON_Q.csv")).set_index("date")
    bq = bq.reindex(C.days).fillna(0.0)
    zpf = zp.reindex(C.days)  # T−1 값이 T 인덱스. blog 는 청산일 표기라 진입일 = 전 거래일
    zen = zpf.shift(1)
    zff = zf.reindex(C.days).shift(1)
    ov = {}
    for nm, w in (("원 장부", pd.Series(1.0, index=C.days)), ("롱은 KQ개인≥0 만, 숏 유지", pd.Series(np.where((bq.pos > 0) & ~(zen >= 0), 0.0, 1.0), index=C.days)), ("롱은 KQ개인≥0 만, 숏 제거", pd.Series(np.where(((bq.pos > 0) & ~(zen >= 0)) | (bq.pos < 0), 0.0, 1.0), index=C.days)),
                  ("롱 유지, 숏 제거", pd.Series(np.where(bq.pos < 0, 0.0, 1.0), index=C.days))):
        x = (bq.net * w).loc["2017-01-02":]
        e = (1 + x).cumprod()
        ov[nm] = dict(cagr=float((e.iloc[-1] ** (252 / len(x)) - 1) * 100), mdd=float((e / e.cummax() - 1).min() * 100), n=int((x != 0).sum()))
    df = post(C)
    print("[위약]")
    pl = placebo(C)
    print(json.dumps(pl, ensure_ascii=False))
    nest = nested(C, df)
    print("[전반 자동 선별 → 후반]")
    print(json.dumps(nest, ensure_ascii=False, default=float, indent=0)[:3000])
    df["notes"] = [notes_str(C, SimpleNamespace(**r)) for r in df.to_dict("records")]
    df["t_std"] = [C.pl.get(i, (np.nan, np.nan))[0] for i in df.id]
    df["p_fwer"] = [C.pl.get(i, (np.nan, np.nan))[1] for i in df.id]
    out = df[CELL_COLS]
    out.to_csv(OUT / "cells.csv", index=False, encoding="utf-8-sig", float_format="%.4f")
    figures(C, df, B, rr)
    res = dict(n_cells=len(df), pass_n=int(df._pass.sum()), placebo=pl, nested=nest, books={k: dict(row={u: v.get(u) if not isinstance(v, pd.Series) else None for u, v in [(u, b["row"]) for u in ("name", "n", "net_bp", "t_nw", "cagr_pct", "mdd_pct", "yearly_pos", "corr_base", "window", "stress_net_bp", "top5_share_pct", "half1_net_bp", "half2_net_bp", "_ban")]},
                                                                                                   yearly=b["yearly"], sharpe=b["sharpe"], active=b["active"], worst=b["worst"], worst_day=b["worst_day"]) for k, b in B.items()},
               rules={k: {u: v.get(u) for u, v in [(u, RR[k]["row"]) for u in ("name", "n", "gross_bp", "net_bp", "t_nw", "cagr_pct", "mdd_pct", "yearly_pos", "top5_share_pct", "_top1", "half1_net_bp", "half2_net_bp", "stress_net_bp", "corr_base", "window", "_delta", "_tdelta", "_ban", "_net1", "_t1", "_n1", "_net2", "_t2", "_n2", "exec_ok")]} for k in RR},
               extra=extra, minute_profile=prof, mech=mech, overlay=ov, ibs_bucket=rr, adj=AD, splits=splits, self_check=SC, price_control=price_control(C), paths=bucket_paths(C))
    (OUT / "summary.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    write_notes(df, res)
    df.to_pickle(SCRATCH / "flows_df.pkl")
    print("통과", int(df._pass.sum()), "/", len(df), f"끝 {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()


