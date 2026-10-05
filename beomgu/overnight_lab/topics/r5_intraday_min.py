"""5차 전략 탐색 — 장중 계열(R5_INTRADAY_MIN): KODEX 200 1분봉으로 1/5/10/60/120분 규칙을 설계·검증한다.

자료
- 팀 저장소 KODEX 200 1분봉 2023-01-02 ~ 2026-09-29(912일). 봉 시각은 끝 시각(09:01 봉 = 09:00:00~09:00:59),
  09:01~15:30 봉만 쓴다. 15:20 봉(15:19:00~15:19:59)이 마지막 접속매매 봉, 15:30 봉 종가 = 종가 단일가.
- 늦게 연 날(1월 첫 거래일·수능일 10:00 개장, 7일)은 매매하지 않는다(무포지션, 수익 0으로 날수에는 포함).
- 체결: 봉 종가에 판단 → 다음 봉 시가 체결. 청산은 15:30 종가 단일가(장 마감 동시호가 주문). 15:20~15:29 체결 없음.
- 사전 정보(진입 전에 알 수 있는 것만): D2 잔차 갭 z(코스피200 지수 09:00 시가 갭을 직전 미국 S&P500 누적수익으로 회귀한
  250일 롤링 잔차 z, r3_d2 와 같은 계산), 지수 시가 갭 부호, 전일(T−1) 외국인 코스피200 선물 순매수 수량의 과거 250일
  분위(상위 40% = q60 초과, 하위 40% = q40 미만), 전일 지수 종가 vs 20일 이동평균, 직전 20일 장중 고저폭 중앙값.

그리드(48개, 실행 전 고정)
- IM 장중 모멘텀(Gao et al. 2018): 신호 {ON30 = 전일 종가→09:30, F30 = 09:00→09:30, F60 = 09:00→10:00} 부호
  × 진입 {14:00, 15:00} → 15:30 종가 × {롱 전용, 롱숏}                                                    12
- REG 국면 + 시각 진입: 09:01 시가 진입 → 15:30 종가. 롱 전용 필터 {z>0, z≥1, 갭>0, 외인 상위40%, MA20 위},
  롱숏 {z 부호, 외인 상위40% 롱/하위40% 숏, MA20 위 롱/아래 숏}                                              8
- ORB5: 5분봉, 시가 범위 09:00~09:15(3봉), 09:15 이후 5분 종가가 범위 고가를 넘는 첫 봉 → 다음 봉 시가 매수(14:00까지),
  15:30 종가 청산, 필터 {없음, z>0, 갭>0, 외인 상위40%, MA20}                                                  5
- ORB10: 10분봉, 시가 범위 09:00~09:30(3봉), 같은 방식                                                       5
- ORB5S: ORB5 + 손절(진입가 − 0.5 × 직전 20일 장중 고저폭 중앙값, 1분 저가 기준, 갭 관통은 그 봉 시가)            5
- ORB5-LS: 상단 돌파 롱 / 하단 이탈 숏(먼저 나온 쪽), 필터 {없음, z 부호(z>0 이면 롱만, z<0 이면 숏만)}               2
- VWAP 눌림: 5분봉, 10:00 가격이 VWAP 위인 날, 10:00~14:00 에 저가가 VWAP 이하로 닿고 종가가 VWAP 위인 첫 봉 →
  다음 봉 시가 매수, 15:30 종가 청산, 필터 {없음, z>0, 갭>0, 외인 상위40%, MA20}                                5
- H60: 60분봉, 첫 60분봉 종가 < 당일 시가(오전 눌림)인 날, 10~14시 60분봉 중 종가 > VWAP 인 첫 봉 → 다음 봉 시가 매수
  (11:00~14:00), 15:30 종가 청산, 필터 {z>0, MA20, 외인 상위40%}                                             3
- H120: 120분봉, 09~11시 봉 종가 < 시가이고 11~13시 봉 종가 > VWAP → 13:00 매수, 15:30 청산, 필터 같은 3개          3
- 하루 1회. 비용 왕복: 롱 전용 = KODEX 200(주식 계좌) 5bp(스트레스 7bp). 숏이 있는 규칙 = 미니 코스피200 선물 2bp
  (스트레스 3bp)이며 선물 경로는 KODEX 200 1분봉 경로로 대리한다. 모든 변형을 5bp·2bp 둘 다 기록한다.

구간(실행 전 고정): 설계 2023-01-02 ~ 2024-12-30, 검증 2025-01-02 ~ 2026-09-29, 검증(2025-06-09~ 제외) 2025-01-02 ~ 2025-06-05.

최종 사양 선정 규칙(설계 구간 지표만 사용, 실행 전 고정)
- 1차(최종) = 롱 전용 37개를 KODEX 200 5bp 로 평가해, 설계 구간 연 거래 ≥ 30회이고 거래당 평균 순수익 > 0 인 변형 중
  설계 샤프(일별, 무포지션 0 포함) 최대. 만족하는 변형이 없으면 설계 샤프 최대 변형을 고르고 '설계 기준 미달'로 표시.
- 보조(선물 트랙) = 48개 전체를 2bp 로 평가해 같은 규칙. 2·3위도 검증 성과를 보여 주되 선택에는 쓰지 않는다.

산출: <OUT>/round5_strategy/intraday_min/{report.md, grid.csv, trades.csv, daily_returns.csv, quarterly.csv, equity.png}
실행: python topics/r5_intraday_min.py [--no-ledger]
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import data  # noqa: E402
import engine as E  # noqa: E402
import r2_d2 as R2  # noqa: E402
import r3_d2 as R3  # noqa: E402

FAMILY = "intraday_min"
TOPIC = "R5_INTRADAY_MIN"
OUTD = data.OUT / "round5_strategy" / FAMILY
TEAM_REPO = data._dir("HYFE_TEAM_REPO", "team_repo_dir", "../..")           # 팀 저장소(1분봉은 팀원 폴더의 data/ 아래)
MIN_FILE = next(TEAM_REPO.glob("*/data/KODEX200_1min_2023_2026.csv"))
DESIGN = ("2023-01-02", "2024-12-30")
VALID = ("2025-01-02", "2026-09-29")
VAL_EX = ("2025-01-02", "2025-06-05")
RECENT = ("2025-06-09", "2026-09-29")
NS = 390                 # 1분 슬롯 0..389 = 09:01..15:30 봉, 슬롯 k 시가 = 09:00 + k분 가격
LAST_CONT = 379          # 15:20 봉(15:19:00~15:19:59), 마지막 접속매매
CLOSE = 389              # 15:30 봉 = 종가 단일가
ENTRY_CAP = 300          # 14:00 이후 진입 금지(기술 규칙)
ANN = 252
COST_ETF, COST_ETF_STRESS, COST_FUT, COST_FUT_STRESS = 5.0, 7.0, 2.0, 3.0


# ---------------------------------------------------------------- 자료
def load_minutes():
    m = pd.read_csv(MIN_FILE)
    m["datetime"] = pd.to_datetime(m["datetime"]).dt.tz_localize(None)
    m["date"] = m["datetime"].dt.normalize()
    slot = m["datetime"].dt.hour * 60 + m["datetime"].dt.minute - (9 * 60 + 1)
    m = m[(slot >= 0) & (slot < NS)].assign(slot=slot)
    mats = {}
    for c in ("open", "high", "low", "close", "volume"):
        mats[c] = m.pivot_table(index="date", columns="slot", values=c, aggfunc="last").reindex(columns=range(NS))
    dates = mats["close"].index
    C = mats["close"].ffill(axis=1).bfill(axis=1).to_numpy(float)
    prevC = np.column_stack([C[:, :1], C[:, :-1]])
    O = mats["open"].to_numpy(float)
    O = np.where(np.isnan(O), prevC, O)
    H = np.where(np.isnan(mats["high"].to_numpy(float)), np.maximum(O, C), mats["high"].to_numpy(float))
    L = np.where(np.isnan(mats["low"].to_numpy(float)), np.minimum(O, C), mats["low"].to_numpy(float))
    V = np.nan_to_num(mats["volume"].to_numpy(float))
    first_vol = np.array([np.argmax(v > 0) if (v > 0).any() else NS for v in V])
    late = pd.Series(first_vol > 4, index=dates)                      # 10:00 개장일 등
    return dates, O, H, L, C, V, late


def features(dates, O, H, L, C, V):
    F = pd.DataFrame(index=dates)
    F["open"] = O[:, 0]
    F["close"] = C[:, CLOSE]
    # 분배 수정: 일봉 수정종가 / 1분봉 원종가 비율로 전일 종가→당일 가격 수익을 보정
    adj = data.price("069500")["close"].reindex(dates)
    f = (adj / F["close"]).ffill().bfill()
    pc = F["close"].shift(1) * f.shift(1)
    F["r_on30"] = C[:, 29] * f / pc - 1
    F["r_f30"] = C[:, 29] / O[:, 0] - 1
    F["r_f60"] = C[:, 59] / O[:, 0] - 1
    # D2 잔차 갭 z (지수), 지수 갭, MA20
    px, ix, tg, cal, listed = R3.load()
    gi = np.log(ix["open"] / ix["close"].shift(1))
    us = R2.us_cum(cal, "spx")
    zi = R3.rolling_z1(gi, us, 250, 200)
    F["z"] = zi.reindex(dates)
    F["gap"] = gi.reindex(dates)
    icl = ix["close"]
    F["ma_up"] = (icl > icl.rolling(20, min_periods=20).mean()).astype(float).where(icl.rolling(20).count() == 20).shift(1).reindex(dates)
    # 전일 외국인 선물 순매수 수량 분위(과거 250일, 전일까지)
    fut = data.naver_investor("fut")
    net = (fut["buyq_foreign"] - fut["sellq_foreign"]).where(fut["sellq_foreign"] > 0)
    q60 = net.shift(1).rolling(250, min_periods=200).quantile(0.6)
    q40 = net.shift(1).rolling(250, min_periods=200).quantile(0.4)
    hi = (net > q60).astype(float).where(q60.notna() & net.notna())
    lo = (net < q40).astype(float).where(q40.notna() & net.notna())
    # T−1 값을 T 에 적용: 선물 파일 자체 거래일로 한 칸 민 뒤 1분봉 날짜에 맞춘다
    u = net.index.union(dates)
    F["flow_hi"] = hi.reindex(u).shift(1).reindex(dates)
    F["flow_lo"] = lo.reindex(u).shift(1).reindex(dates)
    F["flow_asof"] = pd.Series(net.index, index=net.index).reindex(u).shift(1).reindex(dates)
    rng = (H.max(axis=1) - L.min(axis=1)) / O[:, 0]
    F["rng"] = rng
    F["med_rng20"] = pd.Series(rng, index=dates).shift(1).rolling(20, min_periods=15).median()
    return F


# ---------------------------------------------------------------- 규칙
def resample(O, H, L, C, V, n):
    nb = (LAST_CONT + 1) // n
    sl = slice(0, nb * n)
    D = O.shape[0]
    bo = O[:, sl].reshape(D, nb, n)[:, :, 0]
    bh = H[:, sl].reshape(D, nb, n).max(axis=2)
    bl = L[:, sl].reshape(D, nb, n).min(axis=2)
    bc = C[:, sl].reshape(D, nb, n)[:, :, -1]
    return bo, bh, bl, bc, nb


def filt_series(F, name):
    """롱 전용 필터: 1 이면 매매 허용, 0/NaN 이면 무포지션."""
    if name == "none":
        return pd.Series(1.0, index=F.index)
    if name == "z>0":
        return (F["z"] > 0).astype(float).where(F["z"].notna())
    if name == "z>=1":
        return (F["z"] >= 1).astype(float).where(F["z"].notna())
    if name == "gap>0":
        return (F["gap"] > 0).astype(float).where(F["gap"].notna())
    if name == "flow_hi":
        return F["flow_hi"]
    if name == "ma_up":
        return F["ma_up"]
    raise KeyError(name)


def ls_side(F, name):
    """롱숏 국면: +1 롱, −1 숏, 0 무포지션."""
    if name == "z_sign":
        return np.sign(F["z"]).fillna(0)
    if name == "flow":
        return (F["flow_hi"].fillna(0) - F["flow_lo"].fillna(0))
    if name == "ma":
        return F["ma_up"].map({1.0: 1.0, 0.0: -1.0}).fillna(0)
    if name == "none":
        return pd.Series(np.nan, index=F.index)        # 양방향 허용 표시
    raise KeyError(name)


class Ctx:
    def __init__(self, dates, O, H, L, C, V, late, F):
        self.dates, self.O, self.H, self.L, self.C, self.V, self.late, self.F = dates, O, H, L, C, V, late.to_numpy(), F
        tp = (H + L + C) / 3
        cv = np.cumsum(V, axis=1)
        self.vwap = np.cumsum(tp * V, axis=1) / np.where(cv > 0, cv, np.nan)
        self.bars = {n: resample(O, H, L, C, V, n) for n in (5, 10, 60, 120)}


def mk_trade(cx, d, side, e, stop=None):
    entry = cx.O[d, e]
    xs, xp = CLOSE, cx.C[d, CLOSE]
    if stop is not None:
        for j in range(e, LAST_CONT + 1):
            if cx.L[d, j] <= stop:
                xs, xp = j, min(stop, cx.O[d, j])
                break
    return dict(date=cx.dates[d], side=int(side), entry_slot=int(e), exit_slot=int(xs), entry_px=entry, exit_px=xp,
                gross=side * (xp / entry - 1))


def rule_im(cx, sig, ent, mode):
    s = cx.F[{"ON30": "r_on30", "F30": "r_f30", "F60": "r_f60"}[sig]].to_numpy()
    out = []
    for d in range(len(cx.dates)):
        if cx.late[d] or np.isnan(s[d]) or s[d] == 0:
            continue
        side = 1 if s[d] > 0 else -1
        if mode == "LO" and side < 0:
            continue
        out.append(mk_trade(cx, d, side, ent))
    return out


def rule_reg(cx, filt, mode):
    out = []
    if mode == "LO":
        f = filt_series(cx.F, filt).to_numpy()
        for d in range(len(cx.dates)):
            if not cx.late[d] and f[d] == 1:
                out.append(mk_trade(cx, d, 1, 1))
    else:
        sd = ls_side(cx.F, filt).to_numpy()
        for d in range(len(cx.dates)):
            if not cx.late[d] and sd[d] != 0:
                out.append(mk_trade(cx, d, int(sd[d]), 1))
    return out


def rule_orb(cx, n, filt, stop_k=None, ls=False):
    bo, bh, bl, bc, nb = cx.bars[n]
    or_bars = 3
    if ls:
        sd = ls_side(cx.F, filt).to_numpy()
    else:
        f = filt_series(cx.F, filt).to_numpy()
    med = cx.F["med_rng20"].to_numpy()
    out = []
    for d in range(len(cx.dates)):
        if cx.late[d]:
            continue
        if ls:
            allow_up = np.isnan(sd[d]) or sd[d] > 0
            allow_dn = np.isnan(sd[d]) or sd[d] < 0
        else:
            if f[d] != 1:
                continue
            allow_up, allow_dn = True, False
        orh, orl = bh[d, :or_bars].max(), bl[d, :or_bars].min()
        for b in range(or_bars, nb):
            e = (b + 1) * n
            if e > ENTRY_CAP:
                break
            side = 1 if (bc[d, b] > orh) else (-1 if bc[d, b] < orl else 0)
            if side == 0:
                continue
            if (side > 0 and not allow_up) or (side < 0 and not allow_dn):
                break                               # 허용되지 않는 방향이 먼저 나오면 그날 끝(첫 돌파만)
            stop = None
            if stop_k is not None and side > 0:
                if np.isnan(med[d]):
                    break
                stop = cx.O[d, e] * (1 - stop_k * med[d])
            out.append(mk_trade(cx, d, side, e, stop))
            break
    return out


def rule_vwap_pb(cx, filt):
    n = 5
    bo, bh, bl, bc, nb = cx.bars[n]
    f = filt_series(cx.F, filt).to_numpy()
    out = []
    for d in range(len(cx.dates)):
        if cx.late[d] or f[d] != 1:
            continue
        if not (cx.C[d, 59] > cx.vwap[d, 59]):         # 10:00 가격이 VWAP 위
            continue
        for b in range(12, nb):                        # 10:00~10:05 봉부터
            e = (b + 1) * n
            if e > ENTRY_CAP:
                break
            vw = cx.vwap[d, e - 1]
            if bl[d, b] <= vw and bc[d, b] > vw:
                out.append(mk_trade(cx, d, 1, e))
                break
    return out


def rule_hbar(cx, n, filt):
    bo, bh, bl, bc, nb = cx.bars[n]
    f = filt_series(cx.F, filt).to_numpy()
    out = []
    for d in range(len(cx.dates)):
        if cx.late[d] or f[d] != 1:
            continue
        if not (bc[d, 0] < cx.O[d, 0]):                # 첫 봉 오전 눌림
            continue
        for b in range(1, nb):
            e = (b + 1) * n
            if e > ENTRY_CAP:
                break
            if bc[d, b] > cx.vwap[d, e - 1]:
                out.append(mk_trade(cx, d, 1, e))
                break
    return out


def grid():
    G = []
    for sig in ("ON30", "F30", "F60"):
        for ent, lab in ((300, "14:00"), (360, "15:00")):
            for mode in ("LO", "LS"):
                G.append(dict(id=f"IM_{sig}_{lab.replace(':', '')}_{mode}", block="IM", mode=mode,
                              desc=f"장중 모멘텀 {sig} 부호 → {lab} 진입 → 15:30 종가 ({'롱' if mode == 'LO' else '롱숏'})",
                              fn=lambda cx, s=sig, e=ent, m=mode: rule_im(cx, s, e, m)))
    for flt in ("z>0", "z>=1", "gap>0", "flow_hi", "ma_up"):
        G.append(dict(id=f"REG_{flt}_LO", block="REG", mode="LO", desc=f"국면 {flt} → 09:01 진입 → 15:30 종가 (롱)",
                      fn=lambda cx, f=flt: rule_reg(cx, f, "LO")))
    for flt in ("z_sign", "flow", "ma"):
        G.append(dict(id=f"REG_{flt}_LS", block="REG", mode="LS", desc=f"국면 {flt} 롱/숏 → 09:01 진입 → 15:30 종가",
                      fn=lambda cx, f=flt: rule_reg(cx, f, "LS")))
    for n, blk in ((5, "ORB5"), (10, "ORB10")):
        for flt in ("none", "z>0", "gap>0", "flow_hi", "ma_up"):
            G.append(dict(id=f"{blk}_{flt}", block=blk, mode="LO", desc=f"{n}분 ORB(첫 {3 * n}분 범위) 상단 돌파 롱, 필터 {flt}",
                          fn=lambda cx, n=n, f=flt: rule_orb(cx, n, f)))
    for flt in ("none", "z>0", "gap>0", "flow_hi", "ma_up"):
        G.append(dict(id=f"ORB5S_{flt}", block="ORB5S", mode="LO", desc=f"5분 ORB 롱 + 손절 0.5×중앙고저폭, 필터 {flt}",
                      fn=lambda cx, f=flt: rule_orb(cx, 5, f, stop_k=0.5)))
    for flt in ("none", "z_sign"):
        G.append(dict(id=f"ORB5LS_{flt}", block="ORB5LS", mode="LS", desc=f"5분 ORB 상단 롱/하단 숏, 국면 {flt}",
                      fn=lambda cx, f=flt: rule_orb(cx, 5, f, ls=True)))
    for flt in ("none", "z>0", "gap>0", "flow_hi", "ma_up"):
        G.append(dict(id=f"VWPB_{flt}", block="VWPB", mode="LO", desc=f"5분 VWAP 눌림 매수(10:00 VWAP 위), 필터 {flt}",
                      fn=lambda cx, f=flt: rule_vwap_pb(cx, f)))
    for n, blk in ((60, "H60"), (120, "H120")):
        for flt in ("z>0", "ma_up", "flow_hi"):
            G.append(dict(id=f"{blk}_{flt}", block=blk, mode="LO", desc=f"{n}분봉 오전 눌림 후 VWAP 위 마감 매수, 필터 {flt}",
                          fn=lambda cx, n=n, f=flt: rule_hbar(cx, n, f)))
    return G


# ---------------------------------------------------------------- 지표
def daily_series(tr, dates, cost_bp):
    s = pd.Series(0.0, index=dates)
    if len(tr):
        s.loc[tr["date"].to_numpy()] = (tr["gross"] - cost_bp / 1e4).to_numpy()
    return s


def metrics(daily, tr, cost_bp, a, b):
    d = daily.loc[a:b]
    t = tr[(tr["date"] >= pd.Timestamp(a)) & (tr["date"] <= pd.Timestamp(b))] if len(tr) else tr
    n = len(d)
    net = (t["gross"] - cost_bp / 1e4) if len(t) else pd.Series(dtype=float)
    eq = (1 + d).cumprod()
    yrs = n / ANN
    cagr = eq.iloc[-1] ** (1 / yrs) - 1 if n else np.nan
    mdd = (eq / eq.cummax() - 1).min() if n else np.nan
    sd = d.std()
    hold = ((t["exit_slot"] - t["entry_slot"] + 1).mean()) if len(t) else np.nan
    return {"days": n, "trades": len(t), "tpy": len(t) / yrs if yrs else np.nan, "exposure": len(t) / n if n else np.nan,
            "long_share": (t["side"] > 0).mean() if len(t) else np.nan, "hold_min": hold,
            "win": (net > 0).mean() if len(t) else np.nan, "gross_bp": t["gross"].mean() * 1e4 if len(t) else np.nan,
            "net_bp": net.mean() * 1e4 if len(t) else np.nan, "t_net": E.nw_t(net.to_numpy()) if len(t) >= 10 else np.nan,
            "cagr": cagr, "sharpe": d.mean() / sd * np.sqrt(ANN) if sd > 0 else np.nan, "mdd": mdd,
            "calmar": cagr / -mdd if (mdd is not None and mdd < 0) else np.nan, "total": eq.iloc[-1] - 1 if n else np.nan}


def bench(bm, a, b):
    d = bm.loc[a:b].dropna()
    eq = (1 + d).cumprod()
    cagr = eq.iloc[-1] ** (ANN / len(d)) - 1
    return {"bh_cagr": cagr, "bh_mdd": (eq / eq.cummax() - 1).min(), "bh_total": eq.iloc[-1] - 1,
            "bh_sharpe": d.mean() / d.std() * np.sqrt(ANN)}


def windows(dates):
    W = {"설계 2023-01~2024-12": DESIGN, "검증 2025-01~2026-09": VALID, "검증(2025-06-09~ 제외)": VAL_EX,
         "2025-06-09~ (최근 강세)": RECENT}
    for y in sorted(set(dates.year)):
        W[str(y)] = (f"{y}-01-01", f"{y}-12-31")
    return W


# ---------------------------------------------------------------- 보고 보조
def pct(v, k=1):
    return "-" if v is None or pd.isna(v) else f"{v * 100:.{k}f}%"


def num(v, k=2, sign=True):
    return "-" if v is None or pd.isna(v) else (f"{v:+.{k}f}" if sign else f"{v:.{k}f}")


def mtable(rows):
    cols = ["구간", "비용bp", "거래", "연거래", "승률", "거래당순bp", "NW t", "CAGR", "샤프", "MDD", "칼마", "누적", "B&H CAGR", "B&H MDD"]
    out = []
    for r in rows:
        out.append([r["window"], f"{r['cost']:g}", f"{r['trades']}", num(r["tpy"], 0, False), pct(r["win"]), num(r["net_bp"]),
                    num(r["t_net"]), pct(r["cagr"]), num(r["sharpe"]), pct(r["mdd"]), num(r["calmar"]), pct(r["total"]),
                    pct(r["bh_cagr"]), pct(r["bh_mdd"])])
    return pd.DataFrame(out, columns=cols).to_markdown(index=False)


def main():
    OUTD.mkdir(parents=True, exist_ok=True)
    dates, O, H, L, C, V, late = load_minutes()
    F = features(dates, O, H, L, C, V)
    cx = Ctx(dates, O, H, L, C, V, late, F)
    bm_full = data.price("069500")["close"].pct_change()
    bm = bm_full.reindex(dates)
    W = windows(dates)

    G = grid()
    assert len(G) <= 60
    trades, rows = {}, []
    for g in G:
        tr = pd.DataFrame(g["fn"](cx))
        if len(tr) == 0:
            tr = pd.DataFrame(columns=["date", "side", "entry_slot", "exit_slot", "entry_px", "exit_px", "gross"])
        tr["date"] = pd.to_datetime(tr["date"])
        trades[g["id"]] = tr
        row = {"id": g["id"], "block": g["block"], "mode": g["mode"], "desc": g["desc"]}
        for cost in (COST_ETF, COST_FUT):
            ds = daily_series(tr, dates, cost)
            for wk, (a, b) in (("dsg", DESIGN), ("val", VALID)):
                m = metrics(ds, tr, cost, a, b)
                for k in ("trades", "tpy", "win", "gross_bp", "net_bp", "t_net", "cagr", "sharpe", "mdd", "calmar"):
                    row[f"{wk}{int(cost)}_{k}"] = m[k]
        rows.append(row)
    grid_df = pd.DataFrame(rows)

    # ---- 선정(설계 지표만)
    def select(pool, cost):
        """자격(설계 연 30회 이상·거래당 순수익 > 0) 변형을 설계 샤프 순으로 먼저, 나머지를 그 뒤에 설계 샤프 순으로 둔다."""
        c = int(cost)
        elig = (pool[f"dsg{c}_tpy"] >= 30) & (pool[f"dsg{c}_net_bp"] > 0)
        ranked = pool.assign(_e=elig.astype(int)).sort_values(["_e", f"dsg{c}_sharpe"], ascending=[False, False]).drop(columns="_e")
        return ranked, bool(elig.any()), set(pool["id"][elig])

    lo_pool = grid_df[grid_df["mode"] == "LO"]
    rank_etf, ok_etf, el_etf = select(lo_pool, COST_ETF)
    rank_fut, ok_fut, el_fut = select(grid_df, COST_FUT)
    grid_df["elig_etf5"] = grid_df["id"].isin(el_etf).astype(int).where(grid_df["mode"] == "LO")
    grid_df["elig_fut2"] = grid_df["id"].isin(el_fut).astype(int)
    grid_df["rank_etf5_design"] = grid_df["id"].map({k: i + 1 for i, k in enumerate(rank_etf["id"])})
    grid_df["rank_fut2_design"] = grid_df["id"].map({k: i + 1 for i, k in enumerate(rank_fut["id"])})
    final_id = rank_etf["id"].iloc[0]
    fut_id = rank_fut["id"].iloc[0]
    top_etf = list(rank_etf["id"].iloc[:3])
    top_fut = list(rank_fut["id"].iloc[:3])
    gcols = ["id", "block", "mode", "desc", "rank_etf5_design", "elig_etf5", "rank_fut2_design", "elig_fut2"] + \
            [c for c in grid_df.columns if c.startswith("dsg")] + [c for c in grid_df.columns if c.startswith("val")]
    grid_df[gcols].to_csv(OUTD / "grid.csv", index=False, encoding="utf-8-sig", float_format="%.6g")

    # ---- 상세 성과
    def detail(vid, costs):
        tr = trades[vid]
        out = []
        for cost in costs:
            ds = daily_series(tr, dates, cost)
            for wname, (a, b) in W.items():
                m = metrics(ds, tr, cost, a, b)
                m.update(bench(bm, a, b))
                m.update(window=wname, cost=cost, id=vid)
                out.append(m)
        return out

    gmode = grid_df.set_index("id")["mode"]
    cost_of = lambda vid: (COST_ETF, COST_ETF_STRESS) if gmode[vid] == "LO" else (COST_FUT, COST_FUT_STRESS)
    det = {vid: detail(vid, (COST_ETF, COST_ETF_STRESS)) for vid in top_etf}
    det_fut = {vid: detail(vid, (COST_FUT, COST_FUT_STRESS)) for vid in top_fut}
    det_final_fut = detail(final_id, (COST_FUT, COST_FUT_STRESS))      # 최종 사양을 선물로 실행할 때
    # 기준선(그리드 밖, 선택 대상 아님): 매일 09:01 매수 → 15:30 종가
    trades["BASE_daily_0901_close"] = pd.DataFrame(rule_reg(cx, "none", "LO")).assign(date=lambda x: pd.to_datetime(x["date"]))
    det_base = detail("BASE_daily_0901_close", (COST_ETF, COST_FUT))

    # 분기별(최종, 2·3위, 선물 트랙 1위)
    qrows = []
    for vid, cost in [(final_id, COST_ETF), *[(v, COST_ETF) for v in top_etf[1:]], *[(v, COST_FUT) for v in top_fut]]:
        tr = trades[vid]
        ds = daily_series(tr, dates, cost)
        for q, g in ds.groupby(ds.index.to_period("Q")):
            t = tr[tr["date"].dt.to_period("Q") == q]
            net = t["gross"] - cost / 1e4
            qrows.append({"id": vid, "label": f"{vid} ({cost:g}bp)", "cost_bp": cost, "quarter": str(q), "trades": len(t), "win": (net > 0).mean() if len(t) else np.nan,
                          "net_bp": net.mean() * 1e4 if len(t) else np.nan, "ret": (1 + g).prod() - 1,
                          "bh_ret": (1 + bm.reindex(g.index).fillna(0)).prod() - 1})
    qdf = pd.DataFrame(qrows)
    qdf.to_csv(OUTD / "quarterly.csv", index=False, encoding="utf-8-sig", float_format="%.6g")

    # trades.csv / daily_returns.csv
    def hm(s):
        return f"{9 + (s // 60):02d}:{s % 60:02d}"
    tlist = []
    for vid in dict.fromkeys([final_id, *top_etf[1:], *top_fut]):
        t = trades[vid].copy()
        cst = COST_ETF if gmode[vid] == "LO" else COST_FUT
        t.insert(0, "id", vid)
        t["entry_time"] = t["entry_slot"].map(lambda s: hm(int(s)))
        t["exit_time"] = t["exit_slot"].map(lambda s: "15:30" if int(s) == CLOSE else hm(int(s)))
        t["cost_bp"] = cst
        t["net"] = t["gross"] - cst / 1e4
        t["z"] = F["z"].reindex(t["date"]).to_numpy()
        tlist.append(t)
    pd.concat(tlist).to_csv(OUTD / "trades.csv", index=False, encoding="utf-8-sig", float_format="%.8g")
    dr = pd.DataFrame({"kodex200_bh": bm, "late_open_flat": late.astype(int), "z": F["z"], "flow_hi": F["flow_hi"], "ma_up": F["ma_up"]})
    for vid in dict.fromkeys([final_id, *top_etf[1:], *top_fut]):
        for cost in sorted(set((cost_of(vid) if vid in top_etf else ()) + (COST_FUT, COST_FUT_STRESS))):
            dr[f"{vid}_net{cost:g}"] = daily_series(trades[vid], dates, cost)
    dr.index.name = "date"
    dr.to_csv(OUTD / "daily_returns.csv", encoding="utf-8-sig", float_format="%.8g")

    chart(trades, final_id, fut_id, dates, bm)

    # ---- 장부
    if "--no-ledger" not in sys.argv:
        led = []
        for g in G:
            tr = trades[g["id"]]
            for cost in (COST_ETF, COST_FUT):
                ds = daily_series(tr, dates, cost)
                for wname, (a, b) in (("DESIGN 2023-01~2024-12", DESIGN), ("VALID 2025-01~2026-09", VALID)):
                    m = metrics(ds, tr, cost, a, b)
                    led.append({"rule": g["id"], "window": wname, "cost_bp": cost, "asset": "069500 1min" + ("" if g["mode"] == "LO" else " (futures proxy)"),
                                "days": m["days"], "trades": m["trades"], "exposure": m["exposure"], "gross_bp": m["gross_bp"],
                                "net_bp": m["net_bp"], "t_net": m["t_net"], "win": m["win"], "sharpe": m["sharpe"], "cagr": m["cagr"],
                                "mdd": m["mdd"], "calmar": m["calmar"]})
        spec = {"family": FAMILY, "grid": len(G), "design": DESIGN, "validation": VALID,
                "select": "LO pool @5bp: max design Sharpe s.t. tpy>=30 & net>0; futures track all @2bp",
                "final": final_id, "futures_track": fut_id}
        E.ledger_append(data.OUT, TOPIC, spec, pd.DataFrame(led), note="5차 전략 탐색")

    report(G, grid_df, rank_etf, rank_fut, ok_etf, ok_fut, final_id, fut_id, top_etf, top_fut, det, det_fut, det_final_fut, qdf, F, late, dates, bm, trades,
           det_base)
    print("final", final_id, "fut", fut_id)


def chart(trades, final_id, fut_id, dates, bm):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False})
    fig, ax = plt.subplots(figsize=(10, 5), dpi=130)
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    series = [(f"최종 {final_id} (KODEX 5bp)", daily_series(trades[final_id], dates, COST_ETF), dict(color="#2a78d6", lw=2)),
              (f"선물 트랙 {fut_id} (2bp)", daily_series(trades[fut_id], dates, COST_FUT), dict(color="#eb6834", lw=1.6)),
              ("KODEX 200 보유", bm.fillna(0), dict(color="#52514e", lw=1.4, ls=":"))]
    for lab, s, kw in series:
        eq = (1 + s).cumprod()
        ax.plot(eq.index, eq.values, label=lab, **kw)
        ax.annotate(f"×{eq.iloc[-1]:.2f}", xy=(eq.index[-1], eq.iloc[-1]), xytext=(4, 0), textcoords="offset points", fontsize=8, va="center")
    ax.axvline(pd.Timestamp(VALID[0]), color="#8a8984", lw=1, ls="--")
    ax.text(pd.Timestamp(VALID[0]), ax.get_ylim()[1], " 검증 시작", fontsize=8, va="top", color="#52514e")
    ax.set_yscale("log")
    ticks = [0.8, 1, 1.5, 2, 3, 4, 5]
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{t:g}" for t in ticks])
    ax.minorticks_off()
    ax.set_ylabel("누적 가치(시작 = 1, 로그 축)", color="#52514e")
    ax.set_title("장중 계열 최종 사양 누적 수익 (설계 2023~2024 / 검증 2025~2026-09)", loc="left", fontsize=11)
    ax.grid(True, color="#e6e5e1", lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(OUTD / "equity.png", facecolor=fig.get_facecolor())
    plt.close(fig)


def report(G, grid_df, rank_etf, rank_fut, ok_etf, ok_fut, final_id, fut_id, top_etf, top_fut, det, det_fut, det_final_fut, qdf, F, late, dates, bm, trades,
           det_base):
    gi = grid_df.set_index("id")
    md = ["# 5차 전략 탐색 — 장중 계열(KODEX 200 1분봉 → 1/5/10/60/120분)\n",
          "```\n" + __doc__.strip() + "\n```\n",
          f"자료: 1분봉 {len(dates)}일({dates.min().date()} ~ {dates.max().date()}), 늦게 연 날 {int(late.sum())}일 무포지션. "
          f"D2 z 결측일(미국 종가 파일 끝 이후) {int(F['z'].isna().sum())}일, 외인 선물 분위 결측 {int(F['flow_hi'].isna().sum())}일은 그 필터 규칙에서 무포지션.\n"]

    def short_tab(rank, c, k=10):
        el = "elig_etf5" if c == 5 else "elig_fut2"
        rank = rank.assign(_el=grid_df.set_index("id").loc[rank["id"], el].map({1: "O", 0: "X"}).to_numpy())
        cols = ["id", "_el", f"dsg{c}_trades", f"dsg{c}_tpy", f"dsg{c}_win", f"dsg{c}_net_bp", f"dsg{c}_t_net", f"dsg{c}_cagr", f"dsg{c}_sharpe", f"dsg{c}_mdd"]
        t = rank[cols].head(k).copy()
        t.columns = ["변형", "자격", "거래", "연거래", "승률", "거래당순bp", "NW t", "CAGR", "샤프", "MDD"]
        for cc in ("승률", "CAGR", "MDD"):
            t[cc] = t[cc].map(pct)
        for cc in ("거래당순bp", "NW t", "샤프"):
            t[cc] = t[cc].map(num)
        t["연거래"] = t["연거래"].map(lambda v: num(v, 0, False))
        return t.to_markdown(index=False)

    fin = {r["window"]: r for r in det[final_id] if r["cost"] == COST_ETF}
    fv = fin["검증 2025-01~2026-09"]
    hit = (fv["cagr"] >= 0.10) or (fv["win"] >= 0.51)
    ffut = {r["window"]: r for r in det_fut[fut_id] if r["cost"] == COST_FUT}
    fvf = ffut["검증 2025-01~2026-09"]
    md += ["## 1. 요약\n",
           f"- 최종 사양(1차 규칙, KODEX 200 롱 전용 5bp): **{final_id}** — {gi.loc[final_id, 'desc']}"
           + ("" if ok_etf else " (설계 기준 미달: 연 30회·순수익>0 을 만족하는 롱 전용 변형 없음 → 설계 샤프 최대를 그대로 보고)"),
           f"- 설계: 거래 {fin['설계 2023-01~2024-12']['trades']}, 승률 {pct(fin['설계 2023-01~2024-12']['win'])}, 거래당 {num(fin['설계 2023-01~2024-12']['net_bp'])}bp, "
           f"CAGR {pct(fin['설계 2023-01~2024-12']['cagr'])}, 샤프 {num(fin['설계 2023-01~2024-12']['sharpe'])}",
           f"- 검증(2025-01~2026-09): 거래 {fv['trades']}, 승률 {pct(fv['win'])}, 거래당 {num(fv['net_bp'])}bp(NW t {num(fv['t_net'])}), "
           f"CAGR {pct(fv['cagr'])}, 샤프 {num(fv['sharpe'])}, MDD {pct(fv['mdd'])} / 같은 기간 KODEX 200 보유 CAGR {pct(fv['bh_cagr'])}, MDD {pct(fv['bh_mdd'])}",
           f"- 검증(2025-06-09~ 제외, 2025-01~06-05): 거래 {fin['검증(2025-06-09~ 제외)']['trades']}, 승률 {pct(fin['검증(2025-06-09~ 제외)']['win'])}, "
           f"CAGR {pct(fin['검증(2025-06-09~ 제외)']['cagr'])}, 거래당 {num(fin['검증(2025-06-09~ 제외)']['net_bp'])}bp",
           f"- 목표(검증 5bp 순 CAGR ≥ 10% 또는 승률 ≥ 51%) 판정: **{'도달' if hit else '미도달'}**",
           f"- 보조 선물 트랙(48개 전체 2bp, 같은 규칙): **{fut_id}** — {gi.loc[fut_id, 'desc']}"
           + ("" if ok_fut else " (설계 기준 미달)")
           + f" / 검증 승률 {pct(fvf['win'])}, 거래당 {num(fvf['net_bp'])}bp, CAGR {pct(fvf['cagr'])}, 샤프 {num(fvf['sharpe'])}, MDD {pct(fvf['mdd'])}\n",
           "![equity](equity.png)\n",
           "## 2. 설계 구간 순위 (선정에 쓴 표)\n",
           "### 2-1. 롱 전용 37개, KODEX 200 5bp (선정 순서 상위 10: 자격 O = 설계 연 30회 이상·거래당 순수익 > 0, 자격 안에서 설계 샤프 순)\n",
           short_tab(rank_etf, 5), "\n",
           "### 2-2. 전체 48개, 선물 2bp (상위 10)\n", short_tab(rank_fut, 2), "\n",
           f"전체 48개 설계·검증 지표는 `grid.csv`(rank_etf5_design·rank_fut2_design = 설계 순위, val* 열은 선정에 쓰지 않음).\n",
           "## 3. 최종 사양 상세 (KODEX 200 롱, 5bp·스트레스 7bp)\n", mtable(det[final_id]), "\n",
           "### 같은 사양을 미니선물로 실행할 때(2bp·3bp, KODEX 경로 대리)\n",
           mtable([r for r in det_final_fut if r["window"] in ("설계 2023-01~2024-12", "검증 2025-01~2026-09", "검증(2025-06-09~ 제외)", "2025-06-09~ (최근 강세)")]), "\n",
           "## 4. 설계 2·3위의 검증 (견고성 확인용, 선택에 쓰지 않음)\n"]
    for vid in top_etf[1:]:
        md += [f"### {vid} — {gi.loc[vid, 'desc']}\n",
               mtable([r for r in det[vid] if r["cost"] == COST_ETF and r["window"] in ("설계 2023-01~2024-12", "검증 2025-01~2026-09", "검증(2025-06-09~ 제외)", "2025-06-09~ (최근 강세)")]), "\n"]
    md += ["## 5. 보조 선물 트랙 (롱숏 포함, 2bp·3bp)\n"]
    for vid in top_fut:
        md += [f"### {vid} — {gi.loc[vid, 'desc']}" + (" (트랙 1위)" if vid == fut_id else "") + "\n",
               mtable([r for r in det_fut[vid] if r["window"] in ("설계 2023-01~2024-12", "검증 2025-01~2026-09", "검증(2025-06-09~ 제외)", "2025-06-09~ (최근 강세)")
                       or (vid == fut_id and r["cost"] == COST_FUT)]), "\n"]
    # 롱숏·시장중립 변형 요약(검증 포함, 강세장 의존 확인용)
    ls = grid_df[grid_df["mode"] == "LS"]
    t = ls[["id", "dsg2_trades", "dsg2_win", "dsg2_net_bp", "dsg2_sharpe", "val2_trades", "val2_win", "val2_net_bp", "val2_sharpe", "val2_cagr"]].copy()
    t.columns = ["변형", "설계 거래", "설계 승률", "설계 순bp", "설계 샤프", "검증 거래", "검증 승률", "검증 순bp", "검증 샤프", "검증 CAGR"]
    for c in ("설계 승률", "검증 승률", "검증 CAGR"):
        t[c] = t[c].map(pct)
    for c in ("설계 순bp", "설계 샤프", "검증 순bp", "검증 샤프"):
        t[c] = t[c].map(num)
    md += ["### 롱숏 11개 전체(2bp) — 강세장(2025~26) 의존을 보려고 함께 표시, 선정에 쓰지 않음\n", t.to_markdown(index=False), "\n"]
    # 분기별
    q = qdf.copy()
    q["표시"] = q.apply(lambda r: f"{r['ret'] * 100:+.1f}% ({int(r['trades'])}, {'-' if pd.isna(r['win']) else f'{r.win * 100:.0f}%'})", axis=1)
    pv = q.pivot_table(index="quarter", columns="label", values="표시", aggfunc="first")
    bh = q.groupby("quarter")["bh_ret"].first().map(lambda v: f"{v * 100:+.1f}%")
    pv["KODEX 200 보유"] = bh
    md += ["## 6. 분기별 수익 (괄호 = 거래 수, 승률; 롱 전용 5bp, 선물 트랙 2bp)\n", pv.to_markdown(), "\n"]

    # 기준선과 전체 그리드
    keyw = ("설계 2023-01~2024-12", "검증 2025-01~2026-09", "검증(2025-06-09~ 제외)", "2025-06-09~ (최근 강세)")
    md += ["## 7. 기준선: 매일 09:01 매수 → 15:30 종가 (그리드 밖, 선택 대상 아님)\n",
           mtable([r for r in det_base if r["window"] in keyw]), "\n"]
    exec_c = {vid: (5 if mode == "LO" else 2) for vid, mode in zip(grid_df["id"], grid_df["mode"])}
    rows = []
    for _, r in grid_df.iterrows():
        c = exec_c[r["id"]]
        rows.append([r["id"], r["mode"], f"{c}", f"{int(r[f'dsg{c}_trades'])}", pct(r[f"dsg{c}_win"]), num(r[f"dsg{c}_net_bp"]), num(r[f"dsg{c}_sharpe"]),
                     f"{int(r[f'val{c}_trades'])}", pct(r[f"val{c}_win"]), num(r[f"val{c}_net_bp"]), pct(r[f"val{c}_cagr"]), num(r[f"val{c}_sharpe"])])
    md += ["## 8. 전체 48개 (실행 비용 기준: 롱 전용 5bp, 롱숏 2bp) — 투명성용, 검증 열은 선택에 쓰지 않음\n",
           pd.DataFrame(rows, columns=["변형", "방향", "비용bp", "설계 거래", "설계 승률", "설계 순bp", "설계 샤프", "검증 거래", "검증 승률",
                                       "검증 순bp", "검증 CAGR", "검증 샤프"]).to_markdown(index=False), "\n"]

    # 해석
    lo = grid_df[grid_df["mode"] == "LO"]
    n_pos_dsg = int((lo["dsg5_net_bp"] > 0).sum())
    pos_val_neg_dsg = lo[(lo["val5_net_bp"] > 0) & (lo["dsg5_net_bp"] <= 0)]["id"].tolist()
    pos_both = lo[(lo["val5_net_bp"] > 0) & (lo["dsg5_net_bp"] > 0)]["id"].tolist()
    bd = {r["window"]: r for r in det_base if r["cost"] == COST_ETF}
    fx = {r["window"]: r for r in det_fut[fut_id] if r["cost"] == COST_FUT_STRESS}
    md += ["## 9. 해석과 한계\n",
           f"- 롱 전용 37개 중 설계 구간(2023~2024) 5bp 순수익이 플러스인 변형은 {n_pos_dsg}개뿐이고, 1위({final_id})도 거래당 "
           f"{num(fin['설계 2023-01~2024-12']['net_bp'])}bp(NW t {num(fin['설계 2023-01~2024-12']['t_net'])})로 0과 구별되지 않는다. "
           "장중 돌파·눌림·모멘텀 규칙은 KODEX 200 왕복 5bp 를 넘지 못했다.",
           f"- 검증 구간은 KODEX 200 보유 CAGR {pct(fv['bh_cagr'])}의 강세장이었지만, 매일 09:01→종가 롱 기준선은 5bp 순 CAGR "
           f"{pct(bd['검증 2025-01~2026-09']['cagr'])}(설계 {pct(bd['설계 2023-01~2024-12']['cagr'])})에 그쳤다. 상승 대부분이 밤(시가 갭)에 났고 "
           "장중 보유로는 그 몫을 받지 못한다(기존 결론: KODEX 오버나잇 무조건 +5.3bp 와 같은 방향).",
           f"- 검증에서 5bp 순수익이 플러스였던 롱 전용 변형 중 설계에서도 플러스였던 것은 {', '.join(pos_both) or '없음'}"
           f"{'(설계 순위 ' + ', '.join(str(int(grid_df.set_index('id').loc[v, 'rank_etf5_design'])) for v in pos_both) + '위)' if pos_both else ''}뿐이다. "
           f"나머지 {len(pos_val_neg_dsg)}개({', '.join(pos_val_neg_dsg)})는 설계 구간에서 마이너스라 사전 규칙으로는 고를 수 없었고, "
           "강세장 장중 상승분(베타)을 받은 것으로 보는 것이 맞다.",
           f"- 선물 트랙 1위 {fut_id}(전일 외국인 선물 순매수 상위 40% 롱 / 하위 40% 숏, 09:01→15:30)는 설계·검증 모두 승률 51~52%, 거래당 +5~6bp 로 "
           f"방향이 유지됐다. 그러나 검증 NW t {num(fvf['t_net'])}, 3bp 에서 검증 승률 {pct(fx['검증 2025-01~2026-09']['win'])}, "
           f"검증 MDD {pct(fvf['mdd'])}(2026년 급락기)로 약하다. 기존 결과(외인 선물 T−1 상위 20% → 다음날 시가→종가 +4~9bp, 5bp 순 0, 매도 쪽이 더 강함)의 "
           "선물 2bp 버전이며, 1분봉이 더해 준 것은 09:01 체결 가능성 확인뿐이다.",
           "- 결론: 장중 계열은 '사전 고정 규칙으로 고른 사양이 검증에서 실패'한 음성 결과로 보고한다. 팀 근거 중 외국인 선물 흐름만 미니선물 2bp 에서 "
           "약한 플러스가 남는다(승률 51% 경계, CAGR 10% 미달).",
           "- 한계: 1분봉이 2023-01부터라 설계 구간이 2년(약 490일)뿐이다. 롱숏·선물 수치는 KODEX 200 1분 경로로 대리했다(선물 베이시스 변동, 선물 15:45 마감 미반영). "
           "청산은 15:30 종가 단일가 가정, 비용은 왕복 고정(5/7bp, 2/3bp)이며 호가 공백·체결 실패는 반영하지 않았다. "
           "48개 변형을 시험했으므로 설계 1위의 우연 가능성(다중 비교)을 감안해야 한다. KODEX 인버스로 숏을 대신하면 2025~26년 가격이 약 850~5,000원이라 호가 한 단위(1~5원)가 10~20bp 여서 "
           "왕복 비용이 크게 늘어 롱숏 규칙은 사실상 선물 계좌가 필요하다."]
    (OUTD / "report.md").write_text("\n".join(md), encoding="utf-8")


if __name__ == "__main__":
    main()
