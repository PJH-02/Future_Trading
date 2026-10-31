# -*- coding: utf-8 -*-
"""16차 단독 알파 두 전략을 톡방 규칙 그대로 2026-10-30 까지 백테스트한다.

전략 1 코스닥 5분봉 장중 전략
  229200 m = 09:05:00 첫 체결가 / 09:00 단일가 − 1, m ≥ +0.3% 롱, m ≤ −0.3% 숏. 09:06:00 첫 체결 진입, 15:00:00 첫 체결 청산.
  롱 233740 전액(비용 ETF 왕복 5bp + 왕복 1틱), 숏 코스닥150 선물 명목 1배(선물 1분봉 실측 1.8bp, 그 전 229200 반전 대리 2.86bp).
전략 2 코스닥 과매도 구간 스윙트레이딩
  코스닥150 지수 IBS < 0.1 인 날 T 15:45 선물 종가 매수, 뒤 IBS > 0.7 또는 10거래일째 15:45 청산, 청산일 진입 신호면 유지. 왕복 1.8bp, 스트레스 3bp, 만기 교체 왕복 1회 추가.

분봉 라벨: TOSS·KIS 산출 파일은 봉 끝(09:01 시가 = 09:00 단일가, 09:06 시가 = 09:05:00, 09:07 시가 = 09:06:00, 15:01 시가 = 15:00:00). 선물 1분봉 time 은 봉 시작.
거래량 0 봉은 다음 거래 봉으로 민다. 신호 봉이 밀리면 진입은 신호 봉 다음 분부터 찾는다(같은 체결로 신호·진입을 겸하지 않는다).
전략 2 는 r16s_swing 의 hold·sim 을 그대로 쓰고 입력 파일과 끝 날짜만 옮긴다.
검산: 기존 공유 시계열 QC_Df(2025-10-21 까지), X_MRq(2026-09-29 까지)와 일수익 assert.
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

sys.stdout.reconfigure(encoding="utf-8")
pd.set_option("display.width", 260, "display.max_columns", 40, "display.max_rows", 300, "display.max_colwidth", 70)
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]
import r16s_swing as SW  # noqa: E402  hold·sim 재사용, import 때 자료를 읽지 않는다
import r16s_swing_data as D  # noqa: E402

JH = Path("<repo>/jehyeong/intraday/data")
RAW = Path("<raw>")
R = RAW / "r16_refresh"
LAB = Path("<OUT>")
R16 = LAB / "round16_alpha_search"
OUT = R16 / "priority_two/main"
OUT.mkdir(parents=True, exist_ok=True)
NM1, NM2 = "코스닥 5분봉 장중 전략", "코스닥 과매도 구간 스윙트레이딩"
SIG, LNG = "229200", "233740"
ETF, FUT, FUT_S, KQ_ADJ = 5e-4, 1.8e-4, 3e-4, 1.06e-4
START, END = pd.Timestamp("2022-11-23"), pd.Timestamp("2026-10-30")
LATE = ["2023-01-02", "2023-11-16", "2024-01-02", "2024-11-14", "2025-01-02", "2025-11-13", "2026-01-02"]
PERIODS = [("2023", "2023-01-01", "2023-12-31"), ("2024", "2024-01-01", "2024-12-31"), ("2025", "2025-01-01", "2025-12-31"), ("2026 연초~끝", "2026-01-01", str(END.date())),
           ("2025~2026 합", "2025-01-01", str(END.date())), ("2023~2024 합", "2023-01-01", "2024-12-31"), ("전체", str(START.date()), str(END.date()))]


def dot(x):
    return pd.Timestamp(x).strftime("%Y.%m.%d")


def hac_t(x):
    x = np.asarray(x, float)
    if len(x) < 5 or x.std() == 0:
        return np.nan
    return float(sm.OLS(x, np.ones(len(x))).fit(cov_type="HAC", cov_kwds={"maxlags": 5}).tvalues[0])


def curve(r):
    """기간 수익률, CAGR, MDD, 최악일, 최악일 수익."""
    e = (1 + r).cumprod()
    return e.iloc[-1] - 1, e.iloc[-1] ** (252 / len(r)) - 1, (e / e.cummax() - 1).min(), r.idxmin(), r.min()


# ---------------------------------------------------------------- 분봉·선물 1분봉 이어 붙이기(겹치면 기존 우선)
def minute(code):
    t = pd.read_csv(JH / f"minute/{code}_TOSS_1m.csv", encoding="utf-8-sig", usecols=["timestamp", "date", "open", "close", "volume"])
    k = pd.read_csv(R / f"{code}_1m_kis.csv", usecols=["timestamp", "date", "open", "close", "volume"])
    m = pd.concat([t, k[k.date > t.date.max()]], ignore_index=True)
    assert not m.duplicated("timestamp").any() and m.date.max() == str(END.date())
    m["hm"] = m.timestamp.str[11:16]
    return m.pivot(index="date", columns="hm", values="open"), m.pivot(index="date", columns="hm", values="volume"), m.groupby("date").close.last(), t.date.max()


_M = {c: minute(c) for c in (SIG, LNG)}
Op, Vo = {c: v[0] for c, v in _M.items()}, {c: v[1] for c, v in _M.items()}
COLS = list(Op[SIG].columns)
Fm = pd.concat([pd.read_csv(RAW / "intraday/kis_minute/kq150_full_day/futures_1min.csv"), pd.read_csv(R / "kq150_fut_1min_ext.csv")], ignore_index=True)
assert not Fm.duplicated(["date", "time"]).any() and Fm.date.max() == str(END.date())
FB = {d: g.sort_values("time").reset_index(drop=True) for d, g in Fm.groupby("date")}
F0 = min(FB)

# 그해 233740 평균 원가격: 깃 원가격 일봉 + 뒤는 이어 붙인 분봉 종가
_rw = pd.read_csv(JH / f"{LNG}_daily.csv", encoding="utf-8-sig", usecols=["date", "close"]).drop_duplicates("date").set_index("date").close
_rw = pd.concat([_rw, _M[LNG][2][_M[LNG][2].index > _rw.index.max()]])
_rw.index = pd.to_datetime(_rw.index)
TICK = 5 / _rw.groupby(_rw.index.year).mean()


def fill(c, d, lab):
    """라벨 lab 이상 첫 거래 봉의 (라벨, 시가)."""
    v = Vo[c].loc[d, lab:]
    k = v[v > 0].index[0]
    return k, float(Op[c].at[d, k])


def lab2time(lab):
    return (pd.Timestamp("2000-01-01 " + lab) - pd.Timedelta(minutes=1)).strftime("%H:%M")


def fut_fill(d, hm):
    """time hm 이상 첫 선물 봉의 (time, 시가, 계약)."""
    x = FB[d].loc[FB[d].time >= hm].iloc[0]
    return x.time, float(x.open), x.contract


# ---------------------------------------------------------------- 전략 1
first = (Vo[SIG] > 0).idxmax(axis=1)
assert list(first.index[first != "09:01"]) == LATE, "늦은 개장일 변경"
days = [d for d in Op[SIG].index if d not in LATE]
assert set(Op[SIG].index) == set(Op[LNG].index)
rows, pushes = [], []
mv = {}
for d in days:
    p0 = float(Op[SIG].at[d, "09:01"])
    ls, ps = fill(SIG, d, "09:06")
    m = ps / p0 - 1
    mv[d] = m
    side = 1 if m >= .003 - 1e-12 else -1 if m <= -.003 + 1e-12 else 0
    if ls != "09:06":
        pushes.append(dict(date=d, 단계="신호", 원래="09:06", 실제=ls, 거래=side != 0))
    if side == 0:
        continue
    le = "09:07" if ls == "09:06" else COLS[COLS.index(ls) + 1]
    src, ct = "", ""
    if side > 0:
        ins, c = LNG, LNG
    elif d in FB:
        ins, c, src = "선물", None, "실측"
    else:
        ins, c, src = SIG, SIG, "대리"
    if c:
        a, e = fill(c, d, le)
        b, x = fill(c, d, "15:01")
        tgt_a, tgt_b = le, "15:01"
    else:
        tgt_a, tgt_b = lab2time(le), "15:00"
        a, e, ct = fut_fill(d, tgt_a)
        b, x, _ = fut_fill(d, tgt_b)
    if a != tgt_a or le != "09:07":
        pushes.append(dict(date=d, 단계="진입", 원래="09:07" if c else "09:06", 실제=a, 거래=True))
    if b != tgt_b:
        pushes.append(dict(date=d, 단계="청산", 원래=tgt_b, 실제=b, 거래=True))
    g = side * (x / e - 1)
    cost = ETF + TICK[int(d[:4])] if side > 0 else (FUT if src == "실측" else FUT + KQ_ADJ)
    rows.append(dict(date=pd.Timestamp(d), side="롱" if side > 0 else "숏", m_pct=m * 100, instrument=ct or ins, short_src=src, sig_bar=ls, entry_bar=a, exit_bar=b, entry_px=e, exit_px=x,
                     gross=g, cost=cost, net=g - cost))
T1 = pd.DataFrame(rows)
PUSH = pd.DataFrame(pushes)
CAL = pd.DatetimeIndex(Op[SIG].index)
s1 = T1.set_index("date").net.reindex(CAL).fillna(0.0)
print("전략 1 거래", len(T1), T1.side.value_counts().to_dict(), T1.short_src.value_counts().to_dict(), "문턱 경계 일수", sum(abs(abs(v) - .003) < 1e-9 for v in mv.values()))
print("민 봉\n", PUSH.to_string())
_v0 = {c: {lb: [d for d in Vo[c].index[Vo[c][lb] == 0] if d not in LATE] for lb in ("09:06", "09:07", "15:01")} for c in Vo}
print("핵심 봉 거래량 0(늦은 개장일 제외)", _v0)

# ---------------------------------------------------------------- 전략 2
D.EXT, D.CUT = R, END  # 입력 파일과 끝 날짜만 옮긴다
ch = D.fut_chain("kq150")
assert ch.index[-1] == END
_git = D.kq150_index()
_ref = pd.read_csv(R / "kq150_index_daily.csv", parse_dates=["date"]).set_index("date")
assert (_ref.close.reindex(_git.index) - _git.close).abs().max() == 0, "깃 지수와 갱신 파일 불일치"
IX = pd.concat([_git, _ref.loc[_ref.index > _git.index.max(), ["open", "high", "low", "close"]]])
assert IX.index.is_unique and IX.index.is_monotonic_increasing
fx = IX.reindex(ch.index)
ibs = ((fx.close - fx.low) / (fx.high - fx.low)).where(fx.high > fx.low)
ent, ex = (ibs < .1).fillna(False).to_numpy(), (ibs > .7).to_numpy()
w_all = SW.hold(ent, ex, 10)
w_all[ch.index < "2016-01-04"] = 0.0  # 전략 2 는 2016 부터 굴리고 2022-11-23 부터 표시한다(그날 보유 중인 거래가 있어 표시 시작에서 끊으면 거래 목록이 달라진다)
assert ch.index[ch.index >= START].equals(CAL), "달력 불일치"
V = SW.Veh(ch.index, ch.on.to_numpy(), ch.oc.to_numpy(), ch.roll.to_numpy(bool), np.full(len(ch), FUT / 2), np.full(len(ch), FUT_S / 2), "Fkq150")
res = SW.sim(w_all, V, "c")
s2, s2s = pd.Series(res["r"], index=ch.index).loc[START:], pd.Series(res["rs"], index=ch.index).loc[START:]
ti, hd = res["ti"], res["hold"]
xi = ti + hd
_fp = pd.read_csv(R / "kis_fut_kq150_daily.csv", dtype={"code": str}, parse_dates=["date"]).set_index(["date", "code"]).close
_rc = np.cumsum(V.roll)
T2 = pd.DataFrame({"entry": ch.index[ti], "exit": ch.index[xi], "hold_days": hd, "ibs_entry": ibs.to_numpy()[ti], "ibs_exit": ibs.to_numpy()[xi],
                   "code_in": ch.code.to_numpy()[ti], "code_out": ch.code.to_numpy()[xi], "rolls": _rc[xi] - _rc[ti], "gross": res["gt"], "net": res["nt"], "stress": res["st"]})
T2["entry_px"] = _fp.reindex(pd.MultiIndex.from_arrays([T2.entry, T2.code_in])).to_numpy()
T2["exit_px"] = _fp.reindex(pd.MultiIndex.from_arrays([T2.exit, T2.code_out])).to_numpy()
T2["reason"] = np.where(w_all[xi] != 0, "미청산", np.where(T2.ibs_exit > .7, "IBS>0.7", "10거래일"))
T2 = T2[T2.exit >= START].reset_index(drop=True)
print("전략 2 거래(표시 구간에 손익 있는 것)", len(T2), T2.reason.value_counts().to_dict(), "유지 거래(보유 > 10일)", int((T2.hold_days > 10).sum()), "교체 수", int(T2.rolls.sum()))
print("표시 시작 전 진입 거래\n", T2[T2.entry < START].to_string())
expo = pd.Series(res["expo"], index=ch.index).loc[START:]

# ---------------------------------------------------------------- 검산
qc = pd.read_csv(R16 / "share/series/QC_Df.csv", parse_dates=["date"]).set_index("date")
mr = pd.read_csv(R16 / "share/series/X_MRq.csv", parse_dates=["date"]).set_index("date")
_d1 = (qc.net.reindex(CAL).loc[:"2025-10-21"] - s1.loc[:"2025-10-21"]).abs().max()
_d2 = (mr.net.reindex(CAL).loc[:"2026-09-29"] - s2.loc[:"2026-09-29"]).abs().max()
print("기존 시계열 대비 일수익 최대 차: QC_Df ~2025-10-21", _d1, "X_MRq ~2026-09-29", _d2)
assert _d1 < 6e-9 and _d2 < 6e-9  # 저장 소수 8자리


# ---------------------------------------------------------------- 표
def row1(name, a, b, t, r):
    x = t[(t.date >= a) & (t.date <= b)]
    w = r.loc[a:b]
    ret, cg, mdd, wd, wv = curve(w)
    return {"구간": name, "시작": str(w.index[0].date()), "끝": str(w.index[-1].date()), "일수": len(w), "거래수": len(x), "롱건수": int((x.side == "롱").sum()), "숏건수": int((x.side == "숏").sum()),
            "숏실측": int((x.short_src == "실측").sum()), "숏대리": int((x.short_src == "대리").sum()), "승률_pct": (x.net > 0).mean() * 100, "총bp": x.gross.mean() * 1e4, "순bp": x.net.mean() * 1e4,
            "NW_t": hac_t(x.net * 1e4), "기간수익률_pct": ret * 100, "CAGR_pct": cg * 100, "MDD_pct": mdd * 100, "최악일": str(wd.date()), "최악일수익_pct": wv * 100}


def tab1(t):
    r = t.set_index("date").net.reindex(CAL).fillna(0.0)
    return pd.DataFrame([row1(n, a, b, t, r) for n, a, b in PERIODS])


def row2(name, a, b):
    x = T2[(T2.entry >= a) & (T2.entry <= b)]
    w = s2.loc[a:b]
    ret, cg, mdd, wd, wv = curve(w)
    return {"구간": name, "시작": str(w.index[0].date()), "끝": str(w.index[-1].date()), "일수": len(w), "거래수": len(x), "평균보유일": x.hold_days.mean(), "승률_pct": (x.net > 0).mean() * 100, "순bp": x.net.mean() * 1e4,
            "NW_t": hac_t(x.net * 1e4), "기간수익률_pct": ret * 100, "CAGR_pct": cg * 100, "MDD_pct": mdd * 100, "최악일": str(wd.date()), "최악일수익_pct": wv * 100,
            "노출일비율_pct": (expo.loc[a:b] > 0).mean() * 100, "스트레스CAGR_pct": curve(s2s.loc[a:b])[1] * 100}


TB1, TB1L, TB1S = tab1(T1), tab1(T1[T1.side == "롱"]), tab1(T1[T1.side == "숏"])
TB2 = pd.DataFrame([row2(*p) for p in PERIODS])
_m = pd.DataFrame({"s1": s1, "s2": s2}).loc["2025-01-01":]
TB3 = ((1 + _m).groupby(_m.index.to_period("M")).prod() - 1) * 100
TB3.index = TB3.index.astype(str)
TB3 = TB3.rename_axis("월").rename(columns={"s1": NM1, "s2": NM2}).reset_index()

for nm, df in (("table1", TB1), ("table1_long", TB1L), ("table1_short", TB1S), ("table2", TB2)):
    df.round(4).to_csv(OUT / f"{nm}.csv", index=False, encoding="utf-8-sig")
    print(f"\n[{nm}]\n", df.round(3).to_string(index=False))
TB3.round(4).to_csv(OUT / "table3_monthly.csv", index=False, encoding="utf-8-sig")
print("\n[table3]\n", TB3.round(2).to_string(index=False))
_t1 = T1.assign(gross_bp=T1.gross * 1e4, cost_bp=T1.cost * 1e4, net_bp=T1.net * 1e4).drop(columns=["gross", "cost", "net"])
_t1.round(6).to_csv(OUT / "trades_s1.csv", index=False, encoding="utf-8-sig")
_t2 = T2.assign(gross_bp=T2.gross * 1e4, net_bp=T2.net * 1e4, stress_bp=T2.stress * 1e4, in_table=T2.entry >= START).drop(columns=["gross", "net", "stress"])
_t2.round(6).to_csv(OUT / "trades_s2.csv", index=False, encoding="utf-8-sig")
pd.DataFrame({"s1_net": s1, "s2_net": s2}).rename_axis("date").round(10).to_csv(OUT / "daily.csv", encoding="utf-8-sig")


# ---------------------------------------------------------------- 정지 추정일, 실측 대 대리, 민감도
def gaps(c, d, mn=10):
    v = Vo[c].loc[d, "09:01":"15:20"]
    z = v.eq(0)
    g = (z != z.shift()).cumsum()
    return [f"{x.index[0]}~{x.index[-1]}" for _, x in v[z].groupby(g[z]) if len(x) >= mn]


def fgaps(d, mn=10):
    if d not in FB:
        return ["자료 없음"]
    t = FB[d].time[(FB[d].time >= "09:00") & (FB[d].time <= "15:30")]
    df = pd.to_datetime(t, format="%H:%M").diff().dt.total_seconds().div(60).to_numpy()
    return [f"{t.iloc[i - 1]}~{t.iloc[i]}" for i in np.flatnonzero(df > mn)]


_ti = T1.set_index("date")
HALT = []
for d in days:
    a_, b_ = gaps(SIG, d), gaps(LNG, d)
    if a_ or b_:
        ts = pd.Timestamp(d)
        tr = _ti.loc[ts] if ts in _ti.index else None
        HALT.append(dict(date=d, g229=" ".join(a_), g233=" ".join(b_), gf=" ".join(fgaps(d)) or "없음", side=tr.side if tr is not None else "없음", m=mv[d] * 100, net=tr.net * 1e4 if tr is not None else np.nan,
                         bars=f"{tr.sig_bar}/{tr.entry_bar}/{tr.exit_bar}" if tr is not None else ""))
HALT = pd.DataFrame(HALT)
print("\n[정지 추정일]\n", HALT.round(2).to_string(index=False))
_fv = (Vo[LNG] > 0).idxmax(axis=1)
VI2 = [d for d in days if _fv[d] != "09:01"]
print("233740 만 첫 체결이 늦은 날", VI2)

_sr = T1[T1.short_src == "실측"].set_index("date")
_px = pd.Series({d: -(fill(SIG, str(d.date()), _b)[1] / fill(SIG, str(d.date()), _a)[1] - 1) for d, _a, _b in zip(_sr.index, ["09:07"] * len(_sr), ["15:01"] * len(_sr))})
_dg = (_sr.gross - _px) * 1e4
RP = dict(n=len(_sr), a=str(_sr.index[0].date()), b=str(_sr.index[-1].date()), real=_sr.gross.mean() * 1e4, proxy=_px.mean() * 1e4, diff=_dg.mean(), t=hac_t(_dg), corr=float(np.corrcoef(_sr.gross, _px)[0, 1]),
          net_real=_sr.net.mean() * 1e4, net_proxy=(_px - FUT - KQ_ADJ).mean() * 1e4, mx=_dg.abs().max(), mxd=str(_dg.abs().idxmax().date()))
print("실측 대 대리", RP)


def mdd1(r):
    e = np.r_[1.0, (1 + r).cumprod().to_numpy()]
    return (e / np.maximum.accumulate(e) - 1).min()


_series = {"table1": (s1, TB1), "table1_long": (T1[T1.side == "롱"].set_index("date").net.reindex(CAL).fillna(0.0), TB1L), "table1_short": (T1[T1.side == "숏"].set_index("date").net.reindex(CAL).fillna(0.0), TB1S), "table2": (s2, TB2)}
MDIF = []
for k, (r, tb) in _series.items():
    for _, o in tb.iterrows():
        m2 = mdd1(r.loc[o["시작"]:o["끝"]]) * 100
        if abs(m2 - o["MDD_pct"]) > 5e-4:
            MDIF.append(f"{k} {o['구간']} {o['MDD_pct']:.2f}% → {m2:.2f}%")
print("시작값 1 포함 MDD 차", MDIF)

_flat = s2.copy()
_flat.iloc[0] = 0.0
FLAT = curve(_flat)
D68 = "2026-06-08"
t68 = _ti.loc[pd.Timestamp(D68)]
assert t68.side == "롱"
_e68 = float(Op[LNG].at[D68, t68.sig_bar])
n68 = (t68.exit_px / _e68 - 1 - t68.cost) * 1e4
ALT = {}
for tag, v in (("같은 봉 체결", n68 / 1e4), ("거래 제외", 0.0)):
    x = s1.copy()
    x.loc[pd.Timestamp(D68)] = v
    ALT[tag] = (curve(x.loc["2025-01-01":]), curve(x))
print("2026-06-08 대안", n68, ALT)

# ---------------------------------------------------------------- 기존 16차 평가 대조
LR = pd.read_csv(R16 / "eval/leg_results.csv", encoding="utf-8-sig")
CMP = []
for leg in ("QC_Df", "X_MRq"):
    for w in ("D", "S", "W", "X", "Y2023", "Y2024", "Y2025", "Y2026"):
        o = LR[(LR.leg == leg) & (LR.window == w) & (LR.cost == "기본")].iloc[0]
        a, b = o.start, o.end
        x, r = (T1[(T1.date >= a) & (T1.date <= b)], s1.loc[a:b]) if leg == "QC_Df" else (T2[(T2.entry >= a) & (T2.entry <= b)], s2.loc[a:b])
        assert len(r) == o.days, (leg, w, len(r), o.days)
        _c = curve(r)
        CMP.append({"다리": leg, "창": w, "시작": dot(a), "끝": dot(b), "n": f"{o.n}/{len(x)}", "순bp": f"{o.net_bp:.2f}/{x.net.mean() * 1e4:.2f}", "NW_t": f"{o.t_nw:.2f}/{hac_t(x.net * 1e4):.2f}",
                    "CAGR": f"{o.cagr_pct:.2f}/{_c[1] * 100:.2f}", "MDD": f"{o.mdd_pct:.2f}/{_c[2] * 100:.2f}"})
CMP = pd.DataFrame(CMP)
print("\n[기존 평가/새 계산]\n", CMP.to_string(index=False))

C_END = pd.Timestamp("2026-09-29")  # 기존 공유 시계열 끝
o1 = qc.loc[START:C_END]
o1 = o1[o1.pos != 0]
n1 = _ti.loc[:C_END]
only_old, only_new = o1.index.difference(n1.index), n1.index.difference(o1.index)
both = o1.index.intersection(n1.index)
dir_diff = both[np.sign(o1.pos[both]).to_numpy() != np.where(n1.side[both] == "롱", 1, -1)]
_dn = (n1.net[both] - o1.net[both]) * 1e4
_g = {"2025.10.21 까지": _dn[both <= "2025-10-21"], "그 뒤 롱(2026 틱 평균 끝날 차이)": _dn[(both > "2025-10-21") & (n1.side[both] == "롱")], "그 뒤 숏(실측 1.8bp 대 대리 2.86bp)": _dn[(both > "2025-10-21") & (n1.side[both] == "숏")]}
G1 = [f"{k}: n {len(v)}, 차 평균 {v.mean():+.2f}bp, 절대 최대 {v.abs().max():.2f}bp" for k, v in _g.items()]
print("전략 1 목록", len(o1), len(n1), list(only_old), list(only_new), list(dir_diff), G1)
_mp = (mr.pos.fillna(0.0) != 0).to_numpy()
_st, _en = mr.index[_mp & ~np.r_[False, _mp[:-1]]], mr.index[~_mp & np.r_[False, _mp[:-1]]]
o2 = pd.DataFrame({"entry": _st, "exit": _en[:len(_st)]})
o2 = o2[(o2.entry >= START) & (o2.entry <= C_END)].reset_index(drop=True)
n2 = T2[(T2.entry >= START) & (T2.entry <= C_END)].reset_index(drop=True)
L2 = o2.equals(n2[["entry", "exit"]])
print("전략 2 목록 일치", L2, len(o2), len(n2))
TAIL1, TAIL2 = T1[T1.date > C_END], T2[T2.entry > C_END]


# ---------------------------------------------------------------- notes.md
def md(df):
    c = list(df.columns)
    return "\n".join(["| " + " | ".join(c) + " |", "|" + "---|" * len(c)] + ["| " + " | ".join(str(v) for v in r) + " |" for r in df.values.tolist()])


_h = HALT.assign(date=HALT.date.map(dot), 순bp=HALT.net.map(lambda v: "" if pd.isna(v) else f"{v:.1f}"), m=HALT.m.map(lambda v: f"{v:.2f}%"))[["date", "g229", "g233", "gf", "m", "side", "bars", "순bp"]]
_h.columns = ["날짜", "229200 공백(라벨)", "233740 공백(라벨)", "선물 공백(봉 시작)", "m", "방향", "신호/진입/청산 봉", "순bp"]
_tb = {k: float(TB1[TB1["구간"] == k].iloc[0]["CAGR_pct"]) for k in ("2025~2026 합", "전체")}


def src_row(name, x):
    return {"구간": name, "건수": len(x), "승률%": f"{(x.net > 0).mean() * 100:.1f}", "총bp": f"{x.gross.mean() * 1e4:.2f}", "순bp": f"{x.net.mean() * 1e4:.2f}", "NW_t": f"{hac_t(x.net * 1e4):.2f}"}


_sh = T1[T1.side == "숏"]
SRC = pd.DataFrame([src_row("대리 2022.11.23~2024.12.30", _sh[(_sh.short_src == "대리") & (_sh.date < "2025-01-01")]), src_row("대리 2025.01.02~2025.10.21", _sh[(_sh.short_src == "대리") & (_sh.date >= "2025-01-01")]),
                    src_row("실측 2025.10.22~2025.12.30", _sh[(_sh.short_src == "실측") & (_sh.date < "2026-01-01")]), src_row(f"실측 2026.01.02~{dot(END)}", _sh[(_sh.short_src == "실측") & (_sh.date >= "2026-01-01")])])
print("\n[숏 출처별]\n", SRC.to_string(index=False))
_sx = s1.loc["2025-01-02":C_END].copy()
_sx.loc[pd.Timestamp(D68)] = 0.0
SX = curve(_sx)
N_P25 = int(((T1.short_src == "대리") & (T1.date >= "2025-01-01")).sum())
_tick = ", ".join(f"{y} {TICK[y] * 1e4:.2f}bp" for y in (2022, 2023, 2024, 2025, 2026))
_o2 = TB2[TB2["구간"] == "전체"].iloc[0]
_c0 = T2[T2.entry < START].iloc[0]
notes = f"""# 16차 우선 두 전략 백테스트 실행 기록

• 코드: r16p_backtest.py, 실행 PYTHONUTF8=1 python r16p_backtest.py. 전략 2 는 r16s_swing 의 hold·sim 을 그대로 쓰고 입력 파일과 끝 날짜만 옮겼다.
• 명칭: 전략 1 {NM1}, 전략 2 {NM2}. 규칙은 톡방 문구 그대로.
• 평가 끝: 두 전략 모두 {dot(END)}. 229200 분봉, 233740 분봉, 선물 1분봉, 선물 일봉, 코스닥150 지수가 모두 같은 날 끝난다. 달력 {len(CAL)}일, 표시 시작 {dot(START)}.
• 이어 붙이기: 229200 은 TOSS {dot(_M[SIG][3])} 뒤, 233740 은 TOSS {dot(_M[LNG][3])} 뒤부터 KIS 산출 파일. 지수는 깃 뒤 {dot(_ref.index.max())} 까지 KIS. {dot(END)} 은 독립 출처 대조가 없다.
• 검산(assert): QC_Df 공유 시계열과 {dot('2025-10-21')} 까지 일수익 최대 차 {_d1:.1e}, X_MRq 공유 시계열과 {dot(C_END)} 까지 {_d2:.1e}. 저장 소수 8자리 안.
• 독립 재계산: r16p_backtest_check.py. 전략 2 상태 기계 일수익 최대 차 5.0e-11, 전략 1 은 {dot(D68)} 외 509건 신호·총수익 전부 일치, 선물 봉 시작 + 1분 = ETF 라벨 분 수익 상관 0.967(다른 시차 0.04 이하).

## 전략 1 적용 세부

• 제외 {len(LATE)}일(229200 첫 거래 봉이 09:01 라벨보다 늦은 날):{', '.join(dot(d) for d in LATE)}.
• 233740 만 첫 체결이 09:03 라벨인 {len(VI2)}일({', '.join(dot(d) for d in VI2)})은 신호·진입 봉이 정상이라 그대로 거래.
• 거래 {len(T1)}건: 롱 {int((T1.side == '롱').sum())}, 숏 {int((T1.side == '숏').sum())}(실측 {len(_sr)}, 대리 {int((T1.short_src == '대리').sum())}). 실측은 {dot(F0)} 이후 모든 숏.
• 거래량 0 봉 밀림: 거래가 걸린 건 {dot(D68)} 하루, 신호 1건·진입 1건·청산 1건. 신호 09:06→{t68.sig_bar} 라벨, 진입 09:07→{t68.entry_bar} 라벨, 청산 15:01→{t68.exit_bar} 라벨. 신호 봉이 밀리면 같은 체결로 신호와 진입을 겸하지 않도록 진입은 다음 분으로 잡았다.
• {dot(D68)} 거래: m {mv[D68] * 100:.2f}%, 롱, {t68.entry_px:.0f}원 → {t68.exit_px:.0f}원, 순 {t68.net * 1e4:.1f}bp. 같은 봉 체결이면 순 {n68:.1f}bp, 2025~2026 합 CAGR {ALT['같은 봉 체결'][0][1] * 100:.2f}%, MDD {ALT['같은 봉 체결'][0][2] * 100:.2f}%. 기존 평가처럼 거래 제외면 2025~2026 합 CAGR {ALT['거래 제외'][0][1] * 100:.2f}%, MDD {ALT['거래 제외'][0][2] * 100:.2f}%, 전체 CAGR {ALT['거래 제외'][1][1] * 100:.2f}%. 본표는 2025~2026 합 {_tb['2025~2026 합']:.2f}%, 전체 {_tb['전체']:.2f}%.
• 2026.06.23: 15:01 라벨 거래량 0, m {mv['2026-06-23'] * 100:.2f}% 로 신호 없음, 영향 없음.
• 틱(5원 ÷ 그해 233740 평균 원가격): {_tick}. 2026 평균은 {dot(END)} 까지.

정지 추정일(장중 거래량 0 이 10분 이상 이어진 날, 늦은 개장일 제외). 진입·청산이 공백에 걸린 날은 {dot(D68)} 뿐이다. 공백 라벨은 거래량 0 봉 범위.

{md(_h)}

## 숏 실측 대 대리

{md(SRC)}

• 같은 날 비교 {dot(RP['a'])}~{dot(RP['b'])}, {RP['n']}건: 총수익 실측 {RP['real']:.2f}bp, 229200 대리 {RP['proxy']:.2f}bp, 차 {RP['diff']:.2f}bp, NW t {RP['t']:.2f}, 상관 {RP['corr']:.4f}, 절대 차 최대 {RP['mx']:.1f}bp({dot(RP['mxd'])}).
• 같은 날 순수익: 실측(1.8bp) {RP['net_real']:.2f}bp, 대리(2.86bp) {RP['net_proxy']:.2f}bp.
• 만기일 실측은 1분봉 파일 계약 그대로 차근월(일봉 front 와 251일 전부 일치).

## 전략 2 적용 세부

• 거래 {int((T2.entry >= START).sum())}건(진입일 기준 배정). 청산 IBS>0.7 {int(((T2.reason == 'IBS>0.7') & (T2.entry >= START)).sum())}건, 10거래일 {int((T2.reason == '10거래일').sum())}건, 만기 교체 {int(T2[T2.entry >= START].rolls.sum())}회, 미청산 {int((T2.reason == '미청산').sum())}건, 청산일 재신호 유지 {int((T2.hold_days > 10).sum())}건.
• 최대 손실 거래: {dot(T2.loc[T2.net.idxmin(), 'entry'])} 진입, {dot(T2.loc[T2.net.idxmin(), 'exit'])} 청산, 순 {T2.net.min() * 1e4:.1f}bp.
• 2016.01.04 부터 굴리고 {dot(START)} 부터 표시. {dot(_c0.entry)} 진입 거래가 표시 첫날 {dot(_c0.exit)} 청산이라 그날 일수익 {s2.iloc[0] * 100:.2f}% 가 곡선에 들어가고 거래 수에는 없다(거래 순 {_c0.net * 1e4:.1f}bp, trades_s2.csv in_table False).
• 첫날을 0 으로 보면 전체 기간 수익률 {_o2['기간수익률_pct']:.2f}% → {FLAT[0] * 100:.2f}%, CAGR {_o2['CAGR_pct']:.2f}% → {FLAT[1] * 100:.2f}%, MDD 동일. 다른 구간 영향 없음.
• 정지 추정일 공백은 장중이고 전략 2 체결은 15:45 종가라 체결 시각에 걸리지 않는다.

## 기존 16차 평가 대조(기본 비용, 칸은 기존/새 계산)

{md(CMP)}

• X_MRq: 같은 창에서 거래 목록과 수치가 같다. {'거래 목록 ' + str(len(o2)) + '건 진입일·청산일 전부 일치' if L2 else '거래 목록 불일치 확인 필요'}.
• QC_Df: {dot('2025-10-21')} 까지 같고 그 뒤 숏 실측 전환과 {dot(D68)} 포함으로 달라진다. 창 S 에서 {dot(D68)} 을 0 으로 두면 CAGR {SX[1] * 100:.2f}%, MDD {SX[2] * 100:.2f}%. 기존 −19.05%·−50.84% 와의 나머지 차이는 실측 전환과 2026 틱 평균 끝날 몫이다.

## 같은 날 거래 목록 비교({dot(START)}~{dot(C_END)})

• 전략 1: 기존 {len(o1)}건, 새 계산 {len(n1)}건, 공통 {len(both)}건, 기존만 {len(only_old)}건, 새 계산만 {len(only_new)}건({', '.join(dot(d) for d in only_new)}), 방향 다른 날 {len(dir_diff)}건.
• 공통일 순수익 차(새 − 기존): {'; '.join(G1)}.
• 전략 2: 기존 {len(o2)}건, 새 계산 {len(n2)}건, {'진입일·청산일 전부 일치' if L2 else '불일치'}.
• 기존 끝 뒤 추가 거래({dot(C_END)} 초과): 전략 1 {len(TAIL1)}건(순 평균 {TAIL1.net.mean() * 1e4:.1f}bp), 전략 2 {len(TAIL2)}건(순 평균 {TAIL2.net.mean() * 1e4:.1f}bp).
• 시작값 1 을 넣은 MDD 와 다른 표 칸: {'; '.join(MDIF) if MDIF else '없음'}. 표는 기존 식 min(누적/누적최고 − 1) 그대로.

## 한계

• 2025.01.02~2025.10.21 숏 {N_P25}건은 선물 1분봉 보관창 밖이라 대리다.
• 전략 1 롱은 233740 전액이라 지수 명목 2배, 숏은 1배다. 롱만 표와 숏만 표의 낙폭은 명목이 다르다.
• 정지 재개 첫 체결가는 봉 시가 그대로 썼다. 호가 공백 슬리피지는 비용에 없다.
• 틱은 그해 평균 원가격 규약이라 비용에 그해 뒤 가격이 들어간다. 신호에는 영향 없다.
"""
notes = re.sub(r"(?<![\w.|-])-(?=\d)", "−", notes)  # 음수 표기만 바꾼다(저장 표 구분선·파일명 유지)
(OUT / "notes.md").write_text(notes, encoding="utf-8")
print(notes)

# ---------------------------------------------------------------- 결과 묶음(StructuredOutput 입력)
BT = []
for strat, side, tb in ((NM1, "전체", TB1), (NM1, "롱", TB1L), (NM1, "숏", TB1S), (NM2, "롱", TB2)):
    for _, o in tb.iterrows():
        BT.append(dict(strategy=strat, side=side, period=o["구간"], n=int(o["거래수"]), net_bp=round(float(o["순bp"]), 2), t_nw=round(float(o["NW_t"]), 2), ret_pct=round(float(o["기간수익률_pct"]), 2), mdd_pct=round(float(o["MDD_pct"]), 2),
                       cagr_pct=round(float(o["CAGR_pct"]), 2), win_pct=round(float(o["승률_pct"]), 2), worst_date=o["최악일"], worst_pct=round(float(o["최악일수익_pct"]), 2)))
print(json.dumps(BT, ensure_ascii=False))
print(md(TB3.round(2)))
