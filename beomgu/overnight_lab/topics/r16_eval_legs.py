# -*- coding: utf-8 -*-
"""16차 봉인 평가 다리 조립: r16s_portfolio_legs.py 의 규칙 코드를 그대로 쓰고 봉인선만 자료 끝으로 옮긴다.

스캔 대비 바꾼 것(규칙 변경 없음)
  • SEAL 2024-12-30 → 2026-09-29(달력 069500 수정가 일봉 끝). r16s_flows.SEAL·r16s_swing_data.CUT·r16s_morning_exec.SEAL 도 import 뒤 같은 값으로 옮긴다.
  • 2022-11 전 첫 1분 평균 차감값은 설계 구간(≤ 2024-12-30) 값으로 고정. 낮 숏 보정 1.06bp 도 고정. 틱은 그해 평균 원가격 규약 그대로(2025·2026 은 그해 값).
  • 다리 추가: M_zfQ(r16s_morning_exec), C1_RB(r16s_rv_arb 규칙, 선물 1분봉 실측 구간만).
  • 끝에서 스캔 다리와 2024-12-30 까지 전 필드 대조(assert), 다리별 평가 끝 표, 거래 단위 흐름 TR 저장.
입력 자료가 평가 끝 전에 끝나면 그 뒤는 NaN 이다(0 으로 두지 않는다).
날짜 규약: 밤 다리 손익 = 청산일. 비활성일 0. 시작일 전은 NaN(유효 전).
비용: ETF 왕복 5bp, 선물 왕복 1.8bp × 명목, 연속 체결이 끼면 왕복 1틱(5원 / 그해 평균 원가격), 낮 숏 선물 대리 보정 1.06bp × 명목(15차 ADR 상수).
스트레스: ETF 7bp + 1틱(단일가 포함), 선물 3bp × 명목.
재현 단언: 13차 book_daily, 형제 스캔 daynight_exec·morning_exec·kospi_mirror·flows·swing 수치.
"""
import json
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent))

SEAL = pd.Timestamp("2026-09-29")  # 봉인선을 달력 끝으로. 더 긴 자료도 여기서 자른다
DESIGN_END = pd.Timestamp("2024-12-30")
GIT = Path("<repo>")
JH = GIT / "jehyeong/intraday/data"
RAW = Path("<raw>")
LAB = Path("<OUT>")
R13 = LAB / "round13_slot_book"
OUT = LAB / "round16_alpha_search/eval"
OUT.mkdir(parents=True, exist_ok=True)
SCRATCH = Path("<local>")
CACHE = SCRATCH / "r16_eval_legs.pkl"
SCAN_CACHE = SCRATCH / "r16s_portfolio_legs.pkl"  # 스캔 코드 재실행 산출(대조 전용)
ETF, ETF_S, FUT, FUT_S, KQ_ADJ = 5e-4, 7e-4, 1.8e-4, 3e-4, 1.06e-4
CAL0 = "2016-01-04"


def seal(df, col="date"):
    df = df.copy()
    df[col] = pd.to_datetime(df[col])
    df = df[df[col] <= SEAL]
    assert len(df) and df[col].max() <= SEAL, "봉인 구간 행 잔존"
    return df


def rd(p, **kw):
    return pd.read_csv(p, encoding="utf-8-sig", **kw)


CAL = pd.DatetimeIndex(seal(rd(GIT / "beomgu/data/etf/069500.csv", usecols=["date"])).date)
CAL = CAL[CAL >= CAL0]
assert CAL.is_monotonic_increasing and CAL.is_unique and CAL[-1] == SEAL
NXT = pd.Series(CAL[1:], index=CAL[:-1])
PREV = pd.Series(CAL[:-1], index=CAL[1:])
L = {}


def add(key, mkt, slot, net, stress, gross, pos, exec_ok, label, note="", start=None, fut=0.0, auc=0.0):
    """일수익 다리 등록. start 전은 NaN. pos = 자본 1 대비 지수 명목(부호 포함), fut = 명목 중 선물 비율, auc = 단일가 체결에 붙일 1틱(자본 대비)."""
    idx = CAL
    net = pd.Series(net).reindex(idx)
    start = pd.Timestamp(start) if start is not None else net.first_valid_index()
    net = net.fillna(0.0).where(idx >= start)
    f = lambda x: pd.Series(x).reindex(idx).fillna(0.0).where(idx >= start)
    stress = f(stress).where((net != 0) | (f(gross) != 0), 0.0)  # 비활성일 스트레스 비용 0
    L[key] = SimpleNamespace(key=key, mkt=mkt, slot=slot, net=net, stress=stress, gross=f(gross), pos=f(pos), exec_ok=exec_ok, label=label, note=note, start=start,
                             fut=pd.Series(float(fut), index=idx) if np.isscalar(fut) else f(fut), auc=pd.Series(0.0, index=idx) if np.isscalar(auc) else f(auc))


# ---------------------------------------------------------------- 13차 series
LAST = {}  # 13차 series 마지막 유효일


def s13(h):
    s = seal(pd.read_csv(R13 / "series" / f"{h}.csv"))
    m = json.loads((R13 / "series" / f"{h}.meta.json").read_text(encoding="utf-8"))
    LAST[h] = pd.Timestamp(m["last_valid"])
    s = s.set_index("date")
    s = s.loc[m["first_valid"]:]
    return s.reindex(CAL[CAL >= pd.Timestamp(m["first_valid"])]).fillna(0.0)


S = {h: s13(h) for h in ("jehyeong_ON", "beomgu_C3VT30", "jehyeong_KP", "blog_ON_Q", "beomgu_C3Q", "jehyeong_KQ")}
book13 = seal(pd.read_csv(R13 / "book/book_daily.csv")).set_index("date")
assert book13.index.isin(CAL).all()


# ---------------------------------------------------------------- 틱·일봉
def daily(code):
    p = GIT / f"beomgu/data/etf/{code}.csv"
    if not p.exists():
        p = JH / f"{code}_daily.csv"
    d = seal(rd(p, usecols=["date", "open", "high", "low", "close"]))
    return d[~d.date.duplicated()].sort_values("date").set_index("date")


PX = {c: daily(c) for c in ("069500", "122630", "229200", "233740", "251340")}
RAWC = {c: seal(rd(JH / f"{c}_daily.csv", usecols=["date", "close"])).drop_duplicates("date").set_index("date").close for c in PX}
TK = {c: 5 / v.groupby(v.index.year).mean() for c, v in RAWC.items()}


def tk(code, idx):
    idx = pd.DatetimeIndex(idx)
    return pd.Series(pd.Series(idx.year).map(TK[code]).values, index=idx)


def mix(x, a1, a2):
    """명목 x ≤ 2 를 1배·2배 ETF 로 나눈 수익·틱(단위 동일)."""
    x = np.asarray(x, float)
    return np.where(x <= 1, x * a1, (x - 1) * a2 + (2 - x) * a1)


def etf_cap(x):
    return np.minimum(np.asarray(x, float), 1.0)


# ---------------------------------------------------------------- 분봉
def read_min(code):
    m = rd(JH / f"minute/{code}_TOSS_1m.csv", usecols=["timestamp", "date", "open", "high", "low", "close", "volume"], parse_dates=["timestamp"])
    m = seal(m)
    m["hm"] = m.timestamp.dt.strftime("%H:%M")
    return {k: m.pivot(index="date", columns="hm", values=k).sort_index() for k in ("open", "high", "low", "close", "volume")}


M = {c: read_min(c) for c in PX}
Op, Cl, Vo = {c: M[c]["open"] for c in M}, {c: M[c]["close"] for c in M}, {c: M[c]["volume"] for c in M}
DAYS = Op["229200"].index
LATE = pd.to_datetime(["2025-01-02", "2025-11-13", "2026-01-02"])  # 2025 이후 늦은 개장일은 파일에 09:01~10:00 봉 자체가 없다(수능일은 16:30 까지). 스캔 구간 4일과 달리 라벨 격자 밖이라 SPECIAL 로 제외됨을 단언
for c in M:
    _odd = (Cl[c].notna().idxmax(axis=1) != "09:01") | (Cl[c].notna().iloc[:, ::-1].idxmax(axis=1) != "15:30")
    assert list(_odd[_odd].index) == list(LATE), f"{c} 첫·끝 라벨"
_v = Vo["229200"]
SPECIAL = DAYS[(Cl["229200"].notna().sum(axis=1) != 390) | (_v.loc[:, "09:01":"10:00"].sum(axis=1) == 0) | ((Op["229200"].loc[:, "15:21":"15:29"] != Cl["229200"].loc[:, "15:21":"15:29"]).sum(axis=1) >= 5)]
assert LATE.isin(SPECIAL).all()

# ---------------------------------------------------------------- 코스피 다리
s = S["jehyeong_ON"]
add("KA_ON", "KOSPI", "A", s.net, s.gross - ETF_S - (s.pos != 0) * tk("069500", s.index), s.gross, s.pos, True, "jehyeong_ON 069500 15:20 신호 → 15:30 단일가 → 09:00", start=s.index[0], auc=(s.pos != 0) * tk("069500", s.index))

s = S["beomgu_C3VT30"]
_t = (s.pos != 0) * mix(s.pos.abs(), tk("069500", s.index), tk("122630", s.index))
add("KB_C3", "KOSPI", "B", s.net, s.net - (s.pos != 0) * 2e-4 * etf_cap(s.pos.abs()) - _t, s.gross, s.pos, False, "beomgu_C3VT30 원형 09:00 단일가 신호·체결", start=CAL0, auc=_t)
s = S["beomgu_C3Q"]
_t = (s.pos != 0) * mix(s.pos.abs(), tk("229200", s.index), tk("233740", s.index))
add("QB_C3", "KOSDAQ", "B", s.net, s.net - (s.pos != 0) * 2e-4 * etf_cap(s.pos.abs()) - _t, s.gross, s.pos, False, "beomgu_C3Q 원형 09:00 단일가 신호·체결", start=CAL0, auc=_t)


def first_minute(c):
    """09:01 라벨 시가(단일가), 09:02 라벨 시가(09:01:00), 10:01 라벨 시가, 15:30 라벨 종가."""
    o = Op[c]
    p0 = o["09:01"].where(Vo[c]["09:01"] > 0)
    p1 = o["09:02"]
    return pd.DataFrame({"p0": p0, "p1": p1, "p10": o["10:01"], "pc": Cl[c]["15:30"]})


FM = {c: first_minute(c) for c in ("069500", "122630", "229200", "233740")}
PROXY = {}


def c3_e1(key, h, c1, c2, stop=None, exit_="close"):
    """C3 09:02 라벨 시가 진입. 분봉 없는 날은 일봉 시가→종가에서 분봉 창 C3 날 평균 첫 1분 변화를 뺀 근사.
    stop = 1배 ETF 기준 손절폭(일봉 저가, 손절 우선). 명목 x 는 원 시리즈 그대로."""
    s_ = S[h]
    idx = s_.index
    on = s_.pos != 0
    x = s_.pos.abs()
    f1, f2 = FM[c1].reindex(idx), FM[c2].reindex(idx)
    win = on & f1.p0.notna() & f2.p0.notna() & (idx <= DESIGN_END)  # 표본 의존 상수는 설계 구간 값으로 고정
    m1 = float((f1.p1 / f1.p0 - 1)[win].mean())
    m2 = float((f2.p1 / f2.p0 - 1)[win].mean())
    PROXY[key] = (m1, m2, int(win.sum()))
    d1, d2 = PX[c1].reindex(idx), PX[c2].reindex(idx)
    oc1, oc2 = d1.close / d1.open - 1, d2.close / d2.open - 1
    has = f1.p1.notna() & f2.p1.notna()
    day_special = idx.isin(SPECIAL) | (f1.p0.isna() & has)
    if exit_ == "close":
        a1 = (f1.pc / f1.p1 - 1).where(has, (1 + oc1) / (1 + m1) - 1)
        a2 = (f2.pc / f2.p1 - 1).where(has, (1 + oc2) / (1 + m2) - 1)
    else:
        a1, a2 = (f1.p10 / f1.p1 - 1).where(has), (f2.p10 / f2.p1 - 1).where(has)
    g = pd.Series(mix(x, a1, a2), index=idx)
    if stop is not None:
        hit = (d1.low / d1.open) <= (f1.p1 / f1.p0).where(has, 1 + m1) * (1 - stop)  # 시가 대비 진입 위치를 반영한 손절선
        g = g.where(~hit, -x * stop)  # shortcut: 1배 ETF 일봉 저가 기준(손절 우선), 2배 몫도 x×손절폭 근사, 서킷·단일가 전환 체결 지연 미반영. 낮 분봉 경로가 필요하면 분봉 창에서 재측정
    t1, t2 = tk(c1, idx), tk(c2, idx)
    tick = pd.Series(mix(x, t1, t2), index=idx)
    act = on & g.notna() & ~day_special
    c_, cs = ETF * etf_cap(x) + tick, ETF_S * etf_cap(x) + tick
    slip = 0.0 if stop is None else 25e-4 * x * hit
    pos = s_.pos.where(act, 0.0)
    return (g - c_).where(act, 0.0), (g - cs - slip).where(act, 0.0), g.where(act, 0.0), pos


k1 = c3_e1("KB_E1", "beomgu_C3VT30", "069500", "122630")
add("KB_E1", "KOSPI", "B", *k1, True, "C3 09:02 봉 시가 → 15:30(분봉 전 근사)", start=CAL0)
q1 = c3_e1("QB_E1", "beomgu_C3Q", "229200", "233740")
add("QB_E1", "KOSDAQ", "B", *q1, True, "C3Q 09:02 봉 시가 → 15:30(분봉 전 근사)", start=CAL0)
k2 = c3_e1("KB_E1x", "beomgu_C3VT30", "069500", "122630", exit_="1001")
add("KB_E1x", "KOSPI", "B", *k2, True, "C3 09:02 봉 시가 → 10:01 봉 시가(분봉 창만)", start=DAYS[0])
q2 = c3_e1("QB_E1x", "beomgu_C3Q", "229200", "233740", exit_="1001")
add("QB_E1x", "KOSDAQ", "B", *q2, True, "C3Q 09:02 봉 시가 → 10:01 봉 시가(분봉 창만)", start=DAYS[0])
for sp in (0.02, 0.03, 0.04, 0.05):
    t = int(sp * 100)
    add(f"KB_E1s{t}", "KOSPI", "B", *c3_e1("KB_E1", "beomgu_C3VT30", "069500", "122630", stop=sp), True, f"C3 09:02 진입 + 1배 ETF −{t}% 손절", start=CAL0)
    add(f"QB_E1s{t}", "KOSDAQ", "B", *c3_e1("QB_E1", "beomgu_C3Q", "229200", "233740", stop=sp), True, f"C3Q 09:02 진입 + 1배 ETF −{t}% 손절", start=CAL0)

s = S["jehyeong_KP"]
tp = pd.Series(mix(s.pos.abs(), tk("069500", s.index), tk("122630", s.index)), index=s.index) * (s.pos != 0)
add("KC_KP0", "KOSPI", "C", s.net, s.gross - 1.4 * (s.gross - s.net), s.gross, s.pos, False, "jehyeong_KP 저장(틱 없음)", start=s.index[0])
add("KC_KP", "KOSPI", "C", s.net - tp, s.gross - 1.4 * (s.gross - s.net) - tp, s.gross, s.pos, True, "jehyeong_KP + 왕복 1틱", start=s.index[0])

# ---------------------------------------------------------------- 코스닥 밤
s = S["blog_ON_Q"]
_veh = pd.Series(np.where(s.pos > 0, tk("233740", s.index), tk("251340", s.index)), index=s.index)
add("QA_ON", "KOSDAQ", "A", s.net, s.gross - ETF_S - (s.pos != 0) * _veh, s.gross, s.pos, False, "blog_ON_Q 원형 15:30 신호·같은 종가", start=s.index[0], auc=(s.pos != 0) * _veh)
add("QA_ONt", "KOSDAQ", "A", s.net - (s.pos < 0) * _veh, s.gross - ETF_S - (s.pos != 0) * _veh, s.gross, s.pos, False, "blog_ON_Q 원형 + 251340 1틱", start=s.index[0], auc=(s.pos > 0) * _veh)

fut = seal(pd.read_csv(RAW / "r15_external/kis_fut_kq150_daily.csv"))
FPX = fut.set_index(["date", "code"]).sort_index()
assert FPX.index.is_unique
FRONT = fut[fut.front == 1].set_index("date").code
assert FRONT.index.is_unique
q = S["blog_ON_Q"]
qon = q.index[q.pos != 0]
NT = pd.DataFrame({"pos": q.pos[qon].values, "T": PREV.reindex(qon).values, "code": FRONT.reindex(qon).values}, index=qon)
NT["front_T"] = FRONT.reindex(NT["T"]).values
_cT = FPX.close.reindex(pd.MultiIndex.from_arrays([NT["T"], NT.front_T])).values
_oD = FPX.open.reindex(pd.MultiIndex.from_arrays([NT.index, NT.front_T])).values
NT["fr"] = _oD / _cT - 1  # 만기 전날 밤은 T 의 front(만기물)


def qa_fut(fc):
    net = pd.Series(np.nan, index=q.index)
    f_ = NT.pos * NT.fr - fc * NT.pos.abs()
    use = NT.fr.notna() & ~((NT.index < "2018-01-01") & (NT.pos > 0))  # 2016~2017 롱 밤은 얇은 선물 대신 233740
    net.loc[NT.index[use]] = f_[use].values
    return net, use


_n, _u = qa_fut(FUT)
_ns, _ = qa_fut(FUT_S)
gr = q.gross.copy()
gr.loc[NT.index[_u]] = (NT.pos * NT.fr)[_u].values
_thin = (q.pos != 0) & ~q.index.isin(NT.index[_u])
add("QA_F", "KOSDAQ", "A", _n.fillna(q.net), _ns.fillna(q.gross - ETF_S - (q.pos != 0) * _veh), gr, q.pos, True, "blog_ON_Q 15:30 신호 → 15:45 선물(2016~2017 롱 밤 233740)", start=q.index[0],
    fut=pd.Series(0.0, index=q.index).where(~q.index.isin(NT.index[_u]), 1.0), auc=_thin * _veh)

# ---------------------------------------------------------------- 코스닥 낮 jehyeong_KQ
s = S["jehyeong_KQ"]
SH = s.pos < 0
tq = (s.pos > 0) * s.pos * tk("229200", s.index)
n_ = (s.net - tq).where(~SH, s.gross - (FUT + KQ_ADJ) * s.pos.abs())
n_s = (s.gross - 1.4 * (s.gross - s.net) - tq).where(~SH, s.gross - (FUT_S + KQ_ADJ) * s.pos.abs())
add("QC_KQ0", "KOSDAQ", "C", s.net, s.gross - 1.4 * (s.gross - s.net), s.gross, s.pos, False, "jehyeong_KQ 저장(251340 틱 없음)", start=s.index[0])
add("QC_KQ", "KOSDAQ", "C", n_, n_s, s.gross, s.pos, True, "jehyeong_KQ 롱 1틱·숏 선물 대리", start=s.index[0], fut=SH.astype(float))
tq251 = SH * s.pos.abs() * tk("251340", s.index)
add("QC_KQt", "KOSDAQ", "C", s.net - tq - tq251, s.gross - 1.4 * (s.gross - s.net) - tq - tq251, s.gross, s.pos, True, "jehyeong_KQ 롱 1틱·숏 251340 1틱", start=s.index[0])

# ---------------------------------------------------------------- daynight 저장 흐름
nav = seal(rd(GIT / "jehyeong/daynight/results/daily_nav.csv"))
nav3 = seal(rd(GIT / "jehyeong/daynight/results_v3_v4/daily_nav.csv"))
for ver, src in (("V2", nav), ("V3", nav3)):
    for acc, slot, key in (("overnight", "A", f"QA_{ver}"), ("intraday", "C", f"QC_{ver}")):
        x = src[(src.version == ver) & (src.account == acc)].set_index("date").daily_return
        if key == "QA_V3":
            continue
        act = x != 0
        tag = "밤 15:20 직전 체결가 가정" if acc == "overnight" else "낮 251340 틱 없음"
        add(key, "KOSDAQ", slot, x, x - 2e-4 * act, x + 5e-4 * act, pd.Series(1.0, index=x.index).where(act, 0.0), False, f"daynight {ver} {acc} 저장({tag})", start=x.index[0])
_c = nav[(nav.version == "V2") & (nav.account == "combined")].set_index("date").daily_return
add("Q_V2C", "KOSDAQ", "A", _c, _c, _c, pd.Series(1.0, index=_c.index).where(_c != 0, 0.0), False, "daynight V2 combined 저장", start=_c.index[0])
_o3 = nav3[(nav3.version == "V3") & (nav3.account == "overnight")].set_index("date").daily_return
print("V3 밤 − V2 밤 최대 차:", float((_o3 - L["QA_V2"].net.reindex(_o3.index)).abs().max()))
BASE = {"ON": nav[(nav.version == "V2") & (nav.account == "overnight")].set_index("date").daily_return,
        "ID": nav[(nav.version == "V2") & (nav.account == "intraday")].set_index("date").daily_return, "KP": book13.KOSPI_book}

# ---------------------------------------------------------------- daynight 실행 보정판
SIG, LNG, SHT = "229200", "233740", "251340"
d229 = rd(JH / f"{SIG}_daily.csv", usecols=["date", "close"])
d229 = seal(d229).drop_duplicates("date").sort_values("date").set_index("date").close
prev, ma10 = d229.shift(1), d229.rolling(10, min_periods=10).mean().shift(1)
UP, DN = (prev > ma10).reindex(DAYS).fillna(False), (prev < ma10).reindex(DAYS).fillna(False)
h20, l20 = M[SIG]["high"].loc[:, "09:01":"15:20"].max(axis=1), M[SIG]["low"].loc[:, "09:01":"15:20"].min(axis=1)
IBS20 = ((Cl[SIG]["15:20"] - l20) / (h20 - l20)).where(h20 > l20)
MV = Op[SIG]["09:06"] / Op[SIG]["09:01"] - 1
NS = pd.Series(np.where(IBS20 < .3, 1, np.where(IBS20 > .7, -1, 0)), index=DAYS).where(~DAYS.isin(SPECIAL), 0)
DS = pd.Series(np.where(MV >= .003 - 1e-12, 1, np.where(MV <= -.003 + 1e-12, -1, 0)), index=DAYS).where(~DAYS.isin(SPECIAL), 0)
NSF = NS.where(((NS == 1) & UP) | ((NS == -1) & DN), 0)
DSF = DS.where(((DS == 1) & UP) | ((DS == -1) & DN), 0)
X09 = {c: Op[c]["09:01"].where(Vo[c]["09:01"] > 0).shift(-1) for c in (LNG, SHT)}


def night_auc(sig, lx=2.0, sx=-1.0):
    """15:20 신호 → 15:30 단일가 → 익일 09:00 단일가. 날짜 = 청산일."""
    out = pd.DataFrame(0.0, index=CAL, columns=["g", "n", "s", "p"])
    for sd, c, x in ((1, LNG, lx), (-1, SHT, sx)):
        ix = sig.index[sig == sd]
        r = (X09[c] / Cl[c]["15:30"] - 1).reindex(ix).dropna()
        d = NXT.reindex(r.index).dropna()
        r = r.reindex(d.index)
        v = r.values
        out.loc[d.values, "g"] += v
        out.loc[d.values, "n"] += v - ETF
        out.loc[d.values, "s"] += v - ETF_S - tk(c, d.index).values
        out.loc[d.values, "p"] += x
    return out


def day_cont(sig, short="etf", long_="etf"):
    """09:05:00 체결가 신호 → 09:07 라벨 시가 진입 → 15:01 라벨 시가 청산."""
    out = pd.DataFrame(0.0, index=CAL, columns=["g", "n", "s", "p"])
    ent, ex_ = "09:07", "15:01"
    for sd in (1, -1):
        ix = sig.index[sig == sd]
        path = long_ if sd == 1 else short
        c = (LNG if sd == 1 else SHT) if path == "etf" else SIG
        E, X = Op[c][ent].where(Vo[c][ent] > 0), Op[c][ex_].where(Vo[c][ex_] > 0)
        r = (X / E - 1).reindex(ix).dropna()
        if path == "etf":
            t_ = tk(c, r.index).values
            out.loc[r.index, ["g", "n", "s", "p"]] = np.c_[r.values, r.values - ETF - t_, r.values - ETF_S - t_, np.full(len(r), 2.0 if sd == 1 else -1.0)]
        else:
            x = 2.0 if sd == 1 else -1.0
            out.loc[r.index, ["g", "n", "s", "p"]] = np.c_[x * r.values, x * r.values - abs(x) * (FUT + KQ_ADJ), x * r.values - abs(x) * (FUT_S + KQ_ADJ), np.full(len(r), x)]
    return out


def addf(key, slot, df, ok, label, fut=0.0, note="", auc=0.0):
    add(key, "KOSDAQ", slot, df.n, df.s, df.g, df.p, ok, label, note=note, start=DAYS[0], fut=fut, auc=auc)


_nb, _nbm = night_auc(NS), night_auc(NSF)
addf("QA_b", "A", _nb, True, "daynight 밤 15:20 신호 → 15:30 단일가, 국면 없음", auc=_nb.n - _nb.s - 2e-4 * (_nb.p != 0))
addf("QA_bm", "A", _nbm, True, "daynight 밤 15:20 신호 → 15:30 단일가, MA10 순방향", auc=_nbm.n - _nbm.s - 2e-4 * (_nbm.p != 0))
addf("QC_Vt", "C", day_cont(DSF), True, "daynight 낮 V2 규칙 ETF 5bp + 1틱")
addf("QC_D", "C", day_cont(DS), True, "낮 국면 필터 없음 ETF 5bp + 1틱(형제 D01)", note="필터 제거는 결과를 본 변형")
addf("QC_Df", "C", day_cont(DS, short="fut"), True, "낮 국면 필터 없음 롱 233740·숏 선물 대리(형제 D03)", fut=(DS.reindex(CAL) == -1).astype(float), note="필터 제거는 결과를 본 변형")

# ---------------------------------------------------------------- 형제 kospi_mirror 강세 눌림 밤 122630
_k, _q = daily("069500"), daily("229200")
rk, rq = rd(JH / "069500_daily.csv", usecols=["date", "close"]), rd(JH / "229200_daily.csv", usecols=["date", "close"])
rk, rq = seal(rk).drop_duplicates("date").set_index("date").close, seal(rq).drop_duplicates("date").set_index("date").close


def bull19(code, adj, raw):
    cx = Cl[code]["15:20"].reindex(adj.index) * (adj / raw.reindex(adj.index))
    return (cx > (19 * adj.shift(1).rolling(19, min_periods=19).mean() + cx) / 20)


h2, l2 = M["069500"]["high"].loc[:, "09:01":"15:20"].max(axis=1), M["069500"]["low"].loc[:, "09:01":"15:20"].min(axis=1)
ik19 = ((Cl["069500"]["15:20"] - l2) / (h2 - l2)).where(h2 > l2)
_b = (bull19("069500", _k.close, rk).reindex(DAYS).fillna(False) & bull19("229200", _q.close, rq).reindex(DAYS).fillna(False) & (ik19 < .3))
P122 = PX["122630"]
_nr = (P122.open.shift(-1) / P122.close - 1)
_e = _nr.reindex(DAYS[_b]).dropna()
_d = NXT.reindex(_e.index).dropna()
_e = _e.reindex(_d.index)
sr = pd.Series(_e.values, index=_d.values)
add("KA_ST", "KOSPI", "A", sr - ETF, sr - ETF_S - tk("122630", sr.index), sr, pd.Series(2.0, index=sr.index), True, "강세 눌림 밤 122630(형제 kospi_mirror)", note="결과를 본 후보", start=DAYS[0], auc=tk("122630", sr.index))

# ---------------------------------------------------------------- 형제 flows
import r16s_flows as fl  # noqa: E402  __main__ 보호, 파일 쓰기 없음

fl.SEAL = SEAL
C = fl.load()
RR = fl.rules(C)
TR = {}  # 거래 단위 흐름(설계 재현·거래 t 용). 낮·밤 다리 날짜 = 청산일, 스윙 날짜 = 진입 결정일
def f_leg(key, s_, H, V, mk, slot, label):
    sim = fl.simulate(np.nan_to_num(s_), H, V, C.n)
    TR[key] = pd.DataFrame({"net": sim["net"], "gross": sim["gross"], "stress": sim["stress"]}, index=C.days[sim["tx"]])
    nn, ss = pd.Series(sim["daily"], index=C.days), pd.Series(sim["daily_s"], index=C.days)
    seg = fl.build_seg(np.nan_to_num(s_), H, C.n)
    p_ = pd.Series(seg[0::2] + seg[1::2], index=C.days)
    add(key, mk, slot, nn, ss, nn + 1.8e-4 * p_.abs(), p_, True, label, note="결과를 본 후보, 선물", start=CAL0, fut=1.0)


_k2 = fl.etf_px("069500", C.days)
_ik = fl.ibs_of(_k2)
_q1 = fl.etf_px("229200", C.days, adj=False)  # daynight 일봉 IBS 는 원가격 파일
_iq = fl.ibs_of(_q1)
_qc = C.kq150.Close
_bull = (_k2.close > _k2.close.rolling(20, min_periods=20).mean()) & (_qc > _qc.rolling(20, min_periods=20).mean()) & (_ik < .3)
print("[재현] 강세 눌림 선물 신호 건수(2010-11-01~):", int(_bull.loc["2010-11-01":].sum()), "기준 341")
f_leg("KA_STf", 2.0 * _bull.values.astype(float), "N0", C.v["F_KP"], "KOSPI", "A", "강세 눌림 밤 코스피200 선물 2배(형제 kospi_mirror, 15:30 신호 → 15:45)")
_s0 = (2.0 * (_iq < .3) - 1.0 * (_iq > .7)).values.astype(float)
f_leg("QA_nf", _s0, "N0", C.v["F_KQ"], "KOSDAQ", "A", "IBS 국면 필터 없음 15:30 신호 → 15:45 선물 +2·−1(daynight 밤 장기 대리)")
for k_ in ("D1", "D2", "D12", "N1", "N2", "N12", "K1", "K2", "K5"):
    r = RR[k_]
    f_leg(f"X_{k_}", r["m"].astype(float) * r["side"] * (1.0 if r["w"] is None else r["w"]), r["H"], r["V"], "KOSPI" if r["V"].name == "F_KP" else "KOSDAQ", "A" if r["H"] == "N0" else "C", f"flows {k_}: {r['desc']}")

# ---------------------------------------------------------------- 형제 swing
import r16s_swing_data as sd  # noqa: E402

sd.CUT = SEAL  # r16s_swing2 가 import 때 자료를 읽으므로 그 전에 옮긴다
import r16s_swing2 as sw  # noqa: E402  __main__ 보호, 파일 쓰기 없음

for nm in ("MRq", "MRqG", "MRp", "FLzVT"):
    w, res, m = sw.sim_rule(nm)
    V = sw.c.V[m]["F"]
    print(nm, "sim keys", sorted(res.keys()))
    r = pd.Series(res["r"], index=V.idx)
    st = pd.Series(res["rs"], index=V.idx) if "rs" in res else r - (FUT_S - FUT) * pd.Series(np.abs(np.diff(np.r_[0, np.nan_to_num(w)])), index=V.idx)
    mk = "KOSPI" if m == "KP" else "KOSDAQ"
    add(f"X_{nm}", mk, "X", r, st, r, pd.Series(np.nan_to_num(w), index=V.idx), True, f"swing {nm}", note="결과를 본 후보, 선물 스윙", start=max(CAL[0], V.idx[0]), fut=1.0)
    TR[f"X_{nm}"] = pd.DataFrame({"net": res["nt"], "gross": res["gt"], "stress": res["st"]}, index=V.idx[res["ti"]])

# ---------------------------------------------------------------- 코스닥 아침 사전 신호 M_zfQ (r16s_morning_exec)
import r16s_morning_exec as me  # noqa: E402  __main__ 보호

me.SEAL = SEAL
_Q = me.market("KOSDAQ")
me.add_futures(_Q, f"{me.EXT}/r15_external/kis_fut_kq150_daily.csv")
_post = Vo["229200"].index[(Vo["229200"][me.AUC] > 0) & (Vo["229200"].index >= me.FUT_0845)]  # 스캔 post 창: 09:01 봉 거래 있는 날, 2023-07-31~(me.minute 는 LATE 날 단언에 걸려 안 쓴다)
_w, _g, _n, _s = (x.reindex(_post).fillna(0.0) for x in me.run_auc(_Q, (_Q["zf"] >= 1).astype(float).where(_Q["zf"].notna())))
add("M_zfQ", "KOSDAQ", "B", _n, _s, _g, _w, True, "08:45 선물 갭 미국 잔차 z_f ≥ 1 → 09:00 단일가 229200·233740 → 15:30", note="설계 상위 5건 61%", start=me.FUT_0845)

# ---------------------------------------------------------------- 코스닥 마감 15분 C1_RB (r16s_rv_arb 규칙, 선물 1분봉 실측)
_f1 = rd(RAW / "intraday/kis_minute/kq150_full_day/futures_1min.csv", usecols=["date", "time", "open", "close"], parse_dates=["date"])
_f1 = _f1[_f1.date <= SEAL]
_ok = _f1.groupby("date").time.last() == "15:45"  # 수능일처럼 마감이 밀린 날은 15:45 봉이 장중이라 제외
_en = _f1[_f1.time >= "15:31"].groupby("date").open.first().where(_ok).dropna()  # 15:31 이후 첫 체결
_ex = _f1[_f1.time == "15:45"].set_index("date").close.reindex(_en.index)
_rs = np.log(PX["229200"].close / PX["229200"].open).reindex(_en.index)
_sd = np.sign(_rs).where((_rs.abs() >= .01) & FRONT.eq(FRONT.shift(1)).reindex(_en.index), 0.0)  # 분기 만기일(근월 교체일) 제외
_g = _sd * (_ex / _en - 1)
add("C1_RB", "KOSDAQ", "X", (_g - FUT * _sd.abs()), (_g - FUT_S * _sd.abs()), _g, _sd, True, "229200 |시가→15:30| ≥ 1% → 15:31 이후 코스닥150 선물 r 방향 → 15:45", note="등급 estimate, 선물 1분봉 있는 구간만", start="2025-10-22", fut=1.0)


# ---------------------------------------------------------------- 재현 단언
def cagr_mdd(r):
    e = (1 + r).cumprod()
    return (e.iloc[-1] ** (252 / len(r)) - 1) * 100, (e / e.cummax() - 1).min() * 100


def near(a, b, tol, what):
    print(f"[재현] {what}: 계산 {a:.3f} / 기준 {b:.3f}")
    assert abs(a - b) <= tol, what


# 13차 W_des 합산
w_ = ("2022-12-13", "2024-12-30")
kb, qb = L["KB_C3"].pos != 0, L["QB_C3"].pos != 0
rk13 = L["KA_ON"].net + L["KB_C3"].net + L["KC_KP0"].net.where(~kb, 0.0)
rq13 = L["QA_ON"].net + L["QB_C3"].net + L["QC_KQ0"].net.where(~qb, 0.0)
d_ = (rk13.loc[w_[0]:w_[1]] - book13.KOSPI_book.loc[w_[0]:w_[1]]).abs().max(), (rq13.loc[w_[0]:w_[1]] - book13.KOSDAQ_book.loc[w_[0]:w_[1]]).abs().max()
print("[재현] 13차 장부 일수익 최대 차", d_)
assert max(d_) < 1e-7
near(cagr_mdd((0.5 * rk13 + 0.5 * rq13).loc[w_[0]:w_[1]])[0], 24.32, .02, "13차 합산 W_des CAGR")
wd = ("2022-11-23", "2024-12-30")


def tr_check(key, n, net, cagr, mdd, w=wd):
    x = L[key].net.loc[w[0]:w[1]]
    a = x[x != 0]
    c_, m_ = cagr_mdd(x)
    print(f"[재현] {key}: n {len(a)}/{n}, 순 {a.mean() * 1e4:.2f}/{net}, CAGR {c_:.2f}/{cagr}, MDD {m_:.2f}/{mdd}")
    assert abs(len(a) - n) <= 6 and abs(a.mean() * 1e4 - net) < 1.2 and abs(c_ - cagr) < 1.5, key


tr_check("QA_b", 334, 11.2, 18.1, -14.6)
tr_check("QC_Vt", 124, 25.4, 14.4, -11.9)
tr_check("QC_D", 234, 31.2, 38.3, -14.0)
tr_check("QC_Df", 234, 37.9, 49.2, -12.6)
tr_check("KA_ST", 54, 46.07, 12.72, -2.32)
tr_check("KB_E1", 48, 24.88, 5.84, -3.47)
tr_check("QB_E1", 51, -20.56, -5.38, -18.50)
tr_check("KB_E1x", 48, 25.32, 6.09, -1.41)
tr_check("QB_E1x", 51, 16.00, 4.01, -2.72)
tr_check("QA_nf", 1517, 6.1, 9.0, -37.2, ("2016-01-04", "2024-12-30"))
_h3 = (L["X_D1"].net + L["X_K2"].net + L["X_N1"].net).loc["2017-01-02":DESIGN_END]
near(cagr_mdd(_h3)[0], 30.78, .05, "flows 핵심3 CAGR")
near(cagr_mdd(_h3)[1], -11.46, .05, "flows 핵심3 MDD")
_mr = L["X_MRq"].net.loc["2016-01-04":DESIGN_END]
print("[재현] swing MRq CAGR·MDD", cagr_mdd(_mr), "기준 15.87 / −18.22")
_p1 = (L["X_MRq"].net + L["X_FLzVT"].net).loc["2016-01-04":DESIGN_END]
print("[재현] swing P1z CAGR·MDD", cagr_mdd(_p1), "기준 30.56 / −18.03")
near(cagr_mdd(_mr)[0], 15.87, .1, "swing MRq CAGR")
near(cagr_mdd(_p1)[0], 30.56, .3, "swing P1z CAGR")
print("첫 1분 근사(평균 변화, 표본 C3 날):", PROXY)
print("특수일:", [str(x.date()) for x in SPECIAL])


# ---------------------------------------------------------------- 무결성: 설계 구간 스캔 대조
scan = pickle.load(open(SCAN_CACHE, "rb"))
assert scan["PROXY"] == {k: PROXY[k] for k in scan["PROXY"]}, "첫 1분 상수 불일치"
assert list(scan["SPECIAL"]) == [d for d in SPECIAL if d <= DESIGN_END], "특수일 불일치"
rows = []
for k, v in scan["L"].items():
    for f in ("net", "stress", "gross", "pos", "fut", "auc"):
        a, b = getattr(v, f), getattr(L[k], f).reindex(v.net.index)
        assert a.isna().equals(b.isna()), (k, f, "유효 구간 불일치")
        rows.append(dict(leg=k, field=f, n=int(a.notna().sum()), max_abs_diff=float((a - b).abs().max())))
INTEG = pd.DataFrame(rows)
print("스캔 대조 필드별 최대 차:", INTEG.groupby("field").max_abs_diff.max().to_dict())
assert INTEG.max_abs_diff.max() < 1e-12, INTEG.nlargest(5, "max_abs_diff")


def dstat(k, a, b, mult=1.0):
    """설계 재현: 거래 단위 순 bp·NW t. mult = 1배 환산."""
    x = TR[k].net.loc[a:b] * 1e4 / mult
    return len(x), x.mean(), fl.nw_t(x.values)


DESIGN = {"KA_STf": ("2010-11-01", 341, 9.75, 3.05, 2.0), "X_K2": ("2011-01-01", 982, 9.45, 3.36, 1.0), "X_D1": ("2016-01-01", 185, 44.22, 3.88, 1.0), "X_N1": ("2015-01-01", 475, 14.77, 4.01, 1.0),
          "X_MRq": ("2016-01-04", 262, 52.64, 3.31, 1.0), "X_FLzVT": ("2016-01-04", 325, 33.21, 3.12, 1.0)}
for k, (a, n, net, t, mu) in DESIGN.items():
    n2, net2, t2 = dstat(k, a, DESIGN_END, mu)
    print(f"[재현] {k}: n {n2}/{n}, 순 {net2:.2f}/{net}, t {t2:.2f}/{t}")
    # X_FLzVT: 봉인선에 걸친 마지막 거래가 스캔에서는 2024-12-30 에서 잘렸고 여기서는 끝까지 간다(일수익은 위에서 차 0 확인). 그 한 건 몫만 허용
    tol = (.4, .05) if k == "X_FLzVT" else (.03, .03)
    assert n2 == n and abs(net2 - net) < tol[0] and abs(t2 - t) < tol[1], k  # 기준은 사전 등록 표기(소수 둘째 자리)
_t = TR["X_FLzVT"].loc[:DESIGN_END]
print("X_FLzVT 마지막 설계 거래 진입 결정일", _t.index[-1].date(), f"순 {_t.net.iloc[-1] * 1e4:.1f}bp, 이 거래 몫 평균 영향 {(_t.net.mean() - _t.net.iloc[:-1].mean()) * 1e4:+.2f}bp")
_x = L["M_zfQ"].net.loc[:DESIGN_END].dropna()
_a = _x[_x != 0] * 1e4
print(f"[재현] M_zfQ: n {len(_a)}/38, 순 {_a.mean():.2f}/73.66, t {me.nw_t(_a):.2f}/2.58")
assert len(_a) == 38 and abs(_a.mean() - 73.66) < .02 and abs(me.nw_t(_a) - 2.58) < .02

# ---------------------------------------------------------------- 평가 끝: 입력 원본 끝(봉인선 전) → 손익일 끝
B_, X_ = GIT / "beomgu/data", RAW / "r15_external"


def file_end(p, col="date"):
    return pd.to_datetime(pd.read_csv(p, encoding="utf-8-sig", usecols=[col])[col]).max()


SRC = {"069500 수정가": file_end(B_ / "etf/069500.csv"), "122630 수정가": file_end(B_ / "etf/122630.csv"), "229200 수정가": file_end(B_ / "etf/229200.csv"), "233740 수정가": file_end(B_ / "etf/233740.csv"),
       "229200 원가격": file_end(JH / "229200_daily.csv"), "069500 분봉": file_end(JH / "minute/069500_TOSS_1m.csv"), "122630 분봉": file_end(JH / "minute/122630_TOSS_1m.csv"),
       "229200 분봉": file_end(JH / "minute/229200_TOSS_1m.csv"), "233740 분봉": file_end(JH / "minute/233740_TOSS_1m.csv"), "코스닥150 지수": file_end(JH / "KOSDAQ_150_ohlcv.csv", "Date"),
       "코스닥150 선물 투자자별": file_end(JH / "KOSDAQ_150_fut_foreign.csv", "Date"), "코스피200 선물 투자자별": file_end(B_ / "naver_fut_investor_daily.csv"), "코스닥 현물 투자자별": file_end(B_ / "naver_kosdaq_investor_daily.csv"),
       "코스닥 프로그램": file_end(B_ / "naver_kosdaq_program_daily.csv"), "SPX": file_end(B_ / "spx_daily.csv"), "코스피200 선물 일봉": file_end(X_ / "kis_fut_k200_daily.csv"), "코스닥150 선물 일봉": file_end(X_ / "kis_fut_kq150_daily.csv"),
       "코스닥150 선물 1분봉": file_end(RAW / "intraday/kis_minute/kq150_full_day/futures_1min.csv"), **{f"series {h}": d for h, d in LAST.items()}}
# 다리 → (자료, 시차). 시차 k = 그 자료의 T 값이 T+k 거래일 손익까지 쓰인다
_F2 = [("코스닥150 선물 일봉", 0)]
LEG_SRC = {"KB_E1s3": [("series beomgu_C3VT30", 0), ("069500 수정가", 0), ("122630 수정가", 0), ("069500 분봉", 0), ("122630 분봉", 0)],
           "KC_KP": [("series jehyeong_KP", 0)], "QA_F": [("series blog_ON_Q", 0)] + _F2, "QC_KQ": [("series jehyeong_KQ", 0)],
           "KA_STf": [("069500 수정가", 1), ("코스닥150 지수", 1), ("코스피200 선물 일봉", 0)], "X_K2": [("코스피200 선물 투자자별", 1), ("코스피200 선물 일봉", 0)],
           "X_D1": [("코스닥150 선물 투자자별", 1)] + _F2, "X_N1": [("코스닥150 선물 투자자별", 2), ("코스피200 선물 투자자별", 2)] + _F2,
           "X_MRq": [("코스닥150 지수", 1)] + _F2, "X_FLzVT": [("코스닥 프로그램", 1), ("코스닥 현물 투자자별", 1), ("코스닥150 지수", 1)] + _F2,
           "QC_Df": [("229200 분봉", 0), ("233740 분봉", 0), ("229200 원가격", 1)], "M_zfQ": [("SPX", 1), ("229200 수정가", 0), ("233740 수정가", 0), ("229200 분봉", 0), ("코스닥150 지수", 1)] + _F2,
           "C1_RB": [("229200 수정가", 0), ("코스닥150 선물 1분봉", 0)]}


def pnl_end(d, lag):
    i = CAL.searchsorted(d, side="right") - 1
    return CAL[min(i + lag, len(CAL) - 1)]


rows, LEG_END = [], {}
for k, src in LEG_SRC.items():
    ends = {nm: pnl_end(SRC[nm], lag) for nm, lag in src}
    LEG_END[k] = min(ends.values())
    rows += [dict(leg=k, 자료=nm, 자료끝=SRC[nm].date(), 시차=lag, 손익끝=ends[nm].date(), 달력끝=CAL[-1].date(), 다리평가끝=LEG_END[k].date(), 묶는자료=ends[nm] == LEG_END[k]) for nm, lag in src]
    v = L[k]
    for f in ("net", "stress", "gross", "pos", "fut", "auc"):
        setattr(v, f, getattr(v, f).where(CAL <= LEG_END[k]))
    v.end = LEG_END[k]
    if k in TR:
        TR[k] = TR[k].loc[:LEG_END[k]]
ENDS = pd.DataFrame(rows)
print(ENDS.groupby("leg", sort=False).다리평가끝.first().to_string())

print("다리 수:", len(L))
with open(CACHE, "wb") as fh:
    pickle.dump(dict(L=L, CAL=CAL, BASE=BASE, PROXY=PROXY, SPECIAL=SPECIAL, TR=TR, INTEG=INTEG, ENDS=ENDS, LEG_END=LEG_END, FRONT=FRONT), fh)
print("저장:", CACHE)
