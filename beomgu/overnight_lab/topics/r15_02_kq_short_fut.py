"""15차 설계 02_kq_short_fut: 코스닥 숏·인버스 다리의 코스닥150 선물 집행 전환(251340 틱 비용 반영).

규칙(사전 등록, 결과 보기 전 고정; alpha-share-prereg-2026-10-30.md, design_shortlist.md §2 02)
  • 신호·시각·노출은 13차 series 그대로. 롱 다리(233740·229200)는 ETF 유지.
  • (a) blog_ON_Q pos=−1 밤(인버스 밤): net = −ln(F_open,T+1 / F_close,T) − 0.00018.
    F = kis_fut_kq150_daily.csv front(종가 15:45, 시가 08:45). 밤 T→T+1 의 계약 = T+1 의 front 코드
    (최종거래일 전날 밤은 자동으로 차근월: KIS front 는 만기 당일부터 차근월물). 2026-10-30 행 제외. pos=+2 밤은 원 series 그대로.
  • (b) jehyeong_KQ pos<0 낮 숏: 코스닥150 선물 1분봉(kq150_full_day, 2025-10-22~)이 있는 날은 거래별 진입·청산 시각의
    선물 봉으로 실측(0.00018×|pos| 차감), 그 전은 229200 TOSS 1분봉 대리(신호가 = 원 거래의 signal 가격과 일치 확인).
    진입 = 신호봉(라벨 entry_time) 종가, 손절 청산 = 라벨 exit_bar_end 봉 시가, 15:20 강제청산 = 15:20 봉 종가.
    선물 봉 time(시작) + 1분 = ETF 라벨(r7 README) → 선물 봉은 라벨 − 1분.
  • 현실 기준선(틱 경로) = 13차 net − 251340 틱 비용(2024 13.6, 2025 14.2, 2026 21.5, 그 외 10bp) × |pos|, 숏 거래마다.
  • 계약수 = 자본 × |pos| ÷ (F × 10,000) 반올림, 자본 1억.
  • 통계: 이벤트별 순수익 bp 상수 회귀 HAC(maxlags 5) → NW t. Δ = 같은 밤 짝지은 차(선물 − 틱) 상수 회귀 HAC t.
    CAGR = Π(1+r)^(252/일수) − 1(비활성일 0, 069500 달력), MDD = min(누적/최고 − 1).
  • 판정(고정): 선물 경로 순수익 > 0 ∧ 활성일 NW t ≥ 2 ∧ Δ(선물 − 틱) > 0 ∧ 장부 결합(코스닥 A·C 교체) W_all CAGR·MDD 가
    13차 원수치(31.4%/−6.1%, 소수 첫째 자리) 대비 비악화 ∧ 세후 비교표. 새 수익원 아님 → 연도 부호는 다리 단독 보고만.
  • 변형(보고용): V1 (b) / V2 jehyeong_ON 롱 밤 미니선물(kis_fut_k200_daily front 가격 대용) / V3 롱 밤 233740 → 선물 2배 /
    V4 환매 08:45 vs 09:00 / V5 blog_ON_Q1x + 숏 선물 / V6 3bp + 만기 주 차근월 롤.
자료(읽기만): round13_slot_book/series/*.csv·meta.json, book/book_daily.csv, r15_external/kis_fut_kq150_daily.csv·kis_fut_k200_daily.csv,
  intraday/kis_minute/kq150_full_day·full_day/futures_1min.csv, jehyeong/intraday/results/final/KOSDAQ150/trades.csv,
  jehyeong/intraday/data/minute/229200·069500_TOSS_1m.csv, etf/229200·233740·069500.csv.
가정: 선물 체결은 KIS 일봉 시가(08:45 단일가)·종가(15:45 단일가) 그대로(슬리피지 1.8bp 왕복에 포함). 미니선물 가격 = 정규 선물 가격.
  단일가 가정 의존: ETF 쪽(원 series·틱 경로)은 13차 그대로 09:00·15:30 단일가, 선물 쪽은 08:45·15:45 단일가 → 창 차이 15분 두 번.
산출: round15_alpha_share/02_kq_short_fut/{cells.csv, trades.csv, notes.md, fig1_legs.png, fig2_book.png, tax_table.csv, summary.json},
  series/{blog_ON_Q_fut, blog_ON_Q_tick, jehyeong_KQ_fut, jehyeong_KQ_tick}.csv + .meta.json
실행: PYTHONUTF8=1 python "<code>/topics/r15_02_kq_short_fut.py"
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.stdout.reconfigure(encoding="utf-8")
plt.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

LAB = Path("<OUT>")
R13 = LAB / "round13_slot_book"
R15 = LAB / "round15_alpha_share"
OUT = R15 / "02_kq_short_fut"
SER = R15 / "series"
RAW = Path("<raw>")
JH = Path("<repo>/jehyeong/intraday")
OUT.mkdir(parents=True, exist_ok=True)
SER.mkdir(parents=True, exist_ok=True)

FUT_BP, FUT_BP_STRESS = 1.8e-4, 3.0e-4
TICK = {2024: 13.6, 2025: 14.2, 2026: 21.5}          # 251340 틱 비용 bp(그 외 연도 10bp)
CAP = 1e8                                             # 계약수 분해능 가정 자본
MULT = 10_000                                         # 코스닥150 선물 승수(원/pt)
W_A = ("2016-08-10", "2026-09-29")                    # (a) 주 창 = blog_ON_Q 유효 구간
W_ALL = ("2022-12-13", "2026-09-17")
W_EX = ("2022-12-13", "2025-12-30")
W_LONG = ("2016-10-04", "2026-09-17")
W_KQ = ("2022-12-13", "2026-09-29")                   # (b) jehyeong_KQ 유효 구간
W_ON = ("2021-12-20", "2026-09-28")                   # V2 jehyeong_ON 유효 구간
YEARS = {"Y2024": ("2024-01-02", "2024-12-30"), "Y2025": ("2025-01-02", "2025-12-30"), "Y2026": ("2026-01-02", "2026-09-17")}
CAL = pd.read_csv(RAW / "etf/069500.csv", parse_dates=["date"]).set_index("date").index.sort_values()
CELLS, TRADES, NOTE = [], [], []


def D(ts):
    """notes.md 용 날짜(하이픈 금지 규약 → 점 표기)."""
    return pd.Timestamp(ts).strftime("%Y.%m.%d")


def W(w):
    return f"{D(w[0])}~{D(w[1])}"


# ---------------------------------------------------------------- 자료
def load_series(h, root=R13):
    s = pd.read_csv(root / "series" / f"{h}.csv", parse_dates=["date"]).set_index("date")
    meta = json.loads((root / "series" / f"{h}.meta.json").read_text(encoding="utf-8"))
    return s, meta


def valid(s, meta):
    return s.loc[meta["first_valid"]:meta["last_valid"]].fillna(0.0)


def read_fut(path):
    f = pd.read_csv(path, parse_dates=["date"])
    f = f[f.date < pd.Timestamp("2026-10-30")]            # 장중 미확정 행 제외
    rows = f.set_index(["date", "code"])[["open", "close", "volume"]].sort_index()
    front = f[f.front == 1].set_index("date").code.sort_index()
    assert front.index.is_unique and (front.index == f[f.front == 1].date.sort_values().values).all()
    return rows, front


def night_table(rows, front, roll_week=False):
    """밤 T→T+1(연속 front 거래일). 계약 = T+1 의 front(만기 전날 밤은 자동 차근월). roll_week: 만기 주(스위치일 7일 전~) 는 새 front."""
    d = front.index
    tab = pd.DataFrame({"T": d[:-1], "date": d[1:], "code": front.values[1:]})
    tab["switch"] = tab.code.values != front.values[:-1]
    if roll_week:
        for E in tab.date[tab.switch]:
            tab.loc[(tab.date > E - pd.Timedelta(days=7)) & (tab.date <= E), "code"] = front.loc[E]
    tab["close_T"] = rows.close.reindex(pd.MultiIndex.from_arrays([tab["T"], tab.code])).values
    tab["open_T1"] = rows.open.reindex(pd.MultiIndex.from_arrays([tab.date, tab.code])).values
    tab["lnret"] = np.log(tab.open_T1 / tab.close_T)
    return tab.set_index("date")


def read_1min(path):
    f = pd.read_csv(path)
    f["date"] = pd.to_datetime(f.date)
    f["ts"] = f.date + pd.to_timedelta(f.time + ":00")
    return f.set_index("ts").sort_index()


def read_toss(code):
    m = pd.read_csv(JH / "data/minute" / f"{code}_TOSS_1m.csv", encoding="utf-8-sig", parse_dates=["timestamp"])
    return m.set_index("timestamp").sort_index()


# ---------------------------------------------------------------- 통계
def hac_t(x, maxlags=5):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) < 3 or np.allclose(x, x[0]):
        return np.nan
    return float(sm.OLS(x, np.ones(len(x))).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags}).tvalues[0])


def daily_from_events(ev_net, w):
    cal = CAL[(CAL >= w[0]) & (CAL <= w[1])]
    s = pd.Series(0.0, index=cal)
    e = ev_net.loc[w[0]:w[1]]
    assert e.index.isin(cal).all(), "이벤트 날짜가 069500 달력 밖"
    s.loc[e.index] = e.values
    return s


def cagr_mdd(r):
    r = r.dropna()
    if len(r) == 0:
        return np.nan, np.nan
    eq = (1 + r).cumprod()
    cum = float(eq.iloc[-1])
    cagr = cum ** (252 / len(r)) - 1 if cum > 0 else -1.0
    return cagr, float((eq / eq.cummax() - 1).min())


def cell(cid, name, market, w, ev, ref=None, verdict=None, notes=""):
    """ev: date 색인, 열 net·gross(자본 1 기준). ref: 같은 날짜의 짝지은 기준 net(Δ = ev − ref)."""
    e = ev.loc[w[0]:w[1]]
    n = len(e)
    row = dict(id=cid, name=name, market=market, window=f"{w[0]}~{w[1]}", n=n, gross_bp=np.nan, net_bp=np.nan, t_nw=np.nan,
               delta_bp=np.nan, t_delta=np.nan, win_pct=np.nan, cagr_pct=np.nan, mdd_pct=np.nan, yearly_pos="", top5_share_pct=np.nan,
               half1_net_bp=np.nan, half2_net_bp=np.nan, verdict=verdict or "estimate", notes=notes)
    if n == 0:
        CELLS.append(row)
        return row
    net, gross = e.net.values * 1e4, e.gross.values * 1e4
    daily = daily_from_events(e.net, w)
    cagr, mdd = cagr_mdd(daily)
    g = e.net.groupby(e.index.year).agg(["sum", "size"])
    yrs = g[g["size"] >= 10]
    ypos, yn = int((yrs["sum"] > 0).sum()), len(yrs)
    tot = net.sum()
    top5 = np.sort(net)[-5:].sum() / tot * 100 if tot > 0 else np.nan
    h = n // 2
    row.update(gross_bp=gross.mean(), net_bp=net.mean(), t_nw=hac_t(net), win_pct=(net > 0).mean() * 100,
               cagr_pct=cagr * 100, mdd_pct=mdd * 100, yearly_pos=f"{ypos}/{yn}" if yn else "",
               top5_share_pct=top5, half1_net_bp=net[:h].mean() if h else np.nan, half2_net_bp=net[h:].mean())
    if ref is not None:
        d = (e.net - ref.reindex(e.index)).values * 1e4
        row.update(delta_bp=float(np.nanmean(d)), t_delta=hac_t(d))
    if verdict is None:
        if len(daily) <= 300 or n < 10:
            v = "estimate"
        elif net.mean() <= 0:
            v = "reject"
        elif row["t_nw"] >= 2 and yn and ypos / yn >= 0.7 and (np.isnan(top5) or top5 < 50) and row["half1_net_bp"] > 0 and row["half2_net_bp"] > 0:
            v = "candidate"
        else:
            v = "reference"
        row["verdict"] = v
    CELLS.append(row)
    return row


def book_cell(cid, name, w, r, notes="", market="COMB"):
    r = r.loc[w[0]:w[1]].dropna()
    cagr, mdd = cagr_mdd(r)
    act = r[r != 0]
    row = dict(id=cid, name=name, market=market, window=f"{w[0]}~{w[1]}", n=int((r != 0).sum()), gross_bp=np.nan,
               net_bp=r.mean() * 1e4, t_nw=hac_t(r.values * 1e4), delta_bp=np.nan, t_delta=np.nan,
               win_pct=(act > 0).mean() * 100 if len(act) else np.nan, cagr_pct=cagr * 100, mdd_pct=mdd * 100, yearly_pos="",
               top5_share_pct=np.nan, half1_net_bp=np.nan, half2_net_bp=np.nan, verdict="diagnostic", notes=notes)
    CELLS.append(row)
    return row


def dist_cell(cid, name, market, w, x_bp, notes=""):
    x = pd.Series(x_bp).dropna()
    row = dict(id=cid, name=name, market=market, window=f"{w[0]}~{w[1]}", n=len(x), gross_bp=np.nan, net_bp=x.mean(), t_nw=hac_t(x),
               delta_bp=np.nan, t_delta=np.nan, win_pct=(x > 0).mean() * 100, cagr_pct=np.nan, mdd_pct=np.nan, yearly_pos="",
               top5_share_pct=np.nan, half1_net_bp=np.nan, half2_net_bp=np.nan, verdict="diagnostic",
               notes=f"std {x.std():.1f} p05 {x.quantile(.05):.1f} p95 {x.quantile(.95):.1f} 평균|x| {x.abs().mean():.1f} | {notes}")
    CELLS.append(row)
    return row


def tick_bp(dates):
    return pd.Index(dates).year.map(lambda y: TICK.get(y, 10.0)).values / 1e4


def save_series(handle, s_full, meta_src, cost_bp, notes, source_files, passed):
    """13차 series 와 같은 달력·열로 저장. 바꾸지 않은 행은 원본과 수치 동일(assert)."""
    orig, _ = load_series(meta_src["handle"])
    assert (orig.index == s_full.index).all()
    chg = ~np.isclose(orig.net.fillna(-9).values, s_full.net.fillna(-9).values)
    assert np.allclose(orig.pos.fillna(-9).values, s_full.pos.fillna(-9).values)
    assert np.allclose(orig.gross.fillna(-9).values[~chg], s_full.gross.fillna(-9).values[~chg])
    s_full.to_csv(SER / f"{handle}.csv", date_format="%Y-%m-%d")
    v = valid(s_full, meta_src)
    cagr, mdd = cagr_mdd(v.net)
    meta = dict(meta_src)
    meta.update(handle=handle, cost_bp=cost_bp, source_files=source_files, replaced_from=meta_src["handle"], design="02_kq_short_fut",
                created="2026-10-30", n_changed_rows=int(chg.sum()), passed=bool(passed),
                repro={"n_active": int((v.pos != 0).sum()), "CAGR": cagr, "MDD": mdd, "비고": "저장 CSV 재계산(유효 구간 전체)"},
                notes=notes)
    (SER / f"{handle}.meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    return int(chg.sum()), cagr, mdd


# ================================================================ (a) blog_ON_Q 인버스 밤
def part_a():
    rows, front = read_fut(RAW / "r15_external/kis_fut_kq150_daily.csv")
    nt = night_table(rows, front)
    nt_roll = night_table(rows, front, roll_week=True)
    s, meta = load_series("blog_ON_Q")
    v = valid(s, meta)
    assert set(np.unique(v.pos)) <= {0.0, 2.0, -1.0}, set(np.unique(v.pos))
    inv = v[v.pos == -1].copy()
    # 달력 대조: 유효 구간에서 069500 달력 == 선물 front 달력
    cal = CAL[(CAL >= W_A[0]) & (CAL <= W_A[1])]
    fcal = front.loc[W_A[0]:W_A[1]].index
    NOTE.append(f"• 자료 점검 kis_fut_kq150_daily: front 유일 {front.index.is_unique}, 유효 창 {W(W_A)} 069500 달력 {len(cal)}일 중 선물 front 결측 "
                f"{len(cal.difference(fcal))}일, 선물에만 있는 날 {len(fcal.difference(cal))}일, 2026.10.30 행 제외, 월물 스위치 {int(nt.switch.sum())}회, "
                f"스위치 밤의 새 front 가 T 에 없는 경우 {int(nt[nt.switch].close_T.isna().sum())}건, 스위치 밤 새 front T 거래량 최소 "
                f"{int(rows.volume.reindex(pd.MultiIndex.from_arrays([nt[nt.switch]['T'], nt[nt.switch].code])).min())}계약.")
    inv = inv.join(nt[["T", "code", "close_T", "open_T1", "lnret", "switch"]])
    inv = inv.join(nt_roll[["lnret", "code"]].rename(columns={"lnret": "lnret_roll", "code": "code_roll"}))
    assert inv.lnret.notna().all(), inv[inv.lnret.isna()]
    n_roll_na = int(inv.lnret_roll.isna().sum())
    inv["lnret_roll"] = inv.lnret_roll.fillna(inv.lnret)
    inv["gross_fut"] = -inv.lnret
    inv["net_fut"] = inv.gross_fut - FUT_BP
    inv["net_fut3"] = inv.gross_fut - FUT_BP_STRESS
    inv["net_roll"] = -inv.lnret_roll - FUT_BP
    inv["tick"] = tick_bp(inv.index)
    inv["net_tick"] = inv.net - inv.tick
    inv["contracts"] = np.round(CAP / (inv.close_T * MULT))
    inv["contract_err"] = inv.contracts * inv.close_T * MULT / CAP - 1
    tr = inv.rename(columns={"gross": "gross_orig", "net": "net_orig"})
    tr["leg"] = "blog_ON_Q"
    tr["source"] = "kis_daily"
    TRADES.append(tr.reset_index())

    paths = {"orig": ("원 series(251340 ETF 15:30→09:00, 5bp)", inv[["gross", "net"]]),
             "tick": ("틱 경로(원 net − 251340 연도별 틱 비용)", pd.DataFrame({"gross": inv.gross, "net": inv.net_tick})),
             "fut": ("선물 경로(코스닥150 선물 15:45 매도→08:45 환매, 1.8bp)", pd.DataFrame({"gross": inv.gross_fut, "net": inv.net_fut}))}
    res = {}
    for wn, w in (("W_A", W_A), ("W_all", W_ALL), ("W_ex2026", W_EX), ("W_long", W_LONG)):
        for p, (label, ev) in paths.items():
            ref = inv.net_tick if p == "fut" else (inv.net if p == "tick" else None)
            r = cell(f"A_{p}_{wn}", f"(a) 인버스 밤 {label}", "KOSDAQ", w, ev, ref=ref,
                     notes="주 셀" if (p == "fut" and wn in ("W_A", "W_all")) else ("기준선" if p != "fut" else "보고 창"))
            res[(p, wn)] = r
        cell(f"A_fut_vs_orig_{wn}", "(a) 선물 경로 Δ vs 원 series(짝지은 차)", "KOSDAQ", w, paths["fut"][1], ref=inv.net, verdict="diagnostic",
             notes="delta_bp = 선물 − 원 series(창 차이 15:30↔15:45·09:00↔08:45 포함)")
    # V6
    for wn, w in (("W_A", W_A), ("W_all", W_ALL)):
        cell(f"V6_fut3bp_{wn}", "V6 선물 경로 비용 3bp(스트레스)", "KOSDAQ", w, pd.DataFrame({"gross": inv.gross_fut, "net": inv.net_fut3}), ref=inv.net_tick,
             notes="delta_bp = 3bp 선물 − 틱 경로")
        cell(f"V6_roll_{wn}", "V6 만기 주 차근월 롤(스위치일 7일 전부터 새 front, 1.8bp)", "KOSDAQ", w,
             pd.DataFrame({"gross": -inv.lnret_roll, "net": inv.net_roll}), ref=inv.net_fut,
             notes=f"delta_bp = 롤 경로 − 기본 선물 경로; 차근월 행 결측으로 기본 경로 대체 {n_roll_na}건(전체)")
    # 계약수 분해능
    dist_cell("A_contracts_1e8", "(a) 계약수 분해능(자본 1억, 반올림 오차 % 명목)", "KOSDAQ", W_A, inv.contract_err * 100,
              notes=f"계약수 평균 {inv.contracts.mean():.2f} 최소 {int(inv.contracts.min())} 최대 {int(inv.contracts.max())}; 단위 %(명목 괴리), 평균|오차| 는 notes 의 평균|x|")
    # 교체본·틱본 series(전체 달력)
    s_fut = s.copy()
    s_fut.loc[inv.index, "gross"] = inv.gross_fut.values
    s_fut.loc[inv.index, "net"] = inv.net_fut.values
    s_tick = s.copy()
    s_tick.loc[inv.index, "net"] = inv.net_tick.values
    return dict(inv=inv, s=s, meta=meta, s_fut=s_fut, s_tick=s_tick, res=res, nt=nt, rows=rows, front=front)


# ================================================================ (b) jehyeong_KQ 낮 숏 (V1)
def price_at(f1, ts, kind):
    """선물 1분봉 가격: kind open/close 를 봉 time == ts 에서, 봉이 없으면 직전 봉 종가(같은 날)."""
    if ts in f1.index:
        return float(f1.loc[ts, kind]), False
    day = f1.loc[ts.normalize():ts]
    assert len(day), ts
    return float(day.close.iloc[-1]), True


def part_b(a):
    s, meta = load_series("jehyeong_KQ")
    v = valid(s, meta)
    tr = pd.read_csv(JH / "results/final/KOSDAQ150/trades.csv", encoding="utf-8-sig")
    tr["date"] = pd.to_datetime(tr.date)
    tr = tr[(tr.direction < 0) & (tr.date >= meta["first_valid"]) & (tr.date <= meta["last_valid"])].copy()
    assert not tr.date.duplicated().any()
    tr = tr.set_index("date")
    lev = v.pos.reindex(tr.index)
    assert np.allclose(lev.abs(), tr.leverage) and (lev < 0).all()
    toss = read_toss("229200")
    te, tx = pd.to_datetime(tr.entry_time), pd.to_datetime(tr.exit_bar_end)
    p_in = toss.close.reindex(te).values
    p_out = np.where(tr.exit_reason == "forced_1520", toss.close.reindex(tx).values, toss.open.reindex(tx).values)
    ok_in, ok_out = np.isclose(p_in, tr.signal_entry_price), np.isclose(p_out, tr.signal_exit_price)
    NOTE.append(f"• 자료 점검 jehyeong_KQ 숏 {len(tr)}건({W(W_KQ)}): 229200 TOSS 신호봉 가격과 원 거래 signal 가격 일치 진입 {int(ok_in.sum())}/{len(tr)}, "
                f"청산 {int(ok_out.sum())}/{len(tr)}(손절 = exit_bar_end 봉 시가, 15:20 강제 = 15:20 봉 종가). 청산 사유 {tr.exit_reason.value_counts().to_dict()}. "
                f"|pos| 평균 {lev.abs().mean():.3f}(변동성 축소 {int((lev.abs() < 0.999).sum())}건).")
    # 원 gross 와 신호가 수익률의 차(원 모형은 체결 모형 포함)
    sig_gross = tr.direction * tr.leverage * (tr.signal_exit_price / tr.signal_entry_price - 1)
    NOTE.append(f"• 원 ETF gross(체결 모형 포함) − 신호가 수익률: 평균 {((tr.gross_pnl / tr.entry_equity - sig_gross) * 1e4).mean():+.2f}bp/건, "
                f"원 비용(슬리피지+수수료) 평균 {((tr.slippage_cost + tr.commission_cost) / tr.entry_equity * 1e4).mean():.2f}bp/건. 선물 경로는 신호 시각 봉 가격 + 1.8bp.")
    L = lev.abs().values
    proxy_gross = -L * np.log(tr.signal_exit_price.values / tr.signal_entry_price.values)
    # 실측(코스닥150 선물 1분봉)
    f1 = read_1min(RAW / "intraday/kis_minute/kq150_full_day/futures_1min.csv")
    f1_days = set(f1.date.unique())
    meas_gross, src, nfb = [], [], 0
    for d, r, e_ts, x_ts in zip(tr.index, tr.itertuples(), te, tx):
        if d not in f1_days:
            meas_gross.append(np.nan)
            src.append("proxy_229200")
            continue
        fin, fb1 = price_at(f1, e_ts - pd.Timedelta(minutes=1), "close")
        fout, fb2 = price_at(f1, x_ts - pd.Timedelta(minutes=1), "close" if r.exit_reason == "forced_1520" else "open")
        nfb += fb1 + fb2
        meas_gross.append(-abs(r.leverage) * np.log(fout / fin))
        src.append("kq150_1min")
    tr["lev"] = L
    tr["gross_proxy"] = proxy_gross
    tr["gross_meas"] = meas_gross
    tr["source"] = src
    tr["gross_fut"] = tr.gross_meas.fillna(tr.gross_proxy)
    tr["net_fut"] = tr.gross_fut - FUT_BP * L
    tr["net_proxy"] = tr.gross_proxy - FUT_BP * L
    tr["net_meas"] = tr.gross_meas - FUT_BP * L
    tr["gross_orig"] = v.gross.reindex(tr.index).values
    tr["net_orig"] = v.net.reindex(tr.index).values
    tr["tick"] = tick_bp(tr.index) * L
    tr["net_tick"] = tr.net_orig - tr.tick
    n_meas = int(tr.gross_meas.notna().sum())
    NOTE.append(f"• (b) 실측 {n_meas}건(선물 1분봉 {D(f1.date.min())}~{D(min(f1.date.max(), pd.Timestamp(meta['last_valid'])))}), 대리 {len(tr) - n_meas}건, "
                f"정확한 봉이 없어 직전 봉 종가로 대체한 가격 {nfb}건.")
    out = tr.reset_index()[["date", "entry_time", "exit_bar_end", "exit_reason", "lev", "signal_entry_price", "signal_exit_price", "source",
                             "gross_orig", "net_orig", "tick", "net_tick", "gross_proxy", "net_proxy", "gross_meas", "net_meas", "gross_fut", "net_fut"]]
    out["leg"] = "jehyeong_KQ"
    out["pos"] = -out.lev
    TRADES.append(out)
    paths = {"orig": ("원 series(229200 숏 신호가 + 5bp 체결 모형)", pd.DataFrame({"gross": tr.gross_orig, "net": tr.net_orig})),
             "tick": ("틱 경로(원 net − 251340 틱 비용×|pos|)", pd.DataFrame({"gross": tr.gross_orig, "net": tr.net_tick})),
             "fut": ("선물 경로(실측 2025.10.22~ + 229200 대리, 1.8bp×|pos|)", pd.DataFrame({"gross": tr.gross_fut, "net": tr.net_fut})),
             "proxy": ("선물 경로 대리만(229200 신호가, 1.8bp×|pos|)", pd.DataFrame({"gross": tr.gross_proxy, "net": tr.net_proxy}))}
    res = {}
    for wn, w in (("W_KQ", W_KQ), ("W_all", W_ALL), ("W_ex2026", W_EX)):
        for p, (label, ev) in paths.items():
            ref = tr.net_tick if p in ("fut", "proxy") else (tr.net_orig if p == "tick" else None)
            res[(p, wn)] = cell(f"V1_{p}_{wn}", f"V1 (b) KQ 낮 숏 {label}", "KOSDAQ", w, ev, ref=ref, notes="V1 주 변형" if p == "fut" else "기준선·대리")
        cell(f"V1_fut_vs_orig_{wn}", "V1 선물 경로 Δ vs 원 series(짝지은 차)", "KOSDAQ", w, paths["fut"][1], ref=tr.net_orig, verdict="diagnostic",
             notes="delta_bp = 선물 − 원 series(원 모형의 체결 모형 약 −3.8bp/건 제거분 + 비용 차 + 가격 차)")
    m = tr[tr.gross_meas.notna()]
    wm = (str(m.index.min().date()), str(m.index.max().date()))
    cell("V1_meas_only", "V1 (b) 실측만(코스닥150 선물 1분봉)", "KOSDAQ", wm, pd.DataFrame({"gross": m.gross_meas, "net": m.net_meas}), ref=m.net_proxy,
         verdict="estimate", notes="delta_bp = 실측 − 229200 대리(같은 거래), 창 < 1년")
    dist_cell("V1_meas_minus_proxy", "V1 실측 − 대리 (같은 숏 거래, bp)", "KOSDAQ", wm, (m.gross_meas - m.gross_proxy) * 1e4, notes="대리 오차 직접 실측")
    # 대리 오차 상한 보고 셀: 코스피200 선물 1분봉 vs 069500, 코스닥150 선물 vs 229200 (09:30 선물 시가/09:31 ETF 라벨 → 15:19/15:20 종가)
    def gap(fut, etf, w):
        fo = fut[fut.time == "09:30"].set_index("date").open
        fc = fut[fut.time == "15:19"].set_index("date").close
        eo = etf[etf.index.strftime("%H:%M") == "09:31"]
        ec = etf[etf.index.strftime("%H:%M") == "15:20"]
        eo = eo.set_index(eo.index.normalize()).open
        ec = ec.set_index(ec.index.normalize()).close
        j = pd.concat([np.log(fc / fo).rename("f"), np.log(ec / eo).rename("e")], axis=1).dropna().loc[w[0]:w[1]]
        return (j.f - j.e) * 1e4, j
    k1 = read_1min(RAW / "intraday/kis_minute/full_day/futures_1min.csv")
    g, j = gap(k1, read_toss("069500"), ("2025-09-22", "2026-10-02"))
    dist_cell("V1_err_k200_069500", "V1 대리 오차 상한: 코스피200 선물 1분봉 − 069500 1분봉(09:30→15:19 vs 09:31→15:20, bp)", "KOSPI", ("2025-09-22", "2026-10-02"), g,
              notes=f"상관 {j.f.corr(j.e):.4f}")
    g2, j2 = gap(f1, toss, ("2025-10-22", "2026-10-07"))
    dist_cell("V1_err_kq150_229200", "V1 대리 오차: 코스닥150 선물 1분봉 − 229200 1분봉(같은 시각, bp)", "KOSDAQ", ("2025-10-22", "2026-10-07"), g2,
              notes=f"상관 {j2.f.corr(j2.e):.4f}")
    s_fut = s.copy()
    s_fut.loc[tr.index, "gross"] = tr.gross_fut.values
    s_fut.loc[tr.index, "net"] = tr.net_fut.values
    s_tick = s.copy()
    s_tick.loc[tr.index, "net"] = tr.net_tick.values
    return dict(tr=tr, s=s, meta=meta, s_fut=s_fut, s_tick=s_tick, res=res, f1=f1, k1=k1, toss=toss)


# ================================================================ V2·V3·V4·V5
def part_v2():
    rows, front = read_fut(RAW / "r15_external/kis_fut_k200_daily.csv")
    nt = night_table(rows, front)
    s, meta = load_series("jehyeong_ON")
    v = valid(s, meta)
    ev = v[v.pos != 0].copy()
    assert (ev.pos == 1).all()
    ev = ev.join(nt[["T", "code", "close_T", "open_T1", "lnret", "switch"]])
    assert ev.lnret.notna().all(), ev[ev.lnret.isna()]
    ev["gross_fut"] = ev.lnret
    ev["net_fut"] = ev.gross_fut - FUT_BP
    for wn, w in (("W_ON", W_ON), ("W_all", W_ALL)):
        cell(f"V2_orig_{wn}", "V2 jehyeong_ON 롱 밤 원 series(069500 15:30→09:00, 5bp)", "KOSPI", w, ev[["gross", "net"]], notes="기준선")
        cell(f"V2_fut_{wn}", "V2 jehyeong_ON 롱 밤 미니선물 15:45 매수→08:45 매도(정규 선물 front 가격 대용, 1.8bp)", "KOSPI", w,
             pd.DataFrame({"gross": ev.gross_fut, "net": ev.net_fut}), ref=ev.net, notes="delta_bp = 선물 − 원 ETF(창 차이 포함); 세전, 세후는 tax_table.csv")
    t = ev.rename(columns={"gross": "gross_orig", "net": "net_orig"})
    t["leg"] = "jehyeong_ON"
    t["source"] = "kis_daily_k200"
    TRADES.append(t.reset_index())
    return ev


def part_v3(a):
    inv_nt = a["nt"]
    v = valid(a["s"], a["meta"])
    lg = v[v.pos == 2].copy().join(inv_nt[["T", "code", "close_T", "open_T1", "lnret"]])
    assert lg.lnret.notna().all()
    lg["gross_fut"] = 2 * lg.lnret
    lg["net_fut"] = lg.gross_fut - 2 * FUT_BP
    e2 = pd.read_csv(RAW / "etf/233740.csv", parse_dates=["date"]).set_index("date")
    e1 = pd.read_csv(RAW / "etf/229200.csv", parse_dates=["date"]).set_index("date")
    co2 = e2.open / e2.close.shift(1) - 1
    co1 = e1.open / e1.close.shift(1) - 1
    resid = (co2 - 2 * co1).dropna()
    for wn, w in (("W_A", W_A), ("W_all", W_ALL)):
        cell(f"V3_orig_{wn}", "V3 blog_ON_Q 롱 밤 원 series(233740 2x ETF 15:30→09:00, 5bp)", "KOSDAQ", w, lg[["gross", "net"]], notes="기준선")
        cell(f"V3_fut2x_{wn}", "V3 blog_ON_Q 롱 밤 코스닥150 선물 2배(15:45→08:45, 2×1.8bp)", "KOSDAQ", w,
             pd.DataFrame({"gross": lg.gross_fut, "net": lg.net_fut}), ref=lg.net, notes="delta_bp = 선물 2x − 233740(창 차이 포함)")
        dist_cell(f"V3_resid233740_{wn}", "V3 근거: 233740 밤 수익 − 2×229200 밤 수익(종가→익일 시가, 전 거래일, bp)", "KOSDAQ", w, resid.loc[w[0]:w[1]] * 1e4)
        dist_cell(f"V3_resid233740_longnights_{wn}", "V3 근거: 같은 잔차, blog_ON_Q 롱 밤만", "KOSDAQ", w, resid.reindex(lg.loc[w[0]:w[1]].index) * 1e4)
    t = lg.rename(columns={"gross": "gross_orig", "net": "net_orig"})
    t["leg"] = "blog_ON_Q_long_V3"
    t["source"] = "kis_daily"
    TRADES.append(t.reset_index())
    return lg


def part_v4(a, b):
    def og(f1):
        o45 = f1[f1.time == "08:45"].set_index("date").open
        o00 = f1[f1.time == "09:00"].set_index("date").open
        return (np.log(o00 / o45)).dropna() * 1e4
    k = og(b["k1"])
    q = og(b["f1"])
    dist_cell("V4_k200_0845_0900", "V4 코스피200 선물 08:45 시가→09:00 시가(bp, 대리; 숏 환매를 09:00 으로 늦추면 −x 만큼 손익 변화)", "KOSPI", ("2025-09-22", "2026-10-02"), k)
    dist_cell("V4_kq150_0845_0900", "V4 코스닥150 선물 08:45 시가→09:00 시가(bp, 직접)", "KOSDAQ", ("2025-10-22", "2026-10-29"), q)
    # 인버스 밤 창 차이 분해(코스닥150 선물 1분봉 구간): 선물 경로(15:45→08:45) − ETF 창(15:30→09:00, 선물 가격)
    f1 = b["f1"]
    inv = a["inv"]
    c29 = f1[f1.time == "15:29"].set_index("date").close
    c45 = f1[f1.time == "15:45"].set_index("date").close
    o45 = f1[f1.time == "08:45"].set_index("date").open
    o00 = f1[f1.time == "09:00"].set_index("date").open
    x = inv[["T"]].copy()
    x["fut"] = -np.log(o45.reindex(x.index).values / c45.reindex(x["T"]).values)
    x["etfwin"] = -np.log(o00.reindex(x.index).values / c29.reindex(x["T"]).values)
    x = x.dropna()
    chk = (x.fut - (-inv.lnret.reindex(x.index))).abs().max()
    dist_cell("V4_inv_window_diff", "V4 인버스 밤 창 차이: 선물 경로(15:45→08:45) − 같은 선물의 ETF 창(15:30→09:00), bp", "KOSDAQ",
              (str(x.index.min().date()), str(x.index.max().date())), (x.fut - x.etfwin) * 1e4, notes=f"1분봉 vs 일봉 선물 경로 최대 차 {chk*1e4:.2f}bp")
    return k, q


def part_v5(a):
    s, meta = load_series("blog_ON_Q1x")
    v = valid(s, meta)
    inv = a["inv"]
    assert (v.pos.reindex(inv.index) == -1).all()
    s_fut = s.copy()
    s_fut.loc[inv.index, "gross"] = inv.gross_fut.values
    s_fut.loc[inv.index, "net"] = inv.net_fut.values
    s_tick = s.copy()
    s_tick.loc[inv.index, "net"] = inv.net_tick.values
    for wn, w in (("W_A", W_A), ("W_all", W_ALL)):
        for p, ss in (("orig", s), ("tick", s_tick), ("fut", s_fut)):
            r = valid(ss, meta).net
            book_cell(f"V5_Q1x_{p}_{wn}", f"V5 다리 blog_ON_Q1x(롱 1배 229200) {p} 경로 전체 다리", w, r, market="KOSDAQ")
    return s_fut


# ================================================================ 장부 결합
def part_book(a, b, q1x_fut):
    book = pd.read_csv(R13 / "book/book_daily.csv", parse_dates=["date"]).set_index("date")
    sB, mB = load_series("beomgu_C3Q")
    B = valid(sB, mB).reindex(book.index).fillna(0.0)
    A0 = valid(a["s"], a["meta"]).reindex(book.index).fillna(0.0)
    C0 = valid(b["s"], b["meta"]).reindex(book.index).fillna(0.0)
    assert np.allclose(A0.net, book.KOSDAQ_A_blog_ON_Q) and np.allclose(C0.net.where(B.pos == 0, 0.0), book.KOSDAQ_C_jehyeong_KQ)

    def comb(A_net, C_net):
        kq = A_net.reindex(book.index).fillna(0.0) + B.net + C_net.reindex(book.index).fillna(0.0).where(B.pos == 0, 0.0)
        return 0.5 * (book.KOSPI_book + kq), kq

    c0, kq0 = comb(A0.net, C0.net)
    assert np.allclose(c0, book.comb_a) and np.allclose(kq0, book.KOSDAQ_book)
    sKB, mKB = load_series("beomgu_C3VT30")
    KB = valid(sKB, mKB).loc[W_LONG[0]:W_LONG[1]]
    BL = valid(sB, mB).loc[W_LONG[0]:W_LONG[1]]

    def comb_long(s_A):
        AL = valid(s_A, a["meta"]).loc[W_LONG[0]:W_LONG[1]]
        idx = KB.index.intersection(AL.index).intersection(BL.index)
        return 0.5 * (KB.net.reindex(idx) + AL.net.reindex(idx) + BL.net.reindex(idx))

    cl0 = comb_long(a["s"])
    cg, md = cagr_mdd(cl0)
    assert len(cl0) == 2444 and abs(cg - 0.150369) < 0.002 and abs(md + 0.096107) < 0.001, (len(cl0), cg, md)
    cg, md = cagr_mdd(c0)
    assert abs(cg - 0.314179) < 0.002 and abs(md + 0.061046) < 0.001, (cg, md)
    variants = {"orig": ("13차 원 장부", a["s"], b["s"]), "tick": ("틱 경로(현실 기준선: A 인버스 밤·C 숏 251340 틱 비용)", a["s_tick"], b["s_tick"]),
                "fut": ("선물 경로(A 인버스 밤·C 숏 → 코스닥150 선물)", a["s_fut"], b["s_fut"]),
                "fut_A_only": ("A 인버스 밤만 선물(C 원본)", a["s_fut"], b["s"]), "fut_C_only": ("C 숏만 선물(A 원본)", a["s"], b["s_fut"]),
                "V5_Q1x_fut": ("V5 A = blog_ON_Q1x 롱 1배 + 숏 선물, C 숏 선물", q1x_fut, b["s_fut"])}
    # V6 3bp: A·C 선물 경로 비용 3bp
    sA3 = a["s_fut"].copy()
    sA3.loc[a["inv"].index, "net"] = a["inv"].net_fut3.values
    sC3 = b["s_fut"].copy()
    sC3.loc[b["tr"].index, "net"] = (b["tr"].gross_fut - FUT_BP_STRESS * b["tr"].lev).values
    variants["V6_fut3bp"] = ("V6 선물 경로 비용 3bp", sA3, sC3)
    metas = {}
    for k, (label, sA, sC) in variants.items():
        A = valid(sA, a["meta"])
        C = valid(sC, b["meta"])
        c, kq = comb(A.net, C.net)
        m = {}
        for wn, w in (("W_all", W_ALL), ("W_ex2026", W_EX), *YEARS.items()):
            cg, md = cagr_mdd(c.loc[w[0]:w[1]])
            cum = float((1 + c.loc[w[0]:w[1]]).prod() - 1)
            m[wn] = (cg, md, cum)
            book_cell(f"BOOK_{k}_{wn}", f"장부 합산 {label}", w, c, notes=f"누적 {cum*100:+.1f}%; 코스닥 장부 CAGR {cagr_mdd(kq.loc[w[0]:w[1]])[0]*100:.1f}% MDD {cagr_mdd(kq.loc[w[0]:w[1]])[1]*100:.1f}%")
        cl = comb_long(sA)
        cg, md = cagr_mdd(cl)
        m["W_long"] = (cg, md, float((1 + cl).prod() - 1))
        book_cell(f"BOOK_{k}_W_long", f"장부 합산(두 장부: 코스피 B + 코스닥 A·B) {label}", W_LONG, cl, notes=f"누적 {m['W_long'][2]*100:+.1f}%")
        metas[k] = m
        if k in ("orig", "tick", "fut"):
            metas[k + "_series"] = (c, kq)
    return metas


# ================================================================ 세후 표
def tax_table(a, v2):
    """자본 1억 가정 연도별 손익(원). 선물: 양도세 11%(소득세 10 + 지방 1), 기본공제 250만원/년(사람·연 단위, 전 파생 합산 → 여기서는 다리마다 따로 적용 = 낙관),
    손실 이월 없음. ETF: 국내주식형(069500·229200·233740)은 매매차익 비과세, 251340 은 과표기준가 증분 15.4%(국내 장내파생 손익은 과표 제외) → 세금 0 으로 두되 '미확인'."""
    rows = []
    for leg, ev_fut, ev_etf in (("(a) blog_ON_Q 인버스 밤", a["inv"].net_fut, a["inv"].net_tick), ("V2 jehyeong_ON 롱 밤", v2.net_fut, v2.net)):
        f = ev_fut.groupby(ev_fut.index.year).sum() * CAP
        e = ev_etf.groupby(ev_etf.index.year).sum() * CAP
        for y in f.index:
            pre = float(f.loc[y])
            post = pre - 0.11 * max(0.0, pre - 2.5e6)
            rows.append(dict(leg=leg, year=int(y), fut_pre_tax_krw=round(pre), fut_post_tax_krw=round(post), fut_tax_krw=round(pre - post),
                             etf_pre_tax_krw=round(float(e.loc[y])), etf_post_tax_krw=round(float(e.loc[y])), etf_tax_note="ETF 과세 미확인(0 가정)",
                             fut_minus_etf_post_krw=round(post - float(e.loc[y]))))
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "tax_table.csv", index=False, encoding="utf-8-sig")
    return t


# ================================================================ 그림
C = {"orig": "#2a78d6", "tick": "#eb6834", "fut": "#1baf7a"}
LAB_ = {"orig": "원 series(ETF)", "tick": "틱 경로(251340 호가 비용)", "fut": "선물 경로(코스닥150 선물)"}


def fig_legs(a, b):
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    panels = [(axes[0, 0], "blog_ON_Q 다리 전체(롱 밤 + 인버스 밤)", a["meta"], {"orig": a["s"], "tick": a["s_tick"], "fut": a["s_fut"]}, None),
              (axes[0, 1], "jehyeong_KQ 다리 전체(롱 + 숏)", b["meta"], {"orig": b["s"], "tick": b["s_tick"], "fut": b["s_fut"]}, None),
              (axes[1, 0], "blog_ON_Q 인버스 밤만(주 셀)", a["meta"], None, {"orig": a["inv"].net, "tick": a["inv"].net_tick, "fut": a["inv"].net_fut}),
              (axes[1, 1], "jehyeong_KQ 숏만(V1)", b["meta"], None, {"orig": b["tr"].net_orig, "tick": b["tr"].net_tick, "fut": b["tr"].net_fut})]
    for ax, title, meta, full, evs in panels:
        for p in ("orig", "tick", "fut"):
            r = valid(full[p], meta).net if full else daily_from_events(evs[p], (meta["first_valid"], meta["last_valid"]))
            eq = (1 + r).cumprod()
            cg, md = cagr_mdd(r)
            ax.plot(eq.index, eq, lw=1.5, color=C[p], label=f"{LAB_[p]}: CAGR {cg*100:.1f}% / MDD {md*100:.1f}%")
        ax.set_title(f"{title} {D(meta['first_valid'])}~{D(meta['last_valid'])}", fontsize=10, loc="left")
        ax.legend(frameon=False, fontsize=8, loc="upper left")
        ax.grid(alpha=0.25)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    fig.suptitle("02_kq_short_fut: 세 경로 누적(자본 1, 비활성일 0). 선물 경로는 15:45→08:45, ETF 경로는 15:30→09:00", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "fig1_legs.png", dpi=130)
    plt.close(fig)


def fig_book(metas):
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1.3]})
    for p in ("orig", "tick", "fut"):
        c, _ = metas[p + "_series"]
        r = c.loc[W_ALL[0]:W_ALL[1]]
        eq = (1 + r).cumprod()
        cg, md = cagr_mdd(r)
        ax.plot(eq.index, eq, lw=1.6, color=C[p], label=f"{LAB_[p]}: CAGR {cg*100:.1f}% / MDD {md*100:.1f}%")
        ax2.plot(eq.index, (eq / eq.cummax() - 1) * 100, lw=1.1, color=C[p])
    ax.set_title(f"13차 합산 장부(코스피 50 + 코스닥 50), 코스닥 A 인버스 밤·C 숏 집행 경로별, W_all {W(W_ALL)}", fontsize=11, loc="left")
    ax.set_ylabel("누적")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax2.set_ylabel("낙폭 %")
    for s_ in (ax, ax2):
        s_.grid(alpha=0.25)
        for sp in ("top", "right"):
            s_.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "fig2_book.png", dpi=130)
    plt.close(fig)


# ================================================================ 메인
def main():
    a = part_a()
    b = part_b(a)
    v2 = part_v2()
    part_v3(a)
    part_v4(a, b)
    q1x_fut = part_v5(a)
    metas = part_book(a, b, q1x_fut)
    tt = tax_table(a, v2)
    fig_legs(a, b)
    fig_book(metas)

    cells = pd.DataFrame(CELLS)[["id", "name", "market", "window", "n", "gross_bp", "net_bp", "t_nw", "delta_bp", "t_delta", "win_pct", "cagr_pct", "mdd_pct",
                                 "yearly_pos", "top5_share_pct", "half1_net_bp", "half2_net_bp", "verdict", "notes"]]
    cells.to_csv(OUT / "cells.csv", index=False, encoding="utf-8-sig", float_format="%.4f")
    pd.concat(TRADES, ignore_index=True).to_csv(OUT / "trades.csv", index=False, encoding="utf-8-sig", float_format="%.8f")

    # ---------------- 판정
    ra, rw = a["res"][("fut", "W_A")], a["res"][("fut", "W_all")]
    mo, mf, mt = metas["orig"], metas["fut"], metas["tick"]
    r1 = lambda x: round(x * 100, 1)
    nonworse = r1(mf["W_all"][0]) >= r1(mo["W_all"][0]) and r1(mf["W_all"][1]) >= r1(mo["W_all"][1])
    cond = dict(net_pos=ra["net_bp"] > 0 and rw["net_bp"] > 0, t2=ra["t_nw"] >= 2 and rw["t_nw"] >= 2, delta_pos=ra["delta_bp"] > 0 and rw["delta_bp"] > 0,
                book_nonworse=nonworse, tax_table=True)
    passed = all(cond.values())
    book_rule = (mf["W_all"][0] > mo["W_all"][0] and mf["W_all"][1] > mo["W_all"][1] and mf["W_long"][0] > mo["W_long"][0] and mf["W_long"][1] > mo["W_long"][1]
                 and mf["Y2024"][2] > mo["Y2024"][2] and mf["Y2025"][2] > mo["Y2025"][2])
    grade = "candidate" if passed else ("reject" if not cond["net_pos"] else "reference")

    # ---------------- series 저장
    fut_src = "r15_external/kis_fut_kq150_daily.csv(front·next, 08:45 시가·15:45 종가)"
    win_note = "창 차이: ETF 15:30 종가→09:00 시가 vs 선물 15:45 종가→08:45 시가(두 번 15분, V4 분해 참조)"
    n1, cg1, md1 = save_series("blog_ON_Q_fut", a["s_fut"], a["meta"], "롱 밤 ETF 5bp / 인버스 밤 선물 1.8bp",
                               f"13차 blog_ON_Q 교체본: pos=−1 밤만 코스닥150 선물 front 15:45 매도→익일 08:45 환매(1.8bp, 계약 = T+1 front), 롱 밤 원본 그대로. {win_note}. passed={passed}",
                               a["meta"]["source_files"] + ", " + fut_src, passed)
    n2, cg2, md2 = save_series("blog_ON_Q_tick", a["s_tick"], a["meta"], "ETF 5bp + 인버스 밤 251340 틱 비용(2024 13.6, 2025 14.2, 2026 21.5, 그 외 10bp)",
                               "13차 blog_ON_Q 현실 기준선: pos=−1 밤의 net 에서 251340 연도별 틱 비용 차감(gross 는 원본). 창·시각 원본 그대로.",
                               a["meta"]["source_files"], False)
    n3, cg3, md3 = save_series("jehyeong_KQ_fut", b["s_fut"], b["meta"], "롱 원본(편도 2.5bp) / 숏 선물 1.8bp×|pos|",
                               "13차 jehyeong_KQ 교체본: pos<0 일만 코스닥150 선물(2025.10.22~ KIS 1분봉 실측, 그 전 229200 TOSS 신호가 대리) 진입·청산 시각 동일, 1.8bp×|pos|. "
                               f"롱 원본 그대로. 창 차이 없음(장중). passed={passed}",
                               list(b["meta"]["source_files"]) + ["intraday/kis_minute/kq150_full_day/futures_1min.csv", "jehyeong/intraday/data/minute/229200_TOSS_1m.csv"], passed)
    n4, cg4, md4 = save_series("jehyeong_KQ_tick", b["s_tick"], b["meta"], "원 비용 + 숏일 251340 틱 비용×|pos|(2024 13.6, 2025 14.2, 2026 21.5, 그 외 10bp)",
                               "13차 jehyeong_KQ 현실 기준선: pos<0 일의 net 에서 251340 연도별 틱 비용×|pos| 차감(gross 원본).", list(b["meta"]["source_files"]), False)

    # ---------------- notes.md
    inv, tr = a["inv"], b["tr"]
    fmt = lambda r: (f"n {r['n']}, gross {r['gross_bp']:+.1f}bp, net {r['net_bp']:+.1f}bp, NW t {r['t_nw']:.2f}, 승률 {r['win_pct']:.0f}%, CAGR {r['cagr_pct']:.1f}%, "
                     f"MDD {r['mdd_pct']:.1f}%, 연도 부호 {r['yearly_pos']}, 상위5 {r['top5_share_pct']:.0f}%, 전·후반 {r['half1_net_bp']:+.1f}/{r['half2_net_bp']:+.1f}bp").replace("-", "−").replace("nan%", "해당 없음")
    cdf = cells.set_index("id")
    L = ["# 02_kq_short_fut 결과 노트 (2026.10.30)", "",
         "## 규칙 요약", "• 신호·시각·노출 13차 그대로. (a) blog_ON_Q pos=−1 밤 → 코스닥150 선물 front 15:45 매도, 익일 08:45 환매, 왕복 1.8bp. 계약 = T+1 의 front(만기 전날 밤은 차근월).",
         "• (b) jehyeong_KQ pos<0 → 거래별 진입·청산 시각의 선물 1분봉 실측(2025.10.22~), 그 전은 229200 TOSS 신호가 대리. 1.8bp×|pos|.",
         "• 틱 경로(현실 기준선) = 13차 net − 251340 틱 비용(2024 13.6, 2025 14.2, 2026 21.5, 그 외 10bp)×|pos|. 세 경로 모두 단일가 가정(ETF 09:00·15:30, 선물 08:45·15:45).",
         "• 통계 HAC(maxlags 5). Δ 는 같은 밤 짝지은 차의 상수 회귀 HAC t. 2026.10.30 행 제외.", "",
         "## 자료 점검", *NOTE, "",
         "## 주 셀 (a) 인버스 밤 세 경로",
         f"• 선물 경로 {W(W_A)}: {fmt(ra)}; Δ(선물 − 틱) {ra['delta_bp']:+.1f}bp t {ra['t_delta']:.2f}".replace("-", "−"),
         f"• 선물 경로 W_all {W(W_ALL)}: {fmt(rw)}; Δ(선물 − 틱) {rw['delta_bp']:+.1f}bp t {rw['t_delta']:.2f}".replace("-", "−"),
         f"• 틱 경로 {W(W_A)}: {fmt(a['res'][('tick', 'W_A')])}; W_all: {fmt(a['res'][('tick', 'W_all')])}",
         f"• 원 series {W(W_A)}: {fmt(a['res'][('orig', 'W_A')])}; W_all: {fmt(a['res'][('orig', 'W_all')])}",
         f"• 선물 − 원 series Δ {W(W_A)}: {cdf.loc['A_fut_vs_orig_W_A', 'delta_bp']:+.1f}bp t {cdf.loc['A_fut_vs_orig_W_A', 't_delta']:.2f}; W_all {cdf.loc['A_fut_vs_orig_W_all', 'delta_bp']:+.1f}bp t {cdf.loc['A_fut_vs_orig_W_all', 't_delta']:.2f} (창 차이 15분 두 번 포함)".replace("-", "−"),
         f"• 계약수(자본 1억): 평균 {inv.contracts.mean():.2f}계약(최소 {int(inv.contracts.min())}, 최대 {int(inv.contracts.max())}), 반올림 오차 평균 {inv.contract_err.mean()*100:+.2f}%, 평균|오차| {inv.contract_err.abs().mean()*100:.2f}%, 최대|오차| {inv.contract_err.abs().max()*100:.2f}% (명목 기준).".replace("-", "−"),
         f"• 다리 전체(롱 밤 원본 + 인버스 밤 경로별) {D(a['meta']['first_valid'])}~{D(a['meta']['last_valid'])}: 원 CAGR {a['meta']['repro']['전체_CAGR'][0]*100:.1f}% MDD {a['meta']['repro']['전체_MDD'][0]*100:.1f}% → 틱 {cg2*100:.1f}%/{md2*100:.1f}% → 선물 {cg1*100:.1f}%/{md1*100:.1f}% (교체 행 {n1}).".replace("-", "−"), "",
         "## V1 (b) jehyeong_KQ 낮 숏",
         f"• 선물 경로(실측 + 대리) {W(W_KQ)}: {fmt(b['res'][('fut', 'W_KQ')])}; Δ(선물 − 틱) {b['res'][('fut', 'W_KQ')]['delta_bp']:+.1f}bp t {b['res'][('fut', 'W_KQ')]['t_delta']:.2f}".replace("-", "−"),
         f"• 선물 경로 W_all: {fmt(b['res'][('fut', 'W_all')])}; Δ(선물 − 틱) {b['res'][('fut', 'W_all')]['delta_bp']:+.1f}bp t {b['res'][('fut', 'W_all')]['t_delta']:.2f}".replace("-", "−"),
         f"• 틱 경로 {W(W_KQ)}: {fmt(b['res'][('tick', 'W_KQ')])}", f"• 원 series {W(W_KQ)}: {fmt(b['res'][('orig', 'W_KQ')])}",
         f"• 실측만 {cdf.loc['V1_meas_only', 'window'].replace('-', '.')}: n {cdf.loc['V1_meas_only', 'n']}, net {cdf.loc['V1_meas_only', 'net_bp']:+.1f}bp, 실측 − 대리 {cdf.loc['V1_meas_minus_proxy', 'net_bp']:+.2f}bp ({cdf.loc['V1_meas_minus_proxy', 'notes'].split('|')[0].strip()}) → estimate(창 < 1년)".replace("-", "−"),
         f"• 대리 오차 상한(코스피200 선물 vs 069500, 2025.09.22~2026.10.02): n {cdf.loc['V1_err_k200_069500', 'n']}, 평균 {cdf.loc['V1_err_k200_069500', 'net_bp']:+.2f}bp, {cdf.loc['V1_err_k200_069500', 'notes']}".replace("-", "−"),
         f"• 대리 오차(코스닥150 선물 vs 229200, 2025.10.22~2026.10.07): n {cdf.loc['V1_err_kq150_229200', 'n']}, 평균 {cdf.loc['V1_err_kq150_229200', 'net_bp']:+.2f}bp, {cdf.loc['V1_err_kq150_229200', 'notes']}".replace("-", "−"),
         f"• 다리 전체 {D(b['meta']['first_valid'])}~{D(b['meta']['last_valid'])}: 원 CAGR {b['meta']['full_valid_stats']['cagr']*100:.1f}% MDD {b['meta']['full_valid_stats']['mdd']*100:.1f}% → 틱 {cg4*100:.1f}%/{md4*100:.1f}% → 선물 {cg3*100:.1f}%/{md3*100:.1f}% (교체 행 {n3}).".replace("-", "−"), "",
         "## V2~V6 (보고용, 주 셀 변경 없음)"]
    for cid, lab in (("V2_fut_W_ON", "V2 jehyeong_ON 롱 밤 미니선물(정규 선물 가격 대용) 유효 구간"), ("V2_fut_W_all", "V2 같은 셀 W_all"),
                     ("V2_orig_W_ON", "V2 원 ETF 유효 구간"), ("V3_fut2x_W_A", "V3 롱 밤 선물 2배 2016.08~"), ("V3_fut2x_W_all", "V3 롱 밤 선물 2배 W_all"),
                     ("V3_orig_W_all", "V3 원 233740 W_all"), ("V6_fut3bp_W_A", "V6 3bp 2016.08~"), ("V6_fut3bp_W_all", "V6 3bp W_all"),
                     ("V6_roll_W_A", "V6 만기 주 롤 2016.08~"), ("V6_roll_W_all", "V6 만기 주 롤 W_all")):
        r = cdf.loc[cid]
        L.append((f"• {lab}: {fmt(r)}" + (f"; Δ {r['delta_bp']:+.1f}bp t {r['t_delta']:.2f}" if pd.notna(r['delta_bp']) else "")).replace("-", "−"))
    for cid, lab in (("V3_resid233740_W_all", "V3 근거 233740 밤 잔차(전 거래일 W_all)"), ("V3_resid233740_longnights_W_all", "V3 근거 잔차(롱 밤만 W_all)"),
                     ("V4_k200_0845_0900", "V4 코스피200 선물 08:45→09:00"), ("V4_kq150_0845_0900", "V4 코스닥150 선물 08:45→09:00"),
                     ("V4_inv_window_diff", "V4 인버스 밤 창 차이(선물 경로 − ETF 창, 선물 1분봉)")):
        r = cdf.loc[cid]
        L.append(f"• {lab}: n {r['n']}, 평균 {r['net_bp']:+.2f}bp, t {r['t_nw']:.2f}, {r['notes']}".replace("-", "−"))
    for cid in ("V5_Q1x_orig_W_all", "V5_Q1x_tick_W_all", "V5_Q1x_fut_W_all", "V5_Q1x_fut_W_A"):
        r = cdf.loc[cid]
        L.append(f"• {r['name']} {r['window'].replace('-', '.')}: 활성 {r['n']}일, CAGR {r['cagr_pct']:.1f}%, MDD {r['mdd_pct']:.1f}%".replace("-", "−"))
    L += ["", "## 장부 결합(코스닥 A·C 교체, 코스피 50 + 코스닥 50)", "| 경로 | W_all CAGR/MDD | W_ex2026 | W_long(두 장부) | 2024 | 2025 | 2026(~09.17) |", "|---|---|---|---|---|---|---|"]
    for k in ("orig", "tick", "fut", "fut_A_only", "fut_C_only", "V5_Q1x_fut", "V6_fut3bp"):
        m = metas[k]
        L.append(f"| {k} | {m['W_all'][0]*100:.1f}% / {m['W_all'][1]*100:.1f}% | {m['W_ex2026'][0]*100:.1f}% / {m['W_ex2026'][1]*100:.1f}% | {m['W_long'][0]*100:.1f}% / {m['W_long'][1]*100:.1f}% | "
                 f"{m['Y2024'][2]*100:+.1f}% | {m['Y2025'][2]*100:+.1f}% | {m['Y2026'][2]*100:+.1f}% |".replace("-", "−"))
    L += ["", "## 세후 비교표(자본 1억, 원; tax_table.csv)", "• 선물: 파생 양도소득세 11%(소득세 10 + 지방 1), 기본공제 연 250만원(사람·연 단위, 전 파생 합산인데 여기서는 다리마다 따로 적용 → 낙관), 손실 이월 없음.",
          "• ETF: 069500·229200·233740 매매차익 비과세, 251340 은 과표기준가 증분 15.4%(국내 장내파생 손익 과표 제외) → 0 으로 두되 미확인.",
          "| 다리 | 연도 | 선물 세전 | 선물 세후 | ETF(틱 경로) | 세후 차(선물 − ETF) |", "|---|---|---|---|---|---|"]
    for r in tt.itertuples():
        L.append(f"| {r.leg} | {r.year} | {r.fut_pre_tax_krw:,} | {r.fut_post_tax_krw:,} | {r.etf_pre_tax_krw:,} | {r.fut_minus_etf_post_krw:,} |".replace("-", "−"))
    wf = OUT / "web_facts.md"
    L += ["", "## 자료 확보 단계(수수료·증거금·계좌 요건·세금)", *(wf.read_text(encoding="utf-8").splitlines() if wf.exists() else ["• 미확인(검색 실패) → 1.8bp/3bp 규약 유지, 세후 표 ETF 쪽 미확인."]),
          "", "## 판정(사전 등록 고정)",
          f"• 선물 경로 순수익 > 0: {cond['net_pos']} / 활성일 NW t ≥ 2(두 창): {cond['t2']} / Δ(선물 − 틱) > 0: {cond['delta_pos']} / 장부 W_all 비악화(소수 첫째 자리, 원 {r1(mo['W_all'][0])}%/{r1(mo['W_all'][1])}% → 선물 {r1(mf['W_all'][0])}%/{r1(mf['W_all'][1])}%): {cond['book_nonworse']} / 세후 표: 첨부".replace("-", "−"),
          f"• passed = {passed}, 등급 {grade}. 장부 결합 일반 규칙(W_all·W_long CAGR·MDD 동시 개선 ∧ 2024·2025 개선) = {book_rule}.",
          f"• 틱 경로(현실 기준선) 장부 W_all {r1(mt['W_all'][0])}%/{r1(mt['W_all'][1])}%, 2024 {mt['Y2024'][2]*100:+.1f}%, 2025 {mt['Y2025'][2]*100:+.1f}% → 251340 호가 비용이 장부에서 깎는 몫.".replace("-", "−"),
          f"• 돌린 셀 수 {len(cells)}(cells.csv 전부; 사전 등록 주 셀 2 + 변형 6 + 창·기준선·진단 분해).", "",
          "## 단서", "• 선물 경로는 창이 다르다(15:45→08:45). 선물 − 원 series Δ 에는 15:30↔15:45·08:45↔09:00 두 창 효과가 섞여 있고 V4 에서 분해했다(코스닥150 1분봉 구간만).",
          "• 단일가 가정: ETF 쪽 09:00·15:30 단일가, 선물 쪽 08:45·15:45 단일가 체결(슬리피지 1.8bp 에 포함). 선물 시가·종가 단일가는 유동성이 두텁지만 코스닥150 선물 일평균 거래대금은 미확인.",
          "• (b) 2025.10.22 이전은 229200 대리. 대리 오차는 위 두 분포로 상한을 잡았다. 실측 구간은 1년 미만(estimate).",
          "• 미니선물(V2)은 정규 코스피200 선물 가격으로 대용(같은 기초·만기, 승수만 다름).", "• 세금은 미확인(세후 표는 11%·250만원 공제 가정). 파생 계좌 전제(사전교육·모의거래·기본예탁금).",
          "• 틱 비용은 14차 실측 연도 평균을 거래마다 상수로 차감했다(호가 스프레드 변동 미반영).",
          f"• V1 선물 − 원 series Δ {cdf.loc['V1_fut_vs_orig_W_KQ', 'delta_bp']:+.1f}bp(t {cdf.loc['V1_fut_vs_orig_W_KQ', 't_delta']:.2f}) 가운데 약 3.8bp/건은 원 ETF 모형의 체결 모형(다음 봉 체결) 제거분이다. 선물 경로는 신호 시각 봉 가격 체결 가정 → 실거래에서는 다음 봉 체결 지연이 다시 붙는다.",
          "• V3 롱 밤 선물 2배의 Δ 에는 창 차이(15:30↔15:45·08:45↔09:00)가 섞여 있어 233740 잔차(롱 밤 −13.7bp)와 분리되지 않는다. 인버스 밤 창 차이 분해(n 34)는 선물 창이 숏에 −14.8bp(t −1.30) 로 유의하지 않다.",
          "• 상위 5건 비중이 100% 를 넘는 셀은 총 순손익이 작아서(평균 수 bp) 상위 5건 합이 총합보다 큰 경우다.",
          "• 다중 비교: 이 설계 셀은 집행 경로 비교라 새 수익원이 아니며, 다리 단독 연도 부호는 보고만 한다."]
    txt = "\n".join(l if l.startswith("|---") else l.replace("-", "−") for l in L)   # 하이픈 금지 규약(표 구분선 제외)
    assert "-" not in txt.replace("|---|", "").replace("---", "")
    (OUT / "notes.md").write_text(txt, encoding="utf-8")

    summary = dict(passed=passed, grade=grade, cond=cond, book_rule=book_rule, n_cells=len(cells),
                   main=dict(W_A={k: ra[k] for k in ("n", "gross_bp", "net_bp", "t_nw", "delta_bp", "t_delta", "win_pct", "cagr_pct", "mdd_pct", "yearly_pos", "top5_share_pct", "half1_net_bp", "half2_net_bp", "verdict")},
                             W_all={k: rw[k] for k in ("n", "gross_bp", "net_bp", "t_nw", "delta_bp", "t_delta", "win_pct", "cagr_pct", "mdd_pct", "yearly_pos", "top5_share_pct", "half1_net_bp", "half2_net_bp", "verdict")}),
                   book={k: {w: list(v) for w, v in m.items()} for k, m in metas.items() if not k.endswith("_series")},
                   series={"blog_ON_Q_fut": (n1, cg1, md1), "blog_ON_Q_tick": (n2, cg2, md2), "jehyeong_KQ_fut": (n3, cg3, md3), "jehyeong_KQ_tick": (n4, cg4, md4)},
                   contracts=dict(mean=float(inv.contracts.mean()), err_mean_pct=float(inv.contract_err.mean() * 100), err_abs_mean_pct=float(inv.contract_err.abs().mean() * 100)))
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    pd.set_option("display.width", 250, "display.max_columns", 30, "display.max_rows", 300)
    print(cells.drop(columns=["name"]).round(2).to_string(index=False))
    print("\n".join(NOTE))
    print(json.dumps(summary, ensure_ascii=False, indent=1, default=float))


if __name__ == "__main__":
    main()
