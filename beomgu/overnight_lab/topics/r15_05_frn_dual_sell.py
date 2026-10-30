"""15차 05_frn_dual_sell · 외인 현물·선물 동시 순매도 → 익일 코스피200 선물 front 숏(낮 08:45→15:45)

규칙(사전 등록 2026-10-30 고정, 결과를 본 뒤 문턱·창·비용·차량을 바꾸지 않음)
  z_spot,t = (x_t − mean_250)/std_250, 직전 250 거래일(t 제외, ddof 1), x = naver_kospi_investor_daily.csv net_foreign(원)
  z_fut,t  = 같은 식, x = naver_fut_investor_daily.csv net_foreign(코스피200 선물 외인 순매수, 계약)
  신호(T 장 마감 뒤 확정치): z_spot ≤ −0.84 ∧ z_fut ≤ −0.84
  집행: T+1 08:45 코스피200 선물 front 매도 → 15:45 환매. 주 셀 손익 = −ln(F_close,T+1 / F_open,T+1) − 0.00018 (스트레스 0.0003).
    명목 1x(증거금 21% 별도), 보고 셀 2x·3x 는 scaled(k·gross − k·cost).
  F = kis_fut_k200_daily.csv 의 front=1 행(시가 08:45 단일가, 종가 15:45 단일가). 최종거래일 규칙: 이 파일의 front 플래그는 최종거래일(만기일,
    둘째 목요일·휴장 조정) 당일부터 차근월물로 옮겨져 있다(예 2025-09-11: 101W09 front 0, 101W12 front 1). 따라서 T+1 이 만기일이면 차근월물의
    08:45→15:45 를 쓰고, 하루 안에서는 항상 같은 계약이다. 밤 보유 대조(V6b)는 T+1 의 front 계약 코드를 고정해 T+2 의 같은 코드 시가를 쓴다
    (만기 전날 → 만기일 아침은 만기 월물 그대로; 만기 월물의 만기일 행이 파일에 남아 있음).
  창: 집행일 T+1 기준 2011-09-01~2026-09-29(설계 2011-09-01~2019-12-30 / 검증 2020-01-02~2026-09-29), W_all 2022-12-13~2026-09-17(필수),
    W_ex2026 2022-12-13~2025-12-30, W_long 2016-10-04~2026-09-17 병기. 2026-10-30 행 제외. 선물 투자자별 자료가 2010-09-20 시작이라
    z_fut 는 250일 뒤부터 정의되고 첫 신호 가능일은 출력에 기록.
  통계: 거래별 순수익 bp 상수 회귀 HAC(maxlags 5) → NW t. Δ = 창 안 전 거래일(같은 방향·같은 비용)에 신호 더미 회귀 HAC t.
    CAGR = Π(1+r)^(252/일수) − 1(비활성일 0, 069500 달력), MDD = min(누적/누적최고 − 1), 연도 부호 = 이벤트 ≥ 10 인 해만,
    상위 5건 비중 = 상위 5건 순손익 합 / 총 순손익 합(총합 > 0), 전·후반 = 건수 반분 평균.
  판정(고정): 공통 규칙(순수익 > 0 ∧ NW t ≥ 2 ∧ 연도 부호 ≥ 70% ∧ 상위 5건 < 50% ∧ 전·후반 양수) ∧ 스트레스 3bp 순수익 > 0
    ∧ 13차 합산 장부(book_daily comb_a) 일수익과 |ρ| ≤ 0.2(W_all 전 거래일, 비활성일 0 포함) ∧ 결합(코스피 장부에 가산, 장부 = r_A + r_B + r_C + r_new)
    W_all·W_long CAGR·MDD 동시 개선 ∧ V6a 2026 제외 양수 → candidate. 순수익 ≤ 0 → reject, 그 밖 → reference, 창 ≤ 300 거래일 또는 이벤트 < 10 → estimate.
    StructuredOutput 의 book_rule = 위 결합 ∧ 2024·2025 연수익 개선. jehyeong_KP 롱과 같은 날 충돌은 파생 자본 별도라 합산 허용, 충돌일 수·상쇄 효과 별도 보고.
변형(보고용, 주 셀 변경 금지): V1 임계 −0.5·−1.0·−1.5(두 z 같은 임계) / V2 069500 시가→종가 대리 숏(09:00 진입, 왕복 5bp) / V3 동시 매수(z ≥ +0.84 둘 다) → 롱 /
  V4 직전 만기 이후 프로그램 차익 순매수 누적(naver_program_daily arb_net, 만기일 = expiry_calendar.csv, 만기일 당일 arb_net 은 누적에서 제외해 0 부터 재시작)의
    z250 으로 명목 가중 w = clip(1 + 0.5·z_arb,T, 0, 2), w = 0 인 날은 미집행(건수 별도) / V5 코스닥판(코스닥 외인 현물 z ≤ −0.84 ∧ 코스피200 선물 외인 z ≤ −0.84 →
    코스닥150 선물 front 숏, kis_fut_kq150_daily, 집행일 2015-11-24~2026-09-29) / V6a 2026 제외(2011-09-01~2025-12-30) / V6b 밤 보유 대조(T+1 15:45 → T+2 08:45 숏, date = T+2).
  X_flow: woohyun_ Flow 정의(외인 선물 (buy−sell)/(buy+sell), 매수·매도 계약은 2010-12-22 부터 유효)를 z250 ≤ −0.84 로 바꿔 넣은 1셀 + 신호일 겹침 대조.
  S_*: 현물 단독·선물 단독(상대 조건 없음) 숏 2셀. G_*: 3×3 격자(현물 z {≤ −0.84, 중간, ≥ +0.84} × 선물 z 동일)의 익일 front 시가→종가 평균(롱 부호, 비용 0) 진단 9셀(그림 1).
자료(원본 수정 없음): 위 파일들 + etf/069500.csv(달력·V2), naver_kosdaq_investor_daily.csv(V5), round13_slot_book/{book/book_daily.csv, series/*}(결합·ρ·충돌),
  같은 폴더 r15_leverage_desk.py 의 장부 재현 함수(import, k=1 재현 assert).
가정·단서: 선물 수수료율 미확인(1.8bp/3bp 규약), 미니 1계약 단위 무시(명목 연속), 증거금 21% 별도, 선물 양도세 11% 미반영, 08:45·15:45 단일가 체결 가정
  (ETF 09:00 가정과 달리 선물은 개장·마감 단일가라 의존 약함), log 수익을 일수익으로 복리 계산.
산출: round15_alpha_share/05_frn_dual_sell/{cells.csv, trades.csv, notes.md, fig1_zgrid.png, fig2_cum.png, fig3_yearly.png}, series/beomgu_FDS_K.{csv,meta.json}
실행: PYTHONUTF8=1 python "<code>/topics/r15_05_frn_dual_sell.py"
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
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

sys.stdout.reconfigure(encoding="utf-8")
TOPICS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOPICS))
import r15_leverage_desk as LD  # noqa: E402  13차 장부 재현(panel_wall/panel_long/books/metrics/series)

RAW = Path("<raw>")
LAB = Path("<OUT>")
OUT = LAB / "round15_alpha_share/05_frn_dual_sell"
SER = LAB / "round15_alpha_share/series"
EXPIRY = LAB / "round14_alpha_sprint/S1_expiry_program/expiry_calendar.csv"
HANDLE = "beomgu_FDS_K"

TH = -0.84
COST, STRESS, COST_ETF = 1.8e-4, 3.0e-4, 5e-4
W_FULL = ("2011-09-01", "2026-09-29")
W_DES, W_VAL = ("2011-09-01", "2019-12-30"), ("2020-01-02", "2026-09-29")
W_EX26F = ("2011-09-01", "2025-12-30")
W_ALL, W_EX26, W_LONG = LD.WIN["W_all"], LD.WIN["W_ex2026"], LD.W_LONG
W_KQ = ("2015-11-24", "2026-09-29")
CUT = pd.Timestamp("2026-10-30")  # KIS 장중 미확정 행 제외
C = {"blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a", "yellow": "#eda100", "red": "#e34948",
     "ink": "#52514e", "mid": "#f0efec", "surface": "#fcfcfb"}  # dataviz 기본 팔레트(검증 통과 슬롯·발산쌍)
CELL_COLS = ["id", "name", "market", "window", "n", "gross_bp", "net_bp", "t_nw", "delta_bp", "t_delta", "win_pct", "cagr_pct",
             "mdd_pct", "yearly_pos", "top5_share_pct", "half1_net_bp", "half2_net_bp", "verdict", "notes"]


# ---------------------------------------------------------------- 자료
def rd(p):
    return pd.read_csv(p, parse_dates=["date"], encoding="utf-8-sig")


def front_panel(path, days):
    """front=1 행의 날짜별 code/open/close/r_oc(롱 log) + 같은 계약의 다음 거래일 시가로 만든 밤 수익(index = T+1)."""
    raw = rd(path)
    raw = raw[raw.date < CUT]
    fr = raw[raw.front == 1].set_index("date").sort_index()
    assert fr.index.is_unique and (fr.open > 0).all() and (fr.close > 0).all(), path
    out = pd.DataFrame({"code": fr.code, "open": fr.open, "close": fr.close, "r_oc": np.log(fr.close / fr.open)})
    pos = days.get_indexer(out.index)
    ok = (pos >= 0) & (pos + 1 < len(days))
    nxt = pd.DatetimeIndex(days[pos[ok] + 1])
    o = raw.set_index(["date", "code"]).open
    open_next = o.reindex(pd.MultiIndex.from_arrays([nxt, out.code[ok].values])).values
    night = pd.Series(np.log(open_next / out.close[ok].values), index=nxt).dropna()
    return out, night


def z250(x):
    prev = x.shift(1)
    return (x - prev.rolling(250).mean()) / prev.rolling(250).std()


def load():
    etf = rd(RAW / "etf/069500.csv").set_index("date").sort_index()
    days = etf.index
    assert days.is_unique and days.is_monotonic_increasing
    d = {"days": days, "etf": etf}
    for key, f, col in [("ks", "naver_kospi_investor_daily.csv", "net_foreign"), ("fu", "naver_fut_investor_daily.csv", "net_foreign"),
                        ("kd", "naver_kosdaq_investor_daily.csv", "net_foreign"), ("pg", "naver_program_daily.csv", "arb_net")]:
        df = rd(RAW / f).set_index("date").sort_index()
        assert df.index.is_unique and df[col].notna().all(), f
        assert df.index.difference(days).empty, f"{f}: 달력 밖 날짜"
        lo = max(df.index.min(), pd.Timestamp("2010-09-20"))
        gap = days[(days >= lo) & (days <= df.index.max())].difference(df.index)
        assert gap.empty, f"{f}: 달력 결측 {list(gap[:5])}"
        d[key] = df
    d["kf"], d["night_k"] = front_panel(RAW / "r15_external/kis_fut_k200_daily.csv", days)
    d["kq"], d["night_q"] = front_panel(RAW / "r15_external/kis_fut_kq150_daily.csv", days)
    for key, lo in (("kf", W_FULL[0]), ("kq", W_KQ[0])):
        gap = days[(days >= lo) & (days <= W_FULL[1])].difference(d[key].index)
        assert gap.empty, f"{key}: front 결측 {list(gap[:5])}"
    d["expiry"] = pd.read_csv(EXPIRY, parse_dates=["expiry"]).expiry.sort_values().reset_index(drop=True)
    return d


# ---------------------------------------------------------------- 거래·통계
def make_trades(sig, ret, days, side, cost, lag=1, w=None):
    """sig: 달력 index 의 bool(신호일 T). 집행일 = T 의 lag 거래일 뒤. ret: 집행일 index 의 롱 log 수익. side −1 숏 / +1 롱. w: 명목 가중(T index)."""
    T = sig.index[sig.fillna(False).values]
    i = days.get_indexer(T)
    assert (i >= 0).all()
    ok = i + lag < len(days)
    T, T1 = T[ok], pd.DatetimeIndex(days[i[ok] + lag])
    g = side * ret.reindex(T1).values
    ww = np.ones(len(T)) if w is None else w.reindex(T).values
    tr = pd.DataFrame({"T": T, "T1": T1, "w": ww, "gross": ww * g, "net": ww * g - ww * cost})
    n_nodata = int(tr.gross.isna().sum())
    tr = tr.dropna(subset=["gross"])
    n_w0 = int((tr.w <= 0).sum())
    return tr[tr.w > 0].sort_values("T1").reset_index(drop=True), n_nodata, n_w0


def hac_t(y):
    y = np.asarray(y, float)
    if len(y) < 3 or np.allclose(y, y[0]):
        return np.nan
    return float(sm.OLS(y, np.ones((len(y), 1))).fit(cov_type="HAC", cov_kwds={"maxlags": 5}).tvalues[0])


def delta_hac(all_net, sig_days):
    s = all_net.dropna()
    dmy = s.index.isin(sig_days).astype(float)
    n_ctrl = int((1 - dmy).sum())
    if dmy.sum() < 3 or n_ctrl < 3:
        return np.nan, np.nan, n_ctrl
    f = sm.OLS(s.values * 1e4, sm.add_constant(dmy)).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
    return float(f.params[1]), float(f.tvalues[1]), n_ctrl


def cell(cid, name, market, win, tr, all_net, days, note="", verdict=None):
    t = tr[(tr.T1 >= win[0]) & (tr.T1 <= win[1])].sort_values("T1")
    cal = days[(days >= win[0]) & (days <= win[1])]
    n = len(t)
    daily = pd.Series(0.0, index=cal)
    if n:
        daily.loc[t.T1.values] = t.net.values
    m = LD.metrics(daily)
    gross_bp, net_bp = t.gross.mean() * 1e4, t.net.mean() * 1e4
    t_nw = hac_t(t.net * 1e4)
    dbp, dt, n_ctrl = delta_hac(all_net.loc[win[0]:win[1]], t.T1) if all_net is not None else (np.nan, np.nan, 0)
    yr = t.groupby(t.T1.dt.year).net.agg(["sum", "count"])
    yr10 = yr[yr["count"] >= 10]
    ypos, ytot = int((yr10["sum"] > 0).sum()), len(yr10)
    ydet = " ".join(f"{y}:{'+' if r['sum'] > 0 else '−'}{abs(r['sum']) * 1e4:.0f}bp({int(r['count'])})" for y, r in yr.iterrows())
    tot = t.net.sum()
    top5 = t.net.nlargest(5).sum() / tot * 100 if tot > 0 else np.nan
    h = n // 2
    h1 = t.net.iloc[:h].mean() * 1e4 if h else np.nan
    h2 = t.net.iloc[h:].mean() * 1e4 if n - h else np.nan
    if verdict is None:
        if len(cal) <= 300 or n < 10:
            verdict = "estimate"
        elif net_bp <= 0:
            verdict = "reject"
        elif t_nw >= 2 and ytot > 0 and ypos / ytot >= 0.7 and top5 < 50 and h1 > 0 and h2 > 0:
            verdict = "candidate"
        else:
            verdict = "reference"
    row = dict(id=cid, name=name, market=market, window=f"{win[0]}~{win[1]}", n=n, gross_bp=gross_bp, net_bp=net_bp, t_nw=t_nw,
               delta_bp=dbp, t_delta=dt, win_pct=float((t.net > 0).mean() * 100) if n else np.nan, cagr_pct=m["cagr"] * 100,
               mdd_pct=m["mdd"] * 100, yearly_pos=f"{ypos}/{ytot}", top5_share_pct=top5, half1_net_bp=h1, half2_net_bp=h2,
               verdict=verdict, notes=f"{note} 연도별(순손익 합, 건수) {ydet}; 비신호일 n {n_ctrl}; 창 거래일 {len(cal)}; 총 순손익 {tot * 100:+.2f}%".strip())
    return row, t, daily


def py(v):
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        return float(v)
    return v


def book_row(cid, name, market, wn, w, r):
    rr = LD.win(r, w)
    m = LD.metrics(rr)
    return dict(id=cid, name=name, market=market, window=f"{w[0]}~{w[1]}", n=m["n_days"], gross_bp=np.nan, net_bp=float(rr.mean() * 1e4),
                t_nw=hac_t(rr.values * 1e4), delta_bp=np.nan, t_delta=np.nan, win_pct=np.nan, cagr_pct=m["cagr"] * 100, mdd_pct=m["mdd"] * 100,
                yearly_pos="", top5_share_pct=np.nan, half1_net_bp=np.nan, half2_net_bp=np.nan, verdict="diagnostic",
                notes=f"{wn} 누적 {m['cum'] * 100:+.2f}% 샤프 {m['sharpe']:.2f} 최악일 {m['worst_day'] * 100:.2f}%({m['worst_day_date']})"), m


# ---------------------------------------------------------------- 메인
def main():
    plt.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    OUT.mkdir(parents=True, exist_ok=True)
    SER.mkdir(parents=True, exist_ok=True)
    d = load()
    days, etf, ks, fu, kd, pg, kf, kq, night_k, night_q = (d[k] for k in ("days", "etf", "ks", "fu", "kd", "pg", "kf", "kq", "night_k", "night_q"))

    # --- 신호 변수
    flow_raw = fu.buyq_foreign - fu.sellq_foreign
    flow = (flow_raw / (fu.buyq_foreign + fu.sellq_foreign)).where((fu.buyq_foreign > 0) & (fu.sellq_foreign > 0))
    exp = d["expiry"]
    gid = np.searchsorted(exp.values, pg.index.values, side="right") - 1  # 직전 만기(≤ t) 번호
    is_exp = pg.index.isin(exp)
    arb_eff = pg.arb_net.where(~is_exp, 0.0)  # 만기일 당일 = 청산분 → 누적 제외
    cum_arb = arb_eff.groupby(gid).cumsum()
    Z = pd.DataFrame({"z_spot": z250(ks.net_foreign), "z_fut": z250(fu.net_foreign), "z_kq": z250(kd.net_foreign),
                      "z_flow": z250(flow), "z_arb": z250(cum_arb)}).reindex(days)
    first_sig = Z.dropna(subset=["z_spot", "z_fut"]).index.min()
    last_sig = min(ks.index.max(), fu.index.max())
    w_arb = (1 + 0.5 * Z.z_arb).clip(0, 2)
    chk = (ks.net_foreign.iloc[300] - ks.net_foreign.iloc[50:300].mean()) / ks.net_foreign.iloc[50:300].std()
    assert abs(chk - Z.z_spot.loc[ks.index[300]]) < 1e-9  # z 정의 수기 대조(t 제외 직전 250일)

    # --- 수익 계열(롱 log) 과 '전 거래일' 비교군
    r_k, r_q = kf.r_oc, kq.r_oc
    r_etf = np.log(etf.close / etf.open)
    all_k = -r_k - COST
    sig = {"main": (Z.z_spot <= TH) & (Z.z_fut <= TH),
           "buy": (Z.z_spot >= -TH) & (Z.z_fut >= -TH),
           "spot_any": Z.z_spot <= TH, "fut_any": Z.z_fut <= TH,
           "kq": (Z.z_kq <= TH) & (Z.z_fut <= TH),
           "flow": (Z.z_spot <= TH) & (Z.z_flow <= TH)}
    for th in (-0.5, -1.0, -1.5):
        sig[f"th{th}"] = (Z.z_spot <= th) & (Z.z_fut <= th)

    def enrich(tr, z1, z2, panel, night=None):
        tr = tr.copy()
        tr["z1"], tr["z2"] = Z[z1].reindex(tr["T"]).values, Z[z2].reindex(tr["T"]).values
        if night is None:
            p = panel.reindex(tr.T1)
            tr["code"], tr["entry"], tr["exit"] = p.code.values, p.open.values, p.close.values
        else:  # T1 = T+2(청산일), 진입 = T+1 종가(같은 계약), 청산 = T+2 시가
            t1 = pd.DatetimeIndex(days[days.get_indexer(tr.T1) - 1])
            p = panel.reindex(t1)
            tr["code"], tr["entry"] = p.code.values, p.close.values
            tr["exit"] = tr.entry * np.exp(-tr.gross / tr.w)
        return tr

    tr_main, nodata, _ = make_trades(sig["main"], r_k, days, -1, COST)
    tr_main = enrich(tr_main, "z_spot", "z_fut", kf)
    assert (tr_main.T1 > tr_main["T"]).all() and tr_main.T1.is_unique and nodata == 0
    print(f"신호 가능일 {first_sig.date()}~{last_sig.date()}, 주 셀 신호(전 기간) n {len(tr_main)}, 집행 2011-09-01 이후 n {(tr_main.T1 >= W_FULL[0]).sum()}, 마지막 집행일 {tr_main.T1.max().date()}")

    rows, trades, dailies = [], [], {}

    def add(cid, name, market, win, tr, all_net, note="", verdict=None, keep_trades=True):
        row, t, daily = cell(cid, name, market, win, tr, all_net, days, note, verdict)
        rows.append(row)
        dailies[cid] = daily
        if keep_trades and len(t):
            trades.append(t.assign(cell=cid))
        print(f"  {cid:<14} n {row['n']:>4} gross {row['gross_bp']:+7.2f} net {row['net_bp']:+7.2f} t {row['t_nw']:+5.2f} Δ {row['delta_bp']:+7.2f}(t {row['t_delta']:+5.2f}) "
              f"win {row['win_pct']:5.1f} CAGR {row['cagr_pct']:+6.2f} MDD {row['mdd_pct']:+6.2f} yr {row['yearly_pos']:>5} top5 {row['top5_share_pct']:6.1f} "
              f"h1 {row['half1_net_bp']:+6.1f} h2 {row['half2_net_bp']:+6.1f} → {row['verdict']}")
        return row

    print("\n[주 셀 M1 과 창·비용·레버리지]")
    mname = "동시 z ≤ −0.84 → T+1 08:45→15:45 K200 선물 front 숏 1.8bp 1x"
    m1 = add("M1", mname, "KOSPI", W_FULL, tr_main, all_k, "주 셀. 선물 front 단일가 체결 가정(08:45·15:45).")
    add("M1_design", mname + " (설계창)", "KOSPI", W_DES, tr_main, all_k, "설계 분할.")
    add("M1_valid", mname + " (검증창)", "KOSPI", W_VAL, tr_main, all_k, "검증 분할.")
    add("M1_Wall", mname + " (W_all)", "KOSPI", W_ALL, tr_main, all_k, "W_all.")
    add("M1_Wex2026", mname + " (W_ex2026)", "KOSPI", W_EX26, tr_main, all_k, "W_ex2026.")
    add("M1_Wlong", mname + " (W_long)", "KOSPI", W_LONG, tr_main, all_k, "W_long.")
    tr_stress = tr_main.assign(net=tr_main.gross - STRESS)
    ms = add("M1_stress", mname.replace("1.8bp", "3bp 스트레스"), "KOSPI", W_FULL, tr_stress, -r_k - STRESS, "스트레스 비용 3bp.", keep_trades=False)
    lev = {}
    for k in (2, 3):
        trk = tr_main.assign(gross=tr_main.gross * k, net=tr_main.net * k)
        lev[k] = add(f"M1_{k}x", mname.replace("1x", f"{k}x scaled"), "KOSPI", W_FULL, trk, (-r_k - COST) * k, f"보고 셀 {k}x(비용 {k}배, 증거금 21%×{k} 별도).", keep_trades=False)

    print("\n[변형]")
    for th in (-0.5, -1.0, -1.5):
        t, _, _ = make_trades(sig[f"th{th}"], r_k, days, -1, COST)
        add(f"V1_th{abs(th) * 10:02.0f}", f"V1 임계 −{abs(th)} 둘 다 → 선물 숏 1.8bp", "KOSPI", W_FULL, enrich(t, "z_spot", "z_fut", kf), all_k, "V1 임계 변형.")
    t, _, _ = make_trades(sig["main"], r_etf, days, -1, COST_ETF)
    t = enrich(t, "z_spot", "z_fut", etf.assign(code="069500"))
    v2 = add("V2_etf", "V2 069500 09:00 시가→15:30 종가 대리 숏 5bp", "KOSPI", W_FULL, t, -r_etf - COST_ETF, "V2 ETF 대리(단일가 가정, 숏 집행 불가라 가격 경로 대리).")
    t, _, _ = make_trades(sig["buy"], r_k, days, +1, COST)
    v3 = add("V3_dualbuy", "V3 동시 매수 z ≥ +0.84 둘 다 → 선물 롱 1.8bp", "KOSPI", W_FULL, enrich(t, "z_spot", "z_fut", kf), r_k - COST, "V3 대칭 롱(보고).")
    t, _, n_w0 = make_trades(sig["main"], r_k, days, -1, COST, w=w_arb)
    v4 = add("V4_arbz", "V4 직전 만기 이후 차익 순매수 누적 z 가중 숏(w = clip(1+0.5z, 0, 2))", "KOSPI", W_FULL, enrich(t, "z_spot", "z_arb", kf), None,
             f"V4 가중(z2 열 = z_arb). w = 0 미집행 {n_w0}건, 평균 w {t.w.mean():.2f}; Δ 는 가중이라 생략.")
    t, _, _ = make_trades(sig["kq"], r_q, days, -1, COST)
    v5 = add("V5_kosdaq", "V5 코스닥 외인 현물 z ≤ −0.84 ∧ K200 선물 외인 z ≤ −0.84 → KQ150 선물 front 숏 1.8bp", "KOSDAQ", W_KQ, enrich(t, "z_kq", "z_fut", kq), -r_q - COST,
             "V5 코스닥판(z1 = 코스닥 현물 z).")
    v6a = add("V6a_ex2026", mname + " (2026 제외)", "KOSPI", W_EX26F, tr_main, all_k, "V6a 2026 제외 민감도.", keep_trades=False)
    t, _, _ = make_trades(sig["main"], night_k, days, -1, COST, lag=2)
    v6b = add("V6b_night", "V6b 밤 보유 대조 T+1 15:45 → T+2 08:45 선물 숏 1.8bp(같은 계약)", "KOSPI", W_FULL, enrich(t, "z_spot", "z_fut", kf, night=True), -night_k - COST,
              "V6b date = T+2 청산일.")
    t, _, _ = make_trades(sig["flow"], r_k, days, -1, COST)
    xf = add("X_flow", "X woohyun_ Flow 비율 z250 ≤ −0.84 ∧ 현물 z ≤ −0.84 → 선물 숏 1.8bp", "KOSPI", W_FULL, enrich(t, "z_spot", "z_flow", kf), all_k, "X 변수 겹침 대조(z2 = z_flow).")
    for key, nm in (("spot_any", "S 현물 단독 z_spot ≤ −0.84(선물 조건 없음) → 선물 숏 1.8bp"), ("fut_any", "S 선물 단독 z_fut ≤ −0.84(현물 조건 없음) → 선물 숏 1.8bp")):
        t, _, _ = make_trades(sig[key], r_k, days, -1, COST)
        add(f"S_{key}", nm, "KOSPI", W_FULL, enrich(t, "z_spot", "z_fut", kf), all_k, "단독 조건(동시 조건의 증분 대조).", keep_trades=False)

    # 변수 겹침: 주 신호 vs Flow 신호
    dom = Z.dropna(subset=["z_spot", "z_fut", "z_flow"]).loc[:last_sig]
    A, B = sig["main"].reindex(dom.index).fillna(False), sig["flow"].reindex(dom.index).fillna(False)
    overlap = dict(n_dom=len(dom), nA=int(A.sum()), nB=int(B.sum()), nAB=int((A & B).sum()), jaccard=float((A & B).sum() / max(1, (A | B).sum())),
                   rho_z=float(dom.z_fut.corr(dom.z_flow)), rho_raw=float(fu.net_foreign.corr(flow)))
    print(f"  겹침: 공통 정의역 {overlap['n_dom']}일, 주 신호 {overlap['nA']}, Flow 신호 {overlap['nB']}, 교집합 {overlap['nAB']}, Jaccard {overlap['jaccard']:.3f}, "
          f"ρ(z_fut, z_flow) {overlap['rho_z']:.3f}, ρ(순매수 계약, Flow 비율) {overlap['rho_raw']:.3f}")

    print("\n[G 3×3 격자: 익일 front 시가→종가 평균(롱 부호, 비용 0), 집행 2011-09-01~2026-09-29]")
    cat = lambda z: np.where(z <= TH, "sell", np.where(z >= -TH, "buy", "mid"))
    cats = pd.DataFrame({"s": cat(Z.z_spot), "f": cat(Z.z_fut)}, index=days).where(Z[["z_spot", "z_fut"]].notna().all(axis=1))
    grid = {}
    for sc in ("sell", "mid", "buy"):
        for fc in ("sell", "mid", "buy"):
            t, _, _ = make_trades((cats.s == sc) & (cats.f == fc), r_k, days, +1, 0.0)
            row = add(f"G_spot{sc}_fut{fc}", f"G 격자 현물 {sc} × 선물 {fc} 익일 롱 평균(비용 0)", "KOSPI", W_FULL, t, r_k, "격자 진단(그림 1).", verdict="diagnostic", keep_trades=False)
            grid[(sc, fc)] = row

    # --- 13차 장부 결합·상관·충돌
    print("\n[장부 결합]")
    daily_full = dailies["M1"]
    legs_wall = LD.panel_wall()
    base = LD.books(legs_wall, {}, "fixed")
    book = pd.read_csv(LD.R13 / "book/book_daily.csv", parse_dates=["date"]).set_index("date")
    assert np.allclose(base["comb"], book.comb_a)
    bm = LD.metrics(base["comb"])
    assert abs(bm["cagr"] - 0.314179) < 0.002 and abs(bm["mdd"] + 0.061046) < 0.001, bm
    r_new = daily_full.reindex(book.index).fillna(0.0)
    kospi_new = base["KOSPI"] + r_new
    comb_new = 0.5 * (kospi_new + base["KOSDAQ"])
    legs_long = LD.panel_long(["KOSPI_B_beomgu_C3VT30", "KOSDAQ_A_blog_ON_Q", "KOSDAQ_B_beomgu_C3Q"], W_LONG)
    bl = LD.books(legs_long, {}, "fixed")
    lm = LD.metrics(bl["comb"])
    assert lm["n_days"] == 2444 and abs(lm["cagr"] - 0.150369) < 0.002 and abs(lm["mdd"] + 0.096107) < 0.001, lm
    r_new_l = daily_full.reindex(bl["comb"].index).fillna(0.0)
    kospi_long_new = bl["KOSPI"] + r_new_l
    comb_long_new = 0.5 * (kospi_long_new + bl["KOSDAQ"])
    print(f"  재현 k=1: W_all comb CAGR {bm['cagr']:.4f} MDD {bm['mdd']:.4f} | W_long comb CAGR {lm['cagr']:.4f} MDD {lm['mdd']:.4f} n {lm['n_days']}")
    bk = {}
    for wn, w, rb, rn, mk in [("W_all", W_ALL, base["comb"], comb_new, "KOSPI+KOSDAQ"), ("W_ex2026", W_EX26, base["comb"], comb_new, "KOSPI+KOSDAQ"),
                              ("Y2024", LD.WIN["Y2024"], base["comb"], comb_new, "KOSPI+KOSDAQ"), ("Y2025", LD.WIN["Y2025"], base["comb"], comb_new, "KOSPI+KOSDAQ"),
                              ("Y2026", LD.WIN["Y2026"], base["comb"], comb_new, "KOSPI+KOSDAQ"), ("W_long", W_LONG, bl["comb"], comb_long_new, "KOSPI+KOSDAQ"),
                              ("W_all_KOSPI", W_ALL, base["KOSPI"], kospi_new, "KOSPI"), ("W_long_KOSPI", W_LONG, bl["KOSPI"], kospi_long_new, "KOSPI")]:
        for tag, r in (("base", rb), ("FDS", rn)):
            row, m = book_row(f"B_{wn}_{tag}", f"13차 장부 {wn} {'원 장부' if tag == 'base' else '+ beomgu_FDS_K(코스피 장부 가산)'}", mk, wn, w, r)
            rows.append(row)
            bk[(wn, tag)] = m
        b, n_ = bk[(wn, "base")], bk[(wn, "FDS")]
        print(f"  {wn:<13} 원 CAGR {b['cagr'] * 100:6.2f} MDD {b['mdd'] * 100:6.2f} 누적 {b['cum'] * 100:7.2f} 샤프 {b['sharpe']:.2f} | +FDS CAGR {n_['cagr'] * 100:6.2f} MDD {n_['mdd'] * 100:6.2f} 누적 {n_['cum'] * 100:7.2f} 샤프 {n_['sharpe']:.2f}")
    act = r_new != 0
    rho_all = float(np.corrcoef(r_new, book.comb_a)[0, 1])
    rho_act = float(np.corrcoef(r_new[act], book.comb_a[act])[0, 1]) if act.sum() > 2 else np.nan
    rho_kospi = float(np.corrcoef(r_new, book.KOSPI_book)[0, 1])
    kp = LD.series("jehyeong_KP").reindex(book.index).fillna(0.0)
    clash = (kp.pos > 0) & act
    cl = dict(n=int(clash.sum()), n_act=int(act.sum()), n_kp=int((kp.pos > 0).sum()), kp_sum=float(kp.net[clash].sum()), fds_sum=float(r_new[clash].sum()),
              fds_mean_clash=float(r_new[clash].mean() * 1e4) if clash.any() else np.nan, fds_mean_noclash=float(r_new[act & ~clash].mean() * 1e4) if (act & ~clash).any() else np.nan)
    rows.append(dict(id="C_rho", name="ρ(beomgu_FDS_K 일수익, 13차 comb_a) W_all 전 거래일", market="KOSPI+KOSDAQ", window=f"{W_ALL[0]}~{W_ALL[1]}", n=len(r_new),
                     gross_bp=np.nan, net_bp=float(r_new.mean() * 1e4), t_nw=hac_t(r_new.values * 1e4), delta_bp=np.nan, t_delta=np.nan, win_pct=np.nan,
                     cagr_pct=np.nan, mdd_pct=np.nan, yearly_pos="", top5_share_pct=np.nan, half1_net_bp=np.nan, half2_net_bp=np.nan, verdict="diagnostic",
                     notes=f"ρ 전일 {rho_all:.4f}, 활성일만(n {int(act.sum())}) {rho_act:.4f}, 코스피 장부와 {rho_kospi:.4f}; net_bp = FDS 일평균(비활성 0 포함)"))
    rows.append(dict(id="C_KP_clash", name="jehyeong_KP 롱 활성일과 FDS 숏 활성일 충돌(W_all)", market="KOSPI", window=f"{W_ALL[0]}~{W_ALL[1]}", n=cl["n"],
                     gross_bp=np.nan, net_bp=cl["fds_mean_clash"], t_nw=np.nan, delta_bp=cl["fds_mean_clash"] - cl["fds_mean_noclash"] if cl["n"] else np.nan, t_delta=np.nan,
                     win_pct=np.nan, cagr_pct=np.nan, mdd_pct=np.nan, yearly_pos="", top5_share_pct=np.nan, half1_net_bp=np.nan, half2_net_bp=np.nan, verdict="diagnostic",
                     notes=f"충돌일 {cl['n']} / FDS 활성 {cl['n_act']} / KP 롱 {cl['n_kp']}; 충돌일 KP 순손익 합 {cl['kp_sum'] * 100:+.2f}% FDS 합 {cl['fds_sum'] * 100:+.2f}% 합산 {(cl['kp_sum'] + cl['fds_sum']) * 100:+.2f}%; "
                           f"net_bp = 충돌일 FDS 평균, delta_bp = 충돌일 − 비충돌 활성일 평균({cl['fds_mean_noclash']:.2f}bp)"))
    print(f"  ρ(FDS, comb_a) 전일 {rho_all:.4f} 활성일 {rho_act:.4f} 코스피 장부 {rho_kospi:.4f}; KP 충돌 {cl}")

    # --- 판정
    wall_ok = bk[("W_all", "FDS")]["cagr"] > bk[("W_all", "base")]["cagr"] and bk[("W_all", "FDS")]["mdd"] > bk[("W_all", "base")]["mdd"]
    wlong_ok = bk[("W_long", "FDS")]["cagr"] > bk[("W_long", "base")]["cagr"] and bk[("W_long", "FDS")]["mdd"] > bk[("W_long", "base")]["mdd"]
    y24_ok = bk[("Y2024", "FDS")]["cum"] > bk[("Y2024", "base")]["cum"]
    y25_ok = bk[("Y2025", "FDS")]["cum"] > bk[("Y2025", "base")]["cum"]
    common_ok = m1["verdict"] == "candidate"
    stress_ok, rho_ok, ex26_ok = ms["net_bp"] > 0, abs(rho_all) <= 0.2, v6a["net_bp"] > 0
    book05 = wall_ok and wlong_ok
    passed = common_ok and stress_ok and rho_ok and book05 and ex26_ok
    grade = "candidate" if passed else ("reject" if m1["net_bp"] <= 0 else ("estimate" if m1["verdict"] == "estimate" else "reference"))
    J = dict(common=common_ok, stress=stress_ok, rho=rho_ok, wall=wall_ok, wlong=wlong_ok, y2024=y24_ok, y2025=y25_ok, ex2026=ex26_ok,
             book05=book05, book_rule=book05 and y24_ok and y25_ok, passed=passed, grade=grade)
    print("\n[판정]", J)

    # --- 저장: cells, trades, series, meta
    cells = pd.DataFrame(rows)[CELL_COLS]
    cells.to_csv(OUT / "cells.csv", index=False, encoding="utf-8-sig", float_format="%.6f")
    tdf = pd.concat(trades, ignore_index=True)[["cell", "T", "T1", "code", "entry", "exit", "z1", "z2", "w", "gross", "net"]]
    tdf.to_csv(OUT / "trades.csv", index=False, encoding="utf-8-sig", float_format="%.8g")
    last_valid = days[days.get_indexer([last_sig])[0] + 1]
    sidx = days[(days >= W_FULL[0]) & (days <= last_valid)]
    ser = pd.DataFrame({"pos": 0.0, "gross": 0.0, "net": 0.0}, index=sidx)
    tm = tr_main[(tr_main.T1 >= sidx[0]) & (tr_main.T1 <= sidx[-1])]
    ser.loc[tm.T1.values, ["pos", "gross", "net"]] = np.column_stack([-np.ones(len(tm)), tm.gross.values, tm.net.values])
    ser.to_csv(SER / f"{HANDLE}.csv", index_label="date", float_format="%.10g")
    chk = pd.read_csv(SER / f"{HANDLE}.csv", parse_dates=["date"]).set_index("date")
    sm_ = LD.metrics(chk.net)
    ex_ = LD.metrics(daily_full.loc[sidx[0]:sidx[-1]])  # 같은 달력(마지막 신호일 다음 거래일까지)의 주 셀 일수익
    assert abs(sm_["cagr"] - ex_["cagr"]) < 1e-9 and abs(sm_["mdd"] - ex_["mdd"]) < 1e-9 and sm_["n_active"] == ex_["n_active"] == m1["n"], (sm_, ex_)
    assert abs(sm_["mdd"] - m1["mdd_pct"] / 100) < 1e-9
    meta = {"handle": HANDLE, "slot": "C_intraday_short", "market": "KOSPI", "instrument": "코스피200 선물 front 숏 (kis_fut_k200_daily front=1, 명목 1x, 증거금 21% 별도)",
            "timing": "intraday_08:45→15:45 (date = 집행일 T+1, 신호 = T 장 마감 뒤 확정치)", "cost_bp": 1.8, "direction": "short_only",
            "first_valid": str(sidx[0].date()), "last_valid": str(sidx[-1].date()), "n_days": int(len(ser)), "n_active": int((ser.pos != 0).sum()),
            "source_files": [str(RAW / "naver_kospi_investor_daily.csv"), str(RAW / "naver_fut_investor_daily.csv"), str(RAW / "r15_external/kis_fut_k200_daily.csv"),
                             str(RAW / "etf/069500.csv"), str(TOPICS / "r15_05_frn_dual_sell.py")],
            "repro": {"expected": {"cagr": ex_["cagr"], "mdd": ex_["mdd"], "days": ex_["n_days"], "trades": ex_["n_active"]},
                      "got": {"cagr": sm_["cagr"], "mdd": sm_["mdd"], "days": sm_["n_days"], "trades": sm_["n_active"]}},
            "full_valid_stats": {k: py(v) for k, v in sm_.items()}, "grade": grade, "judging": {k: py(v) for k, v in J.items()},
            "notes": "규칙: T 확정치 z_spot(코스피 외인 순매수 원, 250일 t 제외) ≤ −0.84 ∧ z_fut(코스피200 선물 외인 순매수 계약, 250일) ≤ −0.84 → T+1 08:45 front 매도 → 15:45 환매, "
                     "gross = −ln(close/open), net = gross − 1.8bp, pos = −1(명목 1x). 만기일은 차근월물(front 플래그 규칙). 첫 신호 가능일 " + str(first_sig.date()) +
                     ", 마지막 신호일 " + str(last_sig.date()) + "(코스피 투자자별 자료 끝). 사전 등록 판정 " + grade + ". 15차 05_frn_dual_sell.",
            "metric_defs": "누적=Π(1+r); CAGR=누적^(252/일수)-1; 연변동성=std(ddof=1)·√252(0 포함); 샤프=mean·252/vol; MDD=min(누적/누적최고-1); 승률=pos≠0 일 중 net>0 비율"}
    (SER / f"{HANDLE}.meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")

    # --- 그림
    fig1(grid)
    fig2(tr_main, tr_stress)
    fig3(tr_main)
    # --- notes
    write_notes(rows, m1, ms, lev, v2, v3, v4, v5, v6a, v6b, xf, overlap, grid, bk, rho_all, rho_act, rho_kospi, cl, J, first_sig, last_sig, len(rows), d, tr_main, nodata, n_w0)
    print(f"\n셀 수 {len(rows)}, 거래 행 {len(tdf)}, 산출 {OUT}")
    return J


# ---------------------------------------------------------------- 그림
def _ax_clean(ax):
    ax.grid(alpha=0.25)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def fig1(grid):
    cats, lab = ["sell", "mid", "buy"], ["순매도 z ≤ −0.84", "중간", "순매수 z ≥ +0.84"]
    M = np.array([[grid[(s, f)]["gross_bp"] for f in cats] for s in cats])
    v = np.nanmax(np.abs(M))
    fig, ax = plt.subplots(figsize=(7.6, 5.8))
    cmap = LinearSegmentedColormap.from_list("div", [C["blue"], C["mid"], C["red"]])
    im = ax.imshow(M, cmap=cmap, norm=TwoSlopeNorm(vcenter=0, vmin=-v, vmax=v))
    for i, s in enumerate(cats):
        for j, f in enumerate(cats):
            g = grid[(s, f)]
            dark = abs(M[i, j]) > 0.55 * v
            ax.text(j, i, f"{M[i, j]:+.1f}bp\nn {g['n']} · t {g['t_nw']:+.2f}", ha="center", va="center", fontsize=10, color="#ffffff" if dark else C["ink"])
    ax.set_xticks(range(3), lab)
    ax.set_yticks(range(3), lab)
    ax.set_xlabel("선물 외인 순매수 z (T, 250일)")
    ax.set_ylabel("현물 외인 순매수 z (T, 250일)")
    ax.set_title("T 외인 수급 조합별 T+1 코스피200 선물 front 08:45→15:45 평균 수익(롱 부호, 비용 0)\n집행일 2011.9.1~2026.9.29, 주 셀 = 왼쪽 위(동시 매도) 의 숏", fontsize=10, loc="left")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("bp")
    fig.tight_layout()
    fig.savefig(OUT / "fig1_zgrid.png", dpi=130)
    plt.close(fig)


def fig2(tr, trs):
    fig, ax = plt.subplots(figsize=(10, 5.2))
    for t, col, ls, lab in ((tr, C["blue"], "-", "순손익 누적 1x, 1.8bp"), (trs, C["orange"], "--", "스트레스 3bp")):
        t = t[(t.T1 >= W_FULL[0]) & (t.T1 <= W_FULL[1])]
        y = t.net.cumsum() * 100
        ax.plot(t.T1, y, lw=2, color=col, ls=ls, label=lab)
        ax.annotate(f"{y.iloc[-1]:+.1f}%", (t.T1.iloc[-1], y.iloc[-1]), xytext=(5, 0), textcoords="offset points", color=C["ink"], fontsize=9, va="center")
    ax.axvspan(pd.Timestamp(W_VAL[0]), pd.Timestamp(W_FULL[1]), color="#000000", alpha=0.04)
    ax.axvline(pd.Timestamp(W_ALL[0]), color=C["ink"], lw=1, ls=":")
    ax.annotate("검증창 2020~", (pd.Timestamp(W_VAL[0]), ax.get_ylim()[1]), xytext=(4, -12), textcoords="offset points", fontsize=9, color=C["ink"])
    ax.annotate("W_all 시작 2022.12.13", (pd.Timestamp(W_ALL[0]), ax.get_ylim()[0]), xytext=(4, 6), textcoords="offset points", fontsize=9, color=C["ink"])
    ax.axhline(0, color=C["ink"], lw=0.8)
    ax.set_ylabel("누적 순손익 % (명목 1x, 단리 합)")
    ax.set_title("05 동시 순매도 → 익일 선물 숏: 거래별 순손익 누적 (집행일 2011.9.1~2026.9.29)", fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    _ax_clean(ax)
    fig.tight_layout()
    fig.savefig(OUT / "fig2_cum.png", dpi=130)
    plt.close(fig)


def fig3(tr):
    t = tr[(tr.T1 >= W_FULL[0]) & (tr.T1 <= W_FULL[1])]
    g = t.groupby(t.T1.dt.year).net.agg(["sum", "count"])
    fig, ax = plt.subplots(figsize=(10, 4.6))
    cols = [C["blue"] if v >= 0 else C["red"] for v in g["sum"]]
    ax.bar(g.index.astype(str), g["sum"] * 100, color=cols, width=0.62)
    for x, (s, n) in enumerate(zip(g["sum"], g["count"])):
        ax.annotate(f"n {int(n)}", (x, s * 100), xytext=(0, 4 if s >= 0 else -12), textcoords="offset points", ha="center", fontsize=8, color=C["ink"])
    ax.axhline(0, color=C["ink"], lw=0.8)
    ax.set_ylabel("연도별 순손익 합 % (1x, 1.8bp)")
    ax.set_title("05 연도별 순손익 합과 건수 (집행일 연도, 2011.9~2026.9); 연도 부호 판정은 건수 ≥ 10 인 해만", fontsize=11, loc="left")
    _ax_clean(ax)
    fig.tight_layout()
    fig.savefig(OUT / "fig3_yearly.png", dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------- notes
def D(s):
    ts = pd.Timestamp(s)
    return f"{ts.year}.{ts.month}.{ts.day}"


def N(x, d=2, sign=True):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    return (f"{x:+.{d}f}" if sign else f"{x:.{d}f}").replace("-", "−")


def line(r, cost):
    w0, w1 = r["window"].split("~")
    return (f"• {r['name']}: n={r['n']}, 창 {D(w0)}~{D(w1)}, 순수익 {N(r['net_bp'])}bp/건(총수익 {N(r['gross_bp'])}bp, 왕복 {cost}), NW t {N(r['t_nw'])}, "
            f"Δ(비신호일 대비) {N(r['delta_bp'], 1)}bp(t {N(r['t_delta'])}), 승률 {N(r['win_pct'], 1, False)}%, CAGR {N(r['cagr_pct'])}%, MDD {N(r['mdd_pct'])}%, "
            f"연도 부호 {r['yearly_pos']}, 상위 5건 비중 {N(r['top5_share_pct'], 1, False)}%, 전반 {N(r['half1_net_bp'], 1)}bp 후반 {N(r['half2_net_bp'], 1)}bp → {r['verdict']}")


def write_notes(rows, m1, ms, lev, v2, v3, v4, v5, v6a, v6b, xf, ov, grid, bk, rho_all, rho_act, rho_kospi, cl, J, first_sig, last_sig, n_cells, d, tr_main, nodata, n_w0):
    R = {r["id"]: r for r in rows}
    yd = lambda r: r["notes"].split("연도별(순손익 합, 건수) ")[1].split(";")[0]
    ks, fu, kf, kq = d["ks"], d["fu"], d["kf"], d["kq"]
    big = kf.r_oc.loc[W_FULL[0]:W_FULL[1]]
    L = []
    L += ["# 05_frn_dual_sell 결과 · 외인 현물·선물 동시 순매도 → 익일 코스피200 선물 front 숏(낮)", "",
          "## 규약(사전 등록 고정)",
          "• 신호(T 장 마감 뒤 확정치): z_spot ≤ −0.84 ∧ z_fut ≤ −0.84. z = (x_T − 직전 250 거래일 평균)/직전 250일 표준편차(T 제외, ddof 1). x_spot = 코스피 외인 순매수(원, naver_kospi_investor_daily), x_fut = 코스피200 선물 외인 순매수(계약, naver_fut_investor_daily).",
          "• 집행: T+1 08:45 코스피200 선물 front 매도 → 15:45 환매(kis_fut_k200_daily front=1 행, 시가·종가 단일가). 손익 = −ln(종가/시가) − 1.8bp(스트레스 3bp). 명목 1x, 증거금 21% 별도. 2x·3x 는 scaled(비용도 k 배).",
          "• 최종거래일 규칙: 파일의 front 플래그는 만기일(둘째 목요일, 휴장 조정) 당일부터 차근월물. T+1 이 만기일이면 차근월물의 08:45→15:45 이며 하루 안에서는 같은 계약. 밤 보유 대조는 T+1 front 코드를 고정해 T+2 같은 코드 시가 사용.",
          f"• 창(집행일 기준): 전체 {D(W_FULL[0])}~{D(W_FULL[1])}, 설계 {D(W_DES[0])}~{D(W_DES[1])}, 검증 {D(W_VAL[0])}~{D(W_VAL[1])}, W_all {D(W_ALL[0])}~{D(W_ALL[1])}, W_ex2026 {D(W_EX26[0])}~{D(W_EX26[1])}, W_long {D(W_LONG[0])}~{D(W_LONG[1])}. 2026.10.30 행 제외.",
          "• 통계: 거래별 순수익 bp 상수 회귀 HAC(maxlags 5) NW t. Δ = 창 안 전 거래일(같은 방향·비용)에 신호 더미 회귀 HAC t. CAGR·MDD 는 069500 거래일 달력에 비활성일 0. 연도 부호는 이벤트 ≥ 10 인 해만. 상위 5건 비중 = 상위 5건 합/총합(총합 > 0). 전·후반 = 건수 반분.",
          "• 판정: 공통 규칙(순수익 > 0 ∧ NW t ≥ 2 ∧ 연도 부호 ≥ 70% ∧ 상위 5건 < 50% ∧ 전·후반 양수) ∧ 3bp 순수익 > 0 ∧ 13차 합산 장부 comb_a 와 |ρ| ≤ 0.2 ∧ 결합(코스피 장부 가산) W_all·W_long CAGR·MDD 동시 개선 ∧ 2026 제외 양수 → candidate. 순수익 ≤ 0 reject, 그 밖 reference, 창 ≤ 300 거래일 또는 이벤트 < 10 estimate.",
          f"• 돌린 셀 수 {n_cells}(주 셀 창·비용·레버리지 9, 변형 V1~V6 9, X 1, 단독 2, 격자 9, 장부 16, ρ·충돌 2). 변형은 보고용이며 주 셀을 바꾸지 않음. 14차 누적 512셀 + 15차 조사 빠른 점검 약 100셀 위에 더해지는 다중 비교.", "",
          "## 자료 점검(다른 역할이 만든 파일 포함, 직접 확인)",
          f"• 코스피 투자자별(네이버): {D(ks.index.min())}~{D(ks.index.max())} {len(ks)}행, net_foreign 결측 0, 069500 달력과 2010.9.20 이후 완전 일치. |순매수| 중앙값 {ks.net_foreign.abs().median() / 1e8:,.0f}억 원.",
          f"• 선물 투자자별(네이버): {D(fu.index.min())}~{D(fu.index.max())} {len(fu)}행, net_foreign 결측 0, 달력 일치. 매수·매도 계약 열은 2010.12.21 까지 매도 0 으로 채워져 있어(순매수만 유효) Flow 비율은 2010.12.22 부터. |순매수| 중앙값 {fu.net_foreign.abs().median():,.0f}계약, 연도별 표준편차 2010년대 3,400~4,700 → 2023년 6,500 계약.",
          f"• kis_fut_k200_daily(KIS 역할 생성): front 행 {len(kf)}일 {D(kf.index.min())}~{D(kf.index.max())}, 날짜 중복 0, 시가·종가 결측·0 없음, {D(W_FULL[0])}~{D(W_FULL[1])} 달력 결측 0. 만기일 front 전환 규칙을 2024.6.13·2025.9.11·2026.9.10 행에서 확인. |ln(종가/시가)| > 5% 인 날 {int((big.abs() > 0.05).sum())}일(2020.3 과 2026년에 집중, 최대 {N(big.min() * 100)}% {D(big.idxmin())}).",
          f"• kis_fut_kq150_daily: front {len(kq)}일 {D(kq.index.min())}~{D(kq.index.max())}, 달력 결측 0, 만기일 규칙 동일. 코스닥 투자자별·프로그램(차익) 파일 결측 0, 달력 일치. 만기 달력 {len(d['expiry'])}행(2005~2026).",
          f"• z 정의 수기 대조 1건 통과(t 제외 직전 250일). 첫 신호 가능일 {D(first_sig)}(z_fut 워밍업), 마지막 신호일 {D(last_sig)}(코스피 투자자별 자료 끝) → 마지막 집행일 {D(tr_main.T1.max())}. 선물 수익 결측으로 버린 신호 {nodata}건.",
          "• 자료 확보 단계: 없음(전부 보유). 선물 수수료율·세율 미확인 → 규약 1.8bp/3bp.", ""]
    L += ["## 핵심 요약",
          f"• 주 셀(동시 z ≤ −0.84 → 익일 선물 front 숏, 1.8bp, 1x): n={m1['n']}, 창 {D(W_FULL[0])}~{D(W_FULL[1])}, 순수익 {N(m1['net_bp'])}bp/건, NW t {N(m1['t_nw'])}, 연도 부호 {m1['yearly_pos']}, 상위 5건 {N(m1['top5_share_pct'], 1, False)}%, 전반 {N(m1['half1_net_bp'], 1)} 후반 {N(m1['half2_net_bp'], 1)}bp, CAGR {N(m1['cagr_pct'])}% MDD {N(m1['mdd_pct'])}% → 독립 셀 {m1['verdict']}.",
          f"• 설계창 {N(R['M1_design']['net_bp'])}bp(n={R['M1_design']['n']}, t {N(R['M1_design']['t_nw'])}) / 검증창 {N(R['M1_valid']['net_bp'])}bp(n={R['M1_valid']['n']}, t {N(R['M1_valid']['t_nw'])}) / W_all {N(R['M1_Wall']['net_bp'])}bp(n={R['M1_Wall']['n']}, t {N(R['M1_Wall']['t_nw'])}) / 2026 제외 {N(v6a['net_bp'])}bp(n={v6a['n']}). 스트레스 3bp {N(ms['net_bp'])}bp.",
          f"• 장부 결합(코스피 장부 가산, 1x): W_all CAGR {N(bk[('W_all', 'base')]['cagr'] * 100)} → {N(bk[('W_all', 'FDS')]['cagr'] * 100)}%, MDD {N(bk[('W_all', 'base')]['mdd'] * 100)} → {N(bk[('W_all', 'FDS')]['mdd'] * 100)}%; W_long CAGR {N(bk[('W_long', 'base')]['cagr'] * 100)} → {N(bk[('W_long', 'FDS')]['cagr'] * 100)}%, MDD {N(bk[('W_long', 'base')]['mdd'] * 100)} → {N(bk[('W_long', 'FDS')]['mdd'] * 100)}%; 2024 {N(bk[('Y2024', 'base')]['cum'] * 100)} → {N(bk[('Y2024', 'FDS')]['cum'] * 100)}%, 2025 {N(bk[('Y2025', 'base')]['cum'] * 100)} → {N(bk[('Y2025', 'FDS')]['cum'] * 100)}%. ρ(comb_a) {N(rho_all, 3)}.",
          f"• 사전 등록 판정: {'통과' if J['passed'] else '미통과'} → 등급 {J['grade']}. 조건별: 공통 {J['common']}, 3bp {J['stress']}, |ρ| ≤ 0.2 {J['rho']}, W_all 개선 {J['wall']}, W_long 개선 {J['wlong']}, 2026 제외 양수 {J['ex2026']}; 장부 규칙(2024·2025 포함) {J['book_rule']}.", ""]
    L += ["## 주 셀과 창·비용·레버리지"]
    for cid in ("M1", "M1_design", "M1_valid", "M1_Wall", "M1_Wex2026", "M1_Wlong"):
        L.append(line(R[cid], "1.8bp"))
    L.append(line(ms, "3bp"))
    for k in (2, 3):
        L.append(line(lev[k], f"{k}×1.8bp"))
    L += [f"• 주 셀 연도별 순손익 합(건수): {yd(m1)}", ""]
    L += ["## 변형(보고용)"]
    for th in (-0.5, -1.0, -1.5):
        L.append(line(R[f"V1_th{abs(th) * 10:02.0f}"], "1.8bp"))
    L.append(line(v2, "5bp"))
    L.append(line(v3, "1.8bp"))
    L.append(line(v4, "1.8bp×w") + f" (w = 0 미집행 {n_w0}건, Δ 생략)")
    L.append(line(v5, "1.8bp"))
    L.append(line(v6a, "1.8bp"))
    L.append(line(v6b, "1.8bp"))
    L.append(line(xf, "1.8bp"))
    L.append(line(R["S_spot_any"], "1.8bp"))
    L.append(line(R["S_fut_any"], "1.8bp"))
    L += [f"• woohyun_ Flow 정의 대조: 외인 선물 (매수−매도)/(매수+매도) 를 같은 z250 ≤ −0.84 로 넣으면 공통 정의역 {ov['n_dom']}일에서 주 신호 {ov['nA']}일, Flow 신호 {ov['nB']}일, 교집합 {ov['nAB']}일(Jaccard {N(ov['jaccard'], 3, False)}), ρ(z_fut, z_flow) {N(ov['rho_z'], 3)}, ρ(순매수 계약, Flow 비율) {N(ov['rho_raw'], 3)}. 같은 변수의 다른 정규화이며 독립 증거가 아님.", ""]
    L += ["## z 조합 격자(그림 1, 익일 front 시가→종가 평균, 롱 부호, 비용 0)"]
    for s in ("sell", "mid", "buy"):
        L.append("• 현물 " + s + ": " + " / ".join(f"선물 {f} {N(grid[(s, f)]['gross_bp'], 1)}bp(n {grid[(s, f)]['n']}, t {N(grid[(s, f)]['t_nw'])})" for f in ("sell", "mid", "buy")))
    L += ["", "## 13차 장부 결합·상관·충돌(1x, 코스피 장부 = r_A + r_B + r_C + r_FDS)"]
    for wn in ("W_all", "W_ex2026", "Y2024", "Y2025", "Y2026", "W_long", "W_all_KOSPI", "W_long_KOSPI"):
        b, n_ = bk[(wn, "base")], bk[(wn, "FDS")]
        L.append(f"• {wn}({b['n_days']}일): 원 장부 CAGR {N(b['cagr'] * 100)}% MDD {N(b['mdd'] * 100)}% 누적 {N(b['cum'] * 100)}% 샤프 {N(b['sharpe'], 2, False)} → +FDS CAGR {N(n_['cagr'] * 100)}% MDD {N(n_['mdd'] * 100)}% 누적 {N(n_['cum'] * 100)}% 샤프 {N(n_['sharpe'], 2, False)}")
    L += [f"• ρ(FDS 일수익, comb_a) W_all 전 거래일 {N(rho_all, 4)}(n {bk[('W_all', 'base')]['n_days']}), 활성일만 {N(rho_act, 4)}(n {cl['n_act']}), 코스피 장부와 {N(rho_kospi, 4)}.",
          f"• jehyeong_KP 롱 충돌(W_all): 충돌일 {cl['n']}일 / FDS 활성 {cl['n_act']}일 / KP 롱 {cl['n_kp']}일. 충돌일 KP 순손익 합 {N(cl['kp_sum'] * 100)}%, FDS 합 {N(cl['fds_sum'] * 100)}%, 합산 {N((cl['kp_sum'] + cl['fds_sum']) * 100)}%. 충돌일 FDS 평균 {N(cl['fds_mean_clash'], 1)}bp vs 비충돌 활성일 {N(cl['fds_mean_noclash'], 1)}bp. 파생 자본 별도라 합산 허용(사전 선택).", ""]
    L += ["## 판정", f"• 주 셀 M1 독립 셀 판정 {m1['verdict']} → 사전 등록 결합 판정 {'통과' if J['passed'] else '미통과'}, 등급 {J['grade']}.",
          f"• 조건: 공통 규칙 {J['common']} / 3bp 양수 {J['stress']} / |ρ| ≤ 0.2 {J['rho']} / W_all CAGR·MDD 개선 {J['wall']} / W_long CAGR·MDD 개선 {J['wlong']} / 2026 제외 양수 {J['ex2026']} / 2024 개선 {J['y2024']} / 2025 개선 {J['y2025']}.",
          "• 변형 결과로 주 셀 문턱·창·차량을 바꾸지 않았다. 탐색한 셀은 모두 cells.csv 에 있다.", "",
          "## 단서",
          "• 선물 수수료율 미확인(1.8bp 규약), 미니 1계약 단위(약 5,400만 원) 무시, 증거금 21% 별도, 선물 양도세 11% 미반영, 파생 계좌 전제.",
          "• 08:45·15:45 단일가 체결 가정. 선물은 개장·마감 단일가가 있어 ETF 09:00 가정보다 의존이 약하나 08:45 단일가 체결량·슬리피지는 미측정.",
          "• 입력 변수가 4·5차 기각 축(코스피 외인 현물 z, 외인 선물 순매수)과 같다. 동시 조건·매도 쪽·낮 보유·선물 집행이 다를 뿐이라 다중 비교 부담이 크다.",
          "• 2026년은 선물 일중 변동이 5% 를 넘는 날이 많아(위 점검) 2026 포함·제외 수치가 크게 다를 수 있다. 2026 제외 셀을 대표값으로 병기.",
          "• 연도 부호는 이벤트 ≥ 10 인 해만 세므로 2011년(9월 시작)과 건수가 적은 해는 빠진다.", "",
          "## 파일",
          "• cells.csv(셀 전부), trades.csv(주 셀·변형 거래, cell 열), fig1_zgrid.png(3×3 격자), fig2_cum.png(누적 순손익 1x·3bp), fig3_yearly.png(연도별 막대).",
          f"• series/{HANDLE}.csv(date = T+1, pos −1, gross, net, 비활성 0, 069500 달력) + {HANDLE}.meta.json(grade 포함). 코드 r15_05_frn_dual_sell.py(같은 폴더 r15_leverage_desk.py 의 장부 함수 import)."]
    text = "\n".join(L) + "\n"
    assert "-" not in text, [l for l in L if "-" in l][:3]
    (OUT / "notes.md").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
