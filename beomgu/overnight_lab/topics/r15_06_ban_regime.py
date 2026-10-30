"""15차 06_ban_regime · 공매도 전면금지 레짐 분해 · 다리 비중 오버레이(진단 + 포워드 규칙)

규칙(사전 등록 design_shortlist.md §2 '### 06_ban_regime', alpha-share-prereg-2026-10-30.md 고정)
  D_ban = 1 : 2008-10-01~2009-05-31, 2011-08-10~2011-11-09, 2020-03-16~2021-05-02, 2023-11-06~2025-03-30
            (금융위 보도자료로 확정, notes.md 출처). 밤 다리(date=청산일)는 진입일(전 거래일) 기준으로 판정.
  진단   : 13차 series 6개 활성일 net(bp) ~ 상수 + D_ban, HAC(maxlags 10) → 다리별 Δ·t.
  2×2    : jehyeong_ON 조건 밤 vs 무조건 069500 밤(종가→익일 시가, 5bp), 금지기 내외.
  비중표 A(사전 고정): 금지기 jehyeong_ON 0, beomgu_C3Q 0, 나머지 1.0.
  비중표 B          : ON·C3Q 0.5, C3VT30·blog_ON_Q 1.5(ETF 명목 한도 롱 2·코스닥 숏 1 안에서 절단 → blog_ON_Q 는 사실상 1.0).
  비중 m 적용 수익  : m ≤ 1 → m×net(비용도 비례). m > 1(C3VT30 만) → m×gross − 5bp×min(m×|pos|, 1)(C3 비용 규칙 그대로).
  주 셀  : 비중표 A 를 W_long(2016-10-04~2026-09-17, 코스피 B + 코스닥 A·B)의 2020-03-16~2021-05-02 에만 표본외 적용 → 결합 CAGR·MDD vs 원 장부.
           W_all(2023-11-06~2025-03-30 포함) 적용은 '사후' 라벨. 2024·2025 연수익 변화.
  변형   : V1 금지기 ON 조건 해제(무조건 069500 밤 롱) / V2 비중표 B / V3 ON 2010~ 재구성(133690 프록시 ∧ 코스피 외인 T−1 순매수)
           / V4 베이시스 z(FUT−KPI200, 250일)·VKOSPI 통제 / V5 부분 재개 더미(2021-05-03~2023-11-05) 분리 / V6 C3 괴리 몫·갭 몫.
  판정   : 표본외 금지기(2020~2021) CAGR·MDD 동시 개선 ∧ 비중을 바꾼 다리(ON·C3Q)의 Δ HAC |t| ≥ 2(부호 음) ∧ W_all 결합 CAGR·MDD 개선 ∧ 2024·2025 개선('사후').
           등급 estimate 고정. 통과해도 다음 금지 공표 전까지 장부 미반영(포워드 규칙). 통계 t 는 이벤트 bp 상수 회귀 HAC(5), Δ 는 더미 회귀 HAC(10).
자료(읽기 전용)
  round13_slot_book/series/*.csv + meta.json, book/book_daily.csv(comb_a), hyfe/data/raw/etf/069500·133690·229200·FUT·KPI200,
  raw/k200_index_daily.csv, raw/etf_extra/KQ150_index_kis.csv, raw/naver_kospi_investor_daily.csv(net_foreign),
  raw/us_etf/yf_i_NDX.csv, raw/r15_external/fdr_usdkrw_daily.csv, raw/r15_external/kis_vkospi_daily.csv.
가정
  단일가 가정(09:00 시가·15:30 종가 체결)은 13차 series 와 무조건 밤·재구성 셀에 그대로 적용. 모든 창은 2026-09-17 에서 끝냄(13차 W_all·W_long 과 동일).
  C3Q 비중 0 인 날에도 13차 겹침 규칙으로 0 처리된 jehyeong_KQ 는 복원하지 않음(비중만 조절).
산출  round15_alpha_share/06_ban_regime/{cells.csv, trades.csv, notes.md, judging.json, fig1~3.png}, series/overlay_06.csv + overlay_06.meta.json
"""
import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.stdout.reconfigure(encoding="utf-8")
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

LAB = Path("<OUT>")
R13 = LAB / "round13_slot_book"
R15 = LAB / "round15_alpha_share"
OUT = R15 / "06_ban_regime"
SER = R15 / "series"
RAW = Path("<raw>")
OUT.mkdir(parents=True, exist_ok=True)
SER.mkdir(parents=True, exist_ok=True)

END = pd.Timestamp("2026-09-17")
BANS = {"2008": ("2008-10-01", "2009-05-31"), "2011": ("2011-08-10", "2011-11-09"),
        "2020": ("2020-03-16", "2021-05-02"), "2023": ("2023-11-06", "2025-03-30")}
ALL_BANS = list(BANS.values())
PARTIAL = [("2021-05-03", "2023-11-05")]
WIN = {"W_all": ("2022-12-13", "2026-09-17"), "W_ex2026": ("2022-12-13", "2025-12-30"),
       "W_long": ("2016-10-04", "2026-09-17"), "W_KB03": ("2003-01-02", "2026-09-17"),
       "Y2024": ("2024-01-02", "2024-12-30"), "Y2025": ("2025-01-02", "2025-12-30"),
       "BAN2020": ("2020-03-16", "2021-05-02")}
# handle → (시장, 슬롯, 밤 다리 여부)
LEGS = {"jehyeong_ON": ("KOSPI", "A", True), "beomgu_C3VT30": ("KOSPI", "B", False), "jehyeong_KP": ("KOSPI", "C", False),
        "blog_ON_Q": ("KOSDAQ", "A", True), "beomgu_C3Q": ("KOSDAQ", "B", False), "jehyeong_KQ": ("KOSDAQ", "C", False)}
CAP = {"jehyeong_ON": (2, 2), "beomgu_C3VT30": (2, 2), "blog_ON_Q": (2, 1), "beomgu_C3Q": (2, 1)}  # ETF 명목 한도(롱, 숏)
TABLES = {"A": {"jehyeong_ON": 0.0, "beomgu_C3Q": 0.0, "beomgu_C3VT30": 1.0, "blog_ON_Q": 1.0},
          "B": {"jehyeong_ON": 0.5, "beomgu_C3Q": 0.5, "beomgu_C3VT30": 1.5, "blog_ON_Q": 1.5}}
BOOK_LEGS = {"W_all": ["jehyeong_ON", "beomgu_C3VT30", "blog_ON_Q", "beomgu_C3Q"],
             "W_long": ["beomgu_C3VT30", "blog_ON_Q", "beomgu_C3Q"], "W_KB03": ["beomgu_C3VT30"]}
COST_BP = 5.0

# ------------------------------------------------------------------ 달력·자료
_cal = pd.read_csv(RAW / "etf/069500.csv", parse_dates=["date"]).set_index("date").sort_index()
CAL = _cal.index
PREV = pd.Series(CAL, index=CAL).shift(1)
_S = {}


def series(h):
    if h not in _S:
        s = pd.read_csv(R13 / "series" / f"{h}.csv", parse_dates=["date"]).set_index("date")
        m = json.loads((R13 / "series" / f"{h}.meta.json").read_text("utf-8"))
        _S[h] = (s.loc[m["first_valid"]:m["last_valid"]].fillna(0.0).loc[:END], m)
    return _S[h]


def entry(idx, night):
    """밤 다리는 date=청산일 → 진입일 = 전 거래일."""
    return pd.DatetimeIndex(PREV.reindex(idx).values) if night else pd.DatetimeIndex(idx)


def in_periods(e, periods):
    d = np.zeros(len(e), bool)
    for a, b in periods:
        d |= (e >= pd.Timestamp(a)) & (e <= pd.Timestamp(b))
    return d


def dummy(idx, night, periods):
    return in_periods(entry(idx, night), periods)


def ban_id(e):
    out = np.array([""] * len(e), dtype=object)
    for k, (a, b) in BANS.items():
        out[(e >= pd.Timestamp(a)) & (e <= pd.Timestamp(b))] = k
    return out


# ------------------------------------------------------------------ 통계
def hac(y, X=None, lags=5):
    y = np.asarray(y, float)
    n = len(y)
    Xm = np.ones((n, 1)) if X is None else np.column_stack([np.ones(n), np.asarray(X, float)])
    if n < Xm.shape[1] + 2 or (Xm.shape[1] > 1 and (Xm[:, 1:].std(axis=0) == 0).any()):
        return np.full(Xm.shape[1], np.nan), np.full(Xm.shape[1], np.nan)
    r = sm.OLS(y, Xm).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return np.asarray(r.params), np.asarray(r.tvalues)


def ev_stats(net, dates):
    """이벤트(활성일) 순수익 통계. net 은 자본 1 기준 분수."""
    net = np.asarray(net, float)
    n = len(net)
    bp = net * 1e4
    o = dict(n=n, net_bp=bp.mean() if n else np.nan, t_nw=hac(bp)[1][0] if n >= 3 else np.nan,
             win_pct=(net > 0).mean() * 100 if n else np.nan)
    if n:
        g = pd.Series(net, index=pd.DatetimeIndex(dates)).groupby(lambda d: d.year)
        cnt, sm_ = g.count(), g.sum()
        ok = cnt >= 10
        o["yearly_pos"] = f"{int((sm_[ok] > 0).sum())}/{int(ok.sum())}" if ok.any() else ""
        tot = net.sum()
        o["top5_share_pct"] = np.sort(net)[-5:].sum() / tot * 100 if (tot > 0 and n >= 5) else np.nan
        h = n // 2
        o["half1_net_bp"] = bp[:h].mean() if h else np.nan
        o["half2_net_bp"] = bp[h:].mean() if n - h else np.nan
    else:
        o.update(yearly_pos="", top5_share_pct=np.nan, half1_net_bp=np.nan, half2_net_bp=np.nan)
    return o


def metrics(r):
    r = r.dropna()
    n = len(r)
    eq = (1 + r).cumprod()
    cum = float(eq.iloc[-1])
    dd = eq / eq.cummax() - 1
    vol = float(r.std(ddof=1)) * np.sqrt(252)
    return dict(n_days=n, n_active=int((r != 0).sum()), cum=cum - 1, cagr=cum ** (252 / n) - 1, mdd=float(dd.min()),
                mdd_date=dd.idxmin().date().isoformat(), vol=vol, sharpe=float(r.mean()) * 252 / vol if vol else np.nan,
                worst=float(r.min()), worst_date=r.idxmin().date().isoformat(), win=float((r[r != 0] > 0).mean()) if (r != 0).any() else np.nan)


# ------------------------------------------------------------------ 비중 적용
def leg_ret(h, m):
    """비중 m(Series, series 달력) 적용 수익. m ≤ 1 → m×net, m > 1 → m×gross − cost×min(m|pos|,1)/min(|pos|,1). ETF 한도로 절단."""
    s, _ = series(h)
    pos = s.pos.abs()
    safe = pos.where(pos > 0, 1.0)
    cap = np.where(s.pos >= 0, CAP[h][0], CAP[h][1])
    me = np.where(pos > 0, np.minimum(m.values, cap / safe.values), m.values)
    cost = (s.gross - s.net).values
    r = np.where(me <= 1, me * s.net.values,
                 me * s.gross.values - cost * np.minimum(me * pos.values, 1) / np.minimum(safe.values, 1))
    return pd.Series(r, index=s.index), pd.Series(me, index=s.index)


def wser(h, table, periods):
    s, _ = series(h)
    m = pd.Series(1.0, index=s.index)
    m[dummy(s.index, LEGS[h][2], periods)] = TABLES[table][h]
    return m


def book_delta(table, periods, legs, idx):
    """합산 장부(코스피 50% + 코스닥 50%) 일수익 변화 = 0.5 × Σ(비중 적용 수익 − 원 net)."""
    tot = pd.Series(0.0, index=idx)
    for h in legs:
        s, _ = series(h)
        r, _ = leg_ret(h, wser(h, table, periods))
        tot = tot + (r - s.net).reindex(idx).fillna(0.0) * 0.5
    return tot


# ------------------------------------------------------------------ 메인
def main():
    cells, trades, log = [], [], []
    P = lambda *a: (print(*a), log.append(" ".join(str(x) for x in a)))

    def add(id_, name, market, window, st, verdict, notes, gross_bp=np.nan, delta_bp=np.nan, t_delta=np.nan, cagr=np.nan, mdd=np.nan):
        cells.append(dict(id=id_, name=name, market=market, window=window, n=st.get("n", np.nan), gross_bp=gross_bp,
                          net_bp=st.get("net_bp", np.nan), t_nw=st.get("t_nw", np.nan), delta_bp=delta_bp, t_delta=t_delta,
                          win_pct=st.get("win_pct", np.nan), cagr_pct=cagr * 100 if pd.notna(cagr) else np.nan,
                          mdd_pct=mdd * 100 if pd.notna(mdd) else np.nan, yearly_pos=st.get("yearly_pos", ""),
                          top5_share_pct=st.get("top5_share_pct", np.nan), half1_net_bp=st.get("half1_net_bp", np.nan),
                          half2_net_bp=st.get("half2_net_bp", np.nan), verdict=verdict, notes=notes))

    BM = {}

    def book_cell(id_, name, wkey, r_new, r_base, verdict, notes, changed=None):
        w = WIN[wkey]
        rn = r_new.loc[w[0]:w[1]]
        mt = metrics(rn)
        BM[id_] = mt
        dlt = t_d = np.nan
        if r_base is not None:
            diff = (rn - r_base.loc[w[0]:w[1]]) * 1e4
            ch = diff[diff.abs() > 1e-12] if changed is None else diff[changed.reindex(diff.index).fillna(False)]
            if len(ch) >= 3:
                p, t = hac(ch.values, lags=10)
                dlt, t_d = p[0], t[0]
            elif len(ch):
                dlt = ch.mean()
        act = rn[rn != 0]
        st = dict(n=mt["n_days"], net_bp=rn.mean() * 1e4, t_nw=hac(rn.values * 1e4)[1][0], win_pct=mt["win"] * 100 if pd.notna(mt["win"]) else np.nan)
        add(id_, name, "comb", f"{w[0]}~{w[1]}", st, verdict,
            f"{notes} | 누적 {mt['cum']*100:.2f}% 활성 {mt['n_active']} MDD일 {mt['mdd_date']} 최악일 {mt['worst']*100:.2f}%({mt['worst_date']}) 샤프 {mt['sharpe']:.2f}",
            delta_bp=dlt, t_delta=t_d, cagr=mt["cagr"], mdd=mt["mdd"])
        return mt

    def leg_cell(id_, name, h, idx_mask, window, periods, verdict, notes, lags=10, extra_X=None, pre=""):
        """활성일 net 을 D_ban(+통제)에 회귀. idx_mask: 창 안 활성일 선택."""
        s, _ = series(h)
        a = s[(s.pos != 0) & idx_mask]
        d = dummy(a.index, LEGS[h][2], periods).astype(float)
        st = ev_stats(a.net.values, a.index)
        X = d if extra_X is None else np.column_stack([d, extra_X])
        p, t = hac(a.net.values * 1e4, X, lags=lags)
        nb, bb = a.net[d == 0] * 1e4, a.net[d == 1] * 1e4
        txt = f"{pre}금지기 n {int(d.sum())} 평균 {bb.mean() if len(bb) else np.nan:.1f}bp / 비금지 n {int((d==0).sum())} 평균 {nb.mean() if len(nb) else np.nan:.1f}bp"
        add(id_, name, LEGS[h][0], window, st, verdict, f"{notes} | {txt}", gross_bp=a.gross.mean() * 1e4 if h != "beomgu_C3Q" else np.nan,
            delta_bp=p[1], t_delta=t[1])
        return dict(delta=p[1], t=t[1], n_ban=int(d.sum()), mean_ban=bb.mean() if len(bb) else np.nan, mean_non=nb.mean() if len(nb) else np.nan,
                    n_non=int((d == 0).sum()), params=p, tvals=t, stats=st)

    fmtd = lambda d: str(pd.Timestamp(d).date())

    # ---------- 자료 점검
    P("[자료 점검]")
    P(f"069500 달력 {fmtd(CAL[0])}~{fmtd(CAL[-1])} {len(CAL)}행, 2026-09-17 이후 제외")
    for h in LEGS:
        s, m = series(h)
        a = s[s.pos != 0]
        c = (a.gross - a.net) * 1e4
        P(f"{h}: 유효 {fmtd(s.index[0])}~{fmtd(s.index[-1])} {len(s)}일 활성 {len(a)} |pos| {a.pos.abs().min():.3f}~{a.pos.abs().max():.3f} "
          f"비용(gross−net) 중앙 {c.median():.2f}bp 범위 {c.min():.1f}~{c.max():.1f} (meta n_active {m['n_active']})")
    book = pd.read_csv(R13 / "book/book_daily.csv", parse_dates=["date"]).set_index("date")
    for col, h in [("KOSPI_A_jehyeong_ON", "jehyeong_ON"), ("KOSPI_B_beomgu_C3VT30", "beomgu_C3VT30"),
                   ("KOSDAQ_A_blog_ON_Q", "blog_ON_Q"), ("KOSDAQ_B_beomgu_C3Q", "beomgu_C3Q")]:
        dmax = (book[col] - series(h)[0].net.reindex(book.index).fillna(0.0)).abs().max()
        assert dmax < 1e-7, (col, dmax)   # book_daily 는 소수 8자리 반올림
        P(f"장부 열 {col} vs series net 최대 차 {dmax:.1e}")
    comb_all = book["comb_a"]
    m0 = metrics(comb_all)
    assert m0["n_days"] == 919 and abs(m0["cagr"] - 0.314179) < 1e-4 and abs(m0["mdd"] + 0.061046) < 1e-4, m0
    idx_long = CAL[(CAL >= WIN["W_long"][0]) & (CAL <= END)]
    comb_long = 0.5 * sum(series(h)[0].net.reindex(idx_long).fillna(0.0) for h in BOOK_LEGS["W_long"])
    m1 = metrics(comb_long)
    assert m1["n_days"] == 2444 and abs(m1["cagr"] - 0.150369) < 1e-4 and abs(m1["mdd"] + 0.096107) < 1e-4, m1
    idx_kb = CAL[(CAL >= WIN["W_KB03"][0]) & (CAL <= END)]
    comb_kb = series("beomgu_C3VT30")[0].net.reindex(idx_kb).fillna(0.0)
    P(f"13차 재현: W_all comb CAGR {m0['cagr']*100:.2f}% MDD {m0['mdd']*100:.2f}%({m0['mdd_date']}) n {m0['n_days']} | "
      f"W_long CAGR {m1['cagr']*100:.2f}% MDD {m1['mdd']*100:.2f}%({m1['mdd_date']}) n {m1['n_days']}")
    y24, y25 = metrics(comb_all.loc[slice(*WIN["Y2024"])]), metrics(comb_all.loc[slice(*WIN["Y2025"])])
    assert abs(y24["cum"] - 0.112277) < 1e-4 and abs(y25["cum"] - 0.113669) < 1e-4
    P(f"13차 연수익 재현: 2024 {y24['cum']*100:.2f}% 2025 {y25['cum']*100:.2f}%")
    # 금지기 거래일 수(069500 달력)
    for k, (a, b) in BANS.items():
        n = int(((CAL >= a) & (CAL <= b)).sum())
        P(f"금지기 {k}: {a}~{b} 거래일 {n}")
    P(f"부분 재개 {PARTIAL[0][0]}~{PARTIAL[0][1]} 거래일 {int(in_periods(CAL, PARTIAL).sum())}")

    # ---------- 기준선 셀
    P("\n[기준선 13차 원 장부]")
    base = {}
    base["W_long"] = book_cell("B_long_orig", "원 장부 W_long(코스피 B + 코스닥 A·B)", "W_long", comb_long, None, "reference", "13차 원 장부")
    base["BAN2020"] = book_cell("B_ban2020_orig", "원 장부 2020 금지기만", "BAN2020", comb_long, None, "reference", "W_long 장부의 2020-03-16~2021-05-02 구간(≤300일 → estimate 급)")
    base["W_all"] = book_cell("B_all_orig", "원 장부 W_all(여섯 다리)", "W_all", comb_all, None, "reference", "13차 원 장부 comb_a")
    base["W_ex2026"] = book_cell("B_ex2026_orig", "원 장부 W_ex2026", "W_ex2026", comb_all, None, "reference", "13차 원 장부 comb_a")
    base["Y2024"] = book_cell("B_Y2024_orig", "원 장부 2024", "Y2024", comb_all, None, "reference", "연수익(누적) = cum")
    base["Y2025"] = book_cell("B_Y2025_orig", "원 장부 2025", "Y2025", comb_all, None, "reference", "연수익(누적) = cum")
    base["W_KB03"] = book_cell("B_KB03_orig", "원 장부 W_KB03(코스피 B 단독)", "W_KB03", comb_kb, None, "reference", "C3VT30 단독, 2003~")

    # ---------- 주 셀: 비중표 A
    P("\n[주 셀 비중표 A]")
    res = {}
    dA_long_oos = book_delta("A", [BANS["2020"]], BOOK_LEGS["W_long"], idx_long)
    chg = dA_long_oos.abs() > 1e-12
    res["M1"] = book_cell("M1_long_A_2020oos", "비중표 A → W_long, 2020 금지기만(표본외) [주 셀]", "W_long", comb_long + dA_long_oos, comb_long, "estimate",
                          f"표본외 적용(C3Q 0, 2020-03-16~2021-05-02 진입일 기준). 비중 변경일 {int(chg.sum())}", changed=chg)
    res["M1b"] = book_cell("M1b_ban2020_A", "비중표 A, 2020 금지기 구간만 평가", "BAN2020", comb_long + dA_long_oos, comb_long, "estimate", "표본외 금지기 구간 안 CAGR·MDD(≤300일)", changed=chg)
    dA_long_all = book_delta("A", ALL_BANS, BOOK_LEGS["W_long"], idx_long)
    res["M1c"] = book_cell("M1c_long_A_all", "비중표 A → W_long, 2020+2023 금지기(2023 은 사후)", "W_long", comb_long + dA_long_all, comb_long, "estimate",
                           "장부 결합 규칙용(W_long 전체 금지기 적용, 2023 사후 포함)", changed=dA_long_all.abs() > 1e-12)
    dA_all = book_delta("A", ALL_BANS, BOOK_LEGS["W_all"], book.index)
    chgA = dA_all.abs() > 1e-12
    res["M2"] = book_cell("M2_all_A_post", "비중표 A → W_all(2023 금지기, 사후)", "W_all", comb_all + dA_all, comb_all, "estimate", "사후(빠른 점검에서 본 구간). ON 0·C3Q 0", changed=chgA)
    res["M3"] = book_cell("M3_Y2024_A_post", "비중표 A → 2024 연수익(사후)", "Y2024", comb_all + dA_all, comb_all, "estimate", "사후. 연수익 = 누적", changed=chgA)
    res["M4"] = book_cell("M4_Y2025_A_post", "비중표 A → 2025 연수익(사후)", "Y2025", comb_all + dA_all, comb_all, "estimate", "사후. 연수익 = 누적", changed=chgA)
    res["M5"] = book_cell("M5_ex2026_A_post", "비중표 A → W_ex2026(사후)", "W_ex2026", comb_all + dA_all, comb_all, "estimate", "사후", changed=chgA)
    for k in ["M1", "M1b", "M1c", "M2", "M3", "M4", "M5"]:
        c = cells[-7 + ["M1", "M1b", "M1c", "M2", "M3", "M4", "M5"].index(k)]
        P(f"{c['id']}: CAGR {c['cagr_pct']:.2f}% MDD {c['mdd_pct']:.2f}% 누적변경 Δ(변경일 평균) {c['delta_bp']:.2f}bp t {c['t_delta']:.2f} | {c['notes'][-80:]}")

    # ---------- 진단: 다리별 Δ(HAC 10)
    P("\n[진단 다리별 Δ = 금지기 − 비금지, 활성일 net, HAC lag 10]")
    diag = {}
    for h in LEGS:
        s, _ = series(h)
        full = (s.index >= s.index[0]) & (s.index <= END)
        w_full = f"{fmtd(s.index[0])}~{fmtd(END)}"
        diag[h] = leg_cell(f"D_{h}_full", f"{h} 금지 더미 회귀(유효 전 구간)", h, full, w_full, ALL_BANS, "diagnostic", "활성일 net ~ D_ban, HAC(10)")
        wa = (s.index >= WIN["W_all"][0]) & (s.index <= END)
        leg_cell(f"D_{h}_Wall", f"{h} 금지 더미 회귀(W_all, 2023 금지기 1회)", h, wa, f"{WIN['W_all'][0]}~{WIN['W_all'][1]}", ALL_BANS, "diagnostic", "활성일 net ~ D_ban, HAC(10)")
        d = diag[h]
        P(f"{h}: Δ {d['delta']:.2f}bp t {d['t']:.2f} | 금지 n {d['n_ban']} {d['mean_ban']:.1f}bp / 비금지 n {d['n_non']} {d['mean_non']:.1f}bp")
    # C3VT30 금지기별
    s, _ = series("beomgu_C3VT30")
    for k, per in BANS.items():
        a = s[(s.pos != 0) & dummy(s.index, False, [per])]
        st = ev_stats(a.net.values, a.index)
        nonb = s[(s.pos != 0) & ~dummy(s.index, False, ALL_BANS)]
        add(f"D_C3VT30_ban{k}", f"beomgu_C3VT30 금지기 {k} 활성일", "KOSPI", f"{per[0]}~{per[1]}", st, "diagnostic",
            f"금지기 {k} 활성일만. 비금지 전 구간 평균 {nonb.net.mean()*1e4:.1f}bp(n {len(nonb)})", gross_bp=a.gross.mean() * 1e4 if len(a) else np.nan)
        P(f"C3VT30 금지기 {k}: n {st['n']} {st['net_bp']:.1f}bp t {st['t_nw']:.2f}")

    # ---------- 2×2 조건 분해(ON 창) + 무조건 밤 2002~
    P("\n[2×2 조건 분해: jehyeong_ON 조건 밤 vs 무조건 069500 밤]")
    unc_g = (_cal.open / _cal.close.shift(1) - 1).dropna()          # date = 청산일(T+1)
    unc = (unc_g - COST_BP * 1e-4).loc[:END]
    unc_g = unc_g.loc[:END]
    sON, _ = series("jehyeong_ON")
    on_act = sON[sON.pos != 0]
    won = f"{fmtd(sON.index[0])}~{fmtd(END)}"
    uw = unc.loc[sON.index[0]:END]
    d_on = dummy(on_act.index, True, ALL_BANS)
    d_u = dummy(uw.index, True, ALL_BANS)
    x22 = {}
    for lab, ser_, d, nm in [("ON", on_act.net, d_on, "jehyeong_ON 조건 밤"), ("unc", uw, d_u, "무조건 069500 밤(5bp)")]:
        for flag, tag in [(True, "ban"), (False, "nonban")]:
            sub = ser_[d == flag]
            st = ev_stats(sub.values, sub.index)
            x22[(lab, tag)] = st
            add(f"X_{lab}_{tag}", f"{nm} · {'금지기' if flag else '비금지'}", "KOSPI", won, st, "diagnostic",
                f"ON 유효 창 안, 진입일 기준 금지 더미. {'금지기' if flag else '비금지'} 밤", gross_bp=(sub.mean() + COST_BP * 1e-4) * 1e4 if len(sub) else np.nan)
        p, t = hac(ser_.values * 1e4, d.astype(float), lags=10)
        x22[(lab, "delta")] = (p[1], t[1])
        P(f"{nm}: 금지 n {int(d.sum())} {x22[(lab,'ban')]['net_bp']:.1f}bp / 비금지 n {int((~d).sum())} {x22[(lab,'nonban')]['net_bp']:.1f}bp | Δ {p[1]:.1f}bp t {t[1]:.2f}")
    # 조건 프리미엄: 무조건 밤 net ~ 1[ON 활성], 금지기 안/밖 따로
    sel = pd.Series(0.0, index=uw.index)
    sel[sel.index.isin(on_act.index)] = 1.0
    for flag, tag in [(True, "ban"), (False, "nonban")]:
        sub = uw[d_u == flag]
        p, t = hac(sub.values * 1e4, sel.reindex(sub.index).values, lags=10)
        st = dict(n=len(sub), net_bp=sub.mean() * 1e4, t_nw=hac(sub.values * 1e4)[1][0], win_pct=(sub > 0).mean() * 100)
        add(f"X_condprem_{tag}", f"조건 프리미엄(ON 선택 밤 − 비선택 밤) · {'금지기' if flag else '비금지'}", "KOSPI", won, st, "diagnostic",
            f"무조건 밤 net ~ 1[ON 활성], HAC(10). ON 활성 {int(sel.reindex(sub.index).sum())} / 전체 {len(sub)}", delta_bp=p[1], t_delta=t[1])
        x22[("prem", tag)] = (p[1], t[1])
        P(f"조건 프리미엄 {tag}: {p[1]:.1f}bp t {t[1]:.2f}")
    # 무조건 밤 2002~: 금지 4회 합산·개별·비금지
    P("\n[무조건 069500 밤 2002~, 금지 4회]")
    du_all = dummy(unc.index, True, ALL_BANS)
    wu = f"{fmtd(unc.index[0])}~{fmtd(END)}"
    pooled = unc[du_all]
    st = ev_stats(pooled.values, pooled.index)
    p, t = hac(unc.values * 1e4, du_all.astype(float), lags=10)
    add("U_unc_allbans", "무조건 069500 밤 · 금지 4회 합산", "KOSPI", wu, st, "diagnostic", "Δ = 금지 − 비금지(HAC 10), 2002-10-15~", gross_bp=(pooled.mean() + COST_BP * 1e-4) * 1e4, delta_bp=p[1], t_delta=t[1])
    P(f"합산: n {st['n']} {st['net_bp']:.1f}bp t {st['t_nw']:.2f} Δ {p[1]:.1f}bp t {t[1]:.2f}")
    unc_ban = {}
    for k, per in BANS.items():
        sub = unc[dummy(unc.index, True, [per])]
        st = ev_stats(sub.values, sub.index)
        unc_ban[k] = st
        add(f"U_unc_{k}", f"무조건 069500 밤 · 금지기 {k}", "KOSPI", f"{per[0]}~{per[1]}", st, "diagnostic", "진입일 기준", gross_bp=(sub.mean() + COST_BP * 1e-4) * 1e4)
        P(f"금지기 {k}: n {st['n']} {st['net_bp']:.1f}bp t {st['t_nw']:.2f}")
    sub = unc[~du_all]
    st = ev_stats(sub.values, sub.index)
    add("U_unc_nonban", "무조건 069500 밤 · 비금지 전 구간", "KOSPI", wu, st, "diagnostic", "2002-10-15~, 금지 4회 제외", gross_bp=(sub.mean() + COST_BP * 1e-4) * 1e4)
    P(f"비금지: n {st['n']} {st['net_bp']:.1f}bp t {st['t_nw']:.2f}")

    # ---------- V1: 금지기 ON 조건 해제 → 무조건 밤 롱(W_all, 사후)
    P("\n[V1 금지기 ON 조건 해제(무조건 069500 밤 롱), W_all 사후]")
    dv1 = pd.Series(0.0, index=book.index)
    nights = dummy(book.index, True, ALL_BANS)
    dv1[nights] = 0.5 * (unc.reindex(book.index)[nights] - sON.net.reindex(book.index).fillna(0.0)[nights])
    chg1 = pd.Series(nights, index=book.index)
    for wk, id_, nm in [("W_all", "V1_all_post", "V1 → W_all(사후)"), ("Y2024", "V1_Y2024_post", "V1 → 2024(사후)"),
                        ("Y2025", "V1_Y2025_post", "V1 → 2025(사후)"), ("W_ex2026", "V1_ex2026_post", "V1 → W_ex2026(사후)")]:
        mt = book_cell(id_, nm, wk, comb_all + dv1, comb_all, "estimate", f"금지기 밤 전부 069500 롱(5bp)로 교체, 밤 {int(nights[(book.index>=WIN[wk][0])&(book.index<=WIN[wk][1])].sum())}", changed=chg1)
        c = cells[-1]
        P(f"{id_}: CAGR {c['cagr_pct']:.2f}% MDD {c['mdd_pct']:.2f}% 누적 {mt['cum']*100:.2f}% Δ {c['delta_bp']:.2f}bp t {c['t_delta']:.2f}")

    # ---------- V2: 비중표 B
    P("\n[V2 비중표 B]")
    dB_long_oos = book_delta("B", [BANS["2020"]], BOOK_LEGS["W_long"], idx_long)
    dB_long_all = book_delta("B", ALL_BANS, BOOK_LEGS["W_long"], idx_long)
    dB_all = book_delta("B", ALL_BANS, BOOK_LEGS["W_all"], book.index)
    dB_kb = book_delta("B", ALL_BANS, BOOK_LEGS["W_KB03"], idx_kb)
    _, meB = leg_ret("beomgu_C3VT30", wser("beomgu_C3VT30", "B", ALL_BANS))
    sC, _ = series("beomgu_C3VT30")
    capped = ((wser("beomgu_C3VT30", "B", ALL_BANS) > 1) & (sC.pos != 0) & (meB < 1.5)).sum()
    P(f"C3VT30 1.5 적용 활성일 중 한도 절단 {int(capped)}일 / 적용 활성일 {int(((wser('beomgu_C3VT30','B',ALL_BANS)>1)&(sC.pos!=0)).sum())}")
    for id_, nm, wk, r, b, note in [
        ("V2_long_B_2020oos", "비중표 B → W_long, 2020 금지기만(표본외)", "W_long", comb_long + dB_long_oos, comb_long, "ON 없음, C3Q 0.5·C3VT30 1.5(한도)·blog_ON_Q 1.5→한도로 1.0"),
        ("V2_ban2020_B", "비중표 B, 2020 금지기 구간만 평가", "BAN2020", comb_long + dB_long_oos, comb_long, "≤300일"),
        ("V2_long_B_all", "비중표 B → W_long, 2020+2023(2023 사후)", "W_long", comb_long + dB_long_all, comb_long, "장부 결합 규칙용"),
        ("V2_all_B_post", "비중표 B → W_all(사후)", "W_all", comb_all + dB_all, comb_all, "사후"),
        ("V2_Y2024_B_post", "비중표 B → 2024(사후)", "Y2024", comb_all + dB_all, comb_all, "사후"),
        ("V2_Y2025_B_post", "비중표 B → 2025(사후)", "Y2025", comb_all + dB_all, comb_all, "사후"),
        ("V2_KB03_B", "비중표 B → W_KB03(C3VT30 1.5, 금지 4회)", "W_KB03", comb_kb + dB_kb, comb_kb, "코스피 B 단독, 2008·2011·2020·2023")]:
        chgB = (r - b).abs() > 1e-12
        mt = book_cell(id_, nm, wk, r, b, "estimate", note, changed=chgB)
        c = cells[-1]
        P(f"{id_}: CAGR {c['cagr_pct']:.2f}% MDD {c['mdd_pct']:.2f}% 누적 {mt['cum']*100:.2f}% Δ {c['delta_bp']:.2f}bp t {c['t_delta']:.2f}")

    # ---------- V3: jehyeong_ON 2010~ 재구성
    P("\n[V3 jehyeong_ON 재구성: 133690 미국 종가 보정 지표 > 0 ∧ 코스피 외인 T−1 순매수 > 0]")
    e133 = pd.read_csv(RAW / "etf/133690.csv", parse_dates=["date"]).set_index("date").close.sort_index()
    ndx = pd.read_csv(RAW / "us_etf/yf_i_NDX.csv", parse_dates=["date"]).set_index("date").close.sort_index()
    fx = pd.read_csv(RAW / "r15_external/fdr_usdkrw_daily.csv", index_col=0, parse_dates=True).sort_index()["Close"]
    frn = pd.read_csv(RAW / "naver_kospi_investor_daily.csv", parse_dates=["date"], encoding="utf-8-sig").set_index("date")["net_foreign"].sort_index()
    T = e133.index
    ndx_last = pd.Series(ndx.reindex(T - pd.Timedelta(days=1), method="ffill").values, index=T)   # T 이전 마지막 미국 종가
    fx_T = fx.reindex(T, method="ffill")
    sig = np.log(e133 / e133.shift(1)) - np.log(ndx_last / ndx_last.shift(1)) - np.log(fx_T / fx_T.shift(1))
    sig_nofx = np.log(e133 / e133.shift(1)) - np.log(ndx_last / ndx_last.shift(1))
    frn_prev = frn.reindex(CAL).shift(1).reindex(T)
    cond = (sig > 0) & (frn_prev > 0)
    cond_nofx = (sig_nofx > 0) & (frn_prev > 0)
    nxt = pd.Series(CAL, index=CAL).shift(-1).reindex(T)              # 청산일
    rec_dates = pd.DatetimeIndex(nxt[cond].dropna().values)
    rec = unc.reindex(rec_dates).dropna()
    rec = rec[(rec.index >= "2010-10-19") & (rec.index <= END)]
    rec_nofx_dates = pd.DatetimeIndex(nxt[cond_nofx].dropna().values)
    P(f"신호 일수 {int(cond.sum())} (FX 미포함 {int(cond_nofx.sum())}), 자료 결측 sig NaN {int(sig.isna().sum())} frn_prev NaN {int(frn_prev.isna().sum())}")
    # 재현 한계: ON 창 겹침
    on_set = set(on_act.index)
    rw = rec[(rec.index >= sON.index[0])]
    rec_set = set(rw.index)
    both = on_set & rec_set
    jac = len(both) / len(on_set | rec_set)
    rn_set = set(rec_nofx_dates[(rec_nofx_dates >= sON.index[0]) & (rec_nofx_dates <= END)])
    jac_nofx = len(on_set & rn_set) / len(on_set | rn_set)
    daily_on = sON.net.reindex(uw.index).fillna(0.0)
    daily_rec = rw.reindex(uw.index).fillna(0.0)
    corr = float(np.corrcoef(daily_on, daily_rec)[0, 1])
    v3_overlap = dict(n_on=len(on_set), n_rec=len(rec_set), n_both=len(both), jaccard=jac, jaccard_nofx=jac_nofx, corr=corr,
                      on_mean=on_act.net.mean() * 1e4, rec_mean=rw.mean() * 1e4, both_mean=rw.reindex(sorted(both)).mean() * 1e4 if both else np.nan)
    P(f"ON 창 겹침: ON {len(on_set)} 재구성 {len(rec_set)} 공통 {len(both)} 자카드 {jac:.2f}(FX 미포함 {jac_nofx:.2f}) 일수익 상관 {corr:.2f} | ON 평균 {v3_overlap['on_mean']:.1f}bp 재구성 평균 {v3_overlap['rec_mean']:.1f}bp")
    wr = f"{fmtd(rec.index[0])}~{fmtd(END)}"
    d_rec = dummy(rec.index, True, ALL_BANS)
    sub = rec[~d_rec]
    st = ev_stats(sub.values, sub.index)
    add("V3_recon_nonban", "재구성 ON · 비금지(2010-10~)", "KOSPI", wr, st, "diagnostic", "133690 프록시 재구성, 5bp, 금지 4회 제외", gross_bp=(sub.mean() + COST_BP * 1e-4) * 1e4)
    P(f"재구성 비금지: n {st['n']} {st['net_bp']:.1f}bp t {st['t_nw']:.2f} 연도 {st['yearly_pos']}")
    v3 = {}
    for k in ["2011", "2020", "2023"]:
        sub = rec[dummy(rec.index, True, [BANS[k]])]
        st = ev_stats(sub.values, sub.index)
        v3[k] = st
        add(f"V3_recon_{k}", f"재구성 ON · 금지기 {k}{'(사후)' if k == '2023' else ''}", "KOSPI", f"{BANS[k][0]}~{BANS[k][1]}", st, "diagnostic",
            f"재구성 규칙, 금지기 {k} 진입일. 부호 {'+' if st['net_bp'] > 0 else '−'}", gross_bp=(sub.mean() + COST_BP * 1e-4) * 1e4)
        P(f"재구성 금지기 {k}: n {st['n']} {st['net_bp']:.1f}bp t {st['t_nw']:.2f}")
    p, t = hac(rec.values * 1e4, d_rec.astype(float), lags=10)
    sub = rec[d_rec]
    st = ev_stats(sub.values, sub.index)
    add("V3_recon_allbans", "재구성 ON · 금지 3회 합산 Δ", "KOSPI", wr, st, "diagnostic", "Δ = 금지 − 비금지, HAC(10)", gross_bp=(sub.mean() + COST_BP * 1e-4) * 1e4, delta_bp=p[1], t_delta=t[1])
    v3["delta"] = (p[1], t[1])
    P(f"재구성 금지 합산: n {st['n']} {st['net_bp']:.1f}bp Δ {p[1]:.1f}bp t {t[1]:.2f}")
    st = ev_stats(rw.values, rw.index)
    p, t = hac(rw.values * 1e4, dummy(rw.index, True, ALL_BANS).astype(float), lags=10)
    add("V3_recon_ONwindow", "재구성 ON · ON 유효 창(실제 ON 과 대조)", "KOSPI", won, st, "diagnostic",
        f"실제 ON {len(on_set)}건 {v3_overlap['on_mean']:.1f}bp vs 재구성 {len(rec_set)}건, 공통 {len(both)}, 자카드 {jac:.2f}, 상관 {corr:.2f}", gross_bp=(rw.mean() + COST_BP * 1e-4) * 1e4, delta_bp=p[1], t_delta=t[1])
    trades.append(pd.DataFrame({"handle": "recon_ON_2010", "date": rec.index, "entry_date": entry(rec.index, True), "pos": 1.0,
                                "gross": unc_g.reindex(rec.index).values, "net": rec.values}))

    # ---------- V4: 베이시스 z·VKOSPI 통제
    P("\n[V4 베이시스 z(FUT−KPI200, 250일)·VKOSPI 통제, HAC 10]")
    fut = pd.read_csv(RAW / "etf/FUT.csv", parse_dates=["date"]).set_index("date").close
    kpi = pd.read_csv(RAW / "etf/KPI200.csv", parse_dates=["date"]).set_index("date").close
    basis = ((fut - kpi) / kpi * 100).dropna()
    z = ((basis - basis.rolling(250, min_periods=120).mean()) / basis.rolling(250, min_periods=120).std()).dropna()
    vk = pd.read_csv(RAW / "r15_external/kis_vkospi_daily.csv", parse_dates=["date"]).set_index("date").close.loc[:END]
    for k, per in BANS.items():
        b = basis.loc[per[0]:per[1]]
        if len(b):
            P(f"베이시스 평균 금지기 {k}: {b.mean():.3f}% (n {len(b)})")
    P(f"베이시스 평균 비금지(2014-07~): {basis[~in_periods(basis.index, ALL_BANS)].mean():.3f}% / 부분 재개 {basis.loc[PARTIAL[0][0]:PARTIAL[0][1]].mean():.3f}%")
    v4 = {}
    for h in LEGS:
        s, _ = series(h)
        a = s[s.pos != 0]
        ctrl = pd.DatetimeIndex(PREV.reindex(entry(a.index, LEGS[h][2])).values)   # 진입일 전 거래일
        zc, vc = z.reindex(ctrl).values, vk.reindex(ctrl).values
        ok = ~(np.isnan(zc) | np.isnan(vc))
        a2 = a[ok]
        d = dummy(a2.index, LEGS[h][2], ALL_BANS).astype(float)
        X = np.column_stack([d, zc[ok], vc[ok]])
        p, t = hac(a2.net.values * 1e4, X, lags=10)
        st = ev_stats(a2.net.values, a2.index)
        w4 = f"{fmtd(a2.index[0])}~{fmtd(a2.index[-1])}"
        add(f"V4_{h}", f"{h} 금지 더미 + 베이시스 z + VKOSPI 통제", LEGS[h][0], w4, st, "diagnostic",
            f"net ~ D_ban + z_basis(진입 전일) + VKOSPI(진입 전일), HAC(10). b_z {p[2]:.2f}bp/σ(t {t[2]:.2f}) b_vk {p[3]:.3f}bp/pt(t {t[3]:.2f}). 통제 없는 Δ {diag[h]['delta']:.2f}bp(t {diag[h]['t']:.2f}, 전 구간)",
            gross_bp=a2.gross.mean() * 1e4 if h != "beomgu_C3Q" else np.nan, delta_bp=p[1], t_delta=t[1])
        v4[h] = (p[1], t[1])
        P(f"{h}: n {len(a2)} Δ_ban(통제) {p[1]:.2f}bp t {t[1]:.2f} | z {p[2]:.2f}(t {t[2]:.2f}) vk {p[3]:.3f}(t {t[3]:.2f})")

    # ---------- V5: 부분 재개 더미 분리
    P("\n[V5 D_ban + D_partial(2021-05-03~2023-11-05), HAC 10]")
    v5 = {}
    for h in LEGS:
        s, _ = series(h)
        a = s[s.pos != 0]
        d = dummy(a.index, LEGS[h][2], ALL_BANS).astype(float)
        dp = dummy(a.index, LEGS[h][2], PARTIAL).astype(float)
        p, t = hac(a.net.values * 1e4, np.column_stack([d, dp]), lags=10)
        st = ev_stats(a.net.values, a.index)
        add(f"V5_{h}", f"{h} 금지 더미 + 부분 재개 더미", LEGS[h][0], f"{fmtd(a.index[0])}~{fmtd(a.index[-1])}", st, "diagnostic",
            f"기준 = 전면 허용기. D_ban {p[1]:.2f}bp(t {t[1]:.2f}), D_partial {p[2]:.2f}bp(t {t[2]:.2f}), 부분 재개 활성일 {int(dp.sum())}, 금지 {int(d.sum())}, 허용 {int(((d==0)&(dp==0)).sum())}",
            gross_bp=a.gross.mean() * 1e4 if h != "beomgu_C3Q" else np.nan, delta_bp=p[2], t_delta=t[2])
        v5[h] = (p[1], t[1], p[2], t[2])
        P(f"{h}: D_ban {p[1]:.2f}(t {t[1]:.2f}) D_partial {p[2]:.2f}(t {t[2]:.2f}) 기준(허용기) 평균 {p[0]:.2f}bp")

    # ---------- V6: C3 괴리 몫·갭 몫
    P("\n[V6 C3 분해: 괴리 몫 = pos×(ETF oc − 지수 oc), 갭 몫 = pos×지수 oc, 금지기 내외]")
    k200 = pd.read_csv(RAW / "k200_index_daily.csv", parse_dates=["date"], encoding="utf-8-sig").set_index("date")
    kq = pd.read_csv(RAW / "etf_extra/KQ150_index_kis.csv", parse_dates=["date"]).set_index("date").loc["2015-07-13":]
    e229 = pd.read_csv(RAW / "etf/229200.csv", parse_dates=["date"]).set_index("date")
    oc = lambda d: (d.close / d.open - 1)
    v6 = {}
    for h, etf, idx_ in [("beomgu_C3VT30", _cal, k200), ("beomgu_C3Q", e229, kq)]:
        s, _ = series(h)
        a = s[s.pos != 0]
        oce, oci = oc(etf).reindex(a.index), oc(idx_).reindex(a.index)
        ok = ~(oce.isna() | oci.isna())
        a, oce, oci = a[ok], oce[ok], oci[ok]
        comp = {"disc": a.pos * (oce - oci), "gap": a.pos * oci}
        resid = a.gross - a.pos * oce
        d = dummy(a.index, False, ALL_BANS).astype(float)
        for cname, lab in [("disc", "괴리 몫"), ("gap", "갭 몫")]:
            c = comp[cname]
            p, t = hac(c.values * 1e4, d, lags=10)
            st = ev_stats(c.values, a.index)
            add(f"V6_{h.split('_')[1]}_{cname}", f"{h} {lab} 금지기 내외", LEGS[h][0], f"{fmtd(a.index[0])}~{fmtd(a.index[-1])}", st, "diagnostic",
                f"{lab}: 금지 n {int(d.sum())} {c[d==1].mean()*1e4:.1f}bp / 비금지 n {int((d==0).sum())} {c[d==0].mean()*1e4:.1f}bp. 잔차(gross − pos×ETF oc) 평균 {resid.mean()*1e4:.1f}bp(금지 {resid[d==1].mean()*1e4:.1f}, 비금지 {resid[d==0].mean()*1e4:.1f}). 수익률만 사용(조정가)",
                delta_bp=p[1], t_delta=t[1])
            v6[(h, cname)] = (c[d == 1].mean() * 1e4, c[d == 0].mean() * 1e4, p[1], t[1], int(d.sum()), int((d == 0).sum()))
            P(f"{h} {lab}: 금지 {c[d==1].mean()*1e4:.1f}bp / 비금지 {c[d==0].mean()*1e4:.1f}bp Δ {p[1]:.1f} t {t[1]:.2f}")

    # ---------- trades.csv
    for h in LEGS:
        s, _ = series(h)
        a = s[s.pos != 0]
        e = entry(a.index, LEGS[h][2])
        trades.append(pd.DataFrame({"handle": h, "date": a.index, "entry_date": e, "pos": a.pos.values, "gross": a.gross.values, "net": a.net.values}))
    trades.append(pd.DataFrame({"handle": "uncond_069500_night", "date": unc.index, "entry_date": entry(unc.index, True), "pos": 1.0,
                                "gross": unc_g.values, "net": unc.values}))
    tr = pd.concat(trades, ignore_index=True)
    tr["ban_id"] = ban_id(pd.DatetimeIndex(tr.entry_date))
    tr["partial"] = in_periods(pd.DatetimeIndex(tr.entry_date), PARTIAL).astype(int)
    tr.to_csv(OUT / "trades.csv", index=False, float_format="%.8f", encoding="utf-8-sig")

    # ---------- cells.csv
    cdf = pd.DataFrame(cells)
    cdf.to_csv(OUT / "cells.csv", index=False, float_format="%.4f", encoding="utf-8-sig")
    P(f"\ncells {len(cdf)} trades {len(tr)}")

    # ---------- 오버레이(포워드 규칙 기록용)
    ov = pd.DataFrame({"date": CAL})
    for h in ["jehyeong_ON", "beomgu_C3Q", "beomgu_C3VT30", "blog_ON_Q"]:
        w = np.ones(len(CAL))
        w[dummy(CAL, LEGS[h][2], ALL_BANS)] = TABLES["A"][h]
        ov[f"w_{h}"] = w
    ov.to_csv(SER / "overlay_06.csv", index=False, float_format="%.2f")
    meta = {"handle": "overlay_06", "slot": "overlay", "market": "KOSPI+KOSDAQ", "instrument": "13차 다리 비중(jehyeong_ON, beomgu_C3Q, beomgu_C3VT30, blog_ON_Q)",
            "timing": "date = 13차 series 의 date(밤 다리는 청산일 → 진입일 기준 금지 판정)", "cost_bp": "비용 변경 없음(m ≤ 1 은 m×net)",
            "first_valid": fmtd(CAL[0]), "last_valid": fmtd(CAL[-1]), "n_active": {c: int((ov[c] != 1).sum()) for c in ov.columns if c != "date"},
            "source_files": ["round13_slot_book/series/*.csv", "hyfe/data/raw/etf/069500.csv(달력)"],
            "repro": {"bans": BANS, "table": "A", "rule": "금지기 jehyeong_ON 0, beomgu_C3Q 0, C3VT30·blog_ON_Q 1.0"},
            "notes": "비중표 A 포워드 규칙 기록용. 결합(최종 장부)에는 쓰지 않음(사전 등록: 통과 여부와 무관하게 다음 금지 공표 전까지 장부 수치 미반영). 금지기 = 금융위 시행일~재개 전일(재개는 사전 공표)."}
    (SER / "overlay_06.meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")

    # ---------- 판정
    g = lambda i: cdf.set_index("id").loc[i]
    orig_long, m1, m1c = g("B_long_orig"), g("M1_long_A_2020oos"), g("M1c_long_A_all")
    orig_all, m2 = g("B_all_orig"), g("M2_all_A_post")
    EPS = 1e-6   # %p, 부동소수 오차 → '불변'
    better = lambda a, b: a > b + EPS
    nonworse = lambda a, b: a >= b - EPS
    state = lambda a, b: "개선" if better(a, b) else ("불변" if abs(a - b) <= EPS else "악화")
    c1 = bool(better(m1.cagr_pct, orig_long.cagr_pct) and better(m1.mdd_pct, orig_long.mdd_pct))
    c1_relax = bool(better(m1.cagr_pct, orig_long.cagr_pct) and nonworse(m1.mdd_pct, orig_long.mdd_pct))
    tON, tC3Q = diag["jehyeong_ON"]["t"], diag["beomgu_C3Q"]["t"]
    c2 = bool(tON <= -2 and tC3Q <= -2)
    c3 = bool(better(m2.cagr_pct, orig_all.cagr_pct) and better(m2.mdd_pct, orig_all.mdd_pct))
    c3_relax = bool(better(m2.cagr_pct, orig_all.cagr_pct) and nonworse(m2.mdd_pct, orig_all.mdd_pct))
    cum = lambda i: BM[i]["cum"] * 100
    c4 = bool(better(cum("M3_Y2024_A_post"), cum("B_Y2024_orig")) and better(cum("M4_Y2025_A_post"), cum("B_Y2025_orig")))
    book_rule = bool(c3 and better(m1c.cagr_pct, orig_long.cagr_pct) and better(m1c.mdd_pct, orig_long.mdd_pct) and c4)
    book_rule_relax = bool(c3_relax and better(m1c.cagr_pct, orig_long.cagr_pct) and nonworse(m1c.mdd_pct, orig_long.mdd_pct) and c4)
    judging = dict(main_cell_id="M1_long_A_2020oos", cond1_oos_long_cagr_mdd=c1, cond1_relaxed_mdd_nonworse=c1_relax,
                   cond2_leg_t=dict(jehyeong_ON=tON, beomgu_C3Q=tC3Q, passed=c2), cond3_wall=c3, cond3_relaxed=c3_relax, cond4_y2024_y2025=c4,
                   passed=bool(c1 and c2 and c3 and c4), passed_relaxed=bool(c1_relax and c2 and c3_relax and c4),
                   book_rule_passed=book_rule, book_rule_relaxed=book_rule_relax, grade="estimate",
                   numbers=dict(W_long_orig=(orig_long.cagr_pct, orig_long.mdd_pct), W_long_A2020=(m1.cagr_pct, m1.mdd_pct), W_long_A_all=(m1c.cagr_pct, m1c.mdd_pct),
                                W_all_orig=(orig_all.cagr_pct, orig_all.mdd_pct), W_all_A=(m2.cagr_pct, m2.mdd_pct),
                                Y2024=(cum("B_Y2024_orig"), cum("M3_Y2024_A_post")), Y2025=(cum("B_Y2025_orig"), cum("M4_Y2025_A_post"))))
    (OUT / "judging.json").write_text(json.dumps(judging, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    P("\n[판정]", json.dumps(judging, ensure_ascii=False, default=float))

    # ---------- 그림
    C1, C2, C3, C4, RED, INK = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e34948", "#52514e"
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw={"width_ratios": [3, 2]})
    names = list(LEGS)
    xs = np.arange(len(names))
    bw = 0.36
    ax.bar(xs - bw / 2, [diag[h]["mean_non"] for h in names], bw, color=C1, label="비금지 평균", linewidth=0)
    ax.bar(xs + bw / 2, [diag[h]["mean_ban"] for h in names], bw, color=C2, label="금지기 평균", linewidth=0)
    for i, h in enumerate(names):
        y = max(diag[h]["mean_non"], diag[h]["mean_ban"], 0)
        ax.annotate(f"Δ {diag[h]['delta']:.0f}bp\nt {diag[h]['t']:.1f}\nn {diag[h]['n_ban']}/{diag[h]['n_non']}", (i, y), xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=8, color=INK)
    ax.set_xticks(xs)
    ax.set_xticklabels(names, rotation=20, ha="right", fontsize=9)
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_ylabel("활성일 평균 순수익 bp")
    ax.set_title("13차 다리별 금지기 내외 활성일 평균(유효 전 구간, HAC lag 10 t)", fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.set_ylim(top=ax.get_ylim()[1] * 1.35)
    labs = ["ON 비금지", "ON 금지기", "무조건 비금지", "무조건 금지기"]
    vals = [x22[("ON", "nonban")]["net_bp"], x22[("ON", "ban")]["net_bp"], x22[("unc", "nonban")]["net_bp"], x22[("unc", "ban")]["net_bp"]]
    ns = [x22[("ON", "nonban")]["n"], x22[("ON", "ban")]["n"], x22[("unc", "nonban")]["n"], x22[("unc", "ban")]["n"]]
    ax2.bar(np.arange(4), vals, 0.6, color=[C1, C2, C1, C2], linewidth=0)
    for i, (v, n) in enumerate(zip(vals, ns)):
        ax2.annotate(f"{v:.1f}bp\nn {n}", (i, max(v, 0)), xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=8, color=INK)
    ax2.set_xticks(np.arange(4))
    ax2.set_xticklabels(labs, fontsize=9)
    ax2.axhline(0, color=INK, lw=0.8)
    ax2.set_title(f"2×2 조건 분해 {won}\nΔON {x22[('ON','delta')][0]:.0f}bp(t {x22[('ON','delta')][1]:.1f}) Δ무조건 {x22[('unc','delta')][0]:.0f}bp(t {x22[('unc','delta')][1]:.1f})", fontsize=10, loc="left")
    ax2.set_ylim(top=ax2.get_ylim()[1] * 1.3)
    for a_ in (ax, ax2):
        a_.grid(alpha=0.25, axis="y")
        for sp in ("top", "right"):
            a_.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "fig1_leg_ban_bars.png", dpi=130)
    plt.close(fig)

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(11, 8))
    for a_, lines, title in [(ax, [("원 장부", comb_long, C1), ("A 2020 표본외", comb_long + dA_long_oos, C2), ("A 2020+2023", comb_long + dA_long_all, C3), ("B 2020+2023", comb_long + dB_long_all, C4)],
                              f"W_long {WIN['W_long'][0]}~{WIN['W_long'][1]} 합산 장부(코스피 B + 코스닥 A·B), 로그 누적"),
                             (ax2, [("원 장부", comb_all, C1), ("A 사후", comb_all + dA_all, C2), ("B 사후", comb_all + dB_all, C3), ("V1 무조건 밤", comb_all + dv1, C4)],
                              f"W_all {WIN['W_all'][0]}~{WIN['W_all'][1]} 합산 장부(여섯 다리), 로그 누적")]:
        for lab, r, col in lines:
            eq = (1 + r).cumprod()
            mt = metrics(r)
            a_.plot(eq.index, eq, lw=1.6, color=col, label=f"{lab}: CAGR {mt['cagr']*100:.1f}% MDD {mt['mdd']*100:.1f}%")
        lo, hi = a_.get_xlim()
        for k, (s_, e_) in BANS.items():
            if pd.Timestamp(e_) >= lines[0][1].index[0]:
                a_.axvspan(max(pd.Timestamp(s_), lines[0][1].index[0]), pd.Timestamp(e_), color=INK, alpha=0.10, lw=0)
                a_.annotate(f"금지 {k}", (max(pd.Timestamp(s_), lines[0][1].index[0]), 1.0), xytext=(2, -12), textcoords="offset points", fontsize=8, color=INK)
        a_.set_yscale("log")
        a_.set_title(title, fontsize=10, loc="left")
        a_.legend(frameon=False, fontsize=8, loc="upper left")
        a_.grid(alpha=0.25)
        for sp in ("top", "right"):
            a_.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "fig2_equity_ban_shading.png", dpi=130)
    plt.close(fig)

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))
    labs = ["비금지", "2011", "2020", "2023(사후)"]
    stv = [cdf.set_index("id").loc["V3_recon_nonban"], v3["2011"], v3["2020"], v3["2023"]]
    vals = [float(stv[0]["net_bp"])] + [x["net_bp"] for x in stv[1:]]
    ns = [int(stv[0]["n"])] + [x["n"] for x in stv[1:]]
    ax.bar(np.arange(4), vals, 0.6, color=[C1, C2, C2, C2], linewidth=0)
    for i, (v, n) in enumerate(zip(vals, ns)):
        ax.annotate(f"{v:.1f}bp\nn {n}", (i, max(v, 0)), xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=8, color=INK)
    ax.set_xticks(np.arange(4))
    ax.set_xticklabels(labs)
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_title(f"V3 재구성 ON(133690 ∧ 외인 T-1) 2010~ 금지기별 평균 bp\nON 창 자카드 {jac:.2f}, 상관 {corr:.2f}", fontsize=10, loc="left")
    ax.set_ylim(top=ax.get_ylim()[1] * 1.3)
    keys = [("beomgu_C3VT30", "disc"), ("beomgu_C3VT30", "gap"), ("beomgu_C3Q", "disc"), ("beomgu_C3Q", "gap")]
    labs2 = ["C3VT30 괴리", "C3VT30 갭", "C3Q 괴리", "C3Q 갭"]
    xs = np.arange(4)
    ax2.bar(xs - 0.18, [v6[k][1] for k in keys], 0.36, color=C1, label="비금지", linewidth=0)
    ax2.bar(xs + 0.18, [v6[k][0] for k in keys], 0.36, color=C2, label="금지기", linewidth=0)
    for i, k in enumerate(keys):
        ax2.annotate(f"Δ {v6[k][2]:.0f} t {v6[k][3]:.1f}", (i, max(v6[k][0], v6[k][1], 0)), xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=8, color=INK)
    ax2.set_xticks(xs)
    ax2.set_xticklabels(labs2)
    ax2.axhline(0, color=INK, lw=0.8)
    ax2.legend(frameon=False, fontsize=9)
    ax2.set_title("V6 C3 활성일 수익 분해(괴리 몫·갭 몫) 금지기 내외 평균 bp", fontsize=10, loc="left")
    ax2.set_ylim(top=ax2.get_ylim()[1] * 1.3)
    for a_ in (ax, ax2):
        a_.grid(alpha=0.25, axis="y")
        for sp in ("top", "right"):
            a_.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "fig3_v3_v6.png", dpi=130)
    plt.close(fig)

    # ---------- notes.md (하이픈 금지: 날짜는 YYYY.MM.DD, 음수는 −)
    import re

    def nd(s):   # 날짜의 하이픈만 점으로, 나머지 하이픈(음수·지수)은 − 로
        s = re.sub(r"(\d{4})-(\d{2})-(\d{2})", r"\1.\2.\3", str(s))
        s = re.sub(r"(\d{4})-(\d{2})(?!\d)", r"\1.\2", s)
        return s.replace("-", "−")
    f1 = lambda x, d=1: ("미정" if pd.isna(x) else f"{x:.{d}f}").replace("-", "−")
    row = lambda i: cdf.set_index("id").loc[i]
    L = []
    L.append("# 06_ban_regime · 공매도 전면금지 레짐 분해 · 다리 비중 오버레이(진단 + 포워드 규칙)")
    L.append("")
    L.append("## 자료 확보 단계(금융위원회 보도자료, WebSearch·WebFetch 2026.10.30)")
    L.append("• 2008 금지: 금융위 정책마당 보도자료 「공매도 및 자사주 관련 제도 개선」 2008.09.30 게시(첨부 080930_시장조치) → 시행 2008.10.01. 금융위 2009.05.20 자료 「비금융주에 대한 공매도 제한조치 해지」(https://www.fsc.go.kr/no010103/24345)에 「작년 10월 1일부터」 명시, 비금융주 2009.06.01 재개 → D_ban 2008.10.01~2009.05.31. 2008.09.30 민감도 셀은 공식 일자 확정으로 생략.")
    L.append("• 2011 금지: 금융위 영문 보도자료 목록 2011.08.09 「Temporary Ban on Short Selling」(https://www.fsc.go.kr/eng/pr010101?curPage=72) → 시행 2011.08.10. 재개: 금융위 보고영상(https://fsc.go.kr/no010103/24438)에 「금지 후 11월 10일부터」 수치, 보험연구원 주간 보고(2011.11.14)에 서면회의 의결(비금융주 11.10 해제) → D_ban 2011.08.10~2011.11.09. 11.09 보도자료 원문은 미열람(간접 확인).")
    L.append("• 2020 금지: 금융위 2020.08.27 「공매도 금지 및 자기주식 취득한도 확대를 6개월 연장」(https://fsc.go.kr/no010101/74509)에 「20.3.16~9.15(20.3.13 의결)」 과 연장 2021.03.15 까지 명시. 금융위 2021.02.03 「5월 2일까지 금지 연장, 5월 3일부터 코스피200·코스닥150 부분 재개」(https://www.fsc.go.kr/no010101/75290) → D_ban 2020.03.16~2021.05.02.")
    L.append("• 2023 금지: 금융위 2023.11.05 「내년 상반기까지 공매도 전면금지」(https://www.fsc.go.kr/no010101/81013) → 시행 2023.11.06. 2024.06.13 「공매도 금지 조치를 25.3.30(일)까지 연장」(https://www.fsc.go.kr/no010101/82454). 2025.03.21 「3월 31일부터 공매도를 전면 재개」(https://www.fsc.go.kr/no010101/84216) → D_ban 2023.11.06~2025.03.30.")
    L.append("• 네 구간 모두 설계 전문의 일자와 일치. 재개 일자는 모두 사전 공표(2009.05.20, 2011.11.09, 2021.02.03, 2025.03.21) → 포워드 적용 가능. 밤 다리(date=청산일)는 진입일 기준으로 금지 판정.")
    L.append("")
    L.append("## 자료 점검")
    for l_ in log[1:19]:
        L.append("• " + nd(l_))
    L.append("• 다른 역할이 만든 파일: kis_vkospi_daily.csv(2008.12.22~2026.10.30, 2026.10.30 행은 통제 변수 지연 때문에 쓰이지 않음), fdr_usdkrw_daily.csv(2003.12~, 열 Close, 영업일 결측은 전일값), yf_i_NDX.csv(미국 거래일, T 이전 마지막 종가), KQ150_index_kis.csv(2015.07.13 이후만 사용). 열·범위 직접 확인.")
    L.append("• beomgu_C3Q 의 gross 는 근사(13차 meta: net 과 최대 오차 1.44e−02)라 비중 m ≤ 1 은 net 에만 곱함(gross 미사용).")
    L.append("")
    L.append("## 규칙(사전 등록 그대로)")
    L.append("• 진단: 활성일 net(bp) ~ 상수 + D_ban, HAC(maxlags 10). 2×2: jehyeong_ON 조건 밤 vs 무조건 069500 밤(종가→익일 시가, 왕복 5bp). 비중표 A: 금지기 jehyeong_ON 0·beomgu_C3Q 0·나머지 1.0. 비중표 B: ON·C3Q 0.5, C3VT30·blog_ON_Q 1.5(ETF 한도 롱 2·코스닥 숏 1 로 절단 → blog_ON_Q 는 1.0 으로 절단, C3VT30 은 min(1.5, 2/w)).")
    L.append("• 비중 적용 수익: m ≤ 1 → m×net. m > 1 → m×gross − 5bp×min(m×|pos|, 1)(C3 비용 규칙). 장부 = 13차 조립 그대로(코스피 50% + 코스닥 50%, 겹침 규칙 유지, C3Q 0 인 날 jehyeong_KQ 복원 안 함).")
    L.append("• 창: W_long 2016.10.04~2026.09.17(2444일, 코스피 B + 코스닥 A·B), W_all 2022.12.13~2026.09.17(919일), W_ex2026, 2024·2025, W_KB03 2003.01.02~(C3VT30 단독). 모든 창 2026.09.17 종료.")
    L.append("• 판정: 표본외 금지기(2020~2021) 장부 CAGR·MDD 동시 개선 ∧ 비중을 바꾼 다리(ON·C3Q) Δ HAC |t| ≥ 2(부호 음) ∧ W_all 결합 CAGR·MDD 개선 ∧ 2024·2025 개선(사후). 등급 estimate 고정. 통과해도 다음 금지 공표 전까지 장부 미반영.")
    L.append("")
    L.append("## 주 셀 결과(비중표 A)")
    bl, m1r, m1b, m1cr, ba, m2r = row("B_long_orig"), row("M1_long_A_2020oos"), row("M1b_ban2020_A"), row("M1c_long_A_all"), row("B_all_orig"), row("M2_all_A_post")
    bb20 = row("B_ban2020_orig")
    L.append(f"• W_long 원 장부: CAGR {f1(bl.cagr_pct)}% MDD {f1(bl.mdd_pct)}%(MDD일 {nd(bl.notes.split('MDD일 ')[1].split(' ')[0])}), n {int(bl.n)}일.")
    L.append(f"• [주 셀] 비중표 A 를 2020.03.16~2021.05.02(진입일 기준, 표본외)에만 적용: CAGR {f1(m1r.cagr_pct)}% MDD {f1(m1r.mdd_pct)}%(MDD일 {nd(m1r.notes.split('MDD일 ')[1].split(' ')[0])}). 비중 변경일 {int(dA_long_oos.abs().gt(1e-12).sum())}일, 변경일 장부 Δ {f1(m1r.delta_bp, 2)}bp/일(HAC10 t {f1(m1r.t_delta, 2)}). CAGR {state(m1r.cagr_pct, bl.cagr_pct)}, MDD {state(m1r.mdd_pct, bl.mdd_pct)} → 동시 개선 {'충족' if c1 else '미충족'}.")
    L.append(f"• 2020 금지기 구간만(2020.03.16~2021.05.02, {int(bb20.n)}일, estimate 급): 원 장부 CAGR {f1(bb20.cagr_pct)}% MDD {f1(bb20.mdd_pct)}% → A 적용 CAGR {f1(m1b.cagr_pct)}% MDD {f1(m1b.mdd_pct)}%.")
    L.append(f"• W_long 에 2020+2023 둘 다 적용(2023 사후 포함, 장부 결합 규칙용): CAGR {f1(m1cr.cagr_pct)}% MDD {f1(m1cr.mdd_pct)}%.")
    L.append(f"• W_all(사후, 2023.11.06~2025.03.30): 원 CAGR {f1(ba.cagr_pct)}% MDD {f1(ba.mdd_pct)}%(MDD일 {nd(ba.notes.split('MDD일 ')[1].split(' ')[0])}) → A CAGR {f1(m2r.cagr_pct)}% MDD {f1(m2r.mdd_pct)}%. 변경일 {int(chgA.sum())}일 Δ {f1(m2r.delta_bp, 2)}bp/일(t {f1(m2r.t_delta, 2)}). MDD 는 2026 구간의 낙폭이라 2023 금지기 비중으로는 바뀌지 않음.")
    L.append(f"• 연수익(누적, 사후): 2024 {f1(cum('B_Y2024_orig'), 2)}% → {f1(cum('M3_Y2024_A_post'), 2)}%, 2025 {f1(cum('B_Y2025_orig'), 2)}% → {f1(cum('M4_Y2025_A_post'), 2)}%. W_ex2026 CAGR {f1(row('B_ex2026_orig').cagr_pct)}% → {f1(row('M5_ex2026_A_post').cagr_pct)}%, MDD {f1(row('B_ex2026_orig').mdd_pct)}% → {f1(row('M5_ex2026_A_post').mdd_pct)}%.")
    L.append("")
    L.append("## 진단(다리별 Δ = 금지기 − 비금지, 활성일 net, HAC lag 10, 유효 전 구간 ~2026.09.17)")
    for h in LEGS:
        d = diag[h]
        L.append(f"• {h}: 금지기 n {d['n_ban']} 평균 {f1(d['mean_ban'])}bp / 비금지 n {d['n_non']} 평균 {f1(d['mean_non'])}bp, Δ {f1(d['delta'])}bp t {f1(d['t'], 2)}. W_all 창 셀은 cells.csv D_{h}_Wall.")
    for k in BANS:
        r_ = row(f"D_C3VT30_ban{k}")
        L.append(f"• beomgu_C3VT30 금지기 {k}: n {int(r_.n)} 평균 {f1(r_.net_bp)}bp t {f1(r_.t_nw, 2)}.")
    L.append("")
    L.append(f"## 2×2 조건 분해(ON 유효 창 {nd(won)})")
    L.append(f"• jehyeong_ON 조건 밤: 금지기 n {x22[('ON','ban')]['n']} {f1(x22[('ON','ban')]['net_bp'])}bp(t {f1(x22[('ON','ban')]['t_nw'], 2)}) / 비금지 n {x22[('ON','nonban')]['n']} {f1(x22[('ON','nonban')]['net_bp'])}bp(t {f1(x22[('ON','nonban')]['t_nw'], 2)}), Δ {f1(x22[('ON','delta')][0])}bp t {f1(x22[('ON','delta')][1], 2)}.")
    L.append(f"• 무조건 069500 밤(5bp): 금지기 n {x22[('unc','ban')]['n']} {f1(x22[('unc','ban')]['net_bp'])}bp(t {f1(x22[('unc','ban')]['t_nw'], 2)}) / 비금지 n {x22[('unc','nonban')]['n']} {f1(x22[('unc','nonban')]['net_bp'])}bp(t {f1(x22[('unc','nonban')]['t_nw'], 2)}), Δ {f1(x22[('unc','delta')][0])}bp t {f1(x22[('unc','delta')][1], 2)}.")
    L.append(f"• 조건 프리미엄(ON 선택 밤 − 비선택 밤, HAC10): 금지기 {f1(x22[('prem','ban')][0])}bp(t {f1(x22[('prem','ban')][1], 2)}), 비금지 {f1(x22[('prem','nonban')][0])}bp(t {f1(x22[('prem','nonban')][1], 2)}).")
    ua = row("U_unc_allbans")
    L.append(f"• 무조건 밤 2002.10.15~: 금지 4회 합산 n {int(ua.n)} {f1(ua.net_bp)}bp(t {f1(ua.t_nw, 2)}), Δ(대 비금지) {f1(ua.delta_bp)}bp t {f1(ua.t_delta, 2)}. 금지기별: " + ", ".join(f"{k} n {unc_ban[k]['n']} {f1(unc_ban[k]['net_bp'])}bp(t {f1(unc_ban[k]['t_nw'], 2)})" for k in BANS) + f". 비금지 n {int(row('U_unc_nonban').n)} {f1(row('U_unc_nonban').net_bp)}bp(t {f1(row('U_unc_nonban').t_nw, 2)}).")
    L.append("")
    L.append("## 변형(보고용, 주 셀 불변)")
    v1a = row("V1_all_post")
    L.append(f"• V1 금지기 ON 조건 해제(무조건 069500 밤 롱, W_all 사후): CAGR {f1(v1a.cagr_pct)}% MDD {f1(v1a.mdd_pct)}%, 2024 {f1(cum('V1_Y2024_post'), 2)}%, 2025 {f1(cum('V1_Y2025_post'), 2)}%, W_ex2026 CAGR {f1(row('V1_ex2026_post').cagr_pct)}%. 변경 밤 Δ {f1(v1a.delta_bp, 2)}bp/일(t {f1(v1a.t_delta, 2)}).")
    L.append(f"• V2 비중표 B: W_long 2020 표본외 CAGR {f1(row('V2_long_B_2020oos').cagr_pct)}% MDD {f1(row('V2_long_B_2020oos').mdd_pct)}%; W_long 전체 금지기 {f1(row('V2_long_B_all').cagr_pct)}%/{f1(row('V2_long_B_all').mdd_pct)}%; W_all 사후 {f1(row('V2_all_B_post').cagr_pct)}%/{f1(row('V2_all_B_post').mdd_pct)}%; 2024 {f1(cum('V2_Y2024_B_post'), 2)}% 2025 {f1(cum('V2_Y2025_B_post'), 2)}%; W_KB03(C3VT30 1.5, 금지 4회) {f1(row('V2_KB03_B').cagr_pct)}%/{f1(row('V2_KB03_B').mdd_pct)}%(원 {f1(row('B_KB03_orig').cagr_pct)}%/{f1(row('B_KB03_orig').mdd_pct)}%). C3VT30 1.5 적용 활성일 중 한도 절단 {int(capped)}일.")
    L.append(f"• V3 재구성(133690 일수익 − 직전 미국 NDX 종가 변화 − USDKRW 변화 > 0 ∧ 코스피 외인 T−1 순매수 > 0, 069500 종가→익일 시가 5bp, {nd(wr)}): 비금지 n {int(row('V3_recon_nonban').n)} {f1(row('V3_recon_nonban').net_bp)}bp(t {f1(row('V3_recon_nonban').t_nw, 2)}, 연도 부호 {row('V3_recon_nonban').yearly_pos}); 금지기 2011 n {v3['2011']['n']} {f1(v3['2011']['net_bp'])}bp(t {f1(v3['2011']['t_nw'], 2)}), 2020 n {v3['2020']['n']} {f1(v3['2020']['net_bp'])}bp(t {f1(v3['2020']['t_nw'], 2)}), 2023(사후) n {v3['2023']['n']} {f1(v3['2023']['net_bp'])}bp(t {f1(v3['2023']['t_nw'], 2)}); 금지 합산 Δ {f1(v3['delta'][0])}bp t {f1(v3['delta'][1], 2)}.")
    L.append(f"• V3 재현 한계: ON 창에서 실제 ON {v3_overlap['n_on']}건(평균 {f1(v3_overlap['on_mean'])}bp) vs 재구성 {v3_overlap['n_rec']}건(평균 {f1(v3_overlap['rec_mean'])}bp), 공통 {v3_overlap['n_both']}건(공통일 평균 {f1(v3_overlap['both_mean'])}bp), 자카드 {f1(v3_overlap['jaccard'], 2)}(FX 항 제외 {f1(v3_overlap['jaccard_nofx'], 2)}), 일수익 상관 {f1(v3_overlap['corr'], 2)}. 원 규칙의 15:20 시점 proxy(304940·219480)·미국 종가 보정 세부는 자료가 없어 일봉 근사. 2011 금지기는 133690 상장(2010.10) 직후라 표본이 짧음.")
    L.append("• V4 베이시스 z(FUT−KPI200, 250일, 진입 전일)·VKOSPI(진입 전일) 통제 후 D_ban 계수(bp, HAC10): " + "; ".join(f"{h} {f1(v4[h][0])}(t {f1(v4[h][1], 2)})" for h in LEGS) + ". 통제 전 Δ 는 진단 절.")
    L.append("• V4 베이시스 평균(%): " + nd("; ".join(l_.split("베이시스 평균 ")[1] for l_ in log if l_.startswith("베이시스 평균"))) + ".")
    L.append("• V5 부분 재개(2021.05.03~2023.11.05) 더미 분리, 기준 = 전면 허용기(bp, HAC10): " + "; ".join(f"{h} D_ban {f1(v5[h][0])}(t {f1(v5[h][1], 2)}) D_partial {f1(v5[h][2])}(t {f1(v5[h][3], 2)})" for h in LEGS) + ".")
    L.append("• V6 C3 분해(활성일, bp): " + "; ".join(f"{h.split('_')[1]} {lab} 금지 {f1(v6[(h, c)][0])}(n {v6[(h, c)][4]}) / 비금지 {f1(v6[(h, c)][1])}(n {v6[(h, c)][5]}) Δ {f1(v6[(h, c)][2])} t {f1(v6[(h, c)][3], 2)}" for h, c, lab in [("beomgu_C3VT30", "disc", "괴리"), ("beomgu_C3VT30", "gap", "갭"), ("beomgu_C3Q", "disc", "괴리"), ("beomgu_C3Q", "gap", "갭")]) + ". 지수: 코스피200 k200_index_daily(시가 2003~ 결측 0), 코스닥150 KQ150_index_kis(2015.07.13~).")
    L.append("")
    L.append("## 판정")
    L.append(f"• 조건 1 표본외(W_long, 2020 금지기) CAGR·MDD 동시 개선: {'충족' if c1 else '미충족'}(MDD 비악화 해석이면 {'충족' if c1_relax else '미충족'}).")
    L.append(f"• 조건 2 비중을 바꾼 다리의 Δ HAC t ≤ −2: jehyeong_ON t {f1(tON, 2)}, beomgu_C3Q t {f1(tC3Q, 2)} → {'충족' if c2 else '미충족'}.")
    L.append(f"• 조건 3 W_all 결합 CAGR·MDD 개선(사후): {'충족' if c3 else '미충족'}(MDD 비악화 해석이면 {'충족' if c3_relax else '미충족'}; MDD 는 2026 낙폭이라 불변).")
    L.append(f"• 조건 4 2024·2025 연수익 개선(사후): {'충족' if c4 else '미충족'}.")
    L.append(f"• 종합: {'통과' if judging['passed'] else '미통과'}(엄격 해석), {'통과' if judging['passed_relaxed'] else '미통과'}(MDD 비악화 해석). 등급 estimate 고정(사건 수: 장부 창 1회, 장기창 2회). 장부 결합 규칙(W_all·W_long 전체 금지기 적용 ∧ 2024·2025): {'통과' if book_rule else '미통과'}(비악화 해석 {'통과' if book_rule_relax else '미통과'}). 통과 여부와 무관하게 다음 금지 공표 전까지 장부 수치 미반영(포워드 규칙, overlay_06.csv 는 기록용).")
    L.append("")
    L.append("## 단서")
    L.append(f"• 다중 비교: 이 설계에서 돌린 셀 {len(cdf)}개(기준선 7 포함). 사건 수가 4회(장부 창 1회)라 더미 계수는 국면(2020 급락·고변동, 2023~2025 횡보·외인 매도)을 흡수할 수 있음. V4 통제 후 계수와 비교할 것.")
    L.append("• 비중표 A 의 ON·C3Q 0 은 13차 빠른 점검(2023 금지기 음수)을 보고 정한 사후 선택. 표본외는 2020 금지기 하나(W_long 에는 ON 이 없어 C3Q 만 바뀜)뿐이라 W_long 주 셀은 C3Q 한 다리의 2020 성과로 결정됨.")
    L.append("• 단일가 가정(09:00 시가·15:30 종가 체결)은 13차 series·무조건 밤·재구성 셀 모두에 그대로 적용. 다음 금지의 형태(대상·기간·예외)가 과거와 다를 수 있음.")
    L.append("• blog_ON_Q 1.5 는 ETF 한도(233740 롱 2, 251340 숏 1)로 1.0 에 절단되어 B 표의 코스닥 밤 가중은 실질 변화 없음. C3VT30 1.5 는 122630 한도 2 안에서만.")
    L.append("• 2011 재개 보도자료 원문은 미열람(보고영상·보험연구원 보고로 간접 확인). 그 밖 세 구간은 금융위 원문 확인.")
    L.append("")
    L.append("## 산출")
    L.append("• cells.csv(셀 전부), trades.csv(13차 여섯 다리 활성일 + 무조건 밤 + 재구성 ON, 열 handle,date,entry_date,pos,gross,net,ban_id,partial), judging.json, fig1_leg_ban_bars.png, fig2_equity_ban_shading.png, fig3_v3_v6.png, series/overlay_06.csv + overlay_06.meta.json(비중표 A, 기록용). 코드 topics/r15_06_ban_regime.py.")
    text = "\n".join(L) + "\n"
    assert "-" not in text, [l_ for l_ in L if "-" in l_][:3]
    (OUT / "notes.md").write_text(text, encoding="utf-8")
    (OUT / "run_log.txt").write_text("\n".join(log), encoding="utf-8")
    print("\n저장:", OUT)


if __name__ == "__main__":
    main()
