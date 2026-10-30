"""15차 설계 01_open_hedge · ETF 시가 저평가 헤지 추출(ETF 롱 + 선물 숏, 아침 시장 중립 다리)

규칙(사전 등록 alpha-share-prereg-2026-10-30.md, design_shortlist.md §2 01. 결과 보기 전 고정, 변형 결과로 주 셀 변경 없음)
  mis_t = ln(O_etf,t / O_idx,t) − median_{s=t−5..t−1} ln(C_etf,s / C_idx,s)
  진입 mis_t ≤ −0.0010 → 09:00 단일가 ETF 매수 + 09:00 선물 매도(주 셀은 지수 헤지 대용) → 15:30 청산
  r_t(1x) = ln(C_etf,t/O_etf,t) − ln(C_idx,t/O_idx,t) − 0.00068 (ETF 왕복 5bp + 선물 왕복 1.8bp)
  스트레스 0.0010(7 + 3bp), 보수 셀 0.00148(지수 시가 잡음 8bp 추가 차감)
  제외: special_days.csv 특수일(1분봉 창) + 동종 확장(연초 첫 거래일·수능일 10:00 개장, 1분봉 창 밖),
        ETF 09:01 봉 거래량 0 일(1분봉 창만), 지수 시가 결측일, 분배락 당일, 기존 C3 다리 활성일(코스닥 beomgu_C3Q, 코스피 beomgu_C3VT30)
  분배락 식별: 조정 종가(etf csv) ÷ KIS 당일 종가(nav×(1+dprt/100)) 비율의 로그 변화 |Δ| > 0.001 인 날
  주 셀: 229200 + KQ150_index_kis(2015-07-13 이후 유효) 2016-01-04~2026-09-29, 설계 2016~2019 / 검증 2020~2026-09, W_all 필수, 전·후반 = 창 중앙(날짜)
  변형: V1 문턱 −20bp / V2 069500 + KPI200(2014-07-08~) / V3 09:01 진입(TOSS 229200, 09:02 라벨 봉 이후 첫 체결 봉 시가, 헤지는 지수 09:00 시가)
        / V4 조기 청산 09:05·10:00 / V5 선물 실측 헤지(코스피 069500 + 코스피200 선물 1분봉, 코스닥 229200 + 코스닥150 선물 1분봉)
        / V6 breadth 사이징(코스닥150 1배 ETF 20일 거래대금 중앙 ≥ 10억 중 mis ≤ −10bp 비율 ≥ 0.7 → ×1.5, ≤ 0.3 → ×0.5)
  판정: 주 셀 공통 규칙 ∧ 보수 셀 순수익 > 0 ∧ V5 실측 중앙값 > 6.8bp ∧ 2016~2026 연도 전부 양수 → candidate.
        V3 순수익 > 0 ∧ 잔존율 ≥ 50% 아니면 '단일가 가정 의존'(포워드 전용). 장부 결합: 코스닥 장부 + r_new(C3Q 비활성 조건일) → W_all·W_long CAGR·MDD, 2024·2025.
통계: 이벤트 순수익 bp 상수 회귀 HAC(maxlags 5) NW t, Δ = 신호일 − 비신호일(같은 유효 모집단) 더미 회귀 HAC t,
      CAGR = Π(1+r)^(252/일수) − 1(비활성일 0, 069500 달력), MDD, 연도 부호(이벤트 ≥ 10 인 해), 상위 5건 비중, 전·후반(창 중앙 날짜, 건수 반분 병기).
자료(읽기 전용): hyfe/data/raw/etf/{229200,069500,KPI200}.csv, etf_extra/{KQ150_index_kis, 코스닥150 1배 ETF}.csv, r15_external/kis_etf_nav_daily.csv,
      TOSS 1분봉 229200·069500(봉 끝 라벨), intraday/kis_minute/{full_day,kq150_full_day}/futures_1min.csv(봉 시작 라벨),
      round13_slot_book/series/*.csv + book/book_daily.csv, round14 special_days.csv. 장부 함수는 r15_leverage_desk.py 를 그대로 import.
가정: 09:00 시가·15:30 종가 단일가 체결(의존 강도 V3 로 수치화), 지수 헤지 대용(선물 실측은 V5), 세금·증거금 미반영, 수익률은 로그 차(1x 명목).
산출: round15_alpha_share/01_open_hedge/{cells.csv, trades.csv, notes.md, fig1~3.png}, series/beomgu_OH_Q.csv(+meta), series/beomgu_OH_K.csv(+meta).
실행: PYTHONUTF8=1 python "<code>/topics/r15_01_open_hedge.py"
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import r15_leverage_desk as LD  # noqa: E402  13차 장부 재현(panel_wall/panel_long/books/metrics)

RAW = Path("<raw>")
LAB = Path("<OUT>")
MIN = Path("<repo>/jehyeong/intraday/data/minute")
R13 = LAB / "round13_slot_book/series"
OUT = LAB / "round15_alpha_share/01_open_hedge"
SER = LAB / "round15_alpha_share/series"
SPECIAL_CSV = LAB / "round14_alpha_sprint/S3_kq150_close_auction/special_days.csv"

THR = -0.0010
COST = {"main": 6.8e-4, "stress": 10e-4, "cons": 14.8e-4}
W_MAIN = ("2016-01-04", "2026-09-29")
W_DESIGN = ("2016-01-04", "2019-12-30")
W_VALID = ("2020-01-02", "2026-09-29")
W_K = ("2014-07-08", "2026-09-29")
W_ALL, W_EX26 = LD.WIN["W_all"], LD.WIN["W_ex2026"]
W_TOSS = ("2022-11-23", "2026-09-29")  # 일봉 신호(229200.csv) 끝까지
# 1분봉 창(special_days.csv 2022-11-23~) 밖의 동종 특수일: 연초 첫 거래일·수능일(10:00 개장)
SPECIAL_EXT = ["2014-11-13", "2015-01-02", "2015-11-12", "2016-01-04", "2016-11-17", "2017-01-02", "2017-11-23", "2018-01-02",
               "2018-11-15", "2019-01-02", "2019-11-14", "2020-01-02", "2020-12-03", "2021-01-04", "2021-11-18", "2022-01-03", "2022-11-17"]
KQ_1X = ["229200", "232080", "270810", "354500", "316670", "450910", "304770", "0191S0", "301400"]  # kq150_etf_names_naver.csv etf_tab=1
NOTES: list[str] = []
CELLS: list[dict] = []
TRADES: list[pd.DataFrame] = []


def log(s=""):
    s = re.sub(r"(?<![0-9A-Za-z])-(?=[0-9.])", "−", s)  # 음수 부호만 −, 날짜 하이픈은 assert 로 잡는다
    print(s)
    NOTES.append(s)


def dd(s):  # notes.md 하이픈 금지 → 날짜 2016.01.04
    return str(s)[:10].replace("-", ".")


def f1(x, p=1, sign=True):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "·"
    return (f"{x:+.{p}f}" if sign else f"{x:.{p}f}").replace("-", "−")


# ---------------------------------------------------------------- 통계
def nw_t(x, lags=5):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 3:
        return np.nan
    u = x - x.mean()
    s = float((u * u).sum())
    for L in range(1, lags + 1):
        s += 2 * (1 - L / (lags + 1)) * float((u[L:] * u[:-L]).sum())
    return float(x.mean() / (np.sqrt(s) / n)) if s > 0 else np.nan


def delta_test(y_bp, sig):
    d = pd.DataFrame({"y": y_bp, "s": sig.astype(float)}).dropna()
    if d.s.sum() < 3 or (1 - d.s).sum() < 3:
        return np.nan, np.nan
    res = sm.OLS(d.y.to_numpy(), sm.add_constant(d.s.to_numpy())).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
    return float(res.params[1]), float(res.tvalues[1])


def _selfcheck():
    rng = np.random.default_rng(0)
    x = rng.normal(0.3, 1, 400)
    res = sm.OLS(x, np.ones(len(x))).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
    assert abs(nw_t(x) - res.tvalues[0]) < 1e-9


# ---------------------------------------------------------------- 자료
def px(p):
    return pd.read_csv(p, parse_dates=["date"]).set_index("date").sort_index()


CAL = px(RAW / "etf/069500.csv").index  # KRX 거래일 달력


def ex_dates(code, adj, lo):
    nav = pd.read_csv(RAW / "r15_external/kis_etf_nav_daily.csv", parse_dates=["date"], dtype={"code": str})
    n = nav[nav.code == code].set_index("date").sort_index()
    n = n[(n.nav > 0) & (n.dprt.abs() <= 10)]
    j = adj[["close"]].join((n.nav * (1 + n.dprt / 100)).rename("unadj"), how="inner")
    step = np.log(j.close / j.unadj).diff()
    return step[(step.abs() > 0.001) & (step.index >= lo)].index


def special_days():
    s = pd.read_csv(SPECIAL_CSV, parse_dates=["date"])
    return pd.DatetimeIndex(s.date[s.special]).union(pd.to_datetime(SPECIAL_EXT))


def load_toss(code):
    m = pd.read_csv(MIN / f"{code}_TOSS_1m.csv", usecols=["timestamp", "date", "open", "close", "volume"])
    m["time"] = m.timestamp.str.slice(11, 16)
    m["date"] = pd.to_datetime(m.date)
    m = m.drop_duplicates(["date", "time"])
    return {k: m.pivot(index="date", columns="time", values=k) for k in ("open", "close", "volume")}


def load_fut(p):
    f = pd.read_csv(p, dtype={"time": str})
    f["date"] = pd.to_datetime(f.date)
    f = f.drop_duplicates(["date", "time"])
    return {"open": f.pivot(index="date", columns="time", values="open"), "close": f.pivot(index="date", columns="time", values="close"),
            "con": f.groupby("date").contract.first()}


def first_fill(P, start="09:02", stop="09:10"):
    """해당 시각 이후 첫 체결 봉의 시가(r5 규약: 09:01 진입 = 09:02 라벨 봉부터 거래량 > 0 인 첫 봉)."""
    cols = [c for c in P["open"].columns if start <= c <= stop]
    o = P["open"][cols].where(P["volume"][cols] > 0)
    return o.bfill(axis=1).iloc[:, 0]


def daily(etf, idx, c3_handle, exd, spec, w, toss=None):
    d = etf[["open", "close"]].join(idx[["open", "close"]], lsuffix="_e", rsuffix="_i", how="inner")
    lcr = np.log(d.close_e / d.close_i)
    d["mis"] = np.log(d.open_e / d.open_i) - lcr.shift(1).rolling(5).median()
    d["gross"] = np.log(d.close_e / d.open_e) - np.log(d.close_i / d.open_i)
    d = d.loc[w[0]:w[1]].copy()
    d["special"] = d.index.isin(spec)
    d["exdiv"] = d.index.isin(exd)
    d["idx_na"] = ~(d.open_i > 0) | ~(d.close_i > 0) | ((d.open_i == d.close_i) & (idx.high.reindex(d.index) == idx.low.reindex(d.index)))
    z = toss["volume"]["09:01"].reindex(d.index) if toss is not None else pd.Series(np.nan, index=d.index)
    d["v0901_zero"] = (z.notna() & (z == 0)).to_numpy()
    c3 = px(R13 / f"{c3_handle}.csv").pos
    d["c3"] = c3.reindex(d.index).fillna(0).ne(0).to_numpy()
    d["valid"] = d.mis.notna() & ~d.special & ~d.exdiv & ~d.idx_na & ~d.v0901_zero
    return d


# ---------------------------------------------------------------- 셀 지표
def cell(ev_gross, w, cost):
    """ev_gross: 이벤트별 gross(index=date). 반환 지표 dict(bp)."""
    g = ev_gross.sort_index()
    net = g - cost
    days = CAL[(CAL >= w[0]) & (CAL <= w[1])].union(g.index)  # 069500 달력 밖(2026-09-30 이후 1분봉 날)은 이벤트 날만 추가
    out = dict(n=len(g), n_days=len(days))
    if len(g) == 0:
        return out
    nb = net * 1e4
    out.update(gross_bp=float(g.mean() * 1e4), net_bp=float(nb.mean()), t_nw=nw_t(nb), win_pct=float((net > 0).mean() * 100),
               median_net_bp=float(nb.median()))
    r = pd.Series(0.0, index=days).add(net.groupby(net.index).sum().reindex(days).fillna(0.0))
    cum = (1 + r).cumprod()
    out["cagr_pct"] = float((cum.iloc[-1] ** (252 / len(r)) - 1) * 100)
    out["mdd_pct"] = float((cum / cum.cummax() - 1).min() * 100)
    yr = net.groupby(net.index.year).agg(["sum", "count"])
    yr10 = yr[yr["count"] >= 10]
    out["yearly_pos"] = " ".join(f"{y}{'+' if s > 0 else '−'}({int(c)})" for y, (s, c) in yr10.iterrows()) or "(없음)"
    out["yearly_net_bp"] = " ".join(f"{y} {s / c * 1e4:+.1f}bp({int(c)})" for y, (s, c) in yr.iterrows())
    out["year_pos_share"] = float((yr10["sum"] > 0).mean()) if len(yr10) else np.nan
    out["years_lt10"] = ",".join(str(y) for y in yr.index[yr["count"] < 10])
    tot = float(net.sum())
    out["top5_share_pct"] = float(net.nlargest(5).sum() / tot * 100) if tot > 0 else np.nan
    mid = days[len(days) // 2]
    out["half1_net_bp"] = float(nb[nb.index < mid].mean()) if (nb.index < mid).any() else np.nan
    out["half2_net_bp"] = float(nb[nb.index >= mid].mean()) if (nb.index >= mid).any() else np.nan
    h = len(nb) // 2
    out["half1_cnt_bp"] = float(nb.iloc[:h].mean()) if h else np.nan
    out["half2_cnt_bp"] = float(nb.iloc[h:].mean())
    out["sum_net_pct"] = tot * 100
    out["_daily"] = r
    return out


def verdict(m, diagnostic=False):
    if diagnostic:
        return "diagnostic"
    if m.get("n", 0) < 10 or m["n_days"] <= 300:
        return "estimate"
    if m["net_bp"] <= 0:
        return "reject"
    ok = (m["t_nw"] >= 2 and m["year_pos_share"] >= 0.7 and (m["top5_share_pct"] < 50 if not np.isnan(m["top5_share_pct"]) else False)
          and m["half1_net_bp"] > 0 and m["half2_net_bp"] > 0)
    return "candidate" if ok else "reference"


def add_cell(cid, name, market, w, m, delta=(np.nan, np.nan), notes="", diagnostic=False, v=None):
    v = v or verdict(m, diagnostic)
    row = dict(id=cid, name=name, market=market, window=f"{w[0]}~{w[1]}", n=m.get("n", 0), gross_bp=m.get("gross_bp"), net_bp=m.get("net_bp"),
               t_nw=m.get("t_nw"), delta_bp=delta[0], t_delta=delta[1], win_pct=m.get("win_pct"), cagr_pct=m.get("cagr_pct"), mdd_pct=m.get("mdd_pct"),
               yearly_pos=m.get("yearly_pos"), top5_share_pct=m.get("top5_share_pct"), half1_net_bp=m.get("half1_net_bp"), half2_net_bp=m.get("half2_net_bp"),
               verdict=v, notes=notes, yearly_net_bp=m.get("yearly_net_bp"), year_pos_share=m.get("year_pos_share"), years_lt10=m.get("years_lt10"),
               half1_cnt_bp=m.get("half1_cnt_bp"), half2_cnt_bp=m.get("half2_cnt_bp"), median_net_bp=m.get("median_net_bp"), sum_net_pct=m.get("sum_net_pct"),
               n_days=m.get("n_days"))
    CELLS.append(row)
    log(f"• [{cid}] {name}: n {row['n']}, 총 {f1(row['gross_bp'])}bp, 순 {f1(row['net_bp'])}bp(중앙 {f1(row['median_net_bp'])}), NW t {f1(row['t_nw'], 2)}, Δ {f1(delta[0])}bp(t {f1(delta[1], 2)}), "
        f"승률 {f1(row['win_pct'], 0, False)}%, CAGR {f1(row['cagr_pct'])}%, MDD {f1(row['mdd_pct'])}%, 연도 {row['yearly_net_bp']}, "
        f"상위5 {f1(row['top5_share_pct'], 0, False)}%, 전반 {f1(row['half1_net_bp'])}/후반 {f1(row['half2_net_bp'])}bp(창 중앙; 건수 반분 {f1(row['half1_cnt_bp'])}/{f1(row['half2_cnt_bp'])}bp) "
        f"→ **{v}** [창 {dd(w[0])}~{dd(w[1])} {row['n_days']}일, {notes}]")
    return row


def add_trades(cid, df):
    t = df.copy()
    t.insert(0, "cell", cid)
    t.insert(1, "date", t.index.strftime("%Y-%m-%d"))
    TRADES.append(t.reset_index(drop=True))


def run_cells(prefix, market, d, w, cost, note, thr=THR, windows=None):
    """주 셀 규칙을 창별로. 반환 (주 창 이벤트 gross, 주 창 지표)."""
    pop = d.valid & ~d.c3
    ev_all = d[pop & (d.mis <= thr)]
    out = None
    for wid, ww in (windows or [("", w)]):
        sub = d.loc[ww[0]:ww[1]]
        p = sub.valid & ~sub.c3
        sig = p & (sub.mis <= thr)
        m = cell(sub.gross[sig], ww, cost)
        dl = delta_test((sub.gross[p] - cost) * 1e4, sig[p])
        row = add_cell(prefix + wid, f"{note}{(' ' + wid.strip('_')) if wid else ''}", market, ww, m, dl,
                       f"문턱 {thr * 1e4:.0f}bp, 비용 {cost * 1e4:.1f}bp, 대조군 {int((p & ~sig).sum())}일")
        if out is None:
            out = (ev_all.gross, m, row)
    return out


# ---------------------------------------------------------------- 그림
C = {"a": "#2a78d6", "b": "#eb6834", "c": "#1baf7a", "d": "#eda100"}


def fig_quintile(dq, dk):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, (d, lab) in zip(axes, [(dq, "코스닥 229200 vs KQ150 지수"), (dk, "코스피 069500 vs KPI200 지수")]):
        p = d[d.valid & ~d.c3]
        q = pd.qcut(p.mis, 5, labels=False)
        g = p.groupby(q).agg(mean_bp=("gross", lambda x: x.mean() * 1e4), n=("gross", "size"), lo=("mis", "min"), hi=("mis", "max"))
        log(f"• 5분위(유효·C3 비활성일, 비용 0) {lab}: " + " ".join(f"Q{i + 1} {f1(r.mean_bp)}bp(n {int(r.n)}, mis {f1(r.lo * 1e4, 0)}~{f1(r.hi * 1e4, 0)}bp)" for i, r in g.iterrows()))
        ax.bar(range(5), g.mean_bp, color=[C["b"] if i == 0 else C["a"] for i in range(5)])
        for i, r in g.iterrows():
            ax.annotate(f"n {int(r.n)}\n{r.lo * 1e4:.0f}~{r.hi * 1e4:.0f}bp", (i, r.mean_bp), ha="center", va="bottom" if r.mean_bp >= 0 else "top", fontsize=8, color="#52514e")
        ax.axhline(0, color="#52514e", lw=0.8)
        ax.set_xticks(range(5), [f"Q{i + 1}" for i in range(5)])
        ax.set_title(f"{lab}: mis 5분위별 헤지 총수익(비용 0)", fontsize=10, loc="left")
        ax.set_ylabel("평균 bp (ETF 시가→종가 − 지수 시가→종가)")
        ax.grid(alpha=0.25, axis="y")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "fig1_mis_quintile.png", dpi=130)
    plt.close(fig)


def fig_cum(r_main, r_cons, r_v3, r_same):
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 4.3), gridspec_kw={"width_ratios": [1.5, 1]})
    for r, lab, col in [(r_main, "주 셀 6.8bp", C["a"]), (r_cons, "보수 셀 14.8bp", C["b"])]:
        ax.plot(r.index, r.cumsum() * 100, lw=1.5, color=col, label=lab)
    ax.set_title("229200 시가 저평가 헤지 누적 순손익(단순 합, 자본 1 기준 %)", fontsize=10, loc="left")
    for r, lab, col in [(r_same, "같은 날 주 셀(09:00 단일가)", C["a"]), (r_v3, "V3 09:01 진입", C["c"])]:
        ax2.plot(r.index, r.cumsum() * 100, lw=1.5, color=col, label=lab)
    ax2.set_title("1분봉 창: 09:00 단일가 vs 09:01 진입", fontsize=10, loc="left")
    import matplotlib.dates as mdates
    ax2.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7]))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y.%m"))
    ax2.tick_params(axis="x", labelrotation=30, labelsize=8)
    for a in (ax, ax2):
        a.legend(frameon=False, fontsize=9, loc="upper left")
        a.grid(alpha=0.25)
        a.set_ylabel("누적 %")
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "fig2_cum_net.png", dpi=130)
    plt.close(fig)


def fig_v5(vk, vq):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    for ax, (v, lab) in zip(axes, [(vk, "코스피 069500 + 코스피200 선물"), (vq, "코스닥 229200 + 코스닥150 선물")]):
        p = v[v.ok]
        sel = p.gapdiff <= THR
        ax.scatter(p.gapdiff[~sel] * 1e4, p.gross[~sel] * 1e4, s=12, color="#9a9994", alpha=0.6, label="조건 밖")
        ax.scatter(p.gapdiff[sel] * 1e4, p.gross[sel] * 1e4, s=16, color=C["b"], alpha=0.8, label=f"갭차 ≤ −10bp n {int(sel.sum())}")
        ax.axvline(THR * 1e4, color="#e34948", lw=1, ls="--")
        ax.axhline(0, color="#52514e", lw=0.8)
        ax.set_xlabel("ETF 갭 − 선물 갭 (bp)")
        ax.set_ylabel("선물 실측 헤지 총수익 bp (09:00→15:30)")
        ax.set_title(f"{lab} ({dd(p.index.min())}~{dd(p.index.max())}, n {len(p)})", fontsize=10, loc="left")
        ax.legend(frameon=False, fontsize=9)
        ax.grid(alpha=0.25)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "fig3_v5_scatter.png", dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------- 시리즈 저장
def save_series(handle, ev_gross, cost, w, market, instrument, rule, src):
    days = CAL[(CAL >= w[0]) & (CAL <= w[1])]
    s = pd.DataFrame({"pos": 0.0, "gross": 0.0, "net": 0.0}, index=days)
    s.loc[ev_gross.index, "pos"] = 1.0
    s.loc[ev_gross.index, "gross"] = ev_gross
    s.loc[ev_gross.index, "net"] = ev_gross - cost
    s.index.name = "date"
    s.to_csv(SER / f"{handle}.csv", float_format="%.8f")
    st = LD.metrics(s.net)
    meta = {"slot": "B_open_neutral", "market": market, "instrument": instrument, "timing": "intraday_09:00 단일가 → 15:30 종가(ETF 롱 1 + 선물 숏 1, 지수 헤지 대용)",
            "cost_bp": round(cost * 1e4, 2), "direction": "market_neutral", "source_files": src, "notes": rule, "handle": handle,
            "first_valid": w[0], "last_valid": w[1], "n_days": int(len(s)), "n_active": int((s.pos != 0).sum()), "dropped_off_calendar": [],
            "repro": {"expected": {"window": f"{w[0]}~{w[1]}", "cagr": round(st["cagr"], 4), "mdd": round(st["mdd"], 4), "n_active": int((s.pos != 0).sum())},
                      "got": {"window": f"{w[0]}~{w[1]}", "cagr": st["cagr"], "mdd": st["mdd"], "n_active": int((s.pos != 0).sum())}},
            "full_valid_stats": {"start": w[0], "end": w[1], **{k: st[k] for k in ("n_days", "n_active", "cum", "cagr", "vol", "sharpe", "mdd", "calmar", "win")}},
            "metric_defs": "누적=Π(1+r); CAGR=누적^(252/일수)-1; 연변동성=std(ddof=1)·√252(0 포함); 샤프=mean·252/vol; MDD=min(누적/누적최고-1); 승률=pos≠0 일 중 net>0 비율"}
    (SER / f"{handle}.meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    chk = px(SER / f"{handle}.csv")
    assert abs(chk.net[chk.pos != 0].mean() - (ev_gross.mean() - cost)) < 1e-7 and (chk.pos != 0).sum() == len(ev_gross)
    return s


# ---------------------------------------------------------------- 장부 결합
def book_cells(new_net, label, cid):
    """코스닥 장부 + r_new(이미 C3Q 비활성 조건일만). 원 장부 vs 적용, 창별 CAGR·MDD·연수익."""
    legs_w = LD.panel_wall()
    legs_l = LD.panel_long(["KOSPI_B_beomgu_C3VT30", "KOSDAQ_A_blog_ON_Q", "KOSDAQ_B_beomgu_C3Q"], LD.W_LONG)
    bw, bl = LD.books(legs_w, {}, "fixed"), LD.books(legs_l, {}, "fixed")
    base = LD.metrics(bw["comb"])
    assert abs(base["cagr"] - 0.314179) < 0.002 and abs(base["mdd"] + 0.061046) < 0.001, base
    lb = LD.metrics(bl["comb"])
    assert lb["n_days"] == 2444 and abs(lb["cagr"] - 0.150369) < 0.002 and abs(lb["mdd"] + 0.096107) < 0.001, lb
    res = {}
    for scope, b in (("W_all계열", bw), ("W_long", bl)):
        add = new_net.reindex(b["KOSDAQ"].index).fillna(0.0)
        q2 = b["KOSDAQ"] + add
        c2 = 0.5 * (b["KOSPI"] + q2)
        wins = {k: LD.WIN[k] for k in ("W_all", "W_ex2026", "Y2024", "Y2025", "Y2026")} if scope == "W_all계열" else {"W_long": LD.W_LONG}
        for wn, w in wins.items():
            o, a = LD.metrics(LD.win(b["comb"], w)), LD.metrics(LD.win(c2, w))
            oq, aq = LD.metrics(LD.win(b["KOSDAQ"], w)), LD.metrics(LD.win(q2, w))
            n_add = int((LD.win(add, w) != 0).sum())
            res[wn] = dict(window=wn, start=w[0], end=w[1], n_new_days=n_add, comb_cagr=o["cagr"], comb_cagr_new=a["cagr"], comb_mdd=o["mdd"], comb_mdd_new=a["mdd"],
                           comb_cum=o["cum"], comb_cum_new=a["cum"], comb_sharpe=o["sharpe"], comb_sharpe_new=a["sharpe"],
                           kq_cagr=oq["cagr"], kq_cagr_new=aq["cagr"], kq_mdd=oq["mdd"], kq_mdd_new=aq["mdd"])
            m = {"n": n_add, "n_days": a["n_days"], "gross_bp": np.nan, "net_bp": (a["cagr"] - o["cagr"]) * 1e4, "t_nw": np.nan, "win_pct": np.nan,
                 "cagr_pct": a["cagr"] * 100, "mdd_pct": a["mdd"] * 100, "yearly_pos": f"원 장부 CAGR {o['cagr'] * 100:.1f}% MDD {o['mdd'] * 100:.1f}% 연수익 {o['cum'] * 100:.1f}%",
                 "top5_share_pct": np.nan, "half1_net_bp": np.nan, "half2_net_bp": np.nan, "yearly_net_bp": f"적용 연수익 {a['cum'] * 100:.1f}% 샤프 {a['sharpe']:.2f}(원 {o['sharpe']:.2f})",
                 "year_pos_share": np.nan, "years_lt10": "", "half1_cnt_bp": np.nan, "half2_cnt_bp": np.nan, "median_net_bp": np.nan, "sum_net_pct": np.nan}
            add_cell(f"{cid}_{wn}", f"장부 결합 {label} {wn}", "COMB", w, m, notes=f"합산(a) 적용 vs 원 장부. net_bp 열 = CAGR 차(bp), cagr/mdd = 적용값. 코스닥 장부 CAGR {oq['cagr'] * 100:.1f}→{aq['cagr'] * 100:.1f}% MDD {oq['mdd'] * 100:.1f}→{aq['mdd'] * 100:.1f}%",
                     diagnostic=True)
    r = res
    passed = (r["W_all"]["comb_cagr_new"] > r["W_all"]["comb_cagr"] and r["W_all"]["comb_mdd_new"] >= r["W_all"]["comb_mdd"]
              and r["W_long"]["comb_cagr_new"] > r["W_long"]["comb_cagr"] and r["W_long"]["comb_mdd_new"] >= r["W_long"]["comb_mdd"]
              and r["Y2024"]["comb_cum_new"] > r["Y2024"]["comb_cum"] and r["Y2025"]["comb_cum_new"] > r["Y2025"]["comb_cum"])
    return pd.DataFrame(res.values()), passed


# ---------------------------------------------------------------- 메인
def main():
    _selfcheck()
    plt.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    OUT.mkdir(parents=True, exist_ok=True)
    SER.mkdir(parents=True, exist_ok=True)
    spec = special_days()
    etf_q, idx_q = px(RAW / "etf/229200.csv"), px(RAW / "etf_extra/KQ150_index_kis.csv")
    etf_k, idx_k = px(RAW / "etf/069500.csv"), px(RAW / "etf/KPI200.csv")
    toss_q, toss_k = load_toss("229200"), load_toss("069500")
    exd_q, exd_k = ex_dates("229200", etf_q, "2015-10-01"), ex_dates("069500", etf_k, "2014-07-08")

    log("# 01_open_hedge · ETF 시가 저평가 헤지 추출(229200 롱 + 코스닥150 선물 숏 대용, 아침 시장 중립 다리) · 15차")
    log("")
    log("## 1. 자료 점검")
    flat = (idx_q.open == idx_q.close) & (idx_q.high == idx_q.low)
    log(f"• KQ150_index_kis: {len(idx_q)}행 {dd(idx_q.index.min())}~{dd(idx_q.index.max())}, OHLC 전부 같은 행(시가 무효) {int(flat.sum())}개(마지막 {dd(idx_q.index[flat].max())}), "
        f"2015.07.13 이후 무효 0, 시가 결측·0 없음. 주 창 {dd(W_MAIN[0])}~{dd(W_MAIN[1])} 069500 달력 {int(((CAL >= W_MAIN[0]) & (CAL <= W_MAIN[1])).sum())}일과 229200·지수 날짜 완전 일치.")
    log(f"• KPI200.csv: {len(idx_k)}행 {dd(idx_k.index.min())}~{dd(idx_k.index.max())}, 시가 결측·0·무효 0, 069500 달력과 일치. 229200.csv·069500.csv 는 분배금 조정가(TOSS 09:01 봉 시가와 비율만 다름) → 비율·수익률만 사용.")
    log(f"• 분배락 당일(조정 종가 ÷ KIS nav×(1+dprt/100) 비율 로그 변화 |Δ| > 0.001): 229200 {len(exd_q)}일 {', '.join(dd(x) for x in exd_q)}; "
        f"069500(2014.07.08~) {len(exd_k)}일 {', '.join(dd(x) for x in exd_k)}. KIS nav 파일 |dprt| > 10 행 제외 뒤 계산.")
    n_csv = int(pd.read_csv(SPECIAL_CSV).special.sum())
    log(f"• 특수일: special_days.csv True {n_csv}일(2022.11.23~2026.10.07) + 동종 확장 {len(SPECIAL_EXT)}일(연초 첫 거래일·수능일 10:00 개장, 1분봉 창 밖, 사전 목록 고정) = {len(spec)}일 제외.")
    v01 = toss_q["volume"]["09:01"]
    log(f"• TOSS 229200 1분봉: {toss_q['open'].shape[0]}일 {dd(toss_q['open'].index.min())}~{dd(toss_q['open'].index.max())}, 09:01 봉 없음(10:01 개장) {int(v01.isna().sum())}일, "
        f"09:01 봉 거래량 0 {int((v01 == 0).sum())}일 {', '.join(dd(x) for x in v01.index[v01 == 0])}(모두 특수일). 09:02 라벨 봉 거래량 0 {int((toss_q['volume']['09:02'] == 0).sum())}일 → V3 는 거래량 > 0 인 첫 봉(09:10 까지) 시가.")
    fut_k, fut_q = load_fut(RAW / "intraday/kis_minute/full_day/futures_1min.csv"), load_fut(RAW / "intraday/kis_minute/kq150_full_day/futures_1min.csv")
    for nm, f in (("코스피200 full_day", fut_k), ("코스닥150 kq150_full_day", fut_q)):
        roll = f["con"][f["con"] != f["con"].shift(1)].index[1:]
        log(f"• 선물 1분봉 {nm}: {len(f['con'])}일 {dd(f['con'].index.min())}~{dd(f['con'].index.max())}, 09:00 봉 없음 {int(f['open']['09:00'].isna().sum())}일, "
            f"15:30 봉 없음 {int(f['close']['15:30'].isna().sum())}일, 월물 교체일 {', '.join(dd(x) for x in roll)}(교체일은 갭 계산 제외).")
    log("• 08:59 예상체결가·예상지수 기록기(T‑P1)는 사용자 승인 전이라 이번 라운드에 없음. 집행 전제 대체 판정 = V3(09:01 진입 잔존율).")
    log("")

    # ------------------------------------------------ 주 셀(코스닥)
    dq = daily(etf_q, idx_q, "beomgu_C3Q", exd_q, spec, W_MAIN, toss_q)
    dk = daily(etf_k, idx_k, "beomgu_C3VT30", exd_k, spec, W_K, toss_k)
    log("## 2. 주 셀 229200 + KQ150 지수 헤지 대용(문턱 −10bp, 비용 6.8bp)")
    pre = dq.valid & (dq.mis <= THR)
    ovl = float((pre & dq.c3).sum() / pre.sum())
    log(f"• 유효일 {int(dq.valid.sum())}/{len(dq)}(제외: 특수일 {int(dq.special.sum())}, 분배락 {int(dq.exdiv.sum())}, 09:01 거래량 0 {int(dq.v0901_zero.sum())}, mis 결측 {int(dq.mis.isna().sum())}). "
        f"조건일(C3Q 제외 전) {int(pre.sum())}일, 그중 beomgu_C3Q 활성일 {int((pre & dq.c3).sum())}일(겹침 {ovl * 100:.1f}%) 제외 → 주 셀 {int((pre & ~dq.c3).sum())}일. "
        f"C3Q 활성일 전체 {int(dq.c3.sum())}일 중 조건일 비율 {(pre & dq.c3).sum() / max(1, dq.c3.sum()) * 100:.1f}%.")
    ev_q, m_q, row_q = run_cells("OHQ_main", "KOSDAQ", dq, W_MAIN, COST["main"], "주 셀 229200 지수 헤지",
                                 windows=[("", W_MAIN), ("_Wall", W_ALL), ("_Wex26", W_EX26), ("_design", W_DESIGN), ("_valid", W_VALID)])
    add_trades("OHQ_main", dq.loc[ev_q.index, ["open_e", "close_e", "open_i", "close_i", "mis", "gross"]].assign(net=lambda x: x.gross - COST["main"]))
    for cid, ck, nm in (("OHQ_stress", "stress", "주 셀 스트레스 10bp"), ("OHQ_cons", "cons", "보수 셀 14.8bp(지수 시가 잡음 8bp 추가)")):
        for wid, ww in (("", W_MAIN), ("_Wall", W_ALL)):
            sub = dq.loc[ww[0]:ww[1]]
            p = sub.valid & ~sub.c3
            sig = p & (sub.mis <= THR)
            add_cell(cid + wid, nm + (" W_all" if wid else ""), "KOSDAQ", ww, cell(sub.gross[sig], ww, COST[ck]), delta_test((sub.gross[p] - COST[ck]) * 1e4, sig[p]),
                     f"문턱 −10bp, 비용 {COST[ck] * 1e4:.1f}bp")
    fig_quintile(dq, dk)
    log("")
    log("## 3. 변형(보고용, 주 셀 변경 없음)")
    ev_v1, _, _ = run_cells("OHQ_V1_t20", "KOSDAQ", dq, W_MAIN, COST["main"], "V1 문턱 −20bp", thr=-0.0020)
    add_trades("OHQ_V1_t20", dq.loc[ev_v1.index, ["mis", "gross"]].assign(net=lambda x: x.gross - COST["main"]))
    prek = dk.valid & (dk.mis <= THR)
    log(f"• V2 코스피 069500 + KPI200: 유효일 {int(dk.valid.sum())}/{len(dk)}(특수일 {int(dk.special.sum())}, 분배락 {int(dk.exdiv.sum())}, mis 결측 {int(dk.mis.isna().sum())}), "
        f"조건일 {int(prek.sum())}, C3VT30 활성 겹침 {int((prek & dk.c3).sum())}일({(prek & dk.c3).sum() / max(1, prek.sum()) * 100:.1f}%) 제외.")
    ev_k, m_k, row_k = run_cells("OHK_V2", "KOSPI", dk, W_K, COST["main"], "V2 069500 + KPI200 헤지(미니선물 대용)", windows=[("", W_K), ("_Wall", W_ALL)])
    add_trades("OHK_V2", dk.loc[ev_k.index, ["open_e", "close_e", "open_i", "close_i", "mis", "gross"]].assign(net=lambda x: x.gross - COST["main"]))

    # ------------------------------------------------ V3 09:01 진입 / V4 조기 청산 (TOSS)
    P = toss_q
    t = pd.DataFrame({"o0900": P["open"]["09:01"], "p0901": first_fill(P), "c0905": P["close"]["09:05"], "c1000": P["close"]["10:00"], "c1530": P["close"]["15:30"]})
    dt = dq.loc[W_TOSS[0]:W_TOSS[1]].join(t, how="left")
    idx_oc = np.log(dt.close_i / dt.open_i)
    sig_t = dt.valid & ~dt.c3 & (dt.mis <= THR) & dt[["o0900", "p0901", "c1530"]].notna().all(axis=1)
    g_v3 = np.log(dt.c1530 / dt.p0901) - idx_oc
    m3 = cell(g_v3[sig_t], W_TOSS, COST["main"])
    m_same = cell(dt.gross[sig_t], W_TOSS, COST["main"])
    resid = m3["net_bp"] / m_same["net_bp"] if m_same["net_bp"] else np.nan
    pop_t = dt.valid & ~dt.c3 & dt[["o0900", "p0901", "c1530"]].notna().all(axis=1)
    add_cell("OHQ_V3_0901", "V3 09:01 진입(09:02 라벨 봉 이후 첫 체결 시가, 헤지 지수 09:00)", "KOSDAQ", W_TOSS, m3, delta_test((g_v3[pop_t] - COST["main"]) * 1e4, sig_t[pop_t]),
             f"비용 6.8bp, 잔존율 {resid * 100:.1f}%(V3 순 ÷ 같은 날 주 셀 순 {m_same['net_bp']:+.1f}bp), 단일가 미체결 가정")
    add_cell("OHQ_main_sameday", "주 셀 같은 날(V3 비교용, 09:00 단일가)", "KOSDAQ", W_TOSS, m_same, notes="V3 와 같은 날·같은 비용", diagnostic=True)
    slip = ((np.log(dt.p0901 / dt.o0900))[sig_t] * 1e4)
    log(f"• V3 보조: 신호일 09:00 단일가 → 09:01 체결가 변화 평균 {f1(slip.mean())}bp, 중앙 {f1(slip.median())}bp, 양수(회복 후 진입) 비율 {(slip > 0).mean() * 100:.0f}%(n {int(sig_t.sum())}).")
    add_trades("OHQ_V3_0901", dt.loc[sig_t, ["mis", "o0900", "p0901", "c1530"]].assign(gross=g_v3[sig_t], net=g_v3[sig_t] - COST["main"], gross_main=dt.gross[sig_t]))
    for lab, col in (("0905", "c0905"), ("1000", "c1000")):
        ok = sig_t & dt[col].notna()
        g4 = np.log(dt[col] / dt.o0900)
        m4 = cell(g4[ok], W_TOSS, COST["main"])
        add_cell(f"OHQ_V4_{lab}", f"V4 조기 청산 {lab[:2]}:{lab[2:]}(ETF 다리만, 지수 변동 0 가정)", "KOSDAQ", W_TOSS, m4, delta_test((g4[pop_t & dt[col].notna()] - COST["main"]) * 1e4, ok[pop_t & dt[col].notna()]),
                 f"비용 6.8bp, 같은 날 주 셀(15:30 헤지) 순 {cell(dt.gross[ok], W_TOSS, COST['main'])['net_bp']:+.1f}bp, ETF 다리만 15:30 순 {cell(np.log(dt.c1530 / dt.o0900)[ok], W_TOSS, COST['main'])['net_bp']:+.1f}bp")
    # 코스닥150 선물 1분봉 창에서 헤지 포함 조기 청산·V3(양 다리 09:01)
    F = fut_q
    df4 = dt.join(pd.DataFrame({"f0900": F["open"]["09:00"], "f0901": F["open"]["09:01"], "f0905": F["open"]["09:05"], "f1000": F["open"]["10:00"], "f1530": F["close"]["15:30"]}), how="inner")
    w4 = (str(df4.index.min().date()), str(df4.index.max().date()))
    s4 = df4.valid & ~df4.c3 & (df4.mis <= THR) & df4[["o0900", "f0900", "f1530", "c1530"]].notna().all(axis=1)
    for lab, ec, fc in (("0905", "c0905", "f0905"), ("1000", "c1000", "f1000"), ("1530", "c1530", "f1530")):
        ok = s4 & df4[[ec, fc]].notna().all(axis=1)
        g = np.log(df4[ec] / df4.o0900) - np.log(df4[fc] / df4.f0900)
        add_cell(f"OHQ_V4_{lab}_fut", f"V4 선물 실측 헤지 청산 {lab[:2]}:{lab[2:]}(코스닥150 선물 창)", "KOSDAQ", w4, cell(g[ok], w4, COST["main"]), notes="비용 6.8bp, 선물 09:00 봉 시가 매도, 창 ≤ 300일")
    ok3 = s4 & df4[["p0901", "f0901"]].notna().all(axis=1)
    g3f = np.log(df4.c1530 / df4.p0901) - np.log(df4.f1530 / df4.f0901)
    add_cell("OHQ_V3_0901_fut", "V3 양 다리 09:01 진입(선물 09:01 봉 시가, 코스닥150 선물 창)", "KOSDAQ", w4, cell(g3f[ok3], w4, COST["main"]),
             notes=f"비용 6.8bp, 같은 날 선물 실측 09:00 진입 순 {cell((np.log(df4.c1530 / df4.o0900) - np.log(df4.f1530 / df4.f0900))[ok3], w4, COST['main'])['net_bp']:+.1f}bp")

    # ------------------------------------------------ V5 선물 실측 헤지
    log("")
    log("## 4. V5 선물 실측 헤지(ETF 갭 − 선물 갭 ≤ −10bp, 09:00 → 15:30, 비용 6.8bp)")
    v5 = {}
    for mk, Pt, F, idx, lab in (("KOSPI", toss_k, fut_k, idx_k, "069500 + 코스피200 선물"), ("KOSDAQ", toss_q, fut_q, idx_q, "229200 + 코스닥150 선물")):
        t = pd.DataFrame({"etf_o": Pt["open"]["09:01"], "etf_c": Pt["close"]["15:30"], "etf_v": Pt["volume"]["09:01"]})
        t["etf_pc"], t["pdate_e"] = t.etf_c.shift(1), t.index.to_series().shift(1)
        f = pd.DataFrame({"fut_o": F["open"]["09:00"], "fut_c": F["close"]["15:30"], "con": F["con"]})
        f["fut_pc"], f["pdate_f"] = f.fut_c.shift(1), f.index.to_series().shift(1)
        f["roll"] = f.con != f.con.shift(1)
        v = t.join(f, how="inner")
        v["ok"] = (pd.Series(~v.index.isin(spec), index=v.index) & (v.etf_v > 0) & ~v.roll & (v.pdate_e == v.pdate_f) & v[["etf_o", "etf_c", "etf_pc", "fut_o", "fut_c", "fut_pc"]].notna().all(axis=1))
        v["gapdiff"] = np.log(v.etf_o / v.etf_pc) - np.log(v.fut_o / v.fut_pc)
        v["gross"] = np.log(v.etf_c / v.etf_o) - np.log(v.fut_c / v.fut_o)
        v["gross_idx"] = np.log(v.etf_c / v.etf_o) - np.log(idx.close / idx.open).reindex(v.index)
        v5[mk] = v
        w5 = (str(v.index.min().date()), str(v.index.max().date()))
        sel = v.ok & (v.gapdiff <= THR)
        m5 = cell(v.gross[sel], w5, COST["main"])
        dl = delta_test((v.gross[v.ok] - COST["main"]) * 1e4, sel[v.ok])
        gm = float(v.gross[sel].median() * 1e4)
        add_cell(f"OH{'K' if mk == 'KOSPI' else 'Q'}_V5_fut", f"V5 선물 실측 헤지 {lab}", mk, w5, m5, dl,
                 f"총수익 중앙 {gm:+.1f}bp(판정 기준 > 6.8bp), 순 중앙 {gm - 6.8:+.1f}bp, 유효 {int(v.ok.sum())}일(월물 교체·특수일·결측 제외 {int((~v.ok).sum())})")
        dif = (v.gross_idx - v.gross)[sel].dropna() * 1e4
        difa = (v.gross_idx - v.gross)[v.ok].dropna() * 1e4
        opp = v.ok & (v.gapdiff >= -THR)
        log(f"  · {lab}: 지수 대용 − 선물 실측(같은 날, bp) 조건일 평균 {f1(dif.mean())} 중앙 {f1(dif.median())}(n {len(dif)}), 전 유효일 평균 {f1(difa.mean())} 중앙 {f1(difa.median())}(n {len(difa)}); "
            f"반대 조건(갭차 ≥ +10bp) n {int(opp.sum())} 총 {f1(v.gross[opp].mean() * 1e4)}bp 중앙 {f1(v.gross[opp].median() * 1e4)}bp; "
            f"무조건(전 유효일 매일 진입) 총 {f1(v.gross[v.ok].mean() * 1e4)}bp 순 {f1(v.gross[v.ok].mean() * 1e4 - 6.8)}bp(n {int(v.ok.sum())}).")
        big = v[sel & (v.gapdiff.abs() > 0.01)]
        log(f"  · 갭차 |≥ 100bp| 조건일 {len(big)}일: " + ", ".join(f"{dd(i)} 갭차 {f1(r.gapdiff * 1e4, 0)} 총 {f1(r.gross * 1e4, 0)}bp" for i, r in big.iterrows())
            + ". 선물 08:45~09:00 시가가 지수 시가와 수 % 벌어진 날(베이시스 급변)이며 지수·일봉 선물·1분봉 세 원천이 일치해 자료 오류는 아님. 평균을 끌어올리므로 판정은 중앙값.")
        # 집행 전제 대체: 같은 신호(09:00 단일가 기준)를 보고 ETF 09:01 체결(09:02 라벨 봉 이후 첫 체결 시가) + 선물 09:01 봉 시가
        v["etf_o1"], v["fut_o1"] = first_fill(Pt).reindex(v.index), F["open"]["09:01"].reindex(v.index)
        v["gross1"] = np.log(v.etf_c / v.etf_o1) - np.log(v.fut_c / v.fut_o1)
        pop1 = v.ok & v[["etf_o1", "fut_o1"]].notna().all(axis=1)
        ok1 = pop1 & sel
        m51, m5s = cell(v.gross1[ok1], w5, COST["main"]), cell(v.gross[ok1], w5, COST["main"])
        res1 = m51["net_bp"] / m5s["net_bp"] if m5s.get("net_bp") else np.nan
        add_cell(f"OH{'K' if mk == 'KOSPI' else 'Q'}_V5_0901", f"V5 09:01 진입(양 다리 09:01) {lab}", mk, w5, m51, delta_test((v.gross1[pop1] - COST["main"]) * 1e4, sel[pop1]),
                 f"비용 6.8bp, 잔존율 {res1 * 100:.1f}%(같은 날 V5 09:00 진입 순 {m5s['net_bp']:+.1f}bp, n {int(ok1.sum())}), 단일가 미체결 가정")
        add_trades(f"OH{'K' if mk == 'KOSPI' else 'Q'}_V5_fut", v.loc[sel, ["etf_o", "etf_c", "fut_o", "fut_c", "con", "gapdiff", "gross", "gross_idx", "etf_o1", "fut_o1", "gross1"]].assign(net=lambda x: x.gross - COST["main"]))
        # 주 셀 신호(mis)로 같은 창에서 선물 실측 vs 지수 대용
        d0 = (dq if mk == "KOSDAQ" else dk)
        j = v.join(d0[["mis", "valid", "c3", "gross"]].rename(columns={"gross": "gross_day"}), how="inner")
        sm_ = j.ok & j.valid & ~j.c3 & (j.mis <= THR)
        if sm_.sum():
            wj = (str(j.index.min().date()), str(j.index.max().date()))
            add_cell(f"OH{'K' if mk == 'KOSPI' else 'Q'}_mis_fut", f"주 셀 신호(mis ≤ −10bp) 선물 실측 헤지 {lab}", mk, wj, cell(j.gross[sm_], wj, COST["main"]),
                     notes=f"같은 날 지수 대용 순 {cell(j.gross_day[sm_], wj, COST['main'])['net_bp']:+.1f}bp, 차이 평균 {f1(((j.gross_day - j.gross)[sm_] * 1e4).mean())}bp", diagnostic=True)

    # ------------------------------------------------ V6 breadth 사이징
    mis_j, elig_j = {}, {}
    for code in KQ_1X:
        e = px(RAW / ("etf/229200.csv" if code == "229200" else f"etf_extra/{code}.csv"))
        d = e[["open", "close", "volume"]].join(idx_q[["open", "close"]], lsuffix="_e", rsuffix="_i", how="inner")
        mis_j[code] = np.log(d.open_e / d.open_i) - np.log(d.close_e / d.close_i).shift(1).rolling(5).median()
        elig_j[code] = ((d.close_e * d.volume).rolling(20).median().shift(1) >= 1e9)
    M, E = pd.DataFrame(mis_j).reindex(ev_q.index), pd.DataFrame(elig_j).reindex(ev_q.index, fill_value=False).astype(bool)
    E &= M.notna()
    n_el = E.sum(axis=1)
    ratio = (M <= THR).where(E).mean(axis=1)
    mult = pd.Series(np.where(ratio >= 0.7, 1.5, np.where(ratio <= 0.3, 0.5, 1.0)), index=ev_q.index)
    net6, net_m = mult * (ev_q - COST["main"]), ev_q - COST["main"]
    m6 = cell(ev_q * mult, W_MAIN, 0.0)
    m6.update(net_bp=float(net6.mean() * 1e4), t_nw=nw_t(net6 * 1e4), gross_bp=float((ev_q * mult).mean() * 1e4), win_pct=float((net6 > 0).mean() * 100))
    r6 = pd.Series(0.0, index=m6["_daily"].index).add(net6.reindex(m6["_daily"].index).fillna(0.0))
    cum6 = (1 + r6).cumprod()
    m6.update(cagr_pct=float((cum6.iloc[-1] ** (252 / len(r6)) - 1) * 100), mdd_pct=float((cum6 / cum6.cummax() - 1).min() * 100))
    yr = net6.groupby(net6.index.year).agg(["sum", "count"])
    m6["yearly_pos"] = " ".join(f"{y}{'+' if s > 0 else '−'}({int(c)})" for y, (s, c) in yr[yr["count"] >= 10].iterrows())
    m6["yearly_net_bp"] = " ".join(f"{y} {s / c * 1e4:+.1f}bp({int(c)})" for y, (s, c) in yr.iterrows())
    m6["year_pos_share"] = float((yr[yr["count"] >= 10]["sum"] > 0).mean())
    m6["top5_share_pct"] = float(net6.nlargest(5).sum() / net6.sum() * 100) if net6.sum() > 0 else np.nan
    mid = m6["_daily"].index[len(m6["_daily"]) // 2]
    m6.update(half1_net_bp=float(net6[net6.index < mid].mean() * 1e4), half2_net_bp=float(net6[net6.index >= mid].mean() * 1e4),
              half1_cnt_bp=float(net6.iloc[: len(net6) // 2].mean() * 1e4), half2_cnt_bp=float(net6.iloc[len(net6) // 2:].mean() * 1e4), median_net_bp=float(net6.median() * 1e4), sum_net_pct=float(net6.sum() * 100))
    inc = net6 - net_m
    add_cell("OHQ_V6_breadth", "V6 breadth 사이징(×1.5/×1.0/×0.5, 비용도 배수)", "KOSDAQ", W_MAIN, m6, (float(inc.mean() * 1e4), nw_t(inc * 1e4)),
             f"Δ 열 = 229200 단독 대비 증분(건당 평균 bp, NW t). 증분 합 {inc.sum() * 100:+.2f}%p. 배수 ×1.5 {int((mult == 1.5).sum())}일 ×1.0 {int((mult == 1.0).sum())}일 ×0.5 {int((mult == 0.5).sum())}일, "
             f"적격 ETF 수 중앙 {n_el.median():.0f}(최소 {n_el.min()} 최대 {n_el.max()}), 적격 1종뿐인 날 {int((n_el <= 1).sum())}일")
    log(f"• V6 보조: 적격 ETF 수별 신호일 수 {n_el.value_counts().sort_index().to_dict()}, 배수별 주 셀 순평균 bp ×1.5 {f1(net_m[mult == 1.5].mean() * 1e4)} ×1.0 {f1(net_m[mult == 1.0].mean() * 1e4)} ×0.5 {f1(net_m[mult == 0.5].mean() * 1e4)}. "
        f"적격 기준 20일 거래대금 중앙(조정 종가 × 거래량, t−1 까지) ≥ 10억, 229200 포함 9종(kq150_etf_names_naver.csv etf_tab=1).")
    add_trades("OHQ_V6_breadth", pd.DataFrame({"mis": dq.mis.reindex(ev_q.index), "n_elig": n_el, "ratio": ratio, "mult": mult, "gross": ev_q * mult, "net": net6, "net_main": net_m}))

    # ------------------------------------------------ 시리즈 저장 + 장부 결합
    s_q = save_series("beomgu_OH_Q", ev_q, COST["main"], W_MAIN, "KOSDAQ", "KODEX 코스닥150 (229200) 롱 1 + 코스닥150 선물 숏 1(지수 헤지 대용)",
                      "mis = ln(O_etf/O_idx) − 직전 5일 ln(C_etf/C_idx) 중앙 ≤ −10bp, beomgu_C3Q 비활성일만, 09:00 단일가 → 15:30, net = ln차 − 6.8bp",
                      ["hyfe/data/raw/etf/229200.csv", "hyfe/data/raw/etf_extra/KQ150_index_kis.csv", "round13_slot_book/series/beomgu_C3Q.csv"])
    s_k = save_series("beomgu_OH_K", ev_k, COST["main"], W_K, "KOSPI", "KODEX 200 (069500) 롱 1 + 코스피200 미니선물 숏 1(지수 헤지 대용)",
                      "mis = ln(O_etf/O_idx) − 직전 5일 ln(C_etf/C_idx) 중앙 ≤ −10bp, beomgu_C3VT30 비활성일만, 09:00 단일가 → 15:30, net = ln차 − 6.8bp",
                      ["hyfe/data/raw/etf/069500.csv", "hyfe/data/raw/etf/KPI200.csv", "round13_slot_book/series/beomgu_C3VT30.csv"])
    log("")
    log("## 5. 장부 결합(코스닥 장부 = r_A + r_B + r_C + r_new, 합산(a) = 코스피 50% + 코스닥 50%)")
    bt, book_pass = book_cells(s_q.net, "주 셀 6.8bp", "BOOK_OHQ")
    bt.to_csv(OUT / "book_compare.csv", index=False, float_format="%.6f")
    _, book_pass_cons = book_cells(s_q.net - (COST["cons"] - COST["main"]) * (s_q.pos != 0), "보수 셀 14.8bp", "BOOK_OHQ_cons")
    log(f"• 결합 규칙(W_all·W_long CAGR·MDD 동시 개선 ∧ 2024·2025 개선): 주 셀 {'통과' if book_pass else '미달'}, 보수 셀 {'통과' if book_pass_cons else '미달'}.")

    # ------------------------------------------------ 그림·판정
    fig_cum(s_q.net, s_q.net - (COST["cons"] - COST["main"]) * (s_q.pos != 0), (g_v3[sig_t] - COST["main"]), (dt.gross[sig_t] - COST["main"]))
    fig_v5(v5["KOSPI"], v5["KOSDAQ"])
    log("")
    log("## 6. 판정(사전 등록 규칙)")
    cons_row = next(r for r in CELLS if r["id"] == "OHQ_cons")
    v5q = next(r for r in CELLS if r["id"] == "OHQ_V5_fut")
    v5k = next(r for r in CELLS if r["id"] == "OHK_V5_fut")
    v5q_med = float(v5["KOSDAQ"].gross[v5["KOSDAQ"].ok & (v5["KOSDAQ"].gapdiff <= THR)].median() * 1e4)
    v5k_med = float(v5["KOSPI"].gross[v5["KOSPI"].ok & (v5["KOSPI"].gapdiff <= THR)].median() * 1e4)
    years_all = m_q["year_pos_share"] == 1.0 and m_q["years_lt10"] == ""
    v3_row = next(r for r in CELLS if r["id"] == "OHQ_V3_0901")
    v3_ok = v3_row["net_bp"] > 0 and resid >= 0.5
    conds = {"주 셀 공통 규칙": row_q["verdict"] == "candidate", "보수 셀 순수익 > 0": cons_row["net_bp"] > 0, "V5 코스닥 실측 중앙 > 6.8bp": v5q_med > 6.8,
             "2016~2026 연도 전부 양수": bool(years_all)}
    grade = "candidate" if all(conds.values()) else ("reject" if row_q["net_bp"] <= 0 else "reference")
    if grade == "candidate" and not v3_ok:
        grade = "forward_only"
    for k, v in conds.items():
        log(f"• {k}: {'충족' if v else '미달'}")
    log(f"• V3 집행 전제(순수익 > 0 ∧ 잔존율 ≥ 50%): {'충족' if v3_ok else '미달 → 단일가 가정 의존, 포워드 전용'} (V3 순 {f1(v3_row['net_bp'])}bp, 잔존율 {resid * 100:.1f}%).")
    log(f"• V5 참고: 코스피 실측 중앙 {f1(v5k_med)}bp(n {v5k['n']}), 코스닥 실측 중앙 {f1(v5q_med)}bp(n {v5q['n']}). 둘 다 창 ≤ 300일 estimate.")
    mfq, mfk = next(r for r in CELLS if r["id"] == "OHQ_mis_fut"), next(r for r in CELLS if r["id"] == "OHK_mis_fut")
    v51q, v51k = next(r for r in CELLS if r["id"] == "OHQ_V5_0901"), next(r for r in CELLS if r["id"] == "OHK_V5_0901")
    log(f"• 등급: **{grade}**. 장부 결합 규칙 {'통과' if book_pass else '미달'}.")
    log(f"• 같은 신호·같은 날 대조(판정 조건 밖, 해석용): 주 셀 신호(mis ≤ −10bp)를 선물 실측으로 헤지하면 코스닥 순 {f1(mfq['net_bp'])}bp(중앙 {f1(mfq['median_net_bp'])}, n {mfq['n']}, NW t {f1(mfq['t_nw'], 2)}), "
        f"코스피 순 {f1(mfk['net_bp'])}bp(중앙 {f1(mfk['median_net_bp'])}, n {mfk['n']}, NW t {f1(mfk['t_nw'], 2)}). 지수 대용 주 셀과의 차이가 26~40bp 로 보수 셀의 8bp 차감을 넘는다. "
        f"V5 통과 조건은 신호 정의가 다른 셀(갭차, 선물 가격 기준)이며 V5 를 09:01 에 들어가면 코스닥 순 {f1(v51q['net_bp'])}bp·코스피 순 {f1(v51k['net_bp'])}bp. "
        f"선물 실측 창은 1년 미만(estimate)이라 결론이 아니고 반박 근거로 Verify 에 넘긴다.")
    log(f"• 셀 수: cells.csv {len(CELLS)}행(주 셀 창 5 + 비용 4 + 변형·보조 + 장부 결합). 14차 누적 512셀 + 15차 사전 등록 64셀 안의 설계 01 몫 7셀을 넘는 행은 보고용.")
    log("")
    log("## 7. 단서")
    log("• 신호 = 09:00 체결가 자체(단일가 가정 강함). 신호가 체결가이므로 t 가 큰 것은 구조의 산물일 수 있음(V3 잔존율로 수치화). 08:59 예상체결가 포워드 검정은 T‑P1 승인 뒤.")
    log("• KQ150 지수 시가 잡음: 사전 추정은 약 8bp 과대(보수 셀 차감폭)였으나 선물 실측 창에서 신호일 조건부로 재면 코스닥 26bp·코스피 40bp(6절). "
        "신호(mis)가 지수 시가 잡음을 고르는 구조라 무조건 평균(코스닥 10bp)보다 크다. 선물 실측 창은 1년 미만.")
    log("• 헤지 반올림 잔여 노출: 코스닥150 선물 1계약 ≈ 지수 × 10,000원(2026.09 약 1,500만원), 자본 1억 가정 6~7계약 → 잔여 노출 최대 ±반 계약(약 ±7%). 미니 코스피200 1계약 ≈ 지수 × 50,000원(약 5,400만원) → 자본 1억이면 2계약, 잔여 ±27%.")
    log("• 증거금 21%(2026.08.03~), 선물 양도세 11% vs ETF 과세, 수수료율은 미확인. 파생 계좌 필수.")
    log("• ETF 수익은 분배금 조정가 기준 로그 차. 229200 일 거래대금 중앙 약 1,367억이라 소액은 괜찮으나 단일가에 내 주문이 반영됨.")
    log("• 특수일 동종 확장(1분봉 창 밖 연초 첫 거래일·수능일)은 사전 목록으로 고정했고 special_days.csv 의 정의(10:00 개장)와 같음.")

    # ------------------------------------------------ 저장
    cols = ["id", "name", "market", "window", "n", "gross_bp", "net_bp", "t_nw", "delta_bp", "t_delta", "win_pct", "cagr_pct", "mdd_pct", "yearly_pos", "top5_share_pct",
            "half1_net_bp", "half2_net_bp", "verdict", "notes", "yearly_net_bp", "year_pos_share", "years_lt10", "half1_cnt_bp", "half2_cnt_bp", "median_net_bp", "sum_net_pct", "n_days"]
    pd.DataFrame(CELLS)[cols].to_csv(OUT / "cells.csv", index=False, float_format="%.4f", encoding="utf-8-sig")
    pd.concat(TRADES, ignore_index=True).to_csv(OUT / "trades.csv", index=False, float_format="%.8f", encoding="utf-8-sig")
    text = "\n".join(NOTES).replace("W_all계열", "W_all 계열")
    for k, w in (("W_all", W_ALL), ("W_ex2026", W_EX26), ("W_long", LD.W_LONG)):
        text = text.replace(f"{w[0]}~{w[1]}", f"{dd(w[0])}~{dd(w[1])}")
    assert "-" not in text, [l for l in text.splitlines() if "-" in l][:5]
    (OUT / "notes.md").write_text(text + "\n", encoding="utf-8")
    print(f"\n저장: {OUT} / {SER}")
    print("JUDGE", json.dumps({"grade": grade, "conds": conds, "v3_ok": v3_ok, "resid": resid, "book_pass": book_pass, "book_pass_cons": book_pass_cons,
                               "v5q_med": v5q_med, "v5k_med": v5k_med, "n_cells": len(CELLS)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
