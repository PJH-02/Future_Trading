"""R3 · D2 3차 확인(H0 = 결정일 2003-01-02~2009-12-30): 잔차 시가 갭 지속 — 실행형 신호·메커니즘 분해

고정 사양: <OUT>/round3_D2/spec.json (탐색 담당이 A 구간 2010-01-04~2019-12-30 만 보고 고정. 여기서는 바꾸지 않는다)
- 한국 달력 = 069500 일봉 행, prev(T) = 직전 행. 069500 상장(2002-10-14) 전 번인은 코스피200 지수(FDR KS200) 행을 달력으로 쓴다
  (시가·종가 > 0 인 행, 2000-01-01 부터. 창 500 이웃도 번인 행이 500개 넘게 차도록 h0_quality 의 2001-01 보다 1년 앞에서 시작한다.
  창 250·120 z 는 두 시작점에서 같다).
- 지수 = 2014-07-07 이전 FDR KS200, 2014-07-08 이후 fchart KPI200(빈 행만 FDR). r2_d2.load_index 를 그대로 쓴다.
- gi(T) = ln(IDX_o(T)/IDX_c(prev T)), ge(T) = ln(ETF_o(T)/ETF_c(prev T)) (069500 네이버 수정주가).
- us(T) = r2_d2.us_cum(SPX): D(X) = 미국 날짜 < X 인 마지막 ^GSPC 세션, ln 종가(D(T)) − ln 종가(D(prev T)). 종가 결측·파일 끝 뒤 평일이면 NaN.
- z: T 제외 직전 W 개 달력 행 중 g·us 유효 행 ≥ 최소면 OLS g = a + b·us, s = sqrt(SSR/(n−2)), z = (g − a − b·us)/s (닫힌 해, r2_d2.rolling_z 와 대조).
- mis(T) = ln(ETF_o/IDX_o)(T) − median_{j=1..5} ln(ETF_c/IDX_c)(T−j) (5개 모두 있어야 함).
- R3-C1: ze ≥ 1 / R3-C2: zi ≥ 1 / R3-C3: zi ≥ 1 & mis ≤ 0. 모두 069500 롱, T 시가 단일가 매수 → T 종가 단일가 매도(r_oc). NaN 은 표본 제외.
- 이웃(⑤): C1·C2 = 창120(최소 96) / 창500(최소 400) / 임계 0.75 / 1.25. C3 = zi ≥ 0.75 & mis ≤ 0 / zi ≥ 1.25 & mis ≤ 0 /
  mis10 ≤ 0(직전 10개 종가 괴리 중앙값) / (ge − gi) ≤ 0.
확인 판정(H0 1회, 사양 confirm_protocol_H0):
 ① Δ(신호−비신호) > 0 이고 NW(lag5) HAC t 의 단측 p(정규 근사)가 셀 3개 Holm(m = 3) 5% 통과
 ② 7bp 차감 신호일 순평균 > 0   ③ 연도 층화 플라시보 p ≤ 0.05 (engine.placebo_year_strat, 2000회, seed 20260929)
 ④ 변동성 조정 Δ > 0 (r_oc / T 제외 직전 60일 SD, common.vol_adj)   ⑤ 이웃 4개 H0 Δ 모두 > 0
 ⑥ 상위 10일 제거 신호일 총평균 > 0
 등급: ①~⑥ 모두 → 통과 / ①(Holm 전 단측 p < 0.05, Δ > 0)·②·③ → 보류 / 그 밖 → 기각.
보고만: A(표본 내), B(2020-01-01~2025-06-05, 2차에서 이미 씀; 2023-07-31 전후), E4(2025-06-09~; 지수 시가 지연 의심일 제외판 병기),
 H0-1(2003~2006)/H0-2(2007~2009), 연도별, 분해(지수 r_oc 몫 + ETF−지수 단일가 몫), 102110 이전(2008-04 상장), 익일 시가 청산,
 민감도(① 2004 제외 ② h0_quality 괴리 > 100bp 23일·지수 지연 후보 11일 제외 ③ 2008-09-01~2009-03-31 제외), 비용 5·7·10bp.
실행: python topics/r3_d2.py [--dry: 장부 기록 생략]   (산출 <OUT>/round3_D2/report.md, verdict.json, 시행 장부 <OUT>/ledger.csv 주제 R3_D2)
"""
import datetime as dt
import json
import sys
from math import erf, sqrt
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
import data  # noqa: E402
import engine as E  # noqa: E402
import r2_d2 as R2  # noqa: E402  (2차 규약 함수 재사용: load_index, us_cum, rolling_z, decomposition, delta_only)

TOPIC = "R3_D2"
OUTD = data.OUT / "round3_D2"
SPEC_FILE = data.result_file("round3_D2/spec.json")          # 출력 폴더에 없으면 저장소 결과 폴더의 사양을 읽음
H0Q_FILE = data.result_file("round3_D2/h0_quality.json")
CAL0 = "2000-01-01"                       # 상장 전 번인 달력 시작
H0 = ("2003-01-01", "2009-12-31")
WIN = {"H0": H0, "A": ("2010-01-01", E.A_END), "B": ("2020-01-01", E.JUDGE_END), "E4": (E.E4_START, None)}
SUB = {"H0-1(2003~2006)": ("2003-01-01", "2006-12-31"), "H0-2(2007~2009)": ("2007-01-01", "2009-12-31"),
       "B 전반(2020-01~2023-07-28)": ("2020-01-01", "2023-07-28"),
       "B 후반(2023-07-31~2025-06-05, 파생 08:45 개장)": ("2023-07-31", E.JUDGE_END)}
M_CELLS = 3
ALPHA = 0.05
F = C._f
md_table, pfmt = R2.md_table, R2.pfmt


def p_one(t):
    return np.nan if t is None or np.isnan(t) else 0.5 * (1 - erf(t / sqrt(2)))


# ---------------------------------------------------------------- 자료·신호
def rolling_z1(g, x, win=250, minr=200):
    """단일 회귀변수 닫힌 해. T 제외 직전 win 행 중 g·x 유효 행 n ≥ minr 이면 OLS, s = sqrt(SSR/(n−2)), z = 잔차(T)/s."""
    G, X = g.to_numpy(float), x.to_numpy(float)
    ok = ~np.isnan(G) & ~np.isnan(X)
    g0, x0, o = np.where(ok, G, 0.0), np.where(ok, X, 0.0), ok.astype(float)
    cs = [np.concatenate([[0.0], np.cumsum(v)]) for v in (o, x0, g0, x0 * x0, x0 * g0, g0 * g0)]
    i = np.arange(len(G))
    lo = np.maximum(0, i - win)
    n, sx, sy, sxx, sxy, syy = [c[i] - c[lo] for c in cs]
    with np.errstate(invalid="ignore", divide="ignore"):
        mx, my = sx / n, sy / n
        vxx, vxy, vyy = sxx - n * mx * mx, sxy - n * mx * my, syy - n * my * my
        b = vxy / vxx
        a = my - b * mx
        z = (G - a - b * X) / np.sqrt((vyy - b * vxy) / (n - 2))
    z[~ok | (n < minr)] = np.nan
    return pd.Series(z, index=g.index)


def load():
    px = data.price("069500")                                    # 원자료 etf/ 캐시(읽기만)
    fdr = pd.read_csv(data.RAW / "k200_index_daily.csv", parse_dates=["date"], index_col="date").sort_index()
    fdr = fdr[(fdr["open"] > 0) & (fdr["close"] > 0)]
    listed = px.index.min()
    cal = fdr.loc[CAL0:listed - pd.Timedelta(days=1)].index.append(px.index)
    assert cal.is_monotonic_increasing and cal.is_unique
    ix = R2.load_index().reindex(cal)
    tg = data.price("102110").reindex(cal)
    return px.reindex(cal), ix, tg, cal, listed


def B_(z, k):
    return (z >= k).astype(float).where(z.notna())


def le0(x):
    return (x <= 0).astype(float).where(x.notna())


def both(a, b):
    return (a * b).where(a.notna() & b.notna())


# ---------------------------------------------------------------- 통계
def wstats(sig, ret, volr, a, b, excl=None, placebo=True):
    d = pd.DataFrame({"s": sig, "r": ret, "v": volr}).loc[a:b]
    if excl is not None:
        d = d[~d.index.isin(excl)]
    d = d[d["s"].notna() & d["r"].notna()]
    s, r = d["s"], d["r"]
    held = r[s == 1]
    out = dict(days=len(d), n_sig=int((s == 1).sum()), first=str(d.index.min().date()) if len(d) else None,
               last=str(d.index.max().date()) if len(d) else None)
    if len(held) < 5 or (s == 0).sum() < 5:
        return {**out, **{k: np.nan for k in ("gross_bp", "t_gross", "median_bp", "win", "net5_bp", "net7_bp", "net10_bp", "base_bp",
                                              "nonsig_bp", "delta_bp", "t", "p_one", "dvol", "placebo_p", "drop5_bp", "drop10_bp")},
                "yrs_pos5": 0, "n_yrs": 0}
    dl, t, _ = E.delta_test(s, r)
    dv = E.delta_test(s[d["v"].notna()], d["v"].dropna())[0]
    hs = held.sort_values()
    yrs = held.groupby(held.index.year).mean()
    return {**out, "gross_bp": held.mean() * 1e4, "t_gross": E.nw_t(held), "median_bp": held.median() * 1e4, "win": (held > 0).mean(),
            "net5_bp": (held.mean() - 5e-4) * 1e4, "net7_bp": (held.mean() - 7e-4) * 1e4, "net10_bp": (held.mean() - 10e-4) * 1e4,
            "base_bp": r.mean() * 1e4, "nonsig_bp": r[s == 0].mean() * 1e4, "delta_bp": dl * 1e4, "t": t, "p_one": p_one(t), "dvol": dv,
            "placebo_p": E.placebo_year_strat(s, r) if placebo else np.nan,
            "drop5_bp": hs.iloc[:-5].mean() * 1e4, "drop10_bp": hs.iloc[:-10].mean() * 1e4 if len(hs) > 10 else np.nan,
            "yrs_pos5": int(((yrs - 5e-4) > 0).sum()), "n_yrs": int(len(yrs))}


def dlt(sig, ret, a, b, excl=None):
    d = pd.DataFrame({"s": sig, "r": ret}).loc[a:b]
    if excl is not None:
        d = d[~d.index.isin(excl)]
    d = d.dropna()
    held = d["r"][d["s"] == 1]
    x, t, n = E.delta_test(d["s"], d["r"])
    return dict(n_sig=int((d["s"] == 1).sum()), days=len(d), gross_bp=held.mean() * 1e4 if len(held) else np.nan, delta_bp=x * 1e4, t=t)


def holm(pvals, deltas, alpha=ALPHA):
    """단측 p 의 Holm 단계 하강. Δ ≤ 0 이거나 p NaN 이면 그 자리에서 멈춘다."""
    order = sorted(pvals, key=lambda k: (np.inf if np.isnan(pvals[k]) else pvals[k]))
    m = len(order)
    res, alive = {}, True
    for i, k in enumerate(order):
        thr = alpha / (m - i)
        ok = alive and not np.isnan(pvals[k]) and deltas[k] > 0 and pvals[k] <= thr
        res[k] = dict(rank=i + 1, threshold=thr, passed=bool(ok))
        alive = alive and ok
    return res


def strat_rows(sig, ret, label, wins):
    rows = []
    for w, (a, b, costs) in wins.items():
        rr = ret.loc[a:b]
        ss = sig.reindex(rr.index)
        keep = ss.notna() & rr.notna()
        if keep.sum() == 0:
            continue
        for cost in costs:
            for name, pos in (("매일 시가→종가(기준선)", pd.Series(1.0, index=rr.index)[keep]), (label, ss[keep])):
                st = E.stats(pos, rr[keep], cost)
                st.update(rule=name, window=w, cost_bp=cost)
                rows.append(st)
    return pd.DataFrame(rows)


def ledger_table(rules, wins):
    """rules: [(이름, 자산, 포지션, 수익률, 기준선 여부)], wins: {이름: (시작, 끝, 제외일, 비용들)} → LEDGER_COLS 형식 행."""
    rows = []
    for w, (a, b, excl, costs) in wins.items():
        for name, asset, pos, ret, base in rules:
            d = pd.DataFrame({"p": pos, "r": ret}).loc[a:b]
            if excl is not None:
                d = d[~d.index.isin(excl)]
            d = d.dropna()
            if len(d) == 0:
                continue
            rp = np.nan if base else E.random_pct(d["p"], d["r"], 5.0)
            dl, t = (np.nan, np.nan) if base else E.delta_test(d["p"], d["r"])[:2]
            for cost in costs:
                st = E.stats(d["p"], d["r"], cost)
                st.update(rule=name, window=w, cost_bp=cost, asset=asset, rand_pct=rp,
                          delta_bp=np.nan if base else dl * 1e4, t_delta=t)
                rows.append(st)
    return pd.DataFrame(rows)


def ledger_once(spec, table, note):
    """같은 주제·같은 사양 문자열이 장부에 이미 있으면 다시 쓰지 않는다(재실행이 시행 수를 부풀리지 않게)."""
    p = data.OUT / "ledger.csv"
    key = json.dumps(spec, ensure_ascii=False)
    if p.exists():
        led = pd.read_csv(p, usecols=["run_at", "topic", "spec"], encoding="utf-8-sig", dtype=str)
        hit = led[(led["topic"] == TOPIC) & (led["spec"] == key)]
        if len(hit):
            return False, hit["run_at"].min()
    E.ledger_append(data.OUT, TOPIC, spec, table, note=note)
    return True, None


# ---------------------------------------------------------------- 본체
def main(dry=False):
    t_run = dt.datetime.now(E.KST).strftime("%Y-%m-%d %H:%M KST")
    spec = json.loads(SPEC_FILE.read_text(encoding="utf-8"))
    h0q = json.loads(H0Q_FILE.read_text(encoding="utf-8"))
    fs = {x["cell_id"]: x for x in spec["frozen_specs"]}
    excl_years = [int(y) for y in spec.get("h0_excluded_years", [])]

    px, ix, tg, cal, listed = load()
    ge = np.log(px["open"] / px["close"].shift(1))
    gi = np.log(ix["open"] / ix["close"].shift(1))
    dev = ge - gi
    us = R2.us_cum(cal, "spx")
    Z = {(g, w): rolling_z1(gg, us, w, mn) for g, gg in (("i", gi), ("e", ge)) for w, mn in ((250, 200), (120, 96), (500, 400))}
    zi, ze = Z[("i", 250)], Z[("e", 250)]
    # 닫힌 해 대조: 2차 r2_d2.rolling_z(행별 lstsq)와 같은지(2002-06~2010-06 구간)
    seg = slice("2002-06-01", "2010-06-30")
    zchk = float(max((zi.loc[seg] - R2.rolling_z(gi.loc[:"2010-06-30"], us.loc[:"2010-06-30"].to_frame(), 250, 200).loc[seg]).abs().max(),
                     (ze.loc[seg] - R2.rolling_z(ge.loc[:"2010-06-30"], us.loc[:"2010-06-30"].to_frame(), 250, 200).loc[seg]).abs().max()))
    assert zchk < 1e-8, zchk
    prem_c = np.log(px["close"] / ix["close"])
    prem_o = np.log(px["open"] / ix["open"])
    mis = prem_o - prem_c.shift(1).rolling(5).median()
    mis10 = prem_o - prem_c.shift(1).rolling(10).median()

    r_oc = px["close"] / px["open"] - 1
    r_oo = px["open"].shift(-1) / px["open"] - 1
    r_on = px["open"].shift(-1) / px["close"] - 1
    ri_oc = ix["close"] / ix["open"] - 1
    rt_oc = tg["close"] / tg["open"] - 1
    volr = C.vol_adj(r_oc.dropna(), 60).reindex(cal)

    cells = {
        "R3-C1": dict(sig=B_(ze, 1.0), label="ze ≥ 1 (069500 자체 갭)",
                      neighbors={"창120": B_(Z[("e", 120)], 1.0), "창500": B_(Z[("e", 500)], 1.0),
                                 "임계0.75": B_(ze, 0.75), "임계1.25": B_(ze, 1.25)}),
        "R3-C2": dict(sig=B_(zi, 1.0), label="zi ≥ 1 (지수 갭, 2차 D2-C1)",
                      neighbors={"창120": B_(Z[("i", 120)], 1.0), "창500": B_(Z[("i", 500)], 1.0),
                                 "임계0.75": B_(zi, 0.75), "임계1.25": B_(zi, 1.25)}),
        "R3-C3": dict(sig=both(B_(zi, 1.0), le0(mis)), label="zi ≥ 1 & mis ≤ 0",
                      neighbors={"임계0.75": both(B_(zi, 0.75), le0(mis)), "임계1.25": both(B_(zi, 1.25), le0(mis)),
                                 "괴리 기준 10일": both(B_(zi, 1.0), le0(mis10)), "갭 차 dev≤0": both(B_(zi, 1.0), le0(dev))}),
    }

    # 판정 구간 제외(사양 h0_excluded_years; 이번 사양은 빈 목록)
    h0_excl = cal[cal.year.isin(excl_years)] if excl_years else None
    # 민감도·보고용 제외일
    anom = sorted({pd.Timestamp(x["date"]) for x in h0q["h0_dev_anomalies"]} | {pd.Timestamp(x) for x in h0q["h0_idx_lag_candidates"]})
    sens = {"① 2004년 제외": cal[cal.year == 2004],
            f"② 괴리 이상 {len(h0q['h0_dev_anomalies'])}일·지수 지연 후보 {len(h0q['h0_idx_lag_candidates'])}일 제외({len(anom)}일)": pd.DatetimeIndex(anom),
            "③ 2008-09-01~2009-03-31 제외": cal[(cal >= "2008-09-01") & (cal <= "2009-03-31")]}
    # 사양 밖 추가 점검(판정 불사용): ETF 시가 = 전일 종가(0 갭, 시가 미체결 의심)인 H0 날 제외
    zero_gap = cal[(ge == 0).to_numpy() & (cal >= H0[0]) & (cal <= H0[1])]
    extra = {f"(사양 밖) ETF 0 갭일 {len(zero_gap)}일 제외": zero_gap}
    # 사양 밖 보조 규칙(보고만): C2 중 C3 가 아닌 날 = zi ≥ 1 & mis > 0
    c2_not_c3 = both(B_(zi, 1.0), (mis > 0).astype(float).where(mis.notna()))
    lagd = (ge.abs() > 0.01) & (gi.abs() < 0.3 * ge.abs())
    e4_lag = lagd.loc[E.E4_START:][lagd.loc[E.E4_START:]].index
    b_lag = lagd.loc[WIN["B"][0]:WIN["B"][1]][lagd.loc[WIN["B"][0]:WIN["B"][1]]].index

    # ------------------------------------------------------------ 계산
    res = {}
    for cid, c in cells.items():
        sig = c["sig"]
        r = {"win": {}, "sub": {}, "sens": {}, "extra": {}, "neighbors": {}, "transfer": {}, "exit_oo": {}, "exit_on": {}, "decomp": {}, "yearly": []}
        for w, (a, b) in WIN.items():
            r["win"][w] = wstats(sig, r_oc, volr, a, b, excl=h0_excl if w == "H0" else None)
            r["neighbors"][w] = {k: dlt(v, r_oc, a, b, excl=h0_excl if w == "H0" else None) for k, v in c["neighbors"].items()}
            r["transfer"][w] = dlt(sig, rt_oc, a, b)
            r["exit_oo"][w] = dlt(sig, r_oo, a, b)
            r["exit_on"][w] = dlt(sig, r_on, a, b)
            r["decomp"][w] = R2.decomposition(sig, r_oc, ri_oc, dev, a, b)
        r["win"]["E4 지연 의심일 제외"] = wstats(sig, r_oc, volr, E.E4_START, None, excl=e4_lag)
        r["decomp"]["E4 지연 의심일 제외"] = R2.decomposition(sig.drop(e4_lag), r_oc.drop(e4_lag), ri_oc.drop(e4_lag), dev.drop(e4_lag),
                                                         E.E4_START, None)
        for k, (a, b) in SUB.items():
            r["sub"][k] = wstats(sig, r_oc, volr, a, b)
            r["decomp"][k] = R2.decomposition(sig, r_oc, ri_oc, dev, a, b)
        for k, ex in sens.items():
            r["sens"][k] = wstats(sig, r_oc, volr, *H0, excl=ex)
            r["sens"][k]["neighbors"] = {kk: dlt(v, r_oc, *H0, excl=ex)["delta_bp"] for kk, v in c["neighbors"].items()}
        for k, ex in extra.items():
            r["extra"][k] = wstats(sig, r_oc, volr, *H0, excl=ex)
            r["extra"][k]["neighbors"] = {kk: dlt(v, r_oc, *H0, excl=ex)["delta_bp"] for kk, v in c["neighbors"].items()}
            r["extra"][k]["sig_days_hit"] = int(((sig == 1) & sig.index.isin(ex)).sum())
        d = pd.DataFrame({"s": sig, "r": r_oc, "ri": ri_oc}).loc["2003-01-01":].dropna(subset=["s", "r"])
        for y, gy in d.groupby(d.index.year):
            h = gy["r"][gy["s"] == 1]
            x = E.delta_test(gy["s"], gy["r"])[0] if len(h) >= 5 else np.nan
            r["yearly"].append(dict(year=int(y), days=len(gy), n=len(h), gross_bp=h.mean() * 1e4 if len(h) else np.nan,
                                    net5_bp=(h.mean() - 5e-4) * 1e4 if len(h) else np.nan, delta_bp=x * 1e4 if not np.isnan(x) else np.nan,
                                    idx_bp=gy["ri"][gy["s"] == 1].mean() * 1e4 if len(h) else np.nan))
        r["strat"] = strat_rows(sig, r_oc, f"{cid} 신호일만", {"H0": (*H0, (5.0, 7.0, 10.0)), "A": (*WIN["A"], E.COSTS_BP),
                                                               "B": (*WIN["B"], E.COSTS_BP), "E4": (*WIN["E4"], E.COSTS_BP)})
        res[cid] = r

    # 판정(H0)
    pv = {cid: res[cid]["win"]["H0"]["p_one"] for cid in cells}
    dv = {cid: res[cid]["win"]["H0"]["delta_bp"] for cid in cells}
    hl = holm(pv, dv)
    for cid in cells:
        r, H = res[cid], res[cid]["win"]["H0"]
        nb = r["neighbors"]["H0"]
        neigh_ok = all(v["delta_bp"] > 0 for v in nb.values())
        crit = {"①": hl[cid]["passed"], "②": H["net7_bp"] > 0, "③": (not np.isnan(H["placebo_p"])) and H["placebo_p"] <= 0.05,
                "④": H["dvol"] > 0, "⑤": neigh_ok, "⑥": (not np.isnan(H["drop10_bp"])) and H["drop10_bp"] > 0}
        raw1 = H["delta_bp"] > 0 and H["p_one"] < 0.05
        grade = "통과" if all(crit.values()) else ("보류" if raw1 and crit["②"] and crit["③"] else "기각")
        r.update(holm=hl[cid], crit=crit, criteria="".join("O" if v else "X" for v in crit.values()), raw1=raw1, grade=grade, neighbors_ok=neigh_ok)

    # 겹침(H0)과 C2 중 C3 아닌 날(보고만)
    sh = pd.DataFrame({k: cells[k]["sig"] for k in cells}).loc[H0[0]:H0[1]]
    overlap = {"C1&C2": int(((sh["R3-C1"] == 1) & (sh["R3-C2"] == 1)).sum()), "C1": int((sh["R3-C1"] == 1).sum()),
               "C3_not_in_C2": int(((sh["R3-C3"] == 1) & (sh["R3-C2"] != 1)).sum())}
    comp = {w: dlt(c2_not_c3, r_oc, a, b) for w, (a, b) in WIN.items()}
    comp_dc = {w: R2.decomposition(c2_not_c3, r_oc, ri_oc, dev, a, b) for w, (a, b) in WIN.items()}

    # A 재현 대조(사양 A_evidence 의 첫 수치) · 2차 B 재현 대조
    r2v = None
    p2 = data.result_file("round2/D2/verdict.json")
    if p2.exists():
        r2v = json.loads(p2.read_text(encoding="utf-8"))["cells"]["D2-C1"]

    # 신호 가용성 대조(h0_quality.json 의 신호 수)
    avail_ref = {"R3-C1": h0q["signal_availability"]["R3-C1 ze≥1"], "R3-C2": h0q["signal_availability"]["R3-C2 zi≥1"],
                 "R3-C3": h0q["signal_availability"]["R3-C3 zi≥1&mis≤0"]}
    avail_chk = {cid: dict(ref_valid=avail_ref[cid]["valid_days"], got_valid=int(cells[cid]["sig"].loc[H0[0]:H0[1]].notna().sum()),
                           ref_first=avail_ref[cid]["first_valid"], got_first=str(cells[cid]["sig"].loc[H0[0]:H0[1]].dropna().index.min().date()),
                           ref_sig=sum(avail_ref[cid]["signals_by_year"].values()), got_sig=int((cells[cid]["sig"].loc[H0[0]:H0[1]] == 1).sum()))
                 for cid in cells}

    # ------------------------------------------------------------ 시행 장부
    rules = [("069500 매일 시가→종가", "069500", pd.Series(1.0, index=cal).where(r_oc.notna()), r_oc, True),
             ("102110 매일 시가→종가", "102110", pd.Series(1.0, index=cal).where(rt_oc.notna()), rt_oc, True),
             ("069500 매일 시가→익일 시가", "069500", pd.Series(1.0, index=cal).where(r_oo.notna()), r_oo, True)]
    for cid, c in cells.items():
        rules.append((f"{cid} {c['label']}", "069500", c["sig"], r_oc, False))
        rules += [(f"{cid} 이웃 {k}", "069500", v, r_oc, False) for k, v in c["neighbors"].items()]
        rules.append((f"{cid} 이전 102110", "102110", c["sig"], rt_oc, False))
        rules.append((f"{cid} 익일 시가 청산(oo)", "069500", c["sig"], r_oo, False))
    rules.append(("(사양 밖 보조) zi ≥ 1 & mis > 0 (C2 중 C3 아닌 날)", "069500", c2_not_c3, r_oc, False))
    lwins = {"H0(2003~2009, 판정)": (*H0, h0_excl, (5.0, 7.0, 10.0)),
             **{k: (a, b, None, E.COSTS_BP) for k, (a, b) in SUB.items()},
             **{f"H0 민감도 {k}": (*H0, ex, E.COSTS_BP) for k, ex in sens.items()},
             **{f"H0 추가 점검 {k}": (*H0, ex, E.COSTS_BP) for k, ex in extra.items()},
             "A(2010~2019, 표본 내)": (*WIN["A"], None, E.COSTS_BP), "B(2020~2025-06-05, 보고)": (*WIN["B"], None, E.COSTS_BP),
             "E4(2025-06-09~, 보고)": (E.E4_START, None, None, E.COSTS_BP),
             "E4 지연 의심일 제외(보고)": (E.E4_START, None, e4_lag, E.COSTS_BP)}
    table = ledger_table(rules, lwins)
    ledger_spec = {"spec_file": "round3_D2/spec.json", "family": "D2 잔차 갭 지속 3차(실행형·분해)", "target": "r_oc(T) 069500",
                   "z": "gap ~ SPX cum(D<T) OLS past250 cal rows excl T, min200, s=sqrt(SSR/(n-2)); pre-listing burn-in on KS200 calendar from 2000-01",
                   "cells": {"R3-C1": "ze(069500 gap)>=1 -> long 069500 O->C", "R3-C2": "zi(K200 gap)>=1 -> long 069500 O->C",
                             "R3-C3": "zi>=1 & mis5<=0 -> long 069500 O->C"},
                   "neighbors": "C1/C2: W120(min96)/W500(min400)/thr0.75/1.25; C3: zi0.75/zi1.25/mis10/dev<=0",
                   "report_only": "102110 transfer, next-open exit(oo), H0 sensitivity x3, A, B, E4(+lag-day excl)",
                   "off_spec_diagnostics": "H0 excl ETF zero-gap days; zi>=1 & mis>0 (C2 minus C3)",
                   "judge": "H0 2003-01-01~2009-12-31, Holm m=3, 7bp net, placebo 2000 seed 20260929, vol-adj, neighbors, drop-top10"}
    wrote, first_run = (False, "건너뜀(--dry)") if dry else ledger_once(ledger_spec, table, "3차 D2 확인(H0)")

    # ------------------------------------------------------------ 보고서
    zv = {w: (int(zi.loc[a:b].notna().sum()), int(r_oc.loc[a:b].notna().sum())) for w, (a, b) in WIN.items()}
    e4_last = zi.loc[E.E4_START:].dropna().index.max().date()
    md = ["# R3 · D2 잔차 시가 갭 지속 3차 확인 (H0 = 2003~2009)\n",
          f"- 실행 {t_run}. 사양 `round3_D2/spec.json`(고정, 변경 없음). 스크립트 `code/overnight_lab/topics/r3_d2.py`.",
          f"- 판정 구간 H0 = 결정일 {res['R3-C2']['win']['H0']['first']}~{res['R3-C2']['win']['H0']['last']}(1회, 제외 연도 {excl_years or '없음'}; "
          f"C1 은 z 번인 때문에 {res['R3-C1']['win']['H0']['first']}부터). "
          f"A = 2010~2019(탐색에 쓴 표본 내), B = 2020-01-01~2025-06-05(2차 확인에 이미 씀, 보고만), E4 = 2025-06-09~{e4_last}(보고만).",
          f"- 달력: 069500 상장({listed.date()}) 전은 FDR KS200 행({CAL0}~)으로 번인, 뒤는 069500 행. "
          f"H0 zi 유효 {zv['H0'][0]}일 / 069500 r_oc {zv['H0'][1]}일.",
          f"- z 닫힌 해와 2차 r2_d2.rolling_z(행별 lstsq) 최대 차이 {zchk:.1e}. "
          "h0_quality.json 신호 가용성과 대조: " + "; ".join(
              f"{cid} 유효 {v['got_valid']}(기준 {v['ref_valid']}), 첫 유효일 {v['got_first']}(기준 {v['ref_first']}), 신호 {v['got_sig']}(기준 {v['ref_sig']})"
              for cid, v in avail_chk.items()) + ".",
          f"- 시행 장부 `ledger.csv`: 주제 {TOPIC}, {len(table)}행(본 셀·이웃·102110 이전·익일 시가 청산·기준선 × {len(lwins)}개 기간 × 비용) "
          f"{'이번 실행에서 추가' if wrote else f'같은 사양으로 {first_run} 에 이미 기록돼 있어 다시 쓰지 않음'}.",
          "- 전제: 신호(09:00 지수·ETF 시가)와 체결(09:00 시가 단일가)이 같은 경매다. H0 결과는 신호가 09:00 시가로 확정된다는 가정 아래의 통계적 확인이며, "
          "당시 예상체결가·예상지수를 실시간으로 볼 수 있었는지와는 무관하다.",
          "\n```\n" + __doc__.strip() + "\n```\n"]

    md += ["## 1. 판정 요약 (H0)\n",
           md_table(["셀", "유효일 / 신호일", "Δ(신호−비신호)", "HAC t", "단측 p", f"① Holm(m={M_CELLS}) 순위·문턱", "② 7bp 순평균", "③ 플라시보 p",
                     "④ 변동성 조정 Δ", "⑤ 이웃 4개 Δ > 0", "⑥ 상위 10일 제거 평균", "기준 ①~⑥", "판정"],
                    [[cid, f"{res[cid]['win']['H0']['days']} / {res[cid]['win']['H0']['n_sig']}", f"{F(res[cid]['win']['H0']['delta_bp'])}bp",
                      F(res[cid]['win']['H0']['t']), pfmt(res[cid]['win']['H0']['p_one']),
                      f"{res[cid]['holm']['rank']}위 ≤ {res[cid]['holm']['threshold']:.4f} → {'통과' if res[cid]['holm']['passed'] else '실패'}",
                      f"{F(res[cid]['win']['H0']['net7_bp'])}bp", pfmt(res[cid]['win']['H0']['placebo_p']), f"{res[cid]['win']['H0']['dvol']:+.4f} SD",
                      "예" if res[cid]["neighbors_ok"] else "아니오", f"{F(res[cid]['win']['H0']['drop10_bp'])}bp", res[cid]["criteria"],
                      f"**{res[cid]['grade']}**"] for cid in cells]),
           "",
           "등급 규칙(사양): ①~⑥ 모두 충족이면 통과, ①(Holm 전 단측 p < 0.05)·②·③ 충족이면 보류, 그 밖은 기각. "
           "Holm 은 세 셀의 단측 p 를 작은 순으로 0.05/3, 0.05/2, 0.05 와 비교하고 처음 실패한 곳에서 멈춘다.\n"]

    for cid, c in cells.items():
        r = res[cid]
        W = r["win"]
        H, A, Bw, E4, E4x = W["H0"], W["A"], W["B"], W["E4"], W["E4 지연 의심일 제외"]
        cols = [("H0 (판정)", H), ("A (표본 내)", A), ("B (보고)", Bw), ("E4 (보고)", E4), ("E4 지연일 제외", E4x)]

        def row(lab, f, crit=""):
            return [lab, *[f(x) for _, x in cols[:1]], crit, *[f(x) for _, x in cols[1:]]]
        md += [f"## {cid}: {fs[cid]['name']}\n",
               f"사양 규칙: {fs[cid]['rule']}\n",
               f"### 판정표: **{r['grade']}** (기준 {r['criteria']})\n",
               md_table(["항목", "H0 (판정)", "기준", "A (표본 내)", "B (보고)", "E4 (보고)", f"E4 지연 의심일 {len(e4_lag)}일 제외"], [
                   row("유효일 / 신호일", lambda x: f"{x['days']} / {x['n_sig']}"),
                   row("신호일 총평균(비용 전), NW t", lambda x: f"{F(x['gross_bp'])}bp ({F(x['t_gross'])})"),
                   row("비신호일 평균 / 전체 평균", lambda x: f"{F(x['nonsig_bp'])} / {F(x['base_bp'])}"),
                   row("① Δ(신호−비신호), HAC t", lambda x: f"{F(x['delta_bp'])}bp (t {F(x['t'])})", "> 0"),
                   row("① 단측 p", lambda x: pfmt(x["p_one"]), f"Holm {r['holm']['rank']}위 ≤ {r['holm']['threshold']:.4f}"),
                   row("순평균 5bp", lambda x: f"{F(x['net5_bp'])}bp"),
                   row("② 순평균 7bp", lambda x: f"{F(x['net7_bp'])}bp", "> 0"),
                   row("순평균 10bp", lambda x: f"{F(x['net10_bp'])}bp"),
                   row("중앙값 / 승률(비용 전)", lambda x: f"{F(x['median_bp'])}bp / {x['win'] * 100:.1f}%" if not np.isnan(x["win"]) else "-"),
                   row("③ 연도 층화 플라시보 p", lambda x: pfmt(x["placebo_p"]), "≤ 0.05"),
                   row("④ 변동성 조정 Δ", lambda x: f"{x['dvol']:+.4f} SD" if not np.isnan(x["dvol"]) else "-", "> 0"),
                   row("상위 5일 제거 평균", lambda x: f"{F(x['drop5_bp'])}bp"),
                   row("⑥ 상위 10일 제거 평균", lambda x: f"{F(x['drop10_bp'])}bp", "> 0"),
                   row("5bp 후 양수 연도", lambda x: f"{x['yrs_pos5']}/{x['n_yrs']}"),
                   ["⑤ 이웃 4개 Δ", "모두 > 0" if r["neighbors_ok"] else "일부 ≤ 0", "모두 > 0", "", "", "", ""]]),
               "",
               "### 이웃(⑤는 H0 부호만)\n",
               md_table(["이웃", "H0 n / 총평균 / Δ (t)", "A n / 총평균 / Δ", "B n / 총평균 / Δ", "E4 n / 총평균 / Δ"],
                        [[k, *[f"{r['neighbors'][w][k]['n_sig']} / {F(r['neighbors'][w][k]['gross_bp'])} / {F(r['neighbors'][w][k]['delta_bp'])}"
                               + (f" ({F(r['neighbors'][w][k]['t'])})" if w == "H0" else "") for w in WIN]] for k in c["neighbors"]]),
               "",
               "### 하위 구간(보고)\n",
               md_table(["구간", "신호/유효일", "총평균", "순평균 7bp", "Δ (t)", "변동성 조정 Δ", "플라시보 p", "상위 10일 제거"],
                        [[k, f"{s['n_sig']}/{s['days']}", F(s["gross_bp"]), F(s["net7_bp"]), f"{F(s['delta_bp'])} ({F(s['t'])})",
                          f"{s['dvol']:+.4f}" if not np.isnan(s["dvol"]) else "-", pfmt(s["placebo_p"]), F(s["drop10_bp"])] for k, s in r["sub"].items()]),
               "",
               "### H0 민감도(판정 불사용; 마지막 행은 사양 밖 추가 점검)\n",
               md_table(["제외", "신호/유효일", "총평균", "순평균 7bp", "Δ (t)", "단측 p", "플라시보 p", "변동성 조정 Δ", "상위 10일 제거", "이웃 Δ"],
                        [[k, f"{s['n_sig']}/{s['days']}", F(s["gross_bp"]), F(s["net7_bp"]), f"{F(s['delta_bp'])} ({F(s['t'])})", pfmt(s["p_one"]),
                          pfmt(s["placebo_p"]), f"{s['dvol']:+.4f}", F(s["drop10_bp"]), ", ".join(f"{a} {F(b, '{:+.1f}')}" for a, b in s["neighbors"].items())]
                         for k, s in {**r["sens"], **r["extra"]}.items()]),
               "",
               "추가 점검 행: ETF 시가가 전일 종가와 같은 날(시가 단일가 미체결 의심)을 뺐다. 그중 신호일은 "
               + ", ".join(f"{x['sig_days_hit']}일" for k, x in r["extra"].items()) + "이다.\n",
               "### 분해: 지수 몫(코스피200 시가→종가, 거래 불가) + 단일가 몫(ETF − 지수)\n",
               "시가 괴리 = ETF 갭 − 지수 갭(bp). 음수면 ETF 시가가 지수 갭을 덜 반영했다는 뜻이다.\n",
               md_table(["구간", "신호일", "ETF r_oc", "지수 몫", "단일가 몫", "지수 몫 Δ (t)", "단일가 몫 Δ (t)", "시가 괴리 신호일 / 전체"],
                        [[w, x["n_sig"], F(x["etf_bp"]), F(x["idx_bp"]), F(x["single_bp"]), f"{F(x['idx_delta_bp'])} ({F(x['idx_t'])})",
                          f"{F(x['single_delta_bp'])} ({F(x['single_t'])})", f"{F(x['dev_sig_bp'])} / {F(x['dev_all_bp'])}"]
                         for w, x in r["decomp"].items()]),
               "",
               "### 102110 이전·익일 시가 청산(보고만)\n",
               md_table(["구간", "102110 n / 총평균 / Δ (t)", "익일 시가 청산 oo n / 총평균 / Δ (t)", "야간 몫 on Δ (t)"],
                        [[w, f"{r['transfer'][w]['n_sig']} / {F(r['transfer'][w]['gross_bp'])} / {F(r['transfer'][w]['delta_bp'])} ({F(r['transfer'][w]['t'])})",
                          f"{r['exit_oo'][w]['n_sig']} / {F(r['exit_oo'][w]['gross_bp'])} / {F(r['exit_oo'][w]['delta_bp'])} ({F(r['exit_oo'][w]['t'])})",
                          f"{F(r['exit_on'][w]['delta_bp'])} ({F(r['exit_on'][w]['t'])})"] for w in WIN]),
               "",
               f"102110 TIGER 200 은 2008-04-03 상장이라 H0 이전 표본은 2008-04~2009-12 뿐이다(유효일 {r['transfer']['H0']['days']}).\n",
               "### 연도별(신호일 총평균·5bp 순평균·Δ·같은 날 지수 몫)\n",
               md_table(["연도", "구간", "신호/유효일", "총평균", "순평균 5bp", "Δ", "지수 몫"],
                        [[y["year"], "H0" if y["year"] <= 2009 else ("A" if y["year"] <= 2019 else ("B" if y["year"] <= 2024 else ("B/E4" if y["year"] == 2025 else "E4"))),
                          f"{y['n']}/{y['days']}", F(y["gross_bp"]), F(y["net5_bp"]), F(y["delta_bp"]), F(y["idx_bp"])] for y in r["yearly"]]),
               "",
               "### 신호일만 보유 vs 매일 시가→종가(기준선, 같은 유효일), 5·7bp(H0 는 10bp 추가)\n",
               E.fmt_table(r["strat"], cols=("rule", "window", "cost_bp", "trades", "exposure", "gross_bp", "net_bp", "t_net", "win", "sharpe", "cagr", "mdd")),
               ""]
        ev = fs[cid]["A_evidence"]
        md += [f"A 재현 점검. 사양 A_evidence 첫머리: \"{ev.split('. ')[0]}. {ev.split('. ')[1]}.\" → 이 스크립트 A = 신호 {A['n_sig']}/{A['days']}일, "
               f"총평균 {F(A['gross_bp'])}bp(NW t {F(A['t_gross'])}), Δ {F(A['delta_bp'])}bp(HAC t {F(A['t'])}), 상위 10일 제거 {F(A['drop10_bp'])}bp. "
               "(A 변동성 조정 Δ 는 A 밖 수익률로 초기 60일 SD 를 채우므로 사양 값과 조금 다를 수 있다.)"
               + (f" 2차 D2-C1 B 값(Δ {F(r2v['delta_B_bp'])}bp, 신호 {r2v['n_sig_B']})과 이번 B(Δ {F(Bw['delta_bp'])}bp, 신호 {Bw['n_sig']})."
                  if cid == "R3-C2" and r2v else "") + "\n"]

    # 해석(수치는 위 표에서)
    c1, c2, c3 = res["R3-C1"], res["R3-C2"], res["R3-C3"]
    h1, h2, h3 = c1["win"]["H0"], c2["win"]["H0"], c3["win"]["H0"]
    late = "B 후반(2023-07-31~2025-06-05, 파생 08:45 개장)"
    md += ["## 해석과 주의\n",
           f"- **등급**: R3-C1 {c1['grade']}({c1['criteria']}), R3-C2 {c2['grade']}({c2['criteria']}), R3-C3 {c3['grade']}({c3['criteria']}). "
           f"H0 Δ는 C1 {F(h1['delta_bp'])}bp(t {F(h1['t'])}), C2 {F(h2['delta_bp'])}bp(t {F(h2['t'])}), C3 {F(h3['delta_bp'])}bp(t {F(h3['t'])})다. "
           f"A 대비 비율은 C1 {h1['delta_bp'] / c1['win']['A']['delta_bp'] * 100:.0f}%, C2 {h2['delta_bp'] / c2['win']['A']['delta_bp'] * 100:.0f}%, "
           f"C3 {h3['delta_bp'] / c3['win']['A']['delta_bp'] * 100:.0f}%다.",
           f"- **분해(H0)**: 신호일 ETF 수익 = 지수 몫 + 단일가 몫. C1 {F(c1['decomp']['H0']['etf_bp'])} = {F(c1['decomp']['H0']['idx_bp'])} + {F(c1['decomp']['H0']['single_bp'])}, "
           f"C2 {F(c2['decomp']['H0']['etf_bp'])} = {F(c2['decomp']['H0']['idx_bp'])} + {F(c2['decomp']['H0']['single_bp'])}, "
           f"C3 {F(c3['decomp']['H0']['etf_bp'])} = {F(c3['decomp']['H0']['idx_bp'])} + {F(c3['decomp']['H0']['single_bp'])}(bp). "
           "지수 몫은 지수를 시가에 살 수 없어 거래 불가 근사이고, 단일가 몫은 시가 단일가 체결이 전제다.",
           f"- **2023-07-31 이후**: 단일가 몫 Δ가 {late}에 C2 {F(c2['decomp'][late]['single_delta_bp'])}(t {F(c2['decomp'][late]['single_t'])}), "
           f"C3 {F(c3['decomp'][late]['single_delta_bp'])}(t {F(c3['decomp'][late]['single_t'])})다. C3 H0 통과 여부와 관계없이 지금 거래 가능성은 이 값이 좌우한다.",
           f"- **꼬리**: 상위 10일 제거 신호일 평균 H0 C1 {F(h1['drop10_bp'])}, C2 {F(h2['drop10_bp'])}, C3 {F(h3['drop10_bp'])}bp.",
           f"- **비용·틱**: H0 5원 1틱은 2003~2005년 3.8~5.7bp(h0_quality)다. 10bp 순평균 C1 {F(h1['net10_bp'])}, C2 {F(h2['net10_bp'])}, C3 {F(h3['net10_bp'])}bp.",
           f"- **E4 지수 시가 지연 의심일**: {len(e4_lag)}일({', '.join(str(x.date()) for x in e4_lag) or '없음'}). 같은 규칙으로 B 에는 {len(b_lag)}일"
           f"({', '.join(str(x.date()) for x in b_lag) or '없음'}). E4 는 보고만 한다.",
           f"- **한 가족**: H0 에서 C3 가 C2 밖에 신호를 낸 날 {overlap['C3_not_in_C2']}일(C3 ⊂ C2), C1 신호 {overlap['C1']}일 중 C2 와 같은 날 {overlap['C1&C2']}일이다. "
           "세 셀 통과를 독립 증거 세 개로 읽으면 안 된다. D2 가족 A 누적 시도는 580개다.",
           f"- **H0 크기**: H0 신호일 총평균이 A 보다 훨씬 크다(C2 {F(h2['gross_bp'])} vs A {F(c2['win']['A']['gross_bp'])}, C3 {F(h3['gross_bp'])} vs A {F(c3['win']['A']['gross_bp'])}bp). "
           f"두 몫이 모두 커졌다. C2 지수 몫 {F(c2['decomp']['H0']['idx_bp'])} vs A {F(c2['decomp']['A']['idx_bp'])}, 단일가 몫 {F(c2['decomp']['H0']['single_bp'])} vs A "
           f"{F(c2['decomp']['A']['single_bp'])}. H0 ETF 시가 잡음이 A 의 약 2배(h0_quality)라 단일가 몫이 커진 것은 예상된 방향이다. "
           "지수 몫이 커진 이유는 이 자료로 가를 수 없다(당시 지수 시가 산출 방식을 확인할 자료와 선물 일봉이 없다). 크기보다 부호·유의성의 재현으로 읽는 편이 안전하다.",
           f"- **C2 중 C3 아닌 날(zi ≥ 1 & mis > 0, 사양 밖 보고)**: H0 {comp['H0']['n_sig']}일 총평균 {F(comp['H0']['gross_bp'])}bp(Δ {F(comp['H0']['delta_bp'])}, t {F(comp['H0']['t'])}), "
           f"A {comp['A']['n_sig']}일 {F(comp['A']['gross_bp'])}, B {comp['B']['n_sig']}일 {F(comp['B']['gross_bp'])}, E4 {comp['E4']['n_sig']}일 {F(comp['E4']['gross_bp'])}bp. "
           f"H0 에서 이 날들의 지수 몫은 {F(comp_dc['H0']['idx_bp'])}, 단일가 몫은 {F(comp_dc['H0']['single_bp'])}bp 다.",
           f"- **ETF 0 갭일(시가 미체결 의심) 제외(사양 밖)**: Δ C1 {F(c1['extra'][next(iter(extra))]['delta_bp'])}, C2 {F(c2['extra'][next(iter(extra))]['delta_bp'])}, "
           f"C3 {F(c3['extra'][next(iter(extra))]['delta_bp'])}bp 로 판정 표본 값과 거의 같다.",
           "- **실행 가정**: 신호와 체결이 같은 09:00 경매다. 실제로는 08:50~09:00 예상체결가·예상지수로 판단해야 하며, 그 오차는 A 잡음 모의로만 가늠했다.",
           "- 인버스(114800)는 2009-09-16 상장이라 H0 에서 검증하지 않았다.",
           ""]
    OUTD.mkdir(parents=True, exist_ok=True)
    (OUTD / "report.md").write_text("\n".join(md), encoding="utf-8")

    verdict = {"topic": TOPIC, "run_at": t_run, "spec_file": "round3_D2/spec.json", "judge_window": list(H0), "excluded_years": excl_years,
               "n_cells": M_CELLS, "holm_alpha": ALPHA,
               "criteria_legend": "① Δ>0 & Holm(m=3) 단측 p ② 7bp 순평균>0 ③ 플라시보 p≤0.05 ④ 변동성 조정 Δ>0 ⑤ 이웃 4개 Δ>0 ⑥ 상위 10일 제거 평균>0",
               "grade_rule": "①~⑥ 모두 → 통과 / ①(Holm 전 p<0.05)·②·③ → 보류 / 그 밖 기각",
               "data": {"calendar": f"FDR KS200 rows {CAL0}~{(listed - pd.Timedelta(days=1)).date()} + 069500 rows {listed.date()}~",
                        "z_closed_form_vs_lstsq_maxdiff": zchk, "signal_availability_check": avail_chk,
                        "E4_last_signal_day": str(e4_last), "E4_lag_days": [str(x.date()) for x in e4_lag], "B_lag_days": [str(x.date()) for x in b_lag],
                        "H0_sensitivity_sets": {k: len(v) for k, v in sens.items()}},
               "ledger": {"rows": len(table), "written": wrote, "first_run": first_run},
               "H0_overlap": overlap,
               "off_spec_C2_not_C3": {"stats": comp, "decomposition": comp_dc},
               "cells": {}}
    for cid in cells:
        r = res[cid]
        verdict["cells"][cid] = {
            "grade": r["grade"], "criteria": r["criteria"], "crit": r["crit"], "raw_p_lt_0.05": r["raw1"], "holm": r["holm"],
            "H0": r["win"]["H0"], "A": r["win"]["A"], "B": r["win"]["B"], "E4": r["win"]["E4"], "E4_ex_lag": r["win"]["E4 지연 의심일 제외"],
            "neighbors_ok": r["neighbors_ok"], "neighbors": r["neighbors"], "subperiods": r["sub"], "sensitivity": r["sens"],
            "off_spec_extra_check": r["extra"],
            "decomposition": r["decomp"], "transfer_102110": r["transfer"], "exit_next_open": r["exit_oo"], "night_part": r["exit_on"],
            "yearly": r["yearly"],
            "strategy_vs_baseline": r["strat"][["rule", "window", "cost_bp", "trades", "gross_bp", "net_bp", "sharpe", "cagr", "mdd"]].to_dict("records")}
    C.save_verdict(OUTD / "verdict.json", verdict)
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print("\n".join(md[:12]))
    for cid in cells:
        H = res[cid]["win"]["H0"]
        print(cid, res[cid]["grade"], res[cid]["criteria"], {k: (round(v, 4) if isinstance(v, float) else v) for k, v in H.items()})


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true", help="시행 장부에 쓰지 않는다(점검용)")
    main(dry=ap.parse_args().dry)
