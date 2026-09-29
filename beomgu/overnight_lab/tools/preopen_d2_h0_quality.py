"""R3 · D2 확인 구간 H0(결정일 2003-01-02~2009-12-30) 자료 품질 점검 — 신호→수익률은 계산하지 않는다.

점검 항목(연도별, 비교용으로 A 2010~2019 도 같은 지표를 낸다. 두 구간 모두 수익률 계열은 만들지 않는다):
- 달력: 069500(KODEX 200, 2002-10-14 상장) 행과 코스피200 지수(FDR KS200) 행의 일치, 거래량 0 일
- 시가 품질: 069500 시가 = 전일 종가(0 갭) 비율, 그중 지수 갭이 30bp 를 넘는 날(시가 미체결 의심),
  지수 시가 결측·0·전일 종가와 같은 날, ETF 갭 − 지수 갭(dev)의 SD·중앙 절대편차·|dev| > 60·100bp 일수(12월 배당락일 제외),
  두 갭 상관, 대형 ETF 갭(|갭| > 1%)인데 지수 갭이 그 30% 미만인 날(지수 시가 지연 또는 ETF 가격 이상)
- 가격 수준: 수정주가 중앙값, 정수 반올림 오차(bp), 원가격 추정(지수 × 100) 기준 5원 호가 1틱(bp)
- 분배락 처리: 네이버 수정주가 / yfinance 원가격(2007-01~, 결측 구간 있음) 비율의 계단 변화(= 분배 수정)와 그날 dev,
  ETF 종가 괴리(ln ETF 종가/지수 종가)의 12월 밖 영구 이동(직전 5일 vs 이후 5일 중앙값 차 > 100bp)
- 12월 코스피200 구성종목 배당락일(12월 끝에서 둘째 거래일): 지수 갭은 배당만큼 내려가고 ETF(수정주가)는 안 내려간다
- 미국 매핑: D(T) = 미국 날짜 < T 인 마지막 ^GSPC 세션. 한국일마다 새로 끝난 미국 세션 수(0 = us 0), 최대 간격, 종가 결측
- 신호 가용성(신호만): 셀별 첫 유효일과 연도별 신호일 수(z 계산에 수익률 불필요)
판정 부적합 연도 규칙(사전 고정): 아래 중 하나라도 해당하면 그 연도를 H0 판정에서 뺀다.
  ① 지수 시가 결측·0 > 1%  ② 069500 거래량 0 일 > 1%  ③ 069500·지수 행 불일치 > 1%  ④ 미국 매핑 실패(us NaN) > 1%
  ⑤ 시가 품질 지표(|dev|>100bp 일수, 0 갭·지수 30bp 초과 일수, dev 중앙 절대편차)가 A 최악 연도의 2배 초과
산출: <OUT>/round3_D2/h0_quality.csv(연도별), h0_quality.json(목록·판정)
실행: python tools/preopen_d2_h0_quality.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common as C  # noqa: E402
import data  # noqa: E402

H0 = ("2003-01-01", "2009-12-31")
A = ("2010-01-01", "2019-12-31")
OUTD = data.OUT / "round3_D2"


def rd(p):
    return pd.read_csv(p, parse_dates=["date"], index_col="date", encoding="utf-8-sig").sort_index()


def rolling_z1(g, x, win=250, minr=200):
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


def main():
    end = A[1]
    etf = rd(data.PX_DIR / "069500.csv").loc[:end]
    fdr = rd(data.RAW / "k200_index_daily.csv").loc[:end]
    fch = rd(data.PX_DIR / "KPI200.csv").loc[:end]
    ix_all = pd.concat([fdr.loc[:"2014-07-07", ["open", "high", "low", "close"]],
                        fch.loc["2014-07-08":, ["open", "high", "low", "close"]].combine_first(fdr.loc["2014-07-08":, ["open", "high", "low", "close"]])])
    ix_all = ix_all.sort_index()
    ix_all = ix_all[ix_all["close"] > 0]
    # 달력: H0 번인을 위해 069500 상장 전에는 지수 행을 쓴다(상장 뒤에는 두 달력이 같음을 아래에서 확인)
    cal = ix_all.loc["2001-01-01":].index
    px = etf.reindex(cal)
    ix = ix_all.reindex(cal)
    yf = rd(data.RAW / "kodex200_daily.csv").loc[:end]
    spx = data.us_daily("spx")["close"].loc[:"2020-01-10"]

    listed = etf.index.min()
    post = cal[cal >= listed]
    mism = dict(index_not_in_etf=[str(d.date()) for d in post.difference(etf.index)],
                etf_not_in_index=[str(d.date()) for d in etf.index.difference(ix_all.index)])

    ge = np.log(px["open"] / px["close"].shift(1))
    gi = np.log(ix["open"] / ix["close"].shift(1))
    dev = ge - gi
    exdiv = []
    for y in range(2001, 2020):
        dd = cal[(cal.year == y) & (cal.month == 12)]
        if len(dd) >= 2:
            exdiv.append(dd[-2])
    isex = pd.Series(cal.isin(exdiv), index=cal)
    raw_est = ix["close"] * 100                                  # KODEX 200 원가격 ≈ 지수 × 100

    # 미국 매핑
    s = spx.dropna()
    pos = np.searchsorted(s.index.values, cal.values, side="left") - 1
    Dt = pd.Series(pd.DatetimeIndex(np.where(pos >= 0, s.index.values[np.clip(pos, 0, None)], np.datetime64("NaT"))), index=cal)
    nsess = pd.Series(pos - np.r_[np.nan, pos[:-1]], index=cal)
    lc = pd.Series(np.where(pos >= 0, np.log(s.to_numpy(float))[np.clip(pos, 0, None)], np.nan), index=cal)
    us = lc - lc.shift(1)

    rows = []
    for y in range(2003, 2020):
        q = slice(f"{y}-01-01", f"{y}-12-31")
        p, g_e, g_i, d, e = px.loc[q], ge.loc[q], gi.loc[q], dev.loc[q], isex.loc[q]
        n = len(cal[(cal.year == y)])
        rows.append(dict(
            year=y, period="H0" if y <= 2009 else "A", days=n, etf_rows=int(p["close"].notna().sum()), vol0=int((p["volume"] == 0).sum()),
            idx_open_na_or_0=int((ix.loc[q, "open"].isna() | (ix.loc[q, "open"] <= 0)).sum()), idx_open_eq_prevclose=int((g_i == 0).sum()),
            etf_gap0_ratio=round(float((g_e == 0).mean()), 3), etf_gap0_idx_move=int(((g_e == 0) & (g_i.abs() > 0.003)).sum()),
            sd_etf_gap_bp=round(g_e.std() * 1e4, 1), sd_idx_gap_bp=round(g_i.std() * 1e4, 1), sd_dev_bp=round(d.std() * 1e4, 1),
            mad_dev_bp=round(float((d - d.median()).abs().median() * 1e4), 1), corr_gaps=round(float(g_e.corr(g_i)), 3),
            dev_gt60=int(((d.abs() > 0.006) & ~e).sum()), dev_gt100=int(((d.abs() > 0.01) & ~e).sum()),
            idx_lag_cand=int(((g_e.abs() > 0.01) & (g_i.abs() < 0.3 * g_e.abs())).sum()),
            adj_px_median=float(p["close"].median()), round_err_bp=round(float(0.5 / p["close"].median() * 1e4), 2),
            tick5_bp=round(float(5 / raw_est.loc[q].median() * 1e4), 2),
            us_nan=int(us.loc[q].isna().sum()), us_zero_days=int((nsess.loc[q] == 0).sum()), us_multi_days=int((nsess.loc[q] >= 2).sum()),
            us_max_gap_days=int((pd.Series(cal[cal.year == y]) - pd.Series(Dt.loc[q].values)).dt.days.max())))
    T = pd.DataFrame(rows).set_index("year")

    # 부적합 연도 규칙 ⑤: A 최악 연도의 2배 초과
    Aw = T[T["period"] == "A"]
    lim = {"dev_gt100": int(2 * Aw["dev_gt100"].max()), "etf_gap0_idx_move": int(2 * Aw["etf_gap0_idx_move"].max()),
           "mad_dev_bp": float(2 * Aw["mad_dev_bp"].max())}
    excl = {}
    for y, r in T[T["period"] == "H0"].iterrows():
        why = []
        if r["idx_open_na_or_0"] > 0.01 * r["days"]:
            why.append("①지수 시가 결측")
        if r["vol0"] > 0.01 * r["days"]:
            why.append("②거래량 0")
        if (r["days"] - r["etf_rows"]) > 0.01 * r["days"]:
            why.append("③행 불일치")
        if r["us_nan"] > 0.01 * r["days"]:
            why.append("④미국 매핑")
        why += [f"⑤{k}" for k, v in lim.items() if r[k] > v]
        excl[int(y)] = why
    T["exclude"] = [", ".join(excl.get(int(y), [])) if p == "H0" else "" for y, p in zip(T.index, T["period"])]

    # 분배락 처리
    j = etf[["close"]].join(yf[["Close"]], how="inner").dropna()
    ratio = np.log(j["close"] / j["Close"])
    step = ratio.diff()
    steps = [dict(date=str(d.date()), step_bp=round(v * 1e4, 1), prev_row=str(j.index[j.index.get_loc(d) - 1].date()),
                  dev_bp=round(float(dev.get(d, np.nan) * 1e4), 1)) for d, v in step[step.abs() > 0.002].items() if d <= pd.Timestamp(A[1])]
    yf_gap = yf["Close"].loc["2007-01-01":"2009-12-31"]
    prem = np.log(px["close"] / ix["close"])
    shift = (prem.shift(-1).rolling(5).median().shift(-4) - prem.shift(1).rolling(5).median())
    perm = shift[(shift.abs() > 0.01) & ~((shift.index.month == 12) | ((shift.index.month == 1) & (shift.index.day <= 10)))]
    perm = perm.loc[H0[0]:A[1]]

    # 12월 배당락일
    exd = [dict(date=str(d.date()), idx_gap_bp=round(float(gi[d] * 1e4), 1), etf_gap_bp=round(float(ge[d] * 1e4), 1))
           for d in exdiv if pd.Timestamp("2002-10-15") <= d <= pd.Timestamp(A[1])]

    # 이상일 목록(H0)
    an = dev.loc[H0[0]:H0[1]]
    an = an[(an.abs() > 0.01) & ~isex.loc[H0[0]:H0[1]]]
    anomalies = [dict(date=str(d.date()), dev_bp=round(v * 1e4, 1), etf_gap_bp=round(float(ge[d] * 1e4), 1), idx_gap_bp=round(float(gi[d] * 1e4), 1))
                 for d, v in an.items()]
    lag = ((ge.abs() > 0.01) & (gi.abs() < 0.3 * ge.abs())).loc[H0[0]:H0[1]]
    stale_idx = [str(d.date()) for d in lag[lag].index]
    gap0 = ((ge == 0) & (gi.abs() > 0.003)).loc[H0[0]:H0[1]]

    # 신호 가용성(신호만): zi(지수 갭~SPX 누적), ze(069500 갭~SPX 누적), mis(시가 괴리 − 직전 5개 종가 괴리 중앙값)
    zi = rolling_z1(gi, us)
    ze = rolling_z1(ge, us)
    mis = np.log(px["open"] / ix["open"]) - prem.shift(1).rolling(5).median()
    sig = {"R3-C1 ze≥1": (ze >= 1).where(ze.notna()), "R3-C2 zi≥1": (zi >= 1).where(zi.notna()),
           "R3-C3 zi≥1&mis≤0": ((zi >= 1) & (mis <= 0)).where(zi.notna() & mis.notna())}
    avail = {}
    for k, v in sig.items():
        vv = v.loc[H0[0]:H0[1]]
        avail[k] = dict(first_valid=str(vv.dropna().index.min().date()), valid_days=int(vv.notna().sum()), h0_days=len(vv),
                        signals_by_year={int(y): int(g.sum()) for y, g in vv.dropna().groupby(vv.dropna().index.year)})
    ov = pd.DataFrame(sig).loc[H0[0]:H0[1]].dropna()

    OUTD.mkdir(parents=True, exist_ok=True)
    T.to_csv(OUTD / "h0_quality.csv", encoding="utf-8-sig")
    info = dict(calendar_note="H0 달력 = FDR KS200 지수 행(2001~). 069500 상장(2002-10-14) 뒤 두 달력 불일치 없음" if not any(mism.values()) else mism,
                calendar_mismatch=mism, etf_listed=str(listed.date()), rule5_limits=lim, excluded_years={k: v for k, v in excl.items() if v},
                distribution_steps_yf=steps, yf_close_nan_2007_2009=int(yf_gap.isna().sum()), yf_rows_2007_2009=int(len(yf_gap)),
                premium_perm_shift_gt100_outside_dec=[dict(date=str(d.date()), bp=round(v * 1e4, 1)) for d, v in perm.items()],
                dec_exdiv=exd, h0_dev_anomalies=anomalies, h0_idx_lag_candidates=stale_idx,
                h0_etf_gap0_idx_move=[str(d.date()) for d in gap0[gap0].index],
                us_sessions_per_kr_day_H0={int(k): int(v) for k, v in nsess.loc[H0[0]:H0[1]].value_counts().sort_index().items()},
                spx_close_nan_2002_2010=int(spx.loc["2002":"2010"].isna().sum()),
                signal_availability=avail, h0_signal_overlap=dict(C1_and_C2=int(((ov.iloc[:, 0] == 1) & (ov.iloc[:, 1] == 1)).sum()),
                                                                 C2_and_C3=int(((ov.iloc[:, 1] == 1) & (ov.iloc[:, 2] == 1)).sum())),
                inverse_114800_first_row=str(rd(data.PX_DIR / "114800.csv").index.min().date()))
    C.save_verdict(OUTD / "h0_quality.json", info)
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    pd.set_option("display.width", 250)
    print(T.drop(columns=["adj_px_median"]).to_string())
    print(json.dumps({k: info[k] for k in ("calendar_note", "rule5_limits", "excluded_years", "us_sessions_per_kr_day_H0", "signal_availability",
                                           "h0_signal_overlap", "inverse_114800_first_row", "yf_close_nan_2007_2009", "yf_rows_2007_2009")},
                     ensure_ascii=False, indent=1))
    print("분배 계단:", steps[:6])
    print("12월 밖 영구 괴리 이동 >100bp:", info["premium_perm_shift_gt100_outside_dec"])
    print("H0 dev 이상일:", len(anomalies), " 지수 지연 후보:", stale_idx)


if __name__ == "__main__":
    main()
