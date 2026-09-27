"""1단계: 데이터 적재와 KRX 거래일 달력 기준 패널 구성.

원칙
- 달력 = K200 지수 거래일(FDR + 네이버 fchart). 모든 자료를 이 달력에 reindex 한 뒤에 시프트한다.
  (행을 먼저 지우고 shift 하면 결측 다음 날의 'T−1'이 실제로는 T−2 이전 값이 된다.)
- 날짜 중복은 즉시 오류로 멈춘다(중복 행이 있으면 위치 기준 shift 가 당일 값을 끌어온다).
- 원자료는 수정하지 않는다. 보완한 값에는 출처 열을 붙인다.
"""
import re

import numpy as np
import pandas as pd

OPT_INVESTOR_FILE = "krx_opt_investor_daily.csv"   # 표준 스키마: date,investor,cp,buy_qty,sell_qty,buy_amt,sell_amt
OPT_VALUE_FILE = "krx_opt_value_daily.csv"         # 선택: date,call_value,put_value
OPT_REQUIRED = ["date", "investor", "cp", "buy_qty", "sell_qty", "buy_amt", "sell_amt"]
OPT_NUMERIC = ["buy_qty", "sell_qty", "buy_amt", "sell_amt"]

# investor 라벨 → 표준 이름. foreign 은 외국인 + 기타외국인 합산.
INVESTOR_ALIASES = {
    "foreign": "foreign", "외국인": "foreign", "other_foreign": "foreign", "기타외국인": "foreign",
    "individual": "individual", "개인": "individual",
    "institution": "institution", "기관": "institution", "기관계": "institution", "기관합계": "institution",
    "total": "total", "전체": "total", "합계": "total", "계": "total",
    "fin_inv": "fininv", "금융투자": "fininv", "insurance": "insurance", "보험": "insurance",
    "trust": "trust", "투신": "trust", "private_fund": "privatefund", "사모": "privatefund",
    "bank": "bank", "은행": "bank", "other_fin": "otherfin", "기타금융": "otherfin",
    "pension": "pension", "연기금": "pension", "연기금등": "pension",
    "gov": "gov", "국가": "gov", "국가·지자체": "gov", "other_corp": "othercorp", "othercorp": "othercorp", "기타법인": "othercorp",
    "fininv": "fininv", "privatefund": "privatefund", "otherfin": "otherfin",
}
# 합산 대상이 이미 합쳐진 라벨과 함께 오면 이중 계산 → 오류
AMBIGUOUS_FOREIGN = {"외국인계", "외국인합계", "foreign_total"}
OPT_COL_RE = re.compile(r"^opt_(?P<inv>[a-z0-9]+)_(?P<cp>[CP])_(?P<side>buy|sell)_(?P<unit>amt|qty)$")


def _read(path):
    df = pd.read_csv(path, parse_dates=["date"]).set_index("date").sort_index()
    dup = df.index[df.index.duplicated()]
    if len(dup):
        raise ValueError(f"{path.name}: 날짜 중복 {len(dup)}건 (예: {[str(d.date()) for d in dup[:5]]}) — 중복 제거 후 다시 실행")
    return df


def load_index(raw):
    """K200 지수 일봉. 네이버 fchart 시작일 이전은 FDR, 이후는 네이버."""
    fdr = _read(raw / "k200_index_daily.csv")[["open", "high", "low", "close"]]
    nav = _read(raw / "k200_naver_fchart_daily.csv")[["open", "high", "low", "close"]]
    cut = nav.index.min()
    df = pd.concat([fdr[fdr.index < cut], nav]).sort_index()
    df = df[(df["open"] > 0) & (df["close"] > 0)].copy()
    df["src"] = np.where(df.index < cut, "fdr", "naver")
    flat = (df["open"] == df["high"]) & (df["high"] == df["low"]) & (df["low"] == df["close"])
    audit = {"rows": len(df), "start": str(df.index.min().date()), "end": str(df.index.max().date()),
             "source_cut": str(cut.date()), "flat_ohlc_days": int(flat.sum()),
             "flat_ohlc_days_since_2010": int(flat[flat.index >= "2010-01-01"].sum())}
    return df, audit


def load_kodex(raw, calendar, window_start):
    """KODEX 200 일봉. yfinance 원가격이 주다.

    달력 대비 빠진 날은 네이버 fchart(069500)로 채운다. 네이버는 분배금 수정주가라 가격 수준이 다르므로,
    직전 겹침일의 원가격/수정가 비율을 곱해 원가격 수준으로 되돌린다(시가→종가 수익률은 비율과 무관).
    """
    yf = _read(raw / "kodex200_daily.csv").rename(columns=str.lower)[["open", "high", "low", "close", "volume"]].copy()
    yf["src"] = "yfinance"
    audit = {"yfinance_rows": len(yf), "window_start": window_start}
    nav_p = raw / "kodex200_naver_fchart_daily.csv"
    df = yf
    if nav_p.exists():
        nv = _read(nav_p)[["open", "high", "low", "close", "volume"]].copy()
        ov = yf.index.intersection(nv.index)
        ratio = yf.loc[ov, "close"] / nv.loc[ov, "close"]
        r_yf = yf.loc[ov, "close"] / yf.loc[ov, "open"] - 1
        r_nv = nv.loc[ov, "close"] / nv.loc[ov, "open"] - 1
        audit["overlap_days"] = len(ov)
        audit["adj_ratio_first_last"] = [round(float(ratio.iloc[0]), 4), round(float(ratio.iloc[-1]), 4)]
        audit["overlap_oc_return_diff_bp_sd"] = round(float(1e4 * (r_yf - r_nv).std()), 3)
        audit["overlap_oc_return_diff_bp_maxabs"] = round(float(1e4 * (r_yf - r_nv).abs().max()), 2)
        miss = calendar.difference(yf.index)
        fill = nv.loc[nv.index.intersection(miss)].copy()
        if len(fill):
            k = ratio.reindex(ratio.index.union(fill.index)).ffill().bfill().loc[fill.index]
            audit["fill_before_first_overlap"] = int((fill.index < ov.min()).sum())   # bfill(미래 비율) 사용 건수
            for c in ("open", "high", "low", "close"):
                fill[c] = (fill[c] * k).round(0)
            fill["src"] = "naver_scaled"
        df = pd.concat([yf, fill]).sort_index()
        audit["filled_from_naver"] = len(fill)
    out = df.reindex(calendar)
    win = out.index >= pd.Timestamp(window_start)
    audit["missing_in_window"] = int(out.loc[win, "close"].isna().sum())
    audit["missing_in_window_dates"] = [str(d.date()) for d in out.index[win & out["close"].isna().to_numpy()]][:40]
    audit["missing_before_window"] = int(out.loc[(~win) & (out.index >= yf.index.min()), "close"].isna().sum())
    audit["zero_volume_days_in_window"] = int((out.loc[win, "volume"] == 0).sum())
    return out, audit


def load_futflow(raw):
    """외인 선물 매수·매도 (네이버 JSON, 계약·금액). 통제변수 전용."""
    f = _read(raw / "naver_fut_investor_daily.csv")
    need = ["buyq_foreign", "sellq_foreign", "buyv_foreign", "sellv_foreign"]
    miss = [c for c in need if c not in f.columns]
    if miss:
        raise ValueError(f"선물 투자자 파일에 열 없음: {miss}")
    out = pd.DataFrame({"fut_f_buy_q": f["buyq_foreign"], "fut_f_sell_q": f["sellq_foreign"],
                        "fut_f_buy_v": f["buyv_foreign"], "fut_f_sell_v": f["sellv_foreign"]})
    # 네이버 초기 자료(2010-09~12 일부)는 매도 수량이 0이고 매수 칸에 순매수가 들어 있다 → 결측 처리
    bad = (out["fut_f_sell_q"] <= 0) | (out["fut_f_buy_q"] <= 0)
    out.loc[bad, :] = np.nan
    audit = {"rows": len(out), "start": str(out.index.min().date()), "end": str(out.index.max().date()),
             "invalid_rows_set_nan": int(bad.sum()),
             "invalid_range": [str(out.index[bad].min().date()), str(out.index[bad].max().date())] if bad.any() else None}
    return out, audit


def normalize_investor(s):
    raw = s.astype(str).str.strip()
    key = raw.str.lower().where(raw.str.match(r"^[A-Za-z_]+$"), raw)
    return key.map(INVESTOR_ALIASES), raw


def load_options(raw):
    """KRX 투자자별 옵션(표준 스키마). 없으면 (None, 상태)."""
    p = raw / OPT_INVESTOR_FILE
    if not p.exists():
        return None, {"status": "missing", "expected_file": str(p), "schema": OPT_REQUIRED}
    d = pd.read_csv(p, parse_dates=["date"])
    miss = [c for c in OPT_REQUIRED if c not in d.columns]
    if miss:
        raise ValueError(f"{p.name}: 필수 열 없음 {miss}")
    d["cp"] = d["cp"].astype(str).str.strip().str.upper().str[0]
    bad = sorted(set(d["cp"]) - {"C", "P"})
    if bad:
        raise ValueError(f"{p.name}: cp 값은 C/P(또는 call/put) 여야 함, 발견 {bad}")
    labels = set(d["investor"].astype(str).str.strip())
    if labels & AMBIGUOUS_FOREIGN and labels & {"외국인", "기타외국인", "foreign", "other_foreign"}:
        raise ValueError(f"{p.name}: 외국인 합계 라벨과 구성 라벨이 함께 있음 {sorted(labels)} — 하나만 남길 것")
    d["investor_std"], d["investor_raw"] = normalize_investor(d["investor"])
    dup = d.duplicated(["date", "investor_raw", "cp"])
    if dup.any():
        raise ValueError(f"{p.name}: (date, investor, cp) 중복 {int(dup.sum())}건 — 기간을 나눠 받은 파일을 이을 때 경계일 확인")
    unknown = sorted(d.loc[d["investor_std"].isna(), "investor_raw"].unique().tolist())
    d = d[d["investor_std"].notna()]
    num_nan = {c: int(d[c].isna().sum()) for c in OPT_NUMERIC}
    g = d.groupby(["date", "investor_std", "cp"])[OPT_NUMERIC].sum(min_count=1)   # 전부 NaN 이면 NaN 유지
    wide = g.unstack(["investor_std", "cp"])
    wide.columns = [f"opt_{inv}_{cp}_{v.split('_')[0]}_{v.split('_')[1]}" for v, inv, cp in wide.columns]
    wide = wide.sort_index()
    audit = {"status": "loaded", "rows": len(wide), "start": str(wide.index.min().date()), "end": str(wide.index.max().date()),
             "investors_raw": sorted(labels), "investors_std": sorted(d["investor_std"].unique().tolist()),
             "unknown_investor_labels_dropped": unknown, "numeric_nan_cells": num_nan}
    vp = raw / OPT_VALUE_FILE
    if vp.exists():
        v = _read(vp)[["call_value", "put_value"]]
        wide = wide.join(v, how="outer")
        audit["value_file"] = True
    return wide, audit


def _dates_outside(idx, cal):
    out = idx.difference(cal)
    return {"count": int(len(out)), "head": [str(x.date()) for x in out[:20]]}


def build_panel(raw, window_start):
    """달력 기준 패널 + 감사 정보."""
    idx, a_idx = load_index(raw)
    cal = idx.index
    kdx, a_kdx = load_kodex(raw, cal, window_start)
    fut, a_fut = load_futflow(raw)
    opt, a_opt = load_options(raw)

    panel = pd.DataFrame(index=cal)
    panel.index.name = "date"
    panel["idx_open"], panel["idx_close"], panel["idx_src"] = idx["open"], idx["close"], idx["src"]
    panel["kdx_open"], panel["kdx_close"], panel["kdx_src"] = kdx["open"], kdx["close"], kdx["src"]
    a_fut["dates_not_in_calendar"] = _dates_outside(fut.index, cal)
    panel = panel.join(fut, how="left")
    if opt is not None:
        a_opt["dates_not_in_calendar"] = _dates_outside(opt.index, cal)
        panel = panel.join(opt, how="left")
        rng = (panel.index >= opt.index.min()) & (panel.index <= opt.index.max())
        need = [c for c in ("opt_foreign_C_buy_amt", "opt_foreign_C_sell_amt", "opt_foreign_P_buy_amt", "opt_foreign_P_sell_amt") if c in panel]
        a_opt["calendar_days_in_range"] = int(rng.sum())
        a_opt["primary_input_missing_days_in_range"] = int(panel.loc[rng, need].isna().any(axis=1).sum()) if need else None
    if not (panel.index.is_unique and panel.index.is_monotonic_increasing):
        raise AssertionError("패널 날짜가 유일·오름차순이 아님")

    # 목표변수 (%, 단순수익률, 원가격)
    panel["y_idx"] = (panel["idx_close"] / panel["idx_open"] - 1) * 100
    panel["y_kdx"] = (panel["kdx_close"] / panel["kdx_open"] - 1) * 100
    # 통제변수 (T일 행에 T−1 정보만)
    panel["r_prev_oc"] = panel["y_idx"].shift(1)                                               # T−1 시가→종가
    panel["r_prev_cc"] = (panel["idx_close"].pct_change(fill_method=None) * 100).shift(1)     # T−1 종가/T−2 종가 − 1
    # 신호 기준일 = 달력상 직전 거래일
    panel["sig_date"] = pd.Series(cal, index=cal).shift(1)
    audit = {"calendar": a_idx, "kodex": a_kdx, "futures_investor": a_fut, "options": a_opt}
    return panel, audit
