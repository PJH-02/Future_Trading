# -*- coding: utf-8 -*-
"""
r13_slot_book.py — 시간대별 알파 장부(코스피·코스닥) 조립과 성과 판정.

독립 실행: PYTHONUTF8=1 python r13_slot_book.py
입력: ROUND13/series/<handle>.csv (date,pos,gross,net; 유효 시작 전 NaN, 오버나잇은 date=청산일)
출력: ROUND13/book/ book_daily.csv, book_metrics.csv, leg_contrib.csv, sensitivity.csv, yearly.csv,
      worst_days.csv, corr.csv, capital_timeline.md, equity_W_all.png, equity_long.png

[장부 정의 — 사전 고정, 바꾸지 말 것]
• 코스피 장부(자본 1): A 슬롯 jehyeong_ON(15:30 종가 매수 → 익일 09:00 매도, 손익은 청산일),
  B 슬롯 beomgu_C3VT30(09:00→15:30), C 슬롯 jehyeong_KP(09:31~15:20).
• 코스닥 장부(자본 1): A 슬롯 blog_ON_Q(15:30→익일 09:00), B 슬롯 beomgu_C3Q(09:00→15:30), C 슬롯 jehyeong_KQ.
• 자본 재사용: 같은 날 A 다리(전날 밤 포지션)는 09:00 시가에 청산되므로 B·C 와 겹치지 않는다.
  B 와 C 는 같은 시간대라 한 날에 하나만: B 다리가 활성(pos ≠ 0)인 날은 C 다리 수익을 0 으로 둔다(우선순위 B).
  장부 일수익 = r_A + r_B + r_C(각각 자본 1 기준), 장부는 일 복리.
• 합산 장부 = 코스피 장부 50% + 코스닥 장부 50%, (a) 매일 동일가중 재조정, (b) 계좌 분리·이체 없음 둘 다.
• 변형(민감도, 각각 하나씩만 바꿈): V1 코스피 B 슬롯에 beomgu_M2inv 추가(z≤−1 날 인버스; C3 와 날짜 배타, 활성일엔 C 다리 0)
  / V2 코스닥 B 슬롯에 pairQ 추가(활성일엔 C 다리 0, beomgu_C3Q 와 같은 날이면 C3Q 우선)
  / V3 C 슬롯을 blog_ID_K·blog_ID_Q 로 교체 / V4 코스닥 A 를 blog_ON_Q1x 로 / V5 jehyeong_ON 을 jehyeong_ON_orig 로
  / V6 모든 다리 비용 +2bp / V7 2026 제외 / V8 C 우선(B 를 0) / V9 A 슬롯 없음 / V10 B 슬롯 없음 / V11 C 슬롯 없음.
• 창: W_all = 2022-12-13~2026-09-17(모든 주 다리 유효) / W_des = 2022-12-13~2024-12-30 / W_val = 2025-01-02~2026-09-17
  / 연도별(2023, 2024, 2025, 2026) / 장기 참고 W_long = 2016-10-04~2026-09-17 에서 A2+B2 만(코스닥 A·B 두 슬롯) 과 코스피 B 만.
• 목표 판정: W_all 합산 장부(a) CAGR ∈ [20%, 50%] ∧ MDD ∈ [−20%, −10%] 이면 '충족', MDD 가 −10% 보다 얕으면
  '목표 초과 달성(위험 여유)', CAGR 미달이면 '미달'. W_des 와 W_val 각각도 같은 기준으로 적는다.

[지표 정의]
누적 Π(1+r), CAGR = 누적^(252/일수)−1, 연변동성 std(ddof=1)·√252(0 포함), 샤프 = mean·252/vol,
MDD = min(누적/누적최고−1), Calmar = CAGR/|MDD|, 활성일 승률 = r≠0 일 중 r>0 비율.

[달력] KRX 거래일 = hyfe/data/raw/etf/069500.csv 의 date. 달력 밖 시리즈 행은 버림(개수 기록).
[V6 비용] 활성일(pos≠0)마다 net − 2bp (자본 1 기준, 하루 두 건인 날도 2bp 한 번).
주의: 단일 백테스트 추정치. 비용은 각 시리즈 메타 가정값. ETF 괴리·호가 슬리피지·09:00 단일가 체결 불확실성 미반영.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams["font.family"] = "Malgun Gothic"
matplotlib.rcParams["axes.unicode_minus"] = False

ROUND13 = Path(os.environ.get(
    "R13_DIR",
    "<OUT>/round13_slot_book"))
CAL_CSV = Path(os.environ.get("KRX_CAL", "<raw>/etf/069500.csv"))
OUT = ROUND13 / "book"
OUT.mkdir(parents=True, exist_ok=True)

WINDOWS = {
    "W_all": ("2022-12-13", "2026-09-17"),
    "W_des": ("2022-12-13", "2024-12-30"),
    "W_val": ("2025-01-02", "2026-09-17"),
    "Y2023": ("2023-01-02", "2023-12-28"),
    "Y2024": ("2024-01-02", "2024-12-30"),
    "Y2025": ("2025-01-02", "2025-12-30"),
    "Y2026": ("2026-01-02", "2026-09-17"),
}
W_LONG = ("2016-10-04", "2026-09-17")
# 슬롯 정의: 각 슬롯은 우선순위 순 핸들 목록(첫 활성 다리만 사용)
BASE = {
    "KOSPI": {"A": ["jehyeong_ON"], "B": ["beomgu_C3VT30"], "C": ["jehyeong_KP"]},
    "KOSDAQ": {"A": ["blog_ON_Q"], "B": ["beomgu_C3Q"], "C": ["jehyeong_KQ"]},
}
HANDLES = ["jehyeong_ON", "jehyeong_ON_orig", "jehyeong_KP", "jehyeong_KQ", "beomgu_C3VT30", "beomgu_C3Q",
           "beomgu_M2inv", "pairQ", "blog_ON_Q", "blog_ON_Q1x", "blog_ID_K", "blog_ID_Q"]
NOTES: list[str] = []


# ---------------------------------------------------------------- 데이터
def load_panel():
    cal = pd.to_datetime(pd.read_csv(CAL_CSV, usecols=["date"])["date"])
    net = pd.DataFrame(index=pd.DatetimeIndex(cal, name="date"))
    pos = net.copy()
    for h in HANDLES:
        s = pd.read_csv(ROUND13 / "series" / f"{h}.csv", parse_dates=["date"]).set_index("date")
        off = s.index.difference(net.index)
        if len(off):
            NOTES.append(f"{h}: 달력 밖 {len(off)}행 제거({off.min().date()}~{off.max().date()})")
        s = s.reindex(net.index)  # 시리즈 밖 날짜 = NaN(무효)
        net[h] = s["net"]
        pos[h] = s["pos"]
    return net, pos


# ---------------------------------------------------------------- 장부 조립
def slot_returns(net, pos, handles, active_mask=None):
    """슬롯 사용 수익: 우선순위 순 첫 활성 다리의 net. active_mask 가 주어지면 그 날은 0."""
    r = pd.Series(0.0, index=net.index)
    act = pd.Series(False, index=net.index)
    n_pri = 0
    for h in handles:
        a = (pos[h].fillna(0) != 0) & net[h].notna()
        take = a & ~act
        n_pri += int((a & act).sum())
        r[take] = net[h][take]
        act |= a
    if n_pri:
        NOTES.append(f"슬롯 {handles}: 같은 날 활성 {n_pri}일 → 앞 다리 우선")
    return r, act


def build_book(net, pos, slots, c_priority=False, extra_cost=0.0):
    """slots = {'A': [...], 'B': [...], 'C': [...]} → DataFrame(rA,rB,rC,actA,actB,actC,r). 사용 핸들 중 NaN 인 날은 NaN."""
    if extra_cost:
        net = net - extra_cost * (pos.fillna(0) != 0)
    out = pd.DataFrame(index=net.index)
    valid = pd.Series(True, index=net.index)
    for s, hs in slots.items():
        r, act = slot_returns(net, pos, hs) if hs else (pd.Series(0.0, index=net.index), pd.Series(False, index=net.index))
        out[f"r{s}"], out[f"act{s}"] = r, act
        for h in hs:
            valid &= net[h].notna()
    # B·C 배타
    if c_priority:
        clash = out["actC"] & out["actB"]
        out.loc[clash, "rB"] = 0.0
        out.loc[clash, "actB"] = False
    else:
        clash = out["actB"] & out["actC"]
        out.loc[clash, "rC"] = 0.0
        out.loc[clash, "actC"] = False
    out["n_clash"] = clash
    out["r"] = out["rA"] + out["rB"] + out["rC"]
    out.loc[~valid, ["rA", "rB", "rC", "r"]] = np.nan
    return out


def combine(rk, rq):
    """(a) 매일 동일가중 / (b) 계좌 분리·이체 없음(창 시작 자본 반씩, 각자 복리)."""
    a = 0.5 * rk + 0.5 * rq
    ek, eq = (1 + rk).cumprod().shift(fill_value=1.0), (1 + rq).cumprod().shift(fill_value=1.0)
    b = (ek * rk + eq * rq) / (ek + eq)
    return a, b


# ---------------------------------------------------------------- 지표
def metrics(r):
    r = r.dropna()
    n = len(r)
    if n == 0:
        return dict(n_days=0)
    eq = (1 + r).cumprod()
    cum = float(eq.iloc[-1])
    cagr = cum ** (252 / n) - 1
    vol = float(r.std(ddof=1)) * np.sqrt(252) if n > 1 else np.nan
    mdd = float((eq / eq.cummax() - 1).min())
    act = r != 0
    return dict(n_days=n, n_active=int(act.sum()), cum=cum - 1, cagr=cagr, vol=vol,
                sharpe=float(r.mean()) * 252 / vol if vol else np.nan, mdd=mdd,
                calmar=cagr / abs(mdd) if mdd else np.nan,
                win=float((r[act] > 0).mean()) if act.any() else np.nan)


def win(r, w):
    return r.loc[w[0]:w[1]]


def verdict(m):
    if m["cagr"] < 0.20:
        return "미달(CAGR)"
    if m["cagr"] > 0.50:
        return "CAGR 범위 초과"
    if m["mdd"] > -0.10:
        return "목표 초과 달성(위험 여유)"
    if m["mdd"] >= -0.20:
        return "충족"
    return "미달(MDD)"


# ---------------------------------------------------------------- 메인
def run_books(net, pos, slots_by_mkt, **kw):
    bk = build_book(net, pos, slots_by_mkt["KOSPI"], **kw)
    bq = build_book(net, pos, slots_by_mkt["KOSDAQ"], **kw)
    return bk, bq


def main():
    net, pos = load_panel()
    bk, bq = run_books(net, pos, BASE)
    def books_in(w):
        rk, rq = win(bk["r"], w), win(bq["r"], w)
        a, b = combine(rk, rq)  # (b) 는 창 시작에서 자본 반씩
        return {"KOSPI": rk, "KOSDAQ": rq, "합산(a)": a, "합산(b)": b}
    ra, rb = combine(win(bk["r"], WINDOWS["W_all"]), win(bq["r"], WINDOWS["W_all"]))

    # 1) book_daily
    daily = pd.DataFrame({
        "KOSPI_A_jehyeong_ON": bk["rA"], "KOSPI_B_beomgu_C3VT30": bk["rB"], "KOSPI_C_jehyeong_KP": bk["rC"],
        "KOSPI_clash": bk["n_clash"].astype(int), "KOSPI_book": bk["r"],
        "KOSDAQ_A_blog_ON_Q": bq["rA"], "KOSDAQ_B_beomgu_C3Q": bq["rB"], "KOSDAQ_C_jehyeong_KQ": bq["rC"],
        "KOSDAQ_clash": bq["n_clash"].astype(int), "KOSDAQ_book": bq["r"],
        "comb_a": ra, "comb_b": rb,
    }).loc[WINDOWS["W_all"][0]:WINDOWS["W_all"][1]]
    assert daily.drop(columns=["KOSPI_clash", "KOSDAQ_clash"]).notna().all().all(), "W_all 안에 NaN"
    daily.to_csv(OUT / "book_daily.csv", float_format="%.8f")
    NOTES.append(f"W_all B·C 겹침(C 0 처리): 코스피 {int(daily.KOSPI_clash.sum())}일, 코스닥 {int(daily.KOSDAQ_clash.sum())}일")

    # 2) book_metrics
    rows = []
    for w, (s, e) in WINDOWS.items():
        for b, r in books_in((s, e)).items():
            rows.append(dict(window=w, start=s, end=e, book=b, **metrics(r)))
    # 장기 참고
    lk = build_book(net, pos, {"A": [], "B": ["beomgu_C3VT30"], "C": []})["r"]
    lq = build_book(net, pos, {"A": ["blog_ON_Q"], "B": ["beomgu_C3Q"], "C": []})["r"]
    la, lb = combine(win(lk, W_LONG), win(lq, W_LONG))
    for b, r in {"KOSPI_B만": lk, "KOSDAQ_A2+B2": lq, "합산(a)": la, "합산(b)": lb}.items():
        rows.append(dict(window="W_long", start=W_LONG[0], end=W_LONG[1], book=b, **metrics(win(r, W_LONG))))
    bm = pd.DataFrame(rows)
    bm.to_csv(OUT / "book_metrics.csv", index=False, float_format="%.6f")

    # 3) leg_contrib (창별 다리별: 사용 수익 단독 CAGR, 산술 기여 비중, 활성일, 승률)
    legs = {("KOSPI", "A", "jehyeong_ON"): bk["rA"], ("KOSPI", "B", "beomgu_C3VT30"): bk["rB"],
            ("KOSPI", "C", "jehyeong_KP"): bk["rC"], ("KOSDAQ", "A", "blog_ON_Q"): bq["rA"],
            ("KOSDAQ", "B", "beomgu_C3Q"): bq["rB"], ("KOSDAQ", "C", "jehyeong_KQ"): bq["rC"]}
    rows = []
    for w, (s, e) in WINDOWS.items():
        tot = win(bk["r"], (s, e)).sum() + win(bq["r"], (s, e)).sum()
        for (mkt, slot, h), r in legs.items():
            m = metrics(win(r, (s, e)))
            m_raw = metrics(win(net[h], (s, e)))
            rows.append(dict(window=w, market=mkt, slot=slot, handle=h, cagr_used=m["cagr"], cagr_standalone=m_raw["cagr"],
                             mdd_used=m["mdd"], sum_share=win(r, (s, e)).sum() / tot if tot else np.nan,
                             n_active=m["n_active"], n_active_raw=m_raw["n_active"], win=m["win"],
                             mean_bp_active=win(r, (s, e)).pipe(lambda x: x[x != 0].mean() * 1e4)))
    lc = pd.DataFrame(rows)
    lc.to_csv(OUT / "leg_contrib.csv", index=False, float_format="%.6f")

    # 4) sensitivity
    def variant(name, slots=None, window=WINDOWS["W_all"], **kw):
        k, q = run_books(net, pos, slots or BASE, **kw)
        rk, rq = win(k["r"], window), win(q["r"], window)
        first = max(rk.first_valid_index(), rq.first_valid_index())
        rk, rq = rk.loc[first:], rq.loc[first:]
        a, b = combine(rk, rq)
        out = {"variant": name, "start": str(a.index[0].date()), "end": str(a.index[-1].date())}
        for lbl, r in {"a": a, "b": b, "KOSPI": rk, "KOSDAQ": rq}.items():
            m = metrics(r)
            out.update({f"{lbl}_cagr": m["cagr"], f"{lbl}_mdd": m["mdd"], f"{lbl}_sharpe": m["sharpe"], f"{lbl}_calmar": m["calmar"]})
        out["verdict_a"] = verdict(metrics(a))
        return out

    def sl(**over):
        s = {m: dict(v) for m, v in BASE.items()}
        for m, d in over.items():
            s[m].update(d)
        return s

    sens = [variant("V0 기준"),
            variant("V1 코스피 B+M2inv", sl(KOSPI={"B": ["beomgu_C3VT30", "beomgu_M2inv"]})),
            variant("V2 코스닥 B+pairQ", sl(KOSDAQ={"B": ["beomgu_C3Q", "pairQ"]})),
            variant("V3 C=blog_ID", sl(KOSPI={"C": ["blog_ID_K"]}, KOSDAQ={"C": ["blog_ID_Q"]})),
            variant("V4 코스닥 A=blog_ON_Q1x", sl(KOSDAQ={"A": ["blog_ON_Q1x"]})),
            variant("V5 A=jehyeong_ON_orig", sl(KOSPI={"A": ["jehyeong_ON_orig"]})),
            variant("V6 비용+2bp", extra_cost=2e-4),
            variant("V7 2026 제외", window=("2022-12-13", "2025-12-30")),
            variant("V8 C 우선", c_priority=True),
            variant("V9 A 없음", sl(KOSPI={"A": []}, KOSDAQ={"A": []})),
            variant("V10 B 없음", sl(KOSPI={"B": []}, KOSDAQ={"B": []})),
            variant("V11 C 없음", sl(KOSPI={"C": []}, KOSDAQ={"C": []}))]
    sens = pd.DataFrame(sens)
    sens.to_csv(OUT / "sensitivity.csv", index=False, float_format="%.6f")

    # 5) yearly
    rows = []
    for y in range(2023, 2027):
        for b, r in books_in((f"{y}-01-01", min(f"{y}-12-31", WINDOWS["W_all"][1]))).items():
            m = metrics(r)
            rows.append(dict(year=y, book=b, ret=m["cum"], mdd=m["mdd"], n_days=m["n_days"], n_active=m["n_active"], win=m["win"]))
    pd.DataFrame(rows).to_csv(OUT / "yearly.csv", index=False, float_format="%.6f")

    # 6) worst_days
    wd = daily.nsmallest(10, "comb_a")[["comb_a", "KOSPI_book", "KOSDAQ_book", "KOSPI_A_jehyeong_ON", "KOSPI_B_beomgu_C3VT30",
                                         "KOSPI_C_jehyeong_KP", "KOSDAQ_A_blog_ON_Q", "KOSDAQ_B_beomgu_C3Q", "KOSDAQ_C_jehyeong_KQ"]]
    wd.to_csv(OUT / "worst_days.csv", float_format="%.6f")

    # 7) corr
    cc = daily.drop(columns=["KOSPI_clash", "KOSDAQ_clash", "comb_b"]).corr()
    cc.to_csv(OUT / "corr.csv", float_format="%.3f")

    # 8) capital_timeline.md
    (OUT / "capital_timeline.md").write_text(CAPITAL_MD, encoding="utf-8")

    # 9) plots
    def plot(series, title, path):
        fig, ax = plt.subplots(figsize=(11, 6))
        for lbl, r in series.items():
            r = r.dropna()
            ax.plot(r.index, (1 + r).cumprod(), label=lbl, lw=2 if "합산" in lbl or "장부" in lbl else 1, alpha=1 if "합산" in lbl else 0.8)
        ax.set_yscale("log")
        ax.set_title(title)
        ax.set_ylabel("누적(배, 로그축)")
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=8, ncol=2)
        fig.tight_layout()
        fig.savefig(path, dpi=130)
        plt.close(fig)

    wa = WINDOWS["W_all"]
    plot({"합산(a)": ra, "코스피 장부": win(bk["r"], wa), "코스닥 장부": win(bq["r"], wa),
          **{f"{m} {s} {h}": win(r, wa) for (m, s, h), r in legs.items()}},
         f"시간대별 알파 장부 W_all {wa[0]}~{wa[1]} (비용 5bp, 사용 수익 기준)", OUT / "equity_W_all.png")
    plot({"합산(a)": la, "코스피 B만": win(lk, W_LONG), "코스닥 A2+B2": win(lq, W_LONG),
          "beomgu_C3VT30": win(net["beomgu_C3VT30"], W_LONG), "blog_ON_Q": win(net["blog_ON_Q"], W_LONG),
          "beomgu_C3Q": win(net["beomgu_C3Q"], W_LONG)},
         f"장기 참고 W_long {W_LONG[0]}~{W_LONG[1]} (A2+B2 와 코스피 B)", OUT / "equity_long.png")

    # 콘솔 요약
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print("== book_metrics ==")
    print(bm[["window", "book", "n_days", "n_active", "cagr", "vol", "sharpe", "mdd", "calmar", "win"]].to_string(index=False))
    print("== verdict ==")
    for w in ("W_all", "W_des", "W_val"):
        m = bm[(bm.window == w) & (bm.book == "합산(a)")].iloc[0]
        print(w, verdict(m), f"CAGR {m.cagr:.2%} MDD {m.mdd:.2%}")
    print("== leg_contrib W_all ==")
    print(lc[lc.window == "W_all"].to_string(index=False))
    print("== sensitivity ==")
    print(sens[["variant", "start", "end", "a_cagr", "a_mdd", "a_sharpe", "a_calmar", "KOSPI_cagr", "KOSPI_mdd", "KOSDAQ_cagr", "KOSDAQ_mdd", "verdict_a"]].to_string(index=False))
    print("== yearly ==")
    print(pd.read_csv(OUT / "yearly.csv").to_string(index=False))
    print("== worst ==")
    print(wd.to_string())
    print("== corr ==")
    print(cc.round(2).to_string())
    print("== notes ==")
    print("\n".join(NOTES))
    (OUT / "notes.json").write_text(json.dumps(NOTES, ensure_ascii=False, indent=1), encoding="utf-8")


CAPITAL_MD = """# 하루 안의 자본 흐름 (장부 1개, 자본 1)

| 시각 | 코스피 장부 | 코스닥 장부 | 자본 상태 |
|---|---|---|---|
| 전일 15:30 | A jehyeong_ON 신호면 069500 종가 매수 | A blog_ON_Q 신호면 233740(2x) 또는 251340(−1x) 종가 매수 | 밤새 A 다리에 묶임(신호 없으면 현금) |
| 09:00 단일가 | A 매도(069500) / B beomgu_C3VT30 신호면 069500+122630 매수 | A 매도 / B beomgu_C3Q 신호면 229200+233740 매수 | A 매도 대금이 같은 단일가 체결로 B 매수에 쓰임 |
| 09:00~09:30 | B 보유 또는 현금 | B 보유 또는 현금 | |
| 09:31 | B 없는 날만 C jehyeong_KP 진입 | B 없는 날만 C jehyeong_KQ 진입 | B 활성일은 C 생략(우선순위 B) |
| 15:20 | C 청산 | C 청산 | 현금 |
| 15:30 종가 | B 청산, 같은 종가로 A 매수 | B 청산, 같은 종가로 A 매수 | B 매도 대금이 A 매수에 쓰임 |

• 09:00 단일가에서 A 매도와 B 매수가 같은 ETF(코스피 069500, 코스닥 233740)면 수량을 상계해 차이만 주문할 수 있다. 상계하면 비용이 줄지만 장부 수치는 두 다리 비용을 모두 뺀 값(보수적).
• 15:30 종가에서도 B 매도와 A 매수가 같은 ETF 면 상계 가능. 단 B 의 레버리지 배수(VT30)와 A 의 수량이 다르므로 차이분은 주문해야 한다.
• 오버나잇 다리 손익은 청산일(익일) 날짜에 기록하므로 장부 일수익의 날짜 = 손익 실현일.
• 주식 계좌만 쓰므로 09:00 단일가 체결은 호가 상황에 따라 불확실. 백테스트는 단일가 시가 체결 가정.
"""

if __name__ == "__main__":
    main()
