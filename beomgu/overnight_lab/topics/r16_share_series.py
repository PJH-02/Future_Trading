# -*- coding: utf-8 -*-
"""16차 공유용 시계열: 단독 알파 상위 3개와 장부 P1~P3 를 팀 융합 입력 규격(date,pos,gross,net + meta.json)으로 저장한다.

입력: r16_eval_legs.py 가 만든 다리 캐시. 장부 조립은 r16_eval.py 의 sp·parts·notional·run 과 같다.
규칙: 자본 1 기준 일수익, 비활성일 0, 시작일 전 NaN, 밤 다리 손익 = 청산일. 장부 pos = 그날 밤·낮 중 큰 총 명목.
검산: 장부 W 기본 비용 CAGR·MDD 가 eval/book_results.csv 와 같아야 한다(assert).
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
HERE = Path(__file__).resolve().parent
LAB = Path("<OUT>")
EVAL = LAB / "round16_alpha_search/eval"
OUT = LAB / "round16_alpha_search/share/series"
OUT.mkdir(parents=True, exist_ok=True)
CACHE = Path("<local>")
sys.path.insert(0, str(HERE))
d0 = pickle.load(open(CACHE, "rb"))
L, CAL = d0["L"], d0["CAL"]
F = {f: pd.DataFrame({k: getattr(v, f) for k, v in L.items()}) for f in ("net", "gross", "pos")}
ZERO = pd.Series(0.0, index=CAL)
FALSE = pd.Series(False, index=CAL)
BOOK_END = pd.Timestamp("2026-09-22")


def sp(K, Q):
    return dict(K=K, Q=Q)


_KX3 = dict(A=["KA_STf"], B=["KB_E1s3"], C=["KC_KP"], X=["X_K2"])
BOOKS = {"P1": (sp(dict(B=["KB_E1s3"], C=["KC_KP"]), dict(A=["QA_F"], C=["QC_KQ"])), 1.25, None, "E3s"),
         "P2": (sp(dict(A=["KA_STf"], B=["KB_E1s3"], C=["KC_KP"]), dict(A=["QA_F"], C=["QC_KQ"])), 1.00, None, "X2s"),
         "P3": (sp(_KX3, dict(A=["QA_F"], C=["QC_KQ"], X=["X_D1"])), 1.25, 3.0, "X10s")}


def parts(s, field):
    P = {}
    for m in "KQ":
        P[m] = {x: ZERO.copy() for x in "ABCX"}
        act = FALSE.copy()
        for slot, items in s[m].items():
            for k in items:
                P[m][slot] = P[m][slot] + F[field][k].fillna(0.0)
                if slot == "B":
                    act = act | (F["pos"][k].fillna(0.0) != 0)
        P[m]["C"] = P[m]["C"].where(~act, 0.0)
    return P


def notional(s, k):
    nn, dd = ZERO.copy(), ZERO.copy()
    for m in "KQ":
        act = FALSE.copy()
        for k2 in s[m].get("B", []):
            act = act | (F["pos"][k2].fillna(0) != 0)
        for slot, items in s[m].items():
            for x in items:
                p = F["pos"][x].fillna(0).abs()
                if slot == "A":
                    nn = nn + .5 * p
                elif slot == "C":
                    dd = dd + .5 * p.where(~act, 0.0)
                elif slot == "B":
                    dd = dd + .5 * p
                else:
                    nn, dd = nn + .5 * p, dd + .5 * p
    return k * nn, k * dd


def run(s, field, k, cap):
    sA, sD = pd.Series(k, index=CAL), pd.Series(k, index=CAL)
    nn, dd = notional(s, k)
    if cap:
        sA = sA * (cap / nn.where(nn > 0, cap)).clip(upper=1.0)
        sD = sD * (cap / dd.where(dd > 0, cap)).clip(upper=1.0)
    P = parts(s, field)
    night = .5 * (P["K"]["A"] + P["K"]["X"]) + .5 * (P["Q"]["A"] + P["Q"]["X"])
    day = .5 * (P["K"]["C"] + P["K"]["B"]) + .5 * (P["Q"]["C"] + P["Q"]["B"])
    start = min(L[x].start for m in "KQ" for items in s[m].values() for x in items)
    r = sA * night + sD * day
    return r.where(CAL >= start, np.nan), (sA / k * nn), (sD / k * dd)


def met(r):
    r = r.dropna()
    eq = (1 + r).cumprod()
    return eq.iloc[-1] ** (252 / len(r)) - 1, (eq / eq.cummax() - 1).min()


def save(h, df, meta):
    df.round(8).to_csv(OUT / f"{h}.csv", index_label="date", encoding="utf-8")
    (OUT / f"{h}.meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")


LEGS = {"QC_Df": ("코스닥 낮 무필터", "KOSDAQ", "day", "229200 09:05 방향 0.3%, 09:06 진입, 15:00 청산, 롱 233740, 숏 코스닥150 선물(229200 대리)"),
        "X_MRq": ("코스닥 과매도 스윙", "KOSDAQ", "swing", "코스닥150 지수 IBS < 0.1 → 15:45 선물 매수, IBS > 0.7 또는 10거래일 15:45 청산"),
        "X_K2": ("코스피 낮 숏", "KOSPI", "day", "코스피200 선물 개인 순매수 z60 ≥ 0.5 → 다음 날 시가 선물 매도, 15:45 환매")}
for k, (nm, mkt, slot, rule) in LEGS.items():
    df = pd.DataFrame({"pos": F["pos"][k], "gross": F["gross"][k], "net": F["net"][k]})
    save(k, df, dict(handle=k, name=nm, market=mkt, slot=slot, rule=rule, unit="자본 1 기준 일수익", first_valid=str(L[k].start.date()), last_valid=str(df.net.last_valid_index().date()),
                     cost="ETF 왕복 5bp + 연속 체결 1틱, 선물 왕복 1.8bp × 명목, 낮 숏 대리 1.06bp × 명목", status="설계 2024-12-30 까지, 봉인 2025-01-02 이후"))
br = pd.read_csv(EVAL / "book_results.csv")
for b, (s, k, cap, spec) in BOOKS.items():
    net, na, da = run(s, "net", k, cap)
    gross, _, _ = run(s, "gross", k, cap)
    net, gross = net.loc[:BOOK_END], gross.loc[:BOOK_END]
    pos = pd.concat([na, da], axis=1).max(axis=1).loc[:BOOK_END]
    c, m = met(net.loc["2022-12-13":])
    ref = br[(br.book == b) & (br["사전등록k"] == True) & (br.cost == "기본") & (br.window == "W")].iloc[0]
    assert abs(c * 100 - ref.cagr_pct) < 0.02 and abs(m * 100 - ref.mdd_pct) < 0.02, (b, c, m, ref.cagr_pct, ref.mdd_pct)
    print(b, f"W 기본 {c*100:.2f}% / {m*100:.2f}%, 평가 {ref.cagr_pct:.2f}% / {ref.mdd_pct:.2f}%")
    save(b, pd.DataFrame({"pos": pos, "gross": gross, "net": net}),
         dict(handle=b, spec=spec, k=k, total_notional_cap=cap, legs={m2: s[m2] for m2 in "KQ"}, market_split="코스피 50% + 코스닥 50% 매일 재조정", rule="같은 시장 아침 활성일 낮 다리 0",
              unit="자본 1 기준 일수익", last_valid=str(BOOK_END.date()), cost="기본 비용", status="설계 2024-12-30 까지, 봉인 2025-01-02 이후"))
print("저장:", OUT, sorted(p.name for p in OUT.glob("*.csv")))
