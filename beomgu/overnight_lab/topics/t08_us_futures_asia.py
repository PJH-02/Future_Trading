"""T08 · K8 미국 지수선물 아시아장(한국 장중) 변화 → 그날 밤 오버나잇 (팀 9/22 회의 지표 목록 ⑦·①)

사전 고정(실행 전 기록, 2026-09-29):
- us_kr(T) = 219480 KODEX 미국S&P500선물(H) 의 T 시가 → T 종가 수익률 (한국 장중 S&P500 선물 움직임의 대리, 환헤지)
- 분위: 과거 60거래일(약 3개월, T 제외) 70분위. us_kr(T) > q70 (상위 30%, 팀 공유 규칙에서 시각 창만 뺌) → 매수
- 셀 K8-kr : KODEX 200 시간외 종가 매수(체결가 = T 종가, 신호는 15:30 종가로 계산) → T+1 시가 매도
- 셀 K8-us : 같은 신호로 360750 TIGER 미국S&P500 을 같은 방식으로 보유(한국 밤 = 미국 정규장, 목록 ① '미장 상승 마감 예측')
- 판정 구간 2018-01-02 ~ 2025-06-05 (219480 2015~2017 가격 품질 문제로 제외), 360750 은 2020-08-10 ~
- 이웃(보고만): 창 40·120, 분위 80, 신호원 360750(2020-08~)
- 이전: 229200(부호만)
- 비용 5bp(7bp 확인). 219480 체결은 거래대금이 얇아 신호 측정 잡음이 있다(무변동일)
- 보고: yfinance ES=F 1시간봉(약 2년)로 09:00~15:00 KST ES 변화를 직접 잰 판
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common as C  # noqa: E402
import data  # noqa: E402
import engine as E  # noqa: E402

TOPIC = "T08_K8_us_futures_asia"
OUT = data.OUT / TOPIC
JUDGE_START = "2018-01-02"
SPEC = {"src": "219480 O->C", "q": 0.7, "win": 60, "cells": ["KODEX200 night", "360750 night"]}


def sig_from(src_code, index, win=60, q=0.7):
    px = data.price(src_code)
    r = px["close"] / px["open"] - 1
    hi = r.shift(1).rolling(win, min_periods=win).quantile(q)
    return (r > hi).astype(float).where(hi.notna()).reindex(index)


def es_hourly_signal(index):
    """보고용: yfinance ES=F 1시간봉으로 09:00~15:00 KST 변화. 최근 약 730일만."""
    import yfinance as yf
    h = yf.download("ES=F", period="730d", interval="1h", progress=False, auto_adjust=False)
    if h is None or h.empty:
        return None
    h.columns = [c[0] if isinstance(c, tuple) else c for c in h.columns]
    h.index = h.index.tz_convert("Asia/Seoul")
    o = h["Open"].between_time("09:00", "09:00")
    c = h["Close"].between_time("14:00", "14:00")                 # 14:00~15:00 봉 종가 = 15:00 KST
    o.index, c.index = o.index.normalize().tz_localize(None), c.index.normalize().tz_localize(None)
    r = (c / o - 1).dropna()
    hi = r.shift(1).rolling(60, min_periods=60).quantile(0.7)
    return (r > hi).astype(float).where(hi.notna()).reindex(index)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    md = ["# T08 · K8 미국 지수선물 한국 장중 변화 → 그날 밤 오버나잇\n", "```\n" + __doc__.strip() + "\n```\n"]
    rules_all = []
    for cell, code, start in (("K8-kr", "069500", JUDGE_START), ("K8-us", "360750", "2020-08-10")):
        px = data.price(code)
        ret = E.legs(px)["on"]
        s = sig_from("219480", ret.index)
        neigh = {"창40": sig_from("219480", ret.index, 40), "창120": sig_from("219480", ret.index, 120),
                 "분위80": sig_from("219480", ret.index, 60, 0.8), "신호원360750": sig_from("360750", ret.index)}
        pxt = data.price("229200")
        rett = E.legs(pxt)["on"].loc[start:]
        v, mdj = C.judge(f"{cell} ({data.NAMES.get(code, code)} 밤)", s, ret, neighbors=neigh,
                         transfer=(sig_from("219480", rett.index), rett), judge_start=start)
        md += [f"## {cell}: {code} {data.NAMES.get(code, '')}\n", *mdj, ""]
        rj = ret.loc[start:]
        t5, tab5 = C.period_block({cell: s}, rj, 5)
        t7, _ = C.period_block({cell: s}, rj, 7)
        md += ["### 기간별 (5bp)\n", tab5, ""]
        t5["asset"], t7["asset"] = code, code
        rules_all += [t5, t7]
        C.save_verdict(OUT / f"verdict_{cell}.json", v)
    # 보고: ES 1시간봉 직접 측정
    try:
        px = data.price("069500")
        ret = E.legs(px)["on"]
        se = es_hourly_signal(ret.index)
        if se is not None and se.notna().sum() > 50:
            d, t, n = E.delta_test(se, ret)
            st = E.stats(se.fillna(0).loc[se.first_valid_index():], ret.loc[se.first_valid_index():], 5)
            md.append(f"## 보고: ES=F 1시간봉 09:00→15:00 KST 상위 30% → KODEX 200 밤\n\n"
                      f"- 기간 {se.first_valid_index().date()} ~ {ret.dropna().index.max().date()}, 신호 {n}밤, Δ {d * 1e4:+.2f}bp (HAC t {t:+.2f}), "
                      f"신호 밤 순평균(5bp) {st['net_bp']:+.2f}bp\n- 판정 밖(표본 약 2년, E4와 겹침)")
            sig_219 = sig_from("219480", ret.index).reindex(se.dropna().index)
            agree = (sig_219 == se.dropna()).mean()
            md.append(f"- 같은 기간 219480 대리 신호와 ES 직접 신호의 일치율 {agree * 100:.1f}%")
    except Exception as e:  # noqa: BLE001
        md.append(f"## 보고: ES 1시간봉 실패 ({e})")
    (OUT / "report.md").write_text("\n".join(md), encoding="utf-8")
    E.ledger_append(data.OUT, TOPIC, SPEC, pd.concat(rules_all, ignore_index=True))
    print("\n".join(md[2:]))


if __name__ == "__main__":
    main()
