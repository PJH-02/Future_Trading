"""T02 · K2 외인 선물 예상외 순매수 → 그날 밤 오버나잇 (팀 9/22 회의 지표 목록 ③)

사전 고정(실행 전 기록, 2026-09-29):
- net_f(T) = 외인 코스피200 선물 매수계약 − 매도계약(네이버 일별 확정치, 매도 0 결함 행은 NaN)
- 예상외 z(T) = (net_f(T) − 과거 20일 평균) / 과거 20일 SD (T 제외)
- z(T) ≥ 0.8416(상위 20% 절대 임계, H-F-1과 같은 값) → KODEX 200 시간외 종가 매수, T+1 시가 매도
- 관측: T일 15:45 선물 마감 후 외인 누적(HTS 실시간 필요, 네이버는 20분 지연) → 15:40~16:00 시간외 종가
- 이웃(보고만): 임계 0.6745·1.0364, 창 60, F2 정의(Flow비율 > 과거 250일 80분위)
- 통제: 당일 장중 수익률 r_id(T)
- 이전: 229200 코스닥150 (부호만). 실행 가능판(T−1 값 → 그날 밤)은 보고만
- 판정 구간 2011-01-03 ~ 2025-06-05, 비용 5bp(7bp 확인)
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common as C  # noqa: E402
import data  # noqa: E402
import engine as E  # noqa: E402

TOPIC = "T02_K2_foreign_futures_surprise"
OUT = data.OUT / TOPIC
JUDGE_START = "2011-01-03"
SPEC = {"x": "net_f contracts", "z": "past20 z", "thr": 0.8416, "entry": "T after-hours close", "exit": "T+1 open"}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fut = data.naver_investor("fut")
    net = (fut["buyq_foreign"] - fut["sellq_foreign"]).where(fut["sellq_foreign"] > 0)
    flow = ((fut["buyq_foreign"] - fut["sellq_foreign"]) / (fut["buyq_foreign"] + fut["sellq_foreign"])).where(fut["sellq_foreign"] > 0)
    z20 = C.zscore_past(net, 20)
    z60 = C.zscore_past(net, 60)
    q80 = flow.shift(1).rolling(250, min_periods=250).quantile(0.8)

    def sig_of(cond, base):
        return cond.astype(float).where(base.notna())

    px = data.price("069500")
    ret = E.legs(px)["on"]
    idx = ret.index
    main_sig = sig_of(z20 >= 0.8416, z20).reindex(idx)
    neighbors = {"z≥0.6745": sig_of(z20 >= 0.6745, z20).reindex(idx), "z≥1.0364": sig_of(z20 >= 1.0364, z20).reindex(idx),
                 "창60": sig_of(z60 >= 0.8416, z60).reindex(idx), "F2 250일80분위": sig_of(flow > q80, q80).reindex(idx)}
    r_id = (px["close"] / px["open"] - 1).rename("r_id")
    pxt = data.price("229200")
    rett = E.legs(pxt)["on"]
    tsig = main_sig.reindex(rett.index)

    v, md_j = C.judge("K2 외인 선물 예상외 순매수 z20 ≥ 0.8416", main_sig, ret, neighbors=neighbors,
                      transfer=(tsig, rett.loc["2016-08-01":]), controls=r_id.to_frame(), judge_start=JUDGE_START)
    lag_sig = main_sig.shift(1)
    v_lag, md_lag = C.judge("보고: 실행 가능판 z20(T−1) ≥ 0.8416 → T 밤", lag_sig, ret, judge_start=JUDGE_START)

    rules = {"K2 z20≥0.84": main_sig, "보고 T−1판": lag_sig}
    t5, tab5 = C.period_block(rules, ret.loc[JUDGE_START:], 5)
    t7, _ = C.period_block(rules, ret.loc[JUDGE_START:], 7)
    md = ["# T02 · K2 외인 선물 예상외 순매수 → 그날 밤 오버나잇\n", "```\n" + __doc__.strip() + "\n```\n",
          "## 판정 (KODEX 200, 2011-01-03 ~ 2025-06-05)\n", *md_j, "", *md_lag, "\n## 기간별 (5bp)\n", tab5]
    (OUT / "report.md").write_text("\n".join(md), encoding="utf-8")
    t5["asset"], t7["asset"] = "069500", "069500"
    E.ledger_append(data.OUT, TOPIC, SPEC, pd.concat([t5, t7], ignore_index=True))
    C.save_verdict(OUT / "verdict.json", v)
    print("\n".join(md[2:]))


if __name__ == "__main__":
    main()
