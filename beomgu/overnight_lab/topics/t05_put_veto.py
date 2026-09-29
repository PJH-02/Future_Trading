"""T05 · K5 외인 옵션 풋 거부권 (옵션 레그) + N5 교차 적용

사전 고정(실행 전 기록, 2026-09-29):
- PutFlow(d) = (외인 풋 매수금액 − 매도금액) / (매수 + 매도), KRX 코스피200 옵션 월물·정규장 일별 확정치
- z(d) = PutFlow(d) 의 과거 60일 z (d 제외), 번인 60일
- 진입일 T 의 신호는 z(T−1). z(T−1) ≥ 1.0 이면 경보 → T 밤 보유 안 함. 그 밖에는 매일 보유(종가 → 익일 시가)
- 판정: Δveto = 평균(비경보 밤) − 평균(경보 밤) > 0, 경보 밤 평균 < 5bp(쉬는 편이 이득)
- 이웃(보고만): 임계 0.8416·1.2816, 창 20·120
- 통제: r_id(T), 외인 선물 순매수 z20(T−1)
- 이전: 229200(교차 시장, 부호만). 360750 TIGER 미국S&P500 은 음성 대조(보고만)
- 판정 구간 2010-11-01 ~ 2025-06-05, 비용 5bp(7bp 확인)
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common as C  # noqa: E402
import data  # noqa: E402
import engine as E  # noqa: E402

TOPIC = "T05_K5_put_veto"
OUT = data.OUT / TOPIC
JUDGE_START = "2010-11-01"
SPEC = {"putflow": "foreign monthly amt", "z": "past60", "alarm": "z(T-1)>=1.0 -> skip night T"}


def putflow():
    o = data.krx_options()
    fo = o[(o["investor"] == "foreign") & (o["cp"] == "P")].groupby("date")[["buy_amt", "sell_amt"]].sum(min_count=1)
    return ((fo["buy_amt"] - fo["sell_amt"]) / (fo["buy_amt"] + fo["sell_amt"])).sort_index()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pf = putflow()
    px = data.price("069500")
    ret = E.legs(px)["on"]
    idx = ret.index
    r_id = (px["close"] / px["open"] - 1).rename("r_id")
    fut = data.naver_investor("fut")
    fz = C.zscore_past((fut["buyq_foreign"] - fut["sellq_foreign"]).where(fut["sellq_foreign"] > 0), 20).shift(1).rename("fut_z20_lag")

    def alarm(win=60, thr=1.0, index=idx):
        z = C.zscore_past(pf, win)
        a = (z >= thr).astype(float).where(z.notna())
        # KRX 옵션 거래일 기준으로 T−1 값 → 다음 KRX 거래일(T)에 적용
        return a.shift(1).reindex(index)

    main_a = alarm()
    neigh = {"임계0.8416": alarm(thr=0.8416), "임계1.2816": alarm(thr=1.2816), "창20": alarm(20), "창120": alarm(120)}
    pxt = data.price("229200")
    rett = E.legs(pxt)["on"].loc["2016-08-01":]
    v, mdj = C.judge("K5 풋 거부권 (z60(T−1) ≥ 1.0 이면 쉼)", main_a, ret, neighbors=neigh, transfer=(alarm(index=rett.index), rett),
                     controls=pd.concat([r_id, fz], axis=1), veto=True, judge_start=JUDGE_START)
    md = ["# T05 · K5 외인 옵션 풋 거부권 (+N5)\n", "```\n" + __doc__.strip() + "\n```\n",
          "## 판정 (KODEX 200, 2010-11-01 ~ 2025-06-05)\n", *mdj, ""]
    # 음성 대조
    pxu = data.price("360750")
    retu = E.legs(pxu)["on"]
    _, mdu = C.judge("N5 음성 대조: 360750 TIGER 미국S&P500", alarm(index=retu.index), retu, veto=True, judge_start="2020-08-10")
    md += mdu + [""]
    # 샤프 비교: 매일 보유 vs 거부권 적용
    rj = ret.loc[JUDGE_START:].where(main_a.notna())
    rules = {"거부권 적용": 1 - main_a}
    t5, tab5 = C.period_block(rules, rj, 5)
    t7, tab7 = C.period_block(rules, rj, 7)
    md += ["## 기간별 (5bp) — '거부권 적용' 은 경보 밤만 뺀 매일 보유\n", tab5, "\n## 기간별 (7bp)\n", tab7]
    (OUT / "report.md").write_text("\n".join(md), encoding="utf-8")
    t5["asset"], t7["asset"] = "069500", "069500"
    E.ledger_append(data.OUT, TOPIC, SPEC, pd.concat([t5, t7], ignore_index=True))
    C.save_verdict(OUT / "verdict.json", v)
    # 팀 제공용 경보 열
    alarm_csv = pd.DataFrame({"putflow_foreign_monthly": pf, "z60": C.zscore_past(pf, 60),
                              "alarm_for_next_trading_day": (C.zscore_past(pf, 60) >= 1.0).astype("Int64").where(C.zscore_past(pf, 60).notna())})
    alarm_csv.to_csv(OUT / "put_veto_alarm.csv", encoding="utf-8-sig")
    print("\n".join(md[2:]))


if __name__ == "__main__":
    main()
