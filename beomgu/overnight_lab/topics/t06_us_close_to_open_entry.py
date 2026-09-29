"""T06 · K6 익일 시가 매수: 전일 밤 미국 마감 → 한국 시가 매수·종가 매도 (팀 9/22 회의 '익일 시가' 트랙)

사전 고정(실행 전 기록, 2026-09-29):
- us(T) = 한국 거래일 T 의 09:00 이전에 끝난 가장 최근 미국 정규장의 S&P500 종가→종가 수익률
  (미국 날짜 D < T 인 마지막 D. 미국 휴장이면 그 전 세션)
- 분위: 과거 60개 미국 세션(해당 세션 제외)의 70·30 분위
- 셀 H-cont: us(T) > q70 → T 시가 단일가 매수, T 종가 단일가 매도
- 셀 H-rev : us(T) < q30 → 같은 방식 (개장 과잉반영 후 반전 가설, 이우백 2015)
- 목표: KODEX 200 T 시가 → T 종가. 기준선: 매일 시가 매수·종가 매도
- 이웃(보고만): 창 40·120, 분위 80/20
- 이전: 229200 코스닥150 에 나스닥 종합 대신 같은 S&P500 신호(부호만)
- 판정 구간 2010-12-16 ~ 2025-06-05, 비용 5bp(7bp 확인)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common as C  # noqa: E402
import data  # noqa: E402
import engine as E  # noqa: E402

TOPIC = "T06_K6_us_close_to_kr_open"
OUT = data.OUT / TOPIC
JUDGE_START = "2010-12-16"
SPEC = {"us": "SPX close-close of last US session before KR T", "q": [0.7, 0.3], "win": 60, "target": "KODEX T open->close"}


def us_signal(kr_index, win=60, qh=0.7, ql=0.3):
    spx = data.us_daily("spx")["close"].dropna()
    r = spx.pct_change().dropna()
    hi = r.shift(1).rolling(win, min_periods=win).quantile(qh)
    lo = r.shift(1).rolling(win, min_periods=win).quantile(ql)
    pos = np.searchsorted(r.index.values, kr_index.values, side="left") - 1      # D < T 인 마지막 미국 세션
    valid = pos >= 0
    # 데이터 끝 이후 날짜에 마지막 미국 세션 신호가 복사되지 않도록: 한국 T 와 미국 D 의 간격이 4일(주말+휴장)을 넘으면 결측
    gap = (kr_index.values - r.index.values[np.clip(pos, 0, None)]).astype("timedelta64[D]").astype(int)
    valid = valid & (gap <= 4)
    take = lambda s: pd.Series(np.where(valid, s.to_numpy()[np.clip(pos, 0, None)], np.nan), index=kr_index)  # noqa: E731
    ru, h, l_ = take(r), take(hi), take(lo)
    ok = h.notna()
    return {"H-cont": (ru > h).astype(float).where(ok), "H-rev": (ru < l_).astype(float).where(ok)}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    px = data.price("069500")
    ret = E.legs(px)["oc"]
    sig = us_signal(ret.index)
    pxt = data.price("229200")
    rett = E.legs(pxt)["oc"].loc["2016-08-01":]
    sigt = us_signal(rett.index)
    md = ["# T06 · K6 전일 밤 미국 마감 → 한국 시가 매수·종가 매도\n", "```\n" + __doc__.strip() + "\n```\n",
          "## 판정 (KODEX 200 시가→종가, 2010-12-16 ~ 2025-06-05)\n"]
    verdicts, rules = {}, {}
    for cell in ("H-cont", "H-rev"):
        neigh = {"창40": us_signal(ret.index, 40)[cell], "창120": us_signal(ret.index, 120)[cell],
                 "분위80/20": us_signal(ret.index, 60, 0.8, 0.2)[cell]}
        v, mdj = C.judge(f"K6 {cell}", sig[cell], ret, neighbors=neigh, transfer=(sigt[cell], rett), judge_start=JUDGE_START)
        verdicts[cell], rules[f"K6 {cell}"] = v, sig[cell]
        md += mdj + [""]
    rj = ret.loc[JUDGE_START:]
    t5, tab5 = C.period_block(rules, rj, 5)
    t7, _ = C.period_block(rules, rj, 7)
    md += ["## 기간별 (5bp, 기준선 = 매일 시가 매수·종가 매도)\n", tab5]
    (OUT / "report.md").write_text("\n".join(md), encoding="utf-8")
    t5["asset"], t7["asset"] = "069500", "069500"
    E.ledger_append(data.OUT, TOPIC, SPEC, pd.concat([t5, t7], ignore_index=True))
    C.save_verdict(OUT / "verdict.json", verdicts)
    print("\n".join(md[2:]))


if __name__ == "__main__":
    main()
