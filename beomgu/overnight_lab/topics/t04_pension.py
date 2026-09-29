"""T04 · K4 연기금 순매수 → 그날 밤 오버나잇 (팀 9/22 회의 지표 목록 ④) + N3 코스닥

사전 고정(실행 전 기록, 2026-09-29):
- 셀 K4-spot: PF5(T) = 연기금 코스피 현물 순매수 5일 합 / 매수+매도 5일 합(금액), 과거 60일 z(T 제외) ≥ 0.8416 → 매수
- 셀 K4-fut : 연기금 코스피200 선물 순매수 5일 합 / 매수+매도 5일 합(계약), 과거 60일 z ≥ 0.8416 → 매수
  (근거: 우민철·김명애 2021 연기금 선물 순매수 5~30일 누적 → 익일 수익률 +)
- 진입: T 시간외 종가(체결가 = T 종가, T일 투자자별 확정치를 본 뒤), 청산: T+1 시가
- 이웃(보고만): 1일, 20일 누적, 임계 1.0364
- 통제: r_id(T)
- 이전(N3): 코스닥 연기금 현물 PF5 → 229200 (부호만). 코스닥 외인 현물 PF5 는 보고만
- 판정 구간 2010-12-16 ~ 2025-06-05, 비용 5bp(7bp 확인)
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common as C  # noqa: E402
import data  # noqa: E402
import engine as E  # noqa: E402

TOPIC = "T04_K4_pension"
OUT = data.OUT / TOPIC
JUDGE_START = "2010-12-16"
SPEC = {"spot": "pension PF5 amt z60>=0.8416", "fut": "pension futures PF5 qty z60>=0.8416"}


def pf(df, who, n, unit):
    b, s = df[f"buy{unit}_{who}"], df[f"sell{unit}_{who}"]
    tot = (b + s).rolling(n).sum()
    return ((b - s).rolling(n).sum() / tot).where(tot > 0)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    k, f, q = data.naver_investor("kospi"), data.naver_investor("fut"), data.naver_kosdaq_investor()
    px = data.price("069500")
    ret = E.legs(px)["on"]
    idx = ret.index
    r_id = (px["close"] / px["open"] - 1).rename("r_id")
    pxt = data.price("229200")
    rett = E.legs(pxt)["on"].loc["2016-08-01":]

    def S(x, thr=0.8416, index=idx):
        z = C.zscore_past(x, 60)
        return (z >= thr).astype(float).where(z.notna()).reindex(index)

    md = ["# T04 · K4 연기금 순매수 → 그날 밤 오버나잇 (+N3 코스닥)\n", "```\n" + __doc__.strip() + "\n```\n",
          "## 판정 (KODEX 200, 2010-12-16 ~ 2025-06-05)\n"]
    verdicts, rules = {}, {}
    cells = {"K4-spot": (k, "v"), "K4-fut": (f, "q")}
    for cell, (df, unit) in cells.items():
        main_sig = S(pf(df, "pension", 5, unit))
        neigh = {"1일": S(pf(df, "pension", 1, unit)), "20일": S(pf(df, "pension", 20, unit)),
                 "임계1.0364": S(pf(df, "pension", 5, unit), 1.0364)}
        tsig = S(pf(q, "pension", 5, "v"), index=rett.index)
        v, mdj = C.judge(f"{cell} (연기금 5일 누적 z60 ≥ 0.8416)", main_sig, ret, neighbors=neigh,
                         transfer=(tsig, rett), controls=r_id.to_frame(), judge_start=JUDGE_START)
        verdicts[cell] = v
        rules[cell] = main_sig
        md += mdj + [""]
    for who, lab in (("pension", "연기금"), ("foreign", "외인")):
        tsig = S(pf(q, who, 5, "v"), index=rett.index)
        _, mdk = C.judge(f"N3 보고: 코스닥 {lab} 현물 5일 누적 z60 ≥ 0.8416 → 229200", tsig, rett, judge_start="2016-08-01")
        md += mdk + [""]
    rj = ret.loc[JUDGE_START:]
    t5, tab5 = C.period_block(rules, rj, 5)
    t7, _ = C.period_block(rules, rj, 7)
    md += ["## 기간별 (5bp)\n", tab5]
    (OUT / "report.md").write_text("\n".join(md), encoding="utf-8")
    t5["asset"], t7["asset"] = "069500", "069500"
    E.ledger_append(data.OUT, TOPIC, SPEC, pd.concat([t5, t7], ignore_index=True))
    C.save_verdict(OUT / "verdict.json", verdicts)
    print("\n".join(md[2:]))


if __name__ == "__main__":
    main()
