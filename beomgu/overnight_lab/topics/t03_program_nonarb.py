"""T03 · K3 프로그램 비차익 순매수/순매도 → 그날 밤 오버나잇 (팀 9/22 회의 지표 목록 ⑥) + N2 코스닥

사전 고정(실행 전 기록, 2026-09-29):
- NF(T) = 코스피 비차익 순매수 / (비차익 매수 + 매도), 네이버 일별 확정치(장 마감 후 공표)
- z(T) = NF(T)의 과거 60일 z (T 제외)
- 셀 H-cont(목록 그대로, 순매수 → 매수): z ≥ +0.8416
- 셀 H-rev(마감 유동성 압력 반전, 설계 F1): z ≤ −0.8416
- 진입: T 시간외 종가(15:40~16:00, 체결가 = T 종가), 청산: T+1 시가
- 이웃(보고만): 임계 0.6745·1.0364, 창 20, 전체 프로그램(차익+비차익) Flow
- 통제: 당일 장중 수익률 r_id(T)
- 이전(N2): 코스닥 비차익 z60 → 229200 코스닥150 (부호만)
- 판정 구간 2010-12-16 ~ 2025-06-05, 비용 5bp(7bp 확인)
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common as C  # noqa: E402
import data  # noqa: E402
import engine as E  # noqa: E402

TOPIC = "T03_K3_program_nonarb"
OUT = data.OUT / TOPIC
JUDGE_START = "2010-12-16"
SPEC = {"x": "nonarb_net/(buy+sell)", "z": "past60", "cells": {"H-cont": ">= +0.8416", "H-rev": "<= -0.8416"}}


def flow(prog, kind="nonarb"):
    return prog[f"{kind}_net"] / (prog[f"{kind}_buy"] + prog[f"{kind}_sell"])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    prog = data.naver_program("KOSPI")
    nf = flow(prog)
    tf = prog["tot_net"] / (prog["tot_buy"] + prog["tot_sell"])
    z60, z20, zt = C.zscore_past(nf, 60), C.zscore_past(nf, 20), C.zscore_past(tf, 60)
    px = data.price("069500")
    ret = E.legs(px)["on"]
    idx = ret.index
    r_id = (px["close"] / px["open"] - 1).rename("r_id")

    kq = data.naver_program("KOSDAQ")
    zk = C.zscore_past(flow(kq), 60)
    pxt = data.price("229200")
    rett = E.legs(pxt)["on"].loc["2016-08-01":]

    def S(cond, base, index=idx):
        return cond.astype(float).where(base.notna()).reindex(index)

    md = ["# T03 · K3 프로그램 비차익 → 그날 밤 오버나잇 (+N2 코스닥)\n", "```\n" + __doc__.strip() + "\n```\n",
          "## 판정 (KODEX 200, 2010-12-16 ~ 2025-06-05)\n"]
    verdicts, rules = {}, {}
    for cell, sgn in (("H-cont", 1), ("H-rev", -1)):
        thr = 0.8416

        def cond(z, t=thr):
            return (z >= t) if sgn > 0 else (z <= -t)
        main_sig = S(cond(z60), z60)
        neigh = {"임계0.6745": S(cond(z60, 0.6745), z60), "임계1.0364": S(cond(z60, 1.0364), z60),
                 "창20": S(cond(z20), z20), "전체프로그램": S(cond(zt), zt)}
        tsig = S(cond(zk), zk, rett.index)
        v, mdj = C.judge(f"K3 {cell} (비차익 z60 {'≥ +' if sgn > 0 else '≤ −'}0.8416)", main_sig, ret, neighbors=neigh,
                         transfer=(tsig, rett), controls=r_id.to_frame(), judge_start=JUDGE_START)
        verdicts[cell] = v
        rules[f"K3 {cell}"] = main_sig
        md += mdj + [""]
        # N2 코스닥 자체 판정(보고)
        vk, mdk = C.judge(f"N2 보고: 코스닥 비차익 {cell} → 229200", tsig, rett, judge_start="2016-08-01")
        md += mdk + [""]
    rj = ret.loc[JUDGE_START:]  # 기간별 표도 판정 시작일부터(버그 수정 재실행, 2026-09-29)
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
