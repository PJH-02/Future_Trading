"""T07 · K7 현선 베이시스 확대 → 그날 밤 오버나잇 (팀 9/22 회의 지표 목록 ②)

사전 고정(실행 전 기록, 2026-09-29):
- basis(T) = 코스피200 선물 연결 종가(네이버 fchart 'FUT', 15:45 정규장 종가) / 코스피200 지수 종가(15:30) − 1
- 같은 월물 구간 안에서 과거 20일(T 제외, 최소 10일) 평균·SD 로 z. 분기 만기일(3·6·9·12월 둘째 목요일, 휴장이면 직전 거래일)과
  다음 거래일, 12월 마지막 5거래일·1월 첫 3거래일(배당 조정)은 신호 없음
- 셀 H-exp: z ≥ 0.8416 → KODEX 200 시간외 종가(15:40~16:00, 체결가 = T 종가) 매수, T+1 시가 매도
- 보고만: 축소 쪽(z ≤ −0.8416)
- 주의: 15:45 선물 종가에는 15:30 이후 15분 선물 움직임이 들어 있다. 이 몫은 이미 알려진 정보로 과거 가격(15:30 종가)을
  사는 것이라 시간외 종가 체결 가능성·역선택을 따로 따져야 한다
- 이웃(보고만): 임계 0.6745·1.0364, 창 10
- 통제: r_id(T)
- 이전: 229200(부호만)
- 판정 구간 2014-10-01 ~ 2025-06-05(선물 연결 일봉 2014-07~), 비용 5bp(7bp 확인)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common as C  # noqa: E402
import data  # noqa: E402
import engine as E  # noqa: E402

TOPIC = "T07_K7_basis"
OUT = data.OUT / TOPIC
JUDGE_START = "2014-10-01"
SPEC = {"basis": "FUT close(15:45)/K200 close(15:30)-1", "z": "within-contract past20 (min10)", "thr": 0.8416}


def expiries(days):
    """분기 만기일: 3·6·9·12월 둘째 목요일, 휴장이면 직전 거래일."""
    out = []
    for y in range(days.min().year, days.max().year + 1):
        for m in (3, 6, 9, 12):
            first = pd.Timestamp(y, m, 1)
            thu = first + pd.Timedelta(days=(3 - first.weekday()) % 7 + 7)
            if thu > days.max():
                continue
            prev = days[days <= thu]
            if len(prev):
                out.append(prev[-1])
    return pd.DatetimeIndex(out)


def basis_z(win=20, minp=10):
    f, k = data.price("FUT"), data.price("KPI200")
    b = (f["close"] / k["close"] - 1).dropna()
    days = b.index
    ex = expiries(days)
    seg = pd.Series(np.searchsorted(ex.values, days.values, side="left"), index=days)   # 만기일까지 같은 구간, 다음날부터 새 구간
    m = b.groupby(seg).transform(lambda s: s.shift(1).rolling(win, min_periods=minp).mean())
    sd = b.groupby(seg).transform(lambda s: s.shift(1).rolling(win, min_periods=minp).std())
    z = (b - m) / sd
    pos = pd.Series(np.arange(len(days)), index=days)
    bad = pd.Series(False, index=days)
    for e in ex:
        i = pos.get(e)
        if i is not None:
            bad.iloc[i:i + 2] = True
    for y in range(days.min().year, days.max().year + 1):
        dec = days[(days.year == y) & (days.month == 12)]
        jan = days[(days.year == y) & (days.month == 1)]
        bad.loc[dec[-5:]] = True
        bad.loc[jan[:3]] = True
    return z.where(~bad)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    px = data.price("069500")
    ret = E.legs(px)["on"]
    idx = ret.index
    r_id = (px["close"] / px["open"] - 1).rename("r_id")
    z = basis_z()
    z10 = basis_z(10, 6)

    def S(zz, thr=0.8416, up=True, index=idx):
        c = (zz >= thr) if up else (zz <= -thr)
        return c.astype(float).where(zz.notna()).reindex(index)

    pxt = data.price("229200")
    rett = E.legs(pxt)["on"].loc["2016-08-01":]
    neigh = {"임계0.6745": S(z, 0.6745), "임계1.0364": S(z, 1.0364), "창10": S(z10)}
    v, mdj = C.judge("K7 H-exp (베이시스 z ≥ 0.8416)", S(z), ret, neighbors=neigh, transfer=(S(z, index=rett.index), rett),
                     controls=r_id.to_frame(), judge_start=JUDGE_START)
    _, mdc = C.judge("보고: 축소 쪽 (z ≤ −0.8416)", S(z, up=False), ret, judge_start=JUDGE_START)
    rules = {"K7 H-exp": S(z), "보고 축소": S(z, up=False)}
    rj = ret.loc[JUDGE_START:]
    t5, tab5 = C.period_block(rules, rj, 5)
    t7, _ = C.period_block(rules, rj, 7)
    # 진단(판정 밖, 결과를 본 뒤 추가): 효과가 15:30 이후 이미 움직인 선물 가격을 당일 종가에 사는 몫인지 분해
    f, k = data.price("FUT"), data.price("KPI200")
    b = np.log(f["close"] / k["close"])
    b_open_next = np.log(f["open"] / k["open"]).shift(-1)
    k_on = np.log(k["open"].shift(-1) / k["close"])
    f_on = np.log(f["open"].shift(-1) / f["close"])
    collapse = b - b_open_next                                   # 항등식: k_on = b + f_on − b_open_next
    sj = S(z).loc[JUDGE_START:"2025-06-05"]
    diag = [("KODEX 200 r_on, 베이시스 z(T−1) 신호(사전 정보만)", S(z).shift(1).loc[JUDGE_START:"2025-06-05"], ret),
            ("코스피200 지수 r_on (15:30 → 익일 시가)", sj, k_on), ("선물 r_on (15:45 종가 → 익일 시가)", sj, f_on),
            ("베이시스 붕괴 b(T) − b_open(T+1)", sj, collapse)]
    md_d = ["\n## 진단 (판정 밖): 15:30 이후 선물 움직임 분해\n", "| 구간 | 신호 밤 | Δ bp (HAC t) |", "|---|---|---|"]
    for lab, sg, y in diag:
        d_, t_, n_ = E.delta_test(sg, y.reindex(sg.index))
        md_d.append(f"| {lab} | {n_} | {d_ * 1e4:+.2f} ({t_:+.2f}) |")
    md_d.append("\n해석: 신호 밤의 초과 수익이 선물 종가 이후 구간(선물 r_on)이 아니라 베이시스 붕괴 몫에서 나오면, "
                "효과는 마감 뒤 이미 공개된 선물 가격을 과거 종가에 사는 몫이다(시간외 종가 체결 가능성·역선택 별도).")
    md = ["# T07 · K7 현선 베이시스 확대 → 그날 밤 오버나잇\n", "```\n" + __doc__.strip() + "\n```\n",
          "## 판정 (KODEX 200, 2014-10-01 ~ 2025-06-05)\n", *mdj, "", *mdc, *md_d, "\n## 기간별 (5bp)\n", tab5]
    (OUT / "report.md").write_text("\n".join(md), encoding="utf-8")
    t5["asset"], t7["asset"] = "069500", "069500"
    E.ledger_append(data.OUT, TOPIC, SPEC, pd.concat([t5, t7], ignore_index=True))
    C.save_verdict(OUT / "verdict.json", v)
    print("\n".join(md[2:]))


if __name__ == "__main__":
    main()
