"""실시간 파일럿(페이퍼 트레이딩). 장 시작 전에 신호를 기록하고, 장 마감 후 정산한다. 실제 주문은 하지 않는다.

라인(규칙은 첫 기록 전에 고정, 바꾸면 새 버전으로)
  BASE 매일 매수(기준선)
  A1   외인 선물 Flow(T−1) > 0                  ← 1차 신호: 외인 선물 순매수
  A2   외인 선물 Flow z20(T−1) ≥ 0.84           ← 강한 순매수만
  B    외인 풋 Flow(T−1) < 0                    ← 옵션 레그 후보(탐색에서 나온 조건)
  C    A1 그리고 B                               ← 1차 신호 + 옵션 조건
매매: KODEX 200 (1) T일 시가 → 종가, (2) T일 09:10 → 종가(옵션 장초 10분 검증과 같은 진입 시각). 비용 왕복 0.03% + 1틱.

사용
  python pilot_daily.py signal --date 2026-09-28     # T 전날 장 마감 후 ~ T 09:00 전에 실행
  python pilot_daily.py settle --date 2026-09-28     # T 장 마감(15:45) 후 실행
출력: <out-dir>/pilot/pilot_signals.csv, pilot_results.csv, pilot_summary.md
"""
import argparse
import datetime as dt
import re
import subprocess
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

from optflow.config import resolve_collector_dir, resolve_out_dir, resolve_raw_dir

KST = ZoneInfo("Asia/Seoul")
RULE_VERSION = "pilot-v0-2026-09-27"
LINES = {"BASE": "매일 매수(기준선)", "A1": "외인 선물 Flow>0", "A2": "외인 선물 Flow z20≥0.84", "B": "외인 풋 Flow<0", "C": "A1 그리고 B"}
COMM, TICK = 0.00015, 5.0


def futflow_series(raw):
    f = pd.read_csv(raw / "naver_fut_investor_daily.csv", parse_dates=["date"]).set_index("date").sort_index()
    ok = (f["sellq_foreign"] > 0) & (f["buyq_foreign"] > 0)
    fl = ((f["buyq_foreign"] - f["sellq_foreign"]) / (f["buyq_foreign"] + f["sellq_foreign"])).where(ok)
    return fl


def putflow_series(raw):
    p = raw / "krx_opt_investor_daily.csv"
    if not p.exists():
        return pd.Series(dtype=float)
    o = pd.read_csv(p, parse_dates=["date"])
    o = o[(o["investor"].astype(str).str.lower() == "foreign") & (o["cp"].astype(str).str.upper().str[0] == "P")].set_index("date").sort_index()
    return ((o["buy_amt"] - o["sell_amt"]) / (o["buy_amt"] + o["sell_amt"]))


def cmd_signal(target, raw, out, refresh):
    if refresh:
        subprocess.run([sys.executable, "collect_naver.py", "fut"], cwd=resolve_collector_dir(), check=True, capture_output=True)
    fl = futflow_series(raw)
    prior = fl[fl.index < target].dropna()
    s = prior.index.max()
    hist = prior[prior.index < s].tail(20)
    z = (prior.loc[s] - hist.mean()) / hist.std() if len(hist) == 20 else np.nan
    pf = putflow_series(raw)
    put = pf.loc[s] if s in pf.index else np.nan
    a1, a2 = bool(prior.loc[s] > 0), bool(z >= 0.8416) if z == z else False
    b = bool(put < 0) if put == put else None
    now = dt.datetime.now(KST)
    row = {"target": target.date(), "signal_date": s.date(), "created_at_kst": now.strftime("%Y-%m-%d %H:%M:%S"),
           "before_open": now < dt.datetime.combine(target.date(), dt.time(9, 0), KST), "rule_version": RULE_VERSION,
           "futflow": round(float(prior.loc[s]), 5), "futflow_z20": round(float(z), 3) if z == z else None,
           "putflow": round(float(put), 5) if put == put else None,
           "BASE": 1, "A1": int(a1), "A2": int(a2), "B": None if b is None else int(b), "C": None if b is None else int(a1 and b),
           "gap_days": int((target - s).days), "opt_last_date": str(pf.index.max().date()) if len(pf) else None}
    out.mkdir(parents=True, exist_ok=True)
    p = out / "pilot_signals.csv"
    df = pd.read_csv(p) if p.exists() else pd.DataFrame()
    if len(df) and str(target.date()) in df["target"].astype(str).tolist():
        print(f"이미 {target.date()} 신호가 기록돼 있음 — 덮어쓰지 않음"); print(df[df["target"].astype(str) == str(target.date())].to_string(index=False)); return
    pd.concat([df, pd.DataFrame([row])]).to_csv(p, index=False, encoding="utf-8-sig")
    print(pd.DataFrame([row]).T.to_string(header=False))
    if b is None:
        print(f"주의: KRX 옵션 자료에 {s.date()} 가 없음 → B·C 미정. KRX 로그인 후 갱신 필요")


def naver_chart(symbol, timeframe, count):
    r = requests.get("https://fchart.stock.naver.com/sise.nhn", params={"symbol": symbol, "timeframe": timeframe, "count": count, "requestType": 0},
                     headers={"User-Agent": "Mozilla/5.0"}, timeout=20)
    r.raise_for_status()
    return [x.split("|") for x in re.findall(r'data="([^"]+)"', r.text)]


def cmd_settle(target, out):
    d = {row[0]: row for row in naver_chart("069500", "day", 10)}
    key = target.strftime("%Y%m%d")
    if key not in d:
        sys.exit(f"{target.date()} 일봉 없음(장 마감 전이거나 휴장)")
    o, c = float(d[key][1]), float(d[key][4])
    m = {row[0]: row for row in naver_chart("069500", "minute", 500)}
    p910 = m.get(key + "0910")
    e910 = float(p910[4]) if p910 and p910[4] != "null" else np.nan
    sig = pd.read_csv(out / "pilot_signals.csv")
    srow = sig[sig["target"].astype(str) == str(target.date())]
    if srow.empty:
        sys.exit(f"{target.date()} 신호 기록 없음 — 사후 기록은 파일럿에서 제외")
    srow = srow.iloc[0]
    rows = []
    for ln in LINES:
        on = srow[ln]
        for entry_name, entry in (("open", o), ("0910", e910)):
            g = c / entry - 1 if entry == entry else np.nan
            cost = 2 * COMM + TICK / entry if entry == entry else np.nan
            net = (g - cost) if on == 1 else (0.0 if on == 0 else np.nan)
            rows.append({"target": target.date(), "line": ln, "entry": entry_name, "signal": on, "entry_px": entry, "close_px": c,
                         "gross_bp": round(1e4 * g, 1) if g == g else None, "net_bp": round(1e4 * net, 1) if net == net else None,
                         "valid_realtime": bool(srow["before_open"])})
    p = out / "pilot_results.csv"
    df = pd.read_csv(p) if p.exists() else pd.DataFrame()
    if len(df):
        df = df[df["target"].astype(str) != str(target.date())]
    df = pd.concat([df, pd.DataFrame(rows)])
    df.to_csv(p, index=False, encoding="utf-8-sig")
    summarize(df, out)
    print(pd.DataFrame(rows).to_string(index=False))


def summarize(df, out):
    L = [f"# 파일럿 누적 ({RULE_VERSION}, 페이퍼, 비용 반영)", "", "| 라인 | 진입 | 기록일 | 매매일 | 누적 순수익 | 평균 순수익/회 | 승률 |", "|---|---|---|---|---|---|---|"]
    v = df[df["valid_realtime"] == True]
    for (ln, en), g in v.groupby(["line", "entry"]):
        t = g[g["signal"] == 1]["net_bp"].dropna()
        cum = (1 + g["net_bp"].fillna(0) / 1e4).prod() - 1
        L.append(f"| {ln} {LINES.get(ln, '')} | {en} | {g['target'].nunique()} | {len(t)} | {100 * cum:+.2f}% | {t.mean() if len(t) else float('nan'):+.1f}bp | {100 * (t > 0).mean() if len(t) else float('nan'):.0f}% |")
    L += ["", "장 시작 전에 기록된 신호(valid_realtime)만 집계. 표본이 수십 일이 되기 전에는 결과를 판단하지 않는다."]
    (out / "pilot_summary.md").write_text("\n".join(L), encoding="utf-8")


def main():
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["signal", "settle"])
    ap.add_argument("--date", required=True)
    ap.add_argument("--raw-dir", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--no-refresh", action="store_true")
    a = ap.parse_args()
    target = pd.Timestamp(a.date)
    out = resolve_out_dir(a.out_dir) / "pilot"
    if a.cmd == "signal":
        cmd_signal(target, resolve_raw_dir(a.raw_dir), out, refresh=not a.no_refresh)
    else:
        cmd_settle(target, out)


if __name__ == "__main__":
    main()
