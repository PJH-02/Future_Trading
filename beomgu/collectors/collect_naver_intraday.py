# collect_naver_intraday.py — 네이버증권 장중 '시간별' 투자자 잠정치 스냅샷 수집 (당일분만 제공되므로 매 거래일 실행 필요)
#   /api/domestic/market/trend/time?marketType=FUT|KOSPI  → 1분 간격 누적 순매수(계약/원) by investorGubun (bizdate 무시, 당일만)
#   /api/domestic/market/trendProgram?periodType=TIME     → 1분 간격 프로그램 누적 (차익/비차익)
# 실행 권장: 장 마감 후(15:50 이후) 1회. 산출: hyfe/data/raw/intraday/naver_time_<YYYYMMDD>.csv (append-safe: 같은 날 재실행 시 덮어씀)
import os, sys, time, datetime as dt
import requests, pandas as pd
from collect_naver import B, H, GUBUN, SLEEP, pages, note
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "raw", "intraday"); os.makedirs(OUT, exist_ok=True)

def snap_investor(market):
    rows = pages("/api/domestic/market/trend/time", {"tradeType": "KRX", "marketType": market, "bizdate": dt.date.today().strftime("%Y%m%d")})
    recs = []
    for r in rows:
        d = {"date": r["bizdate"], "time": r["time"], "market": market}
        for a in r.get("netAmounts", []):
            g = GUBUN.get(a["investorGubun"], a["investorGubun"]); d[f"net_{g}"] = float(a["diffValue"]); d[f"buyq_{g}"] = float(a["buyQuant"]); d[f"sellq_{g}"] = float(a["sellQuant"])
        recs.append(d)
    return pd.DataFrame(recs)

def snap_program():
    rows = pages("/api/domestic/market/trendProgram", {"tradeType": "KRX", "krxMarketType": "KOSPI", "bizdate": dt.date.today().strftime("%Y%m%d"), "periodType": "TIME"})
    df = pd.DataFrame(rows); df["market"] = "PROGRAM_KOSPI"; return df.rename(columns={"bizdate": "date"})

if __name__ == "__main__":
    today = dt.date.today().strftime("%Y%m%d")
    parts = [snap_investor("FUT"), snap_investor("KOSPI"), snap_program()]
    df = pd.concat(parts, ignore_index=True)
    dates = sorted(set(df["date"].astype(str)))
    if dates != [today]: print(f"주의: 응답 날짜 {dates} ≠ 오늘 {today} (휴장일/장전 실행?)")
    p = os.path.join(OUT, f"naver_time_{dates[-1]}.csv"); df.sort_values(["market", "time"]).to_csv(p, index=False, encoding="utf-8-sig")
    print(f"{p}: {len(df)} rows; FUT={len(parts[0])} KOSPI={len(parts[1])} PROGRAM={len(parts[2])}; FUT time range={parts[0]['time'].min()}~{parts[0]['time'].max()}")
    note(f"collect_naver_intraday.py {dates[-1]} rows={len(df)}")
