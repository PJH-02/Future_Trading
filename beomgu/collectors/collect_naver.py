# collect_naver.py — 네이버증권(stock.naver.com) 비공식 JSON API로 투자자별 매매동향 일별 수집 (로그인 불필요)
#
# 확인된 엔드포인트 (2026-09-23, stock.naver.com 호스트, GET, 인증 없음):
#   /api/domestic/market/trend/daily?tradeType=KRX&marketType={KOSPI|KOSDAQ|FUT}&bizdate=YYYYMMDD&startIdx=<page,0-based>&pageSize=<=100>
#       FUT = 코스피200 선물 투자자별 (diffValue = 순매수 '계약', sellQuant/buyQuant 계약, sellPrice/buyPrice 금액), 이력 2010-09-27~
#       KOSPI = 코스피 주식 투자자별 (diffValue 원), 이력 2005-01-03~
#   /api/domestic/market/trendProgram?tradeType=KRX&krxMarketType=KOSPI&bizdate=..&startIdx=<page>&pageSize=..&periodType=DAY
#       프로그램매매 (diff*=차익, biDiff*=비차익, total*), 이력 2005-01-21~
# investorGubun: 1000 금융투자 2000 보험 3000 투신 3100 사모 4000 은행 5000 기타금융 6000 연기금등 7000 국가·지자체 7100 기타법인
#                8000 개인 9000 외국인 9001 기타외국인 9999 합계(FUT만)
# 산출: hyfe/data/raw/naver_fut_investor_daily.csv, naver_kospi_investor_daily.csv, naver_program_daily.csv (+ _collected_on.txt)
import os, sys, time, datetime as dt
import requests, pandas as pd

RAW = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "raw"); os.makedirs(RAW, exist_ok=True)
B = "https://stock.naver.com"
H = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/128 Safari/537.36",
     "Referer": "https://stock.naver.com/", "Accept": "application/json, text/plain, */*"}
GUBUN = {"1000": "fin_inv", "2000": "insurance", "3000": "trust", "3100": "private_fund", "4000": "bank", "5000": "other_fin",
         "6000": "pension", "7000": "gov", "7100": "other_corp", "8000": "individual", "9000": "foreign", "9001": "other_foreign", "9999": "total"}
SLEEP = 0.3

def note(msg):
    with open(os.path.join(RAW, "_collected_on.txt"), "a", encoding="utf-8") as f: f.write(f"{dt.date.today()} {msg}\n")

def pages(path, params, page_size=100, max_pages=400):
    """startIdx는 0-based 페이지 인덱스. content가 빌 때까지 순회."""
    out, s = [], requests.Session(); s.headers.update(H)
    for pg in range(max_pages):
        r = s.get(B + path, params={**params, "startIdx": pg, "pageSize": page_size}, timeout=30)
        if r.status_code != 200 or r.text[:1] != "{":
            raise RuntimeError(f"{path} page {pg}: status={r.status_code} body={r.text[:80]!r}")
        c = r.json().get("content", [])
        if not c: break
        out.extend(c); time.sleep(SLEEP)
    return out

def investor_daily(market):
    rows = pages("/api/domestic/market/trend/daily", {"tradeType": "KRX", "marketType": market, "bizdate": dt.date.today().strftime("%Y%m%d")})
    recs = []
    for r in rows:
        d = {"date": r["bizdate"]}
        for a in r.get("netAmounts", []):
            g = GUBUN.get(a["investorGubun"], a["investorGubun"])
            d[f"net_{g}"] = float(a["diffValue"]); d[f"buyq_{g}"] = float(a["buyQuant"]); d[f"sellq_{g}"] = float(a["sellQuant"])
            d[f"buyv_{g}"] = float(a["buyPrice"]); d[f"sellv_{g}"] = float(a["sellPrice"])
        recs.append(d)
    df = pd.DataFrame(recs); df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    df = df.drop_duplicates("date").sort_values("date").set_index("date")
    return df

def program_daily():
    rows = pages("/api/domestic/market/trendProgram", {"tradeType": "KRX", "krxMarketType": "KOSPI", "bizdate": dt.date.today().strftime("%Y%m%d"), "periodType": "DAY"})
    df = pd.DataFrame(rows); df["date"] = pd.to_datetime(df["bizdate"], format="%Y%m%d")
    df = df.drop(columns=["bizdate"]).drop_duplicates("date").sort_values("date").set_index("date")
    for c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.rename(columns={"diffBuyAmt": "arb_buy", "diffSellAmt": "arb_sell", "diffPureBuyAmt": "arb_net", "biDiffBuyAmt": "nonarb_buy",
                              "biDiffSellAmt": "nonarb_sell", "biDiffPureBuyAmt": "nonarb_net", "totalDiffBuyAmt": "tot_buy", "totalDiffSellAmt": "tot_sell", "totalDiffPureBuyAmt": "tot_net"})

def save(df, name):
    p = os.path.join(RAW, f"{name}.csv"); df.to_csv(p, encoding="utf-8-sig")
    print(f"{name}: {len(df)} rows {df.index.min().date()}~{df.index.max().date()} -> {p}")
    note(f"collect_naver.py {name} rows={len(df)} {df.index.min().date()}~{df.index.max().date()}")

if __name__ == "__main__":
    which = sys.argv[1:] or ["fut", "kospi", "program"]
    if "fut" in which:
        f = investor_daily("FUT")
        # 무결성: 합계 제외 순매수 합 = 0 (계약 단위, 매수=매도), 외국인 컬럼 존재
        chk = f[[c for c in f.columns if c.startswith("net_") and c != "net_total"]].sum(axis=1)
        print(f"  FUT net-sum check: max|sum|={chk.abs().max():.0f} (0이어야 함), foreign col={'net_foreign' in f.columns}")
        save(f, "naver_fut_investor_daily")
    if "kospi" in which: save(investor_daily("KOSPI"), "naver_kospi_investor_daily")
    if "program" in which: save(program_daily(), "naver_program_daily")
