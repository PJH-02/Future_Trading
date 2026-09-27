# collect_fdr.py — 로그인 불필요 파이썬 소스로 K200 지수 일봉 수집 (FinanceDataReader 주, 네이버 fchart 교차검증)
# 산출: hyfe/data/raw/k200_index_daily.csv (1990~), hyfe/data/raw/k200_naver_fchart_daily.csv (2014~)
import os, datetime as dt, warnings; warnings.filterwarnings("ignore")
import requests, pandas as pd

RAW = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "raw"); os.makedirs(RAW, exist_ok=True)
def note(msg):
    with open(os.path.join(RAW, "_collected_on.txt"), "a", encoding="utf-8") as f: f.write(f"{dt.date.today()} {msg}\n")

# 1) FinanceDataReader KS200 (KRX 지수, 1990-01-03~)
import FinanceDataReader as fdr
k = fdr.DataReader("KS200", "1990-01-01")
k = k.rename(columns=str.lower)[["open", "high", "low", "close", "volume", "amount"]]
k.index.name = "date"; k = k[k["close"] > 0]
p1 = os.path.join(RAW, "k200_index_daily.csv"); k.to_csv(p1, encoding="utf-8-sig")
print(f"fdr KS200: {len(k)} rows {k.index.min().date()}~{k.index.max().date()} -> {p1}"); note(f"collect_fdr.py k200_index_daily (FDR KS200) rows={len(k)} {k.index.min().date()}~{k.index.max().date()}")

# 2) 네이버 fchart KPI200 (2014~, OHLCV) — 교차검증용
r = requests.get("https://fchart.stock.naver.com/sise.nhn", params={"symbol": "KPI200", "timeframe": "day", "count": 8000, "requestType": 0},
                 headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
import re
rows = [x.split("|") for x in re.findall(r'data="([^"]+)"', r.text)]
n = pd.DataFrame(rows, columns=["date", "open", "high", "low", "close", "volume"])
n["date"] = pd.to_datetime(n["date"], format="%Y%m%d"); n = n.set_index("date").astype(float)
p2 = os.path.join(RAW, "k200_naver_fchart_daily.csv"); n.to_csv(p2, encoding="utf-8-sig")
print(f"naver fchart KPI200: {len(n)} rows {n.index.min().date()}~{n.index.max().date()} -> {p2}"); note(f"collect_fdr.py k200_naver_fchart_daily rows={len(n)}")

# 3) 교차검증: 겹치는 기간 종가 차이
j = k[["close"]].join(n[["close"]], how="inner", lsuffix="_fdr", rsuffix="_naver")
d = (j["close_fdr"] - j["close_naver"]).abs()
print(f"cross-check overlap={len(j)} days, |close diff| max={d.max():.3f} mean={d.mean():.4f}, mismatches(>0.05)={int((d > 0.05).sum())}")
if (d > 0.05).sum(): print(j[d > 0.05].tail(5))
