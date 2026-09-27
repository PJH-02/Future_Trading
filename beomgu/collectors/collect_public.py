# collect_public.py — 로그인 불필요 소스(yfinance) 일봉 확보
#   기본(인자 없음 또는 kr): KODEX200(069500.KS), KOSPI(^KS11)  ※ K200 지수는 yfinance ^KS200이 깨져 collect_fdr.py가 담당 (이 파일은 k200_index_daily.csv를 건드리지 않음)
#   us: ES=F, NQ=F, ^GSPC, ^IXIC (오버나잇 트랙 인계용)
import sys, os, datetime as dt
import pandas as pd, yfinance as yf
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "raw")
os.makedirs(OUT, exist_ok=True)

def collect_kr():
    stamp = dt.date.today().isoformat()
    for tk, name in [("069500.KS", "kodex200_daily"), ("^KS11", "kospi_daily")]:
        df = yf.download(tk, start="1996-01-01", auto_adjust=False, progress=False)
        if isinstance(df.columns, pd.MultiIndex): df.columns = [c[0] for c in df.columns]
        df.index.name = "date"; p = os.path.join(OUT, f"{name}.csv"); df.to_csv(p, encoding="utf-8")
        print(f"{name}: {len(df)} rows {df.index.min().date()} ~ {df.index.max().date()} -> {p}")
    open(os.path.join(OUT, "_collected_on.txt"), "a", encoding="utf-8").write(stamp + " collect_public.py yfinance kr" + chr(10))

# --- 추가 (2026-09-23): 미국 지수선물·지수 일봉 (오버나잇 트랙 인계용; yfinance, 로그인 불필요) ---
def collect_us():
    import yfinance as yf
    for sym, name in (("ES=F", "es_daily"), ("NQ=F", "nq_daily"), ("^GSPC", "spx_daily"), ("^IXIC", "ndx_comp_daily")):
        d = yf.download(sym, start="1990-01-01", progress=False, auto_adjust=False)
        if isinstance(d.columns, pd.MultiIndex): d.columns = d.columns.get_level_values(0)
        d = d.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]; d.index.name = "date"
        p = os.path.join(OUT, f"{name}.csv"); d.to_csv(p, encoding="utf-8-sig")
        print(f"{sym}: {len(d)} rows {d.index.min().date()}~{d.index.max().date()} -> {p}")
        with open(os.path.join(OUT, "_collected_on.txt"), "a", encoding="utf-8") as f: f.write(f"{dt.date.today()} collect_public.py {name} ({sym}) rows={len(d)}\n")

if __name__ == "__main__":
    if "us" in sys.argv: collect_us()
    else: collect_kr()
