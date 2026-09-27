"""KODEX 200(069500) 일봉을 네이버 fchart에서 받아 원자료 폴더에 저장한다(yfinance 결측일 보완용).

사용: python scripts/fetch_kodex_naver.py [--raw-dir PATH]
산출: kodex200_naver_fchart_daily.csv (date,open,high,low,close,volume). 분배금 수정주가다.
네이버 fchart 는 최근 약 3,000행만 준다. 기존 파일이 있으면 과거 행을 유지하고 새 행과 병합한다(겹치는 날은 새 값).
수집 기록은 _collected_on.txt 에 남긴다.
"""
import argparse
import datetime as dt
import re
import sys
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from optflow.config import resolve_raw_dir  # noqa: E402


def main():
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default=None)
    a = ap.parse_args()
    raw = resolve_raw_dir(a.raw_dir)
    r = requests.get("https://fchart.stock.naver.com/sise.nhn",
                     params={"symbol": "069500", "timeframe": "day", "count": 8000, "requestType": 0},
                     headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
    r.raise_for_status()
    rows = [x.split("|") for x in re.findall(r'data="([^"]+)"', r.text)]
    if not rows:
        raise RuntimeError("네이버 fchart 응답에 데이터 없음")
    new = pd.DataFrame(rows, columns=["date", "open", "high", "low", "close", "volume"])
    new["date"] = pd.to_datetime(new["date"], format="%Y%m%d")
    new = new.set_index("date").astype(float).sort_index()
    out = raw / "kodex200_naver_fchart_daily.csv"
    kept = 0
    if out.exists():
        old = pd.read_csv(out, parse_dates=["date"]).set_index("date")
        keep = old.loc[old.index.difference(new.index)]
        kept = len(keep)
        new = pd.concat([keep, new]).sort_index()
    new.to_csv(out, encoding="utf-8-sig")
    msg = f"fetch_kodex_naver.py kodex200_naver_fchart_daily rows={len(new)} {new.index.min().date()}~{new.index.max().date()} (기존 과거행 유지 {kept})"
    with open(raw / "_collected_on.txt", "a", encoding="utf-8") as f:
        f.write(f"{dt.date.today()} {msg}\n")
    print(f"{out}: {msg}")


if __name__ == "__main__":
    main()
