"""오버나잇·종가 베팅 랩 데이터 로더.

- 원자료 폴더(RAW): 네이버 투자자별·프로그램, KRX 옵션, 미국 지수 일봉 등
- 가격: 네이버 siseJson 일봉(분배 수정주가, 상장일~). <RAW>/etf/<symbol>.csv 에 캐시한다.
  수정주가라 분배락 하락이 시가 갭에 섞이지 않는다. 지수(KPI200)·선물 연결(FUT)은 fchart(2014-07~)로 받는다.
- 장중에 받은 당일 행은 확정치가 아니므로 16:00 KST 이전 수집이면 당일 행을 버린다.

폴더 결정 순서: 환경변수(OVERNIGHT_RAW_DIR, OVERNIGHT_OUT_DIR, OVERNIGHT_COLLECTOR_DIR) → 이 폴더의 config.local.json
(raw_dir, out_dir, collector_dir; 깃 제외) → 기본값 ../data, ../results/overnight_lab, ../collectors
"""
import datetime as dt
import json
import os
import re
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests

KST = ZoneInfo("Asia/Seoul")
PKG = Path(__file__).resolve().parent


def _dir(env, key, default):
    cfg = PKG / "config.local.json"
    val = os.environ.get(env) or (json.loads(cfg.read_text(encoding="utf-8")).get(key) if cfg.exists() else None) or default
    p = Path(val)
    return (p if p.is_absolute() else PKG / p).resolve()


RAW = _dir("OVERNIGHT_RAW_DIR", "raw_dir", "../data")
OUT = _dir("OVERNIGHT_OUT_DIR", "out_dir", "../results/overnight_lab")
COLLECTORS = _dir("OVERNIGHT_COLLECTOR_DIR", "collector_dir", "../collectors")
PX_DIR = RAW / "etf"

NAMES = {
    "069500": "KODEX 200", "229200": "KODEX 코스닥150", "122630": "KODEX 레버리지", "233740": "KODEX 코스닥150레버리지",
    "091160": "KODEX 반도체", "219480": "KODEX 미국S&P500선물(H)", "KPI200": "코스피200 지수", "KOSDAQ": "코스닥 지수", "KOSPI": "코스피 지수",
    "102110": "TIGER 200", "360750": "TIGER 미국S&P500", "133690": "TIGER 미국나스닥100",
}


def fetch_fchart(symbol, count=8000):
    r = requests.get("https://fchart.stock.naver.com/sise.nhn",
                     params={"symbol": symbol, "timeframe": "day", "count": count, "requestType": 0}, timeout=30)
    r.raise_for_status()
    rows = [x.split("|") for x in re.findall(r'data="([^"]+)"', r.text)]
    if not rows:
        raise RuntimeError(f"fchart {symbol}: 빈 응답")
    df = pd.DataFrame(rows, columns=["date", "open", "high", "low", "close", "volume"])
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    df = df.set_index("date").astype(float)
    now = dt.datetime.now(KST)
    if now.hour < 16:                                   # 장중·마감 직후 당일 행 제외
        df = df[df.index.date < now.date()]
    return df


def fetch_sisejson(symbol, start="19900101"):
    end = dt.datetime.now(KST).strftime("%Y%m%d")
    r = requests.get("https://api.finance.naver.com/siseJson.naver",
                     params={"symbol": symbol, "requestType": 1, "startTime": start, "endTime": end, "timeframe": "day"},
                     headers={"User-Agent": "Mozilla/5.0"}, timeout=60)
    r.raise_for_status()
    rows = re.findall(r'\["(\d{8})",\s*([\d.]+),\s*([\d.]+),\s*([\d.]+),\s*([\d.]+),\s*([\d.]+)', r.text)
    if not rows:
        raise RuntimeError(f"siseJson {symbol}: 빈 응답")
    df = pd.DataFrame(rows, columns=["date", "open", "high", "low", "close", "volume"])
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    df = df.set_index("date").astype(float)
    now = dt.datetime.now(KST)
    if now.hour < 16:
        df = df[df.index.date < now.date()]
    return df


def price(symbol, refresh=False):
    """일봉 OHLCV. 종목코드(6자리 숫자)는 siseJson 수정주가, 지수 심볼은 fchart. 캐시는 통째로 교체
    (수정주가는 분배 때마다 과거값이 바뀌므로 병합하지 않는다)."""
    PX_DIR.mkdir(parents=True, exist_ok=True)
    p = PX_DIR / f"{symbol}.csv"
    if refresh or not p.exists():
        df = fetch_sisejson(symbol) if symbol.isdigit() else fetch_fchart(symbol)
        df.to_csv(p)
        return df
    return pd.read_csv(p, parse_dates=["date"], index_col="date")


def naver_investor(kind):
    """kind: 'fut' (코스피200 선물, 계약·금액) | 'kospi' (코스피 현물, 원)"""
    f = {"fut": "naver_fut_investor_daily.csv", "kospi": "naver_kospi_investor_daily.csv"}[kind]
    return pd.read_csv(RAW / f, parse_dates=["date"], index_col="date").sort_index()


def naver_program(market="KOSPI"):
    if market == "KOSPI":
        return pd.read_csv(RAW / "naver_program_daily.csv", parse_dates=["date"], index_col="date").sort_index()
    return _collected(f"naver_{market.lower()}_program_daily", lambda: _program(market))


def naver_kosdaq_investor():
    return _collected("naver_kosdaq_investor_daily", lambda: _cn().investor_daily("KOSDAQ"))


def _cn():
    """수집기 폴더의 collect_naver.py 수집 함수를 그대로 쓴다(엔드포인트·페이지 규약 공유)."""
    sys.path.insert(0, str(COLLECTORS))
    import collect_naver
    return collect_naver


def _program(market):
    cn = _cn()
    rows = cn.pages("/api/domestic/market/trendProgram", {"tradeType": "KRX", "krxMarketType": market,
                                                          "bizdate": dt.date.today().strftime("%Y%m%d"), "periodType": "DAY"})
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["bizdate"], format="%Y%m%d")
    df = df.drop(columns=["bizdate"]).drop_duplicates("date").sort_values("date").set_index("date")
    df = df.apply(pd.to_numeric, errors="coerce")
    return df.rename(columns={"diffBuyAmt": "arb_buy", "diffSellAmt": "arb_sell", "diffPureBuyAmt": "arb_net", "biDiffBuyAmt": "nonarb_buy",
                              "biDiffSellAmt": "nonarb_sell", "biDiffPureBuyAmt": "nonarb_net", "totalDiffBuyAmt": "tot_buy",
                              "totalDiffSellAmt": "tot_sell", "totalDiffPureBuyAmt": "tot_net"})


def _collected(name, fetch):
    """원자료 폴더에 없으면 수집해 저장한다. 16:00 KST 이전 수집이면 당일 행을 버린다."""
    p = RAW / f"{name}.csv"
    if not p.exists():
        df = fetch()
        now = dt.datetime.now(KST)
        if now.hour < 16:
            df = df[df.index.date < now.date()]
        df.to_csv(p, encoding="utf-8-sig")
        with open(RAW / "_collected_on.txt", "a", encoding="utf-8") as f:
            f.write(f"{dt.date.today()} overnight_lab/data.py {name} rows={len(df)} {df.index.min().date()}~{df.index.max().date()}\n")
    return pd.read_csv(p, parse_dates=["date"], index_col="date").sort_index()


def us_daily(name):
    """name: es | nq | spx | ndx_comp (미국 현지 날짜 기준 일봉)"""
    return pd.read_csv(RAW / f"{name}_daily.csv", parse_dates=["date"], index_col="date").sort_index()


def krx_options():
    return pd.read_csv(RAW / "krx_opt_investor_daily.csv", parse_dates=["date"])


def result_file(rel):
    """<OUT>/rel 이 없으면 저장소에 올린 결과 폴더(../results/2026-09-29-overnight/rel)에서 찾는다(사전 고정 사양 등 읽기용)."""
    cands = [OUT / rel, (PKG / ".." / "results" / "2026-09-29-overnight" / rel).resolve()]
    return next((p for p in cands if p.exists()), cands[0])


def spec_file(family):
    """2차 사전 고정 사양 위치: 출력 폴더(<OUT>/round2/<family>/spec.json)에 없으면 저장소에 올린 결과 폴더에서 찾는다."""
    cands = [OUT / "round2" / family / "spec.json",
             (PKG / ".." / "results" / "2026-09-29-overnight" / "round2" / family / "spec.json").resolve()]
    return next((p for p in cands if p.exists()), cands[0])
