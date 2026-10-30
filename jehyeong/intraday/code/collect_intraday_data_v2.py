#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Collect ONLY the KOSPI 200 cash index DAILY OHLCV (not KOSPI, ETFs or futures).

Default dates, inclusive: 2026-09-22 through 2026-10-13 (KRX closing prices).
Source: Korea Investment & Securities (KIS) domestic index daily-chart API.
  TR: FHKUP03500100, FID_COND_MRKT_DIV_CODE=U, FID_INPUT_ISCD=2001.

No account, holdings, order, or trading APIs are called.

Install: pip install pandas requests python-dotenv
Run:
    python collect_intraday_data.py --dry-run
    python collect_intraday_data.py --test-run       # offline; no credentials
    python collect_intraday_data.py                  # live KOSPI 200 daily only
    python collect_intraday_data.py --prepare-only   # rebuild CSV from saved raw pages

Env: KIS_APP_KEY and KIS_APP_SECRET, from environment or
     ~/Desktop/ETF_iNAV/.env (environment values take precedence).
Output: ../data/KOSPI200_daily.csv when script is in intraday/code/;
        otherwise ./data/KOSPI200_daily.csv next to this script.
Raw index responses are saved without any auth/token/credential data.
"""
from __future__ import annotations

import argparse
import json
import os
import secrets
import tempfile
import time
from pathlib import Path
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from unittest.mock import patch

import pandas as pd
import requests

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent if SCRIPT_DIR.name.lower() == "code" else SCRIPT_DIR
DATA_DIR = PROJECT_DIR / "data"
RAW_DIR = DATA_DIR / "raw" / "KIS" / "KOSPI200" / "daily"
OUT_CSV = DATA_DIR / "KOSPI200_daily.csv"
START_DEFAULT = "2000-09-22"
END_DEFAULT = "2026-10-13"
KST = ZoneInfo("Asia/Seoul")
KIS_BASE = "https://openapi.koreainvestment.com:9443"
INDEX_CODE = "2001"  # KOSPI 200 cash index, NOT 0001 (KOSPI composite)
TR_ID = "FHKUP03500100"
CHART_PATH = "/uapi/domestic-stock/v1/quotations/inquire-daily-indexchartprice"
COLS = ["date", "open", "high", "low", "close", "volume", "trading_value", "symbol", "source"]


def atomic_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{secrets.token_hex(4)}.tmp")
    try:
        with tmp.open("xb") as file:
            file.write(value)
            file.flush()
            os.fsync(file.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def save_json(path: Path, data: dict) -> None:
    atomic_bytes(path, json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8"))


def parse_index_rows(rows: list[dict], start: str, end: str) -> pd.DataFrame:
    """KIS output2 -> exact daily index OHLCV. Reject malformed candles."""
    if not rows:
        return pd.DataFrame(columns=COLS)
    df = pd.DataFrame(rows)
    fields = {
        "stck_bsop_date": "date",
        "bstp_nmix_oprc": "open",
        "bstp_nmix_hgpr": "high",
        "bstp_nmix_lwpr": "low",
        "bstp_nmix_prpr": "close",
        "acml_vol": "volume",
    }
    missing = [k for k in fields if k not in df.columns]
    if missing:
        raise ValueError(f"Missing KIS index response fields: {missing}")
    df = df.rename(columns=fields)
    dates = pd.to_datetime(df["date"], format="%Y%m%d", errors="raise")
    df["date"] = dates.dt.strftime("%Y-%m-%d")
    for field in ("open", "high", "low", "close", "volume"):
        df[field] = pd.to_numeric(
            df[field].astype(str).str.replace(",", "", regex=False), errors="raise"
        )
    # KIS acml_tr_pbmn is cumulative trading value in KRW if supplied.
    if "acml_tr_pbmn" in df.columns:
        df["trading_value"] = pd.to_numeric(
            df["acml_tr_pbmn"].astype(str).str.replace(",", "", regex=False), errors="coerce"
        )
    else:
        df["trading_value"] = float("nan")
    price = df[["open", "high", "low", "close"]]
    if (price <= 0).any().any() or price.isna().any().any():
        raise ValueError("Invalid KOSPI200 index price")
    if (df.volume < 0).any() or df.volume.isna().any():
        raise ValueError("Invalid KOSPI200 index volume")
    if (df.high < price[["open", "low", "close"]].max(axis=1)).any():
        raise ValueError("High is below open/low/close")
    if (df.low > price[["open", "close"]].min(axis=1)).any():
        raise ValueError("Low is above open/close")
    df["symbol"] = "KOSPI200"
    df["source"] = "KIS"
    df = df.loc[df["date"].between(start, end), COLS].copy()
    if df.duplicated("date").any():
        conflicts = df.groupby("date")[["open", "high", "low", "close", "volume"]].nunique()
        if conflicts.gt(1).any().any():
            raise ValueError("Conflicting duplicate daily index candles")
        df = df.drop_duplicates("date")
    return df.sort_values("date").reset_index(drop=True)


def load_saved(start: str, end: str) -> pd.DataFrame:
    rows = []
    # Use ONLY raw KOSPI200 index pages, never stale KOSPI or ETF data.
    for path in sorted(RAW_DIR.glob("page_*.json")):
        saved = json.loads(path.read_text(encoding="utf-8"))
        if saved.get("index_code") != INDEX_CODE or saved.get("tr_id") != TR_ID:
            raise ValueError(f"Unexpected raw archive identity in {path.name}")
        rows.extend(saved["output2"])
    return parse_index_rows(rows, start, end)


def request_with_retry(session, method, url, *, headers=None, params=None, data=None):
    for attempt in range(6):
        try:
            resp = session.request(method, url, headers=headers, params=params,
                                   json=data, timeout=30)
        except requests.RequestException as exc:
            if attempt == 5:
                raise RuntimeError("KIS network request failed after retries") from exc
            time.sleep(min(2 ** attempt, 20))
            continue
        if resp.status_code == 429 or resp.status_code >= 500:
            if attempt == 5:
                raise RuntimeError(f"KIS HTTP {resp.status_code} after retries")
            time.sleep(min(2 ** attempt, 20))
            continue
        if resp.status_code >= 400:
            # Do not log URLs/responses: may contain sensitive diagnostics.
            raise RuntimeError(f"KIS HTTP {resp.status_code}")
        result = resp.json()
        if str(result.get("rt_cd", "0")) == "0":
            return result
        if result.get("msg_cd") == "EGW00201" or result.get("msg_cd") in {"1700", "8005"}:
            if attempt == 5:
                raise RuntimeError("KIS API rate-limit retries exhausted")
            time.sleep(min(2 ** attempt, 20))
            continue
        raise RuntimeError(f"KIS API response code {result.get('msg_cd', 'UNKNOWN')}")
    raise RuntimeError("KIS request retries exhausted")


def fetch_kis_index(start: str, end: str, appkey: str, secret: str) -> pd.DataFrame:
    """Index-specific daily TR, request pagination by decreasing end date."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    with requests.Session() as session:
        auth = request_with_retry(session, "POST", KIS_BASE + "/oauth2/tokenP", data={
            "grant_type": "client_credentials", "appkey": appkey, "appsecret": secret,
        })
        token = auth.get("access_token")
        if not token:
            raise RuntimeError("KIS auth succeeded without access_token")
        headers = {"authorization": f"Bearer {token}", "appkey": appkey,
                   "appsecret": secret, "tr_id": TR_ID, "custtype": "P"}
        cursor_end = end
        seen = set()
        raw_all = []
        for page_no in range(100):  # safety cap; 2026-09-22..2026-10-13 needs ~1 page
            params = {
                "FID_COND_MRKT_DIV_CODE": "U",
                "FID_INPUT_ISCD": INDEX_CODE,
                "FID_INPUT_DATE_1": start.replace("-", ""),
                "FID_INPUT_DATE_2": cursor_end.replace("-", ""),
                "FID_PERIOD_DIV_CODE": "D",
            }
            obj = request_with_retry(session, "GET", KIS_BASE + CHART_PATH,
                                     headers=headers, params=params)
            batch = obj.get("output2") or []
            if not isinstance(batch, list):
                raise ValueError("KIS output2 must be an array")
            if not batch:
                break
            dates = [str(x.get("stck_bsop_date", "")) for x in batch]
            if not all(len(d) == 8 and d.isdigit() for d in dates):
                raise ValueError("Malformed daily dates in KIS output2")
            oldest = min(dates)
            if oldest in seen:
                raise RuntimeError("Repeated index-history cursor; aborting")
            seen.add(oldest)
            # No token, headers, or credentials are included in the raw archive.
            archive = {"index_code": INDEX_CODE, "tr_id": TR_ID,
                       "requested_start": start, "requested_end": cursor_end,
                       "output2": batch}
            save_json(RAW_DIR / f"page_{page_no:03d}.json", archive)
            raw_all.extend(batch)
            if oldest <= start.replace("-", ""):
                break
            new_cursor = (datetime.strptime(oldest, "%Y%m%d") - timedelta(days=1)).strftime("%Y-%m-%d")
            if new_cursor >= cursor_end:
                raise RuntimeError("KIS pagination not moving backwards")
            cursor_end = new_cursor
            time.sleep(0.3)
        else:
            raise RuntimeError("100-page safety limit; collection may be incomplete")
    return parse_index_rows(raw_all, start, end)


def save_output(df: pd.DataFrame, start: str, end: str, *, source_mode: str) -> None:
    if df.empty:
        raise RuntimeError("KOSPI200 index history is empty; no CSV overwritten")
    atomic_bytes(OUT_CSV, df.to_csv(index=False).encode("utf-8-sig"))
    meta = {"symbol": "KOSPI200", "name": "KOSPI 200 cash index",
            "source": "KIS", "index_code": INDEX_CODE, "interval": "1d",
            "requested_start": start, "requested_end": end, "first_date": df.date.iloc[0],
            "last_date": df.date.iloc[-1], "rows": len(df), "mode": source_mode,
            "created_kst": datetime.now(KST).isoformat()}
    save_json(DATA_DIR / "KOSPI200_manifest.json", meta)
    print(f"KOSPI200 daily saved: {OUT_CSV} ({len(df)} rows, {df.date.iloc[0]}..{df.date.iloc[-1]})")
    if df.date.iloc[0] > start or df.date.iloc[-1] < end:
        print("[WARNING] One or both boundary dates have no available index bar; check market calendar/source availability.")


def test_run():
    # Entirely offline; do not load .env and do not use network.
    sample = [
        {"stck_bsop_date": "20260922", "bstp_nmix_oprc": "400.00", "bstp_nmix_hgpr": "404.00",
         "bstp_nmix_lwpr": "397.50", "bstp_nmix_prpr": "403.00", "acml_vol": "1000", "acml_tr_pbmn": "250000"},
        {"stck_bsop_date": "20261013", "bstp_nmix_oprc": "420.00", "bstp_nmix_hgpr": "423.00",
         "bstp_nmix_lwpr": "418.00", "bstp_nmix_prpr": "421.00", "acml_vol": "1800"},
    ]
    df = parse_index_rows(sample, START_DEFAULT, END_DEFAULT)
    assert len(df) == 2 and df.date.tolist() == [START_DEFAULT, END_DEFAULT]
    assert df.iloc[0]["close"] == 403.0 and df.iloc[-1]["close"] == 421.0
    assert df["symbol"].eq("KOSPI200").all() and df["source"].eq("KIS").all()
    assert df.loc[1, "trading_value"] != df.loc[1, "trading_value"]  # absent -> NaN
    try:
        parse_index_rows([dict(sample[0], bstp_nmix_hgpr="390.00")], START_DEFAULT, END_DEFAULT)
    except ValueError:
        pass
    else:
        raise AssertionError("Invalid OHLC was accepted")
    # Mock both API requests: validate that ONLY the index endpoint is queried.
    global RAW_DIR
    real_raw_dir = RAW_DIR
    try:
        with tempfile.TemporaryDirectory() as tmp:
            RAW_DIR = Path(tmp) / "KOSPI200" / "daily"
            seen = []

            def mock_request(session, method, url, **kwargs):
                seen.append((method, url, kwargs))
                if url.endswith("/oauth2/tokenP"):
                    assert method == "POST"
                    return {"access_token": "offline-test-token"}
                assert url.endswith(CHART_PATH) and method == "GET"
                assert kwargs["headers"]["tr_id"] == "FHKUP03500100"
                params = kwargs["params"]
                assert params["FID_COND_MRKT_DIV_CODE"] == "U"
                assert params["FID_INPUT_ISCD"] == "2001"
                assert params["FID_PERIOD_DIV_CODE"] == "D"
                return {"rt_cd": "0", "output2": sample}

            with patch(__name__ + ".request_with_retry", side_effect=mock_request):
                received = fetch_kis_index(START_DEFAULT, END_DEFAULT, "TEST_KEY", "TEST_SECRET")
            assert len(seen) == 2 and len(received) == 2
            recovered = load_saved(START_DEFAULT, END_DEFAULT)
            assert recovered.equals(received)
    finally:
        RAW_DIR = real_raw_dir
    print("PASS: offline KIS KOSPI200-only endpoint, parser, dates, OHLCV validation, raw reconstruction; network_calls=0")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--start", default=START_DEFAULT)
    p.add_argument("--end", default=END_DEFAULT)
    p.add_argument("--env-file", type=Path, default=Path.home() / "Desktop" / "ETF_iNAV" / ".env")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--test-run", action="store_true")
    p.add_argument("--prepare-only", action="store_true")
    args = p.parse_args()
    if args.test_run:
        test_run()
        return
    try:
        first, last = pd.Timestamp(args.start), pd.Timestamp(args.end)
    except (ValueError, TypeError):
        p.error("Dates must be YYYY-MM-DD")
    if first > last:
        p.error("--start must be <= --end")
    if args.dry_run:
        print(json.dumps({"mode": "DRY_RUN", "network_calls": 0, "symbol": "KOSPI200",
                          "interval": "1d", "source": "KIS", "index_code": INDEX_CODE,
                          "start": args.start, "end": args.end, "output": str(OUT_CSV)},
                         ensure_ascii=False, indent=2))
        return
    if args.prepare_only:
        df = load_saved(args.start, args.end)
        save_output(df, args.start, args.end, source_mode="PREPARE_ONLY")
        return
    if args.env_file.is_file():
        try:
            from dotenv import load_dotenv
        except ImportError:
            p.error("Install python-dotenv to read --env-file")
        load_dotenv(args.env_file, override=False)
    key, secret = os.getenv("KIS_APP_KEY"), os.getenv("KIS_APP_SECRET")
    if not key or not secret:
        p.error("Missing KIS_APP_KEY / KIS_APP_SECRET in environment or --env-file")
    df = fetch_kis_index(args.start, args.end, key, secret)
    save_output(df, args.start, args.end, source_mode="KIS_LIVE")


if __name__ == "__main__":
    main()
