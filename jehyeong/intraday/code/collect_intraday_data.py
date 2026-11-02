"""Collect maximum available KRX ETF history. No account/order APIs.

Usage (from any working directory)::
    python collect_intraday_data.py --dry-run
    python collect_intraday_data.py --test-run       # offline, no credentials
    python collect_intraday_data.py                 # user starts real collection
    python collect_intraday_data.py --prepare-only  # rebuild from saved pages

Dependencies: pandas numpy requests python-dotenv (pip install ...).
Default .env: sibling Desktop/ETF_iNAV/.env; environment variables take precedence.
Optional --env-file can point to another existing .env. Credentials required:
TOSS_CLIENT_ID/TOSS_CLIENT_SECRET, KIS_APP_KEY/KIS_APP_SECRET,
KIWOOM_APP_KEY/KIWOOM_APP_SECRET. Default collection is Toss only. Set
KIS_KIWOOM=True below to include Kiwoom 30m and Kiwoom/KIS 1m as supplements.
No credentials or authentication responses are written to disk.
If Kiwoom keys are absent there, read ONLY Kiwoom keys from Desktop/VibeTrading/.env
(the same configuration location used by the original Kiwoom research collector).

Adapted from ETF_iNAV/intraday_research/code/collect_multiasset_1m.py,
collect_kiwoom_research.py and overnight_research/code/kodex.py.
Toss index 1m (KOSPI/KOSDAQ) comes from market-indicators candles; no 30m is aggregated from 1m.
Kiwoom requests native 30m and 1m only when KIS_KIWOOM is enabled.
KIS uses stock historical minute TR FHKST03010230 (not futures TR).
Official KIS example documents a maximum one-year minute retention; every daily
calendar date in the last 366 days is probed, with no empty-day early cutoff.
--kis-lookback-days can extend that probe if the provider changes its retention.

Timestamps: canonical 30m timestamp = decision/end time. TOSS API minute labels
are bar-start, but canonical output convention is per symbol: KOSPI index is
converted to bar-end, while TOSS ETF labels remain bar-start, consistent with
empirical cross-instrument alignment. KIS minute and Kiwoom bar timestamps use
provider bar-start labels. Provider prices are never cross-validated.
15:20 means last continuous minute CLOSE at/before 15:20, never auction close.
Missing minute bars are NOT forward-filled.

All output stays in jehyeong/intraday/data or results; raw pages/checkpoints allow
same-query resume. --max-pages 0 (default) has NO page cap. A cap/error/stalled
cursor is explicitly reported and is never called a provider retention limit.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import secrets
import time
import tempfile
from unittest.mock import patch

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
DESKTOP = ROOT.parents[2]
ENV_DEFAULT = DESKTOP / "ETF_iNAV" / ".env"
KST = "Asia/Seoul"
UNIVERSE = {
    "KOSPI": "코스피200"
}
TIMES = pd.date_range("2000-01-01 09:30", "2000-01-01 15:00", freq="30min").strftime("%H:%M").tolist()
FIELDS = ["open", "high", "low", "close", "volume"]
BASES = {"TOSS": "https://openapi.tossinvest.com",
         "KIS": "https://openapi.koreainvestment.com:9443",
         "KIWOOM": "https://api.kiwoom.com"}
KEYS = {"TOSS": ("TOSS_CLIENT_ID", "TOSS_CLIENT_SECRET"),
        "KIS": ("KIS_APP_KEY", "KIS_APP_SECRET"),
        "KIWOOM": ("KIWOOM_APP_KEY", "KIWOOM_APP_SECRET")}
# False: Toss only. True: also collect Kiwoom native 30m and Kiwoom/KIS 1m.
KIS_KIWOOM = False


def save_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(obj, ensure_ascii=False, indent=2, default=str)
    # Windows Defender/search indexing can briefly hold the previous file.
    for attempt in range(7):
        tmp = path.with_name(f"{path.name}.{os.getpid()}.{secrets.token_hex(4)}.tmp")
        try:
            with tmp.open("x", encoding="utf-8", newline="\n") as f:
                f.write(payload)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)
            return path
        except PermissionError:
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass
            if attempt < 6:
                time.sleep(min(.1 * (2 ** attempt), 2.0))
    if path.name == "checkpoint.json":
        # If Windows keeps the canonical name locked, preserve this revision
        # in an immutable file. The reader below resumes from the newest copy.
        revision = int(obj.get("checkpoint_revision", time.time_ns()))
        fallback = path.with_name(f"checkpoint.recovery.{revision}.{secrets.token_hex(2)}.json")
        with fallback.open("x", encoding="utf-8", newline="\n") as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        print(f"[CHECKPOINT RECOVERY] progress saved to {fallback.name}", flush=True)
        return fallback
    raise PermissionError(f"Could not atomically save {path.name} after retries")


def read_checkpoint(path):
    path = Path(path)
    recovery = list(path.parent.glob("checkpoint.recovery.*.json"))
    recovery.sort(key=lambda p: int(p.name.split(".")[2]) if p.name.split(".")[2].isdigit() else 0)
    candidates = [path, *recovery[-1:]]
    available = []
    for candidate in candidates:
        try:
            value = json.loads(candidate.read_text(encoding="utf-8"))
            available.append((int(value.get("checkpoint_revision", 0)), candidate.stat().st_mtime_ns, value))
        except (OSError, ValueError, json.JSONDecodeError):
            continue
    return max(available, key=lambda item: (item[0], item[1]))[2] if available else None


def save_checkpoint(path, state):
    state["checkpoint_revision"] = time.time_ns()
    return save_json(path, state)


def csv(frame, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(7):
        tmp = path.with_name(f"{path.name}.{os.getpid()}.{secrets.token_hex(4)}.tmp")
        try:
            frame.to_csv(tmp, index=False, encoding="utf-8-sig")
            os.replace(tmp, path)
            return
        except PermissionError:
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass
            if attempt < 6:
                time.sleep(min(.1 * (2 ** attempt), 2.0))
    raise PermissionError(f"Could not atomically save {path.name} after retries")


def ts_kst(values):
    return pd.to_datetime(values, utc=True).dt.tz_convert(KST).dt.tz_localize(None)


def normalize(rows, provider, kind, symbol):
    """Keep raw timestamps; never infer trading value from cumulative turnover."""
    if not rows:
        return pd.DataFrame(columns=["timestamp", "date", *FIELDS, "trading_value", "symbol", "source"])
    x = pd.DataFrame(rows)
    if provider == "TOSS":
        x = x.rename(columns={"openPrice": "open", "highPrice": "high", "lowPrice": "low", "closePrice": "close", "tradingValue": "trading_value"})
        x["timestamp"] = ts_kst(x.timestamp)
    elif provider == "KIWOOM":
        key, fmt = ("dt", "%Y%m%d") if kind == "daily" else ("cntr_tm", "%Y%m%d%H%M%S")
        x["timestamp"] = pd.to_datetime(x[key], format=fmt, errors="raise")
        x = x.rename(columns={"cur_prc": "close", "open_pric": "open", "high_pric": "high", "low_pric": "low", "trde_qty": "volume"})
    else:
        dates = x.stck_bsop_date.astype(str)
        x["timestamp"] = pd.to_datetime(dates if kind == "daily" else dates + x.stck_cntg_hour.astype(str).str.zfill(6), format="%Y%m%d" if kind == "daily" else "%Y%m%d%H%M%S")
        x = x.rename(columns={"stck_oprc": "open", "stck_hgpr": "high", "stck_lwpr": "low", "stck_clpr": "close", "stck_prpr": "close", "acml_vol": "volume"} if kind == "daily" else {"stck_oprc": "open", "stck_hgpr": "high", "stck_lwpr": "low", "stck_prpr": "close", "cntg_vol": "volume"})
    for field in FIELDS:
        x[field] = pd.to_numeric(x[field].astype(str).str.replace(",", "", regex=False), errors="raise")
        if provider == "KIWOOM" and field != "volume":
            x[field] = x[field].abs()  # Kiwoom price sign means change, not side.
    if "trading_value" not in x:
        x["trading_value"] = np.nan
    else:
        x["trading_value"] = pd.to_numeric(x.trading_value, errors="raise")
    x["date"] = x.timestamp.dt.strftime("%Y-%m-%d")
    x["symbol"], x["source"] = symbol, provider
    x = x[["timestamp", "date", *FIELDS, "trading_value", "symbol", "source"]]
    validate(x)
    return x


def validate(x):
    if x.empty:
        return
    if x.timestamp.isna().any() or not np.isfinite(x[FIELDS].to_numpy(dtype=float)).all():
        raise ValueError("Invalid timestamps/nonfinite OHLCV")
    if (x[["open", "high", "low", "close"]] <= 0).any().any() or (x.volume < 0).any():
        raise ValueError("Nonpositive prices or negative volume")
    if (x.high < x[["open", "low", "close"]].max(axis=1)).any() or (x.low > x[["open", "close"]].min(axis=1)).any():
        raise ValueError("Invalid OHLC range")


class API:
    def __init__(self, provider, gap=.25):
        self.provider, self.base = provider, BASES[provider]
        self.session, self.token, self.last, self.gap = requests.Session(), None, 0., gap

    def auth(self):
        a, b = (os.environ[k] for k in KEYS[self.provider])
        kwargs = {"data": {"grant_type": "client_credentials", "client_id": a, "client_secret": b}} if self.provider == "TOSS" else {"json": {"grant_type": "client_credentials", **({"appkey": a, "appsecret": b} if self.provider == "KIS" else {"appkey": a, "secretkey": b})}}
        path = "/oauth2/tokenP" if self.provider == "KIS" else "/oauth2/token"
        for attempt in range(6):
            try:
                r = self.session.post(self.base + path, timeout=30, **kwargs)
                if r.status_code == 429 or r.status_code >= 500:
                    time.sleep(min(2 ** attempt, 30)); continue
                if r.status_code >= 400:
                    raise RuntimeError(f"{self.provider} authentication HTTP {r.status_code}")
                body = r.json()
                self.token = body.get("access_token") or body.get("accessToken") or body.get("token")
                if not self.token:
                    raise RuntimeError(f"{self.provider} token missing")
                return
            except requests.RequestException:
                if attempt == 5:
                    raise RuntimeError(f"{self.provider} auth network retries exhausted") from None
                time.sleep(min(2 ** attempt, 30))
        raise RuntimeError(f"{self.provider} auth retries exhausted")

    def request(self, path, *, params=None, body=None, api_id=None, cursor=None):
        if not self.token:
            self.auth()
        for attempt in range(8):
            time.sleep(max(0., self.gap - (time.monotonic() - self.last)))
            headers = {"Authorization": f"Bearer {self.token}"}
            if self.provider == "KIS":
                headers.update(appkey=os.environ[KEYS["KIS"][0]], appsecret=os.environ[KEYS["KIS"][1]], tr_id=api_id, custtype="P")
            if self.provider == "KIWOOM":
                headers.update({"api-id": api_id, "cont-yn": "Y" if cursor else "N", "next-key": cursor or "", "Content-Type": "application/json;charset=UTF-8"})
            try:
                self.last = time.monotonic()
                r = self.session.request("POST" if body is not None else "GET", self.base + path, params=params, json=body, headers=headers, timeout=40)
                if r.status_code == 401 and attempt < 2:
                    self.auth(); continue
                if r.status_code == 429 or r.status_code >= 500:
                    time.sleep(min(float(r.headers.get("Retry-After", 2 ** attempt)), 45)); continue
                if r.status_code >= 400:
                    raise RuntimeError(f"{self.provider} {api_id or 'candles'} HTTP {r.status_code}")
                value = r.json()
                code = str(value.get("return_code", value.get("rt_cd", "0")))
                if code in {"1700", "8005"} or value.get("msg_cd") == "EGW00201":
                    time.sleep(min(2 ** attempt, 30)); continue
                if code != "0" or value.get("error"):
                    raise RuntimeError(f"{self.provider} {api_id or 'candles'} API error code {code}")
                return value, r.headers
            except requests.RequestException:
                if attempt == 7:
                    raise RuntimeError(f"{self.provider} network retries exhausted") from None
                time.sleep(min(2 ** attempt, 30))
        raise RuntimeError(f"{self.provider} retries exhausted")

    def page(self, kind, symbol, cursor, end, dates):
        """Return raw rows, next cursor, terminal reason. Cursors are market data."""
        if self.provider == "TOSS":
            payload, _ = self.request(f"/api/v1/market-indicators/{symbol}/candles", params={"interval": "1d" if kind == "daily" else "1m", "count": 200, "before": cursor or end + "T23:59:59+09:00"})
            result = payload.get("result", payload)
            nxt = result.get("nextBefore")
            return result.get("candles", []), nxt, "SERVER_END" if not nxt else None
        if self.provider == "KIWOOM":
            body = {"stk_cd": symbol, "upd_stkpc_tp": "0"}
            if kind == "daily":
                body["base_dt"] = end.replace("-", "")
                api_id, key = "ka10081", "stk_dt_pole_chart_qry"
            else:
                body["tic_scope"] = "30" if kind == "30m" else "1"
                # ka10080 requires a base date. Without it the chart request
                # silently defaults to the server's current reference date,
                # which can make an apparently completed continuation chain
                # stop at a shorter rolling window than the requested query.
                body["base_dt"] = end.replace("-", "")
                api_id, key = "ka10080", "stk_min_pole_chart_qry"
            payload, h = self.request("/api/dostk/chart", body=body, api_id=api_id, cursor=cursor)
            nxt = h.get("next-key") if h.get("cont-yn") == "Y" else None
            return payload.get(key, []), nxt, "SERVER_END" if not nxt else None
        if kind == "daily":
            end_cursor = pd.Timestamp(cursor or end)
            payload, _ = self.request("/uapi/domestic-stock/v1/quotations/inquire-daily-itemchartprice", api_id="FHKST03010100", params={"FID_COND_MRKT_DIV_CODE": "J", "FID_INPUT_ISCD": symbol, "FID_INPUT_DATE_1": (end_cursor - pd.Timedelta(days=180)).strftime("%Y%m%d"), "FID_INPUT_DATE_2": end_cursor.strftime("%Y%m%d"), "FID_PERIOD_DIV_CODE": "D", "FID_ORG_ADJ_PRC": "1"})
            rows = payload.get("output2", [])
            if not rows:
                return [], None, "SERVER_EMPTY"
            oldest = min(r["stck_bsop_date"] for r in rows)
            return rows, (pd.Timestamp(oldest) - pd.Timedelta(days=1)).strftime("%Y-%m-%d"), None
        if not dates:
            return [], None, "NO_DAILY_CALENDAR"
        state = cursor or {"index": 0, "clock": "153000"}
        day = dates[state["index"]]
        payload, _ = self.request("/uapi/domestic-stock/v1/quotations/inquire-time-dailychartprice", api_id="FHKST03010230", params={"FID_COND_MRKT_DIV_CODE": "J", "FID_INPUT_ISCD": symbol, "FID_INPUT_DATE_1": day.replace("-", ""), "FID_INPUT_HOUR_1": state["clock"], "FID_PW_DATA_INCU_YN": "N", "FID_FAKE_TICK_INCU_YN": ""})
        rows = [r for r in payload.get("output2", []) if r.get("stck_bsop_date") == day.replace("-", "")]
        if rows:
            oldest = min(r["stck_cntg_hour"] for r in rows)
            if oldest > "090000":
                nxtclock = (pd.Timestamp(day + " " + oldest) - pd.Timedelta(minutes=1)).strftime("%H%M%S")
                return rows, {"index": state["index"], "clock": nxtclock}, None
        nxt = state["index"] + 1
        return rows, {"index": nxt, "clock": "153000"} if nxt < len(dates) else None, "CALENDAR_PROBE_END" if nxt == len(dates) else None


def collect_stream(api, kind, symbol, args, dates=None):
    query = {"provider": api.provider, "kind": kind, "symbol": symbol, "start": args.start, "end": args.end, "kis_lookback_days": args.kis_lookback_days, "adjusted": False, "toss_endpoint": "market-indicators"}
    # Invalidate prior Kiwoom intraday checkpoints because older versions
    # omitted ka10080's required base_dt; reuse of a SERVER_END checkpoint
    # would otherwise prevent the corrected request from ever being sent.
    if api.provider == "KIWOOM" and kind != "daily":
        query["kiwoom_base_dt"] = args.end.replace("-", "")
    jobid = hashlib.sha256(json.dumps(query, sort_keys=True).encode()).hexdigest()[:16]
    dest = DATA / "raw" / api.provider / symbol / kind / jobid
    checkpoint = dest / "checkpoint.json"
    # Reuse completed Toss maximum-history streams across later --end dates.
    # If the query end moved forward, fetch only until new pages overlap the
    # completed history, then combine both sets of raw pages.
    prior_complete_latest = None
    if api.provider == "TOSS" and kind == "1m":
        terminal_status = {"SERVER_END", "START_REACHED", "PRIOR_COMPLETE_HISTORY_OVERLAP"}
        for old_job in (DATA / "raw" / api.provider / symbol / kind).glob("*"):
            if old_job == dest:
                continue
            old_state = read_checkpoint(old_job / "checkpoint.json")
            if old_state and old_state.get("status") in terminal_status and old_state.get("latest"):
                old_latest = pd.Timestamp(old_state["latest"])
                prior_complete_latest = max(prior_complete_latest, old_latest) if prior_complete_latest is not None else old_latest
        if prior_complete_latest is not None and prior_complete_latest.date() >= pd.Timestamp(args.end).date():
            frame = load_saved_stream(api.provider, kind, symbol, args.start, args.end)
            return frame, {"provider": api.provider, "kind": kind, "symbol": symbol, "start": args.start, "end": args.end, "status": "CACHED_PRIOR_SERVER_END", "earliest": str(frame.timestamp.min()) if not frame.empty else None, "latest": str(frame.timestamp.max()) if not frame.empty else None, "pages": 0, "rows": len(frame)}
    state = read_checkpoint(checkpoint) or {**query, "cursor": None, "pages": 0, "rows": 0, "status": "RUNNING", "fingerprints": []}
    terminal = {"SERVER_END", "SERVER_EMPTY", "START_REACHED", "CALENDAR_PROBE_END", "NO_DAILY_CALENDAR", "PRIOR_COMPLETE_HISTORY_OVERLAP"}
    if state["status"] not in terminal:
        seen = set(state["fingerprints"])
        try:
            consumed = 0
            while not args.max_pages or consumed < args.max_pages:
                raw, nxt, stop = api.page(kind, symbol, state["cursor"], args.end, dates or [])
                x = normalize(raw, api.provider, kind, symbol)
                fingerprint = hashlib.sha256(json.dumps(raw, sort_keys=True).encode()).hexdigest()
                if raw and fingerprint in seen:
                    raise RuntimeError("Repeated response page; not a history retention limit")
                if raw and nxt == state["cursor"] and not stop:
                    raise RuntimeError("Stalled cursor; not a history retention limit")
                if raw and state.get("last_oldest") and x.timestamp.min() >= pd.Timestamp(state["last_oldest"]):
                    raise RuntimeError("Historical timestamps did not advance backwards")
                if raw:
                    seen.add(fingerprint)
                    dest.mkdir(parents=True, exist_ok=True)
                    with gzip.open(dest / f"page_{state['pages']:07d}.json.gz", "wt", encoding="utf-8") as f:
                        json.dump(raw, f, ensure_ascii=False)
                    state["pages"] += 1
                    state["rows"] += len(x)
                    state["earliest"] = min(str(x.timestamp.min()), state.get("earliest", str(x.timestamp.min())))
                    state["latest"] = max(str(x.timestamp.max()), state.get("latest", str(x.timestamp.max())))
                    state["last_oldest"] = str(x.timestamp.min())
                    if api.provider != "KIS" or kind == "daily":
                        if x.date.min() <= args.start:
                            stop = "START_REACHED"
                    if prior_complete_latest is not None and x.timestamp.min() <= prior_complete_latest:
                        stop = "PRIOR_COMPLETE_HISTORY_OVERLAP"
                elif api.provider != "KIS" or kind == "daily":
                    stop = stop or "SERVER_EMPTY"
                recent = (state["fingerprints"] + [fingerprint])[-16:] if raw else state["fingerprints"]
                state.update(cursor=nxt, status=stop or "RUNNING", fingerprints=recent)
                save_checkpoint(checkpoint, state)
                consumed += 1
                if consumed == 1 or consumed % 100 == 0 or stop:
                    print(f"[{api.provider} {symbol} {kind}] pages={state['pages']} rows={state['rows']} status={state['status']}", flush=True)
                if stop:
                    break
            else:
                state["status"] = "CLIENT_PAGE_CAP"
        except Exception as exc:
            state["status"] = "ERROR"
            state["error_type"] = type(exc).__name__
            # Do not print request/response/URL/secret text from exceptions.
            detail = f": {exc}" if isinstance(exc, RuntimeError) else ""
            print(f"[{api.provider} {symbol} {kind}] ERROR: {type(exc).__name__}{detail}; saved pages can resume", flush=True)
        save_checkpoint(checkpoint, state)
    frame = load_saved_stream(api.provider, kind, symbol, args.start, args.end)
    return frame, {k: v for k, v in state.items() if k not in {"cursor", "fingerprints"}}


def load_saved_stream(provider, kind, symbol, start, end):
    """Keep older saved queries when provider retention rolls forward."""
    jobs = list((DATA / "raw" / provider / symbol / kind).glob("*"))
    def job_order(job):
        checkpoint = read_checkpoint(job / "checkpoint.json")
        if not checkpoint:
            return ("", 0, job.stat().st_mtime_ns)
        return (str(checkpoint.get("end", "")), int(checkpoint.get("checkpoint_revision", 0)), job.stat().st_mtime_ns)
    jobs.sort(key=job_order)
    frames = []
    for job in jobs:
        page_paths = sorted(job.glob("page_*.json.gz"))
        for path in page_paths:
            with gzip.open(path, "rt", encoding="utf-8") as f:
                frames.append(normalize(json.load(f), provider, kind, symbol))
    # A provider's later query can overlap its older saved pages. Keep the
    # newest returned bar for each timestamp; never compare it to another API.
    frame = pd.concat(frames, ignore_index=True) if frames else normalize([], provider, kind, symbol)
    if not frame.empty:
        frame = frame.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
    return frame[frame.date.between(start, end)].reset_index(drop=True)


def minute_end(x, convention):
    x = x.copy()
    if convention == "start":
        x["timestamp"] += pd.Timedelta(minutes=1)
    x["date"] = x.timestamp.dt.strftime("%Y-%m-%d")
    return x


def save_daily_csv(symbol, day_frames):
    """Persist available daily history before any potentially long minute sweep."""
    frames = [day_frames[p] for p in ["KIWOOM", "TOSS", "KIS"] if p in day_frames and not day_frames[p].empty]
    if not frames:
        return 0
    combined = pd.concat(frames, ignore_index=True).drop_duplicates(["source", "date"], keep="last")
    priority = {"KIWOOM": 0, "TOSS": 1, "KIS": 2}
    combined["_priority"] = combined.source.map(priority).fillna(99)
    selected = combined.sort_values(["date", "_priority"]).drop_duplicates("date", keep="first").drop(columns="_priority")
    csv(selected.sort_values("date"), DATA / f"{symbol}_daily.csv")
    return len(selected)


def prepare_toss_minutes(minute, symbol):
    """Convert TOSS's minute labels to the verified per-symbol output convention."""
    out = minute.copy()
    if symbol in {"KOSPI", "KOSDAQ"}:
        # Index output labels are bar-end; the API timestamp labels the minute start.
        out = minute_end(out, "start")
        semantics = "bar_end"
    else:
        # ETF output labels are bar-start; preserve the API timestamp.
        semantics = "bar_start"
    out["timestamp_semantics"] = semantics
    out["source_timestamp_convention"] = "start"
    return out


def prepare_symbol(symbol, streams, args):
    day_frames, mframes, thirty, forced, labels = {}, {}, [], [], []
    for provider in ["TOSS", "KIWOOM", "KIS"]:
        daily = streams.get((provider, "daily"), pd.DataFrame())
        if not daily.empty:
            day_frames[provider] = daily
        minute = streams.get((provider, "1m"), pd.DataFrame())
        if minute.empty:
            continue
        # Preserve each vendor's documented/chart convention. Price/clock
        # values from another vendor are never used to validate or discard it.
        if provider == "TOSS":
            mframes[provider] = prepare_toss_minutes(minute, symbol)
            label = "start"
        else:
            label = "start" if provider == "KIWOOM" else args.kis_1m_label
            mframes[provider] = minute_end(minute, label)
            mframes[provider]["timestamp_semantics"] = "bar_end"
            mframes[provider]["source_timestamp_convention"] = label
        labels.append({"symbol": symbol, "stream": provider + "_1m", "convention": label, "canonical_semantics": str(mframes[provider].timestamp_semantics.iloc[0]), "alignment_basis": "provider timestamp convention; no cross-provider price matching"})
        csv(mframes[provider], DATA / "minute" / f"{symbol}_{provider}_1m.csv")
    for provider, minute in mframes.items():
        f = minute[minute.timestamp.dt.strftime("%H:%M:%S").eq("15:20:00")].copy()
        f["timestamp_semantics"], f["price_semantics"] = "bar_end", "15:19..15:20 minute close"
        forced.append(f)
    native = streams.get(("KIWOOM", "30m"), pd.DataFrame())
    if not native.empty:
        label = args.kiwoom_30m_label
        labels.append({"symbol": symbol, "stream": "KIWOOM_30m", "convention": label, "alignment_basis": "Kiwoom minute/30-minute chart labels mark bar start"})
        if label:
            native = native.copy()
            if label == "start":
                native["timestamp"] += pd.Timedelta(minutes=30)
            native["date"] = native.timestamp.dt.strftime("%Y-%m-%d")
            native = native[native.timestamp.dt.strftime("%H:%M").isin(TIMES)]
            native["bar_origin"], native["timestamp_semantics"] = "native_30m", "bar_end"
            # Native Kiwoom 30m has highest priority; Toss/KIS minutes fill
            # missing dates after aggregation, without price comparison.
            thirty.append(native)
        else:
            print(f"[{symbol}] Kiwoom native 30m excluded: timestamp label not configured")
    all_bars = pd.concat(thirty, ignore_index=True) if thirty else pd.DataFrame()
    all_forced = pd.concat(forced, ignore_index=True) if forced else pd.DataFrame()
    selected, snapshots, chosen_days, audits = [], [], {}, []
    if not all_bars.empty:
        all_bars = all_bars.drop_duplicates(["source", "timestamp"], keep="first")
        priority = {"KIWOOM": 3, "TOSS": 2, "KIS": 1}
        for day, groups in all_bars.groupby("date", sort=True):
            candidates = []
            for provider, g in groups.groupby("source"):
                f = all_forced[(all_forced.date == day) & (all_forced.source == provider)] if not all_forced.empty else pd.DataFrame()
                complete = len(g) == 12 and set(g.timestamp.dt.strftime("%H:%M")) == set(TIMES)
                score = (int(complete and not f.empty), int(complete), len(g), priority[provider])
                candidates.append((score, provider, g, f))
                audits.append({"symbol": symbol, "date": day, "source": provider, "bars_30m": len(g), "has_1520": not f.empty, "complete_30m": complete})
            _, provider, g, f = max(candidates, key=lambda z: z[0])
            selected.append(g)
            chosen_days[day] = provider
    # Preserve maximum 15:20 range independently, including dates for which an
    # initial partial API page contains 15:20 but no complete 30m decision bar.
    if not all_forced.empty:
        for day, g in all_forced.groupby("date", sort=True):
            same = g[g.source.eq(chosen_days.get(day))]
            selected_force = same if not same.empty else g.sort_values("source", key=lambda z: z.map({"KIWOOM": 0, "TOSS": 1, "KIS": 2}))
            snapshots.append(selected_force.iloc[[0]])
    daily_choices = []
    if day_frames:
        daily_all = pd.concat(day_frames.values(), ignore_index=True)
        for day, g in daily_all.groupby("date", sort=True):
            pref = chosen_days.get(day)
            same = g[g.source.eq(pref)] if pref else pd.DataFrame()
            chosen = same if not same.empty else g.sort_values("source", key=lambda s: s.map({"KIWOOM": 0, "TOSS": 1, "KIS": 2}))
            daily_choices.append(chosen.iloc[[0]])
    if selected:
        csv(pd.concat(selected).sort_values("timestamp"), DATA / f"{symbol}_30m.csv")
    if snapshots:
        csv(pd.concat(snapshots).sort_values("timestamp"), DATA / f"{symbol}_1520.csv")
    if daily_choices:
        csv(pd.concat(daily_choices).sort_values("date"), DATA / f"{symbol}_daily.csv")
    coverage = []
    for kind, chunks in [("30m", selected), ("1520", snapshots), ("daily", daily_choices)]:
        x = pd.concat(chunks) if chunks else pd.DataFrame()
        coverage.append({"symbol": symbol, "name": UNIVERSE[symbol], "kind": kind, "rows": len(x), "days": x.date.nunique() if not x.empty else 0, "start": x.date.min() if not x.empty else None, "end": x.date.max() if not x.empty else None})
    return coverage, audits, labels


def test_run():
    """Offline parser, pagination/checkpoint, clock alignment and boundary tests."""
    global DATA
    old_data = DATA
    DATA = DATA / "_test_run" / "collector" / pd.Timestamp.now(tz=KST).strftime("%Y%m%d_%H%M%S_%f")
    checks = []
    try:
        times = pd.date_range("2026-09-01 09:00", "2026-09-01 15:29", freq="1min")
        prices = 10000 + np.arange(len(times)) * 5
        raw = pd.DataFrame({"timestamp": times, "date": "2026-09-01", "open": prices, "high": prices + 5, "low": prices - 5, "close": prices + 2, "volume": 100, "trading_value": np.nan, "symbol": "069500", "source": "KIWOOM"})
        end = minute_end(raw, "start")
        assert end[end.timestamp.dt.strftime("%H:%M").eq("15:20")].close.iloc[0] == prices[379] + 2
        assert minute_end(raw, "start").timestamp.iloc[0] == pd.Timestamp("2026-09-01 09:01")
        assert minute_end(end, "end").timestamp.iloc[0] == pd.Timestamp("2026-09-01 09:01")
        raw_toss = raw.copy()
        index_minutes = prepare_toss_minutes(raw_toss, "KOSPI")
        etf_minutes = prepare_toss_minutes(raw_toss, "069500")
        assert index_minutes.timestamp.iloc[0] == raw_toss.timestamp.iloc[0] + pd.Timedelta(minutes=1)
        assert index_minutes.timestamp_semantics.eq("bar_end").all()
        assert etf_minutes.timestamp.iloc[0] == raw_toss.timestamp.iloc[0]
        assert etf_minutes.timestamp_semantics.eq("bar_start").all()
        checks.extend(["1m minute_end conventions are explicit", "1520 excludes auction", "TOSS index bar-end and ETF bar-start conventions are distinct"])
        daily_sample = pd.DataFrame([{"timestamp": pd.Timestamp("2026-09-01"), "date": "2026-09-01", "open": 100, "high": 110, "low": 90, "close": 105, "volume": 10, "trading_value": np.nan, "symbol": "069500", "source": "TOSS"}])
        assert save_daily_csv("069500", {"TOSS": daily_sample}) == 1
        assert pd.read_csv(DATA / "069500_daily.csv").iloc[0].close == 105
        checks.append("daily CSVs are persisted independently before minute collection")
        tossrows = [{"timestamp": "2026-09-01T09:01:00+09:00", "openPrice": "10000", "highPrice": "10010", "lowPrice": "9990", "closePrice": "10005", "volume": "100"}]
        kwrows = [{"cntr_tm": "20260901090000", "open_pric": "-10000", "high_pric": "-10010", "low_pric": "-9990", "cur_prc": "-10005", "trde_qty": "100"}]
        kisrows = [{"stck_bsop_date": "20260901", "stck_cntg_hour": "090000", "stck_oprc": "10000", "stck_hgpr": "10010", "stck_lwpr": "9990", "stck_prpr": "10005", "cntg_vol": "100", "acml_tr_pbmn": "1000000"}]
        for p, rows in [("TOSS", tossrows), ("KIWOOM", kwrows), ("KIS", kisrows)]:
            z = normalize(rows, p, "1m", "069500")
            assert z.close.iloc[0] == 10005 and z.trading_value.isna().all()
        checks.append("three provider parsers; cumulative turnover not treated as per-bar value")
        with patch.object(requests.sessions.Session, "request", side_effect=AssertionError("OFFLINE TEST: network prohibited")):
            for provider in ["TOSS", "KIWOOM", "KIS"]:
                api = API(provider)
                called = []
                def stub(path, **kwargs):
                    called.append((path, kwargs))
                    if provider == "TOSS":
                        return {"result": {"candles": tossrows, "nextBefore": "2026-09-01T09:01:00+09:00"}}, {}
                    if provider == "KIWOOM":
                        return {"stk_min_pole_chart_qry": kwrows}, {"cont-yn": "Y", "next-key": "next_page"}
                    return {"output2": kisrows}, {}
                api.request = stub
                kind = "30m" if provider == "KIWOOM" else "1m"
                rr, nxt, stop = api.page(kind, "069500", None, "2026-09-01", ["2026-09-01", "2026-08-31"])
                assert rr
                if provider == "TOSS":
                    assert called[0][0] == "/api/v1/market-indicators/069500/candles" and called[0][1]["params"]["interval"] == "1m" and nxt
                elif provider == "KIWOOM":
                    assert called[0][1]["body"]["tic_scope"] == "30" and called[0][1]["body"]["base_dt"] == "20260901" and called[0][1]["api_id"] == "ka10080" and nxt == "next_page"
                else:
                    assert called[0][1]["api_id"] == "FHKST03010230" and called[0][1]["params"]["FID_COND_MRKT_DIV_CODE"] == "J"
                    assert nxt == {"index": 1, "clock": "153000"}
                    api.request = lambda *a, **kw: ({"output2": []}, {})
                    _, nxt, stop = api.page("1m", "069500", nxt, "2026-09-01", ["2026-09-01", "2026-08-31"])
                    assert nxt is None and stop == "CALENDAR_PROBE_END"
                api.session.close()
        checks.append("mocked real page methods: Toss 1m, Kiwoom native30, KIS stock-history TR and holiday calendar advance")
        # A standalone 15:20-only date must survive canonical preparation even
        # though it has no complete 30m decision window.
        single = raw.iloc[[379]].copy()
        single["timestamp"] += pd.Timedelta(days=1)
        single["date"] = single.timestamp.dt.strftime("%Y-%m-%d")
        prep_args = argparse.Namespace(kis_1m_label="start", kiwoom_30m_label="start")
        prepare_symbol("KOSPI", {("KIWOOM", "1m"): pd.concat([raw, single])}, prep_args)
        assert not (DATA / "069500_30m.csv").exists()
        assert len(pd.read_csv(DATA / "KOSPI_1520.csv")) == 2
        checks.append("Kiwoom native 30m and exact 15:20 snapshots preserved independently")
        class Fake:
            provider = "TOSS"
            calls = 0
            def page(self, kind, symbol, cursor, end, dates):
                self.calls += 1
                return (tossrows, "older", None) if cursor is None else ([], None, "SERVER_END")
        args = argparse.Namespace(start="1990-01-01", end="2026-09-01", kis_lookback_days=366, max_pages=1)
        fake = Fake()
        with patch.object(requests.sessions.Session, "request", side_effect=AssertionError("OFFLINE TEST: network prohibited")):
            _, manifest = collect_stream(fake, "1m", "069500", args)
            assert manifest["status"] == "CLIENT_PAGE_CAP"
            args.max_pages = 0
            x, manifest = collect_stream(fake, "1m", "069500", args)
            assert manifest["status"] == "SERVER_END" and len(x) == 1 and fake.calls == 2
            collect_stream(fake, "1m", "069500", args)
            assert fake.calls == 2
        checks.append("uncapped pagination, cap classification, resume and finished-query cache")
        prior = DATA / "raw" / "TOSS" / "114800" / "1m" / "prior-complete"
        prior.mkdir(parents=True, exist_ok=True)
        save_json(prior / "checkpoint.json", {"provider": "TOSS", "kind": "1m", "symbol": "114800", "status": "SERVER_END", "earliest": "2022-01-01 09:01:00", "latest": "2026-09-01 09:01:00", "pages": 1, "rows": 1})
        class Overlap:
            provider = "TOSS"
            calls = 0
            def page(self, kind, symbol, cursor, end, dates):
                self.calls += 1
                return tossrows, "older", None
        overlap = Overlap()
        args.end = "2026-09-02"
        _, manifest = collect_stream(overlap, "1m", "114800", args)
        assert manifest["status"] == "PRIOR_COMPLETE_HISTORY_OVERLAP" and overlap.calls == 1
        checks.append("completed Toss maximum history reused across end dates; only new pages fetched to overlap")
        class Repeated:
            provider = "TOSS"
            def page(self, kind, symbol, cursor, end, dates):
                return tossrows, "second" if cursor else "first", None
        x, manifest = collect_stream(Repeated(), "1m", "122630", args)
        assert manifest["status"] == "ERROR" and len(x) == 1
        checks.append("repeated response cannot be mislabeled as history exhaustion")
        with tempfile.TemporaryDirectory(dir=str(DATA)) as tempdir:
            cp = Path(tempdir) / "checkpoint.json"
            old = {"status": "RUNNING", "pages": 5, "checkpoint_revision": 1}
            cp.write_text(json.dumps(old), encoding="utf-8")
            new = {"status": "RUNNING", "pages": 6}
            with patch("os.replace", side_effect=PermissionError("simulated Windows file lock")), patch("time.sleep"):
                recovery = save_checkpoint(cp, new)
            assert recovery != cp and recovery.exists()
            assert read_checkpoint(cp)["pages"] == 6
        checks.append("Windows checkpoint lock retries then resumes from recovery state")
        report = {"status": "PASS", "network_calls": 0, "synthetic_only": True, "checks": checks}
        save_json(RESULTS / "test_run" / "collector_test.json", report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
    finally:
        DATA = old_data


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--start", default="1990-01-01", help="default precedes every ETF listing; collect to server end")
    p.add_argument("--end", default=(pd.Timestamp.now(tz=KST).normalize() - pd.Timedelta(days=1)).strftime("%Y-%m-%d"))
    p.add_argument("--providers", default=None, help="override the top-level KIS_KIWOOM setting: comma-separated TOSS,KIWOOM,KIS")
    p.add_argument("--symbols", default=",".join(UNIVERSE))
    p.add_argument("--env-file", type=Path, default=ENV_DEFAULT)
    p.add_argument("--kiwoom-env-file", type=Path, default=DESKTOP / "VibeTrading" / ".env", help="fallback: load only Kiwoom credentials from original collector configuration")
    p.add_argument("--max-pages", type=int, default=0, help="0 = unlimited; positive is a resumable debugging cap per stream")
    p.add_argument("--kis-lookback-days", type=int, default=366)
    p.add_argument("--kis-1m-label", choices=["start", "end"], default="start")
    p.add_argument("--kiwoom-30m-label", choices=["start", "end"], default="start")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--prepare-only", action="store_true")
    p.add_argument("--test-run", action="store_true")
    args = p.parse_args()
    if args.test_run:
        test_run(); return
    if pd.Timestamp(args.start) > pd.Timestamp(args.end) or args.max_pages < 0 or args.kis_lookback_days < 1:
        p.error("Invalid dates/page cap/lookback")
    symbols = args.symbols.split(",")
    if not set(symbols) <= set(UNIVERSE):
        p.error("Only the seven requested ETFs are supported")
    default_providers = ["TOSS", "KIWOOM", "KIS"] if KIS_KIWOOM else ["TOSS"]
    providers = [s.strip().upper() for s in args.providers.split(",")] if args.providers else default_providers
    if args.dry_run:
        print(json.dumps({"mode": "DRY_RUN", "network_calls": 0, "output": str(DATA), "start": args.start, "end": args.end, "symbols": {s: UNIVERSE[s] for s in symbols}, "providers": providers, "kis_kiwoom_enabled": KIS_KIWOOM, "phases": ["save/reuse daily CSV for all symbols", "if enabled, Kiwoom native 30m for all seven ETFs", "Toss 1m for all symbols; reuse complete history, resume partial streams", "if enabled, Kiwoom/KIS 1m supplementation"] , "max_pages": args.max_pages, "env_file": str(args.env_file)}, ensure_ascii=False, indent=2)); return
    if args.env_file.exists():
        from dotenv import load_dotenv
        load_dotenv(args.env_file, override=False)
    if args.kiwoom_env_file.exists() and not all(os.getenv(k) for k in KEYS["KIWOOM"]):
        from dotenv import dotenv_values
        fallback = dotenv_values(args.kiwoom_env_file)
        if not os.getenv("KIWOOM_APP_KEY") and fallback.get("KIWOOM_APP_KEY"):
            os.environ["KIWOOM_APP_KEY"] = fallback["KIWOOM_APP_KEY"]
        if not os.getenv("KIWOOM_APP_SECRET"):
            secret = fallback.get("KIWOOM_APP_SECRET") or fallback.get("KIWOON_APP_SECRET")
            if secret:
                os.environ["KIWOOM_APP_SECRET"] = secret
    if not providers or any(s not in KEYS for s in providers):
        p.error("No providers configured/found; use --env-file or --providers")
    if not args.prepare_only and any(not all(os.getenv(k) for k in KEYS[s]) for s in providers):
        p.error("Missing credentials for an explicitly requested provider")
    streams_manifest, coverage, audits, labels = [], [], [], []
    # Share authenticated clients across symbols to respect token-issuance limits.
    clients = {provider: API(provider) for provider in providers} if not args.prepare_only else {}
    streams = {symbol: {} for symbol in symbols}

    def load_phase(provider, kind, symbol, dates=None):
        if args.prepare_only:
            frame = load_saved_stream(provider, kind, symbol, args.start, args.end)
            for job in (DATA / "raw" / provider / symbol / kind).glob("*"):
                checkpoint = read_checkpoint(job / "checkpoint.json")
                if checkpoint:
                    streams_manifest.append({k: v for k, v in checkpoint.items() if k not in {"cursor", "fingerprints"}})
            return frame
        frame, manifest = collect_stream(clients[provider], kind, symbol, args, dates)
        streams_manifest.append(manifest)
        return frame

    # Global phases keep the long-retention native 30m history ahead of minute
    # downloads. Raw minute pages are subsequently retained by provider and
    # supply 15:20 plus any missing 30m dates. The API itself does not expose
    # an efficient single-price endpoint for historical 15:20 across vendors.
    print("[PHASE 1/3] Collect daily calendars for all symbols/providers", flush=True)
    for symbol in symbols:
        for provider in providers:
            streams[symbol][(provider, "daily")] = load_phase(provider, "daily", symbol)
        day_frames = {provider: streams[symbol][(provider, "daily")] for provider in providers}
        rows = save_daily_csv(symbol, day_frames)
        print(f"[DAILY CSV {symbol}] rows={rows}", flush=True)

    print("[PHASE 2/3] Collect native Kiwoom 30m history for all seven ETFs", flush=True)
    if "KIWOOM" in providers:
        for symbol in symbols:
            streams[symbol][("KIWOOM", "30m")] = load_phase("KIWOOM", "30m", symbol)

    print("[PHASE 3/3] Collect each provider's available 1m history; build 15:20 and fill missing 30m coverage", flush=True)
    for symbol in symbols:
        for provider in providers:
            daily = streams[symbol].get((provider, "daily"), pd.DataFrame())
            dates = sorted(daily.loc[daily.date.ge((pd.Timestamp(args.end) - pd.Timedelta(days=args.kis_lookback_days)).strftime("%Y-%m-%d")), "date"].unique(), reverse=True) if provider == "KIS" and not daily.empty else []
            streams[symbol][(provider, "1m")] = load_phase(provider, "1m", symbol, dates)
        c, a, l = prepare_symbol(symbol, streams[symbol], args)
        coverage.extend(c); audits.extend(a); labels.extend(l)
        csv(pd.DataFrame(coverage), DATA / "coverage.csv")
        csv(pd.DataFrame(audits), DATA / "provider_day_coverage.csv")
        save_json(DATA / "collection_manifest.json", {"updated_kst": str(pd.Timestamp.now(tz=KST)), "streams": streams_manifest, "timestamp_alignment": labels, "selection": "one provider per symbol-day by coverage only; KIWOOM native30 first, TOSS then KIS supplemental; no cross-provider price checks; 15:20 selected by availability", "universe": UNIVERSE, "adjusted": False})
    for api in clients.values():
        api.session.close()
    incomplete = [m for m in streams_manifest if m["status"] in {"ERROR", "CLIENT_PAGE_CAP", "RUNNING"}]
    if incomplete:
        raise SystemExit(f"{len(incomplete)} incomplete streams; raw data/checkpoints saved. Rerun same query to resume; see data/collection_manifest.json")
    print(f"Collection/preparation complete: {DATA}")


if __name__ == "__main__":
    main()
