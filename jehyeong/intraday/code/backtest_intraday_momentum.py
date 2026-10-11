"""Final two Toss intraday strategies, with ETF buy-and-hold comparisons.
30-minute signal/normal exits; next-minute OPEN fills; forced 15:20 CLOSE.
Per side: commission 1.5bp, adverse slippage 1bp. No data collection.
Run from any directory: python backtest_intraday_momentum.py
Only results/final is overwritten; no research/history directories created.
Historical evaluation dates are frozen in results/final/selected_rules.json.
"""
from __future__ import annotations
import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from collect_intraday_data import DATA, RESULTS, FIELDS, csv, save_json, validate, aggregate_30m

VERSIONS = ("V1_next_open",)
MARKETS = {
    "KOSPI200": {"signal": "069500", "long": ("069500", "122630"), "short": (), "ceiling": 2.},
    "KOSDAQ150": {"signal": "229200", "long": ("229200",), "short": ("251340",), "ceiling": 1.},
}
TIMES = pd.date_range("2000-01-01 09:30", "2000-01-01 15:00", freq="30min").strftime("%H:%M").tolist()
SESSION_TIMES = pd.date_range("2000-01-01 09:01", "2000-01-01 15:20", freq="1min").strftime("%H:%M").tolist()
LOOKBACK, VOL_TARGET = 14, .02
TRADE_COLUMNS = ["trade_id", "date", "entry_time", "exit_time", "direction", "signal_strength", "strength_bucket", "vol14", "vol_bucket", "desired_leverage", "leverage", "entry_equity", "invested_capital", "signal_entry_price", "signal_exit_price", "future_signal_return", "holding_minutes", "exit_reason", "gross_pnl", "slippage_cost", "commission_cost", "net_pnl", "trade_return", "invested_return", "turnover_notional", "entry_source", "exit_source"]
ORDER_COLUMNS = ["trade_id", "date", "timestamp", "symbol", "side", "quantity", "reference_price", "fill_price", "slippage_cost", "commission_cost", "filled_notional", "reference_notional", "target_weight", "source"]
TRADE_COLUMNS += ["entry_signal_time", "exit_signal_time", "entry_bar_end", "exit_bar_end", "execution_version"]
ORDER_COLUMNS += ["signal_timestamp", "execution_bar_end", "execution_field", "execution_version"]


def safe_ratio(a, b):
    return float(a / b) if np.isfinite(b) and b != 0 else np.nan


def load_table(path, kind, symbol, required_source="TOSS"):
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")
    x = pd.read_csv(path, dtype={"symbol": str, "date": str})
    needed = {"date", "timestamp", "symbol", "source", *FIELDS}
    if kind == "1m":
        needed.add("timestamp_semantics")
    if not needed <= set(x):
        raise ValueError(f"{path.name}: missing columns {sorted(needed - set(x))}")
    if required_source and (x.source.isna().any() or not x.source.eq(required_source).all()):
        raise ValueError(f"{path.name}: requires source={required_source}")
    x["timestamp"] = pd.to_datetime(x.timestamp, errors="raise")
    if x.timestamp.dt.tz is not None:
        x["timestamp"] = x.timestamp.dt.tz_convert("Asia/Seoul").dt.tz_localize(None)
    x["date"] = pd.to_datetime(x.date).dt.strftime("%Y-%m-%d")
    for c in FIELDS:
        x[c] = pd.to_numeric(x[c], errors="raise")
    validate(x)
    if not x.timestamp.dt.strftime("%Y-%m-%d").eq(x.date).all() or not x.symbol.str.zfill(6).eq(symbol).all():
        raise ValueError(f"{path.name}: date/symbol mismatch")
    key = "date" if kind == "daily" else "timestamp"
    if x.duplicated(key).any():
        raise ValueError(f"{path.name}: duplicate {key}")
    if kind == "1m":
        if not x.timestamp_semantics.eq("bar_end").all() or not x.timestamp.dt.second.eq(0).all():
            raise ValueError(f"{path.name}: requires exact minute-end timestamps")
        x["time"] = x.timestamp.dt.strftime("%H:%M")
        x = x[x.time.isin(SESSION_TIMES)].copy()
        x["slot"] = x.timestamp.dt.hour * 60 + x.timestamp.dt.minute - 541
    x["trading_value"] = pd.to_numeric(x.get("trading_value", np.nan), errors="raise")
    return x.sort_values("timestamp").reset_index(drop=True)


def load_data(directory, required_source="TOSS"):
    symbols = sorted({s for m in MARKETS.values() for s in m["long"] + m["short"]})
    result = {}
    for s in symbols:
        minute = load_table(directory / "minute" / f"{s}_TOSS_1m.csv", "1m", s, required_source)
        result[s] = {"1m": minute, "daily": load_table(directory / f"{s}_daily.csv", "daily", s, required_source), "1520": minute[minute.time.eq("15:20")].copy()}
        print(f"[LOAD {s}] minute_rows={len(minute)} {minute.date.min()}..{minute.date.max()}", flush=True)
    return result


def strength_bucket(s):
    return "1.00-1.25" if s < 1.25 else "1.25-1.50" if s < 1.5 else "1.50+"


def make_features(minutes, daily):
    """Original 30m Noise/VWAP from complete Toss minute windows; no look-ahead."""
    dd = daily.set_index("date").sort_index()
    bars = aggregate_30m(minutes)
    bars["time"] = bars.timestamp.dt.strftime("%H:%M")
    bars["slot"] = bars.timestamp.dt.hour * 60 + bars.timestamp.dt.minute - 541
    grid = bars.pivot(index="date", columns="time", values="close").reindex(index=dd.index, columns=TIMES)
    noise = grid.div(dd.open, axis=0).sub(1).abs().shift(1).rolling(LOOKBACK, min_periods=LOOKBACK).mean()
    vol = dd.close.pct_change(fill_method=None).rolling(LOOKBACK, min_periods=LOOKBACK).std(ddof=1).shift(1)
    x = bars[bars.date.isin(dd.index)].copy()
    valid = x.trading_value.notna() & np.isfinite(x.trading_value) & x.trading_value.ge(0)
    implied = x.trading_value / x.volume.replace(0, np.nan)
    if (valid & x.volume.gt(0) & ((implied < x.low - 1e-6) | (implied > x.high + 1e-6))).any():
        raise ValueError("Minute trading_value units/range invalid")
    denom = x.volume.groupby(x.date).cumsum().replace(0, np.nan)
    proxy = (((x.high + x.low + x.close) / 3) * x.volume).groupby(x.date).cumsum() / denom
    actual = valid.groupby(x.date).cummin()
    real = x.trading_value.fillna(0).groupby(x.date).cumsum() / denom
    x["vwap_proxy"], x["vwap_actual"] = proxy, real.where(actual)
    x["vwap_used"] = real.where(actual, proxy)
    x["vwap_kind"] = np.where(actual, "actual", "30m_HLC3_proxy")
    x = x[x.time.isin(TIMES)].copy()
    x["noise"] = noise.to_numpy()[dd.index.get_indexer(x.date), pd.Index(TIMES).get_indexer(x.time)]
    x["open_day"], x["previous_close"], x["vol14"] = x.date.map(dd.open), x.date.map(dd.close.shift(1)), x.date.map(vol)
    highanchor = x[["open_day", "previous_close"]].max(axis=1)
    lowanchor = x[["open_day", "previous_close"]].min(axis=1)
    x["upper_band"], x["lower_band"] = highanchor * (1 + x.noise), lowanchor * (1 - x.noise)
    x["usable"] = np.isfinite(x[["noise", "vol14", "previous_close", "vwap_used"]]).all(axis=1) & x.noise.gt(0) & x.vol14.gt(0)
    x["price"] = x.close
    x["entry_signal"] = np.where(x.usable & x.close.gt(x.upper_band), 1, np.where(x.usable & x.close.lt(x.lower_band), -1, 0))
    x["signal_strength"] = np.where(x.entry_signal.eq(1), (x.close / highanchor - 1) / x.noise, np.where(x.entry_signal.eq(-1), (1 - x.close / lowanchor) / x.noise, np.nan))
    x["strength_bucket"] = np.where(x.entry_signal.eq(0), "none", np.where(x.signal_strength < 1.25, "1.00-1.25", np.where(x.signal_strength < 1.5, "1.25-1.50", "1.50+")))
    # Research controls: strictly prior daily information and completed opening range.
    prior_ma = dd.close.rolling(20, min_periods=20).mean().shift(1)
    prior_close = dd.close.shift(1)
    trend = np.sign(prior_close - prior_ma).where(prior_ma.notna())
    median_range = ((dd.high-dd.low)/dd.open).shift(1).rolling(20, min_periods=20).median()
    opening = minutes[minutes.time.between("09:01", "09:10")].groupby("date")
    opening_high, opening_low = opening.high.max(), opening.low.min()
    complete_opening = opening.size().eq(10)
    x["prior_ma20_direction"] = x.date.map(trend)
    x["prior_median_range20"] = x.date.map(median_range)
    x["or10_high"] = x.date.map(opening_high.where(complete_opening))
    x["or10_low"] = x.date.map(opening_low.where(complete_opening))
    return x.reset_index(drop=True)


def eligible_days(market, data, features, start=None, end=None):
    symbols = set(MARKETS[market]["long"] + MARKETS[market]["short"])
    lower = max(data[s]["1m"].date.min() for s in symbols)
    upper = min(data[s]["1m"].date.max() for s in symbols)
    lower = max(lower, start) if start else lower
    upper = min(upper, end) if end else upper
    grouped = {s: {d: g for d, g in data[s]["1m"].groupby("date")} for s in symbols}
    daily = {s: data[s]["daily"].set_index("date") for s in symbols}
    panels = {d: g for d, g in features.groupby("date")}
    audits, valid = [], []
    calendar = data[MARKETS[market]["signal"]]["daily"].date
    for day in calendar[calendar.between(lower, upper)]:
        reasons = []
        for s in sorted(symbols):
            g = grouped[s].get(day)
            if g is None or len(g) != 380 or set(g.time) != set(SESSION_TIMES):
                reasons.append(f"{s}:missing_minute_or_special_session")
                continue
            if day not in daily[s].index:
                reasons.append(f"{s}:missing_daily")
            elif not np.isclose(g.open.iloc[0], daily[s].loc[day, "open"], rtol=1e-8, atol=1e-6):
                reasons.append(f"{s}:daily_minute_open_mismatch")
            if g.volume.iloc[-1] <= 0:
                reasons.append(f"{s}:zero_volume_forced_1520")
        panel = panels.get(day)
        if panel is None or len(panel) != len(TIMES) or not panel.usable.all():
            reasons.append("signal:insufficient_exact_14_session_30m_history")
        audits.append({"market": market, "date": day, "eligible": not reasons, "reason": ";".join(reasons), "sample_lower": lower, "sample_upper": upper})
        if not reasons:
            valid.append(day)
    if not valid:
        raise ValueError(f"{market}: no eligible days in {lower}..{upper}")
    return valid, pd.DataFrame(audits)


def add_vol_buckets(features, days):
    x = features.copy()
    vals = x[x.date.isin(days)].groupby("date").vol14.first()
    cuts = vals.quantile([1 / 3, 2 / 3]).to_numpy()
    x["vol_bucket"] = np.where(x.vol14 <= cuts[0], "Low", np.where(x.vol14 <= cuts[1], "Mid", "High"))
    return x, cuts.tolist()


def weights(direction, leverage, market):
    vehicles = MARKETS[market]["long" if direction == 1 else "short"]
    if leverage <= 1:
        return {vehicles[0]: leverage}
    if len(vehicles) != 2:
        raise AssertionError("KOSDAQ leverage not allowed")
    return {vehicles[0]: 2 - leverage, vehicles[1]: leverage - 1}



@dataclass
class Config:
    initial_capital: float = 100_000_000.
    commission_bps: float = 1.5
    whole_shares: bool = False
    slippage_bps: float = 1.0


def simulate(market, model, features, data, days, execution_version, config, direction_filter=0, *, exposure_cap=None, sizing_mode="full", exit_policy="normal", max_daily_entries=None, cooldown_minutes=0, entry_confirmation="none", entry_cutoff=None, minute_safety_stop_k=None):
    """Execute prior-close pending decisions, then observe current close signal."""
    if execution_version not in VERSIONS:
        raise ValueError("Unknown execution version")
    if sizing_mode not in ("full", "vol_only", "strength_only") or exit_policy not in ("normal", "hold_1520"):
        raise ValueError("Unknown diagnostic sizing/exit policy")
    if exposure_cap is not None and (not np.isfinite(exposure_cap) or exposure_cap <= 0):
        raise ValueError("Exposure cap must be positive")
    if max_daily_entries is not None and max_daily_entries < 1:
        raise ValueError("Daily entry limit must be positive")
    if not np.isfinite(cooldown_minutes) or cooldown_minutes < 0 or cooldown_minutes % 30:
        raise ValueError("Cooldown must be a nonnegative multiple of 30 minutes")
    if entry_confirmation not in ("none", "ma20", "or10_vwap", "ma20_or10_vwap"):
        raise ValueError("Unknown entry confirmation")
    if entry_cutoff is not None and entry_cutoff not in TIMES:
        raise ValueError("Entry cutoff must be a 30-minute decision time")
    if minute_safety_stop_k is not None and (not np.isfinite(minute_safety_stop_k) or minute_safety_stop_k<=0):
        raise ValueError("Safety stop multiple must be positive")
    field = "open"
    fee = config.commission_bps / 10000
    slip = config.slippage_bps / 10000
    cash, position = config.initial_capital, None
    orders, trades, daily_rows, marks, rejects = [], [], [], [], []
    last_exit_time, cooldown_until = None, None
    cooldown_blocks = []
    entry_filter_blocks = []
    symbols = set(MARKETS[market]["long"] + MARKETS[market]["short"])
    signal = MARKETS[market]["signal"]
    caches = {}
    for s in symbols:
        if "_minute_cache" not in data[s]:
            data[s]["_minute_cache"] = {day: {c: g[c].to_numpy() for c in ("open", "close", "volume", "timestamp")} for day, g in data[s]["1m"].groupby("date", sort=False)}
        caches[s] = data[s]["_minute_cache"]
    cached = data[signal].get("_event_rows_cache")
    feature_key = hashlib.sha256(pd.util.hash_pandas_object(features, index=True).to_numpy().tobytes()).digest()
    if cached is None or cached[0] != feature_key:
        cached = (feature_key, {day: list(g.itertuples(index=False)) for day, g in features.groupby("date", sort=False)})
        data[signal]["_event_rows_cache"] = cached
    grouped = cached[1]
    current = {}

    def price(s, slot, kind):
        return float(current[s][kind][slot])

    def equity(slot):
        return cash + (sum(q * price(s, slot, "close") for s, q in position["shares"].items()) if position else 0.)

    def order(s, q, side, slot, kind, signal_time, tid, weight):
        nonlocal cash
        bar_end = pd.Timestamp(current[s]["timestamp"][slot])
        actual_time = bar_end - pd.Timedelta(minutes=1) if kind == "open" else bar_end
        if signal_time is not None and not bar_end > signal_time:
            raise AssertionError("Look-ahead/same-signal-bar fill")
        ref = price(s, slot, kind)
        fill = ref * (1 + slip if side == "BUY" else 1 - slip)
        notional, commission = q * fill, q * fill * fee
        slip_cost = q * abs(fill - ref)
        cash += -(notional + commission) if side == "BUY" else notional - commission
        source = str(data[s]["1m"].source.iloc[0])
        row = {"trade_id": tid, "date": actual_time.strftime("%Y-%m-%d"), "timestamp": actual_time, "symbol": s, "side": side, "quantity": q, "reference_price": ref, "fill_price": fill, "slippage_cost": slip_cost, "commission_cost": commission, "filled_notional": notional, "reference_notional": q * ref, "target_weight": weight, "source": source, "signal_timestamp": signal_time, "execution_bar_end": bar_end, "execution_field": kind, "execution_version": execution_version}
        orders.append(row)
        return row

    def enter(r, slot):
        nonlocal position
        desired = 1. if model.startswith("M0") else VOL_TARGET / r.vol14 * r.signal_strength
        if not model.startswith("M0"):
            if sizing_mode == "vol_only": desired = VOL_TARGET / r.vol14
            elif sizing_mode == "strength_only": desired = r.signal_strength
        ceiling = min(MARKETS[market]["ceiling"], exposure_cap) if exposure_cap is not None else MARKETS[market]["ceiling"]
        leverage = min(ceiling, desired)
        target = {s: w for s, w in weights(r.entry_signal, leverage, market).items() if w > 0}
        if any(current[s]["volume"][slot] <= 0 for s in target):
            rejects.append({"date": r.date, "signal_time": r.timestamp, "bar_end": pd.Timestamp(current[signal]["timestamp"][slot]), "reason": "entry_cancelled_zero_volume"})
            return
        cap, tid, shares, entry_orders = cash, len(trades) + 1, {}, []
        for s, w in target.items():
            q = cap * w / (price(s, slot, field) * (1 + slip) * (1 + fee))
            q = np.floor(q) if config.whole_shares else q
            if q <= 0:
                continue
            shares[s] = q
            entry_orders.append(order(s, q, "BUY", slot, field, r.timestamp, tid, w))
        if not shares:
            return
        if cash < -1e-6 * config.initial_capital:
            raise AssertionError("Implicit borrowing")
        position = {"trade_id": tid, "shares": shares, "weights": target, "entry_orders": entry_orders, "entry_time": entry_orders[0]["timestamp"], "entry_signal_time": r.timestamp, "entry_bar_end": entry_orders[0]["execution_bar_end"], "direction": r.entry_signal, "entry_equity": cap, "signal_strength": r.signal_strength, "strength_bucket": r.strength_bucket, "vol14": r.vol14, "vol_bucket": r.vol_bucket, "desired_leverage": desired, "leverage": leverage, "signal_entry_price": r.price}
        if minute_safety_stop_k is not None:
            position["entry_row"] = r
            position["safety_stop"] = price(signal, slot, field)*(1-r.entry_signal*minute_safety_stop_k*r.prior_median_range20)

    def close(slot, kind, signal_time, reason):
        nonlocal position, last_exit_time, cooldown_until
        pos = position
        exits = [order(s, q, "SELL", slot, kind, signal_time, pos["trade_id"], pos["weights"][s]) for s, q in pos["shares"].items()]
        entry = pos["entry_orders"]
        gross = sum(e["reference_notional"] - b["reference_notional"] for b, e in zip(entry, exits))
        commission = sum(o["commission_cost"] for o in entry + exits)
        invested = sum(o["filled_notional"] + o["commission_cost"] for o in entry)
        slippage = sum(o["slippage_cost"] for o in entry + exits)
        net = gross - slippage - commission
        row = {k: pos[k] for k in ("trade_id", "entry_time", "direction", "signal_strength", "strength_bucket", "vol14", "vol_bucket", "desired_leverage", "leverage", "entry_equity", "signal_entry_price", "entry_signal_time", "entry_bar_end")}
        exit_time = exits[0]["timestamp"]
        sp = price(signal, slot, kind)
        row.update(date=exit_time.strftime("%Y-%m-%d"), exit_time=exit_time, exit_signal_time=signal_time, exit_bar_end=exits[0]["execution_bar_end"], execution_version=execution_version, invested_capital=invested, signal_exit_price=sp, future_signal_return=pos["direction"] * (sp / pos["signal_entry_price"] - 1), holding_minutes=(exit_time - pos["entry_time"]).total_seconds() / 60, exit_reason=reason, gross_pnl=gross, slippage_cost=slippage, commission_cost=commission, net_pnl=net, trade_return=net / pos["entry_equity"], invested_return=net / invested, turnover_notional=sum(o["reference_notional"] for o in entry + exits), entry_source="|".join(o["source"] for o in entry), exit_source="|".join(o["source"] for o in exits))
        trades.append(row)
        position = None
        last_exit_time = exit_time
        cooldown_until = exit_time + pd.Timedelta(minutes=cooldown_minutes)

    def cooldown_active(timestamp):
        return cooldown_minutes > 0 and cooldown_until is not None and timestamp < cooldown_until

    def record_cooldown_block(r, timestamp, stage):
        cooldown_blocks.append({"date": r.date, "signal_time": r.timestamp, "direction": r.entry_signal,
                                "checked_time": timestamp, "last_exit_time": last_exit_time,
                                "cooldown_until": cooldown_until, "stage": stage})

    def execute(pending, slot):
        if pending is None:
            return None
        r = pending["row"]
        if pending["exit"] and position is not None:
            if any(current[s]["volume"][slot] <= 0 for s in position["shares"]):
                rejects.append({"date": r.date, "signal_time": r.timestamp, "bar_end": pd.Timestamp(current[signal]["timestamp"][slot]), "reason": "exit_deferred_zero_volume"})
                return pending
            close(slot, field, r.timestamp, pending["exit"])
        if pending["enter"] and position is None and slot < 379:
            actual_time = pd.Timestamp(current[signal]["timestamp"][slot]) - pd.Timedelta(minutes=1)
            if cooldown_active(actual_time):
                record_cooldown_block(r, actual_time, "execution")
            else:
                enter(r, slot)
        return None

    for day in days:
        if position:
            raise AssertionError("Overnight position")
        current = {s: caches[s][day] for s in symbols}
        start_equity, first_trade, first_order, pending = cash, len(trades), len(orders), None
        decision_rows = {r.slot: r for r in grouped[day]}
        for slot in range(29, 379):
            pending = execute(pending, slot)
            timestamp = pd.Timestamp(current[signal]["timestamp"][slot])
            marks.append({"date": day, "timestamp": timestamp, "equity": equity(slot), "cash": cash, "direction": position["direction"] if position else 0})
            r = decision_rows.get(slot)
            if pending is not None:  # A zero-volume exit remains queued.
                continue
            if position and minute_safety_stop_k is not None:
                side, stop = position["direction"], position["safety_stop"]
                observed = price(signal, slot, "close")
                if (observed<=stop if side==1 else observed>=stop):
                    risk_row = position["entry_row"]._replace(timestamp=timestamp, time=timestamp.strftime("%H:%M"), slot=slot)
                    pending = {"row": risk_row, "exit": "minute_safety_stop", "enter": False}
                    continue
            if r is None:
                continue
            exit_reason, exited_side = None, 0
            if position and exit_policy == "normal":
                side = position["direction"]
                stop = (max(r.upper_band, r.vwap_used) if side == 1 else min(r.lower_band, r.vwap_used)) if model.endswith("P") else (max(r.lower_band, r.vwap_used) if side == 1 else min(r.upper_band, r.vwap_used))
                if (r.price <= stop if side == 1 else r.price >= stop):
                    exit_reason, exited_side = ("paper_stop" if model.endswith("P") else "krx_stop"), side
            allowed_entry = r.time <= TIMES[-1] and r.entry_signal != 0 and (not direction_filter or r.entry_signal == direction_filter)
            if max_daily_entries is not None:
                allowed_entry = allowed_entry and len(trades) - first_trade + int(position is not None) < max_daily_entries
            if allowed_entry:
                reasons = []
                if entry_cutoff is not None and r.time>entry_cutoff:
                    reasons.append("entry_cutoff")
                if entry_confirmation in ("ma20", "ma20_or10_vwap") and r.prior_ma20_direction!=r.entry_signal:
                    reasons.append("prior_ma20_direction")
                if entry_confirmation in ("or10_vwap", "ma20_or10_vwap"):
                    confirmed = r.price>max(r.or10_high,r.vwap_used) if r.entry_signal==1 else r.price<min(r.or10_low,r.vwap_used)
                    if not np.isfinite([r.or10_high,r.or10_low,r.vwap_used]).all() or not confirmed:
                        reasons.append("or10_vwap")
                if minute_safety_stop_k is not None and (not np.isfinite(r.prior_median_range20) or r.prior_median_range20<=0):
                    reasons.append("missing_prior_range20")
                if reasons:
                    entry_filter_blocks.append({"date":day,"signal_time":r.timestamp,"direction":r.entry_signal,"reason":"|".join(reasons)})
                    allowed_entry = False
            if allowed_entry and position is None and cooldown_active(r.timestamp):
                record_cooldown_block(r, r.timestamp, "decision")
                allowed_entry = False
            # A cooldown experiment cannot create a reversal entry while still holding.
            # Zero cooldown retains the existing baseline's simultaneous reversal rule.
            enter_now = allowed_entry and (position is None or bool(not cooldown_minutes and exit_reason and r.entry_signal == -exited_side))
            if exit_reason or enter_now:
                pending = {"row": r, "exit": exit_reason, "enter": bool(enter_now)}
        final_slot = 379
        pending = execute(pending, final_slot)
        if position:
            close(final_slot, "close", None, "forced_1520")
        endtime = pd.Timestamp(day + " 15:20")
        marks.append({"date": day, "timestamp": endtime, "equity": cash, "cash": cash, "direction": 0})
        tt, oo = trades[first_trade:], orders[first_order:]
        gross, commission = sum(t["gross_pnl"] for t in tt), sum(t["commission_cost"] for t in tt)
        net = cash - start_equity
        slippage = sum(t["slippage_cost"] for t in tt)
        if not np.isclose(net, gross - slippage - commission, rtol=1e-9, atol=1e-5):
            raise AssertionError("Daily cash/trade PnL reconciliation failed")
        daily_rows.append({"date": day, "start_equity": start_equity, "equity": cash, "daily_return": net / start_equity, "gross_pnl": gross, "slippage_cost": slippage, "commission_cost": commission, "net_pnl": net, "long_pnl": sum(t["net_pnl"] for t in tt if t["direction"] == 1), "short_pnl": sum(t["net_pnl"] for t in tt if t["direction"] == -1), "trades": len(tt), "order_count": len(oo), "turnover": sum(o["reference_notional"] for o in oo) / start_equity})
    d = pd.DataFrame(daily_rows)
    d["drawdown"] = d.equity / np.maximum.accumulate(np.r_[config.initial_capital, d.equity.to_numpy()])[1:] - 1
    d.attrs["rejections"] = pd.DataFrame(rejects, columns=["date", "signal_time", "bar_end", "reason"])
    d.attrs["cooldown_blocks"] = pd.DataFrame(cooldown_blocks, columns=["date", "signal_time", "direction", "checked_time", "last_exit_time", "cooldown_until", "stage"])
    d.attrs["entry_filter_blocks"] = pd.DataFrame(entry_filter_blocks, columns=["date", "signal_time", "direction", "reason"])
    return d, pd.DataFrame(trades, columns=TRADE_COLUMNS), pd.DataFrame(orders, columns=ORDER_COLUMNS), pd.DataFrame(marks)


def return_metrics(returns):
    r = np.asarray(returns, dtype=float)
    if len(r) == 0:
        return {k: np.nan for k in ("total_return", "cagr", "annualized_volatility", "sharpe", "mdd", "calmar")}
    equity = np.r_[1., np.cumprod(1 + r)]
    total = equity[-1] - 1
    cagr = equity[-1] ** (252 / len(r)) - 1 if equity[-1] > 0 else np.nan
    std = float(np.std(r, ddof=1)) if len(r) > 1 else np.nan
    mdd = float(np.min(equity / np.maximum.accumulate(equity) - 1))
    return {"total_return": total, "cagr": cagr, "annualized_volatility": std * np.sqrt(252), "sharpe": safe_ratio(float(r.mean()) * np.sqrt(252), std), "mdd": mdd, "calmar": safe_ratio(cagr, abs(mdd))}


def summarize(daily, trades, marks, config):
    metrics = return_metrics(daily.daily_return)
    wins, losses = trades[trades.net_pnl.gt(0)], trades[trades.net_pnl.lt(0)]
    gross, slip, fee = daily.gross_pnl.sum(), daily.slippage_cost.sum(), daily.commission_cost.sum()
    metrics.update(days=len(daily), first_date=daily.date.min(), last_date=daily.date.max(), final_equity=daily.equity.iloc[-1], number_of_trades=len(trades), long_trades=int(trades.direction.eq(1).sum()), short_trades=int(trades.direction.eq(-1).sum()), win_rate=trades.net_pnl.gt(0).mean() if len(trades) else np.nan, average_trade_return=trades.trade_return.mean(), median_trade_return=trades.trade_return.median(), average_winning_trade=wins.trade_return.mean(), average_losing_trade=losses.trade_return.mean(), profit_factor=safe_ratio(wins.net_pnl.sum(), abs(losses.net_pnl.sum())), average_holding_minutes=trades.holding_minutes.mean(), median_holding_minutes=trades.holding_minutes.median(), turnover=daily.turnover.sum(), annualized_turnover=daily.turnover.mean() * 252, average_trades_per_day=len(trades) / len(daily), gross_pnl=gross, slippage_cost=slip, commission_cost=fee, net_pnl=daily.net_pnl.sum(), cost_over_gross_pnl=(slip + fee) / gross if gross > 0 else np.nan, order_count=int(daily.order_count.sum()))
    e = np.r_[config.initial_capital, marks.equity.to_numpy()]
    metrics["decision_time_mdd"] = float(np.min(e / np.maximum.accumulate(e) - 1))
    return metrics


def breakdown(trades, column, categories=None):
    rows = []
    groups = categories or sorted(trades[column].unique())
    for name in groups:
        g = trades[trades[column].eq(name)]
        rows.append({column: name, "n": len(g), "avg_trade_return": g.trade_return.mean(), "avg_future_signal_return": g.future_signal_return.mean(), "win_rate": g.net_pnl.gt(0).mean() if len(g) else np.nan, "gross_pnl": g.gross_pnl.sum(), "net_pnl": g.net_pnl.sum()})
    return pd.DataFrame(rows)


def period_summary(d, trades):
    x = d.copy()
    dt = pd.to_datetime(x.date)
    x["year"], x["month"] = dt.dt.strftime("%Y"), dt.dt.strftime("%Y-%m")
    x["half_year"] = dt.dt.strftime("%Y") + "-H" + np.where(dt.dt.month <= 6, "1", "2")
    rows = []
    for kind in ["year", "half_year", "month"]:
        for period, g in x.groupby(kind):
            t = trades[trades.date.isin(g.date)]
            rows.append({"period_type": kind, "period": period, "days": len(g), "trades": len(t), **return_metrics(g.daily_return), "gross_pnl": g.gross_pnl.sum(), "slippage_cost": g.slippage_cost.sum(), "commission_cost": g.commission_cost.sum(), "net_pnl": g.net_pnl.sum(), "win_rate": t.net_pnl.gt(0).mean() if len(t) else np.nan})
    return pd.DataFrame(rows)


def vol_diagnostics(features, d, trades):
    daily_vol = features.groupby("date").vol_bucket.first()
    x = d.copy(); x["vol_bucket"] = x.date.map(daily_vol)
    rows = []
    for bucket in ["Low", "Mid", "High"]:
        t, dd = trades[trades.vol_bucket.eq(bucket)], x[x.vol_bucket.eq(bucket)]
        signal = features[features.date.isin(d.date) & features.vol_bucket.eq(bucket) & features.entry_signal.ne(0)]
        rows.append({"vol_bucket": bucket, "days": len(dd), "signal_count": len(signal), "n": len(t), "avg_trade_return": t.trade_return.mean(), "gross_pnl": t.gross_pnl.sum(), "net_pnl": t.net_pnl.sum(), **return_metrics(dd.daily_return)})
    return pd.DataFrame(rows)




def input_hashes(directory):
    symbols = sorted({s for m in MARKETS.values() for s in m["long"] + m["short"]})
    paths = [directory / f"{s}_daily.csv" for s in symbols] + [directory / "minute" / f"{s}_TOSS_1m.csv" for s in symbols]
    return {str(p.relative_to(directory)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def compressed_csv(frame, path):
    """Atomic gzip CSV output for large minute equity ledgers."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temp, index=False, encoding="utf-8-sig", compression="gzip")
    temp.replace(path)


def benchmark_buy_hold(market, data, days, config):
    """Buy first-day 09:30 OPEN, hold overnight, sell last-day 15:20 CLOSE.

    End-of-day observations match the strategy's frozen evaluation dates.
    Missing/excluded intermediate dates cause no sales or rebalancing.
    """
    symbol = MARKETS[market]['signal']
    minute = data[symbol]['1m'].set_index('timestamp')
    entry_bar = pd.Timestamp(days[0] + ' 09:31')
    exit_bar = pd.Timestamp(days[-1] + ' 15:20')
    entry_ref, exit_ref = float(minute.loc[entry_bar, 'open']), float(minute.loc[exit_bar, 'close'])
    fee, slip = config.commission_bps / 10000, config.slippage_bps / 10000
    entry_fill, exit_fill = entry_ref * (1 + slip), exit_ref * (1 - slip)
    quantity = config.initial_capital / (entry_fill * (1 + fee))
    if config.whole_shares:
        quantity = np.floor(quantity)
    entry_fee, exit_fee = quantity * entry_fill * fee, quantity * exit_fill * fee
    cash = config.initial_capital - quantity * entry_fill - entry_fee
    terminal = cash + quantity * exit_fill - exit_fee
    gross = quantity * (exit_ref - entry_ref)
    slippage = quantity * ((entry_fill - entry_ref) + (exit_ref - exit_fill))
    assert np.isclose(terminal - config.initial_capital, gross - slippage - entry_fee - exit_fee)
    timestamps = pd.to_datetime([day + ' 15:20' for day in days])
    equity = cash + quantity * minute.loc[timestamps, 'close'].to_numpy(dtype=float)
    equity[-1] = terminal
    returns = equity / np.r_[config.initial_capital, equity[:-1]] - 1
    peaks = np.maximum.accumulate(np.r_[config.initial_capital, equity])[1:]
    daily = pd.DataFrame({'date': days, 'equity': equity, 'daily_return': returns,
                          'net_pnl': equity - np.r_[config.initial_capital, equity[:-1]],
                          'cumulative_pnl': equity - config.initial_capital,
                          'drawdown': equity / peaks - 1})
    orders = pd.DataFrame([
        {'symbol': symbol, 'side': 'BUY', 'timestamp': entry_bar - pd.Timedelta(minutes=1),
         'execution_bar_end': entry_bar, 'execution_field': 'open', 'quantity': quantity,
         'reference_price': entry_ref, 'fill_price': entry_fill, 'commission_cost': entry_fee,
         'slippage_cost': quantity * (entry_fill - entry_ref)},
        {'symbol': symbol, 'side': 'SELL', 'timestamp': exit_bar, 'execution_bar_end': exit_bar,
         'execution_field': 'close', 'quantity': quantity, 'reference_price': exit_ref,
         'fill_price': exit_fill, 'commission_cost': exit_fee,
         'slippage_cost': quantity * (exit_ref - exit_fill)}])
    stats = {'market': market, 'benchmark_symbol': symbol, 'series': 'ETF buy and hold',
             **return_metrics(returns), 'final_equity': terminal,
             'net_pnl': terminal - config.initial_capital, 'gross_pnl': gross,
             'commission_cost': entry_fee + exit_fee, 'slippage_cost': slippage,
             'days': len(days), 'first_date': days[0], 'last_date': days[-1],
             'number_of_trades': 1, 'order_count': 2}
    return daily, orders, stats


def plot_final_comparison(market, strategy, benchmark, stats, benchmark_stats, path, config):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
    dates = pd.to_datetime(strategy.date)
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True,
                             gridspec_kw={'height_ratios': [1.4, 1]}, layout='constrained')
    colors = ['#137b69', '#d78227']
    names = ['Final strategy', f"Buy & hold: {MARKETS[market]['signal']} (1x ETF)"]
    for i, (frame, metrics) in enumerate([(strategy, stats), (benchmark, benchmark_stats)]):
        label = f"{names[i]} | return {metrics['total_return']:.2%} | Calmar {metrics['calmar']:.3f}"
        axes[0].plot(dates, (frame.equity - config.initial_capital) / 1e6,
                     color=colors[i], linewidth=1.7, label=label)
        axes[1].plot(dates, frame.drawdown * 100, color=colors[i], linewidth=1.4,
                     label=f"{names[i]} | MDD {metrics['mdd']:.2%}")
    for ax in axes:
        ax.axhline(0, color='#48535b', linewidth=.6)
        ax.grid(alpha=.18)
        ax.spines[['top', 'right']].set_visible(False)
        ax.legend(loc='best', fontsize=9)
    axes[0].set_ylabel('Cumulative net PnL (million KRW)')
    axes[1].set_ylabel('Drawdown from peak (%)')
    axes[0].yaxis.set_major_formatter(FuncFormatter(lambda value, _: f'{value:,.0f}'))
    axes[1].set_xlabel('Date (KST); matched end-of-day observations at 15:20')
    display = 'Long only / one entry per day' if market == 'KOSPI200' else 'Both directions / one entry per day / minute safety stop'
    fig.suptitle(f'{market} | {display}\n{days_label(strategy)} | initial {config.initial_capital/1e6:,.0f}m KRW | fee 1.5bp + slip 1bp per side', fontsize=12)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, facecolor='white')
    plt.close(fig)


def days_label(frame):
    return f'{frame.date.iloc[0]} to {frame.date.iloc[-1]} ({len(frame)} evaluation days)'


def write_strategy_description(out, summary, benchmarks, selected, config):
    stats = summary.set_index('market')
    bh = benchmarks.set_index('market')
    rows = []
    for market in MARKETS:
        for label, values in [('최종 전략', stats.loc[market]), ('ETF 매수 후 보유', bh.loc[market])]:
            rows.append(f"| {market} | {label} | {values.total_return:.2%} | {values.cagr:.2%} | {values.mdd:.2%} | {values.calmar:.3f} | {values.net_pnl/1e4:,.1f} |")
    p, q = stats.loc['KOSPI200'], stats.loc['KOSDAQ150']
    text = f'''# 최종 장중 모멘텀 전략 2개

평가 기간: {p.first_date}~{p.last_date}. 초기자본은 시장별 {config.initial_capital:,.0f}원이다.
최종 전략은 거래당 매수/매도 수수료 각 1.5bp, 불리한 슬리피지 각 1bp를 반영한
실험 중 시장별 Calmar(CAGR / |일별 MDD|)가 가장 높은 전략이다.
코스피와 코스닥을 각각 독립적으로 운용한 결과이며, 두 시장을 합친 포트폴리오 결과는 아니다.

## 성과와 벤치마크

| 시장 | 구분 | 누적수익률 | CAGR | 일별 MDD | Calmar | 누적 순손익(만원) |
|---|---|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

![코스피 전략과 매수 후 보유](KOSPI200/pnl_mdd_vs_buy_hold.png)

![코스닥 전략과 매수 후 보유](KOSDAQ150/pnl_mdd_vs_buy_hold.png)

위 그래프의 위쪽은 초기자본 대비 누적 순손익, 아래쪽은 직전 최고자산 대비 낙폭이다.
MDD는 아래쪽 곡선의 최저점이다. 두 곡선 모두 초기자본을 최초 고점에 포함한다.
전략과 벤치마크는 같은 날짜의 15:20에 자산을 평가한다.
벤치마크는 코스피 **069500 KODEX 200**, 코스닥 **229200 KODEX 코스닥150**의
비레버리지 ETF 가격수익률이다. 지수 자체 또는 분배금 재투자 총수익지수가 아니다.
첫 평가일 09:30의 다음 1분봉 open에 전액 매수하고, 마지막 평가일 15:20 close에 전량 매도한다.
그 사이에는 야간과 중간 제외일에도 계속 보유하며, 재매수·리밸런싱하지 않는다.
실제 매수 1회·매도 1회에 각각 동일한 수수료와 슬리피지를 적용한다.
중간 날짜에는 청산비용을 가상 공제하지 않고 종가로 평가하며, 마지막 날에 실제 청산비용을 반영한다.

CAGR는 기존 선정 기준을 유지하여 `(최종자산/초기자산) ** (252/평가일수) - 1`로 계산했다.
코스피 {int(p.days)}일, 코스닥 {int(q.days)}일이며, 두 시장의 평가 날짜가 서로 다를 수 있다.
특수 개장일·누락 봉·가격 불일치·워밍업 부족일 등은 제외된다. 제외일은 그래프에 새 관측점을 추가하지 않는다.
이 연율화는 달력 경과연수 기반 CAGR와 다르다. 벤치마크에도 동일한 시장별 평가일수를 적용했다.
일별 MDD보다 장중 낙폭이 크다. 전략의 1분봉 close 평가 MDD는
코스피 **{p.decision_time_mdd:.2%}**, 코스닥 **{q.decision_time_mdd:.2%}**다.
1분봉 내 고가·저가를 이용한 최악의 낙폭까지 측정한 값은 아니다.

## 공통 시그널·체결·비중

토스 1분봉을 09:00부터 30개씩 묶어 30분봉을 메모리에서 재구성한다.
신규 진입과 기본 청산 판단은 09:30, 10:00, …, 15:00에만 한다.
오늘 같은 시각의 가격과 당일 시가 간 절대수익률의 과거 **14거래일 평균**을 Noise로 사용한다.
상단밴드 = `max(당일 시가, 전일 종가) × (1+Noise)`;
하단밴드 = `min(당일 시가, 전일 종가) × (1-Noise)`이다.
30분봉 close가 상단을 넘으면 롱, 하단 아래면 숏 방향 시그널이다.
당일 데이터는 Noise와 변동성 추정에 포함하지 않는다.

Strength는 밴드 기준점 대비 돌파 폭을 Noise로 나눈 값이다.
Vol14는 전일까지의 일봉 close 수익률 14개 표본표준편차(ddof=1)이다.
목표 기초지수 노출 `L = min(시장별 상한, 0.02 / Vol14 × Strength)`를 사용한다.
노출은 시그널 때 정하고, ETF 수량은 체결 때 정해 청산까지 고정한다.
{'소수점 수량을 허용한 이상적 시뮬레이션이다. 정수 수량 실험은 --whole-shares로 실행한다.' if not config.whole_shares else '정수 수량만 사용한 시뮬레이션이다.'}
수수료·슬리피지를 살 수량에 미리 반영하여 현금 차입을 허용하지 않는다.

일반 청산은 M1-P 규칙이다. 롱은 `close ≤ max(상단밴드, VWAP)`,
숏은 `close ≥ min(하단밴드, VWAP)`일 때 청산 의도를 만든다.
VWAP은 토스에 거래대금이 없어 **누적 30분봉 HLC3×거래량 / 누적 거래량** 근사치를 사용한다.
일반 진입·청산은 판단 봉이 끝난 직후 다음 1분봉 open에 체결한다.
토스 시간은 봉 종료 시각이므로 09:30 판단은 **09:31 행의 open(실제 09:30)** 체결이다.
거래량 0인 진입 봉은 취소하고, 거래량 0인 청산 봉은 다음 유효 봉까지 연기한다.
미청산 포지션은 요청대로 **15:20 봉 close**에 강제청산한다. 전략은 익일로 포지션을 넘기지 않는다.
매수가는 기준가×1.0001, 매도가는 기준가×0.9999, 수수료는 실제 체결금액×0.00015다.

## 코스피: 롱온리 + 하루 한 번 진입

- 대상: 069500 KODEX 200, 122630 KODEX 레버리지.
- 숏 시그널은 무시한다. 하루 첫 실제 진입만 허용하며, 청산 후 그날 재진입하지 않는다.
- `L ≤ 1`이면 069500에 자본의 L만큼 투자하고 나머지는 현금이다.
- `1 < L ≤ 2`이면 069500 비중 `2-L`, 122630 비중 `L-1`이다. 기초지수 명목노출 상한은 2배다.
- 기본 30분 청산과 15:20 강제청산을 사용한다. 별도 1분 보호청산은 없다.
- 총 {int(p.number_of_trades)}회 진입, 모두 롱이다. 시간 쿨다운은 0분이지만 하루 1회 제한이 재진입을 막는다.

개선의 핵심은 약했던 숏과 첫 거래 이후 추가 거래를 제거한 것이다.
이전 추가 진입 109회의 가격손익은 약 +190.5만원이었으나 거래비용 약 612.8만원으로 순손실 약 422.3만원이었다.
첫 롱 진입을 유지해 누적수익률과 일별 Calmar가 개선되었다.
남은 손실은 롱 돌파 실패, 거래비용, 레버리지 ETF로 확대한 장중 변동에서 발생한다.
평균 진입 목표노출은 {read_mean_leverage(out, 'KOSPI200'):.3f}배다. 일별 MDD가 작아도 장중 위험까지 작다는 뜻은 아니다.

## 코스닥: 양방향 + 하루 한 번 진입 + 1분 보호청산

- 롱은 229200 KODEX 코스닥150, 숏 방향은 251340 KODEX 코스닥150선물인버스를 **매수**한다.
- ETF 공매도는 하지 않는다. 233740 레버리지 ETF는 사용하지 않는다.
- 첫 실제 진입만 허용하고, 목표노출 상한은 1배다. 투자하지 않은 자본은 현금이다.
- 기본 청산에 추가로 매분 close에서 보호청산을 판단한다. 기본 진입 판단은 여전히 30분이다.
- `R = 과거 20거래일 (일봉 high-low)/open의 중앙값`, 보호거리 `d = 0.25×R`이다.
- E는 진입 시각의 신호 ETF **229200** 다음 1분봉 open이다.
  롱은 신호 ETF close가 `E×(1-d)` 이하, 숏은 `E×(1+d)` 이상이면 보호청산 의도를 만든다.
  실제 보유 ETF(229200 또는 251340)를 다음 1분봉 open에 매도한다.
- 총 {int(q.number_of_trades)}회: 롱 {int(q.long_trades)}회, 숏 방향 {int(q.short_trades)}회다.

하루 1회만 진입하므로 보호청산은 신규 진입을 바꾸지 않고 큰 역행 손실의 출구를 앞당긴다.
과거 하루 1회 양방향 버전의 일별 MDD 12.50%가 9.07%, Calmar 0.964가 1.488로 개선되었다.
2023-07-26의 일손익률은 약 -6.27%에서 -1.42%로 줄었다.
보호청산은 모든 거래를 개선하지 않으며, 갭·체결 연기로 손실이 정해진 거리보다 커질 수 있다.
2025년 연간 손익은 여전히 음수였고, 인버스 방향 손실과 높은 거래비용이 남았다.
평균 진입 목표노출은 {read_mean_leverage(out, 'KOSDAQ150'):.3f}배다.

## 보존 파일과 재실행

최종 전략의 보존 대상은 `data/`의 토스 일봉 **4개**와 `data/minute/`의 토스 1분봉 **4개**다.
종목은 069500, 122630, 229200, 251340이다. 원본 CSV를 수정하지 않았으며 SHA256을 확인했다.
원시 1분봉 가용 기간은 2022-11-23~2026-10-07이다. 일봉은 이전 구간도 보존하여 사전 지표 계산에 사용한다.
30분봉·15:20 봉은 1분봉에서 재구성하므로 별도 입력 CSV가 필요하지 않다.
KIS·키움 보조 데이터, 미사용 ETF 데이터, 수집 중간 파일, 합성 테스트 데이터, 이전 실험 결과는 삭제 대상이다.
실제 정리 상태는 아래의 정리 상태 항목과 `cleanup_manifest.json`에서 확인한다.

Python 코드는 기존 수집기와 최종 백테스터 **2개**다. 수집기는 이번 작업에서 실행하지 않았다.
최종 백테스터 기본 실행은 이 두 전략만 돌리고 `results/final/`의 결과를 갱신한다.
새 타임스탬프 실험 폴더를 만들지 않는다.

```powershell
& C:\\Users\\박제형\\AppData\\Local\\Programs\\Python\\Python312\\python.exe .\\jehyeong\\intraday\\code\\backtest_intraday_momentum.py
```

현재 작업 디렉터리와 관계없이 스크립트 위치 기준으로 data와 results를 찾는다.
`--start YYYY-MM-DD --end YYYY-MM-DD`로 평가 기간을 제한할 수 있다.
수집기를 다시 실행하면 원래 7개 ETF의 수집 중간 파일과 파생 데이터가 생성될 수 있다.

`selected_rules.json`은 최종 규칙, 원본 데이터 해시, 선정 시점의 평가 날짜와 참조 성과를 보존한다.
과거 공유 데이터 품질 필터로 제외했던 날짜는 입력을 4종목으로 줄였어도 다시 넣지 않았다.
향후 업데이트된 데이터에서는 참조 구간 종료일 이후의 정상 날짜만 추가한다.
변경된 입력 때문에 과거 평가일이 유효하지 않으면 조용히 다른 표본으로 바꾸지 않고 오류로 중단한다.
`run_manifest.json`과 `verification.json`에 실제 사용 날짜·데이터/코드 해시·검증 결과를 남긴다.
각 시장 폴더에는 일별 손익, 거래/주문 원장, 1분 자산, 기간별 성과, 품질 검사,
벤치마크 일별 자산/주문, PnL/MDD 그래프가 있다. `comparison_summary.csv`가 비교 성과표다.

검증은 종전 참조 성과 재현, 종전 거래/주문 원장 재현, 현금과 손익 일치,
수수료·슬리피지 계산, 하루 진입수·방향·노출 제한, 벤치마크 매수/매도 현금흐름을 포함한다.
최종 전략은 이미 관찰한 전체 기간에서 선정했다. 과거 기간을 나눠 본 결과가 있어도
독립적인 미관찰 구간 검증이나 향후 수익 보장은 아니다.
'''
    cleanup_file = out / 'cleanup_manifest.json'
    if cleanup_file.exists():
        cleanup = json.loads(cleanup_file.read_text(encoding='utf-8-sig'))
        if cleanup.get('status') == 'COMPLETE':
            text += '\n## 정리 상태\n\n정리를 완료했다. 원본 CSV 8개와 코드 2개, 최종 전략 2개의 결과만 보존했다.\n'
        else:
            text += f'''
## 정리 상태: 삭제 미완료

자동 승인 검토가 재귀 일괄 삭제와 단일 CSV 삭제를 모두 정책상 차단했다.
차단 사유는 "blocked by policy"로만 제공되었다. 실제로 삭제한 파일은 **0개**다.
이전 파일들은 아직 intraday에 있으며, 최종 백테스터는 그 이력을 읽지 않는다.
삭제 대상은 {cleanup['planned_deleted_file_count']:,}개 파일, 약 {cleanup['planned_deleted_bytes']/1024**2:,.2f} MiB다.
`cleanup_manifest.json`에 경로·보존 파일·정리 상태를 기록했다.
아래 명령은 사용자가 PowerShell에서 직접 실행할 수 있도록 제공한 것이며, 이번 작업에서 실행하지 않았다.
원본 CSV 8개의 해시와 모든 삭제 경로를 확인한 뒤 지정된 intraday 내부 대상만 제거한다.
'''
            text += r'''
```powershell
$ErrorActionPreference = 'Stop'
$root = (Resolve-Path -LiteralPath 'C:\Users\박제형\Desktop\Future_Trading\jehyeong\intraday').Path
$planPath = Join-Path $root 'results/final/cleanup_manifest.json'
$plan = Get-Content -LiteralPath $planPath -Raw -Encoding UTF8 | ConvertFrom-Json
if ($root -ne $plan.root) { throw 'Cleanup root mismatch' }
$prefix = $root.TrimEnd('\') + '\'
$items = @(Get-ChildItem -LiteralPath $root -Recurse -Force)
if (@($items | Where-Object { $_.Attributes -band [IO.FileAttributes]::ReparsePoint }).Count) {
    throw 'Reparse point detected; cleanup stopped'
}
$rules = Get-Content -LiteralPath (Join-Path $root 'results/final/selected_rules.json') -Raw -Encoding UTF8 | ConvertFrom-Json
foreach ($entry in $rules.input_sha256.PSObject.Properties) {
    $file = Join-Path (Join-Path $root 'data') $entry.Name
    if ((Get-FileHash -LiteralPath $file -Algorithm SHA256).Hash -ne $entry.Value) {
        throw "Input hash changed: $file"
    }
}
$targets = @($plan.deleted_targets | ForEach-Object {
    $path = Join-Path $root $_
    if (Test-Path -LiteralPath $path) {
        $resolved = (Resolve-Path -LiteralPath $path).Path
        if (-not $resolved.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase)) {
            throw "Unsafe target: $resolved"
        }
        $resolved
    }
})
foreach ($path in $targets) { Remove-Item -LiteralPath $path -Recurse -Force }
$plan.status = 'COMPLETE'
$plan.actual_deleted_file_count = $plan.planned_deleted_file_count
$plan.blocked_reason = $null
$plan | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $planPath -Encoding UTF8
Write-Output 'Cleanup complete. Rerun the final backtester to refresh this document.'
```
'''
    (out / '최종_전략_설명.md').write_text(text, encoding='utf-8', newline='\n')


def read_mean_leverage(out, market):
    return pd.read_csv(out / market / 'trades.csv').leverage.mean()


def run_selected_strategies(selection_file, config, start=None, end=None):
    """Reproduce only the final selections in one fixed result directory."""
    import gzip
    selected = json.loads(selection_file.read_text(encoding='utf-8'))
    if set(selected['markets']) != set(MARKETS):
        raise ValueError('Selection requires exactly the two final markets')
    expected = {'KOSPI200': (1, None), 'KOSDAQ150': (0, .25)}
    for market, rule in selected['markets'].items():
        options = rule['options']
        if (rule['model'] != 'M1-P' or options['max_daily_entries'] != 1 or
            (options['direction_filter'], options['minute_safety_stop_k']) != expected[market] or
            options['cooldown_minutes'] != 0 or options['sizing_mode'] != 'full' or
            options['exit_policy'] != 'normal' or options['entry_confirmation'] != 'none' or
            options['exposure_cap'] is not None or options['entry_cutoff'] is not None):
            raise ValueError(f'{market}: this executable supports only the frozen final strategy')
    if config.commission_bps != 1.5 or config.slippage_bps != 1.:
        raise ValueError('Final strategies require fee1.5bp and slip1bp per side')
    out = RESULTS / 'final'
    out.mkdir(parents=True, exist_ok=True)
    hashes = input_hashes(DATA)
    source = Path(__file__).read_bytes()
    (out / 'executed_source.txt.gz').write_bytes(gzip.compress(source, mtime=0))
    manifest = {'status': 'RUNNING', 'created_kst': str(pd.Timestamp.now(tz='Asia/Seoul')),
                'selection_file': str(selection_file), 'selected_rules': selected['markets'],
                'selection_provenance': selected['selection_provenance'], 'config': vars(config),
                'input_sha256': hashes, 'code_sha256': hashlib.sha256(source).hexdigest(),
                'sample_dates': {}, 'benchmark': 'Signal ETF buy first day 09:30 open, hold overnight, sell last day 15:20 close; same costs and evaluation dates; excludes distributions',
                'historical_sample_policy': selected['historical_sample_policy']}
    save_json(out / 'run_manifest.json', manifest)
    try:
        data = load_data(DATA)
        rows, benchmark_rows, comparison_rows, checks = [], [], [], []
        for market in MARKETS:
            rule = selected['markets'][market]
            signal = MARKETS[market]['signal']
            features = make_features(data[signal]['1m'], data[signal]['daily'])
            valid, audit = eligible_days(market, data, features, start, end)
            reference_days = selected['sample_dates'][market]
            historical_end = reference_days[-1]
            required_days = [day for day in reference_days
                             if (not start or day >= start) and (not end or day <= end)]
            missing = sorted(set(required_days) - set(valid))
            if missing:
                raise ValueError(f'{market}: invalid frozen historical dates: {missing[:10]}')
            days = sorted(required_days + [day for day in valid if day > historical_end])
            if not days:
                raise ValueError(f'{market}: no evaluation dates requested')
            legacy_excluded = audit.eligible & audit.date.le(historical_end) & ~audit.date.isin(reference_days)
            audit.loc[legacy_excluded, 'eligible'] = False
            audit.loc[legacy_excluded, 'reason'] = 'frozen_original_shared_coverage_exclusion'
            audit['used_in_final_sample'] = audit.date.isin(days)
            features, _ = add_vol_buckets(features, days)
            manifest['sample_dates'][market] = days
            d, t, o, m = simulate(market, rule['model'], features, data, days, VERSIONS[0], config, **rule['options'])
            stats = {'market': market, 'scenario': rule['scenario'], 'model': rule['model'],
                     'execution_version': VERSIONS[0], 'options': json.dumps(rule['options'], sort_keys=True),
                     'commission_bps': 1.5, 'slippage_bps': 1., **summarize(d, t, m, config)}
            assert np.isclose(t.net_pnl.sum(), d.equity.iloc[-1] - config.initial_capital)
            assert np.allclose(o.commission_cost, o.filled_notional * .00015)
            assert np.allclose(o.slippage_cost, o.reference_notional * .0001)
            assert t.groupby('date').size().le(1).all()
            if market == 'KOSPI200':
                assert t.direction.eq(1).all()
            assert t.leverage.le(MARKETS[market]['ceiling']).all()
            if hashes == selected['input_sha256'] and days == reference_days and vars(config) == selected['config']:
                for key, value in rule['reference_metrics'].items():
                    assert np.isclose(stats[key], value, rtol=1e-9, atol=1e-9), (market, key)
                checks.append(f'{market}: selected historical performance reproduced exactly')
            checks.append(f'{market}: PnL, fee, slippage, entry count, direction and exposure verified')
            rows.append(stats)
            dest = out / market
            csv(d, dest / 'daily_pnl.csv')
            csv(t, dest / 'trades.csv')
            csv(o, dest / 'orders.csv')
            compressed_csv(m, dest / 'minute_equity.csv.gz')
            csv(audit, dest / 'data_quality_audit.csv')
            csv(period_summary(d, t), dest / 'period_summary.csv')
            csv(d.attrs['rejections'], dest / 'execution_rejections.csv')
            b, bo, bs = benchmark_buy_hold(market, data, days, config)
            assert list(b.date) == list(d.date) and len(bo) == 2
            checks.append(f'{market}: buy-and-hold cash flow, costs, exactly two orders and matched dates verified')
            csv(b, dest / 'benchmark_daily.csv')
            csv(bo, dest / 'benchmark_orders.csv')
            benchmark_rows.append(bs)
            comparison_rows.extend([{'series': 'Final strategy', **stats}, bs])
            plot_final_comparison(market, d, b, stats, bs, dest / 'pnl_mdd_vs_buy_hold.png', config)
            print(f'[FINAL {market}] strategy return={stats["total_return"]:.2%} MDD={stats["mdd"]:.2%} Calmar={stats["calmar"]:.3f}; B&H return={bs["total_return"]:.2%} MDD={bs["mdd"]:.2%}', flush=True)
        summary, benchmarks = pd.DataFrame(rows), pd.DataFrame(benchmark_rows)
        csv(summary, out / 'performance_summary.csv')
        csv(pd.DataFrame(comparison_rows), out / 'comparison_summary.csv')
        write_strategy_description(out, summary, benchmarks, selected, config)
        save_json(out / 'verification.json', {'status': 'PASS', 'checks': checks, 'network_calls': 0})
        manifest['status'] = 'COMPLETE'
        save_json(out / 'run_manifest.json', manifest)
        print(f'Saved final strategies: {out}', flush=True)
        return out
    except Exception as exc:
        manifest.update(status='INCOMPLETE', error_type=type(exc).__name__, error=str(exc))
        save_json(out / 'run_manifest.json', manifest)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--start')
    parser.add_argument('--end')
    parser.add_argument('--initial-capital', type=float, default=100_000_000.)
    parser.add_argument('--commission-bps', type=float, default=1.5)
    parser.add_argument('--slippage-bps', type=float, default=1.)
    parser.add_argument('--whole-shares', action='store_true')
    parser.add_argument('--run-selected', action='store_true', help='Compatibility alias; final strategies are now the default')
    parser.add_argument('--selection-file', type=Path, default=RESULTS / 'final' / 'selected_rules.json')
    args = parser.parse_args()
    if not np.isfinite(args.initial_capital) or args.initial_capital <= 0:
        parser.error('Initial capital must be positive and finite')
    for name in ['start', 'end']:
        if getattr(args, name):
            setattr(args, name, pd.Timestamp(getattr(args, name)).strftime('%Y-%m-%d'))
    if args.start and args.end and args.start > args.end:
        parser.error('start after end')
    config = Config(args.initial_capital, args.commission_bps, args.whole_shares, args.slippage_bps)
    run_selected_strategies(args.selection_file.resolve(), config, args.start, args.end)


if __name__ == '__main__':
    main()


