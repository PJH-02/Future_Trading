"""Synthetic behavioral checks; the real-data independent audit is separate."""
import unittest
from types import SimpleNamespace
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

from backtest_daynight import (Config, Market, buy, sell, day_direction, ibs_direction,
                              ibs_window, minute_start, prior_features, next_sessions,
                              make_plans, metrics, performance_tables, simulate, stamp, validate_frame)


def plan(day, strategy="intraday", entry=100., exit_=110., next_day=None, mark=100.):
    entry_time = stamp(day, "09:06" if strategy == "intraday" else "15:20")
    exit_time = stamp(day if strategy == "intraday" else next_day, "15:00" if strategy == "intraday" else "09:00")
    return dict(strategy=strategy, symbol="233740", direction=1, entry_time=entry_time, exit_time=exit_time,
                entry_reference_bar_end=entry_time + pd.Timedelta(minutes=1),
                exit_reference_bar_end=exit_time + pd.Timedelta(minutes=1), entry_reference=entry,
                exit_reference=exit_, mark_close=mark, signal_available_time=entry_time)


def minute_frame(day):
    t = pd.date_range(stamp(day, "09:01"), stamp(day, "15:30"), freq="min")
    return pd.DataFrame(dict(timestamp=t, date=pd.Timestamp(day), open=5., high=10., low=0., close=1., volume=1.))


class SignalTests(unittest.TestCase):
    def test_ma10_completed_sessions_and_future_invariance(self):
        dates = pd.bdate_range("2023-01-02", periods=13)
        frame = pd.DataFrame({"close": np.arange(1., 14.)}, index=dates)
        a = prior_features(frame)
        self.assertTrue(np.isnan(a.ma10.iloc[9]))
        self.assertEqual(a.ma10.iloc[10], 5.5)
        self.assertEqual(a.previous_close.iloc[10], 10)
        changed = frame.copy()
        changed.loc[dates[10]:, "close"] = 999999
        self.assertEqual(prior_features(changed).loc[dates[10], "ma10"], a.ma10.iloc[10])

    def test_inclusive_momentum_boundary_and_regime(self):
        self.assertEqual(day_direction(11, 10, 100, 100.3), 1)
        self.assertEqual(day_direction(9, 10, 100, 99.7), -1)
        self.assertEqual(day_direction(11, 10, 100, 100.299), 0)
        self.assertEqual(day_direction(9, 10, 100, 99.701), 0)
        self.assertEqual(day_direction(10, 10, 100, 101), 0)
        self.assertEqual(day_direction(9, 10, 100, 101), 0)
        self.assertEqual(day_direction(11, 10, 100, 99), 0)

    def test_ibs_strict_boundaries_and_zero_range(self):
        self.assertEqual(ibs_direction(3, 10, 0), (.3, 0))
        self.assertEqual(ibs_direction(7, 10, 0), (.7, 0))
        self.assertEqual(ibs_direction(2, 10, 0)[1], 1)
        self.assertEqual(ibs_direction(8, 10, 0)[1], -1)
        self.assertTrue(np.isnan(ibs_direction(5, 5, 5)[0]))

    def test_bar_end_conversion(self):
        self.assertEqual(minute_start("2024-01-02 09:07"), pd.Timestamp("2024-01-02 09:06"))
        self.assertEqual(minute_start("2024-01-02 15:20"), pd.Timestamp("2024-01-02 15:19"))

    def test_ibs_cutoff_and_future_mutation(self):
        day = pd.Timestamp("2024-01-02")
        frame = minute_frame(day)
        frame.loc[frame.timestamp.eq(stamp(day, "15:20")), "close"] = 9
        self.assertEqual(ibs_window(frame, day, "15:15")[:2], (.1, 1))
        self.assertEqual(ibs_window(frame, day, "15:19")[:2], (.9, -1))
        frame.loc[frame.timestamp.gt(stamp(day, "15:16")), "high"] = 1000
        self.assertEqual(ibs_window(frame, day, "15:15")[:2], (.1, 1))

    def test_incomplete_signal_window_is_rejected(self):
        frame = minute_frame("2024-01-02").drop(index=10)
        self.assertEqual(ibs_window(frame, pd.Timestamp("2024-01-02"), "15:15")[2], "incomplete_ibs_window_or_delayed_open")

    def test_next_calendar_session_does_not_jump_missing_execution(self):
        dates = pd.to_datetime(["2024-01-05", "2024-01-08", "2024-01-09"])
        self.assertEqual(next_sessions(dates)[dates[0]], dates[1])
        frames = {d: minute_frame(d) for d in dates}
        groups = {s: {d: f.copy() for d, f in frames.items()} for s in ("229200", "233740", "251340")}
        lookup = {(s, t): row for s in groups for frame in groups[s].values() for t, row in frame.set_index("timestamp", drop=False).iterrows()}
        lookup.pop(("233740", stamp(dates[1], "09:01")))
        market = SimpleNamespace(groups=groups, features=pd.DataFrame({"previous_close": 10., "ma10": 9.}, index=dates),
                                 next=next_sessions(dates), bar=lambda s, t: lookup.get((s, t)))
        plans, signals, skips = make_plans(market, dates)
        self.assertIsNone(plans["V1", dates[0]][1])
        self.assertIn("missing_or_zero_volume_next_session_0900_exit", set(skips.reason))
        self.assertEqual(plans["V1", dates[1]][1]["exit_time"], stamp(dates[2], "09:00"))
        self.assertIsNone(plans["V1", dates[-1]][1])

    def test_proxy_uses_latest_positive_volume_close_not_stale_open(self):
        dates = pd.to_datetime(["2024-01-02", "2024-01-03"])
        groups = {s: {d: minute_frame(d) for d in dates} for s in ("229200", "233740", "251340")}
        g = groups["233740"][dates[0]]
        g.loc[g.timestamp.eq(stamp(dates[0], "15:20")), ["close", "volume"]] = [123., 1.]
        g.loc[g.timestamp.eq(stamp(dates[0], "15:21")), ["open", "volume"]] = [777., 0.]
        lookup = {(s, t): row for s in groups for f in groups[s].values() for t, row in f.set_index("timestamp", drop=False).iterrows()}
        market = SimpleNamespace(groups=groups, features=pd.DataFrame({"previous_close": 10., "ma10": 9.}, index=dates), next=next_sessions(dates), bar=lambda s, t: lookup.get((s, t)))
        plans, _, _ = make_plans(market, dates)
        self.assertEqual(plans["V1", dates[0]][1]["entry_reference"], 123.)
        self.assertEqual(plans["V1", dates[0]][1]["entry_reference_bar_end"], stamp(dates[0], "15:20"))


class AccountingTests(unittest.TestCase):
    def test_cash_sizing_and_exact_cost(self):
        config = Config(initial_capital=10000)
        q, fill, fee = buy(10000, 100, config)
        self.assertAlmostEqual(q * fill + fee, 10000)
        proceeds, _, _ = sell(q, 100, config)
        expected = 10000 * .9999 * .99985 / (1.0001 * 1.00015)
        self.assertAlmostEqual(proceeds, expected)
        self.assertAlmostEqual((1 - proceeds / 10000) * 10000, 5, delta=.002)

    def test_combined_account_sequential_compound_not_average(self):
        dates = pd.to_datetime(["2024-01-02", "2024-01-03"])
        plans = {("V1", dates[0]): (plan(dates[0]), plan(dates[0], "overnight", exit_=120, next_day=dates[1], mark=110)),
                 ("V1", dates[1]): (plan(dates[1], exit_=90), None)}
        config = Config(initial_capital=1000, commission_bps=0, slippage_bps=0)
        nav, orders, trades = simulate("V1", "combined", dates, plans, config)
        self.assertAlmostEqual(nav.nav.iloc[-1], 1000 * 1.1 * 1.2 * .9)
        self.assertEqual(len(trades), 3)
        self.assertTrue(orders.timestamp.is_monotonic_increasing)
        self.assertEqual(list(orders.side), ["BUY", "SELL", "BUY", "SELL", "BUY", "SELL"])

    def test_cross_year_mark_to_market_and_turnover(self):
        dates = pd.to_datetime(["2023-12-28", "2024-01-02"])
        plans = {("V1", dates[0]): (None, plan(dates[0], "overnight", exit_=120, next_day=dates[1], mark=110)),
                 ("V1", dates[1]): (None, None)}
        nav, orders, trades = simulate("V1", "overnight", dates, plans, Config(initial_capital=1000, commission_bps=0, slippage_bps=0))
        self.assertAlmostEqual(nav.nav.iloc[0], 1100)
        self.assertAlmostEqual(nav.daily_return.iloc[0], .1)
        self.assertAlmostEqual(nav.daily_return.iloc[1], 1200 / 1100 - 1)
        self.assertAlmostEqual(nav.turnover.iloc[0], 1.)
        self.assertAlmostEqual(nav.turnover.iloc[1], 1200 / 1100)
        summary, yearly = performance_tables(nav, trades)
        self.assertAlmostEqual(yearly.total_return.iloc[0], .1)
        self.assertAlmostEqual(yearly.total_return.iloc[1], 1200 / 1100 - 1)
        self.assertAlmostEqual(np.prod(1 + yearly.total_return) - 1, .2)
        self.assertAlmostEqual(summary.turnover.iloc[0], 1 + 1200 / 1100)

    def test_initial_capital_counts_for_drawdown_and_cash_days_for_sharpe(self):
        dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
        frame = pd.DataFrame(dict(date=dates, previous_nav=[100, 90, 90], nav=[90, 90, 99],
                                  daily_return=[-.1, 0, .1], turnover=[2., 0, 2.], orders=[2, 0, 2]))
        m = metrics(frame, pd.DataFrame())
        self.assertAlmostEqual(m["mdd"], -.1)
        self.assertAlmostEqual(m["total_return"], -.01)
        self.assertAlmostEqual(m["sharpe"], 0)
        self.assertEqual(m["sessions"], 3)
        self.assertEqual(m["annualized_turnover"], 4 * 252 / 3)

    def test_zero_trade_account_remains_cash(self):
        dates = pd.bdate_range("2024-01-02", periods=2)
        plans = {("V1", d): (None, None) for d in dates}
        nav, orders, trades = simulate("V1", "combined", dates, plans, Config())
        self.assertTrue(nav.nav.eq(100000000).all())
        self.assertTrue(orders.empty)
        self.assertTrue(trades.empty)
        self.assertTrue(np.isnan(metrics(nav, trades)["sharpe"]))

    def test_invalid_configuration(self):
        for initial in (0, -1, np.inf):
            with self.assertRaises(ValueError):
                Config(initial_capital=initial)
        with self.assertRaises(ValueError):
            Config(start="2026-01-01", end="2025-01-01")
        for value in (-1, np.nan, 10000):
            with self.assertRaises(ValueError):
                Config(slippage_bps=value)


class ValidationTests(unittest.TestCase):
    def frame(self):
        return pd.DataFrame(dict(timestamp=["2024-01-02 09:01:00"], date=["2024-01-02"], symbol=["229200"], source=["TOSS"],
                                 timestamp_semantics=["bar_end"], open=[100], high=[102], low=[99], close=[101], volume=[10]))

    def test_duplicate_wrong_semantics_and_bad_ohlc_rejected(self):
        frame = self.frame()
        self.assertEqual(validate_frame(frame, "229200", True).bar_start.iloc[0], pd.Timestamp("2024-01-02 09:00"))
        for bad in (pd.concat([frame, frame]), frame.assign(timestamp_semantics="bar_start"), frame.assign(high=90), frame.assign(volume=-1), frame.assign(date="2024-01-03")):
            with self.assertRaises(ValueError):
                validate_frame(bad, "229200", True)

    def test_loaded_bar_preserves_original_label_for_execution_audit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "minute").mkdir()
            for symbol in ("229200", "233740", "251340"):
                frame = self.frame().assign(symbol=symbol)
                frame.to_csv(root / "minute" / f"{symbol}_TOSS_1m.csv", index=False)
                frame.drop(columns="timestamp_semantics").assign(timestamp="2024-01-02").to_csv(root / f"{symbol}_daily.csv", index=False)
            market = Market(root)
            bar = market.bar("233740", pd.Timestamp("2024-01-02 09:01"))
            self.assertEqual(bar.timestamp, pd.Timestamp("2024-01-02 09:01"))
            self.assertEqual(bar.bar_start, pd.Timestamp("2024-01-02 09:00"))


if __name__ == "__main__":
    unittest.main()
