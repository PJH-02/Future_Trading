"""Clock, shadow-account ownership, and actual inverse-ETF execution tests."""
import unittest
from types import SimpleNamespace

import numpy as np
import pandas as pd

import backtest_daynight as base
from backtest_v3_v4 import execution_plan, shadow_gates, build_v4_plans, corrected_simulate


class ExtensionTests(unittest.TestCase):
    def market(self, dates):
        rows = {}
        for day in dates:
            for symbol in ("233740", "251340"):
                for end in ("09:06", "15:01"):
                    price = 100. if end == "09:06" else (120. if symbol == "233740" else 90.)
                    rows[symbol, base.stamp(day, end)] = pd.Series(dict(timestamp=base.stamp(day, end), open=105. if end == "09:06" else price, close=price, volume=10))
        return SimpleNamespace(bar=lambda symbol, time: rows.get((symbol, time)), next={}, groups={"233740": {}, "251340": {}})

    def test_v3_entry_is_close_not_open_of_correct_source_bar(self):
        dates = pd.to_datetime(["2024-01-02"])
        p, reason = execution_plan(self.market(dates), dates, dates[0], "intraday", 1)
        self.assertEqual(reason, "")
        self.assertEqual(p["entry_reference"], 100.)
        self.assertEqual(p["entry_reference_bar_end"], base.stamp(dates[0], "09:06"))
        self.assertEqual(p["entry_time"], base.stamp(dates[0], "09:06"))
        d, o, _ = corrected_simulate("V3", "intraday", dates, {("V3", dates[0]): (p, None)}, base.Config(initial_capital=1000, commission_bps=0, slippage_bps=0))
        self.assertEqual(d.nav.iloc[-1], 1200.)
        self.assertEqual(o.reference_field.iloc[0], "close")

    def test_inversion_executes_inverse_etf_not_negative_normal_return(self):
        dates = pd.to_datetime(["2024-01-02"])
        p, _ = execution_plan(self.market(dates), dates, dates[0], "intraday", 1, -1)
        self.assertEqual(p["symbol"], "251340")
        self.assertEqual(p["base_direction"], 1)
        self.assertEqual(p["direction"], -1)
        d, _, _ = corrected_simulate("V4", "intraday", dates, {("V4", dates[0]): (p, None)}, base.Config(initial_capital=1000, commission_bps=0, slippage_bps=0))
        self.assertAlmostEqual(d.nav.iloc[-1], 900.)
        self.assertNotEqual(d.nav.iloc[-1], 800.)

    def test_per_leg_exit_clock_and_combined_actual_pnl_are_separate(self):
        dates = pd.bdate_range("2024-01-02", periods=21)
        records = []
        for account, hour in [("intraday", 15), ("overnight", 9)]:
            records.append(dict(account=account, exit_time=dates[19] + pd.Timedelta(hours=hour), exit_equity=90.))
        # Deliberately conflicting combined P&L must have no effect on either gate.
        records.append(dict(account="combined", exit_time=dates[19]+pd.Timedelta(hours=15), exit_equity=9999999.))
        gates = shadow_gates(dates, pd.DataFrame(records), initial_capital=100.)
        g = gates.set_index(["date", "strategy"])
        self.assertEqual(g.loc[(dates[19], "intraday"), "entry_mode"], 1)
        self.assertEqual(g.loc[(dates[19], "overnight"), "entry_mode"], -1)
        self.assertEqual(g.loc[(dates[20], "intraday"), "entry_mode"], -1)
        self.assertEqual(g.loc[(dates[19], "intraday"), "reference_pnl"], -10.)
        self.assertEqual(g.loc[(dates[19], "overnight"), "reference_pnl"], -10.)
        self.assertAlmostEqual(g.loc[(dates[19], "intraday"), "daily_ma20"], -.5)

    def test_reversed_instrument_has_independent_execution_eligibility(self):
        dates = pd.to_datetime(["2024-01-02"])
        market = self.market(dates)
        original = market.bar
        market.bar = lambda symbol, time: None if symbol == "233740" else original(symbol, time)
        normal, _ = execution_plan(market, dates, dates[0], "intraday", 1, 1)
        inverted, _ = execution_plan(market, dates, dates[0], "intraday", 1, -1)
        self.assertIsNone(normal)
        self.assertIsNotNone(inverted)

    def test_empty_normal_accounts_still_produce_cash_warmup_gates(self):
        dates = pd.bdate_range("2024-01-02", periods=21)
        plans = {("V3", d): (None, None) for d in dates}
        _, _, trades = corrected_simulate("V3", "intraday", dates, plans, base.Config())
        gates = shadow_gates(dates, trades, initial_capital=100.)
        self.assertEqual(len(gates), 42)
        self.assertTrue(gates.entry_mode.eq(1).all())
        self.assertTrue(gates.reference_pnl.eq(0).all())


if __name__ == "__main__":
    unittest.main()
