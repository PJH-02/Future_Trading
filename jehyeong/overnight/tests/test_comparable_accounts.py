"""Financial accounting invariants for overnight/intraday comparisons."""

from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jehyeong.overnight.jehyeong_backtest_utils import account_returns, performance_metrics, ROUND_TRIP_COST
from jehyeong.overnight.compare_committed_strategies import etf_weights


class ComparableAccountsTests(unittest.TestCase):
    def test_accounts_start_in_cash_and_reject_exit_after_period(self):
        calendar = pd.to_datetime(["2024-06-14", "2024-06-17", "2024-06-18", "2024-06-19"])
        trades = pd.DataFrame({
            "entry_date": calendar[:3], "exit_date": calendar[1:],
            "net_return": [0.10, 0.02, 0.30],
        })
        daily, selected = account_returns(trades, calendar, "2024-06-17", "2024-06-18")
        self.assertEqual(len(selected), 1)
        np.testing.assert_allclose(daily, [0.0, 0.02])
        self.assertEqual(selected.iloc[0]["entry_date"], pd.Timestamp("2024-06-17"))

    def test_exit_calendar_missing_is_an_error(self):
        trades = pd.DataFrame({
            "entry_date": pd.to_datetime(["2024-06-14"]),
            "exit_date": pd.to_datetime(["2024-06-15"]), "net_return": [0.01],
        })
        with self.assertRaisesRegex(ValueError, "exit date"):
            account_returns(trades, pd.to_datetime(["2024-06-14", "2024-06-17"]), "2024-06-14", "2024-06-17")

    def test_metrics_include_idle_days_and_initial_capital_drawdown(self):
        r = pd.Series([-0.10, 0.0, 0.20], index=pd.bdate_range("2024-06-10", periods=3))
        trades = pd.DataFrame({"net_return": [-0.10, 0.20]})
        metrics = performance_metrics(r, trades)
        self.assertAlmostEqual(metrics["MDD"], -0.10)
        self.assertAlmostEqual(metrics["total_return"], 0.08)
        self.assertAlmostEqual(metrics["CAGR"], 1.08 ** (252 / 3) - 1.0)
        self.assertAlmostEqual(metrics["Sharpe"], r.mean() / r.std(ddof=1) * np.sqrt(252))
        self.assertAlmostEqual(metrics["Calmar"], metrics["CAGR"] / 0.10)
        self.assertEqual(metrics["days"], 3)
        self.assertEqual(metrics["trades"], 2)

    def test_leveraged_etf_cost_uses_cash_turnover_not_index_exposure(self):
        w = pd.Series([0.0, 0.5, 1.0, 1.5, 2.0])
        a, b = etf_weights(w)
        np.testing.assert_allclose(a, [0.0, 0.5, 1.0, 0.5, 0.0])
        np.testing.assert_allclose(b, [0.0, 0.0, 0.0, 0.5, 1.0])
        np.testing.assert_allclose(a + 2 * b, w)
        np.testing.assert_allclose((a + b) * ROUND_TRIP_COST, [0, 0.00025, 0.0005, 0.0005, 0.0005])


if __name__ == "__main__":
    unittest.main()
