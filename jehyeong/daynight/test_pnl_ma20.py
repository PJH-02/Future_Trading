import math
import unittest

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from pnl_ma20 import COLUMNS, build_gate


def calendar(n, start="2024-01-01"):
    return pd.date_range(start, periods=n, freq="D")


def exits_for(cal, values, start=0):
    return pd.DataFrame(
        {
            "exit_time": cal[list(range(start, start + len(values)))],
            "reference_pnl": values,
        }
    )


class TestBuildGate(unittest.TestCase):
    def test_warmup_and_twentieth_observation(self):
        cal = calendar(25)
        values = list(range(1, 20)) + [0]
        result = build_gate(cal, exits_for(cal, values), window=20)

        self.assertEqual(list(result.columns), COLUMNS)
        for i in range(19):
            self.assertTrue(math.isnan(result.loc[i, "ma20"]))
            self.assertEqual(result.loc[i, "mode_after_exit"], 1)
            self.assertEqual(result.loc[i, "observations"], i + 1)

        twentieth = result.loc[19]
        self.assertEqual(twentieth["mode_before_exit"], 1)
        self.assertEqual(twentieth["mode_after_exit"], -1)
        self.assertAlmostEqual(twentieth["ma20"], 9.5)
        self.assertEqual(twentieth["observations"], 20)
        self.assertTrue(twentieth["has_exit"])

        for i in range(20, 25):
            self.assertFalse(result.loc[i, "has_exit"])
            self.assertEqual(result.loc[i, "mode_before_exit"], -1)
            self.assertEqual(result.loc[i, "mode_after_exit"], -1)
            self.assertAlmostEqual(result.loc[i, "ma20"], 9.5)
            self.assertEqual(result.loc[i, "reference_pnl"], 0.0)

    def test_strict_less_and_equality_restore(self):
        cal = calendar(5)
        result = build_gate(cal, exits_for(cal, [10, 10, 10, 0, 5]), window=3)

        self.assertEqual(result.loc[2, "mode_after_exit"], 1)
        self.assertAlmostEqual(result.loc[2, "ma20"], 10.0)

        self.assertEqual(result.loc[3, "mode_before_exit"], 1)
        self.assertEqual(result.loc[3, "mode_after_exit"], -1)
        self.assertAlmostEqual(result.loc[3, "ma20"], 20.0 / 3.0)

        self.assertEqual(result.loc[4, "mode_before_exit"], -1)
        self.assertEqual(result.loc[4, "mode_after_exit"], 1)
        self.assertAlmostEqual(result.loc[4, "ma20"], 5.0)

    def test_nontrade_flat_sessions_count_in_lookback(self):
        cal = calendar(4)
        exits = pd.DataFrame(
            {
                "exit_time": [cal[0], cal[3]],
                "reference_pnl": [10.0, 5.0],
            }
        )
        result = build_gate(cal, exits, window=3)

        for i in (1, 2):
            self.assertFalse(result.loc[i, "has_exit"])
            self.assertEqual(result.loc[i, "reference_pnl"], 10.0)
            self.assertTrue(math.isnan(result.loc[i, "ma20"]))
            self.assertEqual(result.loc[i, "mode_after_exit"], 1)

        self.assertEqual(result.loc[3, "observations"], 3)
        self.assertEqual(result.loc[3, "mode_before_exit"], 1)
        self.assertEqual(result.loc[3, "mode_after_exit"], -1)
        self.assertAlmostEqual(result.loc[3, "ma20"], 25.0 / 3.0)

    def test_no_exit_carries_mode_and_ma(self):
        cal = calendar(4)
        exits = exits_for(cal, [1, 10, 0])
        result = build_gate(cal, exits, window=3)

        ma_at_exit = result.loc[2, "ma20"]
        self.assertEqual(result.loc[2, "mode_after_exit"], -1)
        self.assertAlmostEqual(ma_at_exit, 11.0 / 3.0)

        no_exit = result.loc[3]
        self.assertFalse(no_exit["has_exit"])
        self.assertEqual(no_exit["mode_before_exit"], -1)
        self.assertEqual(no_exit["mode_after_exit"], -1)
        self.assertAlmostEqual(no_exit["ma20"], ma_at_exit)
        self.assertEqual(no_exit["reference_pnl"], 0.0)
        self.assertEqual(no_exit["observations"], 3)

    def test_past_rows_invariant_to_future_exit_changes(self):
        cal = calendar(6)
        base = build_gate(cal, exits_for(cal, [1, 2, 3, 4, 5]), window=3)
        altered = build_gate(cal, exits_for(cal, [1, 2, 3, 4, 100]), window=3)

        assert_frame_equal(base.iloc[:4].reset_index(drop=True),
                           altered.iloc[:4].reset_index(drop=True))
        self.assertNotEqual(base.loc[4, "reference_pnl"],
                            altered.loc[4, "reference_pnl"])

    def test_empty_exits(self):
        cal = calendar(3)
        exits = pd.DataFrame(columns=["exit_time", "reference_pnl"])
        result = build_gate(cal, exits, window=20)

        self.assertEqual(len(result), 3)
        self.assertTrue((result["reference_pnl"] == 0.0).all())
        self.assertTrue(result["ma20"].isna().all())
        self.assertFalse(result["has_exit"].any())
        self.assertTrue((result["mode_before_exit"] == 1).all())
        self.assertTrue((result["mode_after_exit"] == 1).all())
        self.assertEqual(list(result["observations"]), [1, 2, 3])

    def test_empty_calendar(self):
        cal = pd.DatetimeIndex([])
        result = build_gate(cal, pd.DataFrame(columns=["exit_time", "reference_pnl"]))
        self.assertEqual(len(result), 0)
        self.assertEqual(list(result.columns), COLUMNS)

    def test_duplicate_exit_rejected(self):
        cal = calendar(3)
        exits = pd.DataFrame(
            {
                "exit_time": [cal[1], cal[1]],
                "reference_pnl": [1.0, 2.0],
            }
        )
        with self.assertRaises(ValueError):
            build_gate(cal, exits)

    def test_same_session_different_times_rejected(self):
        cal = calendar(3)
        same_day = cal[1] + pd.Timedelta(hours=9)
        also_same_day = cal[1] + pd.Timedelta(hours=15)
        exits = pd.DataFrame(
            {
                "exit_time": [same_day, also_same_day],
                "reference_pnl": [1.0, 2.0],
            }
        )
        with self.assertRaises(ValueError):
            build_gate(cal, exits)

    def test_exit_outside_calendar_rejected(self):
        cal = calendar(3)
        exits = pd.DataFrame(
            {
                "exit_time": [cal[-1] + pd.Timedelta(days=1)],
                "reference_pnl": [1.0],
            }
        )
        with self.assertRaises(ValueError):
            build_gate(cal, exits)

    def test_nonfinite_pnl_rejected(self):
        cal = calendar(3)
        for bad in (np.nan, np.inf, -np.inf):
            exits = pd.DataFrame(
                {"exit_time": [cal[0]], "reference_pnl": [bad]}
            )
            with self.assertRaises(ValueError):
                build_gate(cal, exits)

    def test_calendar_validation(self):
        duplicate = pd.DatetimeIndex(["2024-01-01", "2024-01-01"])
        with self.assertRaises(ValueError):
            build_gate(duplicate, pd.DataFrame(columns=["exit_time", "reference_pnl"]))

        unordered = pd.DatetimeIndex(["2024-01-02", "2024-01-01"])
        with self.assertRaises(ValueError):
            build_gate(unordered, pd.DataFrame(columns=["exit_time", "reference_pnl"]))

    def test_sequential_strategies_are_independent(self):
        cal = calendar(3)
        exits_a = exits_for(cal, [1, 2, 3])
        exits_b = exits_for(cal, [-1, -2, -3])

        result_a = build_gate(cal, exits_a, window=3)
        result_b = build_gate(cal, exits_b, window=3)

        self.assertEqual(result_a.loc[2, "mode_after_exit"], 1)
        self.assertEqual(result_b.loc[2, "mode_after_exit"], -1)
        self.assertEqual(list(result_a["reference_pnl"]), [1.0, 2.0, 3.0])
        self.assertEqual(list(result_b["reference_pnl"]), [-1.0, -2.0, -3.0])

        again_a = build_gate(cal, exits_a, window=3)
        assert_frame_equal(result_a, again_a)


if __name__ == "__main__":
    unittest.main()
