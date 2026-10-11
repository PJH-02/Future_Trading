"""Check source replacement and signal timing on small, independent calendars."""

import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd


SCRIPT = Path(__file__).resolve().parents[1] / "f1_f2_f3_minute_eda_aligned_causal.py"
SPEC = importlib.util.spec_from_file_location("proxy_backtest", SCRIPT)
backtest = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(backtest)


class ProxyBacktestTests(unittest.TestCase):
    def make_inputs(self):
        # First 2021 session follows a holiday/weekend, not a calendar-day shift.
        dates = pd.to_datetime([
            "2020-12-30", "2021-01-04", "2021-01-05",
            "2021-01-06", "2021-01-07", "2021-01-08",
        ])
        research = pd.DataFrame({
            "KODEX_close": 100.0,
            "kr_gap": 0.01,
            "net_gap": 0.0097,
        }, index=dates)
        proxy = pd.DataFrame(0.01, index=dates, columns=backtest.ANCHORED_RETURN_COLS)
        # Missing proxy data must not compress the KRX lag calendar.
        proxy = proxy.drop(pd.Timestamp("2021-01-05"))
        proxy.loc["2021-01-07", backtest.ANCHORED_RETURN_COLS[1]] = 0.0
        foreign = pd.DataFrame({
            "foreign_netbuy": [10.0, -10.0, np.nan, 10.0, 10.0, -10.0],
        }, index=dates)
        return research, proxy, foreign

    def test_research_needs_only_kodex_columns(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "research.csv"
            pd.DataFrame({
                "date": ["2021-01-04"], "KODEX_close": [100],
                "kr_gap": [0.01], "net_gap": [0.0097],
            }).to_csv(path, index=False)
            with patch.object(backtest, "RESEARCH_CANDIDATES", [path]):
                research, _ = backtest.load_research()
        self.assertEqual(len(research), 1)
        self.assertFalse(any("QQQ" in col or "SPY" in col for col in research))

    def test_asof_proxy_rejects_future_prior_day_unusable_and_infinite_prices(self):
        dates = [f"2021-01-{day:02d}" for day in range(4, 10)]
        data = {"date": dates}
        for code, return_col in zip(backtest.PROXY_CODES, backtest.PROXY_RETURN_COLS):
            data[return_col] = [0.01, 0.01, 0.01, 0.01, np.inf, 0.01]
            data[f"{code}_asof_usable_1520"] = [True, True, True, False, True, True]
            data[f"{code}_asof_last_trade_available_at_1520"] = [
                "2021-01-04T15:19:00+09:00",  # earlier same-day trade is valid
                "2021-01-05T15:21:00+09:00",  # after the signal cutoff
                "2021-01-05T15:20:00+09:00",  # prior session, not January 6
                "2021-01-07T15:20:00+09:00",
                "2021-01-08T15:20:00+09:00",
                "2021-01-09T15:20:00+09:00",  # exactly at cutoff is valid
            ]
            # Post-signal daily prices must never supply signal returns.
            data[f"{code}_daily_close"] = [999.0] * len(dates)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "proxy.csv"
            pd.DataFrame(data).to_csv(path, index=False)
            with patch.object(backtest, "PROXY_CANDIDATES", [path]):
                proxy, _ = backtest.load_proxy()
        for return_col in backtest.PROXY_RETURN_COLS:
            self.assertEqual(proxy[return_col].notna().tolist(), [True, False, False, False, False, True])

    def test_full_krx_lag_and_missing_features(self):
        research, proxy, foreign = self.make_inputs()
        result = backtest.build_backtest_dataset(research, proxy, foreign)
        self.assertEqual(result.index.min(), pd.Timestamp("2021-01-04"))
        self.assertEqual(result.loc["2021-01-04", "foreign_prev_date"], pd.Timestamp("2020-12-30"))
        self.assertEqual(result.loc["2021-01-04", "foreign_prev"], 10.0)
        self.assertEqual(result["available"].tolist(), [True, False, False, True, True])
        self.assertEqual(result["signal"].tolist(), [True, False, False, False, True])
        self.assertTrue(pd.isna(result.loc["2021-01-06", "foreign_prev"]))
        self.assertEqual(len(result), 5)  # retain both missing-feature days
        np.testing.assert_allclose(result["net_gap"], 0.0095)

    def test_session_without_next_open_is_excluded(self):
        research, proxy, foreign = self.make_inputs()
        research.loc["2021-01-08", ["kr_gap", "net_gap"]] = np.nan
        result = backtest.build_backtest_dataset(research, proxy, foreign)
        self.assertEqual(result.index.max(), pd.Timestamp("2021-01-07"))

    def test_no_common_data_after_2021_raises(self):
        research, proxy, foreign = self.make_inputs()
        with self.assertRaisesRegex(RuntimeError, "2021"):
            backtest.build_backtest_dataset(research.iloc[:1], proxy.iloc[:1], foreign)

    def test_comparison_uses_same_calendar_and_shared_feature_trade_mask(self):
        research, proxy, foreign = self.make_inputs()
        research[backtest.ORIGINAL_RETURN_COLS] = 0.02
        research.loc["2021-01-07", backtest.ORIGINAL_RETURN_COLS[0]] = np.nan
        proxy.loc["2021-01-07", backtest.ANCHORED_RETURN_COLS[1]] = 0.01
        result = backtest.build_comparison_dataset(research, proxy, foreign)
        self.assertEqual(len(result), 5)
        self.assertEqual(result.index.min(), pd.Timestamp("2021-01-04"))
        self.assertEqual(result["original_signal"].tolist(), [True, False, False, False, True])
        self.assertEqual(result["anchored_signal"].tolist(), [True, False, False, True, True])
        self.assertEqual(result["common_available"].tolist(), [True, False, False, False, True])
        for prefix in ("original", "anchored"):
            self.assertEqual(result[f"{prefix}_common_signal"].tolist(), [True, False, False, False, True])
        calendar = research.index.append(pd.DatetimeIndex(["2021-01-11"]))
        shared = result.index[result["common_available"]]
        for cols in (backtest.ORIGINAL_RETURN_COLS, backtest.ANCHORED_RETURN_COLS):
            model = backtest.build_backtest_dataset(research, proxy, foreign, cols)
            ledger = backtest.overnight_trades(model, calendar, "test")
            daily, trades = backtest.account_returns(ledger, calendar, "2021-01-04", "2021-01-11", shared)
            self.assertEqual(len(trades), 2)
            self.assertEqual(len(daily), 6)
            self.assertAlmostEqual(daily.loc["2021-01-05"], 0.0095)
            self.assertAlmostEqual(daily.loc["2021-01-11"], 0.0095)
            self.assertEqual(daily.loc["2021-01-08"], 0.0)

    def test_original_period_is_not_backfilled_from_proxy(self):
        research, proxy, foreign = self.make_inputs()
        research[backtest.ORIGINAL_RETURN_COLS] = 0.02
        research.loc["2021-01-04", backtest.ORIGINAL_RETURN_COLS] = np.nan
        result = backtest.build_comparison_dataset(research, proxy, foreign)
        self.assertEqual(result.index.min(), pd.Timestamp("2021-01-05"))
        self.assertEqual(result.loc["2021-01-05", "foreign_prev"], -10.0)
        full_proxy = backtest.build_backtest_dataset(research, proxy, foreign)
        self.assertEqual(full_proxy.index.min(), pd.Timestamp("2021-01-04"))
        self.assertTrue(full_proxy.loc["2021-01-04", "signal"])


class UsCloseAlignmentTests(unittest.TestCase):
    def daily_data(self, dates, closes, available_at):
        daily = pd.DataFrame({
            "date": pd.to_datetime(dates), "close": closes,
            "available_at": pd.to_datetime(available_at, utc=True).tz_convert("Asia/Seoul"),
        })
        return {symbol: daily.copy() for symbol in backtest.US_SYMBOLS}

    def proxy_data(self, dates, gross_returns):
        return pd.DataFrame({
            col: np.asarray(gross_returns) - 1.0 for col in backtest.PROXY_RETURN_COLS
        }, index=pd.to_datetime(dates))

    def march_data(self):
        calendar = pd.to_datetime(["2022-03-11", "2022-03-14", "2022-03-15"])
        daily = self.daily_data(
            ["2022-03-10", "2022-03-11", "2022-03-14", "2022-03-15"],
            [100.0, 102.0, 105.0, 1e9],
            ["2022-03-11T06:00:00+09:00", "2022-03-12T06:00:00+09:00",
             "2022-03-15T05:00:00+09:00", "2022-03-16T05:00:00+09:00"],
        )
        return calendar, daily

    def test_us_daily_loader_handles_dst_and_retains_missing_close(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "QQQ_daily.csv"
            pd.DataFrame({"date": ["2022-03-11", "2022-03-14", "2022-03-15"], "close": [102, 105, np.nan]}).to_csv(path, index=False)
            with patch.dict(backtest.US_DAILY_CANDIDATES, {"QQQ": [path]}):
                daily, _ = backtest.load_us_daily("QQQ")
        self.assertEqual(daily.iloc[0]["available_at"], pd.Timestamp("2022-03-12T06:00:00+09:00"))
        self.assertEqual(daily.iloc[1]["available_at"], pd.Timestamp("2022-03-15T05:00:00+09:00"))
        self.assertTrue(pd.isna(daily.iloc[2]["close"]))

    def test_same_krx_day_alignment_weekend_and_future_close_exclusion(self):
        calendar, daily = self.march_data()
        proxy = self.proxy_data(["2022-03-14", "2022-03-15"], [1.03, 1.04])
        result = backtest.add_us_close_anchored_returns(proxy, calendar, daily)
        for symbol, col in zip(backtest.US_SYMBOLS, backtest.ANCHORED_RETURN_COLS):
            self.assertEqual(result.loc["2022-03-14", f"{symbol}_us_session_date_at_signal"], pd.Timestamp("2022-03-11"))
            self.assertEqual(result.loc["2022-03-14", f"{symbol}_us_session_date_at_previous_krx_close"], pd.Timestamp("2022-03-10"))
            self.assertEqual(result.loc["2022-03-15", f"{symbol}_us_session_date_at_signal"], pd.Timestamp("2022-03-14"))
            np.testing.assert_allclose(result[col], [np.log(1.03 / 1.02), np.log(1.04 / (105.0 / 102.0))])
        for symbol in backtest.US_SYMBOLS:
            daily[symbol].loc[3, "close"] = 0.001
        changed = backtest.add_us_close_anchored_returns(proxy, calendar, daily)
        np.testing.assert_allclose(result[backtest.ANCHORED_RETURN_COLS], changed[backtest.ANCHORED_RETURN_COLS])

    def test_us_holiday_uses_same_latest_close_on_consecutive_krx_days(self):
        calendar = pd.to_datetime(["2022-05-27", "2022-05-30", "2022-05-31"])
        daily = self.daily_data(
            ["2022-05-26", "2022-05-27", "2022-05-31"], [100, 101, 103],
            ["2022-05-27T05:00:00+09:00", "2022-05-28T05:00:00+09:00", "2022-06-01T05:00:00+09:00"],
        )
        proxy = self.proxy_data(["2022-05-30", "2022-05-31"], [1.02, 1.01])
        result = backtest.add_us_close_anchored_returns(proxy, calendar, daily)
        for symbol, col in zip(backtest.US_SYMBOLS, backtest.ANCHORED_RETURN_COLS):
            self.assertEqual(result.loc["2022-05-31", f"{symbol}_us_close_at_signal"], 101)
            self.assertEqual(result.loc["2022-05-31", f"{symbol}_us_close_at_previous_krx_close"], 101)
            self.assertAlmostEqual(result.loc["2022-05-31", col], np.log(1.01))

    def test_krx_holiday_removes_multiple_us_sessions_since_actual_previous_krx_close(self):
        calendar = pd.to_datetime(["2022-09-08", "2022-09-13"])
        daily = self.daily_data(
            ["2022-09-07", "2022-09-08", "2022-09-09", "2022-09-12"], [100, 101, 102, 103],
            ["2022-09-08T05:00:00+09:00", "2022-09-09T05:00:00+09:00",
             "2022-09-10T05:00:00+09:00", "2022-09-13T05:00:00+09:00"],
        )
        result = backtest.add_us_close_anchored_returns(self.proxy_data(["2022-09-13"], [1.04]), calendar, daily)
        for symbol, col in zip(backtest.US_SYMBOLS, backtest.ANCHORED_RETURN_COLS):
            self.assertEqual(result.iloc[0][f"{symbol}_us_session_date_at_previous_krx_close"], pd.Timestamp("2022-09-07"))
            self.assertAlmostEqual(result.iloc[0][col], np.log(1.04 / 1.03))

    def test_missing_latest_us_close_does_not_fall_back_to_older_price(self):
        calendar, daily = self.march_data()
        for symbol in backtest.US_SYMBOLS:
            daily[symbol].loc[2, "close"] = np.nan
        result = backtest.add_us_close_anchored_returns(self.proxy_data(["2022-03-15"], [1.04]), calendar, daily)
        self.assertTrue(result[backtest.ANCHORED_RETURN_COLS].isna().all().all())

    def test_source_end_allows_monday_but_does_not_extrapolate_to_tuesday(self):
        calendar, daily = self.march_data()
        for symbol in backtest.US_SYMBOLS:
            daily[symbol] = daily[symbol].iloc[:2]
        result = backtest.add_us_close_anchored_returns(self.proxy_data(["2022-03-14", "2022-03-15"], [1.03, 1.04]), calendar, daily)
        self.assertTrue(result.loc["2022-03-14", backtest.ANCHORED_RETURN_COLS].notna().all())
        self.assertTrue(result.loc["2022-03-15", backtest.ANCHORED_RETURN_COLS].isna().all())

    def test_signal_is_invariant_to_us_price_units_and_missing_proxy_stays_missing(self):
        calendar, daily = self.march_data()
        proxy = self.proxy_data(["2022-03-14", "2022-03-15"], [1.03, 1.04])
        proxy.loc["2022-03-15", backtest.PROXY_RETURN_COLS[0]] = np.nan
        result = backtest.add_us_close_anchored_returns(proxy, calendar, daily)
        for symbol in backtest.US_SYMBOLS:
            daily[symbol]["close"] *= 1000.0
        scaled = backtest.add_us_close_anchored_returns(proxy, calendar, daily)
        np.testing.assert_allclose(result[backtest.ANCHORED_RETURN_COLS], scaled[backtest.ANCHORED_RETURN_COLS], equal_nan=True)
        self.assertTrue(pd.isna(result.loc["2022-03-15", backtest.ANCHORED_RETURN_COLS[0]]))


if __name__ == "__main__":
    unittest.main()
