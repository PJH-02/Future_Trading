"""R05 evaluation-only future-anchor tests."""
import numpy as np
import pandas as pd

from regime_r02_path import add_future_anchors, build_path_features
from regime_r05_anchors import (
    _apply_anchor_labels,
    _daily_block_summary,
    _evaluate_fold,
    _fit_anchor_thresholds,
    _training_cutoff,
    add_future_path_er,
)
from test_regime_r02 import _frame


def test_future_path_er_matches_horizon_definition():
    x = build_path_features(_frame("2024-01-02", n=100))
    got = add_future_path_er(add_future_anchors(x))
    expected_path = np.abs(got.loc[21:25, "ret_1m"]).sum()
    expected_net = abs(np.log(got.loc[25, "close"] / got.loc[20, "close"]))
    assert np.isclose(got.loc[20, "future_er_5"], expected_net / expected_path)
    assert pd.isna(got.loc[got.index[-1], "future_er_5"])


def test_future_targets_do_not_cross_missing_minute_segment():
    x = _frame("2024-01-02", n=160).drop(index=80).reset_index(drop=True)
    x = build_path_features(x)
    got = add_future_anchors(x)
    # A target before the gap cannot look through it; the next segment may form its own target.
    assert pd.isna(got.loc[79, "future_ret_5"])
    assert pd.isna(got.loc[79, "future_rv_5"])
    assert np.isclose(got.loc[80, "future_ret_5"], np.log(got.loc[85, "close"] / got.loc[80, "close"]))


def test_thresholds_are_fit_only_from_passed_training_frame():
    x = build_path_features(_frame("2024-01-02", n=200))
    x = add_future_path_er(add_future_anchors(x))
    x["clock_bucket"] = x["minute"] // 10
    a = _fit_anchor_thresholds(x.iloc[:130])
    holdout = x.iloc[130:].copy()
    holdout.loc[:, "future_rv_5"] = 999.0
    # Fitting receives only the explicit training slice, never the altered holdout.
    b = _fit_anchor_thresholds(x.iloc[:130].copy())
    assert len(holdout) > 0
    assert a["5"]["global"]["rv_q40"] == b["5"]["global"]["rv_q40"]
    assert a["5"]["global"]["rv_q75"] == b["5"]["global"]["rv_q75"]


def test_training_thresholds_apply_clock_bucket_and_preserve_ineligible_label():
    train = pd.DataFrame({"clock_bucket": [54] * 120})
    for h in (5, 15, 30, 60):
        train[f"future_rv_{h}"] = np.arange(120, dtype=float) + 1
        train[f"future_ret_{h}"] = np.linspace(-.02, .02, 120)
    thresholds = _fit_anchor_thresholds(train)
    test = pd.DataFrame({"clock_bucket": [54, 54], "future_rv_5": [1.0, np.nan],
                         "future_ret_5": [-.03, np.nan], "future_er_5": [.8, np.nan],
                         "ma_z": [1.7, np.nan]})
    for h in (15, 30, 60):
        test[f"future_rv_{h}"] = np.nan
        test[f"future_ret_{h}"] = np.nan
        test[f"future_er_{h}"] = np.nan
    labelled = _apply_anchor_labels(test, thresholds)
    assert labelled.loc[0, "future_vol_class_5"] == "CALM"
    assert labelled.loc[0, "future_path_label_5"] == "DOWN"
    assert pd.isna(labelled.loc[1, "future_vol_class_5"])
    assert pd.isna(labelled.loc[1, "future_path_label_5"])


def test_purge_cutoff_counts_observed_minutes_not_wall_clock():
    observed = pd.date_range("2024-01-01 09:01", periods=100, freq="min").append(
        pd.date_range("2024-01-02 09:01", periods=100, freq="min")
    )
    features = pd.DataFrame({"timestamp": observed})
    fold_start = pd.Timestamp("2024-01-02 09:01")
    cutoff = _training_cutoff(fold_start, features)
    assert cutoff == pd.Timestamp("2024-01-01 09:41")


def test_daily_block_bootstrap_reports_day_count_and_ci():
    frame = pd.DataFrame({"date": np.repeat([f"2024-01-{d:02d}" for d in range(1, 11)], 3),
                          "x": np.repeat(np.arange(10, dtype=float), 3)})
    result = _daily_block_summary(frame, "x", seed=10)
    assert result["n_days"] == 10
    assert result["bootstrap_block_days"] == 5
    assert result["ci95"][0] <= result["mean"] <= result["ci95"][1]


def test_fold_evaluation_keeps_each_state_candidate_family_separate():
    rows = []
    for day in range(1, 4):
        for state1, state2, state5 in [("TREND", "UNCERTAIN", "TREND"), ("UNCERTAIN", "TREND", "MEAN_REVERTING")]:
            row = {"date": f"2024-01-{day:02d}", "state_p1": state1, "state_p2": state2, "state_p5": state5}
            for h in (5, 15, 30, 60):
                for metric in ("future_ret", "future_rv", "future_er", "continuation", "reversion"):
                    row[f"{metric}_{h}"] = .001
                row[f"future_path_label_{h}"] = "UP"
                row[f"future_vol_class_{h}"] = "MID"
            rows.append(row)
    result = _evaluate_fold(pd.DataFrame(rows), "2024Q1")
    assert set(result["states"]) == {"state_p1", "state_p2", "state_p5"}
    assert set(result["states"]["state_p1"]["by_state"]) == {"TREND", "UNCERTAIN"}
    assert set(result["states"]["state_p5"]["by_state"]) == {"TREND", "MEAN_REVERTING"}
