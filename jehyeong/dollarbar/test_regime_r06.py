"""R06 detector integration and leakage tests."""
import numpy as np
import pandas as pd
import pytest

from regime_r06_detector import _load_r03_scores, build_diagnostic_stream, runtime_view


def _path_frame():
    ts = pd.date_range("2024-01-02 09:01", periods=12, freq="min")
    return pd.DataFrame({
        "timestamp": ts, "date": ts.strftime("%Y-%m-%d"), "symbol": "KOSPI",
        "state_p1": "TREND", "state_p2": "UNCERTAIN", "state_p5": "UNCERTAIN",
        "trend_direction_p1": "UP", "close": np.arange(12) + 100,
    })


def _shift_frame():
    ts = pd.to_datetime(["2024-01-02 09:05", "2024-01-02 09:10"])
    return pd.DataFrame({
        "timestamp": ts, "date": ts.strftime("%Y-%m-%d"), "recent_start_30": ts - pd.Timedelta(minutes=30),
        "baseline_end_30": ts - pd.Timedelta(minutes=31), "w1_norm_30": [.1, .2],
        "w1_norm_null_pct_30": [.5, .99], "ad2_stat_30": [.0, 2.0],
        "ad2_stat_null_pct_30": [.4, .98], "ks_stat_30": [.1, .3],
        "ks_stat_null_pct_30": [.5, .99], "js_divergence_30": [.02, .1],
        "js_divergence_null_pct_30": [.5, .99], "vol_ratio_30": [1.0, 2.0],
        "timestamp_30": ts,
    })


def test_runtime_view_excludes_future_labels():
    path = _path_frame()
    scores = _shift_frame()
    got = runtime_view(build_diagnostic_stream(path, scores))
    assert not any(c.startswith(("future_", "continuation_", "reversion_", "target_")) for c in got)
    assert not {"future_return", "future_rv", "future_er", "reversion_return"}.intersection(got.columns)


def test_path_state_remains_uncertain_until_model_selected():
    got = runtime_view(build_diagnostic_stream(_path_frame(), _shift_frame()))
    assert set(got["path_state"]) == {"UNCERTAIN"}
    assert set(got["path_candidate_p1"]) == {"TREND"}
    assert set(got["model_version"]) == {"diagnostic-unselected-r06"}


def test_shift_join_is_asof_and_never_uses_future_score():
    path = _path_frame()
    scores = _shift_frame()
    got = runtime_view(build_diagnostic_stream(path, scores))
    before = got.loc[got["timestamp"] < pd.Timestamp("2024-01-02 09:05")]
    at = got.loc[got["timestamp"] == pd.Timestamp("2024-01-02 09:05")].iloc[0]
    assert before["shift_available_at"].isna().all()
    assert at["shift_available_at"] == pd.Timestamp("2024-01-02 09:05")
    assert at["shift_candidate_alarm"] is np.bool_(False) or bool(at["shift_candidate_alarm"]) is False
    after = got.loc[got["timestamp"] == pd.Timestamp("2024-01-02 09:10")].iloc[0]
    assert after["shift_available_at"] == pd.Timestamp("2024-01-02 09:10")
    assert bool(after["shift_candidate_alarm"]) is True


def test_future_target_in_detector_input_fails_closed():
    path = _path_frame()
    path["future_rv_5"] = 1.0
    with pytest.raises(ValueError, match="future/evaluation columns forbidden"):
        build_diagnostic_stream(path, _shift_frame())


def test_r06_loader_does_not_discard_scores_after_r03_calibration(tmp_path):
    rows = []
    for day in pd.date_range("2024-01-01", periods=61, freq="D"):
        for recent in (30, 60):
            rows.append({"timestamp": day + pd.Timedelta(hours=10), "date": day.strftime("%Y-%m-%d"),
                         "recent_n": recent, "w1_norm": .1, "ad2_stat": .2, "ks_stat": .1,
                         "js_divergence": .02, "vol_ratio": 1.0,
                         "w1_norm_null_pct": .5, "ad2_stat_null_pct": .5,
                         "ks_stat_null_pct": .5, "js_divergence_null_pct": .5})
    pd.DataFrame(rows).to_csv(tmp_path / "r03_distribution_scores_dev_kospi.csv", index=False)
    scores = _load_r03_scores("KOSPI", tmp_path)
    assert scores["date"].nunique() == 61
    assert scores["date"].iloc[0] == "2024-01-01"
    assert scores["shift_available_at"].min() == pd.Timestamp("2024-01-01 10:00")
