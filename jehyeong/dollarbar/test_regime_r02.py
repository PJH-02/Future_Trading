"""R02 path-feature tests. Run: python3 -m pytest -q test_regime_r0*.py"""
import numpy as np
import pandas as pd

from regime_r02_path import add_future_anchors, build_path_features


def _frame(day: str, start_minute: int = 9 * 60 + 1, n: int = 180, trend: float = 0.0001):
    times = pd.date_range("2000-01-01 00:00", periods=n, freq="min")
    times = pd.Timestamp(day) + pd.to_timedelta(np.arange(n) + start_minute, unit="min")
    close = 100 * np.exp(np.arange(n) * trend)
    return pd.DataFrame({
        "timestamp": times,
        "date": day,
        "minute": times.hour * 60 + times.minute,
        "open": close,
        "high": close * 1.0002,
        "low": close * 0.9998,
        "close": close,
        "volume": np.full(n, 100.0),
    })


def test_features_reset_at_session_boundary():
    x = pd.concat([_frame("2024-01-02"), _frame("2024-01-03")], ignore_index=True)
    got = build_path_features(x)
    assert pd.isna(got.loc[got["date"] == "2024-01-03", "ret_1m"].iloc[0])
    assert pd.isna(got.loc[got["date"] == "2024-01-03", "er_10"].iloc[0])


def test_missing_minute_does_not_create_one_minute_return_or_bridge_rolling_window():
    x = _frame("2024-01-02", n=240)
    x = x.drop(index=50).reset_index(drop=True)
    got = build_path_features(x)
    # The row immediately after the deleted minute is not treated as a 1-minute return.
    assert pd.isna(got.loc[50, "ret_1m"])
    assert pd.isna(got.loc[50, "er_10"])
    # A full window only becomes available after ten contiguous returns from the gap.
    assert pd.isna(got.loc[59, "er_10"])
    assert pd.notna(got.loc[60, "er_10"])


def test_er_is_bounded_and_trend_candidate_is_not_mr():
    got = build_path_features(_frame("2024-01-02", n=200))
    values = got["er_10"].dropna()
    assert values.between(0, 1).all()
    assert not got[["state_p1", "state_p2"]].eq("MEAN_REVERTING").any().any()


def test_future_anchor_columns_are_evaluation_only_and_future_shifted():
    base = build_path_features(_frame("2024-01-02", n=100))
    augmented = add_future_anchors(base)
    assert "future_ret_5" not in base.columns
    assert np.isclose(augmented.loc[20, "future_ret_5"], np.log(augmented.loc[25, "close"] / augmented.loc[20, "close"]))
    assert np.isclose(augmented.loc[20, "future_rv_5"], np.square(augmented.loc[21:25, "ret_1m"]).sum())
    assert pd.isna(augmented.loc[augmented.index[-1], "future_ret_5"])


def test_future_perturbation_does_not_change_past_path_features():
    original = _frame("2024-01-02", n=180)
    altered = original.copy()
    cutoff = 120
    altered.loc[cutoff + 1:, ["open", "high", "low", "close"]] *= 1.4
    a = build_path_features(original)
    b = build_path_features(altered)
    causal = ["ret_1m", "er_10", "er_30", "er_60", "ema20", "ema60", "ma_z", "vr_60_2", "acf1_60", "state_p2"]
    pd.testing.assert_frame_equal(a.loc[:cutoff, causal], b.loc[:cutoff, causal], check_dtype=False)


def test_prefix_replay_matches_full_path_features():
    full = _frame("2024-01-02", n=180)
    prefix = full.iloc[:120].copy()
    a = build_path_features(full).iloc[:120].reset_index(drop=True)
    b = build_path_features(prefix).reset_index(drop=True)
    causal = ["ret_1m", "er_10", "er_30", "er_60", "ema20", "ema60", "ma_z", "vr_60_2", "acf1_60", "state_p2"]
    pd.testing.assert_frame_equal(a[causal], b[causal], check_dtype=False)


def test_variance_ratio_distinguishes_positive_and_negative_autocorrelation():
    rng = np.random.default_rng(941)
    n = 390
    eps_pos = rng.normal(size=n)
    eps_neg = rng.normal(size=n)
    ret_pos = np.zeros(n)
    ret_neg = np.zeros(n)
    for i in range(1, n):
        ret_pos[i] = .65 * ret_pos[i - 1] + eps_pos[i]
        ret_neg[i] = -.65 * ret_neg[i - 1] + eps_neg[i]
    def feature_for(returns):
        close = 100 * np.exp(np.cumsum(returns))
        t = pd.Timestamp("2024-01-02") + pd.to_timedelta(np.arange(n) + 9 * 60 + 1, unit="min")
        frame = pd.DataFrame({"timestamp": t, "date": "2024-01-02", "minute": t.hour * 60 + t.minute,
                              "open": close, "high": close * 1.001, "low": close * .999,
                              "close": close, "volume": np.full(n, 10.0)})
        return build_path_features(frame)["vr_120_5"].dropna().median()
    vr_pos = feature_for(ret_pos)
    vr_neg = feature_for(ret_neg)
    assert vr_pos > 1.0
    assert vr_neg < 1.0
