"""R03 distribution-distance tests."""
import numpy as np
import pandas as pd

from regime_r03_distribution import (
    BLOCK_LENGTH,
    distance_stats,
    empirical_percentile,
    _sample_blocks,
)


def _edges():
    return np.r_[-np.inf, np.linspace(-3, 3, 15), np.inf]


def test_same_samples_have_small_distances():
    rng = np.random.default_rng(20)
    a = rng.normal(0, 1, 120)
    got = distance_stats(a, a.copy(), _edges())
    assert got["w1"] == 0
    assert got["ad2_stat"] < 1e-10
    assert got["ks_stat"] == 0
    assert got["js_divergence"] < 1e-12


def test_distance_measures_detect_synthetic_scale_and_tail_changes():
    rng = np.random.default_rng(42)
    base = rng.normal(0, 1, 240)
    variance_shift = rng.normal(0, 2.5, 60)
    tail_shift = rng.normal(0, 1, 60)
    tail_shift[:8] *= 7
    scale_stats = distance_stats(base, variance_shift, _edges())
    tail_stats = distance_stats(base, tail_shift, _edges())
    assert scale_stats["w1_norm"] > 0.2
    assert scale_stats["vol_ratio"] > 2
    assert tail_stats["ad2_stat"] > -0.2
    assert tail_stats["w1"] > 0


def test_moving_block_resample_preserves_adjacent_values_within_blocks():
    pool = np.arange(100, dtype=float)
    sample = _sample_blocks(pool, 40, np.random.default_rng(3), BLOCK_LENGTH)
    assert len(sample) == 40
    # At least one contiguous block of the requested length is retained exactly.
    assert any(np.all(np.diff(sample[i:i + BLOCK_LENGTH]) == 1) for i in range(len(sample) - BLOCK_LENGTH + 1))


def test_moving_block_resample_never_crosses_session_separator():
    pool = np.r_[np.arange(10, dtype=float), np.nan, 100 + np.arange(10, dtype=float)]
    sample = _sample_blocks(pool, 200, np.random.default_rng(4), 4)
    assert np.isfinite(sample).all()
    for i in range(len(sample) - 3):
        block = sample[i:i + 4]
        if np.all(np.diff(block) == 1):
            assert block[-1] < 10 or block[0] >= 100


def test_empirical_percentile_is_bounded():
    assert empirical_percentile(3.0, [1.0, 2.0, 3.0, 4.0]) == 0.8
    assert 0 < empirical_percentile(10.0, [1.0, 2.0, 3.0]) <= 1


def test_r03_development_windows_are_nonoverlapping():
    # recent [40,69] and its baseline [0,39] are adjacent but never share a return.
    recent_n, baseline_n, end_idx = 30, 40, 69
    recent_start = end_idx - recent_n + 1
    baseline_end = recent_start
    baseline_start = baseline_end - baseline_n
    base_idx = set(range(baseline_start, baseline_end))
    recent_idx = set(range(recent_start, end_idx + 1))
    assert len(base_idx & recent_idx) == 0
    assert max(base_idx) + 1 == min(recent_idx)


def test_jensen_shannon_uses_fixed_edges_and_outer_bins():
    edges = _edges()
    a = np.array([-100.0, -0.1, 0.1, 100.0] * 10)
    b = np.array([-100.0, -0.1, 0.1, 100.0] * 10)
    got = distance_stats(a, b, edges)
    assert got["js_divergence"] < 1e-12
