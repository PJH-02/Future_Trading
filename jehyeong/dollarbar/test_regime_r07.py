"""R07 model tournament unit tests."""
import numpy as np
import pandas as pd

from regime_r05_anchors import _training_cutoff
from regime_r07_walkforward import (
    CANDIDATES,
    REFERENCES,
    _classification_metrics,
    _fit_probability_table,
    _paired_daily_block_ci,
    _predict_probabilities,
    _select_candidate,
)


def test_clock_prior_predicts_smoothed_training_class_frequencies():
    train = pd.DataFrame({"clock_bucket": [54] * 6, "label": ["DOWN", "DOWN", "SIDEWAYS", "UP", "UP", "UP"]})
    table, priors = _fit_probability_table(train, "label", ("DOWN", "SIDEWAYS", "UP"), ())
    test = pd.DataFrame({"clock_bucket": [54, 99]})
    p = _predict_probabilities(test, (), table, priors)
    assert p.shape == (2, 3)
    assert np.allclose(p.sum(axis=1), 1)
    assert p[0, 2] > p[0, 0]
    assert np.allclose(p[1], priors["__global__"])


def test_candidate_features_use_train_only_tables_and_smoothed_probabilities():
    train = pd.DataFrame({"clock_bucket": [54] * 6, "state": ["TREND"] * 3 + ["UNCERTAIN"] * 3,
                          "label": ["UP", "UP", "UP", "DOWN", "SIDEWAYS", "SIDEWAYS"]})
    table, priors = _fit_probability_table(train, "label", ("DOWN", "SIDEWAYS", "UP"), ("state",))
    test = pd.DataFrame({"clock_bucket": [54, 54], "state": ["TREND", "NEW_STATE"]})
    p = _predict_probabilities(test, ("state",), table, priors)
    assert np.allclose(p.sum(axis=1), 1)
    assert p[0, 2] > p[0, 0]
    assert np.allclose(p[1], priors["54"])


def test_classification_metrics_score_perfect_probabilities_correctly():
    y = pd.Series(["DOWN", "SIDEWAYS", "UP"])
    p = np.eye(3) * .98 + (np.ones((3, 3)) - np.eye(3)) * .01
    scores = _classification_metrics(y, p, ("DOWN", "SIDEWAYS", "UP"))
    assert scores["accuracy"] == 1.0
    assert scores["macro_f1"] == 1.0
    assert scores["brier"] < .001


def test_observed_minute_cutoff_leaves_exact_purge_observations():
    t = pd.date_range("2024-01-01 09:01", periods=300, freq="min")
    frame = pd.DataFrame({"timestamp": t})
    fold_start = pd.Timestamp("2024-01-01 13:01")
    cutoff = _training_cutoff(fold_start, frame)
    assert (frame.loc[frame.timestamp >= cutoff, "timestamp"] < fold_start).sum() == 60
    assert cutoff < fold_start
    assert frame.loc[frame.timestamp < cutoff, "timestamp"].max() < cutoff


def test_paired_daily_block_ci_is_reproducible_and_signed():
    daily = pd.date_range("2024-01-01", periods=20, freq="D")
    rows = pd.DataFrame({"date": np.repeat(daily.strftime("%Y-%m-%d"), 2)})
    candidate = np.tile([.1, .2], 20)
    reference = candidate + .1
    a = _paired_daily_block_ci(rows, candidate, reference, seed=8)
    b = _paired_daily_block_ci(rows, candidate, reference, seed=8)
    assert a == b
    assert a["mean_difference"] < 0
    assert a["ci95"][1] < 0


def _synthetic_report(passing: bool) -> list[dict]:
    reports = []
    for q in range(8):
        models = {name: {"brier": .2} for name in CANDIDATES}
        paired = {candidate: {"ci95": [-.02, -.001] if passing else [-.01, .01]}
                  for candidate in REFERENCES}
        for candidate in REFERENCES:
            models[candidate]["brier"] = .18 if passing else .21
        reports.append({"status": "EVALUATED", "target": "path", "horizon": 15,
                        "models": models, "paired_daily_brier_difference": paired})
    return reports


def test_selection_gate_returns_no_model_without_quarterly_ci_support():
    selection = _select_candidate(_synthetic_report(False))
    assert selection["status"] == "NO_MODEL"
    assert selection["selected_model"] is None


def test_selection_gate_selects_only_simplest_predeclared_passing_candidate():
    selection = _select_candidate(_synthetic_report(True))
    assert selection["status"] == "PASS_CANDIDATE"
    assert selection["selected_model"] == "G1_clock_MA"
