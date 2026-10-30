"""R07 quarterly DEV walk-forward tournament; never loads 2026 or legacy results.

The tournament compares clock-prior forecasts with added path and shift evidence
for future path class and volatility class. Targets/thresholds are evaluation-only;
model inputs contain only as-of state/shift candidates. Candidate scores use
proper probability losses, paired daily moving-block intervals, and an explicit
complexity tie-break. If evidence is insufficient, the only valid decision is
NO_MODEL and R08 must remain gated.

Run: python3 regime_r07_walkforward.py --market KOSPI
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from regime_r00_validate import DEV_END, DEV_START, MINUTE, OUT
from regime_r02_path import HORIZONS, add_future_anchors, build_path_features, valid_dev_minutes
from regime_r05_anchors import (
    BOOTSTRAP_REPLICATES,
    FOLDS,
    _apply_anchor_labels,
    _daily_block_summary,
    _fit_anchor_thresholds,
    _training_cutoff,
    add_future_path_er,
)
from regime_r06_detector import _load_r03_scores

MIN_TRAIN_ROWS = 10_000
SMOOTH_ALPHA = 20.0
BOOTSTRAP_SEED = 42
PATH_CLASSES = ("DOWN", "SIDEWAYS", "UP")
VOL_CLASSES = ("CALM", "MID", "VOLATILE")

# Complexity ladder. The reference candidate is the preceding simpler model.
CANDIDATES = {
    "G0_clock_prior": (),
    "G1_clock_MA": ("state_p1",),
    "G2_clock_ER_MA": ("state_p2",),
    "G5_clock_path5": ("state_p5",),
    "G6_path5_W1": ("state_p5", "w1_bin"),
    "G7_path5_W1_AD": ("state_p5", "w1_bin", "ad_bin"),
    "G8_path5_W1_AD_KS_JS": ("state_p5", "w1_bin", "ad_bin", "ks_bin", "js_bin"),
}
REFERENCES = {
    "G1_clock_MA": "G0_clock_prior",
    "G2_clock_ER_MA": "G1_clock_MA",
    "G5_clock_path5": "G2_clock_ER_MA",
    "G6_path5_W1": "G5_clock_path5",
    "G7_path5_W1_AD": "G6_path5_W1",
    "G8_path5_W1_AD_KS_JS": "G7_path5_W1_AD",
}


def _attach_shift_scores(features: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    x = features.sort_values(["timestamp", "date"]).copy()
    d = scores.sort_values(["timestamp", "date"]).copy()
    x["timestamp"] = pd.to_datetime(x["timestamp"])
    d["timestamp"] = pd.to_datetime(d["timestamp"])
    x["date"] = x["timestamp"].dt.strftime("%Y-%m-%d")
    d["date"] = d["timestamp"].dt.strftime("%Y-%m-%d")
    value_cols = [c for c in d.columns if c not in {"timestamp", "date", "shift_available_at"}]
    d = d[["timestamp", "date", *value_cols]]
    return pd.merge_asof(x, d, on="timestamp", by="date", direction="backward",
                         tolerance=pd.Timedelta(minutes=5), suffixes=("", "_shift"))


def add_shift_bins(frame: pd.DataFrame) -> pd.DataFrame:
    """Turn train-calibrated empirical percentiles into stable coarse bins."""
    x = frame.copy()
    for metric, name in (("w1_norm", "w1"), ("ad2_stat", "ad"),
                         ("ks_stat", "ks"), ("js_divergence", "js")):
        cols = [c for c in x.columns if c.startswith(f"{metric}_null_pct_")]
        if cols:
            value = x[cols].mean(axis=1, skipna=True)
            x[f"{name}_bin"] = pd.cut(value, bins=[-np.inf, .5, .8, .9, .95, np.inf], labels=False).astype("float")
        else:
            x[f"{name}_bin"] = np.nan
    return x


def _feature_key(frame: pd.DataFrame, fields: tuple[str, ...]) -> pd.Series:
    """Vectorized pipe-joined key; identical strings to a row-wise join."""
    cols = ["clock_bucket", *fields]
    key = frame[cols[0]].astype("string").fillna("MISSING")
    for col in cols[1:]:
        key = key.str.cat(frame[col].astype("string").fillna("MISSING"), sep="|")
    return key


def _fit_probability_table(train: pd.DataFrame, target: str, classes: tuple[str, ...], fields: tuple[str, ...]) -> tuple[dict, dict]:
    """Fit smoothed conditional class probabilities, shrunk to clock-bucket priors."""
    valid = train.dropna(subset=[target]).copy()
    valid[target] = valid[target].astype(str)
    valid = valid[valid[target].isin(classes)]
    if valid.empty:
        raise ValueError(f"no valid training labels for {target}")
    prior_counts = pd.crosstab(valid["clock_bucket"].astype(str), valid[target]).reindex(columns=classes, fill_value=0)
    global_counts = valid[target].value_counts().reindex(classes, fill_value=0).to_numpy(dtype=float)
    global_prior = global_counts / global_counts.sum()
    priors = {}
    for bucket, row in prior_counts.iterrows():
        c = row.to_numpy(dtype=float)
        priors[str(bucket)] = c / c.sum() if c.sum() else global_prior
    key = _feature_key(valid, fields)
    counts = pd.crosstab(key, valid[target]).reindex(columns=classes, fill_value=0)
    key_to_bucket = pd.Series(valid["clock_bucket"].astype(str).to_numpy(), index=key.to_numpy()).to_dict()
    table = {}
    for feature_key, row in counts.iterrows():
        bucket = key_to_bucket.get(feature_key, "MISSING")
        prior = priors.get(bucket, global_prior)
        c = row.to_numpy(dtype=float)
        table[str(feature_key)] = (c + SMOOTH_ALPHA * prior) / (c.sum() + SMOOTH_ALPHA)
    priors["__global__"] = global_prior
    return table, priors


def _predict_probabilities(frame: pd.DataFrame, fields: tuple[str, ...], table: dict, priors: dict) -> np.ndarray:
    """Vectorized lookup: exact key first, then clock-bucket prior, then global."""
    keys = _feature_key(frame, fields).to_numpy(dtype=object)
    fallback = frame["clock_bucket"].astype(str).to_numpy(dtype=object)
    global_prior = priors["__global__"]
    n_classes = len(global_prior)
    output = np.empty((len(keys), n_classes), dtype=float)
    for i, (key, bucket) in enumerate(zip(keys, fallback)):
        row = table.get(key)
        if row is None:
            row = priors.get(bucket)
        output[i] = row if row is not None else global_prior
    return output


def _classification_metrics(y: pd.Series, probability: np.ndarray, classes: tuple[str, ...]) -> dict:
    truth = y.astype(str).to_numpy()
    class_index = {v: i for i, v in enumerate(classes)}
    yi = np.array([class_index[v] for v in truth], dtype=int)
    onehot = np.eye(len(classes))[yi]
    p = np.clip(probability, 1e-12, 1.0)
    predicted = np.argmax(p, axis=1)
    recalls, f1s = [], []
    for ci in range(len(classes)):
        tp = np.sum((predicted == ci) & (yi == ci))
        fp = np.sum((predicted == ci) & (yi != ci))
        fn = np.sum((predicted != ci) & (yi == ci))
        recall = tp / (tp + fn) if tp + fn else np.nan
        precision = tp / (tp + fp) if tp + fp else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        recalls.append(recall)
        f1s.append(f1)
    return {
        "rows": int(len(y)),
        "brier": float(np.mean(np.sum((p - onehot) ** 2, axis=1))),
        "log_loss": float(-np.log(p[np.arange(len(yi)), yi]).mean()),
        "accuracy": float(np.mean(predicted == yi)),
        "balanced_accuracy": float(np.nanmean(recalls)),
        "macro_f1": float(np.mean(f1s)),
        "class_rate": {c: float(np.mean(truth == c)) for c in classes},
        "predicted_class_rate": {c: float(np.mean(predicted == i)) for i, c in enumerate(classes)},
    }


def _paired_daily_block_ci(test: pd.DataFrame, candidate_loss: np.ndarray, reference_loss: np.ndarray,
                           seed: int = BOOTSTRAP_SEED) -> dict:
    paired = pd.DataFrame({"date": test["date"].to_numpy(), "difference": candidate_loss - reference_loss})
    daily = paired.groupby("date")["difference"].mean().to_numpy(dtype=float)
    if not len(daily):
        return {"n_days": 0, "mean_difference": None, "ci95": [None, None], "block_days": 5}
    rng = np.random.default_rng(seed)
    block = min(5, len(daily))
    blocks_needed = int(np.ceil(len(daily) / block))
    starts = rng.integers(0, len(daily), size=(BOOTSTRAP_REPLICATES, blocks_needed))
    offsets = np.arange(block)
    indices = (starts[:, :, None] + offsets[None, None, :]) % len(daily)
    sampled = daily[indices].reshape(BOOTSTRAP_REPLICATES, -1)[:, :len(daily)]
    draws = sampled.mean(axis=1)
    return {"n_days": int(len(daily)), "mean_difference": float(daily.mean()),
            "ci95": [float(np.quantile(draws, .025)), float(np.quantile(draws, .975))],
            "block_days": int(block)}


def _run_fold(all_features: pd.DataFrame, period: pd.Period) -> tuple[list[dict], pd.DataFrame]:
    start = period.start_time
    end = (period + 1).start_time
    if end > DEV_END + pd.Timedelta(days=1):
        end = DEV_END + pd.Timedelta(days=1)
    cutoff = _training_cutoff(start, all_features)
    train = all_features[(all_features["timestamp"] < cutoff) & (all_features["date"] < start.strftime("%Y-%m-%d"))].copy()
    test = all_features[(all_features["timestamp"] >= start) & (all_features["timestamp"] < end)].copy()
    if len(train) < MIN_TRAIN_ROWS or test.empty:
        return [{"fold": str(period), "status": "INSUFFICIENT_TRAIN_OR_TEST", "train_rows": len(train), "test_rows": len(test)}], pd.DataFrame()
    # Fit label thresholds on train outcomes only. Test outcomes are never used to fit thresholds.
    train = train[train["future_rv_60"].notna()].copy()
    thresholds = _fit_anchor_thresholds(train)
    train = _apply_anchor_labels(train, thresholds)
    test = _apply_anchor_labels(test, thresholds)
    reports = []
    for horizon in HORIZONS:
        for target_kind, target_col, classes in (
            ("path", f"future_path_label_{horizon}", PATH_CLASSES),
            ("vol", f"future_vol_class_{horizon}", VOL_CLASSES),
        ):
            train_valid = train.dropna(subset=[target_col]).copy()
            test_valid = test.dropna(subset=[target_col]).copy()
            if len(train_valid) < MIN_TRAIN_ROWS or test_valid.empty:
                reports.append({"fold": str(period), "horizon": horizon, "target": target_kind,
                                "status": "INSUFFICIENT_LABELS", "train_rows": len(train_valid), "test_rows": len(test_valid)})
                continue
            test_valid = test_valid.reset_index(drop=True)
            y_test = test_valid[target_col].astype(str)
            model_metrics = {}
            loss_vectors = {}
            for candidate, fields in CANDIDATES.items():
                table, priors = _fit_probability_table(train_valid, target_col, classes, fields)
                probability = _predict_probabilities(test_valid, fields, table, priors)
                metrics = _classification_metrics(y_test, probability, classes)
                model_metrics[candidate] = metrics
                truth_idx = pd.Categorical(y_test, categories=classes).codes
                row_loss = np.sum((probability - np.eye(len(classes))[truth_idx]) ** 2, axis=1)
                loss_vectors[candidate] = row_loss
            paired = {}
            for candidate, reference in REFERENCES.items():
                paired[candidate] = _paired_daily_block_ci(test_valid, loss_vectors[candidate], loss_vectors[reference],
                                                           seed=BOOTSTRAP_SEED + horizon)
            reports.append({
                "fold": str(period), "status": "EVALUATED", "horizon": horizon, "target": target_kind,
                "train_end_exclusive": str(cutoff), "train_rows": int(len(train_valid)),
                "test_rows": int(len(test_valid)), "test_dates": int(test_valid["date"].nunique()),
                "purge_observations": 60, "label_thresholds_fit_on_train_only": True,
                "models": model_metrics, "paired_daily_brier_difference": paired,
            })
    return reports, pd.DataFrame()


def _select_candidate(reports: list[dict]) -> dict:
    evaluated = [r for r in reports if r.get("status") == "EVALUATED" and r["target"] in {"path", "vol"}]
    decisions = {}
    for candidate, reference in REFERENCES.items():
        fold_results = [r for r in evaluated if candidate in r.get("paired_daily_brier_difference", {})]
        wins, total = 0, 0
        relative_improvements = []
        for row in fold_results:
            ci = row["paired_daily_brier_difference"][candidate]
            ref_score = row["models"][reference]["brier"]
            cand_score = row["models"][candidate]["brier"]
            total += 1
            wins += int(ci["ci95"][1] < 0)
            relative_improvements.append((ref_score - cand_score) / ref_score if ref_score > 0 else 0.0)
        median_improvement = float(np.median(relative_improvements)) if relative_improvements else None
        # Conservative gate: at least 6 of 8 path-horizon folds improve with CI below zero and >=5% median gain.
        by_fold_path15 = [r for r in evaluated if r.get("target") == "path" and r.get("horizon") == 15 and candidate in r.get("paired_daily_brier_difference", {})]
        pass_folds = sum(r["paired_daily_brier_difference"][candidate]["ci95"][1] < 0 for r in by_fold_path15)
        med15 = float(np.median([(r["models"][reference]["brier"] - r["models"][candidate]["brier"]) / r["models"][reference]["brier"]
                                 for r in by_fold_path15 if r["models"][reference]["brier"] > 0])) if by_fold_path15 else 0.0
        decisions[candidate] = {"reference": reference, "compared_fold_target_horizons": total,
                                "ci_improvement_count": wins, "median_relative_brier_improvement_all": median_improvement,
                                "path_15m_folds_ci_below_zero": int(pass_folds),
                                "path_15m_median_relative_brier_improvement": med15,
                                "gate_pass": bool(len(by_fold_path15) == 8 and pass_folds >= 6 and med15 >= .05)}
    passing = [name for name, d in decisions.items() if d["gate_pass"]]
    # Complexity ladder: choose the simplest passing model; if none, do not select.
    selected = passing[0] if passing else None
    return {"candidates": decisions, "selected_model": selected,
            "status": "PASS_CANDIDATE" if selected else "NO_MODEL",
            "reason": "Requires at least 6/8 path-15m quarterly paired daily-block CIs below zero and median relative Brier improvement >=5%; no candidate passed." if not selected else "Simplest candidate meeting the predeclared R07 path-15m gate."}


def run(market: str = "KOSPI") -> tuple[pd.DataFrame, dict]:
    if market != "KOSPI":
        raise ValueError("R07 primary execution is KOSPI only; KOSDAQ needs its own eligible 1m file and separate run.")
    minutes = valid_dev_minutes(MINUTE / "KOSPI_TOSS_1m.csv")
    features = add_future_path_er(add_future_anchors(build_path_features(minutes)))
    features["timestamp"] = pd.to_datetime(features["timestamp"])
    features["clock_bucket"] = (features["minute"] // 10).astype(int)
    scores = _load_r03_scores(market)
    features = _attach_shift_scores(features, scores)
    features = add_shift_bins(features)
    all_reports = []
    for period in FOLDS:
        fold_reports, _ = _run_fold(features, period)
        all_reports.extend(fold_reports)
    selection = _select_candidate(all_reports)
    report = {
        "stage": "R07_QUARTERLY_EXPANDING_ORIGIN_TOURNAMENT",
        "status": selection["status"],
        "market": market,
        "dev_start": str(DEV_START.date()),
        "dev_end": str(DEV_END.date()),
        "folds": [str(p) for p in FOLDS],
        "training": "Expanding DEV through the fold cutoff; 60 observed-minute purge; no 2026 rows read.",
        "targets": {"path": list(PATH_CLASSES), "volatility": list(VOL_CLASSES)},
        "models": {name: list(fields) for name, fields in CANDIDATES.items()},
        "smoothing_alpha": SMOOTH_ALPHA,
        "bootstrap": {"unit": "paired daily mean Brier loss", "block_days": 5, "replicates": BOOTSTRAP_REPLICATES, "seed": BOOTSTRAP_SEED},
        "selection_rule": "For 15-minute path class: at least 6/8 quarterly paired daily-block 95% CIs show Brier improvement and median relative Brier improvement >=5%; simplest passing model wins. Otherwise NO_MODEL.",
        "selection": selection,
        "fold_results": all_reports,
        "prediction_rows_persisted": 0,
        "caveats": [
            "This is a predictive-label tournament, not evidence that future path labels are the true latent regime.",
            "Candidate state rules and R03 distance null-percentiles are diagnostic inputs; the winner must not be treated as a trading approval without R08.",
            "Future targets are used only as training labels/held-out outcomes and never as feature keys.",
            "No event-time true change point is observable; this R07 run does not claim natural-market detection latency/FPR.",
            "No model is selected if evidence fails the predeclared fold-consistency and effect-size rule.",
            "All computations are limited to KOSPI DEV; no KOSDAQ file or 2026 period is accessed."
        ],
        "decision": selection["status"],
    }
    return pd.DataFrame(), report


def main() -> None:
    parser = argparse.ArgumentParser(description="R07 DEV expanding-origin model tournament")
    parser.add_argument("--market", choices=("KOSPI",), default="KOSPI")
    args = parser.parse_args()
    _, report = run(args.market)
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "r07_walkforward_report_dev_kospi.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps({"status": report["status"], "fold_results": len(report["fold_results"]),
                      "selected_model": report["selection"]["selected_model"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
