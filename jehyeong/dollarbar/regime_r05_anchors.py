"""R05 future-anchor evaluation, kept strictly separate from model inputs.

For each 2024-2025 quarter, fit operational label thresholds only on earlier DEV
observations whose 60-minute target end precedes the fold. Evaluate descriptive
P1/P2/P5 path states on that quarter's future outcomes. No 2026 input or label is
loaded. Results are evaluation summaries, not a final model selection.

Run: python3 regime_r05_anchors.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from regime_r00_validate import DEV_END, DEV_START, MINUTE, OUT
from regime_r02_path import HORIZONS, add_future_anchors, build_path_features, valid_dev_minutes

PURGE_MINUTES = 60
CLOCK_BUCKET_MINUTES = 10
BOOTSTRAP_REPLICATES = 1000
FOLDS = pd.period_range("2024Q1", "2025Q4", freq="Q")
STATE_COLUMNS = ("state_p1", "state_p2", "state_p5")


def add_future_path_er(x: pd.DataFrame) -> pd.DataFrame:
    """Add path efficiency of each future horizon strictly as target columns."""
    y = x.copy()
    groups = y.groupby("segment_id", sort=False)
    for h in HORIZONS:
        net = np.log(groups["close"].shift(-h) / y["close"])
        path = sum(y.groupby("segment_id")["ret_1m"].shift(-j).abs() for j in range(1, h + 1))
        y[f"future_er_{h}"] = (net.abs() / path.replace(0, np.nan)).clip(0, 1)
    return y


def _clock_bucket(minute: pd.Series) -> pd.Series:
    return (minute // CLOCK_BUCKET_MINUTES).astype(int)


def _training_cutoff(fold_start: pd.Timestamp, features: pd.DataFrame) -> pd.Timestamp:
    """Decision cutoff 60 valid intraday observations before held-out fold start.

    Count only prior complete session observations on the chronological DEV index;
    this avoids treating overnight/weekend wall-clock minutes as purge samples.
    """
    prior = features.loc[features["timestamp"] < fold_start, "timestamp"].drop_duplicates().sort_values()
    if len(prior) <= PURGE_MINUTES:
        return fold_start
    return pd.Timestamp(prior.iloc[-PURGE_MINUTES])


def _fit_anchor_thresholds(train: pd.DataFrame) -> dict:
    """Fit clock/horizon RV and direction scales from prior DEV outcomes only."""
    thresholds = {}
    for h in HORIZONS:
        col = f"future_rv_{h}"
        valid = train[["clock_bucket", col, f"future_ret_{h}"]].dropna()
        if valid.empty:
            thresholds[str(h)] = {"fallback": True, "global": None}
            continue
        global_sigma = float(valid[f"future_ret_{h}"].std(ddof=1))
        global_q40, global_q75 = (float(v) for v in valid[col].quantile([.40, .75]))
        buckets = {}
        for bucket, g in valid.groupby("clock_bucket"):
            if len(g) < 100:
                continue
            buckets[str(int(bucket))] = {
                "sigma_ret": float(g[f"future_ret_{h}"].std(ddof=1)),
                "rv_q40": float(g[col].quantile(.40)),
                "rv_q75": float(g[col].quantile(.75)),
                "n": int(len(g)),
            }
        thresholds[str(h)] = {
            "global": {"sigma_ret": global_sigma, "rv_q40": global_q40, "rv_q75": global_q75, "n": int(len(valid))},
            "by_clock_bucket": buckets,
        }
    return thresholds


def _apply_anchor_labels(test: pd.DataFrame, thresholds: dict) -> pd.DataFrame:
    x = test.copy()
    for h in HORIZONS:
        cfg = thresholds[str(h)]
        global_values = cfg.get("global")
        for key in ("sigma_ret", "rv_q40", "rv_q75"):
            lookup = {int(k): v[key] for k, v in cfg.get("by_clock_bucket", {}).items()}
            fallback = global_values[key] if global_values else np.nan
            x[f"train_{key}_{h}"] = x["clock_bucket"].map(lookup).fillna(fallback)
        rv = x[f"future_rv_{h}"]
        eligible_rv = rv.notna() & x[f"train_rv_q40_{h}"].notna() & x[f"train_rv_q75_{h}"].notna()
        vol_label = pd.Series(pd.NA, index=x.index, dtype="string")
        vol_label.loc[eligible_rv & rv.le(x[f"train_rv_q40_{h}"])] = "CALM"
        vol_label.loc[eligible_rv & rv.ge(x[f"train_rv_q75_{h}"])] = "VOLATILE"
        vol_label.loc[eligible_rv & vol_label.isna()] = "MID"
        x[f"future_vol_class_{h}"] = vol_label

        ret = x[f"future_ret_{h}"]
        eligible_path = ret.notna() & x[f"future_er_{h}"].notna() & x[f"train_sigma_ret_{h}"].notna()
        clear = eligible_path & x[f"future_er_{h}"].gt(.5) & ret.abs().ge(.5 * x[f"train_sigma_ret_{h}"])
        path_label = pd.Series(pd.NA, index=x.index, dtype="string")
        path_label.loc[clear & ret.gt(0)] = "UP"
        path_label.loc[clear & ret.lt(0)] = "DOWN"
        path_label.loc[eligible_path & path_label.isna()] = "SIDEWAYS"
        x[f"future_path_label_{h}"] = path_label
        # These labels exist only in this evaluation copy; feature construction never sees them.
    return x


def _daily_block_summary(frame: pd.DataFrame, value: str, seed: int) -> dict:
    """Circular moving-block bootstrap of daily means; intraday rows stay grouped."""
    daily = frame.groupby("date")[value].mean().dropna().to_numpy(dtype=float)
    if not len(daily):
        return {"n_days": 0, "mean": None, "ci95": [None, None], "bootstrap_block_days": 5}
    rng = np.random.default_rng(seed)
    block_days = min(5, len(daily))
    means = np.empty(BOOTSTRAP_REPLICATES)
    for i in range(BOOTSTRAP_REPLICATES):
        draw: list[float] = []
        while len(draw) < len(daily):
            start = int(rng.integers(0, len(daily)))
            draw.extend(daily[(start + np.arange(block_days)) % len(daily)].tolist())
        means[i] = np.mean(draw[:len(daily)])
    return {
        "n_days": int(len(daily)),
        "mean": float(daily.mean()),
        "ci95": [float(np.quantile(means, .025)), float(np.quantile(means, .975))],
        "bootstrap_block_days": int(block_days),
    }


def _evaluate_fold(test: pd.DataFrame, fold: str) -> dict:
    result: dict = {"fold": fold, "rows": int(len(test)), "dates": int(test["date"].nunique()), "states": {}}
    for state_col in STATE_COLUMNS:
        result["states"][state_col] = {"by_state": {}}
        for state, group in test.groupby(state_col, sort=True):
            s = {"rows": int(len(group)), "coverage": float(len(group) / max(1, len(test)))}
            # Evaluate horizons separately. Same-session targets must be fully eligible.
            for h in HORIZONS:
                eligible = group[group[f"future_ret_{h}"].notna()].copy()
                s[f"eligible_rows_{h}"] = int(len(eligible))
                s[f"eligible_dates_{h}"] = int(eligible["date"].nunique())
                for metric in (f"future_ret_{h}", f"future_rv_{h}", f"future_er_{h}",
                               f"continuation_{h}", f"reversion_{h}"):
                    s[metric] = _daily_block_summary(eligible, metric, seed=42 + h)
                path_label = eligible[f"future_path_label_{h}"].dropna()
                s[f"future_path_label_{h}_rate"] = {str(k): float(v) for k, v in path_label.value_counts(normalize=True).items()}
                vol_label = eligible[f"future_vol_class_{h}"].dropna()
                s[f"future_vol_class_{h}_rate"] = {str(k): float(v) for k, v in vol_label.value_counts(normalize=True).items()}
            result["states"][state_col]["by_state"][str(state)] = s
    return result


def run() -> tuple[pd.DataFrame, dict]:
    minutes = valid_dev_minutes(MINUTE / "KOSPI_TOSS_1m.csv")
    features = add_future_path_er(add_future_anchors(build_path_features(minutes)))
    features["timestamp"] = pd.to_datetime(features["timestamp"])
    features["clock_bucket"] = _clock_bucket(features["minute"])
    fold_reports = []
    evaluated = []
    fold_thresholds = {}
    for period in FOLDS:
        start = period.start_time
        end = (period + 1).start_time
        if end > DEV_END + pd.Timedelta(days=1):
            end = DEV_END + pd.Timedelta(days=1)
        cutoff = _training_cutoff(start, features)
        # Label windows must be wholly observable before the held-out quarter begins.
        train = features[(features["timestamp"] < cutoff) & (features["date"] < start.strftime("%Y-%m-%d"))].copy()
        test = features[(features["timestamp"] >= start) & (features["timestamp"] < end)].copy()
        if train.empty or test.empty:
            fold_reports.append({"fold": str(period), "status": "INSUFFICIENT_TRAIN_OR_TEST", "train_rows": len(train), "test_rows": len(test)})
            continue
        train = train[train[f"future_rv_{max(HORIZONS)}"].notna()]
        thresholds = _fit_anchor_thresholds(train)
        fold_thresholds[str(period)] = thresholds
        # Keep full held-out evaluation rows in memory only; save only compact summaries.
        test = _apply_anchor_labels(test, thresholds)
        fold_reports.append({
            "fold": str(period),
            "status": "EVALUATED",
            "train_end_exclusive": str(cutoff),
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
            "test_dates": int(test["date"].nunique()),
            "threshold_fit_scope": "prior DEV only; clock-bucket/horizon thresholds with global fallback",
            "purge_minutes": PURGE_MINUTES,
            "summary": _evaluate_fold(test, str(period)),
        })
        keep = ["timestamp", "date", "minute", "state_p1", "state_p2", "state_p5", "clock_bucket"]
        keep += [f"future_{name}_{h}" for h in HORIZONS for name in ("ret", "rv", "er")]
        keep += [f"continuation_{h}" for h in HORIZONS] + [f"reversion_{h}" for h in HORIZONS]
        keep += [f"future_vol_class_{h}" for h in HORIZONS] + [f"future_path_label_{h}" for h in HORIZONS]
        evaluated.append(test[keep].assign(fold=str(period)))
    eval_frame = pd.concat(evaluated, ignore_index=True) if evaluated else pd.DataFrame()
    report = {
        "stage": "R05_FUTURE_ANCHOR_EVALUATION",
        "status_detail": "All results are DEV quarterly expanding-origin evaluation only; no model selected.",
        "status": "DEV_EVALUATION_ONLY; NO_MODEL_SELECTION",
        "market": "KOSPI",
        "dev_start": str(DEV_START.date()),
        "dev_end": str(DEV_END.date()),
        "folds": fold_reports,
        "thresholds_by_fold": fold_thresholds,
        "metrics": "Daily-mean moving block bootstrap CI (1000 resamples); conditional state summaries, not independent per-minute inference.",
        "caveats": [
            "Future return/RV/path labels are evaluation-only outputs and were not supplied to feature/state construction.",
            "No 2026 source rows or outcomes were loaded.",
            "Thresholds are fit within each expanding DEV fold on prior outcomes only, with clock bucket/horizon grouping and global fallback.",
            "A 60-observation purge counts observed intraday timestamps (not overnight wall-clock minutes) before fold start; targets also end within their same contiguous session segment.",
            "P1/P2/P5 states remain preliminary diagnostic rules. No state is selected and no strategy is approved.",
            "Bootstrap resamples daily state means; overlapping target horizons remain dependent within dates, and quarter-level heterogeneity is retained in separate fold reports."
        ],
        "decision": "R05_ANCHORS_AVAILABLE_FOR_R07; NO_TRADING_OR_TRUE_OOS_APPROVAL",
    }
    return eval_frame, report


def main() -> None:
    frame, report = run()
    OUT.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT / "r05_future_anchor_eval_quarterly_dev_kospi.csv", index=False)
    with open(OUT / "r05_future_anchor_report_dev_kospi.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps({"status": report["status"], "folds": len(report["folds"]), "evaluated_rows": len(frame), "decision": report["decision"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
