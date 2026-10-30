"""R03 causal distribution-shift diagnostics for KOSPI/KOSDAQ (spec section 5).

No threshold/model is adopted here. Distances are diagnostic inputs for later
R06/R07 ablation. Windows are within a contiguous session segment, with a
non-overlapping baseline immediately preceding the recent window. A fixed JS
histogram is fit from the first 60 DEV trading sessions only. Scores are sampled
on a five-minute clock grid to keep the exploratory computation reproducible.

Run: python3 regime_r03_distribution.py [--market KOSPI|KOSDAQ]
Outputs are written only under jehyeong/dollarbar/results/regime_research/.
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import anderson_ksamp, ks_2samp, wasserstein_distance

from regime_r00_validate import DEV_END, DEV_START, MINUTE, OUT, load_minute
from regime_r02_path import valid_dev_minutes

TRAINING_SESSIONS = 60
BLOCK_LENGTH = 5
NULL_REPLICATES = 200
GRID_MINUTES = 5
WINDOWS = ((30, 120), (60, 120))  # (recent, immediately preceding baseline)


def _returns_by_segment(market: str) -> pd.DataFrame:
    path = MINUTE / f"{market}_TOSS_1m.csv"
    x = valid_dev_minutes(path)
    prev_date = x["date"].shift(1)
    prev_min = x["minute"].shift(1)
    continuous = x["date"].eq(prev_date) & x["minute"].sub(prev_min).eq(1)
    x["ret"] = np.log(x["close"] / x["close"].shift(1)).where(continuous)
    x["segment_id"] = (~continuous).cumsum()
    return x


def _fit_js_edges(x: pd.DataFrame) -> np.ndarray:
    """Fit and freeze 16 histogram bins from the initial training sessions."""
    dates = sorted(x["date"].unique())[:TRAINING_SESSIONS]
    sample = x.loc[x["date"].isin(dates), "ret"].dropna().to_numpy()
    if len(sample) < 1000:
        raise ValueError(f"insufficient training returns for fixed JS edges: {len(sample)}")
    lo, hi = np.quantile(sample, [0.005, 0.995])
    if not np.isfinite([lo, hi]).all() or lo >= hi:
        raise ValueError("degenerate fixed JS training range")
    edges = np.linspace(lo, hi, 17)
    edges[0], edges[-1] = -np.inf, np.inf
    return edges


def distance_stats(baseline: np.ndarray, recent: np.ndarray, edges: np.ndarray) -> dict[str, float]:
    """Compute raw distances; no SciPy asymptotic p-values are used for inference."""
    if len(baseline) < 20 or len(recent) < 20:
        raise ValueError("both samples need at least 20 returns")
    if not np.isfinite(baseline).all() or not np.isfinite(recent).all():
        raise ValueError("nonfinite return in distance sample")
    mad = float(np.median(np.abs(baseline - np.median(baseline)))) * 1.4826
    w1 = float(wasserstein_distance(baseline, recent))
    # The approximate SciPy p-value is deliberately ignored; avoid emitting its
    # clipping warning for every rolling window while preserving the statistic.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        ad = anderson_ksamp([baseline, recent], variant="midrank")
    ks = ks_2samp(baseline, recent, method="auto")
    hb, _ = np.histogram(baseline, bins=edges)
    hr, _ = np.histogram(recent, bins=edges)
    pb = hb.astype(float) + 0.5
    pr = hr.astype(float) + 0.5
    pb /= pb.sum()
    pr /= pr.sum()
    js = float(jensenshannon(pb, pr, base=2.0) ** 2)
    return {
        "w1": w1,
        "w1_norm": w1 / max(mad, np.finfo(float).eps),
        "ad2_stat": float(ad.statistic),
        "ks_stat": float(ks.statistic),
        "js_divergence": js,
        "vol_ratio": float(np.mean(recent ** 2) / max(np.mean(baseline ** 2), np.finfo(float).eps)),
    }


def _sample_blocks(pool: np.ndarray, n: int, rng: np.random.Generator, block_length: int) -> np.ndarray:
    """Moving-block resample only from finite contiguous runs; never cross sessions/gaps."""
    a = np.asarray(pool, dtype=float)
    if len(a) < block_length:
        raise ValueError("null pool shorter than block length")
    valid = np.isfinite(a).astype(np.int64)
    valid_counts = np.convolve(valid, np.ones(block_length, dtype=np.int64), mode="valid")
    starts = np.flatnonzero(valid_counts == block_length)
    if len(starts) == 0:
        raise ValueError("null pool has no finite block of requested length")
    out: list[float] = []
    while len(out) < n:
        start = int(rng.choice(starts))
        out.extend(a[start:start + block_length].tolist())
    return np.asarray(out[:n], dtype=float)


def build_block_null(pool: np.ndarray, recent_n: int, baseline_n: int, seed: int = 42) -> dict:
    """Estimate no-change reference distributions with moving blocks from DEV train data."""
    rng = np.random.default_rng(seed + recent_n * 1000 + baseline_n)
    edges = _fit_js_edges(pd.DataFrame({"date": np.repeat("train", len(pool)), "ret": pool}))
    # For this helper's synthetic tests/training subset, use the supplied pool directly.
    draws: dict[str, list[float]] = {k: [] for k in ("w1_norm", "ad2_stat", "ks_stat", "js_divergence")}
    for _ in range(NULL_REPLICATES):
        # One concatenated null sample preserves the adjacency of the two windows.
        combined = _sample_blocks(pool, baseline_n + recent_n, rng, BLOCK_LENGTH)
        b = combined[:baseline_n]
        r = combined[baseline_n:]
        stats = distance_stats(b, r, edges)
        for key in draws:
            draws[key].append(stats[key])
    return {
        "replicates": NULL_REPLICATES,
        "block_length": BLOCK_LENGTH,
        "training_returns": int(np.isfinite(pool).sum()),
        "threshold_p95": {k: float(np.quantile(v, 0.95)) for k, v in draws.items()},
        "null_values": draws,
    }


def empirical_percentile(value: float, null_values: list[float]) -> float:
    a = np.asarray(null_values, dtype=float)
    return float((1 + np.count_nonzero(a <= value)) / (len(a) + 1))


def compute_scores(market: str = "KOSPI", path: Path | None = None) -> tuple[pd.DataFrame, dict]:
    if market not in {"KOSPI", "KOSDAQ"}:
        raise ValueError(f"unsupported market: {market}")
    if path is None:
        path = MINUTE / f"{market}_TOSS_1m.csv"
    x = _returns_by_segment(market) if path == MINUTE / f"{market}_TOSS_1m.csv" else _load_custom(path)
    # The initial 60 trading sessions are threshold/null training only; no scores there.
    training_dates = set(sorted(x["date"].unique())[:TRAINING_SESSIONS])
    training = x[x["date"].isin(training_dates)]
    train_segments = [g["ret"].dropna().to_numpy(dtype=float) for _, g in training.groupby("segment_id", sort=False)]
    train_segments = [a for a in train_segments if len(a) >= BLOCK_LENGTH]
    # NaN separators make session/gap boundaries unavailable to the block sampler.
    train_returns = np.concatenate([np.r_[a, np.nan] for a in train_segments])
    n_train_returns = int(np.isfinite(train_returns).sum())
    if n_train_returns < 1000:
        raise ValueError(f"insufficient R03 calibration data: {n_train_returns} training returns")
    edges = _fit_js_edges(x)
    nulls = {f"{r}_{b}": build_block_null(train_returns, r, b) for r, b in WINDOWS}
    rows: list[dict] = []
    for segment, g in x.groupby("segment_id", sort=False):
        g = g.reset_index(drop=True)
        # Decision scores only on a fixed 5-minute KST grid. No row after t is accessed.
        for i in range(len(g)):
            if g.loc[i, "date"] in training_dates or int(g.loc[i, "minute"]) % GRID_MINUTES:
                continue
            for recent_n, baseline_n in WINDOWS:
                ret_pos = i  # current confirmed bar's return is available at its bar_end.
                recent_start = ret_pos - recent_n + 1
                baseline_end = recent_start
                baseline_start = baseline_end - baseline_n
                if baseline_start < 0:
                    continue
                baseline = g.loc[baseline_start:baseline_end - 1, "ret"].to_numpy(dtype=float)
                recent = g.loc[recent_start:ret_pos, "ret"].to_numpy(dtype=float)
                if len(baseline) != baseline_n or len(recent) != recent_n:
                    continue
                if not np.isfinite(baseline).all() or not np.isfinite(recent).all():
                    continue
                # Segment boundaries guarantee these windows do not cross session/gaps.
                scores = distance_stats(baseline, recent, edges)
                null = nulls[f"{recent_n}_{baseline_n}"]
                row = {"timestamp": g.loc[i, "timestamp"], "date": g.loc[i, "date"],
                       "minute": int(g.loc[i, "minute"]), "segment_id": int(segment),
                       "recent_n": recent_n, "baseline_n": baseline_n,
                       "recent_start": g.loc[recent_start, "timestamp"],
                       "baseline_start": g.loc[baseline_start, "timestamp"],
                       "baseline_end": g.loc[baseline_end - 1, "timestamp"],
                       **scores}
                for key, value in scores.items():
                    if key in null["null_values"]:
                        row[f"{key}_null_pct"] = empirical_percentile(value, null["null_values"][key])
                rows.append(row)
    scores = pd.DataFrame(rows)
    report = {
        "stage": "R03_RAW_DISTANCES_AND_BLOCK_NULL_DIAGNOSTICS",
        "status": "DIAGNOSTIC_ONLY_R07_INCREMENTAL_SELECTION_PENDING",
        "market": market,
        "dev_start": str(DEV_START.date()),
        "dev_end": str(DEV_END.date()),
        "training_sessions_for_fixed_edges_and_null": TRAINING_SESSIONS,
        "training_returns": n_train_returns,
        "grid_minutes": GRID_MINUTES,
        "window_pairs_recent_baseline": [list(x) for x in WINDOWS],
        "js_edges": edges.tolist(),
        "block_null": {k: {kk: vv for kk, vv in v.items() if kk != "null_values"} for k, v in nulls.items()},
        "rows": int(len(scores)),
        "score_dates": int(scores["date"].nunique()) if len(scores) else 0,
        "coverage_by_clock": ({str(k): int(v) for k, v in scores.groupby(scores["timestamp"].dt.strftime("%H:%M")).size().items()} if len(scores) else {}),
        "distance_summary": ({k: {"median": float(scores[k].median()), "p95": float(scores[k].quantile(.95)),
                                   "p99": float(scores[k].quantile(.99))}
                              for k in ("w1_norm", "ad2_stat", "ks_stat", "js_divergence", "vol_ratio")} if len(scores) else {}),
        "decision": "NO_DISTANCE_ADOPTED; REQUIRE_PAIRED_R07_ABLATION_AT_FIXED_FPR_PER_DAY",
        "caveats": [
            "Initial 60 DEV sessions train fixed JS edges and empirical no-change block null; no 2026 values are read.",
            "Block null uses contiguous five-return moving blocks and paired adjacent baseline/recent samples; sensitivity to block length and day-level null calibration remain required before operating thresholds.",
            "SciPy AD/KS p-values are not used for inference; raw statistics and block-null percentiles are recorded.",
            "Scores are computed every five minutes, not each minute; do not use as a one-minute detector without a separately validated carry-forward policy.",
            "This stage does not compare predictive scores against S0/S1/S2; feature adoption remains pending R07.",
            "Return windows are non-overlapping and do not cross a session or missing-minute segment."
        ],
    }
    return scores, report


def _load_custom(path: Path) -> pd.DataFrame:
    """Schema-compatible helper for tests; applies DEV/session/segment rules."""
    x = load_minute(path)
    x = x[(x["timestamp"] >= DEV_START) & (x["timestamp"] < DEV_END + pd.Timedelta(days=1))].copy()
    x = x[x["timestamp"].dt.strftime("%H:%M").between("09:01", "15:30")].copy()
    x["minute"] = x["timestamp"].dt.hour * 60 + x["timestamp"].dt.minute
    counts = x.groupby("date")["timestamp"].nunique()
    bad = set(counts.index[(390 - counts) >= 30])
    x = x[~x["date"].isin(bad)].sort_values("timestamp").reset_index(drop=True)
    continuous = x["date"].eq(x["date"].shift()) & x["minute"].sub(x["minute"].shift()).eq(1)
    x["ret"] = np.log(x["close"] / x["close"].shift(1)).where(continuous)
    x["segment_id"] = (~continuous).cumsum()
    return x


def main() -> None:
    parser = argparse.ArgumentParser(description="DEV-only R03 distribution diagnostics")
    parser.add_argument("--market", choices=("KOSPI", "KOSDAQ"), default="KOSPI")
    args = parser.parse_args()
    scores, report = compute_scores(args.market)
    OUT.mkdir(parents=True, exist_ok=True)
    suffix = args.market.lower()
    scores.to_csv(OUT / f"r03_distribution_scores_dev_{suffix}.csv", index=False)
    with open(OUT / f"r03_distribution_report_dev_{suffix}.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps({k: report[k] for k in ("market", "status", "rows", "score_dates", "decision")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
