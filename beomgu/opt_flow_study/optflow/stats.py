"""3~8단계 통계: 5분위, Newey-West HAC 회귀, 연도별, 순환 시프트 플라시보, 사전등록 판정.

HAC는 외부 패키지 없이 구현한다(statsmodels OLS cov_type='HAC', use_correction=True 와 같은 값이 나오도록
Bartlett 가중 + n/(n−k) 보정, 정규분포 p). selftest.py 가 statsmodels 와 수치를 대조한다.
"""
import math

import numpy as np
import pandas as pd


def _dstr(v):
    return str(v.date()) if hasattr(v, "date") else str(v)


def _nw_cov(X, e, lags):
    n, k = X.shape
    Xe = X * e[:, None]
    S = Xe.T @ Xe
    for lag in range(1, lags + 1):
        w = 1.0 - lag / (lags + 1.0)
        G = Xe[lag:].T @ Xe[:-lag]
        S = S + w * (G + G.T)
    inv = np.linalg.inv(X.T @ X)
    return inv @ S @ inv * (n / (n - k))


def hac_ols(y, X, lags=5, min_n=30):
    """y: Series, X: DataFrame. 상수항 자동 추가, 공통 결측 제거."""
    if isinstance(X, pd.Series):
        X = X.to_frame()
    d = pd.concat([y.rename("_y"), X], axis=1).dropna()
    n = len(d)
    if n < min_n:
        return {"n": n}
    names = ["const"] + list(X.columns)
    yv = d["_y"].to_numpy(float)
    xv = d.drop(columns="_y").to_numpy(float)
    Xv = np.column_stack([np.ones(n), xv])
    if np.linalg.matrix_rank(Xv) < Xv.shape[1]:
        return {"n": n, "singular": True}        # 상수열·완전 공선 → 추정 불가(예외 대신 표시)
    b, *_ = np.linalg.lstsq(Xv, yv, rcond=None)
    e = yv - Xv @ b
    se = np.sqrt(np.diag(_nw_cov(Xv, e, lags)))
    t = b / se
    p = np.array([math.erfc(abs(v) / math.sqrt(2.0)) for v in t])
    yc = yv - yv.mean()
    r2 = 1.0 - (e @ e) / (yc @ yc)
    sdx = xv.std(axis=0, ddof=1)
    sdy = yv.std(ddof=1)
    return {"n": n, "start": _dstr(d.index.min()), "end": _dstr(d.index.max()), "r2": float(r2),
            "coef": dict(zip(names, b.tolist())), "se": dict(zip(names, se.tolist())),
            "t": dict(zip(names, t.tolist())), "p": dict(zip(names, p.tolist())),
            "beta_per_1sd": dict(zip(names[1:], (b[1:] * sdx).tolist())),
            "std_beta": dict(zip(names[1:], (b[1:] * sdx / sdy).tolist()))}


def quintile_table(y, x, q=5, min_per_bin=10):
    """전체표본 분위(서술용). 동점은 순위 우선으로 나눔."""
    d = pd.concat([y.rename("y"), x.rename("x")], axis=1).dropna()
    if len(d) < q * min_per_bin:
        return None
    d["q"] = pd.qcut(d["x"].rank(method="first"), q, labels=list(range(1, q + 1)))
    g = d.groupby("q", observed=True).agg(mean=("y", "mean"), n=("y", "size"), sd=("y", "std"),
                                          x_lo=("x", "min"), x_hi=("x", "max"))
    means = g["mean"].to_numpy()
    return {"table": g, "q5_q1": float(means[-1] - means[0]), "steps_up": int((np.diff(means) > 0).sum()), "n": len(d)}


def yearly_table(y, x, lags=5, min_n=30):
    d = pd.concat([y.rename("y"), x.rename("x")], axis=1).dropna()
    rows = []
    for yr, g in d.groupby(d.index.year):
        r = hac_ols(g["y"], g[["x"]], lags=lags, min_n=min_n)
        qt = quintile_table(g["y"], g["x"])
        rows.append({"year": int(yr), "n": len(g),
                     "beta": r.get("coef", {}).get("x", np.nan), "p": r.get("p", {}).get("x", np.nan),
                     "q5_q1": qt["q5_q1"] if qt else np.nan})
    return pd.DataFrame(rows)


def circular_shift_placebo(y, x, n_perm=5000, min_shift=20, seed=20260927):
    """신호열을 무작위 오프셋만큼 회전(자기상관·변동성 군집 보존). 단측 p = (1+#{β* ≥ β})/(N+1)."""
    d = pd.concat([y.rename("y"), x.rename("x")], axis=1).dropna()
    n = len(d)
    if n < 2 * min_shift + 10:
        return {"n": n}
    yc = d["y"].to_numpy(float) - d["y"].mean()
    xc = d["x"].to_numpy(float) - d["x"].mean()
    sxx = xc @ xc
    beta = (xc @ yc) / sxx
    rng = np.random.default_rng(seed)
    ks = rng.integers(min_shift, n - min_shift, size=n_perm)
    betas = np.fromiter(((np.roll(xc, int(k)) @ yc) / sxx for k in ks), dtype=float, count=n_perm)
    return {"n": n, "beta_obs": float(beta), "perm_mean": float(betas.mean()), "perm_sd": float(betas.std()),
            "p_one_sided": float((1 + (betas >= beta).sum()) / (n_perm + 1)), "n_perm": n_perm, "min_shift": min_shift, "seed": seed}


def circular_shift_placebo_studentized(y, x, n_perm=2000, min_shift=20, seed=20260927, lags=5):
    """보고용: 같은 회전을 HAC t 통계량으로. 신호 크기와 수익률 변동성이 함께 움직이는 이분산을 반영한다.
    (β 회전은 이분산을 무시해 귀무분포가 OLS 표준오차 폭으로 좁아진다 — 2026-09-27 실데이터에서 확인)"""
    d = pd.concat([y.rename("y"), x.rename("x")], axis=1).dropna()
    n = len(d)
    if n < 2 * min_shift + 10:
        return {"n": n}
    t_obs = hac_ols(d["y"], d[["x"]], lags=lags)["t"]["x"]
    rng = np.random.default_rng(seed)
    xv = d["x"].to_numpy(float)
    ts = np.array([hac_ols(d["y"], pd.DataFrame({"x": np.roll(xv, int(k))}, index=d.index), lags=lags)["t"]["x"]
                   for k in rng.integers(min_shift, n - min_shift, size=n_perm)])
    return {"n": n, "t_obs": float(t_obs), "perm_t_sd": float(ts.std()), "p_one_sided": float((1 + (ts >= t_obs).sum()) / (n_perm + 1)), "n_perm": n_perm}


def evaluate_prereg(res, prereg):
    """사전등록 기준 6개. res 는 run_study 가 만든 결과 딕셔너리."""
    a = prereg["alpha"]
    reg, qt, yr, ca, cb, sub, pl = (res.get(k) for k in ("reg", "quintile", "yearly", "control_prev", "control_prev_fut", "subperiod", "placebo"))
    sig = "x"

    def pos_sig(r, name):
        return bool(r and "coef" in r and r["coef"][name] > 0 and r["p"][name] < a)

    if yr is not None and len(yr):
        est = yr[(yr["n"] >= prereg["year_min_n"]) & yr["beta"].notna()]
        yr_share = float((est["beta"] > 0).mean()) if len(est) else float("nan")
        yr_note = f"실제 {yr_share:.0%}, 추정 {len(est)}개 연도" + (f", 제외 {sorted(set(yr['year']) - set(est['year']))}" if len(est) < len(yr) else "")
    else:
        yr_share, yr_note = float("nan"), "연도 없음"
    crit = [
        ("1 전체 β>0, HAC p<0.05", pos_sig(reg, sig)),
        ("2 Q5−Q1>0", bool(qt and qt["q5_q1"] > 0)),
        (f"3 β>0 연도 비율 ≥ {prereg['year_share_min']:.0%} ({yr_note})", bool(yr_share >= prereg["year_share_min"])),
        ("4 전일수익률 통제 후 β>0·p<0.05, 외인선물까지 통제 후 β>0·p<0.05", pos_sig(ca, sig) and pos_sig(cb, sig)),
        (f"5 {prereg['subperiod_start'][:4]}~ 구간 β>0", bool(sub and "coef" in sub and sub["coef"][sig] > 0)),
        ("6 순환 시프트 플라시보 p ≤ 0.05", bool(pl and "p_one_sided" in pl and pl["p_one_sided"] <= a)),
    ]
    return crit, all(v for _, v in crit)
