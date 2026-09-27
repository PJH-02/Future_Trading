"""엔진 무결성 검사. 옵션 자료 없이 실행된다(합성 데이터 + 실데이터 달력 검사).

사용: python selftest.py [--raw-dir PATH] [--quick] [--require-real]
  --quick        시뮬레이션 횟수를 줄인 참고용 실행(T3 판정 구간이 넓어짐)
  --require-real 원자료가 없어 T7·T11 을 건너뛰면 실패로 처리

검사
  T1 Flow 식·3일 누적               T2 HAC 표준오차 = statsmodels (설치 시)
  T3 귀무(효과 0) 기각률 ≈ 5%        T4 심은 효과 검출
  T5 결측이 있어도 T−1 정렬 유지      T6 룩어헤드 가드(출처 날짜 대조)
  T7 실데이터 달력·KODEX·선물 Flow    T8 옵션 로더: 한글 라벨·기타외국인 합산·total 제외·NaN 유지
  T9 날짜 중복 입력은 오류            T10 상수열 회귀는 예외 대신 singular 표시
  T11 합성 옵션 CSV(total 포함)로 run_study 1~8 끝까지 실행
"""
import argparse
import math
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from optflow import stats as st
from optflow.indicators import build_indicators, flow, lag_to_target

RESULTS = []
HERE = Path(__file__).resolve().parent
RAW_FILES = ["k200_index_daily.csv", "k200_naver_fchart_daily.csv", "kodex200_daily.csv", "kodex200_naver_fchart_daily.csv", "naver_fut_investor_daily.csv"]


def check(name, status, detail=""):
    """status: True=PASS, False=FAIL, None=SKIP"""
    RESULTS.append((name, status, detail))
    tag = {True: "PASS", False: "FAIL", None: "SKIP"}[status]
    print(f"[{tag}] {name} {detail}")


def garch_returns(n, rng):
    s2, e_prev, out = 1.0, 0.0, np.empty(n)
    for t in range(n):
        s2 = 0.05 + 0.10 * e_prev ** 2 + 0.85 * s2
        e = np.sqrt(s2) * rng.standard_t(5) / np.sqrt(5 / 3)
        out[t] = e
        e_prev = e
    return out


def ar1(n, phi, rng):
    x = np.empty(n)
    x[0] = rng.standard_normal()
    for t in range(1, n):
        x[t] = phi * x[t - 1] + rng.standard_normal()
    return x


def t1_flow():
    f = flow(pd.Series([3.0, 0.0, 1.0]), pd.Series([1.0, 0.0, 3.0]))
    ok = np.isclose(f.iloc[0], 0.5) and np.isnan(f.iloc[1]) and np.isclose(f.iloc[2], -0.5)
    idx = pd.bdate_range("2024-01-01", periods=6)
    p = pd.DataFrame({"fut_f_buy_q": [1, 2, 3, 4, 5, 6.0], "fut_f_sell_q": [1, 1, 1, 1, 1, 1.0]}, index=idx)
    ind = build_indicators(p)
    b3, s3 = p["fut_f_buy_q"].rolling(3).sum(), p["fut_f_sell_q"].rolling(3).sum()
    ok3 = np.allclose(ind["futflow_3d"].dropna().to_numpy(), ((b3 - s3) / (b3 + s3)).dropna().to_numpy())
    check("T1 Flow 식·3일 누적", bool(ok and ok3 and np.isnan(ind["futflow_3d"].iloc[1])))


def t2_hac_vs_statsmodels():
    try:
        import statsmodels.api as sm
    except ImportError:
        check("T2 HAC = statsmodels", None, "(statsmodels 미설치)")
        return
    rng = np.random.default_rng(1)
    n = 800
    X = pd.DataFrame({"a": ar1(n, 0.5, rng), "b": rng.standard_normal(n)})
    y = pd.Series(0.2 * X["a"] + garch_returns(n, rng))
    mine = st.hac_ols(y, X, lags=5)
    ref = sm.OLS(y.to_numpy(), sm.add_constant(X.to_numpy())).fit(cov_type="HAC", cov_kwds={"maxlags": 5, "use_correction": True})
    se_ok = np.allclose([mine["se"][k] for k in ("const", "a", "b")], ref.bse, rtol=1e-8)
    p_ok = np.allclose([mine["p"][k] for k in ("const", "a", "b")], ref.pvalues, rtol=1e-6)
    check("T2 HAC = statsmodels", bool(se_ok and p_ok), f"se {np.round(ref.bse, 5)}")


def t3_null(sims, perm, quick):
    rng = np.random.default_rng(2)
    n = 1000
    rej_hac = rej_pl = 0
    for _ in range(sims):
        idx = pd.bdate_range("2010-01-01", periods=n)
        y = pd.Series(garch_returns(n, rng), index=idx)
        x = pd.Series(ar1(n, 0.4, rng), index=idx)
        rej_hac += st.hac_ols(y, x.to_frame("x"), lags=5)["p"]["x"] < 0.05                     # 양측
        rej_pl += st.circular_shift_placebo(y, x, n_perm=perm, min_shift=20, seed=int(rng.integers(1e9)))["p_one_sided"] <= 0.05
    fh, fp = rej_hac / sims, rej_pl / sims
    band = 3 * math.sqrt(0.05 * 0.95 / sims)                                                   # 이항 3SE
    lo, hi = max(0.0, 0.05 - band), 0.05 + band
    ok = lo <= fh <= hi and lo <= fp <= hi
    check("T3 귀무 기각률" + (" (quick: 참고용)" if quick else ""), bool(ok),
          f"HAC {fh:.3f}, 플라시보(단측) {fp:.3f}, 허용 [{lo:.3f}, {hi:.3f}], {sims}회")


def t4_planted(perm):
    rng = np.random.default_rng(3)
    n = 3000
    idx = pd.bdate_range("2010-01-01", periods=n)
    x = pd.Series(ar1(n, 0.3, rng), index=idx)
    x = (x - x.mean()) / x.std()
    y = pd.Series(0.10 * x.to_numpy() + garch_returns(n, rng), index=idx)
    r = st.hac_ols(y, x.to_frame("x"), lags=5)
    q = st.quintile_table(y, x)
    pl = st.circular_shift_placebo(y, x, n_perm=perm)
    ok = r["p"]["x"] < 0.01 and abs(r["coef"]["x"] - 0.10) < 0.05 and q["q5_q1"] > 0 and pl["p_one_sided"] <= 0.01
    check("T4 심은 효과 검출", bool(ok), f"β {r['coef']['x']:.3f} (참값 0.10), p {r['p']['x']:.2e}, 플라시보 p {pl['p_one_sided']:.4f}")


def _mini_panel(n=10):
    cal = pd.bdate_range("2024-01-01", periods=n)
    p = pd.DataFrame(index=cal)
    p["sig_date"] = pd.Series(cal, index=cal).shift(1)
    return cal, p


def t5_alignment_with_gaps():
    cal, p = _mini_panel()
    p["fut_f_buy_q"] = np.arange(1, 11, dtype=float)
    p["fut_f_sell_q"] = 1.0
    p["y_idx"] = 0.0
    p.loc[cal[4], "y_idx"] = np.nan
    ind = build_indicators(p)
    X = lag_to_target(ind, p)
    ok = all(np.isclose(X.loc[cal[i], "futflow_1d"], ind.loc[cal[i - 1], "futflow_1d"]) for i in range(1, 10))
    ok = ok and all(X.loc[cal[i], "sig_date"] == cal[i - 1] for i in range(1, 10))
    check("T5 결측 있어도 T−1 정렬", bool(ok))


def t6_guard():
    cal, p = _mini_panel(5)
    p["fut_f_buy_q"], p["fut_f_sell_q"] = 2.0, 1.0
    p["sig_date"] = pd.Series(cal, index=cal)                   # 일부러 같은 날(룩어헤드)
    try:
        lag_to_target(build_indicators(p), p)
        check("T6 룩어헤드 가드", False, "예외가 나야 함")
    except AssertionError:
        check("T6 룩어헤드 가드", True)


def _synthetic_options(cal, rng, with_total=True, korean=False, split_foreign=False):
    rows = []
    labels = {"foreign": "외국인" if korean else "foreign", "individual": "개인" if korean else "individual",
              "institution": "기관계" if korean else "institution", "total": "전체" if korean else "total"}
    for d in cal:
        tot = {"C": [0.0, 0.0, 0.0, 0.0], "P": [0.0, 0.0, 0.0, 0.0]}
        for inv in ("foreign", "individual", "institution"):
            for cp in ("C", "P"):
                b, s = rng.uniform(1e9, 5e9, 2)
                parts = [(labels[inv], 1.0)]
                if inv == "foreign" and split_foreign:
                    parts = [("외국인" if korean else "foreign", 0.8), ("기타외국인" if korean else "other_foreign", 0.2)]
                for lab, w in parts:
                    rows.append((d, lab, cp, int(w * b / 1e5), int(w * s / 1e5), w * b, w * s))
                tot[cp] = [tot[cp][0] + b / 1e5, tot[cp][1] + s / 1e5, tot[cp][2] + b, tot[cp][3] + s]
        if with_total:
            for cp in ("C", "P"):
                m = (tot[cp][2] + tot[cp][3]) / 2                    # 전체는 매수 = 매도
                rows.append((d, labels["total"], cp, int(m / 1e5), int(m / 1e5), m, m))
    return pd.DataFrame(rows, columns=["date", "investor", "cp", "buy_qty", "sell_qty", "buy_amt", "sell_amt"])


def t8_option_loader():
    from optflow.io import load_options
    rng = np.random.default_rng(8)
    cal = pd.bdate_range("2024-01-01", periods=40)
    with tempfile.TemporaryDirectory() as td:
        raw = Path(td)
        df = _synthetic_options(cal, rng, with_total=True, korean=True, split_foreign=True)
        df.loc[(df["investor"] == "개인") & (df["cp"] == "C") & (df["date"] == cal[3]), "buy_amt"] = np.nan
        df.to_csv(raw / "krx_opt_investor_daily.csv", index=False)
        wide, audit = load_options(raw)
        f = df[(df["investor"].isin(["외국인", "기타외국인"])) & (df["cp"] == "C")].groupby("date")["buy_amt"].sum()
        sum_ok = np.allclose(wide["opt_foreign_C_buy_amt"].to_numpy(), f.reindex(wide.index).to_numpy())
        nan_ok = np.isnan(wide.loc[cal[3], "opt_individual_C_buy_amt"])
        p = wide.copy()
        p["sig_date"] = pd.Series(p.index, index=p.index).shift(1)
        ind = build_indicators(p)
        total_ok = not any(c.startswith("total_") for c in ind.columns) and "cpr_1d" in ind and "optflow_1d" in ind
        ok = sum_ok and nan_ok and total_ok and set(audit["investors_std"]) == {"foreign", "individual", "institution", "total"}
        check("T8 옵션 로더", bool(ok), f"기타외국인 합산 {sum_ok}, NaN 유지 {nan_ok}, total Flow 제외 {total_ok}")


def t9_duplicates():
    from optflow.io import load_options
    rng = np.random.default_rng(9)
    cal = pd.bdate_range("2024-01-01", periods=10)
    with tempfile.TemporaryDirectory() as td:
        raw = Path(td)
        df = _synthetic_options(cal, rng)
        pd.concat([df, df[df["date"] == cal[5]]]).to_csv(raw / "krx_opt_investor_daily.csv", index=False)
        try:
            load_options(raw)
            check("T9 날짜 중복은 오류", False, "예외가 나야 함")
        except ValueError:
            check("T9 날짜 중복은 오류", True)


def t10_singular():
    idx = pd.bdate_range("2024-01-01", periods=100)
    r = st.hac_ols(pd.Series(np.random.default_rng(10).standard_normal(100), index=idx), pd.Series(0.0, index=idx).to_frame("x"))
    check("T10 상수열 회귀", bool(r.get("singular")), f"{r}")


def _raw_or_none(raw_dir):
    from optflow.config import resolve_raw_dir
    try:
        raw = resolve_raw_dir(raw_dir)
    except FileNotFoundError:
        return None
    return raw if all((raw / f).exists() for f in RAW_FILES) else None


def t7_real(raw):
    from optflow.config import PREREG
    from optflow.io import build_panel
    if raw is None:
        check("T7 실데이터", None, "(원자료 없음)")
        return
    panel, audit = build_panel(raw, PREREG["window_start"])
    cal_ok = panel.index.is_monotonic_increasing and panel.index.is_unique
    X = lag_to_target(build_indicators(panel), panel)
    f = pd.read_csv(raw / "naver_fut_investor_daily.csv", parse_dates=["date"]).set_index("date")
    rng = np.random.default_rng(4)
    dates = X.index[X["futflow_1d"].notna()]
    pick = rng.choice(dates, size=min(200, len(dates)), replace=False)
    mism = 0
    for d in pick:
        sd = X.loc[d, "sig_date"]
        b, s = f.loc[sd, "buyq_foreign"], f.loc[sd, "sellq_foreign"]
        mism += not np.isclose(X.loc[d, "futflow_1d"], (b - s) / (b + s))
    kdx_missing = audit["kodex"]["missing_in_window"]
    mx = X["futflow_1d"].abs().max()
    check("T7 실데이터 달력·시프트", bool(cal_ok and mism == 0 and kdx_missing == 0 and mx < 0.5),
          f"달력 {len(panel)}일, 선물 Flow 재계산 불일치 {mism}/{len(pick)}, 분석 구간 KODEX 결측 {kdx_missing}일, 선물 Flow 최대 |x| {mx:.3f}")


def t11_end_to_end(raw):
    if raw is None:
        check("T11 합성 옵션 전 과정", None, "(원자료 없음)")
        return
    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        (tdp / "raw").mkdir()
        for f in RAW_FILES:
            shutil.copy(raw / f, tdp / "raw" / f)
        cal = pd.read_csv(raw / "naver_fut_investor_daily.csv", parse_dates=["date"])["date"]
        _synthetic_options(cal[cal >= "2022-01-01"], np.random.default_rng(11), with_total=True, korean=True, split_foreign=True) \
            .to_csv(tdp / "raw" / "krx_opt_investor_daily.csv", index=False)
        r = subprocess.run([sys.executable, str(HERE / "run_study.py"), "--steps", "1-8", "--raw-dir", str(tdp / "raw"),
                            "--out-dir", str(tdp / "out"), "--perm", "199"], capture_output=True, text=True, encoding="utf-8", errors="replace")
        reports = list((tdp / "out").glob("*/report_optflow_1d.md"))
        ok = r.returncode == 0 and len(reports) == 1 and "NON-PREREG" in reports[0].parent.name
        check("T11 합성 옵션 전 과정", bool(ok), f"exit {r.returncode}, 보고서 {len(reports)}개" + ("" if ok else f" · {r.stderr[-300:]}"))


def main():
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default=None)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--require-real", action="store_true")
    a = ap.parse_args()
    sims, perm = (100, 199) if a.quick else (1000, 499)
    raw = _raw_or_none(a.raw_dir)
    t1_flow()
    t2_hac_vs_statsmodels()
    t3_null(sims, perm, a.quick)
    t4_planted(perm)
    t5_alignment_with_gaps()
    t6_guard()
    t7_real(raw)
    t8_option_loader()
    t9_duplicates()
    t10_singular()
    t11_end_to_end(raw)
    fail = [n for n, s, _ in RESULTS if s is False]
    skip = [n for n, s, _ in RESULTS if s is None]
    passed = len(RESULTS) - len(fail) - len(skip)
    print(f"\n{passed} 통과 · {len(skip)} 건너뜀 · {len(fail)} 실패" + (f" · 실패: {fail}" if fail else ""))
    if fail or (a.require_real and skip):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
