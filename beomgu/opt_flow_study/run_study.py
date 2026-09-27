"""HYFE 옵션 매매 지표 → 코스피200 방향성 조사

사용
  python run_study.py --steps 1-8                 # 1·2단계는 항상, 3~8단계는 옵션 자료가 있을 때
  python run_study.py --steps 1 --raw-dir D:/data/raw --out-dir D:/runs

단계
  1 데이터·기간(달력 패널, 감사)   2 지표 계산(1일·3일 누적, T−1 정렬)
  3 5분위   4 HAC 회귀   5 연도별 안정성   6 통제(전일 수익률 / + 외인 선물)
  7 콜/풋 분해   8 순환 시프트 플라시보   → 사전등록 기준 6개 판정

출력은 --out-dir(기본 ./outputs)/<실행ID>/ 에 쓴다. 이전 실행을 덮어쓰지 않는다.
사전등록 값과 다른 인자(--perm)로 실행하면 파일명·제목에 NON-PREREG 를 붙인다.
"""
import argparse
import datetime as dt
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import optflow
from optflow import report as rp
from optflow import stats as st
from optflow.config import PKG_ROOT, PREREG, PREREG_CHANGES, resolve_out_dir, resolve_raw_dir
from optflow.indicators import build_indicators, describe, lag_to_target
from optflow.io import OPT_INVESTOR_FILE, OPT_REQUIRED, build_panel

INPUT_FILES = ["k200_index_daily.csv", "k200_naver_fchart_daily.csv", "kodex200_daily.csv", "kodex200_naver_fchart_daily.csv",
               "naver_fut_investor_daily.csv", OPT_INVESTOR_FILE, "krx_opt_value_daily.csv"]


def parse_steps(s):
    out = set()
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(part))
    return out


def provenance(raw, args, run_id):
    files = {}
    for name in INPUT_FILES:
        p = raw / name
        if p.exists():
            h = hashlib.sha256(p.read_bytes()).hexdigest()
            try:
                d = pd.read_csv(p, usecols=["date"], parse_dates=["date"])["date"]
                span = [str(d.min().date()), str(d.max().date())]
                rows = int(len(d))
            except Exception:
                span, rows = None, None
            files[name] = {"sha256": h, "rows": rows, "span": span}
    log = raw / "_collected_on.txt"
    last_log = log.read_text(encoding="utf-8").strip().splitlines()[-1] if log.exists() else None
    try:
        rev = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=PKG_ROOT, capture_output=True, text=True, timeout=10).stdout.strip() or None
    except Exception:
        rev = None
    return {"run_id": run_id, "raw_dir": str(raw), "files": files, "last_collection_log": last_log,
            "code_version": optflow.__version__, "git_rev": rev, "python": platform.python_version(),
            "numpy": np.__version__, "pandas": pd.__version__, "args": vars(args), "prereg_version": PREREG["version"]}


def _x(index, **cols):
    return pd.DataFrame(cols, index=index)


def run_core(panel, X, sig, target, pr):
    y, x, L = panel[target], X[sig], pr["hac_lags"]
    res = {"signal": sig, "target": target}
    res["quintile"] = st.quintile_table(y, x)                                                     # 3
    res["reg"] = st.hac_ols(y, _x(panel.index, x=x), lags=L)                                      # 4
    res["yearly"] = st.yearly_table(y, x, lags=L, min_n=pr["year_min_n"])                         # 5
    prev = panel[pr["control_prev_return"]]
    fut = X[pr["control_futflow"]] if pr["control_futflow"] in X else pd.Series(np.nan, index=panel.index)
    res["control_prev"] = st.hac_ols(y, _x(panel.index, x=x, r_prev=prev), lags=L)                # 6
    res["control_prev_fut"] = st.hac_ols(y, _x(panel.index, x=x, r_prev=prev, fut=fut), lags=L)
    res["control_prev_alt"] = st.hac_ols(y, _x(panel.index, x=x, r_prev=panel[pr["control_prev_return_alt"]]), lags=L)   # 탐색 보고
    for key, start in (("subperiod", pr["subperiod_start"]), ("subperiod_0845", "2023-07-31")):
        m = panel.index >= pd.Timestamp(start)
        res[key] = st.hac_ols(y[m], _x(panel.index[m], x=x[m]), lags=L)
    return res


def explore_list(panel, X, sig, target, pr):
    """판정 외 모든 지표(중복 별칭·상수열·통제 제외). 셀 수 보고용."""
    out, seen = [], []
    for c in X.columns:
        if c in (sig, "sig_date") or c.startswith("futflow") or c.startswith("total_"):
            continue
        s = X[c]
        if s.notna().sum() < 100 or s.dropna().nunique() <= 1:
            continue
        if any(s.equals(X[o]) for o in seen):
            continue
        seen.append(c)
        r = st.hac_ols(panel[target], _x(panel.index, x=s), lags=pr["hac_lags"])
        q = st.quintile_table(panel[target], s)
        out.append({"indicator": c, "n": r.get("n"), "beta": r.get("coef", {}).get("x"), "p": r.get("p", {}).get("x"),
                    "r2": r.get("r2"), "q5_q1": q["q5_q1"] if q else None, "singular": r.get("singular", False)})
    return out


def main():
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", default="1-8")
    ap.add_argument("--raw-dir", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--perm", type=int, default=PREREG["placebo_perm"])
    a = ap.parse_args()
    steps = parse_steps(a.steps)
    raw = resolve_raw_dir(a.raw_dir)
    non_prereg = a.perm != PREREG["placebo_perm"]
    run_id = dt.datetime.now().strftime("%Y%m%d-%H%M%S") + ("-NON-PREREG" if non_prereg else "")
    out = resolve_out_dir(a.out_dir) / run_id
    out.mkdir(parents=True, exist_ok=False)
    prov = provenance(raw, a, run_id)
    (out / "provenance.json").write_text(json.dumps(prov, ensure_ascii=False, indent=1), encoding="utf-8")

    # 1. 데이터·기간
    panel, audit = build_panel(raw, PREREG["window_start"])
    if 1 in steps:
        (out / "step1_data_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
        (out / "step1_data_audit.md").write_text(rp.md_audit(audit, panel, raw, prov), encoding="utf-8")
        print(f"[1] 패널 {len(panel)}일 {panel.index.min().date()}~{panel.index.max().date()} · 옵션: {audit['options']['status']}")

    # 2. 지표 (전체 달력에서 계산·시프트한 뒤 판정 창으로 자른다)
    ind = build_indicators(panel)
    X = lag_to_target(ind, panel)
    if 2 in steps:
        desc = describe(ind)
        (out / "step2_indicators.md").write_text(rp.md_describe(desc), encoding="utf-8")
        print(f"[2] 지표 {len(desc)}개: {', '.join(desc['indicator'])}")

    if not steps & set(range(3, 9)):
        print(f"→ {out}")
        return 0
    sig = PREREG["primary_signal"]
    if sig not in X.columns or X[sig].notna().sum() == 0:
        (out / "steps3-8_blocked.md").write_text(rp.md_blocked(raw / OPT_INVESTOR_FILE, OPT_REQUIRED, audit["options"]), encoding="utf-8")
        why = "파일 없음" if audit["options"]["status"] == "missing" else f"파일은 있으나 foreign 콜·풋 입력 없음(투자자 {audit['options'].get('investors_std')})"
        print(f"[3-8] 대기: 주 지표 {sig} — {why} → {out / 'steps3-8_blocked.md'}")
        return 2

    w = (panel.index >= pd.Timestamp(PREREG["window_start"])) & (panel.index <= pd.Timestamp(PREREG["window_end"]))
    P, XW = panel[w], X[w]
    target = PREREG["primary_target"]
    res = run_core(P, XW, sig, target, PREREG)
    res_kdx = run_core(P, XW, sig, "y_kdx", PREREG)
    res["decomp"] = {}                                                                             # 7
    for name, cols in (("call_only", ["callflow_1d"]), ("put_only", ["putflow_1d"]), ("joint", ["callflow_1d", "putflow_1d"])):
        if all(c in XW for c in cols):
            res["decomp"][name] = st.hac_ols(P[target], XW[cols], lags=PREREG["hac_lags"])
    res["placebo"] = st.circular_shift_placebo(P[target], XW[sig], n_perm=a.perm,                  # 8
                                               min_shift=PREREG["placebo_min_shift"], seed=PREREG["placebo_seed"])
    res["placebo_t"] = st.circular_shift_placebo_studentized(P[target], XW[sig], n_perm=2000, min_shift=PREREG["placebo_min_shift"],
                                                             seed=PREREG["placebo_seed"], lags=PREREG["hac_lags"])   # 보고용(판정 아님)
    crit, overall = st.evaluate_prereg(res, PREREG)
    explore = explore_list(P, XW, sig, target, PREREG)
    used = pd.concat([P[target], XW[sig]], axis=1).dropna()
    res["sample"] = {"window": [PREREG["window_start"], PREREG["window_end"]], "n": len(used),
                     "start": str(used.index.min().date()) if len(used) else None, "end": str(used.index.max().date()) if len(used) else None}

    tag = " [NON-PREREG]" if non_prereg else ""
    md = rp.md_results(res, res_kdx, crit, overall, explore, PREREG, prov, tag)
    (out / f"report_{sig}.md").write_text(md, encoding="utf-8")
    js = {"prereg": PREREG, "prereg_changes": PREREG_CHANGES, "non_prereg": non_prereg, "criteria": crit, "overall": overall,
          "explore_count": len(explore), "explore": explore, "result": rp.jsonable(res), "result_kodex": rp.jsonable(res_kdx)}
    (out / f"report_{sig}.json").write_text(json.dumps(js, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(f"[3-8] 완료{tag} · 사전등록 판정: {'통과' if overall else '미통과'} · 탐색 {len(explore)}개 → {out / f'report_{sig}.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
