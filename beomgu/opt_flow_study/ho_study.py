"""옵션 장초 10분 Flow → 09:10~장마감. 입력은 장초 체결을 하루 한 줄로 합친 표다.

입력 ho_daily.csv (하루 한 줄)
  date                       거래일
  c_buy, c_sell              09:00~09:10 콜 체결대금: 매수주도 / 매도주도 (ATM 또는 지정한 행사가 범위)
  p_buy, p_sell              같은 창의 풋 체결대금
  px_0910, px_close          KODEX 200 09:10 가격(09:10 이후 첫 체결 또는 매도1호가), 종가
  px_open (선택)             KODEX 200 시가 → 장초 10분 가격 움직임 통제용
체결 방향은 거래소 체결구분을 쓰고, 없으면 직전 최우선 매도호가 이상 = 매수주도, 매수호가 이하 = 매도주도로 분류한다.

지표(일별 조사와 같은 모양)
  CallFlow10 = (c_buy − c_sell)/(c_buy + c_sell), PutFlow10 같은 식, OptFlow10 = CallFlow10 − PutFlow10
목표: R = px_close / px_0910 − 1 (%)

출력(방향성 통계 + 결합 비교)
  5분위(표본 50일 이상일 때), HAC 회귀, 장초 가격 움직임·외인 선물 Flow(T−1) 통제, 콜/풋 분해,
  HAC t 순환 시프트 플라시보, 규칙 백테스트(OptFlow10>0 → 09:10 매수·종가, 비용 왕복 0.03%+1틱),
  외인 선물 1차 신호(Flow(T−1)>0)일 중 옵션 조건 충족일 대 불충족일 비교

사용: python ho_study.py --input ho_daily.csv [--raw-dir PATH] [--out-dir PATH]
      python ho_study.py --selftest
"""
import argparse
import datetime as dt
import sys

import numpy as np
import pandas as pd

from optflow import stats as st
from optflow.config import resolve_out_dir, resolve_raw_dir
from optflow.indicators import flow

COMM, TICK = 0.00015, 5.0
REQUIRED = ["date", "c_buy", "c_sell", "p_buy", "p_sell", "px_0910", "px_close"]


def build(d, futflow_prev=None):
    d = d.sort_values("date").set_index("date")
    if not d.index.is_unique:
        raise ValueError("날짜 중복")
    out = pd.DataFrame(index=d.index)
    out["callflow10"] = flow(d["c_buy"], d["c_sell"]).to_numpy()
    out["putflow10"] = flow(d["p_buy"], d["p_sell"]).to_numpy()
    out["optflow10"] = out["callflow10"] - out["putflow10"]
    out["y"] = (d["px_close"] / d["px_0910"] - 1) * 100
    out["cost"] = 2 * COMM + TICK / d["px_0910"]
    if "px_open" in d:
        out["early_ret"] = (d["px_0910"] / d["px_open"] - 1) * 100            # 09:00~09:10 가격 움직임(같은 날, 신호 이전 정보)
    if futflow_prev is not None:
        out["futflow_prev"] = futflow_prev.reindex(out.index)
    return out


def futflow_prev_series(raw):
    f = pd.read_csv(raw / "naver_fut_investor_daily.csv", parse_dates=["date"]).set_index("date").sort_index()
    ok = (f["sellq_foreign"] > 0) & (f["buyq_foreign"] > 0)
    fl = ((f["buyq_foreign"] - f["sellq_foreign"]) / (f["buyq_foreign"] + f["sellq_foreign"])).where(ok)
    return fl.shift(1)                                                        # 선물 투자자 파일은 거래일 달력 → 한 칸 = 직전 거래일


def rule(y_pct, cost, sig):
    net = pd.Series(np.where(sig, y_pct / 100 - cost, 0.0), index=y_pct.index)
    tr = net[sig]
    sd = net.std()
    return {"trades": int(sig.sum()), "days": len(net), "win_pct": 100 * (tr > 0).mean() if len(tr) else np.nan,
            "avg_net_bp": 1e4 * tr.mean() if len(tr) else np.nan, "cum_net_pct": 100 * ((1 + net).prod() - 1),
            "sharpe": net.mean() / sd * np.sqrt(252) if sd > 0 else np.nan}


def analyze(t):
    L = [f"# 옵션 장초 10분 Flow → 09:10~장마감 (표본 {len(t)}일, {t.index.min().date()} ~ {t.index.max().date()})", ""]
    y, x = t["y"], t["optflow10"]
    q = st.quintile_table(y, x, min_per_bin=10)
    if q:
        L += ["## 5분위 (다음 09:10→종가 평균, %)", "| 분위 | n | 평균 |", "|---|---|---|"]
        L += [f"| Q{k} | {int(r['n'])} | {r['mean']:+.3f} |" for k, r in q["table"].iterrows()] + [f"\nQ5−Q1 = {q['q5_q1']:+.3f}%p", ""]
    else:
        L += ["5분위: 표본 50일 미만이라 생략", ""]

    def reg_line(name, r, key="x"):
        if "coef" not in r:
            return f"- {name}: 표본 부족(n={r.get('n')})"
        return f"- {name}: β {r['coef'][key]:+.3f}, HAC p {r['p'][key]:.3f}, R² {r['r2']:.4f}, n {r['n']}"
    L += ["## 회귀 R = α + β·OptFlow10 (+ 통제)", reg_line("단독", st.hac_ols(y, x.to_frame("x"), min_n=20))]
    ctrl = {"x": x}
    if "early_ret" in t:
        ctrl["early"] = t["early_ret"]
        L.append(reg_line("장초 가격 움직임 통제", st.hac_ols(y, pd.DataFrame(ctrl), min_n=20)))
    if "futflow_prev" in t:
        ctrl["fut"] = t["futflow_prev"]
        L.append(reg_line("+ 외인 선물 Flow(T−1) 통제", st.hac_ols(y, pd.DataFrame(ctrl), min_n=20)))
    for nm, col in (("콜 단독", "callflow10"), ("풋 단독", "putflow10")):
        L.append(reg_line(nm, st.hac_ols(y, t[[col]].rename(columns={col: "x"}), min_n=20)))
    pl = st.circular_shift_placebo_studentized(y, x, n_perm=2000, min_shift=max(5, len(t) // 20))
    L += ["", f"## 플라시보 (HAC t 순환 시프트): " + (f"t {pl['t_obs']:+.2f}, 단측 p {pl['p_one_sided']:.3f}" if "t_obs" in pl else f"표본 부족(n={pl['n']})"), ""]
    L += ["## 규칙 백테스트 (KODEX 200 09:10 매수 → 종가, 비용 왕복 0.03% + 1틱)", "| 규칙 | 매매일 | 승률 | 평균 순수익/회 | 누적 | 샤프 |", "|---|---|---|---|---|---|"]
    rules = {"매일 09:10 매수": pd.Series(True, index=t.index), "OptFlow10 > 0": x > 0, "PutFlow10 < 0": t["putflow10"] < 0}
    if "futflow_prev" in t:
        a1 = t["futflow_prev"] > 0
        rules.update({"외인 선물 Flow(T−1)>0 (1차 신호)": a1, "1차 신호 ∧ OptFlow10>0": a1 & (x > 0), "1차 신호 ∧ OptFlow10≤0": a1 & (x <= 0)})
    for nm, s in rules.items():
        r = rule(y, t["cost"], s.fillna(False).astype(bool))
        L.append(f"| {nm} | {r['trades']}/{r['days']} | {r['win_pct']:.1f}% | {r['avg_net_bp']:+.1f}bp | {r['cum_net_pct']:+.1f}% | {r['sharpe']:+.2f} |")
    n = len(t)
    L += ["", f"- 표본 {n}일: 1년 미만 표본의 샤프 표준오차는 약 √(252/n) ≈ {np.sqrt(252 / max(n, 1)):.1f} 이다. 판정이 아니라 추정치로 읽는다",
          "- '1차 신호 ∧ 옵션 충족일' 대 '불충족일' 차이가 옵션 조건의 증분이다"]
    return "\n".join(L)


def synthetic(n=180, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2026-01-02", periods=n)
    c_b, c_s, p_b, p_s = (rng.uniform(1e9, 3e9, n) for _ in range(4))
    of = (c_b - c_s) / (c_b + c_s) - (p_b - p_s) / (p_b + p_s)
    px0 = 100000 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    ret = 4.0 * of / 100 + rng.normal(0, 0.012, n)                         # 심은 효과(1표준편차당 약 +0.9%p)
    return pd.DataFrame({"date": idx, "c_buy": c_b, "c_sell": c_s, "p_buy": p_b, "p_sell": p_s,
                         "px_open": px0 * (1 + rng.normal(0, 0.003, n)), "px_0910": px0, "px_close": px0 * (1 + ret)})


def main():
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=None)
    ap.add_argument("--raw-dir", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        syn = synthetic()
        fut = pd.Series(np.random.default_rng(1).normal(0, 0.02, len(syn)), index=pd.DatetimeIndex(syn["date"]))
        t = build(syn, fut)
        md = analyze(t)
        print(md)
        r = st.hac_ols(t["y"], t[["optflow10"]].rename(columns={"optflow10": "x"}), min_n=20)
        null = build(syn.assign(px_close=syn["px_0910"] * (1 + np.random.default_rng(2).normal(0, 0.012, len(syn)))), fut)
        r0 = st.hac_ols(null["y"], null[["optflow10"]].rename(columns={"optflow10": "x"}), min_n=20)
        ok = r["coef"]["x"] > 0 and r["p"]["x"] < 0.01 and r0["p"]["x"] > 0.01 and "1차 신호 ∧ OptFlow10>0" in md
        print(f"\nSELFTEST {'PASS' if ok else 'FAIL'}: 심은 효과 β {r['coef']['x']:+.2f} p {r['p']['x']:.1e} / "
              f"효과 없음 p {r0['p']['x']:.2f} / 결합 비교 경로 {'있음' if '1차 신호 ∧' in md else '없음'}")
        return 0 if ok else 1
    d = pd.read_csv(a.input, parse_dates=["date"])
    miss = [c for c in REQUIRED if c not in d.columns]
    if miss:
        sys.exit(f"입력에 필수 열 없음: {miss}")
    fut = None
    try:
        fut = futflow_prev_series(resolve_raw_dir(a.raw_dir))
    except Exception as e:
        print("외인 선물 Flow 통제 없이 진행:", e)
    md = analyze(build(d, fut))
    out = resolve_out_dir(a.out_dir) / (dt.datetime.now().strftime("%Y%m%d-%H%M%S") + "-ho")
    out.mkdir(parents=True, exist_ok=False)
    (out / "ho_report.md").write_text(md, encoding="utf-8")
    print(md, f"\n→ {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
