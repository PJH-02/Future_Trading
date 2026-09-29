"""T01 · K1 장중 강도 → 그날 밤 오버나잇 (+N1 자산 교체, N4 미국 지수 ETF)

사전 고정(실행 전 기록, 2026-09-29):
- r_id(T) = C(T)/O(T) − 1 (수정주가, 같은 날 비율)
- q_hi, q_lo = 직전 60거래일 r_id 의 70·30 분위(T 제외, 선형보간), 번인 60일
- H-mom: r_id(T) > q70 → T 종가(시간외 종가 체결 가정) 매수, T+1 시가 매도
- H-rev: r_id(T) < q30 → 같은 방식
- 이웃(보고만): 창 40·120, 분위 80·60(mom) / 20·40(rev)
- 판정 자산 069500, 판정 구간 2010-12-16 ~ 2025-06-05, 비용 5bp(판정은 7bp에서도 확인)
- 이전: 229200(2016-08 ~ 2025-06-05) Δ>0 부호만. 102110·091160·122630·233740 재현·배율, 360750·133690 2018~ 보고만
- 2010-12 이전(2003~2010)은 추가 보고(판정 밖)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common as C  # noqa: E402
import data  # noqa: E402
import engine as E  # noqa: E402

TOPIC = "T01_K1_intraday_strength"
OUT = data.OUT / TOPIC
JUDGE_START = "2010-12-16"
MAIN, TRANSFER = "069500", "229200"
REPLICA = ["102110", "091160", "122630", "233740"]
US = ["360750", "133690"]
SPEC = {"r_id": "C/O-1", "window": 60, "q_mom": 0.70, "q_rev": 0.30, "exit": "T+1 open", "entry": "T close(after-hours close)",
        "judge": [JUDGE_START, E.JUDGE_END], "cost_bp": [5, 7]}


def signals(px, win=60, q_mom=0.70, q_rev=0.30):
    r_id = px["close"] / px["open"] - 1
    hi = r_id.shift(1).rolling(win, min_periods=win).quantile(q_mom, interpolation="linear")
    lo = r_id.shift(1).rolling(win, min_periods=win).quantile(q_rev, interpolation="linear")
    ok = hi.notna()
    return {"H-mom": ((r_id > hi) & ok).astype(float).where(ok), "H-rev": ((r_id < lo) & ok).astype(float).where(ok)}


def vol_adj(ret, win=60):
    """과거 60일 오버나잇 SD 로 나눈 수익률(T 시점에 알려진 r_on(T−1)까지 사용)."""
    sd = ret.shift(1).rolling(win, min_periods=win).std()
    return ret / sd


def quality_flag(px_etf):
    """ETF 갭 = 0 이면서 |K200 지수 갭| > 20bp 인 밤(시가 품질 의심)."""
    k = pd.read_csv(data.RAW / "k200_index_daily.csv", parse_dates=["date"], index_col="date")
    kg = k["open"].shift(-1) / k["close"] - 1
    eg = px_etf["open"].shift(-1) / px_etf["close"] - 1
    return ((eg == 0) & (kg.reindex(eg.index).abs() > 0.002))


def run_asset(code, start=None, end=None):
    px = data.price(code)
    L = E.legs(px)
    ret = L["on"]
    sig = signals(px)
    if start:
        ret = ret.loc[start:]
    if end:
        ret = ret.loc[:end]
    sig = {k: v.reindex(ret.index) for k, v in sig.items()}
    rows = []
    for c in E.COSTS_BP:
        t = E.evaluate(sig, ret, cost_bp=c)
        t.insert(0, "asset", code)
        rows.append(t)
    return px, ret, sig, pd.concat(rows, ignore_index=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    md = [f"# T01 · K1 장중 강도 → 그날 밤 오버나잇\n", "```\n" + __doc__.strip() + "\n```\n"]
    all_tables = []

    # ---- 판정 자산 ----
    px, ret, sig, tab = run_asset(MAIN, start=JUDGE_START)
    all_tables.append(tab)
    J = ret.loc[:E.JUDGE_END]
    verdict = {}
    for name in ("H-mom", "H-rev"):
        s = sig[name].loc[J.index]
        d, t, n = E.delta_test(s, J)
        from math import erf, sqrt
        p1 = 0.5 * (1 - erf(t / sqrt(2)))                       # 단측(Δ>0)
        net7 = (J[s == 1] - 7e-4).mean() * 1e4
        dA = E.delta_test(s.loc[:E.A_END], J.loc[:E.A_END])[0]
        dB = E.delta_test(s.loc["2020-01-01":], J.loc["2020-01-01":])[0]
        dv = E.delta_test(s, vol_adj(E.legs(px)["on"]).reindex(J.index))[0]
        plc = E.placebo_year_strat(s, J)
        # 이웃
        neigh = {}
        for lab, kw in (("창40", {"win": 40}), ("창120", {"win": 120}),
                        ("분위80/20", {"q_mom": 0.80, "q_rev": 0.20}), ("분위60/40", {"q_mom": 0.60, "q_rev": 0.40})):
            sn = signals(px, **kw)[name].reindex(J.index)
            neigh[lab] = E.delta_test(sn, J)[0] * 1e4
        # 품질 플래그 제외판
        qf = quality_flag(px).reindex(J.index).fillna(False)
        d_q = E.delta_test(s[~qf], J[~qf])[0]
        verdict[name] = dict(delta_bp=d * 1e4, t=t, p_one=p1, p_bonf5=min(1, p1 * 5), n_sig=n, net7_bp=net7,
                             dA_bp=dA * 1e4, dB_bp=dB * 1e4, dvol=dv, placebo_p=plc, neighbors=neigh,
                             delta_ex_quality_bp=d_q * 1e4, n_quality_flag=int(qf.sum()))

    # ---- 이전 판정 ----
    _, ret_t, sig_t, tab_t = run_asset(TRANSFER, start="2016-08-01")
    all_tables.append(tab_t)
    Jt = ret_t.loc[:E.JUDGE_END]
    for name in ("H-mom", "H-rev"):
        verdict[name]["transfer_229200_delta_bp"] = E.delta_test(sig_t[name].loc[Jt.index], Jt)[0] * 1e4

    # ---- 판정 요약 ----
    md.append("## 판정 (KODEX 200, 2010-12-16 ~ 2025-06-05, 종가 → 익일 시가)\n")
    md.append("| 항목 | H-mom (강한 날) | H-rev (약한 날) | 기준 |\n|---|---|---|---|")
    keys = [("n_sig", "신호 밤 수", "{:.0f}", ""), ("delta_bp", "Δ(bp) 신호−비신호", "{:+.2f}", "> 0"),
            ("t", "HAC t", "{:+.2f}", ""), ("p_one", "단측 p", "{:.4f}", ""), ("p_bonf5", "Bonferroni×5 p(Holm 상한)", "{:.4f}", "< 0.05"),
            ("net7_bp", "신호 밤 순평균 7bp(bp)", "{:+.2f}", "> 0"), ("dA_bp", "Δ A(~2019)", "{:+.2f}", "> 0"),
            ("dB_bp", "Δ B(2020~2025-06)", "{:+.2f}", "> 0, ≥ 0.5·A"), ("dvol", "변동성 조정 Δ(SD 단위)", "{:+.4f}", "> 0"),
            ("placebo_p", "연도 층화 플라시보 p", "{:.4f}", "≤ 0.05"), ("transfer_229200_delta_bp", "이전 229200 Δ(bp)", "{:+.2f}", "> 0"),
            ("delta_ex_quality_bp", "품질 플래그 제외 Δ(bp)", "{:+.2f}", "보고")]
    for k, lab, f, crit in keys:
        md.append(f"| {lab} | {f.format(verdict['H-mom'][k])} | {f.format(verdict['H-rev'][k])} | {crit} |")
    md.append(f"| 이웃 Δ(bp) | {', '.join(f'{a} {b:+.1f}' for a, b in verdict['H-mom']['neighbors'].items())} | "
              f"{', '.join(f'{a} {b:+.1f}' for a, b in verdict['H-rev']['neighbors'].items())} | 부호 동일 |")
    md.append(f"\n품질 플래그 밤 수(판정 구간): {verdict['H-mom']['n_quality_flag']}\n")
    for name, v in verdict.items():
        ok = [v["delta_bp"] > 0 and v["p_bonf5"] < 0.05, v["net7_bp"] > 0, v["dA_bp"] > 0 and v["dB_bp"] > 0 and v["dB_bp"] >= 0.5 * v["dA_bp"],
              v["placebo_p"] <= 0.05, v["dvol"] > 0, all(np.sign(x) == np.sign(v["delta_bp"]) for x in v["neighbors"].values()),
              v["transfer_229200_delta_bp"] > 0]
        grade = "통과" if all(ok) else ("보류" if ok[0] and ok[1] and ok[3] else "기각")
        v["criteria"], v["grade"] = ok, grade
        md.append(f"- **{name}: {grade}** (①~⑦ 충족 여부: {''.join('O' if x else 'X' for x in ok)}, ①은 Bonferroni×5 상한으로 판정)")

    # ---- 기간별 표 ----
    md.append("\n## 기간별 표 (KODEX 200, 비용 5bp)\n")
    md.append(E.fmt_table(tab[(tab.cost_bp == 5)]))
    md.append("\n## 연도별 (KODEX 200, 5bp)\n")
    yr = pd.concat({n: E.yearly(sig[n], ret) for n in ("H-mom", "H-rev")}, axis=1)
    yb = E.yearly(pd.Series(1.0, index=ret.index), ret)
    ytab = pd.DataFrame({"매일보유_net": yb["net_bp"], "mom_n": yr[("H-mom", "trades")], "mom_net": yr[("H-mom", "net_bp")],
                         "rev_n": yr[("H-rev", "trades")], "rev_net": yr[("H-rev", "net_bp")]})
    md.append(ytab.round(2).to_markdown())

    # ---- 자산 교체·재현·미국 ETF (보고) ----
    md.append("\n## 자산 교체·재현 (보고, 5bp, 종가→익일 시가)\n")
    rep_rows = []
    for code, start in [(TRANSFER, "2016-08-01")] + [(c, "2010-12-16") for c in REPLICA] + [(c, "2018-01-01") for c in US]:
        try:
            _, r2, s2, t2 = run_asset(code, start=start)
        except Exception as e:  # noqa: BLE001
            md.append(f"- {code}: 실패 {e}")
            continue
        if code != TRANSFER:                      # 229200 은 위 이전 판정에서 이미 장부에 넣음
            all_tables.append(t2)
        for name in ("H-mom", "H-rev"):
            for wname, (a, b) in {"판정끝까지": (start, E.JUDGE_END), "E4": (E.E4_START, None), "전체": (start, None)}.items():
                rr = r2.loc[a:b]
                ss = s2[name].reindex(rr.index)
                dlt, tt, n = E.delta_test(ss, rr)
                st = E.stats(ss, rr, 5)
                rep_rows.append(dict(asset=f"{code} {data.NAMES.get(code, '')}", rule=name, window=wname, n=n,
                                     delta_bp=dlt * 1e4, t=tt, net_bp=st["net_bp"], sharpe=st["sharpe"]))
    rep = pd.DataFrame(rep_rows)
    md.append(rep.round(2).to_markdown(index=False))

    # ---- 2003~2010 추가 보고 ----
    px0, ret0, sig0, _ = run_asset(MAIN, start="2003-01-01", end="2010-12-15")
    md.append("\n## 판정 이전 구간 2003-01 ~ 2010-12-15 (추가 보고, 판정 밖)\n")
    for name in ("H-mom", "H-rev"):
        dlt, tt, n = E.delta_test(sig0[name].reindex(ret0.index), ret0)
        md.append(f"- {name}: 신호 {n}밤, Δ {dlt * 1e4:+.2f}bp, HAC t {tt:+.2f}")

    (OUT / "report.md").write_text("\n".join(md), encoding="utf-8")
    E.ledger_append(data.OUT, TOPIC, SPEC, pd.concat(all_tables, ignore_index=True))
    C.save_verdict(OUT / "verdict.json", verdict)
    print("\n".join(md[:40]))


if __name__ == "__main__":
    main()
