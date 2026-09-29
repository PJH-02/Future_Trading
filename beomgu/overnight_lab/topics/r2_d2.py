"""R2 · D2 개장 반영 오류(미국 마감으로 설명 안 되는 코스피200 시가 갭 잔차 → T 시가→종가) 2차 확인

고정 사양: <OUT>/round2/D2/spec.json (탐색 담당이 A 구간 2010-01-04~2019-12-30 만 보고 고정. 여기서는 바꾸지 않는다)
- 한국 달력 = 069500 일봉 행, prev(T) = 직전 행
- 지수 = 코스피200 일봉. 2014-07-07 이전 FDR KS200(k200_index_daily.csv), 2014-07-08 이후 fchart KPI200(etf/KPI200.csv).
  fchart 에 없는 행만 FDR 로 채운다. (B 구간 두 원천 시가·종가 완전 일치. FDR 마지막 행 2026-09-17 종가만 fchart 와 달라
  장중 스냅샷으로 보고 E4 에서 fchart 를 우선했다.)
- g(T) = ln(지수 시가(T) / 지수 종가(prev(T)))
- D(X) = 미국 날짜 < X 인 마지막 ^GSPC 세션, us(T) = ln SPX종가(D(T)) − ln SPX종가(D(prev(T))), 같은 세션이면 0.
  종가가 비어 있는 세션(파일 마지막 행 2026-09-22)이나 파일 끝 뒤에 평일이 끼면 NaN.
- T 제외 직전 250행 중 g·us 유효 행 ≥ 200 이면 OLS g = a + b·us, s = sqrt(SSR/(n−2)), z(T) = (g(T) − a − b·us(T)) / s.
  모자라면 NaN(표본 제외). 2변수 이웃은 s = sqrt(SSR/(n−3)).
- D2-C1: z ≥ +1.0 → 069500 T 시가 매수·T 종가 매도(r_oc). D2-C2: z ≤ −1.0 → 114800 KODEX 인버스, 같은 방식.
  114800 은 data.price('114800') (원자료 etf/ 캐시, 없으면 siseJson 으로 받아 캐시).
- 비용 왕복 5bp(스트레스 7bp). 신호·체결 모두 09:00 시가라고 가정(실거래는 08:59 예상지수로 근사해야 함).
- 이웃(보고, 판정 ⑤는 부호만): 창 120(최소 96) / 창 500(최소 400), 임계 0.75 / 1.25, 회귀변수 SPX+^IXIC 누적.
- 이전 자산(보고만): C1 → 102110 TIGER 200·122630 KODEX 레버리지, C2 → 123310 TIGER 인버스.
확인 판정(B = 2020-01-01~2025-06-05, 1회):
 ① Δ(신호−비신호) > 0, NW(lag5) 단측 p 원값 보고(Holm 보정은 2차 종합에서)
 ② 7bp 차감 후 신호 거래 순평균 > 0   ③ 연도 층화 플라시보 p ≤ 0.05
 ④ 변동성 조정 Δ > 0 (r_oc / 과거 60일 SD)   ⑤ 이웃 Δ 부호가 본 셀과 같음
A(표본 내)·E4(2025-06-09~, 보고만) 값을 병기한다.
"""
import datetime as dt
import hashlib
import json
import sys
from math import erf, sqrt
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common as C  # noqa: E402
import data  # noqa: E402
import engine as E  # noqa: E402

TOPIC = "R2_D2"
OUT = data.OUT / "round2" / "D2"
SPEC_FILE = data.spec_file("D2")   # 출력 폴더에 없으면 저장소 결과 폴더의 사양을 읽음
WIN = {"A": ("2010-01-01", E.A_END), "B": ("2020-01-01", E.JUDGE_END), "E4": (E.E4_START, None)}
SUB = {"B 전반(2020-01~2023-07-28)": ("2020-01-01", "2023-07-28"),
       "B 후반(2023-07-31~2025-06-05, 파생 08:45 개장)": ("2023-07-31", E.JUDGE_END),
       "B 끝(2025-03-04~2025-06-05, NXT 프리마켓)": ("2025-03-04", E.JUDGE_END)}
N_CELLS_R2 = 8          # 2차 전체 셀 수 상한(Holm 은 종합에서)


def p_one(t):
    return np.nan if t is None or np.isnan(t) else 0.5 * (1 - erf(t / sqrt(2)))


# ---------------------------------------------------------------- 자료
def load_index():
    fdr = pd.read_csv(data.RAW / "k200_index_daily.csv", parse_dates=["date"], index_col="date")[["open", "close"]]
    fch = pd.read_csv(data.PX_DIR / "KPI200.csv", parse_dates=["date"], index_col="date")[["open", "close"]]
    late = fch.loc["2014-07-08":].combine_first(fdr.loc["2014-07-08":])
    ix = pd.concat([fdr.loc[:"2014-07-07"], late]).sort_index()
    return ix[(ix["open"] > 0) & (ix["close"] > 0)]


def us_log_close(kr_idx, name):
    """각 한국일 T 에 대해 ln 종가(D(T)), D(T) = 미국 날짜 < T 인 마지막 세션. 종가를 모르면 NaN."""
    s = data.us_daily(name)["close"]
    pos = np.searchsorted(s.index.values, kr_idx.values, side="left") - 1
    v = np.log(s.to_numpy(dtype=float))
    last = np.where(pos >= 0, v[np.clip(pos, 0, None)], np.nan)
    L = s.index.max()
    beyond = np.array([len(pd.bdate_range(L + pd.Timedelta(days=1), t - pd.Timedelta(days=1))) > 0 for t in kr_idx])
    last[beyond] = np.nan                  # 파일 끝 뒤 평일(미국 세션 가능)이 있으면 종가 미확인
    return pd.Series(last, index=kr_idx)


def us_cum(kr_idx, name):
    last = us_log_close(kr_idx, name)
    return last - last.shift(1)            # 같은 세션이면 0, 여러 세션이면 누적


def rolling_z(g, X, win=250, min_rows=None):
    """T 제외 직전 win 행에서 g·X 가 모두 있는 행 ≥ min_rows 이면 OLS 후 z(T) = 잔차(T) / sqrt(SSR/(n−p))."""
    min_rows = int(round(win * 0.8)) if min_rows is None else min_rows
    G = g.to_numpy(float)
    Xv = X.to_numpy(float).reshape(len(G), -1)
    ok = ~np.isnan(G) & ~np.isnan(Xv).any(axis=1)
    z = np.full(len(G), np.nan)
    for i in range(len(G)):
        if not ok[i]:
            continue
        lo = max(0, i - win)
        m = ok[lo:i]
        n = int(m.sum())
        if n < min_rows:
            continue
        A = np.column_stack([np.ones(n), Xv[lo:i][m]])
        y = G[lo:i][m]
        b, *_ = np.linalg.lstsq(A, y, rcond=None)
        e = y - A @ b
        s = np.sqrt(e @ e / (n - A.shape[1]))
        z[i] = (G[i] - np.r_[1.0, Xv[i]] @ b) / s
    return pd.Series(z, index=g.index)


def fingerprint(df, a, b):
    x = df.loc[a:b, ["open", "close"]].round(4).to_csv().encode()
    return hashlib.sha256(x).hexdigest()[:16]


# ---------------------------------------------------------------- 통계
def window_stats(sig, ret, volr, a, b, placebo=True):
    d = pd.DataFrame({"s": sig, "r": ret, "v": volr}).loc[a:b]
    d = d[d["s"].notna() & d["r"].notna()]
    s, r = d["s"], d["r"]
    held = r[s == 1]
    dl, t, n = E.delta_test(s, r)
    dv = E.delta_test(s[d["v"].notna()], d["v"].dropna())[0]
    return dict(days=len(d), n_sig=n, gross_bp=held.mean() * 1e4, t_gross=E.nw_t(held), median_bp=held.median() * 1e4,
                win=(held > 0).mean(), net5_bp=(held.mean() - 5e-4) * 1e4, net7_bp=(held.mean() - 7e-4) * 1e4,
                base_bp=r.mean() * 1e4, nonsig_bp=r[s == 0].mean() * 1e4, delta_bp=dl * 1e4, t=t, p_one=p_one(t), dvol=dv,
                placebo_p=E.placebo_year_strat(s, r) if placebo else np.nan,
                first=str(d.index.min().date()) if len(d) else None, last=str(d.index.max().date()) if len(d) else None)


def delta_only(sig, ret, a, b):
    d = pd.DataFrame({"s": sig, "r": ret}).loc[a:b].dropna()
    dl, t, n = E.delta_test(d["s"], d["r"])
    held = d["r"][d["s"] == 1]
    return dict(n_sig=n, gross_bp=held.mean() * 1e4, delta_bp=dl * 1e4, t=t)


def decomposition(sig, etf_oc, idx_oc_signed, dev, a, b):
    """신호일 ETF r_oc = 지수 몫(부호 맞춘 지수 시가→종가) + 단일가 몫(ETF − 지수). 시가 괴리 = ETF 갭 − 부호 맞춘 지수 갭."""
    d = pd.DataFrame({"s": sig, "etf": etf_oc, "idx": idx_oc_signed, "dev": dev}).loc[a:b]
    d = d[d["s"].notna() & d["etf"].notna() & d["idx"].notna()]
    h = d[d["s"] == 1]
    di, ti, _ = E.delta_test(d["s"], d["idx"])
    de, te, _ = E.delta_test(d["s"], d["etf"] - d["idx"])
    return dict(n_sig=len(h), etf_bp=h["etf"].mean() * 1e4, idx_bp=h["idx"].mean() * 1e4,
                single_bp=(h["etf"] - h["idx"]).mean() * 1e4, idx_delta_bp=di * 1e4, idx_t=ti, single_delta_bp=de * 1e4,
                single_t=te, dev_sig_bp=h["dev"].mean() * 1e4, dev_all_bp=d["dev"].mean() * 1e4)


def strat_rows(sig, ret, label):
    rows = []
    for w, (a, b) in WIN.items():
        for cost in E.COSTS_BP:
            rr = ret.loc[a:b]
            ss = sig.reindex(rr.index)
            keep = ss.notna() & rr.notna()
            for name, pos in (("매일 시가→종가(기준선)", pd.Series(1.0, index=rr.index)[keep]), (label, ss[keep])):
                st = E.stats(pos, rr[keep], cost)
                st.update(rule=name, window=w, cost_bp=cost)
                rows.append(st)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 보고 도우미
F = C._f


def md_table(head, rows):
    return "\n".join(["| " + " | ".join(head) + " |", "|" + "---|" * len(head)] + ["| " + " | ".join(map(str, r)) + " |" for r in rows])


def pfmt(p):
    if p is None or np.isnan(p):
        return "-"
    if p == 0:
        return "< 0.0005(2000회 중 0)"
    return f"{p:.2e}" if p < 1e-3 else f"{p:.4f}"


def ledger_once(spec, table, note):
    """같은 주제·같은 사양 문자열이 장부에 이미 있으면 다시 쓰지 않는다(같은 시행의 재실행이 시행 수를 부풀리지 않게)."""
    p = data.OUT / "ledger.csv"
    key = json.dumps(spec, ensure_ascii=False)
    if p.exists():
        led = pd.read_csv(p, usecols=["run_at", "topic", "spec"], encoding="utf-8-sig", dtype=str)
        hit = led[(led["topic"] == TOPIC) & (led["spec"] == key)]
        if len(hit):
            return False, hit["run_at"].min()
    E.ledger_append(data.OUT, TOPIC, spec, table, note=note)
    return True, None


def main():
    t_run = dt.datetime.now(E.KST).strftime("%Y-%m-%d %H:%M KST")
    OUT.mkdir(parents=True, exist_ok=True)              # 저장소 새 클론에서도 출력 폴더가 있도록
    spec = json.loads(SPEC_FILE.read_text(encoding="utf-8"))

    # 자료
    px = data.price("069500")                           # 캐시 사용(기존 파일 수정 없음)
    cal = px.index
    ix = load_index().reindex(cal)
    inv = data.price("114800").reindex(cal)             # 원자료 etf/ 캐시(저장소 재현용, 2026-09-29 캐시 전환)
    g = np.log(ix["open"] / ix["close"].shift(1))
    idx_oc = ix["close"] / ix["open"] - 1
    us_s = us_cum(cal, "spx")
    us_n = us_cum(cal, "ndx_comp")

    z = {"base": rolling_z(g, us_s.to_frame(), 250, 200),
         "창120": rolling_z(g, us_s.to_frame(), 120, 96),
         "창500": rolling_z(g, us_s.to_frame(), 500, 400),
         "SPX+IXIC": rolling_z(g, pd.concat([us_s, us_n], axis=1), 250, 200)}

    def S(zz, cond):
        return cond(zz).astype(float).where(zz.notna())

    cells = {
        "D2-C1": dict(asset="069500", name="KODEX 200", px=px, up=True,
                      transfer={"102110 TIGER 200": data.price("102110"), "122630 KODEX 레버리지": data.price("122630")}),
        "D2-C2": dict(asset="114800", name="KODEX 인버스", px=inv, up=False,
                      transfer={"123310 TIGER 인버스": data.price("123310")}),
    }
    for cid, c in cells.items():
        up = c["up"]
        thr = (lambda k: (lambda x: x >= k)) if up else (lambda k: (lambda x: x <= -k))
        c["sig"] = S(z["base"], thr(1.0))
        c["neighbors"] = {"창120": S(z["창120"], thr(1.0)), "창500": S(z["창500"], thr(1.0)),
                          f"임계 {'+' if up else '−'}0.75": S(z["base"], thr(0.75)),
                          f"임계 {'+' if up else '−'}1.25": S(z["base"], thr(1.25)), "SPX+IXIC": S(z["SPX+IXIC"], thr(1.0))}
        p = c["px"].reindex(cal)
        c["ret"] = p["close"] / p["open"] - 1
        c["volr"] = C.vol_adj(c["ret"].dropna(), 60).reindex(cal)
        sgn = 1 if up else -1
        c["idx_signed"] = sgn * idx_oc
        c["dev"] = np.log(p["open"] / p["close"].shift(1)) - sgn * g
        c["transfer_ret"] = {k: (lambda q: q["close"] / q["open"] - 1)(v.reindex(cal)) for k, v in c["transfer"].items()}

    # 계산
    res = {}
    for cid, c in cells.items():
        r = {"win": {}, "neighbors": {}, "transfer": {}, "decomp": {}, "sub": {}, "yearly": []}
        for w, (a, b) in WIN.items():
            r["win"][w] = window_stats(c["sig"], c["ret"], c["volr"], a, b)
            r["neighbors"][w] = {k: delta_only(v, c["ret"], a, b) for k, v in c["neighbors"].items()}
            r["transfer"][w] = {k: delta_only(c["sig"], v, a, b) for k, v in c["transfer_ret"].items()}
            r["decomp"][w] = decomposition(c["sig"], c["ret"], c["idx_signed"], c["dev"], a, b)
        for k, (a, b) in SUB.items():
            r["sub"][k] = dict(**window_stats(c["sig"], c["ret"], c["volr"], a, b, placebo=False),
                               **{f"dc_{kk}": vv for kk, vv in decomposition(c["sig"], c["ret"], c["idx_signed"], c["dev"], a, b).items()})
        d = pd.DataFrame({"s": c["sig"], "r": c["ret"]}).loc["2010-01-01":].dropna()
        for y, gy in d.groupby(d.index.year):
            h = gy["r"][gy["s"] == 1]
            dl = E.delta_test(gy["s"], gy["r"])[0] if len(h) >= 5 else np.nan
            r["yearly"].append(dict(year=int(y), n=len(h), gross_bp=h.mean() * 1e4, net5_bp=(h.mean() - 5e-4) * 1e4,
                                    delta_bp=dl * 1e4 if not np.isnan(dl) else np.nan))
        B = r["win"]["B"]
        nb = r["neighbors"]["B"]
        same = all(np.sign(v["delta_bp"]) == np.sign(B["delta_bp"]) for v in nb.values())
        crit = [B["delta_bp"] > 0, B["net7_bp"] > 0, (not np.isnan(B["placebo_p"])) and B["placebo_p"] <= 0.05, B["dvol"] > 0, same]
        r["criteria"] = "".join("O" if x else "X" for x in crit)
        r["neighbors_same_sign"] = bool(same)
        r["grade"] = "통과 후보(Holm 보정은 종합에서)" if all(crit) else "기각"
        r["strat"] = strat_rows(c["sig"], c["ret"], f"{cid} 신호일만")
        tick = 5 if c["px"]["close"].loc["2020":E.JUDGE_END].median() >= 2000 else 1
        r["px_med"] = {w: float(c["px"]["close"].loc[a:b].median()) for w, (a, b) in WIN.items()}
        r["tick_bp"] = {w: 1e4 * (5 if v >= 2000 else 1) / v for w, v in r["px_med"].items()}
        del tick
        res[cid] = r

    # 시행 장부(모든 규칙·자산·기간)
    tabs = []
    for cid, c in cells.items():
        rj = c["ret"].loc["2010-01-01":]
        rules = {cid: c["sig"], **{f"{cid} 이웃 {k}": v for k, v in c["neighbors"].items()}}
        for cost in E.COSTS_BP:
            t = E.evaluate(rules, rj, cost_bp=cost, baseline=f"{c['asset']} 매일 시가→종가")
            t["asset"] = c["asset"]
            tabs.append(t)
            for k, v in c["transfer_ret"].items():
                tt = E.evaluate({f"{cid} 이전 {k}": c["sig"]}, v.loc["2010-01-01":], cost_bp=cost, baseline=f"{k.split()[0]} 매일 시가→종가")
                tt["asset"] = k.split()[0]
                tabs.append(tt)
    table = pd.concat(tabs, ignore_index=True)
    ledger_spec = {"spec_file": str(SPEC_FILE), "family": "D2 잔차 갭 지속(개장 과소 반영)", "target": "r_oc(T)",
                   "z": "KS200/KPI200 open gap ~ SPX cum(D<T) OLS past250 rows excl T, min200, s=sqrt(SSR/(n-2))",
                   "cells": {"D2-C1": "z>=+1 -> long 069500 O->C", "D2-C2": "z<=-1 -> long 114800 O->C"},
                   "neighbors": "W120/W500 (min 0.8W), thr 0.75/1.25, SPX+IXIC", "transfer": "102110,122630 / 123310",
                   "judge": "B 2020-01-01~2025-06-05"}
    wrote, first_run = ledger_once(ledger_spec, table, "2차 확인")

    # ------------------------------------------------------------ 보고서
    fs = {cid: next(x for x in spec["frozen_specs"] if x["cell_id"] == cid) for cid in cells}
    zb = z["base"]
    zcov = {w: (int(zb.loc[a:b].notna().sum()), int(len(zb.loc[a:b]))) for w, (a, b) in WIN.items()}
    e4_last = zb.dropna().index.max().date()
    md = ["# R2 · D2 개장 반영 오류(잔차 갭 지속) 2차 확인\n",
          f"- 실행 {t_run}. 사양 `round2/D2/spec.json`(고정, 변경 없음). 스크립트 `code/overnight_lab/topics/r2_d2.py`.",
          f"- 판정 구간 B = 2020-01-01~2025-06-05(1회). A = 2010-01-04~2019-12-30(탐색에 쓴 표본 내 값), E4 = 2025-06-09~{e4_last}(보고만).",
          f"- z 유효일: A {zcov['A'][0]}/{zcov['A'][1]}, B {zcov['B'][0]}/{zcov['B'][1]}, E4 {zcov['E4'][0]}/{zcov['E4'][1]}"
          f"(E4 끝은 ^GSPC 파일의 마지막 확정 종가 {data.us_daily('spx')['close'].dropna().index.max().date()} 에 막힘).",
          f"- 시행 장부 `ledger.csv`: 주제 {TOPIC}, {len(table)}행(본 셀·이웃·이전 자산 × 8개 기간 × 5·7bp) "
          f"{'이번 실행에서 추가' if wrote else f'같은 사양으로 {first_run} 에 이미 기록돼 있어 다시 쓰지 않음'}.",
          f"- 114800 은 원자료 etf/ 캐시(네이버 siseJson 수정주가)에서 읽음: {int(inv['close'].notna().sum())}행, B 가격 지문 {fingerprint(inv, '2020-01-01', E.JUDGE_END)}"
          f"(069500 캐시 B 지문 {fingerprint(px, '2020-01-01', E.JUDGE_END)}).",
          "\n```\n" + __doc__.strip() + "\n```\n"]

    # 요약
    md += ["## 1. 판정 요약 (B)\n",
           md_table(["셀", "신호/유효일", "Δ(신호−비신호)", "HAC t", "단측 p(원값)", "② 7bp 순평균", "③ 플라시보 p", "④ 변동성 조정 Δ",
                     "⑤ 이웃 부호", "기준 ①~⑤", "판정", f"p ≤ 0.05/{N_CELLS_R2}"],
                    [[cid, f"{res[cid]['win']['B']['n_sig']}/{res[cid]['win']['B']['days']}", f"{F(res[cid]['win']['B']['delta_bp'])}bp",
                      F(res[cid]['win']['B']['t']), pfmt(res[cid]['win']['B']['p_one']), f"{F(res[cid]['win']['B']['net7_bp'])}bp",
                      pfmt(res[cid]['win']['B']['placebo_p']), f"{res[cid]['win']['B']['dvol']:+.4f} SD",
                      "동일" if res[cid]["neighbors_same_sign"] else "불일치", res[cid]["criteria"], f"**{res[cid]['grade']}**",
                      "예" if res[cid]['win']['B']['p_one'] <= 0.05 / N_CELLS_R2 else "아니오"] for cid in cells]),
           "",
           f"Holm 참고: 2차 전체 셀 수 상한 {N_CELLS_R2} 기준으로 가장 엄한 문턱은 0.05/{N_CELLS_R2} = {0.05 / N_CELLS_R2:.5f}, "
           "가장 느슨한 문턱은 0.05다. 보정은 종합 단계에서 한다.\n"]

    for cid, c in cells.items():
        r = res[cid]
        A, B, E4 = r["win"]["A"], r["win"]["B"], r["win"]["E4"]
        ev = fs[cid]["A_evidence"]
        md += [f"## {cid}: {'z ≥ +1 → 069500 KODEX 200' if c['up'] else 'z ≤ −1 → 114800 KODEX 인버스'} 시가 매수·종가 매도\n",
               f"사양 규칙: {fs[cid]['rule']}\n",
               f"### 판정표: **{r['grade']}** (기준 {r['criteria']})\n",
               md_table(["항목", "B (판정)", "기준", "A (표본 내)", "E4 (보고)"], [
                   ["유효일 / 신호일", f"{B['days']} / {B['n_sig']}", "", f"{A['days']} / {A['n_sig']}", f"{E4['days']} / {E4['n_sig']}"],
                   ["신호일 총평균(비용 전), NW t", f"{F(B['gross_bp'])}bp ({F(B['t_gross'])})", "", f"{F(A['gross_bp'])}bp ({F(A['t_gross'])})",
                    f"{F(E4['gross_bp'])}bp ({F(E4['t_gross'])})"],
                   ["비신호일 평균 / 전체 평균", f"{F(B['nonsig_bp'])} / {F(B['base_bp'])}", "", f"{F(A['nonsig_bp'])} / {F(A['base_bp'])}",
                    f"{F(E4['nonsig_bp'])} / {F(E4['base_bp'])}"],
                   ["① Δ(신호−비신호), HAC t", f"{F(B['delta_bp'])}bp (t {F(B['t'])})", "> 0", f"{F(A['delta_bp'])}bp (t {F(A['t'])})",
                    f"{F(E4['delta_bp'])}bp (t {F(E4['t'])})"],
                   ["① 단측 p(원값, Holm 은 종합)", pfmt(B["p_one"]), f"보고(≤{0.05 / N_CELLS_R2:.5f}면 어느 순위든 Holm 통과)", pfmt(A["p_one"]), pfmt(E4["p_one"])],
                   ["순평균 5bp", f"{F(B['net5_bp'])}bp", "", f"{F(A['net5_bp'])}bp", f"{F(E4['net5_bp'])}bp"],
                   ["② 순평균 7bp", f"{F(B['net7_bp'])}bp", "> 0", f"{F(A['net7_bp'])}bp", f"{F(E4['net7_bp'])}bp"],
                   ["중앙값 / 승률(비용 전)", f"{F(B['median_bp'])}bp / {B['win'] * 100:.1f}%", "", f"{F(A['median_bp'])}bp / {A['win'] * 100:.1f}%",
                    f"{F(E4['median_bp'])}bp / {E4['win'] * 100:.1f}%"],
                   ["③ 연도 층화 플라시보 p", pfmt(B["placebo_p"]), "≤ 0.05", pfmt(A["placebo_p"]), pfmt(E4["placebo_p"])],
                   ["④ 변동성 조정 Δ", f"{B['dvol']:+.4f} SD", "> 0", f"{A['dvol']:+.4f} SD", f"{E4['dvol']:+.4f} SD"],
                   ["⑤ 이웃 Δ 부호", "동일" if r["neighbors_same_sign"] else "불일치", "모두 동일", "", ""],
                   ["손익분기 왕복 비용(= 총평균)", f"{F(B['gross_bp'])}bp", "", f"{F(A['gross_bp'])}bp", f"{F(E4['gross_bp'])}bp"]]),
               "",
               f"A 재현 점검. 사양 A_evidence: \"{ev[:min(i for i in (ev.find('연도 층화'), ev.find('플라시보'), len(ev)) if i >= 0)].strip()}\" "
               f"→ 이 스크립트 A = 신호 {A['n_sig']}/{A['days']}일, "
               f"총평균 {F(A['gross_bp'])}bp(NW t {F(A['t_gross'])}), Δ {F(A['delta_bp'])}bp(HAC t {F(A['t'])}).\n",
               "### 이웃(보고, B 에서 부호만 판정 ⑤)\n",
               md_table(["이웃", "A n / 총평균 / Δ", "B n / 총평균 / Δ (t)", "E4 n / 총평균 / Δ"],
                        [[k, f"{r['neighbors']['A'][k]['n_sig']} / {F(r['neighbors']['A'][k]['gross_bp'])} / {F(r['neighbors']['A'][k]['delta_bp'])}",
                          f"{r['neighbors']['B'][k]['n_sig']} / {F(r['neighbors']['B'][k]['gross_bp'])} / {F(r['neighbors']['B'][k]['delta_bp'])} ({F(r['neighbors']['B'][k]['t'])})",
                          f"{r['neighbors']['E4'][k]['n_sig']} / {F(r['neighbors']['E4'][k]['gross_bp'])} / {F(r['neighbors']['E4'][k]['delta_bp'])}"]
                         for k in c["neighbors"]]),
               "",
               "### 이전 자산(보고만, 같은 신호·같은 시가→종가)\n",
               md_table(["자산", "A n / 총평균 / Δ", "B n / 총평균 / Δ (t)", "E4 n / 총평균 / Δ"],
                        [[k, f"{r['transfer']['A'][k]['n_sig']} / {F(r['transfer']['A'][k]['gross_bp'])} / {F(r['transfer']['A'][k]['delta_bp'])}",
                          f"{r['transfer']['B'][k]['n_sig']} / {F(r['transfer']['B'][k]['gross_bp'])} / {F(r['transfer']['B'][k]['delta_bp'])} ({F(r['transfer']['B'][k]['t'])})",
                          f"{r['transfer']['E4'][k]['n_sig']} / {F(r['transfer']['E4'][k]['gross_bp'])} / {F(r['transfer']['E4'][k]['delta_bp'])}"]
                         for k in c["transfer_ret"]]),
               "",
               "### 분해: 지수 몫(거래 불가) vs ETF 단일가 몫\n",
               f"신호일 ETF 시가→종가 = 지수 몫({'코스피200' if c['up'] else '−코스피200'} 시가→종가) + 단일가 몫(ETF − 지수). "
               f"시가 괴리 = ETF 갭 − {'' if c['up'] else '(−1)×'}지수 갭(로그, bp). 음수면 ETF 시가가 지수 갭을 덜 반영했다는 뜻이다.\n",
               md_table(["구간", "신호일", "ETF r_oc", "지수 몫", "단일가 몫", "지수 몫 Δ (t)", "단일가 몫 Δ (t)", "시가 괴리 신호일 / 전체"],
                        [[w, r["decomp"][w]["n_sig"], F(r["decomp"][w]["etf_bp"]), F(r["decomp"][w]["idx_bp"]), F(r["decomp"][w]["single_bp"]),
                          f"{F(r['decomp'][w]['idx_delta_bp'])} ({F(r['decomp'][w]['idx_t'])})",
                          f"{F(r['decomp'][w]['single_delta_bp'])} ({F(r['decomp'][w]['single_t'])})",
                          f"{F(r['decomp'][w]['dev_sig_bp'])} / {F(r['decomp'][w]['dev_all_bp'])}"] for w in WIN]
                        + [[k, s["dc_n_sig"], F(s["dc_etf_bp"]), F(s["dc_idx_bp"]), F(s["dc_single_bp"]),
                            f"{F(s['dc_idx_delta_bp'])} ({F(s['dc_idx_t'])})", f"{F(s['dc_single_delta_bp'])} ({F(s['dc_single_t'])})",
                            f"{F(s['dc_dev_sig_bp'])} / {F(s['dc_dev_all_bp'])}"] for k, s in r["sub"].items()]),
               "",
               "### 하위 구간(보고, 판정은 B 전체)\n",
               md_table(["구간", "신호/유효일", "총평균", "순평균 7bp", "Δ (t)", "변동성 조정 Δ"],
                        [[k, f"{s['n_sig']}/{s['days']}", F(s["gross_bp"]), F(s["net7_bp"]), f"{F(s['delta_bp'])} ({F(s['t'])})", f"{s['dvol']:+.4f}"]
                         for k, s in r["sub"].items()]),
               "",
               "### 연도별(비용 전 총평균, 5bp 순평균, Δ)\n",
               md_table(["연도", "구간", "신호일", "총평균", "순평균 5bp", "Δ"],
                        [[y["year"], "A" if y["year"] <= 2019 else ("B" if y["year"] <= 2024 else ("B/E4" if y["year"] == 2025 else "E4")), y["n"],
                          F(y["gross_bp"]), F(y["net5_bp"]), F(y["delta_bp"])] for y in r["yearly"]]),
               "",
               f"B 연도별 5bp 후 양수: {sum(1 for y in r['yearly'] if 2020 <= y['year'] <= 2024 and y['net5_bp'] > 0)}/5 (2020~2024, 2025 는 B·E4 가 섞여 제외).\n",
               "### 신호일만 보유 vs 매일 시가→종가(기준선), 5·7bp\n",
               E.fmt_table(r["strat"], cols=("rule", "window", "cost_bp", "trades", "exposure", "gross_bp", "net_bp", "t_net", "win", "sharpe", "cagr", "mdd")),
               f"가격 수준(중앙값) A {r['px_med']['A']:,.0f}원 / B {r['px_med']['B']:,.0f}원 / E4 {r['px_med']['E4']:,.0f}원. "
               f"호가단위를 2,000원 미만 1원·이상 5원으로 가정하면 1틱 = A {r['tick_bp']['A']:.1f} / B {r['tick_bp']['B']:.1f} / E4 {r['tick_bp']['E4']:.1f}bp.\n"]

    # 해석(수치는 모두 위 표에서 가져온다)
    c1, c2 = res["D2-C1"], res["D2-C2"]
    w1, w2 = c1["win"], c2["win"]
    d1, d2 = c1["decomp"], c2["decomp"]
    s1, s2 = c1["sub"], c2["sub"]
    late = "B 후반(2023-07-31~2025-06-05, 파생 08:45 개장)"
    early = "B 전반(2020-01~2023-07-28)"

    def holm_note(p):
        return (f"단측 p {pfmt(p)} 는 0.05/{N_CELLS_R2} 이하라 2차 셀 {N_CELLS_R2}개 Holm 에서 순위와 무관하게 살아남는다"
                if p <= 0.05 / N_CELLS_R2 else f"단측 p {pfmt(p)} 는 0.05/{N_CELLS_R2} 를 넘어 Holm 통과 여부가 다른 셀 p 에 달렸다")

    md += ["## 해석과 주의\n",
           f"- **C1(z ≥ +1 → 069500)**: 기준 {c1['criteria']}. B Δ {F(w1['B']['delta_bp'])}bp(HAC t {F(w1['B']['t'])}), "
           f"A {F(w1['A']['delta_bp'])}bp 의 {w1['B']['delta_bp'] / w1['A']['delta_bp'] * 100:.0f}% 수준. B 신호일 총평균 {F(w1['B']['gross_bp'])}bp, "
           f"7bp 후 {F(w1['B']['net7_bp'])}bp. {holm_note(w1['B']['p_one'])}. E4 는 Δ {F(w1['E4']['delta_bp'])}bp(t {F(w1['E4']['t'])})로 부호가 같다.",
           f"- **C2(z ≤ −1 → 114800)**: 기준 {c2['criteria']}. B Δ {F(w2['B']['delta_bp'])}bp(HAC t {F(w2['B']['t'])}), A {F(w2['A']['delta_bp'])}bp 보다 크다. "
           f"B 신호일 총평균 {F(w2['B']['gross_bp'])}bp, 7bp 후 {F(w2['B']['net7_bp'])}bp. {holm_note(w2['B']['p_one'])}. "
           f"E4 Δ {F(w2['E4']['delta_bp'])}bp(t {F(w2['E4']['t'])}).",
           f"- **효과의 구성이 A 와 다르다.** C1 의 B 신호일 수익 {F(d1['B']['etf_bp'])}bp 중 지수 몫 {F(d1['B']['idx_bp'])}, 단일가 몫 {F(d1['B']['single_bp'])}"
           f"(A: {F(d1['A']['idx_bp'])} / {F(d1['A']['single_bp'])}). 단일가 몫 Δ 는 {early} {F(s1[early]['dc_single_delta_bp'])}bp(t {F(s1[early]['dc_single_t'])})에서 "
           f"{late} {F(s1[late]['dc_single_delta_bp'])}bp(t {F(s1[late]['dc_single_t'])})로 줄었다. 08:45 파생 개장 뒤 ETF 시가 단일가 과소 반영이 거의 사라졌다는 사양의 우려와 맞는다. "
           f"C2 는 A 에서 Δ 대부분이 단일가 몫이었는데(단일가 Δ {F(d2['A']['single_delta_bp'])}, t {F(d2['A']['single_t'])} / 지수 Δ {F(d2['A']['idx_delta_bp'])}, "
           f"t {F(d2['A']['idx_t'])}) B 에서는 지수 Δ {F(d2['B']['idx_delta_bp'])}(t {F(d2['B']['idx_t'])}) / 단일가 Δ {F(d2['B']['single_delta_bp'])}"
           f"(t {F(d2['B']['single_t'])})로 뒤집혔다. 신호일 평균으로도 지수 몫 {F(d2['B']['idx_bp'])}bp, 단일가 몫 {F(d2['B']['single_bp'])}bp 다. "
           "B 의 C2 는 인버스 시가 괴리보다 지수 하방 지속에서 나왔다.",
           f"- **시가 동시성.** z 는 09:00 지수 시가로 계산하고 체결도 09:00 ETF 시가 단일가다. 일봉으로는 08:59 예상지수 기반 주문을 검증할 수 없다. "
           f"단일가 몫은 시가 단일가에서 체결하지 못하면 사라진다고 봐야 한다. 지수 몫만 남긴다고 가정하면 B 신호일 7bp 후 C1 {F(d1['B']['idx_bp'] - 7)}bp, "
           f"C2 {F(d2['B']['idx_bp'] - 7)}bp 다(지수 시가→종가는 거래 불가라 근사일 뿐).",
           f"- **하위 구간.** C1 {late} Δ {F(s1[late]['delta_bp'])}bp(t {F(s1[late]['t'])}), C2 {F(s2[late]['delta_bp'])}bp(t {F(s2[late]['t'])}). "
           "NXT 프리마켓 이후 B 안 표본(2025-03-04~06-05)은 신호가 한 자릿수라 판단 근거가 못 된다.",
           "- **방향 전환.** 두 셀 모두 A 에서 원 가설(되돌림)이 기각된 뒤 반대 방향(지속)으로 고정됐다. 이번 B 가 첫 표본 밖 검증이고, 사양·창·임계·지수 원천은 바꾸지 않았다.",
           "- **두 셀은 한 신호의 양 꼬리다.** 같은 z 를 쓰므로 독립 발견 두 개가 아니다. 같은 날 동시 신호는 없다(z ≥ 1 과 z ≤ −1 은 서로 배타).",
           f"- **체결 단위.** 114800 가격이 B 중앙값 {c2['px_med']['B']:,.0f}원, E4 {c2['px_med']['E4']:,.0f}원이라 1틱이 비용 5bp 와 비슷하거나 크다"
           f"(가정 호가단위 기준 B {c2['tick_bp']['B']:.1f}bp). 시가·종가 단일가 체결이라 스프레드를 내지는 않지만 수익률 잡음이 크다.",
           "- E4 는 보고만 한다. 2025-06-09 이후 국면이 급변했으므로(1차 요약) 판정에 쓰지 않는다.",
           ""]
    (OUT / "report.md").write_text("\n".join(md), encoding="utf-8")

    verdict = {"topic": TOPIC, "run_at": t_run, "spec_file": str(SPEC_FILE), "judge_window": WIN["B"], "n_cells_round2_max": N_CELLS_R2,
               "criteria_legend": "① Δ>0 ② 7bp 순평균>0 ③ 플라시보 p≤0.05 ④ 변동성 조정 Δ>0 ⑤ 이웃 부호 동일 (p_one 은 Holm 전 원값)",
               "data": {"114800_rows": int(inv["close"].notna().sum()), "114800_B_fingerprint": fingerprint(inv, "2020-01-01", E.JUDGE_END),
                        "069500_B_fingerprint": fingerprint(px, "2020-01-01", E.JUDGE_END), "z_valid": zcov, "E4_last_signal_day": str(e4_last)},
               "cells": {}}
    for cid in cells:
        r = res[cid]
        B = r["win"]["B"]
        verdict["cells"][cid] = {
            "grade": r["grade"], "criteria": r["criteria"], "n_sig_B": B["n_sig"], "n_days_B": B["days"], "delta_B_bp": B["delta_bp"],
            "t_B": B["t"], "p_one_B": B["p_one"], "gross_B_bp": B["gross_bp"], "net5_B_bp": B["net5_bp"], "net7_B_bp": B["net7_bp"],
            "placebo_p_B": B["placebo_p"], "dvol_B": B["dvol"], "neighbors_same_sign": r["neighbors_same_sign"],
            "neighbors_B_delta_bp": {k: v["delta_bp"] for k, v in r["neighbors"]["B"].items()},
            "transfer_B_delta_bp": {k: v["delta_bp"] for k, v in r["transfer"]["B"].items()},
            "A": r["win"]["A"], "E4": r["win"]["E4"], "B": B, "decomposition": r["decomp"], "subperiods": r["sub"], "yearly": r["yearly"],
            "strategy_vs_baseline": r["strat"][["rule", "window", "cost_bp", "trades", "net_bp", "sharpe", "cagr", "mdd"]].to_dict("records")}
    C.save_verdict(OUT / "verdict.json", verdict)
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print("\n".join(md[:12]))


if __name__ == "__main__":
    main()
