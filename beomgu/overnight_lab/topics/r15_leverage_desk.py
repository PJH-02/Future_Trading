"""15차 레버리지 데스크: 13차 장부 × k 스케일링이 'CAGR 30 이상 ∧ MDD 20 이내' 를 얼마나 채우는지.

읽기 전용 입력: round13_slot_book/book/book_daily.csv, series/*.csv, hyfe/data/raw/etf*.
출력: round15_alpha_share/survey/leverage_desk/ (csv + png). 원본 수정 없음.

수익 규약
  net = gross − cost (series 그대로), cost = 자본 기준 왕복 5bp(jehyeong 장중은 편도 2.5bp).
  mode fixed : k·gross − cost  (레버리지 ETF 교체 → 같은 자본에 같은 비용) = k·net + (k−1)·cost
  mode scaled: k·net           (비용도 k 배, 선물·신용 등 명목 비례 비용의 보수적 상한)
  mode etf   : 명목 |pos|·k 를 ETF 한도(롱 2, 코스닥 숏 1)로 자른 fixed. 한도 밖 몫 = 선물 필요.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LAB = Path("<OUT>")
R13 = LAB / "round13_slot_book"
OUT = LAB / "round15_alpha_share/survey/leverage_desk"
RAW = Path("<raw>")
OUT.mkdir(parents=True, exist_ok=True)

KS = [1, 1.25, 1.5, 1.75, 2, 2.5, 3]
WIN = {"W_all": ("2022-12-13", "2026-09-17"), "Y2023": ("2023-01-02", "2023-12-28"),
       "Y2024": ("2024-01-02", "2024-12-30"), "Y2025": ("2025-01-02", "2025-12-30"),
       "Y2026": ("2026-01-02", "2026-09-17"), "W_ex2026": ("2022-12-13", "2025-12-30")}
W_LONG = ("2016-10-04", "2026-09-17")   # 13차 W_long 과 동일(코스피 B, 코스닥 A+B)
W_KB03 = ("2003-01-02", "2026-09-17")   # 코스피 B 단독 최장
# book 열 → (핸들, 시장, 슬롯, 롱 명목 한도, 숏 명목 한도) — 한도는 ETF 만으로 가능한 최대 명목(자본 100%)
LEGS = {"KOSPI_A_jehyeong_ON": ("jehyeong_ON", "KOSPI", "A", 2, 2),
        "KOSPI_B_beomgu_C3VT30": ("beomgu_C3VT30", "KOSPI", "B", 2, 2),
        "KOSPI_C_jehyeong_KP": ("jehyeong_KP", "KOSPI", "C", 2, 2),
        "KOSDAQ_A_blog_ON_Q": ("blog_ON_Q", "KOSDAQ", "A", 2, 1),
        "KOSDAQ_B_beomgu_C3Q": ("beomgu_C3Q", "KOSDAQ", "B", 2, 1),
        "KOSDAQ_C_jehyeong_KQ": ("jehyeong_KQ", "KOSDAQ", "C", 2, 1)}
COMBOS = {"A": "밤만", "B": "아침만", "C": "낮만", "AB": "밤+아침", "AC": "밤+낮", "BC": "아침+낮", "ABC": "셋 다"}


# ---------------------------------------------------------------- 자료
def series(h):
    s = pd.read_csv(R13 / "series" / f"{h}.csv", parse_dates=["date"]).set_index("date")
    meta = json.loads((R13 / "series" / f"{h}.meta.json").read_text(encoding="utf-8"))
    s = s.loc[meta["first_valid"]:meta["last_valid"]]
    return s.fillna(0.0)


def panel_wall():
    """W_all: book_daily 의 다리별 net(B·C 겹침 처리 후) + series 의 cost·|pos|."""
    book = pd.read_csv(R13 / "book/book_daily.csv", parse_dates=["date"]).set_index("date")
    legs = {}
    for col, (h, mkt, slot, lc, sc) in LEGS.items():
        s = series(h).reindex(book.index).fillna(0.0)
        net = book[col]
        act = net != 0
        # 겹침으로 0 처리된 날은 비용·명목도 0. 활성일 net 은 series net 과 같아야 한다.
        assert np.allclose(net[act], s["net"][act]), col
        legs[col] = pd.DataFrame({"net": net, "cost": (s["gross"] - s["net"]).where(act, 0.0),
                                  "pos": s["pos"].where(act, 0.0)})
    chk = sum(l["net"] for c, l in legs.items() if c.startswith("KOSPI")), sum(l["net"] for c, l in legs.items() if c.startswith("KOSDAQ"))
    assert np.allclose(chk[0], book["KOSPI_book"]) and np.allclose(chk[1], book["KOSDAQ_book"])
    assert np.allclose(0.5 * (chk[0] + chk[1]), book["comb_a"])
    return legs


def panel_long(cols, w):
    """장기창: series 만으로(겹침 없음). cols = book 열 이름들."""
    legs = {}
    for col in cols:
        h = LEGS[col][0]
        s = series(h).loc[w[0]:w[1]]
        legs[col] = pd.DataFrame({"net": s["net"], "cost": s["gross"] - s["net"], "pos": s["pos"]})
    idx = None
    for l in legs.values():
        idx = l.index if idx is None else idx.intersection(l.index)
    return {c: l.reindex(idx) for c, l in legs.items()}


# ---------------------------------------------------------------- 스케일링
def lev_leg(l, k, mode, lc, sc):
    if mode == "scaled":
        return k * l["net"]
    if mode == "fixed":
        return k * l["net"] + (k - 1) * l["cost"]
    if mode == "fixed_res2bp":  # 레버리지 ETF 잔차(T4 평균 −2bp/일 수준)를 추가 노출분에 부과
        return k * l["net"] + (k - 1) * (l["cost"] - 2e-4 * (l["pos"] != 0))
    pos = l["pos"].abs()
    cap = np.where(l["pos"] >= 0, lc, sc)
    m = np.where(pos > 0, np.minimum(k * pos, cap) / pos.where(pos > 0, 1.0), 1.0)
    return m * l["net"] + (m - 1) * l["cost"]


def books(legs, kmap, mode):
    """kmap: 열 → k. 반환 {KOSPI, KOSDAQ, comb}; 시장이 하나뿐이면 comb = 그 시장."""
    out = {}
    for mkt in ("KOSPI", "KOSDAQ"):
        cols = [c for c in legs if c.startswith(mkt)]
        if cols:
            out[mkt] = sum(lev_leg(legs[c], kmap.get(c, 1.0), mode, LEGS[c][3], LEGS[c][4]) for c in cols)
    out["comb"] = 0.5 * (out["KOSPI"] + out["KOSDAQ"]) if len(out) == 2 else next(iter(out.values()))
    return out


def etf_overflow(legs, k):
    """mode etf 에서 ETF 한도로 잘린 활성일 수·비율(선물이 필요한 날)."""
    n_act = n_cut = 0
    for c, l in legs.items():
        pos = l["pos"].abs()
        act = pos > 0
        cap = np.where(l["pos"] >= 0, LEGS[c][3], LEGS[c][4])
        n_act += int(act.sum())
        n_cut += int((act & (k * pos > cap + 1e-12)).sum())
    return n_act, n_cut


# ---------------------------------------------------------------- 지표
def metrics(r):
    r = r.dropna()
    n = len(r)
    eq = (1 + r).cumprod()
    cum = float(eq.iloc[-1])
    cagr = cum ** (252 / n) - 1 if cum > 0 else -1.0
    vol = float(r.std(ddof=1)) * np.sqrt(252)
    dd = eq / eq.cummax() - 1
    act = r != 0
    ra = r[act]
    # 연속 손실: 활성일만 이어 붙여 음수 연속 구간 중 누적 손실이 가장 큰 것
    worst_run, run_n, run_c, best_n = 0.0, 0, 1.0, 0
    for x in ra:
        if x < 0:
            run_n += 1
            run_c *= 1 + x
            if run_c - 1 < worst_run:
                worst_run, best_n = run_c - 1, run_n
        else:
            run_n, run_c = 0, 1.0
    return dict(n_days=n, n_active=int(act.sum()), cum=cum - 1, cagr=cagr, vol=vol,
                sharpe=float(r.mean()) * 252 / vol if vol else np.nan, mdd=float(dd.min()),
                mdd_date=dd.idxmin().date().isoformat(), calmar=cagr / abs(dd.min()) if dd.min() else np.nan,
                worst_day=float(r.min()), worst_day_date=r.idxmin().date().isoformat(),
                loss_run_n=best_n, loss_run_ret=worst_run, win=float((ra > 0).mean()) if act.any() else np.nan,
                ruin=bool((eq <= 0).any()))


def win(r, w):
    return r.loc[w[0]:w[1]]


def verdict(m):
    if m["ruin"]:
        return "파산"
    ok_c, ok_m = m["cagr"] >= 0.30, m["mdd"] >= -0.20
    return "충족" if ok_c and ok_m else ("미달(CAGR)" if ok_m else ("미달(MDD)" if ok_c else "미달(둘 다)"))


# ---------------------------------------------------------------- 표
def table_k(legs_wall, legs_long, legs_kb):
    rows = []
    def add(scope, window, w, legs, book, k, mode):
        r = win(books(legs, {c: k for c in legs}, mode)[book], w)
        m = metrics(r)
        row = dict(scope=scope, window=window, start=w[0], end=w[1], book=book, mode=mode, k=k, **m, verdict=verdict(m))
        if mode == "etf":
            na, nc = etf_overflow({c: l.loc[w[0]:w[1]] for c, l in legs.items()}, k)
            row["etf_cut_days"], row["etf_cut_share"] = nc, nc / na if na else np.nan
        rows.append(row)
    for mode in ("fixed", "scaled", "etf", "fixed_res2bp"):
        for k in KS:
            for wn, w in WIN.items():
                for book in ("KOSPI", "KOSDAQ", "comb"):
                    add("W_all계열", wn, w, legs_wall, book, k, mode)
            for book in ("KOSPI", "KOSDAQ", "comb"):
                add("W_long", "W_long", W_LONG, legs_long, book, k, mode)
            add("코스피B_2003", "W_KB03", W_KB03, legs_kb, "comb", k, mode)
    return pd.DataFrame(rows)


def table_maxk(legs_wall, legs_long, legs_kb):
    """MDD ≥ −20% 인 최대 k 와 CAGR ≥ 30% 인 최소 k (격자 0.05)."""
    grid = np.round(np.arange(1.0, 6.0001, 0.05), 2)
    rows = []
    for name, legs, w, book in [("W_all", legs_wall, WIN["W_all"], "comb"), ("W_ex2026", legs_wall, WIN["W_ex2026"], "comb"),
                                ("W_long", legs_long, W_LONG, "comb"), ("W_KB03", legs_kb, W_KB03, "comb"),
                                ("W_all_KOSPI", legs_wall, WIN["W_all"], "KOSPI"), ("W_all_KOSDAQ", legs_wall, WIN["W_all"], "KOSDAQ")]:
        for mode in ("fixed", "scaled", "etf"):
            ms = {k: metrics(win(books(legs, {c: k for c in legs}, mode)[book], w)) for k in grid}
            ok = [k for k in grid if ms[k]["mdd"] >= -0.20 and not ms[k]["ruin"]]
            kmax = max(ok) if ok else np.nan
            need = [k for k in grid if ms[k]["cagr"] >= 0.30]
            kmin = min(need) if need else np.nan
            m = ms[kmax] if ok else {}
            # 같은 k 를 2026 제외창에 적용했을 때
            m26 = metrics(win(books(legs, {c: kmax for c in legs}, mode)[book], WIN["W_ex2026"])) if ok and legs is legs_wall else {}
            rows.append(dict(window=name, book=book, mode=mode, k_max_mdd20=kmax,
                             cagr_at_kmax=m.get("cagr"), mdd_at_kmax=m.get("mdd"), sharpe_at_kmax=m.get("sharpe"),
                             worst_day_at_kmax=m.get("worst_day"), loss_run_n=m.get("loss_run_n"), loss_run_ret=m.get("loss_run_ret"),
                             k_min_cagr30=kmin, mdd_at_kmin=ms[kmin]["mdd"] if need else np.nan,
                             ex2026_cagr_at_kmax=m26.get("cagr"), ex2026_mdd_at_kmax=m26.get("mdd"),
                             etf_cut_share=(etf_overflow({c: l.loc[w[0]:w[1]] for c, l in legs.items()}, kmax)[1]
                                            / max(1, etf_overflow({c: l.loc[w[0]:w[1]] for c, l in legs.items()}, kmax)[0])) if mode == "etf" and ok else np.nan))
    return pd.DataFrame(rows)


def table_combo(legs_wall, legs_long):
    rows = []
    for combo, label in COMBOS.items():
        for mode in ("fixed", "etf"):
            kmap_w = {c: (2.0 if LEGS[c][2] in combo else 1.0) for c in legs_wall}
            for wn, w in WIN.items():
                for book in ("KOSPI", "KOSDAQ", "comb"):
                    m = metrics(win(books(legs_wall, kmap_w, mode)[book], w))
                    rows.append(dict(combo=combo, label=label, mode=mode, window=wn, book=book, **m, verdict=verdict(m)))
            kmap_l = {c: (2.0 if LEGS[c][2] in combo else 1.0) for c in legs_long}
            for book in ("KOSPI", "KOSDAQ", "comb"):
                m = metrics(win(books(legs_long, kmap_l, mode)[book], W_LONG))
                rows.append(dict(combo=combo, label=label, mode=mode, window="W_long", book=book, **m, verdict=verdict(m)))
    return pd.DataFrame(rows)


def table_asym(legs_wall, legs_long):
    """시장별 k 비대칭 (kK, kQ) 격자. 기각 축 '배분 최적화' 와 겹치는 표본 내 선택이므로 참고용."""
    rows = []
    for kk in np.arange(1.0, 3.01, 0.25):
        for kq in np.arange(1.0, 2.01, 0.25):
            for mode in ("fixed", "etf"):
                for wn, legs, w in [("W_all", legs_wall, WIN["W_all"]), ("W_ex2026", legs_wall, WIN["W_ex2026"]), ("W_long", legs_long, W_LONG)]:
                    kmap = {c: (kk if c.startswith("KOSPI") else kq) for c in legs}
                    m = metrics(win(books(legs, kmap, mode)["comb"], w))
                    rows.append(dict(kK=kk, kQ=kq, mode=mode, window=wn, **m, verdict=verdict(m)))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 레버리지 ETF 잔차
def px(code, sub="etf"):
    p = RAW / sub / f"{code}.csv"
    return pd.read_csv(p, parse_dates=["date"]).set_index("date")[["open", "close"]].sort_index()


def table_etf_resid():
    """lev ETF 수익 − k × 기초 ETF 수익 (종가→종가 cc, 시가→종가 oc, 종가→익일 시가 co), bp."""
    pairs = [("122630", "etf", "069500", "etf", 2, "KODEX 레버리지 vs 2×KODEX200"),
             ("233740", "etf", "229200", "etf", 2, "KODEX 코스닥150레버리지 vs 2×코스닥150"),
             ("252670", "etf_extra", "069500", "etf", -2, "KODEX 200선물인버스2X vs −2×KODEX200"),
             ("114800", "etf", "069500", "etf", -1, "KODEX 인버스 vs −1×KODEX200"),
             ("251340", "etf_extra", "229200", "etf", -1, "KODEX 코스닥150인버스 vs −1×코스닥150")]
    rows = []
    for lev, ls, base, bs, k, label in pairs:
        a, b = px(lev, ls), px(base, bs)
        idx = a.index.intersection(b.index)
        a, b = a.loc[idx], b.loc[idx]
        rets = {"cc": lambda d: d["close"].pct_change(), "oc": lambda d: d["close"] / d["open"] - 1,
                "co": lambda d: d["open"] / d["close"].shift(1) - 1}
        for wn, w in [("W_all", WIN["W_all"]), ("since_2016-10", ("2016-10-04", "2026-09-17")), ("full", (str(idx.min().date()), "2026-09-17"))]:
            for hz, f in rets.items():
                res = (f(a) - k * f(b)).loc[w[0]:w[1]].dropna()
                rows.append(dict(pair=label, lev=lev, base=base, k=k, window=wn, start=w[0], end=w[1], horizon=hz, n=len(res),
                                 mean_bp=res.mean() * 1e4, median_bp=res.median() * 1e4, std_bp=res.std() * 1e4,
                                 mean_abs_bp=res.abs().mean() * 1e4, p95_abs_bp=res.abs().quantile(0.95) * 1e4,
                                 corr=float(np.corrcoef(f(a).loc[w[0]:w[1]].dropna().reindex(res.index), (k * f(b)).reindex(res.index))[0, 1])))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 그림
C = {"k1": "#2a78d6", "k1.5": "#eb6834", "k2": "#1baf7a", "k3": "#eda100"}  # dataviz 기본 팔레트 슬롯 1~4(검증 통과)


def fig_equity(legs_wall):
    w = WIN["W_all"]
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1.3]})
    for k, col in zip([1, 1.5, 2, 3], C.values()):
        r = win(books(legs_wall, {c: k for c in legs_wall}, "fixed")["comb"], w)
        eq = (1 + r).cumprod()
        m = metrics(r)
        lab = f"k={k}: CAGR {m['cagr']*100:.1f}% / MDD {m['mdd']*100:.1f}%"
        ax.plot(eq.index, eq, lw=1.6, color=col, label=lab)
        ax.annotate(f"k={k}", (eq.index[-1], eq.iloc[-1]), xytext=(4, 0), textcoords="offset points", color="#52514e", fontsize=9, va="center")
        ax2.plot(eq.index, (eq / eq.cummax() - 1) * 100, lw=1.2, color=col)
    ax.set_yscale("log")
    ax.set_title(f"13차 합산 장부 × k (mode fixed, 왕복 5bp) W_all {w[0]}~{w[1]}", fontsize=11, loc="left")
    ax.set_ylabel("누적(로그)")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax2.axhline(-20, color="#e34948", lw=1, ls="--")
    ax2.annotate("MDD 한도 −20%", (eq.index[5], -19), color="#e34948", fontsize=9, va="bottom")
    ax2.set_ylabel("낙폭 %")
    for a in (ax, ax2):
        a.grid(alpha=0.25)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "fig1_equity_k.png", dpi=130)
    plt.close(fig)


def fig_frontier(legs_wall, legs_long):
    grid = np.round(np.arange(0.5, 4.0001, 0.1), 2)
    fig, ax = plt.subplots(figsize=(8.5, 6))
    sets = [("W_all 2022-12~2026-09", legs_wall, WIN["W_all"], C["k1"]), ("2026 제외 2022-12~2025-12", legs_wall, WIN["W_ex2026"], C["k1.5"]),
            ("W_long 2016-10~2026-09 (두 장부)", legs_long, W_LONG, C["k2"])]
    for label, legs, w, col in sets:
        pts = [(k, metrics(win(books(legs, {c: k for c in legs}, "fixed")["comb"], w))) for k in grid]
        xs = [-m["mdd"] * 100 for _, m in pts]
        ys = [m["cagr"] * 100 for _, m in pts]
        ax.plot(xs, ys, lw=1.8, color=col, label=label)
        for k, m in pts:
            if k in (1.0, 2.0, 3.0):
                ax.scatter([-m["mdd"] * 100], [m["cagr"] * 100], s=36, color=col, zorder=3, edgecolor="#fcfcfb")
                ax.annotate(f"k={k:g}", (-m["mdd"] * 100, m["cagr"] * 100), xytext=(5, -3), textcoords="offset points", fontsize=8, color="#52514e")
    ax.axvspan(0, 20, color="#1baf7a", alpha=0.06)
    ax.axhline(30, color="#e34948", lw=1, ls="--")
    ax.axvline(20, color="#e34948", lw=1, ls="--")
    ax.annotate("목표 영역: MDD ≤ 20 (왼쪽) ∧ CAGR ≥ 30 (위)", (20.4, 2), fontsize=9, color="#e34948")
    ax.set_xlabel("MDD % (오른쪽이 큰 낙폭)")
    ax.set_ylabel("CAGR %")
    ax.set_title("13차 합산 장부 × k (0.5~4, 0.1 간격), mode fixed, 왕복 5bp", fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.grid(alpha=0.25)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "fig2_cagr_mdd_frontier.png", dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------- 메인
def main():
    plt.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    legs_wall = panel_wall()
    legs_long = panel_long(["KOSPI_B_beomgu_C3VT30", "KOSDAQ_A_blog_ON_Q", "KOSDAQ_B_beomgu_C3Q"], W_LONG)
    legs_kb = panel_long(["KOSPI_B_beomgu_C3VT30"], W_KB03)
    # 13차 재현 확인(k=1)
    base = metrics(books(legs_wall, {}, "fixed")["comb"])
    assert abs(base["cagr"] - 0.314179) < 0.002 and abs(base["mdd"] + 0.061046) < 0.001, base
    lb = metrics(books(legs_long, {}, "fixed")["comb"])
    assert lb["n_days"] == 2444 and abs(lb["cagr"] - 0.150369) < 0.002 and abs(lb["mdd"] + 0.096107) < 0.001, lb
    print(f"재현 k=1: W_all comb CAGR {base['cagr']:.4f} MDD {base['mdd']:.4f} | W_long comb CAGR {lb['cagr']:.4f} MDD {lb['mdd']:.4f} n {lb['n_days']}")
    # 다리 명목 분포(ETF 한도 판단용)
    for c, l in legs_wall.items():
        p = l["pos"][l["pos"] != 0]
        print(f"{c}: 활성 {len(p)} |pos| max {p.abs().max():.3f} mean {p.abs().mean():.3f} 숏일 {(p<0).sum()} cost mean {l['cost'][l['pos']!=0].mean()*1e4:.2f}bp")

    t1 = table_k(legs_wall, legs_long, legs_kb)
    t1.to_csv(OUT / "t1_k_scaling.csv", index=False, float_format="%.6f")
    t2 = table_maxk(legs_wall, legs_long, legs_kb)
    t2.to_csv(OUT / "t2_max_k.csv", index=False, float_format="%.6f")
    t3 = table_combo(legs_wall, legs_long)
    t3.to_csv(OUT / "t3_slot_combo_2x.csv", index=False, float_format="%.6f")
    t4 = table_etf_resid()
    t4.to_csv(OUT / "t4_lev_etf_residual.csv", index=False, float_format="%.4f")
    t5 = table_asym(legs_wall, legs_long)
    t5.to_csv(OUT / "t5_asym_kK_kQ.csv", index=False, float_format="%.6f")
    fig_equity(legs_wall)
    fig_frontier(legs_wall, legs_long)

    pd.set_option("display.width", 250, "display.max_columns", 40, "display.max_rows", 500)
    pct = lambda x: f"{x*100:.1f}"
    print("\n[T1 합산 comb, mode fixed] window × k → CAGR/MDD/최악일/샤프/연속손실")
    v = t1[(t1.book == "comb") & (t1["mode"] == "fixed")].copy()
    for col in ("cum", "cagr", "mdd", "worst_day", "loss_run_ret"):
        v[col] = v[col].map(pct)
    print(v[["window", "k", "cum", "cagr", "mdd", "worst_day", "sharpe", "loss_run_n", "loss_run_ret", "verdict"]].to_string(index=False))
    print("\n[T1 mode 비교, W_all·W_ex2026·W_long comb] CAGR/MDD")
    v = t1[(t1.book == "comb") & t1.window.isin(["W_all", "W_ex2026", "W_long", "W_KB03"])].pivot_table(index=["window", "k"], columns="mode", values=["cagr", "mdd"])
    print((v * 100).round(1).to_string())
    print("\n[T1 연수익(cum) etf·fixed·fixed_res2bp 비교, comb] Y2024·Y2025·W_ex2026")
    v = t1[(t1.book == "comb") & t1.window.isin(["Y2024", "Y2025", "W_ex2026", "W_all"]) & t1.k.isin([1, 1.5, 2, 3])].pivot_table(index=["window", "k"], columns="mode", values=["cum", "cagr", "mdd"])
    print((v * 100).round(1).to_string())
    print("\n[T5 비대칭 (kK,kQ) — MDD ≤ 20% 안에서 CAGR 최대, 창·mode 별 상위 3]")
    for (wn, mode), g in t5.groupby(["window", "mode"]):
        ok = g[(g.mdd >= -0.20) & ~g.ruin].nlargest(3, "cagr")
        print(f"  {wn} {mode}: " + " | ".join(f"kK {r.kK:.2f} kQ {r.kQ:.2f} → CAGR {r.cagr*100:.1f} MDD {r.mdd*100:.1f} 최악일 {r.worst_day*100:.1f}" for r in ok.itertuples()))
    print("  같은 (kK,kQ)=(2,1) 세 창: " + " | ".join(f"{r.window} {r['mode']} CAGR {r.cagr*100:.1f} MDD {r.mdd*100:.1f}" for _, r in t5[(t5.kK == 2.0) & (t5.kQ == 1.0)].iterrows()))
    print("  같은 (kK,kQ)=(2,1.5) 세 창: " + " | ".join(f"{r.window} {r['mode']} CAGR {r.cagr*100:.1f} MDD {r.mdd*100:.1f}" for _, r in t5[(t5.kK == 2.0) & (t5.kQ == 1.5)].iterrows()))
    print("\n[T1 etf 한도 초과 활성일 비율, comb]")
    print(t1[(t1.book == "comb") & (t1["mode"] == "etf") & t1.window.isin(["W_all", "W_long"])][["window", "k", "etf_cut_days", "etf_cut_share"]].to_string(index=False))
    print("\n[T2 최대 k]")
    print(t2.to_string(index=False))
    print("\n[T3 슬롯 2배 조합, comb, mode fixed]")
    v = t3[(t3.book == "comb") & (t3["mode"] == "fixed")].pivot_table(index=["combo", "label"], columns="window", values=["cagr", "mdd"])
    print((v * 100).round(1).to_string())
    print("\n[T3 슬롯 2배 조합, comb, mode etf]")
    v = t3[(t3.book == "comb") & (t3["mode"] == "etf")].pivot_table(index=["combo", "label"], columns="window", values=["cagr", "mdd"])
    print((v * 100).round(1).to_string())
    print("\n[T4 레버리지 ETF 잔차 bp]")
    print(t4[["pair", "window", "horizon", "n", "mean_bp", "median_bp", "std_bp", "mean_abs_bp", "p95_abs_bp", "corr"]].round(2).to_string(index=False))
    ter = {"069500": 0.15, "122630": 0.64, "233740": 0.64, "229200": 0.25, "252670": 0.64, "251340": 0.64, "114800": 0.64}
    print("\n[TER 일 환산 bp/일] " + ", ".join(f"{k} {v:.2f}%→{v/252*100:.3f}bp" for k, v in ter.items()))
    print("1일 보유 복리 괴리: 일 리셋 ETF 의 1일 수익 = k×기초 1일 수익 − 비용. 보유 n 일 괴리 ≈ −(k²−k)/2·σ_d²·n 은 n=1 에서 0(정의상). "
          "참고 σ_d=1.2%, k=2, n=1 → 1.44bp 상당의 분산항이 생기려면 n≥2 필요.")


if __name__ == "__main__":
    main()
