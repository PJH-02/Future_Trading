# -*- coding: utf-8 -*-
"""17차 실행판 장부 최종 표.
OUT 의 최종 csv(books, yearly, legs, waterfall, sensitivity, verify/*)를 읽어 OUT/final_tables.md 를 쓴다.
수치는 csv 에서 코드로 넣고, 일수익 csv(daily_13_exec, daily_15_exec, daily_15_exec_long)로 다시 계산해 대조한다.
실행: PYTHONUTF8=1 python r17_final_table.py
쓰는 파일: OUT/final_tables.md, OUT/verify/final_table_checks.csv (기존 파일은 읽기만 한다)
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd

LAB = Path("<OUT>")
OUT = LAB / "round17_exec_books"
RAW = Path("<raw>")
MINUS = "\u2212"
MID = "중간(09:01 진입, 15:30 청산)"


def rd(p, **k):
    return pd.read_csv(p, encoding="utf-8-sig", **k)


B, Y, LG, WF, SE = (rd(OUT / f) for f in ("books.csv", "yearly.csv", "legs.csv", "waterfall.csv", "sensitivity.csv"))
D13, D15, D15L = (rd(OUT / f, parse_dates=["date"]).set_index("date") for f in ("daily_13_exec.csv", "daily_15_exec.csv", "daily_15_exec_long.csv"))
RC, RV = rd(OUT / "verify/recompute_compare.csv"), rd(OUT / "verify/review.csv")
TR = rd(OUT / "c3_exec_trades.csv", parse_dates=["date"])

WIN = {"W_all": ("2022-12-13", "2026-09-17"), "W_des": ("2022-12-13", "2024-12-30"), "W_val": ("2025-01-02", "2026-09-17"),
       "W_ex2026": ("2022-12-13", "2025-12-30"), "Y2023": ("2023-01-01", "2023-12-31"), "Y2024": ("2024-01-01", "2024-12-31"),
       "Y2025": ("2025-01-01", "2025-12-31"), "Y2026": ("2026-01-01", "2026-09-17"), "W_long": ("2016-10-04", "2026-09-17")}
W4 = ["W_all", "W_des", "W_val", "W_ex2026"]
YRS = ["Y2023", "Y2024", "Y2025", "Y2026"]


# ---------------------------------------------------------------- 지표(사양: 누적 Π(1+r), CAGR = 누적^(252/일수) − 1, MDD = min(누적/누적최고 − 1))
def met(r):
    eq = (1 + r).cumprod()
    n, cum, sd = len(r), float(eq.iloc[-1]), float(r.std(ddof=1))
    return dict(cagr_pct=(cum ** (252 / n) - 1) * 100, mdd_pct=float((eq / eq.cummax() - 1).min()) * 100, ret_pct=(cum - 1) * 100,
                sharpe=float(r.mean()) * 252 / (sd * np.sqrt(252)), worst_day=str(r.idxmin().date()), worst_pct=float(r.min()) * 100, n_days=n)


def nw_t(x, lag=5):
    x = np.asarray(x, float)
    n = len(x)
    e = x - x.mean()
    s = e @ e / n
    for j in range(1, min(lag, n - 1) + 1):
        s += 2 * (1 - j / (lag + 1)) * (e[j:] @ e[:-j]) / n
    return float(x.mean() / np.sqrt(s / n))


def win(s, w):
    a, b = WIN[w]
    return s.loc[a:b]


# ---------------------------------------------------------------- 재대조
CHK = []


def chk(name, got, exp, tol):
    try:
        d = float(got) - float(exp)
        ok = abs(d) <= tol
    except (TypeError, ValueError):
        d, ok = np.nan, str(got) == str(exp)
    CHK.append(dict(check=name, got=got, exp=exp, diff=d, tol=tol, ok=ok))


def brow(book, ver, cost, sub, w):
    r = B[(B.book == book) & (B.version == ver) & (B.cost == cost) & (B["sub"] == sub) & (B.window == w)]
    assert len(r) == 1, (book, ver, cost, sub, w)
    return r.iloc[0]


DCOL = {("13차", "원형", "13차 비용", "합산"): D13.comb_orig13, ("13차", "실행판", "13차 비용", "합산"): D13.comb_13cost,
        ("13차", "실행판", "기본", "합산"): D13.comb, ("13차", "실행판", "하한", "합산"): D13.comb_low,
        ("13차", "실행판", "기본", "코스피"): D13.KOSPI, ("13차", "실행판", "기본", "코스닥"): D13.KOSDAQ,
        ("15차 v3", "대표", "기본(15차)", "합산"): D15.comb_rep, ("15차 v3", "실행판", "기본", "합산"): D15.comb, ("15차 v3", "실행판", "하한", "합산"): D15.comb_low}
LCOL = {("15차 v3", "대표", "기본(15차)", "합산"): D15L.comb_long_rep, ("15차 v3", "실행판", "기본", "합산"): D15L.comb_long_base,
        ("15차 v3", "실행판", "하한", "합산"): D15L.comb_long_low}
RECHECKED = set()
for _, r in B.iterrows():
    key = (r.book, r.version, r.cost, r["sub"])
    src = (LCOL if r.window == "W_long" else DCOL).get(key)
    if src is None:
        continue
    m = met(win(src, r.window))
    RECHECKED.add(key)
    tag = f"books {r.book} {r.version} {r.cost} {r['sub']} {r.window}"
    for k, tol in (("cagr_pct", 2e-3), ("mdd_pct", 2e-3), ("ret_pct", 5e-3), ("sharpe", 2e-3), ("worst_pct", 1e-3), ("n_days", 0)):
        chk(f"{tag} {k}", m[k], r[k], tol)
    chk(f"{tag} worst_day", m["worst_day"], r.worst_day, 0)

for _, r in Y.iterrows():                                                   # yearly.csv = books.csv 연도 행
    b = brow(r.book, r.version, r.cost, "합산", f"Y{r.year}")
    for k in ("cagr_pct", "mdd_pct", "ret_pct", "sharpe", "worst_pct", "n_days"):
        chk(f"yearly {r.book} {r.version} {r.cost} {r.year} {k}", r[k], b[k], 1e-9)

ENDS = {("13차", 0): ("원형", "13차 비용"), ("13차", 5): ("실행판", "13차 비용"), ("13차", 7): ("실행판", "기본"), ("13차", 8): ("실행판", "하한"),
        ("15차 v3", 0): ("대표", "기본(15차)"), ("15차 v3", 1): (MID, "기본(15차)"), ("15차 v3", 3): ("실행판", "기본"), ("15차 v3", 4): ("실행판", "하한")}
for _, r in WF.iterrows():                                                  # 폭포 끝점 = books, Δ = 앞 단계 차
    if (r.book, r.step) in ENDS:
        v, c = ENDS[(r.book, r.step)]
        for w in ("W_all", "W_val"):
            b = brow(r.book, v, c, "합산", w)
            chk(f"waterfall {r.book} step{r.step} {w} cagr", r[f"{w}_cagr"], b.cagr_pct, 1e-9)
            chk(f"waterfall {r.book} step{r.step} {w} mdd", r[f"{w}_mdd"], b.mdd_pct, 1e-9)
    if r.step > 0:
        p = WF[(WF.book == r.book) & (WF.step == r.step - 1)].iloc[0]
        z = WF[(WF.book == r.book) & (WF.step == 0)].iloc[0]
        for w in ("W_all", "W_val"):
            chk(f"waterfall {r.book} step{r.step} {w} d_cagr", r[f"{w}_d_cagr"], r[f"{w}_cagr"] - p[f"{w}_cagr"], 2e-6)
            chk(f"waterfall {r.book} step{r.step} {w} d_mdd", r[f"{w}_d_mdd"], r[f"{w}_mdd"] - p[f"{w}_mdd"], 2e-6)
            chk(f"waterfall {r.book} step{r.step} {w} d_cagr_from0", r[f"{w}_d_cagr_from0"], r[f"{w}_cagr"] - z[f"{w}_cagr"], 2e-6)

for _, r in SE.iterrows():                                                  # 민감도 기준 = 실행판 기본, s3 k 1.40 = books
    b = brow(r.book, "실행판", "기본", "합산", r.window)
    chk(f"sens {r.book} {r["item"]} {r.variant} {r.window} base_cagr", r.base_cagr_pct, b.cagr_pct, 1e-9)
    chk(f"sens {r.book} {r["item"]} {r.variant} {r.window} base_mdd", r.base_mdd_pct, b.mdd_pct, 1e-9)
    chk(f"sens {r.book} {r["item"]} {r.variant} {r.window} d_cagr", r.d_cagr, r.cagr_pct - r.base_cagr_pct, 2e-6)  # csv 6자리 반올림 두 번
    if r["item"] == "s3" and r.variant.startswith("k 1.40"):
        c = "기본" if r.variant.endswith("기본") else "하한"
        chk(f"sens {r.variant} {r.window} = books", r.cagr_pct, brow(r.book, "실행판", c, "합산", r.window).cagr_pct, 1e-9)
        chk(f"sens {r.variant} W_long = books", r.W_long_cagr_pct, brow(r.book, "실행판", c, "합산", "W_long").cagr_pct, 1e-9)

for book, df in (("13차", D13), ("15차 v3", D15)):                        # 다리(실행판·기본) 활성일 n·순 bp·NW t·창 CAGR/MDD
    for leg in ("K_A", "K_B", "Q_A", "Q_B"):
        for w in ("W_all", "W_val"):
            s = win(df[leg], w)
            a = s[s != 0]
            g = LG[(LG.book == book) & (LG.version == "실행판·기본") & (LG.leg == leg) & (LG.window == w)].iloc[0]
            m = met(s)
            tag = f"legs {book} 실행판·기본 {leg} {w}"
            chk(f"{tag} n", len(a), g.n, 0)
            chk(f"{tag} net_bp", a.mean() * 1e4, g.net_bp, 2e-3)
            chk(f"{tag} t_nw", nw_t(a.values), g.t_nw, 2e-3)
            chk(f"{tag} cagr", m["cagr_pct"], g.cagr_pct, 2e-3)
            chk(f"{tag} mdd", m["mdd_pct"], g.mdd_pct, 2e-3)
NC = {leg: int((win(D13[leg], "W_all") != 0).sum()) for leg in ("K_C", "Q_C")}   # 겹침 처리 뒤 장부 기여 일수


# 틱(5원 / 그해 평균 수정가, 틀 tk 규약) — RAW 수정가로 재계산, c3_exec_trades 의 cost_tick/cash 와 대조
def etf(code):
    p = RAW / "etf" / f"{code}.csv"
    if not p.exists():
        p = RAW / "etf_extra" / f"{code}.csv"
    return rd(p, parse_dates=["date"]).set_index("date").sort_index()


ETFS = ["069500", "122630", "229200", "233740", "251340"]
TICK = pd.DataFrame({c: (5 / etf(c).close.groupby(etf(c).index.year).mean() * 1e4).loc[2022:2026] for c in ETFS})  # 거래 파일에 2022.12 거래 1건 있음
tr = TR[(TR.cash > 0) & (~TR.rest)].assign(year=lambda d: d.date.dt.year, bp=lambda d: d.cost_tick / d.cash * 1e4)
for (c, y), v in tr.groupby([tr.code.astype(str).str.zfill(6), "year"]).bp.mean().items():
    chk(f"tick {c} {y} = c3_exec_trades cost_tick/cash", TICK.loc[y, c], v, 2e-3)

CK = pd.DataFrame(CHK)
N_OK, N_ALL = int(CK.ok.sum()), len(CK)


# ---------------------------------------------------------------- 서식
def num(x, d=2, sign=False):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    s = f"{abs(float(x)):.{d}f}"
    if float(s) == 0:
        return s
    return (MINUS if float(x) < 0 else ("+" if sign else "")) + s


def kd(s):
    y, m, d = str(s)[:10].split("-")
    return f"{y}.{int(m)}.{int(d)}"


def cm(c, m):
    return f"{num(c)} / {num(m)}"


HANDLE = {"jehyeong_ON": "코스피 밤 K_A", "beomgu_C3VT30": "코스피 아침 C3 K_B", "jehyeong_KP": "코스피 낮 K_C",
          "blog_ON_Q": "코스닥 밤 Q_A", "beomgu_C3Q": "코스닥 아침 C3Q Q_B", "jehyeong_KQ": "코스닥 낮 Q_C"}


def kt(s):
    """csv 문자열 → 보고 서식(날짜 2026.9.17, 음수 −, 다리 기호)."""
    s = "" if s is None or (isinstance(s, float) and np.isnan(s)) else str(s)
    for h, v in HANDLE.items():
        s = s.replace(h, v)
    s = re.sub(r"(\d{4})-(\d{2})-(\d{2})", lambda m: f"{m[1]}.{int(m[2])}.{int(m[3])}", s)
    s = re.sub(r"e-(\d)", "e" + MINUS + r"\1", s)
    s = re.sub(r"(?<![\w.])-(?=\d)", MINUS, s)
    return s


def rv(sub, col="value"):
    r = RV[RV.check.str.contains(sub, regex=False)]
    assert len(r) >= 1, sub
    return kt(r.iloc[0][col])


def table(head, rows, right_from=1):
    al = ["---"] * right_from + ["---:"] * (len(head) - right_from)
    out = ["| " + " | ".join(head) + " |", "|" + "|".join(al) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def nd(w):
    return int(brow("15차 v3", "대표", "기본(15차)", "합산", w).n_days)


V13 = [("원형", "13차 비용", "원형"), ("실행판", "13차 비용", "실행판(체결만 바꿈)"), ("실행판", "기본", "실행판 기본"), ("실행판", "하한", "실행판 하한")]
V15 = [("대표", "기본(15차)", "대표"), (MID, "기본(15차)", MID), ("실행판", "기본", "실행판 기본"), ("실행판", "하한", "실행판 하한")]
LEGN13 = {"K_A": "코스피 밤 ETF", "K_B": "코스피 아침 C3", "K_C": "코스피 낮(팀)", "Q_A": "코스닥 밤 ETF", "Q_B": "코스닥 아침 C3Q", "Q_C": "코스닥 낮(팀)"}
LEGN15 = dict(LEGN13, Q_A="코스닥 밤 선물", Q_C="코스닥 낮(팀, 숏 선물 대리)")
LV13 = ["원형·13차 비용", "실행판·13차 비용", "실행판·기본", "실행판·하한"]
LV15 = ["대표·기본(15차)", MID + "·기본(15차)", "실행판·기본", "실행판·하한"]


def book_rows(book, vers, wins, years, long=False):
    rows = []
    for v, c, lab in vers:
        row = [lab, c] + [cm(brow(book, v, c, "합산", w).cagr_pct, brow(book, v, c, "합산", w).mdd_pct) for w in wins]
        if long:
            row.append(cm(brow(book, v, c, "합산", "W_long").cagr_pct, brow(book, v, c, "합산", "W_long").mdd_pct))
        row += [cm(brow(book, v, c, "합산", y).ret_pct, brow(book, v, c, "합산", y).mdd_pct) for y in years]
        rows.append(row)
    return rows


def sharpe_line(book, vers):
    return " · ".join(f"{lab}({c}) 샤프 {num(brow(book, v, c, '합산', 'W_all').sharpe)}, 최악일 {kd(brow(book, v, c, '합산', 'W_all').worst_day)} "
                      f"{num(brow(book, v, c, '합산', 'W_all').worst_pct)}%" for v, c, lab in vers)


def leg_rows(book, vers, names):
    rows = []
    for leg in ("K_A", "K_B", "K_C", "Q_A", "Q_B", "Q_C"):
        for v in vers:
            cells = [f"{leg} {names[leg]}", v]
            for w in ("W_all", "W_val"):
                g = LG[(LG.book == book) & (LG.version == v) & (LG.leg == leg) & (LG.window == w)].iloc[0]
                cells += [int(g.n), num(g.net_bp, 1, sign=True), num(g.t_nw), cm(g.cagr_pct, g.mdd_pct)]
            rows.append(cells)
    return rows


def wf_rows(book):
    rows = []
    for _, r in WF[WF.book == book].sort_values("step").iterrows():
        d = (lambda k: "" if r.step == 0 else num(r[k], sign=True))
        rows.append([f"{int(r.step)}. {kt(r.label)}", cm(r.W_all_cagr, r.W_all_mdd), d("W_all_d_cagr"), d("W_all_d_mdd"), d("W_all_d_cagr_from0"),
                     cm(r.W_val_cagr, r.W_val_mdd), d("W_val_d_cagr"), d("W_val_d_mdd"), d("W_val_d_cagr_from0")])
    return rows


def sens_rows():
    rows = []
    for _, r in SE.iterrows():
        wl = cm(r.W_long_cagr_pct, r.W_long_mdd_pct) if pd.notna(r.W_long_cagr_pct) else ""
        rows.append([r.book, r["item"], kt(r.variant), r.window, cm(r.cagr_pct, r.mdd_pct), cm(r.base_cagr_pct, r.base_mdd_pct),
                     num(r.d_cagr, sign=True), num(r.d_mdd, sign=True), wl])
    return rows


def top5_rows():
    rows = []
    for _, r in RV[RV.section == "6_집중"].iterrows():
        m = re.match(r"^(13차|15차 v3) (원형|대표|실행판 기본|실행판 하한) (W_all|W_val) CAGR", r.check)
        v = re.match(r"^([\d.+-]+) → ([\d.+-]+) \(([\d.+-]+) → ([\d.+-]+)\)", r.value)
        assert m and v, r.check
        cost = {"원형": "13차 비용", "대표": "기본(15차)", "실행판 기본": "기본", "실행판 하한": "하한"}[m[2]]
        rows.append([m[1], m[2], cost, m[3], f"{kt(v[1])} → {kt(v[2])}", f"{kt(v[3])} → {kt(v[4])}", kt(r.note).replace("상위 5일: ", "")])
    return rows


def rrows(section):
    out = []
    for _, r in RV[RV.section == section].iterrows():
        st = "참고" if pd.isna(r["pass"]) else ("통과" if str(r["pass"]) == "True" else "불일치")
        exp = f" (기준 {kt(r.expected)})" if pd.notna(r.expected) else ""
        note = f" {kt(r.note)}" if pd.notna(r.note) else ""
        out.append(f"  · {kt(r.check)}: {kt(r.value)}{exp} → {st}.{note}")
    return out


def minor_rows():
    out = []
    for _, r in RV[RV.flag == "minor"].iterrows():
        exp = f" (본 산출 {kt(r.expected).replace('r17 ', '')})" if pd.notna(r.expected) else ""
        out.append(f"  · {kt(r.check)}: {kt(r.value)}{exp}.")
    return out


def b(book, v, c, w, k="cagr_pct"):
    return brow(book, v, c, "합산", w)[k]


def cmb(book, v, c, w):
    return cm(b(book, v, c, w), b(book, v, c, w, "mdd_pct"))


def wfl(book, step):
    r = WF[(WF.book == book) & (WF.step == step)].iloc[0]
    return r


# ---------------------------------------------------------------- 본문
tick_rows = [[c] + [num(TICK.loc[y, c], 1) for y in (2023, 2024, 2025, 2026)] for c in ETFS]
n_fail = int((RV["pass"].astype(str) == "False").sum())
n_minor, n_note = int((RV.flag == "minor").sum()), int((RV.flag == "note").sum())
rc_dc, rc_dm = RC.d_cagr_pp.abs().max(), RC.d_mdd_pp.abs().max()
ck_fail = CK[~CK.ok]
wf13_3 = wfl("13차", 3)

L = []
P = L.append
P("# 17차 실행판 장부 최종 표")
P("")
P("• 대상: 13차 장부(합산, k 1.00, ETF 만)와 15차 v3 장부(k 1.40). 신호를 본 가격에 그대로 체결한다고 가정한 다리를 실행 가능한 체결로 바꿔 다시 백테스트한 결과. 규칙·신호·비중·배수는 바꾸지 않았고 결과를 보고 고치지 않았다.")
P("• 출처: round17_exec_books/books.csv, yearly.csv, legs.csv, waterfall.csv, sensitivity.csv, verify/recompute_compare.csv, verify/review.csv. 산출 r17_exec_books.py, 표 작성 r17_final_table.py.")
P(f"• 재대조: 이 표의 수치를 csv 에서 읽어 넣고, 일수익 csv 로 다시 계산해 대조한 결과 {N_ALL}건 중 {N_OK}건 일치(7절).")
P("")
P("## 결과 요약(합산 장부, CAGR % / MDD %)")
P("")
P(f"• 13차 장부: W_all 원형(13차 비용) {cmb('13차', '원형', '13차 비용', 'W_all')} → 체결만 바꿈(13차 비용) {cmb('13차', '실행판', '13차 비용', 'W_all')} → 실행판 기본 {cmb('13차', '실행판', '기본', 'W_all')} → 실행판 하한 {cmb('13차', '실행판', '하한', 'W_all')}.")
P(f"• 13차 장부: W_val(봉인) 원형(13차 비용) {cmb('13차', '원형', '13차 비용', 'W_val')} → 실행판 기본 {cmb('13차', '실행판', '기본', 'W_val')} → 실행판 하한 {cmb('13차', '실행판', '하한', 'W_val')}.")
P(f"• 15차 v3 장부(k 1.40): W_all 대표(기본(15차)) {cmb('15차 v3', '대표', '기본(15차)', 'W_all')} → 중간(기본(15차)) {cmb('15차 v3', MID, '기본(15차)', 'W_all')} → 실행판 기본 {cmb('15차 v3', '실행판', '기본', 'W_all')} → 실행판 하한 {cmb('15차 v3', '실행판', '하한', 'W_all')}.")
P(f"• 15차 v3 장부(k 1.40): W_val(봉인) 대표 {cmb('15차 v3', '대표', '기본(15차)', 'W_val')} → 실행판 기본 {cmb('15차 v3', '실행판', '기본', 'W_val')} → 실행판 하한 {cmb('15차 v3', '실행판', '하한', 'W_val')}. W_long 대표 {cmb('15차 v3', '대표', '기본(15차)', 'W_long')} → 실행판 기본 {cmb('15차 v3', '실행판', '기본', 'W_long')} → 실행판 하한 {cmb('15차 v3', '실행판', '하한', 'W_long')}.")
P(f"• 13차 하락의 가장 큰 몫은 팀 낮 다리 1틱 단계(W_all {num(wfl('13차', 7).W_all_d_cagr, sign=True)}%p, 기본 비용 규약), 그다음이 아침 C3·C3Q 09:01 진입(W_all {num(wfl('13차', 1).W_all_d_cagr, sign=True)}%p·{num(wfl('13차', 3).W_all_d_cagr, sign=True)}%p). 15차 v3 하락의 가장 큰 몫은 09:01 진입(W_all {num(wfl('15차 v3', 1).W_all_d_cagr, sign=True)}%p).")
P("")
P("### 창·지표·기호")
P("")
P(f"• 창: W_all 2022.12.13~2026.9.17({nd('W_all')}일), W_des 2022.12.13~2024.12.30({nd('W_des')}일), W_val 2025.1.2~2026.9.17({nd('W_val')}일), W_ex2026 2022.12.13~2025.12.30({nd('W_ex2026')}일), 연도 2023·2024·2025·2026(~9.17), W_long 2016.10.4~2026.9.17({nd('W_long'):,}일, 15차 장부만, 0.5 × (K_B + Q_A + Q_B)).")
P("• 지표: 누적 Π(1+r), CAGR = 누적^(252/일수) − 1, MDD = min(누적/누적최고 − 1), 샤프 = 평균×252 / (표준편차×√252), 최악일. 다리는 활성일 순수익 평균(bp), NW t(HAC 5), 창 전체(비활성일 0) CAGR/MDD.")
P("• 다리 기호: K_A 코스피 밤 ETF(069500), K_B 코스피 아침 C3(VT30 비중), K_C 코스피 낮 팀 장중, Q_A 코스닥 밤(13차 ETF판 233740·251340, 15차 선물 15:45), Q_B 코스닥 아침 C3Q, Q_C 코스닥 낮 팀 장중(13차 롱 229200·숏 251340, 15차 숏 선물 대리).")
P("• 합산 = 코스피 50% + 코스닥 50% 매일 재조정. K_C·Q_C 는 아침 다리 활성일(원 series 기준) 0.")
P("")
P("## 1. 바꾼 체결 규약")
P("")
P("• 코스피 아침 C3(K_B)·코스닥 아침 C3Q(Q_B)")
P("  · 원형: 09:00 단일가로 신호 확정, 같은 09:00 가격에 체결, 15:30 종가 청산.")
P("  · 실행판: 09:00 단일가(09:01 봉 시가)로 신호 확정 → 09:01:00 첫 연속 체결(09:02 봉 시가) 진입 → 15:19:00(15:20 봉 시가) 청산.")
P(f"  · 09:01 봉 거래량 0 이면 그날 쉼(W_all: K_B {kt(rv('아침 쉼 날 원형 net', 'check')).split('K_B ')[1].split(' /')[0]}, Q_B {kt(rv('아침 쉼 날 원형 net', 'check')).split('Q_B ')[1].split(' (')[0]} 각 1일). 09:02 봉 거래량 0 이면 다음 거래 봉 시가로 민다(0건). 15:20 봉 거래량 0 이면 15:30 단일가(0건).")
P("  · 비중 x = 13차 series |pos| 그대로(VT30, 최대 2). x ≤ 1 은 1배 ETF(069500·229200) x 배, 1 < x ≤ 2 는 (x−1)×2배 ETF(122630·233740) + (2−x)×1배 ETF. 수익은 TOSS 1분봉 원가격 안에서만 잰다.")
P("  · 15차 k 1.40 에서 명목이 2 를 넘는 몫(선물)은 1배 ETF 의 09:02→15:20 체결 수익으로 근사(선물 왕복 1.8bp, 틱 없음). 2022.11.23 전(W_long 만)은 (1+oc)/(1+m)/(1+e) − 1 근사, m = 첫 1분 평균 변화, e = W_all 활성일 평균(15:30 종가 / 15:19:00 시가 − 1).")
P(f"  · 근사 계수(bp) K_B m / e: {rv('15차 2022-11 전 근사 계수(bp) K_B')}. Q_B m / e: {rv('15차 2022-11 전 근사 계수(bp) Q_B')}.")
P("• 코스닥 밤 ETF판(Q_A, 13차 장부)")
P("  · 원형: 15:30 종가로 IBS 계산, 같은 종가에 체결, 다음날 시가 청산.")
P("  · 실행판: 229200 의 09:01~15:20 봉 고가·저가와 15:20 봉 종가로 IBS20 계산(15:20 확정) → 15:30 단일가(15:30 봉 종가) 체결 → 다음날 09:00 단일가(09:01 봉 시가) 청산. 손익 날짜 = 청산일.")
P("  · IBS20 < 0.3 ∧ 전일 종가 > 전일까지 MA10 → 233740 현금 1(명목 2). IBS20 > 0.7 ∧ 전일 종가 < MA10 → 251340 현금 1(명목 −1). MA10·전일 종가는 229200 원가격 일봉.")
P("  · 특수일 7일(늦은 개장·수능일, 사전 공지 일정) 신호 제외. 청산일 09:01 봉 거래량 0 이면 그 밤 제외(3건). 16차 캐시 QA_bm 과 net 일치.")
P("• 그대로 둔 다리")
P("  · 코스피 밤(K_A): 15:20 지표 신호, 15:30 종가 매수, 다음날 09:00 시가 매도. 신호 시각 < 체결 시각.")
P("  · 15차 v3 의 코스닥 밤 선물(15:45 체결)과 코스닥 낮 숏 선물 대리.")
P("  · 팀 장중 다리 K_C·Q_C: 자체 1분봉 체결. 저장 일수익에 틱만 더한다.")
P("• 비용")
P("  · 13차 비용(원형·체결만 바꾼 단계): 아침·밤 ETF 다리 왕복 5bp, 팀 다리 자체 비용, 틱 없음.")
P("  · 기본(15차)(15차 대표·중간): 15차 v3 규약 그대로. ETF 왕복 5bp, 팀 장중 다리 1틱 왕복, 선물 왕복 1.8bp×명목, 낮 숏 선물 대리 +1.06bp, 아침 ETF 틱 없음.")
P("  · 기본(실행판): ETF 왕복 5bp(현금 min(x,1) 기준) + 연속 체결이 끼는 왕복마다 1틱 왕복(아침 C3·C3Q 양쪽 연속 체결, 팀 장중 K_C 혼합·Q_C 롱 229200·숏 251340) + 선물 왕복 1.8bp×명목 + 낮 숏 선물 대리 +1.06bp. 밤 ETF 단일가 다리는 틱 없음.")
P("  · 하한: 기본 + 단일가 체결 ETF 다리(코스피 밤 069500, 코스닥 밤 233740·251340, 15차 2016~2017 233740 롱 밤 포함) 1틱 왕복 + 선물 왕복 3bp(1.8bp 를 3bp 로). 15차 아침 초과(선물) 몫은 틱 없음.")
P("  · 틱 = 5원 / 그해 평균 수정가(틀 tk 규약, RAW 수정가로 재계산, c3_exec_trades 의 cost_tick/cash 와 일치). 왕복 1틱(bp):")
P("")
P(table(["ETF", "2023", "2024", "2025", "2026"], tick_rows))
P("")
P("## 2. 표 1 13차 장부(합산, k 1.00, ETF 만)")
P("")
P("• 칸 = CAGR % / MDD %. 연도 칸 = 그 해 누적 수익 % / 그 해 MDD %. 비용은 둘째 열.")
P("")
P(table(["판", "비용", "W_all", "W_des", "W_val", "W_ex2026", "2023", "2024", "2025", "2026(~9.17)"], book_rows("13차", V13, W4, YRS), 2))
P("")
P(f"• W_all 샤프·최악일: {sharpe_line('13차', V13)}.")
P("")
P("### 표 1b 13차 코스피·코스닥 장부(CAGR % / MDD %)")
P("")
rows1b = []
for v, c, lab in V13:
    for sub in ("코스피", "코스닥"):
        rows1b.append([lab, c, sub] + [cm(brow("13차", v, c, sub, w).cagr_pct, brow("13차", v, c, sub, w).mdd_pct) for w in W4])
P(table(["판", "비용", "장부", "W_all", "W_des", "W_val", "W_ex2026"], rows1b, 3))
P("")
P("## 3. 표 2 15차 v3 장부(합산, k 1.40)")
P("")
P("• 칸 = CAGR % / MDD %. 연도 칸 = 그 해 누적 수익 % / 그 해 MDD %. 중간 = 09:01 진입·15:30 청산(틀 근사, 15차 v3 중간 묶음).")
P("")
P(table(["판", "비용", "W_all", "W_des", "W_val", "W_ex2026", "W_long", "2023", "2024", "2025", "2026(~9.17)"], book_rows("15차 v3", V15, W4, YRS, long=True), 2))
P("")
P(f"• W_all 샤프·최악일: {sharpe_line('15차 v3', V15)}.")
P("")
P("## 4. 표 3 다리별 원형 대 실행판")
P("")
P(f"• n = 활성일 수, 순 bp = 활성일 순수익 평균, t = NW t(HAC 5), CAGR / MDD = 창 전체(비활성일 0). K_C·Q_C 의 n·순 bp 는 겹침(아침 활성일 0) 처리 전 다리 기준이고 장부 기여 일수는 W_all K_C {NC['K_C']}일·Q_C {NC['Q_C']}일.")
P("")
P("### 13차 장부 다리(13차 비용 → 실행판)")
P("")
P(table(["다리", "판·비용", "W_all n", "순 bp", "t", "CAGR / MDD", "W_val n", "순 bp", "t", "CAGR / MDD"], leg_rows("13차", LV13, LEGN13), 2))
P("")
P("### 15차 v3 장부 다리(k 1.40 배수 포함)")
P("")
P(table(["다리", "판·비용", "W_all n", "순 bp", "t", "CAGR / MDD", "W_val n", "순 bp", "t", "CAGR / MDD"], leg_rows("15차 v3", LV15, LEGN15), 2))
P("")
P("## 5. 표 4 폭포(합산 장부, CAGR % / MDD %, Δ 는 %p)")
P("")
P("• 13차: 원형 → 체결 변경(13차 비용 유지, 1~5단계) → 기본 비용 규약(6~7단계) → 하한. 15차 v3: 대표 → 중간 → 15:19 청산 → 기본 → 하한.")
P("")
P("### 13차")
P("")
P(table(["단계", "W_all", "ΔCAGR", "ΔMDD", "0단계 대비 ΔCAGR", "W_val", "ΔCAGR", "ΔMDD", "0단계 대비 ΔCAGR"], wf_rows("13차"), 1))
P("")
P(f"• 3단계({kt(wf13_3.label)})에는 x > 1 몫의 233740 혼합(원형은 229200 x 배)이 함께 들어 있음: {rv('13차 Q_B 실행판 차량 혼합', 'check').split(': ')[-1]} = {rv('13차 Q_B 실행판 차량 혼합')}.")
P("")
P("### 15차 v3(k 1.40)")
P("")
P(table(["단계", "W_all", "ΔCAGR", "ΔMDD", "0단계 대비 ΔCAGR", "W_val", "ΔCAGR", "ΔMDD", "0단계 대비 ΔCAGR"], wf_rows("15차 v3"), 1))
P("")
P("## 6. 표 5 민감도(기준 값 병기)")
P("")
P("• 기준 = 같은 창의 실행판 기본(13차 k 1.00 / 15차 v3 k 1.40). 비용은 변형 설명대로, 적지 않은 것은 기본. 헤드라인으로 쓰지 않는다. s4 는 사후 변형이라 보고만 한다.")
P("")
P(table(["장부", "항목", "변형", "창", "CAGR / MDD", "기준 CAGR / MDD", "ΔCAGR", "ΔMDD", "W_long CAGR / MDD"], sens_rows(), 4))
P("")
P("• s1 아침 청산을 15:30 단일가로 되돌리면 두 장부 모두 낮아진다(마지막 11분 평균 변화가 음수). s2 밤 ETF 청산을 09:01:00 연속 체결로 바꾸면 13차는 낮아지고 15차는 높아진다(첫 1분 상승을 코스피 밤 롱이 받음). s3 은 k 에 거의 비례. s4 는 봉인 W_val 에서 더 크게 낮아진다.")
P("")
P("## 7. 검증 결과")
P("")
P("### 출발점 재현 R1~R4(review.csv 독립 재실행)")
P("")
for sec in ("5_R1", "5_R2", "5_R3", "5_R4"):
    P(f"• {sec[2:]}")
    L.extend(rrows(sec))
P(f"• R3 차이 원인: 16차 격자는 09:01 봉 무거래를 다음 봉으로 밀고 특수일만 뺐으나 실행판 사양은 09:01 무거래면 쉼(코스피 C3 2026.6.8 1일, n 99 대 100). 틱 분모도 격자 원가격, 실행판 수정가(틱 0.05~0.13bp 차). 격자 규약으로 맞추면 순 bp 차 0.01bp 이내.")
P("")
P("### 독립 재계산(verify_r17_recompute.py, 사양·원자료만으로 다리를 다시 짬)")
P("")
P(f"• recompute_compare.csv {len(RC)}행(13차 실행판 13차 비용·기본·하한, 15차 v3 실행판 기본·하한 × W_all·W_des·W_val) 모두 match {bool(RC.match.all())}. 최대 |ΔCAGR| {rc_dc:.6f}%p, 최대 |ΔMDD| {rc_dm:.6f}%p.")
P(f"• 일수익 최대 차: 13차 실행판 합산 기본 {rv('13차 실행판 합산 기본: 독립 재계산')}, 하한 {rv('13차 실행판 합산 하한: 독립 재계산')}, 15차 v3 실행판 기본 다리 {rv('15차 v3 실행판 기본 다리(daily_15_exec)')}(저장 8자리 반올림 한도).")
P("")
rc_rows = [[r.book, r.cost, r.window, cm(r.main_cagr_pct, r.main_mdd_pct), cm(r.mine_cagr_pct, r.mine_mdd_pct), f"{r.d_cagr_pp:+.1e}".replace("-", MINUS), f"{r.d_mdd_pp:+.1e}".replace("-", MINUS)] for _, r in RC.iterrows()]
P(table(["장부", "비용", "창", "본 산출 CAGR / MDD", "재계산 CAGR / MDD", "ΔCAGR(%p)", "ΔMDD(%p)"], rc_rows, 3))
P("")
P("### 반박 검토(verify_r17_review.py, review.csv)")
P("")
P(f"• {len(RV)}행, 불일치 {n_fail}, minor {n_minor}, note {n_note}. 실행판을 부당하게 좋게 만든 주요 오류 없음. minor 4건:")
L.extend(minor_rows())
P(f"• 보수 쪽 항목: 아침 쉼 날 원형 net K_B 2026.6.8 / Q_B 2026.1.2 = {rv('아침 쉼 날 원형 net')}(각 장부 %, 둘 다 양수라 실행판에 불리). 비용 식(5bp = 현금×5e−4, 틱 = 현금×5원/평균 수정가, 하한 − 기본 = 0.5·(069500 1틱 + 코스닥 밤 ETF 1틱)) csv 반올림 안에서 일치. 기존 파일·깃 변경 없음.")
P("")
P("### 이 표의 재대조(r17_final_table.py)")
P("")
P(f"• books.csv 합산·코스피·코스닥 행 가운데 일수익이 저장된 {len(RECHECKED)}개 묶음(13차 원형·체결만 바꿈·기본·하한 합산, 13차 기본 코스피·코스닥, 15차 v3 대표·기본·하한 합산과 W_long)을 daily_13_exec·daily_15_exec·daily_15_exec_long 으로 다시 계산해 CAGR·MDD·누적·샤프·최악일·일수 대조. yearly.csv = books.csv 연도 행, waterfall.csv 끝점 = books.csv 와 Δ 식, sensitivity.csv 기준 열 = 실행판 기본과 s3 k 1.40 = books.csv, legs.csv 실행판·기본 K_A·K_B·Q_A·Q_B 의 n·순 bp·NW t·CAGR/MDD, 틱 표 = c3_exec_trades.")
P(f"• 결과 {N_ALL}건 중 {N_OK}건 일치" + ("." if ck_fail.empty else f", 불일치 {len(ck_fail)}건: " + "; ".join(ck_fail.check.head(10)) + "."))
P("• 15차 중간 묶음과 13차 원형·하한의 코스피·코스닥 장부는 일수익이 저장돼 있지 않아 csv 값 그대로 실었다(중간은 R2 k_picks_v3 대조로 확인).")
P("")
P("## 8. 단서")
P("")
P("### 상위 5일 의존(상위 5일 수익을 0 으로 두었을 때, review.csv)")
P("")
P(table(["장부", "판", "비용", "창", "CAGR → 상위 5일 0", "MDD → 상위 5일 0", "상위 5일"], top5_rows(), 4))
P("")
P("• 상위 5일은 두 장부에 공통(2026.7.31, 2026.2.3, 2026.5.21, 2026.3.24, 2023.11.6 등). 봉인 W_val 수익의 절반 이상이 5일에 실려 있어, 원형과 같은 단서(소수 극단일 의존)가 실행판에도 그대로 붙는다.")
P("")
P("### 거래량 0 처리와 쉼")
P("")
P(f"• 09:01 봉 무거래 쉼(W_all): K_B 2026.6.8, Q_B 2026.1.2 각 1일. 원형 net {rv('아침 쉼 날 원형 net')}(각 장부 %, 둘 다 양수라 쉼은 실행판에 불리).")
P("• 09:02 봉 무거래 진입 밀림 0건, 15:20 봉 무거래 청산 밀림 0건, 신호일 15:30 봉 무거래 0건(W_all).")
P(f"• 코스닥 밤 청산일 09:01 무거래로 뺀 밤 3건(신호 2023.11.15, 2024.12.30, 2025.11.12), 원형 net(코스닥 장부): {rv('뺀 밤 3건')}. 10:00 단일가(10:01 봉 시가) 청산 대안이면 13차 실행판 기본 W_all / W_val {rv('대안(뺀 밤을 청산일 첫 거래 봉 10:01 시가')}(본 산출 {rv('대안(뺀 밤을 청산일 첫 거래 봉 10:01 시가', 'expected').replace('r17 ', '')}).")
P("• 2026.6.8 은 069500·122630 의 09:01·09:02 봉이 전일 종가 수준 평탄·무거래이고 첫 체결이 09:03 봉(−9% 갭). 개장 지연 또는 자료 결손으로 보고 사양대로 쉼. 16차 격자는 이 날을 다음 봉으로 밀어 거래(R3 n 99 대 100).")
P(f"• 시가 결정 지연 추정일(늦은 개장 아닌데 09:01 봉 무거래)에 걸린 원 활성: {rv('시가 결정 지연 추정일에 걸린 원 활성')}. 코스피 밤 K_A 는 일봉 시가 그대로(그대로 둔 다리).")
P("")
P("### 근사·미확인")
P("")
P(f"• 15차 아침 초과(선물) 몫을 1배 ETF 체결 수익으로 근사: 명목 > 2 인 날 K_B {rv('15차 k1.40 K_B W_all 명목 > 2 일수')}, Q_B {rv('15차 k1.40 Q_B W_all 명목 > 2 일수')}(일수 / 초과 명목 평균 / 합산 장부 누적 기여 %). 그 날들의 ETF 시가 괴리(1배 ETF oc − 지수 oc) 평균 bp / 누적 % K_B {rv('15차 k1.40 K_B 초과 몫 날의 ETF 괴리')}, Q_B {rv('15차 k1.40 Q_B 초과 몫 날의 ETF 괴리')}. 괴리를 지수 수준으로 빼면 15차 실행판 기본 W_all / W_val {rv('15차 실행판 기본에서 초과(선물) 몫의 ETF 괴리')}(본 산출 {rv('15차 실행판 기본에서 초과(선물) 몫의 ETF 괴리', 'expected').replace('r17 ', '')}). 대표·중간도 같은 근사라 15차 안의 비교는 공정.")
P(f"• W_long 근사 e 는 W_all 활성일 99·84일 평균을 2022.11 전 229일에 외삽한 값. e = 0 이면 15차 실행판 기본 W_long {rv('15차 실행판 기본 W_long CAGR / MDD: 근사 e 적용')}. W_long 열만 영향.")
P(f"• 13차 실행판 Q_B 는 x > 1 몫을 233740 혼합(원형은 229200 x 배): {rv('13차 Q_B 실행판 차량 혼합', 'check').split(': ')[-1]} = {rv('13차 Q_B 실행판 차량 혼합')}. 폭포 3단계에 체결 시각 변경과 차량 혼합이 함께 들어 있다.")
P(f"• 코스닥 낮 숏 251340 1틱 = 왕복 {num(TICK['251340'].loc[2023:].min(), 1)}~{num(TICK['251340'].loc[2023:].max(), 1)}bp. 13차 실행판 기본 W_all CAGR 하락 중 가장 큰 몫이 팀 낮 다리 1틱 단계({num(wfl('13차', 7).W_all_d_cagr, sign=True)}%p, W_val {num(wfl('13차', 7).W_val_d_cagr, sign=True)}%p). 사양대로 두고 바꾸지 않았다. 15차 v3 는 같은 숏이 선물 대리라 이 몫이 없다.")
P(f"• 틱 분모 수정가 연평균은 원가격 기준보다 틱이 조금 크다(보수, W_all 아침 활성일 1배 ETF 1틱 누적 차 bp 069500 / 229200 {rv('틱 분모 수정가 vs 원가격')}). 원가격 틱이면 장부 차 0.04%p 이내(recompute.csv 대안 행).")
P("• 팀 장중 다리 K_C·Q_C 는 저장 일수익 + 1틱만 반영. 자체 체결 규약·1분봉 재생성은 확인하지 않았다.")
P("• 체결가는 봉 시가 1건 전량 체결 가정. 호가 깊이·부분 체결 미반영. 선물 1분봉이 없어 15차 초과 몫과 코스닥 밤 선물은 일봉 근사.")
P("• 모든 수치는 단일 백테스트 추정치. 규칙은 설계 구간에서 고른 것이고 봉인 결과는 소수 극단일에 의존한다.")
P("")

TEXT = "\n".join(L)

# ---------------------------------------------------------------- 작성 원칙 점검(하이픈·대시, 금지 표현, 핸들)
BAN = ["가 아니라", "다만", "오히려", "지만", "대신", "\u2013", "\u2014", "jehyeong", "beomgu"]
bad = []
for i, ln in enumerate(TEXT.splitlines(), 1):
    if re.fullmatch(r"\|(\s*:?-{3,}:?\s*\|)+", ln.strip()):
        continue
    if "-" in ln:
        bad.append((i, "하이픈", ln[:90]))
    bad += [(i, w, ln[:90]) for w in BAN if w in ln]
assert not bad, bad

(OUT / "final_tables.md").write_text(TEXT, encoding="utf-8")
CK.to_csv(OUT / "verify" / "final_table_checks.csv", index=False, encoding="utf-8-sig")
print(f"final_tables.md {len(TEXT.splitlines())}줄, 재대조 {N_OK}/{N_ALL} 일치, 서식 점검 통과")
if not ck_fail.empty:
    print(ck_fail.to_string())
