"""R5 closebet: 종가 단일가 매수 → 익일 시가 매도(오버나잇) 전략 5차 탐색

세 갈래(A·B는 선택 대상, C는 맥락 보고)
A. 팀원 종가베팅(미국 지수선물 ETF 장 막판 상대강도) 재검증 - 변형 수를 줄이고 기간을 나눈다.
   - 자료: Future_Trading 저장소 팀원 data/ 폴더의 219480(KODEX 미국S&P500선물(H))·304940(KODEX 미국나스닥100선물(H))
     1분봉 14:30~15:30. 봉 시각은 시작 시각 표기(15:19 봉 = 15:19:00~15:19:59, 15:20~15:29 = 종가 단일가 대기, 15:30 = 종가).
   - 창 수익 x = 창 끝 직전 봉 종가 / 창 시작 봉 시가 - 1  ([시작, 끝) 구간). 끝 = 15:20 이면 15:19 봉 종가까지만 쓴다
     → 15:20:00 에 알 수 있고 15:20~15:30 종가 단일가에 주문 가능.
   - 백분위 q = 직전 L개 유효 x 중 오늘 x 보다 작은 비율(팀원 정의와 같음, 오늘 제외). q ≥ 2/3 → 매수.
   - 팀원 원본 정의(확인용 1개, 선택 대상 아님): 창 양 끝 포함(14:40 봉 시가 → 14:50 봉 종가), 비용 3bp, 비수정 KODEX 일봉.
     팀원 96개 변형 중 1520_1530·1500_1530·1430_1530 창은 15:30 종가 단일가 가격을 신호에 써서 종가 매수와 같은 시점에 알 수 없다.
   - 격자 30개 = 대리 {SP, NQ, AVG(두 수익 평균)} × 창 {14:30-15:00, 14:40-15:10, 14:30-15:20, 15:00-15:20, 14:40-14:50} × L {20, 63}
B. KODEX 200 1분봉(2023-01~2026-09, 끝 시각 표기: 09:01 봉 = 09:00:00~09:00:59, 15:20 봉 = 15:19:00~15:19:59)으로 만든
   60/120분 신호 → 종가 단일가 매수·익일 시가 매도. 신호 가격 P(hh:mm) = 그 표기 봉의 종가. P(15:20) = 15:19:59 가격.
   규칙 8개 × 추세 필터 3개 {없음, P(15:20) > 직전 20일 종가 평균, P(15:20) > 직전 60일 종가 평균} = 24개
     ALWAYS : 매일(필터만 작동)
     L60UP  : P(15:20)/P(14:20) - 1 > 0          L120UP : P(15:20)/P(13:20) - 1 > 0
     L120PCT: 120분 수익이 직전 63일 백분위 ≥ 2/3   PMREV  : 오전(시가→13:20) < 0 그리고 120분 > 0 (오후 반전)
     L60DN  : 60분 수익 < 0 (막판 약세 되돌림)      DAYDN  : 시가→15:20 < 0
     CLVHI  : (P(15:20) - 당일 저가)/(고가 - 저가) ≥ 0.7 (15:20 봉까지의 고저)
   시가는 네이버 일봉 시가(09:00 단일가). 장 마감이 15:30 이 아닌 날(수능일 등)은 쉬는 날로 둔다.
C. 맥락: KODEX 200 무조건 오버나잇 연도별(2003~2026) 5bp·2bp(선물 대리).

공통 규약
- 수익: KODEX 200(069500) 네이버 수정주가 일봉, r_on(T) = 시가(T+1)/종가(T) - 1. 매수 체결 = T 종가 단일가, 매도 = T+1 시가 단일가.
- 비용(왕복): KODEX 5bp(스트레스 7bp). 미니 코스피200 선물 2bp(스트레스 3bp)는 KODEX 경로로 대리
  (선물은 15:45 마감·08:45 개장이고 2025-06-09 이후 야간장이 있어 실제 경로와 다르다).
- 기간(보기 전에 고정): A 설계 2021-12~2023-12(63일 준비 기간이 끝나는 날부터 모든 변형 공통 시작), 검증 2024-01~끝.
  B 설계 2023-01-02~2024-12-30, 검증 2025-01-02~2026-09-28. 검증에서 2025-06-09 이후를 뺀 값도 보고.
- 지표: 무포지션 날 0 포함 일별 수익, 복리, 252일 연환산. 샤프 = 평균/SD·√252. NW t(5시차) = 거래별 순수익.

선택 규칙(사전 고정, 설계 지표만 사용)
 1) 각 갈래(A, B)에서 설계 구간 거래 ≥ 30회/년 이고 설계 거래당 순수익(5bp) > 0 인 변형 중 설계 샤프 최대를 갈래 대표로 고른다.
    A의 팀원 원본 확인용 행은 선택 대상이 아니다.
 2) 계열 최종 = 두 갈래 대표 중 설계 샤프가 더 큰 것(같으면 A).
 3) 갈래별 설계 샤프 2·3위 변형의 검증 값도 보고한다(검증 값으로 고르지 않는다).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import data  # noqa: E402
import engine as E  # noqa: E402

FAM = "closebet"
TOPIC = "R5_CLOSEBET"
OUT = data.OUT / "round5_strategy" / FAM
FT = data._dir("HYFE_TEAM_REPO", "team_repo_dir", "../..")   # 팀 저장소 루트: config team_repo_dir, 저장소 안에서는 ../..
ANN = 252
COST = 5.0
COSTS = {"KODEX 5bp": 5.0, "KODEX 스트레스 7bp": 7.0, "미니선물 대리 2bp": 2.0, "미니선물 스트레스 3bp": 3.0}
RECENT = pd.Timestamp("2025-06-09")
A_DES_END, A_VAL_START = pd.Timestamp("2023-12-31"), pd.Timestamp("2024-01-01")
B_DES = (pd.Timestamp("2023-01-02"), pd.Timestamp("2024-12-30"))
B_VAL_START = pd.Timestamp("2025-01-02")
MIN_TPY = 30
HI = 2 / 3


# ---------------------------------------------------------------- 자료
def find_one(pattern):
    hits = sorted(FT.glob(pattern))
    if not hits:
        raise FileNotFoundError(pattern)
    return hits


def load_us_etf_1m(code):
    fs = find_one(f"*/data/{code}_1*_1m.csv")
    df = pd.concat([pd.read_csv(f) for f in fs])
    df["ts"] = pd.to_datetime(df["timestamp"].str[:19])
    df = df.drop_duplicates("ts")
    df["d"], df["hm"] = df["ts"].dt.normalize(), df["ts"].dt.strftime("%H:%M")
    return df.pivot(index="d", columns="hm", values="open"), df.pivot(index="d", columns="hm", values="close")


def load_kodex_1m():
    f = find_one("*/data/KODEX200_1min_2023_2026.csv")[0]
    k = pd.read_csv(f, encoding="utf-8-sig")
    k["ts"] = pd.to_datetime(k["datetime"].str[:19])
    k["d"], k["hm"] = k["ts"].dt.normalize(), k["ts"].dt.strftime("%H:%M")
    last = k.groupby("d")["hm"].max()
    k = k[(k["hm"] >= "09:01") & (k["hm"] <= "15:20")]           # 신호는 15:20 봉(15:19:59)까지만
    C = k.pivot(index="d", columns="hm", values="close")
    hi = k.groupby("d")["high"].max()
    lo = k.groupby("d")["low"].min()
    return C, hi, lo, last


def minus1(hm):
    t = pd.Timestamp("2000-01-01 " + hm) - pd.Timedelta(minutes=1)
    return t.strftime("%H:%M")


def pct_past(x, L):
    """오늘 x 가 직전 L개 유효값 중 몇 %보다 큰가(오늘 제외, 엄격 부등호)."""
    v = x.to_numpy(float)
    out = np.full(len(v), np.nan)
    hist = []
    for i, xi in enumerate(v):
        if np.isnan(xi):
            continue
        if len(hist) >= L:
            past = np.array(hist[-L:])
            out[i] = (past < xi).mean()
        hist.append(xi)
    return pd.Series(out, index=x.index)


# ---------------------------------------------------------------- 지표
def metrics(pos, on, cc, a, b, cost=COST):
    """pos: 결정일 T 포지션(0/1, NaN=0), on: r_on, cc: 종가-종가(벤치마크). 구간 [a, b] 의 결정일."""
    idx = on.loc[a:b].dropna().index
    if len(idx) == 0:
        return {}
    p = pos.reindex(idx).fillna(0)
    r = on.reindex(idx)
    daily = p * r - (p != 0) * cost / 1e4
    tr = daily[p != 0]
    n = len(idx)
    eq = (1 + daily).cumprod()
    cagr = eq.iloc[-1] ** (ANN / n) - 1
    mdd = (eq / eq.cummax() - 1).min()
    sd = daily.std()
    bh = cc.reindex(idx).fillna(0)
    beq = (1 + bh).cumprod()
    return {"start": idx.min().date().isoformat(), "end": idx.max().date().isoformat(), "days": n,
            "trades": int(len(tr)), "tpy": len(tr) / (n / ANN), "exposure": len(tr) / n,
            "win": (tr > 0).mean() if len(tr) else np.nan, "gross_bp": (p * r)[p != 0].mean() * 1e4 if len(tr) else np.nan,
            "net_bp": tr.mean() * 1e4 if len(tr) else np.nan, "t_net": E.nw_t(tr.to_numpy()) if len(tr) >= 10 else np.nan,
            "cagr": cagr, "sharpe": daily.mean() / sd * np.sqrt(ANN) if sd > 0 else np.nan, "mdd": mdd,
            "calmar": cagr / -mdd if mdd < 0 else np.nan,
            "bh_cagr": beq.iloc[-1] ** (ANN / n) - 1, "bh_mdd": (beq / beq.cummax() - 1).min()}


def daily_net(pos, on, idx, cost=COST):
    p = pos.reindex(idx).fillna(0)
    return p * on.reindex(idx) - (p != 0) * cost / 1e4


def delta(pos, on, a, b):
    idx = on.loc[a:b].dropna().index
    s = pos.reindex(idx)
    d, t, n = E.delta_test(s.where(s.notna()), on.reindex(idx))
    return d * 1e4, t


# ---------------------------------------------------------------- 신호
def build_A(idx_k):
    """A 갈래 변형 30개 + 팀원 원본 확인용 1개. 반환: {vid: (meta, pos)}"""
    O, C = {}, {}
    for nm, code in (("SP", "219480"), ("NQ", "304940")):
        O[nm], C[nm] = load_us_etf_1m(code)
    wins = [("14:30", "15:00"), ("14:40", "15:10"), ("14:30", "15:20"), ("15:00", "15:20"), ("14:40", "14:50")]
    x = {}
    for nm in ("SP", "NQ"):
        for a, b in wins:
            x[(nm, a, b)] = C[nm][minus1(b)] / O[nm][a] - 1
    for a, b in wins:
        x[("AVG", a, b)] = (x[("SP", a, b)] + x[("NQ", a, b)]) / 2
    out = {}
    i = 0
    for prox in ("SP", "NQ", "AVG"):
        for a, b in wins:
            for L in (20, 63):
                i += 1
                q = pct_past(x[(prox, a, b)], L)
                pos = (q >= HI - 1e-9).astype(float).where(q.notna())
                meta = dict(part="A", vid=f"A{i:02d}", rule=f"{prox} {a}-{b} ROLL{L} q>=2/3", proxy=prox, window=f"{a}-{b}",
                            lookback=L, filter="-", selectable=True)
                out[meta["vid"]] = (meta, pos.reindex(idx_k))
    # 팀원 원본 정의(양 끝 포함) 확인용
    xt = C["SP"]["14:50"] / O["SP"]["14:40"] - 1
    q = pct_past(xt, 63)
    pos = (q >= HI - 1e-9).astype(float).where(q.notna())
    meta = dict(part="A", vid="A31", rule="팀원 원본 SP 14:40-14:50(양끝 포함) ROLL63 q>=2/3", proxy="SP", window="14:40-14:50 incl",
                lookback=63, filter="-", selectable=False)
    out["A31"] = (meta, pos.reindex(idx_k))
    return out, x


def build_B(px):
    C, hi, lo, last = load_kodex_1m()
    ok = (last == "15:30") & C[["13:20", "14:20", "15:20"]].notna().all(axis=1)
    p1520, p1420, p1320 = C["15:20"], C["14:20"], C["13:20"]
    op = px["open"].reindex(C.index)
    r60, r120, ram, rday = p1520 / p1420 - 1, p1520 / p1320 - 1, p1320 / op - 1, p1520 / op - 1
    clv = (p1520 - lo) / (hi - lo)
    q120 = pct_past(r120.where(ok), 63)
    rules = {
        "ALWAYS": pd.Series(1.0, index=C.index),
        "L60UP": (r60 > 0).astype(float), "L120UP": (r120 > 0).astype(float),
        "L120PCT": (q120 >= HI - 1e-9).astype(float).where(q120.notna()),
        "PMREV": ((ram < 0) & (r120 > 0)).astype(float), "L60DN": (r60 < 0).astype(float),
        "DAYDN": (rday < 0).astype(float), "CLVHI": (clv >= 0.7).astype(float).where((hi - lo) > 0, 0.0),
    }
    close = px["close"]
    ma = {n: close.shift(1).rolling(n).mean().reindex(C.index) for n in (20, 60)}
    filt = {"-": pd.Series(1.0, index=C.index), "MA20": (p1520 > ma[20]).astype(float), "MA60": (p1520 > ma[60]).astype(float)}
    out = {}
    i = 0
    for rn, s in rules.items():
        for fn, f in filt.items():
            i += 1
            pos = (s * f).where(ok, 0.0)
            meta = dict(part="B", vid=f"B{i:02d}", rule=f"{rn}" + ("" if fn == "-" else f" & P1519>{fn}"), proxy="KODEX1m", window=rn,
                        lookback=np.nan, filter=fn, selectable=True)
            out[meta["vid"]] = (meta, pos)
    feats = pd.DataFrame({"r60": r60, "r120": r120, "ram": ram, "rday": rday, "clv": clv, "ok": ok})
    return out, feats


# ---------------------------------------------------------------- 표 서식
def pct(v, d=1):
    return "-" if pd.isna(v) else f"{v * 100:.{d}f}%"


def num(v, f="{:+.2f}"):
    return "-" if pd.isna(v) else f.format(v)


def mrow(name, m):
    return (f"| {name} | {m.get('start', '-')}~{m.get('end', '-')} | {m.get('trades', 0)} | {num(m.get('tpy'), '{:.0f}')} | "
            f"{pct(m.get('exposure'), 0)} | {pct(m.get('win'))} | {num(m.get('net_bp'))} | {num(m.get('t_net'))} | {pct(m.get('cagr'))} | "
            f"{num(m.get('sharpe'))} | {pct(m.get('mdd'))} | {num(m.get('calmar'))} | {pct(m.get('bh_cagr'))} / {pct(m.get('bh_mdd'))} |")


MHEAD = ("| 구간 | 기간 | 거래 | 거래/년 | 노출 | 승률 | 거래당 순bp | NW t | CAGR | 샤프 | MDD | 칼마 | KODEX 보유 CAGR / MDD |\n"
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|")


def pick(grid, part):
    g = grid[(grid["part"] == part) & grid["selectable"] & (grid["d_tpy"] >= MIN_TPY) & (grid["d_net_bp"] > 0)]
    return g.sort_values(["d_sharpe", "vid"], ascending=[False, True])


# ---------------------------------------------------------------- 본체
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    px = data.price("069500")
    on = (px["open"].shift(-1) / px["close"] - 1).rename("on")
    cc = (px["close"] / px["close"].shift(1) - 1).rename("cc")

    # ---- A
    O_sp, _ = load_us_etf_1m("219480")
    idx_a = on.reindex(O_sp.index).dropna().index
    A, xA = build_A(idx_a)
    a_start = max(p.first_valid_index() for m, p in A.values() if m["selectable"])
    A_DES, A_VAL = (a_start, A_DES_END), (A_VAL_START, idx_a.max())
    # ---- B
    B, feats = build_B(px)
    idx_b = on.reindex(feats.index).dropna().index
    B = {k: (m, p.reindex(idx_b)) for k, (m, p) in B.items()}
    B_VAL = (B_VAL_START, idx_b.max())

    periods = {"A": {"design": A_DES, "val": A_VAL}, "B": {"design": B_DES, "val": B_VAL}}
    rows, ledger_rows = [], []
    for part, VAR in (("A", A), ("B", B)):
        P = periods[part]
        for vid, (meta, pos) in VAR.items():
            d = metrics(pos, on, cc, *P["design"])
            v = metrics(pos, on, cc, *P["val"])
            vx = metrics(pos, on, cc, P["val"][0], RECENT - pd.Timedelta(days=1))
            dd, dt_ = delta(pos, on, *P["design"])
            vd, vt = delta(pos, on, *P["val"])
            row = {**meta, **{f"d_{k}": val for k, val in d.items()}, "d_delta_bp": dd, "d_t_delta": dt_,
                   **{f"v_{k}": val for k, val in v.items()}, "v_delta_bp": vd, "v_t_delta": vt,
                   **{f"vx_{k}": val for k, val in vx.items()}}
            rows.append(row)
            for wn, m in (("design", d), ("validation", v), ("validation_ex_2025-06-09", vx)):
                ledger_rows.append({"asset": "069500", "days": m["days"], "trades": m["trades"], "exposure": m["exposure"],
                                    "gross_bp": m["gross_bp"], "net_bp": m["net_bp"], "t_net": m["t_net"], "win": m["win"],
                                    "sharpe": m["sharpe"], "cagr": m["cagr"], "mdd": m["mdd"], "calmar": m["calmar"],
                                    "rule": f"{vid} {meta['rule']}", "window": f"{part}-{wn} {m['start']}~{m['end']}", "cost_bp": COST,
                                    "delta_bp": dd if wn == "design" else vd, "t_delta": dt_ if wn == "design" else vt})
    grid = pd.DataFrame(rows)

    # ---- 선택(설계 지표만)
    selA, selB = pick(grid, "A"), pick(grid, "B")
    repA, repB = selA.iloc[0], selB.iloc[0]
    final = repA if repA["d_sharpe"] >= repB["d_sharpe"] else repB
    grid["d_rank_in_part"] = grid.groupby("part")["d_sharpe"].rank(ascending=False, method="first")
    grid["chosen_part_rep"] = grid["vid"].isin([repA["vid"], repB["vid"]])
    grid["chosen_final"] = grid["vid"] == final["vid"]
    grid.to_csv(OUT / "grid.csv", index=False, encoding="utf-8-sig")
    rank_corr = {p: grid[(grid.part == p) & grid.selectable][["d_sharpe", "v_sharpe"]].corr(method="spearman").iloc[0, 1] for p in ("A", "B")}

    VAR_ALL = {**A, **B}
    fpart = final["part"]
    fpos = VAR_ALL[final["vid"]][1]
    P = periods[fpart]
    idx_f = idx_a if fpart == "A" else idx_b

    # ---- 최종 사양 상세
    always = pd.Series(1.0, index=idx_f)
    blocks = {}
    for cname, cst in COSTS.items():
        blocks[cname] = {wn: metrics(fpos, on, cc, *w, cost=cst) for wn, w in
                         (("설계", P["design"]), ("검증", P["val"]), ("검증(2025-06-09 이전)", (P["val"][0], RECENT - pd.Timedelta(days=1))),
                          ("검증(2025-06-09 이후)", (RECENT, P["val"][1])))}
    alw = {wn: metrics(always, on, cc, *w) for wn, w in (("설계", P["design"]), ("검증", P["val"]),
                                                         ("검증(2025-06-09 이전)", (P["val"][0], RECENT - pd.Timedelta(days=1))))}
    yr = {}
    for y in sorted(set(idx_f.year)):
        a, b = max(pd.Timestamp(f"{y}-01-01"), P["design"][0]), pd.Timestamp(f"{y}-12-31")
        yr[y] = (metrics(fpos, on, cc, a, b), metrics(always, on, cc, a, b), metrics(fpos, on, cc, a, b, cost=2.0))

    # 최근 랠리 의존도: 전체 표본 순수익 합 중 2025-06-09 이후 비중
    dn = daily_net(fpos, on, idx_f[idx_f >= P["design"][0]])
    share_recent = dn[dn.index >= RECENT].sum() / dn.sum() if dn.sum() != 0 else np.nan
    dn_v = dn[dn.index >= P["val"][0]]
    share_recent_val = dn_v[dn_v.index >= RECENT].sum() / dn_v.sum() if dn_v.sum() != 0 else np.nan

    # ---- C: 무조건 오버나잇 연도별
    allidx = on.loc["2003-01-01":].dropna().index
    one = pd.Series(1.0, index=allidx)
    crow = []
    for y in sorted(set(allidx.year)):
        a, b = pd.Timestamp(f"{y}-01-01"), pd.Timestamp(f"{y}-12-31")
        m5, m2 = metrics(one, on, cc, a, b, 5.0), metrics(one, on, cc, a, b, 2.0)
        crow.append({"year": y, "days": m5["days"], "gross_bp": m5["gross_bp"], "win_gross": (on.loc[a:b].dropna() > 0).mean(),
                     "net5_bp": m5["net_bp"], "cagr5": m5["cagr"], "sharpe5": m5["sharpe"], "mdd5": m5["mdd"], "win5": m5["win"],
                     "net2_bp": m2["net_bp"], "cagr2": m2["cagr"], "sharpe2": m2["sharpe"], "win2": m2["win"], "bh_cagr": m5["bh_cagr"]})
    ctab = pd.DataFrame(crow)
    csub = {}
    for nm, (a, b) in {"2003-2009": ("2003-01-01", "2009-12-31"), "2010-2019": ("2010-01-01", "2019-12-31"),
                       "2020-01~2025-06-05": ("2020-01-01", "2025-06-05"), "2025-06-09~": ("2025-06-09", None),
                       "2010-2026 전체": ("2010-01-01", None)}.items():
        csub[nm] = (metrics(one, on, cc, a, b, 5.0), metrics(one, on, cc, a, b, 2.0))
    ctab.to_csv(OUT / "unconditional_overnight_by_year.csv", index=False, encoding="utf-8-sig")

    # ---- 산출물: 일별 수익·거래
    idx_full = idx_f[idx_f >= P["design"][0]]
    dr = pd.DataFrame({"period": np.where(idx_full <= P["design"][1], "design", "validation"),
                       "pos": fpos.reindex(idx_full).fillna(0), "r_on": on.reindex(idx_full),
                       "net_5bp": daily_net(fpos, on, idx_full, 5.0), "net_7bp": daily_net(fpos, on, idx_full, 7.0),
                       "net_2bp_fut_proxy": daily_net(fpos, on, idx_full, 2.0),
                       "always_on_net_5bp": on.reindex(idx_full) - 5e-4, "kodex_bh_cc": cc.reindex(idx_full)}, index=idx_full)
    for k in (2, 3):
        if len((selA if fpart == "A" else selB)) >= k:
            alt = (selA if fpart == "A" else selB).iloc[k - 1]
            dr[f"net_5bp_rank{k}_{alt['vid']}"] = daily_net(VAR_ALL[alt["vid"]][1], on, idx_full)
    dr.index.name = "date"
    dr.to_csv(OUT / "daily_returns.csv", encoding="utf-8-sig")
    t = dr[dr["pos"] != 0]
    trades = pd.DataFrame({"decision_date": t.index.date, "period": t["period"].values,
                           "entry": "T 15:30 종가 단일가", "entry_px": px["close"].reindex(t.index).values,
                           "exit_date": [px.index[px.index.get_loc(d) + 1].date() for d in t.index],
                           "exit": "T+1 09:00 시가 단일가", "exit_px": px["open"].shift(-1).reindex(t.index).values,
                           "gross_bp": t["r_on"].values * 1e4, "net_5bp": t["net_5bp"].values * 1e4, "net_2bp": t["net_2bp_fut_proxy"].values * 1e4})
    trades.to_csv(OUT / "trades.csv", index=False, encoding="utf-8-sig")

    # ---- 그림
    try:
        plot_equity(dr, P, final, OUT / "equity.png")
    except Exception as e:  # 그림 실패는 결과에 영향 없음
        print("plot fail", e)

    # ---- 장부
    spec = {"family": FAM, "selection": "설계 거래≥30/년·설계 순bp>0 중 설계 샤프 최대, 갈래 대표 중 설계 샤프 큰 것",
            "A_design": [str(A_DES[0].date()), str(A_DES[1].date())], "A_val": [str(A_VAL[0].date()), str(A_VAL[1].date())],
            "B_design": [str(B_DES[0].date()), str(B_DES[1].date())], "B_val": [str(B_VAL[0].date()), str(B_VAL[1].date())],
            "n_variants": int(len(grid)), "final": final["vid"], "final_rule": final["rule"], "cost_bp": COST}
    E.ledger_append(data.OUT, TOPIC, spec, pd.DataFrame(ledger_rows), note="5차 전략 탐색")

    # ---- 보고서
    res = dict(grid=grid, selA=selA, selB=selB, repA=repA, repB=repB, final=final, blocks=blocks, alw=alw, yr=yr, P=P, periods=periods,
               share_recent=share_recent, share_recent_val=share_recent_val, ctab=ctab, csub=csub, rank_corr=rank_corr,
               xA=xA, A=A, B=B, on=on, cc=cc, idx_a=idx_a, idx_b=idx_b)
    write_report(res)
    summ = {"final": final["vid"], "rule": final["rule"], "part": fpart,
            "design": blocks["KODEX 5bp"]["설계"], "val": blocks["KODEX 5bp"]["검증"], "val_ex": blocks["KODEX 5bp"]["검증(2025-06-09 이전)"],
            "val_recent": blocks["KODEX 5bp"]["검증(2025-06-09 이후)"], "val_7bp": blocks["KODEX 스트레스 7bp"]["검증"],
            "val_2bp": blocks["미니선물 대리 2bp"]["검증"], "share_recent": share_recent, "share_recent_val": share_recent_val,
            "rank_corr": rank_corr, "repA": repA["vid"], "repB": repB["vid"], "n_variants": len(grid)}
    (OUT / "summary.json").write_text(json.dumps(summ, ensure_ascii=False, indent=1, default=lambda o: None if pd.isna(o) else float(o)),
                                      encoding="utf-8")
    print(json.dumps(summ, ensure_ascii=False, indent=1, default=lambda o: None if pd.isna(o) else float(o)))


def plot_equity(dr, P, final, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(10, 5.2), dpi=130)
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    series = [("전략(5bp)", (1 + dr["net_5bp"]).cumprod(), "#2a78d6"),
              ("매일 오버나잇(5bp)", (1 + dr["always_on_net_5bp"]).cumprod(), "#eb6834"),
              ("KODEX 200 보유", (1 + dr["kodex_bh_cc"].fillna(0)).cumprod(), "#1baf7a")]
    for lab, s, col in series:
        ax.plot(s.index, s.values, color=col, lw=2, label=lab)
        ax.annotate(lab, (s.index[-1], s.values[-1]), xytext=(4, 0), textcoords="offset points", fontsize=8, color="#52514e", va="center")
    ax.set_yscale("log")
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
    lo_, hi_ = ax.get_ylim()
    ax.yaxis.set_major_locator(FixedLocator([v for v in (0.8, 0.9, 1, 1.25, 1.5, 2, 3, 4, 5, 6) if lo_ <= v <= hi_]))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    for x, lab in ((P["val"][0], "검증 시작"), (RECENT, "2025-06-09")):
        ax.axvline(x, color="#a3a29d", lw=1, ls="--")
        ax.text(x, ax.get_ylim()[1], " " + lab, fontsize=8, color="#52514e", va="top")
    ax.grid(axis="y", color="#e6e5e0", lw=0.8)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color("#c3c2b7")
    ax.tick_params(colors="#52514e", labelsize=8)
    ax.set_title(f"{final['vid']} {final['rule']} — 누적 성장(로그 축, 1 = 시작)", fontsize=10, color="#0b0b0b", loc="left")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, facecolor=fig.get_facecolor())
    plt.close(fig)


def write_report(R):
    g, final, P = R["grid"], R["final"], R["P"]
    L = []
    w = L.append
    w("# 5차 전략 탐색 — closebet(종가 단일가 매수 → 익일 시가 매도)\n")
    w("작성: 2026-10-05. 코드 `code/overnight_lab/topics/r5_closebet.py`. 수익은 KODEX 200(069500) 수정주가 일봉 `r_on = 시가(T+1)/종가(T) − 1`, "
      "매수 = T 15:30 종가 단일가, 매도 = T+1 09:00 시가 단일가, 비용 왕복 5bp(스트레스 7bp). 미니선물 2bp(3bp)는 KODEX 경로로 대리한 값이다"
      "(선물은 15:45 마감·08:45 개장, 2025-06-09 이후 야간장 → 실제 경로와 다름).\n")
    fb, fv, fx = R["blocks"]["KODEX 5bp"]["설계"], R["blocks"]["KODEX 5bp"]["검증"], R["blocks"]["KODEX 5bp"]["검증(2025-06-09 이전)"]
    fr = R["blocks"]["KODEX 5bp"]["검증(2025-06-09 이후)"]
    hit = (fv["cagr"] >= 0.095) or (fv["win"] >= 0.51)
    w("## 한눈에\n")
    w(f"- 최종 사양(사전 규칙으로 설계 구간에서만 선택): **{final['vid']} {final['rule']}** (갈래 {final['part']}).")
    w(f"- 설계 {fb['start']}~{fb['end']}: 거래 {fb['trades']}회, 승률 {pct(fb['win'])}, 거래당 순 {num(fb['net_bp'])}bp, CAGR {pct(fb['cagr'])}, 샤프 {num(fb['sharpe'])}, MDD {pct(fb['mdd'])}.")
    w(f"- 검증 {fv['start']}~{fv['end']}: 거래 {fv['trades']}회({num(fv['tpy'], '{:.0f}')}회/년), 승률 {pct(fv['win'])}, 거래당 순 {num(fv['net_bp'])}bp (NW t {num(fv['t_net'])}), "
      f"CAGR {pct(fv['cagr'])}, 샤프 {num(fv['sharpe'])}, MDD {pct(fv['mdd'])}.")
    w(f"- 검증에서 2025-06-09 이후를 빼면: 승률 {pct(fx['win'])}, 거래당 순 {num(fx['net_bp'])}bp, CAGR {pct(fx['cagr'])}, 샤프 {num(fx['sharpe'])}. "
      f"2025-06-09 이후만: 승률 {pct(fr['win'])}, 거래당 순 {num(fr['net_bp'])}bp, CAGR {pct(fr['cagr'])}.")
    w(f"- 목표(검증 5bp 기준 CAGR ≈10% 또는 승률 ≥ 51%) 충족: **{'예' if hit else '아니오'}**. "
      f"전체 표본 순수익 합 중 2025-06-09 이후 비중 {pct(R['share_recent'], 0)}, 검증 구간 안에서는 {pct(R['share_recent_val'], 0)}.")
    w(f"- 과적합 점검: 변형별 설계 샤프와 검증 샤프의 순위상관(Spearman) A {num(R['rank_corr']['A'])}, B {num(R['rank_corr']['B'])}.\n")

    w("## 선택 규칙(실행 전에 코드 머리말에 고정)\n")
    w(f"1. 갈래 A·B 각각에서 설계 구간 거래 ≥ {MIN_TPY}회/년, 설계 거래당 순수익(5bp) > 0 인 변형 중 설계 샤프 최대를 대표로 고른다(A31 팀원 원본 확인용 행 제외).")
    w("2. 계열 최종 = 두 대표 중 설계 샤프가 큰 쪽(같으면 A).")
    w("3. 설계 샤프 2·3위 변형의 검증 값을 함께 본다(검증 값으로 고르지 않는다).\n")
    w(f"- A 대표: {R['repA']['vid']} {R['repA']['rule']} (설계 샤프 {num(R['repA']['d_sharpe'])}) / B 대표: {R['repB']['vid']} {R['repB']['rule']} (설계 샤프 {num(R['repB']['d_sharpe'])})\n")

    w("## 최종 사양 성과\n")
    for cname, bl in R["blocks"].items():
        w(f"### 비용 {cname}\n")
        w(MHEAD)
        for wn, m in bl.items():
            w(mrow(wn, m))
        w("")
    w("### 비교: 같은 기간 매일 오버나잇(무조건, 5bp)\n")
    w(MHEAD)
    for wn, m in R["alw"].items():
        w(mrow(wn, m))
    w("")
    w(f"신호 밤 − 비신호 밤 평균 차(Δ, 비용 전, HAC t): 설계 {num(final['d_delta_bp'])}bp (t {num(final['d_t_delta'])}), 검증 {num(final['v_delta_bp'])}bp (t {num(final['v_t_delta'])}).\n")

    w("### 연도별(최종 사양, 5bp / 같은 해 매일 오버나잇 5bp / 선물 대리 2bp)\n")
    w("| 연도 | 거래 | 승률 | 거래당 순bp | CAGR | 샤프 | MDD | 매일 오버나잇 CAGR | 선물 2bp CAGR | KODEX 보유 CAGR |\n|---|---|---|---|---|---|---|---|---|---|")
    for y, (m, a, f2) in R["yr"].items():
        if not m:
            continue
        w(f"| {y} | {m['trades']} | {pct(m['win'])} | {num(m['net_bp'])} | {pct(m['cagr'])} | {num(m['sharpe'])} | {pct(m['mdd'])} | {pct(a['cagr'])} | {pct(f2['cagr'])} | {pct(m['bh_cagr'])} |")
    w("\n(연도 행은 해당 연도 안의 거래일만으로 연환산. 첫 해는 설계 시작일부터.)\n")

    for part, sel in (("A", R["selA"]), ("B", R["selB"])):
        w(f"## 갈래 {part}: 설계 샤프 상위 3개의 검증(강건성 확인, 선택에는 쓰지 않음)\n")
        w("| 순위 | 변형 | 규칙 | 설계 거래/년 | 설계 승률 | 설계 순bp | 설계 CAGR | 설계 샤프 | 검증 거래 | 검증 승률 | 검증 순bp | 검증 NW t | 검증 CAGR | 검증 샤프 | 검증 MDD | 검증(25-06-09 이전) CAGR | 같은 이전 승률 |")
        w("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for k, (_, r) in enumerate(sel.head(3).iterrows(), 1):
            w(f"| {k} | {r['vid']} | {r['rule']} | {r['d_tpy']:.0f} | {pct(r['d_win'])} | {num(r['d_net_bp'])} | {pct(r['d_cagr'])} | {num(r['d_sharpe'])} | "
              f"{r['v_trades']} | {pct(r['v_win'])} | {num(r['v_net_bp'])} | {num(r['v_t_net'])} | {pct(r['v_cagr'])} | {num(r['v_sharpe'])} | {pct(r['v_mdd'])} | "
              f"{pct(r['vx_cagr'])} | {pct(r['vx_win'])} |")
        w("")

    w("## 전체 격자(설계 지표로 정렬, 검증은 참고)\n")
    for part in ("A", "B"):
        gp = g[g.part == part].sort_values("d_sharpe", ascending=False)
        per = R["periods"][part]
        w(f"### 갈래 {part} — 설계 {per['design'][0].date()}~{per['design'][1].date()}, 검증 {per['val'][0].date()}~{per['val'][1].date()}\n")
        w("| 변형 | 규칙 | 선택대상 | 설계 거래/년 | 설계 승률 | 설계 순bp | 설계 CAGR | 설계 샤프 | 설계 Δbp(t) | 검증 거래/년 | 검증 승률 | 검증 순bp | 검증 CAGR | 검증 샤프 | 검증 Δbp(t) | 검증 이전 CAGR |")
        w("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for _, r in gp.iterrows():
            w(f"| {r['vid']} | {r['rule']} | {'O' if r['selectable'] else '확인용'} | {r['d_tpy']:.0f} | {pct(r['d_win'])} | {num(r['d_net_bp'])} | {pct(r['d_cagr'])} | "
              f"{num(r['d_sharpe'])} | {num(r['d_delta_bp'], '{:+.1f}')}({num(r['d_t_delta'], '{:+.1f}')}) | {r['v_tpy']:.0f} | {pct(r['v_win'])} | {num(r['v_net_bp'])} | "
              f"{pct(r['v_cagr'])} | {num(r['v_sharpe'])} | {num(r['v_delta_bp'], '{:+.1f}')}({num(r['v_t_delta'], '{:+.1f}')}) | {pct(r['vx_cagr'])} |")
        w("")

    w("## 맥락: KODEX 200 무조건 오버나잇(매일 종가 매수 → 익일 시가 매도)\n")
    w("| 구간 | 거래일 | 비용 전 평균bp | 5bp 승률 | 5bp 순bp | 5bp CAGR | 5bp 샤프 | 5bp MDD | 2bp 승률 | 2bp 순bp | 2bp CAGR | 2bp 샤프 | KODEX 보유 CAGR |")
    w("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for nm, (m5, m2) in R["csub"].items():
        w(f"| {nm} | {m5['days']} | {num(m5['gross_bp'])} | {pct(m5['win'])} | {num(m5['net_bp'])} | {pct(m5['cagr'])} | {num(m5['sharpe'])} | {pct(m5['mdd'])} | "
          f"{pct(m2['win'])} | {num(m2['net_bp'])} | {pct(m2['cagr'])} | {num(m2['sharpe'])} | {pct(m5['bh_cagr'])} |")
    w("")
    w("| 연도 | 거래일 | 비용 전 평균bp | 비용 전 상승 비율 | 5bp 순bp | 5bp CAGR | 5bp 샤프 | 2bp 순bp | 2bp CAGR | 2bp 샤프 | KODEX 보유 CAGR |")
    w("|---|---|---|---|---|---|---|---|---|---|---|")
    for _, r in R["ctab"].iterrows():
        w(f"| {int(r['year'])} | {int(r['days'])} | {num(r['gross_bp'])} | {pct(r['win_gross'])} | {num(r['net5_bp'])} | {pct(r['cagr5'])} | {num(r['sharpe5'])} | "
          f"{num(r['net2_bp'])} | {pct(r['cagr2'])} | {num(r['sharpe2'])} | {pct(r['bh_cagr'])} |")
    w("")
    w("## 팀원 원본과의 차이(재검증 메모)\n")
    a31 = g[g.vid == "A31"].iloc[0]
    w(f"- 팀원 최상위 `SIG_SP_ROLL63_10m_1440_1450_High`(전체 표본·3bp·비수정 일봉: CAGR 12.3%, 칼마 1.33)를 같은 정의(양 끝 포함)로 다시 만든 A31: "
      f"설계 CAGR {pct(a31['d_cagr'])}·샤프 {num(a31['d_sharpe'])}·승률 {pct(a31['d_win'])}, 검증 CAGR {pct(a31['v_cagr'])}·샤프 {num(a31['v_sharpe'])}·승률 {pct(a31['v_win'])} (5bp, 수정주가).")
    w("- 팀원 백분위는 오늘을 뺀 직전 63(20)일 중 오늘보다 작은 비율이고 기준 2/3 이다(진단 파일로 확인). 미래 정보 없음.")
    w("- 팀원 96개 변형 중 `1520_1530`, `30m_1500_1530`, `60m_1430_1530` 창은 15:30 종가 단일가 체결가를 신호에 쓰므로 종가 매수 시점에 알 수 없다(실행 불가). "
      "`1510_1520` 은 15:20 봉(단일가 대기 시작)까지 포함한다. 여기서는 모든 창을 15:19 봉 종가까지만 쓴다.")
    w("- 팀원은 비용 3bp·비수정 KODEX 일봉(분배락일 시가 갭이 손실로 잡힘)을 썼고, 여기서는 5bp·수정주가를 쓴다. 96개 변형을 전체 표본에서 칼마로 고른 결과라 표본 내 선택 편향이 크다.")
    w("- 219480·304940 1분봉의 2021~2022년 거래량 값은 15:20~15:29에도 0이 아닌 등 품질이 고르지 않다(가격만 사용).\n")
    w("## 해석·주의\n")
    w("- 오버나잇 롱 규칙은 무조건 오버나잇 자체가 강한 국면(2025-06~ 야간 선물·랠리)에 성과가 몰리기 쉽다. 위 `검증(2025-06-09 이전)` 행과 Δ(신호−비신호)로 조건의 순수 기여를 따로 보라.")
    w("- 실행: 종가 단일가(15:20~15:30)에 KODEX 200 시장가/지정가 매수, 다음 날 08:30~09:00 시가 단일가 매도. 신호 계산은 15:19:59 가격까지라 10분 여유. "
      "1분봉 자료가 체결가 기준이라 실제 15:19 시점 호가·체결 지연은 반영하지 못했다.")
    w("- B 갈래 검증에서 2025-06-09 이전은 약 5개월뿐이라 그 값의 표본 오차가 크다.")
    w("- 그림: `equity.png`(로그 축). 표: `grid.csv`(전체 변형, 설계·검증 지표), `daily_returns.csv`, `trades.csv`, `unconditional_overnight_by_year.csv`.")
    (OUT / "report.md").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
