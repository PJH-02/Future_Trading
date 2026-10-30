"""15차 설계 03_nav_dprt: ETF 종가 공식 NAV 괴리율(dprt) → 밤 다리 비중 오버레이 + 독립 셀.

규칙(사전 등록 alpha-share-prereg-2026-10-30.md, design_shortlist.md ### 03_nav_dprt, 결과 보기 전 고정)
  dprt_{v,t} = kis_etf_nav_daily.csv 의 dprt(%). nav ≤ 0 행·|dprt| > 10 행·2026-10-30 행 제외.
  종가 P_t = nav_t × (1 + dprt_t/100). 괴리 축소분 g_t = ln((1 + dprt_{t+1}/100) / (1 + dprt_t/100)).
  주 셀(오버레이): blog_ON_Q 진입 밤 t 의 차량 v = 233740(pos +2) / 251340(pos −1), jehyeong_ON 진입 밤 v = 069500.
    w_t = 0.5 if dprt_{v,t} ≥ +0.30 else 1.0. net_new = w_t × net_t(비용 비례).
    밤 다리 series 의 date 는 청산일(T+1)이므로 진입일 t = 069500 달력의 직전 거래일. w 는 t 의 dprt 로 정해 청산일 행에 곱한다.
    장부 재계산(13차 조립 규칙: 같은 시장 B 활성일 C=0, 장부 = r_A+r_B+r_C, 합산 = 코스피 50%+코스닥 50% 매일 재조정).
    결합 규칙 = W_all·W_long CAGR·MDD 동시 개선 ∧ 2024·2025 연수익 개선(모두 strict).
  변형: V1 할인 밤 1.5 / V2 독립 셀 229200 dprt ≤ −0.30 종가→익일 종가 / V3 시가 최저 괴리 차량(12차 규칙) /
        V4 C3 잔차 증분 회귀 / V5 60일 z ±1.5 / V6 NAV 평가 시각(선물 15:30→15:45 × L) 점검.
  통계: 이벤트별 순수익 bp 상수 회귀 HAC(maxlags 5) NW t. Δ = 더미 회귀 HAC t. CAGR·MDD 는 069500 거래일 달력(비활성일 0).
  판정: candidate = 순수익 > 0 ∧ t ≥ 2 ∧ 연도 부호 ≥ 70%(이벤트 ≥ 10 인 해) ∧ 상위 5건 < 50% ∧ 전·후반 양수.
        순수익 ≤ 0 reject, 그 밖 reference, 창 ≤ 300 거래일 또는 이벤트 < 10 estimate.
  전제 게이트: KRX 공식 괴리율 대조 불가(로그인) → ① ETF 일봉 close(조정가) / (nav×(1+dprt/100)) 비율이 분배락일 외 상수인지
        ② 네이버 ETF 페이지 당일 괴리율 1점(WebFetch, 실패 시 생략 → naver_check.json 이 없으면 생략으로 기록).
        대조 실패 시 1배(229200·069500) 셀만 판정, 레버리지·인버스(233740·251340·122630) 셀은 estimate.
자료(원본 수정 없음): hyfe/data/raw/r15_external/kis_etf_nav_daily.csv, etf/*.csv, etf_extra/*.csv, KQ150_index_kis.csv,
  intraday/kis_minute/{full_day,full_day_ext,kq150_full_day}/futures_1min.csv, round13_slot_book/series·book,
  round9_c3_etfs/kq150_etfs/daily_returns.csv(z_spx·mis_229200), round5_strategy/c3_levered/daily_returns.csv(z_index·mis).
가정: 단일가 체결(15:30 종가·09:00 시가), ETF 왕복 5bp(자본 1 단위당), 오버레이 비용은 w 에 비례. V1 코스피 1.5 는 50% 122630 + 50% 069500
  (비용 5bp 고정, 122630 밤 잔차 = r_122630 − 2·r_069500 를 실측 차감). 코스닥 1.5 는 ETF 한도 밖이라 scaled(보고만).
산출: round15_alpha_share/03_nav_dprt/{cells.csv, trades.csv, notes.md, fig1~3.png}, series/overlay_03.csv,
  V2 통과 시 series/beomgu_NAV_Q.csv + meta, V3 통과 시 series/beomgu_C3Q_v3.csv + meta.
실행: PYTHONUTF8=1 python "<code>/topics/r15_03_nav_dprt.py"
"""
import os, re, sys, json
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

RAW = "<raw>"
LAB = "<OUT>"
R13 = LAB + "/round13_slot_book"
OUT = LAB + "/round15_alpha_share/03_nav_dprt"
SER = LAB + "/round15_alpha_share/series"
os.makedirs(OUT, exist_ok=True)
os.makedirs(SER, exist_ok=True)

W_ALL = ("2022-12-13", "2026-09-17")
W_EX = ("2022-12-13", "2025-12-30")
W_LONG = ("2016-10-04", "2026-09-17")
Y24 = ("2024-01-02", "2024-12-30")
Y25 = ("2025-01-02", "2025-12-30")
Y26 = ("2026-01-02", "2026-09-17")
THR = 0.30
COST = 0.0005
LAST = "2026-10-29"          # 2026-10-30 행 제외
LEV = {"233740": 2, "251340": -1, "122630": 2, "114800": -1, "069500": 1, "229200": 1}
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#8e6bd6"]   # dataviz 기본 팔레트 슬롯(검증 통과, r15_leverage_desk 와 동일)

# ------------------------------------------------------------------ 통계
def hac_t(y, X=None):
    y = np.asarray(y, float)
    if len(y) < 10:
        return np.nan
    X = np.ones((len(y), 1)) if X is None else X
    r = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
    return float(r.tvalues[-1])


def delta_test(df):
    """df: 날짜순 DataFrame, y(bp)·s(0/1). Δ = b, HAC t."""
    df = df.dropna().sort_index()
    if df["s"].sum() < 10 or (1 - df["s"]).sum() < 10:
        return np.nan, np.nan
    X = sm.add_constant(df["s"].to_numpy(float))
    r = sm.OLS(df["y"].to_numpy(float), X).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
    return float(r.params[1]), float(r.tvalues[1])


def ev_stats(net_bp, gross_bp=None):
    """이벤트별 순수익 bp(날짜 인덱스) → 공통 지표."""
    net = net_bp.sort_index()
    n = len(net)
    o = dict(n=n, gross_bp=float(gross_bp.mean()) if gross_bp is not None and n else np.nan,
             net_bp=float(net.mean()) if n else np.nan, t_nw=hac_t(net) if n else np.nan,
             win_pct=float((net > 0).mean() * 100) if n else np.nan)
    if n:
        yr = net.groupby(net.index.year).agg(["sum", "size"])
        yr = yr[yr["size"] >= 10]
        o["yearly_pos"] = f"{int((yr['sum'] > 0).sum())}/{len(yr)}"
        o["yearly_pos_pct"] = float((yr["sum"] > 0).mean() * 100) if len(yr) else np.nan
        tot = net.sum()
        o["top5_share_pct"] = float(net.nlargest(5).sum() / tot * 100) if tot > 0 else np.nan
        h = n // 2
        o["half1_net_bp"] = float(net.iloc[:h].mean()) if h else np.nan
        o["half2_net_bp"] = float(net.iloc[h:].mean())
    else:
        o.update(yearly_pos="0/0", yearly_pos_pct=np.nan, top5_share_pct=np.nan, half1_net_bp=np.nan, half2_net_bp=np.nan)
    return o


def verdict(o, days):
    if o["n"] < 10 or days <= 300:
        return "estimate"
    if not o["net_bp"] > 0:
        return "reject"
    ok = (o["t_nw"] >= 2 and o["yearly_pos_pct"] >= 70 and o["top5_share_pct"] < 50
          and o["half1_net_bp"] > 0 and o["half2_net_bp"] > 0)
    return "candidate" if ok else "reference"


def metrics(r):
    r = r.dropna()
    n = len(r)
    eq = (1 + r).cumprod()
    cum = float(eq.iloc[-1])
    vol = float(r.std(ddof=1)) * np.sqrt(252)
    return dict(n_days=n, cum=cum - 1, cagr=cum ** (252 / n) - 1 if cum > 0 else -1.0,
                mdd=float((eq / eq.cummax() - 1).min()), sharpe=float(r.mean()) * 252 / vol if vol else np.nan,
                mean_bp=float(r.mean() * 1e4), t_nw=hac_t(r * 1e4))


def win(r, w):
    return r.loc[w[0]:w[1]]


# ------------------------------------------------------------------ 자료
def px(path):
    return pd.read_csv(path, parse_dates=["date"]).set_index("date").sort_index()


cal = px(RAW + "/etf/069500.csv").index                      # KRX 거래일 달력 2002-10-14~2026-09-29
prev_day = pd.Series(cal[:-1], index=cal[1:])                 # 청산일 → 진입일(직전 거래일)
next_day = pd.Series(cal[1:], index=cal[:-1])

nav_raw = pd.read_csv(RAW + "/r15_external/kis_etf_nav_daily.csv", dtype={"code": str}, parse_dates=["date"])
n_raw = len(nav_raw)
nav = nav_raw[(nav_raw["nav"] > 0) & (nav_raw["dprt"].abs() <= 10) & (nav_raw["date"] <= LAST)].copy()
nav["P"] = nav["nav"] * (1 + nav["dprt"] / 100)
NAV = {c: g.set_index("date").sort_index() for c, g in nav.groupby("code")}
for c, g in NAV.items():
    nxt = g["dprt"].reindex(next_day.reindex(g.index).values)
    g["g"] = np.log((1 + nxt.values / 100) / (1 + g["dprt"].values / 100))          # 괴리 축소분(t→t+1)
    g["lnP"] = np.log(g["P"].reindex(next_day.reindex(g.index).values).values / g["P"].values)   # ln(P_{t+1}/P_t)
    g["lnNAV"] = np.log(g["nav"].reindex(next_day.reindex(g.index).values).values / g["nav"].values)
    g["z60"] = (g["dprt"] - g["dprt"].rolling(60).mean()) / g["dprt"].rolling(60).std()
ETF = {c: px(RAW + f"/etf/{c}.csv") for c in ["069500", "122630", "229200", "233740", "114800"]}
ETF["251340"] = px(RAW + "/etf_extra/251340.csv")


def series(h):
    s = pd.read_csv(f"{R13}/series/{h}.csv", parse_dates=["date"]).set_index("date")
    meta = json.loads(open(f"{R13}/series/{h}.meta.json", encoding="utf-8").read())
    return s.loc[meta["first_valid"]:meta["last_valid"]].fillna(0.0), meta


LEGS = {"KOSPI_A_jehyeong_ON": ("jehyeong_ON", "KOSPI"), "KOSPI_B_beomgu_C3VT30": ("beomgu_C3VT30", "KOSPI"),
        "KOSPI_C_jehyeong_KP": ("jehyeong_KP", "KOSPI"), "KOSDAQ_A_blog_ON_Q": ("blog_ON_Q", "KOSDAQ"),
        "KOSDAQ_B_beomgu_C3Q": ("beomgu_C3Q", "KOSDAQ"), "KOSDAQ_C_jehyeong_KQ": ("jehyeong_KQ", "KOSDAQ")}


def panel_wall():
    """W_all: book_daily 의 다리별 net(B·C 겹침 처리 후). r15_leverage_desk.panel_wall 과 같은 검산."""
    book = pd.read_csv(R13 + "/book/book_daily.csv", parse_dates=["date"]).set_index("date")
    legs = {}
    for col, (h, mkt) in LEGS.items():
        s = series(h)[0].reindex(book.index).fillna(0.0)
        net = book[col]
        act = net != 0
        assert np.allclose(net[act], s["net"][act]), col
        legs[col] = pd.DataFrame({"net": net, "cost": (s["gross"] - s["net"]).where(act, 0.0)})
    K = sum(l["net"] for c, l in legs.items() if c.startswith("KOSPI"))
    Q = sum(l["net"] for c, l in legs.items() if c.startswith("KOSDAQ"))
    assert np.allclose(0.5 * (K + Q), book["comb_a"])
    return legs


def panel_long(cols, w):
    legs, idx = {}, None
    for col in cols:
        s = series(LEGS[col][0])[0].loc[w[0]:w[1]]
        legs[col] = pd.DataFrame({"net": s["net"], "cost": s["gross"] - s["net"]})
        idx = legs[col].index if idx is None else idx.intersection(legs[col].index)
    return {c: l.reindex(idx) for c, l in legs.items()}


def comb(legs, override=None):
    """override: col → net Series(청산일 인덱스). 주어진 날만 바꾼다."""
    nets = {}
    for c, l in legs.items():
        n = l["net"].copy()
        if override and c in override:
            o = override[c].reindex(n.index)
            n = o.where(o.notna(), n)
        nets[c] = n
    K = sum(n for c, n in nets.items() if c.startswith("KOSPI"))
    Q = sum(n for c, n in nets.items() if c.startswith("KOSDAQ"))
    return 0.5 * (K + Q)


def book_metrics(override=None):
    r_all = comb(LEGS_WALL, override)
    r_long = comb(LEGS_LONG, override)
    o = {}
    for k, r, w in [("W_all", r_all, W_ALL), ("W_ex2026", r_all, W_EX), ("Y2024", r_all, Y24), ("Y2025", r_all, Y25),
                    ("Y2026", r_all, Y26), ("W_long", r_long, W_LONG)]:
        o[k] = metrics(win(r, w))
    return o


def book_rule(new, base):
    return bool(all(new[w]["cagr"] > base[w]["cagr"] and new[w]["mdd"] > base[w]["mdd"] for w in ("W_all", "W_long"))
                and new["Y2024"]["cum"] > base["Y2024"]["cum"] and new["Y2025"]["cum"] > base["Y2025"]["cum"])


def bm_str(m):
    return (f"W_all {m['W_all']['cagr']*100:.2f}%/{m['W_all']['mdd']*100:.2f}% 샤프 {m['W_all']['sharpe']:.2f} | "
            f"W_ex2026 {m['W_ex2026']['cagr']*100:.2f}%/{m['W_ex2026']['mdd']*100:.2f}% | W_long {m['W_long']['cagr']*100:.2f}%/{m['W_long']['mdd']*100:.2f}% | "
            f"2024 {m['Y2024']['cum']*100:+.2f}% 2025 {m['Y2025']['cum']*100:+.2f}% 2026 {m['Y2026']['cum']*100:+.2f}%")


LEGS_WALL = panel_wall()
LEGS_LONG = panel_long(["KOSPI_B_beomgu_C3VT30", "KOSDAQ_A_blog_ON_Q", "KOSDAQ_B_beomgu_C3Q"], W_LONG)
BASE = book_metrics()
assert abs(BASE["W_all"]["cagr"] - 0.314179) < 0.002 and abs(BASE["W_all"]["mdd"] + 0.061046) < 0.001, BASE["W_all"]
assert BASE["W_long"]["n_days"] == 2444 and abs(BASE["W_long"]["cagr"] - 0.150369) < 0.002, BASE["W_long"]
print("13차 재현:", bm_str(BASE))

CELLS, TRADES, NOTE = [], [], []


def cell(id_, name, market, window, st, verdict_, notes, **extra):
    row = dict(id=id_, name=name, market=market, window=window, n=st.get("n", np.nan), gross_bp=st.get("gross_bp", np.nan),
               net_bp=st.get("net_bp", np.nan), t_nw=st.get("t_nw", np.nan), delta_bp=extra.pop("delta_bp", np.nan),
               t_delta=extra.pop("t_delta", np.nan), win_pct=st.get("win_pct", np.nan), cagr_pct=extra.pop("cagr_pct", np.nan),
               mdd_pct=extra.pop("mdd_pct", np.nan), yearly_pos=st.get("yearly_pos", ""), top5_share_pct=st.get("top5_share_pct", np.nan),
               half1_net_bp=st.get("half1_net_bp", np.nan), half2_net_bp=st.get("half2_net_bp", np.nan), verdict=verdict_, notes=notes)
    row.update(extra)
    CELLS.append(row)
    return row


def book_cond(m):
    arrow = lambda a, b: "↑" if a > b else ("=" if a == b else "↓")
    return (f"W_all CAGR {arrow(m['W_all']['cagr'], BASE['W_all']['cagr'])} MDD {arrow(m['W_all']['mdd'], BASE['W_all']['mdd'])}, "
            f"W_long CAGR {arrow(m['W_long']['cagr'], BASE['W_long']['cagr'])} MDD {arrow(m['W_long']['mdd'], BASE['W_long']['mdd'])}, "
            f"2024 {arrow(m['Y2024']['cum'], BASE['Y2024']['cum'])}, 2025 {arrow(m['Y2025']['cum'], BASE['Y2025']['cum'])} (= 은 동률, 개선 아님)")


def book_cell(id_, name, m, notes, grade=None, n=np.nan, **extra):
    ok = book_rule(m, BASE)
    v = grade or ("candidate" if ok else "reference")
    st = dict(n=n, net_bp=m["W_all"]["mean_bp"], t_nw=m["W_all"]["t_nw"], gross_bp=BASE["W_all"]["mean_bp"])
    return cell(id_, name, "KOSPI+KOSDAQ", "W_all·W_long·2024·2025", st, v,
                f"결합 규칙 {'통과' if ok else '미달'} [{book_cond(m)}] | 적용: {bm_str(m)} | 원 장부: {bm_str(BASE)} | {notes}",
                cagr_pct=m["W_all"]["cagr"] * 100, mdd_pct=m["W_all"]["mdd"] * 100, book_pass=ok, cond=book_cond(m), **extra)


# ------------------------------------------------------------------ 게이트 ① 정합성
gate = {}
for c in ["069500", "229200", "233740", "251340", "122630"]:
    g = NAV[c]
    e = ETF[c]["close"].reindex(g.index)
    ratio = np.log(e / g["P"])
    d = ratio.diff().dropna()
    jumps = d[d.abs() > 0.001]
    ratio_kis = np.log(g["close_adj"] / g["P"]).diff().dropna()
    d16 = d.loc["2016-10-04":]
    gate[c] = dict(n=int(len(d)), p_stable_2bp=float((d.abs() < 0.0002).mean() * 100), n_jump=int(len(jumps)),
                   jump_dates=[x.date().isoformat() for x in jumps.index[-6:]], max_abs=float(d.abs().max()),
                   med_abs=float(d.abs().median()), n_jump_kisadj=int((ratio_kis.abs() > 0.001).sum()),
                   n16=int(len(d16)), p_stable_2bp_16=float((d16.abs() < 0.0002).mean() * 100), n_jump_16=int((d16.abs() > 0.001).sum()),
                   jump_by_year={int(k): int(v) for k, v in jumps.groupby(jumps.index.year).size().items()})
    print(f"게이트① {c}: n {gate[c]['n']} |Δln 비율|<2bp {gate[c]['p_stable_2bp']:.1f}% 점프(>10bp) {gate[c]['n_jump']} (KIS close_adj 기준 {gate[c]['n_jump_kisadj']}) 최근 {gate[c]['jump_dates']} | 2016.10~ n {gate[c]['n16']} 안정 {gate[c]['p_stable_2bp_16']:.1f}% 점프 {gate[c]['n_jump_16']} | 연도별 {gate[c]['jump_by_year']}")
naver = None
if os.path.exists(OUT + "/naver_check.json"):
    naver = json.loads(open(OUT + "/naver_check.json", encoding="utf-8").read())

# ------------------------------------------------------------------ 주 셀: 프리미엄 게이트 오버레이
def leg_nights(h, veh):
    """활성 밤(청산일 D) → 진입일 t, 차량, dprt·z60 at t."""
    s, meta = series(h)
    a = s[s["pos"] != 0].copy()
    a["t"] = prev_day.reindex(a.index).values
    a["veh"] = [veh(p) for p in a["pos"]]
    for col in ("dprt", "z60"):
        a[col] = [NAV[v][col].get(t, np.nan) for v, t in zip(a["veh"], a["t"])]
    return a, meta


Q, Q_META = leg_nights("blog_ON_Q", lambda p: "233740" if p > 0 else "251340")
K, K_META = leg_nights("jehyeong_ON", lambda p: "069500")
# 차량 수익 정합: blog_ON_Q gross = 차량의 종가→익일 시가 수익(자본 1), jehyeong_ON gross ≈ 069500 동일 구간(출처 다름)
co = lambda c: ETF[c]["open"] / ETF[c]["close"].shift(1) - 1
chk = np.array([co(v).get(D, np.nan) for v, D in zip(Q["veh"], Q.index)])
assert np.nanmax(np.abs(chk - Q["gross"].values)) < 1e-6, "blog_ON_Q gross ≠ 차량 밤 수익"
chk_k = co("069500").reindex(K.index)
print(f"정합: blog_ON_Q gross = 차량 밤 수익 (최대 오차 {np.nanmax(np.abs(chk - Q['gross'].values)):.1e}); jehyeong_ON gross vs 069500 밤 수익 중앙 오차 {(chk_k - K['gross']).abs().median()*1e4:.2f}bp 최대 {(chk_k - K['gross']).abs().max()*1e4:.1f}bp")
for a, h in ((Q, "blog_ON_Q"), (K, "jehyeong_ON")):
    print(f"{h}: 활성 {len(a)} dprt 결측 {int(a['dprt'].isna().sum())} 프리미엄≥0.30 {int((a['dprt'] >= THR).sum())} 할인≤−0.30 {int((a['dprt'] <= -THR).sum())}")


def weights(a, thr=THR, col="dprt"):
    return pd.Series(np.where(a[col] >= thr, 0.5, 1.0), index=a.index)


def gated_cells(a, h, w, tag, grade=None):
    """게이트 밤(w<1) 의 원 순수익 vs 비게이트 밤 Δ, 창별."""
    rows = {}
    for wn, ww in [("full", (str(a.index.min().date()), str(a.index.max().date()))), ("W_all", W_ALL)]:
        x = a.loc[ww[0]:ww[1]]
        wx = w.reindex(x.index)
        gN = x[wx < 1]
        st = ev_stats(gN["net"] * 1e4, gN["gross"] * 1e4)
        d, t = delta_test(pd.DataFrame({"y": x["net"] * 1e4, "s": (wx < 1).astype(float)}))
        days = len(cal[(cal >= ww[0]) & (cal <= ww[1])])
        v = grade or "diagnostic"
        nf = lambda v_: "n/a(게이트 밤 < 10)" if np.isnan(v_) else f"{v_:.2f}"
        rows[wn] = cell(f"{tag}_{h}_{wn}", f"{tag} {h} 게이트 밤(원 순수익)", "KOSDAQ" if h == "blog_ON_Q" else "KOSPI", f"{ww[0]}~{ww[1]}", st, v,
                        f"게이트 밤 n {st['n']} / 활성 {len(x)}, 비게이트 밤 평균 {x[wx >= 1]['net'].mean()*1e4:.1f}bp, Δ(게이트−비게이트) {nf(d)}bp t {nf(t)}. 오버레이는 이 밤의 net 을 절반으로 줄임",
                        delta_bp=d, t_delta=t, n_active=len(x))
    return rows


wQ, wK = weights(Q), weights(K)
gated_cells(Q, "blog_ON_Q", wQ, "M", grade="estimate")
gated_cells(K, "jehyeong_ON", wK, "M")
OV = {"KOSDAQ_A_blog_ON_Q": Q["net"] * wQ, "KOSPI_A_jehyeong_ON": K["net"] * wK}
M_BOTH = book_metrics(OV)
M_K = book_metrics({"KOSPI_A_jehyeong_ON": OV["KOSPI_A_jehyeong_ON"]})
M_Q = book_metrics({"KOSDAQ_A_blog_ON_Q": OV["KOSDAQ_A_blog_ON_Q"]})
main_row = book_cell("M_overlay_book", "주 셀 프리미엄 게이트 w=0.5(dprt≥+0.30) blog_ON_Q+jehyeong_ON 장부", M_BOTH,
                     "레버리지·인버스 dprt 외부 대조 실패 → 사전 규칙대로 estimate(수치는 그대로)", grade="estimate",
                     n=int((wQ < 1).sum() + (wK < 1).sum()))
k_row = book_cell("M_overlay_1x_book", "주 셀 1배 부분만(jehyeong_ON 069500 게이트) 장부", M_K, "1배 셀, 판정 가능", n=int((wK < 1).sum()))
q_row = book_cell("M_overlay_lev_book", "주 셀 레버리지·인버스 부분만(blog_ON_Q 233740/251340 게이트) 장부", M_Q, "레버리지·인버스 셀 estimate", grade="estimate", n=int((wQ < 1).sum()))
for a, w, h in ((Q, wQ, "blog_ON_Q"), (K, wK, "jehyeong_ON")):
    for D, r in a[w < 1].iterrows():
        TRADES.append(dict(cell="M_overlay", date=D.date(), entry=r["t"].date(), leg=h, vehicle=r["veh"], pos=r["pos"], dprt=r["dprt"], w=0.5,
                           gross=r["gross"], net=r["net"], net_new=r["net"] * 0.5))
# 다리 자체 지표(오버레이 전후)
for a, w, h in ((Q, wQ, "blog_ON_Q"), (K, wK, "jehyeong_ON")):
    s = series(h)[0]
    r0, r1 = s["net"], s["net"] * w.reindex(s.index).fillna(1.0)
    for wn, ww in [("full", (str(s.index.min().date()), str(s.index.max().date()))), ("W_all", W_ALL)]:
        m0, m1 = metrics(win(r0, ww)), metrics(win(r1, ww))
        NOTE.append(f"{h} 다리 {wn} {ww[0]}~{ww[1]}: CAGR {m0['cagr']*100:.2f}→{m1['cagr']*100:.2f}% MDD {m0['mdd']*100:.2f}→{m1['mdd']*100:.2f}%")
        print(NOTE[-1])
print("주 셀:", bm_str(M_BOTH), "| 결합 규칙", book_rule(M_BOTH, BASE))
print("1배만:", bm_str(M_K), "| 결합 규칙", book_rule(M_K, BASE))
print("lev만:", bm_str(M_Q), "| 결합 규칙", book_rule(M_Q, BASE))

# overlay_03.csv (date = 청산일, 전 거래일 행, 기본 1.0)
ov_idx = cal[(cal >= min(Q.index.min(), K.index.min())) & (cal <= max(Q.index.max(), K.index.max()))]
ov = pd.DataFrame({"date": ov_idx, "w_blog_ON_Q": wQ.reindex(ov_idx).fillna(1.0).values, "w_jehyeong_ON": wK.reindex(ov_idx).fillna(1.0).values})
ov.to_csv(SER + "/overlay_03.csv", index=False, date_format="%Y-%m-%d", float_format="%.1f")

# ------------------------------------------------------------------ V1 할인 밤 1.5
resid = (co("122630") - 2 * co("069500"))                      # 122630 밤 잔차(청산일 인덱스)
K1 = K.copy()
disc_k = K1["dprt"] <= -THR
K1["resid"] = resid.reindex(K1.index)
K1["cost"] = K1["gross"] - K1["net"]
boost_ok = disc_k & K1["resid"].notna()
net_v1_k = K["net"] * wK
net_v1_k[boost_ok] = 1.5 * K1["gross"][boost_ok] + 0.5 * K1["resid"][boost_ok] - K1["cost"][boost_ok]
disc_q = Q["dprt"] <= -THR
net_v1_q = Q["net"] * wQ
net_v1_q[disc_q] = 1.5 * Q["net"][disc_q]                       # scaled(비용 비례), 선물 필요
V1a = book_metrics({"KOSPI_A_jehyeong_ON": net_v1_k, "KOSDAQ_A_blog_ON_Q": OV["KOSDAQ_A_blog_ON_Q"]})
V1b = book_metrics({"KOSPI_A_jehyeong_ON": net_v1_k, "KOSDAQ_A_blog_ON_Q": net_v1_q})
st_k = ev_stats(K["net"][disc_k] * 1e4, K["gross"][disc_k] * 1e4)
st_q = ev_stats(Q["net"][disc_q] * 1e4, Q["gross"][disc_q] * 1e4)
dk = delta_test(pd.DataFrame({"y": K["net"] * 1e4, "s": disc_k.astype(float)}))
dq = delta_test(pd.DataFrame({"y": Q["net"] * 1e4, "s": disc_q.astype(float)}))
cell("V1_disc_jehyeong_ON", "V1 할인 밤(dprt≤−0.30) jehyeong_ON 원 순수익", "KOSPI", f"{K.index.min().date()}~{K.index.max().date()}", st_k, "diagnostic",
     f"할인 밤 n {st_k['n']}/{len(K)}, Δ(할인−그 외) {dk[0]:.1f}bp t {dk[1]:.2f}, 122630 잔차 평균 {K1['resid'][boost_ok].mean()*1e4:.2f}bp(n {int(boost_ok.sum())})", delta_bp=dk[0], t_delta=dk[1])
cell("V1_disc_blog_ON_Q", "V1 할인 밤(dprt≤−0.30) blog_ON_Q 원 순수익", "KOSDAQ", f"{Q.index.min().date()}~{Q.index.max().date()}", st_q, "estimate",
     f"할인 밤 n {st_q['n']}/{len(Q)}, Δ(할인−그 외) {dq[0]:.1f}bp t {dq[1]:.2f}. 코스닥 1.5 는 ETF 한도 밖(선물 필요)", delta_bp=dq[0], t_delta=dq[1])
book_cell("V1a_book", "V1 주 셀 + 코스피 할인 밤 1.5(50% 122630 + 50% 069500, 잔차 실측) 장부", V1a, f"코스피 부스트 밤 {int(boost_ok.sum())}, 코스닥은 주 셀 그대로", n=int(boost_ok.sum()))
book_cell("V1b_book", "V1 + 코스닥 할인 밤 1.5 scaled(선물 필요, 보고만) 장부", V1b, f"코스닥 부스트 밤 {int(disc_q.sum())}, 레버리지·인버스 포함", grade="estimate", n=int(boost_ok.sum() + disc_q.sum()))
print("V1a:", bm_str(V1a), book_rule(V1a, BASE), "| V1b:", bm_str(V1b), book_rule(V1b, BASE))

# ------------------------------------------------------------------ V2 독립 셀 229200 종가→익일 종가
g229 = NAV["229200"].loc["2015-10-13":"2026-09-29"].copy()
g229 = g229[g229["lnP"].notna() & (next_day.reindex(g229.index) <= pd.Timestamp("2026-09-29")).values]
g229["net"] = g229["lnP"] - COST
V2_WIN = {"full": ("2015-10-13", "2026-09-29"), "design": ("2015-10-13", "2019-12-30"), "verify": ("2020-01-02", "2026-09-29"), "W_all": W_ALL}


def v2_cells(sig_col, thr, tag, name, src="lnP"):
    out = {}
    for wn, ww in V2_WIN.items():
        x = g229.loc[ww[0]:ww[1]]
        x = x[(next_day.reindex(x.index) <= pd.Timestamp(ww[1])).values]      # 청산일도 창 안
        s = (x[sig_col] <= thr)
        ev = x[s]
        netv = (ev[src] - COST) * 1e4
        st = ev_stats(netv, ev[src] * 1e4)
        d, t = delta_test(pd.DataFrame({"y": x[src] * 1e4, "s": s.astype(float)}))
        days = len(cal[(cal >= ww[0]) & (cal <= ww[1])])
        # 달력 일수익(청산일 = t+1, 단순수익)으로 CAGR·MDD
        exit_idx = next_day.reindex(ev.index).values
        daily = pd.Series(0.0, index=cal[(cal >= ww[0]) & (cal <= ww[1])])
        daily.loc[exit_idx] = (np.exp(ev[src].values) - 1) - COST
        mm = metrics(daily)
        gst = ev_stats(ev["g"] * 1e4)
        v = verdict(st, days)
        out[wn] = dict(cell(f"{tag}_{wn}", f"{name} [{wn}]", "KOSDAQ", f"{ww[0]}~{ww[1]}", st, v,
                            f"이벤트 {st['n']}/{len(x)}일, Δ(신호−비신호, ln P 수익) {d:.1f}bp t {t:.2f}; g 평균 {gst['net_bp']:.2f}bp NW t {gst['t_nw']:.2f}; NAV 수익 평균 {ev['lnNAV'].mean()*1e4:.1f}bp; 중앙 {netv.median():.1f}bp",
                            delta_bp=d, t_delta=t, cagr_pct=mm["cagr"] * 100, mdd_pct=mm["mdd"] * 100, g_bp=gst["net_bp"], g_t=gst["t_nw"], median_bp=float(netv.median())))
        out[wn]["ev"] = ev          # 복사본에만 붙임(cells.csv 에는 안 들어감)
    return out


V2 = v2_cells("dprt", -THR, "V2", "V2 독립 229200 dprt≤−0.30 종가 매수→익일 종가, ln 수익 − 5bp")
# 진단: 네이버 조정 종가 기준(분배락 영향 점검)
g229["lnC"] = np.log(ETF["229200"]["close"].reindex(next_day.reindex(g229.index).values).values / ETF["229200"]["close"].reindex(g229.index).values)
xx = g229.loc["2015-10-13":"2026-09-29"]
s = xx["dprt"] <= -THR
st = ev_stats((xx["lnC"][s] - COST) * 1e4, xx["lnC"][s] * 1e4)
cell("V2diag_adjclose_full", "V2 진단: 체결가를 ETF 일봉 조정 종가로(분배락 영향)", "KOSDAQ", "2015-10-13~2026-09-29", st, "diagnostic",
     f"KIS P 기준 셀과 차이 = 분배락일 효과; 이벤트 {st['n']}")
for wn, r in V2.items():
    for t_, e in r["ev"].iterrows():
        if wn == "full":
            TRADES.append(dict(cell="V2", date=next_day[t_].date(), entry=t_.date(), leg="beomgu_NAV_Q", vehicle="229200", pos=1, dprt=e["dprt"], w=1.0,
                               gross=e["lnP"], net=e["lnP"] - COST, net_new=np.nan, g=e["g"]))
print("V2:", {k: (r["n"], round(r["net_bp"], 2), round(r["t_nw"], 2), r["verdict"], round(r["g_bp"], 2), round(r["g_t"], 2)) for k, r in V2.items()})

# V2 series(통과 시 저장; 판정은 full 창 공통 규칙, V4 결과는 meta 에 기록)
ev_full = V2["full"]["ev"]
nav_q = pd.DataFrame({"date": cal, "pos": np.nan, "gross": np.nan, "net": np.nan}).set_index("date")
first_valid = next_day[ev_full.index.min()] if len(ev_full) else None

# ------------------------------------------------------------------ V3 시가 최저 괴리 차량(12차 규칙)
V8 = ["229200", "232080", "270810", "354500", "316670", "304770", "301400", "450910"]
KQ = px(RAW + "/etf_extra/KQ150_index_kis.csv").loc["2015-07-13":]
gi_o = np.log(KQ["open"] / KQ["close"].shift(1))
gi_c = np.log(KQ["close"] / KQ["close"].shift(1))
MIS, OC, LIQ = {}, {}, {}
for v in V8:
    d = ETF["229200"] if v == "229200" else px(RAW + f"/etf_extra/{v}.csv")
    d = d.reindex(KQ.index)
    mo = np.log(d["open"] / d["close"].shift(1)) - gi_o
    mc = np.log(d["close"] / d["close"].shift(1)) - gi_c
    bad = (mo.abs() > 0.03) | (mc.abs() > 0.03)
    mo, mc = mo.mask(bad), mc.mask(bad)
    MIS[v] = mo - mc.shift(1).rolling(5).median()
    OC[v] = d["close"] / d["open"] - 1
    LIQ[v] = (d["close"] * d["volume"]).shift(1).rolling(20).median()
MIS, OC, LIQ = pd.DataFrame(MIS), pd.DataFrame(OC), pd.DataFrame(LIQ)
elig = MIS.notna() & (LIQ >= 1e9)
m = MIS.where(elig)
cnt = elig.sum(axis=1)
med, mn = m.median(axis=1), m.min(axis=1)
sig3 = (cnt >= 3) & ((med - mn) >= 0.0010)
sig3 = sig3.loc["2017-01-02":"2026-09-29"]
v3 = pd.DataFrame(index=sig3[sig3].index)
v3["vmin"] = m.loc[v3.index].idxmin(axis=1)
v3["vmax"] = m.loc[v3.index].idxmax(axis=1)
v3["mis_min"], v3["mis_med"] = mn.loc[v3.index], med.loc[v3.index]
v3["rel"] = [OC.at[d, v] - OC.at[d, "229200"] for d, v in zip(v3.index, v3["vmin"])]
v3["rel_max"] = [OC.at[d, v] - OC.at[d, "229200"] for d, v in zip(v3.index, v3["vmax"])]
v3["n_elig"] = cnt.loc[v3.index]
v3 = v3.dropna(subset=["rel"])
C3Q, C3Q_META = series("beomgu_C3Q")
v3["c3q_pos"] = C3Q["pos"].reindex(v3.index).fillna(0.0)
print(f"V3: 신호일 {len(v3)} (2017-01-02~2026-09-29), 자격 차량 평균 {v3['n_elig'].mean():.2f}, 선택 비중 {v3['vmin'].value_counts(normalize=True).round(2).to_dict()}, C3Q 활성 겹침 {int((v3['c3q_pos'] != 0).sum())}")
V3_WIN = {"full": ("2017-01-02", "2026-09-29"), "design": ("2017-01-02", "2022-12-29"), "verify": ("2023-01-02", "2026-09-29")}
V3 = {}
for wn, ww in V3_WIN.items():
    x = v3.loc[ww[0]:ww[1]]
    days = len(cal[(cal >= ww[0]) & (cal <= ww[1])])
    st = ev_stats(x["rel"] * 1e4)
    stp = ev_stats(x["rel_max"] * 1e4)
    ok12 = bool(st["n"] >= 10 and st["net_bp"] > 5 and x["rel"].median() > 0 and st["t_nw"] >= 2)
    sym = bool(stp["net_bp"] < 0 and abs(stp["net_bp"]) >= 0.5 * st["net_bp"])
    V3[wn] = cell(f"V3_all_{wn}", f"V3 최저 괴리 차량 − 229200 시가→종가(모든 신호일) [{wn}]", "KOSDAQ", f"{ww[0]}~{ww[1]}", st, "diagnostic",
                  f"12차 조건(평균>5 ∧ 중앙>0 ∧ t≥2) {'충족' if ok12 else '미달'}, 중앙 {x['rel'].median()*1e4:.1f}bp; 위약(최고 괴리 차량) {stp['net_bp']:.1f}bp t {stp['t_nw']:.2f} {'대칭' if sym else '비대칭'}; 단일가 체결량 10% 조건은 자료 없어 미검정; 비용 0(차량 교체)",
                  median_bp=float(x["rel"].median() * 1e4), ok12=ok12, placebo_bp=stp["net_bp"], placebo_t=stp["t_nw"], placebo_sym=sym)
    xa = x[x["c3q_pos"] != 0]
    sta = ev_stats(xa["rel"] * 1e4)
    stpa = ev_stats(xa["rel_max"] * 1e4)
    ok12a = bool(sta["n"] >= 10 and sta["net_bp"] > 5 and xa["rel"].median() > 0 and sta["t_nw"] >= 2)
    syma = bool(stpa["n"] >= 10 and stpa["net_bp"] < 0 and abs(stpa["net_bp"]) >= 0.5 * sta["net_bp"])
    V3[wn + "_c3q"] = cell(f"V3_c3q_{wn}", f"V3 C3Q 활성일 ∩ 신호일 상대이득(단위당) [{wn}]", "KOSDAQ", f"{ww[0]}~{ww[1]}", sta,
                           "estimate" if sta["n"] < 10 else ("reference" if (ok12a and syma) else ("candidate" if ok12a else "reference")),
                           f"12차 조건 {'충족' if ok12a else '미달'}, 중앙 {xa['rel'].median()*1e4 if len(xa) else np.nan:.1f}bp, pos 가중 평균 {(xa['rel']*xa['c3q_pos']).mean()*1e4 if len(xa) else np.nan:.1f}bp; 위약 {stpa['net_bp']:.1f}bp t {stpa['t_nw']:.2f} {'대칭→reference' if syma else '비대칭'}; 체결량 10% 조건 미검정(자료 없음)",
                           median_bp=float(xa["rel"].median() * 1e4) if len(xa) else np.nan, ok12=ok12a, placebo_bp=stpa["net_bp"], placebo_t=stpa["t_nw"], placebo_sym=syma)
for d, r in v3.iterrows():
    TRADES.append(dict(cell="V3", date=d.date(), entry=d.date(), leg="beomgu_C3Q_v3", vehicle=r["vmin"], pos=r["c3q_pos"], dprt=np.nan, w=np.nan,
                       gross=r["rel"], net=r["rel"], net_new=r["rel"] * r["c3q_pos"], mis_min=r["mis_min"], mis_med=r["mis_med"], vmax=r["vmax"], rel_max=r["rel_max"]))
# 교체본 장부
rep = v3[v3["c3q_pos"] != 0]
c3q_new = C3Q["net"].copy()
c3q_new.loc[rep.index] = C3Q["net"].loc[rep.index] + rep["c3q_pos"] * rep["rel"]
V3B = book_metrics({"KOSDAQ_B_beomgu_C3Q": c3q_new.loc[rep.index]})
book_cell("V3_book", "V3 C3Q 를 최저 괴리 차량으로 교체한 장부", V3B, f"교체 {len(rep)}일, 비용 0(차량만 교체, 단일가 가정)", n=len(rep))
print("V3 book:", bm_str(V3B), book_rule(V3B, BASE))

# ------------------------------------------------------------------ V4 C3 잔차 증분
def v4(df, zc, misc, oc, wc, code, tag, start):
    d = df.copy()
    d["dprt_lag"] = NAV[code]["dprt"].reindex(prev_day.reindex(d.index).values).values
    d = d.loc[start:"2026-09-22"].dropna(subset=[zc, misc, oc, "dprt_lag"])
    for sub, x in (("all", d), ("active", d[d[wc] > 0])):
        y = x[oc].to_numpy(float) * 1e4
        X0 = sm.add_constant(np.column_stack([x[zc], x[misc] * 1e4]))
        X1 = np.column_stack([X0, x["dprt_lag"]])
        r0 = sm.OLS(y, X0).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
        r1 = sm.OLS(y, X1).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
        cz, cm = x["dprt_lag"].corr(x[zc]), x["dprt_lag"].corr(x[misc])
        st = dict(n=len(x), net_bp=float(r1.params[-1]), t_nw=float(r1.tvalues[-1]))
        cell(f"V4_{tag}_{sub}", f"V4 {tag} 시가→종가 회귀 (z, mis) + dprt_{{t−1}} 증분 [{sub}]", "KOSDAQ" if tag == "C3Q" else "KOSPI",
             f"{x.index.min().date()}~{x.index.max().date()}", st, "diagnostic",
             f"dprt_{{t−1}} 계수 {r1.params[-1]:.2f}bp/%p HAC t {r1.tvalues[-1]:.2f}; corr(dprt_{{t−1}}, z_t) {cz:.3f}, corr(dprt_{{t−1}}, mis_t) {cm:.3f}; R² {r0.rsquared:.4f}→{r1.rsquared:.4f}; mis 계수 {r1.params[2]:.3f} t {r1.tvalues[2]:.2f}; z 계수 {r1.params[1]:.2f}bp t {r1.tvalues[1]:.2f}",
             corr_z=cz, corr_mis=cm, r2_base=r0.rsquared, r2_aug=r1.rsquared)
        print(f"V4 {tag} {sub}: n {len(x)} dprt_lag {r1.params[-1]:.2f}bp/%p t {r1.tvalues[-1]:.2f} corr(z) {cz:.3f} corr(mis) {cm:.3f}")


r9 = px(LAB + "/round9_c3_etfs/kq150_etfs/daily_returns.csv")
v4(r9, "z_spx", "mis_229200", "oc_229200", "w_KODEX_vt30", "229200", "C3Q", "2015-10-14")
r5 = px(LAB + "/round5_strategy/c3_levered/daily_returns.csv")
v4(r5, "z_index", "mis", "r_069500_oc", "w_chosen", "069500", "C3VT30", "2005-08-23")
V4_C3Q_T = [r for r in CELLS if r["id"] == "V4_C3Q_all"][0]["t_nw"]

# ------------------------------------------------------------------ V5 60일 z 문턱
wQ5, wK5 = weights(Q, 1.5, "z60"), weights(K, 1.5, "z60")
V5B = book_metrics({"KOSDAQ_A_blog_ON_Q": Q["net"] * wQ5, "KOSPI_A_jehyeong_ON": K["net"] * wK5})
V5K = book_metrics({"KOSPI_A_jehyeong_ON": K["net"] * wK5})
book_cell("V5_overlay_book", "V5 오버레이 w=0.5(z60≥+1.5) 두 다리 장부", V5B, f"게이트 밤 blog_ON_Q {int((wQ5 < 1).sum())} jehyeong_ON {int((wK5 < 1).sum())}", grade="estimate", n=int((wQ5 < 1).sum() + (wK5 < 1).sum()))
book_cell("V5_overlay_1x_book", "V5 오버레이 1배(jehyeong_ON z60≥+1.5)만 장부", V5K, f"게이트 밤 {int((wK5 < 1).sum())}", n=int((wK5 < 1).sum()))
gated_cells(Q, "blog_ON_Q", wQ5, "V5", grade="estimate")
gated_cells(K, "jehyeong_ON", wK5, "V5")
V5 = v2_cells("z60", -1.5, "V5_ind", "V5 독립 229200 z60≤−1.5 종가→익일 종가")
print("V5 overlay:", bm_str(V5B), book_rule(V5B, BASE), "| V5 ind:", {k: (r["n"], round(r["net_bp"], 2), round(r["t_nw"], 2), r["verdict"]) for k, r in V5.items()})

# ------------------------------------------------------------------ V6 NAV 평가 시각(선물 15:30→15:45)
def fut_r(paths):
    f = pd.concat([pd.read_csv(p, parse_dates=["date"]) for p in paths]).sort_values(["date", "time"])
    out = {}
    for d, g in f.groupby("date"):
        pre = g[g["time"] <= "15:29"]
        if len(pre) == 0:
            continue
        out[d] = g["close"].iloc[-1] / pre["close"].iloc[-1] - 1
    return pd.Series(out)


RF = {"KQ": fut_r([RAW + "/intraday/kis_minute/kq150_full_day/futures_1min.csv"]),
      "K2": fut_r([RAW + "/intraday/kis_minute/full_day/futures_1min.csv", RAW + "/intraday/kis_minute/full_day_ext/futures_1min.csv"])}
print(f"V6 선물 15:30→15:45 수익: KQ150 n {len(RF['KQ'])} {RF['KQ'].index.min().date()}~{RF['KQ'].index.max().date()} 평균 {RF['KQ'].mean()*1e4:.1f}bp std {RF['KQ'].std()*1e4:.1f}bp | K200 n {len(RF['K2'])} 평균 {RF['K2'].mean()*1e4:.1f}bp std {RF['K2'].std()*1e4:.1f}bp")
V6 = {}
for code, key in [("233740", "KQ"), ("251340", "KQ"), ("229200", "KQ"), ("122630", "K2"), ("114800", "K2"), ("069500", "K2")]:
    d = pd.DataFrame({"dprt": NAV[code]["dprt"], "rf": RF[key] * 100}).dropna()
    r = sm.OLS(d["dprt"].to_numpy(), sm.add_constant(d["rf"].to_numpy())).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
    L = LEV[code]
    V6[code] = dict(n=len(d), slope=float(r.params[1]), t=float(r.tvalues[1]), expect=-L if code not in ("069500", "229200") else 0, corr=float(d["dprt"].corr(d["rf"])))
    cell(f"V6_slope_{code}", f"V6 dprt(%) ~ 선물 15:30→15:45 수익(%) 기울기 {code}", "KOSDAQ" if key == "KQ" else "KOSPI",
         f"{d.index.min().date()}~{d.index.max().date()}", dict(n=len(d), net_bp=V6[code]["slope"], t_nw=V6[code]["t"]), "diagnostic",
         f"기울기 {V6[code]['slope']:.3f} HAC t {V6[code]['t']:.2f}, corr {V6[code]['corr']:.3f}; NAV 가 15:45 선물로 평가되면 기대 기울기 −L = {V6[code]['expect']}", slope_expect=V6[code]["expect"])
    print(f"V6 {code}: n {len(d)} 기울기 {V6[code]['slope']:.3f} t {V6[code]['t']:.2f} (기대 −L {V6[code]['expect']})")
# 보정 dprt 로 주 셀 재계산(코스닥 선물 창)
Q6 = Q.copy()
rf_t = RF["KQ"].reindex(Q6["t"].values).values
Q6["rf"] = rf_t
Q6["L"] = [LEV[v] for v in Q6["veh"]]
sub = Q6["rf"].notna()
Q6["dprt_adj"] = Q6["dprt"] + Q6["L"] * Q6["rf"] * 100          # dprt = π − L·r_f → π = dprt + L·r_f
Q6["dprt_adj2"] = Q6["dprt"] - Q6["L"] * Q6["rf"] * 100         # 지시문 문자 그대로(차감)
Q6["dprt_adj_fit"] = Q6["dprt"] - np.array([V6[v]["slope"] for v in Q6["veh"]]) * Q6["rf"] * 100   # 실측 기울기로 오염항 제거(보고용)
FW = (str(RF["KQ"].index.min().date()), W_ALL[1])
w_raw = weights(Q6[sub])
w_adj = weights(Q6[sub], col="dprt_adj")
w_adj2 = weights(Q6[sub], col="dprt_adj2")
w_adj3 = weights(Q6[sub], col="dprt_adj_fit")
r_base = win(comb(LEGS_WALL), FW)
for tag, w_ in (("raw", w_raw), ("adj", w_adj), ("adj_literal", w_adj2), ("adj_fit", w_adj3)):
    r_new = win(comb(LEGS_WALL, {"KOSDAQ_A_blog_ON_Q": Q["net"] * w_.reindex(Q.index).fillna(1.0), "KOSPI_A_jehyeong_ON": OV["KOSPI_A_jehyeong_ON"]}), FW)
    m0, m1 = metrics(r_base), metrics(r_new)
    gN = Q6[sub][w_ < 1]
    st = ev_stats(gN["net"] * 1e4, gN["gross"] * 1e4)
    cell(f"V6_overlay_{tag}", f"V6 주 셀 재계산 blog_ON_Q 게이트 dprt {tag} (코스닥 선물 1분봉 창)", "KOSPI+KOSDAQ", f"{FW[0]}~{FW[1]}", st, "estimate",
         f"게이트 밤 {st['n']}/{int(sub.sum())}; 합산 장부 창 {m0['n_days']}일 원 CAGR {m0['cagr']*100:.1f}%/MDD {m0['mdd']*100:.1f}% → {m1['cagr']*100:.1f}%/{m1['mdd']*100:.1f}%; {dict(adj='π = dprt + L·r_f(오염항 제거, 사전 등록 L)', adj_literal='dprt − L·r_f(문자 그대로)', adj_fit='dprt − b̂·r_f(실측 기울기)', raw='원 dprt')[tag]}; 창 ≤ 300 거래일 → estimate",
         cagr_pct=m1["cagr"] * 100, mdd_pct=m1["mdd"] * 100)
    print(f"V6 overlay {tag}: 게이트 {st['n']}/{int(sub.sum())} 장부 {m0['cagr']*100:.1f}/{m0['mdd']*100:.1f} → {m1['cagr']*100:.1f}/{m1['mdd']*100:.1f}")

# ------------------------------------------------------------------ 판정·저장
v2_pass = V2["full"]["verdict"] == "candidate"
g_ok = bool(V2["full"]["g_t"] >= 2)
v4_same = bool(V4_C3Q_T < 2)
main_pass = bool(main_row["book_pass"])
k_pass = bool(k_row["book_pass"])
gate_ok = all(gate[c]["p_stable_2bp_16"] >= 95 for c in ("069500", "229200"))     # 장부 창(2016.10~) 기준 1배 두 종
design_pass = bool(gate_ok and main_pass and v2_pass and g_ok)
if v2_pass:
    ev = ev_full
    exit_idx = next_day.reindex(ev.index).values
    nav_q.loc[first_valid:, ["pos", "gross", "net"]] = 0.0
    nav_q.loc[exit_idx, "pos"] = 1.0
    nav_q.loc[exit_idx, "gross"] = np.exp(ev["lnP"].values) - 1
    nav_q.loc[exit_idx, "net"] = nav_q.loc[exit_idx, "gross"] - COST
    nav_q.to_csv(SER + "/beomgu_NAV_Q.csv", date_format="%Y-%m-%d", float_format="%.8f")
    mm = metrics(nav_q["net"].loc[first_valid:])
    meta = dict(handle="beomgu_NAV_Q", slot="A_close_to_close", market="KOSDAQ", instrument="KODEX 코스닥150 (229200) 1배",
                timing="T 15:30 종가 매수(dprt_T ≤ −0.30) → T+1 15:30 종가 매도, 손익은 T+1 기록(밤+낮, 같은 시장 B·C 와 자본 겹침)",
                cost_bp=5.0, first_valid=str(first_valid.date()), last_valid="2026-09-29", n_active=int(nav_q["pos"].sum()),
                source_files="r15_external/kis_etf_nav_daily.csv(229200)",
                repro={"full_CAGR": mm["cagr"], "full_MDD": mm["mdd"], "n_days": mm["n_days"], "비고": "series 는 단순수익(P_{t+1}/P_t − 1 − 5bp), cells.csv 는 ln 수익"},
                notes=f"dprt_T ≤ −0.30% 이면 종가 P=nav×(1+dprt/100) 매수 → 익일 종가 매도, 5bp. V4 C3Q 증분 HAC t {V4_C3Q_T:.2f} → {'C3 와 같은 원천(오버레이만 유지)' if v4_same else '증분 있음'}")
    json.dump(meta, open(SER + "/beomgu_NAV_Q.meta.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
v3_main = V3["verify_c3q"]
v3_pass = bool(v3_main["ok12"] and not v3_main["placebo_sym"])
if v3_main["ok12"]:   # 검정 가능한 조건 충족 시 교체본 저장(미검정 조건은 passed 플래그로 전달)
    lines = open(f"{R13}/series/beomgu_C3Q.csv", encoding="utf-8").read().splitlines()
    rep_map = {d.strftime("%Y-%m-%d"): (C3Q["gross"].loc[d] + p * r, n) for d, p, r, n in zip(rep.index, rep["c3q_pos"], rep["rel"], c3q_new.loc[rep.index])}
    out_lines = [lines[0]]
    n_rep = 0
    for ln in lines[1:]:
        d = ln.split(",")[0]
        if d in rep_map:
            g_, n_ = rep_map[d]
            out_lines.append(f"{d},{ln.split(',')[1]},{g_:.10g},{n_:.10g}")      # pos 는 원본 문자열 그대로
            n_rep += 1
        else:
            out_lines.append(ln)
    assert n_rep == len(rep)
    open(SER + "/beomgu_C3Q_v3.csv", "w", encoding="utf-8").write("\n".join(out_lines) + "\n")
    meta3 = dict(C3Q_META)
    mm3 = metrics(c3q_new)
    n_self = int((rep["vmin"] == "229200").sum())
    meta3.update(handle="beomgu_C3Q_v3", instrument="코스닥150 1배 ETF 중 시가 최저 괴리 차량(자격: 20일 거래대금 중앙 ≥ 10억, 자격 ≥ 3, 그룹 중앙 − 최저 ≥ 10bp), 그 외 날 229200",
                 source_files=["round13_slot_book/series/beomgu_C3Q.csv", "etf_extra/<8종>.csv", "KQ150_index_kis.csv"], n_replaced=int(len(rep)), n_self_229200=n_self,
                 repro={"W_all": {k: V3B["W_all"][k] for k in ("cagr", "mdd")}, "원본_대비": f"바꾸지 않은 행은 원본과 동일(값이 바뀐 행 {len(rep) - n_self})"},
                 full_valid_stats={"start": C3Q_META["first_valid"], "end": C3Q_META["last_valid"], "n_days": mm3["n_days"], "n_active": int((C3Q["pos"] != 0).sum()),
                                   "cum": mm3["cum"], "cagr": mm3["cagr"], "sharpe": mm3["sharpe"], "mdd": mm3["mdd"]},
                 notes=f"V3: C3Q 활성일 ∩ 신호일 {len(rep)}일(그중 229200 자체가 최저 {n_self}일) net += pos×(oc_선택차량 − oc_229200). 단일가 체결량 10% 조건 미검정(자료 없음) → passed=false 전달. 위약 {'대칭' if v3_main['placebo_sym'] else '비대칭'}")
    json.dump(meta3, open(SER + "/beomgu_C3Q_v3.meta.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)

cells = pd.DataFrame(CELLS)
cells.to_csv(OUT + "/cells.csv", index=False, float_format="%.6g")
pd.DataFrame(TRADES).to_csv(OUT + "/trades.csv", index=False, float_format="%.8g")

# ------------------------------------------------------------------ 그림
fig, ax = plt.subplots(figsize=(9, 5.5))
for i, c in enumerate(["229200", "233740", "251340", "069500"]):
    g = NAV[c].loc[:"2026-09-29"].dropna(subset=["g"])
    q = pd.qcut(g["dprt"], 10, duplicates="drop")
    b = g.groupby(q, observed=True).agg(x=("dprt", "mean"), y=("g", lambda s: s.mean() * 1e4), n=("g", "size"))
    ax.plot(b["x"], b["y"], marker="o", ms=5, lw=2, color=C[i], label=f"{c} (n {len(g):,})")
ax.axhline(0, color="#9a9994", lw=1)
ax.axvline(THR, color="#e34948", lw=1, ls="--")
ax.axvline(-THR, color="#e34948", lw=1, ls="--")
ax.set_xlabel("종가 괴리율 dprt_t (%), 10분위 평균")
ax.set_ylabel("익일 괴리 축소분 g_t (bp)  = ln((1+dprt_t+1)/(1+dprt_t))")
ax.set_title("dprt 분위별 익일 ETF - NAV 수익 (KIS 공식 NAV, 상장~2026.9.29, 비용 0)", fontsize=11, loc="left")
ax.legend(frameon=False, fontsize=9)
ax.grid(alpha=0.25)
for s_ in ("top", "right"):
    ax.spines[s_].set_visible(False)
fig.tight_layout()
fig.savefig(OUT + "/fig1_dprt_decile_g.png", dpi=130)
plt.close(fig)

fig, axes = plt.subplots(2, 1, figsize=(10, 7.5))
for ax, (lab, legs, w) in zip(axes, [("W_all 합산", LEGS_WALL, W_ALL), ("W_long 두 장부 합산", LEGS_LONG, W_LONG)]):
    for i, (nm, o) in enumerate([("원 장부", None), ("오버레이(주 셀, 두 다리)", OV), ("오버레이 1배만(jehyeong_ON)", {"KOSPI_A_jehyeong_ON": OV["KOSPI_A_jehyeong_ON"]})]):
        r = win(comb(legs, o), w)
        mm = metrics(r)
        ax.plot(r.index, (1 + r).cumprod(), lw=1.6, color=C[i], label=f"{nm}: CAGR {mm['cagr']*100:.1f}% MDD {mm['mdd']*100:.1f}%")
    ax.set_title(f"{lab} {w[0]}~{w[1]}: 프리미엄 게이트 오버레이 vs 원 장부", fontsize=11, loc="left")
    ax.set_ylabel("누적")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.grid(alpha=0.25)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
fig.tight_layout()
fig.savefig(OUT + "/fig2_overlay_vs_book.png", dpi=130)
plt.close(fig)

fig, ax = plt.subplots(figsize=(10, 4.5))
ev = V2["full"]["ev"]
eq = (1 + (np.exp(ev["lnP"]) - 1 - COST)).cumprod()
ax.plot(eq.index, eq, lw=1.8, color=C[0], label=f"V2 229200 dprt <= -0.30 종가→익일 종가, 5bp (n {len(ev)})")
ax.plot(ev.index, np.exp(ev["g"].cumsum()), lw=1.6, color=C[2], label="괴리 축소분 g 누적(비용 0)")
ax.plot(ev.index, np.exp(ev["lnNAV"].cumsum()), lw=1.4, color=C[1], label="NAV 수익 누적(지수 노출)")
ax.axvline(pd.Timestamp("2020-01-02"), color="#9a9994", lw=1, ls="--")
ax.annotate("설계 | 검증", (pd.Timestamp("2020-01-02"), ax.get_ylim()[1] * 0.98 if ax.get_ylim()[1] > 1 else 1.0), fontsize=9, color="#52514e", va="top")
ax.set_title("V2 독립 셀 분해 (2015.10~2026.9, 이벤트일 누적)", fontsize=11, loc="left")
ax.legend(frameon=False, fontsize=9, loc="upper left")
ax.grid(alpha=0.25)
for s_ in ("top", "right"):
    ax.spines[s_].set_visible(False)
fig.tight_layout()
fig.savefig(OUT + "/fig3_v2_decomp.png", dpi=130)
plt.close(fig)

# ------------------------------------------------------------------ notes.md
def f1(x, nd=1, sign=False):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    return (f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}").replace("-", "−")


def dd(s):  # 2026-09-17 → 2026.9.17
    y, m, d = str(s)[:10].split("-")
    return f"{y}.{int(m)}.{int(d)}"


def wtxt(w):
    return f"{dd(w[0])}~{dd(w[1])}"


def bm_note(m):
    return (f"W_all CAGR {f1(m['W_all']['cagr']*100, 2)}% / MDD {f1(m['W_all']['mdd']*100, 2)}% / 샤프 {f1(m['W_all']['sharpe'], 2)}, "
            f"W_ex2026 {f1(m['W_ex2026']['cagr']*100, 2)}% / {f1(m['W_ex2026']['mdd']*100, 2)}%, W_long {f1(m['W_long']['cagr']*100, 2)}% / {f1(m['W_long']['mdd']*100, 2)}%, "
            f"2024 {f1(m['Y2024']['cum']*100, 2, True)}%, 2025 {f1(m['Y2025']['cum']*100, 2, True)}%, 2026(~9.17) {f1(m['Y2026']['cum']*100, 2, True)}%")


def cs(r):
    return (f"n {r['n']} 순 {f1(r['net_bp'], 2)}bp(총 {f1(r['gross_bp'], 2)}bp) NW t {f1(r['t_nw'], 2)} 승률 {f1(r['win_pct'])}% 연도 {r['yearly_pos']} "
            f"상위5 {f1(r['top5_share_pct'])}% 전·후반 {f1(r['half1_net_bp'])}/{f1(r['half2_net_bp'])}bp Δ {f1(r['delta_bp'])}bp t {f1(r['t_delta'], 2)} "
            f"CAGR {f1(r['cagr_pct'], 2)}% MDD {f1(r['mdd_pct'], 2)}% → {r['verdict']}")


V = {r["id"]: r for r in CELLS}
L = []
L.append("# 03_nav_dprt · ETF 종가 공식 NAV 괴리율 → 밤 다리 비중 오버레이 + 독립 셀 (15차, 2026.10.30)")
L.append("")
L.append("## 규약")
L.append(f"• 자료 kis_etf_nav_daily.csv {n_raw:,}행 → nav ≤ 0 제외 {int((nav_raw['nav'] <= 0).sum())}행(069500 2004.7~2005.8), |dprt| > 10 제외 {int((nav_raw['dprt'].abs() > 10).sum())}행(069500 2007.8.31 1행, 114800·122630 2011.5.26~6.3 각 7행, 252670 2026.7.31 1행), 2026.10.30 행 {int((nav_raw['date'] > LAST).sum())}행 제외(nav 가 10.29 와 같음, 장중 미확정). 남은 {len(nav):,}행. KIS 날짜는 069500 달력(KRX 거래일) 안에 전부 포함(2026.9.29 까지), 229200 2015.10.13~2026.9.29 결측 0.")
L.append("• 종가 P = nav × (1 + dprt/100), g_t = ln((1 + dprt_t+1/100)/(1 + dprt_t/100)). 밤 다리 series 의 date 는 청산일이라 진입일 t = 직전 거래일의 dprt 로 w 를 정해 청산일 행에 곱함(코드 leg_nights). 비용 ETF 왕복 5bp, 오버레이 비용은 w 비례. 단일가 체결 가정(15:30 종가·09:00 시가)에 전부 의존.")
L.append("• 통계 이벤트별 순수익 bp 상수 회귀 HAC(maxlags 5) NW t, Δ 더미 회귀 HAC t. CAGR·MDD 069500 거래일 달력 비활성일 0. 판정 규칙 사전 등록 그대로(결과 뒤 문턱·창·차량 변경 없음).")
L.append(f"• 13차 장부 재현(k=1): {bm_note(BASE)} (원수치 31.4%/−6.1%, 15.0%/−9.6%, 2024 +11.2%, 2025 +11.4% 와 일치).")
L.append(f"• 돌린 셀 수 {len(cells)} (cells.csv 전부 기록, 다중 비교 부담 그대로).")
L.append("")
L.append("## 자료 점검·전제 게이트")
L.append("• KRX 공식 괴리율 대조: 로그인 필요라 불가(사전 등록대로 대체 점검).")
for c in ["069500", "229200", "233740", "251340", "122630"]:
    gg = gate[c]
    L.append(f"• 게이트① {c}: 네이버 일봉 조정 종가 / (nav×(1+dprt/100)) 의 일별 로그 변화 n {gg['n']:,}, |변화| < 2bp 비율 {f1(gg['p_stable_2bp'])}%, 중앙 {f1(gg['med_abs']*1e4, 2)}bp, 점프(> 10bp) {gg['n_jump']}일(최근 {', '.join(dd(x) for x in gg['jump_dates'])}), KIS close_adj 기준 점프 {gg['n_jump_kisadj']}일; 2016.10.4 이후 n {gg['n16']:,} 안정 비율 {f1(gg['p_stable_2bp_16'])}% 점프 {gg['n_jump_16']}일; 연도별 점프 {gg['jump_by_year']}.")
L.append("• 게이트① 해석: 2016.10 이후 점프는 069500 연 4회(1·4·7·10월 말 분배락)·229200 연 1회(4월 말 분배락)에 몰리고 그 밖의 날은 비율이 상수(중앙 0.3~0.7bp = 호가·반올림) → nav×(1+dprt/100) 가 당시 종가와 정합. 069500 2005~2011·122630 2010~2011 의 점프 다수는 네이버 조정 이력 차이로 보이며 장부 창(2016.10~) 밖. KIS close_adj 기준 점프 수가 같은 것은 두 조정가가 같은 분배락을 반영한다는 뜻.")
if naver:
    L.append(f"• 게이트② 네이버 당일 괴리율 대조: {naver}")
else:
    L.append("• 게이트② 네이버 금융 ETF 페이지(finance.naver.com main.naver, m.stock.naver.com basic API, etfItemList API) WebFetch 전부 실패(호스트 차단) → 생략. 레버리지·인버스(233740·251340·122630) dprt 는 외부 대조 없음 → 사전 규칙대로 그 셀은 estimate, 1배(229200·069500) 셀만 판정.")
L.append(f"• 정합: blog_ON_Q gross = 차량(233740/251340) 종가→익일 시가 수익(최대 오차 1e−6 미만, 820밤), jehyeong_ON gross vs 069500 같은 구간 중앙 오차 {f1((chk_k - K['gross']).abs().median()*1e4, 2)}bp(출처 가격 다름). 청산일 = 진입일 + 1 거래일 정렬 확인.")
L.append(f"• 발생 빈도: blog_ON_Q 활성 {len(Q)}밤 중 차량 dprt ≥ +0.30 {int((Q['dprt'] >= THR).sum())}밤, ≤ −0.30 {int((Q['dprt'] <= -THR).sum())}밤, 결측 {int(Q['dprt'].isna().sum())}; jehyeong_ON {len(K)}밤 중 ≥ +0.30 {int((K['dprt'] >= THR).sum())}, ≤ −0.30 {int((K['dprt'] <= -THR).sum())}.")
L.append(f"• V6 선물 15:30→15:45 수익(1분봉): 코스닥150 n {len(RF['KQ'])} {dd(RF['KQ'].index.min())}~{dd(RF['KQ'].index.max())} 평균 {f1(RF['KQ'].mean()*1e4)}bp 표준편차 {f1(RF['KQ'].std()*1e4)}bp, 코스피200 n {len(RF['K2'])} 평균 {f1(RF['K2'].mean()*1e4)}bp 표준편차 {f1(RF['K2'].std()*1e4)}bp. 봉 라벨 = 봉 시작, 15:29 봉 종가 → 15:45 종가 단일가 봉.")
L.append("")
L.append("## 주 셀 (프리미엄 게이트 오버레이)")
L.append(f"• 두 다리 적용(사전 등록 주 셀): {bm_note(M_BOTH)} → 결합 규칙 {'통과' if main_pass else '미달'}. 등급 estimate(233740·251340 dprt 외부 대조 없음).")
L.append(f"• 1배 부분만(jehyeong_ON 069500 게이트 {int((wK < 1).sum())}밤): {bm_note(M_K)} → 결합 규칙 {'통과' if k_pass else '미달'}.")
L.append(f"• 레버리지·인버스 부분만(blog_ON_Q 게이트 {int((wQ < 1).sum())}밤): {bm_note(M_Q)} → 결합 규칙 {'통과' if q_row['book_pass'] else '미달'} (estimate).")
for id_ in ["M_blog_ON_Q_full", "M_blog_ON_Q_W_all", "M_jehyeong_ON_full", "M_jehyeong_ON_W_all"]:
    r = V[id_]
    L.append(f"• {r['name']} {wtxt(r['window'].split('~'))}: 게이트 밤 {cs(r)} · {r['notes'].split('. ')[0]}")
for s_ in NOTE:
    L.append("• " + s_)
L.append("")
L.append("## 변형")
L.append(f"• V1 할인 밤 1.5: jehyeong_ON 할인 밤 {cs(V['V1_disc_jehyeong_ON'])} ({V['V1_disc_jehyeong_ON']['notes']}); blog_ON_Q 할인 밤 {cs(V['V1_disc_blog_ON_Q'])}.")
L.append(f"• V1a 장부(코스피만 ETF 실현): {bm_note(V1a)} → 결합 규칙 {'통과' if V['V1a_book']['book_pass'] else '미달'}. V1b(코스닥 1.5 scaled 포함, 선물 필요·보고만): {bm_note(V1b)} → {'통과' if V['V1b_book']['book_pass'] else '미달'}.")
for wn in V2_WIN:
    r = V2[wn]
    L.append(f"• V2 독립 229200 dprt ≤ −0.30 [{wn} {wtxt(V2_WIN[wn])}]: {cs(r)}; g 평균 {f1(r['g_bp'], 2)}bp NW t {f1(r['g_t'], 2)}, NAV 수익 평균 {f1(r['ev']['lnNAV'].mean()*1e4)}bp, 중앙 {f1(r['median_bp'])}bp.")
L.append(f"• V2 진단(체결가를 네이버 조정 종가로): {cs(V['V2diag_adjclose_full'])}.")
for wn in V3_WIN:
    r, ra = V3[wn], V3[wn + "_c3q"]
    L.append(f"• V3 [{wn} {wtxt(V3_WIN[wn])}] 모든 신호일: {cs(r)} | 중앙 {f1(r['median_bp'])}bp, 위약 {f1(r['placebo_bp'])}bp t {f1(r['placebo_t'], 2)} {'대칭' if r['placebo_sym'] else '비대칭'} | C3Q 활성 ∩ 신호: {cs(ra)} | 중앙 {f1(ra['median_bp'])}bp, 위약 {f1(ra['placebo_bp'])}bp {'대칭' if ra['placebo_sym'] else '비대칭'}.")
L.append(f"• V3 선택 차량 비중 {', '.join(f'{k} {v*100:.0f}%' for k, v in v3['vmin'].value_counts(normalize=True).items())}, 자격 차량 평균 {f1(v3['n_elig'].mean(), 2)}개, 신호일 {len(v3)} 중 C3Q 활성 {len(rep)}일(229200 자체가 최저인 날 {int((rep['vmin'] == '229200').sum())}일은 변화 없음). 거래대금 = 종가 × 거래량 근사(파일에 거래대금 열 없음). 교체 장부: {bm_note(V3B)} → 결합 규칙 {'통과' if V['V3_book']['book_pass'] else '미달'}. 단일가 체결량 10% 조건은 운용사별 단일가 체결량 자료가 없어 미검정 → 12차 통과 조건 전체 충족 불가.")
for id_ in ["V4_C3Q_all", "V4_C3Q_active", "V4_C3VT30_all", "V4_C3VT30_active"]:
    r = V[id_]
    L.append(f"• {r['name']} {wtxt(r['window'].split('~'))}: n {r['n']}, {r['notes']}")
L.append(f"• V4 판정: C3Q 전체일 dprt_t−1 증분 HAC t {f1(V4_C3Q_T, 2)} → {'< 2 → C3 와 같은 원천으로 분류, 오버레이만 유지(V2 독립 다리 추가 안 함)' if v4_same else '≥ 2 → 증분 있음'}.")
L.append(f"• V5 60일 z 문턱 오버레이(z60 ≥ +1.5 → 0.5): {bm_note(V5B)} → {'통과' if V['V5_overlay_book']['book_pass'] else '미달'}(estimate); 1배만 {bm_note(V5K)} → {'통과' if V['V5_overlay_1x_book']['book_pass'] else '미달'}.")
for wn in V2_WIN:
    r = V5[wn]
    L.append(f"• V5 독립 229200 z60 ≤ −1.5 [{wn}]: {cs(r)}; g {f1(r['g_bp'], 2)}bp t {f1(r['g_t'], 2)}.")
for code in ["233740", "251340", "229200", "122630", "114800", "069500"]:
    r = V6[code]
    L.append(f"• V6 기울기 dprt(%) ~ 선물 15:30→15:45 수익(%) {code}: n {r['n']}, 기울기 {f1(r['slope'], 3)} HAC t {f1(r['t'], 2)}, corr {f1(r['corr'], 3)} (15:45 평가면 기대 {f1(float(r['expect']), 0)}).")
for tag in ("raw", "adj", "adj_literal", "adj_fit"):
    r = V[f"V6_overlay_{tag}"]
    L.append(f"• V6 주 셀 재계산 {tag} {wtxt(FW)}: {r['notes']}")
L.append("")
L.append("## 판정")
L.append(f"• 전제 게이트: ① {'통과' if gate_ok else '미달'}(1배 두 종 2016.10.4 이후 |Δln 비율| < 2bp 비율 ≥ 95% 기준: 069500 {f1(gate['069500']['p_stable_2bp_16'])}%, 229200 {f1(gate['229200']['p_stable_2bp_16'])}%), ② 생략(네이버 차단) → 1배 셀만 판정, 레버리지·인버스 셀 estimate.")
L.append(f"• 주 셀 결합 규칙: 두 다리 {'통과' if main_pass else '미달'}(estimate), 1배만 {'통과' if k_pass else '미달'}.")
L.append(f"• V2 공통 규칙 [full]: {V2['full']['verdict']}; g NW t {f1(V2['full']['g_t'], 2)} {'≥ 2' if g_ok else '< 2'}; 검증 창 {V2['verify']['verdict']}.")
L.append(f"• V4: {'C3 와 같은 원천' if v4_same else '증분 있음'} (t {f1(V4_C3Q_T, 2)}).")
L.append(f"• V3: 검증 창 C3Q 활성 ∩ 신호 12차 조건 {'충족' if v3_main['ok12'] else '미달'}, 위약 {'대칭' if v3_main['placebo_sym'] else '비대칭'}, 체결량 조건 미검정 → {'reference(교체본 저장, passed=false)' if v3_main['ok12'] else '미저장'}.")
L.append(f"• 설계 종합(게이트 ∧ 주 셀 결합 ∧ V2 공통 ∧ g t ≥ 2): {'통과' if design_pass else '미달'}.")
L.append("")
L.append("## 단서")
L.append("• 모든 체결은 단일가 가정(15:30 종가·09:00 시가). 오버레이는 추가 매매 없이 비중만 바꾸므로 의존도는 원 다리와 같음. V2·V3 는 체결가 자체가 신호라 의존 강함.")
L.append("• 레버리지·인버스 dprt 수준은 외부 대조 없음(V6 기울기로 15:45 평가 여부만 점검). 1배 dprt 도 KRX 공식치와 미대조(네이버 종가 일치 표본 8일은 조사 단계).")
L.append("• V2 는 밤+낮 보유라 같은 시장 B·C 다리와 자본이 겹침. V4 가 '같은 원천' 이면 다리로 더하지 않음.")
L.append("• 다중 비교: 이 설계 안 셀 수 위 참조, 15차 사전 등록 64셀, 14차 누적 512셀.")
L.append("• 2026 의존: W_ex2026 수치 병기. 2026.10.30 행 제외.")
txt = "\n".join(L) + "\n"
txt = re.sub(r"(\d{4})-(\d{2})-(\d{2})", lambda m_: f"{m_[1]}.{int(m_[2])}.{int(m_[3])}", txt).replace("-", "−")   # 하이픈 금지(날짜는 점, 음수는 −)
assert "-" not in txt, "notes.md 에 하이픈"
open(OUT + "/notes.md", "w", encoding="utf-8").write(txt)
json.dump(dict(gate_ok=gate_ok, main_pass=main_pass, k_pass=k_pass, v2_pass=v2_pass, g_ok=g_ok, v4_same=v4_same, v3_ok12=bool(v3_main["ok12"]),
               v3_placebo_sym=bool(v3_main["placebo_sym"]), design_pass=design_pass, n_cells=int(len(cells)), V4_C3Q_T=V4_C3Q_T,
               base=BASE, main=M_BOTH, main_1x=M_K, main_lev=M_Q, V1a=V1a, V1b=V1b, V3B=V3B, V5B=V5B),
          open(OUT + "/summary.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
print(f"판정: gate {gate_ok} main {main_pass} 1x {k_pass} V2 {v2_pass} g {g_ok} V4same {v4_same} V3ok12 {bool(v3_main['ok12'])} sym {bool(v3_main['placebo_sym'])} → design {design_pass}; cells {len(cells)}")
