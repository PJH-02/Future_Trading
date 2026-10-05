"""R5 · c3_levered: D2 잔차 갭 신호(C3/C2)를 바꾸지 않고 실행 차량·크기·필터 1개로 연 10% 근처를 노린다 (5차 전략 탐색)

신호(변경 없음): bt_d2_2010_2026.signals() 의 R3-C2(지수 갭 잔차 z ≥ 1), R3-C3(C2 + KODEX 시가가 지수 대비 싸게 열림).
  C3 ⊂ C2 이므로 'C3 ∪ C2' = C2. 계층(TIER) = C3 날 노출 2배, C2 전용 날 노출 1배.
매매: T 09:00 시가 단일가 매수 → T 15:30 종가 단일가 매도(1일 1회, 롱 전용). 신호도 09:00 시가를 쓴다(기존 가정 그대로).
  실거래 가능성은 아래 '분봉 실행 점검'에서 09:01·09:02·09:05·09:10 진입으로 따로 잰다.
노출 w (자본 대비 배수, 0~2):
  1x: w = 신호 / lev: w = 2·신호 / tier: w = C2 + C3
  zlin: w = 신호·clip(z_index, 1, 2) / zstep: w = 신호·(1 + 1[z_index ≥ 1.5])
  vt20·vt30: w = 신호·clip(목표 / σ̂, 0, 2), σ̂ = 코스피200 일간 종가수익 직전 20일 SD×√252 (T−1 종가까지)
  필터(T−1 종가까지 정보, 코스피200 지수 종가): ma20up 종가 > MA20 / ma60up 종가 > MA60 / noup2 전일 수익 ≤ +2% / ma20dn 종가 < MA20(눌림)
차량:
  ETF(주식계좌): w ≤ 1 → 069500 비중 w, 1 < w ≤ 2 → 069500 (2−w) + 122630 (w−1). 수익 = 각 ETF 시가→종가. 비용 = 비중 × 왕복 5bp(스트레스 069500 7bp, 122630 8bp)
  IDX(미니 코스피200 선물 프록시): 수익 = w × 코스피200 지수 시가→종가, 비용 = w × 2bp(스트레스 3bp).
    ETF 시가 할인 몫을 잃고, 지수 시가를 같은 지수 시가에 체결한다고 두므로 선물 실체결과 다를 수 있다(아래 진단).
  2003~2009 추가 표본 외: 122630 이 없어 2x KODEX 경로 합성(lev = (1 + 2·cc)/(1 + 2·gap) − 1, KODEX 수정주가)으로 대체.
구간(사전 고정): 설계 2010-02-22(122630 상장, 모든 변형 공통 시작)~2019-12-31 / 검증 2020-01-01~ 마지막 신호일(미국 종가 자료 끝)
  / 검증 중 2025-06-05 까지(최근 급등 제외) / 2025-06-09~ / 추가 표본 외 2003~2009(지수·KODEX 만).
  주의: C2·C3 신호 자체는 2차(2020~2025-06 B 구간)·3차(2003~2009)에서 이미 본 신호다. 여기서 새로 고르는 것은 차량·크기·필터뿐이다.
그리드(42개): A 차량·계층 10, B z 크기 8, C 변동성 목표 8, D 필터 16 (전부 grid.csv 에 기록).
최종 선택 규칙(설계 지표만, 실행 전 고정):
  후보 = 차량 ETF(주식계좌로 실행 가능한 069500/122630 조합). IDX 변형은 파생계좌가 필요하고 지수 시가 편향이 있어 후보에서 빼고 참고로만 보고한다.
  조건: 설계 거래 ≥ 15회/년, 설계 MDD ≥ −25%.
  1순위: 조건을 만족하고 설계 CAGR ≥ 10% 인 후보 중 설계 샤프 최대. 없으면 조건을 만족하는 후보 중 설계 CAGR 최대.
  동률(샤프 소수 둘째 자리까지 같음)이면 더 단순한 쪽(필터 없음 > 크기 규칙 없음). 2·3위는 같은 순서의 다음 변형. 검증 성과로는 고르지 않는다.
분봉 실행 점검: KODEX 200 1분봉(끝시각 라벨, 09:01 봉 = 09:00:00~09:00:59), 2023-01-02~2026-09-29 신호일.
  진입 = 해당 시각 이후 첫 체결(거래량 > 0 인 첫 봉의 시가; 09:01 진입 = 09:02 라벨 봉), 청산 = 15:30 봉 종가(종가 단일가).
  레버리지 경로는 2x KODEX 경로로 근사: (1 + 2·cc)/(1 + 2·(K_t/K_전일종가 − 1)) − 1.
실행: python topics/r5_c3_levered.py [--no-ledger]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import data  # noqa: E402
import engine as E  # noqa: E402
import r3_d2 as R3  # noqa: E402
import bt_d2_2010_2026 as BT  # noqa: E402

FAMILY = "c3_levered"
TOPIC = "R5_C3_LEVERED"
OUTD = data.OUT / "round5_strategy" / FAMILY
MIN1 = data._dir("HYFE_TEAM_REPO", "team_repo_dir", "../..") / "woohyun" / "data" / "KODEX200_1min_2023_2026.csv"   # 팀 저장소 루트: config team_repo_dir, 저장소 안에서는 ../..
ANN = 252
LEV_LISTED = "2010-02-22"
COST = {"K": (5.0, 7.0), "L": (5.0, 8.0), "I": (2.0, 3.0)}      # 왕복 bp (기준, 스트레스)
MIN_TPY, MIN_MDD, TARGET_CAGR = 15.0, -0.25, 0.10
ENTRY_CLOCKS = {"09:00 단일가": None, "09:00 단일가(2x 합성)": "SYN", "09:01": "09:02", "09:02": "09:03", "09:05": "09:06", "09:10": "09:11"}   # 시각 → 첫 라벨 봉


# ---------------------------------------------------------------- 자료
def load():
    px, sig, feats = BT.signals()                     # px = 069500(달력 cal 재색인), 신호는 기존 정의 그대로
    _, ix, _, cal, _ = R3.load()
    lev = data.price("122630").reindex(cal)
    fut = data.price("FUT").reindex(cal)
    d = pd.DataFrame(index=cal)
    d["k_o"], d["k_c"] = px["open"], px["close"]
    d["r_k"] = px["close"] / px["open"] - 1
    d["r_l"] = lev["close"] / lev["open"] - 1
    d["r_i"] = ix["close"] / ix["open"] - 1
    d["r_f"] = fut["close"] / fut["open"] - 1
    d["k_cc"] = px["close"] / px["close"].shift(1) - 1
    gap = px["open"] / px["close"].shift(1) - 1
    d["k_gap"] = gap
    d["r_ls"] = (1 + 2 * d["k_cc"]) / (1 + 2 * gap) - 1           # 2x KODEX 경로 합성
    d["C2"], d["C3"] = sig["R3-C2"], sig["R3-C3"]
    d["z"], d["mis"] = feats["z_index"], feats["mis"]
    icc = ix["close"] / ix["close"].shift(1) - 1
    d["i_cc"] = icc
    d["sig_hat"] = icc.rolling(20).std().shift(1) * np.sqrt(ANN)
    ma20, ma60 = ix["close"].rolling(20).mean(), ix["close"].rolling(60).mean()
    d["F_ma20up"] = (ix["close"] > ma20).astype(float).where(ma20.notna()).shift(1)
    d["F_ma60up"] = (ix["close"] > ma60).astype(float).where(ma60.notna()).shift(1)
    d["F_noup2"] = (icc <= 0.02).astype(float).where(icc.notna()).shift(1)
    d["F_ma20dn"] = (ix["close"] < ma20).astype(float).where(ma20.notna()).shift(1)
    # 평가 표본: 신호·KODEX·지수 시가→종가가 모두 있는 날(2003~). 2010-02-22 이후에는 122630 도 있어야 함
    ok = d["C2"].notna() & d["C3"].notna() & d["r_k"].notna() & d["r_i"].notna() & (d.index >= "2003-01-01")
    ok &= (d.index < LEV_LISTED) | d["r_l"].notna()
    return d[ok].copy()


# ---------------------------------------------------------------- 변형
def variants():
    V = []
    for s in ("C3", "C2"):                             # A: 차량
        V += [dict(grp="A", sig=s, size="1x", veh="ETF", filt=None), dict(grp="A", sig=s, size="lev", veh="ETF", filt=None),
              dict(grp="A", sig=s, size="1x", veh="IDX", filt=None), dict(grp="A", sig=s, size="lev", veh="IDX", filt=None)]
    V += [dict(grp="A", sig="TIER", size="tier", veh="ETF", filt=None), dict(grp="A", sig="TIER", size="tier", veh="IDX", filt=None)]
    for s in ("C3", "C2"):                             # B: z 크기
        for z in ("zlin", "zstep"):
            for v in ("ETF", "IDX"):
                V.append(dict(grp="B", sig=s, size=z, veh=v, filt=None))
    for s in ("C3", "C2"):                             # C: 변동성 목표
        for t in ("vt20", "vt30"):
            for v in ("ETF", "IDX"):
                V.append(dict(grp="C", sig=s, size=t, veh=v, filt=None))
    for f in ("ma20up", "ma60up", "noup2", "ma20dn"):  # D: 필터 1개
        V += [dict(grp="D", sig="C3", size="lev", veh="ETF", filt=f), dict(grp="D", sig="C2", size="lev", veh="ETF", filt=f),
              dict(grp="D", sig="C3", size="lev", veh="IDX", filt=f), dict(grp="D", sig="C3", size="1x", veh="ETF", filt=f)]
    for v in V:
        v["name"] = f"{v['sig']}|{v['size']}|{v['veh']}" + (f"|{v['filt']}" if v["filt"] else "")
    assert len(V) == len({v["name"] for v in V}) == 42
    return V


def exposure(d, v):
    """결정일 T 의 자본 대비 노출 w(0~2)."""
    if v["sig"] == "TIER":
        b = d["C2"]
    else:
        b = d[v["sig"]]
    if v["filt"]:
        b = b * d[f"F_{v['filt']}"].fillna(0)
    sz = v["size"]
    if sz == "1x":
        w = b
    elif sz == "lev":
        w = 2 * b
    elif sz == "tier":
        w = b + d["C3"] * b
    elif sz == "zlin":
        w = b * d["z"].clip(1, 2)
    elif sz == "zstep":
        w = b * (1 + (d["z"] >= 1.5).astype(float))
    elif sz in ("vt20", "vt30"):
        tgt = {"vt20": 0.20, "vt30": 0.30}[sz]
        w = b * (tgt / d["sig_hat"]).clip(0, 2).fillna(1.0)
    return w.fillna(0).astype(float)


def daily_net(d, w, veh, stress=False, lev_col="r_l"):
    """자본 대비 일 순수익(무포지션 0)과 총수익."""
    j = 1 if stress else 0
    if veh == "IDX":
        gross = w * d["r_i"]
        cost = w * COST["I"][j] / 1e4
    else:
        a = np.where(w <= 1, w, 2 - w)
        bl = np.where(w <= 1, 0.0, w - 1)
        rl = d[lev_col].where(d.index >= LEV_LISTED, d["r_ls"]) if lev_col == "r_l" else d[lev_col]
        gross = a * d["r_k"] + bl * rl.fillna(0)
        cost = (a * COST["K"][j] + bl * COST["L"][j]) / 1e4
    return (gross - cost), gross


def metrics(net, w, gross=None):
    net = net.dropna()
    w = w.reindex(net.index).fillna(0)
    tr = net[w > 0]
    n = len(net)
    if n == 0:
        return {}
    eq = (1 + net).cumprod()
    yrs = n / ANN
    cagr = eq.iloc[-1] ** (1 / yrs) - 1 if eq.iloc[-1] > 0 else np.nan
    mdd = (eq / eq.cummax() - 1).min()
    sd = net.std()
    return {"start": str(net.index.min().date()), "end": str(net.index.max().date()), "days": n, "trades": int((w > 0).sum()),
            "trades_yr": (w > 0).sum() / yrs, "exposure": (w > 0).mean(), "avg_w": w[w > 0].mean() if (w > 0).any() else np.nan,
            "win": (tr > 0).mean() if len(tr) else np.nan, "net_bp": tr.mean() * 1e4 if len(tr) else np.nan,
            "gross_bp": gross.reindex(tr.index).mean() * 1e4 if gross is not None and len(tr) else np.nan,
            "t_net": E.nw_t(tr.to_numpy()) if len(tr) >= 10 else np.nan, "cum": eq.iloc[-1] - 1, "cagr": cagr,
            "sharpe": net.mean() / sd * np.sqrt(ANN) if sd > 0 else np.nan, "mdd": mdd, "calmar": cagr / -mdd if mdd < 0 else np.nan}


def bench(d_all_close, a, b):
    """KODEX 200 매수 보유(같은 구간 달력 전체, 비용 없음)."""
    c = d_all_close.loc[:b]
    r = (c / c.shift(1) - 1).loc[a:b].dropna()
    eq = (1 + r).cumprod()
    yrs = len(r) / ANN
    cagr = eq.iloc[-1] ** (1 / yrs) - 1
    mdd = (eq / eq.cummax() - 1).min()
    return {"bh_cagr": cagr, "bh_mdd": mdd, "bh_sharpe": r.mean() / r.std() * np.sqrt(ANN)}


# ---------------------------------------------------------------- 분봉 실행 점검
def load_min1():
    m = pd.read_csv(MIN1)
    t = pd.to_datetime(m["datetime"])
    t = t.dt.tz_convert("Asia/Seoul").dt.tz_localize(None) if t.dt.tz is not None else t
    m["d"], m["hm"] = t.dt.normalize(), t.dt.strftime("%H:%M")
    m = m[(m["hm"] >= "09:01") & (m["hm"] <= "15:30")]
    m = m[~((m["hm"] > "15:20") & (m["hm"] < "15:30"))]          # 15:20~15:29 체결 없음(종가 단일가 접수)
    return m


def entry_table(m, days):
    """일자별 진입가(시각별 첫 체결)와 15:30 종가(원 가격)."""
    rows = []
    g = {d: x for d, x in m[m["d"].isin(days)].groupby("d")}
    for d in days:
        x = g.get(d)
        if x is None:
            continue
        x = x.sort_values("hm")
        close = x.loc[x["hm"] == "15:30", "close"]
        if close.empty:
            continue
        row = {"date": d, "c_raw": float(close.iloc[0]), "first_hm": x.loc[x["volume"] > 0, "hm"].min()}
        for k, lab in ENTRY_CLOCKS.items():
            if lab in (None, "SYN"):
                continue
            y = x[(x["hm"] >= lab) & (x["volume"] > 0)]
            row[f"e_{k}"] = float(y["open"].iloc[0]) if len(y) else np.nan
        rows.append(row)
    return pd.DataFrame(rows).set_index("date")


def delayed_returns(d, et):
    """진입 시각별 KODEX 1x 와 2x 경로 근사 수익(시가→종가 대체). 09:00 단일가는 일봉 값."""
    out = {}
    cc = d["k_cc"].reindex(et.index)
    for k in ENTRY_CLOCKS:
        if ENTRY_CLOCKS[k] in (None, "SYN"):
            rk = d["r_k"].reindex(et.index)
            rl2 = d["r_ls"].reindex(et.index)
        else:
            ratio = et[f"e_{k}"] / et["c_raw"]                    # 진입가 / 당일 종가 (원 가격끼리라 수정 무관)
            rk = 1 / ratio - 1
            kt = (1 + cc) * ratio - 1                              # 진입 시점 KODEX 의 전일 종가 대비 수익
            rl2 = (1 + 2 * cc) / (1 + 2 * kt) - 1
        out[k] = pd.DataFrame({"r_k": rk, "r_l2": rl2})
    return out


# ---------------------------------------------------------------- 보조
def pct(v, nd=1):
    return "-" if v is None or pd.isna(v) else f"{v * 100:.{nd}f}%"


def num(v, nd=2, sign=True):
    return "-" if v is None or pd.isna(v) else (f"{v:+.{nd}f}" if sign else f"{v:.{nd}f}")


def mrow(label, m, extra=None):
    r = {"구분": label, "기간": f"{m['start']}~{m['end']}", "거래": m["trades"], "거래/년": num(m["trades_yr"], 1, False),
         "노출": pct(m["exposure"]), "평균w": num(m["avg_w"], 2, False), "승률": pct(m["win"]), "거래당순bp": num(m["net_bp"], 1),
         "NW t": num(m["t_net"]), "CAGR": pct(m["cagr"]), "샤프": num(m["sharpe"]), "MDD": pct(m["mdd"]), "칼마": num(m["calmar"])}
    if extra:
        r.update(extra)
    return r


def md(df):
    return df.to_markdown(index=False)


# ---------------------------------------------------------------- 본체
def main():
    OUTD.mkdir(parents=True, exist_ok=True)
    d = load()
    kclose = data.price("069500")["close"]
    last = d.index.max()
    WINS = {"설계": (LEV_LISTED, "2019-12-31"), "검증": ("2020-01-01", last), "검증(~2025-06-05)": ("2020-01-01", "2025-06-05"),
            "2025-06-09~": ("2025-06-09", last), "추가OOS 2003~2009": ("2003-01-01", "2009-12-31")}
    V = variants()

    # ---- 그리드
    rows, series = [], {}
    for v in V:
        w = exposure(d, v)
        net, gross = daily_net(d, w, v["veh"])
        nets, _ = daily_net(d, w, v["veh"], stress=True)
        series[v["name"]] = (net, w, gross, nets)
        r = {k: v[k] for k in ("name", "grp", "sig", "size", "veh", "filt")}
        for wn, (a, b) in WINS.items():
            m = metrics(net.loc[a:b], w.loc[a:b], gross.loc[a:b])
            ms = metrics(nets.loc[a:b], w.loc[a:b])
            pre = {"설계": "des", "검증": "val", "검증(~2025-06-05)": "valx", "2025-06-09~": "rec", "추가OOS 2003~2009": "oos"}[wn]
            for k in ("trades", "trades_yr", "exposure", "avg_w", "win", "net_bp", "gross_bp", "t_net", "cagr", "sharpe", "mdd", "calmar"):
                r[f"{pre}_{k}"] = m.get(k)
            r[f"{pre}_cagr_stress"] = ms.get("cagr")
        rows.append(r)
    grid = pd.DataFrame(rows)

    # ---- 사전 규칙으로 선택(설계 지표만)
    cand = grid[(grid["veh"] == "ETF") & (grid["des_trades_yr"] >= MIN_TPY) & (grid["des_mdd"] >= MIN_MDD)].copy()
    cand["simplicity"] = cand["filt"].isna().astype(int) * 2 + cand["size"].isin(["1x", "lev", "tier"]).astype(int)
    hit = cand[cand["des_cagr"] >= TARGET_CAGR]
    if len(hit):
        hit = hit.assign(sh2=hit["des_sharpe"].round(2)).sort_values(["sh2", "simplicity"], ascending=[False, False])
        rest = cand.drop(hit.index).sort_values("des_cagr", ascending=False)
        order = list(hit["name"]) + list(rest["name"])
        rule_used = "설계 CAGR ≥ 10% 후보 중 설계 샤프 최대"
    else:
        order = list(cand.sort_values("des_cagr", ascending=False)["name"])
        rule_used = "CAGR ≥ 10% 후보 없음 → 설계 CAGR 최대"
    chosen, second, third = order[0], order[1], order[2]
    grid["rank_rule"] = grid["name"].map({n: i + 1 for i, n in enumerate(order)})
    # 참고: 같은 규칙을 IDX(선물 프록시)에 적용
    ci = grid[(grid["veh"] == "IDX") & (grid["des_trades_yr"] >= MIN_TPY) & (grid["des_mdd"] >= MIN_MDD)]
    hi = ci[ci["des_cagr"] >= TARGET_CAGR]
    idx_pick = (hi.assign(sh2=hi["des_sharpe"].round(2)).sort_values("sh2", ascending=False)["name"].iloc[0] if len(hi)
                else ci.sort_values("des_cagr", ascending=False)["name"].iloc[0])
    grid.to_csv(OUTD / "grid.csv", index=False, encoding="utf-8-sig", float_format="%.6g")

    # ---- 최종·2·3위·기준선 상세
    base = "C3|1x|ETF"
    show = [chosen, second, third, base, idx_pick]
    show = list(dict.fromkeys(show))
    detail = []
    for nm in show:
        net, w, gross, nets = series[nm]
        for wn, (a, b) in WINS.items():
            m = metrics(net.loc[a:b], w.loc[a:b], gross.loc[a:b])
            ms = metrics(nets.loc[a:b], w.loc[a:b])
            bh = bench(kclose, m["start"], m["end"])
            detail.append({"variant": nm, "window": wn, **m, "cagr_stress": ms["cagr"], "win_stress": ms["win"], **bh})
    detail = pd.DataFrame(detail)

    # 연도별(최종 + 기준선 + 보유)
    net, w, gross, nets = series[chosen]
    bnet, bw, _, _ = series[base]
    yrows = []
    for y in sorted(set(d.index.year)):
        s = net[net.index.year == y]
        ww = w[w.index.year == y]
        m = metrics(s, ww)
        mb = metrics(bnet[bnet.index.year == y], bw[bw.index.year == y])
        c = kclose[kclose.index.year <= y]
        r = (c / c.shift(1) - 1)
        r = r[r.index.year == y].dropna()
        yrows.append({"연도": y, "구간": "추가OOS" if y < 2010 else ("설계" if y < 2020 else "검증"), "거래": m["trades"],
                      "승률": pct(m["win"]), "거래당순bp": num(m["net_bp"], 1), "연수익": pct(m["cum"]), "MDD": pct(m["mdd"]),
                      "C3 1x 연수익": pct(mb["cum"]), "KODEX200 보유": pct((1 + r).prod() - 1)})
    yearly = pd.DataFrame(yrows)

    # ---- 분봉 실행 점검(2023-01~)
    m1 = load_min1()
    a1, b1 = "2023-01-02", last
    ex_rows, ex_daily = [], {}
    et_all = entry_table(m1, list(d.loc[a1:b1].index[(d.loc[a1:b1, "C2"] == 1)]))
    dr = delayed_returns(d, et_all)
    span = d.loc[a1:b1].index
    have = span[span.isin(m1["d"].unique())]
    for nm in list(dict.fromkeys([base, "C3|lev|ETF", "C2|1x|ETF", "C2|lev|ETF", chosen])):
        v = next(x for x in V if x["name"] == nm)
        w = exposure(d, v).reindex(have).fillna(0)
        if v["veh"] != "ETF":
            continue
        for k in ENTRY_CLOCKS:
            if ENTRY_CLOCKS[k] == "SYN" and w.max() <= 1:
                continue
            r = dr[k].reindex(have)
            a = np.where(w <= 1, w, 2 - w)
            bl = np.where(w <= 1, 0.0, w - 1)
            rl = d["r_l"].reindex(have) if ENTRY_CLOCKS[k] is None else r["r_l2"]
            gross = pd.Series(a * r["r_k"].fillna(0) + bl * rl.fillna(0), index=have)
            net = gross - (a * COST["K"][0] + bl * COST["L"][0]) / 1e4
            nets = gross - (a * COST["K"][1] + bl * COST["L"][1]) / 1e4
            traded = (w > 0) & r["r_k"].notna()
            net, nets, gross = net.where(traded | (w == 0)), nets.where(traded | (w == 0)), gross.where(traded | (w == 0))
            mm = metrics(net.dropna(), w.where(traded | (w == 0)).dropna(), gross)
            ex_daily[(nm, k)] = net
            for sub, (sa, sb) in {"2023-01~2026-09": (a1, b1), "2023~2024": (a1, "2024-12-31"), "2025~2026-09": ("2025-01-01", b1)}.items():
                ms = metrics(net.loc[sa:sb].dropna(), w.loc[sa:sb], gross.loc[sa:sb])
                mst = metrics(nets.loc[sa:sb].dropna(), w.loc[sa:sb])
                ex_rows.append({"변형": nm, "진입": k, "구간": sub, "거래": ms["trades"], "승률": ms["win"], "거래당총bp": ms["gross_bp"],
                                "거래당순bp": ms["net_bp"], "NW t": ms["t_net"], "CAGR": ms["cagr"], "CAGR스트레스": mst["cagr"], "MDD": ms["mdd"], "샤프": ms["sharpe"]})
    ex = pd.DataFrame(ex_rows)
    # 09:00 대비 남는 몫(총수익 기준)
    ex["남는몫"] = np.nan
    for (nm, sub), g in ex.groupby(["변형", "구간"]):
        basek = "09:00 단일가(2x 합성)" if (g["진입"] == "09:00 단일가(2x 합성)").any() else "09:00 단일가"
        g0 = g.loc[g["진입"] == basek, "거래당총bp"].iloc[0]
        ex.loc[g.index, "남는몫"] = g["거래당총bp"] / g0 if g0 else np.nan
    # 단일가 대비 1분 후 KODEX 가격 변화(신호일, C3)
    c3d = d.loc[a1:b1].index[d.loc[a1:b1, "C3"] == 1]
    et3 = et_all.reindex(c3d).dropna(subset=["c_raw"])
    drift = {}
    for k in ENTRY_CLOCKS:
        if ENTRY_CLOCKS[k] in (None, "SYN"):
            continue
        open_raw = et3["c_raw"] / (1 + d["r_k"].reindex(et3.index))           # 단일가 시가(원 가격 환산)
        drift[k] = ((et3[f"e_{k}"] / open_raw - 1) * 1e4).describe()[["count", "mean", "50%"]]
    drift = pd.DataFrame(drift).T
    late_open = et_all[et_all["first_hm"] > "09:01"].index
    same_day = pd.DataFrame([{"신호": nm_, "날수": len(x), "KODEX 09:00 단일가→종가 bp": x["r_k"].mean() * 1e4,
                              "KODEX 09:01 진입→종가 bp": dr["09:01"]["r_k"].reindex(x.index).mean() * 1e4,
                              "코스피200 지수 시가→종가 bp": x["r_i"].mean() * 1e4, "122630 시가→종가 bp": x["r_l"].mean() * 1e4}
                             for nm_, x in (("C3", d.loc[a1:b1][d.loc[a1:b1, "C3"] == 1]), ("C2", d.loc[a1:b1][d.loc[a1:b1, "C2"] == 1]))])

    # 검증 전체에 분봉 지연 손실을 덧씌운 추정(진입 시각별, 2023~2026 평균 손실 bp/노출 1 을 거래마다 차감)
    hc_rows = []
    nm = chosen
    net, w, gross, nets = series[nm]
    for k in ("09:01", "09:02", "09:05"):
        sel = (ex["변형"] == base) & (ex["구간"] == "2023-01~2026-09")
        loss = (ex.loc[sel & (ex["진입"] == "09:00 단일가"), "거래당총bp"].iloc[0] - ex.loc[sel & (ex["진입"] == k), "거래당총bp"].iloc[0])
        adj = net - (w * loss / 1e4).where(w > 0, 0)
        for wn in ("설계", "검증", "검증(~2025-06-05)"):
            a, b = WINS[wn]
            m = metrics(adj.loc[a:b], w.loc[a:b])
            hc_rows.append({"진입": k, "노출1당 차감bp": round(loss, 1), "구간": wn, "승률": pct(m["win"]), "거래당순bp": num(m["net_bp"], 1),
                            "CAGR": pct(m["cagr"]), "MDD": pct(m["mdd"]), "샤프": num(m["sharpe"])})
    haircut = pd.DataFrame(hc_rows)

    # ---- 선물 프록시 진단: C3 날 지수 vs 선물(09:00 개장 시기 2014-07-08~2023-07-28) vs KODEX
    q = d.loc["2014-07-08":"2023-07-28"]
    q3 = q[(q["C3"] == 1) & q["r_f"].notna()]
    q2 = q[(q["C2"] == 1) & q["r_f"].notna()]
    fx = pd.DataFrame([{"신호": nm_, "날수": len(x), "지수 시가→종가 bp": x["r_i"].mean() * 1e4, "선물 시가→종가 bp": x["r_f"].mean() * 1e4,
                        "KODEX 시가→종가 bp": x["r_k"].mean() * 1e4, "122630 시가→종가 bp": x["r_l"].mean() * 1e4}
                       for nm_, x in (("C3", q3), ("C2", q2))])

    # 122630 실제 vs 2x 합성(2010-02-22~, C3 날)
    c3all = d[(d["C3"] == 1) & (d.index >= LEV_LISTED)]
    levchk = {"C3 날수": len(c3all), "122630 실제 bp": c3all["r_l"].mean() * 1e4, "2x KODEX 합성 bp": c3all["r_ls"].mean() * 1e4,
              "KODEX 1x bp": c3all["r_k"].mean() * 1e4, "지수 bp": c3all["r_i"].mean() * 1e4}

    # ---- 산출
    out_daily = pd.DataFrame({"C2": d["C2"], "C3": d["C3"], "z_index": d["z"], "mis": d["mis"], "sig_hat20": d["sig_hat"],
                              "r_069500_oc": d["r_k"], "r_122630_oc": d["r_l"], "r_lev_synth_oc": d["r_ls"], "r_k200idx_oc": d["r_i"]})
    for nm, tag in ((chosen, "chosen"), (second, "second"), (third, "third"), (base, "base_C3_1x"), (idx_pick, "idx_ref")):
        out_daily[f"w_{tag}"] = series[nm][1]
        out_daily[f"net5_{tag}"] = series[nm][0]
    out_daily["bh_069500_cc"] = d["k_cc"]
    out_daily.index.name = "date"
    out_daily.to_csv(OUTD / "daily_returns.csv", encoding="utf-8-sig", float_format="%.8g")

    net, w, gross, nets = series[chosen]
    v = next(x for x in V if x["name"] == chosen)
    tr = d.loc[w > 0, ["z", "mis", "C2", "C3", "r_k", "r_l", "r_ls", "r_i"]].copy()
    tr["w"] = w[w > 0]
    tr["w_069500"] = np.where(tr["w"] <= 1, tr["w"], 2 - tr["w"])
    tr["w_122630"] = np.where(tr["w"] <= 1, 0.0, tr["w"] - 1)
    tr["gross"], tr["net5"], tr["net_stress"] = gross[w > 0], net[w > 0], nets[w > 0]
    tr["window"] = np.where(tr.index < "2010-01-01", "추가OOS", np.where(tr.index < "2020-01-01", "설계", "검증"))
    for k in ENTRY_CLOCKS:
        if ENTRY_CLOCKS[k] not in (None, "SYN") and (chosen, k) in ex_daily:
            tr[f"net5_entry_{k}"] = ex_daily[(chosen, k)].reindex(tr.index)
    tr.index.name = "date"
    tr.to_csv(OUTD / "trades.csv", encoding="utf-8-sig", float_format="%.8g")
    ex.to_csv(OUTD / "entry_delay.csv", index=False, encoding="utf-8-sig", float_format="%.6g")
    detail.to_csv(OUTD / "detail.csv", index=False, encoding="utf-8-sig", float_format="%.6g")

    chart(series, chosen, base, kclose, d)

    # ---- 장부
    if "--no-ledger" not in sys.argv:
        led = []
        for _, r in grid.iterrows():
            for pre, wn in (("des", "설계"), ("val", "검증"), ("valx", "검증(~2025-06-05)"), ("rec", "2025-06-09~"), ("oos", "추가OOS 2003~2009")):
                led.append({"asset": "069500/122630" if r["veh"] == "ETF" else "K200 index(fut proxy)", "rule": r["name"], "window": wn,
                            "cost_bp": "ETF5/IDX2", "trades": r[f"{pre}_trades"], "exposure": r[f"{pre}_exposure"], "gross_bp": r[f"{pre}_gross_bp"],
                            "net_bp": r[f"{pre}_net_bp"], "t_net": r[f"{pre}_t_net"], "win": r[f"{pre}_win"], "sharpe": r[f"{pre}_sharpe"],
                            "cagr": r[f"{pre}_cagr"], "mdd": r[f"{pre}_mdd"], "calmar": r[f"{pre}_calmar"]})
        spec = {"family": FAMILY, "signal": "bt_d2_2010_2026.signals() R3-C2/C3 unchanged", "design": [LEV_LISTED, "2019-12-31"],
                "validation": ["2020-01-01", str(last.date())], "n_variants": len(V), "selection": rule_used, "chosen": chosen}
        E.ledger_append(data.OUT, TOPIC, spec, pd.DataFrame(led), note="5차 전략 탐색")

    report(d, grid, detail, yearly, ex, drift, haircut, fx, levchk, chosen, second, third, base, idx_pick, rule_used, WINS, late_open, len(V), same_day)
    print("chosen", chosen, "second", second, "third", third, "idx_pick", idx_pick)
    print(detail[detail["variant"] == chosen][["window", "trades", "trades_yr", "win", "net_bp", "t_net", "cagr", "sharpe", "mdd", "calmar",
                                              "cagr_stress", "bh_cagr", "bh_mdd"]].to_string())


def chart(series, chosen, base, kclose, d):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False})
    fig, ax = plt.subplots(figsize=(10, 5.2), dpi=130)
    bg = "#fcfcfb"
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)
    a = LEV_LISTED
    lines = [(f"최종: {chosen}", series[chosen][0].loc[a:], dict(color="#2a78d6", lw=2)),
             (f"기준: {base}(기존 C3)", series[base][0].loc[a:], dict(color="#eb6834", lw=1.6)),
             ("KODEX 200 보유", (kclose / kclose.shift(1) - 1).reindex(d.index).loc[a:], dict(color="#52514e", lw=1.4, ls=":"))]
    for lab, s, kw in lines:
        eq = (1 + s.fillna(0)).cumprod()
        ax.plot(eq.index, eq.values, label=lab, **kw)
        ax.annotate(f"×{eq.iloc[-1]:.2f}", xy=(eq.index[-1], eq.iloc[-1]), xytext=(4, 0), textcoords="offset points",
                    fontsize=8, color="#0b0b0b", va="center")
    ax.axvspan(pd.Timestamp("2020-01-01"), d.index.max(), color="#e6e5e1", alpha=0.5, lw=0)
    ax.text(pd.Timestamp("2020-02-01"), 0.97, "검증 구간", transform=ax.get_xaxis_transform(), fontsize=8, color="#52514e", va="top")
    ax.text(pd.Timestamp("2010-04-01"), 0.97, "설계 구간", transform=ax.get_xaxis_transform(), fontsize=8, color="#52514e", va="top")
    ax.set_yscale("log")
    ticks = [0.7, 1, 1.5, 2, 3, 5, 8, 12]
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{t:g}" for t in ticks])
    ax.minorticks_off()
    ax.set_ylabel("누적 가치(시작 = 1, 로그 축)", color="#52514e")
    ax.set_title(f"C3 레버리지 실행 누적 수익, 비용 왕복 5bp ({LEV_LISTED} ~ {d.index.max().date()})", color="#0b0b0b", fontsize=11, loc="left")
    ax.grid(True, color="#e6e5e1", lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c3c2b7")
    ax.tick_params(colors="#52514e", labelsize=8)
    ax.legend(loc="upper left", fontsize=8, frameon=False, bbox_to_anchor=(0, 0.92))
    ax.set_xlim(right=ax.get_xlim()[1] + 300)
    fig.tight_layout()
    fig.savefig(OUTD / "equity.png", facecolor=bg)
    plt.close(fig)


def report(d, grid, detail, yearly, ex, drift, haircut, fx, levchk, chosen, second, third, base, idx_pick, rule_used, WINS, late_open, nv, same_day):
    def drow(nm, wn):
        r = detail[(detail["variant"] == nm) & (detail["window"] == wn)].iloc[0]
        return r

    L = ["# 5차 전략 탐색 · c3_levered: D2 잔차 갭(C3/C2) 레버리지·크기·필터\n"]
    ch_v, ch_vx = drow(chosen, "검증"), drow(chosen, "검증(~2025-06-05)")
    ch_d = drow(chosen, "설계")
    full = ex[(ex["변형"] == chosen) & (ex["구간"] == "2023-01~2026-09")].set_index("진입")
    ex_c0, ex_c1 = full.loc["09:00 단일가"], full.loc["09:01"]
    lv = ex[(ex["변형"] == "C3|lev|ETF") & (ex["구간"] == "2023-01~2026-09")].set_index("진입")["거래당총bp"]
    lev_gap = lv["09:00 단일가(2x 합성)"] - lv["09:00 단일가"]
    L += ["## 요약\n",
          f"- 최종 사양(설계 구간 규칙으로 선택): **{chosen}** — 선택 근거: {rule_used}.",
          f"- 설계({ch_d['start']}~{ch_d['end']}): CAGR {pct(ch_d['cagr'])}, 승률 {pct(ch_d['win'])}, 샤프 {num(ch_d['sharpe'])}, MDD {pct(ch_d['mdd'])}, 거래 {ch_d['trades_yr']:.1f}회/년.",
          f"- 검증({ch_v['start']}~{ch_v['end']}, 비용 5bp): CAGR {pct(ch_v['cagr'])}, 승률 {pct(ch_v['win'])}, 거래당 순 {num(ch_v['net_bp'], 1)}bp(NW t {num(ch_v['t_net'])}), "
          f"샤프 {num(ch_v['sharpe'])}, MDD {pct(ch_v['mdd'])}. 같은 구간 KODEX 200 보유 CAGR {pct(ch_v['bh_cagr'])}, MDD {pct(ch_v['bh_mdd'])}.",
          f"- 최근 급등(2025-06-09~) 제외 검증: CAGR {pct(ch_vx['cagr'])}, 승률 {pct(ch_vx['win'])}, MDD {pct(ch_vx['mdd'])}.",
          "- 단, 위 수치는 '09:00 시가로 신호를 계산하고 같은 09:00 단일가에 체결'한다는 가정(공통 규약 4의 '봉 마감 결정 → 다음 봉 시가 체결'을 어김) 위에 있다.",
          f"- 규약대로 09:00 봉(09:00:00~09:00:59) 마감 뒤 09:01 첫 체결에 들어가면(KODEX 1분봉, {ex_c0['거래']}거래, 2023-01~2026-09): "
          f"최종 사양 CAGR {pct(ex_c1['CAGR'])}·승률 {pct(ex_c1['승률'])}·거래당 순 {num(ex_c1['거래당순bp'], 1)}bp(NW t {num(ex_c1['NW t'])}) — 같은 기간 09:00 단일가 가정 CAGR {pct(ex_c0['CAGR'])}·승률 {pct(ex_c0['승률'])}. "
          f"거래당 총수익의 {ex_c1['남는몫'] * 100:.0f}%만 남는다. 첫 1분 동안 KODEX 가 단일가보다 평균 {drift.loc['09:01', 'mean']:+.1f}bp 오른 몫(시가 단일가 저평가)이 엣지의 큰 부분이다.",
          f"- 목표 점검(검증 구간, 비용 5bp): 시가 단일가 가정 CAGR {pct(ch_v['cagr'])}(≥10% {'충족' if ch_v['cagr'] >= 0.10 else '미달'}), 승률 {pct(ch_v['win'])}(≥51% {'충족' if ch_v['win'] >= 0.51 else '미달'}). "
          f"규약상 실행형 09:01 진입(2023-01~2026-09만 가능) CAGR {pct(ex_c1['CAGR'])}({'충족' if ex_c1['CAGR'] >= 0.10 else '미달'}), 승률 {pct(ex_c1['승률'])}({'충족' if ex_c1['승률'] >= 0.51 else '미달'}). "
          "시가 단일가를 잡으려면 08:59 예상지수·ETF 예상체결가로 미리 주문해야 하며, 그 근사 오차는 과거 자료가 없어 검증하지 못했다.\n"]

    L += ["## 1. 규칙\n", "```\n" + __doc__.strip() + "\n```\n"]

    L += ["## 2. 그리드 설계 성과(상위 15, 규칙 순위) — 전체 42개는 grid.csv\n"]
    g = grid.sort_values("rank_rule").head(15)
    gt = pd.DataFrame({"순위": g["rank_rule"].astype(int), "변형": g["name"], "설계 거래/년": g["des_trades_yr"].map(lambda x: num(x, 1, False)),
                       "설계 승률": g["des_win"].map(pct), "설계 거래당순bp": g["des_net_bp"].map(lambda x: num(x, 1)),
                       "설계 CAGR": g["des_cagr"].map(pct), "설계 샤프": g["des_sharpe"].map(num), "설계 MDD": g["des_mdd"].map(pct)})
    L += [md(gt), "\n"]
    gi = grid[grid["veh"] == "IDX"].sort_values("des_cagr", ascending=False)
    L += ["참고: 지수(선물 프록시, 2bp) 변형 설계 성과(후보 아님)\n",
          md(pd.DataFrame({"변형": gi["name"], "설계 거래/년": gi["des_trades_yr"].map(lambda x: num(x, 1, False)), "설계 승률": gi["des_win"].map(pct),
                           "설계 CAGR": gi["des_cagr"].map(pct), "설계 샤프": gi["des_sharpe"].map(num), "설계 MDD": gi["des_mdd"].map(pct)})), "\n"]

    L += ["## 3. 최종 사양과 2·3위, 기준선의 구간별 성과(비용 5bp, 괄호 밖 CAGR 스트레스는 ETF 7·8bp / 지수 3bp)\n"]
    rows = []
    for nm, tag in ((chosen, "최종"), (second, "2위"), (third, "3위"), (base, "기준선(기존 C3 1x)"), (idx_pick, "지수 참고")):
        for wn in WINS:
            r = drow(nm, wn)
            rows.append(mrow(f"{tag} {nm}", r, {"구간": wn, "CAGR(스트레스)": pct(r["cagr_stress"]), "보유 CAGR": pct(r["bh_cagr"]), "보유 MDD": pct(r["bh_mdd"])}))
    t = pd.DataFrame(rows)[["구분", "구간", "기간", "거래", "거래/년", "평균w", "승률", "거래당순bp", "NW t", "CAGR", "CAGR(스트레스)", "샤프", "MDD", "칼마", "보유 CAGR", "보유 MDD"]]
    L += [md(t), "\n",
          "- 추가OOS 2003~2009 의 레버리지 몫은 122630 이 없어 2x KODEX 경로 합성값이다. 이 시기 KODEX 시가 저평가 몫이 커서(기준선 거래당 순 +80bp 대) 수치가 높다. 신호 자체의 이 시기 판정은 3차 보고서에 있다.\n"]

    L += ["## 4. 연도별(최종 사양, 비용 5bp)\n", md(yearly), "\n"]

    L += ["## 5. 차량 점검\n",
          "122630 실제 시가→종가와 2x KODEX 합성의 C3 날 평균(2010-02-22~):\n",
          md(pd.DataFrame([{k: (f"{v:+.1f}" if isinstance(v, float) else v) for k, v in levchk.items()}])), "\n",
          "선물 프록시 진단(선물이 09:00 에 같이 열리던 2014-07-08~2023-07-28, 신호일 평균 시가→종가 bp, 총수익):\n",
          md(fx.round(1)), "\n",
          "- 신호일에는 지수 자체의 시가→종가가 작고(수 bp), 선물·KODEX·122630 의 시가 단일가가 지수보다 싸게 형성된 몫이 수익의 대부분이다. 엣지의 핵심은 '갭 지속'보다 '시가 단일가 저평가 회복'에 가깝다.",
          "- 그래서 지수 프록시(IDX) 변형은 선물을 09:00 단일가에 잡는 경로보다 낮게 나오고, 단일가 이후 선물 진입에 더 가깝다. 2023-07-31 이후 선물은 08:45 에 열려 09:00 에는 단일가가 없다.\n"]

    L += ["## 6. 분봉 실행 점검(KODEX 200 1분봉, 2023-01-02~)\n",
          f"신호일 중 장 시작이 09:00 이 아닌 날(첫 체결 > 09:01 라벨, 예: 1월 첫 거래일·수능일 10:00 개장): {', '.join(str(x.date()) for x in late_open) or '없음'}. 이런 날은 지연 진입가가 사실상 그날 첫 체결이다.\n",
          "C3 날 단일가 대비 지연 진입가 변화(bp, KODEX 원 가격):\n", md(drift.round(1).reset_index().rename(columns={"index": "진입"})), "\n",
          "같은 신호일(2023-01~)의 평균 총수익 비교(bp):\n", md(same_day.round(1)), "\n"]
    e = ex.copy()
    for c in ("승률", "CAGR", "CAGR스트레스", "MDD"):
        e[c] = e[c].map(pct)
    for c in ("거래당총bp", "거래당순bp", "샤프", "NW t"):
        e[c] = e[c].map(lambda x: num(x, 1) if c != "NW t" else num(x))
    e["남는몫"] = e["남는몫"].map(lambda x: "-" if pd.isna(x) else f"{x * 100:.0f}%")
    L += ["진입 시각별 성과(비용 5bp). 레버리지 몫: '09:00 단일가'는 122630 실제, '(2x 합성)'과 지연 진입은 2x KODEX 경로 근사. '남는몫' = 거래당 총수익 / 같은 경로의 09:00 단일가 총수익(레버리지 포함 변형은 2x 합성 기준). 09:01 진입이 공통 규약(봉 마감 결정 → 다음 봉 시가)에 맞는 실행형이다:\n",
          md(e[e["구간"] == "2023-01~2026-09"].drop(columns=["구간"])), "\n",
          "하위 구간(2023~2024 / 2025~2026-09):\n", md(e[e["구간"] != "2023-01~2026-09"]), "\n",
          "추정: 2023~2026 에서 잰 지연 손실(기존 C3 1x 거래당 총수익 차이)을 노출 1 당 거래마다 빼서 최종 사양 전 구간에 덧씌운 값(가정에 기반한 추정):\n",
          md(haircut), "\n"]

    L += ["## 7. 읽는 법·주의\n",
          "- 신호(C2·C3)는 바꾸지 않았다. 신호 자체는 이전 차수에서 2020~2025-06 과 2003~2009 를 이미 본 것이라, 여기서의 '검증'은 차량·크기·필터 선택에 대해서만 표본 외다.",
          "- 레버리지(122630)는 수익과 손실을 거의 두 배로 키울 뿐 샤프를 바꾸지 않는다. CAGR 10% 는 C3 의 1x 엣지(연 6~8%)를 1.4~2배 노출로 키운 결과다. 변동성 목표(vt30)는 변동성이 큰 시기(2020, 2025~26)에 노출을 줄여 검증 MDD 를 레버리지 고정(−18%)보다 낮췄다(−8%).",
          f"- 분봉 점검의 레버리지 몫은 2x KODEX 경로 근사다. 2023~2026 C3 날 09:00 단일가 기준 근사가 122630 실제보다 거래당 {lev_gap:+.1f}bp(2x 노출 기준) 높아, 지연 진입 레버리지 수치는 그만큼 낙관적일 수 있다.",
          "- 검증 구간 KODEX 200 보유 CAGR 이 25%(2025~26 급등)라 절대 수익은 보유보다 낮다. 이 전략의 장점은 노출 약 10%·MDD 한 자릿수라는 점이다.",
          "- 가장 큰 위험은 실행이다. 백테스트는 09:00 시가로 계산한 신호를 같은 09:00 단일가에 체결한다. 실거래에서 08:59 예상지수로 미리 주문하지 않으면 09:01 이후 진입이 되고, 6절처럼 거래당 총수익이 C3 는 약 절반, C2 는 약 60% 줄어든다(2023~2026 분봉).",
          "- 레버리지 ETF 매수에는 금융투자교육원 사전교육 이수와 증권사 기본예탁금 요건이 있다(증권사에서 확인 필요). 미니 코스피200 선물은 파생상품 계좌(사전교육·모의거래·기본예탁금)가 필요하다.",
          f"- 시행한 변형 수: {nv}개(전부 grid.csv·장부 R5_C3_LEVERED). 선택은 설계 지표 규칙만 썼다.",
          "- 파일: daily_returns.csv(일별 노출·순수익), trades.csv(최종 사양 거래, 분봉 지연 진입 수익 포함), grid.csv, detail.csv, entry_delay.csv, equity.png."]
    L.insert(2, "![equity](equity.png)\n")
    (OUTD / "report.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
