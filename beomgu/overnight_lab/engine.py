"""종가 베팅·오버나잇 백테스트 엔진.

수익률 구간(결정일 T 기준 인덱스):
  on  = T 종가 → T+1 시가   (종가 단일가 매수, 익일 시가 단일가 매도)
  oc  = T 시가 → T 종가
  cc  = T 종가 → T+1 종가
포지션은 결정일 T에 알 수 있는 정보로만 정한다. 매 거래는 왕복 1회 비용.
"""
import datetime as dt
import json
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

KST = ZoneInfo("Asia/Seoul")
COSTS_BP = (5.0, 7.0)                 # 팀 기준 왕복 3bp + 슬리피지 2bp = 5bp, 스트레스 7bp
JUDGE_END = "2025-06-05"              # 판정 구간 끝. 2025-06-09 KRX 야간 이후(E4)는 보고만
A_END = "2019-12-31"                  # 하위 구간 A = ~2019, B = 2020-01~판정 끝
E4_START = "2025-06-09"
ANN = 252


def legs(px):
    o, c = px["open"], px["close"]
    return pd.DataFrame({"on": o.shift(-1) / c - 1, "oc": c / o - 1, "cc": c.shift(-1) / c - 1})


def nw_t(x, lags=5):
    """평균의 Newey-West t (x: 결측 제거된 1차원)."""
    x = np.asarray(x, float)
    n = len(x)
    if n < 10:
        return np.nan
    e = x - x.mean()
    s = e @ e / n
    for k in range(1, min(lags, n - 1) + 1):
        s += 2 * (1 - k / (lags + 1)) * (e[k:] @ e[:-k]) / n
    return x.mean() / np.sqrt(s / n) if s > 0 else np.nan


def stats(pos, ret, cost_bp):
    """pos: 결정일 T의 포지션(0/1 또는 −1/0/1), ret: 같은 인덱스의 구간 수익률."""
    d = pd.DataFrame({"p": pos, "r": ret}).dropna()
    traded = d["p"] != 0
    daily = d["p"] * d["r"] - traded * cost_bp / 1e4
    tr = daily[traded]
    eq = (1 + daily).cumprod()
    years = len(d) / ANN
    cagr = eq.iloc[-1] ** (1 / years) - 1 if years > 0 and eq.iloc[-1] > 0 else np.nan
    mdd = (eq / eq.cummax() - 1).min()
    sd = daily.std()
    return {
        "days": len(d), "trades": int(traded.sum()), "exposure": traded.mean(),
        "gross_bp": (d["p"] * d["r"])[traded].mean() * 1e4, "net_bp": tr.mean() * 1e4,
        "t_net": nw_t(tr), "win": (tr > 0).mean(),
        "sharpe": daily.mean() / sd * np.sqrt(ANN) if sd > 0 else np.nan,
        "cagr": cagr, "mdd": mdd, "calmar": cagr / -mdd if mdd < 0 else np.nan,
    }


def random_pct(pos, ret, cost_bp, draws=2000, seed=0):
    """같은 거래일 수를 무작위로 고른 규칙 대비 거래당 평균 수익 백분위(롱 전용, 비용은 상수라 순위 동일)."""
    d = pd.DataFrame({"p": pos, "r": ret}).dropna()
    k = int((d["p"] != 0).sum())
    if k < 5 or k >= len(d):
        return np.nan
    actual = (d["p"] * d["r"])[d["p"] != 0].mean()
    rng = np.random.default_rng(seed)
    r = d["r"].to_numpy()
    sims = np.array([r[rng.choice(len(r), k, replace=False)].mean() for _ in range(draws)])
    return (sims < actual).mean()


def windows(idx, end=None):
    end = pd.Timestamp(end or idx.max())
    T = pd.Timestamp
    return {
        "판정(~2025-06-05)": (idx.min(), T(JUDGE_END)), "A(~2019)": (idx.min(), T(A_END)),
        "B(2020~2025-06)": (T("2020-01-01"), T(JUDGE_END)), "E4(2025-06-09~)": (T(E4_START), end),
        "3개월": (end - pd.DateOffset(months=3), end), "1년": (end - pd.DateOffset(years=1), end),
        "3년": (end - pd.DateOffset(years=3), end), "전체": (idx.min(), end),
    }


def delta_test(sig, ret, lags=5):
    """Δ = 평균(신호 밤) − 평균(비신호 밤), 회귀 r = a + b·1[sig] 의 Newey-West t (b = Δ)."""
    d = pd.DataFrame({"s": sig.astype(float), "r": ret}).dropna()
    if d["s"].sum() < 5 or (1 - d["s"]).sum() < 5:
        return np.nan, np.nan, int(d["s"].sum())
    X = np.column_stack([np.ones(len(d)), d["s"].to_numpy()])
    y = d["r"].to_numpy()
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    u = y - X @ beta
    n = len(y)
    XtX_inv = np.linalg.inv(X.T @ X)
    S = (X * u[:, None]).T @ (X * u[:, None])
    for k in range(1, lags + 1):
        w = 1 - k / (lags + 1)
        G = (X[k:] * u[k:, None]).T @ (X[:-k] * u[:-k, None])
        S += w * (G + G.T)
    V = XtX_inv @ S @ XtX_inv * n / (n - 2)
    return beta[1], beta[1] / np.sqrt(V[1, 1]), int(d["s"].sum())


def placebo_year_strat(sig, ret, draws=2000, seed=20260929):
    """연도 층화 무작위 밤 플라시보: 연도별 신호 수를 유지한 채 밤을 무작위로 골라 Δ의 HAC t 분포와 비교(단측 p)."""
    d = pd.DataFrame({"s": sig.astype(float), "r": ret}).dropna()
    _, t0, _ = delta_test(d["s"], d["r"])
    if np.isnan(t0):
        return np.nan
    rng = np.random.default_rng(seed)
    years = d.index.year
    groups = [np.where(years == y)[0] for y in np.unique(years)]
    ks = [int(d["s"].iloc[g].sum()) for g in groups]
    ts = []
    for _ in range(draws):
        s = np.zeros(len(d))
        for g, k in zip(groups, ks):
            if k:
                s[rng.choice(g, k, replace=False)] = 1
        ts.append(delta_test(pd.Series(s, index=d.index), d["r"])[1])
    return float((np.array(ts) >= t0).mean())


def evaluate(rules, ret, cost_bp=COSTS_BP[0], baseline="매일 보유"):
    """rules: {이름: 포지션 Series}. 포지션 NaN(신호 계산 불가) 밤은 그 규칙의 표본에서 뺀다. 기준선(매일 보유) 자동 포함."""
    rules = {baseline: pd.Series(1.0, index=ret.index), **rules}
    rows = []
    for wname, (a, b) in windows(ret.dropna().index).items():
        for name, pos in rules.items():
            s = stats(pos.loc[a:b], ret.loc[a:b], cost_bp)
            s.update(rule=name, window=wname, cost_bp=cost_bp)
            if name != baseline:
                s["rand_pct"] = random_pct(pos.loc[a:b], ret.loc[a:b], cost_bp)
                p = pos.loc[a:b]
                dlt, t, _ = delta_test((p != 0).astype(float).where(p.notna()), ret.loc[a:b])
                s["delta_bp"], s["t_delta"] = dlt * 1e4, t
            rows.append(s)
    return pd.DataFrame(rows)


def yearly(pos, ret, cost_bp=COSTS_BP[0]):
    out = []
    for y, g in ret.groupby(ret.index.year):
        s = stats(pos.reindex(g.index), g, cost_bp)
        s["year"] = y
        out.append(s)
    return pd.DataFrame(out).set_index("year")


def fmt_table(df, cols=("rule", "window", "trades", "delta_bp", "t_delta", "net_bp", "win", "sharpe", "cagr", "mdd", "rand_pct")):
    cols = [c for c in cols if c in df.columns]
    head = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
    body = ""
    for _, r in df[cols].iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, str):
                cells.append(v)
            elif c in ("win", "cagr", "mdd", "exposure", "rand_pct"):
                cells.append("" if pd.isna(v) else f"{v * 100:.1f}%")
            elif c in ("trades", "days"):
                cells.append(f"{int(v)}")
            else:
                cells.append("" if pd.isna(v) else f"{v:+.2f}")
        body += "| " + " | ".join(cells) + " |\n"
    return head + body


LEDGER_COLS = ["run_at", "topic", "spec", "asset", "days", "trades", "exposure", "gross_bp", "net_bp", "t_net", "win", "sharpe",
               "cagr", "mdd", "calmar", "rule", "window", "cost_bp", "rand_pct", "delta_bp", "t_delta", "run_note"]


def ledger_append(out_dir, topic, spec, table, note=""):
    """시행 장부: 돌린 모든 규칙·자산·기간을 남긴다(사후 선택 방지). 열 순서는 LEDGER_COLS 로 고정."""
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / "ledger.csv"
    t = table.copy()
    t["run_at"] = dt.datetime.now(KST).strftime("%Y-%m-%d %H:%M")
    t["topic"], t["spec"], t["run_note"] = topic, json.dumps(spec, ensure_ascii=False), note
    t.reindex(columns=LEDGER_COLS).to_csv(p, mode="a", header=not p.exists(), index=False, encoding="utf-8-sig")
