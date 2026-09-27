"""2단계: 지표 계산과 T−1 정렬.

Flow = (매수 − 매도) / (매수 + 매도)   범위 [−1, 1]
OptFlow = CallFlow − PutFlow          범위 [−2, 2]  (부호 +: 콜 쪽 매수세 우위 → 상승 가정)
CPR = (콜 거래대금 − 풋 거래대금) / (콜 + 풋)
3일 누적 = 최근 3거래일 매수·매도를 각각 합한 뒤 같은 식으로 계산

'total'(시장 전체)은 매수 = 매도라 Flow 가 항상 0 이다 → Flow 를 만들지 않고 CPR 에만 쓴다.
"""
import pandas as pd

from optflow.io import OPT_COL_RE


def flow(buy, sell):
    buy = pd.Series(buy, dtype=float)
    sell = pd.Series(sell, dtype=float)
    tot = buy + sell
    return ((buy - sell) / tot).where(tot > 0)


def _cum(s, n):
    return s.rolling(n, min_periods=n).sum()


def _pair_flows(p, buy_col, sell_col, prefix):
    out = {}
    if buy_col in p.columns and sell_col in p.columns:
        out[f"{prefix}_1d"] = flow(p[buy_col], p[sell_col])
        out[f"{prefix}_3d"] = flow(_cum(p[buy_col], 3), _cum(p[sell_col], 3))
    return out


def option_investors(columns):
    return sorted({m.group("inv") for c in columns if (m := OPT_COL_RE.match(c))})


def build_indicators(panel):
    """패널(달력 기준, 날짜 = 데이터 발생일)에서 지표 계산. 아직 시프트하지 않는다."""
    p = panel
    ind = {}
    ind.update(_pair_flows(p, "fut_f_buy_q", "fut_f_sell_q", "futflow"))       # 통제: 외인 선물(계약)

    for inv in option_investors(p.columns):
        if inv == "total":
            continue
        for unit, suf in (("amt", ""), ("qty", "_qty")):
            ind.update(_pair_flows(p, f"opt_{inv}_C_buy_{unit}", f"opt_{inv}_C_sell_{unit}", f"{inv}_callflow{suf}"))
            ind.update(_pair_flows(p, f"opt_{inv}_P_buy_{unit}", f"opt_{inv}_P_sell_{unit}", f"{inv}_putflow{suf}"))
            for h in ("1d", "3d"):
                kc, kp = f"{inv}_callflow{suf}_{h}", f"{inv}_putflow{suf}_{h}"
                if kc in ind and kp in ind:
                    ind[f"{inv}_optflow{suf}_{h}"] = ind[kc] - ind[kp]
    # 주 지표 별칭(외인 금액 기준). 탐색 목록에서는 foreign_* 원본을 빼고 별칭만 센다.
    for h in ("1d", "3d"):
        for base in ("optflow", "callflow", "putflow"):
            if f"foreign_{base}_{h}" in ind:
                ind[f"{base}_{h}"] = ind.pop(f"foreign_{base}_{h}")

    vc = vp = None
    if "call_value" in p.columns and "put_value" in p.columns:
        vc, vp = p["call_value"], p["put_value"]
    elif "opt_total_C_buy_amt" in p.columns and "opt_total_P_buy_amt" in p.columns:
        vc, vp = p["opt_total_C_buy_amt"], p["opt_total_P_buy_amt"]            # 전체 매수금액 = 전체 거래대금
    if vc is not None:
        ind["cpr_1d"] = flow(vc, vp)
        ind["cpr_3d"] = flow(_cum(vc, 3), _cum(vp, 3))
    return pd.DataFrame(ind, index=p.index)


def lag_to_target(ind, panel):
    """T−1 신호를 T일 행에 둔다. 달력 전체 행에서 shift 하므로 목표 결측과 무관하게 직전 거래일 값이 온다.

    가드: shift 로 실제로 끌려온 출처 날짜(src)를 같은 연산으로 만들어, 패널의 sig_date 와 같고 목표일보다 앞서는지 행 단위로 확인한다.
    """
    if not (ind.index.equals(panel.index) and ind.index.is_unique and ind.index.is_monotonic_increasing):
        raise AssertionError("지표와 패널의 날짜 축이 다름/중복/비정렬")
    lagged = ind.shift(1)
    src = pd.Series(ind.index, index=ind.index).shift(1)
    both = src.notna() & panel["sig_date"].notna()
    if not bool((src[both] == panel.loc[both, "sig_date"]).all()):
        raise AssertionError("룩어헤드: 신호 출처 날짜가 패널 sig_date 와 다름")
    if not bool((src[src.notna()] < src.index[src.notna()]).all()):
        raise AssertionError("룩어헤드: 신호 출처 날짜가 목표 날짜보다 늦거나 같음")
    lagged["sig_date"] = src
    return lagged


def describe(ind):
    rows = []
    for c in ind.columns:
        s = ind[c].dropna()
        if s.empty:
            continue
        rows.append({"indicator": c, "n": len(s), "start": str(s.index.min().date()), "end": str(s.index.max().date()),
                     "mean": s.mean(), "sd": s.std(), "min": s.min(), "max": s.max(), "ar1": s.autocorr(1)})
    return pd.DataFrame(rows)
