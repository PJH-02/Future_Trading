"""16차 스윙 관점 공용 자료 로더 + 점검(python r16s_swing_data.py).

봉인: 모든 로더는 읽은 직후 date <= 2024-12-30 로 자르고 assert 한다(수익 계산 전).
깃 밖 자료(로컬 보조)는 함수 이름에 _ext 를 붙이고 결과에 '깃 밖 자료'로 표기한다.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

CUT = pd.Timestamp("2024-12-30")
G = Path("<repo>")
B = G / "beomgu/data"
JI = G / "jehyeong/intraday/data"
RAW = Path("<raw>")
EXT = RAW / "r15_external"
LAB = Path("<OUT>")


def seal(df, col="date"):
    """날짜 열을 Timestamp 로 바꾸고 봉인 시점으로 자른다."""
    df = df.copy()
    df[col] = pd.to_datetime(df[col])
    df = df[df[col] <= CUT].sort_values(col).reset_index(drop=True)
    assert df[col].max() <= CUT
    return df


def rd(path, datecol="date", **kw):
    df = pd.read_csv(path, encoding="utf-8-sig", **kw)
    if datecol != "date":
        df = df.rename(columns={datecol: "date"})
    return seal(df).set_index("date")


def k200_index():
    """코스피200 지수 일봉. 2014-07-08 이후는 KPI200(fchart), 그 전은 k200_index_daily, 빈 행은 서로 보충(r5 와 같은 규칙)."""
    a = rd(B / "k200_index_daily.csv")[["open", "high", "low", "close"]]
    b = rd(B / "etf/KPI200.csv")[["open", "high", "low", "close"]]
    late = b.loc["2014-07-08":].combine_first(a.loc["2014-07-08":])
    x = pd.concat([a.loc[:"2014-07-07"], late]).sort_index()
    return x[(x["close"] > 0)]


def kq150_index():
    x = rd(JI / "KOSDAQ_150_ohlcv.csv", datecol="Date").rename(columns=str.lower)
    x = x[~x.index.duplicated(keep="last")]        # 원본에 2024-10-08 동일 행 중복
    return x[["open", "high", "low", "close"]].astype(float)


def etf(code):
    """분배금 조정 일봉(수익률 전용). 깃 beomgu 우선, 없으면 깃 밖 etf_extra."""
    p = B / f"etf/{code}.csv"
    if not p.exists():
        p = RAW / f"etf_extra/{code}.csv"
    return rd(p)[["open", "high", "low", "close", "volume"]].astype(float)


def etf_raw(code):
    """jehyeong 원가격 일봉(호가 비용 계산 전용)."""
    return rd(JI / f"{code}_daily.csv")[["open", "high", "low", "close", "volume"]].astype(float)


def fut_ext(name):
    """깃 밖 자료. name: 'k200' | 'kq150'. 날짜별 front 계약 행. 일중 비정상 행 제거."""
    f = pd.read_csv(EXT / f"kis_fut_{name}_daily.csv", dtype={"code": str})
    return seal(f)


def fut_chain(name):
    """근월 연결 수익률(체크리스트 5). KIS front 는 만기 당일부터 차근월. 만기 전날 밤은 만기물을 들고, 만기일 시가에 차근월로 롤한다.
    on = 전일 front 계약의 전일 종가→당일 시가, oc = 당일 front 계약의 시가→종가, cc = (1+on)(1+oc)−1, roll = 당일 시가에 계약 교체."""
    f = fut_ext(name)
    for k in ("open", "high", "low", "close", "volume"):
        f[k] = pd.to_numeric(f[k], errors="coerce")
    fr = f[f["front"] == 1].drop_duplicates("date").set_index("date").sort_index()
    px = f.pivot_table(index="date", columns="code", values="close", aggfunc="last")
    ox = f.pivot_table(index="date", columns="code", values="open", aggfunc="last")
    idx = fr.index
    code = fr["code"].to_numpy()
    rows = []
    for i in range(1, len(idx)):
        d, p, c, cp = idx[i], idx[i - 1], code[i], code[i - 1]
        on = ox.at[d, cp] / px.at[p, cp] - 1
        oc = px.at[d, c] / ox.at[d, c] - 1
        rows.append((d, c, (1 + on) * (1 + oc) - 1, on, oc, int(c != cp), fr["volume"].iloc[i]))
    return pd.DataFrame(rows, columns=["date", "code", "cc", "on", "oc", "roll", "vol"]).set_index("date")


def investor(kind):
    return rd(B / {"kospi": "naver_kospi_investor_daily.csv", "kosdaq": "naver_kosdaq_investor_daily.csv", "fut": "naver_fut_investor_daily.csv"}[kind])


def program(kind):
    return rd(B / {"kospi": "naver_program_daily.csv", "kosdaq": "naver_kosdaq_program_daily.csv"}[kind])


def kq_fut_foreign():
    return rd(JI / "KOSDAQ_150_fut_foreign.csv", datecol="Date")


def us(name):
    return rd(B / f"{name}_daily.csv")[["open", "high", "low", "close"]].astype(float)


def vkospi_ext():
    return rd(EXT / "kis_vkospi_daily.csv").astype(float)


def usdkrw_ext():
    x = pd.read_csv(EXT / "fdr_usdkrw_daily.csv", encoding="utf-8-sig")
    x = x.rename(columns={x.columns[0]: "date"}).rename(columns=str.lower)
    return seal(x).set_index("date")


def kosdaq_comp_ext():
    """깃 밖 자료. 코스닥 종합지수 일봉(1996~). 사전 구간 검증 전용, 거래 차량 아님."""
    x = pd.read_csv(EXT / "naver_kosdaq_index_daily.csv", encoding="utf-8-sig")
    x["date"] = pd.to_datetime(x["localDate"].astype(str), format="%Y%m%d")
    x = x.rename(columns={"closePrice": "close", "openPrice": "open", "highPrice": "high", "lowPrice": "low"})
    return seal(x).set_index("date")[["open", "high", "low", "close"]].astype(float)


def base_streams():
    """기준 흐름: V2 overnight·intraday, 13차 KOSPI_book·KOSDAQ_book (봉인 시점까지 일수익)."""
    n = pd.read_csv(G / "jehyeong/daynight/results/daily_nav.csv", encoding="utf-8-sig")
    n = seal(n[n["version"] == "V2"])
    w = n.pivot_table(index="date", columns="account", values="daily_return")
    b = rd(LAB / "round13_slot_book/book/book_daily.csv")
    return pd.DataFrame({"V2_ON": w["overnight"], "V2_ID": w["intraday"], "KP_book": b["KOSPI_book"], "KQ_book": b["KOSDAQ_book"]})


def main():
    print("== 코스피200 지수")
    k = k200_index()
    x = k.loc["2010-01-01":]
    print(len(x), x.index.min().date(), x.index.max().date(), "결측 O/H/L", x[["open", "high", "low"]].isna().sum().to_dict(),
          "0값", (x[["open", "high", "low"]] <= 0).sum().to_dict(), "H<max(O,C)", int((x["high"] < x[["open", "close"]].max(axis=1) - 1e-6).sum()),
          "L>min(O,C)", int((x["low"] > x[["open", "close"]].min(axis=1) + 1e-6).sum()))
    print("== 코스닥150 지수")
    q = kq150_index()
    x = q.loc["2015-01-01":]
    print(len(q), q.index.min().date(), q.index.max().date(), "2015~ 결측", x.isna().sum().to_dict(), "0값", (x <= 0).sum().to_dict())
    q2 = rd(RAW / "etf_extra/KQ150_index_kis.csv") if (RAW / "etf_extra/KQ150_index_kis.csv").exists() else None
    if q2 is not None:
        print("KIS 지수 열", list(q2.columns)[:8], len(q2))
        c = q2[[c for c in q2.columns if "close" in c.lower()][0]].astype(float)
        j = pd.concat([q["close"], c], axis=1, keys=["a", "b"]).dropna()
        print("공통", len(j), "일수익 상관", round(j.pct_change().corr().iloc[0, 1], 5), "절대차 최대", float((j["a"] / j["b"] - 1).abs().max()))
    print("== ETF")
    for c in ("069500", "122630", "114800", "229200", "233740", "251340", "252670"):
        e = etf(c)
        r = etf_raw(c) if c != "x" else None
        j = pd.concat([e["close"], r["close"]], axis=1, keys=["adj", "raw"]).dropna()
        ratio = (j["adj"] / j["raw"])
        print(c, len(e), e.index.min().date(), e.index.max().date(), "조정/원 비율 처음·끝", round(ratio.iloc[0], 4), round(ratio.iloc[-1], 4),
              "연 평균 원가격", {y: round(v) for y, v in r["close"].groupby(r.index.year).mean().items() if y >= 2015})
    print("== 선물 연결")
    for nm in ("k200", "kq150"):
        fc = fut_chain(nm)
        print(nm, len(fc), fc.index.min().date(), fc.index.max().date(), "결측", fc[["cc", "on", "oc"]].isna().sum().to_dict(), "롤 수", int(fc["roll"].sum()))
        print(" 연도별 front 거래량 중앙값", {y: int(v) for y, v in fc["vol"].groupby(fc.index.year).median().items()})
        big = fc[fc["cc"].abs() > 0.08][["code", "cc", "roll"]]
        print(" 절대 8% 초과일", len(big), big.tail(12).round(4).to_dict("index") if len(big) < 30 else "")
        rr = fc[fc["roll"] == 1]["cc"].abs()
        print(" 롤일 평균 절대 cc", round(rr.mean(), 5), "비롤일", round(fc[fc["roll"] == 0]["cc"].abs().mean(), 5))
    ix = {"k200": k, "kq150": q}
    for nm in ix:
        fc = fut_chain(nm)
        j = pd.concat([fc["cc"], ix[nm]["close"].pct_change()], axis=1, keys=["f", "i"]).dropna()
        print(nm, "선물 cc 와 지수 cc 상관", round(j.corr().iloc[0, 1], 4), "평균 차 bp", round((j["f"] - j["i"]).mean() * 1e4, 2))
    print("== 수급")
    for nm, d in (("kospi", investor("kospi")), ("kosdaq", investor("kosdaq")), ("fut", investor("fut"))):
        print(nm, len(d), d.index.min().date(), d.index.max().date(), "net_foreign 결측", int(d["net_foreign"].isna().sum()),
              "연 중앙 절대값", {y: float(v) for y, v in d["net_foreign"].abs().groupby(d.index.year).median().items() if y in (2011, 2017, 2024)})
    for nm in ("kospi", "kosdaq"):
        d = program(nm)
        print(nm, "프로그램", len(d), d.index.min().date(), d.index.max().date(), d.isna().sum().to_dict())
    d = kq_fut_foreign()
    print("KQ 선물 외인", len(d), d.index.min().date(), d.index.max().date(), d.isna().sum().to_dict(), d.tail(2).to_dict("index"))
    print("합계 total 0 비율", round((d["total"] == 0).mean(), 3))
    print("== 미국")
    for nm in ("spx", "ndx_comp", "nq", "es"):
        d = us(nm)
        print(nm, len(d), d.index.min().date(), d.index.max().date())
    print("== 기준 흐름")
    b = base_streams()
    print(b.count().to_dict(), b.index.min().date(), b.index.max().date())
    print("== 깃 밖")
    print(vkospi_ext().index.min().date(), usdkrw_ext().columns.tolist())


if __name__ == "__main__":
    main()
