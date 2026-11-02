# -*- coding: utf-8 -*-
"""15차 설계 04_us_calendar: 밤 다리 미국 달력 오버레이(매크로 발표 밤 가중·미국 휴장 밤 제외) + 독립 발표 밤 다리.

규칙(사전 등록 alpha-share-prereg-2026-10-30.md, design_shortlist.md '### 04_us_calendar', 결과 보기 전 고정)
  H = NYSE 휴장일(전일 휴장) 중 한국 거래일(069500 거래일 달력).
  A = FOMC 성명일 ∪ BLS CPI 발표일 ∪ BLS 고용(Employment Situation) 발표일 중 한국 거래일.
      CPI·고용 08:30 ET = KST 21:30/22:30, FOMC 14:00 ET = KST 익일 03:00/04:00 → 모두 발표일 D(미국 달력일) 의 한국 밤 창
      (D 15:30 → 다음 거래일 09:00) 안. 발표일이 한국 휴장이면 제외(이동 없음).
  밤 수익 ret_t = ln(O_069500,t' / C_069500,t), t' = 다음 069500 거래일. 발표 밤 롱 순수익 = ret − 5bp(왕복, 자본 1 단위).
  Δ = mean(ret | A) − mean(ret | ¬A ∧ ¬H): 더미 회귀 HAC(maxlags 5).
  회귀 ret ~ 1 + 1[A] + 1[H] + r_us,t + 요일 더미(월 기준), HAC(maxlags 5).
  r_us,t = ln(C_us(≤ t'−1일) / C_us(≤ t−1일)): 한국 밤 창 안에 든 미국 정규장 누적 로그 수익
      (spx_daily.csv, 2026.9.22 이후는 yf_SPY adj 로 이어 붙임; 미국 휴장 밤은 0).
  창: 주 2005-01-03~2026-09-29(FOMC 표 시작), W_all·W_ex2026·W_long 병기. 창 적용은 청산일(t') 기준. 2026-10-30 행 없음.
  변형(사전 등록 6): V1 휴장 밤 단독 Δ(요일 대조, estimate) / V2 FOMC 만 / V3 CPI·고용 만(CPI·고용 각각도 보고용으로 추가)
      / V4 2016 이후 분리 / V5 229200 판(2015-10~) / V6 오버레이 w = 0(H), 1.5(A, 코스피만), 1.0.
  판정(고정): 주 셀 공통 규칙(순수익 > 0 ∧ NW t ≥ 2 ∧ 연도 부호 ≥ 70% ∧ 상위 5건 < 50% ∧ 전·후반 양수)
      ∧ Δ NW t ≥ 2 ∧ 1[A] 계수 HAC t ≥ 2 ∧ V4(2016 이후) 순수익 양수 → candidate. 순수익 ≤ 0 reject, 그 밖 reference,
      창 ≤ 300 거래일 또는 이벤트 < 10 estimate. V1 은 estimate 고정.
      V6 오버레이 = 13차 합산 장부에 적용해 W_all·W_long CAGR·MDD 동시 개선 ∧ 2024·2025 연수익 개선. H→0 몫만의 오버레이는
      CAGR·MDD 비악화로 완화.
  오버레이 적용: jehyeong_ON·blog_ON_Q 의 date = 청산일, 진입일 = 직전 거래일. 진입일 ∈ H → 그 밤 0(비용도 0).
      진입일 ∈ A(∧ ∉ H) ∧ 코스피 → 자본 50% 를 122630 으로 바꾼 1.5 배: net + 0.5 × (r_122630,co − r_069500,co)
      (122630 밤 잔차를 실측값으로 차감). 'simple' 판은 1.5 × net(잔차 무시, 진단용). 코스닥은 2x 한도라 A 가중 1.0.
자료(원본 수정 없음)
  hyfe/data/raw/etf/069500.csv(거래일 달력, 분배금 조정가 → 수익률만), 229200.csv, 122630.csv, spx_daily.csv, us_etf/yf_SPY.csv,
  round13_slot_book/series/jehyeong_ON·blog_ON_Q(+meta), book/book_daily.csv.
  달력(2026-10-30 WebFetch 로 확보, 아래 상수에 내장. 산출 폴더에 calendar_*.csv 로 저장):
    NYSE 공식 2025~2027 https://www.nyse.com/markets/hours-calendars ; 2002~2024 는 사전 등록 대체 규칙
      (spx_daily·yf_SPY 결측 평일) 로 채우고 특별 휴장(2004-06-11, 2007-01-02, 2012-10-29·30, 2018-12-05, 2025-01-09) 일치 확인.
    FOMC 2005~2020 fomchistorical{YYYY}.htm(회의 마지막 날 + 성명이 나온 비정례 회의의 성명일), 2021~2026 fomccalendars.htm.
    BLS CPI·고용: 보도자료 아카이브 파일명(cpi_MMDDYYYY, empsit_MMDDYYYY = 실제 발표일, 셧다운 지연·미발표 반영)
      https://www.bls.gov/bls/news-release/cpi.htm , /empsit.htm ; 2005년 1월 발표분은 https://www.bls.gov/schedule/2005/home.htm .
가정: 단일가 체결(15:30 종가 매수·09:00 시가 매도, 13차 밤 다리 규약 그대로). 달력은 사전 공표 일정이라 사전 신호.
      통계(평균·t·Δ·회귀)는 로그 수익, series·장부 결합은 단순 수익(Π(1+r)) 사용.
실행: PYTHONUTF8=1 python "<code>/topics/r15_04_us_calendar.py"
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.stdout.reconfigure(encoding="utf-8")
plt.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

LAB = Path("<OUT>")
R13 = LAB / "round13_slot_book"
R15 = LAB / "round15_alpha_share"
OUT = R15 / "04_us_calendar"
SER = R15 / "series"
RAW = Path("<raw>")
COST = 5e-4
WIN = {"full": ("2005-01-03", "2026-09-29"), "W_all": ("2022-12-13", "2026-09-17"), "W_ex2026": ("2022-12-13", "2025-12-30"),
       "W_long": ("2016-10-04", "2026-09-17"), "since2016": ("2016-01-04", "2026-09-29"), "pre2016": ("2005-01-03", "2015-12-30"),
       "V5": ("2015-10-01", "2026-09-29")}
HANDLE = "beomgu_USCAL_K"

# ------------------------------------------------------------------ 달력 상수(WebFetch 2026-10-30)
NYSE_URL = "https://www.nyse.com/markets/hours-calendars"
NYSE_OFFICIAL = """2025-01-01 New Year's Day|2025-01-20 Martin Luther King, Jr. Day|2025-02-17 Washington's Birthday|2025-04-18 Good Friday
2025-05-26 Memorial Day|2025-06-19 Juneteenth|2025-07-04 Independence Day|2025-09-01 Labor Day|2025-11-27 Thanksgiving Day|2025-12-25 Christmas Day
2026-01-01 New Year's Day|2026-01-19 Martin Luther King, Jr. Day|2026-02-16 Washington's Birthday|2026-04-03 Good Friday|2026-05-25 Memorial Day
2026-06-19 Juneteenth|2026-07-03 Independence Day (observed)|2026-09-07 Labor Day|2026-11-26 Thanksgiving Day|2026-12-25 Christmas Day
2027-01-01 New Year's Day|2027-01-18 Martin Luther King, Jr. Day|2027-02-15 Washington's Birthday|2027-03-26 Good Friday|2027-05-31 Memorial Day
2027-06-18 Juneteenth (observed)|2027-07-05 Independence Day (observed)|2027-09-06 Labor Day|2027-11-25 Thanksgiving Day|2027-12-24 Christmas Day (observed)"""
NYSE_SPECIAL = {"2004-06-11": "Reagan funeral", "2007-01-02": "Ford funeral", "2012-10-29": "Hurricane Sandy", "2012-10-30": "Hurricane Sandy",
                "2018-12-05": "G.H.W. Bush funeral", "2025-01-09": "Carter funeral"}
FOMC_URL = {y: f"https://www.federalreserve.gov/monetarypolicy/fomchistorical{y}.htm" for y in range(2005, 2021)}
FOMC_URL.update({y: "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm" for y in range(2021, 2028)})
FOMC_SCHED = """2005: 02-02 03-22 05-03 06-30 08-09 09-20 11-01 12-13
2006: 01-31 03-28 05-10 06-29 08-08 09-20 10-25 12-12
2007: 01-31 03-21 05-09 06-28 08-07 09-18 10-31 12-11
2008: 01-30 03-18 04-30 06-25 08-05 09-16 10-29 12-16
2009: 01-28 03-18 04-29 06-24 08-12 09-23 11-04 12-16
2010: 01-27 03-16 04-28 06-23 08-10 09-21 11-03 12-14
2011: 01-26 03-15 04-27 06-22 08-09 09-21 11-02 12-13
2012: 01-25 03-13 04-25 06-20 08-01 09-13 10-24 12-12
2013: 01-30 03-20 05-01 06-19 07-31 09-18 10-30 12-18
2014: 01-29 03-19 04-30 06-18 07-30 09-17 10-29 12-17
2015: 01-28 03-18 04-29 06-17 07-29 09-17 10-28 12-16
2016: 01-27 03-16 04-27 06-15 07-27 09-21 11-02 12-14
2017: 02-01 03-15 05-03 06-14 07-26 09-20 11-01 12-13
2018: 01-31 03-21 05-02 06-13 08-01 09-26 11-08 12-19
2019: 01-30 03-20 05-01 06-19 07-31 09-18 10-30 12-11
2020: 01-29 04-29 06-10 07-29 09-16 11-05 12-16
2021: 01-27 03-17 04-28 06-16 07-28 09-22 11-03 12-15
2022: 01-26 03-16 05-04 06-15 07-27 09-21 11-02 12-14
2023: 02-01 03-22 05-03 06-14 07-26 09-20 11-01 12-13
2024: 01-31 03-20 05-01 06-12 07-31 09-18 11-07 12-18
2025: 01-29 03-19 05-07 06-18 07-30 09-17 10-29 12-10
2026: 01-28 03-18 04-29 06-17 07-29 09-16 10-28 12-09"""
# 비정례 회의·전화회의 중 정책 성명이 나온 날(성명일 기준). 2010-05-09·2020-03-15 는 일요일이라 한국 거래일 필터에서 자동 제외.
# 제외: 성명 없는 전화회의(2007-12-06, 2008-01-09, 2008-07-24, 2008-09-29, 2009 4회, 2010-10-15, 2011 2회, 2013-10-16, 2014-03-04),
#       통지투표 보도자료(2020-03-19, 2020-03-31), 장기 목표 성명(2020-08-27, 2025-08-22), 취소 회의(2020-03-17·18).
FOMC_UNSCHED = [("2007-08-10", "전화회의 08-10 성명 당일"), ("2007-08-17", "전화회의 08-16 → 성명 08-17(monetary20070817b)"),
                ("2008-01-22", "전화회의 01-21 → 성명 01-22 08:20 ET(monetary20080122b)"), ("2008-03-11", "전화회의 03-10 → 성명 03-11(monetary20080311a)"),
                ("2008-10-08", "전화회의 10-07 → 성명 10-08 07:00 ET(monetary20081008a)"), ("2010-05-09", "전화회의 05-09 성명 당일(일요일)"),
                ("2019-10-11", "화상회의 10-04 → 성명 10-11"), ("2020-03-03", "전화회의 03-02 → 성명 03-03"),
                ("2020-03-15", "전화회의 03-15 성명 당일(일요일)"), ("2020-03-23", "통지투표 03-23 성명 당일")]
BLS_CPI_URL = "https://www.bls.gov/bls/news-release/cpi.htm"
BLS_EMP_URL = "https://www.bls.gov/bls/news-release/empsit.htm"
BLS_2005_URL = "https://www.bls.gov/schedule/2005/home.htm"
BLS_CPI = """01192005 02232005 03232005 04202005 05182005 06152005 07142005 08162005 09152005 10142005 11162005 12152005
01182006 02222006 03162006 04192006 05172006 06142006 07192006 08162006 09152006 10182006 11162006 12152006
01182007 02212007 03162007 04172007 05152007 06152007 07182007 08152007 09192007 10172007 11152007 12142007
01162008 02202008 03142008 04162008 05142008 06132008 07162008 08142008 09162008 10162008 11192008 12162008
01162009 02202009 03182009 04152009 05152009 06172009 07152009 08142009 09162009 10152009 11182009 12162009
01152010 02192010 03182010 04142010 05192010 06172010 07162010 08132010 09172010 10152010 11172010 12152010
01142011 02172011 03172011 04152011 05132011 06152011 07152011 08182011 09152011 10192011 11162011 12162011
01192012 02172012 03162012 04132012 05152012 06142012 07172012 08152012 09142012 10162012 11152012 12142012
01162013 02212013 03152013 04162013 05162013 06182013 07162013 08152013 09172013 10302013 11202013 12172013
01162014 02202014 03182014 04152014 05152014 06172014 07222014 08192014 09172014 10222014 11202014 12172014
01162015 02262015 03242015 04172015 05222015 06182015 07172015 08192015 09162015 10152015 11172015 12152015
01202016 02192016 03162016 04142016 05172016 06162016 07152016 08162016 09162016 10182016 11172016 12152016
01182017 02152017 03152017 04142017 05122017 06142017 07142017 08112017 09142017 10132017 11152017 12132017
01122018 02142018 03132018 04112018 05102018 06122018 07122018 08102018 09132018 10112018 11142018 12122018
01112019 02132019 03122019 04102019 05102019 06122019 07112019 08132019 09122019 10102019 11132019 12112019
01142020 02132020 03112020 04102020 05122020 06102020 07142020 08122020 09112020 10132020 11122020 12102020
01132021 02102021 03102021 04132021 05122021 06102021 07132021 08112021 09142021 10132021 11102021 12102021
01122022 02102022 03102022 04122022 05112022 06102022 07132022 08102022 09132022 10132022 11102022 12132022
01122023 02142023 03142023 04122023 05102023 06132023 07122023 08102023 09132023 10122023 11142023 12122023
01112024 02132024 03122024 04102024 05152024 06122024 07112024 08142024 09112024 10102024 11132024 12112024
01152025 02122025 03122025 04102025 05132025 06112025 07152025 08122025 09112025 10242025 12182025
01132026 02132026 03112026 04102026 05122026 06102026 07142026 08122026 09112026 10142026"""
BLS_EMP = """01072005 02042005 03042005 04012005 05062005 06032005 07082005 08052005 09022005 10072005 11042005 12022005
01062006 02032006 03102006 04072006 05052006 06022006 07072006 08042006 09012006 10062006 11032006 12082006
01052007 02022007 03092007 04062007 05042007 06012007 07062007 08032007 09072007 10052007 11022007 12072007
01042008 02012008 03072008 04042008 05022008 06062008 07032008 08012008 09052008 10032008 11072008 12052008
01092009 02062009 03062009 04032009 05082009 06052009 07022009 08072009 09042009 10022009 11062009 12042009
01082010 02052010 03052010 04022010 05072010 06042010 07022010 08062010 09032010 10082010 11052010 12032010
01072011 02042011 03042011 04012011 05062011 06032011 07082011 08052011 09022011 10072011 11042011 12022011
01062012 02032012 03092012 04062012 05042012 06012012 07062012 08032012 09072012 10052012 11022012 12072012
01042013 02012013 03082013 04052013 05032013 06072013 07052013 08022013 09062013 10222013 11082013 12062013
01102014 02072014 03072014 04042014 05022014 06062014 07032014 08012014 09052014 10032014 11072014 12052014
01092015 02062015 03062015 04032015 05082015 06052015 07022015 08072015 09042015 10022015 11062015 12042015
01082016 02052016 03042016 04012016 05062016 06032016 07082016 08052016 09022016 10072016 11042016 12022016
01062017 02032017 03102017 04072017 05052017 06022017 07072017 08042017 09012017 10062017 11032017 12082017
01052018 02022018 03092018 04062018 05042018 06012018 07062018 08032018 09072018 10052018 11022018 12072018
01042019 02012019 03082019 04052019 05032019 06072019 07052019 08022019 09062019 10042019 11012019 12062019
01102020 02072020 03062020 04032020 05082020 06052020 07022020 08072020 09042020 10022020 11062020 12042020
01082021 02052021 03052021 04022021 05072021 06042021 07022021 08062021 09032021 10082021 11052021 12032021
01072022 02042022 03042022 04012022 05062022 06032022 07082022 08052022 09022022 10072022 11042022 12022022
01062023 02032023 03102023 04072023 05052023 06022023 07072023 08042023 09012023 10062023 11032023 12082023
01052024 02022024 03082024 04052024 05032024 06072024 07052024 08022024 09062024 10042024 11012024 12062024
01102025 02072025 03072025 04042025 05022025 06062025 07032025 08012025 09052025 11202025 12162025
01092026 02112026 03062026 04032026 05082026 06052026 07022026 08072026 09042026 10022026"""


def mmddyyyy(s):
    return pd.to_datetime([f"{t[4:8]}-{t[0:2]}-{t[2:4]}" for t in s.split()])


def build_calendars(us_days, last_us):
    """달력 4종 DataFrame(date, source_url, ...) 과 집합. us_days = 미국 정규장이 있었던 날(spx ∪ SPY)."""
    off = [(pd.Timestamp(x[:10]), x[11:]) for line in NYSE_OFFICIAL.splitlines() for x in line.split("|")]
    official = pd.DataFrame({"date": [d for d, _ in off], "name": [n for _, n in off], "source_url": NYSE_URL})
    bdays = pd.bdate_range("2002-10-14", last_us)
    fb = bdays.difference(pd.DatetimeIndex(us_days))
    fallback = pd.DataFrame({"date": fb, "name": [NYSE_SPECIAL.get(d.strftime("%Y-%m-%d"), "") for d in fb],
                             "source_url": "spx_daily.csv·yf_SPY.csv 결측 평일(사전 등록 대체 규칙)"})
    chk_off = official[official.date <= last_us]
    missing = chk_off[~chk_off.date.isin(fb)]
    assert missing.empty, f"공식 휴장일인데 미국 시세가 있음: {missing.date.tolist()}"
    extra = fallback[(fallback.date >= "2025-01-01") & ~fallback.date.isin(official.date)]
    nyse = pd.concat([official, fallback[~fallback.date.isin(official.date)]]).sort_values("date").reset_index(drop=True)
    nyse["kind"] = np.where(nyse.date.isin(official.date), "official", "fallback")
    # FOMC
    rows = []
    for line in FOMC_SCHED.splitlines():
        y, days = line.split(":")
        for md in days.split():
            rows.append((pd.Timestamp(f"{y.strip()}-{md}"), "scheduled", "회의 마지막 날", FOMC_URL[int(y)]))
    for d, note in FOMC_UNSCHED:
        rows.append((pd.Timestamp(d), "unscheduled", note, FOMC_URL[int(d[:4])]))
    fomc = pd.DataFrame(rows, columns=["date", "kind", "note", "source_url"]).sort_values("date").reset_index(drop=True)
    assert not fomc.date.duplicated().any()
    cpi = pd.DataFrame({"date": mmddyyyy(BLS_CPI)}).sort_values("date").reset_index(drop=True)
    cpi["source_url"] = np.where(cpi.date == "2005-01-19", BLS_2005_URL, BLS_CPI_URL)
    emp = pd.DataFrame({"date": mmddyyyy(BLS_EMP)}).sort_values("date").reset_index(drop=True)
    emp["source_url"] = np.where(emp.date == "2005-01-07", BLS_2005_URL, BLS_EMP_URL)
    assert not cpi.date.duplicated().any() and not emp.date.duplicated().any()
    return nyse, fomc, cpi, emp, extra


# ------------------------------------------------------------------ 시세·밤 패널
def us_log_level():
    spx = pd.read_csv(RAW / "spx_daily.csv", parse_dates=["date"]).set_index("date")["close"].dropna().sort_index()
    spy = pd.read_csv(RAW / "us_etf/yf_SPY.csv", parse_dates=["date"]).set_index("date")["adj"].dropna().sort_index()
    lr = np.log(spx).diff().dropna()
    lr_spy = np.log(spy).diff().dropna()
    lr = pd.concat([lr, lr_spy[lr_spy.index > lr.index[-1]]])
    assert lr.index.is_monotonic_increasing and not lr.index.duplicated().any()
    us_days = sorted(set(spx.index) | set(spy.index))
    return lr.cumsum(), us_days, spx.index[-1], spy.index[-1]


def nights(code, sub, level, sets):
    px = pd.read_csv(RAW / sub / f"{code}.csv", parse_dates=["date"]).set_index("date").sort_index()
    assert not px.index.duplicated().any() and (px[["open", "close"]] > 0).all().all()
    d = pd.DataFrame({"entry": px.index[:-1], "exit": px.index[1:],
                      "ret": np.log(px["open"].values[1:] / px["close"].values[:-1]),
                      "simple": px["open"].values[1:] / px["close"].values[:-1] - 1})
    d["wd"] = d.entry.dt.weekday
    for k, s in sets.items():
        d[k] = d.entry.isin(s)
    d["A"] = d.fomc | d.cpi | d.emp
    d["cpiemp"] = d.cpi | d.emp
    d["ctrl"] = ~d.A & ~d.H
    one = pd.Timedelta(days=1)
    d["r_us"] = level.asof(d.exit - one).values - level.asof(d.entry - one).values
    d["gap_days"] = (d.exit - d.entry).dt.days
    return d.set_index("exit")


# ------------------------------------------------------------------ 통계
def hac(y, X=None):
    y = np.asarray(y, float)
    X = np.ones((len(y), 1)) if X is None else np.asarray(X, float)
    return sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 5})


def dummy_delta(d, treat, control, weekday=False):
    """Δ = 처리 평균 − 대조 평균(더미 회귀 HAC). 표본 = 처리 ∪ 대조."""
    s = d[treat | control]
    X = pd.DataFrame({"const": 1.0, "T": treat[s.index].astype(float)}, index=s.index)
    if weekday:
        for w in (1, 2, 3, 4):
            X[f"wd{w}"] = (s.wd == w).astype(float)
    r = hac(s.ret * 1e4, X)
    return float(r.params[1]), float(r.tvalues[1]), int(treat[s.index].sum()), int(control[s.index].sum())


def full_reg(d, flags):
    """ret(bp) ~ 1 + 더미들 + r_us(bp) + 요일 더미. 반환 {이름: (계수, t)}."""
    X = pd.DataFrame({"const": 1.0}, index=d.index)
    for f in flags:
        X[f] = d[f].astype(float)
    X["r_us"] = d.r_us * 1e4
    for w in (1, 2, 3, 4):
        X[f"wd{w}"] = (d.wd == w).astype(float)
    r = hac(d.ret * 1e4, X)
    return {c: (float(r.params[i]), float(r.tvalues[i])) for i, c in enumerate(X.columns)}, int(r.nobs), float(r.rsquared)


def cell_stats(tr, cal, w, cost=COST):
    w0, w1 = pd.Timestamp(w[0]), pd.Timestamp(w[1])
    tr = tr[(tr.index >= w0) & (tr.index <= w1)].sort_index()
    cal_w = cal[(cal >= w0) & (cal <= w1)]
    assert tr.index.isin(cal_w).all()
    n, ndays = len(tr), len(cal_w)
    net = (tr.ret - cost) * 1e4
    gross = tr.ret * 1e4
    daily = pd.Series(0.0, index=cal_w)
    daily.loc[tr.index] = tr.simple - cost
    eq = (1 + daily).cumprod()
    cagr = eq.iloc[-1] ** (252 / ndays) - 1
    mdd = float((eq / eq.cummax() - 1).min())
    t_nw = float(hac(net).tvalues[0]) if n >= 3 else np.nan
    ys = [(y, g) for y, g in tr.groupby(tr.index.year) if len(g) >= 10]
    ypos, ytot = sum(float((g.ret - cost).mean()) > 0 for _, g in ys), len(ys)
    tot = net.sum()
    top5 = net.nlargest(5).sum() / tot * 100 if tot > 0 else np.nan
    h = n // 2
    return dict(n=n, ndays=ndays, gross_bp=float(gross.mean()) if n else np.nan, net_bp=float(net.mean()) if n else np.nan, t_nw=t_nw,
                win_pct=float((net > 0).mean() * 100) if n else np.nan, cagr_pct=cagr * 100, mdd_pct=mdd * 100,
                yearly_pos=f"{ypos}/{ytot}({ypos / ytot * 100:.0f}%)" if ytot else "0/0", ypct=ypos / ytot * 100 if ytot else np.nan,
                top5_share_pct=top5, half1_net_bp=float(net.iloc[:h].mean()) if h else np.nan, half2_net_bp=float(net.iloc[h:].mean()) if n - h else np.nan,
                daily=daily)


def verdict_common(s):
    if s["n"] < 10 or s["ndays"] <= 300:
        return "estimate"
    if s["net_bp"] <= 0:
        return "reject"
    ok = s["t_nw"] >= 2 and s["ypct"] >= 70 and s["top5_share_pct"] < 50 and s["half1_net_bp"] > 0 and s["half2_net_bp"] > 0
    return "candidate" if ok else "reference"


# ------------------------------------------------------------------ 13차 장부(leverage_desk 와 같은 재현)
LEGS = {"KOSPI_A_jehyeong_ON": "jehyeong_ON", "KOSPI_B_beomgu_C3VT30": "beomgu_C3VT30", "KOSPI_C_jehyeong_KP": "jehyeong_KP",
        "KOSDAQ_A_blog_ON_Q": "blog_ON_Q", "KOSDAQ_B_beomgu_C3Q": "beomgu_C3Q", "KOSDAQ_C_jehyeong_KQ": "jehyeong_KQ"}


def series13(h):
    s = pd.read_csv(R13 / "series" / f"{h}.csv", parse_dates=["date"]).set_index("date")
    meta = json.loads((R13 / "series" / f"{h}.meta.json").read_text(encoding="utf-8"))
    return s.loc[meta["first_valid"]:meta["last_valid"]].fillna(0.0)


def panel_wall():
    book = pd.read_csv(R13 / "book/book_daily.csv", parse_dates=["date"]).set_index("date")
    legs = {}
    for col, h in LEGS.items():
        s = series13(h).reindex(book.index).fillna(0.0)
        net = book[col]
        act = net != 0
        assert np.allclose(net[act], s["net"][act]), col
        legs[col] = pd.DataFrame({"net": net, "pos": s["pos"].where(act, 0.0)})
    k = sum(l["net"] for c, l in legs.items() if c.startswith("KOSPI"))
    q = sum(l["net"] for c, l in legs.items() if c.startswith("KOSDAQ"))
    assert np.allclose(k, book["KOSPI_book"]) and np.allclose(q, book["KOSDAQ_book"]) and np.allclose(0.5 * (k + q), book["comb_a"])
    return legs


def panel_long(cols, w):
    legs = {c: series13(LEGS[c]).loc[w[0]:w[1]][["net", "pos"]] for c in cols}
    idx = None
    for l in legs.values():
        idx = l.index if idx is None else idx.intersection(l.index)
    return {c: l.reindex(idx) for c, l in legs.items()}


def comb(legs):
    k = sum(l["net"] for c, l in legs.items() if c.startswith("KOSPI"))
    q = sum(l["net"] for c, l in legs.items() if c.startswith("KOSDAQ"))
    return 0.5 * (k + q)


def metrics(r):
    eq = (1 + r).cumprod()
    return dict(n_days=len(r), cagr=eq.iloc[-1] ** (252 / len(r)) - 1, mdd=float((eq / eq.cummax() - 1).min()),
                sharpe=float(r.mean()) * 252 / (float(r.std(ddof=1)) * np.sqrt(252)))


def yr(r, y):
    return float((1 + r.loc[str(y)]).prod() - 1)


def apply_overlay(legs, mode, cal, Hset, Aset, resid):
    """mode ∈ {H, A, full, full_simple}. 반환 (legs', 증분 comb 수익 시계열, 영향 밤 수)."""
    out = {c: l.copy() for c, l in legs.items()}
    inc = pd.Series(0.0, index=next(iter(legs.values())).index)
    n_aff = 0
    for col in ("KOSPI_A_jehyeong_ON", "KOSDAQ_A_blog_ON_Q"):
        if col not in out:
            continue
        l = out[col]
        act = l["pos"] != 0
        pos_in_cal = cal.get_indexer(l.index)
        assert (pos_in_cal > 0).all()
        entry = cal[pos_in_cal - 1]
        Hn = pd.Series(entry.isin(Hset), index=l.index)
        An = pd.Series(entry.isin(Aset), index=l.index)
        zero = act & Hn if mode in ("H", "full", "full_simple") else pd.Series(False, index=l.index)
        up = act & An & ~Hn if (mode in ("A", "full", "full_simple") and col.startswith("KOSPI")) else pd.Series(False, index=l.index)
        new = l["net"].copy()
        new[zero] = 0.0
        new[up] = 1.5 * l["net"][up] if mode == "full_simple" else l["net"][up] + 0.5 * resid.reindex(l.index)[up]
        inc += 0.5 * (new - l["net"])
        out[col]["net"] = new
        out[col]["pos"] = l["pos"].where(~zero, 0.0).where(~up, l["pos"] * 1.5)
        n_aff += int((zero | up).sum())
    return out, inc, n_aff


# ------------------------------------------------------------------ 그림(팔레트: 레버리지 데스크와 같은 dataviz 슬롯 1~4)
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
GRAY = "#8a8985"


def style(ax):
    ax.grid(alpha=0.25)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def fig_bars(groups, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=False)
    for ax, (wname, rows) in zip(axes, groups.items()):
        labels = [r[0] for r in rows]
        vals = [r[1] for r in rows]
        cols = [GRAY if r[0].startswith("대조") else C[0] for r in rows]
        bars = ax.bar(range(len(rows)), vals, width=0.6, color=cols)
        ax.axhline(0, color="#52514e", lw=0.8)
        lo, hi = min(0.0, min(vals)), max(0.0, max(vals))
        ax.set_ylim(lo - 0.32 * (hi - lo), hi + 0.32 * (hi - lo))
        for b, r in zip(bars, rows):
            ax.annotate(f"{r[1]:+.1f}bp\nt {r[2]:+.2f}\nn {r[3]}", (b.get_x() + b.get_width() / 2, r[1]), xytext=(0, 4 if r[1] >= 0 else -4),
                        textcoords="offset points", ha="center", va="bottom" if r[1] >= 0 else "top", fontsize=8, color="#52514e")
        ax.set_xticks(range(len(rows)))
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_title(f"069500 밤 평균 로그 수익(비용 0, NW t HAC 5) · {wname}", fontsize=9.5, loc="left")
        ax.set_ylabel("bp/밤")
        style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fig_book(curves, path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax, (wname, series) in zip(axes, curves.items()):
        for (label, r), col in zip(series, C):
            eq = (1 + r).cumprod()
            m = metrics(r)
            ax.plot(eq.index, eq, lw=1.6, color=col, label=f"{label}: CAGR {m['cagr'] * 100:.1f}% / MDD {m['mdd'] * 100:.1f}%")
        ax.set_yscale("log")
        ax.set_title(f"13차 합산 장부 vs 달력 오버레이 · {wname}", fontsize=10, loc="left")
        ax.set_ylabel("누적(로그)")
        ax.legend(frameon=False, fontsize=8, loc="upper left")
        style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fig_yearly(tr, path):
    g = tr.groupby(tr.index.year)
    years = list(g.groups)
    vals = [float(((x.ret - COST) * 1e4).mean()) for _, x in g]
    ns = [len(x) for _, x in g]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(years, vals, width=0.6, color=C[0])
    ax.axhline(0, color="#52514e", lw=0.8)
    lo, hi = min(0.0, min(vals)), max(0.0, max(vals))
    ax.set_ylim(lo - 0.15 * (hi - lo), hi + 0.1 * (hi - lo))
    for y, v, n in zip(years, vals, ns):
        ax.annotate(f"n{n}", (y, v), xytext=(0, 3 if v >= 0 else -11), textcoords="offset points", ha="center", fontsize=7, color="#52514e")
    ax.set_title("주 셀 069500 발표 밤 롱 연도별 평균 순수익(5bp 차감) bp/건", fontsize=10, loc="left")
    ax.set_ylabel("bp/건")
    style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ------------------------------------------------------------------ 메인
def fmt(x, nd=2, sign=False):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    s = f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"
    return s.replace("-", "−")


def dstr(d):
    return pd.Timestamp(d).strftime("%Y.%m.%d")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    SER.mkdir(parents=True, exist_ok=True)
    level, us_days, spx_last, spy_last = us_log_level()
    nyse, fomc, cpi, emp, nyse_extra = build_calendars(us_days, spy_last)
    for name, df in [("calendar_nyse_holidays", nyse), ("calendar_fomc", fomc), ("calendar_bls_cpi", cpi), ("calendar_bls_emp", emp)]:
        df.assign(date=df.date.dt.strftime("%Y-%m-%d")).to_csv(OUT / f"{name}.csv", index=False, encoding="utf-8-sig")
    sets = {"fomc": set(fomc.date), "cpi": set(cpi.date), "emp": set(emp.date), "H": set(nyse.date)}
    k = nights("069500", "etf", level, sets)
    cal = pd.read_csv(RAW / "etf/069500.csv", parse_dates=["date"])["date"]
    cal = pd.DatetimeIndex(cal.sort_values())
    q = nights("229200", "etf", level, sets)
    assert q.index.isin(cal).all() and q.entry.isin(cal).all()
    # r_us 점검: 통상 밤(다음 달력일이 거래일, 미국 개장) 은 spx 당일 로그 수익과 같아야 함
    spx = pd.read_csv(RAW / "spx_daily.csv", parse_dates=["date"]).set_index("date")["close"].dropna()
    spx_lr = np.log(spx).diff()
    norm = k[(k.gap_days == 1) & ~k.H & (k.entry <= spx_last)]
    chk = norm.r_us.values - spx_lr.reindex(norm.entry).values
    chk = chk[~np.isnan(chk)]
    assert np.nanmax(np.abs(chk)) < 1e-9, np.nanmax(np.abs(chk))
    assert np.allclose(k.r_us[k.H & (k.gap_days == 1)], 0.0)
    k.to_csv(OUT / "nights_069500.csv", float_format="%.8f", encoding="utf-8-sig")
    q.to_csv(OUT / "nights_229200.csv", float_format="%.8f", encoding="utf-8-sig")

    # 자료 점검 출력
    full = WIN["full"]
    kf = k[(k.index >= full[0]) & (k.index <= full[1])]
    print(f"[자료] 069500 {cal[0].date()}~{cal[-1].date()} {len(cal)}일, 밤 패널 {len(k)}, 주 창 밤 {len(kf)}; spx 마지막 유효 {spx_last.date()}, SPY 로 {spy_last.date()} 까지 연장")
    print(f"[달력] NYSE 휴장 {len(nyse)}(공식 {int((nyse.kind == 'official').sum())}, 대체 {int((nyse.kind == 'fallback').sum())}), "
          f"2025 이후 대체에만 있는 날 {nyse_extra.date.dt.strftime('%Y-%m-%d').tolist()}; FOMC {len(fomc)}(비정례 {int((fomc.kind == 'unscheduled').sum())}); CPI {len(cpi)}; 고용 {len(emp)}")
    for nm, df in [("CPI", cpi), ("고용", emp), ("FOMC", fomc)]:
        wd = df.date.dt.weekday.value_counts().sort_index().to_dict()
        print(f"   {nm} 요일 분포(월=0) {wd}, 범위 {df.date.min().date()}~{df.date.max().date()}")
    in_kr = {nm: int(pd.Index(s).isin(cal).sum()) for nm, s in sets.items()}
    print(f"[한국 거래일 안] {in_kr}; 주 창 밤: A {int(kf.A.sum())}(FOMC {int(kf.fomc.sum())}, CPI {int(kf.cpi.sum())}, 고용 {int(kf.emp.sum())}, 겹침 {int((kf.fomc & kf.cpiemp).sum() + (kf.cpi & kf.emp).sum())}), "
          f"H {int(kf.H.sum())}, A∩H {int((kf.A & kf.H).sum())}, 대조 {int(kf.ctrl.sum())}")

    # ---------------- 독립 셀
    cells, trades, regs = [], [], []
    main_extra = {}

    def add(cid, name, market, wname, tr, d, treat_col, ctrl_col="ctrl", weekday=False, forced=None, note="", cost=COST, cal_=cal, reg_flags=None):
        w = WIN[wname]
        s = cell_stats(tr, cal_, w, cost)
        dw = d[(d.index >= w[0]) & (d.index <= w[1])]
        delta, t_delta, n_t, n_c = dummy_delta(dw, dw[treat_col], dw[ctrl_col], weekday)
        reg, nobs, r2 = full_reg(dw, reg_flags or ["A", "H"])
        coef_name = treat_col if treat_col in reg else ("cpiemp" if treat_col in ("cpi", "emp") else "A")
        v = forced or verdict_common(s)
        hpart = "" if coef_name == "H" else f", 1[H] {fmt(reg['H'][0], 2, True)}bp t {fmt(reg['H'][1], 2, True)}"
        rnote = f"회귀 1[{coef_name}] 계수 {fmt(reg[coef_name][0], 2, True)}bp t {fmt(reg[coef_name][1], 2, True)}{hpart}, r_us {fmt(reg['r_us'][0], 3, True)} t {fmt(reg['r_us'][1], 2, True)}, n {nobs}"
        cells.append(dict(id=cid, name=name, market=market, window=f"{w[0]}~{w[1]}", n=s["n"], gross_bp=s["gross_bp"], net_bp=s["net_bp"], t_nw=s["t_nw"],
                          delta_bp=delta, t_delta=t_delta, win_pct=s["win_pct"], cagr_pct=s["cagr_pct"], mdd_pct=s["mdd_pct"], yearly_pos=s["yearly_pos"],
                          top5_share_pct=s["top5_share_pct"], half1_net_bp=s["half1_net_bp"], half2_net_bp=s["half2_net_bp"], verdict=v,
                          notes=f"비용 {cost * 1e4:.0f}bp; 대조군 n {n_c}; {rnote}; {note}".strip("; ")))
        regs.append(dict(id=cid, window=wname, nobs=nobs, r2=r2, **{f"{c}_coef": v_[0] for c, v_ in reg.items()}, **{f"{c}_t": v_[1] for c, v_ in reg.items()}))
        return s, delta, t_delta, reg

    kA = k[k.A]
    sM, dM, tdM, regM = add("M_full", "주 셀 069500 발표 밤 롱(FOMC+CPI+고용)", "KOSPI", "full", kA, k, "A")
    main_extra["delta_t"] = tdM
    main_extra["coefA_t"] = regM["A"][1]
    main_extra["coefA"] = regM["A"][0]
    add("M_wall", "주 셀 · W_all", "KOSPI", "W_all", kA, k, "A")
    add("M_ex2026", "주 셀 · W_ex2026", "KOSPI", "W_ex2026", kA, k, "A")
    add("M_long", "주 셀 · W_long", "KOSPI", "W_long", kA, k, "A")
    add("M_full_7bp", "주 셀 · 스트레스 7bp", "KOSPI", "full", kA, k, "A", cost=7e-4, note="진단(비용 스트레스)", forced=None)
    cells[-1]["verdict"] = "diagnostic"
    add("V1_H", "V1 미국 휴장 밤 롱(H, 요일 대조)", "KOSPI", "full", k[k.H], k, "H", weekday=True, forced="estimate", note="사전 등록 estimate 고정, Δ 는 요일 더미 포함")
    add("V1_H_wall", "V1 · W_all", "KOSPI", "W_all", k[k.H], k, "H", weekday=True, forced="estimate", note="사전 등록 estimate 고정")
    add("V2_fomc", "V2 FOMC 성명 밤만", "KOSPI", "full", k[k.fomc], k, "fomc", reg_flags=["fomc", "cpiemp", "H"])
    add("V2_fomc_wall", "V2 · W_all", "KOSPI", "W_all", k[k.fomc], k, "fomc", reg_flags=["fomc", "cpiemp", "H"])
    add("V3_cpiemp", "V3 CPI·고용 발표 밤만", "KOSPI", "full", k[k.cpiemp], k, "cpiemp", reg_flags=["fomc", "cpiemp", "H"])
    add("V3_cpiemp_wall", "V3 · W_all", "KOSPI", "W_all", k[k.cpiemp], k, "cpiemp", reg_flags=["fomc", "cpiemp", "H"])
    add("V3a_cpi", "V3a CPI 만(보고용)", "KOSPI", "full", k[k.cpi], k, "cpi", reg_flags=["fomc", "cpi", "emp", "H"])
    add("V3b_emp", "V3b 고용만(보고용)", "KOSPI", "full", k[k.emp], k, "emp", reg_flags=["fomc", "cpi", "emp", "H"])
    s16, _, _, _ = add("V4_2016plus", "V4 주 셀 2016 이후", "KOSPI", "since2016", kA, k, "A")
    main_extra["v4_net"] = s16["net_bp"]
    add("V4_pre2016", "V4 주 셀 2015 까지", "KOSPI", "pre2016", kA, k, "A")
    qA = q[q.A]
    add("V5_229200", "V5 229200 판 발표 밤 롱", "KOSDAQ", "V5", qA, q, "A")
    add("V5_229200_wall", "V5 · W_all", "KOSDAQ", "W_all", qA, q, "A")
    # FOMC 는 정례 연 8회라 '사건 ≥ 10 인 해' 가 2007·2008 뿐 → 건수 제한 없는 연도 부호를 보고용으로 덧붙임
    def yearly_all(tr, w):
        t = tr[(tr.index >= w[0]) & (tr.index <= w[1])]
        g = t.groupby(t.index.year).apply(lambda x: float((x.ret - COST).mean()) > 0)
        return f"{int(g.sum())}/{len(g)}"
    for cell in cells:
        if cell["id"].startswith("V2_fomc"):
            ya = yearly_all(k[k.fomc], WIN["full"] if cell["id"] == "V2_fomc" else WIN["W_all"])
            cell["notes"] += f"; 연도 부호는 사건 ≥ 10 인 해가 2007·2008 뿐(정례 연 8회) → 건수 제한 없이 {ya}"
            main_extra["fomc_yr_all" if cell["id"] == "V2_fomc" else "fomc_yr_all_wall"] = ya
    # 주 셀 추가 조건
    m = cells[0]
    extra_ok = tdM >= 2 and regM["A"][1] >= 2 and s16["net_bp"] > 0
    if m["verdict"] == "candidate" and not extra_ok:
        m["verdict"] = "reference"
    m["notes"] += f"; 추가 조건(Δ t ≥ 2 {tdM >= 2}, 1[A] t ≥ 2 {regM['A'][1] >= 2}, 2016 이후 양수 {s16['net_bp'] > 0}) → {'통과' if extra_ok else '미달'}"

    for cid, tr in [("M_full", kA), ("V1_H", k[k.H]), ("V5_229200", qA)]:
        t = tr.copy()
        t["cell"] = cid
        trades.append(t)
    trades = pd.concat(trades).reset_index().rename(columns={"exit": "exit"})
    trades = trades[["cell", "entry", "exit", "wd", "fomc", "cpi", "emp", "A", "H", "gap_days", "ret", "simple", "r_us"]]
    trades["net"] = trades.ret - COST
    trades.to_csv(OUT / "trades.csv", index=False, float_format="%.8f", encoding="utf-8-sig")

    # ---------------- V6 오버레이 · 장부 결합
    legs_wall = panel_wall()
    legs_long = panel_long(["KOSPI_B_beomgu_C3VT30", "KOSDAQ_A_blog_ON_Q", "KOSDAQ_B_beomgu_C3Q"], WIN["W_long"])
    base_w, base_l = comb(legs_wall), comb(legs_long)
    mb, ml = metrics(base_w), metrics(base_l)
    assert abs(mb["cagr"] - 0.314179) < 0.002 and abs(mb["mdd"] + 0.061046) < 0.001, mb
    assert ml["n_days"] == 2444 and abs(ml["cagr"] - 0.150369) < 0.002 and abs(ml["mdd"] + 0.096107) < 0.001, ml
    base = dict(cagr_all=mb["cagr"], mdd_all=mb["mdd"], y2024=yr(base_w, 2024), y2025=yr(base_w, 2025), cagr_long=ml["cagr"], mdd_long=ml["mdd"], sharpe_all=mb["sharpe"])
    print(f"[재현] W_all comb CAGR {base['cagr_all'] * 100:.2f}% MDD {base['mdd_all'] * 100:.2f}% 2024 {base['y2024'] * 100:.2f}% 2025 {base['y2025'] * 100:.2f}% | W_long CAGR {base['cagr_long'] * 100:.2f}% MDD {base['mdd_long'] * 100:.2f}%")
    lev = pd.read_csv(RAW / "etf/122630.csv", parse_dates=["date"]).set_index("date").sort_index()
    k_px = pd.read_csv(RAW / "etf/069500.csv", parse_dates=["date"]).set_index("date").sort_index()
    resid = (lev["open"] / lev["close"].shift(1) - 1) - (k_px["open"] / k_px["close"].shift(1) - 1).reindex(lev.index)
    Aset = sets["fomc"] | sets["cpi"] | sets["emp"]
    ov_rows, curves_all, curves_long = [], [("원 장부", base_w)], [("원 장부", base_l)]
    ov_results = {}
    for mode, label in [("H", "V6a H→0 만"), ("A", "V6b A→1.5(122630 잔차 차감) 만"), ("full", "V6c 전체(H→0 ∧ A→1.5, 잔차 차감)"), ("full_simple", "V6d 전체 simple(1.5×net, 잔차 무시)")]:
        lw, inc_w, n_w = apply_overlay(legs_wall, mode, cal, sets["H"], Aset, resid)
        ll, inc_l, n_l = apply_overlay(legs_long, mode, cal, sets["H"], Aset, resid)
        cw, cl = comb(lw), comb(ll)
        mw, mlg = metrics(cw), metrics(cl)
        res = dict(cagr_all=mw["cagr"], mdd_all=mw["mdd"], y2024=yr(cw, 2024), y2025=yr(cw, 2025), cagr_long=mlg["cagr"], mdd_long=mlg["mdd"], sharpe_all=mw["sharpe"], n_w=n_w, n_l=n_l)
        strict = res["cagr_all"] > base["cagr_all"] and res["mdd_all"] > base["mdd_all"] and res["cagr_long"] > base["cagr_long"] and res["mdd_long"] > base["mdd_long"] and res["y2024"] > base["y2024"] and res["y2025"] > base["y2025"]
        relaxed = res["cagr_all"] >= base["cagr_all"] - 1e-12 and res["mdd_all"] >= base["mdd_all"] - 1e-12 and res["cagr_long"] >= base["cagr_long"] - 1e-12 and res["mdd_long"] >= base["mdd_long"] - 1e-12 and res["y2024"] >= base["y2024"] - 1e-12 and res["y2025"] >= base["y2025"] - 1e-12
        res["strict"], res["relaxed"] = bool(strict), bool(relaxed)
        ov_results[mode] = res
        if mode == "H":
            v = "candidate" if relaxed else "reject"
            rule = "완화 규칙(비악화)"
        elif mode == "full_simple":
            v, rule = "diagnostic", "진단"
        else:
            v = "candidate" if strict else "reject"
            rule = "결합 규칙(동시 개선)"
        aff_w = inc_w[inc_w != 0]
        aff_l = inc_l[inc_l != 0]
        for wname, incs, n_aff, mm, cagr_, mdd_ in [("W_all", aff_w, n_w, mw, res["cagr_all"], res["mdd_all"]), ("W_long", aff_l, n_l, mlg, res["cagr_long"], res["mdd_long"])]:
            tnw = float(hac(incs * 1e4).tvalues[0]) if len(incs) >= 3 else np.nan
            if n_aff == 0:
                v, rule = "diagnostic", "변경 없음(이 창에 jehyeong_ON 없음)"
            note = (f"{rule}; 합산 장부 W_all CAGR {fmt(base['cagr_all'] * 100, 2)}→{fmt(res['cagr_all'] * 100, 2)}% MDD {fmt(base['mdd_all'] * 100, 2)}→{fmt(res['mdd_all'] * 100, 2)}%, "
                    f"2024 {fmt(base['y2024'] * 100, 2)}→{fmt(res['y2024'] * 100, 2)}%, 2025 {fmt(base['y2025'] * 100, 2)}→{fmt(res['y2025'] * 100, 2)}%, "
                    f"W_long CAGR {fmt(base['cagr_long'] * 100, 2)}→{fmt(res['cagr_long'] * 100, 2)}% MDD {fmt(base['mdd_long'] * 100, 2)}→{fmt(res['mdd_long'] * 100, 2)}%; "
                    f"동시 개선 {strict}, 비악화 {relaxed}; n = 영향 밤(활성 ∧ w≠1), net_bp = 영향 밤당 합산 장부 증분")
            ov_rows.append(dict(id=f"V6{ {'H': 'a', 'A': 'b', 'full': 'c', 'full_simple': 'd'}[mode] }_{wname}", name=f"{label} · {wname}", market="BOOK",
                                window=f"{WIN[wname][0]}~{WIN[wname][1]}", n=n_aff, gross_bp=np.nan, net_bp=float(incs.mean() * 1e4) if len(incs) else 0.0, t_nw=tnw,
                                delta_bp=np.nan, t_delta=np.nan, win_pct=float((incs > 0).mean() * 100) if len(incs) else np.nan, cagr_pct=cagr_ * 100, mdd_pct=mdd_ * 100,
                                yearly_pos="", top5_share_pct=np.nan, half1_net_bp=np.nan, half2_net_bp=np.nan, verdict=v, notes=note))
        if mode in ("H", "full"):
            curves_all.append((label.split("(")[0].strip(), cw))
        if mode == "full":  # W_long 에서는 V6a 와 V6c 가 같은 곡선(H→0 만 작동)
            curves_long.append((label.split("(")[0].strip() + "(= V6a)", cl))
    cells += ov_rows
    cells_df = pd.DataFrame(cells)[["id", "name", "market", "window", "n", "gross_bp", "net_bp", "t_nw", "delta_bp", "t_delta", "win_pct", "cagr_pct", "mdd_pct",
                                    "yearly_pos", "top5_share_pct", "half1_net_bp", "half2_net_bp", "verdict", "notes"]]
    cells_df.to_csv(OUT / "cells.csv", index=False, float_format="%.4f", encoding="utf-8-sig")
    pd.DataFrame(regs).to_csv(OUT / "reg_summary.csv", index=False, float_format="%.4f", encoding="utf-8-sig")

    # ---------------- 시리즈·오버레이 파일
    w = WIN["full"]
    cal_w = cal[(cal >= w[0]) & (cal <= w[1])]
    ser = pd.DataFrame({"date": cal_w, "pos": 0, "gross": 0.0, "net": 0.0}).set_index("date")
    kAw = kA[(kA.index >= w[0]) & (kA.index <= w[1])]
    ser.loc[kAw.index, "pos"] = 1
    ser.loc[kAw.index, "gross"] = kAw.simple
    ser.loc[kAw.index, "net"] = kAw.simple - COST
    ser.reset_index().assign(date=lambda x: x.date.dt.strftime("%Y-%m-%d")).to_csv(SER / f"{HANDLE}.csv", index=False, float_format="%.8f")
    back = pd.read_csv(SER / f"{HANDLE}.csv", parse_dates=["date"]).set_index("date")
    mrep = metrics(back["net"])
    meta = {"handle": HANDLE, "slot": "A_overnight", "market": "KOSPI", "instrument": "KODEX 200 (069500) 종가 매수 → 익일 시가 매도(발표 밤만)",
            "timing": "overnight_close_to_next_open (date=청산일, 진입 = 직전 거래일 15:30)", "cost_bp": 5, "direction": "long_only",
            "first_valid": w[0], "last_valid": str(cal_w[-1].date()), "n_days": int(len(ser)), "n_active": int(ser.pos.sum()),
            "source_files": ["hyfe/data/raw/etf/069500.csv", "round15_alpha_share/04_us_calendar/calendar_fomc.csv", "calendar_bls_cpi.csv", "calendar_bls_emp.csv"],
            "repro": {"expected": {"cagr": sM["cagr_pct"] / 100, "mdd": sM["mdd_pct"] / 100, "trades": sM["n"]}, "got": {"cagr": mrep["cagr"], "mdd": mrep["mdd"], "trades": int((back.pos != 0).sum())}},
            "verdict": m["verdict"],
            "notes": "진입일 ∈ A(FOMC 성명일 ∪ BLS CPI 발표일 ∪ 고용 발표일, 미국 달력일 = 한국 거래일) 이면 069500 종가 매수 → 익일 시가 매도, 왕복 5bp. gross = 단순 수익 O_{t'}/C_t − 1. A∩H(미국 휴장) 밤도 포함."}
    (SER / f"{HANDLE}.meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    assert abs(mrep["cagr"] - sM["cagr_pct"] / 100) < 1e-6 and abs(mrep["mdd"] - sM["mdd_pct"] / 100) < 1e-6, (mrep, sM["cagr_pct"], sM["mdd_pct"])
    ov = pd.DataFrame({"date": cal[1:]})
    entry = cal[:-1]
    inH, inA = entry.isin(sets["H"]), entry.isin(Aset)
    ov["w_jehyeong_ON"] = np.where(inH, 0.0, np.where(inA, 1.5, 1.0))
    ov["w_blog_ON_Q"] = np.where(inH, 0.0, 1.0)
    ov.assign(date=ov.date.dt.strftime("%Y-%m-%d")).to_csv(SER / "overlay_04.csv", index=False)

    # ---------------- 그림
    groups = {}
    for wname in ("full", "W_all"):
        ww = WIN[wname]
        dw = k[(k.index >= ww[0]) & (k.index <= ww[1])]
        rows = []
        for lab, mask in [("FOMC", dw.fomc), ("CPI", dw.cpi), ("고용", dw.emp), ("A 전체", dw.A), ("H 휴장", dw.H), ("대조 ¬A¬H", dw.ctrl)]:
            y = dw.ret[mask] * 1e4
            rows.append((lab, float(y.mean()), float(hac(y).tvalues[0]), int(mask.sum())))
        groups[f"{ww[0][:4]}~{ww[1][:4]}" if wname == "full" else "W_all 2022.12~2026.9"] = rows
    fig_bars(groups, OUT / "fig1_event_means.png")
    fig_book({"W_all 2022.12.13~2026.9.17": curves_all, "W_long 2016.10.4~2026.9.17": curves_long}, OUT / "fig2_overlay_book.png")
    fig_yearly(kA[(kA.index >= w[0]) & (kA.index <= w[1])], OUT / "fig3_yearly.png")

    # ---------------- 출력 요약
    pd.set_option("display.width", 250, "display.max_columns", 30, "display.max_colwidth", 60)
    print("\n[cells]")
    print(cells_df.drop(columns=["notes"]).round(2).to_string(index=False))
    print("\n[주 셀 회귀]", {c: (round(v_[0], 3), round(v_[1], 2)) for c, v_ in regM.items()})
    print("[주 셀 추가 조건]", main_extra, "→", m["verdict"])
    print("[V6]", {k_: {a: (round(b, 4) if isinstance(b, float) else b) for a, b in v_.items()} for k_, v_ in ov_results.items()})
    write_notes(cells_df, base, ov_results, main_extra, k, kf, q, nyse, fomc, cpi, emp, nyse_extra, spx_last, spy_last, cal, in_kr, regM, sM, s16)
    print(f"\n[저장] {OUT} / {SER / (HANDLE + '.csv')} / {SER / 'overlay_04.csv'} ; 셀 수 {len(cells_df)}")


def write_notes(cells, base, ov, mx, k, kf, q, nyse, fomc, cpi, emp, nyse_extra, spx_last, spy_last, cal, in_kr, regM, sM, s16):
    c = cells.set_index("id")

    def row(cid):
        r = c.loc[cid]
        return (f"n={int(r.n)}, 창 {r.window.replace('-', '.')}, 총수익 {fmt(r.gross_bp, 2, True)}bp, 순수익 {fmt(r.net_bp, 2, True)}bp/건(5bp), NW t {fmt(r.t_nw, 2, True)}, "
                f"Δ {fmt(r.delta_bp, 2, True)}bp(t {fmt(r.t_delta, 2, True)}), 승률 {fmt(r.win_pct, 1)}%, CAGR {fmt(r.cagr_pct, 2)}%, MDD {fmt(r.mdd_pct, 2)}%, "
                f"연도 부호 {r.yearly_pos}, 상위 5건 {fmt(r.top5_share_pct, 1)}%, 전·후반 {fmt(r.half1_net_bp, 2, True)}/{fmt(r.half2_net_bp, 2, True)}bp → {r.verdict}")

    def ovl(mode):
        o = ov[mode]
        return (f"W_all CAGR {fmt(base['cagr_all'] * 100, 2)}→{fmt(o['cagr_all'] * 100, 2)}%, MDD {fmt(base['mdd_all'] * 100, 2)}→{fmt(o['mdd_all'] * 100, 2)}%, "
                f"2024 {fmt(base['y2024'] * 100, 2)}→{fmt(o['y2024'] * 100, 2)}%, 2025 {fmt(base['y2025'] * 100, 2)}→{fmt(o['y2025'] * 100, 2)}%, "
                f"W_long CAGR {fmt(base['cagr_long'] * 100, 2)}→{fmt(o['cagr_long'] * 100, 2)}%, MDD {fmt(base['mdd_long'] * 100, 2)}→{fmt(o['mdd_long'] * 100, 2)}% "
                f"(영향 밤 W_all {o['n_w']}, W_long {o['n_l']}; 동시 개선 {'예' if o['strict'] else '아니오'}, 비악화 {'예' if o['relaxed'] else '아니오'})")

    m = c.loc["M_full"]
    n_cells = len(cells)
    wd_name = ["월", "화", "수", "목", "금", "토", "일"]
    wd_emp = emp.date.dt.weekday.value_counts().sort_index()
    wd_cpi = cpi.date.dt.weekday.value_counts().sort_index()
    wd_fomc = fomc.date.dt.weekday.value_counts().sort_index()
    txt = f"""# 04_us_calendar 밤 다리 미국 달력 오버레이(발표 밤 가중·미국 휴장 밤 제외) 결과

## 규약
• 사전 등록(알파 공유 사전 등록 문서 2026.10.30, design_shortlist.md 04 절) 그대로. 결과를 본 뒤 문턱·창·비용·차량 변경 없음.
• H = NYSE 휴장일 중 한국 거래일. A = FOMC 성명일 ∪ BLS CPI 발표일 ∪ 고용(Employment Situation) 발표일 중 한국 거래일(발표일 = 미국 달력일, 한국 휴장이면 제외, 이동 없음).
• 밤 수익 ret = ln(익일 069500 시가 / 당일 종가), 롱 순수익 = ret − 5bp(왕복, 자본 1 단위). 창 적용은 청산일 기준. 단일가 체결 가정(13차 밤 다리 규약).
• Δ = 발표 밤 평균 − 대조군(¬A ∧ ¬H) 평균(더미 회귀 HAC lag 5). 회귀 ret ~ 1[A] + 1[H] + r_us + 요일 더미(월 기준), HAC lag 5. r_us = 밤 창 안 미국 정규장 누적 로그 수익(spx, {dstr(spx_last)} 이후 SPY 조정가로 연장).
• 통계(평균·t·Δ·회귀)는 로그 수익, series·장부 결합은 단순 수익(Π(1+r)). CAGR·MDD 는 비활성일 0 의 069500 거래일 달력.
• 판정(고정): 주 셀 공통 규칙 ∧ Δ NW t ≥ 2 ∧ 1[A] 계수 HAC t ≥ 2 ∧ V4(2016 이후) 양수 → candidate. V6 오버레이는 결합 규칙(W_all·W_long CAGR·MDD 동시 개선 ∧ 2024·2025 개선), H→0 몫만은 비악화. V1 은 estimate 고정.
• 돌린 셀 수 {n_cells}(독립 셀·창 변형 {int((cells.market != 'BOOK').sum())}, 오버레이 결합 {int((cells.market == 'BOOK').sum())}). 사전 등록 7(주 1 + 변형 6) 외의 행은 창 병기·보고용 분해·진단이며 전부 cells.csv 에 남김.

## 핵심 요약
• 주 셀 M_full: {row('M_full')}.
• 추가 조건: Δ NW t {fmt(mx['delta_t'], 2, True)}(≥ 2 {'충족' if mx['delta_t'] >= 2 else '미달'}), spx·요일 통제 뒤 1[A] 계수 {fmt(mx['coefA'], 2, True)}bp t {fmt(mx['coefA_t'], 2, True)}(≥ 2 {'충족' if mx['coefA_t'] >= 2 else '미달'}), 2016 이후 순수익 {fmt(mx['v4_net'], 2, True)}bp/건({'양수' if mx['v4_net'] > 0 else '양수 아님'}).
• W_all 병기 M_wall: {row('M_wall')}.
• V6 오버레이 전체(V6c, H→0 ∧ 코스피 A→1.5 122630 잔차 차감): {ovl('full')}.
• V6 H→0 만(V6a, 완화 규칙): {ovl('H')}.
• 변형 V2 FOMC 성명 밤만: {row('V2_fomc')}. 공통 규칙은 통과로 찍히나 연도 부호 검정은 사건 ≥ 10 인 해가 2007·2008 뿐(정례 연 8회)이라 사실상 미검정(건수 제한 없이 {mx.get('fomc_yr_all', 'n/a')}), W_all 에서는 n=28, t +1.65 로 reference. 변형이므로 주 셀 판정을 바꾸지 않으며 다음 라운드 사전 등록 후보(FOMC 단독 밤 다리)로만 기록. trades.csv 의 fomc 열로 재현 가능.
• 변형 V3 CPI·고용만 {fmt(c.loc['V3_cpiemp', 'net_bp'], 2, True)}bp/건(n={int(c.loc['V3_cpiemp', 'n'])}) reject(CPI 만 {fmt(c.loc['V3a_cpi', 'net_bp'], 2, True)}bp reference, 고용만 {fmt(c.loc['V3b_emp', 'net_bp'], 2, True)}bp reject). V4 2016 이후 {fmt(c.loc['V4_2016plus', 'net_bp'], 2, True)}bp/건(n={int(c.loc['V4_2016plus', 'n'])}) reject vs 2015 까지 {fmt(c.loc['V4_pre2016', 'net_bp'], 2, True)}bp(n={int(c.loc['V4_pre2016', 'n'])}, t {fmt(c.loc['V4_pre2016', 't_nw'], 2, True)}) reference → 효과가 2015 이전에 집중(Kurov 외 2021 의 2015 이후 약화와 정합). V5 229200 판 {fmt(c.loc['V5_229200', 'net_bp'], 2, True)}bp/건(n={int(c.loc['V5_229200', 'n'])}) reject. V1 미국 휴장 밤 총수익 {fmt(c.loc['V1_H', 'gross_bp'], 2, True)}bp, 요일 대조 Δ {fmt(c.loc['V1_H', 'delta_bp'], 2, True)}bp(t {fmt(c.loc['V1_H', 't_delta'], 2, True)}, n={int(c.loc['V1_H', 'n'])}) estimate.
• 독립 밤 다리 series/{HANDLE}.csv(발표 밤만 pos 1, date = 청산일, 2005.1.3~2026.9.29, 활성 {int(sM['n'])}) 와 series/overlay_04.csv(w_jehyeong_ON ∈ {{0, 1, 1.5}}, w_blog_ON_Q ∈ {{0, 1}}, 전 거래일 행) 저장. 판정 등급은 위 주 셀·결합 결과를 따름(reference, 오버레이 미통과).

## 독립 셀 전부(cells.csv)
""" + "\n".join(f"• {cid}: {c.loc[cid, 'name']} · {row(cid)}" for cid in c.index if c.loc[cid, "market"] != "BOOK") + f"""

## V6 오버레이 결합(13차 합산 장부, 코스피 50% + 코스닥 50% 매일 재조정)
• 적용: jehyeong_ON·blog_ON_Q 의 date = 청산일, 진입일 = 직전 거래일. 진입일 ∈ H → 그 밤 수익·비용 0. 진입일 ∈ A(∉ H) ∧ 코스피 → 자본 50% 를 122630 으로 바꾼 1.5 배 = net + 0.5 × (r_122630 − r_069500)(밤 잔차 실측 차감). 코스닥은 2x 한도라 A 가중 1.0. W_long 에는 jehyeong_ON 이 없어 blog_ON_Q 의 H→0 만 작동.
• 원 장부 재현: W_all CAGR {fmt(base['cagr_all'] * 100, 2)}% MDD {fmt(base['mdd_all'] * 100, 2)}% 샤프 {fmt(base['sharpe_all'], 2)}, 2024 {fmt(base['y2024'] * 100, 2)}%, 2025 {fmt(base['y2025'] * 100, 2)}%, W_long CAGR {fmt(base['cagr_long'] * 100, 2)}% MDD {fmt(base['mdd_long'] * 100, 2)}%(13차 원수치 31.4/−6.1, 11.2, 11.4, 15.0/−9.6 과 일치).
• V6a H→0 만: {ovl('H')}.
• V6b A→1.5 만(잔차 차감): {ovl('A')}.
• V6c 전체: {ovl('full')}.
• V6d 전체 simple(1.5 × net, 잔차 무시, 진단): {ovl('full_simple')}.
• 영향 밤당 합산 장부 증분(bp)과 NW t 는 cells.csv 의 V6 행(net_bp, t_nw) 참조.

## 자료 확보 단계 결과(WebFetch, 2026.10.30)
• NYSE: 공식 휴장표 페이지(URL 은 calendar_nyse_holidays.csv 의 source_url)는 2025~2027 만 게시 → 공식 {int((nyse.kind == 'official').sum())}일. 2002~2024 는 사전 등록 대체 규칙(spx_daily·yf_SPY 결측 평일) {int((nyse.kind == 'fallback').sum())}일로 채움. 대체 목록은 특별 휴장 2004.6.11(레이건), 2007.1.2(포드), 2012.10.29·30(샌디), 2018.12.5(부시), 2025.1.9(카터)를 모두 포함하고, 공식 게시 연도(2025~{dstr(spy_last)[:4]})에서 공식 휴장일은 전부 미국 시세 결측과 일치(검증 assert). 대체에만 있고 공식에 없는 날: {', '.join(dstr(d) for d in nyse_extra.date) if len(nyse_extra) else '없음'}. 2001.9.11~14 는 자료 창(2002.10.14~) 밖.
• FOMC: 2005~2020 fomchistorical{{YYYY}}.htm 16쪽, 2021~2026 fomccalendars.htm. 정례 회의 마지막 날 {int((fomc.kind == 'scheduled').sum())}(2020.3.17·18 취소 제외) + 정책 성명이 나온 비정례 회의 성명일 {int((fomc.kind == 'unscheduled').sum())}(2007.8.10, 2007.8.17, 2008.1.22, 2008.3.11, 2008.10.8, 2010.5.9 일, 2019.10.11, 2020.3.3, 2020.3.15 일, 2020.3.23; 익일 성명 4건은 보도자료 페이지로 날짜 확인). 성명 없는 전화회의·통지투표 보도자료·장기 목표 성명(2020.8.27, 2025.8.22)은 제외. 요일 분포(월~일) {', '.join(f'{wd_name[i]} {int(v)}' for i, v in wd_fomc.items())}.
• BLS: 보도자료 아카이브 파일명(cpi_MMDDYYYY, empsit_MMDDYYYY)이 실제 발표일 → CPI {len(cpi)}건({dstr(cpi.date.min())}~{dstr(cpi.date.max())}), 고용 {len(emp)}건({dstr(emp.date.min())}~{dstr(emp.date.max())}). 2005년 1월 발표분(CPI 2005.1.19, 고용 2005.1.7)은 2005 일정표에서 보충. 셧다운 지연(2013.10.22 고용·10.30 CPI, 2025.10.24 CPI, 2025.11.20·12.16 고용, 2026.2.11 고용)과 2025년 10월 CPI 미발표가 반영됨. 고용 요일 분포 {', '.join(f'{wd_name[i]} {int(v)}' for i, v in wd_emp.items())}; CPI {', '.join(f'{wd_name[i]} {int(v)}' for i, v in wd_cpi.items())}.
• 발표 시각: CPI·고용 08:30 ET(KST 21:30/22:30), FOMC 정례 14:00 ET(KST 익일 03:00/04:00), 비정례는 07:00~10:00 ET → 모두 발표일의 한국 밤 창 안. 네 표는 calendar_*.csv(열 date, source_url 외)로 저장.

## 자료 점검
• 069500.csv {dstr(cal[0])}~{dstr(cal[-1])} {len(cal)}행, 결측·중복·0 없음, 밤 패널 {len(k)}건(주 창 {len(kf)}건). spx_daily.csv 마지막 유효 종가 {dstr(spx_last)}(09.22 행 close 결측) → yf_SPY 조정가 로그수익으로 {dstr(spy_last)} 까지 이어 붙임(SPY 결측 평일 = spx 결측 평일 일치 확인). 통상 밤의 r_us 가 spx 당일 로그수익과 같음을 assert.
• 한국 거래일 안 달력: FOMC {in_kr['fomc']}, CPI {in_kr['cpi']}, 고용 {in_kr['emp']}, NYSE 휴장 {in_kr['H']}. 주 창 밤: A {int(kf.A.sum())}(FOMC {int(kf.fomc.sum())}, CPI {int(kf.cpi.sum())}, 고용 {int(kf.emp.sum())}), H {int(kf.H.sum())}, A∩H {int((kf.A & kf.H).sum())}(성금요일 고용 발표 등; 주 셀·독립 다리는 A 에 포함, 오버레이는 H 우선 0), 대조군 {int(kf.ctrl.sum())}.
• 229200.csv 2015.10.1~ {len(q)}밤, 거래일이 069500 달력 안에 있음을 assert.
• 13차 series 대조: blog_ON_Q gross 는 233740·251340 종가→익일 시가 단순 수익과 정확히 일치(820건). jehyeong_ON gross 는 069500.csv 단순 수익과 중앙값 차 0.0bp, 상관 0.998 이나 4일(2022.4.28 −142.5bp, 2022.7.28, 2022.10.28, 2026.7.30)에 분배금 조정 차이가 있음(13차 원자료는 비조정가로 보임, 이번 설계는 series net 을 그대로 쓰고 1.5 배 몫만 조정가 CSV 의 122630·069500 차이로 계산). 122630 밤 잔차(122630 − 069500 종가→시가 단순 수익) 는 leverage_desk t4 와 같은 정의.
• 2026.10.30 행 없음(모든 자료가 2026.9.29 이전 또는 SPY 10.5 까지만 사용).

## 단서
• 주 셀 유의성과 Δ 는 비용 0 의 요일·spx 통제 회귀로 분리해 보고. 발표 밤 효과가 미국 당일 수익 경로를 지나는 몫은 r_us 계수가 흡수하므로, 1[A] 계수는 '발표일이라는 사전 정보' 만의 몫.
• 미국 휴장 밤은 월요일 편중(요일 교란) → V1 은 요일 더미 포함 Δ 로만 보고, estimate 고정.
• 2020.3 비정례 성명 등 위기 국면 사건이 평균을 끌어올릴 수 있어 상위 5건 비중·전후반·V4(2016 이후) 분리를 같이 봐야 함.
• 발표 일정은 사전 공표라 사전 신호이나, 셧다운 지연·미발표는 사후 확정 일정(실제 발표일)을 사용 → 포워드에서는 공표 일정 변경을 매일 반영해야 함.
• 오버레이 1.5 배는 122630 경유(자본 50%)·단일가 체결 가정, 잔차는 과거 실측 평균이 아니라 밤별 실측값 차감. 코스닥 레버리지(233740 2x)는 한도라 가중 없음.
• 상위 5건 비중이 100% 를 넘거나 n/a 인 셀은 총 순손익이 0 근처이거나 음수인 경우(정의상 비율이 커짐). 연도 부호 0/0 은 사건 ≥ 10 인 해가 없는 셀.
• BLS 일정 아카이브 URL(schedule/archives/cpi_nr.htm, empsit_nr.htm)은 404 → 보도자료 아카이브 파일명으로 대체(실제 발표일이라 지연·미발표가 자동 반영).
• 다중 비교: 이 설계 {n_cells}셀, 15차 사전 등록 64셀, 14차 누적 512셀.
"""
    assert "-" not in txt, [l for l in txt.splitlines() if "-" in l][:3]
    (OUT / "notes.md").write_text(txt, encoding="utf-8")


if __name__ == "__main__":
    main()
