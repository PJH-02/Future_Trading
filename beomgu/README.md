# 백테스트 개요 — 옵션 레그

**검증한 것.** 전일 KOSPI200 옵션시장 외국인 콜/풋 매수세가 다음날 KOSPI200 현물 방향(시가→종가)을 예측하는가.

## 데이터
| 항목 | 출처 | 기간 | 내용 |
|---|---|---|---|
| 옵션 투자자별 일별 | KRX 정보데이터시스템 [15007] | 2010-01 ~ 2026-09 | 월물·정규장. 외국인(기타외국인 포함)·개인·기관·기타법인 × 콜/풋 × 매수/매도(금액·계약) |
| 선물 투자자별 일별 | 네이버증권 | 2010-09 ~ 2026-09 | 외국인 매수·매도 계약(통제변수) |
| K200 지수 일봉 | FinanceDataReader + 네이버 | 1990 ~ 2026-09 | 목표 수익률 |
| KODEX 200 일봉 | yfinance + 네이버 | 2007 ~ 2026-09 | 매매 백테스트 |

## 지표
- CallFlow = (외인 콜 매수대금 − 매도대금) / (매수 + 매도), PutFlow 같은 식
- OptFlow = CallFlow − PutFlow. 콜 쪽이 세면 +
- T−1 값으로 T일을 봄. 1일 값과 3일 누적

## 분석 (`opt_flow_study/run_study.py`)
1. KRX 거래일 달력으로 패널 구성
2. 지표 계산
3. 5분위: 다음날 평균 시가→종가
4. 회귀: R(T) = α + β·OptFlow(T−1), HAC(lag 5)
5. 연도별 β·p·Q5−Q1
6. 통제: 전일 수익률(T−1 시가→종가), 외인 선물 Flow(T−1)
7. 콜/풋 분해
8. 순환 시프트 플라시보

판정 기준 6개는 옵션 데이터를 보기 전에 정했음(`opt_flow_study/README.md`).

## 매매 백테스트 (`backtest_periods.py`, `explore_putflow_thresholds.py`)
- KODEX 200 T일 시가 매수 → 종가 청산. 비용 왕복 0.03% + 1틱
- 규칙: 매일 매수(기준선), OptFlow > 0, OptFlow z20 ≥ 0.84, 외인 풋 순매도(탐색)
- 기간: 3개월·1년·3년·최대, 종료 2026-09-23

## 결과
**방향성 (K200 지수 시가→종가, 비용 전)**

| 기간 | 거래일 | Q5−Q1 | β | HAC p | R² |
|---|---|---|---|---|---|
| 3개월 | 64 | +0.217%p | +2.65 | 0.563 | 0.38% |
| 1년 | 243 | +0.332%p | +2.75 | 0.396 | 0.40% |
| 3년 | 728 | +0.161%p | +2.22 | 0.385 | 0.27% |
| 최대 | 3,938 | +0.056%p | +1.72 | 0.258 | 0.14% |

**샤프 (연환산, 비용 후)**

| 규칙 | 3개월 | 1년 | 3년 | 최대 |
|---|---|---|---|---|
| 매일 매수 (기준선) | −2.80 | +0.12 | −0.19 | −1.02 |
| OptFlow > 0 | −0.44 | −0.13 | −0.13 | −0.46 |
| OptFlow z ≥ 0.84 | −0.62 | +0.26 | +0.26 | −0.16 |
| 외인 풋 순매도 (탐색) | +0.10 | +0.58 | +0.28 | −0.40 |

- 판정: 기준 1·3·4 미충족 → 1차 신호로 미채택
- 콜 매수세에는 정보가 거의 없고, 외인 풋 순매수 다음날 약세(p 0.03, 탐색)
- 같은 회귀에서 외인 선물 Flow(통제변수)는 p < 0.001(비용 전)
- 외인 풋 순매도가 아주 강한 날(PutFlow < −0.05)은 모든 기간에서 손실
- 1년 샤프의 표준오차는 약 1

## 파일
| 경로 | 내용 |
|---|---|
| `opt_flow_study/` | 코드. 실행 `python run_study.py --steps 1-8`, `python backtest_periods.py`, `python explore_putflow_thresholds.py`, 검사 `python selftest.py` |
| `collectors/` | 원자료 수집기 |
| `results/2026-09-27/` | `report_optflow_1d.md`(판정), `periods_report.md`(기간별), `putflow_thresholds.md`(풋 강도별) |
| `data/krx_k200_option_foreign_callput_net_daily_2010-2026.csv` | KRX 코스피200 옵션(월물·정규장) 외국인 콜/풋 순매수 일별, 2010-01-04 ~ 2026-09-23. 대금 원, 거래량 계약 |
| `data/krx_opt_investor_daily.csv` | 백테스트 입력: KRX 옵션 투자자별 일별(외국인·개인·기관·기타법인·전체 × 콜/풋 × 매수·매도, 금액·계약) |
| `data/naver_fut_investor_daily.csv` | 백테스트 입력: 코스피200 선물 투자자별 일별(네이버) |
| `data/k200_index_daily.csv`, `data/k200_naver_fchart_daily.csv` | 백테스트 입력: K200 지수 일봉(FinanceDataReader, 네이버) |
| `data/kodex200_daily.csv`, `data/kodex200_naver_fchart_daily.csv` | 백테스트 입력: KODEX 200 일봉(yfinance 원가격, 네이버 수정주가) |
| `overnight_lab/` | 종가 베팅·오버나잇 백테스트 코드(1차 t01~t08, 2차 r2_d1~d4, 3차 r3_d2·bt_d2_2010_2026, 3차 탐색 tools/preopen_d2_*). 실행 `python topics/t01_intraday_strength.py` |
| `results/2026-09-29-overnight/` | 종가 베팅·오버나잇 1차 결과(`README.md` 요약, 주제별 `report.md`·`verdict*.json`, 시행 장부 `ledger.csv`, 풋 경보 열 `put_veto_alarm.csv`) |
| `results/2026-09-29-overnight/round2/` | 2차 디벨롭 D1~D4: 셀별 `spec.json`(사전 고정 사양)·`explore.md`(설계 구간 탐색 메모)·`report.md`·`verdict.json`. 코드 `overnight_lab/topics/r2_d1~d4.py` |
| `results/2026-09-29-overnight/round3_D2/` | D2 3차 디벨롭: 사양(`spec.json`)·탐색 메모와 변형 목록·2003~2009 자료 품질·판정 보고(`report.md`). 코드 `overnight_lab/topics/r3_d2.py`, `overnight_lab/tools/preopen_d2_*.py` |
| `results/2026-09-29-overnight/strategy_d2_2010_2026/` | D2 3차 셀 전략 백테스트 2010~2026: `report.md`(구간·연도별 표), `equity.png`, `daily_returns.csv`, `summary.csv`, `yearly.csv`. 코드 `overnight_lab/topics/bt_d2_2010_2026.py` |
| `data/etf/` | 네이버 siseJson 분배 수정주가 일봉(069500·229200·102110·091160·122630·233740·360750·133690·219480·114800·123310), 네이버 fchart 코스피200 지수(KPI200)·선물 연결(FUT) |
| `data/naver_kospi_investor_daily.csv`, `data/naver_kosdaq_investor_daily.csv` | 코스피·코스닥 현물 투자자별 일별(네이버) |
| `data/naver_program_daily.csv`, `data/naver_kosdaq_program_daily.csv` | 코스피·코스닥 프로그램 매매 일별(차익·비차익, 네이버) |
| `data/spx_daily.csv`, `data/ndx_comp_daily.csv`, `data/es_daily.csv`, `data/nq_daily.csv` | S&P500·나스닥 종합 지수, S&P500·나스닥100 선물 연결 일봉(미국 날짜) |

재현: `cd opt_flow_study && python run_study.py --steps 1-8 --raw-dir ../data`
