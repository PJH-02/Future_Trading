# opt_flow_study — 옵션 매매 지표로 코스피200 방향 조사

HYFE 파생퀀트 연합, 인트라데이 트랙(옵션 레그). 백테스트 개요: `../README.md`

**질문.** 전일 코스피200 옵션 시장의 외인 콜/풋 매수세가 당일 코스피200 시가→종가 방향을 예측하는가?


## 상태 (2026-09-27)
- [x] 1단계 데이터 패널·감사, 2단계 지표 계산 — 실데이터로 실행
- [x] 3~8단계 코드 + 자체 검증 `selftest.py` T1~T11 — 합성 데이터로 검증, 독립 코드 검토 12건 반영
- [x] 3~8단계 실데이터 실행 (2026-09-27) — 사전등록 판정 **미통과**. 결과: `../results/2026-09-27/`

## 구조
```
opt_flow_study/
  run_study.py            1~8단계 실행 → outputs/<실행ID>/ 에 마크다운·JSON·provenance
  selftest.py             엔진 검사 T1~T11
  optflow/config.py       경로, 사전등록 파라미터 PREREG(+ 변경 이력)
  optflow/io.py           1단계: KRX 거래일 달력 패널, 감사, 옵션 표준 스키마 로더
  optflow/indicators.py   2단계: Flow·OptFlow·CPR, T−1 정렬과 룩어헤드 가드
  optflow/stats.py        3~8단계: 5분위·HAC·연도별·순환 시프트 플라시보·판정
  optflow/report.py       결과 보고서(마크다운)
  scripts/fetch_kodex_naver.py   KODEX 결측일 보완용 일봉(병합 저장)
  scripts/krx_browser_collect.js KRX [15007] 투자자별 옵션 일별 수집(로그인한 브라우저 탭에서 실행)
  scripts/local_receiver.py      브라우저 다운로드가 막힐 때 쓰는 1회용 로컬 수신기
  backtest_periods.py     기간별(3개월·1년·3년·최대) 방향성·매매 규칙 표
  explore_putflow_thresholds.py  외인 풋 순매도 강도별 탐색
  pilot_daily.py          실시간 페이퍼 파일럿(장 전 신호 기록 → 장 후 정산)
  ho_study.py             옵션 장초 10분 Flow → 09:10~장마감 (장초 체결 데이터 필요)
```

## 옵션 장초 10분 검증에 필요한 데이터 (`ho_study.py`)
장초 체결을 하루 한 줄로 합친 `ho_daily.csv`: `date, c_buy, c_sell, p_buy, p_sell, px_0910, px_close[, px_open]`
- `c_buy/c_sell`: 09:00~09:10 콜 체결대금의 매수주도·매도주도 합(풋도 같은 방식)
- 체결 단위 원자료에 필요한 필드: 거래소 체결시각, 종목(만기·행사가·콜/풋), 체결가, 체결수량, 체결구분(매수/매도 주도) 또는 직전 최우선 매수·매도호가
- KODEX 200: 09:10 이후 첫 체결가 또는 매도1호가, 종가(시가는 선택)
- 수집 시작일과 빠진 날 목록
- 확인: `python ho_study.py --selftest`

## 실행
검증 환경: Python 3.14.4, numpy 2.4.4, pandas 2.3.3, requests 2.33.1, statsmodels 0.15.0 (`requirements.lock`).
```bash
pip install -r requirements.txt            # 분석
pip install -r requirements-dev.txt        # selftest T2 교차검증
python selftest.py                         # 전체 검사(약 1분). --quick 은 참고용
python run_study.py --steps 1-8
```
원자료 폴더 지정 순서: `--raw-dir`, 환경변수 `OPTFLOW_RAW_DIR`(둘 다 현재 작업 폴더 기준 상대경로) → `config.local.json`(이 폴더 기준, 깃 제외) → `./data/raw`.
Windows 콘솔 한글은 스크립트가 stdout 을 UTF-8 로 바꿔 처리한다.

## 데이터
생성 명령은 저장소 루트(`tradingview-mcp/`) 기준이다. 수집기는 `requirements-collect.txt` 가 필요하다.

| 파일 | 내용 | 필수 열 | 생성 |
|---|---|---|---|
| `k200_index_daily.csv` | K200 지수 일봉(달력·목표, 2014-07 이전) | date,open,high,low,close | `python hyfe/collect_fdr.py` (FinanceDataReader `KS200`) |
| `k200_naver_fchart_daily.csv` | K200 지수 일봉 2014-07~ | date,open,high,low,close | 위와 같음 (네이버 fchart `KPI200`) |
| `kodex200_daily.csv` | KODEX 200 원가격 일봉 | date,Open,High,Low,Close,Volume | `python hyfe/collect_public.py` (yfinance `069500.KS`) |
| `kodex200_naver_fchart_daily.csv` | KODEX 200 결측일 보완(수정주가) | date,open,high,low,close,volume | `python <이 폴더>/scripts/fetch_kodex_naver.py` |
| `naver_fut_investor_daily.csv` | 외인 선물 매수·매도(통제변수) | date,buyq_foreign,sellq_foreign,buyv_foreign,sellv_foreign | `cd hyfe && python collect_naver.py fut` (네이버증권 JSON) |
| **`krx_opt_investor_daily.csv`** | **투자자별 옵션(주 지표)** | 아래 표준 스키마 | **KRX 정보데이터시스템, 로그인 필요** |

표준 스키마 `date, investor, cp, buy_qty, sell_qty, buy_amt, sell_amt`
- investor: 한글 라벨 그대로 가능. 외국인·기타외국인 → `foreign` 으로 합산, 개인 → `individual`, 기관계 → `institution`, 전체 → `total`(CPR 에만 사용)
- cp: C/P 또는 call/put · 금액 단위 원 · 날짜는 KRX 거래일 · (date, investor, cp) 중복은 오류

주의: 장중에 수집한 당일 행은 확정치가 아니다. 장 마감 후 다시 받는다(2026-09-23 행을 장중에 받았다가 재수집한 전례).

## 사전등록 기준 (v1.1, 옵션 자료 수령 전 고정)
주 지표 `optflow_1d` = 외인 CallFlow − PutFlow(금액, T−1). 목표는 K200 지수 시가→종가. 판정 창 2010-09-20 ~ 2026-09-23(목표일 기준).
1. 전체 기간 β > 0, HAC(lag 5) 양측 p < 0.05
2. Q5−Q1 > 0
3. β > 0 인 연도 비율 ≥ 75% (n<30 연도는 분모에서 제외하고 보고)
4. 전일 수익률 R(T−1)(= T−1 시가→종가) 통제 후, 그리고 외인 선물 Flow까지 통제 후 모두 β > 0, p < 0.05
5. 2023-01-01 이후 구간 β > 0
6. 순환 시프트 플라시보(5,000회) 단측 p ≤ 0.05

나머지(3일 누적, 계약 기준, 콜·풋 단독, 개인·기관, CPR, 종가→종가 통제, 2023-07-31~ 구간)는 탐색·보고이며 판정에 쓰지 않는다.

## KRX 옵션 자료 받기
1. 본인 계정으로 data.krx.co.kr 에 로그인한 브라우저 탭을 연다
2. 그 탭의 개발자 콘솔에서 `scripts/krx_browser_collect.js` 전체를 실행한다(1년 단위 136회 조회, 약 4분)
3. 내려받은 `krx_opt_investor_daily.csv` 를 원자료 폴더에 둔다
4. 브라우저가 다운로드를 막으면 `scripts/local_receiver.py` 를 띄우고 탭에서 127.0.0.1 로 보낸다

## 깃
- 이 폴더의 `.gitignore`: `data/`, `outputs/`, `config.local.json`, 캐시 제외
- 원자료(csv)는 커밋하지 않는다. 수집기는 `../collectors/`

## 변경 이력
- 0.1.0 (2026-09-27) 최초 작성. 사전등록 v1.
- 0.2.0 (2026-09-27) 독립 코드 검토 반영: total 행 Flow 제외·상수열 방어, 날짜 중복 오류, 투자자 라벨 별칭(기타외국인 합산), NaN 유지, 판정 창 고정, 전일 수익률 정의를 설계 표기와 일치(v1.1), 연도 n<30 제외 규칙, 출력 실행ID 폴더·NON-PREREG 표시·provenance, 룩어헤드 가드를 출처 날짜 대조로 강화, stdout UTF-8, selftest SKIP 구분·이항 판정·T8~T11 추가.
- 0.2.1 (2026-09-27) 실데이터 실행 후: 기타법인 라벨 별칭 추가(탐색만 영향), 보고용 HAC t 순환 시프트 플라시보 추가(β 회전의 이분산 한계 확인, 판정 불변). KRX 수집 `scripts/krx_browser_collect.js`, 1회용 로컬 수신기 `scripts/local_receiver.py`.
