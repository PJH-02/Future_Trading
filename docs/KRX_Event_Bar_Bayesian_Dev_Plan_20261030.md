# KOSPI Event-Bar 베이지안 전략 — 검증·백테스트 계획

> 작성일: 2026-10-30 · 근거 명세: `docs/KRX_Event_Bar_Bayesian_Trading_Research_Spec_v2_Final_20261029.md`
> 상태: **계획서** (코드 미작성). 확정된 범위와 데이터 규약에 따라 작성했으며, 실행 전 "6. 승인 전 확인" 항목만 남긴다.

## 0. 확정된 결정 (사용자 승인)

| 항목 | 결정 |
|---|---|
| 시장 | KOSPI 단일. KOSDAQ은 1분 지수 파일이 없으므로 제외 |
| 상품 | 1x만 사용. +1x = `069500`, −1x = `114800` (TOSS 1분봉). 2x 계열은 G08 이후 과제 |
| 타임스탬프 | 파일별 규약이 다르다. `KOSPI_TOSS_1m.csv`는 `bar_end`: 라벨 `T`는 `[T-1분, T)` 구간의 완료 바. `069500_TOSS_1m.csv`, `114800_TOSS_1m.csv`는 `bar_start`: 라벨 `T`는 `[T, T+1분)` 구간의 바. 두 규약은 데이터로 확인했으며(`jehyeong/probability/check_bar_convention.py`, 1분 수익률 상관 피크가 시프트 0에서만 나타남) 파일을 이동하지 않는다 |
| 실행 시가 | ETF 15:19 **시작** 바의 시가 = ETF 라벨 `15:19` 행의 `open`. 15:20~15:30 종가 단일가는 사용하지 않음 |
| DEV 시작 | 2023-09-21 (KOSPI 1분봉 최초 거래일). 명세의 2022-11 시작은 사용하지 않음 |
| 점프 구간 | ETF 가격의 로그수익률 절대값 > 15%인 1분 구간과 그 거래일은 신호·체결에서 **제외(censor)**. 조정하지 않음 |
| 2026 | 2026-01-01 이후 데이터는 True OOS로 봉인. LOCK 전 생성·조회 금지 |

### 0.1. 데이터 실측 (계획 근거)

| 파일 | 기간 | 거래일 | 확인 사항 |
|---|---|---|---|
| `KOSPI_TOSS_1m.csv` | 2023-09-21 ~ 2026-10-26 | 748 | `bar_end`(실측 확인). 헤더 `source_timestamp_convention=start`는 잘못된 값. 15:31 행 존재 |
| `069500_TOSS_1m.csv` | 2022-11-23 ~ 2026-10-07 | 944 | `bar_start`(실측 확인). 일 390바, 15:19 행 존재. 15:20 이후 행 있음 |
| `114800_TOSS_1m.csv` | 2022-11-23 ~ 2026-10-08 | 945 | `bar_start`(실측 확인). 15:19 행 존재 |

- 두 ETF 파일 모두 KOSPI DEV 구간을 덮는다. 15:19 실행 바 존재 여부는 G00에서 거래일별로 전수 확인한다.
- `trading_value`는 비어 있으므로 원화 거래대금은 `(O+C)/2 × V` 근사로만 만든다. 지수 Won Bar는 만들지 않는다.
- 축약본 `069500_1m.csv` 등은 사용하지 않는다(TOSS 파일과 겹치는 행은 일치하나 중복 소스이므로 혼합 금지).
- **규약 확정:** KOSPI 파일은 `bar_end`, ETF 파일은 `bar_start`로 처리한다. 자동 시프트는 하지 않으며, 파일별 규약을 로더에 명시하고 G00에서 시작 분(09:01 등) 존재 여부로 재확인한다.

## 1. 기간 분할

| 구간 | 기간 | 용도 |
|---|---|---|
| TRAIN | 2023-09-21 ~ 2023-12-31 | 초기 임계값·W1 기준값 교정. 성과 평가 금지 |
| DEV | 2024-01-01 ~ 2025-12-31 | 8개 분기 expanding-origin walk-forward. 모든 선택은 여기서만 |
| LOCK | 2025-12-31 기준 | 설정·코드 해시 동결 |
| TRUE OOS | 2026-01-01 ~ 마지막 완료 거래일 | 동결 모델 1회 평가 |

## 2. 단계 계획

명세 G00~G09 게이트를 그대로 따른다. 실패 시 명세의 폐기 기준을 적용한다.

| 단계 | 내용 | 산출물 | 통과 기준 | 실패 시 |
|---|---|---|---|---|
| **P0 환경** | Python, numpy, pandas, scipy, pytest 준비 | requirements 목록, 빈 테스트 실행 | 기존 스크립트 import 가능, 테스트 0건 실패 | 다음 단계 금지 |
| **G00 데이터·시간** | 중복, 정렬, OHLC 부등식, V≥0, 1분 간격, 세션 결측, 15:19 실행 바, 점프 구간 검출, 규약 판정 | 데이터 검증 리포트 | 규약 판정 완료, 실행 바 존재율 확인 | 해당 구간 BLOCKED |
| **G01 샘플링** | Time Bar 1분·8분 기준선, KOSPI Volume Bar(전일까지 20일 중앙값/50 고정 임계값), ETF Won Bar(`(O+C)/2×V`) | 일 바 수, duration, overshoot, 누출 테스트 | 정확성 통과. 통계·경제 우위 없으면 Time Bar 유지 | Time Bar 유지 |
| **G02 기초 알파** | 공통 충격 `Z_k`(3바, σ_pre 20바, \|Z\|≥1)에서 T/R 후보 생성, 비용 0/2/5/10bp별 순EV | T/R 순EV 표, 날짜 집중도 | 비용 5bp에서 양의 순EV가 DEV 분기에서 일관 | 해당 가설 기각 |
| **G03 청산 3종** | EXIT_60, EXIT_NO_60, EXIT_CLOSE_ONLY를 같은 진입가에서 paired 비교 | 정책별 paired P&L, 보유시간, EOD 편향 | 우위 근거 있을 때만 정책 선택 | EXIT_60 유지 또는 기각 |
| **G04 Activity 분할** | 과거 동일 30분대 중앙값 기준 High/Low | High/Low 차이, 추정 안정성 | 차이가 안정적 | 분할 제거 |
| **G05 Beta** | `Beta(0.5,0.5)`, 최근 100 완료 사건, 최소 30사건·10일·승패 각 5. P0 vs P1, 정적 TRAIN 빈도 vs 온라인, Brier·calibration | P1 vs P0, calibration | P1이 P0 대비 DEV에서 일관 개선 | Beta 제거 |
| **G06 W1/RV Guard** | W1(최근 30 / 이전 120), TRAIN 일별 최대 q90 경보, 이벤트 q75 해제, 3바 정지·2바 해제·probation | P3−P1, P3−P2, 회피손실·놓친 이익 | 위험예산 대비 개선 | W1 제거 |
| **G07 비중 조절** | P4(Beta 0/5/10%), P5, P6(W1 soft). 노출·거래 횟수 동등화 | P4−P1, P5−P4, P6−P3 | 노출 통제 후에도 우위 | 고정 10% 유지 |
| **G08 2x ETF** | 본 계획 범위 밖 | — | — | 1x만 사용 |
| **LOCK** | DEV 종료 시점 설정·코드·데이터 해시 저장 | lock 파일 | 이후 변경 없음 | — |
| **G09 True OOS** | LOCK된 최선 모델 1회 평가 | `TRUE_OOS_PASS/FAIL/INCONCLUSIVE` | 사전 기준 판정 | 재튜닝 금지 |

## 3. 필수 테스트 (pytest)

명세 14.2 기준. 실제 CSV의 짧은 구간 또는 합성 데이터로 실행한다. `skip`이나 assertion 완화는 금지.

1. **Future perturbation**: t 이후 데이터를 바꿔도 t 시점의 임계값·신호·Beta·W1·비중이 불변.
2. **Preopen threshold**: 전일까지 완료된 정상 20거래일만 사용. 당일 총량으로 재계산 금지.
3. **Whole-minute stopping**: 임계 초과 시 해당 1분봉 전체 포함, 초과분 이월 없음.
4. **Shock source**: σ_pre에 최근 3바 미포함, overnight gap 제외, 결측 시 무신호.
5. **Exit exactness**: 3종 정책의 종료 시각, 진입+60분, 15:19 시가, EOD 누락, 다음 시가 누출 없음.
6. **Label consistency**: 성공은 `net>0`, 비용 시나리오별 라벨 분리, 비용 재차감 금지.
7. **Delayed Beta**: exit 체결 전 Y는 카운트에 미반영. 정책·비용 간 posterior 혼합 금지.
8. **Same-event pairing**: T/R은 독립 2거래로 세지 않음. 3개 청산 정책은 같은 진입가.
9. **Guard hysteresis**: q90 일별 최대, q75 해제, 3바 정지, 2바 해제, probation.
10. **Sizing separation**: P0~P3은 10% 고정, P4·P6은 0/5/10%.
11. **Data cutoff**: 점프 구간·2026 행이 DEV 산출에 들어가지 않음.
12. **Seal/reproducibility**: 동일 해시·seed에서 결과 동일. LOCK 없이는 2026 호출 불가.

## 4. 백테스트 설계

- **신호**: 09:30~15:00 완료 바 기준. 신규 진입은 15:00 이전만.
- **체결**: 신호 바 `available_at` 이후 ETF 다음 1분봉 시가. 같은 바 종가 체결 금지.
- **청산**: EXIT_60 = min(3 Event 완료, 진입+60분, 15:19 시가), EXIT_NO_60 = min(3 Event 완료, 15:19), EXIT_CLOSE_ONLY = 15:19 시가. 오버나이트 금지.
- **상품**: 069500(+1x 매수), 114800(−1x 매수). 지수 수익률로 인버스 P&L을 만들지 않음.
- **비용**: 왕복 0/2/5/10bp. 기본 5bp.
- **비중**: 최초 P0~P3은 현금 10% 고정. 일일 손실 −1% 도달 시 신규 진입 중지.
- **지표**: 순손익, 거래당 net EV(bp), 일별 Sharpe, MDD, VaR99/ES99, 거래 수, 거래일 수, 노출 시간, turnover, 승률과 손익비 함께 보고.
- **통계**: 날짜 블록 부트스트랩 1,000회 이상(seed 42). 후보 수에 따른 FDR 또는 Reality Check 보정.
- **기준선**: 069500 buy-and-hold, 시간 블록 보존 무작위 진입, 무거래.

## 5. 산출물

- 코드: `jehyeong/intraday/code/` 아래 신규 모듈 1개 + 테스트 파일. 신규 디렉터리는 만들지 않음. 기존 `collect_intraday_data.py`의 `validate`·`aggregate_30m` 재사용 검토.
- 결과: `jehyeong/intraday/results/regime_research/` (DEV와 LOCK 분리).
- 2026 결과: `results/regime_research/true_oos_2026/`에만 저장. LOCK 이전 생성 금지.

## 6. 승인 전 확인

1. **패키지 추가**: scipy(AD·KS·Beta 분위수), pytest(테스트). 현재 `daynight/requirements.txt`에 scipy 없음.
2. **타임스탬프 규약**: KOSPI `bar_end`, ETF `bar_start`로 확정됨(실측). 로더에 파일별 규약을 명시하고 G00에서 재확인.
3. **기존 결과 격리**: 기존 `results/final`(2026 포함)은 그대로 두고, 본 연구는 별도 봉인 절차로 수행.
4. **첫 착수 범위**: G00까지 구현 후 결과 검토, 이후 G01 진행 여부.

## 7. 실행 순서

```text
P0 환경 → G00 데이터·시간 → G01 샘플링 → G02 알파 → G03 청산
  → G04 Activity → G05 Beta → G06 W1/RV → G07 비중
  → LOCK → G09 True OOS 1회
```
