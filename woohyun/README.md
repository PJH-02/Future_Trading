# KOSPI200 Foreign Flow Backtest

## 1. 연구 목적

KOSPI200 선물시장에서 외국인 투자자의 순매수·순매도 흐름이
다음 거래일 KOSPI200 지수 수익률에 유의한 정보를 제공하는지 검증한다.

## 2. 데이터

- 기간: 2023-09-27 ~ 2026-09-22
- KOSPI200 가격 데이터
- 외국인 매수/매도 데이터
- 일별 데이터 사용

## 3. Foreign Flow 정의

Foreign Flow는 다음과 같이 계산한다.

Foreign Flow =
(Foreign Buy - Foreign Sell) /
(Foreign Buy + Foreign Sell)

값이 클수록 외국인 매수 우위,
값이 작을수록 외국인 매도 우위를 의미한다.

## 4. 백테스트 방법

### 1-Day Foreign Flow

당일 외국인 Flow를 기준으로 다음 거래일 수익률을 분석한다.

Foreign Flow를 5개 분위수(Q1~Q5)로 나누어
각 그룹의 평균 다음날 수익률을 비교한다.

### 3-Day Foreign Flow

최근 3거래일의 Foreign Flow를 누적하여
다음 거래일 수익률과의 관계를 분석한다.

## 5. 주요 결과

### 1-Day Foreign Flow

| Quintile | 다음날 평균 수익률 |
|---|---:|
| Q1 | -0.276% |
| Q2 | -0.096% |
| Q3 | +0.080% |
| Q4 | +0.099% |
| Q5 | +0.141% |

Foreign Flow가 증가할수록 다음날 평균 수익률이
전반적으로 증가하는 패턴이 나타났다.

회귀분석 결과:

- Beta: +0.100
- HAC t-stat: 2.719
- p-value: 0.0065
- R²: 0.0078

## 6. 해석

Foreign Flow는 다음날 수익률과 양(+)의 관계를 보였다.

다만 R²가 약 0.8%로 낮기 때문에 Foreign Flow만으로
수익률을 설명하기는 어렵다.

따라서 본 연구에서는 Foreign Flow를 독립적인
매매전략이라기보다 방향성 필터로 활용하는 것을 고려한다.

## 7. 주의사항

본 결과는 현재 표본에 대한 탐색적 백테스트 결과이며,
거래비용·슬리피지 및 완전한 Out-of-Sample 검증을
반영하지 않았다.

따라서 실제 투자성과를 의미하지 않는다.