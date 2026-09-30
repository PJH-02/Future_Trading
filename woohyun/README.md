# KOSPI200 Futures Foreign Flow & Intraday Strategy

## Research Overview

본 연구는 **KOSPI200 선물시장의 외국인 Flow가 다음 거래일 현물시장의 방향성에 유의미한 정보를 제공하는지** 검증하고, 이를 KODEX 200 장중 매매의 방향성 필터로 활용하는 것을 목적으로 한다.

전일 외국인 선물 Flow를 이용해 당일 매매 여부를 판단하고, 실제 매수 시점은 KODEX 200 1분봉의 **ORB(Opening Range Breakout) + VWAP**을 이용한다.

---

## Strategy

**1. Foreign Flow Filter**

\[
Flow = \frac{Foreign\ Buy - Foreign\ Sell}{Foreign\ Buy + Foreign\ Sell}
\]

전 거래일 Flow가 과거 Rolling 250일 기준 **상위 40%**인 경우에만 당일 매매를 허용한다.

**2. Intraday Entry**

- 대상: KODEX 200 (069500)
- 데이터: 1분봉
- Opening Range: 09:01~09:10
- Entry Condition:
  - 현재가 > OR High
  - 현재가 > VWAP
  - 전일 Foreign Flow ≥ Rolling 60th Percentile
- 진입: 신호 발생 다음 1분봉 시가
- 방향: Long Only

**3. Exit**

5 / 10 / 20 / 30 / 60분 보유 후 청산.

---

## Flow Analysis

전일 외국인 Flow와 다음 거래일 장중 수익률의 관계를 분석한 결과:

| Flow Group | Next-day Intraday Return |
|---|---:|
| Q1 | -0.2765% |
| Q2 | -0.0963% |
| Q3 | +0.0795% |
| Q4 | +0.0991% |
| Q5 | +0.1407% |

Q5-Q1 차이: **약 +0.417%p**

1-day Flow 회귀분석:

- Beta: **+0.1001**
- HAC t-stat: **2.719**
- p-value: **0.0065**
- R²: **0.0078**

Flow는 통계적으로 유의한 양(+)의 관계를 보였지만, 설명력은 낮아 **독립적인 매매신호보다는 방향성 필터로 활용**하였다.

---

## Backtest Results

### Foreign Flow Top 40% + ORB + VWAP

| Holding | Trades | Win Rate | Avg Return | PF | Sharpe | MDD |
|---|---:|---:|---:|---:|---:|---:|
| 5 min | 152 | 54.61% | +0.0535% | 1.678 | 2.479 | -2.40% |
| 10 min | 152 | 51.32% | +0.0394% | 1.327 | 1.491 | -4.62% |
| 20 min | 152 | 51.32% | +0.0185% | 1.114 | 0.586 | -4.66% |
| **30 min** | **152** | **56.58%** | **+0.0693%** | **1.417** | **1.841** | **-3.56%** |
| 60 min | 151 | 52.32% | +0.1313% | 1.605 | 2.389 | -4.07% |

---

## Flow Threshold Comparison

30분 보유 기준:

| Filter | Trades | Win Rate | Avg Return | PF |
|---|---:|---:|---:|---:|
| Top 20% | 78 | 57.69% | +0.1281% | 1.736 |
| Top 30% | 117 | 54.70% | +0.0895% | 1.524 |
| **Top 40%** | **152** | **56.58%** | **+0.0693%** | **1.417** |
| Flow > 0 | 262 | 49.62% | +0.0200% | 1.124 |
| Price Only | 651 | 49.92% | +0.0248% | 1.181 |

Flow threshold를 높일수록 거래 횟수는 감소하지만 거래당 성과가 개선되는 패턴이 나타났다.

---

## Key Finding

전일 **KOSPI200 선물 외국인 Flow**를 방향성 필터로 사용하고, **KODEX 200 1분봉의 ORB + VWAP**을 이용해 매수 시점을 결정하는 전략에서 양(+)의 성과 패턴을 확인하였다.

핵심 구조:

**Foreign Futures Flow → Directional Filter → KODEX200 1-minute → ORB + VWAP → Long Entry**

다만 거래비용, 슬리피지 및 Out-of-Sample 검증이 추가적으로 필요하다.