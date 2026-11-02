# KRX best policy and portfolio study

Primary portfolio evaluation: 2025-12-09 through 2026-09-17 (190 KRX sessions).
Individual policy selection window: full-history; selected by maximum CALMAR over each team's native history through the six-source common end.

The allocator keeps all six sleeves in a fixed universe for every method and date. EW targets 1/6 per sleeve; a known source NA is logged and treated as suspended cash for execution. LP and Convex use the latest 252 complete common observations available before each target.

Policy choice is exploratory and ex-post. It is not an untouched out-of-sample result. CAGR is annualized as (1 + total return)^(252 / sessions) − 1; short samples are explicitly labeled in the CSV. The CSV includes both shared post-warmup and full-history-through-common-end metrics.

The reported rebalance proxy charges 1.5 bp commission plus 1 bp adverse slippage against each sleeve's absolute capital-budget delta. It is a conservative capital-budget resize proxy, not actual ETF fills or T+2 settlement. Treasury/internal cash transfers incur no cost. Ordinary strategy trade costs are already included in sleeve returns. The no-extra-order-cost sensitivity isolates the funding-transfer case.

## Individual policy results

S0 is the saved original policy benchmark. The selected row uses the configured selection window.

| Strategy | Policy | CAGR | MDD | Sharpe | Calmar |
|---|---:|---:|---:|---:|---:|---:|
| daynight_v1_combined | S0 | 0.4390 | -0.2073 | 1.334 | 2.118 |
| daynight_v1_combined | S0 | 0.4390 | -0.2073 | 1.334 | 2.118 |
| intraday_kospi_final | S0 | 0.0756 | -0.0435 | 0.929 | 1.737 |
| intraday_kospi_final | S0 | 0.0756 | -0.0435 | 0.929 | 1.737 |
| intraday_kosdaq_final | S0 | 0.1256 | -0.0907 | 1.015 | 1.385 |
| intraday_kosdaq_final | S0 | 0.1256 | -0.0907 | 1.015 | 1.385 |

Selected-policy entry leverage audit (closed trade events):

| Strategy | Policy | Mean L | Max L | L=0 hits | L=2 hits |
|---|---:|---:|---:|---:|---:|

## Best main portfolio rows by method

| Method | Cap | Band | CAGR | MDD | Sharpe | Calmar |
|---|---:|---:|---:|---:|---:|---:|
| EW | 0.00 | 0.05 | 0.2230 | -0.0436 | 2.229 | 5.110 |
| CONVEX | 0.05 | 0.10 | 0.0945 | -0.0771 | 0.857 | 1.226 |
| LP | 0.05 | 0.05 | 0.0728 | -0.0857 | 0.665 | 0.849 |

## Metric-specific portfolio winners

| Metric | Method | Cap | Band | Value |
|---|---:|---:|---:|---:|
| Highest CAGR | EW | 0.00 | 0.10 | 0.2241 |
| Smallest absolute MDD | EW | 0.00 | 0.05 | -0.0436 |
| Highest Sharpe | EW | 0.00 | 0.05 | 2.2291 |
| Highest Calmar | EW | 0.00 | 0.05 | 5.1095 |

## Limitations

- 중앙 EW/LP/Convex는 overnight gap 이후 다음 세션 funding을 반영하는 일별 TWR surrogate이며, 실제 T+2 settlement·수량/호가단위·ETF resize fill 감사가 아닙니다.
- 중앙 internal transfer 자체 비용은 0입니다. CENTRAL_RESIZE_COST_BPS는 ETF 주문비용 대신 쓰는 명시적 notional proxy이며, 기본값 0이고 ETF gross/net delta·세금·분배금은 별도 검증이 필요합니다.
- Beomgu의 상장 전 leveraged history는 synthetic ETF 경로를 포함합니다. Sunny는 실제 ETF fill이 아닌 index/mini-futures proxy입니다.
- Beomgu 원본은 같은 09:00 시가로 신호와 체결을 가정합니다. 원본을 보존한 비교이며, 실행 가능한 인과적 매매 성과로 인증하지 않습니다.
- S0_MATCHED_DEV는 개발 구간의 평균 진입 노출로 고정한 대조군이며 개발 종료 다음 날부터만 평가합니다. 노출의 일중 평균이나 완전한 위험 일치를 뜻하지 않습니다.
- 252일은 현재 비교 가능한 유효 관측일 기준입니다. 전략별 누락 때문에 실제 거래일 범위가 더 길 수 있으며 estimator_start/end에 기록합니다. 분기 표는 사전 고정 정책의 순차 결과이며 중첩 정책 선정 검증을 수행하지 않습니다.
- intraday 최종 규칙은 2026-10-07까지의 전체 sample을 사용해 선정되었으므로 2026-10 이후는 untouched holdout으로 볼 수 없습니다.
- 동일일 신호가 없는 고정형 sleeve의 원천 NA는 optimizer에서 NA로 유지하고 execution funding surrogate에서는 suspended cash로 처리하며 감사 파일에 남깁니다.

All daily returns, target vectors, solver statuses, and budget-delta charges are in the adjacent CSV artifacts.
