# V3 / V4 추가 백테스트

```bash
python3 jehyeong/daynight/backtest_v3_v4.py
python3 -m unittest discover -s jehyeong/daynight -p 'test_*.py' -v
python3 jehyeong/daynight/independent_v3_v4.py
```

기존 V1·V2의 코드·결과는 변경하지 않습니다. 새 결과는 `results_v3_v4/report.html`과 같은 폴더의 CSV·PNG·SVG에 저장합니다. HTML에는 그래프가 포함되어 오프라인으로 열 수 있습니다.

사용자 확인 사항:

- V3는 V2를 기준으로 장중 진입 가격만 09:05 시작 분봉의 종가로 변경합니다. 원본 종료 라벨은 09:06이며, 신호 판단은 기존 09:05 시가를 유지합니다. 청산은 15:00 시가입니다.
- 오버나이트는 V2와 같은 15:19 시작 분봉 종가 IBS, 승인된 15:20 직전 마지막 체결가 참고가격, 다음 거래일 09:00 시가 청산입니다.
- V4는 V3에 두 독립적인 정상 전략 모의 계좌를 둡니다. 각 계좌는 초기 1억원과 실제 수수료·슬리피지로 정상 방향 거래를 계속합니다. 누적 **확정** P&L만 MA20의 입력이며, 실제 V4나 결합 계좌 손익을 사용하지 않습니다.
- 20일은 거래 횟수가 아닌 거래소 20거래일입니다. 거래 없는 날은 마지막 확정 P&L을 유지합니다. 청산 시 현재 값과 직전 19거래일 말의 값을 평균하며, 20개 관측치 미만이면 정상 방향입니다.
- 정상 모의 P&L < MA20이면 반전, >= MA20이면 정상 복귀입니다. 상태는 정상 모의 포지션 청산 시에만 갱신합니다. 장중은 전날까지의 갱신 상태, 오버나이트는 당일 09:00 청산 후 상태로 진입합니다.
- 반전은 레버리지(233740) 매수와 인버스(251340) 매수를 교환하며, 각각의 실제 ETF 가격을 사용합니다. 정상 손익에 단순히 -1을 곱하지 않습니다.
- 매 거래 현금 전액 재투자, 소수점 수량, 매수·매도 편도 수수료 1.5bp·슬리피지 1bp를 유지합니다. 성과 지표는 비용 차감 후 15:30 종가 평가 NAV 기준입니다. 분봉 종가와 15:20 참고가격 체결은 이상적 가격 가정입니다.

주요 결과 파일:

- `summary.csv`, `yearly.csv`: V3·V4의 장중·오버나이트·결합 전체 및 연도별 모든 지표.
- `comparison_summary.csv`, `comparison_yearly.csv`: 동일 설정·입력 데이터인 경우 보존된 V1·V2와 합친 비교표.
- `shadow_gate.csv`: 각 정상 모의 전략의 확정 P&L, 실제 일별 MA20, 청산 시 결정 MA20, 전후 모드와 진입 모드.
- `execution_signals.csv`: V4의 원래 방향·반전 배수·실제 방향·ETF 및 진입 시점에 이용 가능한 판단 자료의 시각. 장중 행에 당일 오후 청산 결과를 진입 판단 자료로 사용하지 않습니다.
- `switch_events.csv`: 정상·반전 상태가 바뀐 청산 시점.
- `orders.csv`, `trades.csv`, `daily_nav.csv`, `skips.csv`, `manifest.json`, `independent_check.json`, `acceptance.json`: 거래·계좌·검증 추적.

`independent_v3_v4.py`는 기존 엔진과 필터 모듈을 import하지 않고 원본 분봉에서 두 정상 모의 계좌·각 청산 시점의 모드·6개 계좌 NAV를 재구성합니다. 거래 방향·실제 ETF·주문 현금 원장·전체 및 연도별 지표·회전율을 대조합니다.

`backtest_v3_v4.py`는 `--data-dir`, `--output-dir`, `--start`, `--end`, `--initial-capital`을 지원합니다. 독립 검사 스크립트는 기본 `results_v3_v4` 폴더를 검사합니다. 재실행 시 해당 폴더의 동일한 파일을 갱신하고 acceptance를 PENDING으로 초기화합니다.
