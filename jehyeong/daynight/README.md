# Kosdaq day & night

로컬 보유 1분봉 및 일봉으로 두 버전을 비교합니다. 시그널 종목은 사용자가 확인한 KODEX 코스닥150 ETF(229200), 매수 종목은 레버리지(233740)·인버스(251340)입니다.

```bash
python3 jehyeong/daynight/backtest_daynight.py
python3 -m unittest discover -s jehyeong/daynight -p 'test_*.py' -v
python3 jehyeong/daynight/independent_check.py
```

Windows에서는 Python 환경에 따라 `python3` 대신 `python`을 사용합니다. 스크립트는 어느 작업 디렉터리에서 실행하든 기본 데이터·출력 경로를 스크립트 위치로부터 결정합니다.

```bash
python3 backtest_daynight.py --start 2022-01-01 --end 2026-12-31 --initial-capital 100000000
```

기본 입력은 `../intraday/data/{229200,233740,251340}_daily.csv` 및 `../intraday/data/minute/{symbol}_TOSS_1m.csv`입니다. 입력은 읽기 전용이며 데이터를 수집하거나 주문하지 않습니다. `--data-dir`, `--output-dir`로 변경할 수 있습니다. 재실행하면 출력 폴더의 동일한 결과 파일을 갱신합니다.

- V1: 15:15 **시작** 분봉 종가로 IBS 판정(15:16 확정).
- V2: 15:19 **시작** 분봉 종가로 IBS 판정(15:20 확정).
- 원본은 종료 시각 라벨입니다. 예: 요청 09:06 시가 = 원본 09:07 봉 시가.
- 사용자 승인에 따라 오버나이트 매수 참고가격은 **15:20 직전 마지막 양의 거래량 분봉 종가**입니다. 이는 실제 15:20 시장가 체결을 가정할 수 없는 데이터 문제를 처리한 연구용 가격 가정입니다. V1에서도 원안과 달라진 승인 사항입니다. V2는 신호 확정과 동시에 지연 없이 해당 가격으로 매수한다고 가정합니다.
- 초기 1억원, 매 거래 현금 전액 재투자, 소수점 수량, 수수료·슬리피지 편도 각각 1.5bp·1bp, 일별 15:30 종가 평가. 배당·현금 이자·별도 세금은 추가하지 않습니다.
- 결합 전략은 하나의 계좌에서 시간순으로 운용합니다. 개별 전략은 각각 별도의 초기 1억원으로 평가하는 비교 대상입니다.

`results/report.html`을 브라우저로 열면 전체·연도별 CAGR, MDD, Sharpe, turnover, P&L 그래프를 모두 확인할 수 있습니다. 그래프가 포함된 단일 HTML이므로 오프라인에서도 열립니다. `report.md`도 제공합니다.

`summary.csv`, `yearly.csv`는 원시 수치입니다. 수익률 0.1은 10%, turnover 2는 양방향 거래대금/NAV 2배입니다. `annualized_turnover`는 연환산 회전율입니다. 연도별 `total_return`과 연환산 `cagr`을 구분해 보십시오. MDD는 일별 종가 평가 기준이며 장중 낙폭이 아닙니다. 상세 산식과 부분연도·누락 데이터 처리는 보고서 및 `manifest.json`에 기록합니다.

`independent_check.py`는 엔진을 import하지 않고 원본 데이터에서 신호와 계좌 NAV를 다시 계산하고, 실제 주문 수량·현금 원장, 모든 전체·연도별 지표 및 입력 해시를 대조합니다. 기본 `results` 폴더를 검사하며 설정은 그 폴더의 manifest에서 읽습니다. 검증 결과는 `independent_check.json`입니다. 백테스트 재실행 시 과거 판정이 현재 결과로 오인되지 않도록 `acceptance.json`은 PENDING으로 초기화됩니다.

거래·감사 파일: `orders.csv`, `trades.csv`, `signals.csv`, `daily_nav.csv`, `skips.csv`, `data_quality.csv`, `manifest.json`, `acceptance.json`. PNG/SVG 그래프는 `pnl_V1`, `pnl_V2`, `pnl_comparison`, `pnl_yearly`입니다.

구현 위임 기록: GLM 연결 실패 및 HTTP429로 진행하지 못했고, 사용자 승인하에 직접 OpenCode DeepSeek 무료 경로로 전환했으나 출력 길이 제한으로 코드 없이 종료됐습니다. 두 차례 재위임 한도 도달 후 Codex가 사용자에게 알리고 직접 구현·검증을 완료했습니다. 상세 실행 정보는 `acceptance.json`에 있습니다.
