# overnight_lab — 종가 베팅·오버나잇 백테스트

- `data.py`: 가격(네이버 siseJson 수정주가, 지수·선물은 fchart), 네이버 투자자별·프로그램(코스피·코스닥), KRX 옵션, 미국 지수 일봉 로더
- `engine.py`: 구간 수익률(on = 종가→익일 시가, oc, cc), 기간 창, Δ·Newey-West t, 연도 층화 플라시보, 기간별 표, 시행 장부
- `common.py`: 셀 판정 블록(공통 규약) + 마크다운
- `topics/t01~t08_*.py`: 주제별 사전 고정 스크립트(docstring 에 규칙 기록 후 실행)

실행: `python topics/t01_intraday_strength.py` (산출: `<OUT>/<주제>/report.md`, 시행 장부 `<OUT>/ledger.csv`)

폴더: 원자료 `../data`(가격 캐시 `../data/etf/`), 산출 `../results/overnight_lab`, 수집기 `../collectors` 가 기본값이다. 바꾸려면 환경변수 `OVERNIGHT_RAW_DIR`·`OVERNIGHT_OUT_DIR`·`OVERNIGHT_COLLECTOR_DIR` 또는 이 폴더의 `config.local.json`(raw_dir, out_dir, collector_dir)을 쓴다.

패키지: pandas, numpy, requests, tabulate (T08 보고용 ES 1시간봉은 yfinance)

규약 요약: 판정 구간 번인 후 ~ 2025-06-05, A(~2019)/B(2020~2025-06)/E4(2025-06-09~, 보고만), 비용 왕복 5bp(스트레스 7bp), 신호 NaN(번인·결측) 밤은 표본 제외.

2차 디벨롭 스크립트 `topics/r2_d1.py`~`r2_d4.py` 는 사전 고정 사양 `spec.json` 을 `<OUT>/round2/<D1~D4>/spec.json` 에서 읽고, 없으면 `../results/2026-09-29-overnight/round2/<D1~D4>/spec.json` 을 읽는다(`data.spec_file`). 결과는 `<OUT>/round2/<D1~D4>/` 에 쓴다.

주의: 기본 출력 폴더 `../results/overnight_lab/` 에는 실행 환경의 절대경로가 기록될 수 있으므로 커밋하지 않는다(`../.gitignore`).
