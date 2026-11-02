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

3차(D2): `topics/r3_d2.py`(2003~2009 판정), `topics/bt_d2_2010_2026.py`(2010~2026 전략 백테스트, matplotlib 필요, 재생성만 할 때는 `--no-ledger`), 탐색 `tools/preopen_d2_explore.py`·자료 품질 `tools/preopen_d2_h0_quality.py`. 사양 등 이전 결과는 `data.result_file()` 로 `<OUT>` 에 없으면 `../results/2026-09-29-overnight/` 에서 읽는다.

## 13·15차 시간대별 장부와 공유 구성 (2026-10-30)

• `topics/r13_slot_book.py`: 13차 시간대별 장부. 코스피·코스닥 각각 밤(A, 15:30 종가 → 익일 09:00 시가)·아침(B, 09:00 → 15:30)·낮(C, 장중) 다리를 같은 자본으로 회전한다(같은 시장에서 B 활성일엔 C 0). 합산 = 코스피 50% + 코스닥 50% 매일 재조정. 다리 시계열 규격은 `date,pos,gross,net`(자본 1 기준 일수익, 밤 다리는 date = 청산일) + `<handle>.meta.json`. 결과 `../results/2026-09-29-overnight/round13_slot_book/`.
• 공유 구성 확정판 `topics/r15_final_book_v3.py`(결과 `../results/2026-09-29-overnight/round15_alpha_share/final_v3/`, k_picks_v3.csv·k_grid_v3.csv): 13차 장부에서 코스닥 밤 다리(롱·인버스 모두)는 15:30 종가로 신호를 확인한 뒤 15:45 코스닥150 선물로 진입해 익일 08:45 청산하고(만기 전날 밤은 만기물, 2016~2017 롱 밤은 233740 유지), 코스닥 낮 숏은 선물로(원 체결 모형 총수익 + 선물 비용), 아침 두 다리의 배수는 2배 ETF 혼합 한도 위만 지수 수익으로 계산하며, 연속 매매 낮 다리에는 호가 1틱을 더한다. 전체 명목은 k배(대표 1.40 = 최근 4년·10년 창 모두 MDD 15% 안 최대 배수). 묶음은 대표(09:00 단일가, 선물 1.8bp), 중간(아침 09:01 체결), 하한(09:01 + 선물 3bp + 단일가 ETF 1틱) 세 가지. 시나리오 계산은 `topics/verify_r15_v2_review.py`(독립 검토 코드)를 그대로 실행하며 그 점검 출력은 `final_v3/from_review/` 에 있다.
• 이전 판: `topics/r15_final_book.py`(결과 `final/`)는 코스닥 롱 밤의 종가 신호·종가 체결, 아침 다리의 일괄 배수, 251340 에만 준 틱 비용 때문에 낙관이라 폐기, `topics/r15_final_book_v2.py`(결과 `final_v2/`)는 롤 밤 처리 때문에 장기창 낙폭이 작게 나와 폐기. 기록용으로만 남긴다.
• `topics/r15_leverage_desk.py`: 13차 장부 × k 기준선(결과 `round15_alpha_share/leverage_desk/`). `topics/r15_01~06_*.py`: 15차 사전 등록 설계 6개(판정은 각 폴더 notes.md·cells.csv, 교체·후보 시계열은 `round15_alpha_share/series/`).
• 입력: 13차·15차 시계열은 결과 폴더 series/ 에 있어 `r15_final_book.py` 의 장부 재현은 저장소 자료만으로 된다(선물 경로 독립 대조에는 코스닥150 선물 일봉이 필요). 15차 신규 원자료(코스피200·코스닥150 선물 일봉·1분봉, ETF NAV, VKOSPI 일봉 등 KIS 시세 조회분, 코스닥150 1배 ETF 일봉)는 넣지 않았다. 코드의 `<OUT>`(결과 루트), `<raw>`(원자료 루트), `<repo>`(이 저장소) 표기는 실행 환경에 맞게 바꾼다.
• 규약: 비용 ETF 왕복 5bp, 선물 왕복 1.8bp, 251340 호가 비용 연도별(2024 13.6, 2025 14.2, 2026 21.5, 그 외 10bp), ETF 09:00·15:30 및 선물 08:45·15:45 단일가 체결 가정, 통계 HAC(maxlags 5).
