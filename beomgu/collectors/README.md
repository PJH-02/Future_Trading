# 원자료 수집기 (로그인 불필요)

| 스크립트 | 산출(`data/raw/`) |
|---|---|
| `collect_fdr.py` | K200 지수 일봉(FDR 1990~ + 네이버 2014~) |
| `collect_public.py` · `collect_public.py us` | KODEX 200·KOSPI 일봉 / ES·NQ·S&P·나스닥 일봉(yfinance) |
| `collect_naver.py fut` | 코스피200 선물 투자자별 일별(네이버증권 JSON, 2010-09~) |
| `collect_naver_intraday.py` | 당일 1분 투자자 잠정치(장 마감 후 매일 실행) |

분석 코드는 `OPTFLOW_RAW_DIR` 로 이 폴더의 `data/raw` 를 가리키면 된다. 패키지: `../opt_flow_study/requirements-collect.txt`
