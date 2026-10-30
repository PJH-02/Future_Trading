# KOSPI·KOSDAQ Event-Bar 기반 조건부 확률 베팅 전략 — 구현 및 검증 명세 v2.0 — 최종 수정

- 작성 기준일: 2026-10-29
- 상태: **사전등록용 연구 설계. 실증 성과·수익성 미확인.**
- 목적: 기존 레포에서 최소 코드로 **가격충격의 지속(Trend) / 복원(MR)** 독립 전략의 성공확률을 추정하고, **Beta–Bernoulli의 순수 증분 효과**, **W1 Guard의 순수 증분 효과**, **청산 규칙 3종**, **확률 기반 비중 조절의 증분 효과**를 순서대로 검증한다.
- 우선순위: (1) 전일 고정 임계값으로 인과적 Event Bar → (2) 경제적 가격충격 시그널과 청산 3종 확정 → (3) 독립적인 기초 알파 → (4) Beta 확률 필터 → (5) W1/RV Guard → (6) 동적 비중/ETF 배수 → (7) 2026 True OOS 1회.
- 절대 제한: 미래 정보 누출, 2026년을 이용한 선택·수정, 성과가 좋도록 조건/승률/손절을 사후 최적화하는 행위, 실제 없는 거래대금·틱·호가·ETF를 만들어 내는 행위.

> **판정의 출발점:** 어떤 기술적 지표에도 '적당한 수익이 안정적으로 나는' 보편적 파라미터는 없다. 따라서 기준 시그널은 **재현성과 해석 가능성**으로 선택하고, 그 시그널에 Beta와 W1을 더한 효과는 **동일 후보 사건·동일 실행가격·동일 위험 예산**으로 측정한다. 이 전략의 경제적 수익은 베타분포나 Event Bar 자체에서 생성되는 것이 아니라, 유의미한 가격 지속/복원의 구조가 실제로 존재할 때 발생한다.

## 0. 확정된 맥락 및 수정하지 않을 가정

1. **거래 범위:** 한국 장중 주가지수 관련 ETF 매수·청산. 스윙, 오버나이트, 선물·옵션 직접거래 없음. +1x/−1x를 1차 연구, +2x/−2x는 2차 선택적 검증. 인버스 ETF는 *매수*하여 하락에 대응한다.
2. **입력:** KOSPI·KOSDAQ 지수의 1분 OHLCV, 이용 가능한 +1x/−1x/+2x/−2x ETF의 1분 OHLCV. 보유 파일과 상품 메타데이터에서 확인된 시장만 활성화. 앞선 실행 요약에서 KOSDAQ 데이터 부재가 보고되었으므로 실제 파일 존재 여부를 먼저 검사한다. KOSPI/KOSDAQ 전체지수와 거래 ETF가 추종하는 KOSPI200/KOSDAQ150/선물지수의 불일치는 한계로 명시한다.
3. **체결의 연구 가정:** 지수·ETF 바가 확정된 다음 ETF **1분봉 시가**에 진입/청산. `available_at`보다 엄격히 미래인 실행가격만 사용. 사용자 지정 이상화이며 실제 체결 가능성이 보장되지 않는다. 비용 스트레스 필수.
4. **신호시간:** 신규 후보 09:30–15:00 KST. 원칙적으로 **15:20 전 연속매매 중** 청산하며 마지막 계획 매도는 ETF 원본 1분봉의 **15:19:00 bar-start open**(해당 데이터에 존재할 때). 15:20–15:30 동시호가 종가를 1분 시가로 대신 사용하지 않는다. 정규장 종료는 15:30이지만 접속매매와 구분한다. 오버나이트 금지.
5. **개발/동결:** 2022-11-01~2025-12-31에서 진단·파라미터 결정·대회·개발 백테스트. **2026-01-01~2026-10-29 중 실제 수집된 기간은 True OOS로 봉인하고 최종 모델/정책에 대해 한 번만 평가.** 2026 자료를 보고 기준선을 바꾸면 해당 해의 True OOS 지위가 소실된다.
6. **모델 크기:** 가격충격 하나에서 T/MR 반대 방향 후보를 생성하고, 상대 거래활동 High/Low 한 축만 조건부 검증. EMA/RSI는 별도 기준선. HMM, 전이확률 학습, 복잡한 ML, Gaussian Kelly는 범위 밖. Beta는 관측된 완료 성공/실패의 저차원 온라인 집계다.

## 1. 경제적 가설·비교 실험의 기본 원칙

### 1.1. 두 개의 독립 경제적 가설과 기준선

**T — Trend Continuation.** 최근 3개 Event Bar에 걸쳐 큰 가격충격이 발생했다면, 정보의 지연 반영 또는 분할 주문 집행 때문에 이후 가격이 **충격과 같은 방향**으로 추가 이동할 수 있다. 기본 시그널은 가격충격의 부호를 그대로 사용한다.

**R — Mean Reversion.** 동일 가격충격이 정보가 아닌 일시적 유동성 수요의 결과였다면, 주문 압력이 사라진 후 **충격과 반대 방향**으로 복원될 수 있다. 기본 시그널은 가격충격 부호를 반전한다.

두 전략은 **같은 이벤트를 반대 방향으로 해석**하므로 최초에는 동일 이벤트에 대해 별도 ETF shadow P&L을 생성한다. 하나의 실제 계좌에서 두 방향을 동시에 실행하지 않는다. `EMA(16,64)` Trend와 `Wilder RSI(14), 30/70` MR은 **복제 가능한 기술적 기준선만** 담당하며, Beta/W1의 순수 증분 효과는 주력인 공통 가격충격 시그널을 고정해 측정한다.

경제적 제약: 지수 OHLCV만으로 정보거래와 강제 유동성 거래를 직접 식별하는 것은 불가능하다. 따라서 가격충격·거래활동은 인과적 가설에서 도출한 대리변수이며, 실제 알파는 조건부 ETF 순손익으로 독립 검증한다.

### 1.2. 순수 증분 비교 정책 (최소 → 확장)

| 정책 | 동일한 기초 가격충격 신호 | Beta 필터 | 분포·변동성 Guard | ETF 노출 비중 |
|---|---|---|---|---|
| `P0_FIXED` | T 또는 R | 없음 | 없음 | 고정 계좌 10% 현금 |
| `P1_BETA` | P0와 동일 | 보수적 순EV | 없음 | 고정 10% |
| `P2_BETA_RV` | P1과 동일 | 동일 | RV Hard Guard | 고정 10% |
| `P3_BETA_W1` | P1과 동일 | 동일 | W1 Hard Guard | 고정 10% |
| `P4_BETA_SIZE` | P1과 동일 | 동일 | 없음 | Beta 신뢰도 기반 **0/5/10%** |
| `P5_BETA_SIZE_W1_HARD` | P4와 동일 | 동일 | W1 Hard Guard | Beta 0/5/10% |
| `P6_BETA_W1_SOFT` | P1과 동일 | 동일 | W1 단계별 축소 | 0/5/10% |

- `P0~P3`는 최초 필수 비교이며 **모두 고정 비중**이다. `P4~P6`는 Beta와 W1의 추가 가치가 입증될 때만 실행하는 **후속 확장 실험**이다. 위험회피적 비중 조절이 꼭 수익률을 높이는 것은 아니다.
- 각 **청산 정책은 완전히 별도의 후보·성과·Beta 모델 키**로 실험한다. 같은 진입시간의 paired 평가와, 현실적으로 달라지는 신규 진입 가능 횟수 비교를 분리한다.
- Guard가 거래를 차단해도 같은 시점의 shadow 결과는 확정시점에 Beta 업데이트에 사용한다. 모든 정책에서 **동일 원본 후보 스트림·동일한 정책별 잠재 손익·동일한 과거 완료 결과**를 사용한다. Beta 리셋은 별도 실험으로만 다룬다.
- `P4/P5/P6`의 기준 자본 위험 예산은 10% 현금의 +1x ETF 익스포저다. **포지션 비중**과 ETF 자체의 **레버리지 배수**는 서로 다른 변수이며, 먼저 비중 0~10% 조절만 검증한다.

## 2. 사전 과제: 사용자 확인이 필요한 사실, 업계 표준과 연구 기본값

### 2.1. 반드시 사실로 확인할 항목 (값을 임의 생성하지 말 것)

| 항목 | 기본 처리 | 선택/확인 주체 |
|---|---|---|
| KOSDAQ 1분 지수/ETF 파일 실제 존재 | 없으면 KOSDAQ만 `DATA_BLOCKED` | 파일 검사(사용자 필요 시 보완) |
| 지수 거래대금 `turnover_krw` 실제 필드 존재 | 없으면 **지수 Won Bar 금지** | 데이터 스키마 확인 |
| ETF 티커·기초지수·레버리지 배수 | 실제 파일/메타데이터에서 자동 발견; 미확인 상품 거래 금지 | 사용자 데이터 |
| ETF 수수료, 상품별 세금, 실제 평균 슬리피지 | 연구 기본비용 0/2/5/10bp **왕복 총액** 스트레스; 실제 기본비용이 있다면 사용자 값으로 대체 | 사용자/계좌 조건 |
| 1분봉 timestamp가 bar-start인지 bar-end인지 | 제공자 규약을 확인하여 완료시각 통일. 자동 상관 극대화 시프트 금지 | 데이터 공급자/사용자 |
| 대상 ETF의 09:30–15:19 유효 분봉 | 없는 상품은 실행 비활성화 | 데이터 검사 |

### 2.2. 확정된 실험 기본값과 표준성 구분

| 변수 | 1차 기본값 | 해석 및 근거 수준 |
|---|---|---|
| Primary 경제적 이벤트 | **최근 3 Event Bar 누적 충격**; 과거 20바 변동성으로 표준화, `|Z| ≥ 1` | 가격 충격의 지속/복원 경제적 가설. 3/20/1은 **연구 초기값**, 업계 검증 최적값 아님 |
| Activity 조건 | 이벤트 거래활동 강도 `A_k / duration_k`의 **과거 동일 시간대 중앙값** 상·하 2구간 | 한 축만 추가. 지수/ETF 거래량의 원인을 직접 판별하는 것은 아님 |
| 기술적 비교 기준선 | Trend EMA(16,64), MR Wilder RSI(14)/30·70 | BTC 발표 및 널리 쓰이는 RSI 관행, **주력 전략 아님** |
| 하루 Event Bar 목표 | **50개/일**, 진단 민감도 30개 | mlfinpy의 1/50 평균 일거래량 예시를 참고한 출발점. **정확히 50개 생성되는 것은 아님** |
| 당일 고정 임계값 | 전일 이전 최근 **20개 완료 거래일** 총 거래량/원화 거래대금 근사치 **중앙값 ÷ 50** | 20일·중앙값은 이상치에 강한 연구 선택. 틱이 아닌 1분봉 집계라는 제약 |
| 주 청산 지평 `H` | **다음 완료 3개 Event Bar** | 사전등록 기본값; 보조 1·5바. `CLOSE_ONLY`에서는 H가 청산에 관여하지 않음 |
| **청산 정책 3종** | `EXIT_60`: min(H,60분, EOD); `EXIT_NO_60`: min(H,EOD); `EXIT_CLOSE_ONLY`: EOD | 사용자 요청에 따라 별도로 비교. **H=3과 60분은 업계 표준 아님** |
| EOD 청산 실행 | ETF의 마지막 유효 **15:19 bar-start open**, 접속매매 종료 전 | 동시호가 종가 대신 분봉에서 재현 가능한 마지막 실행가격. 데이터 규약 확인 필수 |
| Beta Prior | **Jeffreys Beta(0.5,0.5)** | 이항 승률의 표준 약정보 Prior 중 하나 |
| Beta 메모리 | 동일 그룹의 마지막 **100개 확정 후보거래** | 비정상성을 고려한 국소 Beta 추정 연구값 |
| Beta 활성화 | **30개 완료 사건**, 최소 10거래일, 성공·실패 각 5개 이상 | 과신 방지 진단값; 충분한 검정력 보장 없음 |
| Beta 성공 조건 | **실제 ETF 비용 차감 순수익률 > 0** | 실제 거래목표에 일치시킴. **비용을 EV에서 두 번 빼지 않음** |
| Beta 보수성 | `BetaPPF(0.05, W+0.5,L+0.5)` | 단측 5% 하한(연구 위험 선호값). 이항 독립성 위반 가능성 별도 검증 |
| W1 분포창 | 최근 30 vs 이전 120 완료 바 비중첩 | 연구값 |
| W1 경보/해제 | TRAIN `일별 W1 최대값 q90` / `개별 W1 q75` | 시장별 관측척도에 맞는 수치. **실제 이상상태 FPR을 보장하지 않음** |
| W1 Hysteresis | 경보 1바, 최소 정지 3바, 해제 2바 연속; probation 10완료 사건·3일 | 정지·재개 오판을 줄이는 연구값 |
| Beta 비중 정책 | `p05>pBE` → 5%; `p05≥pBE+0.05` → 10%; 이외 0 | 5%p의 **사후 확률 마진은 연구 초기값**, 실무 표준 아님. 고정 10%와 비교 |
| W1 Soft 비중 정책 | `D<q75_event`: 100%, `q75_event≤D<q90_dailymax`: 50%, `D≥q90_dailymax`: 0% | 동일 노출 상한 내 부분 축소 가설, 시장 수익률 확률로 해석 금지 |
| 2배 ETF | 초기 사용 금지, 후속 검증만. 같은 **기초지수 등가 노출** 비교 | 2x ETF 5% 현금 ≈ 1x ETF 10% 기초지수 명목노출. ETF 실현 손익·괴리 위험은 실제 가격 사용 |
| 비용/위험 | 왕복 0/2/5/10bp; 첫 기본 5bp, 현금 10%·상품 25% 상한·일일 -1% 신규 진입 중지 | 연구 기본값이며 계좌 실거래 비용/슬리피지 확인 전 실거래 승인 금지 |

**표준 vs 실험값:** Volume Bar의 누적량 임계값 방식, Jeffreys Prior, 사건 확정 후 베이지안 갱신은 이론적으로 뒷받침된다. 그러나 50개/일, 20일 중앙값, 3개 미래 바, 60분, 5%p 신뢰 마진, W1 q90/q75는 모두 사전등록한 연구 선택값이지 검증된 최적값이 아니다.

## 3. R00 — 데이터와 시간의 인과적 정렬

- 지수 및 ETF 1분 OHLCV에서 중복·비양의 가격·OHLC 불변조건·거래량 음수·휴장일·중간 누락을 검사한다.
- 기준시각은 **바가 실제 확정되어 이용 가능한 시각**. 09:00–09:01 바는 09:01에 확정된다. bar-end 규약의 09:01 행을 bar-start 09:01로 오해하면 즉시 누출.
- 같은 거래일의 연속된 확정 바만 합산한다. KOSPI 이전 종가→금일 시가 갭을 분봉 장중 수익률에 혼합하지 않는다. 결측분은 가격채우기 없이 윈도우 리셋/무효 처리.
- 지수 가격 × 지수 거래량 = 원화 거래대금으로 계산하지 않는다. 지수의 거래량 기반 **Volume Bar**와 대표 +1x ETF의 `(O+C)/2×V` 근사 **Won Bar**를 분리한다. 지수 Won Bar는 실측 원화 거래대금 필드가 없으면 사용 금지.
- 전체 KOSPI/KOSDAQ 지수 시그널과 ETF 추종지수의 불일치를 감사 기록에 명시. 사용자의 요구대로 호가·선물 리드래그·추적 오차 별도 모형은 만들지 않으나 이로 인해 상품 수익률이 단순 지수 수익률×배수가 된다고 가정하지 않는다.
- 테스트: 원본 t시점 이후 데이터를 무작위로 변경해도 t시점까지 샘플링/신호/W1/베타 업데이트가 불변이어야 한다(`future_perturbation`).

## 4. R01 — Event Sampling 비교 및 바 생성 정확성

### 4.1. 비교 파이프라인

1. `TIME`: **분당 원본 1분봉**은 정보표현 기준선. 공정한 거래 빈도 비교용으로 별도로 **완료 8개 1분 바(약 8분) Time Bar**를 둔다. 1분 기준선과 약 50 bars/day의 Event Bar를 같은 독립 표본수로 취급하지 않는다.
2. `VOL`: 지수 volume로 목표 바 수 50/일인 Minute-Aggregated Volume Bar.
3. `WON`: 실제 대표 +1x ETF의 `a_m=((O_m+C_m)/2)×V_m` 거래대금 근사로 목표 50/일의 Minute-Aggregated Won Bar. Event 완성 시각의 지수 확정가격을 이용해 지수 특징과 조인.

### 4.2. 개장 전에 고정하는 Volume/Won Bar의 **절대 임계값**

**중요: 당일 총 거래량은 필요하지 않다.** 목표 `50 bars/day`는 실제 당일 거래량을 50등분한다는 뜻이 아니라 **전일까지 관측한 통상 거래량을 50등분한 고정 누적량을 그날 사용**한다는 뜻이다.

전일까지 완료한 정상 거래일의 실제 하루 총 거래량을 `A_d`라 하면, 당일 개장 전에 다음을 계산하여 **장중 변경하지 않는다**.

\[
Q^{VOL}_d=\frac{\operatorname{median}(A^{VOL}_{d-20},\ldots,A^{VOL}_{d-1})}{50}
\]

\[
Q^{WON}_d=\frac{\operatorname{median}(A^{WON}_{d-20},\ldots,A^{WON}_{d-1})}{50}
\]

예: 이전 20개 정상거래일의 **일일 지수 거래량 중앙값 5억 단위**라면, 개장 전 당일 `Q_VOL=1,000만 단위`. 거래가 활발하면 바 60개, 뜸하면 35개처럼 변한다. **실제 50개 정확히 강제하지 않는다.** 또 ETF의 이전 20일 일 거래대금 근사 중앙값이 1,000억 원이라면 `Q_WON=20억 원`을 하루 동안 사용한다. 예시 숫자는 사용자의 관측값이 아니다.

순차 생성:

```python
# Pre-market: include only earlier COMPLETE and valid sessions
q_today = median(last_20_completed_daily_totals) / target_bars_per_day
accum = 0.0
pending_rows = []
for completed_minute in today's_session_minute_rows:
    measure = completed_minute.volume if mode == 'VOL' else (
        (completed_minute.open + completed_minute.close) / 2
        * completed_minute.volume
    )
    accum += measure
    pending_rows.append(completed_minute)
    if accum >= q_today:
        emit_whole_minute_aggregate(pending_rows, available_at=completed_minute.end_ts)
        pending_rows = []
        accum = 0.0
# Never turn an incomplete end-of-day remainder into a live signal.
```

- 최소 20개 **사전에 완료된 유효 거래일** 확보 전에는 `WARMUP`으로 선언. 결측/반일장과 유효하지 않은 거래일의 총량은 이전 정상일로 대체하거나 0으로 채우지 않는다; 사전 정의한 유효 거래일만 기록한다.
- 초과분(`overshoot`)은 포함한 마지막 **1분봉 전체에 귀속**, 다음 바로 넘기지 않음. 1분봉 하나를 둘로 쪼개거나 가상의 틱 체결을 만들지 않음. 오전에 생성된 바의 임계값·경계가 오후 수량에 의존하면 **미래정보 누출**.
- 동일 분에서 volume 0이면 새 이벤트가 형성되지 않을 수 있다. 이벤트 완성 순간의 `available_at`은 완료 1분봉 시각 이후다.
- 익일 임계값만 새로 계산하고 누적은 장 시작에 0으로 리셋. 미완성 일말 바는 거래 신호로 사용하지 않되 진단용 잔여량을 보고한다.
- 기록: `threshold_set_at, threshold_source_dates, threshold_volume_or_krw, target_bars_per_day, actual_bars_today, event_end_ts, available_at, overshoot_ratio, duration_minutes, incomplete_remainder`.
- `WON`은 대표 ETF `((O+C)/2)×V`라는 분봉 거래대금 **근사**이고, 정확한 틱별 KRW 체결대금이 아니다. 지수 OHLCV에는 실측 원화 거래대금이 없으면 지수×지수거래량을 임의 계산하지 않는다.

**바 수 50개 목표의 통계적 동기:** 이벤트 간 관측 빈도를 거래활동에 따라 조정하려는 것이지 정규분포·독립 동일분포를 보장하는 것이 아니다. 20일 median/50, 20일 mean/50 등 여러 임계값을 사후 성과로 조합 선택하지 않는다. 최초에는 median/50만 고정한다.

### 4.3. 검증 지표

- 바 개수/일 중앙값, 10/90 분위수; `overshoot_ratio` p50/p95; duration median/p95; 거래대금/거래량 연속성; 동일 시간 구간 중복 여부.
- 바 수익률의 왜도·첨도·QQ·Anderson–Darling 정규성, ACF/Ljung–Box, 제곱수익률 ACF/ARCH-LM, ADF/KPSS, `Var(r_event²)` 및 `Var(r_event²/duration)`.
- 이 수치는 **샘플링 품질 진단**일 뿐 통계적 정규성을 성공조건으로 강제하지 않는다. 최종 선별은 E1–E4에서 조건부 확률 예측의 OOS 품질과 ETF 경제적 성능이 개선되는지로 결정한다.
- 비동일 duration의 평가를 분리: **이벤트 지평 n개**와 **동일 실제 15/30/60분** 성과를 나란히 보고한다.

**R01 게이트:** 정확성 불합격→수정 전 연구 중단. 단순 통계적 우수성만 있으면 `EDA_ONLY`; P1/P3에서 OOS 개선 없으면 Time 바 유지. Event 소스가 바뀌어도 **모델 비교에서 시그널 규칙은 그대로** 둔다.

## 5. R02 — 동일 가격충격으로 Trend/MR 후보 생성 (기술적 지표는 기준선)

### 5.1. 완료 Event Bar의 공통 충격·부호

시각 `k`에서 완료된 Event Bar의 **지수** 종가 `C_k`, 로그수익률 `r_k=log(C_k/C_{k-1})`를 사용한다. overnight gap이나 결측을 건너뛴 수익률은 계산하지 않는다.

\[
R_k^{(3)}=r_k+r_{k-1}+r_{k-2},\quad
Z_k=\frac{R_k^{(3)}}{\sqrt3\,\widehat\sigma_{\mathrm{pre},20}}
\]

- `σ_pre,20`: 충격 윈도우 이전의 **완료 20개 Event Bar** 수익률 표준편차. 최근 충격 3개 바는 표준편차 추정에 포함하지 않는다. 고정 표준편차가 0/결측이면 후보 생성 금지.
- `|Z_k| ≥ 1`인 경우에만 **하나의 가격충격 사건 ID**를 만들며, 그 사건을 다음 두 방향으로 가상 평가한다:
  - `T`: `sign(Z_k)` 방향 ETF 매수. 정보의 지연 가격 반영에 따른 지속 가설.
  - `R`: `-sign(Z_k)` 방향 ETF 매수. 일시적 가격 압력의 복원 가설.
- `Z=1`은 사전설정 효과크기 게이트일 뿐 정규이론의 유의수준 또는 한국시장에서 최적인 값은 아니다.
- 최소 20+3개 유효 과거 이벤트가 필요. 서로 다른 거래일의 유효 Event 수익률을 이어 붙여 sigma를 추정할 수 있지만, **장중 갭은 넣지 않는다**. 충격 `R_k^(3)`은 같은 거래일의 연속 3개 이벤트에서만 계산한다.
- 신호 확정 후 다음 실행 가능한 ETF 1분봉 시가를 진입가로 사용한다. 신호 발생시각에 ETF 진입가를 이미 안다고 가정하지 않는다.
- 같은 가격충격에서 `T`, `R`의 두 잠재 성과는 서로 반대 경제적 가설을 테스트하며, 하나의 실계좌에서 두 방향을 동시에 체결하지 않는다.

### 5.2. **사전에 고정한 조건은 거래활동 강도 한 축**

Event Bar는 대부분 비슷한 누적 volume/거래대금으로 끝나므로, 정보 도착 강도의 대리변수는 **Event Duration**이 더 적합하다.

\[
I_k=\frac{A_k}{\max(\Delta t_k,1\text{ minute})}
\]

- `A_k`: 이번 바의 실제 지수 volume 또는 ETF 원화 거래대금 근사. `Δt_k`: 해당 바가 형성된 실제 분 수.
- **High Activity**: 현재 event 강도 `I_k`가 과거 동일 시각대(`09:30~`, `...`) 확정 이벤트의 **20거래일 중앙값**보다 큼. 그렇지 않으면 Low/Normal. 시간대별 표본이 부족하면 사전에 정의한 전체 장중 과거 중앙값을 사용하고 별도 플래그 기록.
- 시간대 bucket은 예컨대 고정 **30분 단위**, 한정된 거래일 자료에서 20×30분 단위가 충분하지 않다면 `ACTIVITY_UNAVAILABLE`로 채우지 않고 전체 당일-전 이전 완료 이벤트의 rolling median으로 fallback. 현재 이벤트 이전까지의 데이터만 사용하며 당일 미래 완성 이벤트를 포함하지 않는다.
- 1분 Time Bar는 대응 시각대의 `volume / 1min`으로 같은 조건을 구현한다.
- 1차 성공확률 그룹 = `(시장, T/R, ETF 방향, sampling, exit_policy, activity_H/L, 비용 시나리오)`. 너무 작은 그룹을 억지로 합치지 않고 `INSUFFICIENT_EVENTS` 처리. 조건 미적용의 T/R 전체 그룹을 별도 기준선으로 유지.
- 가설: 높은 거래활동은 정보 반영 지속 또는 강제 유동성 수요 어느 쪽에서도 나타날 수 있다. **High→Trend, Low→MR를 강제하지 않는다.** 검증 결과가 조건 분할의 효과를 보이지 않으면 이 한 축도 제거한다.

### 5.3. EMA / RSI는 별도 기술적 기준선

- `T_TECH`: EMA(16,64) 방향(sign)만 사용.
- `R_TECH`: Wilder RSI(14) ≤30이면 +1x, ≥70이면 −1x ETF 매수 후보.
- 이 기준선의 결과는 경제적 가격충격 시그널과 **별도 보고**하고, 기준선 종류를 Beta 추가 여부와 동시에 바꾸지 않는다.
- 기준선에서 가장 높은 Sharpe가 나왔다는 이유로 경제적 가설의 가격충격 규칙을 사후 교체하지 않는다. 새로운 승격은 개발기간 내 별도 등록된 가설로 취급한다.

### 5.4. 사건 비중첩·조건 충돌·계좌 실행

- 연구 후보의 원본 스트림은 **모든 정책에서 동일하게 생성**한다. 적격 이벤트가 진행 중이면 같은 `market×strategy×exit_policy`의 다음 이벤트를 열지 않으며, shadow 후보도 독립 사건으로 증식시키지 않는다.
- 다만 **EXIT 정책마다 실제 보유시간이 다르므로**, 전체 신규 진입 횟수는 달라질 수 있다. 이를 감추지 말고 공통 첫 신호 비교와 실시간 이벤트 캘린더 비교를 별도로 보고한다.
- 전략별·방향별 Beta 그룹은 표본이 희소해질 수 있다. 30완료 사건·10거래일·양방향 결과 최소 기준 충족 전에는 `INSUFFICIENT_EVENTS`로 거래하지 않는다.
- T/R 충돌 시에는 별도 독립 연구가 우선. 결합 포트폴리오는 양쪽 검증 후 별도 연구로 남긴다.
- `09:30~15:00` 신규 후보, 이미 청산이 불가능한 시각에는 신규 진입 금지. 이벤트 신호는 완료된 원본 1분봉 이후에만 사용한다.

## 6. R03 — 청산 3종, 미래 지평, Beta 성공 사건 (사용자 확정 수정)

### 6.1. 동일 `H=3`에서 청산 규칙 3종을 **완전히 분리**

`H`는 **완료되는 미래 Event Bar 수**, `M`은 Beta의 과거 완료 사건 기억 수다. 기본 `H=3`, 민감도 `H=1,5` (개발기간에서만; 사후 최고 Sharpe의 H를 무작정 고르지 않음).

| Exit Policy ID | 계획된 청산 조건 | `H=3`의 의미 | 비교 목적 |
|---|---|---|---|
| `EXIT_60` | 다음 **3개 Event Bar 완료**, **진입 후 60분 도달**, **당일 15:19 실행 가능 시각** 중 첫 시점 | 최대 이벤트 지평 | 체류시간 위험 제한 효과 |
| `EXIT_NO_60` | 다음 **3개 Event Bar 완료** 또는 **당일 15:19** 중 첫 시점 | `60분 cap`만 삭제 | 지연된 바를 기다릴 가치 |
| `EXIT_CLOSE_ONLY` | **당일 15:19 실행 가능 시각**에만 계획 청산 | `H=3`은 신호 정의·평가 참조용이고 **실제 청산에는 사용하지 않음** | 당일 잔여시간 보유효과 |

주의: `EXIT_NO_60`도 **오버나이트하지 않는다**. 사용자 요청의 `시간제한 삭제`는 **60분 상한 삭제**이지 일말 강제 청산 삭제가 아니다. `EXIT_CLOSE_ONLY` 역시 일반적으로 15:30 동시호가 종가가 아니라, 분봉으로 확인 가능한 **연속매매 마지막 15:19분 bar-start open**에 체결한다는 보수적 규약이다. 데이터가 15:19 open을 제공하지 않으면 미리 정한 마지막 유효 접속매매 시가를 사용하고 그 시각/결측 원인을 모두 보고한다. **장마감 동시호가 15:30 종가**로 실제 청산하려면 별도의 체결 경로·동시호가 가격 데이터가 있어야 하므로 이번 기본 실험과 혼합하지 않는다.

- Event 완료 시각이 `t_event`, 진입 `t_in > t_event`인 최초 ETF 원본 1분봉 open.
- H번째 이벤트 종료 시각 `t_H` 이후 최초 ETF 원본 1분봉 open을 청산가로 이용한다. 60분 상한은 **진입시각+60분 이후** 처음 체결 가능한 1분봉 open이다.
- `EXIT_CLOSE_ONLY`는 진입 직후 다음 신호 발생이나 중간 H 완료와 무관하게 예정된 EOD 접속매매 마지막 open에 청산한다. 중간 신호로 탈출하는 규칙은 없음.
- 일일 위험 kill-switch·데이터 이상으로 인한 강제청산은 **세 정책 모두 공통**이며 별도 `exit_reason`에 기록한다. 연구의 계획 청산과 섞지 않는다.
- EOD 실행가가 없는 날은 가짜 종가를 만들지 않는다. `CENSORED_EXEC`로 기록하고 거래가능성·결측률을 보고한다. 청산 불가능한 후보를 사후 P&L이 나쁘다는 이유로 제외하지 않는다.

### 6.2. **성공은 ETF 비용 차감 순수익률이 플러스인 경우**

각 `(시장, 전략, sampling, ETF, Exit Policy, roundtrip_cost)` 조합에 대해 그 정책의 실제 다음 시가 진입·청산 손익을 만든다.

\[
R_{i}^{gross}=\frac{P^{ETF}_{out,i}}{P^{ETF}_{in,i}}-1,
\quad R_i^{net}=R_i^{gross}-C_i,
\quad Y_i=\mathbf 1(R_i^{net}>0).
\]

- Beta는 **`P(해당 Exit Policy의 net ETF 수익률 >0 | 사전 정의된 조건)`**를 추정한다. 방향만 맞춘 경우나 이벤트 bar마다 1회를 세는 방식이 아니다.
- ETF가 +1x 또는 −1x **매수**라는 뜻이므로 손익에는 ETF 자체의 실제 open 가격을 사용한다. 지수 `-수익률`로 인버스 ETF 수익을 만들지 않는다.
- `G_net = mean(R_net | Y=1)`, `L_net=-mean(R_net | Y=0)`. `R_net=0`은 실패(손실크기 0).
- **비용 차감 성공 라벨을 사용하므로** 아래 기대수익 계산에서 비용 `C`를 다시 차감하지 않는다:

\[
EV_{net}=pG_{net}-(1-p)L_{net},\quad
p_{BE}=\frac{L_{net}}{G_{net}+L_{net}},\quad
EV_{cons}=p_{0.05}G_{net}-(1-p_{0.05})L_{net}.
\]

- `G+L=0`이거나 승/패 표본 부족이면 `EV_UNAVAILABLE`; 확률 하한만 확정했다고 손익폭의 신뢰도가 확보된 것은 아니므로 개발 OOS 날짜 블록 리샘플로 `G/L` 민감도도 기록.
- 비용 시나리오 0/2/5/10 bp **각각** 다른 `Y`, Beta 카운트, `G/L`, `pBE`를 만들 수 있다. 5bp를 기본 표로 보고하고 나머지는 민감도. 비용을 고정한 상태에서 정책만 비교한다.
- `EXIT_CLOSE_ONLY`는 당일 남은 시간이 얼마냐에 따라 보유 지평이 달라진다. 동일 3바 성공확률로 주문 크기를 정하면 **레이블 불일치**다. 따라서 **청산 정책별 별도 Beta**가 필수다.

### 6.3. 정책별 후보의 비교 가능성 (필수 2개의 표)

**A. Matched-entry exit tournament:** 시장×전략×거래일마다 최초 적격 충격 이벤트 1개를 채택해 **세 청산 정책을 완전히 같은 ETF 진입가**에서 가상 체결하고, 청산 시점·손익·Beta 대상 이벤트를 paired 비교한다. 첫 후보 기준이므로 표본 손실은 있지만 **청산 효과의 원인 분리**에 필수다.

**B. Continuous policy simulation:** 각 Exit Policy별로 하나의 포지션만 개방하며 청산 이후 발생한 새로운 충격은 거래 가능하게 처리한다. 이때 정책마다 하루 신호 수·보유시간·익스포저가 달라지는 것은 **전략의 일부**다. 포트폴리오 성과와 turnover을 별도로 비교한다.

- 같은 정책/전략의 중복된 3바 평가 기간을 동시에 열지 않는다. 단 T/R 반대 방향 가상 결과는 같은 가격충격을 공유하므로 T와 R 승패를 **독립 관측 두 개로 합산하지 않는다**.
- Guard나 Beta가 후보를 차단해도 해당 후보의 가상 청산 손익은 maturity 시각에 공개하여 **같은 정책 키의** 베이지안 집계에 반영한다.
- 계획된 3개 바 완성이 다음 날로 넘어가면 안 되며, 장말에 해당 정책대로 강제 종료한다.

## 7. R04 — Beta–Bernoulli 확률 추정, 정적 기준선, 비중 분리

### 7.1. 조건부 집단·충분통계량

- `beta_key=(market,strategy_T_or_R,ETF_side,sampling_mode,activity_H_or_L,exit_policy,cost_scenario)`를 기본 키로 하며, 표본이 충분하지 않으면 **미승인**. 시장·방향·청산·비용의 성공사건은 혼합하지 않는다. 별도로 activity 미분할 기준 집단도 평가한다.
- 사전 `Beta(0.5,0.5)`; 완료된 **최근 100개 적격 shadow 사건**에서 성공 `W`, 실패 `L`을 집계한다:

\[
p_k\mid\mathcal E_k\sim\operatorname{Beta}(W_k+0.5,L_k+0.5).
\]

- 확률 해석: *동일 조건에서 관측한 최근 사건들의 국소 성공확률*. 금융시장의 비정상성과 거래일 단위 종속성을 제거해 주는 마법의 추정기는 아니다. 날짜 블록 교정과 Calibration 점검 필수.
- 활성화 최소 30개 완료 사건·10거래일·승패 각 5개. `EXIT_CLOSE_ONLY`는 하루 표본이 적어 조건 분리 후 미달할 수 있으며, 이를 임의로 보정해 성과를 만들어내지 않는다.
- **시점 규칙:** candidate 신호 시점에 `Y`는 미래값이다. **ETF exit open을 관측해 비용 차감 결과가 확정된 시각 이후부터만** Beta 집계에 넣는다. `PAUSED` 중에도 shadow 결과는 같은 규칙에 따라 완료되면 반영한다.

### 7.2. Beta 진입 게이트

\[
p_{mean}=\frac{W+0.5}{W+L+1},\quad
p_{05}=\operatorname{BetaPPF}(0.05,W+0.5,L+0.5).
\]

동일 그룹에서 과거 완료 사건으로 추정한 `G_net,L_net`을 사용:

\[
EV_{cons}=p_{05}G_{net}-(1-p_{05})L_{net}.
\]

- `EV_cons>0`인 경우만 Beta 필터 PASS. `p05>pBE`와 동치(단 G_net+L_net>0). W1 경보는 Beta 추정치를 정확하게 검증하지 않으며, 단지 분포 이동에 대한 별도 위험 신호.
- `G_net/L_net` 추정오차는 개발 OOS 블록 부트스트랩 및 다양한 가격충격/거래비용 민감도에서 감사. 베팅 크기 확대는 이 근거가 없으면 금지.
- `Beta posterior mean`은 Brier/Calibration에서의 확률 예측값이고 `p05`는 **위험회피적 의사결정 통계량**이다. 두 값을 혼동하지 않는다.
- 정적인 동일 전략 `P_STATIC`: 해당 fold의 과거 TRAIN에서만 산출한 **고정 빈도·고정 G/L**을 평가 기간 전체에 적용. `P1`은 계속 완료되는 사건으로 갱신한다. `P1−P_STATIC`이 온라인 업데이트의 순수 가치를 보여준다.

### 7.3. Beta 확률을 **비중**에 쓰는 2단계 후속 실험

초기 `P0~P3`은 **계좌 현금 10% 고정**(필터 불통과 시 0%)이다. Beta 사이징을 동시에 켜면 '확률 필터' 효과와 '노출 변화' 효과가 뒤섞이므로 **최초 성능 검정 후에만** `P4`를 허용한다.

단순 사전등록 2단계 크기:

\[
w^{Beta}_k=
\begin{cases}
0, & EV_{cons}\le0\\
0.05, & p_{05}>p_{BE}\ \text{and}\ p_{05}<p_{BE}+0.05\\
0.10, & p_{05}\ge p_{BE}+0.05
\end{cases}
\]

- `pBE+5%p`는 **확률 추정 마진 연구값**이다. 표본이 부족해 불안정하면 사이징 자체를 기각. 단, 경계가 `>1`이라도 정책은 최고 5% tier까지만 발생할 수 있으며 이를 억지 조정하지 않는다.
- 고정 10% 통과 포지션에 대한 `P1` 대비 `P4`에서 **Brier를 개선하지 못해도 위험조정 성과가 개선**될 수는 있다. 그러나 성과의 원인은 고정 비중과 다른 익스포저뿐일 수 있으므로 **같은 평균 시장 노출·같은 총 위험예산**으로 재비교한다.
- Kelly `f=μ/σ²`를 이산 베팅 확률에 기계적으로 대입하지 않는다. ETF 레버리지 배수를 Beta 승률만으로 바로 높이지 않는다.
- `P4/P5/P6`의 **10%는 최대 현금 비중**이고, +2x ETF를 10% 사면 기초지수 등가 익스포저는 약 20%다. ETF 배수 실험에서는 +1x 10% 현금과 +2x **5% 현금**을 비교해 목표 지수 익스포저를 우선 통제한다. 실제 ETF 가격 경로·수수료·괴리·거래가능성은 별도 확인한다.

### 7.4. 단위·중복 주의

**Beta 1회는 `H` 이벤트가 아니라 ETF 매매 1건의 완료된 결과**다. `EXIT_60`, `EXIT_NO_60`, `EXIT_CLOSE_ONLY`의 결과는 같은 진입에서도 다를 수 있으므로 하나의 Beta로 합치지 않는다. 서로 다른 비용 시나리오도 같은 Beta 카운트로 가정하지 않는다.

## 8. R05 — W1 Distribution-Shift Guard 및 Hysteresis

### 8.1. W1의 의미와 입력

`W1`은 과거 성공확률이 **반드시** 변했다는 증거가 아니라, **최근 수익률 분포와 직전 참조 분포 사이의 차이**다. 가격충격·변동성 확대·평균 위치 변화 등 다양한 변화에 반응하며, Beta의 conditional P(win) 안정성을 직접 검정하는 것은 아니다.

Event `k`의 무중첩 수익률 구간:

\[
R_k=\{r_{k-29},\ldots,r_k\},\quad
B_k=\{r_{k-149},\ldots,r_{k-30}\},
\quad D_k=\frac{W_1(R_k,B_k)}{\max(\mathrm{sd}(B_k),\epsilon)}.
\]

- 가격은 **같은 Event 종료시각에서 확인 가능한 지수 종가**. ETF Won Bar 생성 시 거래량 임계값은 ETF로 계산하지만 W1의 수익률은 해당 시점의 **지수 수익률**로 계산해 예측목표와 일치시킨다.
- 장중 event close-to-close 수익률은 **같은 거래일 안에서만** 계산. 전일 마지막 Event close→당일 첫 Event close 구간은 W1 window에서 제외하고, 유효한 과거 거래일의 내부 바 수익률만 연결하여 최근·기준 표본을 구성한다.
- `sd(B)`로 나누어 시장별/시기별 스케일을 표준화하되, 이는 변동성 변화에 대한 민감도를 없애는 것이 아니다. `raw W1`, `reference volatility`, `event duration`도 모두 저장한다. duration 기반 `r/sqrt(duration)` 결과는 후속 감도분석.
- 최근·기준 30/120개 모두 성숙할 때까지 `W1_UNAVAILABLE`: 거래 Guard가 기준선을 막아버리는 불공정한 비교를 피하기 위해 P3은 동일 공통 워밍업 이후부터 P0–P2와 비교하는 **공통 평가 구간**을 사용한다.

### 8.2. 경보 임계값은 어떻게 수치화하는가?

**보편적인 W1=0.3이나 p=0.05 같은 숫자는 없다.** 분포의 스케일·윈도우 길이·자기상관·이벤트 빈도에 따라 결정된다.

초기 연구용 기본 교정 절차:

1. **개발 구간의 초기 TRAIN(예: 2022-11~2023-12)**에서 매 이벤트에 W1 점수를 계산한다. TRAIN 데이터의 각 거래일에서 **하루 W1 최댓값** `M_d=max(D_k in day)`를 기록한다.
2. `alarm_threshold=Q_0.90({M_d})`를 정한다. `release_threshold=Q_0.75({D_k})`는 이벤트 단위 분포에서 정한다. 두 임계값 모두 TRAIN 이후 **학습/선정 대상 fold의 시작 전에만 fit**한다. 2026 도입 전에는 2025년까지의 개발기간에서 확정해 **고정**한다.
3. 이런 교정은 개발 표본에서 일별 최고경보가 약 10%의 날짜에서 넘는 관찰을 유도할 수 있지만, **실제 안정된 시장에서의 참 오경보율(FPR)** 을 보증하지 않는다. W1이 구조 변화를 탐지했는지를 알려줄 '참의 null'이 없기 때문이다.
4. W1 알람 수/일, 알람 발생 거래일 비율, 경보 후 shadow 평균수익과 tail loss, 알람 전후 duration·RV를 보고한다. 원한다면 안정 구간 또는 stationary synthetic/블록부트스트랩에서 **실제 시퀀스 단위** 경보 빈도를 보정한다. 개별 W1 95백분위와 일별 오경보율 5%를 동일시하지 않는다.
5. 임계값 후보가 필요하면 Validation에서 사전등록된 `Q0.80/Q0.90/Q0.95` **일별 최대치** 세 개까지만 비교하고, 최종에는 하나를 동결한다. 계속 경보가 발생하도록 임의적으로 낮추지 않는다.

### 8.3. Guard State Machine (단순 위험관리용)

`NORMAL → PAUSED → PROBATION → NORMAL`.

- **NORMAL:** W1 `D_k >= alarm_threshold`인 완료 이벤트 발생 → 즉시 다음 신규 진입부터 `PAUSED`. 이미 보유 중인 포지션은 사전 정의된 만기/위험규칙에 따라 청산하며 알람 발생 순간 체결됐다고 가정하지 않는다.
- **PAUSED:** 최소 3개 완성 이벤트 동안 신규 진입 금지. `D_k <= release_threshold`가 2개 연속 확인되면 `PROBATION`. 2개 연속은 바의 완료 이벤트 기준이며 실제 시간은 기록한다.
- **PROBATION:** 신규 진입을 계속 막되 모든 T/R 후보를 shadow 기록한다. 안정화 이후 최소 10개 **완료** 후보·3개 거래일을 관측하고, W1이 다시 경보를 넘지 않으며 현재 rolling Beta의 `EV_cons>0`일 때 `NORMAL` 재개. 조건을 만족하지 못하면 중단 유지. Shadow의 성과를 2026 최종 테스트에서 보고 재개 임계값을 바꾸면 안 된다.
- **재경보:** `PROBATION`에서 W1이 다시 알람 초과하면 `PAUSED`로 재진입, 확인 카운터 초기화. `NORMAL`에서도 신호가 없으면 거래하지 않음.
- W1이 오랜 기간 높거나 새 정상분포가 형성돼도 이 Guard는 오래 멈출 수 있다. 그 자체가 전략의 실패 가능성으로 계량한다(`time_paused_pct`, `missed_profit`, `avoided_loss`). 이전 분포가 '안정적으로 회귀'했다는 보장이 없으므로 **분포가 안정됐다는 이유만으로 Beta 카운트를 자동 초기화하지 않는다**.
- W1 guard를 사용하지 않는 P0/P1도 동일하게 shadow outcome을 계속 기록한다. **모든 정책의 Beta 시계열은 동일해야 한다**. 그래야 경보가 줄인 거래를 인과적으로 비교할 수 있다.

### 8.4. 직접 비교해야 할 정책

- `P1_BETA` vs `P3_BETA_W1`: W1만의 순수 거래 선택 효과.
- `P2_BETA_RV` vs `P3_BETA_W1`: W1이 단순 변동성 이상을 설명하는지.
- `P0_FIXED` vs `P3_BETA_W1`: 전체 제안 구조의 최종 효용.
- `P4_RESET`은 **후속/선택적**: W1을 장기적인 분포 재편 신호로 보고 Beta를 Jeffreys prior로 리셋하는 정책. 위 P3에 섞어 한 번에 검증하면 중단 효과와 리셋 효과를 구분할 수 없으므로 **1차 토너먼트에서 제외**.

### 8.5. W1을 연속적인 신뢰도나 직접 레버리지 공식으로 사용하지 않는 이유

W1은 **분포 간 거리**이지 `P(trade wins)` 또는 `VaR`가 아니다. `size = 1 / W1`이나 `leverage ∝ 1−W1` 같은 공식에는 일반적으로 정당화되는 확률적 해석이 없다. 따라서 초기 비교는 `P3` **Hard Guard(0 또는 고정 10%)**다.

후속 `P6_BETA_W1_SOFT`는 비교 가능한 단순 단계형 실험이다.

- `D_k < Q75_event`: Beta PASS 시 고정 **10%** 현금 비중.
- `Q75_event ≤ D_k < Q90_daily_max`: Beta PASS 시 **5%**로 축소.
- `D_k ≥ Q90_daily_max`: **신규 진입 0%**, 기존 PAUSED/PROBATION 재진입 규칙 유지.
- 3개 상태에 사용한 문턱은 사전 TRAIN에서만 설정한다. Soft가 Hard보다 낫다는 증거 없으면 유지하지 않는다.
- `P5`: Beta 0/5/10%와 W1 Hard를 결합하며, `P4`와의 차이만 W1 효과다. `P6`: Beta **이진 필터·고정 상한 10%** + W1 soft이고, `P3`와 비교해 부분 축소의 효과를 측정한다. 두 사이징을 한꺼번에 합친 복잡한 `P7`은 먼저 각각 통과하기 전에는 만들지 않는다.

Guard는 Beta가 진짜로 맞는지 확인하는 검정이 아니다. 별도로 posterior probability calibration 및 `W1 high/low` 그룹의 Beta 예측 잔차·실제 승률 차이를 평가한다.

## 9. R06 — RV 경보 대조군

- W1과 똑같은 `PAUSED/PROBATION` 상태 처리 구조를 사용한다.
- 차이점은 경보 점수만 이전 완료 Event Bar `30개`의 제곱수익률 평균 `RVbar_k=(1/30)sum(r_i²)`로 교체한다.
- Threshold 교정은 **TRAIN 일별 최대 RVbar**의 90백분위, 해제는 TRAIN 이벤트별 RVbar 75백분위. 동일한 집계·동일한 최소 중단/복귀 규칙.
- RV와 W1은 동일한 알람 빈도일 필요는 없으므로 `time_paused_pct`, `signal_coverage`, `avoided_loss`, `missed_profit`을 보고하고, 동일 중단비율의 후속 민감도도 비교한다.
- **판정:** W1이 P2보다 실질적 이점이 없으면 더 단순한 RV Guard 또는 No Guard를 선호.

## 10. R07 — Walk-forward 및 확률 모델 평가

### 10.1. 개발 데이터 내부 split

- 2022-11~2023-12: 초기 TRAIN·데이터 진단·전처리/threshold fit.
- 2024-01~2025-12: **8개 분기 rolling/expanding-origin** DEV 검증. 매 분기 평가 시작 전에 그때까지 실제 확정된 outcome만으로 Beta rolling100 카운트, W1 기준 및 G/L을 업데이트할 수 있다. 
- fold별 성과·확률의 통계 신뢰구간을 날짜 기준 stationary/moving block bootstrap(고정 seed 42, 최소 1,000 반복)으로 평가. 청산정책 3종 중 가장 긴 당일 보유시간, 장내 미래 라벨 중첩 및 거래일 경계에 맞춰 purge 처리.
- 매 분기의 데이터가 다음 분기의 TRAIN에 들어가는 expanding 검증은 허용한다. 그러나 성과를 확인한 뒤 신호 임계값을 해당 분기에서 다시 선택·수정하면 별도 experiment ID로 등록.
- 2025-12-31에 signal, sampling, H, exit_policy, Beta prior/M/min_n, W1 thresholds/pause, Beta/W1 sizing tiers, cost/risk policy를 모두 **LOCK**. 2026년은 최종 단 한 번 실행.

### 10.2. 확률 측정: 반드시 정량 보고

- 모든 잠재 후보 i의 진입 시점에 `p_mean`, `p_05`, `p_BE`, `EV_point`, `EV_conservative`를 기록하고, 뒤늦게 확정된 `y_i`와 비교.
- 지표: Brier `mean((p_i-y_i)^2)`, Log Loss(확률을 `eps`로 clip), reliability bins, 시장·전략·방향별 calibration, hit/miss 평균 gross 수익, net EV.
- 기준선: (i) 이전 모든 완료 사건의 global 빈도, (ii) 직전 100개 빈도(Jeffreys 없는 평활 기준), (iii) Beta 평균, (iv) Beta 하위 분위수. **하위 분위수는 보수적 진입 기준이지 calibration용 예측확률 자체는 아니다.**
- Beta와 빈도 기반 baseline은 모두 완료 사건만 사용. 실현 가능성 없는 `이전 n-bar 미래 사건 적중`을 조기에 Beta에 반영하지 않는다.
- 희소 표본일 때 매매하지 못했다는 사실과 평가 사건 자체가 부족하다는 사실을 분리한다. `30개 완료 기회·10거래일` 기준을 만족하지 못하는 그룹은 `INCONCLUSIVE`.
- 복수 후보 비교 편향, 동일 세션 의존성과 중첩된 시장상황을 고려해 paired daily block bootstrap 신뢰구간과 분기별 부호 일관성을 보고한다.

### 10.3. 소수의 주 결과만 사전 확정

1. **경제적 1차 지표:** 순수익/거래당 net EV 및 일별 Sharpe, MaxDD, ES. 진입을 줄인 정책의 총수익만 비교하지 말고 **동일 위험 예산 및 거래시간 노출**도 같이 보고.
2. **확률 1차 지표:** 각 exit_policy별 Beta 평균의 OOS Brier vs 단순 rolling 성공빈도·고정 TRAIN 빈도. Lower bound를 사용한 거래 중단은 경제적 지표로 별도 평가.
3. **W1 1차 지표:** P3−P1의 **위험 예산 대비** 순손익 개선, 추가로 P3−P2(RV Guard), 알람 후 실제 전략 기대값, 경보 빈도, 놓친 이익, ES.
4. **Exit 비교:** matched 최초 후보 기준 정책 간 paired net P&L, 지속시간/실제 바 개수, EOD-only의 장말 편향.
5. **Sizing:** P4−P1(Beta 사이징), P5−P4(W1 Hard 추가), P6−P3(Soft vs Hard). 평균 익스포저·목표 변동성·거래횟수 통제 후에도 경제적 이점이 있는지 확인.

임의의 `Sharpe≥1`만 통과기준으로 삼지 않는다. 목표 Sharpe≥1을 참고하되, **OOS 개선의 부호와 불확실성**이 더 중요하다. 개발 구간의 실패는 재탐색·개발 결과이지 2026 OOS를 열어 다시 최적화할 근거가 아니다.

## 11. R08 — ETF 백테스트: 모든 정책 동일 실행 시뮬레이터

### 11.1. 체결·청산·레버리지 실험 규약

- **진입:** Event Bar 또는 완료된 Time Bar 확정 `available_at` 이후 ETF의 **다음 실제 1분봉 bar-start open**. 동일 완성 바 종가 체결 금지.
- **EXIT_60:** `3개 미래 Event 완료` / `진입+60분` / `15:19 ETF open` 중 먼저 도달하는 실행시각에 청산.
- **EXIT_NO_60:** `3개 미래 Event 완료` / `15:19 ETF open` 중 먼저 도달하는 실행시각에 청산.
- **EXIT_CLOSE_ONLY:** 다른 신호와 3바 완료에 관계없이 `15:19 ETF open`에 계획 청산. 오버나이트·동시호가 체결 가정 없음.
- 데이터 공급자의 bar-start/bar-end 규약을 통일하고 15:19 bar-start open이 실제 존재하는지 먼저 검증한다. 15:20–15:30 종가 단일가와 15:19 접속매매 시가는 다른 가격이다. KRX 기준 ETF 접속매매는 09:00~15:20, 종가 단일가는 15:20~15:30.
- ETF의 유효한 실측 OHLCV로 gross/net P&L을 계산한다. `+2x ETF 수익률 = 지수 수익률×2` 등으로 대체하지 않는다.
- 최초 `P0~P3` **현금 10%** 고정, +1x/−1x ETF만 사용. 모든 정책은 비용 0/2/5/10bp의 **동일** 시나리오로 비교.
- 후속 `P4~P6`는 최대 10% 현금 내 0/5/10% tier만 검증. 별도 **ETF 자체 배수 +2x/−2x**의 위험조정/등가노출 실험은 우선 모델 승격 후에만 시행.
- 모든 비교에서 cash weight, underlying-equivalent exposure(ETF 현금 비중×절대배수), 일별 realized volatility, 위험 예산, 거래시간 노출 및 turnover을 별도로 기록한다.

### 11.2. 필수 손익·리스크 지표

`net_total_return, annualized_daily_sharpe, max_drawdown, daily_VaR99, daily_ES99, gross_and_net_EV_bp, trade_count, trades_per_day, win_rate_net, payoff_ratio_net, profit_factor, turnover, exposure_time_pct, mean_underlying_equivalent_exposure, days_active, long_inverse_breakdown, strategy_breakdown, exit_policy_breakdown, matched_exit_pnl, cost_sensitivity, guarded_missed_profit, guarded_avoided_loss, paused_pct, beta_sizing_tradeoff, posterior_calibration`.

- Daily Sharpe는 일별 계좌 수익률로 계산한다. 매매당 샤프를 일별로 연율화하지 않는다.
- VaR/ES는 꼬리표본 수와 CI를 기록하며, 비정규 분포를 기본으로 가정한다.
- `P3`의 MDD가 낮아졌더라도 단순 거래기회 축소 효과일 수 있으므로 동일 리스크 예산/거래 시간 조건 비교를 병행.
- 2026 OOS에서도 개별 요소를 다시 선택하지 않는다.

### 11.3. Risk Gate

- daily account loss ≥ 1% 도달 시 신규 진입 금지; 만기와 시간종료 강제청산은 유지.
- 계좌 위험 0.2%/거래 99% VaR 예산 및 instrument 최대 25% 상한. VaR 추정에 필요한 완료 사건이 없으면 10% 고정 연구 시뮬레이션과 별도로 실거래 승격 `BLOCKED`.
- 결측 데이터, 바/ETF 미정렬, 데이터 지연, 후보 방향 충돌, 15:00 이후 신규 신호는 신규 진입 금지.
- W1 Guard는 **신규 주문 중단**을 기본으로 하며, 기존 포지션을 시가 외 임의 가격에 즉시 청산했다고 간주하지 않는다.

## 12. R09 — 2026 True OOS 1회와 최종 판정

- 2025년 말까지 선정한 **시장·샘플링·가격충격 T/R 시그널·Activity 기준·H·청산정책 3종 중 최종 선택·Beta 메모리·W1 경보·비중 tier·확률 및 위험 정책**을 immutable config/hash로 저장.
- 2026년을 처음 읽은 이후에는 OOS 결과를 보고 숫자를 수정하지 않는다. 2026년 관측이 도착할 때, 이미 확정된 거래 결과로 **사전에 정의된 rolling100 Beta 집계가 온라인 갱신되는 것**은 허용. 모델/임계값 재최적화는 금지.
- 분석 보고는 ① 바 샘플링 ② 충격 T/R 신호 ③ 세 청산정책 ④ Beta 증분 ⑤ W1 vs RV/No Guard ⑥ 비중/ETF 배수 증분 ⑦ 비용·기간 강건성으로 분리한다.
- 최종 상태: `DATA_BLOCKED`, `BASE_SIGNAL_REJECTED`, `BETA_NOT_USEFUL`, `W1_NOT_USEFUL`, `STATISTICALLY_SUPPORTED`, `PAPER_ELIGIBLE`, `OOS_FAILED` 중 구체적인 근거와 함께 선택.

## 13. 연구 게이트: 가설·청산·Beta·W1·레버리지의 증분 효과

| 단계 | 실험 목적 | 핵심 보고 | 실패 시 |
|---|---|---|---|
| G00 | 데이터·시간·바 임계값 인과성 | 전일 고정 threshold, 1분 초과율, 누출검사 | 데이터 오류 해결 전 중단 |
| G01 | Time / Volume / Won Bar 비교 | 실제 바 수/일, duration, 첨도, OOS Brier·EV | 통계·경제 우위 없으면 Time 유지 |
| G02 | 경제적 가격충격 Trend/MR | 순EV/손익비·거래빈도·단일 날짜 집중도 | 기초 알파 없으면 해당 전략 기각 |
| G03 | 청산 3종 | 같은 진입 matched P&L, 총 보유시간·EOD 편향 | 기본 `EXIT_60` 유지 또는 모두 기각 |
| G04 | 조건부 Activity 분리 | High/Low의 차이·추정 안정성 | 분할 제거 |
| G05 | Beta 추가 | P1 vs P0, P1 vs TRAIN 정적확률, Calibration | Beta 제거 |
| G06 | W1/RV Guard | P3−P1, P3−P2, 회피손실·놓친이익 | W1 제거 |
| G07 | 확률/분포 비중 조절 | P4−P1, P5−P4, P6−P3; 노출 동등화 | 고정 10% 유지 |
| G08 | +2x/−2x ETF | **같은 지수 등가노출** 비교, 추적오차·수수료·위험 | 1x ETF만 |
| G09 | 2026 True OOS | 모든 설정 LOCK 1회 평가 | 실거래 승격 중지 |

1차 실험에서 가장 단순한 통과 모델을 우선한다. 청산 3종을 평가한다고 2026에서 세 후보를 다 돌려 수익이 좋은 버전을 고르는 것은 **True OOS 선별 편향**이다. 2025년 개발기간에서 하나의 모델·정책을 선정해 2026년 검증은 1회로 봉인한다.

## 14. 기존 레포 최소 변경 구현 계약 (파일/폴더 신설 강제 없음)

기존 코드를 재사용하고 최소한 아래 인터페이스만 추가·수정한다. 기존 레포의 함수명이 다르면 대응시키면 된다. 별도 프로젝트 아키텍처는 설계하지 않는다.

```python
set_preopen_threshold(prior_20_daily_totals, target_bars_per_day=50)
build_minute_event_bars(minute_ohlcv, metric, fixed_threshold_today)
make_shock_signals(index_event_close, shock_bars=3, sigma_pre_bars=20, z_abs_min=1.0)
classify_activity(event_volume_or_won, event_duration, past_activity_reference)
make_candidate_events(signals, etf_minute_ohlcv, H=3, exit_policy='EXIT_60')
update_beta_from_matured_events(candidate_log, asof_ts, group_key, last_M=100, prior=(0.5, 0.5))
score_w1(index_event_returns, recent=30, reference=120)
update_w1_guard_state(previous_state, score, thresholds, completed_shadow_events)
choose_cash_weight(beta_p05, p_break_even, w1_guard_state, policy_id)
backtest_policy(candidate_log, beta_snapshot, guard_state, policy_id, costs, risk_limits)
compare_exit_policies(matched_first_entry_log, continuous_policy_runs)
compare_walk_forward(policy_runs, folds, daily_block_bootstrap=1000)
```

### 14.1. 결과 로그: 최소 공통 필드

`run_id, git_sha, data_hash, config_hash, market, sampling_mode, source_metric, threshold_set_at, threshold_value, threshold_history_dates, actual_bars_today, event_id, event_end_ts, available_at_ts, event_duration_min, overshoot_ratio, activity_score, activity_group, shock_R3, shock_pre_sigma20, shock_Z, candidate_id, strategy_T_or_R, side, etf_ticker, etf_leverage, entry_ts, entry_open, exit_policy, H, max_hold_minutes, exit_trigger_ts, exit_ts, exit_open, exit_reason, gross_return, cost_scenario_bp, net_return, y_net_positive, y_available_at, beta_group_key, beta_n, beta_w, beta_l, beta_p_mean, beta_p05, G_net, L_net, p_BE, EV_cons_net, W1_score, W1_alarm_q90_dailymax, W1_release_q75_event, W1_state, RV_state, policy_id, cash_weight, equivalent_underlying_exposure, decision_reason, model_locked_date, train_end_date`.

`y_net_positive`와 미래 청산 가격·수익률은 as-of 의사결정 입력에 절대 들어가면 안 된다. 고정 진입시간의 Exit 3종 paired 비교와 각 정책의 독립 재진입 로그를 따로 저장한다.

### 14.2. 필수 테스트

1. **Future perturbation:** 미래 1분봉/당일 이후 거래량을 바꿔도 현재 event threshold/경계/신호/베타/W1/size는 불변.
2. **Preopen threshold:** 전일까지 정상 20일 총량만 사용, 오늘 실제 일총량/최종 바 수로 재계산 금지. 50개 목표라도 50 미만/초과 결과를 허용.
3. **Whole-minute stopping:** 최초 임계 초과 시 원본 1분봉 전체 포함·초과분 이월 없음·1분 1바 이하·일말 잔여량 무신호.
4. **Shock source:** Z의 σ에 충격 최근 3바 미포함, Overnight gap 제외, 23바 미성숙/결측 무신호.
5. **Exit exactness:** EXIT_60 vs NO_60 vs CLOSE_ONLY의 예정 종료 시각, H 이벤트 지연, 진입+60분, 15:19 시가, EOD 누락, 다음 시가 누출 없음.
6. **Label consistency:** 성공은 `R_net>0`, 비용 시나리오별 다른 라벨, `G_net/L_net` 사용 시 비용 재차감 금지.
7. **Delayed Beta:** 해당 exit_policy의 ETF exit 체결 전 `Y`가 베타 카운트에 미반영, 다른 exit_policy·비용 scenario의 posterior 혼합 금지.
8. **Same-event pairing:** T/R 같은 가격충격은 독립 2거래로 착각 금지; 세 Exit는 첫 일일 후보 진입가 동일.
9. **Guard hysteresis:** q90 daily-max 경보, q75 event 해제, 최소 3바 정지·2바 연속·probation 규칙.
10. **Sizing separation:** P0~P3은 10% 고정, P4는 0/5/10%, P6는 W1 0/5/10%. Beta 확률 또는 W1으로 방향 반전 금지.
11. **ETF leverage equivalence:** 1x 현금 10% vs 2x 현금 5%의 기초지수 명목노출 일치; ETF P&L은 실제 원본 가격.
12. **2026 sealing/reproducibility:** 과거 데이터/설정 hash 동일, 2026 파라미터 선택 접근 금지, 2025 LOCK 그대로 재실행.

## 15. 최종 결정: 바로 연구 시작할 수 있는 사전등록 설정

### 15.1. 기본 후보 및 사전등록 설정

- **샘플링:** 1분 Time Bar 기준선 + 8분 Time Bar, 지수 Volume Bar, ETF 근사 Won Bar. 당일 임계값은 이전 완료 20거래일 중앙값/50으로 **개장 전 고정**. 30 bars/day는 보조 진단만.
- **독립 전략:** 동일 충격 `R3/(sqrt3·sigma_pre20)`의 `|Z|≥1`에서 T는 충격 방향, MR은 반대 방향; EMA16/64·RSI14/30·70은 기술적 대조군.
- **단 하나의 사전 조건:** 동일 시각대 과거 거래활동 강도 중앙값 기준 High vs Low. 표본 부족 시 activity 조건을 제거할 수 있으나 개발 중 미리 규칙 고정.
- **청산 3종:** `EXIT_60 = min(3바,60분,EOD)`, `EXIT_NO_60=min(3바,EOD)`, `EXIT_CLOSE_ONLY=EOD(15:19분 open)`; 모든 기간 중 보유 금지. 민감도 `H=1,5`.
- **Beta:** 매 청산정책×시장×방향×전략×activity×비용 별 `Beta(0.5,0.5)`, 완료 최근 100사건, 최소 30사건/10일/승패 5회, 성공 `ETF net>0`, EV 하한 `p05`.
- **W1:** 최근30/이전120, TRAIN 일별 최대 q90 경보 / Event별 q75 해제, 정지/재개/Hysteresis 유지. RV Guard 대조군.
- **비중:** 최초 10% 현금 고정으로 Beta/W1 순효과 검증. 그 이후만 Beta `0/5/10%` tier와 W1 soft `0/5/10%`의 증분 가치 평가; ETF +2배 사용 여부는 **등가 지수 노출**로 후속 연구.
- **데이터:** 2025년까지 개발·선정, **2026년 봉인 후 True OOS 1회**. 사용자 미확인 실제 ETF·비용·타임스탬프는 자료 검증에서만 BLOCKED 처리.

### 15.2. 사용자 또는 레포에서 사실 확인이 필요한 사항

| 항목 | 확인 전 조치 |
|---|---|
| 실제 KOSPI/KOSDAQ 지수·ETF OHLCV, 거래량 단위, 분봉 timestamp | 자동 스키마 확인 전 대상 시장 사용 금지 |
| 대표 ETF의 정확한 +1/−1/+2/−2 배수 및 기초지수 | 임의 상품매핑 금지 |
| 15:19 ETF bar-start open이 실제 존재하는지 | 없으면 EOD 청산을 임의 동시호가 종가로 대체하지 않음 |
| 사용자 계좌의 수수료·세금·실제 스프레드 및 시장충격 | 연구 0/2/5/10bp 비교만 허용. 실거래 승인 전 실측 필요 |

그 외 수치는 이번 **연구 기본값으로 확정**되어 있으므로 추가 질문 없이 2025 DEV에서 시작할 수 있다. 통과 못한 모델을 임계값 조작으로 살리는 것은 금지한다.

## 16. 참조 출처·적용 한계

**첨부 BTC PDF/노트북에서 얻은 설계 힌트(이번 연구가 새로 증명하는 내용 아님):**

- 첨부 `확률 기반 크립토 퀀트 트레이딩_최종 (1)(1).pdf`, pp.5–7: 거래대금 임계값의 Event Bar 생성 개념. pp.8–11: W1·JS·KS와 hysteresis. pp.12–17: Beta–Bernoulli 및 EWMAC/RSI 시그널, Kelly/비중. pp.18–19: Beta ON/OFF 성과. **첨부 노트북의 미래 일거래대금 사용, 미래 결과 즉시 Beta 갱신, 포지션 부호 소실 등 감사에서 확인된 오류 때문에 발표 성과를 입증된 OOS 알파로 인용하지 않는다.**
- López de Prado, *Advances in Financial Machine Learning* (2018): Event/volume/dollar bars, triple barrier, time-series 검증의 출발점. https://agorism.dev/book/finance/ml/Marcos%20Lopez%20de%20Prado%20-%20Advances%20in%20Financial%20Machine%20Learning-Wiley%20%282018%29.pdf
- `mlfinpy` Financial Data Structures: 이전 일별 volume/거래대금의 약 1/50을 초기 누적 임계값으로 삼는 예시(실제 bar 수 50 보장 X). 정적/동적 threshold 모두 설명. **틱 기준**과 1분봉 집계 근사는 다름. https://mlfinpy.readthedocs.io/en/latest/FinancialDataStructure.html
- 한국거래소 KRX ETF 접속매매 09:00–15:20, 종가 단일가 15:20–15:30: https://global.krx.co.kr/contents/GLB/06/0602/0602010204/GLB0602010204T9.jsp
- Wilder RSI 기간과 30/70의 관행: https://download.esignal.com/products/workstation/help/charts/studies/rsi.htm
- Beta(0.5,0.5) Jeffreys prior의 표준적인 이항확률 해석·구간: https://pmc.ncbi.nlm.nih.gov/articles/PMC7531056/
- 비정상성 감지의 표본·거짓 경보·신호와 거래 효용 차이: https://www.sciencedirect.com/science/article/pii/S092523122602045X

## 17. 실패 시 폐기/퇴각 기준

- **바 생성 기각:** 당일 미래 총 거래량을 써야만 50개에 맞출 수 있다면 그 실험을 폐기. 이전 20완료일의 **고정 임계값**을 지켜 생성. 이벤트의 분포 특성만 나아졌지만 OOS Beta/ETF EV가 Time 대비 개선되지 않으면 Event Bar 제거.
- **경제적 가설 기각:** 동일 가격충격의 T/MR 중 비용 차감 EV>0이 안정적으로 관측되지 않고, 사전 정의한 Activity 조건도 개선하지 못하면 해당 가설 기각. 이 경우 Beta/W1은 알파를 만들어주지 못한다.
- **청산 비교:** EXIT_CLOSE_ONLY에서만 수익이 나고 장말 잔여시간/시장 방향 노출에 의해 설명된다면 우수한 Event-Bar 예측으로 오인 금지. EXIT_NO_60의 긴 보유로 꼬리손실이 커지면 60분 상한 삭제 기각.
- **Beta 기각:** `P1`이 고정 TRAIN 확률/단순 완료 빈도 및 `P0`의 경제적 성능을 일관되게 개선하지 못하거나 Posterior가 과신되면 Beta 별도 계층 제거.
- **W1 기각:** `P3`가 `P1` 및 RV Guard 대비 순EV·ES에서 실질 개선 없거나 회피손실보다 놓친 이익이 크면 W1 제거.
- **비중 조절 기각:** `P4/P5/P6`가 고정 10%와 **동일한 위험·노출 예산** 비교에서도 우위가 없으면 고정비중 유지. W1 수치 자체로 레버리지 배수를 직접 확대하는 공식은 채택하지 않는다.
- **실행 기각:** ETF next-open 가정·왕복비용 스트레스에서 순EV가 무너지고, 막대한 가격충격·동시호가 종가에만 의존하면 실거래 후보 아님.
- **2026 최종:** 동결한 한 가지 최선 모델의 True OOS가 실패하면 실거래 승격 금지. 2026을 보며 임계값을 바꾸고 다시 True OOS라 부르지 않는다.

**최종 연구 질문:** "**당일을 미리 알지 않고** 만든 Time/Volume/Won Bar에서 같은 가격충격의 지속·복원 가설을 검증할 때, **청산 3종 중 어떤 실현손익 정의가 강건한가**, 그 조건부 성공확률에 Beta를 붙이면 고정 신호보다 EV가 개선되는가, W1 Guard가 위험을 추가로 줄이는가, 그리고 **Beta/W1 기반 포지션 비중 조절**이 고정 10%보다 OOS에서 실제 우수한가?" 각 질문에 별도 Yes/No를 기록하고, 2026 True OOS는 최종 선정 한 모델에 한 번만 사용한다.
