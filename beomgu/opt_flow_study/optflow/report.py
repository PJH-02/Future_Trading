"""결과를 5분위·회귀·연도별·통제·플라시보 순서의 마크다운으로 쓴다."""
import numpy as np
import pandas as pd


def _f(v, d=3):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return f"{v:+.{d}f}" if isinstance(v, (int, float, np.floating)) else str(v)


def _p(v):
    return "—" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.3f}"


def jsonable(obj):
    if isinstance(obj, dict):
        return {k: jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, pd.DataFrame):
        return obj.reset_index().to_dict(orient="records")
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


def _prov_lines(prov):
    return [f"실행 {prov['run_id']} · 코드 {prov['code_version']} (git {prov['git_rev'] or '없음'}) · Python {prov['python']}, "
            f"pandas {prov['pandas']}, numpy {prov['numpy']} · 사전등록 {prov['prereg_version']}",
            f"원자료 `{prov['raw_dir']}` · 마지막 수집 기록: {prov['last_collection_log'] or '없음'} · 파일 해시는 provenance.json", ""]


def md_audit(audit, panel, raw, prov):
    c, k, f, o = audit["calendar"], audit["kodex"], audit["futures_investor"], audit["options"]
    L = ["# 1단계 — 데이터·기간 감사", ""] + _prov_lines(prov) + [
        "| 항목 | 값 |", "|---|---|",
        f"| 달력(K200 지수 거래일) | {c['rows']}일, {c['start']} ~ {c['end']} (FDR < {c['source_cut']} ≤ 네이버) |",
        f"| 지수 시가=고가=저가=종가인 날 | 전체 {c['flat_ohlc_days']}일, 2010년 이후 {c['flat_ohlc_days_since_2010']}일 |",
        f"| KODEX 200 | yfinance 원가격 {k['yfinance_rows']}행, 네이버 보완 {k.get('filled_from_naver', '파일 없음')}일(원가격 수준으로 환산, 첫 겹침일 이전 보완 {k.get('fill_before_first_overlap', 0)}일) · "
        f"분석 구간({k['window_start']}~) 결측 {k['missing_in_window']}일, 거래량 0 {k['zero_volume_days_in_window']}일 · 구간 이전 결측 {k['missing_before_window']}일 |",
        f"| KODEX 겹침 대조(yfinance vs 네이버) | {k.get('overlap_days', '—')}일. 네이버는 분배금 수정주가(원가격/수정가 비율 {k.get('adj_ratio_first_last')}), "
        f"시가→종가 수익률 차이 SD {k.get('overlap_oc_return_diff_bp_sd')}bp, 최대 {k.get('overlap_oc_return_diff_bp_maxabs')}bp |",
        f"| 외인 선물 매수·매도(통제) | {f['rows']}일, {f['start']} ~ {f['end']}, 달력 밖 날짜 {f['dates_not_in_calendar']['count']}일 · "
        f"매도=0 결함 {f['invalid_rows_set_nan']}일 결측 처리 {f['invalid_range']} |"]
    if o["status"] == "loaded":
        L.append(f"| KRX 투자자별 옵션(주 지표) | {o['rows']}일 {o['start']} ~ {o['end']} · 원 라벨 {o['investors_raw']} → 표준 {o['investors_std']} · "
                 f"알 수 없어 뺀 라벨 {o['unknown_investor_labels_dropped'] or '없음'} · 수치 결측 셀 {o['numeric_nan_cells']} · "
                 f"달력 밖 날짜 {o['dates_not_in_calendar']['count']}일 · 기간 내 주 지표 입력 결측 {o['primary_input_missing_days_in_range']}/{o['calendar_days_in_range']}일 |")
    else:
        L.append(f"| KRX 투자자별 옵션(주 지표) | **없음** — `{o['expected_file']}` 필요 |")
    L.append("")
    if k["missing_in_window_dates"]:
        L += [f"분석 구간 KODEX 결측일(최대 40): {', '.join(k['missing_in_window_dates'])}", ""]
    ws = k["window_start"]
    y = panel[["y_idx", "y_kdx"]].dropna()
    y = y[y.index >= ws]
    L += [f"## 목표변수 기초 ({ws} 이후, 비용 전, 단순수익률 %)", "",
          "| 목표 | n | 평균 | 표준편차 |", "|---|---|---|---|",
          f"| K200 지수 시가→종가 | {len(y)} | {y['y_idx'].mean():+.4f} | {y['y_idx'].std():.4f} |",
          f"| KODEX 200 시가→종가 | {len(y)} | {y['y_kdx'].mean():+.4f} | {y['y_kdx'].std():.4f} |",
          f"| 차이(KODEX − 지수) | {len(y)} | {(y['y_kdx'] - y['y_idx']).mean():+.4f} | {(y['y_kdx'] - y['y_idx']).std():.4f} |", ""]
    return "\n".join(L)


def md_describe(desc):
    L = ["# 2단계 — 지표 기술통계 (신호 발생일 기준, 시프트 전)", "",
         "| 지표 | n | 기간 | 평균 | 표준편차 | 최소 | 최대 | AR(1) |", "|---|---|---|---|---|---|---|---|"]
    for _, r in desc.iterrows():
        L.append(f"| {r['indicator']} | {r['n']} | {r['start']} ~ {r['end']} | {r['mean']:+.4f} | {r['sd']:.4f} | {r['min']:+.3f} | {r['max']:+.3f} | {r['ar1']:+.3f} |")
    L += ["", "futflow_* 는 6단계 통제변수 전용이다. 이 조사에서 신호로 판정하지 않는다."]
    return "\n".join(L)


def md_blocked(path, schema, opt_audit):
    L = ["# 3~8단계 대기", ""]
    if opt_audit["status"] == "missing":
        L += [f"주 지표(외인 옵션 OptFlow)를 만들 파일이 없다: `{path}`", ""]
    else:
        L += [f"파일은 있으나 외인(foreign) 콜·풋 매수·매도 입력이 없다. 표준 투자자: {opt_audit.get('investors_std')}, "
              f"원 라벨: {opt_audit.get('investors_raw')}, 알 수 없는 라벨: {opt_audit.get('unknown_investor_labels_dropped')}", ""]
    L += [f"표준 스키마: `{', '.join(schema)}`", "",
          "- investor: 외국인·기타외국인(합산되어 foreign), 개인(individual), 기관계(institution), 전체(total) 등 — 한글 라벨 그대로 가능",
          "- cp: C/P 또는 call/put · 금액 단위 원 · 날짜는 KRX 거래일", "",
          "자료 확보 절차: README '사용자 절차'."]
    return "\n".join(L)


def _reg_lines(r, lags, name="x"):
    if not r or "coef" not in r:
        why = "추정 불가(상수열·공선)" if r and r.get("singular") else "표본 부족"
        return [f"- {why} (n={r.get('n') if r else 0})"]
    return [f"- β = {_f(r['coef'][name], 4)} (1표준편차당 {_f(r['beta_per_1sd'][name], 4)}%p, 표준화 {_f(r['std_beta'][name], 4)})",
            f"- HAC(lag {lags}) p = {r['p'][name]:.4f}, t = {r['t'][name]:+.2f}",
            f"- R² = {r['r2']:.4f}, n = {r['n']} ({r['start']} ~ {r['end']})"]


def _qt_lines(qt):
    if not qt:
        return ["표본 부족"]
    L = ["| 분위 | 지표 범위 | n | 다음날 평균 시가→종가(%) |", "|---|---|---|---|"]
    for q, row in qt["table"].iterrows():
        L.append(f"| Q{q} | {row['x_lo']:+.3f} ~ {row['x_hi']:+.3f} | {int(row['n'])} | {row['mean']:+.3f} |")
    L.append(f"\nQ5−Q1 = {qt['q5_q1']:+.3f}%p, 증가 단계 {qt['steps_up']}/4 (전체표본 분위 — 서술용)")
    return L


def md_results(res, res_kdx, crit, overall, explore, pr, prov, tag=""):
    sig, L0 = res["signal"], pr["hac_lags"]
    s = res["sample"]
    L = [f"# 옵션 매매 지표 조사 결과 — {sig}{tag}", ""] + _prov_lines(prov) + [
        f"판정 창 {s['window'][0]} ~ {s['window'][1]} → 실제 표본 n {s['n']} ({s['start']} ~ {s['end']}) · 목표 K200 지수 시가→종가(%) · KODEX 200 병기 · 비용 전", "",
        "## ③ 5분위", "### K200 지수"] + _qt_lines(res["quintile"]) + ["", "### KODEX 200"] + _qt_lines(res_kdx["quintile"])
    L += ["", "## ④ 회귀  R(T) = α + β·Flow(T−1) + ε", "### K200 지수"] + _reg_lines(res["reg"], L0) + ["### KODEX 200"] + _reg_lines(res_kdx["reg"], L0)
    L += ["", f"## ⑤ 연도별 안정성 (K200 지수, n<{pr['year_min_n']} 연도는 판정 분모에서 제외)", "| 연도 | n | β | p | Q5−Q1(연내 분위) |", "|---|---|---|---|---|"]
    for _, r in res["yearly"].iterrows():
        L.append(f"| {int(r['year'])} | {int(r['n'])} | {_f(r['beta'], 4)} | {_p(r['p'])} | {_f(r['q5_q1'], 3)} |")
    L += ["", "## ⑥ 통제", "### 전일 수익률 R(T−1) = T−1 시가→종가 통제 (사전등록)"] + _reg_lines(res["control_prev"], L0)
    if "coef" in (res["control_prev"] or {}):
        L.append(f"- 전일 수익률 계수 {_f(res['control_prev']['coef']['r_prev'], 4)}, p = {_p(res['control_prev']['p']['r_prev'])}")
    L += ["### 전일 수익률 + 외인 선물 Flow 통제 (사전등록)"] + _reg_lines(res["control_prev_fut"], L0)
    if "coef" in (res["control_prev_fut"] or {}):
        L.append(f"- 외인 선물 Flow 계수 {_f(res['control_prev_fut']['coef']['fut'], 4)}, p = {_p(res['control_prev_fut']['p']['fut'])}")
    L += ["### (탐색) 전일 종가→종가 수익률 통제"] + _reg_lines(res["control_prev_alt"], L0)
    L += ["", f"### 비교 구간 {pr['subperiod_start']}~ (사전등록 기준 5)"] + _reg_lines(res["subperiod"], L0)
    L += ["### (보고) 08:45 개장 이후 2023-07-31~"] + _reg_lines(res["subperiod_0845"], L0)
    L += ["", "## ⑦ 콜/풋 분해"]
    for name, r in res.get("decomp", {}).items():
        if "coef" in r:
            terms = ", ".join(f"{k} β {_f(v, 4)} (p {r['p'][k]:.3f})" for k, v in r["coef"].items() if k != "const")
            L.append(f"- {name}: {terms}, R² {r['r2']:.4f}")
    pl = res["placebo"]
    L += ["", "## ⑧ 순환 시프트 플라시보",
          f"- 실제 β {_f(pl.get('beta_obs'), 4)} vs 회전 β 평균 {_f(pl.get('perm_mean'), 4)} ± {pl.get('perm_sd', float('nan')):.4f}",
          f"- 단측 p = {pl.get('p_one_sided', float('nan')):.4f} ({pl.get('n_perm')}회, 최소 회전 {pl.get('min_shift')}일, seed {pl.get('seed')}) — 사전등록 통계량(β)",
          *( [f"- (보고) HAC t 로 같은 회전: t {res['placebo_t']['t_obs']:+.2f}, 단측 p = {res['placebo_t']['p_one_sided']:.4f} ({res['placebo_t']['n_perm']}회). "
              "β 회전은 신호 크기와 변동성이 함께 움직이는 이분산을 무시해 귀무분포가 좁아지므로, 해석은 이 값을 우선한다"] if res.get("placebo_t", {}).get("t_obs") is not None else [] ),
          "", "## 사전등록 기준 판정", "| 기준 | 결과 |", "|---|---|"]
    L += [f"| {name} | {'충족' if ok else '미충족'} |" for name, ok in crit]
    L += ["", f"**종합: {'1차 신호로 쓸 만함(2단계 매매 백테스트 진행 가능)' if overall else '기준 미충족 — 1차 신호로 채택하지 않음'}**", "",
          f"## 탐색 ({len(explore)}개, 중복 별칭·상수열 제외, 판정에 쓰지 않음·다중비교 미보정)", "| 지표 | n | β | p | R² | Q5−Q1 |", "|---|---|---|---|---|---|"]
    for e in explore:
        L.append(f"| {e['indicator']} | {e['n']} | {_f(e['beta'], 4)} | {_p(e['p'])} | {'—' if e['r2'] is None else round(e['r2'], 4)} | {_f(e['q5_q1'], 3)} |")
    return "\n".join(L)
