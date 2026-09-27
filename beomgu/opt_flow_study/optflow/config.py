"""경로·사전등록 파라미터.

사전등록 값은 옵션 자료를 보기 전에 정한 것이다.
옵션 자료를 받은 뒤에는 이 값을 바꾸지 않는다. 바꿔야 하면 새 버전을 만들고 PREREG_CHANGES 와 README 변경 이력에 이유를 남긴다.
"""
import json
import os
from pathlib import Path

PKG_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT_DIR = PKG_ROOT / "outputs"


def resolve_raw_dir(cli_value=None):
    """원자료 폴더 결정 순서와 상대경로 기준.

    1) --raw-dir, 2) 환경변수 OPTFLOW_RAW_DIR : 현재 작업 폴더 기준
    3) config.local.json 의 raw_dir             : 이 패키지 폴더(PKG_ROOT) 기준
    4) ./data/raw                                : PKG_ROOT 기준
    """
    base = Path.cwd()
    cand = cli_value or os.environ.get("OPTFLOW_RAW_DIR")
    if not cand:
        base = PKG_ROOT
        cfg = PKG_ROOT / "config.local.json"
        if cfg.exists():
            cand = json.loads(cfg.read_text(encoding="utf-8")).get("raw_dir")
    if not cand:
        base, cand = PKG_ROOT, "data/raw"
    p = Path(cand)
    if not p.is_absolute():
        p = (base / p).resolve()
    if not p.exists():
        raise FileNotFoundError(f"원자료 폴더 없음: {p} (README '데이터' 절 참고)")
    return p


def _local_cfg():
    cfg = PKG_ROOT / "config.local.json"
    return json.loads(cfg.read_text(encoding="utf-8")) if cfg.exists() else {}


def _resolve(cli_value, env_name, cfg_key, default):
    """CLI·환경변수는 현재 작업 폴더 기준, config.local.json 값과 기본값은 PKG_ROOT 기준."""
    base, cand = Path.cwd(), cli_value or os.environ.get(env_name)
    if not cand:
        base, cand = PKG_ROOT, _local_cfg().get(cfg_key) or default
    p = Path(cand)
    return p if p.is_absolute() else (base / p).resolve()


def resolve_out_dir(cli_value=None):
    """출력 폴더: --out-dir → OPTFLOW_OUT_DIR → config.local.json out_dir → ./outputs"""
    return _resolve(cli_value, "OPTFLOW_OUT_DIR", "out_dir", "outputs")


def resolve_collector_dir(cli_value=None):
    """원자료 수집기(collect_naver.py 등) 폴더: --collector-dir → OPTFLOW_COLLECTOR_DIR → config.local.json collector_dir → ../collectors"""
    return _resolve(cli_value, "OPTFLOW_COLLECTOR_DIR", "collector_dir", "../collectors")


# ---- 사전등록 v1.1 (2026-09-27, 옵션 자료 수령 전) ----
PREREG = {
    "version": "v1.1-2026-09-27",
    "primary_signal": "optflow_1d",          # 외인(외국인+기타외국인) 콜 Flow − 풋 Flow, 금액 기준, T−1
    "primary_target": "y_idx",               # T일 K200 지수 시가→종가(%)
    "window_start": "2010-09-20",            # 판정 표본 창(목표일 기준). 옵션 자료가 더 짧으면 짧은 대로 쓰고 감사에 기록
    "window_end": "2026-09-23",
    "hac_lags": 5,
    "alpha": 0.05,                           # 양측 HAC p
    "year_share_min": 0.75,                  # β>0 연도 비율
    "year_min_n": 30,                        # n<30 인 연도는 분모에서 제외하고 보고
    "subperiod_start": "2023-01-01",         # 비교 구간(08:45 개장 이후 구간은 따로 보고만)
    "placebo_perm": 5000,
    "placebo_min_shift": 20,
    "placebo_seed": 20260927,
    "control_prev_return": "r_prev_oc",      # R(T−1) = T−1 시가→종가 (목표 R(T)와 같은 정의)
    "control_prev_return_alt": "r_prev_cc",  # T−1 종가/T−2 종가 − 1 → 탐색 보고만
    "control_futflow": "futflow_1d",         # 외인 선물 (매수−매도)/(매수+매도), 계약, T−1
}

PREREG_CHANGES = [
    ("v1-2026-09-27", "최초"),
    ("v1.1-2026-09-27", "옵션 자료 수령 전 코드 검토 반영: 판정 창 고정(2010-09-20~2026-09-23), "
                        "전일 수익률 통제를 설계 표기 R(T−1)=T−1 시가→종가로 일치(종가→종가는 탐색), "
                        "연도 비율에서 n<30 연도 제외 규칙 명시"),
]
