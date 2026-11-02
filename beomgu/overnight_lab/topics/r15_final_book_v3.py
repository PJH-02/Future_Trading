"""15차 공유 구성 확정판(v3): v2 반박 검토의 시나리오 틀(verify_r15_v2_review.py)을 그대로 실행하고 대표·중간·하한 묶음을 계산한다.

v2(r15_final_book_v2.py) 대비 대표 수치에 넣은 것(자료 처리 선택)
  • 롤 밤(청산일 d = 만기일): 차근월 대신 T 의 front(만기물)로 잡는다. 차근월 T 종가가 얇아 한쪽으로 쏠린 밤(2021-09-09 등)이 장기창 낙폭을 줄이는 문제 제거.
  • 2016~2017 코스닥 롱 밤: 선물 상장 초기 얇은 시가 체결(2017-10-16 등) 대신 13차 233740 경로 유지.
중간 묶음: 대표 + 아침 C3 두 다리 09:01 첫 체결 진입(분봉 2022-11-23~, 그 전은 W_all 실측 첫 1분 평균 변화로 근사).
하한 묶음: 중간 + 선물 비용 3bp(밤 롱·인버스, 낮 숏, 아침 초과 몫) + 단일가 체결 ETF 다리(밤 069500, 아침 혼합)에도 1틱.
k_rec = W_all·W_long 둘 다 MDD ≤ 15% 최대 k, k_max = ≤ 20%(격자 0.05). 산출 round15_alpha_share/final_v3/.
실행: PYTHONUTF8=1 python r15_final_book_v3.py  (검토 코드의 점검 출력도 final_v3/from_review/ 에 함께 쌓인다)
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
HERE = Path(__file__).resolve().parent
src = (HERE / "verify_r15_v2_review.py").read_text(encoding="utf-8")
old = 'OUT = R15 / "final_v2" / "verify"'
assert src.count(old) == 1
src = src.replace(old, 'OUT = R15 / "final_v3" / "from_review"')
ns = {"__name__": "r15_v3_runner", "__file__": str(HERE / "verify_r15_v2_review.py")}
exec(compile(src, "verify_r15_v2_review.py", "exec"), ns)

scenario, lg, C3OC, PROXY, R15 = ns["scenario"], ns["lg"], ns["C3OC"], ns["PROXY"], ns["R15"]
OUT = R15 / "final_v3"
OUT.mkdir(parents=True, exist_ok=True)
thin = lg.index[lg.index < "2018-01-01"]
BUNDLES = {
    "대표(롤 밤 만기물 + 2016~2017 롱 밤 233740)": dict(qa_etf_dates=thin, qa_roll_frontT=True),
    "중간(대표 + 아침 09:01 체결, 2022-11 전 근사)": dict(qa_etf_dates=thin, qa_roll_frontT=True, c3oc=C3OC, c3proxy=PROXY),
    "하한(중간 + 선물 3bp + 단일가 ETF 1틱)": dict(qa_etf_dates=thin, qa_roll_frontT=True, c3oc=C3OC, c3proxy=PROXY, fut=3e-4, auc_tick=True),
}
rows, grids = [], []
for tag, kw in BUNDLES.items():
    G, kr, km = scenario(tag, "9_v3", **kw)
    G = G.copy()
    G.insert(0, "bundle", tag)
    G["k_rec"], G["k_max"] = kr, km
    grids.append(G)
    for pick, kk in (("k1", 1.0), ("k_rec", kr), ("k_max", km), ("k1.40", 1.40)):
        r = G[np.isclose(G.k, kk)].iloc[0].to_dict()
        r["pick"] = pick
        rows.append(r)
pd.concat(grids).to_csv(OUT / "k_grid_v3.csv", index=False, encoding="utf-8-sig")
P = pd.DataFrame(rows)
cols = ["bundle", "pick", "k", "W_all_cagr", "W_all_mdd", "W_ex2026_cagr", "W_ex2026_mdd", "Y2024_ret", "Y2025_ret", "Y2026_ret",
        "W_long_cagr", "W_long_mdd", "W_all_worst", "W_long_worst", "W_long_worst_date"]
P = P[cols]
P.to_csv(OUT / "k_picks_v3.csv", index=False, encoding="utf-8-sig")
pd.set_option("display.width", 250)
pd.set_option("display.max_colwidth", 40)
show = P.copy()
for c in cols[3:-1]:
    show[c] = (show[c].astype(float) * 100).round(1)
print(show.to_string(index=False))
print("저장:", OUT)
