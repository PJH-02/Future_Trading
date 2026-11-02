#!/usr/bin/env python3
"""오프라인 KRX sizing/allocation 비교기 (연구용; 주문 전송 없음).

설치: pip install pandas numpy scipy cvxpy openpyxl matplotlib requests
실행: python compare_strategy_allocation.py
경로/선택/수수료/비용/정책은 아래 CONFIG와 STRATEGIES에서 수정합니다.
기본값은 저장된 결과 장부를 재사용하고, 장부가 없으면 필요한 원본 함수만
메모리에서 실행합니다(검색/최적화/파일쓰기/네트워크 없음). 중앙 계정 경로는
갭 인지 daily TWR funding surrogate이며 ETF/T+2 결제·체결 감사가 아닙니다.
"""
from __future__ import annotations

import ast
import contextlib
import hashlib
import io
import importlib.util
import json
import math
import argparse
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

# ── 사용자 설정 ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "compare_strategy_allocation_output"
DEVELOPMENT_END = "2026-09-30"
DAYNIGHT_VERSION = "V1"
DAYNIGHT_ACCOUNT = "combined"
BEOMGU_VARIANT = "C3|vt30|ETF"
INTRADAY_SELECTED_RULES_PATH = "jehyeong/intraday/results/final/selected_rules.json"
HISTORY_DAYS = 252
MIN_TRADES = 30
KELLY_FRACTIONS = (0.25, 0.50, 1.00)
KELLY_SELECTED = {"daynight_v1_combined": "S1", "intraday_kospi_final": "S1", "intraday_kosdaq_final": "S1"}
KELLY_WINDOW = "expanding"  # expanding | rolling
ELIGIBILITY_FROM = None     # None means data-availability research only
COMMON_CALENDAR = True
MDD_CAPS = (0.05, 0.10)
REBALANCE_BANDS = (0.05, 0.10)
COMMISSION_BPS = 1.5
SLIPPAGE_BPS = 1.0
CENTRAL_RESIZE_COST_BPS = 0.0  # optional notional cost proxy; transfer itself remains free
WOOHYUN_HOLDING_BARS = 30
SUNNY_POSITION_COL = "pos_C"  # pos_A/B/C/D; default final choice C
SUNNY_ROUND_TRIP_COST = 0.0003
HOLDOUT_TOUCHED = True
ETF_MINUTE_DIR = ROOT / "jehyeong/intraday/data/minute"
# 중앙 book/account NAV는 초기자본=1인 연구 단위입니다. 원화 금액은 원하는 초기자본을 곱합니다.
# allocation_targets.csv에는 전략별 목표/실제 비중 열이 함께 저장됩니다.

# 장기 WF: 데이터 최초 가용일을 가상 개발일로 사용하며 아래 날짜로 덮어쓸 수 있습니다.
# EW는 즉시, LP/Convex는 252개 개별·공동 관측치 확보 후 편입합니다.
WF_OUTPUT_DIR = ROOT / "results/krx_staggered_walkforward_20261102"
WF_DEVELOPED_FROM = {
    "beomgu_c3_vt30_etf": None,
    "woohyun_top40_orb_vwap_30": None,
    "sunny_strategy_c": None,
    "daynight_v1_combined": None,
    "intraday_kospi_final": None,
    "intraday_kosdaq_final": None,
}
WF_NEW_STRATEGY_RULE = "mature_only"
WF_MIN_OBSERVATIONS = 252
WF_MDD_CAPS = (0.05, 0.10)
WF_BANDS = (0.05, 0.10)
WF_END = "2026-09-30"
WF_KERNEL_VERSION = "staggered-v2-mature-only"

# Registry extension: adapter='daily_csv' requires date + daily_return columns;
# trades_path is optional and provenance text should explain return construction.
STRATEGIES = [{'id': 'beomgu_c3_vt30_etf',
  'label': 'beomgu C3|VT30|ETF',
  'enabled': True,
  'adapter': 'daily_csv',
  'source_code': 'beomgu/overnight_lab/topics/r5_c3_levered.py',
  'daily_path': 'beomgu/results/2026-09-29-overnight/round5_strategy/c3_levered/daily_returns.csv',
  'trades_path': 'beomgu/results/2026-09-29-overnight/round5_strategy/c3_levered/trades.csv',
  'return_col': 'net5_chosen',
  'can_adjust_leverage': False,
  'cash_when_missing': True,
  'eligible_from': None,
  'provenance': 'Saved chosen series; includes pre-launch synthetic leveraged history per source '
                'limitations.'},
 {'id': 'woohyun_top40_orb_vwap_30',
  'label': 'woohyun Top40 ORB/VWAP 30m',
  'enabled': True,
  'adapter': 'woohyun_trades',
  'source_code': 'woohyun/Results/vwap+orb+flow40%.py',
  'daily_path': 'woohyun/data/KODEX200_1min_2023_2026.csv',
  'trades_path': 'woohyun/Results/flow_top20_30_40_strategy_trades.csv',
  'can_adjust_leverage': False,
  'cash_when_missing': True,
  'eligible_from': None,
  'provenance': 'Positive-volume minute calendar; gross trade returns less configured round-trip cost.'},
 {'id': 'sunny_strategy_c',
  'label': 'sunny Strategy C',
  'enabled': True,
  'adapter': 'sunny_signals',
  'source_code': 'sunny/intraday_strategy_C/strategy3_smart_vs_dumb.py',
  'daily_path': 'sunny/intraday_strategy_C/output/strategy3_signals.csv',
  'can_adjust_leverage': False,
  'cash_when_missing': True,
  'eligible_from': None,
  'provenance': 'Saved full signal/price history; index/mini-futures proxy, not physical ETF fills.'},
 {'id': 'daynight_v1_combined',
  'label': 'jehyeong daynight V1 combined',
  'enabled': True,
  'adapter': 'daily_csv',
  'source_code': 'jehyeong/daynight/backtest_daynight.py',
  'daily_path': 'jehyeong/daynight/results/daily_nav.csv',
  'trades_path': 'jehyeong/daynight/results/trades.csv',
  'return_col': 'daily_return',
  'can_adjust_leverage': True,
  'cash_when_missing': False,
  'market': 'KOSDAQ',
  'orders_path': 'jehyeong/daynight/results/orders.csv',
  'eligible_from': None,
  'provenance': 'Saved V1 combined account NAV and immutable trade/order ledger.'},
 {'id': 'intraday_kospi_final',
  'label': 'jehyeong intraday final KOSPI200',
  'enabled': True,
  'adapter': 'daily_csv',
  'source_code': 'jehyeong/intraday/code/backtest_intraday_momentum.py',
  'daily_path': 'jehyeong/intraday/results/final/KOSPI200/daily_pnl.csv',
  'trades_path': 'jehyeong/intraday/results/final/KOSPI200/trades.csv',
  'return_col': 'daily_return',
  'can_adjust_leverage': True,
  'cash_when_missing': True,
  'market': 'KOSPI',
  'orders_path': 'jehyeong/intraday/results/final/KOSPI200/orders.csv',
  'eligible_from': None,
  'provenance': 'Saved final daily P&L/trade/order ledgers; selected_rules used full sample through '
                '2026-10-07.'},
 {'id': 'intraday_kosdaq_final',
  'label': 'jehyeong intraday final KOSDAQ150',
  'enabled': True,
  'adapter': 'daily_csv',
  'source_code': 'jehyeong/intraday/code/backtest_intraday_momentum.py',
  'daily_path': 'jehyeong/intraday/results/final/KOSDAQ150/daily_pnl.csv',
  'trades_path': 'jehyeong/intraday/results/final/KOSDAQ150/trades.csv',
  'return_col': 'daily_return',
  'can_adjust_leverage': True,
  'cash_when_missing': True,
  'market': 'KOSDAQ',
  'orders_path': 'jehyeong/intraday/results/final/KOSDAQ150/orders.csv',
  'eligible_from': None,
  'provenance': 'Saved final daily P&L/trade/order ledgers; no physical inverse-2x exists, cap short-side '
                'leverage at 1x.'}]

LIMITATIONS = [
    "중앙 EW/LP/Convex는 overnight gap 이후 다음 세션 funding을 반영하는 일별 TWR surrogate이며, 실제 T+2 settlement·수량/호가단위·ETF resize fill 감사가 아닙니다.",
    "중앙 internal transfer 자체 비용은 0입니다. CENTRAL_RESIZE_COST_BPS는 ETF 주문비용 대신 쓰는 명시적 notional proxy이며, 기본값 0이고 ETF gross/net delta·세금·분배금은 별도 검증이 필요합니다.",
    "Beomgu의 상장 전 leveraged history는 synthetic ETF 경로를 포함합니다. Sunny는 실제 ETF fill이 아닌 index/mini-futures proxy입니다.",
    "Beomgu 원본은 같은 09:00 시가로 신호와 체결을 가정합니다. 원본을 보존한 비교이며, 실행 가능한 인과적 매매 성과로 인증하지 않습니다.",
    "S0_MATCHED_DEV는 개발 구간의 평균 진입 노출로 고정한 대조군이며 개발 종료 다음 날부터만 평가합니다. 노출의 일중 평균이나 완전한 위험 일치를 뜻하지 않습니다.",
    "252일은 현재 비교 가능한 유효 관측일 기준입니다. 전략별 누락 때문에 실제 거래일 범위가 더 길 수 있으며 estimator_start/end에 기록합니다. 분기 표는 사전 고정 정책의 순차 결과이며 중첩 정책 선정 검증을 수행하지 않습니다.",
    "intraday 최종 규칙은 2026-10-07까지의 전체 sample을 사용해 선정되었으므로 2026-10 이후는 untouched holdout으로 볼 수 없습니다.",
    "동일일 신호가 없는 고정형 sleeve의 원천 NA는 optimizer에서 NA로 유지하고 execution funding surrogate에서는 suspended cash로 처리하며 감사 파일에 남깁니다.",
]


def _path(rel):
    p = ROOT / rel
    if not p.is_file():
        raise FileNotFoundError(f"필수 입력 누락: {p}; 데이터 없는 날짜를 0으로 채우지 않습니다.")
    return p


def _date(s):
    return pd.to_datetime(s, errors="raise").dt.normalize()


def _load_daily_csv(spec):
    p = _path(spec["daily_path"])
    x = pd.read_csv(p)
    adapter = spec["adapter"]
    if adapter == "daily_csv" and spec["id"] == "daynight_v1_combined":
        x = x[(x.version == DAYNIGHT_VERSION) & (x.account == DAYNIGHT_ACCOUNT)].copy()
        if x.empty:
            raise ValueError("daynight ledger lacks V1/combined account rows")
    if adapter == "daily_csv":
        x["date"] = _date(x.date)
        col = spec.get("return_col", "daily_return")
        if col not in x: raise ValueError(f"{p}: missing {col}")
        if spec["id"] == "beomgu_c3_vt30_etf":
            needed={"w_chosen","r_069500_oc","r_122630_oc"}
            if not needed.issubset(x): raise ValueError(f"{p}: missing Beomgu gross inputs {needed-set(x)}")
            w=pd.to_numeric(x.w_chosen,errors="coerce").fillna(0.).clip(0,2)
            r1=pd.to_numeric(x.r_069500_oc,errors="coerce")
            r2=pd.to_numeric(x.r_122630_oc,errors="coerce")
            # Fractional leverage is a cash + 1x/2x sleeve blend; leave cash
            # untouched by trading costs.
            w1=w.where(w.le(1),2-w); w2=(w-1).clip(lower=0); cash=1-w1-w2
            buy=1/((1+SLIPPAGE_BPS/10000)*(1+COMMISSION_BPS/10000))
            sell=(1-SLIPPAGE_BPS/10000)*(1-COMMISSION_BPS/10000)
            leg1=np.where(w1.to_numpy()>0,(1+r1.to_numpy())*buy*sell,1.)
            leg2=np.where(w2.to_numpy()>0,(1+r2.to_numpy())*buy*sell,1.)
            x[col]=cash.to_numpy()+w1.to_numpy()*leg1+w2.to_numpy()*leg2-1
        if x.date.duplicated().any(): raise ValueError(f"{p}: duplicate daily dates")
        s = pd.Series(pd.to_numeric(x[col], errors="coerce").to_numpy(), index=x.date, name=spec["id"]).sort_index()
        return s, {"path":str(p),"sha256":sha256(p),"rows":len(x),"return_col":col}
    if adapter == "sunny_signals":
        x["date"] = _date(x.date)
        required = ["oc_ret", SUNNY_POSITION_COL]
        if not set(required).issubset(x): raise ValueError(f"{p}: missing {set(required)-set(x)}")
        signal_cols={"pos_A":["ind_call_z_lag"],"pos_B":["ind_put_z_lag"],"pos_C":["for_put_z_lag","ind_call_z_lag"],"pos_D":["for_put_z_lag","ind_call_z_lag","for_call_z_lag","ind_put_z_lag"]}[SUNNY_POSITION_COL]
        if not set(signal_cols).issubset(x): raise ValueError(f"{p}: missing signal warmup columns {set(signal_cols)-set(x)}")
        valid = x[required+signal_cols].notna().all(axis=1)
        pos=pd.to_numeric(x[SUNNY_POSITION_COL], errors="coerce")
        direction=np.sign(pos)
        op=pd.to_numeric(x.open,errors="coerce"); cl=pd.to_numeric(x.close,errors="coerce")
        slip=SLIPPAGE_BPS/10000.; fee=COMMISSION_BPS/10000.
        entry=op*(1+direction*slip); exit_=cl*(1-direction*slip)
        ret=direction*(exit_-entry)/op-fee*(entry+exit_)/op
        ret=ret.where(direction.ne(0),0.)
        # Invalid warm-up rows stay NA; valid no-position rows remain true zero.
        ret = ret.where(valid)
        if x.date.duplicated().any(): raise ValueError(f"{p}: duplicate daily dates")
        s = pd.Series(ret.to_numpy(), index=x.date, name=spec["id"])
        return s.sort_index(), {"path":str(p),"sha256":sha256(p),"rows":len(x),"return_col":f"oc_ret*{SUNNY_POSITION_COL}-0.0003*abs({SUNNY_POSITION_COL})"}
    raise ValueError(adapter)



def _woohyun_ready_dates(minute_calendar):
    bp=_path("woohyun/data/KRX foreign_buy.xlsx"); sp=_path("woohyun/data/KRX foreign_sell.xlsx")
    buy=pd.read_excel(bp); sell=pd.read_excel(sp)
    for frame in (buy,sell): frame["date"]=pd.to_datetime(frame["일자"],errors="raise").dt.normalize()
    buy=buy[["date","외국인 합계"]].rename(columns={"외국인 합계":"buy"})
    sell=sell[["date","외국인 합계"]].rename(columns={"외국인 합계":"sell"})
    flow=buy.merge(sell,on="date",how="inner").sort_values("date")
    denom=pd.to_numeric(flow.buy,errors="coerce")+pd.to_numeric(flow.sell,errors="coerce")
    flow["foreign_flow"]=(pd.to_numeric(flow.buy,errors="coerce")-pd.to_numeric(flow.sell,errors="coerce"))/denom.replace(0,np.nan)
    flow["previous_day_flow"]=flow.foreign_flow.shift(1)
    flow["top40_threshold"]=flow.foreign_flow.rolling(250,min_periods=250).quantile(.60).shift(1)
    available=flow.previous_day_flow.notna()&flow.top40_threshold.notna()&flow.date.isin(minute_calendar)
    return pd.DatetimeIndex(flow.loc[available,"date"]),{"foreign_buy_path":str(bp),"foreign_buy_sha256":sha256(bp),"foreign_sell_path":str(sp),"foreign_sell_sha256":sha256(sp)}

def _load_woohyun(spec):
    tp, cp = _path(spec["trades_path"]), _path(spec["daily_path"])
    tr = pd.read_csv(tp)
    tr = tr[tr.strategy.astype(str).eq(f"E_FlowTop40_ORB_VWAP_{WOOHYUN_HOLDING_BARS}min")].copy()
    tr["date"] = _date(tr.date)
    gross=pd.to_numeric(tr["return"],errors="coerce")
    buy=1/((1+SLIPPAGE_BPS/10000)*(1+COMMISSION_BPS/10000))
    sell=(1-SLIPPAGE_BPS/10000)*(1-COMMISSION_BPS/10000)
    tr["return"]=(1+gross)*buy*sell-1
    # Calendar from observed positive-volume minute bars, not from trade days.
    m = pd.read_csv(cp, usecols=["datetime","volume"])
    m = m[pd.to_numeric(m.volume, errors="coerce").gt(0)].copy()
    # The archived minute timestamps carry Asia/Seoul offsets while trade and
    # ETF calendars are naive local dates. Normalize to Seoul wall time first.
    minute_dt = pd.to_datetime(m.datetime, errors="raise", utc=True).dt.tz_convert("Asia/Seoul").dt.tz_localize(None)
    m["date"] = minute_dt.dt.normalize()
    idx = pd.DatetimeIndex(m.date.drop_duplicates().sort_values())
    ready,flow_meta=_woohyun_ready_dates(idx)
    s = pd.Series(0.0, index=idx, name=spec["id"]); s.loc[~s.index.isin(ready)]=np.nan
    if tr.date.duplicated().any(): raise ValueError("multiple selected Woohyun trades/day; daily adapter requires explicit aggregation")
    if not tr.date.isin(ready).all(): raise ValueError("saved Woohyun trade precedes flow/top40 warmup or lacks minute calendar")
    s.loc[tr.date.to_numpy()] = tr["return"].to_numpy()
    return s, {"trades_path":str(tp),"trades_sha256":sha256(tp),"calendar_path":str(cp),"calendar_sha256":sha256(cp),**flow_meta,"trades":len(tr),"cost_bps_each_side":COMMISSION_BPS+SLIPPAGE_BPS,"holding_bars":WOOHYUN_HOLDING_BARS}


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""): h.update(chunk)
    return h.hexdigest()


def _load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    if spec is None or spec.loader is None: raise ImportError(f"cannot load {path}")
    module=importlib.util.module_from_spec(spec); sys.modules[name]=module; spec.loader.exec_module(module)
    return module


def fallback_sunny(spec):
    source=_path("sunny/intraday_strategy_C/strategy3_smart_vs_dumb.py")
    m=_load_module("allocation_fallback_sunny",source)
    panel=m.build_positions(m.build_signals(m.load_option_flow(),m.load_prices()))
    signal_cols={"pos_A":["ind_call_z_lag"],"pos_B":["ind_put_z_lag"],"pos_C":["for_put_z_lag","ind_call_z_lag"],"pos_D":["for_put_z_lag","ind_call_z_lag","for_call_z_lag","ind_put_z_lag"]}[SUNNY_POSITION_COL]
    panel=panel.dropna(subset=signal_cols)
    _,net=m.backtest(panel,SUNNY_POSITION_COL,cost=SUNNY_ROUND_TRIP_COST)
    s=pd.Series(net.to_numpy(float),index=pd.DatetimeIndex(net.index).normalize(),name=spec["id"])
    trades=pd.DataFrame({"date":s.index[s.ne(0)],"net_return":s[s.ne(0)].to_numpy()})
    return s,{"fallback":"original functions load_option_flow/load_prices/build_signals/build_positions/backtest in memory","source":str(source),"source_sha256":sha256(source)},trades,None


def fallback_woohyun(spec):
    source=_path("woohyun/Results/vwap+orb+flow40%.py")
    paths={"KODEX_FILE":ROOT/"woohyun/data/KODEX200_1min_2023_2026.csv","BUY_FILE":ROOT/"woohyun/data/KRX foreign_buy.xlsx","SELL_FILE":ROOT/"woohyun/data/KRX foreign_sell.xlsx"}
    nodes=[]
    for node in ast.parse(source.read_text(encoding="utf-8")).body:
        if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id in paths:
            node.value=ast.Constant(str(paths[node.targets[0].id]))
        nodes.append(node)
        if isinstance(node,ast.FunctionDef) and node.name=="backtest": break
    else: raise ValueError("Woohyun backtest function not found")
    ns={"__name__":"allocation_fallback_woohyun","__file__":str(source)}
    tree=ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[]))
    with contextlib.redirect_stdout(io.StringIO()): exec(compile(tree,str(source),"exec"),ns)
    minute=ns["kodex"]
    tr=ns["backtest"](minute,WOOHYUN_HOLDING_BARS,"top40")
    tr=pd.DataFrame(tr)
    date=pd.to_datetime(tr["date"],errors="raise").dt.normalize()
    tr["date"]=date
    gross=pd.to_numeric(tr["return"],errors="coerce")
    buy=1/((1+SLIPPAGE_BPS/10000)*(1+COMMISSION_BPS/10000)); sell=(1-SLIPPAGE_BPS/10000)*(1-COMMISSION_BPS/10000)
    tr["net_return"]=(1+gross)*buy*sell-1
    cal=pd.DatetimeIndex(pd.to_datetime(minute.loc[pd.to_numeric(minute.volume,errors="coerce").gt(0),"date"]).drop_duplicates().sort_values())
    ready,flow_meta=_woohyun_ready_dates(cal)
    ret=pd.Series(0.,index=cal,name=spec["id"]); ret.loc[~ret.index.isin(ready)]=np.nan
    if tr.date.duplicated().any(): raise ValueError("Woohyun fallback produced multiple trades/day")
    if not tr.date.isin(ready).all(): raise ValueError("Woohyun fallback trade before Top40 warmup/calendar")
    ret.loc[tr.date.to_numpy()]=tr.net_return.to_numpy()
    return ret,{"fallback":"AST-extracted original preparation/backtest only; fixed holding/top40","source":str(source),"source_sha256":sha256(source),"data_paths":[str(x) for x in paths.values()],**flow_meta},tr,None


def fallback_beomgu(spec):
    # Guard every cached price before importing; patched price never fetches/downloads.
    required=[ROOT/f"beomgu/data/etf/{x}.csv" for x in ("069500","122630","102110","KPI200","FUT")]
    required += [ROOT/"beomgu/data/k200_index_daily.csv",ROOT/"beomgu/data/spx_daily.csv"]
    missing=[p for p in required if not p.is_file()]
    if missing: raise FileNotFoundError(f"Beomgu offline fallback caches missing: {missing[:5]}")
    source=ROOT/"beomgu/overnight_lab/topics/r5_c3_levered.py"
    m=_load_module("allocation_fallback_beomgu",source)
    m.data.RAW=ROOT/"beomgu/data"; m.data.PX_DIR=m.data.RAW/"etf"
    original=m.data.price
    def cached_price(symbol,refresh=False):
        p=m.data.PX_DIR/f"{symbol}.csv"
        if refresh or not p.is_file(): raise FileNotFoundError(f"offline-only Beomgu fallback refuses uncached {symbol}")
        return original(symbol,False)
    m.data.price=cached_price
    original_get=m.data.requests.get
    def network_forbidden(*args,**kwargs): raise RuntimeError("offline fallback forbids network access")
    m.data.requests.get=network_forbidden
    try:
        d=m.load(); v=next(v for v in m.variants() if v["name"]==BEOMGU_VARIANT)
    finally:
        m.data.requests.get=original_get
    w=m.exposure(d,v); net,_=m.daily_net(d,w,"ETF")
    s=pd.Series(net.to_numpy(float),index=pd.DatetimeIndex(d.index).normalize(),name=spec["id"])
    tr=pd.DataFrame({"date":s.index[w.to_numpy()>0],"net_return":s.to_numpy()[w.to_numpy()>0]})
    return s,{"fallback":f"cached-only original load/exposure/daily_net; frozen {BEOMGU_VARIANT}","source":str(source),"source_sha256":sha256(source)},tr,None


def fallback_daynight(spec):
    source=ROOT/"jehyeong/daynight/backtest_daynight.py"
    m=_load_module("allocation_fallback_daynight",source)
    market=m.Market(ROOT/"jehyeong/intraday/data"); config=m.Config(); dates=market.calendar(config)
    plans,_,_=m.make_plans(market,dates)
    daily,orders,trades=m.simulate(DAYNIGHT_VERSION,DAYNIGHT_ACCOUNT,dates,plans,config)
    daily["date"]=pd.to_datetime(daily.date).dt.normalize()
    trades["date"]=pd.to_datetime(trades.exit_time).dt.normalize() if not trades.empty else pd.Series(dtype="datetime64[ns]")
    return pd.Series(daily.daily_return.to_numpy(float),index=daily.date,name=spec["id"]),{"fallback":"in-memory Market/calendar/make_plans/simulate V1 combined","source":str(source),"source_sha256":sha256(source)},trades,orders


def fallback_intraday(spec):
    source=ROOT/"jehyeong/intraday/code/backtest_intraday_momentum.py"
    code_dir=str(source.parent)
    if code_dir not in sys.path: sys.path.insert(0,code_dir)
    m=_load_module("allocation_fallback_intraday_"+spec["id"],source)
    selection=_path(INTRADAY_SELECTED_RULES_PATH)
    selected=json.loads(_path(str(selection.relative_to(ROOT))).read_text(encoding="utf-8"))
    market="KOSPI200" if spec["id"]=="intraday_kospi_final" else "KOSDAQ150"
    rule=selected["markets"][market]
    data=m.load_data(ROOT/"jehyeong/intraday/data")
    signal=m.MARKETS[market]["signal"]
    features=m.make_features(data[signal]["1m"],data[signal]["daily"])
    valid,_=m.eligible_days(market,data,features)
    reference_dates=[pd.Timestamp(d).strftime("%Y-%m-%d") for d in selected["sample_dates"][market]]
    valid_dates=[pd.Timestamp(d).strftime("%Y-%m-%d") for d in valid]
    missing=sorted(set(reference_dates)-set(valid_dates))
    if missing: raise ValueError(f"{market}: invalid frozen sample dates {missing[:5]}")
    frozen_end=max(reference_dates)
    days=sorted(set(reference_dates)|{d for d in valid_dates if d>frozen_end})
    features,_=m.add_vol_buckets(features,days)
    config=m.Config()
    daily,trades,orders,_=m.simulate(market,rule["model"],features,data,days,"V1_next_open",config,**rule["options"])
    daily["date"]=pd.to_datetime(daily.date).dt.normalize()
    return pd.Series(daily.daily_return.to_numpy(float),index=daily.date,name=spec["id"]),{"fallback":"in-memory local load_data/features/frozen eligible dates/simulate; run_selected_strategies not called","source":str(source),"source_sha256":sha256(source)},trades,orders


def fallback_adapter(spec):
    aid=spec["id"]
    if aid=="beomgu_c3_vt30_etf": return fallback_beomgu(spec)
    if aid=="woohyun_top40_orb_vwap_30": return fallback_woohyun(spec)
    if aid=="sunny_strategy_c": return fallback_sunny(spec)
    if aid=="daynight_v1_combined": return fallback_daynight(spec)
    if aid in ("intraday_kospi_final","intraday_kosdaq_final"): return fallback_intraday(spec)
    raise ValueError(f"no original in-memory fallback for adapter {aid}")


def load_universe():
    series, provenance, trades, order_ledgers = {}, {}, {}, {}
    for spec in STRATEGIES:
        if not spec.get("enabled"): continue
        missing_saved=not (ROOT/spec["daily_path"]).is_file()
        if spec["id"]=="woohyun_top40_orb_vwap_30" and WOOHYUN_HOLDING_BARS!=30: missing_saved=True
        if spec["id"]=="beomgu_c3_vt30_etf" and BEOMGU_VARIANT!="C3|vt30|ETF": missing_saved=True
        for key in ("trades_path","orders_path"):
            if key in spec and not (ROOT/spec[key]).is_file(): missing_saved=True
        try:
            if missing_saved: raise FileNotFoundError("saved ledger missing")
            if spec["adapter"] == "woohyun_trades":
                if WOOHYUN_HOLDING_BARS!=30: raise FileNotFoundError("saved ledger is fixed at 30 bars")
                s, src = _load_woohyun(spec)
            else: s, src = _load_daily_csv(spec)
            tx=pd.DataFrame(); ox=pd.DataFrame()
            if spec.get("trades_path"):
                tx=pd.read_csv(_path(spec["trades_path"]))
                if spec["id"]=="daynight_v1_combined": tx=tx[(tx.version==DAYNIGHT_VERSION)&(tx.account==DAYNIGHT_ACCOUNT)].copy()
                if spec["id"]=="woohyun_top40_orb_vwap_30": tx=tx[tx.strategy.astype(str).eq(f"E_FlowTop40_ORB_VWAP_{WOOHYUN_HOLDING_BARS}min")].copy()
            if spec.get("orders_path"):
                op=_path(spec["orders_path"]); ox=pd.read_csv(op)
                if spec["id"]=="daynight_v1_combined": ox=ox[(ox.version==DAYNIGHT_VERSION)&(ox.account==DAYNIGHT_ACCOUNT)].copy()
                src={**src,"orders_path":str(op),"orders_sha256":sha256(op)}
        except FileNotFoundError:
            s,src,tx,ox=fallback_adapter(spec)
            src={**src,"generated_in_memory":True}
        start=spec.get("eligible_from",ELIGIBILITY_FROM)
        if start: s=s.loc[pd.Timestamp(start):]
        series[spec["id"]]=s
        source_meta={}
        if spec.get("source_code") and (ROOT/spec["source_code"]).is_file(): source_meta={"source_code_path":str(ROOT/spec["source_code"]),"source_code_sha256":sha256(ROOT/spec["source_code"])}
        provenance[spec["id"]]={**src,**source_meta,"label":spec["label"],"provenance":spec["provenance"],"can_adjust_leverage":spec["can_adjust_leverage"]}
        if not tx.empty: trades[spec["id"]]=tx
        if not ox.empty: order_ledgers[spec["id"]]=ox
    if not series: return pd.DataFrame(),provenance,trades,order_ledgers
    panel=pd.concat(series.values(),axis=1).sort_index()
    return panel,provenance,trades,order_ledgers

ETF_PRICE_CACHE = {}
ETF_PRICE_INPUTS = {}


def price_store(symbol):
    """Raw TOSS minute rows; keep the 15:30 close row for daynight marks."""
    symbol = str(symbol).zfill(6)
    if symbol in ETF_PRICE_CACHE: return ETF_PRICE_CACHE[symbol]
    path = ETF_MINUTE_DIR / f"{symbol}_TOSS_1m.csv"
    if not path.is_file(): ETF_PRICE_CACHE[symbol] = None; return None
    x = pd.read_csv(path, usecols=["timestamp", "open", "high", "low", "close", "volume"])
    ETF_PRICE_INPUTS[symbol]={"path":str(path),"sha256":sha256(path),"rows":len(x)}
    x["timestamp"] = pd.to_datetime(x.timestamp, errors="raise")
    x = x.sort_values("timestamp").drop_duplicates("timestamp").set_index("timestamp")
    for c in ("open", "high", "low", "close", "volume"): x[c] = pd.to_numeric(x[c], errors="coerce")
    ETF_PRICE_CACHE[symbol] = x
    return x


def _market_symbol(symbol, market, direction):
    """Map original security to the same-direction physical 1x and available 2x ETF."""
    sym = str(symbol).zfill(6)
    if market == "KOSPI":
        if sym in ("114800", "252670") or direction < 0: return ("114800", "252670")
        return ("069500", "122630")
    if sym == "251340" or direction < 0: return ("251340", None)
    return ("229200", "233740")


def _bar_price(symbol, bar_end, field, executable=True):
    x = price_store(symbol)
    if x is None: return None
    t = pd.Timestamp(bar_end)
    if t not in x.index: return None
    row = x.loc[t]
    if executable and (not np.isfinite(row.get("volume", np.nan)) or row.volume <= 0): return None
    p = row.get(field, np.nan)
    return float(p) if np.isfinite(p) and p > 0 else None


def normalize_trade_events(spec, trades, orders):
    """Join saved trades/orders by immutable trade id; never infer a bar from clock time."""
    if trades.empty or orders.empty: return []
    tr = trades.copy(); od = orders.copy()
    if spec["id"] == "daynight_v1_combined":
        tr = tr[(tr.version == DAYNIGHT_VERSION) & (tr.account == DAYNIGHT_ACCOUNT)].copy()
        od = od[(od.version == DAYNIGHT_VERSION) & (od.account == DAYNIGHT_ACCOUNT)].copy()
    if "trade_id" not in tr or "trade_id" not in od: raise ValueError(f"{spec['id']}: trade_id required for physical replay")
    events=[]
    for _, t in tr.iterrows():
        matching=od[od.trade_id.astype(str).eq(str(t.trade_id))].copy()
        if matching.empty: raise ValueError(f"{spec['id']}: no order ledger for trade {t.trade_id}")
        time_col="timestamp"
        matching[time_col]=pd.to_datetime(matching[time_col],errors="raise")
        matching=matching.sort_values(time_col)
        entry=matching.iloc[0]; exit_=matching.iloc[-1]
        if "execution_bar_end" in matching:
            eb=pd.to_datetime(entry.execution_bar_end,errors="coerce")
            xb=pd.to_datetime(exit_.execution_bar_end,errors="coerce")
            ef=str(entry.get("execution_field","open")); xf=str(exit_.get("execution_field","close"))
        else:
            eb=pd.to_datetime(entry.reference_bar_end,errors="coerce")
            xb=pd.to_datetime(exit_.reference_bar_end,errors="coerce")
            ef=str(entry.get("reference_field","close")); xf=str(exit_.get("reference_field","close"))
        if pd.isna(eb) or pd.isna(xb): raise ValueError(f"{spec['id']} trade {t.trade_id}: missing exact reference bar end")
        direction=int(t.get("direction",1))
        # Original order symbol identifies long/inverse side; direction is a fallback.
        raw_symbol=str(entry.get("symbol",t.get("symbol","")))
        if raw_symbol.endswith(".0"): raw_symbol=raw_symbol[:-2]
        orig=raw_symbol.zfill(6)
        d=-1 if orig in ("114800","252670","251340") else direction
        one,two=_market_symbol(orig,spec["market"],d)
        exit_raw=str(exit_.get("symbol",orig))
        if exit_raw.endswith(".0"): exit_raw=exit_raw[:-2]
        events.append({"trade_id":str(t.trade_id),"entry_date":eb.normalize(),"exit_date":xb.normalize(),"entry_bar_end":eb,"exit_bar_end":xb,"entry_field":ef,"exit_field":xf,"original_symbol":orig,"exit_order_symbol":exit_raw.zfill(6),"entry_reference_price":float(entry.get("reference_price",np.nan)),"exit_reference_price":float(exit_.get("reference_price",np.nan)),"one_x":one,"two_x":two,"direction":d,"orders":matching,"trade":t})
    return sorted(events,key=lambda e:(e["entry_bar_end"],e["exit_bar_end"]))


def _event_price_ratio(event,symbol):
    sym=str(symbol).zfill(6)
    ep=event["entry_reference_price"] if sym==event["original_symbol"] and np.isfinite(event["entry_reference_price"]) else _bar_price(sym,event["entry_bar_end"],event["entry_field"],True)
    xp=event["exit_reference_price"] if sym==event["exit_order_symbol"] and np.isfinite(event["exit_reference_price"]) else _bar_price(sym,event["exit_bar_end"],event["exit_field"],True)
    return (float(xp)/float(ep)) if ep is not None and xp is not None and ep>0 else np.nan


def _event_path_uncached(event, symbol):
    """Daily 1x/2x ETF wealth marks from exact entry/exit bars and 15:30 closes."""
    sym=str(symbol).zfill(6)
    ep=event["entry_reference_price"] if sym==event["original_symbol"] and np.isfinite(event["entry_reference_price"]) else _bar_price(sym,event["entry_bar_end"],event["entry_field"],True)
    xp=event["exit_reference_price"] if sym==event["exit_order_symbol"] and np.isfinite(event["exit_reference_price"]) else _bar_price(sym,event["exit_bar_end"],event["exit_field"],True)
    if ep is None or xp is None: return None,"missing_positive_volume_entry_or_exit"
    if event["exit_bar_end"] < event["entry_bar_end"]: return None,"exit_before_entry"
    buy_factor=1/((1+SLIPPAGE_BPS/10000)*(1+COMMISSION_BPS/10000))
    sell_factor=(1-SLIPPAGE_BPS/10000)*(1-COMMISSION_BPS/10000)
    if event["entry_date"] == event["exit_date"]:
        return {event["entry_date"]: float(xp/ep*buy_factor*sell_factor)},"ok"
    x=price_store(symbol)
    if x is None: return None,"missing_minute_price_file"
    # Only held positions require an EOD mark; an exited position needs no later quote.
    observed=x.index.normalize()
    days=pd.DatetimeIndex(observed[(observed >= event["entry_date"]) &
                                  (observed <= event["exit_date"])].unique()).sort_values()
    if event["entry_date"] not in days or event["exit_date"] not in days:
        return None,"missing_market_session_1530_mark"
    marks={}
    for day in days:
        close_t=day+pd.Timedelta(hours=15,minutes=30)
        if day==event["exit_date"] and event["exit_bar_end"]<=close_t: marks[day]=xp
        elif day==event["entry_date"] and event["entry_bar_end"]>=close_t: marks[day]=ep
        else:
            marks[day]=_bar_price(symbol,close_t,"close",False)
            if marks[day] is None: return None,"missing_exact_1530_mark"
    wealth={}; prior=1.0; previous_price=ep
    for day in days:
        if day==event["entry_date"]:
            # Entry cost is paid once; a same-day exit also pays the sell-side cost.
            value=(marks[day]/ep)*buy_factor
            if day==event["exit_date"]: value*=sell_factor
        elif day==event["exit_date"]:
            value=prior*(marks[day]/previous_price)*sell_factor
        else:
            value=prior*(marks[day]/previous_price)
        wealth[day]=float(value); prior=float(value); previous_price=marks[day]
    return wealth,"ok"


_EVENT_PATH_CACHE = {}


def _event_path(event, symbol):
    """Memoize exact reference-bar paths, including their physical inputs/costs."""
    key=(str(event["trade_id"]),str(symbol).zfill(6),event["entry_bar_end"],
         event["exit_bar_end"],event["entry_field"],event["exit_field"],
         event["entry_reference_price"],event["exit_reference_price"],
         COMMISSION_BPS,SLIPPAGE_BPS)
    if key not in _EVENT_PATH_CACHE:
        path,status=_event_path_uncached(event,symbol)
        _EVENT_PATH_CACHE[key]=(None if path is None else dict(path),status)
    path,status=_EVENT_PATH_CACHE[key]
    return (None if path is None else dict(path)),status


def _wealth_daily_returns(wealth):
    x=pd.Series(wealth).sort_index().astype(float)
    if x.empty: return {}
    prior=1.0; out={}
    for d,v in x.items(): out[d]=float(v/prior-1); prior=float(v)
    return out


def physical_unit_panels(spec, trades, orders, calendar):
    """Replay unchanged saved signals on real same-direction 1x and 2x ETFs.

    Returns unit-capital daily 1x,2x paths plus per-trade feasibility audit.
    """
    events=normalize_trade_events(spec,trades,orders)
    r1=pd.Series(0.0,index=calendar,name=spec["id"]+"_unit1x")
    r2=pd.Series(0.0,index=calendar,name=spec["id"]+"_unit2x")
    valid1=pd.Series(True,index=calendar); valid2=pd.Series(True,index=calendar)
    audit=[]
    occupied=[]
    for ev in events:
        for a,b in occupied:
            if ev["entry_bar_end"]<b and ev["exit_bar_end"]>a:
                raise ValueError(f"{spec['id']}: overlapping source trades need portfolio-level unit allocation ({ev['trade_id']})")
        occupied.append((ev["entry_bar_end"],ev["exit_bar_end"]))
        p1,s1=_event_path(ev,ev["one_x"])
        if p1 is None:
            days=calendar[(calendar>=ev["entry_date"])&(calendar<=ev["exit_date"])]
            valid1.loc[days]=False
            if ev["two_x"]: valid2.loc[days]=False
            audit.append({"trade_id":ev["trade_id"],"status":"unit1x_infeasible","reason":s1}); continue
        p2=None; s2="no_physical_2x_or_short_inverse2x"
        if ev["two_x"]:
            p2,s2=_event_path(ev,ev["two_x"])
        for day,val in _wealth_daily_returns(p1).items():
            if day in r1.index: r1.loc[day]=(1+r1.loc[day])*(1+val)-1
        if p2 is not None:
            for day,val in _wealth_daily_returns(p2).items():
                if day in r2.index: r2.loc[day]=(1+r2.loc[day])*(1+val)-1
        elif ev["two_x"] is None:
            for day,val in _wealth_daily_returns(p1).items():
                if day in r2.index: r2.loc[day]=(1+r2.loc[day])*(1+val)-1
        elif ev["two_x"]:
            days=calendar[(calendar>=ev["entry_date"])&(calendar<=ev["exit_date"])]
            valid2.loc[days]=False
        audit.append({"trade_id":ev["trade_id"],"entry_date":ev["entry_date"],"exit_date":ev["exit_date"],"unit1x_symbol":ev["one_x"],"unit2x_symbol":ev["two_x"],"status":"ok" if p2 is not None else "unit1x_ok_unit2x_infeasible","reason":s2})
    r1.loc[~valid1]=np.nan; r2.loc[~valid2]=np.nan
    # A short direction with no inverse-2x has feasible 1x only, so 2x history is NA.
    return r1,r2,pd.DataFrame(audit)



def replay_sizing_policies(spec, trades, orders, unit1_history, unit2_history, original_returns):
    """Standalone shadow accounts with frozen signals and causal per-entry sizing."""
    events=normalize_trade_events(spec,trades,orders)
    idx=pd.DatetimeIndex(unit1_history.index).sort_values().unique()
    policies={"S1":pd.Series(0.,index=idx),"S2":pd.Series(0.,index=idx),"S3":pd.Series(0.,index=idx),"S_ETF":pd.Series(0.,index=idx)}
    gap_paths={k:pd.Series(0.,index=idx) for k in policies}
    prepared=[]; audit=[]; previous_exit=None
    for ev in events:
        if previous_exit is not None and ev["entry_bar_end"] < previous_exit:
            raise ValueError(f"{spec['id']}: overlapping trades cannot be replayed as standalone full-capital events ({ev['trade_id']})")
        previous_exit=ev["exit_bar_end"]
        p1,st1=_event_path(ev,ev["one_x"])
        p2,st2=(None,"no_physical_2x")
        if ev["two_x"]: p2,st2=_event_path(ev,ev["two_x"])
        p0,st0=(None,"original_vehicle_missing")
        if p1 is None: p0,st0=_event_path(ev,ev["original_symbol"])
        prepared.append((ev,p1,st1,p2,st2,p0,st0))
    for ev,p1,st1,p2,st2,p0,st0 in prepared:
        hist=unit1_history.loc[unit1_history.index < ev["entry_date"]].dropna()
        if KELLY_WINDOW == "rolling": hist=hist.tail(HISTORY_DAYS)
        ntr=sum(e2["exit_date"] < ev["entry_date"] for e2, *_ in prepared)
        ready=len(hist)>=HISTORY_DAYS and ntr>=MIN_TRADES and hist.var(ddof=1)>1e-12
        if ready:
            mu=float(hist.mean()); var=float(hist.var(ddof=1))
            Ls={f"S{i+1}":float(np.clip(frac*max(0.,mu)/var,0,2)) for i,frac in enumerate(KELLY_FRACTIONS)}
            kelly_status="kelly_ready"
        else:
            kelly_status="warmup_original_sizing_fallback"
            Ls={"S1":1.,"S2":1.,"S3":1.}
        original_symbol=ev["original_symbol"]
        if spec["id"] == "daynight_v1_combined":
            L0=2.0 if original_symbol in ("233740","122630","252670") else 1.0
        else:
            L0=float(np.clip(float(ev["trade"].get("leverage",1.0)),0,2))
        if not ready: Ls={k:L0 for k in ("S1","S2","S3")}
        if p1 is None:
            if p0 is None:
                raise ValueError(f"{spec['id']}/{ev['trade_id']}: cannot mark either 1x or original ETF: {st1}; {st0}")
            # Preserve the original physical vehicle when a 1x replacement is not tradable.
            for pol in policies:
                daily_returns=_wealth_daily_returns(p0)
                for d,r in daily_returns.items():
                    if d in policies[pol].index: policies[pol].loc[d]=(1+policies[pol].loc[d])*(1+r)-1
                if str(ev["trade"].get("strategy",""))=="overnight" and ev["exit_date"] in daily_returns:
                    gap_paths[pol].loc[ev["exit_date"]]=(1+gap_paths[pol].loc[ev["exit_date"]])*(1+daily_returns[ev["exit_date"]])-1
            audit.append({"trade_id":ev["trade_id"],"policy":"S1","L":L0,"status":"original_vehicle_fallback","reason":st1,"kelly_status":kelly_status})
            continue
        if p2 is None and not ready and L0>1 and p0 is not None:
            for pol in ("S1","S2","S3","S_ETF"):
                daily_returns=_wealth_daily_returns(p0)
                for d,r in daily_returns.items():
                    if d in policies[pol].index: policies[pol].loc[d]=(1+policies[pol].loc[d])*(1+r)-1
                if str(ev["trade"].get("strategy",""))=="overnight" and ev["exit_date"] in daily_returns:
                    gap_paths[pol].loc[ev["exit_date"]]=(1+gap_paths[pol].loc[ev["exit_date"]])*(1+daily_returns[ev["exit_date"]])-1
            audit.append({"trade_id":ev["trade_id"],"policy":"S1","L":L0,"status":"warmup_original_vehicle_fallback","reason":st2,"kelly_status":kelly_status})
            continue
        if p2 is None:
            Ls={k:min(1.,v) for k,v in Ls.items()}
            kelly_status += ";physical_cap_1x:"+st2
        L_etf=L0 if not ready else 1.0; etf_status="warmup_original_mix" if not ready else "insufficient_history_or_2x"
        if ready:
            pair=pd.concat([unit1_history,unit2_history],axis=1)
            pair=pair.loc[(pair.index<ev["entry_date"])].dropna().tail(HISTORY_DAYS)
            if len(pair)==HISTORY_DAYS:
                from scipy.optimize import minimize_scalar
                hist1=pair.iloc[:,0].to_numpy(float); hist2=pair.iloc[:,1].to_numpy(float)
                upper=2.0 if p2 is not None else 1.0
                def objective(L):
                    w1,w2=(L,0.) if L<=1 else (2-L,L-1)
                    R=w1*hist1+w2*hist2
                    return -float(np.log1p(R).sum()) if np.all(R>-1) else 1e9
                segments=[(0.,1.)] if upper<=1 else [(0.,1.),(1.,2.)]
                candidates=[(objective(x),float(x)) for x in (0.,upper)]
                for lo,hi in segments:
                    opt=minimize_scalar(objective,bounds=(lo,hi),method="bounded",options={"xatol":1e-5})
                    if opt.success: candidates.append((float(opt.fun),float(opt.x)))
                _,L_etf=min(candidates,key=lambda z:(z[0],z[1])); etf_status="physical_1x_2x_log_growth" if p2 is not None else "physical_1x_cash_log_growth_capped1"
        Ls["S_ETF"]=L_etf
        for policy,L in Ls.items():
            w1,w2=(L,0.) if L<=1 else (2-L,L-1)
            if w2>0 and p2 is None: L=min(L,1.); w1,w2=L,0.
            cash=max(0.,1-w1-w2); previous=1.0
            event_returns={}
            event_days=sorted(set(p1)| (set(p2) if p2 else set()))
            for day in event_days:
                # Hold the same ETF units for the entire signal; no daily rebalance.
                wealth=cash+w1*p1.get(day,1.0)+w2*(p2.get(day,1.0) if p2 else 1.0)
                ret=wealth/previous-1
                event_returns[day]=ret
                if day in policies[policy].index: policies[policy].loc[day]=(1+policies[policy].loc[day])*(1+ret)-1
                previous=wealth
            if str(ev["trade"].get("strategy",""))=="overnight" and ev["exit_date"] in event_returns:
                gap_paths[policy].loc[ev["exit_date"]]=(1+gap_paths[policy].loc[ev["exit_date"]])*(1+event_returns[ev["exit_date"]])-1
            buy_factor=1/((1+SLIPPAGE_BPS/10000)*(1+COMMISSION_BPS/10000))
            sell_factor=(1-SLIPPAGE_BPS/10000)*(1-COMMISSION_BPS/10000)
            components=[(w1,ev["one_x"])]
            if w2>0 and p2 is not None: components.append((w2,ev["two_x"]))
            entry_cost=exit_cost=turnover=0.
            for weight,symbol in components:
                ratio=_event_price_ratio(ev,symbol)
                if not np.isfinite(ratio): continue
                entry_cost+=weight*(1-buy_factor)
                exit_cost+=weight*buy_factor*ratio*(1-sell_factor)
                turnover+=weight*buy_factor*((1+SLIPPAGE_BPS/10000)+ratio*(1-SLIPPAGE_BPS/10000))
            audit.append({"trade_id":ev["trade_id"],"policy":policy,"entry_date":ev["entry_date"],"exit_date":ev["exit_date"],"one_x_symbol":ev["one_x"],"two_x_symbol":ev["two_x"],"history_days":len(hist),"closed_trades_prior":ntr,"kelly_status":kelly_status,"L":L,"target_gross_exposure_at_entry":w1+2*w2,"turnover_fraction_of_NAV":turnover,"entry_cost_fraction_of_NAV":entry_cost,"exit_cost_fraction_of_NAV":exit_cost,"status":"ok","S_ETF_status":etf_status if policy=="S_ETF" else "not_applicable"})
    return policies,gap_paths,pd.DataFrame(audit)



def replay_fixed_exposure(spec,trades,orders,calendar,original_returns,leverage):
    """S0 risk-matched control using a leverage mean frozen from development only."""
    events=normalize_trade_events(spec,trades,orders)
    out=pd.Series(0.,index=calendar,name=spec["id"]+"_S0_MATCHED_DEV"); gap=pd.Series(0.,index=calendar)
    audit=[]; prior_exit=None
    for ev in events:
        if prior_exit is not None and ev["entry_bar_end"]<prior_exit:
            raise ValueError(f"{spec['id']}: overlapping trades prevent matched-exposure replay")
        prior_exit=ev["exit_bar_end"]
        p1,st1=_event_path(ev,ev["one_x"]); p2=None
        if ev["two_x"]: p2,_=_event_path(ev,ev["two_x"])
        if p1 is None:
            p0,_=_event_path(ev,ev["original_symbol"])
            if p0 is None:
                raise ValueError(f"{spec['id']}/{ev['trade_id']}: missing physical marks for fixed-exposure control")
            daily=_wealth_daily_returns(p0); gross_multiple=np.nan
        else:
            L=min(float(leverage),2.)
            if p2 is None: L=min(L,1.)
            w1,w2=(L,0.) if L<=1 else (2-L,L-1); cash=1-w1-w2
            daily={}; prev=1.
            for d in sorted(set(p1)|(set(p2) if p2 else set())):
                wealth=cash+w1*p1.get(d,1.)+w2*(p2.get(d,1.) if p2 else 1.)
                daily[d]=wealth/prev-1; prev=wealth
            gross_multiple=w1+2*w2
        for d,r in daily.items():
            if d in out.index: out.loc[d]=(1+out.loc[d])*(1+r)-1
        if str(ev["trade"].get("strategy",""))=="overnight" and ev["exit_date"] in daily:
            gap.loc[ev["exit_date"]]=(1+gap.loc[ev["exit_date"]])*(1+daily[ev["exit_date"]])-1
        audit.append({"trade_id":ev["trade_id"],"entry_date":ev["entry_date"],"exit_date":ev["exit_date"],"status":"ok","L_dev":leverage,"realized_gross_exposure":gross_multiple})
    return out,gap,pd.DataFrame(audit)

def metric_row(r, policy, window, allow_short=False):
    x = pd.Series(r).dropna().astype(float)
    if x.empty: return {"policy":policy,"window":window,"days":0}
    nav=(1+x).cumprod(); peak=pd.concat([pd.Series([1.]),nav.reset_index(drop=True)]).cummax().iloc[1:].to_numpy()
    dd=nav.to_numpy()/peak-1; years=len(x)/252
    cagr=(nav.iloc[-1]**(1/years)-1) if (years>=1 or allow_short) and nav.iloc[-1]>0 else np.nan
    return {"policy":policy,"window":window,"days":len(x),"start":str(x.index[0].date()),"end":str(x.index[-1].date()),"total_return":float(nav.iloc[-1]-1),"cagr":float(cagr) if np.isfinite(cagr) else np.nan,"mdd":float(dd.min()),"calmar":float(cagr/abs(dd.min())) if np.isfinite(cagr) and dd.min()<0 else np.nan,"sharpe":float(x.mean()/x.std(ddof=1)*np.sqrt(252)) if len(x)>1 and x.std(ddof=1)>0 else np.nan,"var95_daily":float(x.quantile(.05)),"es95_daily":float(x[x<=x.quantile(.05)].mean()),"log_growth":float(np.log1p(x).sum()) if (x>-1).all() else -np.inf}


def lp_weights(R, cap, max_iter=25):
    from scipy.optimize import linprog
    R=np.asarray(R,float); n=R.shape[1]
    U=np.vstack([np.ones(n),np.cumprod(1+R,axis=0)])
    if not np.isfinite(U).all() or (U<=0).any(): return None,"invalid_compounded_path"
    rows=[]; rhs=[]
    # Constraint generation over drawdown pairs V_t >= (1-cap)V_s.
    for _ in range(max_iter):
        A=np.asarray(rows) if rows else np.empty((0,n)); b=np.asarray(rhs)
        base_A = [np.ones(n)]
        base_b = [1.0]
        if len(A): base_A.extend(A); base_b.extend(b)
        res=linprog(-(U[-1]-1), A_ub=np.asarray(base_A), b_ub=np.asarray(base_b), bounds=[(0,1)]*n, method="highs")
        if not res.success: return None,"solver:"+res.message
        v=1+(U[1:]-1)@res.x
        v=np.r_[1.0,v]  # V0=1 is included in every peak-to-trough check.
        peak_before=np.r_[-np.inf,np.maximum.accumulate(v)[:-1]]
        deficits=v-(1-cap)*peak_before
        deficits[0]=0.
        t=int(np.argmin(deficits))
        if deficits[t]>=-1e-8: return res.x,"optimal"
        s=int(np.argmax(v[:t]))
        row=(1-cap)*(U[s]-1)-(U[t]-1)
        rows.append(row); rhs.append(cap)
    return res.x,"constraint_generation_limit"


def convex_weights(R, cap, max_iter=30):
    import cvxpy as cp
    R=np.asarray(R,float); n=R.shape[1]; w=cp.Variable(n,nonneg=True)
    wealth=1+R@w
    if np.any(1+R.min(axis=0)<0): return None,"invalid_input_domain"
    constraints=[cp.sum(w)<=1,wealth>=1e-8]
    # Add violated peak/trough log intervals iteratively.
    added=set()
    for _ in range(max_iter):
        problem=cp.Problem(cp.Maximize(cp.sum(cp.log(wealth))),constraints)
        try: problem.solve(solver="CLARABEL",verbose=False,tol_gap_abs=1e-8,tol_gap_rel=1e-8,tol_feas=1e-8,max_iter=500)
        except Exception as e: return None,"solver:"+str(e)
        if w.value is None or problem.status not in ("optimal","optimal_inaccurate"): return None,"status:"+str(problem.status)
        value=np.asarray(w.value).reshape(-1)
        if not np.isfinite(value).all() or value.min() < -1e-7 or value.sum() > 1+1e-7:
            return None,"infeasible_weight_budget"
        value=np.maximum(value,0.)
        if value.sum()>1: value/=value.sum()
        growth=1+R@value
        if np.any(growth<=0): return None,"nonpositive_growth"
        vals=np.log(growth); cum=np.r_[0,np.cumsum(vals)]
        peak_before=np.r_[-np.inf,np.maximum.accumulate(cum)[:-1]]
        deficits=cum-peak_before
        t=int(np.argmin(deficits[1:]))+1
        if deficits[t]>=math.log(1-cap)-1e-7: return value,str(problem.status)
        s=int(np.argmax(cum[:t]))
        if (s,t) in added: return np.asarray(w.value).reshape(-1),"unresolved_drawdown_constraint"
        added.add((s,t)); constraints.append(cp.sum(cp.log(wealth[s:t]))>=math.log(1-cap))
    return np.asarray(w.value).reshape(-1),"constraint_generation_limit"



def candidate_schedule(panel, method, cap, ids, dates=None, fixed_universe=False):
    """Causal EOD targets, retaining pre-comparison history for warm-up."""
    rows=[]; vectors={}; last=np.r_[np.zeros(len(ids)),1.]
    for day in (panel.index if dates is None else pd.DatetimeIndex(dates)):
        eligible=(np.ones(len(ids),dtype=bool) if fixed_universe else
                  np.array([pd.notna(panel.loc[day,s]) and
                            (method=="EW" or panel.loc[:day,s].notna().sum()>=HISTORY_DAYS)
                            for s in ids]))
        active=[sid for sid,ok in zip(ids,eligible) if ok]
        status="scheduled"; cand=np.r_[np.zeros(len(ids)),1.]; hist=pd.DataFrame()
        if method=="EW":
            if active:
                for sid in active: cand[ids.index(sid)]=1/len(active)
                cand[-1]=0.
        elif active:
            hist=panel.loc[panel.index<=day,active].dropna(how="any").tail(HISTORY_DAYS)
            if len(hist)>=HISTORY_DAYS:
                R=hist.to_numpy(float)
                if method=="LP": w,status=lp_weights(R,cap)
                else: w,status=convex_weights(R,cap)
                if w is not None and status in ("optimal","optimal_inaccurate"):
                    for sid,value in zip(active,w): cand[ids.index(sid)]=float(value)
                    cand[-1]=1-float(np.sum(w))
                else: cand=last.copy()
            else:
                status="insufficient_complete_common_history"; cand=last.copy()
        else: status="no_eligible_sleeves"
        vectors[day]=cand.copy()
        rows.append({"date":day,"method":method,"mdd_cap":cap,"candidate":json.dumps(cand.tolist()),"solver_status":status,"eligible":json.dumps(active),"estimator_start":str(hist.index.min()) if len(hist) else "","estimator_end":str(hist.index.max()) if len(hist) else "","estimator_observations":len(hist)})
        last=cand.copy()
    return vectors,pd.DataFrame(rows)


def simulate_funding(panel, gap_panel, vectors, band, policy_name, ids, no_transfer=False, initial_weights=None, resize_proxy=False):
    """Segregated account + treasury daily funding surrogate.

    Old holdings first earn the physical overnight gap. A previously-triggered
    EOD target is transferred after that gap, before the day's remaining return.
    Transfers net to zero and contribute no P&L. T+2 and resize fills are outside
    this daily funding surrogate.
    """
    n=len(ids)
    if no_transfer:
        initial=np.asarray(initial_weights if initial_weights is not None else np.r_[np.repeat(1./n,n),0.],float)
        if len(initial)!=n+1 or np.any(initial<0) or not np.isclose(initial.sum(),1.): raise ValueError("invalid no-transfer initial weights")
        accounts=initial[:n].copy(); treasury=float(initial[-1])
    else: accounts=np.zeros(n); treasury=1.
    dates=panel.index; daily=[]; flows=[]; account_rows=[]; contrib=[]; targets=[]
    last_target=np.r_[accounts,treasury]; pending=None
    for k,day in enumerate(dates):
        start_nav=float(accounts.sum()+treasury)
        daily_pnl=np.zeros(n); twr_gap=np.ones(n); twr_rest=np.ones(n); transfer_in=np.zeros(n); transfer_out=np.zeros(n); resize_cost=0.0
        account_cost=np.zeros(n); treasury_cost=0.; executed_turnover=0.
        proxy_delta=np.zeros(n); proxy_fee=np.zeros(n); proxy_slippage=np.zeros(n)
        gap=gap_panel.loc[day,ids].to_numpy(float) if day in gap_panel.index else np.zeros(n)
        ret=panel.loc[day,ids].to_numpy(float)
        for i,sid in enumerate(ids):
            if accounts[i]>1e-12:
                if not np.isfinite(gap[i]) or not np.isfinite(ret[i]): raise ValueError(f"{policy_name}: funded {sid} missing return/gap mark at {day.date()}")
                gp=accounts[i]*gap[i]; accounts[i]+=gp; daily_pnl[i]+=gp; twr_gap[i]=1+gap[i]
        # EOD decision from prior session executes only after old positions earn today's gap.
        if pending is not None and not no_transfer:
            current_nav=float(accounts.sum()+treasury); before=np.r_[accounts,treasury].copy(); desired_gross=pending*current_nav
            traded_delta=desired_gross[:n]-before[:n]
            executed_turnover=.5*float(np.abs(np.r_[traded_delta,desired_gross[-1]-before[-1]]).sum())/current_nav
            if resize_proxy:
                proxy_delta=traded_delta.copy()
                proxy_notional=np.abs(proxy_delta)
                proxy_fee=proxy_notional*COMMISSION_BPS/10000.
                proxy_slippage=proxy_notional*SLIPPAGE_BPS/10000.
                account_cost=proxy_fee+proxy_slippage
                net_nav=current_nav-float(account_cost.sum())
                if net_nav<=0: raise ValueError(f"{policy_name}: resize proxy exhausts book NAV")
                desired_net=pending*net_nav
                # Charge each sleeve's proxy cost, with the internal flow
                # restoring the post-cost target budget. Sellers can fund a
                # fee from sale proceeds without creating a negative sleeve.
                accounts=desired_net[:n]+account_cost
                treasury=float(desired_net[-1])
                funding_delta=np.r_[accounts,treasury]-before
                if not np.isclose(funding_delta.sum(),0.,atol=1e-9): raise AssertionError("fee-adjusted internal funding did not net to zero")
                transfer_in=np.maximum(funding_delta[:n],0.); transfer_out=np.maximum(-funding_delta[:n],0.)
                for i,sid in enumerate(ids):
                    if abs(funding_delta[i])>1e-14: flows.append({"date":day,"policy":policy_name,"from_account":"TREASURY" if funding_delta[i]>0 else sid,"to_account":sid if funding_delta[i]>0 else "TREASURY","amount":abs(float(funding_delta[i])),"net_book_cashflow":0.,"status":"TWR_SURROGATE_UNSETTLED_TRANSFER"})
                twr_rest*=np.where(accounts>0,1-account_cost/np.maximum(accounts,1e-300),1.)
                accounts-=account_cost
                daily_pnl-=account_cost
                resize_cost=float(account_cost.sum())
            else:
                accounts=desired_gross[:n].copy(); treasury=float(desired_gross[-1])
                funding_delta=np.r_[accounts,treasury]-before
                transfer_in=np.maximum(funding_delta[:n],0.); transfer_out=np.maximum(-funding_delta[:n],0.)
                for i,sid in enumerate(ids):
                    if abs(funding_delta[i])>1e-14: flows.append({"date":day,"policy":policy_name,"from_account":"TREASURY" if funding_delta[i]>0 else sid,"to_account":sid if funding_delta[i]>0 else "TREASURY","amount":abs(float(funding_delta[i])),"net_book_cashflow":0.,"status":"TWR_SURROGATE_UNSETTLED_TRANSFER"})
                if not np.isclose(funding_delta.sum(),0.,atol=1e-9): raise AssertionError("internal funding transfer did not net to zero")
            if not resize_proxy and CENTRAL_RESIZE_COST_BPS>0:
                resize_cost=executed_turnover*current_nav*CENTRAL_RESIZE_COST_BPS/10000.
                remaining_cost=resize_cost
                treasury_cost=min(treasury,remaining_cost); treasury-=treasury_cost; remaining_cost-=treasury_cost
                if remaining_cost>0:
                    total_accounts=accounts.sum()
                    if total_accounts<=remaining_cost: raise ValueError(f"{policy_name}: resize cost proxy exceeds available cash")
                    account_cost=accounts*(remaining_cost/total_accounts)
                    twr_rest*=np.where(accounts>0,1-remaining_cost/total_accounts,1.)
                    accounts-=account_cost; daily_pnl-=account_cost
            last_target=pending.copy(); pending=None
        # Recover the post-gap remainder so full daily return remains exact.
        for i,sid in enumerate(ids):
            if accounts[i]>1e-12:
                if not np.isfinite(ret[i]) or not np.isfinite(gap[i]): raise ValueError(f"{policy_name}: funded {sid} missing mark at {day.date()}")
                if 1+gap[i]<=0: raise ValueError(f"{policy_name}: invalid gap return for {sid} on {day.date()}")
                rem=(1+ret[i])/(1+gap[i])-1
                if accounts[i]>0: twr_rest[i]*=1+rem
                rp=accounts[i]*rem; accounts[i]+=rp; daily_pnl[i]+=rp
        book_nav=float(accounts.sum()+treasury); book_return=book_nav/start_nav-1 if start_nav>0 else np.nan
        pnl=daily_pnl
        weights=np.r_[accounts,treasury]/book_nav if book_nav>0 else np.r_[np.zeros(n),1.]
        candidate=vectors.get(day,last_target).copy()
        if no_transfer: candidate=weights.copy()
        distance=.5*float(np.abs(candidate-weights).sum())
        trigger=distance>band and not no_transfer
        if trigger: pending=candidate.copy()
        if no_transfer: candidate=weights.copy(); distance=0.; trigger=False
        for i,sid in enumerate(ids):
            account_rows.append({"date":day,"policy":policy_name,"strategy_id":sid,"account_equity":float(accounts[i]),"daily_pnl":float(pnl[i]),"time_weighted_return":float(twr_gap[i]*twr_rest[i]-1),"transfer_in":float(transfer_in[i]),"transfer_out":float(transfer_out[i]),"resize_cost_proxy":float(account_cost[i]),"budget_delta_signed":float(proxy_delta[i]),"proxy_buy_notional":float(max(proxy_delta[i],0.)),"proxy_sell_notional":float(max(-proxy_delta[i],0.)),"proxy_fee":float(proxy_fee[i]),"proxy_slippage":float(proxy_slippage[i]),"funding_status":"TWR_SURROGATE"})
            contrib.append({"date":day,"policy":policy_name,"strategy_id":sid,"contribution":float(pnl[i])})
        account_rows.append({"date":day,"policy":policy_name,"strategy_id":"TREASURY","account_equity":treasury,"daily_pnl":-treasury_cost,"resize_cost_proxy":treasury_cost,"funding_status":"TWR_SURROGATE"})
        contrib.append({"date":day,"policy":policy_name,"strategy_id":"TREASURY","contribution":-treasury_cost})
        if not np.isclose(book_nav-start_nav,float(pnl.sum())-treasury_cost,rtol=1e-9,atol=1e-10):
            raise AssertionError("Funding/P&L reconciliation failed")
        team_weights=weights[:n]
        ordered=np.sort(team_weights)[::-1]
        invested=float(team_weights.sum())
        concentration=float(np.square(team_weights/invested).sum()) if invested>0 else 0.
        targets.append({"date":day,"policy":policy_name,"candidate":json.dumps(candidate.tolist()),"last_applied":json.dumps(last_target.tolist()),"actual_equity_weight":json.dumps(weights.tolist()),"half_l1_distance":distance,"rebalance_trigger":trigger,"pending_target_next_session":json.dumps(pending.tolist()) if pending is not None else "","band":band,"top1_weight":float(ordered[0]) if len(ordered) else 0.,"top2_weight":float(ordered[:2].sum()),"effective_team_count":float(1/concentration) if concentration>0 else 0.})
        for j,sid in enumerate(ids+["TREASURY"]):
            targets[-1]["target_"+sid]=float(candidate[j])
            targets[-1]["actual_"+sid]=float(weights[j])
            targets[-1]["last_applied_"+sid]=float(last_target[j])
        daily.append({"date":day,"policy":policy_name,"book_return":book_return,"book_nav":book_nav,"market_pnl":float(pnl.sum()+account_cost.sum()),"rebalance_turnover":executed_turnover,"resize_cost_proxy":resize_cost,"treasury_equity":treasury})
    return pd.DataFrame(daily),pd.DataFrame(account_rows),pd.DataFrame(flows),pd.DataFrame(contrib),pd.DataFrame(targets)



def valuation_calendar(panel):
    """Choose an observed KRX valuation clock within common or staggered boundaries."""
    valid_starts=[panel[s].first_valid_index() for s in panel]
    valid_ends=[panel[s].last_valid_index() for s in panel]
    if any(x is None for x in valid_starts+valid_ends): raise ValueError("strategy has no valid source returns")
    if COMMON_CALENDAR:
        lower=max(valid_starts); upper=min(valid_ends)
    else:
        lower=min(valid_starts); upper=max(valid_ends)
    dates=panel.index[(panel.index>=lower)&(panel.index<=upper)]
    if dates.empty: raise ValueError("no observed KRX valuation dates in configured calendar bounds")
    return dates


def execution_returns(panel, dates, registry):
    """Convert only same-day missing source sessions to explicit cash suspension.

    Original NA remains untouched for estimation. A missing daynight daily mark is
    fatal because an overnight holding may be open; it cannot be called flat.
    """
    x=panel.reindex(dates).copy(); susp=[]
    for sid in x:
        missing=x[sid].isna()
        policy=next((s for s in registry if s["id"]==sid),{})
        if missing.any() and not policy.get("cash_when_missing",False):
            first=panel[sid].first_valid_index(); last=panel[sid].last_valid_index()
            interior=missing & (x.index>=first) & (x.index<=last)
            if interior.any():
                ds=x.index[interior].strftime("%Y-%m-%d").tolist()
                raise ValueError(f"{sid} has missing daily mark while it may carry a position: {ds[:5]}")
        if missing.any():
            for d in x.index[missing]: susp.append({"date":d,"strategy_id":sid,"execution_treatment":"suspended_cash_no_entry","estimation_value":"NA_preserved"})
            x.loc[missing,sid]=0.0
    return x,pd.DataFrame(susp,columns=["date","strategy_id","execution_treatment","estimation_value"])


BEST_OUTPUT_DIR = ROOT / "results/krx_best_policies_20261102"
BEST_POLICIES = ("S0", "S1", "S2", "S3", "S_ETF")



def _best_metrics(r, name, window):
    row=metric_row(r,name,window,allow_short=True)
    row["short_sample_annualization"]="252 / observed sessions; exploratory" if row.get("days",0)<252 else "252 / observed sessions"
    return row


def _best_write_csv(frame, name):
    frame.to_csv(BEST_OUTPUT_DIR/name,index=False)


def _compress_zero_rows(returns):
    """Drop all-sleeve zero observations only after the complete lookback is formed."""
    R=np.asarray(returns,dtype=float)
    if R.ndim!=2: raise ValueError("returns must be a 2D matrix")
    keep=np.any(R!=0.,axis=1)
    return np.ascontiguousarray(R[keep],dtype="<f8"),int(keep.sum())


def _wf_equal_weights(active_ids, all_ids):
    out=np.zeros(len(all_ids)+1,dtype=float)
    if active_ids:
        w=1./len(active_ids)
        for sid in active_ids: out[all_ids.index(sid)]=w
    else: out[-1]=1.
    return out


def staggered_candidate_schedule(panel, ids, developed, method, cap, dates,
                                 min_observations=WF_MIN_OBSERVATIONS,
                                 solver_cache=None):
    """EW admits immediately; optimizers incubate entrants while old teams continue."""
    cache={} if solver_cache is None else solver_cache
    rows=[]; vectors={}; last=np.r_[np.zeros(len(ids)),1.]
    admitted=[]; admitted_at={}; previous_known=[]; logged_year=None
    valid_dates={sid:panel.index[(panel.index>=developed[sid]) & panel[sid].notna()] for sid in ids}
    ready_at={sid:(days[min_observations-1] if len(days)>=min_observations else pd.Timestamp.max)
              for sid,days in valid_dates.items()}
    for day in pd.DatetimeIndex(dates):
        if logged_year!=day.year:
            print(f"[staggered] {method} cap={cap:.2f} scheduling {day.year}",flush=True)
            logged_year=day.year
        known=[sid for sid in ids if developed[sid]<=day]
        known_entrants=[sid for sid in known if sid not in previous_known]
        previous_known=known.copy()
        entrants=[]
        if method=="EW":
            active=known.copy(); entrants=known_entrants
            for sid in entrants: admitted_at[sid]=day
        else:
            # An immature entrant must not suspend an already operational optimizer roster.
            newcomers=sorted((sid for sid in known if sid not in admitted and ready_at[sid]<=day),
                             key=lambda sid:(ready_at[sid],developed[sid],sid))
            for sid in newcomers:
                trial=admitted+[sid]
                joint=panel.loc[(panel.index>=max(developed[x] for x in trial)) &
                                (panel.index<=day),trial].dropna(how="any")
                if len(joint)>=min_observations:
                    admitted.append(sid); admitted_at[sid]=day; entrants.append(sid)
            active=admitted.copy()
        newest=max((developed[sid] for sid in active),default=day)
        obs=0; solve_rows=0; window_start=""; window_end=""
        candidate=_wf_equal_weights(active,ids)
        status="equal_weight" if method=="EW" else "warmup_cash"
        if method in ("LP","CONVEX") and active:
            hist=panel.loc[(panel.index>=newest)&(panel.index<=day),active].dropna(how="any").tail(min_observations)
            obs=len(hist)
            if obs<min_observations: raise AssertionError("admitted optimizer roster lacks its required history")
            window_start=str(hist.index[0].date()); window_end=str(hist.index[-1].date())
            compact,solve_rows=_compress_zero_rows(hist.to_numpy(float))
            if not solve_rows:
                status="all_zero_objective_equal_weight"
            else:
                key=(method,float(cap),tuple(active),compact.shape,compact.tobytes())
                cached=cache.get(key); cache_hit=cached is not None
                if cached is None:
                    result,solve_status=(lp_weights(compact,cap) if method=="LP" else convex_weights(compact,cap))
                    cached=(None if result is None else np.asarray(result,float),str(solve_status))
                    cache[key]=cached
                result,solve_status=cached
                if result is not None and solve_status in ("optimal","optimal_inaccurate"):
                    candidate=np.r_[np.zeros(len(ids)),1.]
                    for sid,w in zip(active,result): candidate[ids.index(sid)]=float(w)
                    status=("cache_hit_" if cache_hit else "optimized_")+str(solve_status)
                else:
                    candidate=last.copy()
                    for sid in set(ids)-set(active): candidate[ids.index(sid)]=0.
                    status="solver_failure_hold:"+str(solve_status)
        # No undeveloped or untrained team receives an optimizer target, even during failures.
        for sid in set(ids)-set(active):
            if abs(candidate[ids.index(sid)])>1e-12:
                raise AssertionError(f"ineligible sleeve {sid} received capital target on {day.date()}")
        if not np.isfinite(candidate).all() or np.any(candidate < -1e-9) or candidate[:-1].sum()>1+1e-7:
            raise AssertionError("invalid staggered target budget")
        candidate=np.maximum(candidate,0.)
        if candidate[:-1].sum()>1: candidate[:-1]/=candidate[:-1].sum()
        candidate[-1]=max(0.,1.-candidate[:-1].sum())
        vectors[day]=candidate.copy()
        rows.append({"date":day,"method":method,"mdd_cap":cap,"candidate":json.dumps(candidate.tolist()),
            "solver_status":status,"known_roster":json.dumps(known),"known_entrants":json.dumps(known_entrants),
            "eligible_roster":json.dumps(active),"entrants":json.dumps(entrants),
            "developed_count":len(known),"eligible_count":len(active),
            "newest_member_developed":str(newest.date()) if active else "",
            "lookback_valid_observations":obs,"nonzero_solve_rows":solve_rows,
            "estimator_start":window_start,"estimator_end":window_end,
            "strategy_eligibledate":json.dumps({sid:str(admitted_at[sid].date()) for sid in active}),
            **{f"target_{sid}":float(candidate[i]) for i,sid in enumerate(ids)},"target_TREASURY":float(candidate[-1])})
        last=candidate.copy()
    return vectors,pd.DataFrame(rows),cache


def _wf_resolve_availability(raw):
    """Resolve counterfactual admission dates from saved, locally available source data."""
    rows=[]; resolved={}
    for spec in STRATEGIES:
        if not spec.get("enabled"): continue
        sid=spec["id"]; data_first=raw[sid].first_valid_index()
        if data_first is None: raise ValueError(f"{sid}: no valid saved S0 return history")
        feasible_first=data_first
        if sid=="beomgu_c3_vt30_etf":
            src=pd.read_csv(_path(spec["daily_path"]),usecols=["date","r_122630_oc"])
            src["date"]=pd.to_datetime(src.date,errors="raise").dt.normalize()
            src["r_122630_oc"]=pd.to_numeric(src.r_122630_oc,errors="coerce")
            feasible=src.loc[src.r_122630_oc.notna(),"date"]
            if feasible.empty: raise ValueError("Beomgu has no observed physical 2x ETF return")
            feasible_first=max(pd.Timestamp(data_first),pd.Timestamp(feasible.min()))
        override=WF_DEVELOPED_FROM.get(sid)
        developed=pd.Timestamp(override).normalize() if override is not None else pd.Timestamp(feasible_first).normalize()
        if developed<feasible_first: raise ValueError(f"{sid}: assumed development {developed.date()} precedes feasible data {feasible_first.date()}")
        resolved[sid]=developed
        rows.append({"strategy_id":sid,"data_first":str(pd.Timestamp(data_first).date()),
            "feasible_first":str(pd.Timestamp(feasible_first).date()),
            "assumed_developed":str(developed.date()),"last_source":str(raw[sid].last_valid_index().date()),
            "first_eod_roster":"","next_session_entry":"","first_actual_funding":""})
    return resolved,pd.DataFrame(rows)


def _wf_daynight_gaps(raw, trades):
    gaps=pd.DataFrame(0.,index=raw.index,columns=raw.columns)
    sid="daynight_v1_combined"
    if sid not in trades: return gaps
    for _,tr in trades[sid].iterrows():
        if str(tr.get("strategy",""))!="overnight": continue
        exit_day=pd.Timestamp(tr.exit_time).normalize(); entry_day=pd.Timestamp(tr.entry_time).normalize()
        sym=str(tr.symbol); sym=sym[:-2] if sym.endswith(".0") else sym; sym=sym.zfill(6)
        mark=_bar_price(sym,entry_day+pd.Timedelta(hours=15,minutes=30),"close",False)
        qty=float(tr.quantity); prev_nav=qty*mark if mark is not None else np.nan
        if mark is None or not np.isfinite(prev_nav) or prev_nav<=0:
            raise ValueError(f"daynight overnight gap mark missing {exit_day.date()} {sym}")
        gap=(qty*(float(tr.exit_fill)-mark)-float(tr.exit_commission))/prev_nav
        if exit_day in gaps.index: gaps.loc[exit_day,sid]+=gap
    return gaps


def run_staggered_walkforward(output_dir=WF_OUTPUT_DIR):
    """Run a long, counterfactual expanding-universe S0 allocation study."""
    out=Path(output_dir); out.mkdir(parents=True,exist_ok=True)
    t0=time.time()
    if WF_NEW_STRATEGY_RULE!="mature_only": raise ValueError(f"unsupported WF_NEW_STRATEGY_RULE={WF_NEW_STRATEGY_RULE}")
    if WF_MIN_OBSERVATIONS<2: raise ValueError("WF_MIN_OBSERVATIONS must be at least two")
    print("[staggered] loading saved S0 strategy histories",flush=True)
    raw,provenance,trades,orders=load_universe()
    ids=list(raw.columns)
    enabled=[s for s in STRATEGIES if s.get("enabled")]
    if set(ids)!={s["id"] for s in enabled}: raise ValueError("source panel and enabled registry differ")
    developed,availability=_wf_resolve_availability(raw)
    wf_end=min(pd.Timestamp(WF_END),min(raw[s].last_valid_index() for s in ids))
    wf_start=min(developed.values())
    calendar=raw.index[(raw.index>=wf_start)&(raw.index<=wf_end)]
    if len(calendar)<WF_MIN_OBSERVATIONS: raise ValueError("observed KRX calendar shorter than warm-up")
    panel=raw.copy()
    for sid in ids: panel.loc[panel.index<developed[sid],sid]=np.nan
    gaps=_wf_daynight_gaps(raw,trades)
    exec_panel,susp=execution_returns(panel.reindex(calendar),calendar,enabled)
    # Pre-development returns remain NA for fitting and are not called suspensions.
    if not susp.empty:
        susp=susp[susp.apply(lambda r: r.date>=developed[r.strategy_id],axis=1)].copy()
    timeline=[]; previous_roster=[]
    for day in calendar:
        roster=[sid for sid in ids if developed[sid]<=day]
        entrants=[sid for sid in roster if sid not in previous_roster]
        previous_roster=roster.copy()
        timeline.append({"date":day,"eligible_roster":json.dumps(roster),"entrants":json.dumps(entrants),
            "eligible_count":len(roster),"newest_member_developed":str(max(developed[s] for s in roster).date()) if roster else ""})
    timeline=pd.DataFrame(timeline)
    timeline["era_id"]=timeline.eligible_roster.ne(timeline.eligible_roster.shift()).cumsum().astype(int)
    for i,row in timeline.iterrows():
        availability.loc[availability.strategy_id.isin(json.loads(row.eligible_roster)) & availability.first_eod_roster.eq(""),"first_eod_roster"]=str(row.date.date())
    next_session={}
    for sid,date in developed.items():
        at=int(calendar.searchsorted(date,side="left"))
        if at>=len(calendar): raise ValueError(f"{sid}: development after evaluation end")
        eod=calendar[at]
        next_session[sid]=calendar[at+1] if at+1<len(calendar) else pd.NaT
        availability.loc[availability.strategy_id.eq(sid),"first_eod_roster"]=str(eod.date())
        availability.loc[availability.strategy_id.eq(sid),"next_session_entry"]=str(next_session[sid].date()) if pd.notna(next_session[sid]) else ""
    # Build the auditable, source-specific cache identity only after price/gap inputs are loaded.
    gap_hash=hashlib.sha256(np.ascontiguousarray(gaps.reindex(calendar).fillna(0.).to_numpy(dtype="<f8")).tobytes()).hexdigest()
    input_manifest={"code_sha256":sha256(Path(__file__)),"kernel_version":WF_KERNEL_VERSION,
        "source_provenance":provenance,"etf_minute_inputs":ETF_PRICE_INPUTS,
        "gap_matrix_sha256":gap_hash,"developed_from":{k:str(v.date()) for k,v in developed.items()},
        "new_strategy_rule":WF_NEW_STRATEGY_RULE,"min_observations":WF_MIN_OBSERVATIONS,
        "mdd_caps":WF_MDD_CAPS,"bands":WF_BANDS,"start":str(calendar[0].date()),
        "end":str(calendar[-1].date()),"commission_bps":COMMISSION_BPS,"slippage_bps":SLIPPAGE_BPS,
        "resize_cost_basis":"sum(abs(sleeve budget delta))*(fee+slippage); cash transfer free"}
    cache_identity=hashlib.sha256(json.dumps(input_manifest,sort_keys=True,default=str).encode()).hexdigest()
    cache_dir=out/"candidate_cache"; cache_dir.mkdir(parents=True,exist_ok=True)
    methods=[("EW",0.0)]+[(method,cap) for method in ("LP","CONVEX") for cap in WF_MDD_CAPS]
    all_daily=[]; all_accounts=[]; all_flows=[]; all_targets=[]; all_resize=[]; all_schedules=[]
    completed=[]; solver_cache_hits=0; optimization_days=0; warmup_days=0; solver_failures=0
    for method,cap in methods:
        stem=f"{method}_cap{cap:.2f}"; cache_csv=cache_dir/f"{stem}.csv"; cache_json=cache_dir/f"{stem}.json"
        vectors={}; schedule=None; schedule_key=hashlib.sha256(f"{cache_identity}|{method}|{cap}|{','.join(ids)}".encode()).hexdigest()
        if cache_csv.is_file() and cache_json.is_file():
            try: cache_match=json.loads(cache_json.read_text()).get("schedule_key")==schedule_key
            except (OSError,ValueError): cache_match=False
        else: cache_match=False
        if cache_match:
            schedule=pd.read_csv(cache_csv,parse_dates=["date"])
            vectors={pd.Timestamp(r.date):np.asarray(json.loads(r.candidate),float) for r in schedule.itertuples()}
            schedule["solver_status_origin"]=schedule.solver_status.astype(str)
            schedule["solver_status"]="persisted_cache_hit:"+schedule.solver_status.astype(str)
            print(f"[staggered] using persisted {stem} schedule",flush=True)
        else:
            print(f"[staggered] computing {stem} schedule from {calendar[0].date()}",flush=True)
            vectors,schedule,local_cache=staggered_candidate_schedule(panel,ids,developed,method,cap,calendar)
            schedule["solver_status_origin"]=schedule.solver_status.astype(str)
            schedule.to_csv(cache_csv,index=False)
            cache_json.write_text(json.dumps({"schedule_key":schedule_key,"inputs":input_manifest,"method":method,"cap":cap,"registry_ids":ids},indent=2,default=str))
        origins=schedule.solver_status_origin.astype(str)
        optimization_days+=int((origins.str.startswith("optimized_")|origins.str.startswith("cache_hit_")).sum())
        solver_cache_hits+=int(origins.str.startswith("cache_hit_").sum())
        if cache_match: solver_cache_hits+=len(schedule)
        warmup_days+=int(origins.eq("warmup_cash").sum())
        solver_failures+=int(origins.str.startswith("solver_failure_hold:").sum())
        schedule["method"]=method; schedule["mdd_cap"]=cap; all_schedules.append(schedule.copy())
        # Expose completed candidate schedules immediately for monitoring/recovery.
        pd.concat(all_schedules,ignore_index=True).to_csv(out/"solver_status.csv",index=False)
        completed.append(stem)
        for band in WF_BANDS:
            for proxy in (True,False):
                mode="budget_delta_proxy" if proxy else "no_extra_order_cost"
                policy=f"{method}_cap{cap:.2f}_band{band:.2f}_{mode}"
                print(f"[staggered] simulate {policy}",flush=True)
                d,a,f,c,t=simulate_funding(exec_panel,gaps.reindex(calendar).fillna(0.),vectors,band,policy,ids,resize_proxy=proxy)
                meta=schedule.set_index("date")
                for col in ("solver_status","known_roster","known_entrants","eligible_roster","entrants","developed_count","eligible_count","newest_member_developed","lookback_valid_observations","nonzero_solve_rows","estimator_start","estimator_end","strategy_eligibledate"):
                    t[col]=t.date.map(meta[col].to_dict())
                d["method"]=method; d["mdd_cap"]=cap; d["band"]=band; d["resize_cost_mode"]=mode
                t["method"]=method; t["mdd_cap"]=cap; t["band"]=band; t["resize_cost_mode"]=mode
                a["method"]=method; a["mdd_cap"]=cap; a["band"]=band; a["resize_cost_mode"]=mode
                if proxy: all_resize.append(a.copy())
                all_daily.append(d); all_accounts.append(a); all_targets.append(t)
                if not f.empty: all_flows.append(f)
    daily=pd.concat(all_daily,ignore_index=True); accounts=pd.concat(all_accounts,ignore_index=True)
    targets=pd.concat(all_targets,ignore_index=True); schedules=pd.concat(all_schedules,ignore_index=True)
    flows=pd.concat(all_flows,ignore_index=True) if all_flows else pd.DataFrame(columns=["date","policy","from_account","to_account","amount","net_book_cashflow","status"])
    resize=pd.concat(all_resize,ignore_index=True) if all_resize else pd.DataFrame()
    daily.to_csv(out/"daily_book_nav.csv",index=False); accounts.to_csv(out/"daily_account_nav.csv",index=False)
    flows.to_csv(out/"account_cash_flows.csv",index=False); targets.to_csv(out/"allocation_targets.csv",index=False)
    schedules.to_csv(out/"solver_status.csv",index=False); resize.to_csv(out/"resize_cost_audit.csv",index=False)
    if not susp.empty: susp.to_csv(out/"data_suspensions.csv",index=False)
    else: pd.DataFrame(columns=["date","strategy_id","execution_treatment","estimation_value"]).to_csv(out/"data_suspensions.csv",index=False)
    timeline.to_csv(out/"universe_timeline.csv",index=False)
    # First actual funding is read from the earliest EW/proxy run, which starts with NAV entirely in reserve.
    ew=accounts[(accounts.method=="EW")&(accounts.resize_cost_mode=="budget_delta_proxy")&(accounts.band==WF_BANDS[0])]
    actual_first={sid:(g.loc[g.account_equity.gt(1e-12),"date"].min() if g.account_equity.gt(1e-12).any() else pd.NaT) for sid,g in ew[ew.strategy_id.isin(ids)].groupby("strategy_id")}
    for sid,date in actual_first.items(): availability.loc[availability.strategy_id.eq(sid),"first_actual_funding"]=str(pd.Timestamp(date).date()) if pd.notna(date) else ""
    for method in ("EW","LP","CONVEX"):
        one=schedules[schedules.method.eq(method)].drop_duplicates("date").sort_values("date")
        for sid in ids:
            eligible=one[one.eligible_roster.str.contains(json.dumps(sid),regex=False)]
            availability.loc[availability.strategy_id.eq(sid),f"{method}_eligible_eod"]=str(eligible.date.min().date()) if len(eligible) else ""
    availability.to_csv(out/"strategy_availability.csv",index=False)
    funding_rows=[]
    for (method,cap,band,sid),g in accounts[(accounts.resize_cost_mode=="budget_delta_proxy") & accounts.strategy_id.isin(ids)].groupby(["method","mdd_cap","band","strategy_id"]):
        funded=g[g.account_equity.gt(1e-12)]
        funding_rows.append({"method":method,"mdd_cap":cap,"band":band,"strategy_id":sid,
                             "first_funding":str(funded.date.min().date()) if len(funded) else ""})
    pd.DataFrame(funding_rows).to_csv(out/"strategy_funding_timeline.csv",index=False)
    # Same full-period economic book, sliced only for descriptive calendar and roster-constant summaries.
    yearly=[]
    daily["year"]=pd.to_datetime(daily.date).dt.year
    for (policy,year),g in daily.groupby(["policy","year"]):
        row=_best_metrics(pd.Series(g.book_return.to_numpy(float),index=pd.to_datetime(g.date)),policy,f"year:{year}")
        row["year"]=year; row["method"]=g.method.iloc[0]; row["mdd_cap"]=g.mdd_cap.iloc[0]; row["band"]=g.band.iloc[0]; row["resize_cost_mode"]=g.resize_cost_mode.iloc[0]; yearly.append(row)
    pd.DataFrame(yearly).to_csv(out/"yearly_metrics.csv",index=False)
    era_metrics=[]
    for (policy,era),g in daily.merge(timeline[["date","era_id","eligible_count"]],on="date").groupby(["policy","era_id"]):
        row=_best_metrics(pd.Series(g.book_return.to_numpy(float),index=pd.to_datetime(g.date)),policy,f"era:{era}")
        row.update({"era_id":era,"eligible_count":int(g.eligible_count.iloc[0]),"roster":timeline.loc[timeline.era_id.eq(era),"eligible_roster"].iloc[0],"method":g.method.iloc[0],"mdd_cap":g.mdd_cap.iloc[0],"band":g.band.iloc[0],"resize_cost_mode":g.resize_cost_mode.iloc[0]}); era_metrics.append(row)
    pd.DataFrame(era_metrics).to_csv(out/"era_metrics.csv",index=False)
    comparisons=[]
    for (policy,method,cap,band,mode),g in daily.groupby(["policy","method","mdd_cap","band","resize_cost_mode"]):
        row=_best_metrics(pd.Series(g.book_return.to_numpy(float),index=pd.to_datetime(g.date)),policy,"full_walkforward")
        row.update({"method":method,"mdd_cap":cap,"band":band,"resize_cost_mode":mode,"total_rebalance_turnover":float(g.rebalance_turnover.sum()),"total_resize_cost_proxy":float(g.resize_cost_proxy.sum())}); comparisons.append(row)
    comp=pd.DataFrame(comparisons); main=comp[comp.resize_cost_mode=="budget_delta_proxy"].copy()
    main.to_csv(out/"portfolio_comparison.csv",index=False)
    main.sort_values(["calmar","cagr"],ascending=False).groupby("method",as_index=False).first().to_csv(out/"best_by_method.csv",index=False)
    schedules.groupby(["method","mdd_cap","solver_status_origin"]).size().rename("days").reset_index().to_csv(out/"schedule_status_summary.csv",index=False)
    sens=comp.pivot_table(index=["method","mdd_cap","band"],columns="resize_cost_mode",values=["cagr","mdd","sharpe","calmar","total_resize_cost_proxy"]).reset_index()
    sens.to_csv(out/"cost_sensitivity.csv",index=False)
    input_manifest.update({"cache_identity":cache_identity,"resolved_development_dates":{k:str(v.date()) for k,v in developed.items()},"actual_end":str(calendar[-1].date()),"observed_sessions":len(calendar),"completed_schedules":completed,"solver_cache_hits":solver_cache_hits,"optimization_days":optimization_days,"warmup_days":warmup_days,"solver_failures":solver_failures,"elapsed_seconds":time.time()-t0,"counterfactual_availability_assumption":True})
    (out/"inputs_manifest.json").write_text(json.dumps(input_manifest,indent=2,default=str))
    report=["# KRX staggered-universe walk-forward","",f"Counterfactual portfolio period: {calendar[0].date()} through {calendar[-1].date()} ({len(calendar)} observed KRX sessions).", "", "Each source's earliest usable date is treated as a hypothetical strategy-development date. These are not actual historical fund launch records. Original strategy rules and S0 sizing are used throughout; no later policy winner is backfilled.", "", "Strategies join the target roster at their assumed development date and first receive capital on a later observed KRX session. EW admits a developed strategy immediately. LP and Convex admit each entrant only after 252 individual and joint known observations; mature teams continue to be optimized while newcomers incubate at zero capital. Before the first trained team is available their capital stays in treasury. The book remains continuous across roster changes.", "", f"Runtime {time.time()-t0:.1f}s; candidate cache hits {solver_cache_hits}; optimized dates {optimization_days}; initial cash warm-up dates {warmup_days}; solver failures {solver_failures}.", "", "## Full-period main results (2.5bp budget-delta proxy)", "", "| Method | Cap | Band | CAGR | MDD | Sharpe | Calmar |", "|---|---:|---:|---:|---:|---:|---:|"]
    for _,r in main.iterrows(): report.append(f"| {r.method} | {r.mdd_cap:.2f} | {r.band:.2f} | {r.cagr:.4f} | {r.mdd:.4f} | {r.sharpe:.3f} | {r.calmar:.3f} |")
    report += ["", "## Source and execution limits", "", "- Same-day strategy returns are reused as saved net S0 histories. The allocation model is an account-level daily funding surrogate, not an ETF resize fill or T+2 settlement ledger.", "- Rebalance proxy charges 1.5bp fee plus 1bp adverse slippage on absolute per-sleeve capital-budget changes. Treasury/internal cash transfer itself remains free. A no-extra-order-cost sensitivity is included.", "- Known source NAs stay NA in fitting histories; execution treats eligible same-day signal gaps as suspended cash. Missing held daynight marks fail.", "- Beomgu pre-listing synthetic 2x ETF history is excluded before physical 2x feasibility.", "", "See `strategy_availability.csv`, `universe_timeline.csv`, `solver_status.csv`, yearly/era metric files, and daily audit files for dates and path details."]
    (out/"report.md").write_text("\n".join(report)+"\n",encoding="utf-8")
    report_ko=["# KRX 전략 확장 유니버스 워크포워드","",f"기간: {calendar[0].date()} ~ {calendar[-1].date()} ({len(calendar)}개 관측 거래일).", "", "각 전략의 최초 사용 가능일을 가상의 개발 완료일로 간주합니다. 실제 펀드 출시 이력이 아닙니다. 전체 기간 동안 저장된 원본 S0 비중만 사용하며, 이후 선정된 정책을 과거로 소급 적용하지 않습니다.", "", "전략은 가정한 개발일 장 마감 후 유니버스에 들어오고, 다음 관측 거래일부터 자금을 배정합니다. EW는 개발 즉시 편입합니다. LP/Convex는 새 전략의 252개 개별 및 공동 관측치가 확보된 뒤 편입하며, 그전에도 이미 학습된 전략의 최적화를 계속합니다. 최초 학습 완료 전에는 현금으로 대기합니다. 전략 추가 시 NAV를 초기화하지 않습니다.", "", f"최적화 실행일 {optimization_days}, 최초 학습 전 현금 대기 일수(방법·제약 합) {warmup_days}, solver 실패 {solver_failures}, 캐시 적중 {solver_cache_hits}.", "", "| 방법 | MDD 한도 | 밴드 | CAGR | MDD | Sharpe | Calmar |", "|---|---:|---:|---:|---:|---:|---:|"]
    for _,r in main.iterrows(): report_ko.append(f"| {r.method} | {r.mdd_cap:.2f} | {r.band:.2f} | {r.cagr:.4f} | {r.mdd:.4f} | {r.sharpe:.3f} | {r.calmar:.3f} |")
    report_ko += ["", "리밸런싱 비용은 계정별 목표 자본 변화액에 수수료 1.5bp와 불리한 슬리피지 1bp를 적용한 보수적 proxy입니다. 실제 ETF 체결/T+2 결제 감사가 아닙니다. 상세 입력, 결측 처리, 일별 목표와 NAV는 CSV를 참고하세요."]
    (out/"report_ko.md").write_text("\n".join(report_ko)+"\n",encoding="utf-8")
    print(f"[staggered] finished {calendar[0].date()}..{calendar[-1].date()} in {time.time()-t0:.1f}s; output={out}",flush=True)


def run_best_comparison(selection_window="shared", output_dir=None):
    """Ex-post CALMAR selection and six-sleeve budget-delta proxy study."""
    global BEST_OUTPUT_DIR
    if selection_window not in {"shared","full-history"}:
        raise ValueError("selection_window must be shared or full-history")
    if output_dir is None and selection_window=="full-history":
        output_dir=BEST_OUTPUT_DIR.parent/"full_history_selected"
    if output_dir is not None: BEST_OUTPUT_DIR=Path(output_dir)
    BEST_OUTPUT_DIR.mkdir(parents=True,exist_ok=True)
    print("[best] loading saved ledgers and verifying input provenance",flush=True)
    raw,provenance,trades,orders=load_universe()
    if len(raw.columns)!=6: raise ValueError(f"--best requires six enabled source strategies, got {len(raw.columns)}")
    adjustable=[s for s in STRATEGIES if s.get("enabled") and s.get("can_adjust_leverage")]
    if len(adjustable)!=3: raise ValueError(f"--best requires three adjustable teams, got {len(adjustable)}")
    raw_gaps=pd.DataFrame(0.,index=raw.index,columns=raw.columns)
    sizing={}; gaps={}; audits=[]; native={sid:raw[sid].copy() for sid in raw}
    price_sources={str(p):sha256(p) for p in sorted(ETF_MINUTE_DIR.glob("*_TOSS_1m.csv"))}
    manifest_inputs={"code_sha256":sha256(Path(__file__)),"sources":provenance,"etf_minute_inputs":price_sources,
        "costs":{"commission_bps_each_leg":COMMISSION_BPS,"slippage_bps_each_leg":SLIPPAGE_BPS},
        "policies":BEST_POLICIES,"history_days":HISTORY_DAYS,"min_trades":MIN_TRADES}
    cache_key=hashlib.sha256(json.dumps(manifest_inputs,sort_keys=True,default=str).encode()).hexdigest()
    cache_file=BEST_OUTPUT_DIR/"standalone_daily_cache.csv"; gap_cache=BEST_OUTPUT_DIR/"standalone_gap_cache.csv"
    cache_meta=BEST_OUTPUT_DIR/"standalone_daily_cache.json"; audit_cache=BEST_OUTPUT_DIR/"sizing_trade_audit.csv"
    try: loaded_cache=(json.loads(cache_meta.read_text()).get("cache_key")==cache_key and cache_file.is_file() and gap_cache.is_file() and audit_cache.is_file())
    except (OSError,ValueError): loaded_cache=False
    if loaded_cache:
        long=pd.read_csv(cache_file,parse_dates=["date"])
        for (sid,p),g in long.groupby(["strategy_id","policy"]):
            series=pd.Series(g.daily_return.to_numpy(float),index=pd.DatetimeIndex(g.date),name=sid)
            if p=="S0": native[sid]=series
            else: sizing[(sid,p)]=series
        gap_long=pd.read_csv(gap_cache,parse_dates=["date"])
        for (sid,p),g in gap_long.groupby(["strategy_id","policy"]):
            series=pd.Series(g.gap_return.to_numpy(float),index=pd.DatetimeIndex(g.date))
            if p=="S0": raw_gaps[sid]=series.reindex(raw.index).fillna(0.)
            else: gaps[(sid,p)]=series
        audits=[pd.read_csv(audit_cache)]
        print("[best] hash-matched physical path cache loaded; skipping minute replay",flush=True)
    else:
        print("[best] no matching physical path cache; minute replay required",flush=True)
    if not loaded_cache:
        for spec in adjustable:
            sid=spec["id"]
            print(f"[best] replaying physical unit and sizing paths: {sid}",flush=True)
            u1,u2,pa=physical_unit_panels(spec,trades[sid],orders[sid],raw[sid].dropna().index)
            pa["strategy_id"]=sid; audits.append(pa)
            paths,gp,audit=replay_sizing_policies(spec,trades[sid],orders[sid],u1,u2,raw[sid])
            for p in BEST_POLICIES[1:]:
                if p not in paths: raise ValueError(f"{sid}: missing required sizing path {p}")
                sizing[(sid,p)]=paths[p]; gaps[(sid,p)]=gp[p]
            audit["strategy_id"]=sid; audits.append(audit)
            if sid=="daynight_v1_combined":
                for _,tr in trades[sid].iterrows():
                    if str(tr.get("strategy",""))!="overnight": continue
                    exit_day=pd.Timestamp(tr.exit_time).normalize(); entry_day=pd.Timestamp(tr.entry_time).normalize()
                    sym=str(tr.symbol); sym=sym[:-2] if sym.endswith(".0") else sym; sym=sym.zfill(6)
                    mark=_bar_price(sym,entry_day+pd.Timedelta(hours=15,minutes=30),"close",False)
                    prev_nav=float(tr.quantity)*mark if mark is not None else np.nan
                    if mark is None or prev_nav<=0: raise ValueError(f"missing daynight gap mark {exit_day.date()} {sym}")
                    gap=(float(tr.exit_fill)-mark-float(tr.exit_commission)/float(tr.quantity))/mark
                    if exit_day in raw_gaps.index: raw_gaps.loc[exit_day,sid]+=gap
    if not loaded_cache:
        rows=[]
        for sid,s in native.items():
            for d,v in s.dropna().items(): rows.append({"date":d,"strategy_id":sid,"policy":"S0","daily_return":float(v)})
        for (sid,p),s in sizing.items():
            for d,v in s.dropna().items(): rows.append({"date":d,"strategy_id":sid,"policy":p,"daily_return":float(v)})
        long=pd.DataFrame(rows)
        long.to_csv(cache_file,index=False)
        gap_rows=[]
        for sid,s in raw_gaps.items():
            for d,v in s.items(): gap_rows.append({"date":d,"strategy_id":sid,"policy":"S0","gap_return":float(v)})
        for (sid,p),s in gaps.items():
            for d,v in s.items(): gap_rows.append({"date":d,"strategy_id":sid,"policy":p,"gap_return":float(v)})
        pd.DataFrame(gap_rows).to_csv(gap_cache,index=False)
        audit_frame=pd.concat(audits,ignore_index=True,sort=False) if audits else pd.DataFrame()
        _best_write_csv(audit_frame,"sizing_trade_audit.csv")
        cache_meta.write_text(json.dumps({"cache_key":cache_key,"inputs":manifest_inputs},indent=2,default=str))
    print("[best] standalone paths ready; deriving common post-warmup window",flush=True)

    # KRX master calendar is the common observed source calendar. Every source
    # is required to have a real observation on a scored date.
    starts=[native[s].first_valid_index() for s in native]; ends=[native[s].last_valid_index() for s in native]
    lo=max(starts); hi=min(ends); master=raw.index[(raw.index>=lo)&(raw.index<=hi)&(raw.index<=pd.Timestamp(DEVELOPMENT_END))]
    if master.empty: raise ValueError("no shared KRX calendar through configured end")
    # Complete 252-day common estimator readiness, strictly before evaluation.
    common=native.keys(); common_frame=pd.concat([native[s].reindex(master) for s in common],axis=1)
    complete=common_frame.notna().all(axis=1)
    common_ready=complete.cumsum()>=HISTORY_DAYS
    warmup_dates=[]
    for spec in adjustable:
        sid=spec["id"]; path=sizing[(sid,"S1")].reindex(master)
        good=path.notna().cumsum()>=HISTORY_DAYS
        evs=normalize_trade_events(spec,trades[sid],orders[sid])
        exits=pd.DatetimeIndex([e["exit_date"] for e in evs])
        nclosed=pd.Series([int((exits<d).sum()) for d in master],index=master)
        ready=good & (nclosed>=MIN_TRADES)
        eligible=master[ready.to_numpy()]
        if eligible.empty: raise ValueError(f"{sid}: never reaches configured sizing warmup")
        warmup_dates.append(eligible[0])
    idx=np.flatnonzero(common_ready.to_numpy())
    if not len(idx): raise ValueError("six-way panel never reaches 252 prior complete observations")
    first=max(max(warmup_dates),master[idx[0]])
    # Evaluation starts on the session after readiness is achieved.
    eval_dates=master[master>first]
    if len(eval_dates)<2: raise ValueError(f"post-warmup evaluation window too short after {first.date()}")
    print(f"[best] evaluation {eval_dates[0].date()}..{eval_dates[-1].date()} ({len(eval_dates)} sessions)",flush=True)
    standalone_rows=[]; suspension_rows=[]
    common_end=master[-1]
    for spec in adjustable:
        sid=spec["id"]
        for p in BEST_POLICIES:
            ser=native[sid] if p=="S0" else sizing[(sid,p)]
            scored,susp=execution_returns(pd.DataFrame({sid:ser.reindex(eval_dates)},index=eval_dates),eval_dates,[spec])
            if not susp.empty:
                susp["strategy_id"]=sid; susp["policy"]=p; suspension_rows.append(susp)
            r=scored[sid]
            standalone_rows.append(_best_metrics(r,f"{sid}:{p}","shared_post_warmup"))
            # Supplementary native-history context is not used for selection.
            native_r=ser.dropna()
            standalone_rows.append(_best_metrics(native_r,f"{sid}:{p}","full_native_supplement"))
            history_start=ser.first_valid_index()
            full_dates=raw.index[(raw.index>=history_start)&(raw.index<=common_end)]
            if full_dates.empty: raise ValueError(f"{sid}/{p}: no full-history dates through common end {common_end.date()}")
            full_exec,full_susp=execution_returns(pd.DataFrame({sid:ser.reindex(full_dates)},index=full_dates),full_dates,[spec])
            if not full_susp.empty:
                full_susp["strategy_id"]=sid; full_susp["policy"]=p; full_susp["window"]="full_history_to_common_end"; suspension_rows.append(full_susp)
            full_row=_best_metrics(full_exec[sid],f"{sid}:{p}","full_history_to_common_end")
            full_row["period_end_common"] = str(common_end.date())
            standalone_rows.append(full_row)
    standalone=pd.DataFrame(standalone_rows)
    standalone["strategy_id"]=standalone.policy.str.split(":").str[0]
    standalone["sizing_policy"]=standalone.policy.str.split(":").str[1]
    _best_write_csv(standalone,"standalone_policy_comparison.csv")
    selected={}
    selection_metric_window="shared_post_warmup" if selection_window=="shared" else "full_history_to_common_end"
    for spec in adjustable:
        sid=spec["id"]
        group=standalone[(standalone.strategy_id==sid)&(standalone.window==selection_metric_window)]
        group=group.sort_values(["calmar","cagr"],ascending=False,na_position="last")
        if group.empty or not np.isfinite(group.iloc[0].calmar): raise ValueError(f"{sid}: no finite CALMAR candidate")
        selected[sid]=group.iloc[0].sizing_policy
    selection_start=min(standalone.loc[(standalone.window==selection_metric_window),"start"])
    selection_end=str(common_end.date()) if selection_window=="full-history" else str(eval_dates[-1].date())
    selection_periods={}
    for sid,pol in selected.items():
        row=standalone[(standalone.strategy_id==sid)&(standalone.sizing_policy==pol)&(standalone.window==selection_metric_window)].iloc[0]
        selection_periods[sid]={"policy":pol,"start":row.start,"end":row.end,"sessions":int(row.days)}
    (BEST_OUTPUT_DIR/"selected_policies.json").write_text(json.dumps({"selection_window":selection_window,"selection_metric":"CALMAR","selection":"exploratory ex-post maximum CALMAR","period_start":str(selection_start),"period_end":selection_end,"selection_periods_by_team":selection_periods,"portfolio_evaluation_start":str(eval_dates[0].date()),"portfolio_evaluation_end":str(eval_dates[-1].date()),"selected":selected},indent=2))
    print(f"[best] {selection_window} CALMAR policies {selected}; caching phase complete",flush=True)

    # Freeze the three selected sleeves and three unchanged fixed strategies.
    portfolio_panel=pd.DataFrame(index=raw.index)
    portfolio_gaps=raw_gaps.copy()
    for spec in STRATEGIES:
        if not spec.get("enabled"): continue
        sid=spec["id"]
        if spec.get("can_adjust_leverage"):
            policy=selected[sid]; portfolio_panel[sid]=native[sid] if policy=="S0" else sizing[(sid,policy)]
            if policy!="S0": portfolio_gaps[sid]=gaps[(sid,policy)].reindex(raw.index).fillna(0.)
        else: portfolio_panel[sid]=native[sid]
    eval_panel=portfolio_panel.reindex(eval_dates)
    registry=[s for s in STRATEGIES if s.get("enabled")]
    exec_panel,susp=execution_returns(eval_panel,eval_dates,registry)
    if not susp.empty:
        susp["policy"]="selected_six_sleeve_portfolio"; suspension_rows.append(susp)
    eval_gaps=portfolio_gaps.reindex(eval_dates).fillna(0.)
    ids=list(exec_panel.columns)

    # Optimize once per method/cap. Reuse the resulting schedule for both bands
    # and both explicit cost scenarios.
    comparisons=[]; daily_out=[]; targets_out=[]; solver_out=[]; resize_out=[]
    configs=[("EW",0.0),("LP",.05),("LP",.10),("CONVEX",.05),("CONVEX",.10)]
    total=len(configs)*len(REBALANCE_BANDS)*2; done=0
    for method,cap in configs:
        vectors,schedule=candidate_schedule(portfolio_panel,method,cap,ids,eval_dates,fixed_universe=True)
        for band in REBALANCE_BANDS:
            for cost_proxy in (True,False):
                done+=1; label=f"{method}_cap{cap:.2f}_band{band:.2f}_"+("budget_delta_proxy" if cost_proxy else "no_extra_order_cost")
                print(f"[best] portfolio {done}/{total}: {label}",flush=True)
                d,a,f,c,t=simulate_funding(exec_panel,eval_gaps,vectors,band,label,ids,resize_proxy=cost_proxy)
                t["solver_status"]=t.date.map(schedule.set_index("date").solver_status.to_dict())
                t["eligible"]=t.date.map(schedule.set_index("date").eligible.to_dict())
                t["estimator_start"]=t.date.map(schedule.set_index("date").estimator_start.to_dict())
                t["estimator_end"]=t.date.map(schedule.set_index("date").estimator_end.to_dict())
                t["estimator_observations"]=t.date.map(schedule.set_index("date").estimator_observations.to_dict())
                metrics=_best_metrics(pd.Series(d.book_return.to_numpy(),index=pd.DatetimeIndex(d.date)),label,"primary_shared_period")
                metrics.update({"method":method,"mdd_cap":cap,"band":band,"resize_cost_mode":"budget_delta_buy_sell_proxy" if cost_proxy else "no_extra_order_cost_sensitivity","period_start":str(eval_dates[0].date()),"period_end":str(eval_dates[-1].date()),"sessions":len(eval_dates),"rebalance_turnover":float(d.rebalance_turnover.sum()),"resize_cost_proxy":float(d.resize_cost_proxy.sum())})
                comparisons.append(metrics); d["method"]=method; d["mdd_cap"]=cap; d["band"]=band; d["resize_cost_mode"]=metrics["resize_cost_mode"]; daily_out.append(d)
                t["method"]=method; t["mdd_cap"]=cap; t["resize_cost_mode"]=metrics["resize_cost_mode"]; targets_out.append(t)
                sch=schedule.copy(); sch["method"]=method; sch["mdd_cap"]=cap; sch["policy"]=label; solver_out.append(sch)
                if cost_proxy:
                    audit=a.copy(); audit["method"]=method; audit["mdd_cap"]=cap; audit["band"]=band; audit["policy"]=label; resize_out.append(audit)
    comp=pd.DataFrame(comparisons)
    main=comp[comp.resize_cost_mode=="budget_delta_buy_sell_proxy"].copy()
    _best_write_csv(main,"portfolio_comparison.csv")
    _best_write_csv(pd.concat(daily_out,ignore_index=True),"daily_nav.csv")
    _best_write_csv(pd.concat(targets_out,ignore_index=True),"targets.csv")
    _best_write_csv(pd.concat(solver_out,ignore_index=True),"solver_status.csv")
    resize=pd.concat(resize_out,ignore_index=True) if resize_out else pd.DataFrame()
    _best_write_csv(resize,"resize_cost_audit.csv")
    _best_write_csv(pd.concat(suspension_rows,ignore_index=True) if suspension_rows else pd.DataFrame(columns=["date","strategy_id","execution_treatment","estimation_value","policy"]),"data_suspensions.csv")
    sensitivity=comp.pivot_table(index=["method","mdd_cap","band"],columns="resize_cost_mode",values=["cagr","mdd","sharpe","calmar","resize_cost_proxy"]).reset_index()
    _best_write_csv(sensitivity,"portfolio_cost_sensitivity.csv")
    best=main.sort_values("calmar",ascending=False).groupby("method",as_index=False).head(1)
    _best_write_csv(best,"best_by_method.csv")
    (BEST_OUTPUT_DIR/"inputs_manifest.json").write_text(json.dumps({**manifest_inputs,"cache_key":cache_key,"selection_window":selection_window,"selection_metric_window":selection_metric_window,"selection_period_start":str(selection_start),"selection_period_end":selection_end,"evaluation_start":str(eval_dates[0].date()),"evaluation_end":str(eval_dates[-1].date()),"sessions":len(eval_dates),"selected_policies":selected,"fixed_six_sleeve_universe":True,"rebalance_cost":"per sleeve abs(account capital delta) * (1.5bp commission + 1bp slippage); conservative sizing proxy, not actual ETF fills","fixed_sleeves":registry,"limitations":LIMITATIONS},indent=2,default=str))
    selection_description=("maximum CALMAR over the shared post-warmup period" if selection_window=="shared" else "maximum CALMAR over each team's native history through the six-source common end")
    report=["# KRX best policy and portfolio study","",f"Primary portfolio evaluation: {eval_dates[0].date()} through {eval_dates[-1].date()} ({len(eval_dates)} KRX sessions).",f"Individual policy selection window: {selection_window}; selected by {selection_description}.", "", "The allocator keeps all six sleeves in a fixed universe for every method and date. EW targets 1/6 per sleeve; a known source NA is logged and treated as suspended cash for execution. LP and Convex use the latest 252 complete common observations available before each target.", "", "Policy choice is exploratory and ex-post. It is not an untouched out-of-sample result. CAGR is annualized as (1 + total return)^(252 / sessions) − 1; short samples are explicitly labeled in the CSV. The CSV includes both shared post-warmup and full-history-through-common-end metrics.", "", "The reported rebalance proxy charges 1.5 bp commission plus 1 bp adverse slippage against each sleeve's absolute capital-budget delta. It is a conservative capital-budget resize proxy, not actual ETF fills or T+2 settlement. Treasury/internal cash transfers incur no cost. Ordinary strategy trade costs are already included in sleeve returns. The no-extra-order-cost sensitivity isolates the funding-transfer case.", "", "## Individual policy results", "", "S0 is the saved original policy benchmark. The selected row uses the configured selection window.", "", "| Strategy | Policy | CAGR | MDD | Sharpe | Calmar |", "|---|---:|---:|---:|---:|---:|---:|"]
    for sid,pol in selected.items():
        for policy in ("S0",pol):
            row=standalone[(standalone.strategy_id==sid)&(standalone.sizing_policy==policy)&(standalone.window==selection_metric_window)].iloc[0]
            report.append(f"| {sid} | {policy} | {row.cagr:.4f} | {row.mdd:.4f} | {row.sharpe:.3f} | {row.calmar:.3f} |")
    report += ["", "Selected-policy entry leverage audit (closed trade events):", "", "| Strategy | Policy | Mean L | Max L | L=0 hits | L=2 hits |", "|---|---:|---:|---:|---:|---:|"]
    audit_all=pd.concat(audits,ignore_index=True,sort=False) if audits else pd.DataFrame()
    for sid,pol in selected.items():
        if "strategy_id" not in audit_all or "policy" not in audit_all or "L" not in audit_all: continue
        ag=audit_all[(audit_all.strategy_id==sid)&(audit_all.policy==pol)]
        lev=pd.to_numeric(ag.L,errors="coerce").dropna()
        if lev.empty: continue
        report.append(f"| {sid} | {pol} | {lev.mean():.3f} | {lev.max():.3f} | {(lev<=1e-8).sum()} | {(lev>=2-1e-8).sum()} |")
    report += ["", "## Best main portfolio rows by method", "", "| Method | Cap | Band | CAGR | MDD | Sharpe | Calmar |", "|---|---:|---:|---:|---:|---:|---:|"]
    for _,row in best.iterrows(): report.append(f"| {row.method} | {row.mdd_cap:.2f} | {row.band:.2f} | {row.cagr:.4f} | {row.mdd:.4f} | {row.sharpe:.3f} | {row.calmar:.3f} |")
    winners=[("Highest CAGR",main.sort_values("cagr",ascending=False).iloc[0]),
             ("Smallest absolute MDD",main.iloc[np.argmin(np.abs(main.mdd.to_numpy(float)))]),
             ("Highest Sharpe",main.sort_values("sharpe",ascending=False).iloc[0]),
             ("Highest Calmar",main.sort_values("calmar",ascending=False).iloc[0])]
    report += ["", "## Metric-specific portfolio winners", "", "| Metric | Method | Cap | Band | Value |", "|---|---:|---:|---:|---:|"]
    for label,row in winners:
        value=row.mdd if label=="Smallest absolute MDD" else (row.cagr if label=="Highest CAGR" else (row.sharpe if label=="Highest Sharpe" else row.calmar))
        report.append(f"| {label} | {row.method} | {row.mdd_cap:.2f} | {row.band:.2f} | {value:.4f} |")
    report += ["", "## Limitations", ""]+[f"- {x}" for x in LIMITATIONS]
    report += ["", "All daily returns, target vectors, solver statuses, and budget-delta charges are in the adjacent CSV artifacts."]
    (BEST_OUTPUT_DIR/"report.md").write_text("\n".join(report)+"\n",encoding="utf-8")
    print(f"[best] completed; artifacts: {BEST_OUTPUT_DIR}",flush=True)

def run():
    if SUNNY_POSITION_COL not in {"pos_A","pos_B","pos_C","pos_D"}: raise ValueError("SUNNY_POSITION_COL must be pos_A, pos_B, pos_C, or pos_D")
    if KELLY_WINDOW not in {"expanding","rolling"}: raise ValueError("KELLY_WINDOW must be expanding or rolling")
    if WOOHYUN_HOLDING_BARS<1: raise ValueError("WOOHYUN_HOLDING_BARS must be positive")
    if CENTRAL_RESIZE_COST_BPS<0: raise ValueError("CENTRAL_RESIZE_COST_BPS must be nonnegative")
    if len(KELLY_FRACTIONS)!=3: raise ValueError("Set three Kelly fractions for S1/S2/S3")
    if HISTORY_DAYS<2 or MIN_TRADES<0: raise ValueError("Invalid estimation warm-up")
    if not all(0<d<1 for d in MDD_CAPS) or not all(0<=b<1 for b in REBALANCE_BANDS):
        raise ValueError("Drawdown caps and rebalance bands must be fractions of NAV")
    if not all(np.isfinite(c) and 0<=c<10000 for c in (COMMISSION_BPS,SLIPPAGE_BPS,CENTRAL_RESIZE_COST_BPS)):
        raise ValueError("Invalid trading costs")
    names=[s["id"] for s in STRATEGIES if s.get("enabled")]
    if len(names)!=len(set(names)): raise ValueError("Strategy ids must be unique")
    raw_panel, provenance, trade_ledgers, order_ledgers = load_universe()
    if len(raw_panel.columns)<1: raise ValueError("no enabled strategies")
    values=raw_panel.to_numpy(float)
    if np.isinf(values).any() or np.any(values[np.isfinite(values)]<=-1):
        raise ValueError("Source return is infinite or <= -100%; cannot simulate an unborrowed solvent account")
    sizing_paths={}; sizing_gaps={}; matched_paths={}; sizing_audits=[]
    enabled=[s for s in STRATEGIES if s.get("enabled")]
    for spec in enabled:
        if not spec.get("can_adjust_leverage"): continue
        sid=spec["id"]
        if "market" not in spec or "orders_path" not in spec:
            raise ValueError(f"{sid}: adjustable registry entry requires market and orders_path physical-replay adapter")
        if sid not in trade_ledgers or sid not in order_ledgers:
            raise ValueError(f"{sid}: saved trade/order ledger missing; no physical sizing run")
        base_dates=raw_panel[sid].dropna().index
        u1,u2,pa=physical_unit_panels(spec,trade_ledgers[sid],order_ledgers[sid],base_dates)
        pa["strategy_id"]=sid; pa["variant"]="unit_price_audit"; sizing_audits.append(pa)
        paths,gaps,audit=replay_sizing_policies(spec,trade_ledgers[sid],order_ledgers[sid],u1,u2,raw_panel[sid])
        for policy,path in paths.items(): sizing_paths[(sid,policy)]=path; sizing_gaps[(sid,policy)]=gaps[policy]
        s1=audit[(audit.get("policy",pd.Series(index=audit.index,dtype=object))=="S1")].copy() if not audit.empty else pd.DataFrame()
        ld=pd.to_numeric(s1.loc[pd.to_datetime(s1.get("entry_date",pd.Series(dtype=object)),errors="coerce")<=pd.Timestamp(DEVELOPMENT_END),"L"],errors="coerce").dropna() if "L" in s1 else pd.Series(dtype=float)
        mean_l=float(ld.mean()) if not ld.empty else 1.0
        mp,mg,ma=replay_fixed_exposure(spec,trade_ledgers[sid],order_ledgers[sid],raw_panel[sid].dropna().index,raw_panel[sid],mean_l)
        # The fitted development mean must never be scored retrospectively as causal.
        mp.loc[mp.index<=pd.Timestamp(DEVELOPMENT_END)]=np.nan
        matched_paths[sid]=mp; ma["strategy_id"]=sid; ma["policy"]="S0_MATCHED_DEV"; ma["development_mean_L"]=mean_l; sizing_audits.append(ma)
        audit["strategy_id"]=sid; sizing_audits.append(audit)
    raw_gaps=pd.DataFrame(0.0,index=raw_panel.index,columns=raw_panel.columns)
    dn_id="daynight_v1_combined"
    if dn_id in trade_ledgers:
        for _,tr in trade_ledgers[dn_id].iterrows():
            if str(tr.get("strategy",""))!="overnight": continue
            exit_day=pd.Timestamp(tr.exit_time).normalize(); entry_day=pd.Timestamp(tr.entry_time).normalize()
            sym=str(tr.symbol); sym=sym[:-2] if sym.endswith(".0") else sym; sym=sym.zfill(6)
            mark=_bar_price(sym,entry_day+pd.Timedelta(hours=15,minutes=30),"close",False)
            prev_nav=float(tr.quantity)*mark if mark is not None else np.nan
            if mark is None or not np.isfinite(prev_nav) or prev_nav<=0: raise ValueError(f"daynight gap mark unavailable for {exit_day.date()} {sym}")
            gap=(float(tr.quantity)*(float(tr.exit_fill)-mark)-float(tr.exit_commission))/prev_nav
            if exit_day in raw_gaps.index: raw_gaps.loc[exit_day,dn_id]+=gap
    # Common mode bounds the comparison to shared source coverage. The valuation
    # clock retains observed NA rows; only same-day missing sessions are explicit
    # cash suspensions for execution, while optimizer histories retain NA.
    calendar=valuation_calendar(raw_panel)
    base_panel=raw_panel.reindex(calendar).copy()
    base_gaps=raw_gaps.reindex(calendar).fillna(0.)
    if ELIGIBILITY_FROM is not None:
        mask=calendar>=pd.Timestamp(ELIGIBILITY_FROM)
        calendar=calendar[mask]; base_panel=base_panel.loc[calendar]; base_gaps=base_gaps.loc[calendar]
    if not len(calendar): raise ValueError("empty calendar after configured eligibility cutoff")
    base_execution,suspensions=execution_returns(base_panel,calendar,enabled)
    OUTPUT_DIR.mkdir(parents=True,exist_ok=True)

    # Standalone S0 and causal physical shadow outcomes.
    metrics=[]; standalone=[]
    for sid in raw_panel.columns:
        variants={"S0":raw_panel[sid]}
        if sid in matched_paths: variants["S0_MATCHED_DEV"]=matched_paths[sid]
        variants.update({p:path for (s,p),path in sizing_paths.items() if s==sid})
        for policy,x in variants.items():
            x=x.dropna()
            for d,r in x.items(): standalone.append({"date":d,"strategy_id":sid,"policy":policy,"daily_return":float(r),"basis":"saved_original_unchanged" if policy=="S0" else "physical_ETF_shadow"})
            metrics.append(metric_row(x.loc[x.index<=DEVELOPMENT_END],f"{sid}:{policy}","development"))
            hold=x.loc[x.index>DEVELOPMENT_END]
            if not hold.empty: metrics.append(metric_row(hold,f"{sid}:{policy}","exploratory_holdout_touched" if HOLDOUT_TOUCHED else "holdout"))
    pd.DataFrame(standalone).to_csv(OUTPUT_DIR/"standalone_sizing.csv",index=False)
    (pd.concat(sizing_audits,ignore_index=True,sort=False) if sizing_audits else pd.DataFrame([{"status":"NO_ADJUSTABLE_STRATEGIES_ENABLED"}])).to_csv(OUTPUT_DIR/"sizing_trade_audit.csv",index=False)

    # Use each team's preconfigured frozen policy, default S1. Non-adjustable S0 is unchanged.
    frozen_history=raw_panel.copy()
    frozen_gap_raw=raw_gaps.reindex(calendar).fillna(0.)
    for spec in enabled:
        sid=spec["id"]
        if spec.get("can_adjust_leverage"):
            policy=KELLY_SELECTED.get(sid,"S1")
            if policy!="S0":
                if (sid,policy) not in sizing_paths: raise ValueError(f"selected sizing variant unavailable: {sid}/{policy}")
                frozen_history[sid]=sizing_paths[(sid,policy)].reindex(raw_panel.index).where(raw_panel[sid].notna())
                frozen_gap_raw[sid]=sizing_gaps[(sid,policy)].reindex(calendar).fillna(0.)
    frozen_panel=frozen_history.reindex(calendar); frozen_gaps=frozen_gap_raw
    frozen_execution,susp2=execution_returns(frozen_panel,calendar,enabled)
    suspensions=pd.concat([suspensions,susp2],ignore_index=True).drop_duplicates(["date","strategy_id"],keep="last")

    # Original sizing candidate vectors are frozen and reused in the conditional Kelly runs.
    all_daily=[]; all_accounts=[]; all_flows=[]; all_contrib=[]; all_targets=[]
    no_vectors={}
    available=base_panel.iloc[0].notna().to_numpy(float)
    init=np.r_[available/available.sum(),0.] if available.sum() else np.r_[np.zeros(len(available)),1.]
    d,a,f,c,t=simulate_funding(base_execution,base_gaps,no_vectors,0.0,"NO_TRANSFER_EQUAL",list(base_panel.columns),no_transfer=True,initial_weights=init)
    all_daily.append(d); all_accounts.append(a); all_flows.append(f); all_contrib.append(c); all_targets.append(t)
    for label,sel in (("development",d[d.date<=pd.Timestamp(DEVELOPMENT_END)]),("exploratory_holdout_touched" if HOLDOUT_TOUCHED else "holdout",d[d.date>pd.Timestamp(DEVELOPMENT_END)])):
        if not sel.empty: metrics.append(metric_row(pd.Series(sel.book_return.to_numpy(),index=pd.DatetimeIndex(sel.date)),"NO_TRANSFER_EQUAL",label))
    for method in ("EW","LP","CONVEX"):
        caps=(0.0,) if method=="EW" else MDD_CAPS
        for cap in caps:
            base_vectors,schedule=candidate_schedule(raw_panel,method,cap,list(base_panel.columns),calendar)
            refit_vectors,refit_schedule=candidate_schedule(frozen_history,method,cap,list(base_panel.columns),calendar)
            for band in REBALANCE_BANDS:
                for variant,returns_panel,refit in (("S0",base_panel,False),("KELLY_FROZEN",frozen_panel,False),("KELLY_REFIT",frozen_panel,True)):
                    vectors=base_vectors
                    solver_rows=schedule
                    if refit:
                        vectors,solver_rows=refit_vectors,refit_schedule
                    policy=f"{variant}_{method}_d{cap:.2f}_b{band:.2f}"
                    gaps_panel=base_gaps if variant=="S0" else frozen_gaps
                    executable=base_execution if variant=="S0" else frozen_execution
                    d,a,f,c,t=simulate_funding(executable,gaps_panel,vectors,band,policy,list(base_panel.columns))
                    solver_rows=solver_rows.copy(); solver_rows["policy"]=policy
                    t["solver_status"]=t.date.map(solver_rows.set_index("date").solver_status.to_dict())
                    for column in ("eligible","estimator_start","estimator_end","estimator_observations"):
                        t[column]=t.date.map(solver_rows.set_index("date")[column].to_dict())
                    all_daily.append(d); all_accounts.append(a); all_flows.append(f); all_contrib.append(c); all_targets.append(t)
                    for window,sel in (("development",d[d.date<=pd.Timestamp(DEVELOPMENT_END)]),("exploratory_holdout_touched" if HOLDOUT_TOUCHED else "holdout",d[d.date>pd.Timestamp(DEVELOPMENT_END)])):
                        if not sel.empty:
                            mr=metric_row(pd.Series(sel.book_return.to_numpy(),index=pd.DatetimeIndex(sel.date)),policy,window)
                            mr["total_rebalance_turnover"]=float(sel.rebalance_turnover.sum())
                            mr["total_resize_cost_proxy"]=float(sel.resize_cost_proxy.sum())
                            metrics.append(mr)
    daily=pd.concat(all_daily,ignore_index=True); accounts=pd.concat(all_accounts,ignore_index=True); flows=pd.concat([x for x in all_flows if not x.empty],ignore_index=True) if any(not x.empty for x in all_flows) else pd.DataFrame(columns=["date","policy","from_account","to_account","amount","net_book_cashflow","status"]); contrib=pd.concat(all_contrib,ignore_index=True); targets=pd.concat(all_targets,ignore_index=True)
    daily.to_csv(OUTPUT_DIR/"daily_book_nav.csv",index=False)
    accounts.to_csv(OUTPUT_DIR/"daily_account_nav.csv",index=False)
    flows.to_csv(OUTPUT_DIR/"account_cash_flows.csv",index=False)
    contrib.to_csv(OUTPUT_DIR/"strategy_contributions.csv",index=False)
    targets.to_csv(OUTPUT_DIR/"allocation_targets.csv",index=False)
    suspensions.to_csv(OUTPUT_DIR/"data_suspensions.csv",index=False)
    # Security-level exposure and position resize fills are not part of this TWR surrogate.
    metrics_frame=pd.DataFrame(metrics)
    trade_summaries=[]
    for (sid,pol),g in pd.concat(sizing_audits,ignore_index=True,sort=False).groupby(["strategy_id","policy"],dropna=True) if sizing_audits else []:
        if "L" in g and g.L.notna().any():
            L=pd.to_numeric(g.L,errors="coerce").dropna()
            trade_summaries.append({"policy":f"{sid}:{pol}","window":"trade_audit","closed_trades":len(L),"mean_L":L.mean(),"max_L":L.max(),"fraction_L_zero":(L<=1e-8).mean(),"fraction_L_two":(L>=2-1e-8).mean(),"mean_target_gross_exposure":pd.to_numeric(g.target_gross_exposure_at_entry,errors="coerce").mean(),"mean_turnover_fraction":pd.to_numeric(g.turnover_fraction_of_NAV,errors="coerce").mean(),"total_entry_cost_fraction":pd.to_numeric(g.entry_cost_fraction_of_NAV,errors="coerce").sum(),"total_exit_cost_fraction":pd.to_numeric(g.exit_cost_fraction_of_NAV,errors="coerce").sum()})
    if trade_summaries: metrics_frame=pd.concat([metrics_frame,pd.DataFrame(trade_summaries)],ignore_index=True,sort=False)
    metrics_frame.to_csv(OUTPUT_DIR/"comparison_summary.csv",index=False)
    metrics_frame[metrics_frame.window.isin(["development","exploratory_holdout_touched","holdout"])].to_csv(OUTPUT_DIR/"wf_metrics.csv",index=False)
    hold=metrics_frame[metrics_frame.window.astype(str).str.contains("holdout")]
    hold.to_csv(OUTPUT_DIR/"holdout_metrics.csv",index=False)
    quarterly=[]
    standalone_frame=pd.DataFrame(standalone)
    if not standalone_frame.empty:
        standalone_frame["quarter"]=pd.to_datetime(standalone_frame.date).dt.to_period("Q").astype(str)
        for (sid,pol,q),g in standalone_frame.groupby(["strategy_id","policy","quarter"]):
            quarterly.append(metric_row(pd.Series(g.daily_return.to_numpy(),index=pd.to_datetime(g.date)),f"{sid}:{pol}",f"quarter:{q}"))
    daily_frame=pd.concat(all_daily,ignore_index=True)
    daily_frame["quarter"]=pd.to_datetime(daily_frame.date).dt.to_period("Q").astype(str)
    for (pol,q),g in daily_frame.groupby(["policy","quarter"]):
        quarterly.append(metric_row(pd.Series(g.book_return.to_numpy(),index=pd.to_datetime(g.date)),pol,f"quarter:{q}"))
    pd.DataFrame(quarterly).to_csv(OUTPUT_DIR/"wf_quarterly_metrics.csv",index=False)
    manifest={"code_sha256":sha256(Path(__file__)),"config":{"development_end":DEVELOPMENT_END,"daynight_version":DAYNIGHT_VERSION,"daynight_account":DAYNIGHT_ACCOUNT,"beomgu_variant":BEOMGU_VARIANT,"intraday_selected_rules_path":INTRADAY_SELECTED_RULES_PATH,"history_days":HISTORY_DAYS,"min_closed_trades":MIN_TRADES,"kelly_fractions":KELLY_FRACTIONS,"kelly_selected_frozen":KELLY_SELECTED,"kelly_window":KELLY_WINDOW,"sunny_position_col":SUNNY_POSITION_COL,"sunny_round_trip_cost":SUNNY_ROUND_TRIP_COST,"woohyun_holding_bars":WOOHYUN_HOLDING_BARS,"mdd_caps":MDD_CAPS,"bands":REBALANCE_BANDS,"common_calendar":COMMON_CALENDAR,"valuation_first":str(calendar.min().date()),"valuation_last":str(calendar.max().date()),"eligibility_from":ELIGIBILITY_FROM,"holdout_touched":HOLDOUT_TOUCHED,"commission_bps":COMMISSION_BPS,"slippage_bps":SLIPPAGE_BPS,"central_resize_cost_bps":CENTRAL_RESIZE_COST_BPS,"original_daily_returns_preserved":True},"sources":provenance,"etf_minute_inputs":ETF_PRICE_INPUTS,"limitations":LIMITATIONS,"execution":"physical standalone ETF trade replay; central TWR funding surrogate without settlement/order execution"}
    (OUTPUT_DIR/"manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2,default=str),encoding="utf-8")
    report=["# KRX 전략 sizing / allocation 비교","","> 독립 sizing은 저장된 신호시각을 실제 1x/2x ETF 분봉으로 재생한 shadow 결과입니다. 중앙 배분은 overnight gap 이후 다음 세션에 funding하는 계정별 TWR surrogate이며 실제 주문/T+2 감사가 아닙니다.","","## 데이터 및 해석"]
    report += [f"- {x}" for x in LIMITATIONS]
    report += ["","## 비교 결과","","전체 수치와 분기별 walk-forward 표는 `comparison_summary.csv`와 `wf_quarterly_metrics.csv`에 있습니다.","","## 입력 경로"]
    report += [f"- {sid}: {'in-memory fallback' if src.get('generated_in_memory') else 'saved ledger'} — {src.get('path',src.get('source',''))}" for sid,src in provenance.items()]
    report += ["","## 산출 파일",""]+[f"- `{p.name}`" for p in sorted(OUTPUT_DIR.iterdir())]
    (OUTPUT_DIR/"report.md").write_text("\n".join(report)+"\n",encoding="utf-8")
    print(f"Loaded {len(base_panel.columns)} strategies, {len(calendar)} EOD dates. Output: {OUTPUT_DIR}")

if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument("--best",action="store_true",help="run the exploratory CALMAR-selected policy and six-sleeve allocator study")
    mode.add_argument("--staggered",action="store_true",help="run the counterfactual expanding-universe S0 walk-forward study")
    parser.add_argument("--best-selection",choices=("shared","full-history"),default="shared",help="choose policies on the shared post-warmup period or each team's full history through the common end")
    parser.add_argument("--best-output",type=Path,default=None,help="output directory (default: standard shared output, or full_history_selected for full-history selection)")
    parser.add_argument("--wf-output",type=Path,default=None,help="staggered study output directory (default: results/krx_staggered_walkforward_20261102)")
    args=parser.parse_args()
    if args.staggered:
        if args.best_output is not None or args.best_selection!="shared": parser.error("--best options cannot be combined with --staggered")
        run_staggered_walkforward(args.wf_output or WF_OUTPUT_DIR)
    elif args.best:
        run_best_comparison(args.best_selection,args.best_output)
    elif args.best_selection!="shared" or args.best_output is not None or args.wf_output is not None:
        parser.error("study output/selection options require --best or --staggered")
    else:
        run()
