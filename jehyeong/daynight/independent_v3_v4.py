"""Independent V3/V4 reference; imports neither engine nor MA gate module.
Rebuilds raw signals, both normal shadow accounts, exit-clock modes and six actual
accounts; independently replays published ETF orders and all performance metrics.
"""
from pathlib import Path
import pandas as pd
import numpy as np
import json
import hashlib
HERE=Path(__file__).resolve().parent
OUT=HERE/'results_v3_v4'
manifest=json.loads((OUT/'manifest.json').read_text())
cfg=manifest['config']
ROOT=Path(manifest['source_files'][0]['path']).parent
INIT=cfg['initial_capital'];FEE=cfg['commission_bps']/10000;SLIP=cfg['slippage_bps']/10000;FACTOR=(1-SLIP)*(1-FEE)/((1+SLIP)*(1+FEE))
minute={s:pd.read_csv(ROOT/'minute'/f'{s}_TOSS_1m.csv',parse_dates=['timestamp']) for s in ['229200','233740','251340']}
daily=pd.read_csv(ROOT/'229200_daily.csv',parse_dates=['date']).sort_values('date').set_index('date')
lo=max(cfg['start'],*(x.date.min() for x in minute.values()));hi=min(cfg['end'],*(x.date.max() for x in minute.values()))
dates=daily.index[(daily.index>=lo)&(daily.index<=hi)]
next_day=dict(zip(daily.index[:-1],daily.index[1:]))
lookup={s:x.set_index('timestamp') for s,x in minute.items()}
groups={s:dict(tuple(x.groupby('date'))) for s,x in minute.items()}
previous=daily.close.shift();ma=daily.close.rolling(10).mean().shift()
def ts(d,t):return pd.Timestamp(str(d.date())+' '+t)
def bar(s,t):
 try:return lookup[s].loc[t]
 except KeyError:return None
def valid(b):return b is not None and b.volume>0 and b.open>0
def prices(day,leg,direction):
 if not direction:return None
 s='233740' if direction==1 else '251340'
 if leg=='intraday':
  b,e=bar(s,ts(day,'09:06')),bar(s,ts(day,'15:01'))
  return (s,float(b.close),float(e.open),None) if valid(b) and valid(e) else None
 nx=next_day.get(day)
 if nx is None or nx>dates[-1]:return None
 e=bar(s,ts(nx,'09:01'));m=bar(s,ts(day,'15:30'));g=groups[s].get(str(day.date()))
 if not valid(e) or m is None or g is None:return None
 g=g[(g.timestamp>=ts(day,'09:01'))&(g.timestamp<=ts(day,'15:20'))&(g.volume>0)]
 return (s,float(g.close.iloc[-1]),float(e.open),float(m.close)) if len(g) else None
intents={}
for day in dates:
 b0,b5=bar('229200',ts(day,'09:01')),bar('229200',ts(day,'09:06'));ds=0
 if b0 is not None and b5 is not None and np.isfinite(ma.loc[day]):
  move=b5.open/b0.open-1
  if previous.loc[day]>ma.loc[day] and move>=.003-1e-12:ds=1
  if previous.loc[day]<ma.loc[day] and move<=-.003+1e-12:ds=-1
 ns=0;g=groups['229200'].get(str(day.date()))
 if g is not None:
  w=g[g.timestamp.between(ts(day,'09:01'),ts(day,'15:20'))]
  expected=pd.date_range(ts(day,'09:01'),ts(day,'15:20'),freq='min')
  if set(w.timestamp)==set(expected) and len(w)==len(expected):
   h,l=w.high.max(),w.low.min()
   ibs=(w.close.iloc[-1]-l)/(h-l) if h>l else np.nan
   ns=1 if ibs<.3 else -1 if ibs>.7 else 0
 intents[day]=(ds,ns)
normal_cash={'intraday':INIT,'overnight':INIT};realized={'intraday':0.,'overnight':0.}
history={'intraday':[],'overnight':[]};mode={'intraday':1,'overnight':1};pending=None
plans={};gate_rows=[];decision_ma={'intraday':np.nan,'overnight':np.nan}
def update(leg):
 if len(history[leg])>=19:
  decision_ma[leg]=sum(history[leg][-19:]+[realized[leg]])/20
  mode[leg]=-1 if realized[leg]<decision_ma[leg] else 1
 else:mode[leg]=1
for day in dates:
 ds,ns=intents[day];before=mode.copy();has_exit={'intraday':False,'overnight':False}
 if pending is not None:
  normal_cash['overnight']=pending[0]*pending[1]*(1-SLIP)*(1-FEE)
  realized['overnight']=normal_cash['overnight']-INIT;has_exit['overnight']=True;pending=None;update('overnight')
 entry_mode={'intraday':mode['intraday'],'overnight':mode['overnight']}
 day_normal=prices(day,'intraday',ds);night_normal=prices(day,'overnight',ns)
 plans['V3',day]=(day_normal,night_normal)
 plans['V4',day]=(prices(day,'intraday',ds*entry_mode['intraday']),prices(day,'overnight',ns*entry_mode['overnight']))
 if day_normal:
  normal_cash['intraday']*=day_normal[2]/day_normal[1]*FACTOR
  realized['intraday']=normal_cash['intraday']-INIT;has_exit['intraday']=True;update('intraday')
 if night_normal:
  q=normal_cash['overnight']/(night_normal[1]*(1+SLIP)*(1+FEE));pending=(q,night_normal[2])
 for leg in history:
  gate_rows.append(dict(date=day,strategy=leg,reference_pnl=realized[leg],has_exit=has_exit[leg],mode_before_exit=before[leg],mode_after_exit=mode[leg],entry_mode=entry_mode[leg],decision_ma20=decision_ma[leg]))
  history[leg].append(realized[leg])
results={}
for version in ['V3','V4']:
 for leg in ['intraday','overnight','combined']:
  cash=INIT;pos=None;rows=[];trades=[]
  for day in dates:
   if pos:cash=pos[0]*pos[1]*(1-SLIP)*(1-FEE);pos=None
   dt,nt=plans[version,day]
   if dt and leg in ['intraday','combined']:
    cash*=dt[2]/dt[1]*FACTOR;trades.append(('intraday',str(day.date()),dt[0]))
   if nt and leg in ['overnight','combined']:
    q=cash/(nt[1]*(1+SLIP)*(1+FEE));pos=q,nt[2];nav=q*nt[3];cash=0.;trades.append(('overnight',str(day.date()),nt[0]))
   else:nav=cash
   rows.append((day,nav))
  nav=pd.Series(dict(rows));r=nav.pct_change();r.iloc[0]=nav.iloc[0]/INIT-1
  peak=np.maximum.accumulate(np.r_[INIT,nav.to_numpy()])[1:]
  stats=dict(cagr=(nav.iloc[-1]/INIT)**(365.25/((dates[-1]-dates[0]).days+1))-1,mdd=float(np.min(nav.to_numpy()/peak-1)),sharpe=r.mean()/r.std(ddof=1)*252**.5,total_return=nav.iloc[-1]/INIT-1)
  results[version,leg]=dict(nav=nav,trades=trades,stats=stats)
  print(version,leg,'trades',len(trades),stats,flush=True)

expected_gate=pd.DataFrame(gate_rows).set_index(['date','strategy']).sort_index()
actual_gate=pd.read_csv(OUT/'shadow_gate.csv',parse_dates=['date']).set_index(['date','strategy']).sort_index()
assert actual_gate.index.equals(expected_gate.index)
for field in ['mode_before_exit','mode_after_exit','entry_mode','has_exit']:
    assert np.array_equal(actual_gate[field],expected_gate[field]),('gate',field)
for field in ['reference_pnl','decision_ma20']:
    assert np.allclose(actual_gate[field],expected_gate[field],atol=.01,rtol=0,equal_nan=True),('gate',field)
actual=pd.read_csv(OUT/'daily_nav.csv',parse_dates=['date'])
actual_trades=pd.read_csv(OUT/'trades.csv',dtype={'symbol':str},parse_dates=['entry_time','exit_time'])
actual_orders=pd.read_csv(OUT/'orders.csv',parse_dates=['timestamp','reference_bar_end'])
summary=pd.read_csv(OUT/'summary.csv');yearly=pd.read_csv(OUT/'yearly.csv')
checks=[]
for (version,account),ref in results.items():
    g=actual[actual.version.eq(version)&actual.account.eq(account)].set_index('date').sort_index()
    assert g.index.equals(ref['nav'].index)
    error=float(np.max(np.abs(g.nav-ref['nav'])))
    assert error<.01,(version,account,'NAV',error)
    t=actual_trades[actual_trades.version.eq(version)&actual_trades.account.eq(account)]
    ids=[(x.strategy,str(x.entry_time.date()),x.symbol.zfill(6)) for x in t.itertuples()]
    assert ids==ref['trades'],(version,account,'trade identities')
    if version=='V4':
        wanted=[int(expected_gate.loc[(x.entry_time.normalize(),x.strategy),'entry_mode']) for x in t.itertuples()]
        assert np.array_equal(t.entry_mode,wanted),(version,account,'actual entry modes')
    orders=actual_orders[actual_orders.version.eq(version)&actual_orders.account.eq(account)].copy()
    orders['date']=orders.timestamp.dt.normalize()
    previous=g.nav.shift();previous.iloc[0]=INIT
    returns=g.nav/previous-1
    turnover=orders.groupby('date').filled_notional.sum().reindex(g.index,fill_value=0)/previous
    assert np.allclose(turnover,g.turnover,atol=1e-10,rtol=0)
    cash=INIT;quantity=0.;symbol=None
    for day,row in g.iterrows():
        for order in orders[orders.date.eq(day)].itertuples():
            sym=str(order.symbol).zfill(6)
            price_bar=lookup[sym].loc[order.reference_bar_end]
            raw=float(price_bar.close if order.side=='BUY' else price_bar.open)
            assert abs(raw-order.reference_price)<1e-8,'reference field price'
            if order.side=='BUY':
                assert quantity==0,'overlap'
                fill=raw*(1+SLIP);wanted=cash/(fill*(1+FEE))
                assert np.isclose(wanted,order.quantity,rtol=1e-12)
                quantity=order.quantity;symbol=sym;cash-=quantity*fill*(1+FEE)
                assert order.reference_field=='close'
            else:
                assert symbol==sym and np.isclose(quantity,order.quantity,rtol=1e-12)
                fill=raw*(1-SLIP);cash+=quantity*fill*(1-FEE);quantity=0.;symbol=None
            assert np.isclose(fill,order.fill_price,rtol=1e-12)
            assert np.isclose(order.commission,order.quantity*fill*FEE,rtol=1e-12)
            assert np.isclose(order.filled_notional,order.quantity*fill,rtol=1e-12)
        marked=cash if not quantity else cash+quantity*lookup[symbol].loc[ts(day,'15:30'),'close']
        assert abs(marked-row.nav)<.01,'cash replay'
    s=summary[summary.version.eq(version)&summary.account.eq(account)].iloc[0]
    expected=dict(ref['stats'],turnover=turnover.sum(),annualized_turnover=turnover.sum()*252/len(g))
    for name,value in expected.items():
        assert np.isclose(s[name],value,atol=1e-10,rtol=1e-10),(version,account,name)
    ys=yearly[yearly.version.eq(version)&yearly.account.eq(account)]
    assert np.isclose(np.prod(1+ys.total_return)-1,ref['stats']['total_return'],atol=1e-10)
    for year,part in g.groupby(g.index.year):
        start=previous.loc[part.index[0]];r=returns.loc[part.index];turn=turnover.loc[part.index]
        published=ys[ys.year.eq(year)].iloc[0]
        peak=np.maximum.accumulate(np.r_[start,part.nav.to_numpy()])[1:]
        want=dict(total_return=part.nav.iloc[-1]/start-1,cagr=(part.nav.iloc[-1]/start)**(365.25/((part.index[-1]-part.index[0]).days+1))-1,
                  mdd=np.min(part.nav.to_numpy()/peak-1),sharpe=r.mean()/r.std(ddof=1)*252**.5,turnover=turn.sum(),annualized_turnover=turn.sum()*252/len(part))
        for name,value in want.items():
            assert np.isclose(published[name],value,atol=1e-10,rtol=1e-10,equal_nan=True),(version,account,year,name)
    checks.append(dict(version=version,account=account,sessions=len(g),trades=len(t),max_absolute_nav_error_krw=error))
signals=pd.read_csv(OUT/'execution_signals.csv',parse_dates=['date','mode_source_time_at_entry'])
for row in signals.itertuples():
    entry=ts(row.date,'09:06' if row.strategy=='intraday' else '15:20')
    assert pd.isna(row.mode_source_time_at_entry) or row.mode_source_time_at_entry<entry,'future mode information'
for src in manifest['source_files']:
    assert hashlib.sha256(Path(src['path']).read_bytes()).hexdigest()==src['sha256']
record=dict(status='PASS',method='Independent raw-source gate and account reconstruction; all orders and performance metrics replayed',
            accounts=checks,gate_session_observations=len(actual_gate),no_future_mode_timestamps=True,source_files_verified=6,
            code_sha256=manifest['code_sha256'],artifact_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.glob('*.csv')})
(OUT/'independent_check.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
print('PASS: V3/V4 six account NAVs, trade identities, separate shadow gate clocks, actual ETF order prices, all metrics and sources')
