"""Independent reference calculation: does NOT import the backtest engine.
Run AFTER the default backtest. Reads results/manifest.json for configuration.
Compares raw-source signals, every account/date NAV, trade identities, metrics,
turnover, source hashes, and calendar-year compounding. Raises on mismatch.
"""
from pathlib import Path
import json
import hashlib
import pandas as pd
import numpy as np
HERE=Path(__file__).resolve().parent
OUT=HERE/'results'
manifest=json.loads((OUT/'manifest.json').read_text())
cfg=manifest['config']
initial=float(cfg['initial_capital'])
root=Path(manifest['source_files'][0]['path']).parent
m={s:pd.read_csv(root/'minute'/f'{s}_TOSS_1m.csv',parse_dates=['timestamp']) for s in ['229200','233740','251340']}
d=pd.read_csv(root/'229200_daily.csv',parse_dates=['date']).sort_values('date').set_index('date')
prior=d.close.shift(); ma=d.close.rolling(10).mean().shift()
lo=max(cfg['start'],*(x.date.min() for x in m.values())); hi=min(cfg['end'],*(x.date.max() for x in m.values()))
dates=d.index[(d.index>=lo)&(d.index<=hi)]
nextday=dict(zip(d.index[:-1],d.index[1:]))
lookup={s:x.set_index('timestamp') for s,x in m.items()}
groups={s:dict(tuple(x.groupby('date'))) for s,x in m.items()}
def bar(s,t):
 try:return lookup[s].loc[t]
 except KeyError:return None
def valid(b):return b is not None and b.volume>0 and b.open>0
signals={}; skips=[]
for ver,cut in [('V1','15:16'),('V2','15:20')]:
 for day in dates:
  stamp=lambda time:pd.Timestamp(str(day.date())+' '+time)
  b0,b5=bar('229200',stamp('09:01')),bar('229200',stamp('09:06'))
  ds=0
  if b0 is not None and b5 is not None and np.isfinite(ma.loc[day]):
   r=b5.open/b0.open-1
   ds=1 if prior.loc[day]>ma.loc[day] and r>=.003-1e-12 else -1 if prior.loc[day]<ma.loc[day] and r<=-.003+1e-12 else 0
  symbol='233740' if ds==1 else '251340'
  ent,ex=bar(symbol,stamp('09:07')),bar(symbol,stamp('15:01'))
  daytrade=(symbol,float(ent.open),float(ex.open)) if ds and valid(ent) and valid(ex) else None
  x=groups['229200'].get(str(day.date())); ns=0; nt=None
  if x is not None:
   w=x[x.timestamp.between(stamp('09:01'),stamp(cut))]
   expected=pd.date_range(stamp('09:01'),stamp(cut),freq='min')
   if len(w)==len(expected) and set(w.timestamp)==set(expected):
    h,l=w.high.max(),w.low.min(); ibs=(w.close.iloc[-1]-l)/(h-l) if h>l else np.nan
    ns=1 if ibs<.3 else -1 if ibs>.7 else 0
    symbol='233740' if ns==1 else '251340'
    nx=nextday.get(day)
    if ns and nx is not None and nx<=dates[-1]:
     p=groups[symbol].get(str(day.date()))
     if p is not None:
      p=p[(p.timestamp>=stamp('09:01')) & (p.timestamp<=stamp('15:20')) & (p.volume>0)]
      ex=bar(symbol,nx+pd.Timedelta(hours=9,minutes=1)); mark=bar(symbol,stamp('15:30'))
      if len(p) and valid(ex) and mark is not None:
       nt=(symbol,float(p.close.iloc[-1]),float(ex.open),float(mark.close),nx)
  signals[ver,day]=(daytrade,nt)
results={}; fee=cfg['commission_bps']/10000; slip=cfg['slippage_bps']/10000
factor=(1-slip)*(1-fee)/((1+slip)*(1+fee))
for ver in ['V1','V2']:
 for sleeve in ['intraday','overnight','combined']:
  cash=initial; pos=None; rows=[]; trades=[]
  for day in dates:
   if pos:
    cash=pos[0]*pos[1]*(1-slip)*(1-fee);pos=None
   dt,nt=signals[ver,day]
   if dt and sleeve in ['intraday','combined']:
    cash*=dt[2]/dt[1]*factor; trades.append(('intraday',str(day.date()),dt[0]))
   if nt and sleeve in ['overnight','combined']:
    qty=cash/(nt[1]*(1+slip)*(1+fee));pos=(qty,nt[2]);nav=qty*nt[3]; trades.append(('overnight',str(day.date()),nt[0]));cash=0
   else:nav=cash
   rows.append((day,nav))
  nav=pd.Series(dict(rows)); ret=nav.pct_change();ret.iloc[0]=nav.iloc[0]/initial-1
  dd=nav/np.maximum.accumulate(np.r_[initial,nav.values])[1:]-1
  cagr=(nav.iloc[-1]/initial)**(365.25/((dates[-1]-dates[0]).days+1))-1
  sharpe=ret.mean()/ret.std(ddof=1)*252**.5
  print(ver,sleeve,len(dates),str(dates[0].date()),str(dates[-1].date()),'trades',len(trades),'return',nav.iloc[-1]/initial-1,'cagr',cagr,'mdd',dd.min(),'sharpe',sharpe,flush=True)
  results[ver,sleeve]={'nav':nav,'trades':trades}

actual=pd.read_csv(OUT/'daily_nav.csv',parse_dates=['date'])
trade_log=pd.read_csv(OUT/'trades.csv',dtype={'symbol':str},parse_dates=['entry_time','exit_time'])
orders=pd.read_csv(OUT/'orders.csv',parse_dates=['timestamp'])
summary=pd.read_csv(OUT/'summary.csv')
yearly=pd.read_csv(OUT/'yearly.csv')
audit=[]
for (version,sleeve),reference in results.items():
    obs=actual[(actual.version==version)&(actual.account==sleeve)].set_index('date').sort_index()
    expected=reference['nav']
    assert obs.index.equals(expected.index), (version,sleeve,'calendar mismatch')
    error=float(np.max(np.abs(obs.nav.to_numpy()-expected.to_numpy())))
    assert error<.01, (version,sleeve,'NAV mismatch',error)
    t=trade_log[(trade_log.version==version)&(trade_log.account==sleeve)]
    identities=[(row.strategy,str(row.entry_time.date()),str(row.symbol).zfill(6)) for row in t.itertuples()]
    assert identities==reference['trades'], (version,sleeve,'trade identity mismatch')
    prev=obs.nav.shift();prev.iloc[0]=initial
    ret=obs.nav/prev-1
    ledger=orders[(orders.version==version)&(orders.account==sleeve)].copy()
    ledger['date']=ledger.timestamp.dt.normalize()
    notionals=ledger.groupby('date').filled_notional.sum().reindex(obs.index,fill_value=0)
    turn=notionals/prev
    assert np.allclose(turn,obs.turnover,atol=1e-11,rtol=0)
    assert np.allclose(ret,obs.daily_return,atol=1e-12,rtol=0)
    # Independent replay of actual order quantities, prices and cash balances.
    cash=initial;quantity=0.;symbol=None
    for day,row in obs.iterrows():
        for order in ledger[ledger.date.eq(day)].itertuples():
            if order.side=='BUY':
                assert quantity==0, 'overlap'
                fill=order.reference_price*(1+slip)
                wanted=cash/(fill*(1+fee))
                assert np.isclose(order.quantity,wanted,rtol=1e-12)
                quantity=order.quantity;symbol=str(order.symbol).zfill(6)
                cash-=quantity*fill*(1+fee)
            else:
                assert np.isclose(quantity,order.quantity,rtol=1e-12)
                fill=order.reference_price*(1-slip)
                cash+=quantity*fill*(1-fee);quantity=0.;symbol=None
            assert np.isclose(fill,order.fill_price,rtol=1e-12)
            assert np.isclose(order.filled_notional,order.quantity*fill,rtol=1e-12)
            assert np.isclose(order.commission,order.quantity*fill*fee,rtol=1e-12)
        marked=cash
        if quantity:
            marked+=quantity*lookup[symbol].loc[day+pd.Timedelta(hours=15,minutes=30),'close']
        assert abs(marked-row.nav)<.01, (version,sleeve,day,'ledger replay mismatch')
    s=summary[(summary.version==version)&(summary.account==sleeve)].iloc[0]
    total=expected.iloc[-1]/initial-1
    peak=np.maximum.accumulate(np.r_[initial,expected.to_numpy()])[1:]
    expected_metrics={'total_return':total,'cagr':(1+total)**(365.25/((expected.index[-1]-expected.index[0]).days+1))-1,
                      'mdd':float(np.min(expected.to_numpy()/peak-1)),
                      'sharpe':float(ret.mean()/ret.std(ddof=1)*252**.5),
                      'turnover':float(turn.sum()),'annualized_turnover':float(turn.sum()*252/len(obs))}
    for field,value in expected_metrics.items():
        assert np.isclose(s[field],value,atol=1e-10,rtol=1e-10,equal_nan=True),(version,sleeve,field)
    y=yearly[(yearly.version==version)&(yearly.account==sleeve)]
    assert np.isclose(np.prod(1+y.total_return)-1,total,atol=1e-10)
    for yr,g in obs.groupby(obs.index.year):
        yr_start=prev.loc[g.index[0]]
        published=y[y.year.eq(yr)].iloc[0]
        yr_returns=ret.loc[g.index]
        yr_dd=g.nav/np.maximum.accumulate(np.r_[yr_start,g.nav.to_numpy()])[1:]-1
        yr_cagr=(g.nav.iloc[-1]/yr_start)**(365.25/((g.index[-1]-g.index[0]).days+1))-1
        checks={'total_return':g.nav.iloc[-1]/yr_start-1,'mdd':yr_dd.min(),'cagr':yr_cagr,
                'sharpe':yr_returns.mean()/yr_returns.std(ddof=1)*252**.5,
                'turnover':turn.loc[g.index].sum(),'annualized_turnover':turn.loc[g.index].sum()*252/len(g)}
        for field,value in checks.items():
            assert np.isclose(published[field],value,atol=1e-10,rtol=1e-10,equal_nan=True),(version,sleeve,yr,field)
    audit.append(dict(version=version,account=sleeve,sessions=len(obs),trades=len(identities),max_nav_absolute_error_krw=error,
                      trade_identities_equal=True,order_cash_replay_equal=True,summary_and_yearly_metrics_equal=True))
for src in manifest['source_files']:
    assert hashlib.sha256(Path(src['path']).read_bytes()).hexdigest()==src['sha256'],'source hash changed'
record=dict(status='PASS',method='Independent raw-data reference, without importing engine; chronological order-ledger replay; all metrics/yearly reconciliation',
            checks=audit,source_hashes_verified=len(manifest['source_files']),
            code_sha256=hashlib.sha256((HERE/'backtest_daynight.py').read_bytes()).hexdigest(),
            daily_nav_sha256=hashlib.sha256((OUT/'daily_nav.csv').read_bytes()).hexdigest())
(OUT/'independent_check.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
print('PASS: all six account NAVs, trade identities, order-ledger cash, full/yearly metrics, turnover and source hashes')
