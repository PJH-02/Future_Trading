// KRX 정보데이터시스템 [15007] 파생상품 투자자별 거래실적(일별추이) → 표준 스키마 CSV 한 파일
//
// 전제: 사용자가 data.krx.co.kr 에 직접 로그인한 브라우저 탭. 이 스크립트는 로그인하지 않으며 자격증명을 다루지 않는다.
// 실행: 그 탭의 개발자 콘솔(또는 브라우저 자동화 도구)에서 전체를 붙여 실행.
// 결과: krx_opt_investor_daily.csv 다운로드 (date,investor,cp,buy_qty,sell_qty,buy_amt,sell_amt) → hyfe/data/raw/ 로 옮김
// 정의는 09-Decisions/2026-09-27-decision-optflow-preregistration.md 와 같다(월물, 정규장, 외국인 합계).
(async () => {
  const CFG = { isuCd: 'KR___OPK2I', aggBasTpCd: '0', startYear: 2010, end: '20260923', sleepMs: 500, download: true };
  const COLS = { A07: 'institution', A08: 'othercorp', A09: 'individual', A12: 'foreign', AMT_OR_QTY: 'total' };
  const sleep = ms => new Promise(r => setTimeout(r, ms));
  const q = async p => {
    const r = await fetch('/comm/bldAttendant/getJsonData.cmd', {
      method: 'POST', credentials: 'include',
      headers: { 'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8', 'X-Requested-With': 'XMLHttpRequest' },
      body: new URLSearchParams(p).toString(),
    });
    const t = await r.text();
    let j;
    try { j = JSON.parse(t); } catch (e) { throw new Error(`KRX 응답 이상(로그인 만료?): ${t.slice(0, 80)}`); }
    if (!Array.isArray(j.output)) throw new Error(`output 없음: ${t.slice(0, 80)}`);
    return j.output;
  };
  const num = v => { const s = String(v ?? '').replace(/,/g, '').trim(); return s === '' || s === '-' ? '' : Number(s); };
  const endYear = Number(CFG.end.slice(0, 4));
  const chunks = [];
  for (let y = CFG.startYear; y <= endYear; y++) chunks.push([`${y}0101`, y === endYear ? CFG.end : `${y}1231`]);

  const data = new Map();
  let requests = 0;
  for (const cp of ['C', 'P']) for (const [unit, prtType] of [['amt', 'AMT'], ['qty', 'QTY']]) for (const [side, prtCheck] of [['buy', 'SU'], ['sell', 'DO']]) {
    for (const [s, e] of chunks) {
      const rows = await q({ bld: 'dbms/MDC/STAT/standard/MDCSTAT13102', locale: 'ko_KR', inqTpCd: '2', isuCd: CFG.isuCd, isuOpt: cp,
        aggBasTpCd: CFG.aggBasTpCd, prtType, prtCheck, money: '1', share: '1', strtDd: s, endDd: e, csvxls_isNo: 'false' });
      requests++;
      for (const r of rows) {
        const d = r.TRD_DD.replace(/\//g, '-');
        for (const [k, inv] of Object.entries(COLS)) {
          const key = `${d}|${inv}|${cp}`;
          if (!data.has(key)) data.set(key, { date: d, investor: inv, cp });
          data.get(key)[`${side}_${unit}`] = num(r[k]);
        }
      }
      await sleep(CFG.sleepMs);
    }
  }
  const recs = [...data.values()].sort((a, b) => (a.date + a.investor + a.cp).localeCompare(b.date + b.investor + b.cp));
  const F = ['buy_qty', 'sell_qty', 'buy_amt', 'sell_amt'];
  const incomplete = recs.filter(r => F.some(f => r[f] === undefined || r[f] === '')).length;
  // 무결성: 날짜·콜풋마다 4개 투자자 합 = 전체
  const byDC = new Map();
  for (const r of recs) { const k = `${r.date}|${r.cp}`; if (!byDC.has(k)) byDC.set(k, {}); byDC.get(k)[r.investor] = r; }
  let sumMismatch = 0;
  for (const g of byDC.values()) for (const f of F) {
    const parts = ['institution', 'othercorp', 'individual', 'foreign'].map(i => (g[i] && typeof g[i][f] === 'number') ? g[i][f] : NaN);
    if (g.total && typeof g.total[f] === 'number' && Math.abs(parts.reduce((a, b) => a + b, 0) - g.total[f]) > 1) sumMismatch++;
  }
  const csv = ['date,investor,cp,' + F.join(','), ...recs.map(r => [r.date, r.investor, r.cp, ...F.map(f => r[f] ?? '')].join(','))].join('\n');
  if (CFG.download) {
    const a = document.createElement('a');
    a.href = URL.createObjectURL(new Blob([csv], { type: 'text/csv' }));
    a.download = 'krx_opt_investor_daily.csv';
    document.body.appendChild(a); a.click(); a.remove();
  }
  const dates = [...new Set(recs.map(r => r.date))];
  return { requests, rows: recs.length, dates: dates.length, first: dates[0], last: dates[dates.length - 1], incomplete, sumMismatch, bytes: csv.length };
})();
