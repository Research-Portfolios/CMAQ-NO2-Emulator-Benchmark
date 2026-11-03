"""Writes the input-check record of the second pass and its JSON summary from the check counts, and prints the verdict."""
import collections
import datetime
import os
import sys
sys.path.insert(0, '/content/emul/run/code')
from s3 import s3_d4 as D
from s3 import s3_d4_record as R
from s3 import s3common as S
from s3p2 import p2common as P
g = R.g

def main():
    st = P.jload(os.path.join(P.WORK_P2, 'd4_stats.json'))
    mc = st['mask_cells']
    A = R.agg(st['answers'], mc)
    L = []

    def add(first, text='', cont='  '):
        if text == '':
            L.append(first)
        else:
            L.extend(S.wrap(first, text, cont))
    add('SECOND PASS INPUT CHECKS (the second version')
    add('step (3)')
    add('). Root .; W is')
    add('work (local machine only).')
    add('Made by run\\code\\s3p2\\p2_d4.py (read %s to %s' % (st['start'][11:], st['end'][11:]))
    add("%d worker processes; the earlier version's s3\\s3_d4.py check_answer and" % st['workers'])
    add('check_extract used as they are) and p2_d4_record.py from')
    add('W\\arrays_p2\\work\\d4_stats.json, before any array of the second pass was')
    add('built. Inputs: the files of run\\config\\')
    add('S2C_INVENTORY.json with s3_input true only. Masked cells (this')
    add("stage's mask W\\arrays_p2\\mask_conus_land.npy, sha16 %s): %d." % (st['mask']['sha256'][:16], mc))
    add('')
    add('1. HOW EACH FILE WAS READ')
    add('- ', 'RSIG answers: read whole, bytes and sha256 against the inventory; decompressed in memory (nothing written), decompressed bytes and sha256 against the inventory; opened from memory with netCDF4; the field read as stored (no mask or scale).')
    nfail = sum((1 for a in st['answers'] if not all(a['checks'].values())))
    add('- ', 'Answers read: %d (T-1, M-1 to M-9 and X-6: 660 monthly; 5 lag answers of 2014-12-31; T-2: 12); with a failed byte or sha256 check: %d.' % (len(st['answers']), nfail))
    add('- ', 'MCIP members and the %d extracts: sha256 and bytes against the inventory (section 5).' % len(st['extracts']))
    add('- ', 'Fill values: a value at or below -9e36 counts as -9.999e+36; NaN counted apart. Dot fields: counted on the 300 x 460 points, and on the masked cells after the four-corner mean (a cell with a fill or NaN corner counts as missing).')
    add('')
    add('2. RSIG SERIES BY YEAR: COVERAGE, FORM, FILL VALUES (masked cells)')
    add('  series        year ans days miss dup  fill+NaN masked  fraction')
    stop = []
    for s in R.MONTHLY:
        for y in S.YEARS:
            r = A[s, y]
            add('  %-13s %4d %3d %4d %4d %3d %16d  %.2e' % (R.SER_LABEL[s], y, r['answers'], r['steps'], len(r['days_missing']), len(r['days_duplicate']), r['fill_or_nan_mask'], r['fill_or_nan_mask_fraction']))
            if r['fill_or_nan_mask_fraction'] > 0.001:
                stop.append('%s %d' % (s, y))
    r2 = A['T2', 2016]
    add('  %-13s %4d %3d %4d %4d %3d %16d  %.2e' % (R.SER_LABEL['T2'], 2016, r2['answers'], r2['steps'], len(r2['days_missing']), len(r2['days_duplicate']), r2['fill_or_nan_mask'], r2['fill_or_nan_mask_fraction']))
    add('  ans: answers; days: steps read; miss: days of the year with no step;')
    add('  dup: dates read more than once; masked cell-days = %d x days.' % mc)
    form = collections.Counter()
    for (s, y), r in A.items():
        for k in ('bad_shape', 'grid_diff', 'bad_nlays', 'bad_tstep', 'tflag_vars_differ', 'dates_vs_request', 'checks_failed'):
            form[k] += r[k]
        if r['days_extra']:
            form['days_extra'] += len(r['days_extra'])
    lag = [a for a in st['answers'] if a['period'] == '20141231']
    ngd = sum((1 for a in st['answers'] if a['grid_diff']))
    add('- ', 'Form, over all %d monthly and T-2 answers: shape not (days, 1, 299, 459) or, for M-4 and M-5, (days, 1, 300, 460): %d; NLAYS not 1: %d; TSTEP not 240000: %d; TFLAG dates not those of the request: %d; TFLAG differing between variables: %d; dates outside the year: %d.' % (sum((r['answers'] for r in A.values())), form['bad_shape'], form['bad_nlays'], form['bad_tstep'], form['dates_vs_request'], form['tflag_vars_differ'], form['days_extra']))
    add('- ', "THE GRID OF EVERY ANSWER: GDTYP, P_ALP, P_BET, P_GAM, XCENT, YCENT, XCELL, YCELL, NCOLS, NROWS, XORIG and YORIG of each of the %d answers (the lag answers included) against EPA's printed 12US1 grid (cross: 459 x 299, XORIG -2556000, YORIG -1728000; dot: 460 x 300, -2562000, -1734000; P_ALP 33, P_BET 45, P_GAM and XCENT -97, YCENT 40, cells 12000 m; the earlier version's GRIDCRO and DOT values, found equal to them: %s): answers differing %d." % (len(st['answers']), st['ad55_grid_values_equal_epa'], ngd))
    tms = sorted(set((t for r in A.values() for t in r['times'])))
    fmts = sorted(set((t for r in A.values() for t in r['formats'])))
    add('- ', 'TFLAG times of the daily steps: %s; file format: %s.' % (tms, ', '.join(fmts)))
    add('- ', 'Coverage: every series T-1, M-1 to M-9 and X-6 holds every day of 2015-01-01 to 2019-12-31 (1826 days) once: %s.' % all((not A[s, y]['days_missing'] and (not A[s, y]['days_duplicate']) for s in R.MONTHLY for y in S.YEARS)))
    add('- ', 'The lag day: %d answers (%s), each %s step(s) dated %s; fill or NaN on masked cells: %d.' % (len(lag), ', '.join(sorted((a['series'] for a in lag))), sorted(set((len(a['tflag_dates']) for a in lag))), sorted(set((d for a in lag for d in a['tflag_dates']))), sum((st2.get('fill_mask', 0) + st2.get('nan_mask', 0) + st2.get('fill_or_nan_mask_cells', 0) for a in lag for st2 in a['steps']))))
    wind = collections.Counter()
    for a in st['answers']:
        if 'var_attrs' in a:
            va = a['var_attrs']
            wind['%s: long_name %s; var_desc %s; units %s' % (a['series'], va.get('long_name'), va.get('var_desc'), va.get('units'))] += 1
    for k, n in sorted(wind.items()):
        add('- ', "the wind's attributes, %s (%d answers)." % (k, n))
    add('')
    add('3. UNITS AGAINST THE CHANNEL TABLE')
    for s in R.MONTHLY + ('T2',):
        seen = sorted(set((u for (s2, y), r in A.items() if s2 == s for u in r['units'])))
        lagu = sorted(set((a['units'] for a in lag if a['series'] == s)))
        c2, ok = S.C2_UNITS[s]
        good = all((u in ok for u in seen + lagu))
        add('  %-16s expected %-11s answers %-10s %s' % (R.SER_LABEL[s], c2, '/'.join(seen), 'ok' if good else 'DIFFERS'))
    add('')
    add('4. RANGES (counted, kept, never altered; all cells / masked cells)')
    add('  series        year  range         out(all) out(mask)  min(mask)  max(mask)')
    add("  (PBLH's range 0..6000 m is applied to M-8 and to X-6, the daily maximum)")
    for s in R.MONTHLY:
        lo, hi = D.RANGES.get(s, (None, None))
        rg = '%s..%s' % (g(lo), g(hi)) if lo is not None else 'none'
        for y in S.YEARS:
            r = A[s, y]
            add('  %-13s %4d  %-13s %8d %9d %10s %10s' % (R.SER_LABEL[s], y, rg, r['below_all'] + r['above_all'], r['below_mask'] + r['above_mask'], g(r['min_mask']), g(r['max_mask'])))
    add('  ', '(M-4 and M-5 have no checked range: their extremes are given for the record; cell values after the four-corner mean.)')
    for s in ('M4', 'M5'):
        mn = min((A[s, y]['dot_min'] for y in S.YEARS))
        mx = max((A[s, y]['dot_max'] for y in S.YEARS))
        add('  ', '%s on the dot points, 2015-2019: min %s, max %s.' % (R.SER_LABEL[s], g(mn), g(mx)))
    add('  ', 'WSPD10 by its range: every value of M-3 lies in 0 to 50 m/s: %s (out of range, all cells: %d).' % (all((A['M3', y]['below_all'] + A['M3', y]['above_all'] == 0 for y in S.YEARS)), sum((A['M3', y]['below_all'] + A['M3', y]['above_all'] for y in S.YEARS))))
    add('')
    add('5. MCIP AND EMISSIONS')
    for name, rec in sorted(st['mcip'].items()):
        if rec.get('used') is False:
            add('  ', '%s: sha256 %s; not used (PURB is in GRIDCRO2D).' % (name, 'ok' if rec['sha256_ok'] else 'DIFFERS'))
            continue
        bits = []
        for n, vr in rec['vars'].items():
            t = '%s %s %s fill %d NaN %d min %s max %s' % (n, vr['units'] or '-', 'x'.join((str(x) for x in vr['shape'])), vr['fill'], vr['nan'], g(vr['min']), g(vr['max']))
            if n == 'LWMASK':
                t += ' values %s' % vr['values']
            if n == 'DLUSE':
                t += ' (%d distinct values, all whole numbers %s, exact in float16 %s; on the masked cells %d of them)' % (vr['n_values'], vr['all_whole_numbers'], vr['float16_exact'], len(vr['values_on_masked_cells']))
            if 'records_used' in vr:
                t += ' records %d-%d (%d, hours %d-%d)' % (vr['records_used'][0], vr['records_used'][1], vr['records_used_n'], vr['record_hours'][0] // 10000, vr['record_hours'][-1] // 10000)
            for k in ('below_0', 'above_1', 'above_100'):
                if k in vr:
                    t += ' %s %d' % (k, vr[k])
            bits.append(t)
        if 'present' in rec:
            bits.append('GSW %s, RGRND %s, ALBEDO %s' % tuple(('present' if rec['present'][k] else 'absent' for k in ('GSW', 'RGRND', 'ALBEDO'))))
        add('  ', '%s: sha256 %s; %s; TFLAG %s to %s, %d steps; grid attributes %s; %s.' % (name, 'ok' if rec['sha256_ok'] else 'DIFFERS', rec['data_model'], rec['tflag_first'], rec['tflag_last'], rec['steps'], 'equal' if not rec['grid_diff'] else 'DIFFER %s' % rec['grid_diff'], '; '.join(bits)))
    dl = st['mcip']['GRIDCRO2D_20160701.nc4']['vars']['DLUSE']
    add('  ', 'L-6 DLUSE, the distinct category codes over all cells: %s; on the masked cells: %s.' % (', '.join((g(v) for v in dl['values'])), ', '.join((g(v) for v in dl['values_on_masked_cells']))))
    ex = st['extracts']
    for kind in ('emis_merged', 'pt_marine'):
        grp = collections.defaultdict(list)
        for p, r in ex.items():
            if r['kind'] != kind:
                continue
            key = (r.get('sector', 'merged'), r['date'][:4])
            grp[key].append(r)
        for (sec, y), lst in sorted(grp.items()):
            dates = collections.Counter((r['date'] for r in lst))
            dup = [d for d, n in dates.items() if n > 1]
            tfl = sum((1 for r in lst if S.yyyyddd(datetime.date(int(r['date'][:4]), int(r['date'][4:6]), int(r['date'][6:]))) != r['tflag_date']))
            units = sorted(set(('%s %s' % (s, u) for r in lst for s, u in zip(r['species'], r['units']))))
            nonf = sum((a['nonfinite'] for r in lst for a in r['arrays'].values()))
            neg = sum((a['negative'] for r in lst for a in r['arrays'].values()))
            zero = collections.Counter((s for r in lst for s, a in r['arrays'].items() if a['all_zero']))
            shp = sorted(set(('%s %s' % ('x'.join(map(str, a['shape'])), a['dtype']) for r in lst for a in r['arrays'].values())))
            bad = sum((1 for r in lst if not (r['sha256_ok'] and r['bytes_ok'])))
            hrs = sorted(set((r['hours'] for r in lst)))
            t = "%s %s: %d files, %d dates, duplicates %d, TFLAG date not the file's date %d; sha256 or bytes differing %d; %s; %s; hours %s; non-finite %d, negative %d" % (sec, y, len(lst), len(dates), len(dup), tfl, bad, ', '.join(shp), ', '.join(units), ', '.join(hrs), nonf, neg)
            if zero:
                t += '; all-zero fields: %s' % ', '.join(('%s %d' % kv for kv in sorted(zero.items())))
            if kind == 'pt_marine':
                outs = sum((int(r.get('n_outside', 0)) for r in lst))
                ns = sorted(set((int(r['n_stacks']) for r in lst)))
                t += '; stacks %s, outside the grid %d' % (ns, outs)
                if sec in S.REP_SECTORS:
                    pm = collections.Counter((r['date'][4:6] for r in lst))
                    t += '; members a month %s' % '/'.join((str(pm[m]) for m in sorted(pm)))
            add('  ', t + '.')
    add('  ', "P-1 is formed from the point and marine extracts' NO, NO2 and HONO above (moles/s, the units of the channel table; finite and not negative as counted), so its inputs pass the input checks; its own values are counted at step (4) (the second pass's report).")
    add('')
    add('6. WHAT IS EXCLUDED, AND WHY')
    for t in ("L-4 ALBEDO: not in METCRO2D; not built (built from GSW and RGRND only if the arm of SALMAN's 18 runs, before its training), so 25 input channels are built: the primary's 17, the advanced set's 23 and the arm's 18 (19 with L-4).", 'T-2 (lstaconc DAILY_NO2, 2016): check only, not an input or a target.', 'The first try of M4_metdot3d_UWIND_L1_DM_201607, the 11 test answers of the download and their 20 decompressed copies in W\\work\\rsig: s3_input false in the inventory; not read.', "The answers' LONGITUDE, LATITUDE and ELEVATION: not inputs (LONGITUDE and LATITUDE serve the grid check only).", "MCIP: LUFRAC_CRO (PURB present), GRIDCRO2D's MSFX2, METCRO2D's other 40 variables and its 25th record (00 UTC of the 16th; the study design's rule for daily files): not used. GRIDCRO2D's DLUSE is now used, as L-6 (the DLUSE instruction).", 'Emissions: the fire sectors and rwc are not read; E-1 to E-4 take the merged files only; the six point and marine sectors enter as P-1.', 'No value of any input is excluded or altered for its range or for a fill value: the counts above are kept as read.'):
        add('- ', t)
    add('')
    add('7. STOP TEST (more than 0.1 percent of masked cell-days missing)')
    verdict = 'STOP' if stop or ngd or nfail else 'PASS'
    add('  ', "%s: series and years over the limit: %s; answers failing a byte or sha256 check: %d; answers whose grid differs from EPA's printed grid: %d." % (verdict, ', '.join(stop) if stop else 'none', nfail, ngd))
    text = '\n'.join(L) + '\n'
    bad = S.check_record_text(text.rstrip('\n'))
    if bad:
        raise ValueError('; '.join(bad[:5]))
    with open(P.D4_RECORD + '.partial', 'w', encoding='ascii', newline='\n') as f:
        f.write(text)
    os.replace(P.D4_RECORD + '.partial', P.D4_RECORD)
    summ = {'verdict': verdict, 'over_limit': stop, 'grid_differ': ngd, 'checks_failed': nfail, 'series_year': {'%s_%d' % k: {kk: vv for kk, vv in v.items() if kk not in ('variables',)} for k, v in A.items()}}
    P.write_json(os.path.join(P.WORK_P2, 'd4_summary.json'), summ)
    print('verdict %s; record %d lines' % (verdict, len(L)))
if __name__ == '__main__':
    P.run_main(main)
