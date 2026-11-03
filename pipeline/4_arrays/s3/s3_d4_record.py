"""Writes the input-check record and its JSON summary from the input-check counts (coverage, duplicates, units, dimensions and grid, fills and NaN, value ranges, the MCIP and emission checks and what was excluded), and prints the verdict."""
import collections
import datetime
import json
import os
import sys
sys.path.insert(0, '/content/emul/run/code')
from s3 import s3common as S
from s3 import s3_d4 as D
OUT = os.path.join(S.LOGS, 'S3_D4_CHECKS_ASCII.txt')
MONTHLY = ('T1', 'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9', 'X6')
SER_LABEL = {'T1': 'T-1 NO2', 'M1': 'M-1 PRES', 'M2': 'M-2 SFC_TMP', 'M3': 'M-3 WSPD10', 'M4': 'M-4 UWIND dot', 'M5': 'M-5 VWIND dot', 'M6': 'M-6 SOL_RAD', 'M7': 'M-7 QV', 'M8': 'M-8 PBLH', 'M9': 'M-9 precip', 'X6': 'X-6 PBLH max', 'T2': 'T-2 DAILY_NO2'}

def g(x):
    if x is None:
        return 'none'
    return '%.6g' % x

def agg(answers, mask_cells):
    out = {}
    for a in answers:
        s, per = (a['series'], a['period'])
        if per == '20141231':
            continue
        y = int(per[:4])
        r = out.setdefault((s, y), {'answers': 0, 'dates': collections.Counter(), 'units': set(), 'formats': set(), 'bad_shape': 0, 'grid_diff': 0, 'bad_nlays': 0, 'bad_tstep': 0, 'times': set(), 'tflag_vars_differ': 0, 'dates_vs_request': 0, 'checks_failed': 0, 'fill_or_nan_mask': 0, 'fill_all': 0, 'nan_all': 0, 'below_all': 0, 'above_all': 0, 'below_mask': 0, 'above_mask': 0, 'min_all': None, 'max_all': None, 'min_mask': None, 'max_mask': None, 'dot_below': 0, 'dot_above': 0, 'dot_min': None, 'dot_max': None, 'steps': 0, 'variables': set()})
        r['answers'] += 1
        r['units'].add(a['units'])
        r['formats'].add(a['format'])
        r['variables'].add(','.join(a['variables']))
        var, _u, dot = S.SERIES[s]
        want = [len(a['expected_dates']), 1] + ([300, 460] if dot else [299, 459])
        r['bad_shape'] += int(a['shape'] != want)
        r['grid_diff'] += int(bool(a['grid_diff']))
        r['bad_nlays'] += int(a['nlays'] != 1)
        r['bad_tstep'] += int(a['tstep'] != 240000)
        r['times'].update(a['tflag_times'])
        r['tflag_vars_differ'] += int(not a['tflag_vars_same'])
        r['dates_vs_request'] += int(a['tflag_dates'] != a['expected_dates'])
        r['checks_failed'] += int(not all(a['checks'].values()))
        for st in a['steps']:
            r['dates'][st['date']] += 1
            r['steps'] += 1
            r['fill_all'] += st['fill_all']
            r['nan_all'] += st['nan_all']
            if dot:
                r['fill_or_nan_mask'] += st['fill_or_nan_mask_cells']
                dd = st['dot']
                r['dot_below'] += dd['below_all']
                r['dot_above'] += dd['above_all']
                for k, f in (('dot_min', min), ('dot_max', max)):
                    v = dd['min_all' if k == 'dot_min' else 'max_all']
                    if v is not None:
                        r[k] = v if r[k] is None else f(r[k], v)
            else:
                r['fill_or_nan_mask'] += st['fill_mask'] + st['nan_mask']
            for k in ('below_all', 'above_all', 'below_mask', 'above_mask'):
                r[k] += st[k]
            for k, f in (('min_all', min), ('max_all', max), ('min_mask', min), ('max_mask', max)):
                v = st[k]
                if v is not None:
                    r[k] = v if r[k] is None else f(r[k], v)
    for (s, y), r in out.items():
        days = [S.yyyyddd(d) for d in S.days_of_year(y)]
        r['days_expected'] = len(days)
        r['days_missing'] = [d for d in days if d not in r['dates']]
        r['days_duplicate'] = [d for d, n in r['dates'].items() if n > 1]
        r['days_extra'] = [d for d in r['dates'] if d not in set(days)]
        r['masked_cell_days'] = mask_cells * len(days)
        r['fill_or_nan_mask_fraction'] = r['fill_or_nan_mask'] / float(r['masked_cell_days'])
        del r['dates']
        for k in ('units', 'formats', 'times', 'variables'):
            r[k] = sorted(r[k])
    return out

def main():
    with open(os.path.join(S.WORK_S3, 'd4_stats.json'), 'r', encoding='ascii') as f:
        st = json.load(f)
    mc = st['mask_cells']
    A = agg(st['answers'], mc)
    L = []

    def add(first, text='', cont='  '):
        if text == '':
            L.append(first)
        else:
            L.extend(S.wrap(first, text, cont))
    add('S3 INPUT CHECKS (stage S3')
    add('step (3)).')
    add('Root .; W is work')
    add('(local machine only). Made by run\\code\\s3\\s3_d4.py (read')
    add('%s to %s, %d worker processes) and s3_d4_record.py from' % (st['start'][11:], st['end'][11:], st['workers']))
    add('W\\work\\s3\\d4_stats.json, before any array was built. Inputs:')
    add('the files of run\\config\\S2C_INVENTORY.json with')
    add('s3_input true only. Masked cells: %d.' % mc)
    add('')
    add('1. HOW EACH FILE WAS READ')
    add('- ', "RSIG answers: read whole, bytes and sha256 against the inventory; decompressed in memory (nothing written), decompressed bytes and sha256 against the inventory's decompressed_bytes and decompressed_sha256; opened from memory with netCDF4 1.7.4; the field read as stored (no mask or scale).")
    nfail = sum((1 for a in st['answers'] if not all(a['checks'].values())))
    add('- ', 'Answers read: %d (T-1, M-1 to M-9 and X-6: 660 monthly; 5 lag answers of 2014-12-31; T-2: 12); with a failed byte or sha256 check: %d.' % (len(st['answers']), nfail))
    add('- ', 'MCIP members and the 3274 extracts: sha256 and bytes against the inventory (section 5).')
    add('- ', 'Fill values: a value at or below -9e36 counts as -9.999e+36 (the null of the answers); NaN counted apart. Dot fields: counted on the 300 x 460 points, and on the masked cells after the four-corner mean (a cell with a fill or NaN corner counts as missing).')
    add('')
    add('2. RSIG SERIES BY YEAR: COVERAGE, FORM, FILL VALUES (masked cells)')
    add('  series        year ans days miss dup  fill+NaN masked  fraction')
    stop = []
    for s in MONTHLY:
        for y in S.YEARS:
            r = A[s, y]
            add('  %-13s %4d %3d %4d %4d %3d %16d  %.2e' % (SER_LABEL[s], y, r['answers'], r['steps'], len(r['days_missing']), len(r['days_duplicate']), r['fill_or_nan_mask'], r['fill_or_nan_mask_fraction']))
            if r['fill_or_nan_mask_fraction'] > 0.001:
                stop.append('%s %d' % (s, y))
    r2 = A['T2', 2016]
    add('  %-13s %4d %3d %4d %4d %3d %16d  %.2e' % (SER_LABEL['T2'], 2016, r2['answers'], r2['steps'], len(r2['days_missing']), len(r2['days_duplicate']), r2['fill_or_nan_mask'], r2['fill_or_nan_mask_fraction']))
    add('  ans: answers; days: steps read; miss: days of the year with no step;')
    add('  dup: dates read more than once; masked cell-days = %d x days.' % mc)
    form = collections.Counter()
    for (s, y), r in A.items():
        for k in ('bad_shape', 'grid_diff', 'bad_nlays', 'bad_tstep', 'tflag_vars_differ', 'dates_vs_request', 'checks_failed'):
            form[k] += r[k]
        if r['days_extra']:
            form['days_extra'] += len(r['days_extra'])
    add('- ', "Form, over all %d monthly and T-2 answers: shape not (days, 1, 299, 459) or, for M-4 and M-5, (days, 1, 300, 460): %d; grid attributes not GRIDCRO2D's (dot: half a cell beyond): %d; NLAYS not 1: %d; TSTEP not 240000: %d; TFLAG dates not those of the request: %d; TFLAG differing between variables: %d; dates outside the year: %d." % (sum((r['answers'] for r in A.values())), form['bad_shape'], form['grid_diff'], form['bad_nlays'], form['bad_tstep'], form['dates_vs_request'], form['tflag_vars_differ'], form['days_extra']))
    tms = sorted(set((t for r in A.values() for t in r['times'])))
    fmts = sorted(set((t for r in A.values() for t in r['formats'])))
    add('- ', 'TFLAG times of the daily steps: %s; file format: %s.' % (tms, ', '.join(fmts)))
    add('- ', 'Coverage: every series T-1, M-1 to M-9 and X-6 holds every day of 2015-01-01 to 2019-12-31 (1826 days) once: %s.' % all((not A[s, y]['days_missing'] and (not A[s, y]['days_duplicate']) for s in MONTHLY for y in S.YEARS)))
    lag = [a for a in st['answers'] if a['period'] == '20141231']
    add('- ', 'The lag day: %d answers (%s), each %s step(s) dated %s; fill or NaN on masked cells: %d.' % (len(lag), ', '.join(sorted((a['series'] for a in lag))), sorted(set((len(a['tflag_dates']) for a in lag))), sorted(set((d for a in lag for d in a['tflag_dates']))), sum((st2.get('fill_mask', 0) + st2.get('nan_mask', 0) + st2.get('fill_or_nan_mask_cells', 0) for a in lag for st2 in a['steps']))))
    add('')
    add('3. UNITS AGAINST THE CHANNEL TABLE')
    for s in MONTHLY + ('T2',):
        seen = sorted(set((u for (s2, y), r in A.items() if s2 == s for u in r['units'])))
        lagu = sorted(set((a['units'] for a in lag if a['series'] == s)))
        c2, ok = S.C2_UNITS[s]
        good = all((u in ok for u in seen + lagu))
        add('  %-16s expected %-11s answers %-10s %s' % (SER_LABEL[s], c2, '/'.join(seen), 'ok' if good else 'DIFFERS'))
    add('')
    add('4. RANGES (counted, kept, never altered; all cells / masked cells)')
    add('  series        year  range         out(all) out(mask)  min(mask)  max(mask)')
    add("  (PBLH's range 0..6000 m is applied to M-8 and to X-6, the daily maximum)")
    for s in MONTHLY:
        lo, hi = D.RANGES.get(s, (None, None))
        rg = '%s..%s' % (g(lo), g(hi)) if lo is not None else 'none'
        for y in S.YEARS:
            r = A[s, y]
            add('  %-13s %4d  %-13s %8d %9d %10s %10s' % (SER_LABEL[s], y, rg, r['below_all'] + r['above_all'], r['below_mask'] + r['above_mask'], g(r['min_mask']), g(r['max_mask'])))
    add('  ', '(M-4 and M-5 have no checked range: their extremes are given for the record; cell values after the four-corner mean.)')
    for s in ('M4', 'M5'):
        mn = min((A[s, y]['dot_min'] for y in S.YEARS))
        mx = max((A[s, y]['dot_max'] for y in S.YEARS))
        add('  ', '%s on the dot points, 2015-2019: min %s, max %s.' % (SER_LABEL[s], g(mn), g(mx)))
    add('  ', 'WSPD10 by its range: every value of M-3 lies in 0 to 50 m/s: %s (out of range, all cells: %d). The hourly WSPD10 and WDIR10 test of the download found the coverage named WSPD10 in 0 to 50 m/s and the one named WDIR10 in 0 to 360 degrees.' % (all((A['M3', y]['below_all'] + A['M3', y]['above_all'] == 0 for y in S.YEARS)), sum((A['M3', y]['below_all'] + A['M3', y]['above_all'] for y in S.YEARS))))
    add('')
    add('5. MCIP AND EMISSIONS')
    for name, rec in sorted(st['mcip'].items()):
        if rec.get('used') is False:
            add('  ', '%s: sha256 %s; not used (PURB is in GRIDCRO2D).' % (name, 'ok' if rec['sha256_ok'] else 'DIFFERS'))
            continue
        bits = []
        for n, vr in rec['vars'].items():
            t = '%s %s %s fill %d NaN %d min %s max %s' % (n, vr['units'] or '-', 'x'.join((str(x) for x in vr['shape'])), vr['fill'], vr['nan'], g(vr['min']), g(vr['max']))
            if 'values' in vr:
                t += ' values %s' % vr['values']
            if 'records_used' in vr:
                t += ' records %d-%d (%d, hours %d-%d)' % (vr['records_used'][0], vr['records_used'][1], vr['records_used_n'], vr['record_hours'][0] // 10000, vr['record_hours'][-1] // 10000)
            for k in ('below_0', 'above_1', 'above_100'):
                if k in vr:
                    t += ' %s %d' % (k, vr[k])
            bits.append(t)
        add('  ', '%s: sha256 %s; %s; TFLAG %s to %s, %d steps; grid attributes %s; %s.' % (name, 'ok' if rec['sha256_ok'] else 'DIFFERS', rec['data_model'], rec['tflag_first'], rec['tflag_last'], rec['steps'], 'equal' if not rec['grid_diff'] else 'DIFFER %s' % rec['grid_diff'], '; '.join(bits)))
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
    add('')
    add('6. WHAT IS EXCLUDED, AND WHY')
    for t in ("L-4 ALBEDO: not in METCRO2D; dropped by the channel table's rule, so 23 input channels are built, 17 of them the baseline's.", 'T-2 (lstaconc DAILY_NO2, 2016): check only, not an input or a target.', 'The first try of M4_metdot3d_UWIND_L1_DM_201607, the 11 test answers of the download and their 20 decompressed copies in W\\work\\rsig: s3_input false in the inventory; not read.', "The answers' LONGITUDE, LATITUDE and ELEVATION: not inputs (LONGITUDE and LATITUDE serve the grid check only).", "MCIP: LUFRAC_CRO (PURB present), GRIDCRO2D's MSFX2 and DLUSE, METCRO2D's other 40 variables and its 25th record (00 UTC of the 16th; the study design's rule for daily files): not used.", 'Emissions: the fire sectors and rwc are not read; E-4 takes the merged files only.', 'No value of any input is excluded or altered for its range or for a fill value: the counts above are kept as read.'):
        add('- ', t)
    add('')
    add('7. STOP TEST (more than 0.1 percent of masked cell-days missing)')
    verdict = 'STOP' if stop else 'PASS'
    add('  ', '%s: series and years over the limit: %s.' % (verdict, ', '.join(stop) if stop else 'none'))
    text = '\n'.join(L) + '\n'
    bad = S.check_record_text(text.rstrip('\n'))
    if bad:
        raise ValueError('; '.join(bad[:5]))
    with open(OUT, 'w', encoding='ascii', newline='\n') as f:
        f.write(text)
    summ = {'verdict': verdict, 'over_limit': stop, 'series_year': {'%s_%d' % k: {kk: vv for kk, vv in v.items() if kk not in ('variables',)} for k, v in A.items()}}
    S.write_json(os.path.join(S.WORK_S3, 'd4_summary.json'), summ)
    print('verdict %s; record %d lines' % (verdict, len(L)))
if __name__ == '__main__':
    main()
